"""Real-time music performance and capture.

Includes extensions of ``jack`` that allow automatic registration of multiple
client ports, easier capturing of buffer arrays from an arbitrary number of
inports, sending of sequences of events to capture, and tracking times of
capture endpoints and buffer overrun/underrun events.
"""

import copy
import itertools
import re
import struct
import time
import jack
import numpy as np
import rtmidi

import muser.utils

JACK_PORT_NAMES = {
    'inports':'in_{}',
    'outports':'out_{}',
    'midi_inports':'midi_in_{}',
    'midi_outports':'midi_out_{}',
}
"""Default naming of JACK port types."""


class MIDIRingBuffer(object):
    """Manages a JACK ringbuffer for thread-safe MIDI event passing.

    Writing to a ringbuffer decreases available write space; reading increases.

    Attributes:
        EVENT_FORMAT (str): The ``struct`` (C) format for each event.
        EVENT_SIZE (int): Number of bytes written to the ringbuffer per event.

    Args:
        size (int): Number of bytes allocated for ringbuffer storage.
            Rounded by ``jack.RingBuffer`` to the next-highest power of 2.
    """
    EVENT_HEADER_FORMAT = "IB"
    EVENT_HEADER_SIZE = struct.calcsize(EVENT_HEADER_FORMAT)

    def __init__(self, size):
        self.ringbuffer = jack.RingBuffer(size)

    def write_event(self, offset, event):
        """Write a MIDI event to the ringbuffer.

        Args:
            offset (uint32): The frame offset of the event.
            event (Tuple[uint8]): Bytes specifying a MIDI event.
        """
        n_bytes_event = len(event)
        n_bytes_write = self.EVENT_HEADER_SIZE + n_bytes_event
        write_format = self.EVENT_HEADER_FORMAT + '{}B'.format(n_bytes_event)
        if self.ringbuffer.write_space < n_bytes_write:
            raise jack.JackError('Low ringbuffer space, discarded event')
        write_data = struct.pack(write_format, offset, n_bytes_event, *event)
        self.ringbuffer.write(write_data)

    def read_events(self):
        """Read MIDI events currently stored in the ringbuffer.

        Returns:
            events (List[tuple]): List of retrieved MIDI offset/event pairs.
        """
        events = []
        while self.ringbuffer.read_space:
            header = self.ringbuffer.read(self.EVENT_HEADER_SIZE)
            offset, n_bytes_event = struct.unpack(self.EVENT_HEADER_FORMAT,
                                                  header)
            event_data = self.ringbuffer.read(n_bytes_event)
            event = struct.unpack("{}B".format(n_bytes_event), event_data)
            events.append((offset, event))
        return events


class AudioRingBuffer(object):
    """Manages a JACK ringbuffer for thread-safe passing of audio data.

    Attributes:
        FRAME_FORMAT (str): Defines the C datatype of each JACK audio frame.
            See ``jack_default_audio_sample_t`` in the JACK C source code.
            Given as a Python format string with an integer field to be
            replaced by the number of frames per buffer stored.
        FRAME_BYTES (int): Number of bytes per frame.
        BUFFER_FORMAT (str): Defines the C datatype of a JACK block buffer.
            Provided as a string template based on ``FRAME_FORMAT``, to be
            formatted with the number of frames per block on instantiation.

    Note:
        With 1024 samples/block and 44100 samples/s, 2 channels, and samples
        stored as C floats (4 bytes), an hour of audio is approximately
        155,000 samples or 1.27 GB, and 1 min is ~21 MB.

    Args:
        blocksize (int): Number of frames per JACK audio buffer.
        channels (int): Number of channels to be stored per block.
        blocks (int): Number of blocks for which to allocate storage.
            The more the read process lags behind the write process, and the
            longer the overall capture time, the larger this value should be
            to avoid losing blocks.
    """
    FRAME_FORMAT = 'f'  # floats
    FRAME_BYTES = struct.calcsize(FRAME_FORMAT)
    BUFFER_FORMAT = '{:d}' + FRAME_FORMAT

    def __init__(self, blocksize, channels, blocks=10000):
        self.buffer_format = self.BUFFER_FORMAT.format(blocksize)
        self.buffer_bytes = struct.calcsize(self.buffer_format)
        self.block_format = channels * self.buffer_format
        self.block_bytes = struct.calcsize(self.block_format)
        self.channels = channels

        self._ringbuffer = jack.RingBuffer(blocks * self.block_bytes)
        self._last = jack.RingBuffer(self.block_bytes + 1)
        self._active = False

    def write_block(self, buffers):
        """If ringbuffer is active, write buffers for a single block.

        Args:
            buffers (List[buffer]): JACK CFFI buffers from audio ports.
                Should have length equal to the number of channels.
        """
        if not len(buffers) == self.channels:
            raise ValueError("Number of buffers passed for block write not "
                             "equal to number of ringbuffer channels")
        data = b''.join(buffer_ for buffer_ in buffers)
        self._last.read_advance(self.block_bytes)
        self._last.write(data)
        if self._active:
            self._ringbuffer.write(data)

    def read_block(self):
        """Read a single block's buffers from the ringbuffer.

        Returns:
            buffers (List[buffer]): JACK CFFI buffers for a single audio block.
        """
        block = self._ringbuffer.read(self.block_bytes)
        buffers = list(map(b''.join, zip(*[iter(block)] * self.buffer_bytes)))
        return buffers

    def reset(self):
        """Empty the ringbuffer. Not thread safe."""
        self._ringbuffer.reset()

    def activate(self):
        """Enable writes of incoming buffers to ringbuffer."""
        self._active = True

    def deactivate(self):
        """Disable writes of incoming buffers to ringbuffer."""
        self._active = False

    @property
    def active(self):
        """bool: Whether incoming buffers are being written."""
        return self._active

    @property
    def last(self):
        """buffer: Copy of all channels of last stored block."""
        while not self._last.read_space:
            pass
        return self._last.peek(self.block_bytes)

    @property
    def n_blocks(self):
        """int: Number of blocks stored in ringbuffer."""
        n_blocks = self._ringbuffer.read_space // self.block_bytes
        return n_blocks


class ExtendedJackClient(jack.Client):
    """A JACK client with added management features.

    Defines a default Xrun callback that logs Xrun details, and properties
    for tracking of Xruns and access to commonly calculated quantities.

    Args:
        name (str): The JACK client name.
    """

    def __init__(self, name):
        super().__init__(name=name)
        self._xruns = []
        self._n_xruns = 0
        self.set_xrun_callback(self._handle_xrun)

    def _handle_xrun(self, delay_usecs):
        # does not need to be suitable for real-time execution
        self._xruns.append((time.perf_counter(), delay_usecs))
        self._n_xruns += 1

    @staticmethod
    def _register_ports(jack_client, **port_args):
        """Register a JACK client's ports of the given type and number.

        Note:
            It is caller's responsibility to properly specify ``**port_args``,
            and to document them!

        Args:
            jack_client (jack.Client): The client to register the ports.
            **port_args: Keys give port type, values give quantity to register.
                Keys of ``port_args`` must be key subset of ``JACK_PORT_NAMES``
        """
        for port_type, n_ports in port_args.items():
            ports = getattr(jack_client, port_type)
            for p_i in range(n_ports):
                ports.register(JACK_PORT_NAMES[port_type].format(p_i))

    def disconnect_all(self):
        """Disconnect all connections of ports belonging to instance."""
        for port in self.get_ports(self.name):
            port.disconnect()

    def dismantle(self):
        """Unregister all ports, deactivate, and close the ``jack`` client."""
        self.transport_stop()
        self.disconnect_all()
        self.deactivate()
        for port_type in JACK_PORT_NAMES:
            getattr(self, port_type).clear()
        self.close()

    @property
    def xruns(self):
        """np.ndarray: Array of logged xrun times."""
        return np.array(self._xruns)

    @property
    def n_xruns(self):
        """int: Number of xruns logged by the client."""
        return self._n_xruns

    @property
    def max_offset(self):
        """int: The largest offset for a buffer frame."""
        return self.blocksize - 1

    @property
    def blocktime(self):
        """float: The number of seconds in one JACK buffer."""
        blocktime = self.blocksize / self.samplerate
        return blocktime


class SynthInterfaceClient(ExtendedJackClient):
    """Extended JACK client with synthesizer interfacing features.

    Includes MIDI event sending to synth MIDI inports and capture of audio
    out of synth outports. Uses ``jack`` ringbuffers for thread-safe exchanges
    of MIDI and audio data with the process callback.

    Note:
        Number of bytes allocated for the MIDIRingBuffer does not need to equal
        number of frames in a JACK block, but assigned as such because at twice
        the blocksize (same samplerate), expect to send twice as many events
        per block.

    Args:
        synth_config (dict): Synthesizer configuration variables.
        audiobuffer_time (float): Minutes of audio to allocate for ringbuffer.
            This should be at least as long as the longest audio capture,
            uninterrupted by a call to ``self.drop_capture()``.
    """

    def __init__(self, synth_config, audiobuffer_time=None):
        super().__init__(name="Muser-{} Interface".format(synth_config['name']))
        ExtendedJackClient._register_ports(
            self, midi_outports=len(synth_config['midi_inports']),
            inports=len(synth_config['outports']),
        )
        self.synth_config = synth_config

        if audiobuffer_time is None:
            audiobuffer_time = 10
        audiobuffer_blocks = int(60 * audiobuffer_time / self.blocktime)
        self.__audiobuffer = AudioRingBuffer(self.blocksize, len(self.inports),
                                             blocks=audiobuffer_blocks)
        self.__eventsbuffer = MIDIRingBuffer(self.blocksize)
        self._capture_log = []
        self.set_process_callback(self.__process)

    @classmethod
    def from_synthname(cls, synth_name, reset_event=None,
                       audiobuffer_time=None):
        """ Return an interface to the active synth with the provided name.

        Searches active JACK ports for client names containing ``synth_name``,
        irrespective of case or surrounding characters.

        Example:
            Pianoteq v5.5.1 is active with port names like 'Pianoteq55:out_1',
            with one MIDI inport and five audio outports enabled by default.
            Calling this constructor with ``synth_name='pianoteq'`` constructs
            and returns a JACK client with 1 MIDI inport and 5 audio outports.

        Note:
            Regular expression strings (not compiled objects) can be passed to
            ``jack.Client.get_ports()``, but ``re`` is used here anyway because
            passing with the case-insensitive flag ``(?i)`` caused segfaults.

        Args:
            synth_name (str): Name of the synthesizer. Case-insensitive.
            reset_event (Tuple[int]): MIDI event that resets the synth.
                Synth behaviour is not universal, so
            audiobuffer_time (float): Minutes of audio buffer time to allocate.
        """
        synth_config = dict(name='', midi_inports=[], outports=[],
                            reset_event=reset_event)
        regex = re.compile(synth_name, re.IGNORECASE)
        with jack.Client('tmp') as client:
            for port in client.get_ports():
                if regex.search(port.name.split(':')[0]):
                    if port.is_midi and port.is_input:
                        synth_config['midi_inports'].append(port.name)
                    if port.is_audio and port.is_output:
                        synth_config['outports'].append(port.name)
            synth_config['name'] = synth_config['outports'][0].split(':')[0]
        return cls(synth_config, audiobuffer_time=audiobuffer_time)

    def __process(self, frames):
        self._capture(frames)
        self._midi_write(frames)

    def _capture(self, dummy):
        """The capture process. Runs continuously with activated client."""
        buffers = [port.get_buffer() for port in self.inports]
        self.__audiobuffer.write_block(buffers)

    def _midi_write(self, dummy):
        self.midi_outports[0].clear_buffer()
        for event in self.__eventsbuffer.read_events():
            self.midi_outports[0].write_midi_event(*event)

    def send_events(self, events):
        """Write events to the ringbuffer for reading by next process cycle.

        If offset is determined by ``frames_since_cycle_start``, which is an
        estimate, then force into ``range(blocksize)`` to prevent JACK errors.
        """
        offset = 0
        for event in events:
            self.__eventsbuffer.write_event(offset, tuple(event))

    def capture_events(self, event_groups, test_rate=25, amp_rel_thres=1e-4,
                       max_xruns=0, attempts=10, cpu_load_thres=15):
        """Send groups of MIDI events in series and capture the result.

        Args:
            event_groups (List[dict]): Groups of MIDI events to send.
                After each group is sent, a condition is awaited before sending
                the next, or stopping audio capture.
            test_rate (float): Frequency (Hz) of sequence continuance tests.
                Lower values will allow the capture to observe ``times`` more
                precisely and more accurately estimate the max. amplitude,
                but too low can cause premature continues or Xruns.
            amp_rel_thres (float): Fraction of the max amplitude (established
                during volume testing) at which to set the threshold for
                continuation.
            max_xruns (int): Max xruns to allow before re-attempting capture.
            attempts (int): Number of Xrun-prompted re-attempts before abort.
            cpu_load_thres (float): Wait for CPU load reported by JACK to
                drop below this threshold before re-attempting capture.
                Higher CPU load is associated with increased chance of Xruns.
        """
        test_period = 1.0 / test_rate
        # pause briefly to ensure inter-capture xruns are logged
        time.sleep(0.1)
        for _ in range(attempts):
            capture_args = (event_groups, test_period,
                            amp_rel_thres, max_xruns)
            self.__audiobuffer.activate()
            attempt = self._capture_loop(*capture_args)
            self.__audiobuffer.deactivate()
            if attempt is None:
                self.__audiobuffer.reset()
                while self.cpu_load() > cpu_load_thres:
                    pass
            else:
                return

    @muser.utils.prepost_method('reset_synth')
    @muser.utils.log_with_timepoints('_capture_log')
    def _capture_loop(self, event_groups, test_period, amp_rel_thres,
                      max_xruns):
        init_xruns = self.n_xruns
        for group in event_groups:
            group_start = time.perf_counter()
            if group['events'] is not None:
                self.send_events(group['events'])
                if group['duration'] is None:
                    amp_max, amp_thres = 0, 0
                    while True:
                        time.sleep(test_period)
                        if (self.n_xruns - init_xruns) > max_xruns:
                            return
                        last = struct.unpack(self.__audiobuffer.buffer_format,
                                             self.__audiobuffer.last)
                        last_max = max(last)
                        if last_max > amp_max:
                            amp_max = last_max
                            amp_thres = amp_max * amp_rel_thres
                            if not last_max > amp_thres:
                                break
                    continue

            while (time.perf_counter() - group_start) < group['duration']:
                if (self.n_xruns - init_xruns) > max_xruns:
                    return
                time.sleep(test_period)
        return event_groups

    def drop_captured(self):
        """Return the audio data captured in the ringbuffer.

        Returns:
            captured (np.ndarray): JACK audio
        """
        blocks = []
        buffer_fmt = self.__audiobuffer.buffer_format
        while self.__audiobuffer.n_blocks:
            block = self.__audiobuffer.read_block()
            blocks.append([struct.unpack(buffer_fmt, b) for b in block])
        captured = np.array(blocks, dtype=np.float32).swapaxes(0, 1)
        return captured

    def connect_synth(self, disconnect=True):
        """Connect interface and synthesizer ports.

        Args:
            disconnect (bool): Whether to disconnect interface first.
                Prevents ``jack.JackError`` if client was auto-connected.
        """
        if disconnect:
            self.disconnect_all()
        for midi_port_pair in zip(self.midi_outports, self.synth_config['midi_inports']):
            self.connect(*midi_port_pair)
        for audio_port_pair in zip(self.synth_config['outports'], self.inports):
            self.connect(*audio_port_pair)

    def reset_synth(self):
        """Send signal to synthesizer to reset output."""
        if self.synth_config['reset_event'] is not None:
            self.send_events((self.synth_config['reset_event'],))

    @property
    def capture_log(self):
        """list: Captured event sequences and their timepoints."""
        return copy.deepcopy(self._capture_log)

    @property
    def capture_times(self):
        """np.ndarray: Start and stop clock of event sequence captures."""
        return np.array([s[1:] for s in self._capture_log])


class SynthClient(ExtendedJackClient):
    """Manages multichannel, functional synthesis of audio.

    TODO: Pre-calculated waveforms to minimize JACK xruns. Threading?
    """

    def __init__(self, name="Muser Synth", channels=1):
        super().__init__(name=name)
        ExtendedJackClient._register_ports(self, midi_inports=channels,
                                           outports=channels)
        self.set_process_callback(self.__process)
        self.synth_functions = [[] for ch in self.outports]
        self.channel_range = range(len(self.outports))

        self._toggle = False
        self._time = itertools.cycle(np.linspace(0, 1, self.samplerate,
                                                 endpoint=False).tolist())

    def __process(self, frames):
        self._play(frames)

    def _play(self, dummy):
        buffers = [memoryview(p.get_buffer()).cast('f') for p in self.outports]
        if self._toggle:
            for i_frame in range(self.blocksize):
                time_i = next(self._time)
                for i_chan in self.channel_range:
                    buffer_ = buffers[i_chan]
                    buffer_[i_frame] = 0
                    for func in self.synth_functions[i_chan]:
                        buffer_[i_frame] += func(time_i)
        else:
            for i_frame in range(self.blocksize):
                for i_chan in self.channel_range:
                    buffer_ = buffers[i_chan]
                    buffer_[i_frame] = 0


    def toggle(self):
        """Toggle audio synthesis on all channels."""
        self._toggle = not self._toggle

    def add_synth_function(self, synth_func, channels_idx=None):
        """Add an audio-generating function to the synthesizer.

        Args:
            synth_func (function): Takes time and returns amplitude.
            channels_idx (list): Indices of channels to which to add.
                If ``None`` (default), adds the function to all channels.
        """
        if channels_idx is None:
            for channel in self.synth_functions:
                channel.append(synth_func)
        else:
            for i_chan in channels_idx:
                self.synth_functions[i_chan].append(synth_func)

    def clear_synth_functions(self, channels_idx=None):
        """Remove all generating functions from the synthesizer.

        Args:
            channels_idx (list): Indices of channels to clear of functions.
        """
        if channels_idx is None:
            self.synth_functions = [[] for ch in self.outports]
        else:
            for i_chan in channels_idx:
                self.synth_functions[i_chan] = []


def jack_client_with_ports(name="Muser", inports=0, outports=0,
                           midi_inports=0, midi_outports=0):
    """Return an inactive ``jack`` client with registered ports."""
    jack_client = jack.Client(name)
    ExtendedJackClient._register_ports(jack_client,
                                       inports=inports, outports=outports,
                                       midi_inports=midi_inports,
                                       midi_outports=midi_outports)
    return jack_client


def init_rtmidi_out(name="MuserRtmidiClient", outport=0):
    """Return an ``rtmidi`` output client with opened port.

    Args:
        name (str): The name of the ``rtmidi`` client.
        outport (int): Virtual output port number to initialize.

    Returns:
        midi_out (rtmidi.MidiOut): ``rtmidi`` output client.
    """
    midi_out = rtmidi.MidiOut(name=name)
    if midi_out.get_ports():
        midi_out.open_port(outport)
    else:
        midi_out.open_virtual_port("out_{}".format(outport))
    return midi_out


def send_events(rtmidi_out, events):
    """Send a series of MIDI events out through ``rtmidi``.

    Events are sent without pause, so intended for sending series of events
    that should be heard simultaneously (chords).

    A single event (not nested in a list of events) may be passed as ``events``.

    Args:
        rtmidi_out (rtmidi.MidiOut): The `rtmidi` output client.
        events: MIDI event data.
    """
    try:
        iter(events[0])
    except TypeError:
        events = (events,)
    for event in events:
        rtmidi_out.send_message(event)


def get_rtmidi_send_events(rtmidi_out):
    """Returns a client-specific ``send_events``."""
    def rtmidi_send_events(events):
        return send_events(rtmidi_out, events)
    return rtmidi_send_events


def unpack_midi_event(event_in):
    """Convert received MIDI event parameters from binary to tuple form.

    Args:
        event_in: Iterable containing sample offset (in buffer) as first
            element and binary MIDI event specifier as second element.

    Returns:
        unpacked_event (tuple): Series of integers specifying the MIDI event.
            The first element is status and is always defined for events. This
            tuple's length is in ``range(1, 4)``.
    """
    _, indata = event_in
    for n_items in range(3, 0, -1):
        try:
            unpacked_event = struct.unpack('{}B'.format(n_items), indata)
        except struct.error:
            pass
    try:
        return unpacked_event
    except NameError:
        raise ValueError("event_in not an unpackable binary representation "
                         "of a MIDI event tuple")
