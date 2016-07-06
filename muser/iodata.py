"""Input and output of music data, including ``.wav`` and live MIDI.

Includes extensions of ``jack`` that allow automatic registration of multiple
client ports, easier capturing of buffer arrays from an arbitrary number of
inports, sending of sequences of events to capture, and tracking times of
capture endpoints and buffer overrun/underrun events.

Conversion of vector representations of notes and chords to MIDI events.
"""

import numpy as np
import scipy.io.wavfile
import muser.utils
import jack
import time
import struct
import copy
import re
import rtmidi

SND_DTYPES = {'int16': 16, np.int16: 16, 'int32': 32, np.int32: 32}
"""Data types that SciPy can import from ``.wav``"""

N_PITCHES = 127
NOTE_ON = 0x90
NOTE_OFF = 0x80
ALL_NOTES_OFF = 0x7B
STATUS_ALIASES = {'NOTE_ON': NOTE_ON, 'ON': NOTE_ON,
                  'NOTE_OFF': NOTE_OFF, 'OFF': NOTE_OFF}
"""MIDI constants."""

JACK_PORT_NAMES = {'inports':'in_{}', 'outports':'out_{}',
                   'midi_inports':'midi_in_{}', 'midi_outports':'midi_out_{}'}
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
        data = b''.join(bytes(buffer_) for buffer_ in buffers)
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
    def n(self):
        """int: Number of blocks stored in ringbuffer."""
        return (self._ringbuffer.read_space // self.block_bytes)


class ExtendedJackClient(jack.Client):
    """A ``jack`` client with added management features.

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
        self._xruns.append((time.time(), delay_usecs))
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
        for port_type, n in port_args.items():
            ports = getattr(jack_client, port_type)
            for p in range(n):
                ports.register(JACK_PORT_NAMES[port_type].format(p))

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
        return (self.blocksize - 1)

    @property
    def blocktime(self):
        """float: The number of seconds in one JACK buffer."""
        return (self.blocksize / self.samplerate)


class SynthInterfaceClient(ExtendedJackClient):
    """Extended ``jack`` client with audio capture and MIDI event sending.

    Uses ``jack`` ringbuffers for thread-safe exchanges of MIDI and audio data
    with the process callback.

    Note:
        Number of bytes allocated for the MIDIRingBuffer does not need to equal
        number of frames in a JACK block, but assigned as such because at twice
        the blocksize (same samplerate), expect to send twice as many events
        per block.

    Args:
        name (str): Client name.
        inports (int): Number of inports to register and capture from.
            Intended to be equal to the number of synthesizer audio outports.
            Two-channel (stereo) capturing is default.
        audiobuffer_time (float): Minutes of audio to allocate for ringbuffer.
            This should be at least as long as the longest audio capture,
            uninterrupted by a call to ``self.drop_capture()``.
        reset_event (tuple): MIDI event to reset synth output.
            Not all synths will reset themselves in response to the same
            status byte, so the user should verify and alter as needed.
    """

    def __init__(self, synth_midi_inports, synth_outports,
                 name="Muser-Synth Interface", reset_event=None,
                 audiobuffer_time=None):

        super().__init__(name=name)
        ExtendedJackClient._register_ports(self, midi_outports=1,
                                           inports=len(synth_outports))
        self.midi_outport = self.midi_outports[0]
        self._inport_enum = list(enumerate(self.inports))
        self.synth_midi_inports = synth_midi_inports
        self.synth_outports = synth_outports

        if audiobuffer_time is None:
            audiobuffer_time = 10
        audiobuffer_blocks = int(60 * audiobuffer_time / self.blocktime)
        self.__audiobuffer = AudioRingBuffer(self.blocksize, len(self.inports),
                                             blocks=audiobuffer_blocks)
        self.__eventsbuffer = MIDIRingBuffer(self.blocksize)
        self._captured_sequences = []
        self._reset_event = reset_event
        self.set_process_callback(self._process)

    @classmethod
    def from_synthname(cls, synth_name, reset_event=None,
                       audiobuffer_time=None):
        """ Return an interface to the active synth with the provided name.

        Searches active JACK ports for client names containing ``synth_name``,
        irrespective of case or surrounding characters.

        Example:
            Pianoteq v5.5.1 is active with port names like 'Pianoteq55::out_1',
            with one MIDI inport and five audio outports enabled by default.
            Calling this constructor with ``synth_name='pianoteq'`` constructs
            and returns a JACK client with one MIDI outport and five audio
            outports, and the name 'Muser/Pianoteq55 Interface'.

        Note:
            Regular expression strings (not compiled objects) can be passed to
            ``jack.Client.get_ports()``, but ``re`` is used here anyway because
            passing with the case-insensitive flag '(?i)' caused segfaults.

        Args:
            synth_name (str): Name of the synthesizer. Case-insensitive.
            reset_event (Tuple[int]):
            audiobuffer_time (float):
        """
        client_name = "Muser/{} Interface"
        synth_midi_inports, synth_outports = [], []
        regex = re.compile(synth_name, re.IGNORECASE)
        with jack.Client('tmp') as client:
            for port in client.get_ports():
                port_name = port.name
                if regex.search(port_name.split(':')[0]):
                    if port.is_midi and port.is_input:
                        synth_midi_inports.append(port_name)
                    if port.is_audio and port.is_output:
                        synth_outports.append(port_name)
        synth_proper_name = synth_outports[0].split(':')[0]
        return cls(synth_midi_inports=synth_midi_inports,
                   synth_outports=synth_outports,
                   name=client_name.format(synth_proper_name),
                   reset_event=reset_event,
                   audiobuffer_time=audiobuffer_time)

    def _process(self, frames):
        self._capture(frames)
        self._midi_write(frames)

    def _capture(self, frames):
        """The capture process. Runs continuously with activated client."""
        buffers = [port[1].get_buffer() for port in self._inport_enum]
        self.__audiobuffer.write_block(buffers)

    def _midi_write(self, frames):
        self.midi_outport.clear_buffer()
        for event in self.__eventsbuffer.read_events():
            self.midi_outport.write_midi_event(*event)

    def send_events(self, events):
        """Write events to the ringbuffer for reading by next process cycle.

        If offset is determined by ``frames_since_cycle_start``, which is an
        estimate, might need to force into [0, blocksize).
        """
        offset = 0
        for event in events:
            self.__eventsbuffer.write_event(offset, tuple(event))

    def capture_events(self, events_sequence, send_events=None, blocks=None,
                       init_blocks=0, amp_testrate=25, amp_rel_thres=1e-4,
                       max_xruns=0, attempts=10, cpu_load_thres=15):
        """Send groups of MIDI events in series and capture the result.

        Args:
            events_sequence (List[np.ndarray]): Groups of MIDI events to send.
                After each group is sent, a condition is awaited before sending
                the next, or stopping audio capture.
            send_events (function): Accepts an iterable of MIDI events and
                sends them to the synthesizer.
            blocks (list): Number of JACK blocks to record for each set of
                events. Wherever ``None``, records the current events until
                the audio amplitude decreases past a threshold.
            init_blocks (int): Number of JACK blocks to record before sending
                the first set of events.
            amp_testrate (float): Frequency (Hz) of volume testing to establish
                a relative volume threshold, and to continue to the next set of
                events (or capture end) upon passing it. If too low, can cause
                premature continuation; too high, inaccuracy due to low
                resolution in time.
            amp_rel_thres (float): Fraction of the max amplitude (established
                during volume testing) at which to set the threshold for
                continuation.
            max_xruns (int): Max xruns to allow before re-attempting capture
            attempts (int): Number of xrun-prompted re-attempts before aborting
            cpu_load_thres (float): Re-attempt capture after CPU load reported
                by JACK drops below this threshold; 15% by default.
                Higher CPU load is associated with increased chance of Xruns.
        """
        try:
            if not len(blocks) == len(events_sequence):
                raise ValueError("List of numbers of blocks to record was "
                                 "given instead of a constant, but its length "
                                 "does not match that of events sequence")
        except TypeError:
            blocks = [blocks] * len(events_sequence)
        if send_events is None:
            send_events = self.send_events
        amp_testperiod = 1. / amp_testrate

        for a in range(attempts):
            capture_args = (events_sequence, send_events, blocks, init_blocks,
                            amp_testperiod, amp_rel_thres,
                            self.n_xruns, max_xruns)
            self.__audiobuffer.activate()
            attempt = self._capture_loop(*capture_args)
            self.__audiobuffer.deactivate()
            self.reset_synth()
            if attempt is None:
                self.__audiobuffer.reset()
                while self.cpu_load() > cpu_load_thres:
                    pass
                continue
            else:
                return

    @muser.utils.record_with_timepoints('_captured_sequences')
    def _capture_loop(self, events_sequence, send_events, blocks, init_blocks,
                      amp_testperiod, amp_rel_thres, init_xruns, max_xruns):
        while self.__audiobuffer.n < init_blocks:
            pass
        for e, events in enumerate(events_sequence):
            send_events(events)
            if blocks[e] is None:
                # wait for amplitude to fall below threshold
                amp_max = 0
                amp_thres = 0
                while True:
                    time.sleep(amp_testperiod)
                    if (self.n_xruns - init_xruns) > max_xruns:
                        return
                    last = struct.unpack(self.__audiobuffer.buffers_format,
                                         self.__audiobuffer.last)
                    last_max = max(last)
                    if last_max > amp_max:
                        amp_max = last_max
                        amp_thres = amp_max * amp_rel_thres
                    if not last_max > amp_thres:
                        break
            else:
                n = self.__audiobuffer.n
                while (self.__audiobuffer.n - n) < blocks[e]:
                    if (self.n_xruns - init_xruns) > max_xruns:
                        return
        return events_sequence

    def drop_captured(self):
        """Return the audio data captured in the ringbuffer.

        Returns:
            np.ndarray: JACK audio
        """
        blocks = []
        buffer_fmt = self.__audiobuffer.buffer_format
        while self.__audiobuffer.n:
            block = self.__audiobuffer.read_block()
            blocks.append([struct.unpack(buffer_fmt, b) for b in block])
        return np.array(blocks, dtype=np.float32).swapaxes(0, 1)

    def reset_synth(self):
        """Send signal to synthesizer to reset output."""
        if self._reset_event is not None:
            self.send_events((self._reset_event,))

    @property
    def captured_sequences(self):
        """list: Captured event sequences and their timepoints."""
        return copy.deepcopy(self._captured_sequences)

    @property
    def capture_timepoints(self):
        """np.ndarray: Start and stop timepoints of event sequence captures."""
        return np.array([s[1] for s in self._captured_sequences])



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

    Args:
        rtmidi_out (rtmidi.MidiOut): The `rtmidi` output client.
        events (List[tuple]): MIDI event data.
    """
    for event in events:
        rtmidi_out.send_message(event)


def get_rtmidi_send_events(rtmidi_out):
    """ Returns an ``rtmidi`` client-specific ``send_events``. """
    def rtmidi_send_events(events):
        return send_events(rtmidi_out, events)
    return client_send_events


def midi_all_notes_off(midi_basic=False, pitch_range=(0, 128)):
    """Return MIDI event(s) to turn off all notes in range.

    Args:
        midi_basic (bool): Switches MIDI event type to turn notes off.
            Use NOTE_OFF events for each note if True, and single
            ALL_NOTES_OFF event if False.
        pitch_range (Tuple[int]): Range of pitches for NOTE_OFF events, if used.
            Defaults to entire MIDI pitch range.
    """
    if midi_basic:
        pitches_off = np.zeros(N_PITCHES)
        pitches_off[slice(*pitch_range)] = 1
        return vector_to_midi_events(NOTE_OFF, pitches_off)

    else:
        return np.array(((ALL_NOTES_OFF, 0, 0),))


def vector_to_midi_events(status, pitch_vector, velocity=0):
    """ Return MIDI event parameters for given pitch vector.

    Status can be specified as one of the keys in ``STATUS_ALIASES``.

    Args:
        status: The status parameter of the returned events.
        pitch_vector (np.ndarray): The hot vector of MIDI pitches in a chord.
        velocity (int): The MIDI velocity of the chord.

    Returns:
        chord_events (np.ndarray): MIDI event parameters, one event per row.
    """

    try:
        status = STATUS_ALIASES[status.upper()]
    except (KeyError, AttributeError):
        pass
    pitches = np.flatnonzero(pitch_vector)
    chord_events = np.zeros((3, len(pitches)), dtype=np.uint8)
    chord_events[0] = status
    chord_events[1] = pitches
    chord_events[2] = velocity
    chord_events = chord_events.transpose()
    return chord_events


def to_sample_index(time, sample_rate):
    """Return sample index closest to given time.

    Args:
        time (float): Time relative to the start of sample indexing.
        sample_rate (int): Rate of sampling for the recording.

    Returns:
        sample_index (int): Index of the sample taken nearest to ``time``.
    """
    sample_index = int(time * sample_rate)
    return sample_index


def snd_norm(snd, factor=None):
    """ Scale elements of an array of (.wav) data from -1 to 1.

    Default factor is determined from ``snd.dtype``, corresponding to the
    format imported by ``scipy.io.wavfile``. Can scale other types of data if
    ``factor`` is appropriately specified and ``snd`` can be scaled
    element-wise with the division operator, as for ``np.ndarray``.

    Args:
        snd (np.ndarray): Data (audio from .wav) to be scaled.
        factor (int): Divide elements of ``snd`` by this number.

    Returns:
        scaled (np.ndarray): Same shape as ``snd``, with elements scaled.
    """
    if factor is None:
        factor = 2. ** (SND_DTYPES[snd.dtype.name] - 1)
    scaled = snd / factor
    return scaled


def wav_read_norm(wavfile_name):
    """ Return contents of .wav as array scaled from -1 to 1.

    Args:
        wavfile_name (str): Name of the .wav file to read.

    Returns:
        sample_rate (int): The file's audio sampling rate.
        snd (np.ndarray): The file's audio samples as ``float``.
    """
    sample_rate, snd = scipy.io.wavfile.read(wavfile_name)
    snd = snd_norm(snd)
    return sample_rate, snd


def buffers_to_snd(buffers, stereo=True, channel_ind=None, dtype=np.int32):
    """ Convert a series of JACK buffers to 2-channel SciPy audio.

    Args:
        buffers (np.ndarray): Series of JACK buffers in a 3D array. Second
            dimension length is number of channels, third is ``buffer_size``.
        stereo (bool): If ``True``, the two channels of ``snd`` are taken by
            default from ``buffers[0:2]``, else both from ``buffers[0]`` (mono).
        channel_ind: If stereo, can be a length-2 ``slice`` or a Numpy advanced
            index selecting two channels in ``buffers``. If mono, an integer,
            slice, or Numpy advanced index for a single channel must be passed.
        dtype (str): Datatype of the returned array.
            Must be a key in ``SND_DTYPES`` to ensure SciPy compatibility.

    Returns:
        snd (np.ndarray): SciPy-compatible array of audio frames.
    """
    if stereo:
        if channel_ind is None:
            channel_ind = slice(0, 2)
        buffers_ = buffers[channel_ind]
    else:
        if channel_ind is None:
            channel_ind = 0
        buffers_ = np.concatenate(np.atleast_2d(buffers[channel_ind]) * 2)
    snd = buffers_.reshape((2, buffers_.size // 2)).T
    snd = snd * 2.**(SND_DTYPES[dtype] - 1)
    snd = snd.astype(dtype)
    return snd


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
