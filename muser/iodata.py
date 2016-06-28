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
import copy
import time
import struct
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

    Acts like a queue. Writing decreases available write space, reading
    increases it.

    Attributes:
        EVENT_FORMAT (str): The ``struct`` (C) format for each event.
        EVENT_SIZE (int): Number of bytes written to the ringbuffer per event.

    Args:
        size (int): Number of bytes allocated for ringbuffer storage.
            Rounded by ``jack.RingBuffer`` to the next-highest power of 2.
    """
    EVENT_FORMAT = "I3B" # 32-bit uint + 3 * 8-bit uint
    EVENT_SIZE = struct.calcsize(EVENT_FORMAT)

    def __init__(self, size):
        self.ringbuffer = jack.RingBuffer(size)

    def write_event(self, offset, event):
        """Write a MIDI event to the ringbuffer.

        Args:
            offset (uint32): The frame offset of the event.
            event (Tuple[uint8]): Bytes specifying a MIDI event.
        """
        if len(event) < 3:
            event += [0] * (3 - len(event))
        if self.ringbuffer.write_space < self.EVENT_SIZE:
            raise jack.JackError('Too little RingBuffer space, event discard')
        data = [offset] + event
        self.ringbuffer.write(struct.pack(self.EVENT_FORMAT, *data))

    def read_events(self):
        """Read MIDI events currently stored in the ringbuffer.

        Returns:
            events_list (List[tuple]): List of retrieved MIDI events.
        """
        events_list = []
        while self.ringbuffer.read_space:
            data = struct.unpack(self.EVENT_FORMAT,
                                 self.ringbuffer.read(self.EVENT_SIZE))
            offset, event = data[0], data[1:]
            events_list.append((offset, event))
        return events_list


class AudioRingBuffer(object):
    """Manages a JACK ringbuffer for thread-safe passing of audio data.

    Attributes:
        FRAME_FORMAT (str): Defines the C datatype of each JACK audio frame.
            See ``jack_default_audio_sample_t`` in the JACK C source code.
            Given as a Python format string with an integer field to be
            replaced by the number of frames per buffer stored.

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
    FRAME_FORMAT = "{:d}f"  # floats
    FRAME_SIZE = struct.calcsize("f")

    def __init__(self, blocksize, channels, blocks=10000):
        self.buffers_format = channels * self.FRAME_FORMAT.format(blocksize)
        self.buffers_size = struct.calcsize(self.buffers_format)
        self._ringbuffer = jack.RingBuffer(blocks * self.buffers_size)
        self._last = jack.RingBuffer(self.buffers_size + 1)
        self.channels = channels

    def write_block(self, buffers):
        """Write buffers for a single block to the ringbuffer.

        Args:
            buffers (List[buffer]): JACK CFFI buffers from audio ports.
                Should have length equal to the number of channels per block.
        """
        if not len(buffers) == self.channels:
            raise ValueError("Number of buffers passed for block write not "
                             "equal to number of ringbuffer channels")
        data = b''.join(bytes(buffer_) for buffer_ in buffers)
        self._last.read(self.buffers_size)
        self._last.write(data)
        self._ringbuffer.write(data)

    def read_block(self):
        """Read a single block's buffers from the ringbuffer.

        Returns:
            buffers (List[buffer]): JACK CFFI buffers for a single audio block.
        """
        buffers = self._ringbuffer.read(self.buffers_size)
        return buffers

    @property
    def last(self):
        """buffer: All channels of last stored block."""
        while not self._last.read_space:
            pass
        return self._last.peek(self.buffers_size)

    @property
    def n(self):
        """int: Number of blocks stored in ringbuffer."""
        return self._ringbuffer.read_space // self.buffers_size


class ExtendedClient(jack.Client):
    """Extended ``jack`` client with audio capture and MIDI event queuing.

    Args:
        name (str): Client name.
        inports (int): Number of inports to register and capture from.
        midi_outports (int): Number of MIDI outports to register.
        ringbuffer_time (float): Minutes of audio to allocate for ringbuffer.
    """

    def __init__(self, name='CapturerClient', inports=1, midi_outports=1,
                 ringbuffer_time=10):
        super().__init__(name=name)
        ExtendedClient.register_ports(self, inports=inports,
                                      midi_outports=midi_outports)
        self._inport_enum = list(enumerate(self.inports))
        self.set_process_callback(self._process)
        self._eventsbuffer = MIDIRingBuffer(self.blocksize)
        blocks = int((ringbuffer_time * 60) / (self.blocksize / self.samplerate))
        self._audiobuffer = AudioRingBuffer(self.blocksize, len(self.inports),
                                            blocks=blocks)

        self.set_xrun_callback(self._handle_xrun)
        self._xruns = []

        self._capture_toggle = False
        self._capture_timepoints = []

    def _handle_xrun(self, delay_usecs):
        self._xruns.append((time.time(), delay_usecs))

    def _process(self, frames):
        self._capture(frames)
        self._play(frames)

    @muser.utils.if_true('_capture_toggle')
    def _capture(self, frames):
        """The capture process. Runs continuously with activated client."""
        buffers = [port[1].get_buffer() for port in self._inport_enum]
        self._audiobuffer.write_block(buffers)

    def _play(self, frames):
        self.midi_outports[0].clear_buffer()
        for event in self._eventsbuffer.read_events():
            self.midi_outports[0].write_midi_event(*event)

    def send_events(self, events):
        """Write events to the ringbuffer for reading by next process cycle."""
        offset = self.frames_since_cycle_start
        for event in events:
            self._eventsbuffer.write_event(offset, event)

    @muser.utils.record_timepoints('_capture_timepoints')
    @muser.utils.set_true('_capture_toggle')
    def capture_events(self, events_sequence, send_events=None,
                       blocks=None, init_blocks=0,
                       amp_testrate=25, amp_rel_thres=1e-4):
        """Send groups of MIDI events in series and capture the result.

        TODO: Times (based on self.blocksize and self.samplerate) instead of
            blocks

        Args:
            events_sequence (List[np.ndarray]): Groups of MIDI events to send.
                After each group is sent, a condition is awaited before sending
                the next, or stopping audio capture.
            send_events (function): Accepts an iterable of MIDI events, and
                sends them to the synthesizer.
            blocks (list): Number of JACK blocks to record for each set of
                events. Wherever ``None``, records the set of events until
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
        """
        try:
            if not len(blocks) == len(events_sequence):
                raise ValueError("List of numbers of blocks to record was "
                                 "given instead of a constant, but its length "
                                 "does not match that of events sequence")
        except TypeError:
            blocks = [blocks] * len(events_sequence)
        events_sequence = [events.tolist() for events in events_sequence]
        if send_events is None:
            send_events = self.send_events

        while self._audiobuffer.n < init_blocks:
            pass
        for e, events in enumerate(events_sequence):
            send_events(events)
            if blocks[e] is None:
                amp_max = 0
                amp_thres = 0
                while True:
                    time.sleep(1. / amp_testrate)
                    last = struct.unpack(self._audiobuffer.buffers_format,
                                         self._audiobuffer.last)
                    last_max = max(last)
                    if last_max > amp_max:
                        amp_max = last_max
                        amp_thres = amp_max * amp_rel_thres
                    if not last_max > amp_thres:
                        break
            else:
                n = self._audiobuffer.n
                while (self._audiobuffer.n - n) < blocks[e]:
                    pass

    def drop_captured(self):
        """Return and empty the array of captured buffers.

        Returns:
            captured (np.ndarray): Previously captured and stored buffer arrays.
        """
        captured = [[] for i in self.inports]
        data_format = self._audiobuffer.buffers_format
        bs = self.blocksize
        while self._audiobuffer.n:
            data = struct.unpack(data_format, self._audiobuffer.read_block())
            for i in range(len(self.inports)):
                captured[i].append(data[i*bs:(i+1)*bs])
        captured = np.array(captured, dtype=np.float32)
        return captured

    @property
    def timepoints(self):
        """np.ndarray: Array of capture start and stop timepoints. """
        return np.array(self._timepoints)

    @property
    def xruns(self):
        """np.ndarray: Array of logged xrun times. """
        return np.array(self._xruns)

    @staticmethod
    def register_ports(jack_client, **port_args):
        """Register a JACK client's ports of the given type and number.

        Note:
            It is the caller's responsibility to properly specify ``**port_args``,
            and to document them!

        Args:
            jack_client (jack.Client): The client to register the ports.
            **port_args: Keywords give port type, args give quantity to register.
                ``port_args.keys()`` must be a subset of ``JACK_PORT_NAMES.keys()``
        """
        for port_type, n in port_args.items():
            ports = getattr(jack_client, port_type)
            for p in range(n):
                ports.register(JACK_PORT_NAMES[port_type].format(p))

    @staticmethod
    def dismantle(jack_client):
        """Unregister all ports, deactivate, and close a ``jack`` client."""
        jack_client.outports.clear()
        jack_client.inports.clear()
        jack_client.midi_outports.clear()
        jack_client.midi_inports.clear()
        jack_client.deactivate()
        jack_client.close()


def jack_client_with_ports(name="MuserClient", inports=0, outports=0,
                           midi_inports=0, midi_outports=0):
    """Return an inactive ``jack`` client with registered ports."""
    jack_client = jack.Client(name)
    ExtendedClient.register_ports(jack_client,
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


def unit_snd(snd, factor=None):
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


def wav_read_unit(wavfile_name):
    """ Return contents of .wav as array scaled from -1 to 1.

    Args:
        wavfile_name (str): Name of the .wav file to read.

    Returns:
        sample_rate (int): The file's audio sampling rate.
        snd (np.ndarray): The file's audio samples as ``float``.
    """
    sample_rate, snd = scipy.io.wavfile.read(wavfile_name)
    snd = unit_snd(snd)
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
