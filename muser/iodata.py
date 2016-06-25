"""Input and output of music data, including ``.wav`` and live MIDI.

Includes extensions of ``jack`` that allow automatic registration of multiple
client ports, easier capturing of buffer arrays from an arbitrary number of
inports, sending of sequences of events to capture, and tracking times of
capture endpoints and buffer overrun/underrun events.

Conversion of vector representations of notes and chords to MIDI events.
"""

import sys
import time
import struct
import numpy as np
import rtmidi
import jack
from scipy.io import wavfile
import muser.utils

SND_DTYPES = {'int16': 16, np.int16: 16, 'int32': 32, np.int32: 32}
"""Data types that SciPy can import from `.wav`"""

N_PITCHES = 127
NOTE_ON = 0x90
NOTE_OFF = 0x80
ALL_NOTES_OFF = 0x7B
STATUS_ALIASES = {'NOTE_ON': NOTE_ON, 'ON': NOTE_ON,
                  'NOTE_OFF': NOTE_OFF, 'OFF': NOTE_OFF}
"""MIDI parameters."""

JACK_PORT_NAMES = {'inports':'in_{}', 'outports':'out_{}',
                   'midi_inports':'midi_in_{}', 'midi_outports':'midi_out_{}'}
"""Types of `jack` ports and their default naming."""


def _register_ports(jack_client, **port_args):
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


def init_jack_client(name="MuserClient", inports=0, outports=0,
                     midi_inports=0, midi_outports=0):
    """Return an inactive `jack` client with registered ports. """
    jack_client = jack.Client(name)
    _register_ports(jack_client, inports=inports, outports=outports,
                   midi_inports=midi_inports, midi_outports=midi_outports)
    return jack_client


class JackAudioCapturer(jack.Client):
    """JACK client binding a process that captures audio inports.

    Args:
        name (str): Client name.
        inports (int): Number of inports to register and capture from.
    """

    def __init__(self, name='CapturerClient', inports=1):
        super().__init__(name=name)
        _register_ports(self, inports=inports)
        self._inport_enum = list(enumerate(self.inports))
        self.set_process_callback(self._capture)
        self._captured = [[] for p in self._inport_enum]
        self.set_xrun_callback(self._handle_xrun)
        self._xruns = []

        self._capture_toggle = False
        self._process_lock = False
        self._timepoints = []

    def _handle_xrun(self, delay_usecs):
        self._xruns.append((time.time(), delay_usecs))

    @muser.utils.if_true('_capture_toggle')
    @muser.utils.set_true('_process_lock')
    def _capture(self, frames):
        """ The capture process. Runs continuously with activated client. """
        for p, inport in self._inport_enum:
            self._captured[p].append(inport.get_array())

    @muser.utils.record_timepoints('_timepoints')
    @muser.utils.set_true('_capture_toggle')
    def capture_events(self, events_sequence, send_events, blocks=None):
        """ Send groups of MIDI events in series and capture the result.

        Args:
            events_sequence (List[np.ndarray]):
            send_events (function):
            blocks (list): Number of JACK buffers to record for each set of
                events. If ``None``, records until silence
        """
        try:
            if not len(blocks) == len(events_sequence):
                raise ValueError("List of numbers of blocks to record was "
                                 "given instead of a constant, but its length "
                                 "does not match that of events sequence")
        except TypeError:
            blocks = [blocks] * len(events_sequence)
        for e, events in enumerate(events_sequence):
            send_events(events)
            while not self.n or not any(self.last[0]):
                pass
            if blocks[e] is None:
                while any(self.last[0]):
                    pass
            else:
                while self.n < blocks[e]:
                    pass

    @muser.utils.wait_while('_process_lock')
    def drop_captured(self):
        """ Return and empty the array of captured buffers.

        Returns:
            captured (np.ndarray): Previously captured and stored buffer arrays.
        """
        captured = np.array(self._captured, dtype=np.float32)
        self._captured = [[] for p in self._inport_enum]
        return captured

    @property
    def timepoints(self):
        """np.ndarray: Array of capture start and stop timepoints. """
        return np.array(self._timepoints)

    @property
    def xruns(self):
        """np.ndarray: Array of logged xrun times. """
        return np.array(self._xruns)

    @property
    @muser.utils.wait_while('_process_lock')
    def n(self):
        """int: The number of blocks captured per inport so far."""
        return len(self._captured[0])

    @property
    @muser.utils.wait_while('_process_lock')
    def last(self):
        """list: The last group of buffer arrays captured. """
        return [ch[-1] for ch in self._captured]


def disable_jack_client(jack_client):
    """Unregister all ports, deactivate, and close a JACK client."""
    jack_client.outports.clear()
    jack_client.inports.clear()
    jack_client.midi_outports.clear()
    jack_client.midi_inports.clear()
    jack_client.deactivate()
    jack_client.close()


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


def get_client_send_events(rtmidi_out):
    """ Returns an ``rtmidi`` client-specific ``send_events``. """
    def client_send_events(events):
        return send_events(rtmidi_out, events)
    return client_send_events


def midi_all_notes_off(midi_basic=False, pitch_range=(0, 128)):
    """Return MIDI event(s) to turn off all notes in range.

    Args:
        midi_basic (bool): Switches MIDI event type to turn notes off.
            Use NOTE_OFF events for each note if True, and single
            ALL_NOTES_OFF event if False.
        midi_range (Tuple[int]): Range of pitches for NOTE_OFF events, if used.
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
    sample_rate, snd = wavfile.read(wavfile_name)
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
