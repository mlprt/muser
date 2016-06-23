"""Input and output of music data, including `.wav` and live MIDI.

Low-latency audio connectivity is provided by python-rtmidi via JACK.
"""

import sys
import time
import binascii
import struct
import numpy as np
import rtmidi
import jack
import music21
from scipy.io import wavfile

SND_DTYPES = {'int16': 16, 'int32': 32}
""" Data types that SciPy can import from `.wav`"""

NOTE_ON = 0x90
NOTE_OFF = 0x80
ALL_NOTES_OFF = 0x7B
""" MIDI parameters. """

JACK_PORT_NAMES = {'inports':'in_{}', 'outports':'out_{}',
                   'midi_inports':'midi_in_{}', 'midi_outports':'midi_out_{}'}
""" Types of `jack` ports and their default naming. """


def _register_ports(jack_client, **port_args):
    """ Register a JACK client's ports of the given type and number.

    Note:
        It is the caller's responsibility to properly specify `**port_args`,
        and to document them!

    Args:
        jack_client (`jack.Client`): The client to register the ports.
        **port_args: Keywords give port type, args give quantity to register.
            `port_args.keys()` must be a subset of `JACK_PORT_NAMES.keys()`
    """
    for port_type, n in port_args.items():
        ports = getattr(jack_client, port_type)
        for p in range(n):
            ports.register(JACK_PORT_NAMES[port_type].format(p))


def init_jack_client(name="MuserClient", inports=0, outports=0,
                     midi_inports=0, midi_outports=0):
    """ Return an inactive `jack` client with registered ports. """
    jack_client = jack.Client(name)
    _register_ports(jack_client, inports=inports, outports=outports,
                   midi_inports=midi_inports, midi_outports=midi_outports)
    return jack_client


class JackAudioCapturer(jack.Client):
    """ JACK client with process capturing audio inports when toggled.

    Attributes:
        capture_toggle (bool): While ``True``, instance captures buffers.
        captured (np.ndarray): Buffers captured since instantiation or drop.

    Args:
        name (str): Client name.
        inports (int): Number of inports to register and capture from.
    """

    def __init__(self, name='CapturerClient', inports=1):
        super().__init__(name=name)
        _register_ports(self, inports=inports)
        self._buffer_array = np.zeros([len(self.inports), 1, self.blocksize],
                                      dtype=np.float64)
        self.capture_toggle = False
        self.captured = self._empty_captured
        self.set_process_callback(self._capture)

    def _capture(self, frames):
        """ The capture process. Runs continuously with activated client. """
        if self.capture_toggle:
            for p, inport in enumerate(self.inports):
                self._buffer_array[p] = inport.get_array()
            self.captured = np.append(self.captured, self._buffer_array, axis=1)

    @property
    def _empty_captured(self):
        """np.ndarray: An (empty) array for storage of captured buffers. """
        return np.ndarray([len(self.inports), 0, self.blocksize])

    def drop_captured(self):
        """ Return and empty the array of captured buffers.

        Returns:
            captured (np.ndarray): Previously captured and stored buffer arrays.
        """
        captured = np.copy(self.captured)
        self.captured = self._empty_captured
        return captured

    @property
    def n(self):
        """int: The number of buffer array groups stored in ``self.captured``.

        Depends on capture time but not the number of inports.
        """
        return self.captured.shape[1]

    @property
    def last(self):
        """np.ndarray: The last group of buffer arrays captured. """
        return self.captured[:, -1]


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


def midi_all_notes_off(rtmidi_out, midi_basic=False, midi_range=(0, 128)):
    """Send MIDI event(s) to release (turn off) all notes.

    Args:
        midi_out (rtmidi.MidiOut): An `rtmidi` MIDI output client
        midi_basic (bool): Switches MIDI event type to turn notes off.
            Use NOTE_OFF events for each note if True, and single
            ALL_NOTES_OFF event if False.
        midi_range (Tuple[int]): Range of pitches for NOTE_OFF events, if used.
            Defaults to entire MIDI pitch range.
    """
    if midi_basic:
        for midi_pitch in range(*midi_range):
            rtmidi_out.send_message((NOTE_OFF, midi_pitch, 0))
    else:
        rtmidi_out.send_message((ALL_NOTES_OFF, 0, 0))


def to_midi_note_events(pitch_vector, velocity_vector=None, velocity=100):
    """ Return tuples specifying on/off MIDI note events based on vectors.

    TODO: General function for events,
          e.g. note_on_events = to_midi_events(NOTE_ON, vector1, vector2)

    Args:
        pitch_vector (np.ndarray): The vector of MIDI pitches
        velocity_vector (np.ndarray): The vector of MIDI velocities
        velocity (int): Constant velocity in place of `velocity_vector`

    Returns:
        note_on_events (np.ndarray): MIDI NOTE_ON event parameters
        note_off_events (np.ndarray): MIDI NOTE_OFF event parameters
    """
    midi_pitches = np.flatnonzero(pitch_vector)
    n = len(midi_pitches)
    if velocity_vector is None:
        velocity_vector = np.full(n, velocity, dtype=np.uint8)
    note_on_events = np.full((3, n), NOTE_ON, dtype=np.uint8)
    note_on_events[1] = midi_pitches
    note_on_events[2] = velocity_vector
    note_off_events = np.copy(note_on_events)
    note_off_events[0] = np.full(n, NOTE_OFF, dtype=np.uint8)

    return note_on_events.T, note_off_events.T


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


def buffers_to_snd(buffers, stereo=True, dtype='int32'):
    """ Convert a series of JACK buffers to 2-channel SciPy audio.

    Args:
        buffers (np.ndarray): Series of JACK buffers in a 3D array.
            Second dimension size is number of buffers, third is `buffer_size`.
        stereo (bool): If `True`, the two channels of `snd` are taken from
            `buffers[0:2]`, else both copied from `buffers[0]` (mono).
        dtype (str): Datatype of the returned array.
            Should be 'int32' or 'int16' for SciPy compatibility.

    Returns:
        snd (np.ndarray): SciPy-compatible array of audio frames.
    """
    if stereo:
        buffers_ = buffers[0:2]
    else:
        buffers_ = np.concatenate((buffers[0], buffers[0]))
    snd = buffers_.reshape((2, buffers_.size // 2)).T
    snd = snd * 2.**(SND_DTYPES[dtype] - 1)
    snd = snd.astype(dtype)

    return snd


def report_midi_event(event, last_frame_time=0, out=sys.stdout):
    """ Print details of a JACK MIDI event.

    Note:
        Does not apply to tuples specifying events, as with ``rtmidi``.
        Retaining this material with intent for further ``jack`` integration.

    Args:
        event ():
        last_frame_time (int):
        out ():
    """
    offset, indata = event
    #print(struct.unpack(str(len(indata))+'B\n', indata))
    try:
        status, pitch, vel = struct.unpack('3B', indata)
    except struct.error:

        return
    rprt = "{0} + {1}:\t0x{2}\n".format(last_frame_time,offset,
                                      binascii.hexlify(indata).decode())
    #rprt += "indata: {0}\n".format(indata)
    rprt += "status: {0},\tpitch: {1},\tvel.: {2}\n".format(status, pitch, vel)
    #rprt += "repacked: {0}".format(struct.pack('3B', status, pitch, vel))
    rprt += "\n"
    out.write(rprt)
