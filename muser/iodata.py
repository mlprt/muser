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


def init_rtmidi_out(name="MuserRtmidiClient", outports=1):
    """ Return `rtmidi` output client with given number of ports.

    Parameters:
        name (str): The name of the `rtmidi` client.
        outports (int): Number of virtual output ports to initialize.

    Returns:
        midi_out (rtmidi.MidiOut): `rtmidi` output client.
    """
    midi_out = rtmidi.MidiOut(name=name)
    for o in range(outports):
        midi_out.open_virtual_port("out_{}".format(o))

    return midi_out


def init_jack_client(name="MuserJACKClient", inports=0, outports=0):
    """ Return an inactive `jack` client with registered audio ports. """
    jack_client = jack.Client(name)
    for i in range(inports):
        jack_client.inports.register("in_{}".format(i))
    for o in range(outports):
        jack_client.outports.register("out_{}".format(o))

    return jack_client


def del_jack_client(jack_client):
    """ Unregister all ports, deactivate, and close a `jack` client. """
    jack_client.outports.clear()
    jack_client.inports.clear()
    jack_client.midi_outports.clear()
    jack_client.midi_inports.clear()
    jack_client.deactivate()
    jack_client.close()


def get_send_events(midi_out):
    """ Returns a function that sends MIDI events through `rtmidi`.

    Parameters:
        midi_out (rtmidi.MidiOut): The `rtmidi` output client.

    Returns:
        send_events (function): Client-specific MIDI event sender.
    """

    def send_events(events):
        """ Send a series of MIDI events out through rtmidi.

        Events sent without pause, so intended for sending series of events that
        should be heard simultaneously (chords).

        Parameters:
            events (list): Tuples specifying MIDI events.
        """
        for event in events:
            midi_out.send_message(event)

    return send_events


def midi_all_notes_off(midi_out, midi_basic=False, midi_range=(0, 128)):
    """ Send MIDI event(s) to release (turn off) all notes.

    Parameters:
        midi_out (rtmidi.MidiOut): An `rtmidi` MIDI output client
        midi_basic (bool): Send NOTE_OFF events for each note if True, and
                           single ALL_NOTES_OFF event if False
        midi_range (tuple): Range of pitches for NOTE_OFF events if midi_basic
                            (defaults to entire MIDI pitch range)
    """
    if midi_basic:
        for midi_pitch in range(*midi_range):
            midi_out.send_message((NOTE_OFF, midi_pitch, 0))
    else:
        midi_out.send_message((ALL_NOTES_OFF, 0, 0))


def to_midi_note_events(pitch_vector, velocity_vector=None, velocity=100):
    """ Return tuples specifying on/off MIDI note events based on vectors.

    TODO: General function for events,
          e.g. note_on_events = to_midi_events(NOTE_ON, vector1, vector2)

    Parameters:
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


def get_to_sample_index(sample_frq):
    """Return function that converts time to sample index for given sample rate.

    Parameters:
        sample_frq (int):

    Returns:
        to_sample_index (function):
    """
    def to_sample_index(time):
        """ Return sample index corresponding to a given time.

        Parameters:
            time (float, int):

        Returns:

        """
        def to_sample(time):
            try:
                return int(time * sample_frq)
            except TypeError:  # time not specified or specified badly
                e = "Real local endpoints must be specified! (t_endp)"
                raise TypeError(e)
        try:
            samples = [to_sample(t) for t in time]

            return samples

        except TypeError:  # time not iterable
            sample = to_sample(time)

            return sample

    return to_sample_index


def unit_snd(snd, factor=None):
    """ Scale elements of an array of wav data from -1 to 1.

    Default factor is determined from `snd.dtype`, corresponding to the format imported by `scipy.io.wavfile`. Can scale other types of data if factor is appropriately specified and data object can be scaled element-wise with the division operator, as for np.ndarray.

    Parameters:
        snd (np.ndarray): Data (audio from `.wav`) to be scaled.
        factor (int): Divide elements of snd by this number.

    Returns:
        scaled (np.ndarray): Same shape as snd, with elements scaled by factor.
    """
    if factor is None:
        factor = 2. ** (SND_DTYPES[snd.dtype.name] - 1)
    scaled = snd / factor

    return scaled


def wav_read_scaled(wavfile_name):
    """ Return contents of `.wav` scaled from -1 to 1.

    Parameters:
        wavfile_name (str):

    Returns:
        sample_rate (int):
        snd (np.ndarray):
    """
    sample_rate, snd = wavfile.read(wavfile_name)
    snd = unit_snd(snd)

    return sample_rate, snd


def report_midi_event(event, last_frame_time=0, out=sys.stdout):
    """ Print details of a `jack` MIDI event.

    NOTE: Does not apply to tuples specifying events, as with `rtmidi`.
    Retaining this material with intent for further `jack` integration.

    Parameters:
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
