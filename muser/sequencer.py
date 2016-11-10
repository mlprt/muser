"""Musical entity representation and manipulation.

MIDI message bytes are defined by the `MMA MIDI Specifications`_ for the sake
of compatibility with the many synthesizers and applications that adhere.

The current implementation only uses MIDI channel 1, but there are actually
16 channels available. For instance, to send a note-on message to channel 2
instead of channel 1, the note-on status byte should be 0x91 instead of 0x90.
To send a control change to channel 16 instead of channel 1, the control status
byte should be 0xBF instead of 0xB0.

.. _MMA MIDI Specifications
   https://www.midi.org/specifications/category/reference-tables
"""

import music21
import numpy as np

import muser.utils

N_PITCHES = 128
VELOCITY_LO = 0
VELOCITY_HI = 127
VELOCITY_LIMS = (VELOCITY_LO, VELOCITY_HI + 1)
"""MIDI constants."""

STATUS_BYTES = dict(
    NOTE_ON=0x90, ON=0x90,
    NOTE_OFF=0x80, OFF=0x80,
    CONTROL=0xB0,
)
CONTROL_BYTES = dict(
    PEDAL_SUSTAIN=0x40,
    PEDAL_PORTAMENTO=0x41,
    PEDAL_SOSTENUTO=0x42,
    PEDAL_SOFT=0x43,
    RESET_ALL_CONTROLLERS=0x79,
    ALL_NOTES_OFF=0x7B,
)
"""MIDI message (event) bytes."""

PITCH_LIMS = dict(
    midi=(0, 127),
    piano=(21, 108),
)
PITCH_RANGES = {name: np.arange(lim[0], lim[1] + 1)
                for name, lim in PITCH_LIMS.items()}
"""MIDI pitch (note number) limits and ranges for different instruments."""


def notation_to_notes(notation):
    """ Parse a notation string and return a list of Note objects.

    Args:
        notation (str): Melody notated in a music21-parsable string format.

    Returns:
        notes (list): Sequence of ``music21.note.Note`` objects.
    """
    sequence = music21.converter.parse(notation)
    notes = list(sequence.flat.getElementsByClass(music21.note.Note))
    return notes


def random_note(pitch=None, pitch_range='midi', velocity=None,
                velocity_lims=VELOCITY_LIMS):
    """Return a ``music21.note.Note`` with specified or randomized attributes.

    Args:
        pitch (int or str): Pitch of the returned ``music21.note.Note``.
            A MIDI note number or music21-parsable string. Randomly assigned
            if ``None`` (default).
        pitch_range (iterable or string): Vector of MIDI pitches for random
            selection. May be a key from ``PITCH_RANGES`` corresponding to a
            pre-defined range vector.
        velocity (int): Defined velocity. Randomly assigned if ``None``.
        velocity_lims (tuple): Range of velocities for random assignment.

    Returns:
        note (music21.note.Note): Note object with assigned properties.
    """

    if pitch is None:
        pitch_range = muser.utils.key_check(pitch_range, PITCH_RANGES, 'lower')
        pitch = np.random.choice(pitch_range)
    if velocity is None:
        velocity = np.random.randint(*velocity_lims)
    note = music21.note.Note(pitch)
    note.volume.velocity = velocity
    return note


def random_chord(chord_size=3, pitch_range='midi', velocity=None,
                 velocity_lims=VELOCITY_LIMS, unique=True):
    """Return a music21 Chord object with random pitches and velocity.

    TODO: MIDI velocity (int) or normalized velocity (/1.0)

    Args:
        chord_size (int): Number of notes in the returned chord.
        pitch_range (iterable or str): Vector of MIDI pitches for
            random selection. May be a key from ``PITCH_RANGES``
            corresponding to a pre-defined range.
        velocity (int): MIDI velocity of the returned chord.
            Randomly assigned if ``None`` (default).
        velocity_lims (tuple): Range of velocities for random assignment.
        unique (bool): If `True`, no duplicate pitches in returned chord.

    Returns:
        chord (music21.chord.Chord): Chord object.
    """
    pitch_range = muser.utils.key_check(pitch_range, PITCH_RANGES, 'lower')
    pitches = np.random.choice(pitch_range, chord_size, replace=not unique)
    notes = [random_note(pitch=p, velocity=velocity,
                         velocity_lims=velocity_lims) for p in pitches]
    chord = music21.chord.Chord(notes)
    return chord


def chord_to_velocity_vector(chord):
    """Return a MIDI velocity vector for a music21 Chord object.

    Args:
        chord (music21.chord.Chord): Chord object to convert to vector form.

    Returns:
        velocity_vector (np.ndarray): Vector of velocities of each MIDI pitch
    """
    chord_velocity = chord.volume.velocity
    if chord_velocity is None:
        chord_velocity = 1.0
    vector = velocity_vector()
    vector[[(p.midi - 1) for p in chord.pitches]] = chord_velocity
    return vector


def note_to_velocity_vector(note):
    """Return a MIDI pitch vector for a music21 Note object."""
    vector = chord_to_velocity_vector(music21.chord.Chord([note]))
    return vector


def random_velocity_vector(n_pitches=1, pitch_range='midi', velocity=None):
    """Return a random velocity vector.

    Args:
        n_pitches (int): Number of pitches in the velocity vector.
            If function is given: assigns call to function as ``n_pitches``.
        pitch_range (iterable or string): Vector of MIDI pitches for
            random selection.
        velocity (int or tuple): MIDI velocities of returned chord.
            If ``None``: random velocities in [0, 1).
            If a number constant is given: constant velocity.
    Returns:
        vector (np.ndarray): Velocities of each MIDI pitch/note number.
    """
    pitch_range = muser.utils.key_check(pitch_range, PITCH_RANGES, 'lower')
    try:
        n_pitches = n_pitches()
    except TypeError:
        pass
    pitches = np.random.choice(pitch_range, n_pitches, replace=False)
    vector = velocity_vector(pitch_range)
    try:
        vector[pitches,] = np.random.rand(n_pitches)
    except TypeError:
        vector[pitches,] = velocity
    return vector


def midi_all_notes_off(midi_basic=False, pitch_range='midi'):
    """Return MIDI event(s) to turn off all notes in range.

    Args:
        midi_basic (bool): Switches MIDI event type to turn notes off.
            Use NOTE_OFF events for each note if True, and single
            ALL_NOTES_OFF event if False.
        pitch_range (Tuple[int]): Range of pitches for NOTE_OFF events, if used.
            Defaults to entire MIDI pitch range.
    """
    pitch_range = muser.utils.key_check(pitch_range, PITCH_RANGES, 'lower')
    if midi_basic:
        pitches_off = np.zeros(N_PITCHES)
        pitches_off[slice(*pitch_range)] = 1
        return vector_to_midi_events(STATUS_BYTES['NOTE_OFF'], pitches_off)
    else:
        return np.array(((STATUS_BYTES['CONTROL'],
                          CONTROL_BYTES['ALL_NOTES_OFF'], 0),))


def vector_to_midi_events(status, vector):
    """ Return MIDI event parameters for given velocity vector.

    Status can be specified as one of the keys in ``STATUS_BYTES``.

    Args:
        status: The status parameter of the returned events.
        velocity_vector (np.ndarray): Vector of velocities of each MIDI pitch

    Returns:
        chord_events (np.ndarray): MIDI event parameters, one event per row.
    """
    status = muser.utils.key_check(status, STATUS_BYTES, 'upper')
    pitches = np.flatnonzero(vector)
    velocities = vector[pitches] * VELOCITY_HI
    chord_events = np.zeros((3, len(pitches)), dtype=np.uint8)
    chord_events[0] = status
    chord_events[1] = pitches
    chord_events[2] = velocities
    chord_events = chord_events.transpose()
    return chord_events


def note_to_midi_onoff(note):
    """Returns MIDI note-on and note-off events for a music21 note.

    Args:
        note (music21.Note.note): The ``music21`` note to convert to events.
    """
    vector = note_to_velocity_vector(note)
    note_on = vector_to_midi_events(STATUS_BYTES['NOTE_ON'], vector)
    note_off = vector_to_midi_events(STATUS_BYTES['NOTE_OFF'], vector)
    return note_on, note_off


def control_event(data_byte1, data_byte2=0, channel=1):
    """Return a MIDI control event with the given data bytes."""
    data_byte1 = muser.utils.key_check(data_byte1, CONTROL_BYTES, 'upper')
    return (STATUS_BYTES['CONTROL'] + channel - 1, data_byte1, data_byte2)


def continuous_event(status, data_byte1, channel=1):
    """Return a function that varies the second data byte of a MIDI event.

    Args:
        status (int or str): The MIDI status byte.
        data_byte1 (int or str): The first MIDI data byte (control byte).
        channel (int): The MIDI channel (1-16).
            Only applies to channel-dependent MIDI messages.
    """
    status = muser.utils.key_check(status, STATUS_BYTES, 'upper')
    def event(data_byte2):
        return (status + channel - 1, data_byte1, data_byte2)
    return event


def continuous_control(data_byte1, channel=1):
    """Return a function that varies the second data byte of a control event."""
    data_byte1 = muser.utils.key_check(data_byte1, CONTROL_BYTES, 'upper')
    return continuous_event(STATUS_BYTES['CONTROL'],
                            data_byte1=data_byte1,
                            channel=channel)


def velocity_vector(pitch_range='midi', dtype=np.float32):
    """Returns a velocity vector of zeros for all MIDI pitches."""
    pitch_range = muser.utils.key_check(pitch_range, PITCH_RANGES, 'lower')
    return np.zeros_like(pitch_range, dtype=dtype)


def beat_bias(beat_float, timesig, beat_biases):
    """Returns a bias depending on relative position in a measure.

    Currently uses linear interpolation between values in ``beat_biases``.

    Args:
        beat_float (float): Relative position in the measure.
            Refers to beats using indices, so a value of 0.5 refers to the
            position halfway between the first and second beats in the measure.
        timesig (Tuple[int]): The time signature of the measure.
            For example, ``(4, 4)`` corresponds to the time signature 4/4.
        beat_biases (dict):
    """
    beats = range(timesig[0])
    return np.interp(beat_float, beats, beat_biases[timesig])


def read_midifile(filepath):
    """Read a MIDI file and return a music21 ``MidiFile`` object."""
    midifile = music21.midi.MidiFile()
    midifile.open(filepath, 'rb')
    midifile.read()
    midifile.close()
    return midifile


def midifile_to_notes(midifile):
    """Read a MIDI file and return."""
    return music21.midi.translate.midiFileToStream(midifile).flat


def m21_midievent_to_event(midievent):
    """Convert a music21 MidiEvent to a tuple of MIDI bytes."""
    status = midievent.data + midievent.channel - 1
    return (status, midievent.pitch, midievent.velocity)
