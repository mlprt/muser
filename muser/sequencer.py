"""Music structure generation and manipulation."""

import music21
import numpy as np


N_PITCHES = 127
PITCH_LO = 0
PITCH_HI = 127
PIANO_LO = 21
PIANO_HI = 108
VELOCITY_LO = 0
VELOCITY_HI = 127
NOTE_ON = 0x90
NOTE_OFF = 0x80
ALL_NOTES_OFF = 0x7B
"""MIDI constants."""

STATUS_ALIASES = {'NOTE_ON': NOTE_ON, 'ON': NOTE_ON,
                  'NOTE_OFF': NOTE_OFF, 'OFF': NOTE_OFF}
"""Aliases for reference to MIDI constants by string arguments."""


def notation_to_notes(notation):
    """ Parse a notation string and return a list of Note objects.

    Parameters:
        notation (str): Melody notated in a `music21`-parsable string format.

    Returns:
        notes (list): Sequence of `music21` Note objects.
    """
    sequence = music21.converter.parse(notation)
    notes = list(sequence.flat.getElementsByClass(music21.note.Note))

    return notes


def random_note(pitch=None, pitch_range=None, velocity=None, velocity_range=None):
    """ Return a random `music21` Note object.

    Parameters:
        pitch (int or str): Defined pitch. Randomly assigned if `None` (default).
        pitch_range (tuple): Range of MIDI pitches for random assignment.
        velocity (int): Defined velocity. Randomly assigned if `None` (default).
        velocity_range (tuple): Range of velocities for random assignment.

    Returns:
        note (music21.note.Note): Note object with assigned properties.
    """

    if pitch is None:
        if pitch_range is None:
            pitch_range = (PIANO_LO, PIANO_HI + 1)
        pitch = np.random.randint(*pitch_range)
    if velocity is None:
        if velocity_range is None:
            velocity_range = (VELOCITY_LO, VELOCITY_HI + 1)
        velocity = np.random.randint(*velocity_range)
    note = music21.note.Note(pitch)
    note.volume.velocity = velocity

    return note


def random_chord(chord_size=3, pitch_range=None, velocity=None,
              velocity_range=None, unique=True):
    """ Return a random `music21` Chord object.

    TODO: Switch from pitch range to pitch vector (easier uniques)

    Parameters:
        chord_size (int): Number of notes in the chord
        pitch_range (tuple):
        velocity (int): Constant velocity. If `None`, randomly assigned.
        velocity_range (tuple): Range of velocities for random assignment
        unique (bool): If `True`, no duplicate pitches in returned chord

    Returns:
        chord (music21.chord.Chord): Chord object
    """
    notes = []
    for n in range(chord_size):
        notes.append(random_note(pitch_range=pitch_range, velocity=velocity,
                 velocity_range=velocity_range))
        if unique:
            # TODO: remove current pitch from vector for next pass
            pass
    chord = music21.chord.Chord(notes)

    return chord


def chord_to_pitch_vector(chord, pitch_range=None):
    """ Return a MIDI pitch vector for a `music21` Chord object. """
    if pitch_range is None:
        pitch_range = (VELOCITY_LO, VELOCITY_HI+1)
    pitch_vector = np.zeros(VELOCITY_HI)
    pitch_vector[[pitch.midi for pitch in chord.pitches]] = 1
    pitch_vector = pitch_vector[slice(*pitch_range)]

    return pitch_vector


def note_to_pitch_vector(note, pitch_range=None):
    """ Return a MIDI pitch vector for a `music21` Note object. """
    chord = music21.chord.Chord([note])
    pitch_vector = chord_to_pitch_vector(chord, pitch_range=pitch_range)

    return pitch_vector


def random_pitch_vector(pitches, pitch_range=None, velocity=None,
                        velocity_range=None):
    """ Return a random pitch vector (chord).

    Parameters:
        pitches (int): Number of pitches in the chord/vector.
        velocity (int): Constant velocity of resulting chord. Random if `None`.
        velocity_range (tuple): Range for random assignment of velocity.
    Returns:
        pitch_vector (np.ndarray): Vector specifying pitches
    """
    chord = random_chord(chord_size=pitches, pitch_range=pitch_range,
                         velocity=velocity, velocity_range=velocity_range)
    pitch_vector = chord_to_pitch_vector(chord)

    return pitch_vector


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


def continuous_controller(status, data_byte1):
    """Return a function that varies the second data byte of a MIDI event.

    Args:
        status (int): The MIDI status byte.
        data_byte1 (int): The first MIDI data byte.
    """
    def event(data_byte2):
        return (status, data_byte1, data_byte2)
    return event
