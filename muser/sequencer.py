""" Music theory structures. """

import music21
import numpy as np

PITCH_LO = 0
PITCH_HI = 127
VELOCITY_LO = 0
VELOCITY_HI = 127
""" Absolute limits of MIDI pitch and velocity. """
PIANO_LO = 21
PIANO_HI = 108
""" MIDI pitch range of 88-key piano. """

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
