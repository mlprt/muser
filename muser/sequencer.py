""" Music theory structures. """

import music21
import numpy as np

VELOCITY_LO = 0
VELOCITY_HI = 127

PIANO_LO = 21
PIANO_HI = 108
""" MIDI pitch range of 88-key piano """

def notation_to_notes(notation):
    """ Parse a notation string and return a list of Note objects.

    Parameters:
        notation (str):

    Returns:
        notes (list):
    """
    sequence = music21.converter.parse(notation)
    notes = list(sequence.flat.getElementsByClass(music21.note.Note))

    return notes

def get_note(pitch=None, pitch_range=None, velocity=None, velocity_range=None):
    """ Return a random `music21` Note object.

    Parameters:
        pitch (int or str): Defined pitch. Randomly assigned if `None` (default)
        pitch_range (tuple): Range of MIDI pitches for random assignment
        velocity (int): Defined velocity. Randomly assigned if `None` (default)
        velocity_range (tuple): Range of velocities for random assignment

    Returns:
        note (music21.note.Note): Note object with assigned properties
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


def get_chord(chord_size=3, pitch_range=None, velocity=None,
              velocity_range=None, unique=True):
    """ Return a random `music21` Chord object.

    TODO: Switch from pitch range to pitch vector (easier uniques)

    Parameters:
        chord_size (int): Number of notes in the chord
        pitch_range (tuple):
        velocity (int): Constant velocity. If `None`, randomly assigned per note
        velocity_range (tuple): Range of velocities for random assignment
                                (if None, relays to `get_note`)
        unique (bool): If `True`, no duplicate pitches in returned chord

    Returns:
        chord (music21.chord.Chord): Chord object
    """
    notes = []
    for n in range(chord_size):
        notes.append(get_note(pitch_range=pitch_range, velocity=velocity,
                 velocity_range=velocity_range))
        if unique:
            # TODO: remove current pitch from vector for next pass
            pass
    chord = music21.chord.Chord(notes)

    return chord
