""" Music theory structures. """

import music21
import numpy as np

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

def get_note_batch(batch_size, pitch_lo=PIANO_LO, pitch_hi=PIANO_HI,
                   velocity_lo=60, velocity_hi=120):
    """ Return a batch of `music21` Note objects.

    TODO: non-MIDI range input

    Arguments:
        batch_size (int): Number of pitches to return
        note_lo (int): MIDI pitch of lowest note in desired range
        note_hi (int): MIDI pitch of highest note in desired range

    Returns:
        notes (list): List of music21 Note objects
    """
    notes = []
    for n in range(batch_size):
        note = music21.note.Note(np.random.randint(pitch_lo, pitch_hi + 1))
        velocity = np.random.randint(velocity_lo, velocity_hi + 1)
        note.volume.velocityScalar = velocity
        notes.append(note)

    return notes
