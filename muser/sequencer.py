""" Music theory structures. """

import music21

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
