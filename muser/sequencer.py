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

PITCH_RANGES = {'piano': (PIANO_LO, PIANO_HI + 1),
                'full': (PITCH_LO, PITCH_HI + 1),
}
PITCH_RANGE_VECTORS = {r: np.arange(*PITCH_RANGES[r]) for r in PITCH_RANGES}
STATUS_ALIASES = {'NOTE_ON': NOTE_ON, 'ON': NOTE_ON,
                  'NOTE_OFF': NOTE_OFF, 'OFF': NOTE_OFF,
}
"""Aliases for reference to MIDI constants by string arguments."""


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


def random_note(pitch=None, pitch_range_vector='full', velocity=None,
                velocity_range=None):
    """Return a ``music21.note.Note`` with specified or randomized attributes.

    TODO: Check music21 API and use existing functions as much as possible.
    ``music21.midi.translate`` has relevant functions but will need to
    implement MIDI note durations (i.e. event times passed with events).

    Args:
        pitch (int or str): Pitch of the returned ``music21.note.Note``.
            A MIDI note number or music21-parsable string. Randomly assigned
            if ``None`` (default).
        pitch_range_vector (iterable or string): Vector of MIDI pitches for
            random selection. May be a key from ``PITCH_RANGE_VECTORS``
            corresponding to a pre-defined range vector.
        velocity (int): Defined velocity. Randomly assigned if ``None``.
        velocity_range (tuple): Range of velocities for random assignment.

    Returns:
        note (music21.note.Note): Note object with assigned properties.
    """

    if pitch is None:
        try:
            pitch_range_vector = PITCH_RANGE_VECTORS[pitch_range_vector.lower()]
        except AttributeError:
            pass
        pitch = np.random.choice(pitch_range_vector)
    if velocity is None:
        if velocity_range is None:
            velocity_range = (VELOCITY_LO, VELOCITY_HI + 1)
        velocity = np.random.randint(*velocity_range)
    note = music21.note.Note(pitch)
    note.volume.velocity = velocity
    return note


def random_chord(chord_size=3, pitch_range_vector='full', velocity=None,
              velocity_range=None, unique=True):
    """Return a music21 Chord object with random pitches and velocity.

    TODO: MIDI velocity (int) or normalized velocity (/1.0)

    Args:
        chord_size (int): Number of notes in the returned chord.
        pitch_range_vector (iterable or str): Vector of MIDI pitches for
            random selection. May be a key from ``PITCH_RANGE_VECTORS``
            corresponding to a pre-defined range vector.
        velocity (int): MIDI velocity of the returned chord.
            Randomly assigned if ``None`` (default).
        velocity_range (tuple): Range of velocities for random assignment.
        unique (bool): If `True`, no duplicate pitches in returned chord.

    Returns:
        chord (music21.chord.Chord): Chord object.
    """
    try:
        pitch_range_vector = PITCH_RANGE_VECTORS[pitch_range_vector.lower()]
    except AttributeError:
        pass
    pitches = np.random.choice(pitch_range_vector, chord_size,
                               replace=not unique)
    notes = [random_note(pitch=p, velocity=velocity,
                         velocity_range=velocity_range) for p in pitches]
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
    velocity_vector = np.zeros(PITCH_HI)
    velocity_vector[[pitch.midi for pitch in chord.pitches]] = chord_velocity
    return velocity_vector


def note_to_velocity_vector(note):
    """Return a MIDI pitch vector for a music21 Note object."""
    velocity_vector = chord_to_velocity_vector(music21.chord.Chord([note]))
    return velocity_vector


def random_velocity_vector(pitches, pitch_range_vector='full', velocity=None,
                           velocity_range=None):
    """Return a random velocity vector.

    Args:
        pitches (int): Number of pitches in the velocity vector.
        pitch_range_vector (iterable or string): Vector of MIDI pitches for
            random selection.
        velocity (int): MIDI velocity of returned chord.
            Chosen randomly if ``None`` (default).
        velocity_range (tuple): Range for random assignment of velocity.
    Returns:
        velocity_vector (np.ndarray): Vector of velocities of each MIDI pitch
    """
    chord = random_chord(chord_size=pitches, pitch_range=pitch_range,
                         velocity=velocity, velocity_range=velocity_range)
    velocity_vector = chord_to_velocity_vector(chord)

    return velocity_vector


def midi_all_notes_off(midi_basic=False, pitch_range='all'):
    """Return MIDI event(s) to turn off all notes in range.

    Args:
        midi_basic (bool): Switches MIDI event type to turn notes off.
            Use NOTE_OFF events for each note if True, and single
            ALL_NOTES_OFF event if False.
        pitch_range (Tuple[int]): Range of pitches for NOTE_OFF events, if used.
            Defaults to entire MIDI pitch range.
    """
    try:
        pitch_range = PITCH_RANGES[pitch_range.lower()]
    except AttributeError:
        pass
    if midi_basic:
        pitches_off = np.zeros(N_PITCHES)
        pitches_off[slice(*pitch_range)] = 1
        return vector_to_midi_events(NOTE_OFF, pitches_off)
    else:
        return np.array(((ALL_NOTES_OFF, 0, 0),))


def vector_to_midi_events(status, velocity_vector):
    """ Return MIDI event parameters for given velocity vector.

    Status can be specified as one of the keys in ``STATUS_ALIASES``.

    Args:
        status: The status parameter of the returned events.
        velocity_vector (np.ndarray): Vector of velocities of each MIDI pitch

    Returns:
        chord_events (np.ndarray): MIDI event parameters, one event per row.
    """
    try:
        status = STATUS_ALIASES[status.upper()]
    except (KeyError, AttributeError):
        pass
    pitches = np.flatnonzero(velocity_vector)
    velocities = velocity_vector[pitches] * VELOCITY_HI
    chord_events = np.zeros((3, len(pitches)), dtype=np.uint8)
    chord_events[0] = status
    chord_events[1] = pitches
    chord_events[2] = velocities
    chord_events = chord_events.transpose()
    return chord_events


def note_to_midi_onoff(note, velocity=100):
    """Returns MIDI note-on and note-off events for a ``music21`` note.

    TODO: Velocity vectors.

    Args:
        note (music21.Note.note): The ``music21`` note to convert to events.
    """
    vector = note_to_velocity_vector(note)
    note_on = vector_to_midi_events(NOTE_ON, vector, velocity=velocity)[0]
    note_off = vector_to_midi_events(NOTE_ON, vector)[0]
    return note_on, note_off


def continuous_controller(status, data_byte1):
    """Return a function that varies the second data byte of a MIDI event.

    Args:
        status (int): The MIDI status byte.
        data_byte1 (int): The first MIDI data byte.
    """
    def event(data_byte2):
        return (status, data_byte1, data_byte2)
    return event
