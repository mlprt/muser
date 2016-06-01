""" Create music objects in music21 and output as live MIDI.

During sends, destroys rtmidi.MidiOut interface after `KeyboardInterrupt` or `SystemExit` to preclude JACK artifacts.
"""

import muser.iodata
import muser.sequencer
import music21

tempo = 90/60.  # quarter notes per second
notes = muser.sequencer.notation_to_notes("tinynotation: C8 D# G c G D#")
durations = [tempo * 0.1] * len(notes)
for i, note in enumerate(notes):
    note.duration.quarterLength = durations[i]
    note.volume.velocityScalar = 0.7
midi_notes_ = muser.iodata.to_midi_notes(notes)
midi_notes = []
for note in midi_notes_:
    midi_notes.append((note[0], durations[i])) # note on and pause
    midi_notes.append((note[1], 0)) # note off

try:
    midi_out, _ = muser.iodata.init_midi_out()
    send_events = muser.iodata.get_send_events(midi_out)
    send_events(midi_notes, loop=5)

except (KeyboardInterrupt, SystemExit):
    try:
        del midi_out
    except NameError:
        pass
    raise
