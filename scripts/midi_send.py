""" Create music objects in music21 and output as live MIDI.

During sends, destroys rtmidi.MidiOut interface after `KeyboardInterrupt` or `SystemExit` to preclude JACK artifacts.
"""

import muser.live as live
import muser.sequencer as sequencer
import time
import music21

loops = 10
tempo = 90/60.  # quarter notes per second
notes = sequencer.notation_to_notes("tinynotation: C8 D# G c G D#")
durations = [tempo * 0.1] * len(notes)
for i, note in enumerate(notes):
    note.duration.quarterLength = durations[i]
midi_notes = [sequencer.note_to_midi_onoff(note) for note in notes]

try:
    rtmidi_out = live.init_rtmidi_out()
    send_events = live.get_rtmidi_send_events(rtmidi_out)
    for i in range(loops):
        for note, duration in zip(midi_notes, durations):
            send_events(note[0])
            time.sleep(duration)
            send_events(note[1])

except (KeyboardInterrupt, SystemExit):
    try:
        del midi_out
    except NameError:
        pass
    raise
