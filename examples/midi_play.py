"""Simple playback of MIDI files through synthesizer.

Only accounts for note on and off events and tempo changes.
"""

import midi
import time
import muser.iodata as iodata

synth_midi_in = 'Pianoteq55:midi_in'

midi_file = '/home/mll/data/midi/goldberg.mid'
midi_pattern = midi.read_midifile(midi_file)

# flatten tracks
midi_pattern.make_ticks_abs()
events = []
for track in midi_pattern:
    for event in track:
        events.append(event)

timesig = {0: (4, 4)}
tempo = {0: 120}
events_sequence = {}

# filter and convert events
for event in events:
    if event.name == 'Time Signature':
        timesig[event.tick] = (event.numerator, event.denominator)
    elif event.name == 'Set Tempo':
        tempo[event.tick] = event.bpm
    # elif control change
    # elif program change
    elif event.name == 'Note On':
        note_tuple = (144, event.pitch, event.velocity)
        if event.tick in events_sequence:
            events_sequence[event.tick].append(note_tuple)
        else:
            events_sequence[event.tick] = [note_tuple]
    elif event.name == 'Note Off':
        note_tuple = (128, note.pitch, 0)
        if event.tick in events_sequence:
            events_sequence[event.tick].append(note_tuple)
        else:
            events_sequence[event.tick] = [note_tuple]

# client with MIDI event sending only
client = iodata.ExtendedClient(inports=0)
client.activate()
client.midi_outports[0].disconnect()
client.connect(client.midi_outports[0], synth_midi_in)

# playback: register tempo changes, send current events to synthesizer,
#     then pause for the current duration of one tick
for tick in range(0, max(events_sequence.keys())):
    if tempo.get(tick, 0):
        tempo_ = tempo[tick]
        s_per_tick = 60.0 / (tempo_ * midi_pattern.resolution)
    current_events = events_sequence.get(tick, [])
    if current_events:
        client.send_events(current_events)
    time.sleep(s_per_tick)
