"""Simple playback of MIDI files through synthesizer.

Only accounts for note on and off events and tempo changes.
"""

import cProfile
import os
import pstats
import midi
import numpy as np
import scipy.io.wavfile as wavfile

import muser.audio as audio
import muser.live as live
import muser.sequencer as sequencer

# configuration of synthesizer and I/O paths
synth_name = 'Pianoteq55'
midi_name = 'chopin-op28-no7'
midi_file = '/home/mll/data/midi/{}.mid'.format(midi_name)
wavfile_path = '/tmp/muser/{}.wav'.format(midi_name)
profile_path = '/tmp/muser/tmp/profile'
os.makedirs(profile_path, exist_ok=True)

# basic velocity (volume) bias for beats in a measure
bias_beats = True
beat_biases = {(4, 4): (1.0, 0.9, 0.95, 0.875),
               (3, 4): (1.0, 0.8, 0.9)}

# read and flatten MIDI file
midi_pattern = midi.read_midifile(midi_file)
ticks_per_beat = midi_pattern.resolution
midi_pattern.make_ticks_abs()
events = []
for track in midi_pattern:
    for event in track:
        events.append(event)

# filter events and convert to 
events_by_tick = {}
timesig, tempo = {}, {}
for event in events:
    if event.name == 'Time Signature':
        timesig[event.tick] = (event.numerator, event.denominator)
        timesig_ = timesig[event.tick]
        ticks_per_measure = ticks_per_beat * event.numerator
    elif event.name == 'Set Tempo':
        tempo[event.tick] = event.bpm
    # elif control change
    # elif program change
    elif event.name == 'Note On':
        note = [sequencer.STATUS_BYTES['NOTE_ON'], event.pitch, event.velocity]
        if bias_beats:
            measure_pos = (event.tick % ticks_per_measure) / ticks_per_beat
            note[2] *= sequencer.beat_bias(measure_pos, timesig_, beat_biases)
            note[2] = int(note[2])
        if event.tick in events_by_tick:
            events_by_tick[event.tick].append(note)
        else:
            events_by_tick[event.tick] = [note]
    elif event.name == 'Note Off':
        note = [sequencer.STATUS_BYTES['NOTE_OFF'], event.pitch, 0]
        if event.tick in events_by_tick:
            events_by_tick[event.tick].append(note)
        else:
            events_by_tick[event.tick] = [note]

# sort events by MIDI tick and calculate ticks between events
events_list = sorted([(tick, group) for tick, group in events_by_tick.items()])
delta_ticks = np.diff(list(zip(*events_list))[0])
event_groups = [{'events': None, 'duration': 1.0}]
# convert ticks to times based on tempo
for i_group, group in enumerate(events_list):
    event_group = {'events': group[1]}
    tempo_i = tempo.get(group[0], 0)
    if tempo_i:
        s_per_tick = 60.0 / (tempo_i * midi_pattern.resolution)
    try:
        event_group['duration'] = delta_ticks[i_group] * s_per_tick
    except IndexError:
        event_group['duration'] = 1.0
    event_groups.append(event_group)

# client with MIDI event sending only
client = live.SynthInterfaceClient.from_synthname(synth_name,
                                                  audiobuffer_time=0.05)
client.synth_config['reset'] = (0xB0, 0, 0)

# capture and save synthesizer audio for the whole MIDI file
with client:
    client.connect_synth()
    cProfile.run('client.capture_events(event_groups)', profile_path)
buffers = client.drop_captured()
snd = audio.buffers_to_snd(buffers)
wavfile.write(wavfile_path, 44100, snd)

# print profile statistics
profile = pstats.Stats(profile_path).strip_dirs()
profile.sort_stats('time').print_stats(10)
