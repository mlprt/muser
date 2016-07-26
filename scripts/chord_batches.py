"""Send random chords of MIDI note events to a synth, and record audio.

By default, prints the number of JACK xruns (buffer overruns or underruns)
produced during the MIDI playback and capture process.
"""

import cProfile
import datetime
import json
import os
import pstats
import time
import numpy as np
import scipy.io.wavfile

import muser.audio as audio
import muser.live as live
import muser.sequencer as sequencer
import muser.utils as utils

rnd = np.random.RandomState()
date = datetime.datetime.now().strftime("%y%m%d-%Hh%M")

# Basic configuration
out_dir = '/tmp/muser/chord_batches'
wav_out = True
profile_capture = False

# Chord generation and capture parameters
batches = 2
batch_size = 2
chord_gen = sequencer.random_velocity_vector
chord_size = lambda: rnd.randint(1, 4)
velocity = (30, 128)
init_silence = 0.1
chord_time = 1.0
release_time = 0.0

# Synthesizer parameters
pianoteq_stereo = dict(
    name='Pianoteq55',
    midi_inports=['Pianoteq55:midi_in'],
    outports=['Pianoteq55:out_1', 'Pianoteq55:out_2'],
    reset=(0xB0, 0, 0),
)

# File name and path formats
out_subdir = os.path.join(out_dir, date)
os.makedirs(out_subdir, exist_ok=True)
names = dict(
    pickle='batch{}.pickle',
    wav='batch{}-chord{}.wav',
    start_log='params.json',
    end_log='end_log',
    capture_profile='capture_events-batch{}_chord{}-profile',
)
paths = {k: os.path.join(out_subdir, name) for k, name in names.items()}

# Write parameter log for monitors
with open(paths['start_log'], 'w') as start_log:
    params = {'paths': paths, 'batches': batches, 'batch_size': batch_size,
              'times': [init_silence, chord_time, release_time]}
    start_log.write(json.dumps(params))


#
chord_dtype = np.dtype([('velocity_vector', np.float32, sequencer.N_PITCHES),
                        ('captured_buffers', object)])
batch = np.ndarray([batch_size], dtype=chord_dtype)

# JACK client initialization
client = live.SynthInterfaceClient(synth_config=pianoteq_stereo)
samplerate = client.samplerate

with client:
    client.connect_synth()
    start_clock = time.perf_counter()
    for i_batch in range(batches):
        batch['velocity_vector'] = [chord_gen(chord_size, velocity=velocity)
                                    for _ in range(batch_size)]
        for i_chord, chord in enumerate(batch):
            init_pause = {'events': None, 'duration': init_silence}
            #
            velocity_vector = chord['velocity_vector']
            notes_on = sequencer.vector_to_midi_events('ON', velocity_vector)
            on_events = {'events': notes_on, 'duration': chord_time}
            notes_off = sequencer.vector_to_midi_events('OFF', velocity_vector)
            off_events = {'events': notes_off, 'duration': release_time}
            #
            event_groups = [init_pause, on_events, off_events]
            if profile_capture:
                name_i = paths['capture_profile'].format(i_batch, i_chord)
                cProfile.run('client.capture_events(event_groups)', name_i)
            else:
                client.capture_events(event_groups)
            chord['captured_buffers'] = client.drop_captured()

            if wav_out:
                snd = audio.buffers_to_snd(chord['captured_buffers'])
                wav_path = paths['wav'].format(i_batch, i_chord)
                scipy.io.wavfile.write(wav_path, samplerate, snd)

        batch.dump(paths['pickle'].format(i_batch))

if profile_capture:
    name = paths['capture_profile'].format(0, 0)
    profile = pstats.Stats(name).strip_dirs()
    profile.sort_stats('time').print_stats(10)

log_str = "Captured {} batches of {} chords, at [s]:\n".format(batches,
                                                               batch_size)
log_str += utils.logs_entryexit(client.capture_log,
                                output_labels={None: 'Xrun'},
                                ref_clock=start_clock,
                                header=('Start', 'End'))
xrun_print_end = ', at:' if client.n_xruns else '.'
log_str += "\n\n{} total Xruns{}\n".format(client.n_xruns, xrun_print_end)
for xrun in client.xruns - start_clock:
    log_str += '{:10.4f} s\n'.format(xrun[0])

print('\n' + log_str)

with open(paths['end_log'], 'w') as end_log:
    end_log.write(log_str)
