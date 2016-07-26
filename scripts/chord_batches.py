"""Send random chords of MIDI note events to a synth, and record audio.

By default, prints the number of JACK xruns (buffer overruns or underruns)
produced during the MIDI playback and capture process.
"""

import cProfile
import datetime
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

# Synth and file configuration
synth_name = 'Pianoteq'
synth_reset_event = (0xB0, 0, 0)
data_dir = '/tmp/muser/chord_batches'
wav_out = False
profile_capture = False

# Paths and filenames
output_dir = os.path.join(data_dir, date)
os.makedirs(output_dir, exist_ok=True)
pickle_paths = os.path.join(output_dir, 'chords_batch{}.pickle')
wav_paths = os.path.join(output_dir, 'wav', 'batch{}-chord{}.wav')
log_path = os.path.join(output_dir, 'log')
profile_names = 'capture_events-batch{}_chord{}-profile'

# Batch generation parameters
chord_size = lambda: rnd.randint(1, 4)
velocity_lims = (30, 128)
batch_size = 2
batches = 2

# Recording parameters
chord_init_silence = 0.1  # >0 to prevent reset overlap...
chord_time = 1.0
chord_release_time = 0.0

# chord batches structure and random generation
chord_gen = sequencer.random_velocity_vector
member_kwargs = {'n_pitches': chord_size, 'velocity': velocity_lims}
chord_dtype = np.dtype([('velocity_vector', np.float32, sequencer.N_PITCHES),
                        ('captured_buffers', object)])
batch = np.ndarray([batch_size], dtype=chord_dtype)

# JACK client initialization
client = live.SynthInterfaceClient.from_synthname(synth_name=synth_name,
                                                  channels=['out_1', 'out_2'])
client.synth_config['reset'] = synth_reset_event
samplerate = client.samplerate

with client:
    client.connect_synth()
    start_clock = time.perf_counter()
    for i_batch in range(batches):
        batch['velocity_vector'] = [chord_gen(**member_kwargs)
                                    for _ in range(batch_size)]
        for i_chord, chord in enumerate(batch):
            init_pause = {'events': None, 'duration': chord_init_silence}
            #
            velocity_vector = chord['velocity_vector']
            notes_on = sequencer.vector_to_midi_events('ON', velocity_vector)
            on_events = {'events': notes_on, 'duration': chord_time}
            notes_off = sequencer.vector_to_midi_events('OFF', velocity_vector)
            off_events = {'events': notes_off, 'duration': chord_release_time}
            #
            event_groups = [init_pause, on_events, off_events]
            if profile_capture:
                name_i = os.path.join(data_dir,
                                      profile_names.format(i_batch, i_chord))
                cProfile.run('client.capture_events(event_groups)', name_i)
            else:
                client.capture_events(event_groups)
            chord['captured_buffers'] = client.drop_captured()

            if wav_out:
                snd = audio.buffers_to_snd(chord['captured_buffers'])
                wav_path = wav_paths.format(i_batch, i_chord)
                scipy.io.wavfile.write(wav_path, samplerate, snd)

        batch.dump(pickle_paths.format(i_batch))

if profile_capture:
    name = os.path.join(data_dir, profile_names.format(0, 0))
    profile = pstats.Stats(name).strip_dirs()
    profile.sort_stats('time').print_stats(10)

log_str += "Captured {} batches of {} chords, at [s]:\n".format(batches,
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

with open(log_path, 'w') as log:
    log.write(log_str)
