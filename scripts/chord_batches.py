"""Capture synthesizer audio for each of a batch of random chords.

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

## Output configuration
out_dir = '/tmp/muser/chord_batches'
# save each chord's captured audio data to a .wav file
wav_out = False
# profile the audio capture operation
profile_capture = False

## Chord generation and capture parameters
batches = 10
batch_size = 32
chord_size = 1 #lambda: rnd.randint(1, 4)
# function to generate random velocity vectors
chord_gen = sequencer.random_velocity_vector
# scalar or range of velocity
velocity = (30, 128)
# duration of silence captured efore sending chord's events
init_silence = 0.1
# duration of capture, before and after chord release
chord_time = 2.0
release_time = 0.0

## Synthesizer parameters
pianoteq_stereo = dict(
    name='Pianoteq55',
    midi_inports=['Pianoteq55:midi_in'],
    outports=['Pianoteq55:out_1', 'Pianoteq55:out_2'],
    reset=(0xB0, 0, 0),
)

## File name and path formats
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

## Data structure for chord batches
chord_dtype = np.dtype([('velocity_vector', np.float32, sequencer.N_PITCHES),
                        ('captured_buffers', object)])
batch = np.ndarray([batch_size], dtype=chord_dtype)

## JACK client initialization
client = live.SynthInterfaceClient(synth_config=pianoteq_stereo)
blocksize, samplerate = client.blocksize, client.samplerate

## Write to parameter log---for file monitors
# TODO: update utils.FileMonitor to use JSON logs
with open(paths['start_log'], 'w') as start_log:
    params = {'paths': paths, 'samplerate': samplerate, 'blocksize': blocksize,
              'batches': batches, 'batch_size': batch_size,
              'times': [init_silence, chord_time, release_time]}
    start_log.write(json.dumps(params))

with client:
    client.connect_synth()
    start_clock = time.perf_counter()
    for i_batch in range(batches):
        # generate batch of random chords (velocity vectors)
        batch['velocity_vector'] = [chord_gen(chord_size, velocity=velocity)
                                    for _ in range(batch_size)]
        for i_chord, chord in enumerate(batch):
            init_pause = {'events': None, 'duration': init_silence}
            # prepare the chord's MIDI events 
            velocity_vector = chord['velocity_vector']
            notes_on = sequencer.vector_to_midi_events('ON', velocity_vector)
            on_events = {'events': notes_on, 'duration': chord_time}
            notes_off = sequencer.vector_to_midi_events('OFF', velocity_vector)
            off_events = {'events': notes_off, 'duration': release_time}
            # collate event groups for client.capture_events
            event_groups = [init_pause, on_events, off_events]
            # send the event groups to the client for capture
            if profile_capture:
                name_i = paths['capture_profile'].format(i_batch, i_chord)
                cProfile.run('client.capture_events(event_groups)', name_i)
            else:
                client.capture_events(event_groups)
            # retrieve the captured audio for the chord
            chord['captured_buffers'] = client.drop_captured()

            # save the chord audio data to a .wav file
            if wav_out:
                snd = audio.buffers_to_snd(chord['captured_buffers'])
                wav_path = paths['wav'].format(i_batch, i_chord)
                scipy.io.wavfile.write(wav_path, samplerate, snd)

        batch.dump(paths['pickle'].format(i_batch))

## print profile of the capture operation
# TODO: statistics across chord profiles
if profile_capture:
    # (currently prints profile for first captured chord only)
    name = paths['capture_profile'].format(0, 0)
    profile = pstats.Stats(name).strip_dirs()
    profile.sort_stats('time').print_stats(10)

## generate and write post-capture log
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
