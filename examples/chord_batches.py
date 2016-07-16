"""Send random chords of MIDI note events to a synth, and record audio.

By default, prints the number of JACK xruns (buffer overruns or underruns)
produced during the MIDI playback and capture process.
"""

import cProfile
import os
import pstats
import time
import numpy as np
import scipy.io.wavfile

import muser.audio as audio
import muser.live as live
import muser.sequencer as sequencer
import muser.utils as utils

# User and synth parameters
data_dir = '/tmp/muser/chord_batches'
os.makedirs(data_dir, exist_ok=True)
print_details = True
profile_capture = True
profile_name = 'capture_events-batch{}_chord{}-profile'
synth_config = dict(
    name="Pianoteq55",
    reset=(0xB0, 0, 0),
    pedal_soft=sequencer.continuous_controller(0xB0, 67),
    pedal_sustain=sequencer.continuous_controller(0xB0, 64),
    pedal_sostenuto=sequencer.continuous_controller(0xB0, 66),
    pedal_harmonic=sequencer.continuous_controller(0xB0, 69),
)

# Batch generation parameters
chord_size = 10
batch_size = 5
batches = 1

# chord batches structure and random generation
chord_dtype = np.dtype([('velocity_vector', np.uint8, sequencer.N_PITCHES),
                        ('captured_buffers', object)])
chord_batches = np.ndarray([batches, batch_size], dtype=chord_dtype)
chord_gen = sequencer.random_velocity_vector
chord_batches['velocity_vector'] = utils.get_batches(chord_gen, batches,
                                                     batch_size, [chord_size])

# JACK client initialization
client = live.SynthInterfaceClient.from_synthname(
    synth_name=synth_config['name'],
    reset_event=synth_config['reset'],
    audiobuffer_time=1,
)
samplerate = client.samplerate

client.activate()
try:
    client.connect_synth()
    start_clock = time.perf_counter()
    for i_batch, batch in enumerate(chord_batches):
        for i_chord, chord in enumerate(batch):
            init_pause = {'events': None, 'duration': 0.5}
            velocity_vector = chord['velocity_vector']
            notes_on = sequencer.vector_to_midi_events('ON', velocity_vector)
            on_events = {'events': notes_on, 'duration': 2.0}
            notes_off = sequencer.vector_to_midi_events('OFF', velocity_vector)
            off_events = {'events': notes_off, 'duration': 0.25}
            event_groups = [init_pause, on_events, off_events]
            if profile_capture:
                name_i = os.path.join(data_dir,
                                      profile_name.format(i_batch, i_chord))
                cProfile.run('client.capture_events(event_groups)', name_i)
            else:
                client.capture_events(event_groups)
            chord['captured_buffers'] = client.drop_captured()

    client.dismantle()

except (KeyboardInterrupt, SystemExit):
    print('\nUser or system interrupt, dismantling JACK clients!\n')
    client.dismantle()
    raise

if profile_capture:
    profile = pstats.Stats(profile_name.format(0, 0)).strip_dirs()
    profile.strip_dirs().sort_stats('time').print_stats(10)

if print_details:
    print("{} Xruns".format(client.n_xruns))
    for xrun in client.xruns - start_clock:
        print('{:.4f} s'.format(xrun[0]))
    print("\nCapture timepoints")
    print('{:>10}  \t{:>10}'.format('Start', 'Stop'))
    for log in client.capture_log:
        group_start = log['entry_clock'] - start_clock
        group_end = log['exit_clock'] - start_clock
        xrun = " (Xrun)" if log['output'] is None else ''
        print("{:10.4f} s\t{:10.4f} s {}".format(group_start, group_end, xrun))

# store chord batches
pickle_path = os.path.join(data_dir, 'chord_batches.pickle')
chord_batches.dump(pickle_path)

for b, batch in enumerate(chord_batches):
    for c, chord in enumerate(batch):
        snd = audio.buffers_to_snd(chord['captured_buffers'])
        wavfile_name = 'batch{}_chord{}.wav'.format(b, c)
        wavfile_path = os.path.join(data_dir, wavfile_name)
        scipy.io.wavfile.write(wavfile_path, samplerate, snd)
