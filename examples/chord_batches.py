"""Send random chords of MIDI note events to a synth, and record audio.

By default, prints the number of JACK xruns (buffer overruns or underruns)
produced during the MIDI playback and capture process.
"""

import numpy as np
import muser.iodata as iodata
import muser.live as live
import muser.sequencer as sequencer
import muser.utils as utils
import scipy.io.wavfile
import os
import time

import cProfile
import pstats

# User and synth parameters
data_dir = '/tmp/muser/'
print_details = True
profile_capture = True
synth_config = dict(
    name="pianoteq",
    reset=(0xB0, 0, 0),
    pedal_soft=iodata.continuous_controller(0xB0, 67),
    pedal_sustain=iodata.continuous_controller(0xB0, 64),
    pedal_sostenuto=iodata.continuous_controller(0xB0, 66),
    pedal_harmonic=iodata.continuous_controller(0xB0, 69),
)

# Batch generation parameters
chord_size = 1
batch_size = 2
batches = 1

# data structure
chord_dtype = np.dtype([('velocity_vector', np.uint8, iodata.N_PITCHES),
                        ('captured_buffers', object)])
chord_batches = np.ndarray([batches, batch_size], dtype=chord_dtype)

# generate chord vectors
chord_gen = sequencer.random_pitch_vector
chord_batches['velocity_vector'] = utils.get_batches(chord_gen, batches,
                                                     batch_size, [chord_size])

# JACK client initialization
client = live.SynthInterfaceClient.from_synthname(synth_config['name'],
                                                    reset_event=(0xB0,0,0))
samplerate = client.samplerate

client.activate()
try:
    # connect MIDI and audio ports of synthesizer and JACK client
    # (disconnect all first to prevent errors if auto-reconnected)
    # TODO: Move to iodata
    client.disconnect_all()
    client.connect(client.midi_outport, client.synth_midi_inports[0])
    for port_pair in zip(client.synth_outports, client.inports):
        client.connect(*port_pair)

    start_time = time.time()
    for batch in chord_batches:
        for chord in batch:
            velocity_vector = chord['velocity_vector']
            notes_on = iodata.vector_to_midi_events('ON', velocity_vector,
                                                    velocity=100)
            notes_off = iodata.vector_to_midi_events('OFF', velocity_vector)
            events_sequence = [notes_on, notes_off]
            capture_exec = ('client.capture_events(events_sequence, '
                            'blocks=(250, 25), init_blocks=25, amp_testrate=50, '
                            'max_xruns=1)')
            if profile_capture:
                cProfile.run(capture_exec, 'capture_events-profile')
            else:
                exec(capture_exec)
            chord['captured_buffers'] = client.drop_captured()

    if profile_capture:
        profile = pstats.Stats('capture_events-profile').strip_dirs()
        profile.strip_dirs().sort_stats('time').print_stats(10)

    if print_details:
        print("{} Xruns".format(client.n_xruns))
        for xrun in (client.xruns - start_time):
            print('{:.4f} s'.format(xrun[0]))
        print("\nCapture timepoints")
        print('{:>10}  \t{:>10}'.format('Start', 'Stop'))
        for item in client.captured_sequences:
            times = np.array(item[1]) - start_time
            xrun = " (Xrun)" if item[0] is None else ''
            print("{:10.4f} s\t{:10.4f} s {}".format(times[0], times[1], xrun))

except (KeyboardInterrupt, SystemExit):
    print('\nUser or system interrupt, dismantling JACK clients!')
    client.dismantle()
    raise

client.dismantle()

# store chord batches
batches_dir = os.path.join(data_dir, 'chord_batches')
os.makedirs(batches_dir, exist_ok=True)
pickle_path = os.path.join(batches_dir, 'chord_batches.pickle')
chord_batches.dump(pickle_path)

for b, batch in enumerate(chord_batches):
    for c, chord in enumerate(batch):
        snd = iodata.buffers_to_snd(chord['captured_buffers'])
        wavfile_name = 'batch{}_chord{}.wav'.format(b, c)
        wavfile_path = os.path.join(batches_dir, wavfile_name)
        scipy.io.wavfile.write(wavfile_path, samplerate, snd)
