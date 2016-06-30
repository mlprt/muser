"""Send random chords of MIDI note events to a synth, and record audio.

By default, prints the number of JACK xruns (buffer overruns or underruns)
produced during the MIDI playback and capture process.
"""

import numpy as np
import muser.iodata as iodata
import muser.sequencer as sequencer
import muser.utils as utils
import scipy.io.wavfile
import os

# User and synth parameters
data_dir = '/tmp/muser/'
os.makedirs(data_dir, exist_ok=True)
synth_outports = ["Pianoteq55:out_1", "Pianoteq55:out_2"]
synth_midi_in = "Pianoteq55:midi_in"

# Batch generation parameters
chord_size = 5
batch_size = 2
batches = 1
print_n_xruns = True

# data structure
chord_dtype = np.dtype([('velocity_vector', np.uint8, iodata.N_PITCHES),
                        ('captured_buffers', object)])
chord_batches = np.ndarray([batches, batch_size], dtype=chord_dtype)

# generate chord vectors
chord_gen = sequencer.random_pitch_vector
chord_batches['velocity_vector'] = utils.get_batches(chord_gen, batches,
                                                     batch_size, [chord_size])

# JACK client initialization
channels = len(synth_outports)
jack_client = iodata.ExtendedClient(inports=channels, midi_outports=1)
samplerate = jack_client.samplerate

jack_client.activate()
try:
    # connect MIDI and audio ports of synthesizer and JACK client
    jack_client.connect(jack_client.midi_outports[0], synth_midi_in)
    for port_pair in zip(synth_outports, jack_client.inports):
        port_pair[1].disconnect()
        jack_client.connect(*port_pair)

    for batch in chord_batches:
        for chord in batch:
            velocity_vector = chord['velocity_vector']
            notes_on = iodata.vector_to_midi_events('ON', velocity_vector,
                                                    velocity=100)
            notes_off = iodata.vector_to_midi_events('OFF', velocity_vector)
            events_sequence = [notes_on, notes_off]
            jack_client.capture_events(events_sequence, blocks=(250, 25),
                                       init_blocks=25)
            chord['captured_buffers'] = jack_client.drop_captured()

except (KeyboardInterrupt, SystemExit):
    print('\nUser or system interrupt, dismantling JACK clients!')
    iodata.ExtendedClient.dismantle(jack_client)
    raise

if print_n_xruns:
    print("xruns: {}".format(len(jack_client.xruns)))
iodata.ExtendedClient.dismantle(jack_client)

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
