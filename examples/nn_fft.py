""" Learn to predict notes from FFT data.

Send notes to MIDI synthesizer, then calculate FFT for resulting audio. FFT
becomes input for neural network, while the known notes that produced it become
training outputs.

Starting with input FFT length equal to JACK buffer size, the output vector has
length 88 corresponding to all notes on a standard piano. Start with one hidden
layer, which by heuristic should have between 88 and 512 neurons.

Each note played will produce many JACK buffers worth of audio data. FFT will
vary over the duration of the note, which can be minimized by disabling certain
synthesizer features. Will initially record isolated notes/chords and take the
FFTs at maximum/average amplitude as inputs. As more complex examples are
investigated, will need to isolate harmonic features.

Temporarily using ``rtmidi`` to send notes to the synthesizer, as it is simpler
to time without involving ``jack`` threading and buffer offsets.
"""

import numpy as np
import tensorflow as tf
import muser.iodata as iodata
import muser.sequencer as sequencer
import muser.utils as utils
import scipy.io.wavfile

# Synthesizer MIDI ports
synth_outports = ["Pianoteq55:out_1", "Pianoteq55:out_2"]
synth_midi_in = "Pianoteq55:midi_in"

channels = len(synth_outports)

# JACK capture client initialization
capturer = iodata.JACKAudioCapturer(inports=channels)
samplerate = capturer.samplerate
print_n_xruns = True

# MIDI output client initialization
rtmidi_out = iodata.init_rtmidi_out()
rtmidi_send_events = iodata.get_client_send_events(rtmidi_out)

# Training parameters
chord_size = 1
batch_size = 2
batches = 1
learning_rate = 0.001

# storage of results
# TODO: velocity vectors
chord_dtype = np.dtype([('pitch_vector', np.uint8, iodata.N_PITCHES),
                        ('captured_buffers', object)])
chord_batches = np.ndarray([batches, batch_size], dtype=chord_dtype)

# generate note batches
chord_batches['pitch_vector'] = utils.get_batches(sequencer.random_pitch_vector,
                                                  batches, batch_size,
                                                  [chord_size])

capturer.activate()
try:
    # connect synthesizer stereo audio outputs to ``jack`` client inputs
    for port_pair in zip(synth_outports, capturer.inports):
        capturer.connect(*port_pair)

    for batch in chord_batches:
        for chord in batch:
            pitch_vector = chord['pitch_vector']
            notes_on = iodata.vector_to_midi_events('ON', pitch_vector,
                                                    velocity=100)
            notes_off = iodata.vector_to_midi_events('OFF', pitch_vector)
            events = [notes_on, notes_off]
            capturer.capture_events(events, rtmidi_send_events, blocks=(50, 25),
                                    init_blocks=25)
            chord['captured_buffers'] = capturer.drop_captured()

except (KeyboardInterrupt, SystemExit):
    print('\nUser or system interrupt, dismantling JACK clients!')
    rtmidi_send_events(iodata.midi_all_notes_off(midi_basic=True))
    del rtmidi_out
    iodata.disable_jack_client(capturer)
    raise

if print_n_xruns:
    print("xruns: {}".format(len(capturer.xruns)))
iodata.disable_jack_client(capturer)

# store audio results
chord_batches.dump('chord_batches.pickle')

for b, batch in enumerate(chord_batches):
    for c, chord in enumerate(batch):
        snd = iodata.buffers_to_snd(chord['captured_buffers'])
        wavfile_name = 'batch{}_chord{}.wav'.format(b, c)
        scipy.io.wavfile.write(wavfile_name, samplerate, snd)

quit()

# Neural network parameters
inputs = chord_batches.shape[2]
hidden1_n = 300
outputs = 88

# Input placeholder variables
x = tf.placeholder(tf.float32, shape=[None, inputs])

with tf.name_scope('hidden1'):
    weights = tf.Variable(
        tf.truncated_normal([inputs, hidden1_n],
                            stddev=1.0 / np.sqrt(float(inputs))),
        name='weights')
    biases = tf.Variable(tf.zeros([hidden1_n]), name='biases')
    hidden1 = tf.nn.relu(tf.matmul(x, weights) + biases)

with tf.name_scope('output'):
    weights = tf.Variable(
        tf.truncated_normal([hidden1_n, outputs],
                            stddev=1.0 / np.sqrt(float(hidden1_n))),
        name='weights')
    biases = tf.Variable(tf.zeros([outputs]),
                         name='biases')
    output = tf.matmul(hidden1, weights) + biases

# compute cost
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(output, labels,
                                                               name='crossent')
cost = tf.reduce_mean(cross_entropy, name='crossent_mean')

# prepare training optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op = optimizer.minimize(cost)


with tf.Session() as sess:
    with tf.device("/cpu:0"):
        pass
