""" Learn to predict notes from FFT data.

Send notes to MIDI synthesizer, then calculate FFT for resulting audio; FFT becomes input for neural network, while the known notes that produced it are training outputs.

Starting with input FFT length equal to JACK buffer size, output vector with length 88 corresponding to all notes on a standard piano, and one hidden layer. By heuristics, number of neurons in single hidden layer likely between 88 and 512.

Each note played will produce many buffers worth of audio data. FFT should vary over the duration of the note, but this can be minimized by disabling certain features of the synthesizer. Initially, play isolated notes/chords and take the FFTs at maximum amplitude as inputs, or use an average over the duration. As more complex examples are investigated, will need more nuanced algorithm to isolate harmonic features.

TODO: Switch to TensorFlow batch 1D FFT when supported by OpenCL
"""

import numpy as np
import muser.iodata
import muser.fft
import jack
import music21
import matplotlib.pyplot as plt
import tensorflow as tf

PIANO_LO = 21
PIANO_HI = 108
""" MIDI pitch range of 88-key piano """

audio_client_name = "MuserAudioClient"
audio_client = jack.Client(audio_client_name)
capture_1 = "Pianoteq55:out_1"
capture_2 = "Pianoteq55:out_2"
inport_1 = audio_client.inports.register("in_1")
inport_2 = audio_client.inports.register("in_2")

buffer_size = audio_client.blocksize
sample_rate = audio_client.samplerate

# Training parameters
batch_size = 64
learning_rate = 0.001
epochs = 10

# Neural network parameters
inputs = buffer_size
hidden1_n = 300
outputs = 88

# Input placeholder variables
x = tf.placeholder(tf.float32, shape=[None, inputs])

with tf.name_scope('hidden1'):
    weights = tf.Variable(
        tf.truncated_normal([inputs, hidden1_n],
                            stddev=1.0 / np.sqrt(float(inputs)))
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

buffers = []

@audio_client.set_process_callback
def process(frames):
    buffer = inport_1.get_array()
    buffers.append(buffer)

with audio_client:
    # connect to synthesizer's audio output
    audio_client.connect(capture_1, "{}:in_1".format(audio_client_name))
    audio_client.connect(capture_2, "{}:in_2".format(audio_client_name))
    input()

with tf.Session() as sess:
    with tf.device("/cpu:0"):
        pass
