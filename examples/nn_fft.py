""" Learn to predict notes from FFT data.

Send notes to MIDI synthesizer, then calculate FFT for resulting audio; FFT becomes input for neural network, while the known notes that produced it are training outputs.

Starting with input FFT length equal to JACK buffer size, output vector with length 88 corresponding to all notes on a standard piano, and one hidden layer. By heuristics, number of neurons in single hidden layer likely between 88 and 512.

Each note played will produce many buffers worth of audio data. Resonance will cause some variability in the FFT over time after the note is played, but this can be minimized by disabling certain features of the synthesizer. Initially, play isolated notes/chords and take as inputs the FFTs at maximum amplitude, or as an average over the evolution of the note. As more complex examples are investigated, will need to use other algorithms to isolate harmonic features. Will likely experiment with RNNs.

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
capture_1 = "Pianoteq55:out_1"
capture_2 = "Pianoteq55:out_2"
audio_client = jack.Client(audio_client_name)
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
hidden1_neurons = 300
outputs = 88

ffts_ = tf.placeholder(tf.float32, shape=[None, inputs])
keys_ = tf.placeholder(tf.float32, shape=[None, outputs])

with tf.name_scope('hidden1'):
    weights = tf.Variable(
        tf.truncated_normal([inputs, hidden1_neurons],
                            stddev=1.0 / np.sqrt(float(inputs)))
        name='weights')
    biases = tf.Variable(tf.zeros([hidden1_neurons]), name='biases')

hidden1 = tf.nn.relu(tf.matmul(ffts, weights) + biases)
output = tf.matmul(hidden1, weights) + biases

# compute cost
# construct training optimizer

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
