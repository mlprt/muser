""" Learn to predict notes from FFT data.

Starting with input FFT length equal to JACK buffer size, the output vector has
length 88 corresponding to all notes on a standard piano. Start with one hidden
layer, which by heuristic should have between 88 and 512 neurons.

FFT will vary over the duration of the note, which can be minimized by disabling
certain synthesizer features. Will initially record isolated notes/chords and
take the FFTs at maximum/average amplitude as inputs. As more complex examples
are investigated, will need to isolate harmonic features.
"""

import tensorflow as tf
import numpy as np

data_file = '/tmp/muser/chord_batches/chord_batches.pickle'
chord_batches = np.load(data_file)

# Parameters
inputs = chord_batches.shape[2]
hidden1_n = 300
outputs = 88
learning_rate = 0.001

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
