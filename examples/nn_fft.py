""" Learn to predict notes from FFT data.

Send notes to MIDI synthesizer, then calculate FFT for resulting audio; FFT becomes input for neural network, while the known notes that produced it are training outputs.

Starting with input FFT length equal to JACK buffer size, output vector with length 88 corresponding to all notes on a standard piano, and one hidden layer. By heuristics, number of neurons in single hidden layer likely between 88 and 512.

Each note played will produce many buffers worth of audio data. FFT should vary over the duration of the note, but this can be minimized by disabling certain features of the synthesizer. Initially, play isolated notes/chords and take the FFTs at maximum amplitude as inputs, or use an average over the duration. As more complex examples are investigated, will need more nuanced algorithm to isolate harmonic features.

Currently using `rtmidi` to send notes to the synthesizer, as this is much simpler to time without messing with threading and buffer offsets through `jack`, which now monitors audio buffers output from the synthesizer. Hope to eliminate one of these dependencies eventually.

TODO: Switch to TensorFlow batch 1D FFT when supported by OpenCL
"""

import numpy as np
import tensorflow as tf
import muser.iodata
import muser.fft
import jack
import music21
import matplotlib.pyplot as plt

PIANO_LO = 21
PIANO_HI = 108
""" MIDI pitch range of 88-key piano """

# Synthesizer MIDI outport names
capture_1 = "Pianoteq55:out_1"
capture_2 = "Pianoteq55:out_2"

audio_client_name = "MuserAudioClient"
audio_client = jack.Client(audio_client_name)
inport_1 = audio_client.inports.register("in_1")
inport_2 = audio_client.inports.register("in_2")
buffer_size = audio_client.blocksize
sample_rate = audio_client.samplerate

midi_out, _ = muser.iodata.init_midi_out()
send_events = muser.iodata.get_send_events(midi_out)

# Training parameters
batch_size = 64
batches = 10
learning_rate = 0.001


def get_note_batch(batch_size, pitch_lo=PIANO_LO, pitch_hi=PIANO_HI,
                   velocity_lo=60, velocity_hi=120):
    """ Return a batch of MIDI notes.

    Arguments:
        batch_size (int): Number of pitches to return
        note_lo (int): MIDI pitch of lowest note in desired range
        note_hi (int): MIDI pitch of highest note in desired range

    Returns:
        notes (list): List of music21 Note objects
    """
    notes = []
    for n in range(batch_size):
        # NOTE: not necessary to use music21 for now... return MIDI event tuples
        note = music21.note.Note(np.random.randint(pitch_lo, pitch_hi + 1))
        velocity = np.random.randint(velocity_lo, velocity_hi + 1)
        note.volume.velocityScalar = velocity
        notes.append(note)
        notes = muser.iodata.to_midi_notes(notes)

    return notes


notes_batches = [get_note_batch(batch_size) for b in range(batches)]

# `jack` monitor to capture audio buffers from synthesizer
buffers = []
@audio_client.set_process_callback
def process(frames):
    buffer = inport_1.get_array()
    # TODO: record output corresponding to rtmidi sends, collate with MIDI notes
    buffers.append(buffer)

# activate `jack` client and connect to synthesizer output
with audio_client:
    audio_client.connect(capture_1, "{}:in_1".format(audio_client_name))
    audio_client.connect(capture_2, "{}:in_2".format(audio_client_name))
    for notes in notes_batches:
        # TODO: fix muser.iodata pausing/offsets
        pass
    input()

# Neural network parameters
inputs = buffer_size
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
