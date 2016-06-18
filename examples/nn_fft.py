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
import muser.sequencer
import muser.fft
import jack
import matplotlib.pyplot as plt
import json

N_MIDI_NOTES = 127

# Synthesizer MIDI ports
synth_out_1 = "Pianoteq55:out_1"
synth_out_2 = "Pianoteq55:out_2"
synth_midi_in = "Pianoteq55:midi_in"

# `jack` initialization
audio_client_name = "MuserAudioClient"
audio_client = jack.Client(audio_client_name)
inport_1 = audio_client.inports.register("in_1")
inport_2 = audio_client.inports.register("in_2")
buffer_size = audio_client.blocksize
sample_rate = audio_client.samplerate

# `rtmidi` initialization
rtmidi_out = muser.iodata.init_midi_out()
send_events = muser.iodata.get_send_events(rtmidi_out)
rtmidi_out_name = "a2j:MuserRtmidiClient [131] (capture): Virtual Port Out 0"

# Training parameters
batch_size = 1
batches = 1
learning_rate = 0.001

# generate note batches
note_batches = muser.utils.get_batches(muser.sequencer.get_note, batches,
                                       batch_size)

# storage of results
rec_dtype = np.dtype([('note_vector', np.uint8, N_MIDI_NOTES),
                      ('buffers', object)])
recordings = np.ndarray([batches, batch_size], dtype=rec_dtype)
buffers = np.ndarray([0, buffer_size])

note_toggle = False

# `jack` monitor
@audio_client.set_process_callback
def process(frames):
    global note_toggle
    global buffers
    if note_toggle:
        buffer_ = inport_1.get_array()
        # record entire note (until silence)
        # loses buffers if check longer than buffer, assuming one monitor thread
        if np.any(buffer_):
            buffers = np.append(buffers, [buffer_], axis=0)
        else:
            note_toggle = False

with audio_client:
    try:
        # connect `rtmidi` MidiOut instance to synthesizer MIDI input
        audio_client.connect(rtmidi_out_name, synth_midi_in)
        # connect synthesizer stereo audio outputs to `jack` client inputs
        audio_client.connect(synth_out_1, "{}:in_1".format(audio_client_name))
        audio_client.connect(synth_out_2, "{}:in_2".format(audio_client_name))

        for b, notes in enumerate(note_batches):
            midi_notes = muser.iodata.to_midi_notes(notes)
            for n, note in enumerate(midi_notes):
                note_toggle = True
                send_events((note[0],))
                while note_toggle:
                    # `jack` listening through process()
                    pass
                send_events((note[1],))
                # TODO: note_vectors used throughout as struct for MIDI chords
                note_vector = np.array([note[0][2] if i==note[0][1] else 0
                                for i in range(N_MIDI_NOTES)], dtype=np.uint8)
                recordings[b][n]['note_vector'] = note_vector
                recordings[b][n]['buffers'] = buffers
                buffers = np.ndarray([0, buffer_size])

    except (KeyboardInterrupt, SystemExit):
        print('\nUser interrupt, quitting!')
        #  synthesizer
        muser.iodata.midi_all_notes_off(rtmidi_out, midi_basic=True)
        # close `rtmidi` and `jack` clients
        del rtmidi_out
        audio_client.deactivate()
        audio_client.close()
        raise

quit()
# FFT for each recorded buffer
# ffts = np.zeros_like(recordings)

# store audio results
#with open('recordings.json', 'w') as recordings_file:
#    json.dump(recordings, recordings_file, sort_keys=True, indent=4,
#              ensure_ascii=False)

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
