""""""

import json
import time
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import RMSprop
import numpy as np

import muser.fft as fft

params_file = '/tmp/muser/chord_batches/160727-10h27/params.json'

with open(params_file, 'r') as params_file:
    params = json.loads(params_file.read())

globals().update(params)

clfft = fft.get_cl_fft(axes=(1,))
batch_ffts = np.ndarray([batch_size], dtype=object)

# define model

for i_batch in range(batches):
    while True:
        try:
            batch = np.load(paths['pickle'].format(i_batch))
            break
        except FileNotFoundError:
            time.sleep(5)
    for i_chord, chord in enumerate(batch):
        captures = chord['captured_buffers'].astype(np.complex64)
        batch_ffts[i_chord] = fft.fft1d_collapse(captures, fft=clfft)
