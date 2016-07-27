"""Experiment to train model to estimate velocity vector from average FFT.

Relies on the file ``params.json`` produced at the beginning of execution of
chord_batches.py. If a certain number of batches are expected based on the
``params.json``, but the corresponding dumps don't exist yet, this script
checks periodically until the next dump is available.
"""

import json
import os
import time
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import RMSprop
import numpy as np

import muser.fft as fft

data_dir = '/tmp/muser/chord_batches/160727-14h17'

params_file = os.path.join(data_dir, 'params.json')
with open(params_file, 'r') as params_file:
    params_json = params_file.read()
    globals().update(json.loads(params_json))

clfft = fft.get_cl_fft(axes=(1,))
vectors = np.ndarray([batch_size, 128], dtype=np.float32)
batch_ffts = np.ndarray([batch_size], dtype=object)
batch_rfft_avg = np.ndarray([batch_size, 2, 513], dtype=np.float32)
rfft_rel_thres = 0.01

# define model
model = Sequential()
model.add(Dense(320, input_dim=513, init='uniform', activation='relu'))
#model.add(Dense(256, init='uniform', activation='relu'))
model.add(Dense(128, init='uniform', activation='sigmoid'))
model.compile(loss='mean_squared_error', optimizer=RMSprop(), metrics=['accuracy'])
# save model architecture
open(os.path.join(data_dir, 'model.json'), 'w').write(model.to_json())

for i_batch in range(batches):
    while True:
        try:
            batch = np.load(paths['pickle'].format(i_batch))
            break
        except (FileNotFoundError, OSError):
            time.sleep(10)
    for i_chord, chord in enumerate(batch):
        vectors[i_chord, :] = chord['velocity_vector']
        captures = chord['captured_buffers'].astype(np.complex64)
        batch_ffts[i_chord] = fft.fft1d_collapse(captures, fft=clfft)
        # average RFFT per channel, excluding quiet buffers:
        chord_rffts = abs(batch_ffts[i_chord])[:, :, : blocksize // 2 + 1]
        avg_rffts = np.average(chord_rffts, axis=2)
        exclude = np.where(avg_rffts < np.max(avg_rffts) * rfft_rel_thres)
        chord_rfft_avg = np.mean(np.delete(chord_rffts, exclude, 1), 1)
        batch_rfft_avg[i_chord, :, :] = chord_rfft_avg
    # use channel average, for now
    train_data = np.mean(batch_rfft_avg, 1)
    model.fit(train_data, vectors, batch_size=batch_size,
              nb_epoch=300, verbose=1, validation_split=0.125)

model.save_weights(os.path.join(data_dir, 'model_weights.h5'))
