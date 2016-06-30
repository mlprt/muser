""" Example FFT of `.wav` with clFFT, and visualization. """

import os
import numpy as np
import muser.iodata as iodata
import muser.fft as fft
import muser.utils as utils
import muser.vis as vis

name = "op28-20"
audio_dir = "/home/mll/dev/muser/examples/"
wavfile_name = audio_dir + name + '.wav'
t_start = 0.5
length = 2**15
length_sqrt = np.sqrt(length)

sample_rate, snd = iodata.wav_read_norm(wavfile_name)
i_start = iodata.to_sample_index(t_start, sample_rate)

rfft = fft.get_cl_rfft(length)
amp, freq = fft.local_rfft(snd.T, i_start, length, units='dB',
                           rfft=rfft, scale=lambda _: length_sqrt)
freq = abs(sample_rate * freq)   # Hz

thres_rel = 0.01  # threshold for peak detection, fraction of max
peaks = utils.get_peaks(amp, freq, thres_rel)

# plot sound amplitude versus frequency
file_name = 'fft_{}_t{:.3f}-{:.3f}_ch{:d}.png'
title = '{}.wav, channel {:d}, t[s] = {:.3f} $-$ {:.3f}'
for i, ch in enumerate(amp):
    t_end = t_start + length / float(sample_rate)
    title = title.format(name, i, t_start, t_end)
    file_name_ = file_name.format(name, t_start, t_end, i)
    vis.plot_fft(freq, ch, title, peaks=peaks[i], save=file_name_)
