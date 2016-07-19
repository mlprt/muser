""" Example FFT of `.wav` with clFFT, and visualization. """

import os
import numpy as np
import muser.iodata as iodata
import muser.fft as fft
import muser.utils as utils
import muser.vis as vis

wav_filename = "op28-20.wav"
audio_dir = "/home/mll/dev/muser/examples/"
wav_filepath = os.path.join(audio_dir, wav_filename)
time_i = 0.5
fft_samples = 2**15

samplerate, snd = iodata.wav_read_norm(wav_filepath)
sample_i = utils.time_to_sample(time_i, samplerate)
sample_f = sample_i + fft_samples
time_f = utils.sample_to_time(sample_f, samplerate)

rfft = fft.get_cl_rfft(fft_samples)
snd_local = snd[sample_i:sample_f]
amp, freq = fft.snd_rfft(snd_local.T, rfft=rfft, amp_convert=utils.amp_to_dB,
                         freq_convert=utils.freq_to_hertz(samplerate))

thres_rel = 0.01  # threshold for peak detection, fraction of max
peaks = utils.get_peaks(amp, freq, thres_rel)

# plot sound amplitude versus frequency
filename = 'fft_{}_t{:.3f}-{:.3f}_ch{:d}.png'
title = '{}.wav, channel {:d}, t[s] = {:.3f} $-$ {:.3f}'
for i, ch in enumerate(amp):
    title = title.format(wav_filename, i, time_i, time_f)
    filename_ = filename.format(os.path.splitext(wav_filename)[0],
                                time_i, time_f, i)
    vis.plot_fft(freq, ch, title, peaks=peaks[i], save=filename_)
