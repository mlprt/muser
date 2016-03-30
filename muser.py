"""
Machine learning music
"""

import numpy as np
import tensorflow as tf
import jack
from scipy.io import wavfile
import peakutils
from matplotlib import pyplot

# Datatypes that SciPy can import from .wav
snd_dtypes = {'int16': 16-1, 'int32': 32-1}
wav_name = 'op28-20.wav'

def jack_client(midi_ins=1, midi_outs=1, name="MuserClient"):
    """ Returns an active JACK client with specified number of inputs and outputs """
    client = jack.Client(name)
    for j in range(midi_ins):
        client.midi_inports.register("midi_in_{}".format(j))
    for k in range(midi_outs):
        client.midi_outports.register("midi_out_{}".format(k))
    client.activate()
    return client


def get_to_f(sf):
    """ Return function that converts time (is s if sf is Hz) to sample index
        for a defined sampling rate.
    """
    def to_f(t):
        try:
            return [int(p * sf) for p in t]
        except TypeError:
            return int(t * sf)

    return to_f

    
def scale_snd(snd, factor=None):
    """ Scale signal from -1 to 1 
    """
    if factor is None:
        return scale_snd(snd, 2.**snd_dtypes[snd.dtype.name])
    else:
        return snd / factor
        

def local_fft(chs, f0, f1, norm="ortho"):
    """ Calculate FFTs for each channel over the interval 
    """
    return np.stack(np.fft.rfft(ch[f0:f1], norm=norm) for ch in chs)

def get_peaks(amp, frq, thres):
    peak_i = peakutils.indexes(amp, thres=thres)
    return (frq[peak_i], amp[peak_i])
    
def main():

    # x = snd.shape[0]             # number of sample points (duration * rate)
    # t = np.arange(0, x, 1) / sf  # sample points as time points (seconds)
    
    # import contents of wav file
    sf, snd = wavfile.read(wav_name)
    snd = scale_snd(snd)
    
    to_f = get_to_f(sf)
    t_endp = (0.5, 1.0)
    f_endp = to_f(t_endp)
    # total number of frames should be even
    f_endp[1] += sum(f_endp) % 2

    amp = local_fft(snd.T, *f_endp)
    amp_sq = abs(amp) ** 2
    amp_pwr = 10. * np.log10(amp_sq) # dB
    amp_out = amp_pwr
    frq = np.fft.fftfreq(f_endp[1] - f_endp[0])
    frq_a = abs(sf * frq)[0:amp.shape[1]]   # Hz

    thres = 0.01  # peak detection threshold as factor of w_max
    peaks = [get_peaks(amp_j, frq_a, thres) for amp_j in amp_out]
    amp_max = [max(amp_j) for amp_j in amp_out]
    # plot sound intensity (dB) versus frequency (Hz)

    frq_max = 0
    clr = 'bgrcmyk'
    for i, ch in enumerate(amp_out):
        pyplot.plot(frq_a, ch, "{}-".format(clr[-i]))
        pyplot.plot(peaks[i][0], peaks[i][1] + i, "{}.".format(clr[-i]))
        frq_max_i = max(peaks[i][0])
        if frq_max_i > frq_max: frq_max = frq_max_i
    
    pyplot.axis([0, frq_max * 1.1, 0, max(amp_max) * 1.1])
    pyplot.xlabel('Hz')
    pyplot.ylabel('dB')
    pyplot.show()

    return peaks
