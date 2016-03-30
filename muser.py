"""
Machine learning music
"""

import numpy as np
from scipy.integrate import trapz
from scipy.io import wavfile
import jack
import tensorflow as tf
from pylab import plot, show 

class MuserJACK():
    """ Manages a JACK client for Muser """
    def __init__(self, midi_ins=1, midi_outs=1, name="MuserClient"):
        self.client = jack.Client(name)
        self.midi_in = [client.midi_inports.register("midi_in_{}".format(j)) for j in range(midi_ins)]
        self.midi_out = [client.midi_outports.register("midi_out_{}".format(k)) for k in range(midi_outs)]
        activate()
    
    def activate(self):
        client.activate()
    def deactivate(self):
        client.deactivate()
    def close():
        client.close()
        
def f(t):
    try:
        return [int(p * sf) for p in t]
    except TypeError:
        return int(t * sf)

def scale_wav(snd, factor=None):
    # scale signal from -1 to 1
    if factor:
        return snd/factor
    else:
        return scale_wav(snd, 2.**15 if snd.dtype == np.dtype('int16') else 2.**31) 
        
# x = snd.shape[0]             # number of sample points (duration * rate)
# t = np.arange(0, x, 1) / sf  # sample points as time points (seconds)

def local_fft(chs, f0, f1, norm="ortho"):
    # calculate FFTs for each channel over the interval
    return np.stack(np.fft.fft(ch[f0:f1], norm=norm) for ch in chs)

# import contents of wav file
sf, snd = wavfile.read('op28-20.wav')
snd = scale_wav(snd)

t0, t1 = (0.5, 1.0)
w = local_fft(snd.T, f(t0), f(t1))

n = (w.shape[1] // 2) + 1
w_a = abs(w)[:][0:n]
w_pwr = w_a ** 2 
frqs = np.fft.fftfreq(f(t1) - f(t0))
frqs_a = abs(sf * frqs)[0:n]   # Hz

frq_max = 5000 # Frequency cutoff
cut = np.where(frqs_a > frq_max)[0][0] 
plot(frqs_a[0:cut], w_pwr[0][0:cut])
show()

# print(sum(ws_p[1]))
