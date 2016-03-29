"""
Machine learning music
"""

import numpy as np
from scipy.integrate import trapz
from scipy.io import wavfile
from pylab import plot, show 

# import contents of wav file
sf, snd = wavfile.read('op28-20.wav')

# scale signal from -1 to 1
snd = snd/(2.**15) if snd.dtype == np.dtype('int16') else snd/(2.**31)

# x = snd.shape[0]             # number of sample points (duration * rate)
# t = np.arange(0, x, 1) / sf  # sample points as time points (seconds)

# time interval containing the frames for FFT
i = int(0.5 * sf)
f = int(1.0 * sf)

# calculate FFTs for each channel over the interval
ws = np.stack(np.fft.fft(ch[i:f], norm="ortho") for ch in snd.T)
ws_a = abs(ws)
ws_p = ws_a**2
fr_a = abs(np.fft.fftfreq(f-i) * sf)   # Hz

fr_max = 5000 # Frequency cutoff
crop = [(fr, ws_p[0][i]) for i, fr in enumerate(fr_a) if fr < fr_max]
plot(*zip(*crop))
show()

print(sum(ws_p[0]))
