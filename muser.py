"""
Machine learning music
"""

import sys
import numpy as np
import tensorflow as tf
from scipy.io import wavfile
import jack
import matplotlib
import matplotlib.pyplot
import peakutils

# Datatypes that SciPy can import from .wav
SND_DTYPES = {'int16': 16-1, 'int32': 32-1} 
# Colours for cycling in plots
CLRS = 'bgrcmyk'
# Set default font for matplotlib
matplotlib.rcParams['font.family'] = 'TeX Gyre Adventor'

def jack_client(midi_ins=1, midi_outs=1, name="MuserClient"):
    """Returns an active JACK client with specified number of inputs and outputs
    """
    client = jack.Client(name)
    for j in range(midi_ins):
        client.midi_inports.register("midi_in_{}".format(j))
    for k in range(midi_outs):
        client.midi_outports.register("midi_out_{}".format(k))
    client.activate()

    return client


def get_to_frames(sample_frq):
    """Return function that converts time (is s if sf is Hz) to sample index
       for a defined sampling rate.
    """
    def to_frames(time):
        def to_f(time):
            try:
                return int(time * sample_frq)
            except TypeError: # time not specified or specified badly
                raise TypeError("Real local endpoints must be specified! (t_endp)")
        try:
            return [to_f(t) for t in time]
        except TypeError: # time not iterable
            return [to_f(time)]
        
    return to_frames


def scale_snd(snd, factor=None):
    """ Scale signal from -1 to 1.
    """
    if factor is None:
        return scale_snd(snd, 2.**SND_DTYPES[snd.dtype.name])

    return snd / factor


def local_rfft(chs, f_endp, units='', norm='ortho'):
    """ Calculate FFTs for each channel over the interval
    """
    rfft = np.stack(np.fft.rfft(ch[slice(*f_endp)], norm=norm) for ch in chs)
    frq = np.fft.fftfreq(f_endp[1] - f_endp[0])[0:rfft.shape[1]]
    
    if units == '':
        amp = rfft
    elif units == 'dB':
        amp = 10. * np.log10(abs(rfft) ** 2.)
    elif units == 'sqr':
        amp = abs(rfft) ** 2.
        
    return amp, frq


def get_peaks(amp, frq, thres):
    """ """
    try:
        iter(amp)
        peaks_idx = [peakutils.indexes(ch, thres=thres) for ch in amp]
        peaks = [(frq[idx], amp[i][idx]) for i, idx in enumerate(peaks_idx)]
    except TypeError:
        return get_peaks([amp], frq, thres=thres)
        
    return peaks


def get_plot(title="", xlabel="", ylabel="", facecolor='w', bgcolor='w'):
    """ """
    fig = matplotlib.pyplot.figure()
    axes = fig.add_subplot(111)
    axes.patch.set_facecolor(facecolor)
    axes.set_axis_bgcolor(bgcolor)
    axes.set_title(title)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    
    return fig, axes


def load_wav(wav_name, scale=True):
    sample_frq, snd = wavfile.read(wav_name)
    if scale: snd = scale_snd(snd)

    return sample_frq, snd


def main(wav_name="op28-20.wav", t_endp=(None, None)):
    # t = np.arange(0, x, 1) / sample_frq  # sample points as time points (seconds)
    # import contents of wav file

    sample_frq, snd = load_wav(wav_name)
    to_frames = get_to_frames(sample_frq)
    f_endp = to_frames(t_endp)
    f_endp[1] += sum(f_endp) % 2  # total number of frames should be even

    amp, frq = local_rfft(snd.T, f_endp, units='dB')
    frq = abs(sample_frq * frq)   # Hz

    thres = 0.01  # peak detection threshold as factor of amp_max
    peaks = get_peaks(amp, frq, thres)
    maxes = zip(*[(max(peak[0]), max(peak[1])) for peak in peaks])
    
    # axes sound intensity (dB) versus frequency (Hz)
    axes_title = '{}, $\Delta$t = {}'.format(wav_name, t_endp)
    fig, axes = get_plot(axes_title, 'Hz', 'dB')
    marg = 1.1
    for i, ch in enumerate(amp):
        axes.plot(frq, ch, "{}-".format(CLRS[-i]))
        axes.plot(peaks[i][0], peaks[i][1] + i, "{}.".format(CLRS[-i]))

    axes.axis([0, max(maxes[0]) * marg, 0, max(maxes[1]) * marg])
    matplotlib.pyplot.show()

    return peaks

if __name__ == '__main__':
    t_endp = (float(sys.argv[1]), float(sys.argv[2]))
    main(t_endp=t_endp)
