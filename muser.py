"""
Machine learning music
"""

import numpy as np
from scipy.io import wavfile
import jack
import peakutils
import tensorflow as tf
import matplotlib
import matplotlib.pyplot

# Datatypes that SciPy can import from .wav
SND_DTYPES = {'int16': 16-1, 'int32': 32-1}
# Colours for cycling in plots
CLRS = 'bgrcmyk'

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
    else:
        return snd / factor


def local_rfft(chs, f_endp, norm="ortho"):
    """ Calculate FFTs for each channel over the interval
    """
    # TODO: Move f0 and f1 out of this function?
    rfft = np.stack(np.fft.rfft(ch[slice(*f_endp)], norm=norm) for ch in chs)
    return rfft


def get_peaks(amp, frq, thres):
    """ """
    peak_i = peakutils.indexes(amp, thres=thres)
    return (frq[peak_i], amp[peak_i])


def get_plot(title="", xlabel="", ylabel="", facecolor='w', bgcolor='w'):
    """ """
    matplotlib.rcParams['font.family'] = 'cmr10'

    plot = matplotlib.pyplot.subplot()
    plot.patch.set_facecolor(facecolor)
    plot.set_axis_bgcolor(bgcolor)
    plot.set_title(title)
    plot.set_xlabel(xlabel)
    plot.set_ylabel(ylabel)

    return plot


def main(wav_name="op28-20.wav", t_endp=(None, None)):

    # t = np.arange(0, x, 1) / sample_frq  # sample points as time points (seconds)

    # import contents of wav file
    sample_frq, snd = wavfile.read(wav_name)
    snd = scale_snd(snd)
    to_frames = get_to_frames(sample_frq)
    f_endp = to_frames(t_endp)
    # total number of frames should be even
    f_endp[1] += sum(f_endp) % 2

    amp = local_rfft(snd.T, f_endp)
    amp_pwr = 10. * np.log10(abs(amp) ** 2) # dB
    amp_out = amp_pwr
    frq = np.fft.fftfreq(f_endp[1] - f_endp[0])
    frq_a = abs(sample_frq * frq)[0:amp.shape[1]]   # Hz

    thres = 0.01  # peak detection threshold as factor of amp_max
    peaks = [get_peaks(amp_j, frq_a, thres) for amp_j in amp_out]
    amp_max = [max(amp_j) for amp_j in amp_out]
    # plot sound intensity (dB) versus frequency (Hz)

    plot_title = u'{}, \u0394t = {}'.format(wav_name, t_endp)
    plot = get_plot(plot_title, 'Hz', 'dB')
    frq_max = 0
    marg = 1.1
    for i, ch in enumerate(amp_out):
        pyplot.plot(frq_a, ch, "{}-".format(CLRS[-i]))
        pyplot.plot(peaks[i][0], peaks[i][1] + i, "{}.".format(CLRS[-i]))
        frq_max_i = max(peaks[i][0])
        if frq_max_i > frq_max:
            frq_max = frq_max_i

    plot.axis([0, frq_max * marg, 0, max(amp_max) * marg])
    pyplot.show()
