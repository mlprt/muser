"""
Machine learning music
"""

import sys
import numpy as np
from scipy.io import wavfile
import jack
import matplotlib
import matplotlib.pyplot
import pyopencl
import pyopencl.array
import gpyfft.fft
import peakutils

# Datatypes that SciPy can import from .wav
SND_DTYPES = {'int16': 16-1, 'int32': 32-1} 
# Set default font for matplotlib
matplotlib.rcParams['font.family'] = 'TeX Gyre Adventor'


def jack_client(midi_ins=1, midi_outs=1, name="MuserClient"):
    """Returns an active JACK client with specified number of inputs and outputs
    """
    client = jack.Client(name)
    ports_in, ports_out = [], []
    for j in range(midi_ins):
        ports_in[j] = client.midi_inports.register("midi_in_{}".format(j))
    for k in range(midi_outs):
        ports_out['out'][j] = client.midi_outports.register("midi_out_{}".format(k))
    client.activate()

    return client, ports


def load_wav(wav_name, scale=True):
    """ """
    sample_frq, snd = wavfile.read(wav_name)
    if scale: snd = scale_snd(snd)

    return sample_frq, snd


def scale_snd(snd, factor=None):
    """ Scale signal from -1 to 1.
    """
    if factor is None:
        return scale_snd(snd, 2.**SND_DTYPES[snd.dtype.name])

    return snd / factor


def get_to_frames(sample_frq):
    """Return function that converts time (is s if sf is Hz) to sample index
       for a defined sampling rate.
    """
    def to_frames(time):
        def to_f(time):
            try:
                return int(time * sample_frq)
            except TypeError: # time not specified or specified badly
                e = "Real local endpoints must be specified! (t_endp)"
                raise TypeError(e)
        try:
            return [to_f(t) for t in time]
        except TypeError: # time not iterable
            return [to_f(time)]
        
    return to_frames


def local_rfft(snd, f_start, length, units='', norm='ortho', rfft_=None):
    """ Calculate FFTs for each channel over the interval
    """
    if rfft_ is None: rfft_ = get_np_rfft()
    loc = slice(f_start, f_start + length)
    local = [ch[loc] for ch in snd]

    rfft = np.stack(rfft_(ch) for ch in local)  
    frq = np.fft.fftfreq(length)[0:rfft.shape[1]]
    
    if units == '':
        amp = rfft
    elif units == 'dB':
        amp = 10. * np.log10(abs(rfft) ** 2.)
    elif units == 'sqr':
        amp = abs(rfft) ** 2.
        
    return amp, frq


def get_np_rfft(norm=None):
    def np_rfft(data):
        return np.fft.rfft(data, norm=norm)

    return np_rfft


def get_cl_rfft(length):
    """ Prepares gpyfft for 1D arrays of known length """
    context = pyopencl.create_some_context(interactive=False)
    queue = pyopencl.CommandQueue(context)
    
    def cl_rfft(data):
        """ """
        data = np.array(data, dtype=np.complex64)
        data_c = pyopencl.array.to_device(queue, data)
        transform = gpyfft.fft.FFT(context, queue, (data_c,))
        events = transform.enqueue()
        for e in events:
            e.wait()
            
        return data_c.get()[:length/2]
        
    return cl_rfft


def get_peaks(amp, frq, thres):
    """ """
    try:
        peaks_idx = [peakutils.indexes(ch, thres=thres) for ch in amp]
        peaks = [(frq[idx], amp[i][idx]) for i, idx in enumerate(peaks_idx)]
    except TypeError: # amp not iterable
        return get_peaks([amp], frq, thres=thres)
        
    return peaks


def get_axes(title="", xlabel="", ylabel="", facecolor='w', bgcolor='w'):
    """ """
    fig = matplotlib.pyplot.figure()
    axes = fig.add_subplot(111)
    axes.patch.set_facecolor(facecolor)
    axes.set_axis_bgcolor(bgcolor)
    axes.set_title(title)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    
    return fig, axes


def plot_fft(frq, amp, sample_freq, title='', labels=['Hz', 'dB'], peaks=None,
             save=False, scale=None, margin=1.1):
    """ Create a plot and plot FFT data, and peak marker points if included """

    fig, axes = get_axes(title, *labels)
    axes.plot(frq, amp, 'k-')
    if peaks:
        axes.plot(peaks[0], peaks[1] + 1, 'r.')
        axes.axis([0, max(peaks[0]) * margin, 0, max(peaks[1]) * margin])
    if save:
        fig.savefig(save)
    
    return axes


def main(wav_name="op28-20", t_start=0.5, length=None):
    # import contents of wav file
    if length is None: length = 2**15
    
    sample_frq, snd = load_wav('{}.wav'.format(wav_name))
    to_frames = get_to_frames(sample_frq)
    f_start = to_frames(t_start)[0]
    
    rffts = [get_cl_rfft(length), get_np_rfft()]
    amp, frq = local_rfft(snd.T, f_start, length, units='dB', rfft_=rffts[1])
    frq = abs(sample_frq * frq)   # Hz

    thres = 0.01  # threshold for peak detection, fraction of max
    peaks = get_peaks(amp, frq, thres)
    
    # plot sound intensity (dB) versus frequency (Hz)
    for i, ch in enumerate(amp):
        title = '{}.wav, t = {} + {} samples @ {} {}'
        title = title.format(wav_name, t_start, length, sample_frq, 'Hz')
        file_name = 'fft_{}_t{}_ch{}.png'.format(wav_name, t_start, i)
        plot_fft(frq, ch, sample_frq, title, peaks=peaks[i], save=file_name)
        

if __name__ == '__main__':
    #t_endp = (float(sys.argv[1]), float(sys.argv[2]))
    main()
