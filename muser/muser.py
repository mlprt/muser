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
                raise TypeError("Real local endpoints must be specified! (t_endp)")
        try:
            return [to_f(t) for t in time]
        except TypeError: # time not iterable
            return [to_f(time)]
        
    return to_frames


def local_rfft(snd, f_start, length, units='', norm='ortho', cl_fft=None):
    """ Calculate FFTs for each channel over the interval
    """
    loc = slice(f_start, f_start + length)
    np_rfft = get_np_rfft(norm)

    local = [ch[loc].reshape((length, 1)) for ch in snd]
    _fft = np_rfft if cl_fft == None else cl_fft
    print local[0].shape
    rfft = np.stack(_fft(ch) for ch in local)

    print rfft
        
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


def get_cl_fft_1d(length):
    context = pyopencl.create_some_context()
    queue = pyopencl.CommandQueue(context)
    #GFFT = gpyfft.GpyFFT(debug=True)
    #plan = GFFT.create_plan(context, (length, ))
    #plan.bake(queue)
    
    def cl_fft_1d(data):
        data = np.array(data, dtype=np.complex64)
        dataC = pyopencl.array.to_device(queue, data)
        #dataF = pyopencl.array.to_device(queue, np.asfortranarray(data))
        result = np.zeros_like(data)
        resultC = pyopencl.array.to_device(queue, result)
        #resultF = pyopencl.array.to_device(queue, np.asfortranarray(result))
        transform = gpyfft.fft.FFT(context, queue, (dataC,))
        events = transform.enqueue()
        #events = plan.enqueue_transform((queue,), (dataC,), (resultC,))
        for e in events:
            e.wait()
            
        return dataC
        
    return cl_fft_1d


def get_peaks(amp, frq, thres):
    """ """
    try:
        iter(amp)
        peaks_idx = [peakutils.indexes(ch, thres=thres) for ch in amp]
        peaks = [(frq[idx], amp[i][idx]) for i, idx in enumerate(peaks_idx)]
    except TypeError:
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


def main(wav_name="op28-20.wav", t_start=0.5, length=8):
    # t = np.arange(0, x, 1) / sample_frq  # sample points as time points (seconds)
    # import contents of wav file

    sample_frq, snd = load_wav(wav_name)
    to_frames = get_to_frames(sample_frq)
    f_start = to_frames(t_start)[0]
    length = 2 ** length
    
    cl_fft = get_cl_fft_1d(length)
    amp, frq = local_rfft(snd.T, f_start, length, units='dB')
    amp, frq = local_rfft(snd.T, f_start, length, units='dB', cl_fft=cl_fft)
    frq = abs(sample_frq * frq)   # Hz

    thres = 0.01  # threshold for peak detection, fraction of max
    peaks = get_peaks(amp, frq, thres)
    maxes = zip(*[(max(peak[0]), max(peak[1])) for peak in peaks])
    
    # plot sound intensity (dB) versus frequency (Hz)
    axes_title = '{}, t = {} + {} samples @ {} Hz'
    axes_title = axes_title.format(wav_name, t_start, length, )
    fig, axes = get_axes(axes_title, 'Hz', 'dB')
    margin = 1.1
    for i, ch in enumerate(amp):
        clr = CLRS[-i]
        
        axes.plot(frq, ch, "{}-".format(clr))
        axes.plot(peaks[i][0], peaks[i][1] + i, "{}.".format(clr))

    axes.axis([0, max(maxes[0]) * margin, 0, max(maxes[1]) * margin])
    matplotlib.pyplot.show()


if __name__ == '__main__':
    t_endp = (float(sys.argv[1]), float(sys.argv[2]))
    main()
