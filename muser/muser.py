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
import moviepy.editor
from moviepy.video.io.bindings import mplfig_to_npimage

# Datatypes that SciPy can import from .wav
SND_DTYPES = {'int16': 16-1, 'int32': 32-1}
# Set default font for matplotlib
matplotlib.rcParams['font.family'] = 'MathJax_Main'
# Latin Modern Math, Latin Modern Mono, Inconsolata, Linux Libertine Mono O


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


# def jack_thru(client, midi_in, midi_out):
#    client.connect()


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
            except TypeError:  # time not specified or specified badly
                e = "Real local endpoints must be specified! (t_endp)"
                raise TypeError(e)
        try:
            return [to_f(t) for t in time]
        except TypeError:  # time not iterable
            return to_f(time)

    return to_frames


def local_rfft(snd, f_start, length, units='', rfft=None, scale=None):
    """ Calculate FFTs for each channel over the interval
    """
    if rfft is None:
        rfft = get_np_rfft()
    loc = slice(f_start, f_start + length)
    local = [ch[loc] for ch in snd]
    amp = np.stack(rfft(ch) for ch in local)
    if scale:
        amp = amp / scale(amp)
    if units == '':
        amp = amp
    elif units == 'dB':
        amp = 10. * np.log10(abs(amp) ** 2.)
    elif units == 'sqr':
        amp = abs(amp) ** 2.
    frq = np.fft.fftfreq(length)[0:amp.shape[1]]

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
        # TODO: could convert to numpy (constant size) if assign peaks
        #       to harmonic object containing all notes
        #       (could also be used for training)
        peaks = [(frq[idx], amp[i][idx]) for i, idx in enumerate(peaks_idx)]
    except TypeError:  # amp not iterable
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


def plot_fft(frq, amp, title='', labels=['Hz', 'dB'], peaks=None,
             save=False, scale=True, margin=1.1):
    """ Create a plot and plot FFT data, and peak marker points if included """

    fig, axes = get_axes(title, *labels)
    axes.plot(frq, amp, 'k-')
    if any(peaks[0]):
        axes.plot(peaks[0], peaks[1] + 1, 'r.')
        # TODO: Generalize for many FFTs (plotting in same video)
        # amp_mean = np.mean(peaks[1])
        # amp_std = np.std(peaks[1])
        # cut = 2 * amp_std + amp_mean
        # frq_cut = peaks[0][0:np.amax(np.where(peaks[1] > cut))]
    else:
        maxes = max(frq), max(amp)
    if scale:
        # axes.axis([0, maxes[0] * margin, 0, maxes[1] * margin])
        axes.axis([0, 4000, 0, 10])
        # Highest note on 88-key piano is C8, 4186.01 Hz
    if save:
        fig.savefig(save)

    return fig, axes


def nearest_pow(num, base, rule=round):
    """ Calculate the power of base nearest to num
        Rounds to the nearest power by default
    """
    return int(rule(np.log10(num) / np.log10(base)))


def setup(wav_name, scale=True):
    sample_frq, snd = wavfile.read(wav_name)
    to_frames = get_to_frames(sample_frq)
    if scale:
        snd = scale_snd(snd)

    return sample_frq, snd.T, to_frames


def get_make_frame(snd, chan, sample_frq, to_frames, rfft=None, rfft_len=None):
    if rfft is None:
        if rfft_len is None:
            res = 2  # 2**res == ~number of fft bins per second
            rfft_len = 2**(nearest_pow(sample_frq, 2) - res)
        rfft = get_cl_rfft(rfft_len)
    elif rfft_len is None:
        # raise TypeError("rfft_len must be provided with rfft")
        res = 2  # 2**res == ~number of fft bins per second
        rfft_len = 2**(nearest_pow(sample_frq, 2) - res)

    # TODO: Change to single figure that is updated each time?
    def make_frame(t):
        amp, frq = local_rfft([snd[chan]], to_frames(t), rfft_len, 'sqr', rfft,
                              lambda x: np.sqrt(rfft_len))
        frq = abs(sample_frq * frq)  # Hz
        thres = 0.01  # threshold for peak detection, fraction of max
        peaks = get_peaks(amp, frq, thres)
        fig, _ = plot_fft(frq, amp[0], title='', peaks=peaks[0])
        frame = mplfig_to_npimage(fig)
        matplotlib.pyplot.close()

        return frame

    return make_frame


def fft_movie(wav_name, chan=0, fps=4, rfft=None, rfft_len=None):
    sample_frq, snd, to_frames = setup(wav_name + '.wav')
    snd_dur = len(snd[chan]) / float(sample_frq)
    make_frame = get_make_frame(snd, chan, sample_frq, to_frames=to_frames,
                                rfft=rfft, rfft_len=rfft_len)
    animation = moviepy.editor.VideoClip(make_frame, duration=snd_dur)
    animation.write_videofile("test.mp4", fps=fps)


def main(wav_name="op28-20", t_start=0.5, length=None):
    # import contents of wav file
    if length is None:
        length = 2**15

    sample_frq, snd, to_frames = setup(wav_name + '.wav')
    f_start = to_frames(t_start)
    rffts = [get_cl_rfft(length), get_np_rfft()]
    rfft_tmp = rffts[0]
    amp, frq = local_rfft(snd.T, f_start, length, units='dB',
                          rfft_=rfft_tmp, scale=lambda x: np.sqrt(length))
    frq = abs(sample_frq * frq)   # Hz

    thres = 0.01  # threshold for peak detection, fraction of max
    peaks = get_peaks(amp, frq, thres)

    # plot sound amplitude versus frequency
    file_name = 'fft_{}_t{:.3f}-{:.3f}_ch{:d}.png'
    title = '{}.wav, channel {:d}, t[s] = {:.3f} $-$ {:.3f}'
    for i, ch in enumerate(amp):
        t_end = t_start + length / float(sample_frq)
        title = title.format(wav_name, i, t_start, t_end)
        file_name_ = file_name.format(wav_name, t_start, t_end, i)
        plot_fft(frq, ch, title, peaks=peaks[i], save=file_name_)


if __name__ == '__main__':
    n_argv = len(sys.argv)
    if n_argv == 2:
        main(t_start=float(sys.argv[1]))
    else:
        main()
