""" Visualizations. """

import numpy as np
import matplotlib
import matplotlib.pyplot
import moviepy.editor
from moviepy.video.io.bindings import mplfig_to_npimage


matplotlib.rcParams['font.family'] = 'MathJax_Main'
"""str: Name of default font used by matplotlib"""


def plot_fft(frq, amp, title='', labels=['Hz', 'dB'], peaks=None,
             save=False, scale=True, margin=1.1):
    """ Plot FFT data, and peak markers if included.

    Parameters:
        frq ():
        amp ():
        title (str):
        labels (list[str]):
        peaks ():
        save (bool):
        scale (bool):
        margin (float):

    Returns:
        fig ():
        axes ():
    """

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


def get_axes(title="", xlabel="", ylabel="", facecolor='w', bgcolor='w'):
    """

    Parameters:
        title (str):
        xlabel (str):
        ylabel (str):
        facecolor ():
        bgcolor ():

    Returns:
        fig ():
        axes ():
    """
    fig = matplotlib.pyplot.figure()
    axes = fig.add_subplot(111)
    axes.patch.set_facecolor(facecolor)
    axes.set_axis_bgcolor(bgcolor)
    axes.set_title(title)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)

    return fig, axes


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
    sample_frq, snd = wavfile.read(wav_name + '.wav')
    snd = scale_snd(snd)
    to_frames = get_to_frames(sample_frq)
    snd_dur = len(snd[chan]) / float(sample_frq)
    make_frame = get_make_frame(snd, chan, sample_frq, to_frames=to_frames,
                                rfft=rfft, rfft_len=rfft_len)
    animation = moviepy.editor.VideoClip(make_frame, duration=snd_dur)
    animation.write_videofile("test.mp4", fps=fps)
