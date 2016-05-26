""" Fourier analysis. """

import numpy as np
import pyopencl
import pyopencl.array
import gpyfft.fft


def local_rfft(snd, f_start, length, units='', rfft=None, scale=None):
    """ Return tuple of FFT amplitudes and frequencies for each channel in snd.
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
