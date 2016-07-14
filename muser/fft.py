"""Fourier analysis of sampled audio."""

import numpy as np
import muser.utils
import pyopencl
import pyopencl.array
import gpyfft.fft


def get_cl_rfft(samples):
    """Return OpenCL FFT function for 1D arrays of known length."""
    context = pyopencl.create_some_context(interactive=False)
    queue = pyopencl.CommandQueue(context)

    def cl_rfft(data):
        """Calculate and return FFT amplitudes for the given data."""
        data = np.array(data, dtype=np.complex64)
        data_c = pyopencl.array.to_device(queue, data)
        transform = gpyfft.fft.FFT(context, queue, (data_c,))
        cl_events = transform.enqueue()
        for event in cl_events:
            event.wait()
        fft = data_c.get()[: samples // 2]
        return fft

    return cl_rfft


def snd_rfft(snd, rfft=None, scale=None, amp_convert=None, freq_convert=None):
    """Return FFT amplitudes and frequencies for each channel in ``snd``.

    TODO: Make this even more modular. Scaling (utils?). Unit conversion.

    Args:
        snd (np.ndarray): Vectors (channels) of audio amplitude samples.
        rfft (function): Returns FFT amplitudes per vector of audio amplitudes.
        amp_scale (function):
        amp_convert (function):

    Returns:
        amp (np.ndarray): FFT amplitudes.
        freq (np.ndarray): FFT frequencies.
    """
    samples = snd.shape[1]
    if rfft is None:
        rfft = np.fft.rfft(data, norm=None)
    if scale is None:
        scale = lambda _: np.sqrt(samples)
    amp = np.apply_along_axis(rfft, 1, snd)
    amp = amp / scale(amp)
    if amp_convert is not None:
        amp = amp_convert(amp)
    freq = np.fft.fftfreq(samples)[: samples // 2]
    if freq_convert is not None:
        freq = freq_convert(freq)
    return amp, freq
