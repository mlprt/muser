"""Fourier analysis of sampled audio."""

import numpy as np
import pyopencl as cl
import pyopencl.array as cla
import gpyfft.fft as gfft


def get_cl_fft(axes=(1,)):
    """Return clFFT transformer with expected transform axes.

    Args:
      axes (tuple): Axes to be transformed. Defaults to 1D transform on axis 1.

    Example:
      With ``axes=(2, 1,)``, performs a 2D transform on axes 1 and 2, with
      batch members along axis 0.
    """
    context = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(context)

    def cl_fft(data):
        """Return FFT of the data ."""
        data_gpu = cla.to_device(queue, data)
        transform = gfft.FFT(context, queue, (data_gpu,), axes=axes)
        event, = transform.enqueue()
        event.wait()
        fft = data_gpu.get()
        return fft

    return cl_fft


def fft1d_collapse(data, fft=np.fft.fft):
    """Return FFTs over the last axis after collapsing the other axes.

    Intended for batch processing by clFFT, which does not accept
    more than 1 non-transformed dimension.

    Args:
      data (np.ndarray):
      fft (function):

    Returns:
      np.ndarray:
    """
    dims = data.shape
    data_batch1D = data.reshape(np.prod(dims[:-1]), dims[-1])
    return fft(data_batch1D).reshape(*dims)
