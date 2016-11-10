"""Audio file I/O and audio data manipulation."""

import numpy as np
import scipy.io.wavfile

SND_DTYPES = {'int16': 16, np.int16: 16, 'int32': 32, np.int32: 32}
"""Data types that SciPy can import from ``.wav``"""


def snd_norm(snd, factor=None):
    """ Scale elements of an array of (.wav) data from -1 to 1.

    Default factor is determined from ``snd.dtype``, corresponding to the
    format imported by ``scipy.io.wavfile``. Can scale other types of data if
    ``factor`` is appropriately specified and ``snd`` can be scaled
    element-wise with the division operator, as for ``np.ndarray``.

    Args:
        snd (np.ndarray): Data (audio from .wav) to be scaled.
        factor (int): Divide elements of ``snd`` by this number.

    Returns:
        scaled (np.ndarray): Same shape as ``snd``, with elements scaled.
    """
    if factor is None:
        factor = 2. ** (SND_DTYPES[snd.dtype.name] - 1)
    scaled = snd / factor
    return scaled


def wav_read_norm(wavfile_name):
    """ Return contents of .wav as array scaled from -1 to 1.

    Args:
        wavfile_name (str): Name of the .wav file to read.

    Returns:
        sample_rate (int): The file's audio sampling rate.
        snd (np.ndarray): The file's audio samples as ``float``.
    """
    sample_rate, snd = scipy.io.wavfile.read(wavfile_name)
    snd = snd_norm(snd)
    return sample_rate, snd


def buffers_to_snd(buffers, stereo=True, channel_ind=None, dtype=np.int32):
    """ Convert a series of JACK buffers to 2-channel SciPy audio.

    Args:
        buffers (np.ndarray): Series of JACK buffers in a 3D array. Second
            dimension length is number of channels, third is ``buffer_size``.
        stereo (bool): If ``True``, the two channels of ``snd`` are taken by
            default from ``buffers[0:2]``, else both from ``buffers[0]`` (mono).
        channel_ind: If stereo, can be a length-2 ``slice`` or a Numpy advanced
            index selecting two channels in ``buffers``. If mono, an integer,
            slice, or Numpy advanced index for a single channel must be passed.
        dtype (str): Datatype of the returned array.
            Must be a key in ``SND_DTYPES`` to ensure SciPy compatibility.

    Returns:
        snd (np.ndarray): SciPy-compatible array of audio frames.
    """
    if stereo:
        if channel_ind is None:
            channel_ind = slice(0, 2)
        buffers_ = buffers[channel_ind]
    else:
        if channel_ind is None:
            channel_ind = 0
        buffers_ = np.concatenate(np.atleast_2d(buffers[channel_ind]) * 2)
    snd = buffers_.reshape((2, buffers_.size // 2)).T
    snd = snd * 2.**(SND_DTYPES[dtype] - 1)
    snd = snd.astype(dtype)
    return snd
