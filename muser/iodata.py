"""Input and output of audio and music sequences.

Low-latency audio connectivity is provided by the JACK Audio Connection Kit.
"""

import sys
import numpy as np
import jack
from scipy.io import wavfile

SND_DTYPES = {'int16': 16, 'int32': 32}
"""dict: Data types that SciPy can import from .wav"""


def get_jack_client(midi_ins=1, midi_outs=1, name="MuserClient"):
    """Returns an active JACK client with MIDI inputs and outputs.

    Args:
        midi_ins (int): Number of MIDI in ports to register. Defaults to 1.
        midi_outs (int): Number of MIDI out ports to register. Defaults to 1.
        name (str): Name of the returned client. Defaults to "MuserClient".
    """
    client = jack.Client(name)
    for j in range(midi_ins):
        client.midi_inports.register("midi_in_{}".format(j))
    for k in range(midi_outs):
        client.midi_outports.register("midi_out_{}".format(k))
    client.activate()

    return client


def get_to_sample_index(sample_frq):
    """Return function that converts time to sample index for given sample rate.

    Args:
        sample_frq (int):
    """
    def to_sample_index(time):
        """ Return sample index """
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


def unit_snd(snd, factor=None):
    """ Scale elements of an array of wav data from -1 to 1.

    Works as a gener

    Args:
        snd (np.ndarray): i
        factor (int):
    """
    if factor is None:
        factor = 2.**(SND_DTYPES[snd.dtype.name] - 1)

    return snd / factor
