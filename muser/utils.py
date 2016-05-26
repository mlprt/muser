""" Utility functions. """

import numpy as np
import peakutils

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


def nearest_pow(num, base, rule=round):
    """ Given a base, return power nearest to num.

        Args:
            num (float):
            base (float):
            rule (function):

    """

    return int(rule(np.log10(num) / np.log10(base)))
