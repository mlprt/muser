""" Utility functions. """

import functools
import time
import numpy as np
import peakutils


def amp_to_decibels(amp):
    """Converts amplitude to decibel units."""
    amp_db = 10.0 * np.log10(np.absolute(amp) ** 2)
    return amp_db


def freq_to_hertz(samplerate):
    """Returns function that converts frequency per-sample to per-second."""
    def to_hertz(freq):
        return np.absolute(freq * samplerate)
    return to_hertz


def pitch_to_hertz(midi_pitch):
    """Converts MIDI note number to its specified audio frequency (Hz).

    Based on 440 Hz concert pitch corresponding to MIDI pitch number of 69,
    and the doubling of frequency with each octave (12 semitones or MIDI note
    numbers).

    Args:
        midi_pitch (int): MIDI note number.

    Returns:
        hz (float): Audio frequency specified by the MIDI note number.
    """
    pitch_hertz = 440 * (2 ** ((midi_pitch - 69) / 12))
    return pitch_hertz


def time_to_sample(time_, samplerate):
    """Return sample index closest to given time.

    Args:
        time (float): Time relative to the start of sample indexing.
        samplerate (int): Rate of sampling for the recording.

    Returns:
        sample (int): Index of the sample taken nearest to ``time``.
    """
    sample = int(time_ * samplerate)
    return sample


def sample_to_time(sample, samplerate):
    """Returns times corresponding to samples in a series."""
    return sample / float(samplerate)


def wait_while(toggle_attr):
    """Returns a decorator that waits for an instance attribute to be false."""
    def wait_while_decorator(instance_method):
        """Waits on a toggle attribute before executing an instance method."""
        @functools.wraps(instance_method)
        def wrapper(self, *args, **kwargs):
            while getattr(self, toggle_attr):
                pass
            return instance_method(self, *args, **kwargs)

        return wrapper
    return wait_while_decorator


def set_true(toggle_attr):
    """Returns a decorator that enables an attribute during execution."""
    def set_true_decorator(instance_method):
        """Enables a boolean attribute while the decorated method executes."""
        @functools.wraps(instance_method)
        def wrapper(self, *args, **kwargs):
            setattr(self, toggle_attr, True)
            output = instance_method(self, *args, **kwargs)
            setattr(self, toggle_attr, False)
            return output

        return wrapper
    return set_true_decorator


def if_true(toggle_attr):
    """Returns a decorator that executes its method if a condition is met."""
    def if_true_decorator(instance_method):
        """Execute the decorated method conditional on a toggle attribute."""
        @functools.wraps(instance_method)
        def wrapper(self, *args, **kwargs):
            if getattr(self, toggle_attr):
                return instance_method(self, *args, **kwargs)
        return wrapper
    return if_true_decorator


def log_with_timepoints(record_attr):
    """Returns a decorator that logs entry and return times."""
    def log_with_timepoints_decorator(instance_method):
        """Append return value and entry and return times to attribute."""
        @functools.wraps(instance_method)
        def wrapper(self, *args, **kwargs):
            entry_abs = time.time()
            entry_clock = time.perf_counter()
            output = instance_method(self, *args, **kwargs)
            exit_clock = time.perf_counter()
            log = {'output': output, 'entry_abs': entry_abs,
                   'entry_clock': entry_clock, 'exit_clock': exit_clock}
            getattr(self, record_attr).append(log)
            return output
        return wrapper
    return log_with_timepoints_decorator


def prepost_method(method_name, *method_args, **method_kwargs):
    """Decorators that call an instance method before and after another."""
    def prepost_method_decorator(instance_method):
        """Call an instance method before and after the decorated method."""
        @functools.wraps(instance_method)
        def wrapper(self, *args, **kwargs):
            getattr(self, method_name)(*method_args, **method_kwargs)
            output = instance_method(self, *args, **kwargs)
            getattr(self, method_name)(*method_args, **method_kwargs)
            return output
        return wrapper
    return prepost_method_decorator


def get_peaks(y, x, thres):
    """Return the peaks in data that exceed a relative threshold."""
    try:
        peaks_idx = [peakutils.indexes(ch, thres=thres) for ch in y]
        peaks = [(x[idx], y[i][idx]) for i, idx in enumerate(peaks_idx)]
    except TypeError:  # amp not iterable
        return get_peaks([y], x, thres=thres)

    return peaks


def nearest_pow(num, base, rule=round):
    """Given a base, return power nearest to num.

    Parameters:
        num (float):
        base (float):
        rule (function):
    """
    return int(rule(np.log10(num) / np.log10(base)))


def series(func, length, *args):
    return list(map(func, args * length))


def get_batches(get_member, batches, batch_size, member_args=(None,)):
    """Return batches of elements generated by ``get_member``.

    Parameters:
        get_member (function): Generates batch members
        batches (int): Number of batches to return
        batch_size (int): Number of members in each batch
        member_args (iterable): Parameters to pass to `get_member`

    Returns:
        batches (list): A series of batches from `get_member`
    """
    def batch(*args):
        return series(get_member, batch_size, *args)
    batches = series(batch, batches, *member_args)
    return batches
