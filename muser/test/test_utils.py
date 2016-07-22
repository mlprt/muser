
import functools
import pytest
import numpy as np
import muser.utils as utils

rnd = np.random.RandomState()


def factors(n):
    """Return all factors of n."""
    candidates = ([i, n // i] for i in range(1, int(n**0.5) + 1) if n % i == 0)
    return set(functools.reduce(list.__add__, candidates))


def test_bytes_split():
    tests1 = 10
    # 1: bytes that can be equally divided
    piece_len_lims = (1, 100000)
    pieces_lims = (1, 1000)
    for _ in range(tests1):
        piece_len = rnd.randint(*piece_len_lims)
        n_pieces = rnd.randint(*pieces_lims)
        pieces = [rnd.bytes(piece_len) for piece in range(n_pieces)]
        assert pieces == utils.bytes_split(b''.join(pieces), piece_len)
    # 2: bytes that can't be equally divided
    prime_len_bytes = rnd.bytes(421)
    for i in range(2, 10):
        with pytest.raises(ValueError):
            utils.bytes_split(prime_len_bytes, i)
    # 3: bytes of length 0
    assert [b''] == utils.bytes_split(b'', 2)
