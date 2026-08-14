"""_AudioRing: the pre-roll supplier. This exact class already paid for one
silent-corruption bug (see vad.py's push() comment about segment handles), so
its invariant -- absolute sample position p lives at p % capacity -- gets
tested directly, including the seams.
"""

from __future__ import annotations

import numpy as np

from src.nobody_flux.turn.vad import _AudioRing


def _arange(start: int, stop: int) -> np.ndarray:
    return np.arange(start, stop, dtype=np.float32)


def test_read_within_capacity_roundtrips():
    ring = _AudioRing(10)
    ring.append(_arange(0, 6))
    np.testing.assert_array_equal(ring.read(0, 6), _arange(0, 6))
    np.testing.assert_array_equal(ring.read(2, 5), _arange(2, 5))


def test_wraparound_seam_preserves_order():
    ring = _AudioRing(10)
    # 25 samples through a 10-slot ring, in odd-sized chunks that force writes
    # to split across the seam.
    ring.append(_arange(0, 7))
    ring.append(_arange(7, 16))
    ring.append(_arange(16, 25))
    # Only the last 10 absolute positions survive.
    np.testing.assert_array_equal(ring.read(15, 25), _arange(15, 25))


def test_read_clamps_to_what_survives():
    ring = _AudioRing(10)
    ring.append(_arange(0, 6))
    ring.append(_arange(6, 12))
    # Asking further back than capacity degrades to the oldest data available,
    # by design (a clipped pre-roll beats a failed turn).
    np.testing.assert_array_equal(ring.read(0, 12), _arange(2, 12))


def test_read_beyond_written_clamps_forward():
    ring = _AudioRing(10)
    ring.append(_arange(0, 4))
    np.testing.assert_array_equal(ring.read(0, 100), _arange(0, 4))


def test_empty_range_returns_empty():
    ring = _AudioRing(10)
    ring.append(_arange(0, 4))
    assert ring.read(3, 3).size == 0
    assert ring.read(5, 2).size == 0


def test_oversized_frame_keeps_tail_phase_correct():
    """Regression for the invariant break flagged in code-review-20260814 #14:
    an over-capacity frame used to be written flat at index 0, which is only
    correct when (written + n) % capacity == 0 -- any other alignment made
    read() return size-correct but shuffled audio, silently."""
    ring = _AudioRing(10)
    ring.append(_arange(0, 3))  # misalign: written=3, so 3+25 % 10 != 0
    ring.append(_arange(3, 28))  # single frame larger than the whole ring
    np.testing.assert_array_equal(ring.read(18, 28), _arange(18, 28))
    # And a subsequent normal append still lands in phase.
    ring.append(_arange(28, 31))
    np.testing.assert_array_equal(ring.read(21, 31), _arange(21, 31))


def test_read_returns_copy_not_view():
    ring = _AudioRing(10)
    ring.append(_arange(0, 8))
    snapshot = ring.read(0, 8)
    ring.append(np.full(8, -1.0, dtype=np.float32))  # overwrite those slots
    np.testing.assert_array_equal(snapshot, _arange(0, 8))


def test_reset_restarts_absolute_origin():
    ring = _AudioRing(10)
    ring.append(_arange(0, 8))
    ring.reset()
    assert ring.written == 0
    ring.append(_arange(100, 104))
    np.testing.assert_array_equal(ring.read(0, 4), _arange(100, 104))
