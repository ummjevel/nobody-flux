"""resample_to: the single conversion boundary between TTS output rates and
the 16kHz session. The no-drift invariant matters most -- streaming playback
concatenates many short chunks, and one sample of per-chunk stretch becomes
audible drift."""

from __future__ import annotations

import numpy as np

from src.nobody_flux.audio.resample import resample_to


def test_output_length_formula():
    x = np.zeros(22_050, dtype=np.float32)
    assert len(resample_to(x, 22_050, 16_000)) == round(22_050 * 16_000 / 22_050)
    x = np.zeros(1_000, dtype=np.float32)
    assert len(resample_to(x, 48_000, 16_000)) == round(1_000 / 3)


def test_identity_rate_passes_through_contiguous():
    x = np.linspace(-1, 1, 480, dtype=np.float32)
    out = resample_to(x, 16_000, 16_000)
    np.testing.assert_array_equal(out, x)
    assert out.flags["C_CONTIGUOUS"]
    assert out.dtype == np.float32


def test_empty_input_returns_empty():
    assert resample_to(np.zeros(0, dtype=np.float32), 48_000, 16_000).size == 0


def test_tiny_input_rounding_to_zero_returns_empty():
    assert resample_to(np.zeros(1, dtype=np.float32), 48_000, 16_000).size == 0


def test_multichannel_downmixes_by_mean():
    stereo = np.stack(
        [np.full(100, 0.5, dtype=np.float32), np.full(100, -0.5, dtype=np.float32)], axis=1
    )
    out = resample_to(stereo, 16_000, 16_000)
    np.testing.assert_allclose(out, np.zeros(100, dtype=np.float32))


def test_chunked_conversion_has_no_length_drift():
    """Sum of per-chunk output lengths must equal the whole-signal output
    length -- the endpoint=False grid guarantees it for chunk sizes that
    divide the ratio evenly, which the 30ms frame path always is."""
    rng = np.random.default_rng(0)
    signal = rng.standard_normal(48_000 * 2).astype(np.float32)  # 2s at 48kHz

    whole = resample_to(signal, 48_000, 16_000)
    chunk = 1_440  # 30ms at 48kHz -> exactly 480 out per chunk
    pieces = [
        resample_to(signal[i : i + chunk], 48_000, 16_000)
        for i in range(0, len(signal), chunk)
    ]
    assert sum(len(p) for p in pieces) == len(whole)
    assert all(len(p) == 480 for p in pieces[:-1])  # the tail chunk is shorter


def test_constant_signal_resamples_to_constant():
    x = np.full(2_205, 0.25, dtype=np.float32)
    out = resample_to(x, 22_050, 16_000)
    np.testing.assert_allclose(out, np.full(len(out), 0.25, dtype=np.float32), rtol=1e-6)
