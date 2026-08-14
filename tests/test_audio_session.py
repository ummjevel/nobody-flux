"""SharedStreamSession._process_block's reference alignment (code-review #5):
the reference handed to the echo canceller must be the played signal delayed
by delay_ms at *sample* granularity. _process_block is a plain method over
numpy arrays by design ("testable without opening a device"), so no stream is
ever started here.
"""

from __future__ import annotations

import numpy as np

from src.nobody_flux.audio.aec import EchoCanceller
from src.nobody_flux.audio.session import FRAME_SAMPLES, SAMPLE_RATE, SharedStreamSession


class RecordingCanceller(EchoCanceller):
    def __init__(self):
        self.refs: list[np.ndarray] = []

    def process(self, mic, ref):
        self.refs.append(ref.copy())
        return mic


def feed_ramp(session, canceller, n_frames: int) -> np.ndarray:
    """Play a strictly-increasing ramp through _process_block frame by frame;
    return the full played signal for comparison."""
    total = n_frames * FRAME_SAMPLES
    played = np.arange(1, total + 1, dtype=np.float32)  # 1-based: 0 means "not played yet"
    mic = np.zeros(FRAME_SAMPLES, dtype=np.float32)
    for i in range(n_frames):
        session._process_block(mic, played[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES])
    return played


def test_reference_is_sample_delayed_not_frame_delayed():
    delay_ms = 28.0  # the Windows box's measured value -- NOT a multiple of 30
    delay_samples = round(SAMPLE_RATE * delay_ms / 1000)  # 448
    canceller = RecordingCanceller()
    session = SharedStreamSession(echo_canceller=canceller, delay_ms=delay_ms)

    played = feed_ramp(session, canceller, n_frames=4)
    got = np.concatenate(canceller.refs)
    expected = np.concatenate(
        [np.zeros(delay_samples, dtype=np.float32), played[: len(got) - delay_samples]]
    )
    np.testing.assert_array_equal(got, expected)


def test_zero_delay_passes_the_current_frame():
    canceller = RecordingCanceller()
    session = SharedStreamSession(echo_canceller=canceller, delay_ms=0.0)
    played = feed_ramp(session, canceller, n_frames=2)
    np.testing.assert_array_equal(np.concatenate(canceller.refs), played)


def test_delay_frames_fallback_when_delay_ms_absent():
    canceller = RecordingCanceller()
    session = SharedStreamSession(echo_canceller=canceller, delay_frames=1)  # 480 samples
    played = feed_ramp(session, canceller, n_frames=3)
    got = np.concatenate(canceller.refs)
    expected = np.concatenate(
        [np.zeros(FRAME_SAMPLES, dtype=np.float32), played[: len(got) - FRAME_SAMPLES]]
    )
    np.testing.assert_array_equal(got, expected)


def test_stop_playback_keeps_inflight_reference():
    """The last played samples are still physically echoing for delay_ms after
    the speaker cuts; the tail must keep serving them as reference."""
    canceller = RecordingCanceller()
    session = SharedStreamSession(echo_canceller=canceller, delay_ms=28.0)
    played = feed_ramp(session, canceller, n_frames=2)
    session.stop_playback()

    silence = np.zeros(FRAME_SAMPLES, dtype=np.float32)
    session._process_block(silence, silence)
    # The frame right after the cut still references the tail of what played.
    post_cut_ref = canceller.refs[-1]
    delay_samples = round(SAMPLE_RATE * 28.0 / 1000)
    np.testing.assert_array_equal(post_cut_ref[:delay_samples], played[-delay_samples:])
    assert not post_cut_ref[delay_samples:].any()
