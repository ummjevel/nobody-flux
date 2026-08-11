"""Sample-rate conversion for the playback path.

Extracted from ``session.py`` so the duplex-stream backends are not the only
possible consumer: the reply players and any future device backend need the
same conversion, and none of them should have to import a session class to get
at it.

Scope note -- this is intentionally *not* a general-purpose resampler. It
handles exactly one job: fitting TTS output (22.05kHz for the Matcha presets,
48kHz for MOSS-TTS-Nano) into the 16kHz duplex stream the microphone side
requires. Both are downward conversions of already-synthesized speech destined
for a small speaker, so the aliasing a naive interpolator introduces above the
new Nyquist limit is inaudible in context. A polyphase/windowed-sinc resampler
would be more correct and would cost a dependency (``scipy``/``soxr``) plus
per-chunk latency in the streaming playback path, for quality nobody in this
pipeline can hear.
"""

from __future__ import annotations

import numpy as np


def resample_to(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Convert mono audio from ``sr_in`` to ``sr_out`` by linear interpolation.

    Accepts multi-channel input (downmixed by averaging) and any dtype numpy
    can cast to float32. Always returns a contiguous mono float32 array, which
    is what both the ``sounddevice`` callback and the echo cancellers require.

    The identity case (``sr_in == sr_out``) and the empty case are handled
    without touching the interpolation path -- this runs once per synthesized
    chunk during streaming playback, and the common configuration has the TTS
    already at the session rate.
    """
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = np.asarray(x, dtype=np.float32)

    if sr_in == sr_out or len(x) == 0:
        # ascontiguousarray rather than a bare return: callers hand the result
        # straight to native code (PortAudio, SpeexDSP), which requires a
        # contiguous buffer, and a slice of a larger array would not be one.
        return np.ascontiguousarray(x, dtype=np.float32)

    n_out = int(round(len(x) * sr_out / sr_in))
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)

    # Both grids are built with endpoint=False so sample *positions* line up
    # proportionally regardless of length -- using endpoint=True would stretch
    # the output by one sample interval, which accumulates into audible drift
    # when many short chunks are concatenated back to back, exactly what the
    # streaming playback path does.
    src = np.linspace(0.0, 1.0, len(x), endpoint=False)
    dst = np.linspace(0.0, 1.0, n_out, endpoint=False)
    return np.interp(dst, src, x).astype(np.float32)
