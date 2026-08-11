"""Sound-device I/O: capture, playback, and echo cancellation.

Everything in this package deals in samples. Nothing here knows what a turn is,
what was said, or which model produced the audio -- that separation is what
lets the turn controller be tested against a fake session, and lets a new
device backend be added without touching conversation logic.

``session``
    ``AudioSession`` and its backends. The important one is
    ``SharedStreamSession``: a single duplex stream that both captures and
    plays. One stream rather than two is what makes the reply available as an
    echo-cancellation reference, and what sidesteps the platform conflicts that
    arise from opening capture and playback independently.
``aec``
    Frame-level echo cancellers (``ReferenceGate``, ``SpeexEchoCanceller``)
    plugged into a session. Given (mic frame, reference frame), return a mic
    frame with the speaker's own output removed or suppressed.
``player``
    Playback of a reply's streamed chunks, interruptibly.
``resample``
    Sample-rate conversion between TTS output and the session rate.

The module-level constants are re-exported here because they are the shared
contract between this package and ``turn/vad.py``: both sides must agree on
16kHz mono in 30ms frames, and there should be exactly one place that says so.
Importing this package costs numpy and nothing else -- ``sounddevice`` is
loaded lazily, when a stream is actually opened, so tooling that only needs the
frame geometry does not have to have a working audio device.
"""

from __future__ import annotations

from .session import FRAME_MS, FRAME_SAMPLES, SAMPLE_RATE

__all__ = ["SAMPLE_RATE", "FRAME_MS", "FRAME_SAMPLES"]
