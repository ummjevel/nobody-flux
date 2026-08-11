"""Playback of a streamed reply, in chunks, interruptibly.

A reply is not one clip. ``pipeline.run_streaming`` emits it as a sequence of
sentence-sized ``AudioChunk``s so playback of the first can begin while the LLM
is still generating the rest. Something has to hold that sequence, play it back
to back without audible seams, and be able to abandon the whole remainder the
instant a barge-in is confirmed. That is what this module is.

Two implementations, one interface (``ReplyPlayer``):

``StreamPlayer``
    Owns a private output stream. Used when there is no duplex audio session --
    the legacy path, and still the right one when capture and playback are on
    separate devices and echo is not a concern.

``SessionPlayer``
    Owns nothing; hands samples to an ``audio.session.AudioSession`` that is
    already running one duplex stream for both directions. Playback then
    doubles as the echo canceller's reference signal, and no second stream is
    opened -- which is what avoids the macOS err -50 conflict.

Both were previously defined inside ``scripts/talk.py``. They are here because
they are audio-device machinery, not conversation logic, and because the turn
controller (``turn/controller.py``) now needs them too -- leaving them in a
script would have meant importing a CLI entry point from library code.

Why StreamPlayer holds one persistent stream
--------------------------------------------

The straightforward implementation is ``sounddevice.play(chunk); sd.wait()``
per chunk. It has two costs that matter here, and this class exists to avoid
both:

*Seams.* ``sd.play`` opens a stream, plays, and closes it, for every chunk.
Between chunks the device is torn down and re-established, which inserts a
short silence at every sentence boundary -- precisely where a listener is most
likely to read it as the speaker having finished.

*Interruption latency.* ``sd.wait()`` takes no timeout, so bounding a stalled
backend requires spawning a watchdog thread per chunk (which the previous
implementation did). Writing block by block into a persistent stream gives the
same bound for free, and tightens the stop response to a single block period
(~30ms) rather than however long the current clip had left to play.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Protocol

import numpy as np

from .resample import resample_to

# Samples per write into the output stream. Bounds two things at once: how long
# a confirmed barge-in waits before playback actually stops (the write loop
# only checks for a stop request between blocks), and how much slack the device
# has before it underruns. 1024 frames is ~21ms at the 48kHz a TTS chunk may
# arrive at and ~64ms at 16kHz -- comfortably below the 250ms barge-in
# confirmation window at either rate, so stopping still reads as immediate,
# while staying large enough that the per-write overhead is negligible.
BLOCK_FRAMES = 1024

# Added to a chunk's own duration when deciding that a write has wedged. Only
# reachable if the audio backend stops consuming samples entirely; the normal
# path never comes near it.
PLAYBACK_TIMEOUT_MARGIN_S = 5.0


class ReplyPlayer(Protocol):
    """What the turn loop needs from a player, and nothing more.

    The lifecycle is one instance per reply: ``start()``, then ``enqueue()``
    per synthesized chunk, then ``done()`` once no more are coming. ``stop()``
    may arrive at any point from another thread.
    """

    def start(self) -> None:
        """Begin accepting chunks for a new reply."""

    def enqueue(self, samples: np.ndarray, sample_rate: int) -> None:
        """Add one synthesized chunk to the end of this reply."""

    def done(self) -> None:
        """Signal that no further chunks will be enqueued for this reply."""

    def stop(self) -> None:
        """Confirmed barge-in: cut what is sounding and discard what is queued."""

    def stop_requested(self) -> bool:
        """True once ``stop()`` has been called.

        The *producer* polls this to stop synthesizing the rest of an
        interrupted reply -- there is no point spending TTS time on audio that
        will never be played.
        """

    def is_active(self) -> bool:
        """True while a reply is sounding or still has chunks pending.

        This is how a barge-in is told apart from an ordinary turn start: the
        user speaking while this is False is simply taking their turn, not
        interrupting.
        """

    def join(self, timeout: float | None = None) -> None:
        """Block until this reply finishes playing, is stopped, or times out."""


class StreamPlayer:
    """Plays a reply's chunks through one persistent output stream.

    ``active`` is set from the moment ``start()`` is called -- before any chunk
    has been enqueued -- so that speech arriving during the gap between "reply
    began" and "first audio synthesized" is still classified as an
    interruption rather than a fresh turn.

    Sample rate: the stream is opened at the first chunk's rate and stays
    there. Later chunks at a different rate are resampled to match rather than
    reopening the device, since reopening would reintroduce exactly the seam
    this class exists to remove. In practice every chunk of a reply comes from
    the same TTS preset at the same rate, so the resample path is a safety net,
    not the common case.
    """

    def __init__(self) -> None:
        self._queue: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._active = threading.Event()
        self._thread: threading.Thread | None = None
        self._stream = None
        self._stream_rate: int | None = None

    # -- producer side ----------------------------------------------------

    def start(self) -> None:
        self._stop.clear()
        self._active.set()
        self._thread = threading.Thread(
            target=self._run, name="reply-playback", daemon=True
        )
        self._thread.start()

    def enqueue(self, samples: np.ndarray, sample_rate: int) -> None:
        self._queue.put((samples, sample_rate))

    def done(self) -> None:
        self._queue.put(None)  # sentinel: end of this reply

    def stop(self) -> None:
        self._stop.set()
        # Abort rather than stop: abort discards whatever the device has
        # already buffered, so the tail that was handed to the driver but not
        # yet sounded is dropped too. stop() would let it play out, which is
        # audible as the reply continuing for a moment after the interruption.
        stream = self._stream
        if stream is not None:
            try:
                stream.abort()
            except Exception:
                # The stream may already be closing on the worker thread. A
                # failure here means playback is ending anyway, which is what
                # was wanted -- there is nothing useful to recover.
                pass

    def stop_requested(self) -> bool:
        return self._stop.is_set()

    def is_active(self) -> bool:
        return self._active.is_set()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    # -- consumer side (worker thread) ------------------------------------

    def _run(self) -> None:
        try:
            while True:
                item = self._queue.get()
                if item is None:
                    break
                if self._stop.is_set():
                    # Post-barge-in: keep draining so the producer never blocks
                    # on a full queue, but play nothing.
                    continue
                self._play(*item)
        finally:
            self._close_stream()
            self._active.clear()

    def _ensure_stream(self, sample_rate: int):
        """Open the output stream on first use, at the first chunk's rate."""
        if self._stream is not None:
            return self._stream
        import sounddevice as sd

        self._stream_rate = sample_rate
        self._stream = sd.OutputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocksize=BLOCK_FRAMES,
        )
        self._stream.start()
        return self._stream

    def _close_stream(self) -> None:
        stream, self._stream = self._stream, None
        self._stream_rate = None
        if stream is None:
            return
        try:
            stream.stop()
            stream.close()
        except Exception:
            pass  # already aborted by stop(); nothing left to release

    def _play(self, samples: np.ndarray, sample_rate: int) -> None:
        """Write one chunk to the device, block by block, checking for a stop
        request between blocks."""
        stream = self._ensure_stream(sample_rate)
        if self._stream_rate is not None and sample_rate != self._stream_rate:
            samples = resample_to(samples, sample_rate, self._stream_rate)

        # float32 mono, contiguous: PortAudio writes the buffer directly, so a
        # non-contiguous view or a wrong dtype would mean a silent copy per
        # block at best and garbage audio at worst.
        samples = np.ascontiguousarray(samples, dtype=np.float32)

        # Wall-clock bound on this chunk: its own duration plus a margin. Only
        # reachable if the backend stops draining, since a healthy stream
        # returns from write() at roughly real time.
        expires_at = time.monotonic() + len(samples) / max(sample_rate, 1) + PLAYBACK_TIMEOUT_MARGIN_S

        for offset in range(0, len(samples), BLOCK_FRAMES):
            if self._stop.is_set():
                return
            if time.monotonic() > expires_at:
                # The device has stopped consuming samples. Abandoning this
                # chunk is strictly better than blocking the reply forever;
                # the next chunk gets a fresh budget.
                return
            try:
                stream.write(samples[offset : offset + BLOCK_FRAMES])
            except Exception:
                # Raised when the stream was aborted from stop() mid-write.
                # That is a normal barge-in, not an error worth propagating
                # into the turn loop.
                return


class SessionPlayer:
    """Routes a reply's chunks into an already-running duplex ``AudioSession``.

    Deliberately owns no thread and no stream. The session's own callback
    drains the playback buffer, which is also what supplies the echo
    canceller's reference -- so playing through here (rather than through a
    second, private stream) is what makes the reply cancellable out of the
    microphone signal.

    ``done()`` is a no-op for the same reason: there is no worker to signal,
    because the session drains itself.
    """

    def __init__(self, session) -> None:
        self._session = session
        self._stop = threading.Event()

    def start(self) -> None:
        self._stop.clear()

    def enqueue(self, samples: np.ndarray, sample_rate: int) -> None:
        self._session.play(samples, sample_rate)

    def done(self) -> None:
        pass

    def stop(self) -> None:
        self._stop.set()
        self._session.stop_playback()

    def stop_requested(self) -> bool:
        return self._stop.is_set()

    def is_active(self) -> bool:
        return self._session.playback_active()

    def join(self, timeout: float | None = None) -> None:
        """Poll until the session's playback buffer drains, it is stopped, or
        ``timeout`` elapses.

        Polling rather than waiting on a condition because the session exposes
        buffer state, not an event -- and at a 20ms interval the imprecision is
        well under one audio block, so nothing is gained by making the session
        signal it.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        while self._session.playback_active() and not self._stop.is_set():
            if deadline is not None and time.monotonic() > deadline:
                return
            time.sleep(0.02)
