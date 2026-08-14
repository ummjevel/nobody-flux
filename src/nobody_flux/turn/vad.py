"""Voice activity detection: where one spoken turn begins and ends.

TEN-VAD, reached through sherpa-onnx's built-in support -- the same runtime
already loaded for ASR and TTS, so it costs no new dependency. It replaced this
project's original hand-rolled RMS-energy threshold: a trained model instead of
two fixed numbers that needed re-tuning per room and per microphone. (The
energy version is still in git history if that trade is ever worth revisiting;
it needed no ONNX file and no per-frame inference.)

Two ways to drive it
--------------------

``VadStream`` is the primitive: push one frame, get back zero or more events.
It owns no thread and no device, so *the caller* decides where frames come from
and what else happens between them. This is what ``turn/controller.py``
requires -- a controller that must keep capturing during reply generation
cannot afford a VAD that blocks until an utterance completes.

``VoiceActivityDetector.listen_for_utterance`` is the convenience wrapper:
block until one utterance has been captured, then return it. Simpler when
blocking is genuinely what is wanted (the diagnostic and calibration scripts),
and implemented entirely on top of ``VadStream`` so the two cannot diverge.

The previous version offered only the blocking form, with the frame loop, the
event callbacks, the pre-roll buffering and the endpoint-continuation logic all
interleaved inside one function. Separating the state machine from the loop
that feeds it is what made Phase 4's continuous capture possible.

Known limits, stated rather than papered over
---------------------------------------------

TEN-VAD's own thresholds are defaults tuned on the TEN Framework's evaluation
set. They are a better starting point than a hand-picked energy cutoff, not a
guarantee for any particular microphone -- ``configs/vad.yaml``'s ``threshold``
is already overridden to 0.25 for exactly this reason, measured with
``scripts/_debug_vad_mic.py``.

``barge_in_confirm_ms`` and the endpoint grace parameters distinguish a real
interruption from a backchannel ("어", "응"). See ``docs/barge-in-design.md``
for why a plain "any detected speech ends the reply" rule is wrong for this
project's casual persona.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterator

import numpy as np
import sherpa_onnx

if TYPE_CHECKING:
    from .detector import TurnDetector

from ..paths import PROJECT_ROOT

SAMPLE_RATE = 16_000
FRAME_MS = 30
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)  # 480

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "ten-vad" / "ten-vad.onnx"


@dataclass
class Utterance:
    """One captured spoken turn, ready for recognition."""

    audio: np.ndarray  # mono float32 at `sample_rate`
    sample_rate: int = SAMPLE_RATE
    # Samples the VAD attributed to speech itself, excluding the pre-roll
    # padding prepended to `audio` (and, for a multi-segment turn, counting
    # only the segments). None when the capture path did not measure it.
    speech_samples: int | None = None

    @property
    def duration_s(self) -> float:
        """Length of the full capture buffer, pre-roll included."""
        return len(self.audio) / self.sample_rate

    @property
    def speech_duration_s(self) -> float:
        """Length of the speech itself, as the VAD segmented it.

        This is the value duration-based judgments must use. ``duration_s``
        includes ``pre_roll_ms`` of padding, so with pre_roll_ms=500 every
        capture reports at least ~0.65s -- which silently disabled the 0.6s
        backchannel gate when pre-roll grew from 300 to 500 (afc0df8).
        """
        if self.speech_samples is None:
            return self.duration_s
        return self.speech_samples / self.sample_rate


class VadEvent(Enum):
    """What a pushed frame caused.

    ``SPEECH_STARTED``
        The VAD's internal "in speech" flag just flipped on. Fires early and
        cheaply -- useful for showing the user that the microphone is live, but
        far too eager to act on, since backchannel trips it just as readily as
        a real interruption.
    ``BARGE_IN_CONFIRMED``
        Speech has now continued for ``barge_in_confirm_ms``. This is stage 1 of
        ``docs/barge-in-design.md``: long enough that most backchannel has
        already ended without ever reaching here. At most once per turn.
    ``UTTERANCE_READY``
        A complete utterance is available from ``VadStream.take_utterance()``.
    """

    SPEECH_STARTED = "speech_started"
    BARGE_IN_CONFIRMED = "barge_in_confirmed"
    UTTERANCE_READY = "utterance_ready"


class _AudioRing:
    """Fixed-capacity ring buffer over the frames fed to the VAD, addressed by
    absolute sample position.

    Exists to supply the pre-roll (see ``pre_roll_ms``). The previous
    implementation kept every frame since the last reset in a Python list and
    called ``np.concatenate`` over all of them each time a segment finalized --
    rebuilding hundreds of small arrays into a new multi-megabyte one, once per
    utterance, purely to slice a fraction of it back out.

    Writing into a preallocated array instead makes the per-frame cost a single
    memcpy and the per-utterance cost one allocation sized to the utterance.
    Absolute positions are what make it usable: sherpa-onnx reports
    ``segment.start`` as a sample index counted from the last VAD reset, so the
    ring tracks the same origin and translates on read.
    """

    def __init__(self, capacity_samples: int) -> None:
        self._buffer = np.zeros(capacity_samples, dtype=np.float32)
        self._capacity = capacity_samples
        self._written = 0  # absolute count of samples ever written since reset

    def reset(self) -> None:
        self._written = 0

    @property
    def written(self) -> int:
        return self._written

    def append(self, frame: np.ndarray) -> None:
        n = len(frame)
        if n >= self._capacity:
            # A single frame larger than the whole ring: keep only its tail,
            # which is all the ring could have held anyway. The tail must be
            # written where the ring's invariant puts it -- absolute position p
            # lives at p % capacity -- not flat at index 0, or read() would
            # return size-correct but phase-shuffled audio with no error.
            # (Unreachable with today's 30ms frames vs a ~20.5s ring, but this
            # class exists to prevent silent audio corruption; see read().)
            tail = frame[-self._capacity :]
            self._written += n
            start = self._written % self._capacity
            split = self._capacity - start
            self._buffer[start:] = tail[:split]
            self._buffer[:start] = tail[split:]
            return
        start = self._written % self._capacity
        end = start + n
        if end <= self._capacity:
            self._buffer[start:end] = frame
        else:
            # Wraps past the end: split the copy across the seam.
            split = self._capacity - start
            self._buffer[start:] = frame[:split]
            self._buffer[: end - self._capacity] = frame[split:]
        self._written += n

    def read(self, start_abs: int, end_abs: int) -> np.ndarray:
        """Samples in the absolute range ``[start_abs, end_abs)``.

        Clamped to what the ring still holds, so a request reaching further
        back than capacity returns the oldest audio still available rather than
        raising. That degradation is deliberate: losing part of the pre-roll
        makes the first syllable slightly clipped, which is far better than
        failing the turn outright.
        """
        oldest = max(0, self._written - self._capacity)
        start_abs = max(start_abs, oldest)
        end_abs = min(end_abs, self._written)
        if end_abs <= start_abs:
            return np.zeros(0, dtype=np.float32)

        start = start_abs % self._capacity
        length = end_abs - start_abs
        if start + length <= self._capacity:
            # Copy, not a view: the caller keeps this past the point where the
            # ring will overwrite the region.
            return self._buffer[start : start + length].copy()
        split = self._capacity - start
        return np.concatenate([self._buffer[start:], self._buffer[: length - split]])


@dataclass
class VoiceActivityDetector:
    """Configuration for TEN-VAD, and a factory for the streams that run it.

    This object is cheap to hold and holds no per-utterance state -- that lives
    in ``VadStream``. One detector can therefore be built once from
    ``configs/vad.yaml`` (see ``registry.build_vad``) and used to open a stream
    per turn, or one long-lived stream for a whole session.
    """

    model_path: Path = DEFAULT_MODEL_PATH

    # -- TEN-VAD's own parameters -----------------------------------------
    # Listed explicitly rather than left to sherpa-onnx's defaults, so they are
    # overridable like every other field and so behaviour cannot change
    # silently if sherpa-onnx ever changes ITS defaults.
    threshold: float = 0.5
    min_silence_duration: float = 0.5  # trailing silence that ends a segment
    # sherpa-onnx defaults this to 0.25s; lowered because this project's persona
    # (see persona.py) invites one-syllable replies, and a quick Korean "네" can
    # run under 250ms. At the stock default those vanish as noise -- the user
    # speaks and nothing happens at all, with no error and no log line.
    min_speech_duration: float = 0.15
    max_speech_duration: float = 20.0  # hard cap; a stuck-open mic cannot hang a turn
    num_threads: int = 1

    # -- Capture shaping ---------------------------------------------------
    # The VAD needs a few frames of evidence before it is confident speech has
    # begun, so `segment.start` lands slightly after the true onset -- without
    # compensation the first word was reliably clipped (confirmed by hand).
    # Padding the returned audio backward by this much recovers it.
    pre_roll_ms: int = 300

    # -- Barge-in (docs/barge-in-design.md, stage 1) -----------------------
    # How long speech must continue past SPEECH_STARTED before it counts as a
    # real interruption. 250ms rather than this project's earlier 400ms guess,
    # recalibrated against LiveKit's published production figure (216ms median
    # to decide, audio only).
    #
    # Deliberately distinct from min_speech_duration above, which answers a
    # different question: that one decides whether a sound is speech at all,
    # this one decides whether speech already known to be real is long enough
    # to be an interruption rather than a backchannel.
    barge_in_confirm_ms: int = 250

    # -- Endpoint grace (Phase 2, only with a turn_detector) ---------------
    # After Smart Turn judges a finished segment *incomplete* -- the user paused
    # mid-thought rather than finishing -- this is the longest we keep waiting
    # for them to resume. Bounds the wait so a wrong "incomplete" verdict cannot
    # hang a turn when the user really was done.
    endpoint_grace_ms: int = 800
    # Lower bound of the same wait. A barely-incomplete verdict (P(complete)
    # just under threshold) waits about this long instead of the full budget, so
    # a real end-of-turn the detector slightly under-scored is not held up.
    endpoint_grace_min_ms: int = 300
    # False pins the wait at endpoint_grace_ms regardless of P(complete) -- the
    # pre-Phase-2 behaviour.
    adaptive_endpoint_grace: bool = True

    _vad: object = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        ten_vad_config = sherpa_onnx.TenVadModelConfig(
            model=str(self.model_path),
            threshold=self.threshold,
            min_silence_duration=self.min_silence_duration,
            min_speech_duration=self.min_speech_duration,
            max_speech_duration=self.max_speech_duration,
        )
        vad_config = sherpa_onnx.VadModelConfig(
            ten_vad=ten_vad_config, sample_rate=SAMPLE_RATE, num_threads=self.num_threads
        )
        # buffer_size_in_seconds is sherpa-onnx's own internal buffer for audio
        # it is still deciding about. 100s is comfortably above
        # max_speech_duration, so it never drops samples mid-utterance.
        self._vad = sherpa_onnx.VoiceActivityDetector(vad_config, buffer_size_in_seconds=100)

    def grace_frames_for_prob(self, prob: float) -> int:
        """How many frames of continued silence to wait for the user to resume,
        given Smart Turn's P(complete) for the segment just judged incomplete.

        Lower probability means a clearer mid-thought pause, so wait longer, up
        to ``endpoint_grace_ms``; a borderline verdict scales down toward
        ``endpoint_grace_min_ms``. Linear in ``1 - prob`` and clamped at both
        ends. This is the adaptive endpointing pattern LiveKit and Kyutai both
        describe (see ``docs/voice-agent-oss-survey.md``).
        """
        if not self.adaptive_endpoint_grace:
            grace_ms = float(self.endpoint_grace_ms)
        else:
            span = max(0, self.endpoint_grace_ms - self.endpoint_grace_min_ms)
            clamped = min(1.0, max(0.0, prob))
            grace_ms = self.endpoint_grace_min_ms + span * (1.0 - clamped)
        return max(1, int(grace_ms / FRAME_MS))

    def open_stream(self, turn_detector: "TurnDetector | None" = None) -> "VadStream":
        """Begin a frame-driven capture session. See ``VadStream``."""
        return VadStream(self, turn_detector=turn_detector)

    # -- blocking convenience ---------------------------------------------

    def listen_for_utterance(
        self,
        on_speech_start: Callable[[], None] | None = None,
        on_barge_in_confirmed: Callable[[], None] | None = None,
        turn_detector: "TurnDetector | None" = None,
        frame_source: Callable[[], np.ndarray] | None = None,
    ) -> Utterance:
        """Block until one spoken turn has been captured, then return it.

        A thin loop over ``VadStream``: pull a frame, push it, dispatch the
        resulting events to the callbacks, stop when an utterance is ready.
        Kept because the diagnostic and calibration scripts genuinely want a
        blocking call, and because it is the smaller change for any caller that
        does not need to do anything while listening.

        ``frame_source``
            Returns the next mono float32 frame. When given, frames come from
            it and no device is opened here -- that is how a duplex
            ``AudioSession`` supplies already-echo-cancelled audio. When None,
            this opens and owns a private input stream.
        ``on_speech_start`` / ``on_barge_in_confirmed``
            Called on the corresponding ``VadEvent``. Without the first, this
            function is silent for however long the user speaks, which from
            outside is indistinguishable from a dead microphone.
        ``turn_detector``
            Optional Smart Turn endpoint detection: a segment TEN-VAD finalized
            on silence is not returned immediately but checked for completeness,
            and a mid-thought pause is waited out and concatenated rather than
            being cut into two turns.
        """
        stream = self.open_stream(turn_detector=turn_detector)

        # Own a device only when nobody else is supplying frames. nullcontext
        # keeps the `with` shape identical in both cases.
        if frame_source is None:
            # Imported here rather than at module scope: opening a device is the
            # only thing in this module that needs PortAudio, and the controller
            # path never reaches it. Keeping it local means importing this
            # module stays possible on a machine with no working audio device.
            import sounddevice as sd

            stream_ctx = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=FRAME_SAMPLES,
            )
        else:
            stream_ctx = contextlib.nullcontext()

        with stream_ctx as device_stream:

            def next_frame() -> np.ndarray:
                if frame_source is not None:
                    return frame_source()
                block, _overflowed = device_stream.read(FRAME_SAMPLES)
                # read() may reuse its internal buffer between calls, so this
                # must be copied before it is retained past this iteration.
                return block[:, 0].copy()

            while True:
                for event in stream.push(next_frame()):
                    if event is VadEvent.SPEECH_STARTED and on_speech_start is not None:
                        on_speech_start()
                    elif (
                        event is VadEvent.BARGE_IN_CONFIRMED
                        and on_barge_in_confirmed is not None
                    ):
                        on_barge_in_confirmed()
                    elif event is VadEvent.UTTERANCE_READY:
                        utterance = stream.take_utterance()
                        if utterance is not None:
                            return utterance


class VadStream:
    """The frame-driven turn state machine.

    Push frames in; get events out. Holds all per-utterance state: whether
    speech is currently active, how much of it has accumulated toward barge-in
    confirmation, the pre-roll ring, and any audio carried over from a segment
    the turn detector judged incomplete.

    Not thread-safe. It is designed to be owned by exactly one capture loop,
    which is the only arrangement that makes sense -- frames arrive in order
    from a single device, and the decisions here depend on that order.
    """

    def __init__(
        self, config: VoiceActivityDetector, turn_detector: "TurnDetector | None" = None
    ) -> None:
        self._config = config
        self._vad = config._vad
        self._turn_detector = turn_detector

        self._pre_roll_samples = int(SAMPLE_RATE * config.pre_roll_ms / 1000)
        self._confirm_samples = int(SAMPLE_RATE * config.barge_in_confirm_ms / 1000)
        self._max_samples = int(SAMPLE_RATE * config.max_speech_duration)

        # Must span a full-length utterance plus its pre-roll, since the
        # pre-roll is read at the moment the segment finalizes -- by which time
        # the segment's own start may be max_speech_duration in the past. At
        # 16kHz float32 that is roughly 1.3MB, allocated once per stream.
        capacity = int(SAMPLE_RATE * (config.max_speech_duration + config.pre_roll_ms / 1000)) + FRAME_SAMPLES
        self._ring = _AudioRing(capacity)

        self._reset_segment()
        # Carried across segments, not reset with them:
        self._carried: np.ndarray | None = None  # audio from "incomplete" segments
        self._barge_in_fired = False  # one barge-in per turn, not per segment
        self._grace_frames = max(1, int(config.endpoint_grace_ms / FRAME_MS))
        self._pending: Utterance | None = None
        # Speech samples accumulated across this turn's finalized segments --
        # what Utterance.speech_duration_s reports. Pre-roll is excluded by
        # construction: sherpa's segment length never includes it.
        self._turn_speech_samples = 0

    def _reset_segment(self) -> None:
        """Clear the state that belongs to a single VAD segment."""
        self._vad.reset()
        self._ring.reset()
        self._speaking = False
        self._speech_samples = 0
        self._silence_frames = 0

    def reset_turn(self) -> None:
        """Discard everything and start a fresh turn.

        Call between turns. Distinct from ``_reset_segment`` because a turn may
        legitimately span several segments -- that is exactly what the endpoint
        detector's "incomplete, keep waiting" verdict produces.
        """
        self._reset_segment()
        self._carried = None
        self._barge_in_fired = False
        self._grace_frames = max(1, int(self._config.endpoint_grace_ms / FRAME_MS))
        self._pending = None
        self._turn_speech_samples = 0

    @property
    def speaking(self) -> bool:
        """Whether the VAD currently believes speech is in progress."""
        return self._speaking

    def take_utterance(self) -> Utterance | None:
        """Retrieve and clear the utterance announced by ``UTTERANCE_READY``.

        Take-once, so a caller cannot accidentally process the same turn twice
        by polling. Returns None if there is nothing pending.
        """
        utterance, self._pending = self._pending, None
        if utterance is not None:
            self.reset_turn()
        return utterance

    def push(self, frame: np.ndarray) -> Iterator[VadEvent]:
        """Feed one frame and yield whatever it caused.

        A generator so callers can react to each event in order without this
        method needing to know about callbacks, queues, or threads. Typical
        frames produce nothing at all.
        """
        self._ring.append(frame)
        self._vad.accept_waveform(frame)

        if not self._speaking and self._vad.is_speech_detected():
            self._speaking = True
            self._silence_frames = 0
            yield VadEvent.SPEECH_STARTED

        # Endpoint grace: only meaningful while carrying a segment the detector
        # called incomplete. If the user does not resume within the (adaptive)
        # window, the verdict was wrong -- they had finished -- so hand back
        # what was already captured.
        if self._carried is not None and not self._speaking:
            self._silence_frames += 1
            if self._silence_frames >= self._grace_frames:
                self._pending = Utterance(
                    audio=self._carried,
                    sample_rate=SAMPLE_RATE,
                    speech_samples=self._turn_speech_samples,
                )
                yield VadEvent.UTTERANCE_READY
                return

        if self._speaking and not self._barge_in_fired:
            self._speech_samples += len(frame)
            if self._speech_samples >= self._confirm_samples:
                self._barge_in_fired = True
                yield VadEvent.BARGE_IN_CONFIRMED

        if self._vad.empty():
            return

        # A segment finalized. Its geometry -- where it starts and how long it
        # is -- must be read BEFORE pop(), and the audio itself comes from our
        # own ring rather than from the segment object.
        #
        # This is not defensive style, it is a bug that was caught by
        # scripts/_smoke_turn.py: `front` returns a handle into sherpa-onnx's
        # internal queue, and `pop()` invalidates it. Reading `segment.samples`
        # afterwards yields an empty array, so the captured turn came out as
        # exactly the pre-roll -- 0.30s of audio for 4.6s of speech, silently,
        # with no error anywhere. Taking the length first and slicing our own
        # buffer sidesteps the lifetime question entirely.
        segment = self._vad.front
        segment_start = segment.start
        segment_length = len(segment.samples)
        self._vad.pop()

        self._turn_speech_samples += segment_length
        seg_audio = self._ring.read(
            segment_start - self._pre_roll_samples, segment_start + segment_length
        )
        combined = (
            seg_audio if self._carried is None else np.concatenate([self._carried, seg_audio])
        )

        if self._turn_detector is None:
            self._pending = Utterance(
                audio=combined,
                sample_rate=SAMPLE_RATE,
                speech_samples=self._turn_speech_samples,
            )
            yield VadEvent.UTTERANCE_READY
            return

        is_complete, prob = self._turn_detector.predict(combined, SAMPLE_RATE)
        if is_complete or len(combined) >= self._max_samples:
            self._pending = Utterance(
                audio=combined,
                sample_rate=SAMPLE_RATE,
                speech_samples=self._turn_speech_samples,
            )
            yield VadEvent.UTTERANCE_READY
            return

        # Incomplete: keep the audio and listen for the continuation. How long
        # to wait scales with how incomplete it looked -- see
        # grace_frames_for_prob. Bounded by that grace and by max_speech_duration.
        self._grace_frames = self._config.grace_frames_for_prob(prob)
        self._carried = combined
        # Segment-level state restarts; turn-level state (carried audio, the
        # once-per-turn barge-in latch) deliberately does not.
        self._reset_segment()
