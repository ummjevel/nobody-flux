"""Voice activity detection: decides when one spoken turn starts and ends,
without a wakeword or full streaming ASR.

TEN-VAD (via sherpa_onnx's built-in support -- same runtime already used for
ASR, see asr.py, so no new dependency) replaces this project's original
hand-rolled RMS-energy-threshold VAD: a real trained model instead of two
fixed thresholds that need per-room/per-mic retuning by hand (that version's
still in git history if the tradeoff ever needs revisiting -- no ML model,
but also no ~330KB onnx file or onnxruntime inference cost per frame).

Known limits (document, don't silently paper over):
  - TEN-VAD's own thresholds (threshold/min_silence_duration/etc, all
    overridable via the constructor) are still just defaults tuned on
    whatever TEN Framework's own eval set looked like -- not guaranteed
    optimal for any specific mic/room either, just a better starting point
    than a hand-picked RMS cutoff.

barge_in_confirm_ms / on_barge_in_confirmed (see listen_for_utterance) exist
to tell a real interruption apart from backchannel ("어", "응") -- see
docs/barge-in-design.md for why a plain "any detected speech = barge-in"
rule is wrong for this project's casual persona, and the research behind
picking a duration threshold over more sophisticated (but out of scope for
this prototype) acoustic classifiers.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from .turn_detector import TurnDetector
import sherpa_onnx
import sounddevice as sd

from .paths import PROJECT_ROOT

SAMPLE_RATE = 16_000
FRAME_MS = 30
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "ten-vad" / "ten-vad.onnx"


@dataclass
class Utterance:
    audio: np.ndarray  # float32, mono, SAMPLE_RATE
    sample_rate: int = SAMPLE_RATE


@dataclass
class VoiceActivityDetector:
    model_path: Path = DEFAULT_MODEL_PATH
    # TenVadModelConfig's own defaults (confirmed by introspecting the
    # installed sherpa_onnx build) -- listed explicitly here rather than left
    # implicit, so they show up as something callers can override the same
    # way as every other field, and so this file doesn't silently change
    # behavior if sherpa_onnx ever changes ITS defaults.
    threshold: float = 0.5
    min_silence_duration: float = 0.5  # seconds of trailing silence -> segment ends
    # TenVadModelConfig's own default is 0.25s -- lowered here. This
    # project's persona (persona.py) is a casual, one/two-word-backchannel
    # kind of conversation ("네", "응"), and a single casual Korean syllable
    # said quickly can plausibly run under 250ms; at the original default
    # those would get silently discarded as noise, i.e. the user says
    # something and nothing happens, no error, no log line, just silence.
    min_speech_duration: float = 0.15
    max_speech_duration: float = 20.0  # hard cap so a stuck-open mic can't hang forever
    num_threads: int = 1
    # The VAD needs a few frames of evidence before it's confident speech has
    # started, so `segment.start` (see listen_for_utterance) tends to land
    # slightly after the true onset -- confirmed by hand: without this, the
    # first word or so of an utterance was reliably clipped. Padding the
    # returned audio backward by this much compensates.
    pre_roll_ms: int = 300
    # docs/barge-in-design.md's stage 1 (delayed-stop): how long speech has
    # to continue past on_speech_start before on_barge_in_confirmed fires.
    # 250ms, not this project's earlier 400ms guess -- recalibrated against
    # LiveKit's Adaptive Interruption Handling (216ms median duration to
    # decide, audio-only, in production), see that doc's "관련 연구" section.
    # Deliberately separate from min_speech_duration above: that one decides
    # whether TEN-VAD's segment finalizer treats a sound as speech at all
    # (too low and a real backchannel gets silently dropped as noise), this
    # one decides whether speech that's already confirmed real is *long
    # enough to be a barge-in rather than a backchannel*.
    barge_in_confirm_ms: int = 250
    # Only used when listen_for_utterance is given a turn_detector (Smart Turn
    # v3 endpoint detection -- see that arg's docstring): after the detector
    # says a just-finished segment is an *incomplete* turn (user paused
    # mid-thought, didn't actually finish), this is how long to keep waiting
    # for them to resume before giving up and returning what we have anyway.
    # Bounds the "wait for continuation" so a wrong "incomplete" verdict can't
    # hang the turn forever when the user really was done.
    endpoint_grace_ms: int = 800

    def __post_init__(self):
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
        # buffer_size_in_seconds: sherpa_onnx's own internal ring buffer for
        # audio it's still deciding about -- 100s is comfortably above
        # max_speech_duration so it never has to drop samples mid-utterance.
        self._vad = sherpa_onnx.VoiceActivityDetector(vad_config, buffer_size_in_seconds=100)

    def listen_for_utterance(
        self,
        on_speech_start: Callable[[], None] | None = None,
        on_barge_in_confirmed: Callable[[], None] | None = None,
        turn_detector: "TurnDetector | None" = None,
        frame_source: Callable[[], np.ndarray] | None = None,
    ) -> Utterance:
        """Block until one spoken turn has been captured, then return it.

        frame_source: optional callable returning the next FRAME_SAMPLES-long
        mono float32 frame. When given, this reads from it instead of opening
        its own sd.InputStream -- that's how the unified duplex AudioSession
        (audio.py, Phase 1.5) feeds already-echo-cancelled mic frames in, so
        capture and playback share one stream (fixes the macOS err -50 duplex
        conflict) and barge-in doesn't false-trip on the reply's own echo. When
        None (default), behaviour is unchanged: this owns a private input stream.

        Feeds mic frames into the VAD's streaming accept_waveform() until it
        queues one finished segment (its own internal state machine handles
        speech/silence timing per the thresholds above). Rather than
        returning that segment's own `samples` directly, this keeps its own
        copy of every frame fed in since the last reset() and slices the
        returned audio out of THAT -- using `segment.start` as the cut point,
        pulled back by pre_roll_ms -- since the VAD's own segment tends to
        start a bit after the true onset (see pre_roll_ms's docstring).

        on_speech_start: called once per captured segment, the instant the
        VAD's internal "currently in speech" flag first flips true. Optional
        -- this function otherwise blocks silently for however long that
        takes, which from the caller's side is indistinguishable from "not
        listening at all"; a caller like talk.py can use this to print
        something so the user isn't staring at an unchanging "... listening
        ..." wondering if the mic works.

        on_barge_in_confirmed: called once per turn, when speech has continued
        for barge_in_confirm_ms past on_speech_start -- see that field's
        docstring and docs/barge-in-design.md. Fires strictly after
        on_speech_start, never instead of it. This is stage 1 of that doc's
        two-stage design (delayed-stop); stage 2 (post-hoc lexical check
        against the ASR result once this call returns) is the caller's job,
        not this function's -- this function only knows audio, never text.

        turn_detector: optional Smart Turn v3 endpoint detector (see
        turn_detector.py). When given, a segment that TEN-VAD finalized on
        silence is NOT immediately returned -- the detector is asked whether
        it's a *complete* turn or a mid-thought pause. On "incomplete," this
        keeps listening (up to endpoint_grace_ms of continued silence, or
        max_speech_duration total) and concatenates any continuation onto the
        same utterance, so a natural pause ("음... 그러니까...") doesn't get
        chopped into separate turns the way pure silence-based endpointing
        does. When None, behaviour is exactly the old single-segment,
        silence-only endpointing (unchanged). Note: the accumulation loop
        below is logic-reviewed but NOT yet validated on a live mic (this dev
        env's WSL2 mic constraint -- see talk.py) -- treat the default
        (turn_detector=None) as the trusted path.
        """
        pre_roll_samples = int(SAMPLE_RATE * self.pre_roll_ms / 1000)
        confirm_samples = int(SAMPLE_RATE * self.barge_in_confirm_ms / 1000)
        grace_frames = max(1, int(self.endpoint_grace_ms / FRAME_MS))
        max_samples = int(SAMPLE_RATE * self.max_speech_duration)

        # Audio from earlier segments the detector judged "incomplete" -- the
        # continuation gets concatenated onto this. None until the first
        # incomplete verdict; stays None entirely when turn_detector is None.
        carried: np.ndarray | None = None
        barge_in_fired = False  # hoisted across segments: one barge-in per turn, not per segment

        # Own a private input stream only when no external frame_source is given
        # (see that arg). With one, the AudioSession owns the (duplex) stream and
        # this just pulls frames from it -- contextlib.nullcontext keeps the
        # `with` shape identical either way.
        if frame_source is None:
            stream_ctx = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=FRAME_SAMPLES,
            )
        else:
            stream_ctx = contextlib.nullcontext()

        with stream_ctx as stream:

            def next_frame() -> np.ndarray:
                if frame_source is not None:
                    return frame_source()
                block, _overflowed = stream.read(FRAME_SAMPLES)
                # InputStream.read() may reuse its internal buffer across calls
                # -- copy before stashing it past this iteration.
                return block[:, 0].copy()

            while True:  # outer loop: one iteration captures one VAD segment
                self._vad.reset()
                speaking = False
                speech_samples_seen = 0
                silence_frames_while_waiting = 0
                frames: list[np.ndarray] = []

                while True:  # inner loop: accumulate frames until one segment finalizes
                    samples = next_frame()
                    frames.append(samples)
                    self._vad.accept_waveform(samples)

                    if not speaking and self._vad.is_speech_detected():
                        speaking = True
                        if on_speech_start is not None:
                            on_speech_start()

                    # Grace timeout: only relevant once we're carrying an
                    # "incomplete" turn and waiting to see if the user resumes.
                    # If they don't start speaking again within endpoint_grace_ms
                    # of silence, treat the carried audio as the final turn --
                    # the detector's "incomplete" was wrong (they were done).
                    if carried is not None and not speaking:
                        silence_frames_while_waiting += 1
                        if silence_frames_while_waiting >= grace_frames:
                            return Utterance(audio=carried, sample_rate=SAMPLE_RATE)

                    if speaking and not barge_in_fired:
                        speech_samples_seen += len(samples)
                        if speech_samples_seen >= confirm_samples:
                            barge_in_fired = True
                            if on_barge_in_confirmed is not None:
                                on_barge_in_confirmed()

                    if not self._vad.empty():
                        segment = self._vad.front
                        self._vad.pop()
                        full_audio = np.concatenate(frames)
                        start = max(0, segment.start - pre_roll_samples)
                        end = segment.start + len(segment.samples)
                        seg_audio = full_audio[start:end]
                        break

                combined = seg_audio if carried is None else np.concatenate([carried, seg_audio])

                if turn_detector is None:
                    return Utterance(audio=combined, sample_rate=SAMPLE_RATE)

                is_complete, _prob = turn_detector.predict(combined, SAMPLE_RATE)
                if is_complete or len(combined) >= max_samples:
                    return Utterance(audio=combined, sample_rate=SAMPLE_RATE)
                # Incomplete: keep this audio and loop to capture the
                # continuation (bounded by grace timeout above and max_samples).
                carried = combined
