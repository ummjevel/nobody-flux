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

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
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
    ) -> Utterance:
        """Block until one spoken turn has been captured, then return it.

        Feeds mic frames into the VAD's streaming accept_waveform() until it
        queues one finished segment (its own internal state machine handles
        speech/silence timing per the thresholds above). Rather than
        returning that segment's own `samples` directly, this keeps its own
        copy of every frame fed in since the last reset() and slices the
        returned audio out of THAT -- using `segment.start` as the cut point,
        pulled back by pre_roll_ms -- since the VAD's own segment tends to
        start a bit after the true onset (see pre_roll_ms's docstring).

        on_speech_start: called once, the instant the VAD's internal
        "currently in speech" flag first flips true. Optional -- this
        function otherwise blocks silently for however long that takes,
        which from the caller's side is indistinguishable from "not
        listening at all"; a caller like talk.py can use this to print
        something so the user isn't staring at an unchanging "... listening
        ..." wondering if the mic works.

        on_barge_in_confirmed: called once, when speech has continued for
        barge_in_confirm_ms past on_speech_start -- see that field's
        docstring and docs/barge-in-design.md. Fires strictly after
        on_speech_start, never instead of it. This is stage 1 of that doc's
        two-stage design (delayed-stop); stage 2 (post-hoc lexical check
        against the ASR result once this call returns) is the caller's job,
        not this function's -- this function only knows audio, never text.
        """
        self._vad.reset()
        speaking = False
        barge_in_confirmed = False
        speech_samples_seen = 0  # samples fed in since on_speech_start fired
        frames: list[np.ndarray] = []  # every frame since reset(), for the pre-roll slice below
        pre_roll_samples = int(SAMPLE_RATE * self.pre_roll_ms / 1000)
        confirm_samples = int(SAMPLE_RATE * self.barge_in_confirm_ms / 1000)

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=FRAME_SAMPLES,
        ) as stream:
            while True:
                block, _overflowed = stream.read(FRAME_SAMPLES)
                # InputStream.read() may reuse its internal buffer across
                # calls -- copy before stashing it past this iteration.
                samples = block[:, 0].copy()
                frames.append(samples)
                self._vad.accept_waveform(samples)

                if not speaking and self._vad.is_speech_detected():
                    speaking = True
                    if on_speech_start is not None:
                        on_speech_start()

                if speaking and not barge_in_confirmed:
                    speech_samples_seen += len(samples)
                    if speech_samples_seen >= confirm_samples:
                        barge_in_confirmed = True
                        if on_barge_in_confirmed is not None:
                            on_barge_in_confirmed()

                if not self._vad.empty():
                    segment = self._vad.front
                    self._vad.pop()
                    full_audio = np.concatenate(frames)
                    start = max(0, segment.start - pre_roll_samples)
                    end = segment.start + len(segment.samples)
                    return Utterance(audio=full_audio[start:end], sample_rate=SAMPLE_RATE)
