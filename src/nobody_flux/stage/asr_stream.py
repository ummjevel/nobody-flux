"""Phase 3: recognition that runs *while* the user is talking.

The problem this solves
-----------------------

Every turn so far has paid for recognition twice over. ``turn/vad.py`` captures
a whole utterance, the caller writes it to a wav, and only then does an ASR
stage read that file back and decode it. Nothing is recognized until the user
has already stopped speaking, so the ASR cost -- ~130ms for the small models,
~2s for VibeASR -- sits squarely on the critical path between "user stops" and
"reply starts", *added to* the silence the VAD had to observe before it could
declare the utterance over.

A streaming transducer decodes incrementally instead. Frames go in as the
microphone produces them, and by the time speech ends the decoder has already
consumed everything but the last few frames. Recognition latency stops being a
term in the turn latency equation and becomes a rounding error.

Why LocalAgreement is needed
----------------------------

Incremental output is not free: a transducer's hypothesis is *unstable*. Having
emitted "그래서 내가" it may, three frames later, revise it to "그래서 냈어" as
more acoustic context arrives. Any consumer that acts on the raw running
hypothesis will act on text that gets retracted.

LocalAgreement (Liu et al. 2020; the stabilization scheme Macháček's
whisper-streaming popularized) resolves this with one rule: **text is committed
only once N consecutive hypotheses agree on it.** Concretely, hold the last N
hypotheses and take their longest common prefix. A prefix that survived N
independent decoder updates is one the decoder has stopped revising, so it can
be treated as final; everything past it stays provisional.

``N=2`` is the default here, matching whisper-streaming's own. Higher N commits
later but retracts less; the trade is directly tunable via ``agreement_n``.

Agreement is computed over **characters**, not words. Korean is the target
language and this checkpoint emits Korean without inter-eojeol spaces (a known
property of the model -- see ``StreamingZipformerAsr`` in ``asr.py``), so there
are no reliable word boundaries to align on. Characters are also strictly finer
grained, which means the committed prefix advances sooner.

Endpointing
-----------

The recognizer carries its own endpoint detector, which decides an utterance is
over from *decoder* state -- trailing silence measured relative to what has
actually been recognized -- rather than from raw acoustic energy the way
TEN-VAD does. The two disagree in useful ways: TEN-VAD sees a breath as speech,
the decoder does not; the decoder holds on through a pause mid-word, TEN-VAD
cuts. ``turn/controller.py`` consumes both and is where the arbitration policy
lives, deliberately not here -- this class reports what its decoder thinks and
nothing more.

Status
------

Logic-complete and unit-testable by feeding it wav frames (see
``transcribe_array``, which is exactly that path). The *live microphone* loop
has not been validated end to end -- that is what the native-Windows
environment was built to make possible, and until it has been done the batch
ASR stages remain the trusted default. See ``docs/FEATURES.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import sherpa_onnx

from ..paths import PROJECT_ROOT

SAMPLE_RATE = 16_000

DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "streaming-zipformer-ko"


def longest_common_prefix(texts: list[str]) -> str:
    """The longest string that every input starts with (``""`` if they share
    no first character, or if fewer than one string was given).

    Pulled out as a free function because it is the entire substance of the
    LocalAgreement rule and deserves to be testable on plain strings, with no
    model, device, or audio anywhere near it.
    """
    if not texts:
        return ""
    if len(texts) == 1:
        return texts[0]

    shortest = min(texts, key=len)
    for index, char in enumerate(shortest):
        # Comparing against every other hypothesis, not just the previous one:
        # with agreement_n > 2 a prefix must survive *all* of the retained
        # hypotheses, which is what makes larger N a stronger stability
        # guarantee rather than merely a longer one.
        if any(other[index] != char for other in texts):
            return shortest[:index]
    return shortest


@dataclass
class StreamingTranscriber:
    """Live incremental Korean recognition over a frame stream.

    Lifecycle, per turn::

        transcriber.reset()
        while capturing:
            transcriber.accept_frame(frame)     # 16kHz mono float32
            if transcriber.endpoint_detected:
                break
        text = transcriber.finalize()

    ``committed`` and ``hypothesis`` may be read at any point between frames:
    ``committed`` is the LocalAgreement-stabilized prefix (safe to act on),
    ``hypothesis`` is the decoder's full current guess (fine to display, not to
    act on).
    """

    model_dir: Path = DEFAULT_MODEL_DIR
    # int8 by default: the fp32 encoder is ~292MB against ~127MB quantized, and
    # every deployment target here is CPU-bound. Accuracy difference on this
    # checkpoint was not measurable against the project's test wavs.
    use_int8: bool = True
    num_threads: int = 2

    # -- LocalAgreement ---------------------------------------------------
    # How many consecutive hypotheses must agree before a prefix is committed.
    # 2 is whisper-streaming's default and the right starting point: it removes
    # the great majority of retractions (a one-off flap never survives a second
    # update) at the cost of one decoder update's delay. Raise it if committed
    # text is still seen to change; each increment costs roughly one frame
    # period of additional commit latency.
    agreement_n: int = 2

    # -- Endpoint detection ------------------------------------------------
    # The recognizer's own endpointing, which is separate from and complementary
    # to TEN-VAD's. Enabled here so the controller has both signals available;
    # which one is allowed to end a turn is the controller's decision.
    enable_endpoint_detection: bool = True
    # Trailing silence required to end a turn when *nothing* has been decoded
    # yet -- i.e. the stream opened, the user never really spoke, and this is
    # the timeout. sherpa-onnx defaults to 2.4s; kept generous because firing
    # early here means cutting off someone who simply paused before starting.
    rule1_min_trailing_silence: float = 2.4
    # Trailing silence required after something *has* been decoded. This is the
    # one that governs conversational responsiveness, and sherpa-onnx's 1.2s
    # default is far too slow for it: a listener reads a 1.2s gap as the other
    # person having finished and then some. Set to 0.6s to sit just above
    # configs/vad.yaml's min_silence_duration (0.5s), so TEN-VAD -- the tighter,
    # better-tuned signal -- normally ends the turn first and this acts as the
    # backstop rather than the primary. Estimated, not yet measured on a real
    # microphone.
    rule2_min_trailing_silence: float = 0.6
    # Hard cap on a single utterance, mirroring vad.py's max_speech_duration so
    # neither component can hold a turn open on its own.
    rule3_min_utterance_length: float = 20.0

    # -- Internal state ----------------------------------------------------
    _recognizer: object = field(init=False, repr=False, default=None)
    _stream: object = field(init=False, repr=False, default=None)
    _recent: list[str] = field(init=False, repr=False, default_factory=list)
    _committed: str = field(init=False, repr=False, default="")
    _endpoint: bool = field(init=False, repr=False, default=False)

    def __post_init__(self) -> None:
        suffix = ".int8.onnx" if self.use_int8 else ".onnx"
        # sample_rate/feature_dim are this checkpoint's training configuration,
        # not tunable parameters -- changing either produces silent garbage
        # rather than an error.
        self._recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(self.model_dir / "tokens.txt"),
            encoder=str(self.model_dir / f"encoder-epoch-99-avg-1{suffix}"),
            decoder=str(self.model_dir / f"decoder-epoch-99-avg-1{suffix}"),
            joiner=str(self.model_dir / f"joiner-epoch-99-avg-1{suffix}"),
            num_threads=self.num_threads,
            sample_rate=SAMPLE_RATE,
            feature_dim=80,
            decoding_method="greedy_search",
            enable_endpoint_detection=self.enable_endpoint_detection,
            rule1_min_trailing_silence=self.rule1_min_trailing_silence,
            rule2_min_trailing_silence=self.rule2_min_trailing_silence,
            rule3_min_utterance_length=self.rule3_min_utterance_length,
        )
        self.reset()

    # -- state ------------------------------------------------------------

    def reset(self) -> None:
        """Discard all state and begin a fresh utterance.

        A new stream rather than the recognizer's ``reset(stream)``: a fresh
        stream cannot carry over decoder or feature state from the previous
        turn, and stream creation is cheap next to a turn's other costs. This
        matters because leaked state shows up as the *previous* turn's words
        reappearing at the head of the next one, which is both confusing and
        hard to trace back.
        """
        self._stream = self._recognizer.create_stream()
        self._recent = []
        self._committed = ""
        self._endpoint = False

    # -- feeding ----------------------------------------------------------

    def accept_frame(self, frame: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
        """Feed one frame of mono float32 audio and advance the decoder.

        Frame length is not constrained -- the recognizer buffers internally --
        so the 30ms frames the audio session produces work unchanged, as does
        any other size. ``sample_rate`` is validated rather than resampled: the
        entire capture path is fixed at 16kHz, so a mismatch is a wiring bug
        that should be reported, not silently corrected.
        """
        if sample_rate != SAMPLE_RATE:
            raise ValueError(
                f"StreamingTranscriber expects {SAMPLE_RATE}Hz audio, got {sample_rate}Hz. "
                "The capture path is fixed at 16kHz (see audio/session.py); resample "
                "before calling if that ever changes."
            )

        self._stream.accept_waveform(sample_rate, frame)

        # Drain every decode step the new audio made available. is_ready() goes
        # false as soon as the decoder needs more input, so this is bounded by
        # the frame just fed -- it is not an open-ended loop.
        while self._recognizer.is_ready(self._stream):
            self._recognizer.decode_stream(self._stream)

        if self.enable_endpoint_detection and self._recognizer.is_endpoint(self._stream):
            self._endpoint = True

        self._observe(self._recognizer.get_result(self._stream))

    def _observe(self, hypothesis: str) -> None:
        """Record one hypothesis and recompute the committed prefix.

        This is the LocalAgreement rule itself: retain the last ``agreement_n``
        hypotheses, commit their longest common prefix.
        """
        self._recent.append(hypothesis)
        if len(self._recent) > self.agreement_n:
            # Drop from the front rather than using a deque: agreement_n is 2 or
            # 3, so the list operation is trivially cheap and keeps the
            # committed-prefix computation a plain slice over a list.
            del self._recent[0]

        if len(self._recent) < self.agreement_n:
            # Not enough evidence yet -- nothing has had the chance to be
            # confirmed by a second observation, so commit nothing.
            return

        agreed = longest_common_prefix(self._recent)
        # Monotonic guard. The committed prefix must never shrink: a consumer
        # that has already acted on committed text cannot un-act on it. If a
        # later agreement is somehow shorter (a mid-utterance decoder reset, a
        # hypothesis that dropped a leading token), keep what was already
        # promised rather than retracting it.
        if len(agreed) > len(self._committed):
            self._committed = agreed

    # -- reading ----------------------------------------------------------

    @property
    def committed(self) -> str:
        """Stabilized prefix -- agreed on by ``agreement_n`` hypotheses, and
        guaranteed never to shrink. This is the text that is safe to act on
        mid-utterance."""
        return self._committed

    @property
    def hypothesis(self) -> str:
        """The decoder's full current guess, committed prefix included. Safe to
        display as a live caption; not safe to act on, since the uncommitted
        tail may still be revised."""
        return self._recent[-1] if self._recent else ""

    @property
    def endpoint_detected(self) -> bool:
        """True once the recognizer's own endpoint rules have fired for this
        utterance. Latched rather than momentary -- the underlying flag is
        transient, and a caller polling between frames would otherwise miss
        it."""
        return self._endpoint

    def finalize(self) -> str:
        """End the utterance and return the complete text.

        Flushes the decoder with trailing silence before reading the result. A
        transducer will not emit the final tokens of an utterance until it has
        seen enough right-context; without this padding the last syllable or
        two are reliably lost -- the same reason ``StreamingZipformerAsr`` pads
        in its batch path.
        """
        tail = np.zeros(int(SAMPLE_RATE * 0.5), dtype=np.float32)
        self._stream.accept_waveform(SAMPLE_RATE, tail)
        self._stream.input_finished()
        while self._recognizer.is_ready(self._stream):
            self._recognizer.decode_stream(self._stream)

        text = self._recognizer.get_result(self._stream)
        # Everything is settled now, so the full text becomes committed --
        # there is no longer any uncommitted tail to be revised.
        self._committed = text
        return " ".join(text.split())

    # -- offline convenience ----------------------------------------------

    def transcribe_array(
        self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE, frame_samples: int = 480
    ) -> str:
        """Decode a complete utterance by replaying it through the live path.

        The point is not convenience but *fidelity*: this feeds the array frame
        by frame through exactly the code a microphone drives, so a test over a
        recorded wav exercises the incremental decoder, the LocalAgreement
        stabilizer and the endpoint logic rather than a separate batch path
        that could drift away from them. It is how the streaming engine can be
        checked without a working microphone.
        """
        self.reset()
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        for offset in range(0, len(audio), frame_samples):
            self.accept_frame(audio[offset : offset + frame_samples], sample_rate)
        return self.finalize()
