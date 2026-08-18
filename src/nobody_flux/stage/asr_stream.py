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
class LocalAgreementStabilizer:
    """The LocalAgreement state machine, model-free.

    Extracted from ``StreamingTranscriber`` so the commit/monotonicity rules
    can be unit-tested on plain strings -- no recognizer, no weights, no audio.
    ``StreamingTranscriber`` owns one and feeds it every decoder hypothesis.
    """

    agreement_n: int = 2
    _recent: list[str] = field(init=False, repr=False, default_factory=list)
    _committed: str = field(init=False, repr=False, default="")

    def reset(self) -> None:
        self._recent = []
        self._committed = ""

    def observe(self, hypothesis: str) -> None:
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

    def force_commit(self, text: str) -> None:
        """Everything is settled (finalize) -- the full text becomes committed."""
        self._committed = text

    @property
    def committed(self) -> str:
        return self._committed

    @property
    def hypothesis(self) -> str:
        return self._recent[-1] if self._recent else ""


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
    _stabilizer: LocalAgreementStabilizer = field(init=False, repr=False, default=None)
    _endpoint: bool = field(init=False, repr=False, default=False)

    def __post_init__(self) -> None:
        self._stabilizer = LocalAgreementStabilizer(agreement_n=self.agreement_n)
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
        self._stabilizer.reset()
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

        self._stabilizer.observe(self._recognizer.get_result(self._stream))

    # -- reading ----------------------------------------------------------

    @property
    def committed(self) -> str:
        """Stabilized prefix -- agreed on by ``agreement_n`` hypotheses, and
        guaranteed never to shrink. This is the text that is safe to act on
        mid-utterance."""
        return self._stabilizer.committed

    @property
    def hypothesis(self) -> str:
        """The decoder's full current guess, committed prefix included. Safe to
        display as a live caption; not safe to act on, since the uncommitted
        tail may still be revised."""
        return self._stabilizer.hypothesis

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
        self._stabilizer.force_commit(text)
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


DEFAULT_SENSE_VOICE_DIR = PROJECT_ROOT / "models" / "sense-voice"


def should_decode(
    buffered_samples: int,
    samples_at_last_decode: int,
    min_decode_s: float,
    hop_s: float,
    sample_rate: int = SAMPLE_RATE,
) -> bool:
    """Whether enough audio has arrived to be worth another full re-decode.

    Pulled out as a free function for the same reason ``longest_common_prefix``
    was: it is the entire substance of the chunking policy, it decides both the
    CPU cost and the partial-transcript latency, and it deserves to be testable
    on plain integers with no model, weights or audio anywhere near it.

    The two gates are not symmetric. ``min_decode_s`` is an absolute floor
    below which a decode is worse than useless -- SenseVoice returns nothing for
    very short input, and *two* empty hypotheses in a row would agree with each
    other and let LocalAgreement commit the empty string. ``hop_s`` is the
    recurring interval after that.

    A consequence worth knowing before tuning either: the earliest a prefix can
    possibly be committed is ``min_decode_s + hop_s`` (one decode to form a
    hypothesis, one more to agree with it at the default agreement_n=2).
    Measured at the defaults that is ~1.29s, which is longer than most turns in
    this project's own capture set -- measured, 8 of 16 committed nothing usable
    before finalize (one of them committed only ".", which is nothing).
    """
    if buffered_samples < int(min_decode_s * sample_rate):
        return False
    return (buffered_samples - samples_at_last_decode) >= int(hop_s * sample_rate)


@dataclass
class ChunkedSenseVoiceTranscriber:
    """Streaming transcripts from SenseVoice, which is not a streaming model.

    ## Why this exists

    ``StreamingTranscriber`` above is the architecturally correct answer -- a
    real streaming transducer, native Korean, built-in endpointing. It does not
    work. On clean test wavs it matches the batch path exactly (similarity 1.00
    via scripts/_smoke_turn.py); on real microphone captures it returns the empty
    string. docs/FEATURES.md records ruling out level (16x amplification still
    empty), noise (clean speech fine at SNR 5dB), lead-in, and length, and the
    *batch* path of the same checkpoint fails identically while SenseVoice reads
    the same audio fine. So the wiring is right and the checkpoint is wrong.

    It is not our bug to fix, either: sherpa-onnx issue #2886 reports the same
    symptom ("always returns empty string") for the Korean streaming models, was
    opened 2025-12-10, is still open with no maintainer fix, and attributes it to
    a malformed encoder ONNX export. Waiting on that is not a plan.

    docs/FEATURES.md left two options -- find another Korean streaming
    checkpoint, or run SenseVoice in chunks. This is the second one.

    ## What it does

    Accumulate the utterance, and every ``hop_s`` of *new* audio re-decode the
    whole thing from the start with the offline recognizer, feeding each result
    to the same ``LocalAgreementStabilizer`` the transducer path uses. Two
    consecutive decodes agreeing on a prefix commits it.

    Re-decoding from the start every hop, rather than decoding each chunk
    independently and concatenating, is the whole point. SenseVoice is a
    non-autoregressive encoder over the full input; a chunk decoded in isolation
    has no left context, and the measurements in docs/FEATURES.md show this
    checkpoint needs roughly 2.8s of audio before it recognizes an utterance
    completely and returns nothing at all below ~0.5s. Independent chunks would
    therefore be individually unreadable no matter how they were stitched.

    ## The cost, stated plainly

    That makes decode work quadratic in utterance length: an N-second utterance
    is decoded N/hop_s times, over a mean of N/2 seconds each. This is the
    central thing to measure before adopting it -- see scripts/_ab_asr.py. On a
    CM4 it may simply be unaffordable, in which case the honest outcome is a
    recorded number and no adoption. ``max_buffer_s`` bounds the worst case.

    ## What is lost relative to the transducer path

    ``endpoint_detected`` is always False. SenseVoice has no decoder state and
    therefore no endpointing of its own, so the deliberate dual-endpointing
    design documented at the top of this module -- the recognizer's decoder
    endpointing *and* TEN-VAD's acoustic endpointing, arbitrated by the
    controller -- loses one of its two signals here. Turn ends fall entirely to
    TEN-VAD and Smart Turn. That is a real reduction: the two disagreed usefully,
    the decoder holding on through a mid-word pause where TEN-VAD cuts.

    Also inherited from the batch path: this checkpoint's Korean tokenization
    inserts spurious mid-eojeol spaces, which is why LocalAgreement here compares
    characters rather than words (see ``longest_common_prefix``).
    """

    model_dir: Path = DEFAULT_SENSE_VOICE_DIR
    use_int8: bool = True
    language: str = "ko"
    # Inverse text normalization ON, matching the batch NobodyASR preset so the
    # two paths produce comparable text. Note this is what makes the recognizer
    # write digits for spoken numerals -- see scripts/_ab_tts.py, where that
    # behaviour had to be excluded from a CER aggregate.
    use_itn: bool = True
    num_threads: int = 2

    agreement_n: int = 2

    # How much new audio to accumulate before re-decoding. 0.48s = 16 frames of
    # 30ms, so it lands on a frame boundary and never splits one. Smaller means
    # fresher partials and more CPU; given the quadratic cost above, this is the
    # main dial for the latency/compute trade.
    hop_s: float = 0.48
    # Below this, do not decode at all. FEATURES.md measured that this
    # checkpoint returns nothing for a 0.5s utterance and only a partial at
    # 0.8s, so decoding earlier spends CPU to produce an empty hypothesis --
    # and an empty hypothesis is not harmless: two of them in a row would agree
    # with each other and commit the empty string as a prefix.
    min_decode_s: float = 0.8
    # Hard cap, mirroring configs/vad.yaml's max_speech_duration. A stuck mic
    # must not be able to grow an unbounded buffer *and* a quadratic decode
    # cost at the same time.
    max_buffer_s: float = 20.0

    def __post_init__(self) -> None:
        model_file = "model.int8.onnx" if self.use_int8 else "model.onnx"
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(self.model_dir / model_file),
            tokens=str(self.model_dir / "tokens.txt"),
            num_threads=self.num_threads,
            language=self.language,
            use_itn=self.use_itn,
        )
        self._stabilizer = LocalAgreementStabilizer(agreement_n=self.agreement_n)
        self._buffer: list[np.ndarray] = []
        self._buffered_samples = 0
        self._samples_at_last_decode = 0
        self.decode_count = 0
        self.decoded_samples_total = 0

    # -- lifecycle ---------------------------------------------------------

    def reset(self) -> None:
        self._stabilizer.reset()
        self._buffer = []
        self._buffered_samples = 0
        self._samples_at_last_decode = 0
        # decode_count / decoded_samples_total deliberately survive a reset:
        # they are cost instrumentation for a whole session, not per-utterance
        # state. scripts/_ab_asr.py reads them to report the quadratic penalty.

    def accept_frame(self, frame: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
        # Validated, not resampled -- same rule as StreamingTranscriber above.
        # The capture path is fixed at 16kHz, so a mismatch here is a wiring bug
        # and silently correcting it would hide the bug while changing the
        # audio the recognizer was tuned on.
        if sample_rate != SAMPLE_RATE:
            raise ValueError(
                f"ChunkedSenseVoiceTranscriber expects {SAMPLE_RATE}Hz audio, got "
                f"{sample_rate}Hz. The capture path is fixed at 16kHz (see "
                "audio/session.py); resample before calling if that ever changes."
            )

        frame = np.asarray(frame, dtype=np.float32)
        if frame.ndim > 1:
            frame = frame.mean(axis=1)

        cap = int(self.max_buffer_s * SAMPLE_RATE)
        if self._buffered_samples >= cap:
            # Drop rather than grow. Losing the tail of a 20s monologue is worse
            # than the alternative only in theory; in practice the turn is
            # already past any useful length, and the decode cost is the thing
            # that will actually break the session.
            return

        self._buffer.append(frame)
        self._buffered_samples += len(frame)

        if should_decode(
            self._buffered_samples,
            self._samples_at_last_decode,
            self.min_decode_s,
            self.hop_s,
        ):
            self._decode_buffer()

    def _decode_buffer(self) -> None:
        audio = np.concatenate(self._buffer) if self._buffer else np.zeros(0, np.float32)
        self._samples_at_last_decode = self._buffered_samples
        self.decode_count += 1
        self.decoded_samples_total += len(audio)
        self._stabilizer.observe(self._decode(audio))

    def _decode(self, audio: np.ndarray) -> str:
        """One offline decode of `audio`. A fresh stream every time.

        The recognizer is stateless across streams by design, which is exactly
        why re-decoding from the start is possible at all -- but it also means
        there is nothing to incrementally reuse, hence the quadratic cost.
        """
        if len(audio) == 0:
            return ""
        stream = self.recognizer.create_stream()
        stream.accept_waveform(SAMPLE_RATE, audio)
        self.recognizer.decode_stream(stream)
        return " ".join(stream.result.text.split())

    # -- readouts ----------------------------------------------------------

    @property
    def committed(self) -> str:
        return self._stabilizer.committed

    @property
    def hypothesis(self) -> str:
        return self._stabilizer.hypothesis

    @property
    def endpoint_detected(self) -> bool:
        """Always False -- see the class docstring. SenseVoice has no decoder
        state to endpoint on, so ending the turn is entirely TEN-VAD's and Smart
        Turn's job under this transcriber."""
        return False

    def finalize(self) -> str:
        """Decode everything once more and treat the result as settled.

        The final decode is unconditional, even if a hop just ran, because the
        last partial hop of audio is often where the sentence-final ending lives
        -- and in Korean that is the most informative part of the utterance.
        """
        if self._buffered_samples == 0:
            return self._stabilizer.committed
        audio = np.concatenate(self._buffer)
        self.decode_count += 1
        self.decoded_samples_total += len(audio)
        text = self._decode(audio)
        self._stabilizer.force_commit(text)
        return text

    def transcribe_array(
        self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE, frame_samples: int = 480
    ) -> str:
        """Replay a complete utterance through the live path -- same contract and
        same rationale as StreamingTranscriber.transcribe_array, so a recorded
        wav exercises the real incremental code rather than a parallel batch one."""
        self.reset()
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        for offset in range(0, len(audio), frame_samples):
            self.accept_frame(audio[offset : offset + frame_samples], sample_rate)
        return self.finalize()
