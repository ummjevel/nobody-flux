"""Semantic end-of-turn / turn-completeness detector: pipecat-ai's Smart Turn
v3, an 8M-param Whisper-tiny-encoder + linear-head ONNX classifier that, given
a captured utterance's audio, scores how likely it is a *complete* spoken turn
(someone actually done talking) vs an incomplete fragment.

Why this exists in a project that already has backchannel.py: backchannel.py's
stage 2 is a hand-maintained lexical wordlist ("어"/"응"/...) matched against
the ASR text -- it only catches words someone remembered to list, and only
after ASR runs. Smart Turn works on the raw audio instead, so it can flag a
short filler/backchannel-y utterance as "not a real turn" without depending on
an exact-string match. See docs/barge-in-design.md's research section for why
audio-only turn models are the production-standard approach and the honest
caveat that Smart Turn is *repurposed* here (its designed job is end-of-turn
detection, and using "incomplete turn" as a proxy for "backchannel" is a
reasonable but unvalidated-on-our-mic heuristic).

CPU-only, ~12ms inference, Korean is one of its 23 supported languages, BSD-2.
Model file (~8.7MB) is fetched by scripts/setup_common.sh, not committed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..paths import PROJECT_ROOT

SAMPLE_RATE = 16_000
# Smart Turn v3's fixed input geometry (confirmed by introspecting the onnx:
# input `input_features` [batch, 80, 800], output `logits` [batch, 1]). 80
# mel bins, 800 frames = 8s at Whisper's 160-sample hop (8 * 16000 / 160).
N_MELS = 80
N_FRAMES = 800
CHUNK_SAMPLES = SAMPLE_RATE * 8  # 128000, the 8s window the model sees

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "smart-turn-v3" / "smart-turn-v3.2-cpu.onnx"


@dataclass
class TurnDetector:
    model_path: Path = DEFAULT_MODEL_PATH
    # Sigmoid(logit) >= this -> "complete turn." 0.5 is the model's own stock
    # decision boundary; talk.py treats "not complete" as a backchannel-like
    # utterance to skip, so lowering this makes the detector *less* eager to
    # skip (a real user turn wrongly dropped is worse than a backchannel
    # wrongly answered -- same conservative stance as backchannel.py). Not
    # tuned against this project's own mic yet -- see docs/barge-in-design.md.
    complete_threshold: float = 0.5
    num_threads: int = 1
    _session: object = field(init=False, repr=False, default=None)
    _feat: object = field(init=False, repr=False, default=None)

    def __post_init__(self):
        import onnxruntime as ort
        from transformers import WhisperFeatureExtractor

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = self.num_threads
        opts.inter_op_num_threads = self.num_threads
        self._session = ort.InferenceSession(
            str(self.model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        # Whisper's log-mel frontend, but configured for the 8s / 80-bin /
        # 800-frame geometry this checkpoint's encoder expects (whisper-tiny's
        # stock extractor is 30s/3000-frame). n_fft/hop are Whisper's usual
        # 400/160.
        self._feat = WhisperFeatureExtractor(
            feature_size=N_MELS,
            sampling_rate=SAMPLE_RATE,
            hop_length=160,
            chunk_length=8,
            n_fft=400,
        )

    def _features(self, audio: np.ndarray) -> np.ndarray:
        """Raw mono 16k float32 audio -> [1, 80, 800] log-mel, with the audio
        pinned to the END of the 8s window (zeros padded at the front, most
        recent audio kept if longer than 8s) -- the end-of-turn decision is
        about how the utterance *finishes*, so the tail is what matters and
        what the reference implementation feeds."""
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio[:, 0]
        if len(audio) >= CHUNK_SAMPLES:
            audio = audio[-CHUNK_SAMPLES:]
        else:
            audio = np.concatenate(
                [np.zeros(CHUNK_SAMPLES - len(audio), dtype=np.float32), audio]
            )
        feats = self._feat(audio, sampling_rate=SAMPLE_RATE, return_tensors="np")
        input_features = np.asarray(feats["input_features"], dtype=np.float32)
        # Defensive: Whisper's centered STFT can yield N_FRAMES(+1); pin to the
        # exact width the onnx input demands rather than trusting the extractor.
        t = input_features.shape[-1]
        if t > N_FRAMES:
            input_features = input_features[:, :, :N_FRAMES]
        elif t < N_FRAMES:
            input_features = np.pad(
                input_features, ((0, 0), (0, 0), (0, N_FRAMES - t)), mode="constant"
            )
        return input_features

    def predict(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> tuple[bool, float]:
        """Returns (is_complete_turn, probability). probability is
        sigmoid(logit) in [0, 1]; is_complete_turn is probability >=
        complete_threshold. Expects 16kHz mono audio (vad.py's utterances
        already are) -- sample_rate is validated, not resampled, since the
        only caller feeds VAD output at the fixed project SAMPLE_RATE."""
        if sample_rate != SAMPLE_RATE:
            raise ValueError(
                f"TurnDetector expects {SAMPLE_RATE}Hz audio, got {sample_rate}Hz "
                "(vad.py produces 16kHz; resample before calling if that changes)."
            )
        input_features = self._features(audio)
        (logits,) = self._session.run(None, {"input_features": input_features})
        prob = float(1.0 / (1.0 + np.exp(-logits.reshape(-1)[0])))
        return prob >= self.complete_threshold, prob
