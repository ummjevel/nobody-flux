"""Local ASR stage: wav in -> Korean text out.

SenseVoice-Small via sherpa-onnx (Apache-2.0, official ko/zh/en/ja/yue support,
non-autoregressive so it's fast even on CPU -- see
docs/output/ondevice_asr_llm_tts_research_20260716.md for why this beat
Vosk/whisper.cpp/sherpa-onnx-zipformer-ko in the research pass).

Model: sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17
  https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from pathlib import Path

# sherpa-onnx's compiled extension dlopen()s a bare "libonnxruntime.so", but the
# onnxruntime pip wheel only ships the versioned "libonnxruntime.so.1.27.0" --
# without a same-directory unversioned symlink, `import sherpa_onnx` raises
# "ImportError: libonnxruntime.so: cannot open shared object file".
# This creates that symlink if missing, but can't fix the other half of the
# problem from inside the process: glibc's dynamic linker parses
# LD_LIBRARY_PATH once at process startup, so setting os.environ here (after
# startup) does NOT make the linker search this directory -- LD_LIBRARY_PATH
# must be set in the shell before `python`/`uv run` launches. See
# scripts/env.sh (source it, or `export $(cat scripts/env.sh | ...)`-style use)
# and the project README.
for _versioned in glob.glob(
    str(
        Path(__file__).resolve().parents[2]
        / ".venv"
        / "lib"
        / "*"
        / "site-packages"
        / "onnxruntime"
        / "capi"
        / "libonnxruntime.so.*"
    )
):
    _unversioned = os.path.join(str(Path(_versioned).parent), "libonnxruntime.so")
    if not os.path.exists(_unversioned):
        os.symlink(os.path.basename(_versioned), _unversioned)

import sherpa_onnx  # noqa: E402 -- needs LD_LIBRARY_PATH set (see above) before this import
import soundfile as sf  # noqa: E402

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "sense-voice"


@dataclass
class NobodyASR:
    model_dir: Path = DEFAULT_MODEL_DIR
    use_int8: bool = True
    language: str = "ko"  # SenseVoice also accepts "auto" for language detection
    use_itn: bool = True  # inverse text normalization: spoken numbers -> digits etc.
    num_threads: int = 2

    def __post_init__(self):
        model_file = "model.int8.onnx" if self.use_int8 else "model.onnx"
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(self.model_dir / model_file),
            tokens=str(self.model_dir / "tokens.txt"),
            num_threads=self.num_threads,
            language=self.language,
            use_itn=self.use_itn,
        )

    def transcribe_file(self, wav_path: str) -> str:
        """Read a wav file (any sample rate; sherpa-onnx resamples internally) and
        return the recognized Korean text.

        Known rough edge: this SenseVoice checkpoint's Korean tokenization inserts
        spurious spaces mid-eojeol ("생각 을" instead of "생각을") -- confirmed against
        the model's own bundled test_wavs/ko.wav. Collapsing whitespace here can't fix
        wrong word boundaries, only repeated spaces; the LLM stage gets the text as-is.
        Revisit with sherpa-onnx-zipformer-ko (pending its #2886 bug fix) or Vosk if
        this proves too noisy for the LLM to work with in practice.
        """
        audio, sample_rate = sf.read(wav_path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        stream = self.recognizer.create_stream()
        stream.accept_waveform(sample_rate, audio)
        self.recognizer.decode_stream(stream)
        return " ".join(stream.result.text.split())
