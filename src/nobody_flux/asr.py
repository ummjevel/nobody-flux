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
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ._procio import LineReader, StderrDrainer, clean_subprocess_env
from .paths import PROJECT_ROOT

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
    str(PROJECT_ROOT / ".venv" / "lib" / "*" / "site-packages" / "onnxruntime" / "capi" / "libonnxruntime.so.*")
):
    _unversioned = os.path.join(str(Path(_versioned).parent), "libonnxruntime.so")
    if not os.path.exists(_unversioned):
        os.symlink(os.path.basename(_versioned), _unversioned)

import sherpa_onnx  # noqa: E402 -- needs LD_LIBRARY_PATH set (see above) before this import
import soundfile as sf  # noqa: E402

DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "sense-voice"


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


VIBEASR_REPO_DIR = PROJECT_ROOT / "external" / "VibeASR.cpp"
DEFAULT_VIBEASR_VAE_MODEL = PROJECT_ROOT / "models" / "vibeasr" / "vibeasr-vae-encoder-i8_s.gguf"
DEFAULT_VIBEASR_LM_MODEL = PROJECT_ROOT / "models" / "vibeasr" / "vibeasr-lm-i2_s-embed-q6_k.gguf"


@dataclass
class VibeAsrBitnet:
    """Second ASR candidate: microsoft/VibeVoice-ASR-BitNet (arXiv:2607.21075),
    run through its official runtime microsoft/VibeASR.cpp (CPU-only ggml/BitNet,
    ARM NEON + x86 AVX2 kernels -- directly relevant to the CM4 on-device target).
    See docs/output/ondevice_asr_llm_tts_research_20260716.md (predates this
    model; added as a follow-up candidate).

    On this exact test wav it doesn't have SenseVoice-Small's spurious
    mid-eojeol spacing bug (see NobodyASR.transcribe_file's docstring) --
    "조금만 생각을 하면서..." vs "조 금만 생각 을 하 면서...".

    Backed by VibeASR.cpp's own `asr_stream_server` binary: a persistent
    process that loads both GGUF models once and then answers requests over
    stdin/stdout (protocol: send a wav path per line, get text back
    terminated by "---END---" -- see asr_server.cpp's header comment). Started
    lazily on first transcribe_file() call, reused across calls, and must be
    shut down via close() (STSPipeline.close() does this) -- otherwise it
    lingers as an orphaned process holding GPU/CPU memory. This cut per-turn
    latency from ~8.7s (previous per-call `asr_infer` subprocess, reloading
    both models every time) to ~2.1s (8 threads, see num_threads below) after
    the one-time ~5.7s startup cost.

    Setup (see scripts/setup_common.sh step 5): clone microsoft/VibeASR.cpp
    --recursive into `repo_dir`, build it (both `asr_infer` and
    `asr_stream_server` CMake targets), and download the two GGUF files from
    huggingface.co/microsoft/VibeVoice-ASR-BitNet into `vae_model`/`lm_model`'s
    directory (~1.7GB total; skip the 10.7GB SafeTensors checkpoint, it's only
    needed for re-quantizing).

    Known local patch: upstream src/vae.cpp hardcodes a 128GB ggml context
    (`vae_ctx_mem_size`) on non-Windows, assuming Linux overcommit -- this
    aborts with ENOMEM in posix_memalign on any box whose RAM+swap is under
    128GB (confirmed on this WSL2 dev box: 64GB RAM + 16GB swap = 80GB). Patched
    the clone down to 8GB (same order as the Windows branch's already-patched
    6GB) -- setup_common.sh reapplies this patch on fresh clones.
    """

    repo_dir: Path = VIBEASR_REPO_DIR
    vae_model: Path = DEFAULT_VIBEASR_VAE_MODEL
    lm_model: Path = DEFAULT_VIBEASR_LM_MODEL
    # Benchmarked on this 28-core dev box: 4 threads -> ~2.7s/request, 8 ->
    # ~2.1s, 16 -> ~2.0s (diminishing returns), 28 -> ~3.4s (*worse* --
    # thread-management overhead outweighs the extra parallelism once you're
    # oversubscribing every logical core on the box). 8 is the sweet spot:
    # most of the speedup, without hogging every core on the machine.
    num_threads: int = 8
    startup_timeout_seconds: float = 60.0
    request_timeout_seconds: float = 60.0

    def __post_init__(self):
        self._proc: subprocess.Popen | None = None
        self._stdout_reader: LineReader | None = None
        self._stderr_drainer: StderrDrainer | None = None

    def _binary(self) -> Path:
        binary = self.repo_dir / "build" / "bin" / "asr_stream_server"
        if not binary.exists():
            raise FileNotFoundError(
                f"asr_stream_server not built at {binary}. Run scripts/setup_common.sh "
                "(step 5) to clone+build microsoft/VibeASR.cpp, or see "
                "src/nobody_flux/asr.py's VibeAsrBitnet docstring."
            )
        return binary

    def _ensure_started(self) -> subprocess.Popen:
        if self._proc is not None and self._proc.poll() is None:
            return self._proc

        # --greedy: deterministic decoding, matching what run_pipeline.py's
        # preset-comparison use case needs (same input -> same output across
        # runs). --no-token-stream: one text blob per request instead of one
        # line per token -- simpler to parse, and we don't need token-level
        # streaming here.
        self._proc = subprocess.Popen(
            [
                str(self._binary()),
                "--vae-model",
                str(self.vae_model),
                "--lm-model",
                str(self.lm_model),
                "-t",
                str(self.num_threads),
                "--greedy",
                "--no-token-stream",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            # bufsize=1: line-buffer *our* end of the pipes (Python-side).
            # This does NOT control the child's own stdio buffering -- that's
            # up to asr_stream_server itself, which flushes after each line
            # it cares about us seeing (---READY---, ---END---, etc).
            bufsize=1,
            # General hygiene, see clean_subprocess_env()'s docstring -- not
            # the fix for the hang below, that's StderrDrainer.
            env=clean_subprocess_env(),
        )
        # Both readers MUST start immediately, before the first get_line()
        # call below -- see LineReader/StderrDrainer's docstrings for the two
        # different (and differently-shaped) hangs this avoids.
        self._stdout_reader = LineReader(self._proc.stdout)
        self._stderr_drainer = StderrDrainer(self._proc.stderr)

        try:
            ready = self._stdout_reader.get_line(self.startup_timeout_seconds)
        except TimeoutError as exc:
            self._proc.kill()
            raise RuntimeError(f"asr_stream_server didn't start in time: {exc}") from exc
        if ready.strip() != "---READY---":
            self._proc.kill()
            raise RuntimeError(
                f"asr_stream_server failed to start ({ready!r}):\n{self._stderr_drainer.tail()[-2000:]}"
            )
        return self._proc

    def transcribe_file(self, wav_path: str) -> str:
        """Read a wav file and return the recognized text."""
        proc = self._ensure_started()
        proc.stdin.write(f"{wav_path}\n")
        proc.stdin.flush()

        lines = []
        while True:
            line = self._stdout_reader.get_line(self.request_timeout_seconds)
            if not line:
                raise RuntimeError(
                    f"asr_stream_server exited unexpectedly:\n{self._stderr_drainer.tail()[-2000:]}"
                )
            if line.strip() == "---END---":
                break
            lines.append(line)

        text = "".join(lines).strip()
        if text.startswith("[ERROR]"):
            raise RuntimeError(f"asr_stream_server: {text}")
        return text

    def close(self) -> None:
        if self._proc is None or self._proc.poll() is not None:
            return
        try:
            self._proc.stdin.write("EXIT\n")
            self._proc.stdin.flush()
            self._proc.wait(timeout=5)
        except Exception:
            self._proc.kill()
