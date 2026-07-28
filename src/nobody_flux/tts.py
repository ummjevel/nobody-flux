"""Local TTS stage: Korean text in -> wav out.

MOSS-TTS-Nano (OpenMOSS/Fudan, Apache-2.0, 100M params, official CPU-realtime
"4-core, zero GPU" spec, Korean in its 20 supported languages -- see
docs/output/ondevice_asr_llm_tts_research_20260716.md). Shelled out to via its
own infer.py rather than imported as a library: its `moss_tts_nano` package is
really a script-style CLI tool (see MOSS-TTS-Nano/pyproject.toml's `py-modules`),
and driving it as a subprocess sidesteps import-order/global-state assumptions
in code we don't control.

Voice: voice_clone mode needs a reference clip (there's no bundled default
voice, and no Korean sample ships with the repo -- see assets/demo.jsonl,
Chinese-only). Standing in with a placeholder enrollment recording for this
prototype; swap for a properly licensed/chosen persona voice before anything
user-facing.

WeTextProcessing is disabled (--disable-wetext-processing): its language
resolver only ever returns "zh" or "en" (see
MOSS-TTS-Nano/text_normalization_pipeline.py:resolve_text_normalization_language)
so plain Korean text would silently fall through to the *Chinese* ITN
normalizer -- and avoiding it also means we don't need the pynini/
WeTextProcessing native-dependency install. The repo's own regex-only
`normalize_tts_text` robustness pass (language-agnostic, no pynini) still runs.

Interpreter: this shells out with `repo_dir/.venv/bin/python`, NOT this
project's own `sys.executable`. MOSS-TTS-Nano pins torch==2.7.0/transformers
exactly; nobody-flux's own venv tracks a newer torch independently (currently
2.13.0) for the LLM stage, and letting `uv sync` reconcile a single shared venv
against both pins would force one project's torch version to lose -- it
already happened once (see scripts/setup_common.sh, which creates this
isolated venv). If repo_dir has no .venv, this falls back to sys.executable
with a warning, which only works if someone manually installed moss_tts_nano
into this project's own venv (fragile -- uv sync will silently uninstall it
again, since it isn't a declared dependency here).
"""

from __future__ import annotations

import json
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

from ._procio import LineReader, StderrDrainer, clean_subprocess_env
from .paths import PROJECT_ROOT

MOSS_TTS_NANO_REPO = PROJECT_ROOT / "external" / "MOSS-TTS-Nano"
DEFAULT_REFERENCE_AUDIO = PROJECT_ROOT / "data" / "reference_voice_16k.wav"


@dataclass
class NobodyTTS:
    repo_dir: Path = MOSS_TTS_NANO_REPO
    reference_audio: Path = DEFAULT_REFERENCE_AUDIO
    # Real default lives in configs/models.yaml's tts preset params, not here --
    # this is just the fallback for constructing NobodyTTS directly (tests,
    # a REPL). "auto" (CUDA if visible) crashes on the RTX 5090 dev box with
    # "no kernel image is available for execution on the device": MOSS-TTS-Nano's
    # own venv pins torch==2.7.0/cu126, whose prebuilt kernels predate Blackwell
    # (sm_120) -- confirmed by hand, not a hypothetical. CPU sidesteps this
    # entirely and matches the model's own official spec ("4-core CPU, zero GPU
    # realtime"), so this isn't even a compromise. Hardware where MOSS-TTS-Nano's
    # pinned torch build actually has matching kernels (e.g. H100/sm_90, well
    # within cu126's range) can override via --tts preset params or a new
    # preset, not a code edit -- that's the point of configs/models.yaml.
    device: str = "cpu"
    # infer.py loads its own model on each invocation (no server/warm process),
    # so a stuck load or hung generation would otherwise block forever with no
    # feedback. 120s covers a cold model load plus generation for this
    # persona's short replies with headroom to spare.
    timeout_seconds: float = 120.0

    def _interpreter(self) -> str:
        venv_python = self.repo_dir / ".venv" / "bin" / "python"
        if venv_python.exists():
            return str(venv_python)
        warnings.warn(
            f"No isolated venv at {venv_python}; falling back to {sys.executable}. "
            "Run scripts/setup_common.sh to create MOSS-TTS-Nano's own venv.",
            stacklevel=2,
        )
        return sys.executable

    def synthesize(self, text: str, out_path: str) -> str:
        """Synthesize `text` (Korean) to a 48kHz stereo wav at `out_path`."""
        cmd = [
            self._interpreter(),
            str(self.repo_dir / "infer.py"),
            "--prompt-audio-path",
            str(self.reference_audio),
            "--text",
            text,
            "--out",
            out_path,
            "--disable-wetext-processing",
            "--device",
            self.device,
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.timeout_seconds
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"MOSS-TTS-Nano timed out after {self.timeout_seconds}s"
            ) from exc
        if result.returncode != 0:
            raise RuntimeError(f"MOSS-TTS-Nano failed:\n{result.stderr[-2000:]}")
        return out_path


FREYATTS_VENV_DIR = PROJECT_ROOT / "external" / "freyatts-venv"
DEFAULT_FREYATTS_MODEL_DIR = PROJECT_ROOT / "models" / "freyatts-ko-voiceA"
FREYATTS_SERVER_SCRIPT = PROJECT_ROOT / "scripts" / "_freyatts_server.py"


@dataclass
class FreyaTtsKo:
    """Second TTS candidate: FreyaTTS (github.com/ummjevel/FreyaTTS), a Korean
    fork distilled from Qwen3-TTS, checkpoint "voiceA" (from a sibling project,
    voice-announce-mcp -- see its models/freyatts-ko-voiceA/, copied into this
    project's models/ dir). Flow-matching DiT + frozen VoxCPM2 AudioVAE; no
    speaker embedding -- the noise seed *is* the voice (seed 9 = voiceA, per
    that project's confirmed_voices/best_seeds.json).

    Unlike MOSS-TTS-Nano, this doesn't pin an exact torch version (just
    "torch" in its deps), so its isolated venv resolved torch==2.13.0+cu130 --
    which DOES have sm_120 (RTX 5090/Blackwell) kernels, unlike MOSS-TTS-Nano's
    forced torch==2.7.0+cu126. CUDA actually works here.

    Backed by scripts/_freyatts_server.py, a persistent process (in FreyaTTS's
    own isolated venv) that loads the model once via FreyaTTS.from_pretrained
    and then answers requests over stdin/stdout (JSON lines in, JSON lines
    out -- see that script's header comment). Started lazily on first
    synthesize() call, reused across calls, and must be shut down via close()
    (STSPipeline.close() does this) -- otherwise it lingers as an orphaned
    process holding GPU memory. This cut per-turn latency from ~7.4s (previous
    per-call subprocess, reloading the model + re-fetching the VoxCPM2 AudioVAE
    every time) to ~0.7-1.4s after the one-time ~5.7s startup cost.

    Setup (see scripts/setup_common.sh step 6): `uv venv --python 3.11
    external/freyatts-venv && uv pip install --python
    external/freyatts-venv/bin/python 'freyatts @
    git+https://github.com/ummjevel/FreyaTTS.git' soundfile` (freyatts is a
    normal pip package, no repo clone needed -- simpler setup than
    MOSS-TTS-Nano's).
    """

    venv_dir: Path = FREYATTS_VENV_DIR
    model_dir: Path = DEFAULT_FREYATTS_MODEL_DIR
    device: str = "cuda"
    steps: int = 32
    seed: int = 9
    # from_pretrained() also fetches the frozen VoxCPM2 AudioVAE from HF on
    # first use (cached after); budget for that plus a cold model load. Only
    # paid once (server startup), not per request.
    startup_timeout_seconds: float = 60.0
    request_timeout_seconds: float = 60.0

    def __post_init__(self):
        self._proc: subprocess.Popen | None = None
        self._stdout_reader: LineReader | None = None
        self._stderr_drainer: StderrDrainer | None = None

    def _interpreter(self) -> str:
        venv_python = self.venv_dir / "bin" / "python"
        if not venv_python.exists():
            raise FileNotFoundError(
                f"No freyatts venv at {venv_python}. Create it with "
                "`uv venv --python 3.11 external/freyatts-venv && uv pip install "
                "--python external/freyatts-venv/bin/python "
                "'freyatts @ git+https://github.com/ummjevel/FreyaTTS.git' soundfile` "
                "(see src/nobody_flux/tts.py's FreyaTtsKo docstring)."
            )
        return str(venv_python)

    def _ensure_started(self) -> subprocess.Popen:
        if self._proc is not None and self._proc.poll() is None:
            return self._proc

        self._proc = subprocess.Popen(
            [
                self._interpreter(),
                str(FREYATTS_SERVER_SCRIPT),
                "--model-dir",
                str(self.model_dir),
                "--device",
                self.device,
                "--steps",
                str(self.steps),
                "--seed",
                str(self.seed),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            # bufsize=1: line-buffer *our* end of the pipes (Python-side);
            # scripts/_freyatts_server.py flushes explicitly after every
            # line it writes, so this is enough to see each response promptly.
            bufsize=1,
            # General hygiene, see clean_subprocess_env()'s docstring -- not
            # the fix for the deadlock avoided below by StderrDrainer.
            env=clean_subprocess_env(),
        )
        # Both readers MUST start immediately, before the first get_line()
        # call below -- see LineReader/StderrDrainer's docstrings in
        # _procio.py for the two different (and differently-shaped) hangs
        # this avoids.
        self._stdout_reader = LineReader(self._proc.stdout)
        self._stderr_drainer = StderrDrainer(self._proc.stderr)

        try:
            ready = self._stdout_reader.get_line(self.startup_timeout_seconds)
        except TimeoutError as exc:
            self._proc.kill()
            raise RuntimeError(f"freyatts server didn't start in time: {exc}") from exc
        if ready.strip() != "---READY---":
            self._proc.kill()
            raise RuntimeError(
                f"freyatts server failed to start ({ready!r}):\n{self._stderr_drainer.tail()[-2000:]}"
            )
        return self._proc

    def synthesize(self, text: str, out_path: str) -> str:
        """Synthesize `text` (Korean) to a 48kHz mono wav at `out_path`."""
        proc = self._ensure_started()
        proc.stdin.write(json.dumps({"text": text, "out": out_path}) + "\n")
        proc.stdin.flush()

        try:
            line = self._stdout_reader.get_line(self.request_timeout_seconds)
        except TimeoutError as exc:
            raise RuntimeError(f"freyatts server request timed out: {exc}") from exc
        if not line:
            raise RuntimeError(
                f"freyatts server exited unexpectedly:\n{self._stderr_drainer.tail()[-2000:]}"
            )

        response = json.loads(line)
        if not response.get("ok"):
            raise RuntimeError(f"freyatts server: {response.get('error')}")
        return out_path

    def close(self) -> None:
        if self._proc is None or self._proc.poll() is not None:
            return
        try:
            self._proc.stdin.write(json.dumps({"cmd": "exit"}) + "\n")
            self._proc.stdin.flush()
            self._proc.wait(timeout=5)
        except Exception:
            self._proc.kill()
