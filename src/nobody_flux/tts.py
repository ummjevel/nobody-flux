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

import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

MOSS_TTS_NANO_REPO = Path(__file__).resolve().parents[2] / "external" / "MOSS-TTS-Nano"
DEFAULT_REFERENCE_AUDIO = (
    Path(__file__).resolve().parents[2] / "data" / "reference_voice_16k.wav"
)


@dataclass
class NobodyTTS:
    repo_dir: Path = MOSS_TTS_NANO_REPO
    reference_audio: Path = DEFAULT_REFERENCE_AUDIO
    # "auto" (i.e. CUDA if visible) crashes on the RTX 5090 dev box with
    # "no kernel image is available for execution on the device": MOSS-TTS-Nano's
    # own venv pins torch==2.7.0/cu126, whose prebuilt kernels predate Blackwell
    # (sm_120) -- confirmed by hand, not a hypothetical. CPU sidesteps this
    # entirely and matches the model's own official spec ("4-core CPU, zero GPU
    # realtime" -- see docs/output/ondevice_asr_llm_tts_research_20260716.md),
    # so this isn't even a compromise. Override to "cuda" explicitly on
    # hardware where MOSS-TTS-Nano's pinned torch build actually has matching
    # kernels (e.g. H100/sm_90, well within cu126's supported range).
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
