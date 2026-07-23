"""Loads configs/models.yaml and instantiates named ASR/LLM/TTS presets.

This is the "swap models via config, not code" layer: scripts/run_pipeline.py
and scripts/talk.py both go through build_asr/build_llm/build_tts instead of
constructing NobodyASR/NobodyLLM/NobodyTTS directly, so a new preset in the
yaml is immediately usable from either entrypoint via --asr/--llm/--tts.

Only one preset per stage exists today (see configs/models.yaml) -- this
module doesn't add new models, just the plumbing for future ones.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from . import asr, llm, tts

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "models.yaml"

# Every class a preset's `class:` field is allowed to name. Deliberately a
# fixed allowlist (not getattr-by-string on the modules) so a typo'd or
# malicious class name in the yaml can't reach for something unrelated.
_CLASSES: dict[str, type] = {
    "NobodyASR": asr.NobodyASR,
    "NobodyLLM": llm.NobodyLLM,
    "NobodyTTS": tts.NobodyTTS,
}

# params keys that hold filesystem paths and should be resolved relative to
# PROJECT_ROOT rather than passed through as-is.
_PATH_PARAM_KEYS = {"model_dir", "repo_dir", "reference_audio"}


def _load_config() -> dict[str, Any]:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_params(params: dict[str, Any]) -> dict[str, Any]:
    resolved = dict(params)
    for key in _PATH_PARAM_KEYS & resolved.keys():
        resolved[key] = PROJECT_ROOT / resolved[key]
    return resolved


def _build(stage: str, preset: str | None, overrides: dict[str, Any] | None = None):
    config = _load_config()
    preset = preset or config["defaults"][stage]
    try:
        entry = config[stage][preset]
    except KeyError:
        available = ", ".join(sorted(config[stage]))
        raise ValueError(
            f"Unknown {stage} preset '{preset}'. Available: {available}"
        ) from None

    cls = _CLASSES[entry["class"]]
    params = _resolve_params(entry.get("params", {}))
    params.update(overrides or {})
    return cls(**params)


def build_asr(preset: str | None = None, **overrides) -> asr.NobodyASR:
    return _build("asr", preset, overrides)


def build_llm(preset: str | None = None, **overrides) -> llm.NobodyLLM:
    return _build("llm", preset, overrides)


def build_tts(preset: str | None = None, **overrides) -> tts.NobodyTTS:
    return _build("tts", preset, overrides)


def list_presets(stage: str) -> list[str]:
    return sorted(_load_config()[stage])


def default_preset(stage: str) -> str:
    """The preset name build_{stage}(None) actually resolves to.

    NOT list_presets(stage)[0] -- that's alphabetical order and only happens
    to match today because each stage has exactly one preset. Once a stage
    has two, that shortcut silently reports the wrong name to callers that
    just want to log/display which preset is active (see scripts/talk.py).
    """
    return _load_config()["defaults"][stage]
