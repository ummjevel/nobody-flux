"""Loads configs/models.yaml and instantiates named ASR/LLM/TTS presets.
Also loads configs/voices.yaml for TTS reference-clip selection (resolve_voice)
and configs/vad.yaml for VoiceActivityDetector's tunable parameters.

This is the "swap models via config, not code" layer: scripts/run_pipeline.py
and scripts/talk.py both go through build_asr/build_llm/build_tts instead of
constructing NobodyASR/NobodyLLM/NobodyTTS directly, so a new preset in the
yaml is immediately usable from either entrypoint via --asr/--llm/--tts.

Only one preset per stage exists today (see configs/models.yaml) -- this
module doesn't add new models, just the plumbing for future ones.
"""

from __future__ import annotations

import typing
from pathlib import Path
from typing import Any

import yaml

from . import asr, audio, llm, tts, turn_detector, vad
from .paths import PROJECT_ROOT

CONFIG_PATH = PROJECT_ROOT / "configs" / "models.yaml"
VOICES_CONFIG_PATH = PROJECT_ROOT / "configs" / "voices.yaml"
VAD_CONFIG_PATH = PROJECT_ROOT / "configs" / "vad.yaml"
TURN_DETECTOR_CONFIG_PATH = PROJECT_ROOT / "configs" / "turn_detector.yaml"
AUDIO_CONFIG_PATH = PROJECT_ROOT / "configs" / "audio.yaml"

# Every class a preset's `class:` field is allowed to name. Deliberately a
# fixed allowlist (not getattr-by-string on the modules) so a typo'd or
# malicious class name in the yaml can't reach for something unrelated.
_CLASSES: dict[str, type] = {
    "NobodyASR": asr.NobodyASR,
    "VibeAsrBitnet": asr.VibeAsrBitnet,
    "StreamingZipformerAsr": asr.StreamingZipformerAsr,
    "NobodyLLM": llm.NobodyLLM,
    "NobodyLLMGguf": llm.NobodyLLMGguf,
    "NobodyTTS": tts.NobodyTTS,
    "FreyaTtsKo": tts.FreyaTtsKo,
    "SherpaMatchaTts": tts.SherpaMatchaTts,
}


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _lookup(mapping: dict[str, Any], key: str, label: str) -> Any:
    """mapping[key], or a ValueError listing what's actually available.

    Shared by preset lookup (_build) and voice lookup (resolve_voice) so both
    fail the same way instead of each hand-rolling its own KeyError->ValueError.
    """
    try:
        return mapping[key]
    except KeyError:
        available = ", ".join(sorted(mapping))
        raise ValueError(f"Unknown {label} '{key}'. Available: {available}") from None


def _path_field_names(cls: type) -> set[str]:
    """Constructor params of cls that are typed Path -- these get resolved
    relative to PROJECT_ROOT. Derived from the dataclass's own annotations
    (via typing.get_type_hints, which resolves the `from __future__ import
    annotations` string form back to real types) instead of a hand-maintained
    set of param names, so a new Path-typed field is picked up automatically
    instead of silently bypassing resolution until someone remembers to list it.
    """
    return {name for name, hint in typing.get_type_hints(cls).items() if hint is Path}


def _build(stage: str, preset: str | None, overrides: dict[str, Any] | None = None):
    config = _load_yaml(CONFIG_PATH)
    preset = preset or config["defaults"][stage]
    entry = _lookup(config[stage], preset, stage)

    cls = _CLASSES[entry["class"]]
    params = dict(entry.get("params", {}))
    params.update(overrides or {})

    # Applied to the merged dict (preset params + overrides) so both go
    # through the same resolution -- previously overrides skipped this
    # entirely and only worked because callers happened to pre-resolve
    # (see resolve_voice). Re-resolving an already-absolute Path is a no-op:
    # PROJECT_ROOT / absolute_path == absolute_path (pathlib joinpath
    # semantics), so this is safe for both relative yaml values and
    # already-resolved override values.
    for key in _path_field_names(cls) & params.keys():
        params[key] = PROJECT_ROOT / params[key]

    return cls(**params)


def build_asr(preset: str | None = None, **overrides) -> asr.NobodyASR:
    return _build("asr", preset, overrides)


def build_llm(preset: str | None = None, **overrides) -> llm.NobodyLLM:
    return _build("llm", preset, overrides)


def build_tts(preset: str | None = None, **overrides) -> tts.NobodyTTS:
    return _build("tts", preset, overrides)


def list_presets(stage: str) -> list[str]:
    return sorted(_load_yaml(CONFIG_PATH)[stage])


def default_preset(stage: str) -> str:
    """The preset name build_{stage}(None) actually resolves to.

    NOT list_presets(stage)[0] -- that's alphabetical order and only happens
    to match today because each stage has exactly one preset. Once a stage
    has two, that shortcut silently reports the wrong name to callers that
    just want to log/display which preset is active.
    """
    return _load_yaml(CONFIG_PATH)["defaults"][stage]


def resolve_voice(name: str | None = None) -> Path:
    """A voice is a TTS *parameter* (reference_audio), not a separate preset --
    pass the result to build_tts(..., reference_audio=resolve_voice(name)).

    Files aren't committed (see .gitignore); this raises FileNotFoundError
    with the expected path if the voice hasn't been dropped in yet, rather
    than letting NobodyTTS fail deep inside a subprocess call with a less
    obvious error.
    """
    config = _load_yaml(VOICES_CONFIG_PATH)
    name = name or config["default"]
    entry = _lookup(config["voices"], name, "voice")

    path = PROJECT_ROOT / entry["path"]
    if not path.exists():
        raise FileNotFoundError(
            f"Voice '{name}' is registered in configs/voices.yaml but {path} "
            "doesn't exist yet -- place the reference wav there first."
        )
    return path


def list_voices() -> list[str]:
    return sorted(_load_yaml(VOICES_CONFIG_PATH)["voices"])


def build_vad(**overrides) -> vad.VoiceActivityDetector:
    """VoiceActivityDetector isn't a named preset like asr/llm/tts (there's
    only one implementation, TEN-VAD -- see vad.py), just a config's worth of
    tunable numeric parameters, so this reads configs/vad.yaml as a flat dict
    straight into the constructor rather than going through _build()'s
    preset-name indirection. **overrides works the same way as
    build_asr/build_llm/build_tts's, for one-off tuning without editing the
    yaml (e.g. from a REPL)."""
    config = _load_yaml(VAD_CONFIG_PATH)
    config.update(overrides)
    return vad.VoiceActivityDetector(**config)


def build_audio_session(backend: str | None = None) -> audio.AudioSession:
    """Build the duplex/AEC audio session for talk.py's mic loop from
    configs/audio.yaml (see audio.py). `backend` overrides the yaml's `backend`
    field (e.g. from talk.py's --aec); 'auto' resolves per platform + installed
    libs. Not a named preset like asr/llm/tts -- there's one config's worth of
    knobs, same flat-yaml pattern as build_vad/build_turn_detector. The session's
    stream isn't opened until .start()."""
    config = _load_yaml(AUDIO_CONFIG_PATH)
    prefer = backend or config.get("backend", "auto")
    resolved = audio.select_backend(prefer)
    return audio.build_session(resolved, delay_frames=int(config.get("delay_frames", 4)))


def build_turn_detector(**overrides) -> turn_detector.TurnDetector:
    """Same flat-config pattern as build_vad -- Smart Turn v3 is the only
    turn-detector implementation, so this reads configs/turn_detector.yaml
    straight into the constructor rather than being a named preset. Loading
    the model (onnxruntime session + Whisper feature extractor) happens in
    TurnDetector.__post_init__, so this is only worth calling when a caller
    actually wants endpoint detection (talk.py builds it lazily)."""
    config = _load_yaml(TURN_DETECTOR_CONFIG_PATH)
    config.update(overrides)
    return turn_detector.TurnDetector(**config)
