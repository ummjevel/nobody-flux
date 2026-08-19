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

import copy
import os
import typing
from pathlib import Path
from typing import Any

import yaml

from .audio import session as audio_session
from .paths import PROJECT_ROOT
from .stage import asr, llm, tts
from .turn import detector as turn_detector
from .turn import vad

CONFIG_PATH = PROJECT_ROOT / "configs" / "models.yaml"
VOICES_CONFIG_PATH = PROJECT_ROOT / "configs" / "voices.yaml"
VAD_CONFIG_PATH = PROJECT_ROOT / "configs" / "vad.yaml"
TURN_DETECTOR_CONFIG_PATH = PROJECT_ROOT / "configs" / "turn_detector.yaml"
AUDIO_CONFIG_PATH = PROJECT_ROOT / "configs" / "audio.yaml"
STREAMING_ASR_CONFIG_PATH = PROJECT_ROOT / "configs" / "streaming_asr.yaml"
RUNTIME_CONFIG_PATH = PROJECT_ROOT / "configs" / "runtime.yaml"

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
    "SherpaSupertonicTts": tts.SherpaSupertonicTts,
}


# (mtime, parsed) per path. models.yaml alone was being parsed six times per
# startup (three _build calls + three _cli.py default lookups); besides the
# waste, uncached reads meant three stages could in principle see three
# different file contents if the yaml were edited mid-startup. The mtime key
# keeps long-lived callers (REPL, calibration scripts) able to pick up edits.
_YAML_CACHE: dict[Path, tuple[float, dict[str, Any]]] = {}


def _load_yaml(path: Path) -> dict[str, Any]:
    """Parsed yaml, mtime-cached, returned as a fresh deep copy every call.

    The copy is not defensive politeness, it is required. Callers here treat the
    returned dict as scratch space -- `config.update(overrides)`,
    `config.pop(engine_block)` -- and handing out the cached object let those
    edits persist into the cache. Measured before the fix: loading
    streaming_asr.yaml, applying one build's overrides, then loading it again
    returned `num_threads: 99` and no engine blocks at all.

    Both halves of that are silent. The leaked override makes a second build in
    the same process quietly inherit the first one's tuning, and the missing
    engine block makes it fall back to dataclass defaults -- which for VAD means
    presenting an uncalibrated threshold as the configured one. Nothing raises;
    the numbers just stop meaning what they say. talk.py builds each stage once
    so it was never hit there, but benchmark.py and the _ab_* scripts build
    repeatedly in one process.

    deepcopy rather than dict(...) because these configs are nested (per-engine
    blocks, per-preset params) and a shallow copy would still share the inner
    dicts, which is exactly what gets popped.
    """
    mtime = os.path.getmtime(path)
    cached = _YAML_CACHE.get(path)
    if cached is not None and cached[0] == mtime:
        return copy.deepcopy(cached[1])
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    _YAML_CACHE[path] = (mtime, data)
    return copy.deepcopy(data)


def stage_threads(stage: str) -> int | None:
    """Thread count for a stage under configs/runtime.yaml's CPU budget, or
    None when the budget doesn't cover that stage.

    threads = clamp(1, cpu_budget * fraction, cap). This is what replaces the
    per-class hardcoded defaults that were tuned on a 28-core dev box and
    added up to ~14 threads on the 4-core CM4 target (code-review #9). A
    preset that sets n_threads explicitly in models.yaml still wins -- see
    _build; this is the *derived default*, not an override.

    NOBODY_CPU_BUDGET overrides the yaml's cpu_budget for one process. That
    exists to answer "what will the 4-core CM4 see" from a dev box --
    `NOBODY_CPU_BUDGET=4 python scripts/_ab_persona.py` -- without editing a
    tracked config mid-measurement. It bounds thread counts only: core count
    is one of several things that differ from the real board (ISA, clock,
    memory bandwidth), so treat the result as an upper bound on CM4 speed,
    not a prediction of it.
    """
    if not RUNTIME_CONFIG_PATH.exists():
        return None
    config = _load_yaml(RUNTIME_CONFIG_PATH) or {}
    entry = (config.get("stages") or {}).get(stage)
    if entry is None:
        return None
    override = os.environ.get("NOBODY_CPU_BUDGET")
    budget = (
        int(override) if override else config.get("cpu_budget") or os.cpu_count() or 4
    )
    threads = max(1, int(budget * float(entry.get("fraction", 1.0))))
    cap = entry.get("cap")
    if cap is not None:
        threads = min(threads, int(cap))
    return threads


def _inject_thread_budget(stage: str, cls: type, params: dict[str, Any]) -> None:
    """Fill in the class's thread-count parameter from the runtime budget,
    unless the preset/override already pinned one (explicit config beats a
    derived default). Field name differs per runtime (llama.cpp: n_threads,
    sherpa/onnxruntime: num_threads), so it's discovered from the class's own
    annotations rather than assumed."""
    hints = typing.get_type_hints(cls)
    field_name = next((n for n in ("n_threads", "num_threads") if n in hints), None)
    if field_name is None or field_name in params:
        return
    threads = stage_threads(stage)
    if threads is not None:
        params[field_name] = threads


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

    _inject_thread_budget(stage, cls, params)
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
    """Build the VAD from configs/vad.yaml.

    VoiceActivityDetector isn't a named preset like asr/llm/tts -- it's a
    config's worth of tunable numeric parameters -- so this reads the yaml into
    the constructor rather than going through _build()'s preset-name
    indirection. **overrides works the same way as build_asr/build_llm/
    build_tts's, for one-off tuning without editing the yaml.

    Two engines now exist, selected by the yaml's `engine` key, for the licence
    reason in vad._ENGINES. Structure mirrors configs/streaming_asr.yaml:
    engine-independent knobs at the top level, engine-specific ones in a block
    named after the engine, and only the selected block is merged. Here that
    matters for `threshold` specifically -- the 0.5 in the ten-vad block is a
    measurement from this room's microphone, and letting it leak onto a
    different model would present an uncalibrated number as a calibrated one.

    Unlike build_streaming_transcriber, overrides are applied *after* the block
    merge. scripts/_calibrate_vad_threshold.py sweeps `threshold` through this
    function, and a block value that outranked the sweep would make every
    iteration measure the same number.
    """
    config = _load_yaml(VAD_CONFIG_PATH)
    engine = overrides.pop("engine", None) or config.pop("engine", "ten-vad")
    config.pop("engine", None)

    engine_params = {}
    for name in vad.VAD_ENGINES:
        block = config.pop(name, None) or {}
        if name == engine:
            engine_params = block
    params = {**config, **engine_params, **overrides, "engine": engine}

    if params.get("model_path"):
        # Written relative to the project root in the yaml, the same convention
        # _build() applies to preset params.
        params["model_path"] = PROJECT_ROOT / params["model_path"]

    # A key the constructor has no field for is a config error worth reporting
    # rather than dropping -- silently dropping is how a knob appears tuned
    # while doing nothing. Same guard as build_streaming_transcriber's.
    accepted = set(vad.VoiceActivityDetector.__dataclass_fields__)
    unknown = sorted(set(params) - accepted)
    if unknown:
        raise ValueError(
            f"configs/vad.yaml sets {unknown} which VoiceActivityDetector "
            f"(engine: {engine}) has no field for. Move them under the engine "
            f"block they belong to, or remove them."
        )
    return vad.VoiceActivityDetector(**params)


def build_audio_session(backend: str | None = None) -> audio_session.AudioSession:
    """Build the duplex/AEC audio session for talk.py's mic loop from
    configs/audio.yaml (see audio.py). `backend` overrides the yaml's `backend`
    field (e.g. from talk.py's --aec); 'auto' resolves per platform + installed
    libs. Not a named preset like asr/llm/tts -- there's one config's worth of
    knobs, same flat-yaml pattern as build_vad/build_turn_detector. The session's
    stream isn't opened until .start()."""
    config = _load_yaml(AUDIO_CONFIG_PATH)
    prefer = backend or config.get("backend", "auto")
    resolved = audio_session.select_backend(prefer)
    delay_ms = config.get("delay_ms")
    return audio_session.build_session(
        resolved,
        delay_frames=int(config.get("delay_frames", 4)),
        delay_ms=float(delay_ms) if delay_ms is not None else None,
    )


def build_streaming_transcriber(**overrides):
    """Build the Phase 3 live recognizer from configs/streaming_asr.yaml.

    Same flat-config pattern as build_vad/build_turn_detector -- a config's
    worth of knobs rather than a named preset. Kept separate from the `asr`
    presets in models.yaml on purpose: those are batch, file-in/text-out stages
    that benchmark.py compares against each other, while this consumes a live
    frame stream and cannot be substituted for one. Conflating them would put an
    object in the preset table that half the callers could not actually use.

    There are now two engines, selected by the yaml's `engine` key, and they are
    not interchangeable in quality -- see configs/streaming_asr.yaml for which
    to use and why. They share the `accept_frame`/`committed`/`hypothesis`/
    `endpoint_detected`/`finalize` contract, so callers need no branch; the one
    behavioural difference is that `chunked-sensevoice` never reports an
    endpoint of its own.

    Imported lazily so that merely importing registry does not construct a
    sherpa-onnx recognizer for the majority of callers who never enable
    streaming ASR.
    """
    from .stage import asr_stream

    config = _load_yaml(STREAMING_ASR_CONFIG_PATH)
    config.update(overrides)

    # A fixed allowlist rather than getattr-by-string, for the same reason
    # _CLASSES above is one: a typo'd or hostile engine name in the yaml must
    # not be able to reach for something unrelated.
    engines = {
        "zipformer": asr_stream.StreamingTranscriber,
        "chunked-sensevoice": asr_stream.ChunkedSenseVoiceTranscriber,
    }
    engine = config.pop("engine", "zipformer")
    cls = _lookup(engines, engine, "streaming ASR engine")

    # Shared knobs live at the top level; each engine's own knobs live in a
    # block named after it. Only the selected engine's block is merged in, and
    # the others are dropped rather than unioned -- both engines have a
    # `model_dir` field pointing at different checkpoints, so unioning them let
    # the zipformer path's model_dir reach the SenseVoice constructor. That
    # happened, and it is the shape of bug that loads the wrong weights quietly
    # somewhere else.
    engine_params = {}
    for name in engines:
        block = config.pop(name, None) or {}
        if name == engine:
            engine_params = block
    params = {**config, **engine_params}

    if "model_dir" in params:
        # Written relative to the project root in the yaml, the same convention
        # _build() applies to preset params.
        params["model_dir"] = PROJECT_ROOT / params["model_dir"]

    # A shared key the selected engine has no field for is a config error worth
    # reporting, not something to silently drop -- silently dropping is how
    # `rule2_min_trailing_silence` would appear to be tuned while doing nothing.
    accepted = set(getattr(cls, "__dataclass_fields__", {}))
    unknown = sorted(set(params) - accepted)
    if unknown:
        raise ValueError(
            f"configs/streaming_asr.yaml sets {unknown} which "
            f"{cls.__name__} (engine: {engine}) has no field for. Move them "
            f"under the engine block they belong to, or remove them."
        )
    return cls(**params)


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
