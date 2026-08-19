"""registry._load_yaml's caching contract.

Split out because the bug it guards against is not about any one stage. Every
builder in registry.py treats the loaded config as scratch space -- update it
with overrides, pop the engine block it doesn't want -- and _load_yaml used to
hand out the cached dict itself. So the first build in a process permanently
edited what every later build would see.

Both symptoms were silent. A leaked override makes the second build inherit the
first one's tuning; a popped engine block makes it fall back to dataclass
defaults, which for VAD means an uncalibrated threshold standing in for a
measured one. Nothing raises. talk.py builds each stage once and never hit it;
benchmark.py and the _ab_* scripts build repeatedly.
"""

import pytest

from src.nobody_flux import registry

PATHS = [
    registry.CONFIG_PATH,
    registry.VAD_CONFIG_PATH,
    registry.STREAMING_ASR_CONFIG_PATH,
    registry.TURN_DETECTOR_CONFIG_PATH,
    registry.AUDIO_CONFIG_PATH,
    registry.VOICES_CONFIG_PATH,
]


@pytest.mark.parametrize("path", PATHS, ids=lambda p: p.name)
def test_each_load_returns_a_distinct_object(path):
    assert registry._load_yaml(path) is not registry._load_yaml(path)


@pytest.mark.parametrize("path", PATHS, ids=lambda p: p.name)
def test_mutating_a_loaded_config_does_not_affect_the_next_load(path):
    first = registry._load_yaml(path)
    baseline = registry._load_yaml(path)
    first["__injected__"] = "should not persist"
    for key in list(first):
        first.pop(key)
    assert registry._load_yaml(path) == baseline


def test_nested_blocks_survive_being_popped_from_a_copy():
    """A shallow copy would not be enough: the engine blocks and preset params
    are nested dicts, and those are exactly what the builders pop into."""
    path = registry.STREAMING_ASR_CONFIG_PATH
    blocks = [k for k, v in registry._load_yaml(path).items() if isinstance(v, dict)]
    assert blocks, "expected at least one engine block to test against"

    first = registry._load_yaml(path)
    for name in blocks:
        first[name].clear()
        first.pop(name)

    again = registry._load_yaml(path)
    for name in blocks:
        assert again.get(name), name


def test_repeated_builds_do_not_inherit_an_earlier_override(monkeypatch):
    """The concrete failure: sweeping a value through a builder, then building
    again without it, used to return the swept value."""
    seen = {}
    real_fields = dict(registry.vad.VoiceActivityDetector.__dataclass_fields__)

    class Stub:
        __dataclass_fields__ = real_fields

        def __init__(self, **kwargs):
            seen.clear()
            seen.update(kwargs)

    monkeypatch.setattr(registry.vad, "VoiceActivityDetector", Stub)

    registry.build_vad(threshold=0.31)
    assert seen["threshold"] == 0.31
    registry.build_vad()
    assert seen["threshold"] != 0.31


def test_repeated_builds_do_not_lose_the_engine_block(monkeypatch):
    seen = {}
    real_fields = dict(registry.vad.VoiceActivityDetector.__dataclass_fields__)

    class Stub:
        __dataclass_fields__ = real_fields

        def __init__(self, **kwargs):
            seen.clear()
            seen.update(kwargs)

    monkeypatch.setattr(registry.vad, "VoiceActivityDetector", Stub)

    registry.build_vad()
    first_threshold = seen["threshold"]
    for _ in range(3):
        registry.build_vad()
    assert seen["threshold"] == first_threshold
    assert "model_path" in seen
