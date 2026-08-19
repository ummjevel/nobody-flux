"""VAD engine selection: the config plumbing, not the models.

Two engines exist because TEN-VAD's licence is Apache-2.0 "with additional
conditions" (a non-compete term -- THIRD-PARTY-NOTICES.md §2.3) while Silero VAD
is plain MIT. Nothing here says which to run; these tests pin that the choice
stays a config value and, more importantly, that one engine's numbers cannot
leak onto the other.

That last part is the whole point. `threshold: 0.5` in the ten-vad block is a
measurement from this room's microphone (scripts/_calibrate_vad_threshold.py),
and the repo's recorded failure is asymmetric: too low and the model reports
speech through silence, so the turn never ends. Presenting that number as if it
applied to a different model is the same class of bug as the flat
streaming_asr.yaml that let zipformer's model_dir reach the SenseVoice
constructor.

No weights required: the merge logic is exercised against a stub, and the
sherpa-config builder is only asked to fail.
"""

import pytest
import yaml

from src.nobody_flux import registry
from src.nobody_flux.paths import PROJECT_ROOT
from src.nobody_flux.turn import vad as vad_module


@pytest.fixture
def captured(monkeypatch):
    """Replace the detector with something that records what it was handed."""
    seen = {}
    # Captured before the patch, not after -- build_vad's unknown-key guard reads
    # __dataclass_fields__ off whatever name it finds, so the stub has to carry
    # the real field set or the guard would accept anything.
    real_fields = dict(vad_module.VoiceActivityDetector.__dataclass_fields__)

    class Stub:
        __dataclass_fields__ = real_fields

        def __init__(self, **kwargs):
            seen.clear()
            seen.update(kwargs)

    monkeypatch.setattr(vad_module, "VoiceActivityDetector", Stub)
    return seen


# ------------------------------------------------------------------- registry

def test_default_engine_comes_from_the_yaml(captured):
    registry.build_vad()
    on_disk = yaml.safe_load(registry.VAD_CONFIG_PATH.read_text(encoding="utf-8"))
    assert captured["engine"] == on_disk["engine"]


def test_selected_engine_block_is_merged(captured):
    registry.build_vad(engine="silero-vad")
    assert captured["engine"] == "silero-vad"
    assert "silero-vad" in str(captured["model_path"])


def test_other_engines_blocks_are_dropped_not_unioned(captured):
    """The failure this mirrors: a flat config let one engine's model path reach
    the other engine's constructor."""
    registry.build_vad(engine="silero-vad")
    assert "ten-vad" not in str(captured["model_path"])
    for name in vad_module.VAD_ENGINES:
        assert name not in captured


def test_each_engine_gets_its_own_threshold(captured):
    """ten-vad's 0.5 is measured; silero's is sherpa's default. They must be
    read from their own blocks even when the numbers happen to coincide."""
    on_disk = yaml.safe_load(registry.VAD_CONFIG_PATH.read_text(encoding="utf-8"))
    for engine in vad_module.VAD_ENGINES:
        registry.build_vad(engine=engine)
        assert captured["threshold"] == on_disk[engine]["threshold"], engine


def test_threshold_is_not_a_top_level_key():
    """Keeping it top-level is what would let the calibrated value leak."""
    on_disk = yaml.safe_load(registry.VAD_CONFIG_PATH.read_text(encoding="utf-8"))
    assert "threshold" not in on_disk
    for engine in vad_module.VAD_ENGINES:
        assert "threshold" in on_disk[engine], engine


def test_overrides_beat_the_engine_block(captured):
    """_calibrate_vad_threshold.py sweeps threshold through build_vad. If the
    block outranked the override every iteration would measure the same value --
    which is exactly the "looks tuned, does nothing" failure."""
    registry.build_vad(threshold=0.37)
    assert captured["threshold"] == 0.37


def test_override_can_switch_engine(captured):
    registry.build_vad(engine="silero-vad")
    assert captured["engine"] == "silero-vad"
    registry.build_vad(engine="ten-vad")
    assert captured["engine"] == "ten-vad"


def test_model_path_is_resolved_against_the_project_root(captured):
    registry.build_vad(engine="ten-vad")
    assert captured["model_path"].is_absolute()
    assert str(captured["model_path"]).startswith(str(PROJECT_ROOT))


def test_unknown_key_raises_rather_than_being_dropped(captured):
    with pytest.raises(ValueError) as e:
        registry.build_vad(no_such_knob=1)
    assert "no_such_knob" in str(e.value)


def test_shared_knobs_still_reach_the_constructor(captured):
    registry.build_vad()
    for knob in ("pre_roll_ms", "barge_in_confirm_ms", "min_silence_duration"):
        assert knob in captured, knob


# ------------------------------------------------------- engine table / builder

def test_every_engine_has_a_default_model_path():
    for engine in vad_module.VAD_ENGINES:
        assert vad_module.default_model_path(engine).name.endswith(".onnx")


def test_engine_names_are_distinct_paths():
    paths = {vad_module.default_model_path(e) for e in vad_module.VAD_ENGINES}
    assert len(paths) == len(vad_module.VAD_ENGINES)


def test_unknown_engine_raises_from_default_model_path():
    with pytest.raises(ValueError) as e:
        vad_module.default_model_path("whisper-vad")
    assert "whisper-vad" in str(e.value)


def test_unknown_engine_raises_from_the_config_builder():
    with pytest.raises(ValueError):
        vad_module.build_sherpa_vad_config(
            "whisper-vad", "x.onnx", threshold=0.5, min_silence_duration=0.5,
            min_speech_duration=0.15, max_speech_duration=20.0,
        )


def test_missing_model_file_names_the_engine(tmp_path):
    """Same reasoning as SherpaMatchaTts._check_data_dir: fail in Python with the
    path in the message, not as a sherpa-onnx log line."""
    with pytest.raises(FileNotFoundError) as e:
        vad_module.build_sherpa_vad_config(
            "silero-vad", tmp_path / "absent.onnx", threshold=0.5,
            min_silence_duration=0.5, min_speech_duration=0.15,
            max_speech_duration=20.0,
        )
    assert "silero-vad" in str(e.value) and "absent.onnx" in str(e.value)


def test_yaml_declares_a_block_for_every_known_engine():
    """A new engine added to the table without a config block would fall back to
    shared values silently -- including a threshold measured for another model."""
    on_disk = yaml.safe_load(registry.VAD_CONFIG_PATH.read_text(encoding="utf-8"))
    for engine in vad_module.VAD_ENGINES:
        assert isinstance(on_disk.get(engine), dict), engine
