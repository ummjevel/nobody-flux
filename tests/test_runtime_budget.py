"""registry's CPU-budget thread allocation (configs/runtime.yaml, code-review
#9): the piece the CM4 port needs first. The math is clamp(1, budget*fraction,
cap); the injection rule is "explicit preset config beats derived default"."""

from __future__ import annotations

from dataclasses import dataclass

from src.nobody_flux import registry


def test_four_core_target_allocation(monkeypatch):
    monkeypatch.setattr(registry.os, "cpu_count", lambda: 4)
    assert registry.stage_threads("llm") == 3   # 4 * 0.75
    assert registry.stage_threads("tts") == 1   # 4 * 0.25
    assert registry.stage_threads("asr") == 3


def test_dev_box_hits_the_measured_caps(monkeypatch):
    # The caps preserve the hand-tuned 28-core values (llm 8 / tts 2 / asr 8);
    # beyond them measurement showed no gain.
    monkeypatch.setattr(registry.os, "cpu_count", lambda: 28)
    assert registry.stage_threads("llm") == 8
    assert registry.stage_threads("tts") == 2
    assert registry.stage_threads("asr") == 8


def test_single_core_never_goes_below_one(monkeypatch):
    monkeypatch.setattr(registry.os, "cpu_count", lambda: 1)
    assert registry.stage_threads("tts") == 1


def test_unbudgeted_stage_returns_none():
    assert registry.stage_threads("vad") is None


def test_env_override_simulates_a_smaller_machine(monkeypatch):
    # The point of the override: measure CM4-like thread counts from the dev
    # box without editing a tracked config mid-run.
    monkeypatch.setattr(registry.os, "cpu_count", lambda: 28)
    monkeypatch.setenv("NOBODY_CPU_BUDGET", "4")
    assert registry.stage_threads("llm") == 3
    assert registry.stage_threads("tts") == 1


def test_injection_fills_only_missing_thread_params(monkeypatch):
    monkeypatch.setattr(registry.os, "cpu_count", lambda: 4)

    @dataclass
    class WithLlamaStyle:
        n_threads: int = 8

    @dataclass
    class WithSherpaStyle:
        num_threads: int = 2

    @dataclass
    class WithoutThreads:
        other: int = 0

    params: dict = {}
    registry._inject_thread_budget("llm", WithLlamaStyle, params)
    assert params == {"n_threads": 3}

    params = {}
    registry._inject_thread_budget("tts", WithSherpaStyle, params)
    assert params == {"num_threads": 1}

    # An explicit preset value must win over the derived default.
    params = {"n_threads": 6}
    registry._inject_thread_budget("llm", WithLlamaStyle, params)
    assert params == {"n_threads": 6}

    # A class with no thread field is left alone.
    params = {}
    registry._inject_thread_budget("llm", WithoutThreads, params)
    assert params == {}
