#!/usr/bin/env python3
"""Import-only smoke test: does the package load, and how long does it take?

Deliberately imports every module but constructs nothing. That split is the
point -- it separates "the refactor is wired correctly" from "the models are
downloaded", so a fresh environment can be checked before any weights exist.
It also surfaces the native-library problems this project has repeatedly hit
(sherpa-onnx failing to find onnxruntime; macOS aborting on a duplicate
libomp), which are all import-time failures.

The reported timings double as a regression check on the lazy-import policy in
nobody_flux/__init__.py: if importing the package as a whole starts costing
seconds, something has begun pulling torch or llama.cpp in eagerly again.

    python scripts/_smoke_imports.py
"""

from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Ordered roughly by dependency weight, so the output reads as a cost profile.
MODULES = [
    "src.nobody_flux",
    "src.nobody_flux.paths",
    "src.nobody_flux.platform_support",
    "src.nobody_flux.persona",
    "src.nobody_flux.textchunk",
    "src.nobody_flux.storage",
    "src.nobody_flux.memory",
    "src.nobody_flux.audio",
    "src.nobody_flux.audio.resample",
    "src.nobody_flux.audio.aec",
    "src.nobody_flux.audio.session",
    "src.nobody_flux.audio.player",
    "src.nobody_flux.turn",
    "src.nobody_flux.turn.backchannel",
    "src.nobody_flux.turn.vad",
    "src.nobody_flux.turn.detector",
    "src.nobody_flux.turn.controller",
    "src.nobody_flux.stage",
    "src.nobody_flux.stage.asr",
    "src.nobody_flux.stage.asr_stream",
    "src.nobody_flux.stage.llm",
    "src.nobody_flux.stage.tts",
    "src.nobody_flux.pipeline",
    "src.nobody_flux.registry",
]


def main() -> int:
    from src.nobody_flux import platform_support as ps

    print(f"platform    : {ps.platform_label()}")
    print(f"interpreter : {sys.executable}")
    for path in ps.site_packages_dirs():
        print(f"site-packages: {path}")
    print()

    failures: list[tuple[str, BaseException]] = []
    for name in MODULES:
        started = time.perf_counter()
        try:
            importlib.import_module(name)
        except BaseException as exc:  # noqa: BLE001 - report, don't abort the sweep
            failures.append((name, exc))
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
            continue
        elapsed_ms = (time.perf_counter() - started) * 1000
        # Only annotate imports slow enough to be worth noticing; a wall of
        # "0.0ms" lines buries the two or three that actually cost something.
        note = f"  ({elapsed_ms:.0f}ms)" if elapsed_ms >= 50 else ""
        print(f"  ok    {name}{note}")

    print()
    if failures:
        print(f"{len(failures)} module(s) failed to import.")
        return 1

    # Config-driven construction is checked separately from imports: it can
    # fail for a completely different reason (a malformed yaml, a preset naming
    # a class that no longer exists) and should not be confused with a broken
    # native library.
    from src.nobody_flux import registry
    from src.nobody_flux.audio import session

    for stage in ("asr", "llm", "tts"):
        presets = ", ".join(registry.list_presets(stage))
        print(f"{stage:>3} presets: {presets}  (default: {registry.default_preset(stage)})")
    print(f"aec backend for 'auto': {session.select_backend('auto')}")
    print("\nAll imports OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
