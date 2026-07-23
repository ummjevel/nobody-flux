"""Shared CLI plumbing for run_pipeline.py and talk.py.

Both entrypoints need the same --asr/--llm/--tts/--voice flags and the same
"resolve preset args -> concrete STSPipeline" wiring; before this module they
each hand-rolled it, byte-for-byte identical in two places. Importable as a
plain sibling module (no package/__init__.py needed) because Python puts the
launched script's own directory at sys.path[0] automatically.
"""

from __future__ import annotations

import argparse

from src.nobody_flux import registry
from src.nobody_flux.pipeline import STSPipeline


def add_pipeline_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--asr", default=None, help="ASR preset name (configs/models.yaml)")
    parser.add_argument("--llm", default=None, help="LLM preset name (configs/models.yaml)")
    parser.add_argument("--tts", default=None, help="TTS preset name (configs/models.yaml)")
    parser.add_argument("--voice", default=None, help="TTS reference voice (configs/voices.yaml)")


def build_pipeline_from_args(args: argparse.Namespace) -> tuple[STSPipeline, dict[str, str]]:
    """Returns (pipeline, {"asr": ..., "llm": ..., "tts": ...}) -- the resolved
    preset names, not just what was passed on the CLI (which may be None),
    so callers that need to log which preset actually ran (e.g. talk.py's
    ConversationStore.log_turn) don't have to separately re-derive "arg or
    default" themselves.
    """
    asr_preset = args.asr or registry.default_preset("asr")
    llm_preset = args.llm or registry.default_preset("llm")
    tts_preset = args.tts or registry.default_preset("tts")

    tts_overrides = {}
    if args.voice:
        tts_overrides["reference_audio"] = registry.resolve_voice(args.voice)

    pipeline = STSPipeline(
        asr=registry.build_asr(asr_preset),
        llm=registry.build_llm(llm_preset),
        tts=registry.build_tts(tts_preset, **tts_overrides),
    )
    return pipeline, {"asr": asr_preset, "llm": llm_preset, "tts": tts_preset}
