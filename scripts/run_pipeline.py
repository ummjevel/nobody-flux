#!/usr/bin/env python3
"""CLI entrypoint for the prototype local STS pipeline -- one wav in, one wav out.

For an actual multi-turn conversation (mic in, speaker out, history persists
across turns), use scripts/talk.py instead; this script exists as a
deterministic, scriptable single-call tool for debugging/regression checks
and for comparing --asr/--llm/--tts presets on a fixed input file.

Usage (from the project root):
    uv run python scripts/run_pipeline.py --wav-in path/to/input.wav --wav-out path/to/output.wav \\
        [--asr PRESET] [--llm PRESET] [--tts PRESET] [--voice VOICE]

Preset names come from configs/models.yaml (registry.py); omit a flag to use
that stage's default preset. --voice picks a TTS reference clip from
configs/voices.yaml (registry.resolve_voice) -- it's a parameter of the TTS
stage, not a separate preset dimension.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _cli import add_pipeline_args, build_pipeline_from_args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav-in", required=True, help="input wav (user's utterance)")
    parser.add_argument(
        "--wav-out", required=True, help="output wav (nobody's spoken reply)"
    )
    add_pipeline_args(parser)
    args = parser.parse_args()

    pipeline, _presets = build_pipeline_from_args(args)
    result = pipeline.run(args.wav_in, args.wav_out)

    print(f"[user]   {result['user_text']}")
    print(f"[nobody] {result['reply_text']}")
    print(f"[wav]    {result['wav_out']}")
    print(
        f"[timing] asr={result['asr_ms']}ms llm={result['llm_ms']}ms tts={result['tts_ms']}ms"
    )


if __name__ == "__main__":
    main()
