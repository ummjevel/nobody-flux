#!/usr/bin/env python3
"""One-shot CLI wrapper around freyatts's FreyaTTS.from_pretrained/synthesize/
save_wav, run inside FreyaTTS's own isolated venv (see src/nobody_flux/tts.py's
FreyaTtsKo docstring for why it's isolated: same reasoning as MOSS-TTS-Nano's
own venv, just installed via pip from the freyatts git package instead of a
cloned repo).

NOT used by FreyaTtsKo at runtime -- that class talks to the persistent
scripts/_freyatts_server.py instead (loads the model once, serves many
requests -- see that script's docstring for why). This script reloads the
model on every invocation, which makes it too slow for real conversation
turns, but it's kept around as a quick manual sanity-check tool ("does this
checkpoint still load and produce a wav at all") that doesn't require
understanding the server's stdin/stdout protocol.

Usage:
    python _freyatts_infer.py --model-dir <dir> --text "..." --out out.wav \\
        --device cuda --steps 32 --seed 9
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=9)
    args = parser.parse_args()

    from freyatts import FreyaTTS

    tts = FreyaTTS.from_pretrained(args.model_dir, device=args.device)
    wav = tts.synthesize(args.text, steps=args.steps, seed=args.seed)
    tts.save_wav(wav, args.out)
    print(args.out)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 -- surfaced via subprocess stderr, caller decides how to react
        print(f"freyatts inference failed: {exc}", file=sys.stderr)
        sys.exit(1)
