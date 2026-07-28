#!/usr/bin/env python3
"""Persistent FreyaTTS server: loads the model once, then serves synthesis
requests over stdin/stdout until told to exit. Run inside FreyaTTS's own
isolated venv (see src/nobody_flux/tts.py's FreyaTtsServerKo docstring).

Protocol (line-based, one JSON object per line):
  stdin:  {"text": "...", "out": "/path/to/out.wav"}  -- one per request
          {"cmd": "exit"}                              -- shut down
  stdout: ---READY---                                  -- once, after model load
          {"ok": true, "out": "..."}                   -- one per successful request
          {"ok": false, "error": "..."}                -- one per failed request
"""

import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=9)
    args = parser.parse_args()

    from freyatts import FreyaTTS

    tts = FreyaTTS.from_pretrained(args.model_dir, device=args.device)

    print("---READY---", flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            print(json.dumps({"ok": False, "error": f"bad request JSON: {exc}"}), flush=True)
            continue

        if request.get("cmd") == "exit":
            break

        text = request.get("text")
        out_path = request.get("out")
        if not text or not out_path:
            print(json.dumps({"ok": False, "error": "request needs 'text' and 'out'"}), flush=True)
            continue

        try:
            wav = tts.synthesize(text, steps=args.steps, seed=args.seed)
            tts.save_wav(wav, out_path)
            print(json.dumps({"ok": True, "out": out_path}), flush=True)
        except Exception as exc:  # noqa: BLE001 -- reported to caller over the protocol, not fatal to the server
            print(json.dumps({"ok": False, "error": str(exc)}), flush=True)


if __name__ == "__main__":
    main()
