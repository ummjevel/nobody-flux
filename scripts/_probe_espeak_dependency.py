"""How hard does sherpa-matcha-ko depend on espeak-ng-data, and how does it fail?

Why this is a script and not a test: it needs the 18MB espeak-ng-data bundle and
a 74MB acoustic model, and it exists so the finding in
docs/output/research-delta-20260818.md §13.4 stays reproducible.

What it now checks is that SherpaMatchaTts._check_data_dir gets there first. Both
bad cases used to abort the interpreter from C -- rc=1, no traceback, nothing for
try/except -- and one of them (a host with a system espeak-ng installed at
/usr/share/espeak-ng-data) would instead have succeeded silently against the
wrong phoneme tables. A C-ABORT line in the output means that guard was bypassed.

⚠️ The methodological trap, which cost a wrong answer the first time: espeak-ng
is a process-global singleton (espeak_ng_Initialize / espeak_ng_InitializePath).
Test a bad data_dir *after* a good one in the same process and the bad case
silently reuses the already-initialized global state -- byte-for-byte identical
audio, reading as "espeak-ng-data is not actually needed". It is needed. Each
case must get a fresh process, which is why this drives subprocesses.

Usage:
    python scripts/_probe_espeak_dependency.py
    python scripts/_probe_espeak_dependency.py --case missing   # internal
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TEXT = "안녕하세요. 오늘 날씨가 좋네요."
CASES = {
    "good": "models/sherpa-matcha-en/espeak-ng-data",
    "missing": "models/__no_such_espeak_dir__",
    "empty": "",
}


def run_case(name: str) -> None:
    import numpy as np

    from nobody_flux.stage.tts import SherpaMatchaTts

    tts = SherpaMatchaTts(
        acoustic_model=Path("models/sherpa-matcha-ko/matcha-ko-voiceA-ep499-steps10.onnx"),
        vocoder=Path("models/sherpa-matcha-ko/vocos-22khz-univ.onnx"),
        tokens=Path("models/sherpa-matcha-ko/tokens.txt"),
        data_dir=Path(CASES[name]),
    )
    samples, sr = tts.synthesize_audio(TEXT)
    a = np.asarray(samples, dtype=np.float32)
    print("__OK__ samples=%d sr=%d dur=%.2fs peak=%.4f" % (a.size, sr, a.size / sr, float(np.abs(a).max())))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=sorted(CASES))
    args = ap.parse_args()

    if args.case:
        run_case(args.case)
        return 0

    # Bad cases first, so a false pass from leftover global state is impossible
    # even if someone later collapses this back into one process.
    for name in ("missing", "empty", "good"):
        p = subprocess.run(
            [sys.executable, __file__, "--case", name],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        ok = next((l for l in p.stdout.splitlines() if l.startswith("__OK__")), None)
        if ok:
            print("%-8s OK       %s" % (name, ok[len("__OK__"):].strip()))
            continue
        combined = p.stdout + p.stderr
        lines = [l for l in combined.splitlines() if l.strip()]
        last = lines[-1][:110] if lines else "(no output)"
        # The distinction that matters: our guard raises a Python exception we
        # can see and act on; espeak-ng aborts the interpreter and leaves only a
        # C-level message. Before SherpaMatchaTts._check_data_dir existed, both
        # of the bad cases were the second kind.
        kind = "PY-RAISE" if "Traceback (most recent call last)" in combined else "C-ABORT "
        print("%-8s %s rc=%d -- %s" % (name, kind, p.returncode, last))
    print()
    print("Expected: missing/empty -> PY-RAISE from SherpaMatchaTts._check_data_dir;")
    print("          good -> OK. A C-ABORT means the Python guard was bypassed.")
    print("See docs/output/research-delta-20260818.md §13.4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
