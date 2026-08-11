#!/usr/bin/env python3
"""Phase 2b: measure this mic/room's actual backchannel vs barge-in duration
distributions and suggest turn-taking thresholds from them, instead of the
"실측 전 추정치" defaults in configs/vad.yaml and src/nobody_flux/backchannel.py.

docs/barge-in-design.md says exactly this: "확정 방법: scripts/_debug_vad_mic.py를
확장(맞장구/barge-in 샘플 녹음 → 지속시간 분포 실측)해 정한다." This is that tool.

What it does: records labeled single utterances (you say "응"/"어" for backchannel,
a real interruption for barge-in, a normal sentence for normal), measures each via
the SAME TEN-VAD segmentation vad.py uses, then reports per-label duration stats and
suggests:
  - barge_in_confirm_ms   -- the cut that best separates backchannel from barge-in
                             (stage 1 delayed-stop threshold, configs/vad.yaml)
  - BACKCHANNEL_MAX_DURATION_S -- backchannel upper bound (src/nobody_flux/backchannel.py)

Suggestions are printed, not auto-applied (except --apply, which writes only
barge_in_confirm_ms into configs/vad.yaml -- the backchannel constant lives in code,
so that one you edit by hand after seeing the numbers).

Interactive; run it on the real device (a Mac's native mic works; WSL2 passthrough
is unreliable -- see talk.py):

    source scripts/env.sh
    uv run python scripts/_calibrate_turn_params.py

Deliberately reads configs/vad.yaml directly rather than importing registry, same
dependency-surface reasoning as scripts/_debug_vad_mic.py.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import yaml

from src.nobody_flux.paths import PROJECT_ROOT
from src.nobody_flux.turn.vad import FRAME_SAMPLES, SAMPLE_RATE, VoiceActivityDetector

VAD_CONFIG_PATH = PROJECT_ROOT / "configs" / "vad.yaml"

LABELS = {"b": "backchannel", "i": "bargein", "n": "normal"}


def _pct(values: list[float], p: float) -> float:
    return float(np.percentile(values, p)) if values else float("nan")


def suggest_thresholds(durations_by_label: dict[str, list[float]]) -> dict:
    """Pure function (unit-testable, no audio): given captured utterance
    durations in seconds per label, return suggested thresholds + the stats they
    came from. barge_in_confirm_ms is placed between the backchannel p90 and the
    barge-in p10 -- the gap between the two distributions -- so most backchannel
    falls under it and most real interruptions over it. `separable` flags whether
    the two actually separate (p90_backchannel < p10_bargein); if not, no
    duration cut cleanly tells them apart and the lexical stage 2 carries more."""
    out: dict = {}
    bc = sorted(durations_by_label.get("backchannel", []))
    bi = sorted(durations_by_label.get("bargein", []))

    if bc:
        out["backchannel_p50_s"] = round(_pct(bc, 50), 3)
        out["backchannel_p90_s"] = round(_pct(bc, 90), 3)
        # A little headroom over p90 so a slightly-long "그으래?" still counts.
        out["BACKCHANNEL_MAX_DURATION_S"] = round(_pct(bc, 90) + 0.1, 2)
    if bi:
        out["bargein_p10_s"] = round(_pct(bi, 10), 3)
        out["bargein_p50_s"] = round(_pct(bi, 50), 3)
    if bc and bi:
        bc90, bi10 = _pct(bc, 90), _pct(bi, 10)
        out["separable"] = bool(bc90 < bi10)
        # Midpoint of the gap (or of the overlap, if they overlap -- still the
        # least-bad single cut).
        out["barge_in_confirm_ms"] = int(round((bc90 + bi10) / 2 * 1000))
    return out


def write_barge_in_confirm_ms(value_ms: int) -> None:
    text = VAD_CONFIG_PATH.read_text(encoding="utf-8")
    new, n = re.subn(r"(?m)^barge_in_confirm_ms:.*$", f"barge_in_confirm_ms: {value_ms}", text)
    if n == 0:
        new = text.rstrip() + f"\nbarge_in_confirm_ms: {value_ms}\n"
    VAD_CONFIG_PATH.write_text(new, encoding="utf-8")


def record_one(vad: VoiceActivityDetector, sd) -> float | None:
    """Capture one utterance and return its TEN-VAD speech duration in seconds
    (None if nothing was detected). Same segmentation as vad.py, minimal loop."""
    vad._vad.reset()
    got: float | None = None
    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=FRAME_SAMPLES
    ) as stream:
        silent_after_speech = 0
        spoke = False
        while True:
            block, _ = stream.read(FRAME_SAMPLES)
            samples = block[:, 0].copy()
            vad._vad.accept_waveform(samples)
            if vad._vad.is_speech_detected():
                spoke = True
                silent_after_speech = 0
            elif spoke:
                silent_after_speech += 1
            if not vad._vad.empty():
                seg = vad._vad.front
                got = len(seg.samples) / SAMPLE_RATE
                vad._vad.pop()
                break
            # Give up if a long silence follows with no finalized segment.
            if spoke and silent_after_speech > int(1.5 * SAMPLE_RATE / FRAME_SAMPLES):
                break
    return got


def main():
    import sounddevice as sd

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply", action="store_true", help="write suggested barge_in_confirm_ms to configs/vad.yaml"
    )
    args = parser.parse_args()

    with open(VAD_CONFIG_PATH, encoding="utf-8") as f:
        vad = VoiceActivityDetector(**yaml.safe_load(f))

    durations: dict[str, list[float]] = {v: [] for v in LABELS.values()}
    print(
        "Record labeled samples. For each: type b (backchannel: 응/어/네), "
        "i (barge-in: a real interruption), n (normal sentence), or q to finish.\n"
        "Aim for 8-10 of each b and i for a stable distribution."
    )
    while True:
        key = input("\n[b/i/n/q] > ").strip().lower()
        if key == "q":
            break
        if key not in LABELS:
            print("  (use b, i, n, or q)")
            continue
        label = LABELS[key]
        print(f"  recording {label} -- speak now...")
        dur = record_one(vad, sd)
        if dur is None:
            print("  (no speech detected, skipped)")
            continue
        durations[label].append(dur)
        print(f"  {label}: {dur * 1000:.0f}ms   (n={len(durations[label])})")

    print("\n=== duration stats (ms) ===")
    for label, xs in durations.items():
        if xs:
            print(
                f"{label:12s} n={len(xs):2d}  "
                f"p10={_pct(xs, 10) * 1000:.0f}  p50={_pct(xs, 50) * 1000:.0f}  "
                f"p90={_pct(xs, 90) * 1000:.0f}  max={max(xs) * 1000:.0f}"
            )

    suggestions = suggest_thresholds(durations)
    print("\n=== suggestions ===")
    for k, v in suggestions.items():
        print(f"{k}: {v}")

    if not suggestions.get("separable", True):
        print(
            "\n[warn] backchannel and barge-in durations overlap -- no clean single "
            "cut. Stage-2 lexical check (backchannel.py) matters more here."
        )

    if args.apply and "barge_in_confirm_ms" in suggestions:
        write_barge_in_confirm_ms(suggestions["barge_in_confirm_ms"])
        print(f"\nwrote barge_in_confirm_ms: {suggestions['barge_in_confirm_ms']} to {VAD_CONFIG_PATH}")
    elif "BACKCHANNEL_MAX_DURATION_S" in suggestions:
        print(
            "\n(edit BACKCHANNEL_MAX_DURATION_S in src/nobody_flux/backchannel.py by hand; "
            "pass --apply to write barge_in_confirm_ms into configs/vad.yaml)"
        )


if __name__ == "__main__":
    main()
