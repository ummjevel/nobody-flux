#!/usr/bin/env python3
"""Measure the speaker->mic acoustic delay and write it to configs/audio.yaml
(delay_frames), so the reference-gate echo canceller (src/nobody_flux/aec.py's
ReferenceGate, used by audio.py's SharedStreamSession) aligns the reply it's
playing with the echo the mic actually hears. Speex models the delay itself, but
the gate needs it explicitly -- an unmeasured delay is why every threshold in
this repo is "실측 전 추정치".

How: play a short chirp while recording simultaneously (sd.playrec keeps the two
sample-aligned), then cross-correlate the recording against the chirp. The lag at
the correlation peak is the round-trip delay; delay_frames = round(lag / 480).

Run it on the actual device (mic + speaker in their real positions), NOT over an
SSH/WSL passthrough that adds its own buffering:

    source scripts/env.sh
    uv run python scripts/_calibrate_aec_delay.py

Needs a working mic + speaker; there's nothing to measure without both, so it
errors clearly rather than writing a bogus value. Requires quiet-ish surroundings
and the speaker audible to the mic (that's the whole signal path being measured).
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import sounddevice as sd

from src.nobody_flux.audio import FRAME_SAMPLES, SAMPLE_RATE
from src.nobody_flux.paths import PROJECT_ROOT

AUDIO_CONFIG = PROJECT_ROOT / "configs" / "audio.yaml"


def make_chirp(duration_s: float = 0.25, f0: float = 300.0, f1: float = 3000.0) -> np.ndarray:
    """Linear sine sweep -- broadband so the cross-correlation peak is sharp
    (a pure tone would autocorrelate into a broad ridge), band-limited to the
    voice range the speaker/mic actually reproduce."""
    n = int(SAMPLE_RATE * duration_s)
    t = np.arange(n) / SAMPLE_RATE
    k = (f1 - f0) / duration_s
    phase = 2 * np.pi * (f0 * t + 0.5 * k * t * t)
    # Fade the ends so speaker transients don't smear the correlation peak.
    env = np.ones(n, dtype=np.float32)
    fade = max(1, n // 20)
    env[:fade] = np.linspace(0, 1, fade)
    env[-fade:] = np.linspace(1, 0, fade)
    return (0.5 * np.sin(phase)).astype(np.float32) * env


def measure_delay_samples(chirp: np.ndarray, tail_s: float = 0.5, repeats: int = 5) -> int:
    """Play chirp+silence while recording; return the median cross-correlation
    lag (samples) over `repeats` trials. Median rejects the odd trial ruined by
    a transient noise."""
    tail = np.zeros(int(SAMPLE_RATE * tail_s), dtype=np.float32)
    playback = np.concatenate([chirp, tail])
    lags = []
    for i in range(repeats):
        rec = sd.playrec(playback, samplerate=SAMPLE_RATE, channels=1, dtype="float32")
        sd.wait()
        rec = rec[:, 0].astype(np.float32)
        # Full cross-correlation; peak index maps to lag via the chirp length.
        corr = np.correlate(rec, chirp, mode="full")
        lag = int(np.argmax(np.abs(corr))) - (len(chirp) - 1)
        lag = max(0, lag)
        lags.append(lag)
        print(f"  trial {i + 1}/{repeats}: lag = {lag} samples ({lag * 1000 / SAMPLE_RATE:.0f}ms)")
    return int(np.median(lags))


def write_delay_frames(delay_frames: int) -> None:
    """Rewrite just the `delay_frames:` line in configs/audio.yaml, preserving
    the file's comments (a full yaml round-trip would strip them)."""
    text = AUDIO_CONFIG.read_text(encoding="utf-8")
    new, n = re.subn(r"(?m)^delay_frames:.*$", f"delay_frames: {delay_frames}", text)
    if n == 0:
        new = text.rstrip() + f"\ndelay_frames: {delay_frames}\n"
    AUDIO_CONFIG.write_text(new, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5, help="number of chirp trials")
    parser.add_argument(
        "--dry-run", action="store_true", help="measure and print but don't write audio.yaml"
    )
    args = parser.parse_args()

    try:
        sd.check_input_settings(samplerate=SAMPLE_RATE, channels=1)
        sd.check_output_settings(samplerate=SAMPLE_RATE, channels=1)
    except Exception as exc:
        raise SystemExit(
            f"Need a working mic AND speaker at {SAMPLE_RATE}Hz to measure delay: {exc}"
        )

    print(f"Measuring speaker->mic delay ({args.repeats} trials). Stay quiet...")
    delay_samples = measure_delay_samples(make_chirp(), repeats=args.repeats)
    delay_frames = round(delay_samples / FRAME_SAMPLES)
    print(
        f"\nmedian delay = {delay_samples} samples "
        f"({delay_samples * 1000 / SAMPLE_RATE:.0f}ms) = {delay_frames} frames"
    )

    if args.dry_run:
        print("(--dry-run: not writing configs/audio.yaml)")
        return
    write_delay_frames(delay_frames)
    print(f"wrote delay_frames: {delay_frames} to {AUDIO_CONFIG}")


if __name__ == "__main__":
    main()
