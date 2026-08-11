#!/usr/bin/env python3
"""Throwaway probe: what does the captured room tone actually look like?

The threshold sweep reported a noise floor of rms 0.035 with a peak of 0.99997,
which is contradictory -- a quiet room does not touch full scale. Either the
capture opens with a transient (a device start-up pop, common on USB audio) or
the microphone's gain is set so high that its own noise is enormous. The two
call for different fixes, so this prints the level per 100ms window.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import soundfile as sf

from src.nobody_flux.paths import PROJECT_ROOT

path = PROJECT_ROOT / "data" / "room_tone.wav"
audio, rate = sf.read(str(path), dtype="float32", always_2d=False)
print(f"{path.name}: {len(audio)/rate:.2f}s at {rate}Hz")

window = int(rate * 0.1)
for index in range(0, len(audio) - window, window):
    block = audio[index : index + window]
    rms = float(np.sqrt(np.mean(block.astype("float64") ** 2)))
    print(f"  t={index/rate:5.2f}s  rms={rms:.5f}  peak={float(np.max(np.abs(block))):.5f}")

tail = audio[rate:]  # everything after the first second
print(
    f"\nexcluding first 1s: rms={float(np.sqrt(np.mean(tail.astype('float64')**2))):.5f} "
    f"peak={float(np.max(np.abs(tail))):.5f}"
)
