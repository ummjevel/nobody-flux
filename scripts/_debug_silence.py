#!/usr/bin/env python3
"""Throwaway probe: does TEN-VAD treat *digital* silence as silence?

_debug_segment.py showed a segment finalizing 6.2s after speech stopped, with
min_silence_duration set to 0.5s. This isolates the suspected cause: the
trailing "silence" in that harness is exact zeros, which no microphone ever
produces. Feature extraction over an all-zero frame is degenerate, so the
model's output there says nothing about its behaviour on real room tone.

Feeds the same utterance twice -- once followed by zeros, once by a quiet noise
floor -- and reports when each finalizes.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import sherpa_onnx
import soundfile as sf
import yaml

from src.nobody_flux.paths import PROJECT_ROOT

SAMPLE_RATE = 16_000
FRAME_SAMPLES = 480

wav = PROJECT_ROOT / "models" / "sense-voice" / "test_wavs" / "ko.wav"
audio, _rate = sf.read(str(wav), dtype="float32")
cfg = yaml.safe_load((PROJECT_ROOT / "configs" / "vad.yaml").read_text(encoding="utf-8"))


def build_vad(threshold: float | None = None):
    ten = sherpa_onnx.TenVadModelConfig(
        model=str(PROJECT_ROOT / "models" / "ten-vad" / "ten-vad.onnx"),
        threshold=cfg["threshold"] if threshold is None else threshold,
        min_silence_duration=cfg["min_silence_duration"],
        min_speech_duration=cfg["min_speech_duration"],
        max_speech_duration=cfg["max_speech_duration"],
    )
    return sherpa_onnx.VoiceActivityDetector(
        sherpa_onnx.VadModelConfig(ten_vad=ten, sample_rate=SAMPLE_RATE, num_threads=1),
        buffer_size_in_seconds=100,
    )


def run(label: str, silence: np.ndarray, threshold: float | None = None) -> None:
    vad = build_vad(threshold)
    lead = np.zeros(int(SAMPLE_RATE * 0.3), dtype=np.float32)
    script = np.concatenate([lead, audio, silence])
    speech_end_s = (len(lead) + len(audio)) / SAMPLE_RATE

    fed = 0
    reported = False
    for offset in range(0, len(script), FRAME_SAMPLES):
        frame = script[offset : offset + FRAME_SAMPLES]
        if len(frame) < FRAME_SAMPLES:
            frame = np.pad(frame, (0, FRAME_SAMPLES - len(frame)))
        vad.accept_waveform(frame)
        fed += len(frame)
        while not vad.empty():
            seg = vad.front
            start, length = seg.start, len(seg.samples)
            vad.pop()
            print(
                f"  [{label}] segment start={start/SAMPLE_RATE:.2f}s "
                f"len={length/SAMPLE_RATE:.2f}s "
                f"finalized at fed={fed/SAMPLE_RATE:.2f}s "
                f"(lag after speech end {fed/SAMPLE_RATE - speech_end_s:.2f}s)"
            )
            reported = True
    if not reported:
        print(f"  [{label}] no segment finalized within {len(script)/SAMPLE_RATE:.2f}s")


seconds = 15.0
n = int(SAMPLE_RATE * seconds)
rng = np.random.default_rng(0)

print(f"speech: {len(audio)/SAMPLE_RATE:.2f}s, trailing silence: {seconds:.0f}s")
noise = (rng.standard_normal(n) * 1e-3).astype(np.float32)
for threshold in (0.25, 0.4, 0.5, 0.7):
    print(f"threshold={threshold}")
    run("digital zeros", np.zeros(n, dtype=np.float32), threshold)
    run("noise 1e-3   ", noise, threshold)
