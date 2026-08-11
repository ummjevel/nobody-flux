#!/usr/bin/env python3
"""Throwaway probe: what does sherpa-onnx's SpeechSegment actually report?

Feeds a known wav plus trailing silence through the raw sherpa-onnx VAD and
prints each segment's start index and length, plus every transition of
`is_speech_detected()`, so the units, the origin of `segment.start` and the
*latency* between "silence began" and "segment finalized" can be confirmed
rather than assumed.

The trailing silence is deliberately far longer than min_silence_duration: the
question this was written to answer is why a segment does not finalize shortly
after speech stops.
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
TRAILING_SILENCE_S = 15.0

wav = PROJECT_ROOT / "models" / "sense-voice" / "test_wavs" / "ko.wav"
audio, rate = sf.read(str(wav), dtype="float32")
print(f"source: {len(audio)} samples = {len(audio)/rate:.2f}s at {rate}Hz")

speech_end_s = 0.3 + len(audio) / SAMPLE_RATE
script = np.concatenate(
    [
        np.zeros(int(SAMPLE_RATE * 0.3), dtype=np.float32),
        audio,
        np.zeros(int(SAMPLE_RATE * TRAILING_SILENCE_S), dtype=np.float32),
    ]
)
print(f"script: {len(script)} samples = {len(script)/SAMPLE_RATE:.2f}s "
      f"(speech ends at {speech_end_s:.2f}s)")

cfg = yaml.safe_load((PROJECT_ROOT / "configs" / "vad.yaml").read_text(encoding="utf-8"))
ten = sherpa_onnx.TenVadModelConfig(
    model=str(PROJECT_ROOT / "models" / "ten-vad" / "ten-vad.onnx"),
    threshold=cfg["threshold"],
    min_silence_duration=cfg["min_silence_duration"],
    min_speech_duration=cfg["min_speech_duration"],
    max_speech_duration=cfg["max_speech_duration"],
)
vad = sherpa_onnx.VoiceActivityDetector(
    sherpa_onnx.VadModelConfig(ten_vad=ten, sample_rate=SAMPLE_RATE, num_threads=1),
    buffer_size_in_seconds=100,
)

fed = 0
speaking = False
for offset in range(0, len(script), FRAME_SAMPLES):
    frame = script[offset : offset + FRAME_SAMPLES]
    if len(frame) < FRAME_SAMPLES:
        frame = np.pad(frame, (0, FRAME_SAMPLES - len(frame)))
    vad.accept_waveform(frame)
    fed += len(frame)

    now = vad.is_speech_detected()
    if now != speaking:
        speaking = now
        print(f"  is_speech_detected -> {now} at fed={fed} ({fed/SAMPLE_RATE:.2f}s)")

    while not vad.empty():
        seg = vad.front
        start, length = seg.start, len(seg.samples)
        vad.pop()
        print(
            f"  segment: start={start} ({start/SAMPLE_RATE:.2f}s) "
            f"len={length} ({length/SAMPLE_RATE:.2f}s) "
            f"end={start+length} ({(start+length)/SAMPLE_RATE:.2f}s) "
            f"| fed so far={fed} ({fed/SAMPLE_RATE:.2f}s) "
            f"| lag after speech end={(fed/SAMPLE_RATE)-speech_end_s:.2f}s"
        )
print(f"total fed: {fed} ({fed/SAMPLE_RATE:.2f}s)")
