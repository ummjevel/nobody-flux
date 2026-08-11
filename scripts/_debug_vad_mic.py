#!/usr/bin/env python3
"""Debug helper: records N seconds from the mic (same sd.InputStream setup as
vad.py) while printing per-frame RMS energy and TEN-VAD's raw
is_speech_detected() state, then saves the raw capture to a wav file.

Not part of the app -- for diagnosing "VAD doesn't detect anything" reports
where the model works fine on file-based tests but not on someone's actual
live mic. Run it, speak into the mic for the recording window, then send
both the printed output and the wav file it writes.

Deliberately does NOT import src.nobody_flux.registry: registry.py does
`from . import asr, llm, tts, vad`, and llm.py pulls in the full
transformers/torch stack just to build a VAD. Reading configs/vad.yaml
directly here keeps this script's dependency surface to just vad.py, so it
stays usable even if something unrelated breaks in the transformers/torch
import chain (confirmed to happen once already -- see git history).

Usage:
    source scripts/env.sh
    uv run python scripts/_debug_vad_mic.py [seconds]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import sherpa_onnx
import sounddevice as sd
import soundfile as sf
import yaml

from src.nobody_flux.paths import PROJECT_ROOT
from src.nobody_flux.turn.vad import FRAME_SAMPLES, SAMPLE_RATE, VoiceActivityDetector

VAD_CONFIG_PATH = PROJECT_ROOT / "configs" / "vad.yaml"
DURATION_S = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0
OUT_PATH = PROJECT_ROOT / "data" / "debug_vad_capture.wav"


def main():
    with open(VAD_CONFIG_PATH, encoding="utf-8") as f:
        vad_config = yaml.safe_load(f)
    vad = VoiceActivityDetector(**vad_config)

    print(f"threshold={vad.threshold} min_speech_duration={vad.min_speech_duration}")
    print(f"Recording {DURATION_S:.1f}s -- speak now...")

    frames: list[np.ndarray] = []
    n_frames_needed = int(DURATION_S * SAMPLE_RATE / FRAME_SAMPLES)

    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=FRAME_SAMPLES
    ) as stream:
        for i in range(n_frames_needed):
            block, _overflowed = stream.read(FRAME_SAMPLES)
            samples = block[:, 0].copy()
            frames.append(samples)

            rms = float(np.sqrt(np.mean(samples.astype("float64") ** 2)))
            vad._vad.accept_waveform(samples)
            speaking = vad._vad.is_speech_detected()
            segment_ready = not vad._vad.empty()

            t = i * FRAME_SAMPLES / SAMPLE_RATE
            print(
                f"t={t:5.2f}s  rms={rms:.4f}  is_speech_detected={speaking}  "
                f"segment_ready={segment_ready}"
            )
            if segment_ready:
                seg = vad._vad.front
                print(f"  -> SEGMENT: start={seg.start} len={len(seg.samples)}")
                vad._vad.pop()

    full_audio = np.concatenate(frames)
    sf.write(str(OUT_PATH), full_audio, SAMPLE_RATE)
    overall_rms = float(np.sqrt(np.mean(full_audio.astype("float64") ** 2)))
    peak = float(np.max(np.abs(full_audio))) if len(full_audio) else 0.0
    print(f"\nSaved raw capture to {OUT_PATH}")
    print(f"Overall RMS: {overall_rms:.4f}  Peak: {peak:.4f}")
    print(f"(sherpa_onnx {sherpa_onnx.__file__})")


if __name__ == "__main__":
    main()
