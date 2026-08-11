#!/usr/bin/env python3
"""Measure this microphone's noise floor and pick the TEN-VAD threshold from it.

Why this exists
---------------

``configs/vad.yaml``'s ``threshold`` was set to 0.25 on the old WSL2 box, where
TEN-VAD was *missing* real speech at the stock 0.5. That fixed the symptom and
created a worse one, which this project only discovered once a real microphone
was available: below roughly 0.4, TEN-VAD keeps reporting speech through
silence, so a segment never finalizes and the turn never ends. Measured on a
recorded utterance followed by 15s of quiet, at threshold 0.25 the segment
closed **6.2 seconds** after speech actually stopped -- and over a faint noise
floor it did not close at all until ``max_speech_duration`` forced it.

That is not a tuning nicety. A turn that never ends is a conversation partner
that never answers.

So the threshold has to be chosen against a *specific* microphone, and the
choice needs both sides of the trade measured:

* too high -- real speech is missed entirely (the original WSL2 failure)
* too low  -- silence reads as speech and turns never close (the failure above)

What it measures
----------------

Two signals, swept over candidate thresholds:

**Room tone** (the negative case). Recorded from the actual microphone, in
whatever room and with whatever fan, keyboard and street noise are really
present. Any frame flagged as speech here is a false positive; any *segment*
finalized here is the VAD hallucinating a whole utterance out of nothing.

**Room tone plus known speech** (the positive case). A recorded Korean
utterance is scaled to a realistic speaking level and mixed onto that same room
tone, then a long stretch of room tone follows it. This yields the three
numbers that matter: whether speech was detected at all, how close the segment
start lands to the true onset, and -- the one that was broken -- how long after
speech ends the segment actually closes.

Mixing recorded speech rather than asking the operator to talk is deliberate.
It keeps the noise this room's real noise while making the run unattended,
repeatable and exactly comparable across thresholds. What it does *not* cover
is this speaker's own voice and mic gain; a threshold chosen here that sits
close to the boundary should still be confirmed by speaking into
``_debug_vad_mic.py``.

    # record fresh room tone and sweep
    .venv-win/Scripts/python.exe scripts/_calibrate_vad_threshold.py

    # reuse a previous capture, and write the recommendation into vad.yaml
    .venv-win/Scripts/python.exe scripts/_calibrate_vad_threshold.py \
        --room-wav data/room_tone.wav --apply

Reads ``configs/vad.yaml`` directly rather than importing ``registry``, for the
same reason ``_debug_vad_mic.py`` does: building a VAD should not drag in the
transformers/torch import chain.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import sherpa_onnx
import soundfile as sf
import yaml

from src.nobody_flux.paths import PROJECT_ROOT
from src.nobody_flux.turn.vad import FRAME_SAMPLES, SAMPLE_RATE

VAD_CONFIG_PATH = PROJECT_ROOT / "configs" / "vad.yaml"
TEN_VAD_MODEL = PROJECT_ROOT / "models" / "ten-vad" / "ten-vad.onnx"
# Ships with the SenseVoice model, so no test asset has to be committed.
DEFAULT_SPEECH_WAV = PROJECT_ROOT / "models" / "sense-voice" / "test_wavs" / "ko.wav"
DEFAULT_ROOM_WAV = PROJECT_ROOT / "data" / "room_tone.wav"

# Peak amplitude the reference utterance is scaled to before being mixed onto
# room tone. 0.25 is an ordinary speaking level into a desktop microphone --
# well clear of the noise floor, nowhere near clipping. Sweeping this is how to
# ask "what if the user is quieter?", which is why it is a flag.
DEFAULT_SPEECH_PEAK = 0.25

# Room tone before and after the utterance. The tail has to be long enough that
# a threshold which closes segments *slowly* still closes them inside the
# window -- otherwise a slow threshold and a broken one look identical.
LEAD_S = 1.0
TAIL_S = 8.0

DEFAULT_THRESHOLDS = (0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7)

# Discarded from the front of any capture before it is used as room tone.
# Opening an input stream produces a transient: measured on this project's USB
# microphone, the first ~150ms came back at rms 0.27 with a full-scale peak,
# against a real floor of rms 0.003 -- a 100x overstatement of the noise level,
# which would drag the whole calibration with it. 0.5s covers it with margin.
# (``audio.session`` drops the same warm-up window on the live path, for the
# related reason: that transient reads as speech to the VAD.)
WARMUP_SKIP_S = 0.5


@dataclass
class ThresholdResult:
    """One threshold's behaviour on both the negative and positive signals."""

    threshold: float
    # -- room tone alone (negative) ---------------------------------------
    room_speech_frame_ratio: float  # fraction of frames flagged as speech
    room_segments: int  # utterances hallucinated out of silence
    # -- room tone + speech (positive) ------------------------------------
    detected: bool
    start_error_s: float | None  # segment start minus true onset; <0 = early
    close_lag_s: float | None  # segment finalized this long after speech ended
    speech_len_s: float | None  # how long the captured segment claims to be

    @property
    def clean(self) -> bool:
        """No speech invented from room tone at all."""
        return self.room_segments == 0 and self.room_speech_frame_ratio == 0.0


def recommend(
    results: list[ThresholdResult], min_silence_duration: float, tolerance_s: float = 0.5
) -> tuple[float | None, str]:
    """Pick a threshold, and say why. Pure function -- no audio, no model, so
    the policy can be tested on synthetic rows.

    The rule, in order:

    1. It must not invent speech from room tone. A hallucinated segment is a
       spurious turn: the assistant answers a question nobody asked.
    2. It must detect the real utterance. Obvious, and the failure the original
       0.25 was reaching for.
    3. It must close the segment promptly -- within ``min_silence_duration``
       plus ``tolerance_s``. This is the check that would have caught the 6.2s
       hang, and it is the reason a "detects speech fine" threshold is not
       automatically a usable one.

    Among the thresholds that pass all three, the **lowest** wins: everything
    above it is a larger margin against missing quiet speech, and quiet speech
    is the failure mode a user notices immediately.
    """
    limit = min_silence_duration + tolerance_s
    passing = [
        r
        for r in results
        if r.clean and r.detected and r.close_lag_s is not None and r.close_lag_s <= limit
    ]
    if passing:
        best = min(passing, key=lambda r: r.threshold)
        return (
            best.threshold,
            f"lowest threshold with no false speech in room tone, the utterance "
            f"detected, and the segment closing {best.close_lag_s:.2f}s after speech "
            f"ended (budget {limit:.2f}s)",
        )

    # Nothing passed. Say which requirement did the killing, rather than
    # returning a bare None -- the two failure modes need opposite responses.
    detected_any = [r for r in results if r.detected]
    if not detected_any:
        return None, (
            "no threshold detected the reference utterance at all. The mix level is "
            "probably too low for this mic -- re-run with a larger --speech-peak, or "
            "check that the capture device is the one you think it is."
        )
    return None, (
        "every threshold that detects speech also fails to close the segment in time "
        f"(budget {limit:.2f}s). Raise the sweep's upper end, or lower "
        "min_silence_duration in configs/vad.yaml."
    )


def build_vad(threshold: float, cfg: dict) -> sherpa_onnx.VoiceActivityDetector:
    """A raw sherpa-onnx VAD with everything from vad.yaml except the threshold
    under test. Raw rather than ``VoiceActivityDetector`` because this measures
    the *model's* behaviour, below the pre-roll/barge-in/carry logic that wraps
    it -- mixing the two would make it unclear which layer a result came from."""
    ten = sherpa_onnx.TenVadModelConfig(
        model=str(TEN_VAD_MODEL),
        threshold=threshold,
        min_silence_duration=cfg["min_silence_duration"],
        min_speech_duration=cfg["min_speech_duration"],
        max_speech_duration=cfg["max_speech_duration"],
    )
    return sherpa_onnx.VoiceActivityDetector(
        sherpa_onnx.VadModelConfig(
            ten_vad=ten, sample_rate=SAMPLE_RATE, num_threads=cfg.get("num_threads", 1)
        ),
        buffer_size_in_seconds=100,
    )


def _feed(vad, audio: np.ndarray):
    """Push audio frame by frame, yielding ``(fed_samples, speaking, segment)``
    where segment is ``(start, length)`` or None. Frame-sized pushes, because
    that is what the live path does and VAD state is path dependent."""
    fed = 0
    for offset in range(0, len(audio), FRAME_SAMPLES):
        frame = audio[offset : offset + FRAME_SAMPLES]
        if len(frame) < FRAME_SAMPLES:
            frame = np.pad(frame, (0, FRAME_SAMPLES - len(frame)))
        vad.accept_waveform(frame)
        fed += FRAME_SAMPLES
        speaking = vad.is_speech_detected()
        segment = None
        if not vad.empty():
            seg = vad.front
            segment = (seg.start, len(seg.samples))
            vad.pop()
        yield fed, speaking, segment


def evaluate(threshold: float, cfg: dict, room: np.ndarray, mixed: np.ndarray,
             speech_start_s: float, speech_end_s: float) -> ThresholdResult:
    room_vad = build_vad(threshold, cfg)
    speech_frames = 0
    total_frames = 0
    room_segments = 0
    for _fed, speaking, segment in _feed(room_vad, room):
        total_frames += 1
        speech_frames += int(speaking)
        room_segments += int(segment is not None)

    mix_vad = build_vad(threshold, cfg)
    start_error = close_lag = speech_len = None
    for fed, _speaking, segment in _feed(mix_vad, mixed):
        if segment is None:
            continue
        start, length = segment
        # First segment only: with a correct threshold there is exactly one, and
        # if a low threshold fragments the utterance the first fragment is still
        # the one that tells us where onset was detected.
        start_error = start / SAMPLE_RATE - speech_start_s
        speech_len = length / SAMPLE_RATE
        close_lag = fed / SAMPLE_RATE - speech_end_s
        break

    return ThresholdResult(
        threshold=threshold,
        room_speech_frame_ratio=speech_frames / total_frames if total_frames else 0.0,
        room_segments=room_segments,
        detected=close_lag is not None,
        start_error_s=start_error,
        close_lag_s=close_lag,
        speech_len_s=speech_len,
    )


def record_room_tone(seconds: float, out_path: Path) -> np.ndarray:
    """Capture room tone from the real microphone and save it.

    Saved because it is the input that makes every later number reproducible:
    re-running the sweep against the same capture is how a change in policy is
    told apart from a change in the room.
    """
    import sounddevice as sd

    print(f"Recording {seconds:.0f}s of room tone -- stay quiet...")
    frames: list[np.ndarray] = []
    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=FRAME_SAMPLES
    ) as stream:
        for _ in range(int(seconds * SAMPLE_RATE / FRAME_SAMPLES)):
            block, _overflowed = stream.read(FRAME_SAMPLES)
            frames.append(block[:, 0].copy())
    audio = np.concatenate(frames) if frames else np.zeros(0, dtype=np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Saved *including* the warm-up transient, and trimmed only on the way into
    # the sweep. The transient is real data about this device -- keeping it in
    # the file is what let it be diagnosed in the first place.
    sf.write(str(out_path), audio, SAMPLE_RATE)
    print(f"Saved to {out_path}")
    return audio


def load_mono_16k(path: Path) -> np.ndarray:
    audio, rate = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if rate != SAMPLE_RATE:
        raise SystemExit(f"{path} is {rate}Hz; this project's capture path is {SAMPLE_RATE}Hz.")
    return audio


def speech_extent(audio: np.ndarray, relative: float = 0.05) -> tuple[int, int]:
    """First and last sample of actual speech in a reference clip.

    A recorded utterance is not speech from sample zero: ``ko.wav`` opens with
    roughly 0.8s of studio silence. Measuring the VAD's onset error against the
    *file's* start would charge it for that silence and report an 0.8s lag that
    is not there. Anything above 5% of peak is speech for this purpose -- the
    clip is clean, so the exact cut hardly matters; what matters is not using 0.
    """
    if len(audio) == 0:
        return 0, 0
    loud = np.flatnonzero(np.abs(audio) >= relative * float(np.max(np.abs(audio))))
    if len(loud) == 0:
        return 0, len(audio)
    return int(loud[0]), int(loud[-1]) + 1


def tile_to(audio: np.ndarray, length: int) -> np.ndarray:
    """Repeat room tone to fill ``length`` samples. Tiling rather than padding
    with zeros: digital zeros are not silence as far as a VAD is concerned (the
    original harness proved that), so every sample of the test signal has to
    carry real noise."""
    if len(audio) == 0:
        return np.zeros(length, dtype=np.float32)
    reps = int(np.ceil(length / len(audio)))
    return np.tile(audio, reps)[:length].astype(np.float32)


def write_threshold(value: float) -> None:
    text = VAD_CONFIG_PATH.read_text(encoding="utf-8")
    new, n = re.subn(r"(?m)^threshold:.*$", f"threshold: {value}", text)
    if n == 0:
        new = text.rstrip() + f"\nthreshold: {value}\n"
    VAD_CONFIG_PATH.write_text(new, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seconds", type=float, default=6.0,
                        help="how much room tone to record (default: 6)")
    parser.add_argument("--room-wav", type=Path, default=None,
                        help="reuse a previous room-tone capture instead of recording")
    parser.add_argument("--speech-wav", type=Path, default=DEFAULT_SPEECH_WAV,
                        help="reference utterance to mix in")
    parser.add_argument("--speech-peak", type=float, default=DEFAULT_SPEECH_PEAK,
                        help=f"peak level to scale it to (default: {DEFAULT_SPEECH_PEAK})")
    parser.add_argument("--thresholds", type=float, nargs="+", default=list(DEFAULT_THRESHOLDS))
    parser.add_argument("--skip-start", type=float, default=WARMUP_SKIP_S,
                        help="discard this many seconds from the front of the room tone "
                             f"(device warm-up transient; default: {WARMUP_SKIP_S})")
    parser.add_argument("--apply", action="store_true",
                        help="write the recommended threshold into configs/vad.yaml")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = yaml.safe_load(VAD_CONFIG_PATH.read_text(encoding="utf-8"))

    if args.room_wav is not None:
        room = load_mono_16k(args.room_wav)
        print(f"Room tone: {args.room_wav} ({len(room)/SAMPLE_RATE:.1f}s)")
    else:
        room = record_room_tone(args.seconds, DEFAULT_ROOM_WAV)

    skip = int(SAMPLE_RATE * args.skip_start)
    if len(room) > skip:
        room = room[skip:]

    if len(room) < SAMPLE_RATE:
        raise SystemExit(
            f"Need at least 1s of room tone after discarding the first "
            f"{args.skip_start:.1f}s of device warm-up."
        )

    room_rms = float(np.sqrt(np.mean(room.astype("float64") ** 2)))
    room_peak = float(np.max(np.abs(room)))
    print(f"noise floor: rms={room_rms:.5f} peak={room_peak:.5f}")
    if room_peak == 0.0:
        raise SystemExit(
            "The capture is digital silence -- the microphone is muted or not "
            "actually being read. Fix that first; a threshold calibrated against "
            "zeros is meaningless."
        )

    speech = load_mono_16k(args.speech_wav)
    peak = float(np.max(np.abs(speech)))
    if peak > 0:
        speech = speech * (args.speech_peak / peak)
    speech_rms = float(np.sqrt(np.mean(speech.astype("float64") ** 2)))
    snr_db = 20 * np.log10(speech_rms / room_rms) if room_rms > 0 else float("inf")
    print(
        f"speech: {args.speech_wav.name} {len(speech)/SAMPLE_RATE:.2f}s "
        f"scaled to peak {args.speech_peak} (SNR {snr_db:.1f} dB)"
    )

    lead, tail = int(SAMPLE_RATE * LEAD_S), int(SAMPLE_RATE * TAIL_S)
    total = lead + len(speech) + tail
    mixed = tile_to(room, total).copy()
    mixed[lead : lead + len(speech)] += speech
    # Against where speech genuinely starts and stops inside the clip, not
    # against the clip's own boundaries -- see speech_extent.
    onset, offset = speech_extent(speech)
    speech_start_s = (lead + onset) / SAMPLE_RATE
    speech_end_s = (lead + offset) / SAMPLE_RATE
    print(
        f"speech extent within clip: {onset/SAMPLE_RATE:.2f}s..{offset/SAMPLE_RATE:.2f}s "
        f"({(offset-onset)/SAMPLE_RATE:.2f}s of actual speech)"
    )

    print(
        f"\n{'thresh':>7} {'room speech%':>13} {'room segs':>10} "
        f"{'detected':>9} {'start err':>10} {'close lag':>10} {'seg len':>8}"
    )
    results: list[ThresholdResult] = []
    for threshold in sorted(args.thresholds):
        r = evaluate(threshold, cfg, room, mixed, speech_start_s, speech_end_s)
        results.append(r)
        print(
            f"{r.threshold:>7.2f} {r.room_speech_frame_ratio*100:>12.1f}% "
            f"{r.room_segments:>10} {str(r.detected):>9} "
            f"{'' if r.start_error_s is None else f'{r.start_error_s:>+9.2f}s'} "
            f"{'' if r.close_lag_s is None else f'{r.close_lag_s:>9.2f}s'} "
            f"{'' if r.speech_len_s is None else f'{r.speech_len_s:>7.2f}s'}"
        )

    choice, why = recommend(results, cfg["min_silence_duration"])
    print()
    if choice is None:
        print(f"No usable threshold: {why}")
        return 1

    print(f"recommended threshold: {choice}\n  because {why}")
    print(f"  (currently in configs/vad.yaml: {cfg['threshold']})")
    if args.apply:
        write_threshold(choice)
        print(f"  written to {VAD_CONFIG_PATH}")
    else:
        print("  re-run with --apply to write it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
