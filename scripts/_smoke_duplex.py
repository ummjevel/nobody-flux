#!/usr/bin/env python3
"""Does the assistant interrupt itself? Measured, on the real device.

The single assumption the duplex/AEC work rests on has never been tested. The
turn controller keeps capturing while a reply plays -- that is the whole point
of Phase 4 -- and the microphone can hear the speaker (28ms of acoustic delay,
measured here by _calibrate_aec_delay.py). So every reply is also an input. If
the echo canceller does not remove it, the VAD sees the assistant's own voice,
calls it an interruption, and cuts the reply off mid-sentence. The conversation
then fails in the most confusing way available: it talks over itself.

Nothing in the repo checks this, because until there was a machine with both a
working microphone and a working speaker there was no way to.

What it does
------------

Plays a known utterance through the duplex session while the turn controller
captures, then reports what the VAD made of it. Twice: once with the echo
canceller the platform actually selects, and once with it disabled.

Reported per run:

* ``barge_in``      false interruptions. Must be 0.
* ``events``        what the VAD saw. ``speech_started`` present while
  ``barge_in`` is 0 means the gate held: the echo registered as speech but
  never lasted long enough to be confirmed as an interruption.
* ``turns``         utterances the echo produced out of nothing. Must be 0.
* ``quiet -> during playback``  the captured level before anything played,
  and while it did. Both from the same stream, seconds apart, because capture
  gain on consumer USB microphones drifts: measured at up to 2.8x between runs
  with the backend held fixed, which is larger than the echo being looked for.
  Comparing two separate runs to each other measures that drift; comparing a
  run against its own baseline measures the echo.

If playback barely lifts the level above the baseline, no echo reached the
microphone (headphones, distance, a quiet speaker) and a clean result is
evidence about *this setup*, not about the canceller. The report says which of
the two it got, rather than reporting a pass it did not earn.

    .venv-win/Scripts/python.exe scripts/_smoke_duplex.py

Needs a real speaker and microphone in the same room. On a machine with neither
(CI, a headless server) it exits 0 with a skip -- there is nothing to measure,
and failing would only teach people to ignore it.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import soundfile as sf

from src.nobody_flux import registry
from src.nobody_flux.audio.player import SessionPlayer
from src.nobody_flux.paths import PROJECT_ROOT
from src.nobody_flux.turn.controller import TurnController
from src.nobody_flux.turn.vad import SAMPLE_RATE, VadEvent

TEST_WAV = PROJECT_ROOT / "models" / "sense-voice" / "test_wavs" / "ko.wav"

# Played back at a normal listening level. Loud enough that the microphone
# genuinely hears it -- the whole point is to create the echo, not to avoid it.
PLAYBACK_PEAK = 0.5

# Quiet capture taken at the start of each run, before anything plays, as that
# run's own reference level.
BASELINE_S = 1.5

# How far playback must lift the captured level above that baseline before the
# echo is considered present at all. Below this, a clean barge-in result says
# only that this particular setup is quiet -- it is not evidence about the
# canceller, and the report says so rather than claiming a pass it did not earn.
ECHO_PRESENT_LIFT = 1.5


def load_reply() -> np.ndarray:
    audio, rate = sf.read(str(TEST_WAV), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if rate != SAMPLE_RATE:
        raise SystemExit(f"{TEST_WAV} is {rate}Hz; expected {SAMPLE_RATE}Hz.")
    peak = float(np.max(np.abs(audio)))
    return audio * (PLAYBACK_PEAK / peak) if peak else audio


def run_once(backend: str, reply: np.ndarray) -> dict:
    """One playback through one backend, with capture running throughout."""
    session = registry.build_audio_session(backend)
    session.start()

    counts = {event: 0 for event in VadEvent}
    captured: list[np.ndarray] = []

    def frame_source() -> np.ndarray:
        frame = session.read_frame()
        # Recorded here rather than inside the session so the measurement sees
        # exactly what the VAD sees -- post-cancellation, post-warm-up.
        captured.append(frame)
        return frame

    controller = TurnController(
        vad=registry.build_vad(),
        frame_source=frame_source,
        player_factory=lambda: SessionPlayer(session),
        on_event=lambda event, _state: counts.__setitem__(event, counts[event] + 1),
    )
    controller.start()
    try:
        # Baseline: the same microphone, through the same stream, with nothing
        # playing. Every level below is measured against this rather than
        # against a separately recorded room tone, because capture gain on this
        # class of device drifts between streams -- measured at up to 2.8x
        # run-to-run with the backend held fixed, which is larger than the
        # effect being looked for. A within-run baseline cannot drift relative
        # to the thing it is the baseline for.
        time.sleep(BASELINE_S)
        baseline_end = len(captured)

        player = controller.begin_response()
        try:
            player.enqueue(reply, SAMPLE_RATE)
            player.done()
            # Bounded well past the clip's own length: if playback wedges, the
            # run should report that rather than hang.
            player.join(timeout=len(reply) / SAMPLE_RATE + 5.0)
        finally:
            controller.finish_response()
        # A confirmed barge-in needs barge_in_confirm_ms of continued speech, so
        # give the tail of the echo a moment to be judged before tearing down.
        time.sleep(0.5)
        turns = 0
        while controller.next_turn(timeout=0.1) is not None:
            turns += 1
    finally:
        controller.stop()
        session.close()

    def rms(frames: list[np.ndarray]) -> float:
        if not frames:
            return 0.0
        signal = np.concatenate(frames).astype("float64")
        return float(np.sqrt(np.mean(signal**2)))

    quiet, during = captured[:baseline_end], captured[baseline_end:]
    return {
        "backend": backend,
        "barge_in": controller.barge_in_count,
        "events": {event.value: count for event, count in counts.items() if count},
        "turns": turns,
        "baseline_rms": rms(quiet),
        "during_rms": rms(during),
        "during_peak": float(np.max(np.abs(np.concatenate(during)))) if during else 0.0,
        "captured_s": sum(len(f) for f in captured) / SAMPLE_RATE,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["auto", "off"],
        help="backends to run, in order. The default pairs the platform's real "
        "choice with a no-cancellation control. Passing the same backend twice "
        "(e.g. --backends off off) measures this device's run-to-run capture "
        "variance, which is what any comparison between backends has to beat "
        "before it means anything.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not TEST_WAV.exists():
        raise SystemExit(f"Missing {TEST_WAV}. Run scripts/setup_windows.ps1 first.")

    try:
        import sounddevice as sd

        devices = sd.query_devices()
        has_input = any(d["max_input_channels"] > 0 for d in devices)
        has_output = any(d["max_output_channels"] > 0 for d in devices)
    except Exception as exc:  # pragma: no cover - depends on the host
        print(f"SKIP: no usable audio backend ({exc})")
        return 0
    if not (has_input and has_output):
        print("SKIP: this machine has no microphone and speaker pair to measure.")
        return 0

    reply = load_reply()
    print(f"reply: {TEST_WAV.name}, {len(reply)/SAMPLE_RATE:.2f}s at peak {PLAYBACK_PEAK}")
    print(
        f"Playing it aloud {len(args.backends)} time(s). Stay quiet -- anything you "
        "say counts as a barge-in.\n"
    )

    results = []
    # The canceller the platform would actually pick, then none at all. Running
    # 'auto' first means a failure is reported against the real configuration
    # rather than against a control that nobody ships.
    for backend in args.backends:
        result = run_once(backend, reply)
        results.append(result)
        base = result["baseline_rms"]
        lift = f"{result['during_rms']/base:.2f}x" if base > 0 else "n/a"
        print(
            f"  {backend:>5}: barge_in={result['barge_in']} turns={result['turns']} "
            f"events={result['events'] or '{}'}\n"
            f"         quiet rms={base:.5f} -> during playback {result['during_rms']:.5f} "
            f"({lift} baseline, peak {result['during_peak']:.3f})"
        )

    live = results[0]
    control = results[-1]
    print()
    ok = True
    if live["barge_in"]:
        print(
            f"  FAIL the reply interrupted itself {live['barge_in']} time(s). The echo "
            "canceller is not removing enough of the playback, or delay_frames in "
            "configs/audio.yaml is wrong (re-run _calibrate_aec_delay.py)."
        )
        ok = False
    if live["turns"]:
        print(f"  FAIL the echo produced {live['turns']} phantom turn(s).")
        ok = False

    # How much the playback lifted the captured level above that same run's own
    # quiet baseline. This is the honest measure of how much echo reached the
    # microphone; a lift near 1.0 means essentially none did.
    lift = (
        control["during_rms"] / control["baseline_rms"] if control["baseline_rms"] > 0 else 0.0
    )
    if lift < ECHO_PRESENT_LIFT:
        print(
            f"  NOTE playback lifted the captured level only {lift:.2f}x above this "
            "run's own baseline, so almost no echo reached the microphone -- "
            "headphones, a quiet speaker, or distance. The clean result above is "
            "therefore evidence that nothing self-interrupts in THIS setup, and "
            "says nothing yet about the echo canceller, which never had anything "
            "to cancel. To exercise it, re-run with speakers rather than "
            "headphones and the volume up."
        )
    else:
        print(f"  echo raised the captured level {lift:.2f}x above baseline")
    if ok:
        print("\nPASS the assistant did not interrupt itself.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
