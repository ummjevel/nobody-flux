#!/usr/bin/env python3
"""Verify Phase 3 (streaming ASR) and Phase 4 (turn controller) without a microphone.

Every turn-taking parameter in this project is currently an estimate, because
the development environment could not open a real microphone. That is a
measurement problem and this script does not solve it. What it does solve is
the *other* half: proving the machinery is correctly wired, by replaying
recorded audio through exactly the code paths a live microphone drives.

Four checks, each isolating a different failure mode:

1. **LocalAgreement**, on plain strings. No model, no audio. If prefix
   stabilization is wrong, it is wrong here, and finding that out costs
   milliseconds instead of a confusing transcript.

2. **Streaming vs batch recognition**, on a real wav. The streaming transducer
   is fed frame by frame through the live path (``transcribe_array``) and its
   output compared against the batch decode of the same file. They will not be
   byte-identical -- different amounts of right-context reach the decoder --
   but a large divergence means the incremental path is broken rather than
   merely different.

3. **The turn controller**, driven by a synthetic frame source that replays
   speech surrounded by this machine's own recorded room tone. This exercises
   the capture thread, the VAD state machine, the pre-roll ring and the queue
   hand-off, and asserts that exactly one turn comes out the other side.

4. **The two together**, with the streaming recognizer attached to the
   controller -- the configuration ``talk.py --streaming-asr`` runs. The turn
   must arrive already carrying its text. Checked separately because each half
   passing says nothing about the hand-off between them.

A note on the "silence" in check 3: it is recorded room tone, not
``np.zeros``. An all-zero frame is degenerate input to the VAD's feature
extractor and TEN-VAD reports *speech* through it, which had this harness
failing while the code under test was correct. Anything that pads a signal for
this model has to pad it with real noise.

What this deliberately does NOT check: whether the thresholds are any good.
The VAD threshold is measured separately against this machine's own noise floor
(scripts/_calibrate_vad_threshold.py); the barge-in and backchannel durations
need real spoken samples (scripts/_calibrate_turn_params.py,
docs/barge-in-design.md).

    python scripts/_smoke_turn.py
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import soundfile as sf

from _calibrate_vad_threshold import speech_extent
from src.nobody_flux.paths import PROJECT_ROOT
from src.nobody_flux.turn.vad import FRAME_SAMPLES, SAMPLE_RATE

# A Korean utterance that ships with the SenseVoice model, so this runs on any
# machine that completed setup without needing a committed test asset.
TEST_WAV = PROJECT_ROOT / "models" / "sense-voice" / "test_wavs" / "ko.wav"


ROOM_TONE_WAV = PROJECT_ROOT / "data" / "room_tone.wav"

# Level of the synthetic noise floor used when no room-tone capture exists.
# Matches what scripts/_calibrate_vad_threshold.py measured on this project's
# Windows USB microphone (rms 0.0032), so the fallback is at least the right
# order of magnitude rather than an invented number.
FALLBACK_NOISE_RMS = 0.003


def silence(seconds: float, room: np.ndarray | None) -> np.ndarray:
    """A stretch of *plausible* silence -- room tone, not digital zeros.

    This distinction is not pedantry, it cost this project a day. The harness
    originally padded with ``np.zeros``, and TEN-VAD reported speech straight
    through it: an all-zero frame is degenerate input to the feature extractor,
    and the model's output over it says nothing about how it behaves in a quiet
    room. The result was a smoke test that failed while the code under test was
    fine, and would equally have passed while it was broken.

    Real room tone from ``data/room_tone.wav`` is used when it exists (write it
    with scripts/_calibrate_vad_threshold.py); otherwise a Gaussian floor at the
    level that capture measured. Fixed seed, so a failure reproduces.
    """
    length = int(SAMPLE_RATE * seconds)
    if room is not None and len(room) > 0:
        reps = int(np.ceil(length / len(room)))
        return np.tile(room, reps)[:length].astype(np.float32)
    rng = np.random.default_rng(0)
    return (rng.standard_normal(length) * FALLBACK_NOISE_RMS).astype(np.float32)


def load_room_tone() -> np.ndarray | None:
    """This machine's recorded noise floor, if it has one. Optional by design:
    the smoke test has to run on a box that has never had a microphone."""
    if not ROOM_TONE_WAV.exists():
        return None
    audio, rate = sf.read(str(ROOM_TONE_WAV), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if rate != SAMPLE_RATE:
        return None
    # Drop the device warm-up transient, which is the loudest thing in the file
    # and is not room tone at all (see audio.session.WARMUP_FRAMES).
    return audio[SAMPLE_RATE // 2 :] if len(audio) > SAMPLE_RATE else None


def load_test_audio() -> np.ndarray:
    """Mono float32 at SAMPLE_RATE. Fails loudly on a rate mismatch rather than
    resampling: the capture path is fixed at 16kHz, and silently converting
    here would hide a genuine configuration problem."""
    audio, rate = sf.read(str(TEST_WAV), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if rate != SAMPLE_RATE:
        raise SystemExit(f"{TEST_WAV} is {rate}Hz; this harness assumes {SAMPLE_RATE}Hz.")
    return audio


def check_local_agreement() -> bool:
    """The LocalAgreement rule, on strings."""
    from src.nobody_flux.stage.asr_stream import longest_common_prefix

    cases = [
        # (hypotheses, expected committed prefix, what it demonstrates)
        (["그래서내가", "그래서내가말"], "그래서내가", "stable prefix, growing tail"),
        (["그래서내가", "그래서냈어"], "그래서", "tail revised -> only the agreed part commits"),
        (["안녕", "안녕"], "안녕", "identical hypotheses commit fully"),
        (["가나다", "라마바"], "", "no agreement -> nothing commits"),
        ([], "", "no hypotheses at all"),
        (["하나"], "하나", "a single hypothesis is its own prefix"),
        (["짧다", "짧다길어짐", "짧"], "짧", "3-way agreement is bounded by the shortest"),
    ]
    ok = True
    for hypotheses, expected, description in cases:
        actual = longest_common_prefix(hypotheses)
        status = "ok  " if actual == expected else "FAIL"
        if actual != expected:
            ok = False
        print(f"  {status} {description}: {hypotheses} -> {actual!r} (expected {expected!r})")
    return ok


def check_streaming_asr(audio: np.ndarray) -> bool:
    """Streaming decode of a real utterance, against the batch decode."""
    from src.nobody_flux import registry
    from src.nobody_flux.stage.asr import StreamingZipformerAsr

    transcriber = registry.build_streaming_transcriber()

    started = time.perf_counter()
    streamed = transcriber.transcribe_array(audio)
    streaming_ms = (time.perf_counter() - started) * 1000

    batch = StreamingZipformerAsr()
    started = time.perf_counter()
    batched = batch.transcribe_file(str(TEST_WAV))
    batch_ms = (time.perf_counter() - started) * 1000

    print(f"  streaming: {streamed!r}  ({streaming_ms:.0f}ms wall)")
    print(f"  batch    : {batched!r}  ({batch_ms:.0f}ms wall)")
    print(f"  committed prefix at end: {transcriber.committed!r}")
    print(f"  endpoint detected      : {transcriber.endpoint_detected}")

    if not streamed:
        print("  FAIL streaming decode produced no text")
        return False

    # Character overlap rather than equality. The two paths legitimately differ:
    # the streaming decoder commits with limited right-context while the batch
    # one sees the whole utterance. A high overlap means the incremental path is
    # tracking the same hypothesis; a low one means it is broken.
    from difflib import SequenceMatcher

    ratio = SequenceMatcher(None, streamed, batched).ratio()
    print(f"  similarity to batch    : {ratio:.2f}")
    if ratio < 0.6:
        print("  FAIL streaming and batch decodes diverge too far")
        return False
    return True


def check_turn_controller(audio: np.ndarray, with_transcriber: bool = False) -> bool:
    """The capture thread and turn state machine, on a synthetic frame source.

    With ``with_transcriber`` this is the Phase 3 + Phase 4 join: the same
    frames drive both the VAD and the streaming recognizer, and the finished
    turn should arrive already carrying its text. That is the configuration
    ``talk.py --streaming-asr`` runs, and the one where recognition costs the
    turn nothing -- worth checking directly, since each half passing alone says
    nothing about the hand-off between them.
    """
    from src.nobody_flux import registry
    from src.nobody_flux.turn.controller import TurnController, TurnState
    from src.nobody_flux.turn.vad import VadEvent

    # Room tone, speech, then enough room tone for the VAD to close the segment.
    # 2.5s of tail is comfortably past configs/vad.yaml's min_silence_duration
    # (0.5s) plus the ~0.3s the model takes to actually let go at the calibrated
    # threshold -- see the close-lag column in _calibrate_vad_threshold.py.
    room = load_room_tone()
    print(f"  noise floor    : {'data/room_tone.wav' if room is not None else 'synthetic'}")
    script = np.concatenate([silence(0.5, room), audio, silence(2.5, room)])

    cursor = 0
    exhausted = threading.Event()
    # Room tone to serve once the script runs out, walked cyclically so the
    # frames vary the way a live device's do. Returning one identical frame over
    # and over would be a periodic signal, which is not what a quiet room sounds
    # like to a model trained on real audio.
    idle_tone = silence(10.0, room)
    idle_cursor = 0

    def frame_source() -> np.ndarray:
        """Replay the script one frame at a time, then block.

        Blocking at the end rather than raising or returning silence forever is
        what a real device does between utterances, so the controller sees the
        same shape of input it will see live. The event lets shutdown release
        it deterministically instead of leaving a thread spinning.
        """
        nonlocal cursor, idle_cursor
        if cursor >= len(script):
            exhausted.set()
            time.sleep(0.05)
            # Room tone, not zeros, for the same reason the script itself uses
            # it: an all-zero frame is degenerate input the VAD reads as speech.
            idle_cursor = (idle_cursor + FRAME_SAMPLES) % (len(idle_tone) - FRAME_SAMPLES)
            return idle_tone[idle_cursor : idle_cursor + FRAME_SAMPLES]
        frame = script[cursor : cursor + FRAME_SAMPLES]
        cursor += FRAME_SAMPLES
        if len(frame) < FRAME_SAMPLES:
            frame = np.pad(frame, (0, FRAME_SAMPLES - len(frame)))
        return frame

    seen: list[tuple[VadEvent, TurnState]] = []
    transcriber = registry.build_streaming_transcriber() if with_transcriber else None
    controller = TurnController(
        vad=registry.build_vad(),
        frame_source=frame_source,
        player_factory=None,  # nothing is played in this harness
        transcriber=transcriber,
        on_event=lambda event, state: seen.append((event, state)),
    )

    controller.start()
    try:
        # Generous relative to the ~11s of scripted audio: the whole point is to
        # distinguish "no turn was produced" from "the machine is just slow".
        turn = controller.next_turn(timeout=30.0)
    finally:
        controller.stop()

    events = [event.value for event, _state in seen]
    print(f"  events observed: {events}")

    if turn is None:
        print("  FAIL no turn was captured")
        return False

    # Against the clip's *speech*, not its file length. ko.wav opens with 0.78s
    # of studio silence and ends with about a second more; measuring against
    # 4.61s of file would demand the VAD capture silence it is specifically
    # built to drop, and the bound would then be satisfied only by accident.
    onset, offset = speech_extent(audio)
    speech_s = (offset - onset) / SAMPLE_RATE
    print(f"  turn index     : {turn.index}")
    print(
        f"  turn duration  : {turn.duration_s:.2f}s "
        f"(speech in clip {speech_s:.2f}s of {len(audio)/SAMPLE_RATE:.2f}s file)"
    )
    expectation = "recognized during capture" if with_transcriber else "no transcriber attached"
    print(f"  streamed text  : {turn.streamed_text!r} ({expectation})")

    if VadEvent.SPEECH_STARTED.value not in events:
        print("  FAIL speech was never detected")
        return False

    if with_transcriber and not turn.streamed_text:
        print(
            "  FAIL the transcriber was attached but the turn arrived with no text -- "
            "the capture thread is not feeding it, or finalize() came back empty"
        )
        return False

    # Checked against speech_duration_s, not the buffer length: the buffer
    # includes pre_roll_ms of padding, and slack wide enough to absorb it is
    # exactly how the 300->500ms pre-roll growth silently killed the
    # backchannel gate (docs/code-review-20260814.md #1). Lower bound: much
    # under the speech itself means the front of the turn was lost. Upper
    # bound: the model's measured let-go lag at the calibrated threshold
    # (~0.8s) -- room tone swept in past that inflates every downstream
    # duration decision.
    if not (speech_s * 0.9 <= turn.speech_duration_s <= speech_s + 1.0):
        print(
            f"  FAIL speech duration {turn.speech_duration_s:.2f}s "
            f"(buffer {turn.duration_s:.2f}s) is implausible for "
            f"{speech_s:.2f}s of speech"
        )
        return False
    return True


def main() -> int:
    results: list[tuple[str, bool]] = []

    # Model-free check first, so a missing wav (below) can never mask a pure
    # logic failure. The same cases also run in tests/test_small_pure.py --
    # kept here too so this script remains a self-contained smoke pass.
    print("\n[1/4] LocalAgreement prefix stabilization")
    results.append(("local-agreement", check_local_agreement()))

    if not TEST_WAV.exists():
        raise SystemExit(
            f"Missing {TEST_WAV}. Run the setup script for your platform first "
            "(scripts/setup_windows.ps1, setup_local.sh, setup_server.sh, or setup_mac.sh)."
        )

    audio = load_test_audio()

    print("\n[2/4] Streaming ASR (Phase 3)")
    results.append(("streaming-asr", check_streaming_asr(audio)))

    print("\n[3/4] Turn controller (Phase 4)")
    results.append(("turn-controller", check_turn_controller(audio)))

    print("\n[4/4] Turn controller + streaming ASR (Phase 3 x 4)")
    results.append(
        ("turn-controller-streaming", check_turn_controller(audio, with_transcriber=True))
    )

    print("\n" + "-" * 60)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    failed = [name for name, ok in results if not ok]
    if failed:
        print(f"\n{len(failed)} check(s) failed: {', '.join(failed)}")
        return 1
    print("\nAll checks passed -- wiring, not tuning. Of the turn-taking parameters,")
    print("configs/vad.yaml's threshold is now measured on this machine's microphone")
    print("(scripts/_calibrate_vad_threshold.py); barge_in_confirm_ms and the")
    print("backchannel duration bound still need real spoken samples, which only")
    print("scripts/_calibrate_turn_params.py can collect.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
