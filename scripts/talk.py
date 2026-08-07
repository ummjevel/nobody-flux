#!/usr/bin/env python3
"""Continuous voice conversation loop: mic in, speaker out, no restart between turns.

Unlike run_pipeline.py (one wav in, one wav out, process exits), this keeps a
single STSPipeline alive for the whole session -- so NobodyLLM.history carries
across turns and it's an actual multi-turn conversation, not N independent
one-shot calls.

Turn boundaries come from vad.py's TEN-VAD-based VAD (no wakeword, no
push-to-talk): it starts recording when you start talking and stops when you
stop. See vad.py's docstring for the tuning knobs and its limits -- if it cuts
you off early or won't stop listening, that's threshold tuning, not a bug to
route around here.

Barge-in: listening for the next utterance starts as soon as this turn's
reply begins playing (not after playback finishes), so speaking while nobody
is still talking can cut the reply off instead of waiting it out. Two-stage
disambiguation from backchannel ("어", "응", common given persona.py's casual
tone) per docs/barge-in-design.md: stage 1 (real time, see
on_barge_in_confirmed/vad.py's barge_in_confirm_ms) only cuts playback once
speech has continued past a short duration threshold, so most backchannel
never trips it at all; stage 2 (after ASR, see is_backchannel below) catches
short utterances stage 1 let through and skips turn processing for them
entirely (no LLM/TTS/storage) rather than treating them as a real reply.

No echo cancellation: the mic is listening the whole time nobody's reply
plays out of the speaker, and there's no acoustic-echo-cancellation step
between them, so on setups where the reply bleeds back into the mic loud
enough for TEN-VAD to mistake it for speech, that will register as a
false-positive barge-in (nobody interrupting itself). Not observed on this
dev box's WSL2/WSLg passthrough setup (playback and mic capture are
different logical devices there), but worth knowing if it happens on a
setup with real speaker-into-mic leakage -- fix would be AEC or a
headset/earbuds, not a VAD threshold tweak.

WSL note: this machine is WSL2. Mic capture through WSLg's audio passthrough
is not guaranteed to work -- if sounddevice can't see an input device, or
listen_for_utterance() hangs, test on the H100 server (native Linux) instead,
or fall back to run_pipeline.py with pre-recorded wav files.

Usage:
    uv run python scripts/talk.py [--asr PRESET] [--llm PRESET] [--tts PRESET] [--voice VOICE]

--voice picks a TTS reference clip from configs/voices.yaml.
"""

import argparse
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import soundfile as sf
import sounddevice as sd
from loguru import logger

from _cli import add_pipeline_args, build_pipeline_from_args
from src.nobody_flux import registry
from src.nobody_flux.backchannel import is_backchannel
from src.nobody_flux.memory import extract_memories, format_recall_block
from src.nobody_flux.paths import PROJECT_ROOT
from src.nobody_flux.storage import ConversationStore

SESSION_AUDIO_DIR = PROJECT_ROOT / "data" / "sessions"

# Safety margin added on top of the reply clip's own duration when deciding
# a playback has hung (see play_with_timeout below).
PLAYBACK_TIMEOUT_MARGIN_S = 5.0

# Spoken once at session start, after models finish loading -- audible
# confirmation the session is actually ready and listening, instead of the
# first sign of life being silence until you happen to say something the VAD
# catches. Plain Korean text (no numbers/English/emoji), same TTS-friendliness
# rules as persona.py's SYSTEM_PROMPT.
GREETING_TEXT = "안녕, 나 지금 듣고 있어."


def play_async(audio, sample_rate: int, playback_active: threading.Event) -> threading.Thread:
    """Like sd.play() + sd.wait(), but runs on a background thread (so the
    caller can start listening for the next utterance immediately, for
    barge-in -- see module docstring) and never blocks that thread forever.

    playback_active is set for the duration of this specific clip's
    playback and cleared right before returning -- it's how on_speech_start
    (below, in main()) decides whether "speech just started" means
    "interrupt the reply that's currently playing" or "just the normal start
    of the next turn." Cleared in a `finally` so a timeout or an explicit
    sd.stop() from a barge-in both still leave it correctly cleared.

    sd.wait() has no timeout parameter -- if the underlying audio backend
    stalls mid-playback, this would otherwise hang forever, and from a caller
    stuck in the main while-loop that's indistinguishable from the session
    simply never returning to "listening" for the next turn.

    First version of this polled sd.get_stream().active with a deadline
    instead of calling sd.wait() at all -- confirmed by hand that was wrong:
    on this dev box's WSL2/WSLg PulseAudio passthrough, .active stayed True
    for the full (clip duration + margin) on EVERY turn regardless of
    whether playback had actually already finished, so every single reply
    paid the full margin as dead time even when nothing was stalled. Calling
    the real sd.wait() on a background thread and joining it with a timeout
    gets fast completion back (wait() returns as soon as playback genuinely
    ends, including a barge-in's early sd.stop()) while still bounding the
    wait if the backend truly does stall.

    Returns the background thread so the caller can join() it before
    starting the *next* clip's playback (sd.play() itself would just cut the
    previous clip off, but leaving the previous watchdog thread running past
    that point serves no purpose).
    """
    playback_active.set()
    sd.play(audio, sample_rate)
    timeout = len(audio) / sample_rate + PLAYBACK_TIMEOUT_MARGIN_S

    def _wait():
        done = threading.Event()

        def _mark_done():
            sd.wait()
            done.set()

        threading.Thread(target=_mark_done, daemon=True).start()
        try:
            if not done.wait(timeout=timeout):
                logger.warning("[playback] timed out, stopping and moving on to the next turn")
                sd.stop()
        finally:
            playback_active.clear()

    thread = threading.Thread(target=_wait, daemon=True)
    thread.start()
    return thread

# Default loguru sink already includes a timestamp; this just tightens the
# format to level + message (module/line noise isn't useful for a live
# conversation transcript -- this is for talk.py's own status line, not a
# stack-trace-grade log).
logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss.SSS}</green> | {message}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_pipeline_args(parser)
    parser.add_argument(
        "--endpoint-detect",
        action="store_true",
        help="use Smart Turn v3 endpoint detection (don't cut the user off on a "
        "mid-thought pause). Off by default -- the real-time accumulation loop is "
        "not yet mic-validated (see vad.py's listen_for_utterance docstring).",
    )
    args = parser.parse_args()

    logger.info("Loading models...")
    pipeline, presets = build_pipeline_from_args(args)
    vad = registry.build_vad()
    # Built only when asked (loading the onnx model + Whisper feature extractor
    # isn't free) -- passed into every listen_for_utterance call below.
    turn_detector = registry.build_turn_detector() if args.endpoint_detect else None
    if turn_detector is not None:
        logger.info("[turn] Smart Turn v3 endpoint detection enabled")
    store = ConversationStore()
    session_id = store.start_session()
    session_dir = SESSION_AUDIO_DIR / str(session_id)
    session_dir.mkdir(parents=True, exist_ok=True)

    # docs/memory-design.md's "다음 세션에 어떻게 반영할까": recall happens
    # once, here, before the first turn -- not re-fetched mid-session, since
    # nothing in *this* session can add to `memories` until it ends (see the
    # extraction call in `finally` below). format_recall_block() returns ""
    # when there's nothing to recall yet (first session ever), and
    # NobodyLLM/NobodyLLMGguf's reply() only appends system_prompt_suffix to
    # the persona prompt when it's non-empty -- so this is a no-op on a
    # fresh install, not a special case to branch on here.
    recall_block = format_recall_block(store.recent_memories())
    if recall_block:
        pipeline.llm.system_prompt_suffix = recall_block
        logger.info(f"[memory] recalled from previous sessions:\n{recall_block}")

    STAGE_LABELS = {
        "asr": "[ASR] transcribing...",
        "llm": "[LLM] generating reply...",
        "tts": "[TTS] synthesizing...",
    }

    # Set for exactly the duration of a reply/greeting clip's playback (see
    # play_async) -- on_barge_in_confirmed uses this to tell "barge-in during
    # a reply" apart from "the normal start of the next turn while nothing is
    # playing." No lock needed: both callbacks only ever read/write this from
    # the single thread running vad.listen_for_utterance's loop, and
    # play_async's own background thread is the only other writer.
    playback_active = threading.Event()
    playback_thread: threading.Thread | None = None

    def on_speech_start():
        # Printed the instant VAD's IDLE->RECORDING transition fires, not
        # after the whole utterance is captured -- otherwise "... listening
        # ..." is the only thing on screen for however long you're mid-turn,
        # which looks identical to the mic just not working at all. Does NOT
        # stop playback itself anymore (see on_barge_in_confirmed) -- any
        # detected speech used to cut the reply off immediately, which meant
        # backchannel ("어", "응") interrupted it too. docs/barge-in-design.md
        # stage 1.
        logger.info("[VAD] speech detected, recording...")

    def on_barge_in_confirmed():
        # Fires once speech has continued past vad.yaml's
        # barge_in_confirm_ms -- see vad.py's field docstring and
        # docs/barge-in-design.md. Backchannel is short enough that this
        # usually never fires for it at all; only stops playback if it's
        # actually still going (a normal turn-start while nothing is
        # playing also reaches here, is_set() is False, and this is a
        # no-op).
        if playback_active.is_set():
            logger.info("[VAD] barge-in confirmed, stopping playback")
            sd.stop()

    def on_stage_start(stage: str):
        logger.info(STAGE_LABELS[stage])

    def on_result(stage: str, text: str, elapsed_ms: int):
        # Fires right after ASR/LLM produce their text, not after the whole
        # turn (asr+llm+tts) finishes -- so the transcript shows up the
        # instant it's known instead of being held back until synthesis and
        # playback are also done.
        if stage == "asr":
            logger.info(f"[user]   {text}  ({elapsed_ms}ms)")
        elif stage == "llm":
            logger.info(f"[nobody] {text}  ({elapsed_ms}ms)")

    logger.info(f"Session {session_id} started. Speak after the beep-less silence; Ctrl+C to end.")

    logger.info("[TTS] synthesizing greeting...")
    greeting_wav = session_dir / "greeting.wav"
    pipeline.tts.synthesize(GREETING_TEXT, out_path=str(greeting_wav))
    logger.info(f"[nobody] {GREETING_TEXT}")
    greeting_audio, greeting_sr = sf.read(str(greeting_wav), dtype="float32")
    playback_thread = play_async(greeting_audio, greeting_sr, playback_active)

    turn_index = 0
    try:
        while True:
            logger.info("... listening ...")
            # Deliberately starts before playback_thread (the previous
            # turn's reply, or the greeting) has necessarily finished --
            # that overlap is what makes barge-in possible. on_barge_in_confirmed
            # is what actually cuts the still-playing clip off (once
            # confirmed real, not on the first frame); this call blocks
            # until an utterance completes either way, so by the time it
            # returns playback has always stopped one way or another.
            utterance = vad.listen_for_utterance(
                on_speech_start=on_speech_start,
                on_barge_in_confirmed=on_barge_in_confirmed,
                turn_detector=turn_detector,
            )
            if utterance.audio.size == 0:
                continue

            # Not strictly necessary (on_speech_start already stopped
            # playback if this was a barge-in, and play_async's own
            # background thread clears playback_active itself once done) --
            # but joining here means the *next* play_async call below can't
            # race with this one's watchdog thread still winding down.
            if playback_thread is not None:
                playback_thread.join()

            duration_s = len(utterance.audio) / utterance.sample_rate
            logger.info(f"[VAD] silence detected, recording ended ({duration_s:.1f}s captured)")

            turn_index += 1
            wav_in = session_dir / f"turn_{turn_index:03d}_in.wav"
            wav_out = session_dir / f"turn_{turn_index:03d}_out.wav"
            sf.write(str(wav_in), utterance.audio, utterance.sample_rate)

            result = pipeline.run(
                str(wav_in),
                str(wav_out),
                on_stage_start=on_stage_start,
                on_result=on_result,
                # docs/barge-in-design.md stage 2: skip LLM/TTS/storage
                # entirely for a short utterance whose ASR result is a bare
                # backchannel word -- duration_s is the same value already
                # logged above (VAD's own segment length, not re-measured).
                should_continue_after_asr=lambda text: not is_backchannel(text, duration_s),
            )
            if result["skipped"]:
                logger.info(f"[VAD] backchannel ignored: {result['user_text']!r}")
                continue

            logger.info(
                f"[timing] asr={result['asr_ms']}ms llm={result['llm_ms']}ms "
                f"tts={result['tts_ms']}ms"
            )

            logger.info("[playback] playing reply...")
            reply_audio, reply_sr = sf.read(str(wav_out), dtype="float32")
            playback_thread = play_async(reply_audio, reply_sr, playback_active)

            store.log_turn(
                session_id,
                turn_index,
                result["user_text"],
                result["reply_text"],
                user_wav_path=str(wav_in),
                reply_wav_path=str(wav_out),
                asr_preset=presets["asr"],
                llm_preset=presets["llm"],
                tts_preset=presets["tts"],
                asr_ms=result["asr_ms"],
                llm_ms=result["llm_ms"],
                tts_ms=result["tts_ms"],
            )
    except KeyboardInterrupt:
        pass
    finally:
        # Cut off whatever's still playing rather than letting it run out on
        # its own after Ctrl+C -- and join so play_async's watchdog thread
        # doesn't outlive the process it's a daemon thread of anyway, but
        # this makes the "playback stopped" state deterministic before the
        # rest of shutdown runs.
        sd.stop()
        if playback_thread is not None:
            playback_thread.join(timeout=1.0)

        # Session-end batch extraction (docs/memory-design.md's recommended
        # timing over per-turn) -- runs before pipeline.close() since it
        # still needs pipeline.llm loaded. Skipped for a 0-turn session
        # (Ctrl+C before saying anything): nothing was said, nothing to
        # extract, and it'd otherwise burn a generation call on an empty
        # transcript. Failure here (model output too malformed for
        # memory.py's defensive parser to salvage anything, or a generation
        # error) is logged and swallowed, not raised -- extraction failing
        # shouldn't take the rest of shutdown down with it.
        if turn_index > 0:
            try:
                session_turns = [
                    (user_text, reply_text)
                    for _asr, _llm, _tts, user_text, reply_text in store.turns_for_session(
                        session_id
                    )
                ]
                logger.info("[memory] extracting from this session...")
                memories = extract_memories(pipeline.llm, session_turns)
                for m in memories:
                    store.save_memory(session_id, m["category"], m["key"], m["value"], m["confidence"])
                logger.info(f"[memory] saved {len(memories)} fact(s)")
            except Exception:
                logger.exception("[memory] extraction failed, skipping")

        # Shut down any server-backed ASR/TTS subprocess (VibeAsrBitnet,
        # FreyaTtsKo) cleanly on exit -- these stay alive across turns for
        # speed (that's the whole point), so nothing else stops them.
        pipeline.close()
        store.end_session(session_id)
        store.close()
        logger.info(f"Session {session_id} ended ({turn_index} turns).")


if __name__ == "__main__":
    main()
