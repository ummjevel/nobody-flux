#!/usr/bin/env python3
"""Continuous voice conversation loop: mic in, speaker out, no restart between turns.

Unlike run_pipeline.py (one wav in, one wav out, process exits), this keeps a
single STSPipeline alive for the whole session -- so NobodyLLM.history carries
across turns and it's an actual multi-turn conversation, not N independent
one-shot calls.

Turn boundaries come from vad.py's simple energy-based VAD (no wakeword, no
push-to-talk): it starts recording when you start talking and stops when you
stop. See vad.py's docstring for the tuning knobs and its limits -- if it cuts
you off early or won't stop listening, that's threshold tuning, not a bug to
route around here.

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


def play_with_timeout(audio, sample_rate: int) -> None:
    """Like sd.play() + sd.wait(), but never blocks the session's turn loop
    forever.

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
    ends) while still bounding the wait if the backend truly does stall.
    """
    sd.play(audio, sample_rate)
    timeout = len(audio) / sample_rate + PLAYBACK_TIMEOUT_MARGIN_S
    done = threading.Event()

    def _wait():
        sd.wait()
        done.set()

    threading.Thread(target=_wait, daemon=True).start()
    if not done.wait(timeout=timeout):
        logger.warning("[playback] timed out, stopping and moving on to the next turn")
        sd.stop()

# Default loguru sink already includes a timestamp; this just tightens the
# format to level + message (module/line noise isn't useful for a live
# conversation transcript -- this is for talk.py's own status line, not a
# stack-trace-grade log).
logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss.SSS}</green> | {message}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_pipeline_args(parser)
    args = parser.parse_args()

    logger.info("Loading models...")
    pipeline, presets = build_pipeline_from_args(args)
    vad = registry.build_vad()
    store = ConversationStore()
    session_id = store.start_session()
    session_dir = SESSION_AUDIO_DIR / str(session_id)
    session_dir.mkdir(parents=True, exist_ok=True)

    STAGE_LABELS = {
        "asr": "[ASR] transcribing...",
        "llm": "[LLM] generating reply...",
        "tts": "[TTS] synthesizing...",
    }

    def on_speech_start():
        # Printed the instant VAD's IDLE->RECORDING transition fires, not
        # after the whole utterance is captured -- otherwise "... listening
        # ..." is the only thing on screen for however long you're mid-turn,
        # which looks identical to the mic just not working at all.
        logger.info("[VAD] speech detected, recording...")

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
    play_with_timeout(greeting_audio, greeting_sr)

    turn_index = 0
    try:
        while True:
            logger.info("... listening ...")
            utterance = vad.listen_for_utterance(on_speech_start=on_speech_start)
            if utterance.audio.size == 0:
                continue

            duration_s = len(utterance.audio) / utterance.sample_rate
            logger.info(f"[VAD] silence detected, recording ended ({duration_s:.1f}s captured)")

            turn_index += 1
            wav_in = session_dir / f"turn_{turn_index:03d}_in.wav"
            wav_out = session_dir / f"turn_{turn_index:03d}_out.wav"
            sf.write(str(wav_in), utterance.audio, utterance.sample_rate)

            result = pipeline.run(
                str(wav_in), str(wav_out), on_stage_start=on_stage_start, on_result=on_result
            )
            logger.info(
                f"[timing] asr={result['asr_ms']}ms llm={result['llm_ms']}ms "
                f"tts={result['tts_ms']}ms"
            )

            logger.info("[playback] playing reply...")
            reply_audio, reply_sr = sf.read(str(wav_out), dtype="float32")
            play_with_timeout(reply_audio, reply_sr)

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
        # Shut down any server-backed ASR/TTS subprocess (VibeAsrBitnet,
        # FreyaTtsKo) cleanly on exit -- these stay alive across turns for
        # speed (that's the whole point), so nothing else stops them.
        pipeline.close()
        store.end_session(session_id)
        store.close()
        logger.info(f"Session {session_id} ended ({turn_index} turns).")


if __name__ == "__main__":
    main()
