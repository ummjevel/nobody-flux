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
    uv run python scripts/talk.py [--asr PRESET] [--llm PRESET] [--tts PRESET]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import soundfile as sf
import sounddevice as sd

from src.nobody_flux import registry
from src.nobody_flux.pipeline import STSPipeline
from src.nobody_flux.storage import ConversationStore
from src.nobody_flux.vad import VoiceActivityDetector

SESSION_AUDIO_DIR = Path(__file__).resolve().parents[1] / "data" / "sessions"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asr", default=None, help="ASR preset name (configs/models.yaml)")
    parser.add_argument("--llm", default=None, help="LLM preset name (configs/models.yaml)")
    parser.add_argument("--tts", default=None, help="TTS preset name (configs/models.yaml)")
    args = parser.parse_args()

    print("Loading models...")
    pipeline = STSPipeline(
        asr=registry.build_asr(args.asr),
        llm=registry.build_llm(args.llm),
        tts=registry.build_tts(args.tts),
    )
    asr_preset = args.asr or registry.default_preset("asr")
    llm_preset = args.llm or registry.default_preset("llm")
    tts_preset = args.tts or registry.default_preset("tts")
    vad = VoiceActivityDetector()
    store = ConversationStore()
    session_id = store.start_session()
    session_dir = SESSION_AUDIO_DIR / str(session_id)
    session_dir.mkdir(parents=True, exist_ok=True)

    print(f"Session {session_id} started. Speak after the beep-less silence; Ctrl+C to end.")
    turn_index = 0
    try:
        while True:
            print("\n... listening ...")
            utterance = vad.listen_for_utterance()
            if utterance.audio.size == 0:
                continue

            turn_index += 1
            wav_in = session_dir / f"turn_{turn_index:03d}_in.wav"
            wav_out = session_dir / f"turn_{turn_index:03d}_out.wav"
            sf.write(str(wav_in), utterance.audio, utterance.sample_rate)

            result = pipeline.run(str(wav_in), str(wav_out))
            print(f"[user]   {result['user_text']}")
            print(f"[nobody] {result['reply_text']}")

            reply_audio, reply_sr = sf.read(str(wav_out), dtype="float32")
            sd.play(reply_audio, reply_sr)
            sd.wait()

            store.log_turn(
                session_id,
                turn_index,
                result["user_text"],
                result["reply_text"],
                user_wav_path=str(wav_in),
                reply_wav_path=str(wav_out),
                asr_preset=asr_preset,
                llm_preset=llm_preset,
                tts_preset=tts_preset,
                asr_ms=result["asr_ms"],
                llm_ms=result["llm_ms"],
                tts_ms=result["tts_ms"],
            )
    except KeyboardInterrupt:
        pass
    finally:
        store.end_session(session_id)
        print(f"\nSession {session_id} ended ({turn_index} turns).")


if __name__ == "__main__":
    main()
