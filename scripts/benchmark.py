#!/usr/bin/env python3
"""Run a fixed set of test wavs through every combination of ASR/LLM/TTS
presets and print a latency table (+ log every turn to conversations.db).

Why this exists: the FEATURES.md "다음 단계" list has wanted this since the
second ASR/TTS presets landed -- `turns` (storage.py) already carries
per-stage ms and which preset produced each row, so a preset comparison was
always just a GROUP BY away, but nothing ran the *same* fixed inputs through
*every* preset combo and reported it back as one table.

Quality isn't scored here -- ASR/TTS output quality needs a human ear/eye,
not a number. --verbose prints every turn's user_text/reply_text alongside
its preset combo so a person can read down the list and judge that part by
hand; this script only automates the part that's actually objective
(latency).

Usage (from the project root):
    uv run python scripts/benchmark.py --wav-dir data/benchmark_wavs
        [--asr PRESET [PRESET ...]] [--llm PRESET [PRESET ...]]
        [--tts PRESET [PRESET ...]] [--voice VOICE] [--repeat N] [--verbose]

Omitting --asr/--llm/--tts benchmarks every registered preset for that stage
(registry.list_presets) -- with N_asr * N_llm * N_tts combinations run for
every wav in --wav-dir, that product grows fast, so pass explicit preset
names to narrow it down once there are more than a couple per stage.

--wav-dir needs to be populated by hand (same reason data/*.wav isn't
committed -- see .gitignore and README's voice-clone note): drop a handful
of representative test utterances there. Nothing here generates them.
"""

import argparse
import itertools
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.nobody_flux import registry
from src.nobody_flux.pipeline import STSPipeline
from src.nobody_flux.storage import ConversationStore


def _collect_wavs(wav_dir: Path) -> list[Path]:
    wavs = sorted(wav_dir.glob("*.wav"))
    if not wavs:
        raise FileNotFoundError(
            f"No .wav files found in {wav_dir} -- this is the fixed test set, "
            "drop a handful of representative test utterances there first "
            "(not committed, same as other audio under data/, see .gitignore)."
        )
    return wavs


def _run_combo(
    store: ConversationStore,
    session_id: int,
    asr_preset: str,
    llm_preset: str,
    tts_preset: str,
    voice: str | None,
    wavs: list[Path],
    repeat: int,
    turn_index: int,
    tmp_dir: Path,
) -> int:
    """Runs every wav (x `repeat`) through one preset combo, logging each as
    a turn. Returns the next unused turn_index (turn_index is threaded
    through combos rather than reset per-combo so every row in this
    benchmark run's session has a unique index, matching log_turn's normal
    one-per-actual-turn meaning elsewhere).
    """
    tts_overrides = {}
    if voice:
        tts_overrides["reference_audio"] = registry.resolve_voice(voice)

    pipeline = STSPipeline(
        asr=registry.build_asr(asr_preset),
        llm=registry.build_llm(llm_preset),
        tts=registry.build_tts(tts_preset, **tts_overrides),
    )
    try:
        for wav_in in wavs:
            for _ in range(repeat):
                turn_index += 1
                wav_out = tmp_dir / f"benchmark_{turn_index:04d}_out.wav"
                result = pipeline.run(str(wav_in), str(wav_out))
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
                print(
                    f"  {wav_in.name}: asr={result['asr_ms']}ms llm={result['llm_ms']}ms "
                    f"tts={result['tts_ms']}ms"
                )
    finally:
        # Same reason run_pipeline.py/talk.py call this -- server-backed
        # presets (VibeAsrBitnet, FreyaTtsKo) leave a subprocess running
        # otherwise, and a benchmark run instantiates far more of these in
        # one process lifetime than either of those scripts ever does.
        pipeline.close()
    return turn_index


def _print_report(store: ConversationStore, session_id: int, verbose: bool) -> None:
    header = f"{'asr':<20} {'llm':<20} {'tts':<20} {'n':>3} {'asr_ms':>8} {'llm_ms':>8} {'tts_ms':>8} {'total_ms':>9}"
    print("\n" + header)
    print("-" * len(header))
    for asr_p, llm_p, tts_p, n, asr_ms, llm_ms, tts_ms, total_ms in store.turns_by_preset(session_id):
        print(
            f"{asr_p:<20} {llm_p:<20} {tts_p:<20} {n:>3} "
            f"{asr_ms:>8.0f} {llm_ms:>8.0f} {tts_ms:>8.0f} {total_ms:>9.0f}"
        )

    if verbose:
        print("\ntranscripts:")
        for asr_p, llm_p, tts_p, user_text, reply_text in store.turns_for_session(session_id):
            print(f"  [{asr_p}/{llm_p}/{tts_p}]")
            print(f"    user:   {user_text}")
            print(f"    nobody: {reply_text}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wav-dir", default="data/benchmark_wavs", help="fixed test-set directory")
    parser.add_argument("--asr", nargs="+", default=None, help="ASR presets (default: all)")
    parser.add_argument("--llm", nargs="+", default=None, help="LLM presets (default: all)")
    parser.add_argument("--tts", nargs="+", default=None, help="TTS presets (default: all)")
    parser.add_argument("--voice", default=None, help="TTS reference voice (configs/voices.yaml)")
    parser.add_argument("--repeat", type=int, default=1, help="repeats per wav per combo (for averaging)")
    parser.add_argument("--verbose", action="store_true", help="also print every turn's transcript")
    args = parser.parse_args()

    wavs = _collect_wavs(Path(args.wav_dir))
    asr_presets = args.asr or registry.list_presets("asr")
    llm_presets = args.llm or registry.list_presets("llm")
    tts_presets = args.tts or registry.list_presets("tts")

    combos = list(itertools.product(asr_presets, llm_presets, tts_presets))
    print(
        f"Benchmarking {len(combos)} preset combo(s) x {len(wavs)} wav(s) x "
        f"{args.repeat} repeat(s) = {len(combos) * len(wavs) * args.repeat} turns"
    )

    store = ConversationStore()
    session_id = store.start_session()
    turn_index = 0
    try:
        # Synthesized reply wavs are throwaway here (this script's output is
        # the latency table + transcripts, not a set of clips to listen
        # back to) -- reply_wav_path still gets logged for consistency with
        # normal turns, it just won't resolve to a file anymore once this
        # `with` block exits.
        with tempfile.TemporaryDirectory(prefix="nobody_flux_benchmark_") as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            for asr_preset, llm_preset, tts_preset in combos:
                print(f"\n== {asr_preset} / {llm_preset} / {tts_preset} ==")
                turn_index = _run_combo(
                    store,
                    session_id,
                    asr_preset,
                    llm_preset,
                    tts_preset,
                    args.voice,
                    wavs,
                    args.repeat,
                    turn_index,
                    tmp_dir,
                )
        _print_report(store, session_id, args.verbose)
    finally:
        store.end_session(session_id)
        store.close()


if __name__ == "__main__":
    main()
