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
import queue
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import soundfile as sf
import sounddevice as sd
from loguru import logger

from _cli import add_pipeline_args, build_pipeline_from_args
from src.nobody_flux import registry
from src.nobody_flux.backchannel import is_backchannel
from src.nobody_flux.memory import consolidate_memories, extract_memories, format_recall_block
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


class ChunkPlayer:
    """Sequential background playback of a streamed reply's chunks
    (pipeline.AudioChunk). Chunks are enqueue()'d as they're synthesized and a
    worker thread plays them back to back, so playback of chunk 1 overlaps
    synthesis of chunk 2+ (Phase 1's point -- see pipeline.run_streaming).
    Replaces the old single-clip play_async: a reply is now N chunks, not one
    wav.

    `active` is set while a reply is playing or still has pending chunks --
    on_barge_in_confirmed reads it exactly like the old playback_active Event
    did, to tell "the user is interrupting a reply" apart from "the user is
    just starting the next turn while nothing is playing."

    stop() clips the whole reply (current clip + everything still queued) for a
    confirmed barge-in. Playback is sd.play/sd.wait (one output stream), same
    as before; the unified duplex AudioSession (Phase 1.5, audio.py) will later
    take over capture+playback+AEC together and supersede this.

    Per-chunk watchdog: sd.wait() has no timeout, so a stalled backend would
    hang the worker forever. Each chunk is bounded by its own duration plus a
    margin (same reasoning as the old play_async), after which it force-stops
    and moves on."""

    def __init__(self):
        self._q: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self.active = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop.clear()
        # Set immediately: a reply is incoming even before the first chunk is
        # enqueued, so a barge-in in that window is still recognized as one.
        self.active.set()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def enqueue(self, samples: np.ndarray, sample_rate: int) -> None:
        self._q.put((samples, sample_rate))

    def done(self) -> None:
        """Signal no more chunks will be enqueued for this reply (sentinel)."""
        self._q.put(None)

    def _run(self) -> None:
        try:
            while True:
                item = self._q.get()
                if item is None:
                    break
                if self._stop.is_set():
                    continue  # post-barge-in: drain remaining chunks unplayed
                self._play_one(*item)
        finally:
            self.active.clear()

    def _play_one(self, samples: np.ndarray, sample_rate: int) -> None:
        sd.play(samples, sample_rate)
        timeout = len(samples) / sample_rate + PLAYBACK_TIMEOUT_MARGIN_S
        finished = threading.Event()

        def _mark_done():
            sd.wait()
            finished.set()

        threading.Thread(target=_mark_done, daemon=True).start()
        if not finished.wait(timeout=timeout):
            logger.warning("[playback] timed out, stopping and moving on")
            sd.stop()

    def stop(self) -> None:
        """Confirmed barge-in: cut current playback and discard queued chunks."""
        self._stop.set()
        sd.stop()

    def stop_requested(self) -> bool:
        """True once stop() has been called (barge-in) -- the producer polls
        this to abandon synthesizing the rest of an interrupted reply."""
        return self._stop.is_set()

    def is_active(self) -> bool:
        return self.active.is_set()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)


class SessionPlayer:
    """ChunkPlayer-compatible facade over an audio.AudioSession's playback, so
    the turn loop is byte-for-byte identical whether or not the duplex/AEC
    session is in use. Playback (and thus the AEC reference) goes through the
    session's single stream instead of a private sd.play, which is what lets
    capture and playback coexist without the macOS err -50 conflict."""

    def __init__(self, session):
        self._session = session
        self._stop = threading.Event()

    def start(self) -> None:
        self._stop.clear()

    def enqueue(self, samples: np.ndarray, sample_rate: int) -> None:
        self._session.play(samples, sample_rate)

    def done(self) -> None:
        # Nothing to signal: the session drains its own playback buffer.
        pass

    def stop(self) -> None:
        self._stop.set()
        self._session.stop_playback()

    def stop_requested(self) -> bool:
        return self._stop.is_set()

    def is_active(self) -> bool:
        return self._session.playback_active()

    def join(self, timeout: float | None = None) -> None:
        # Wait for the session's playback buffer to drain (or a barge-in stop),
        # bounded by timeout -- the session plays on its own callback thread, so
        # there's no worker thread to join, just a state to poll.
        deadline = None if timeout is None else time.monotonic() + timeout
        while self._session.playback_active() and not self._stop.is_set():
            if deadline is not None and time.monotonic() > deadline:
                break
            time.sleep(0.02)

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
    parser.add_argument(
        "--no-barge-in",
        action="store_true",
        help="sequential mode: play each reply to completion BEFORE listening, "
        "instead of listening during playback. Disables barge-in, but avoids "
        "opening a playback stream and a mic stream at the same time -- which on "
        "macOS/CoreAudio conflicts (PaMacCore err -50) and corrupts mic capture. "
        "Use this to get a clean turn on a Mac until real AEC/duplex handling lands.",
    )
    parser.add_argument(
        "--aec",
        default="none",
        choices=["none", "auto", "off", "refgate", "speex", "os", "vpio"],
        help="route mic+speaker through the unified duplex AudioSession "
        "(src/nobody_flux/audio.py) with echo cancellation, instead of the "
        "legacy separate mic/speaker streams. 'none' (default) = legacy path. "
        "A single duplex stream also fixes the macOS err -50 conflict, so with "
        "--aec you generally don't need --no-barge-in. 'auto' picks per platform "
        "+ installed libs; refgate=dependency-free suppression gate, speex=real "
        "AEC (needs speexdsp), os=Linux module-echo-cancel, vpio=macOS (not wired). "
        "See configs/audio.yaml.",
    )
    args = parser.parse_args()

    # --aec short names -> audio.build_session backend names (see audio.py).
    AEC_BACKENDS = {
        "auto": "auto",
        "off": "off",
        "refgate": "shared-refgate",
        "speex": "shared-speex",
        "os": "os-echocancel",
        "vpio": "vpio",
    }

    logger.info("Loading models...")
    pipeline, presets = build_pipeline_from_args(args)
    vad = registry.build_vad()
    # Built only when asked (loading the onnx model + Whisper feature extractor
    # isn't free) -- passed into every listen_for_utterance call below.
    turn_detector = registry.build_turn_detector() if args.endpoint_detect else None
    if turn_detector is not None:
        logger.info("[turn] Smart Turn v3 endpoint detection enabled")

    # Unified duplex/AEC audio session (audio.py), opt-in via --aec. When on, it
    # owns one stream doing both capture and playback: vad reads frames from it
    # (frame_source below) and replies play through it (SessionPlayer), so the
    # reply's echo is cancelled out of the mic and the macOS err -50 duplex
    # conflict never arises. When off (default), capture/playback stay on the
    # legacy separate streams (vad's own InputStream + ChunkPlayer's sd.play).
    audio_session = None
    if args.aec != "none":
        audio_session = registry.build_audio_session(AEC_BACKENDS[args.aec])
        audio_session.start()
        logger.info(f"[audio] duplex AEC session active (--aec {args.aec})")
    frame_source = audio_session.read_frame if audio_session is not None else None

    def make_player():
        """A ChunkPlayer (legacy) or SessionPlayer (AEC session) exposing the
        same interface, so the turn loop below doesn't branch on which is used."""
        if audio_session is not None:
            player = SessionPlayer(audio_session)
        else:
            player = ChunkPlayer()
        player.start()
        return player

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

    # The reply/greeting currently playing (or None). Reassigned each turn; the
    # barge-in callback reads whichever ChunkPlayer is active through this
    # single-element holder so it always targets the current reply. A plain
    # local wouldn't work -- the nested callbacks would close over its initial
    # value, not the per-turn reassignment.
    state: dict[str, ChunkPlayer | None] = {"player": None}

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
        # usually never fires for it at all; only clips the reply if one is
        # actually still playing (a normal turn-start while nothing is playing
        # also reaches here, active is clear, and this is a no-op). With
        # streaming playback this clips ALL remaining queued chunks, not just
        # the one clip that happens to be sounding (ChunkPlayer.stop()).
        player = state["player"]
        if player is not None and player.is_active():
            logger.info("[VAD] barge-in confirmed, stopping playback")
            player.stop()

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

    def produce(utterance, turn_index: int, player: ChunkPlayer) -> None:
        """Drive run_streaming for one turn: enqueue each synthesized chunk to
        `player` as it's produced (so playback overlaps synthesis), then log the
        turn. Runs on the main thread on purpose -- playback happens on the
        player's own worker thread, and the *next* listen() overlaps this
        reply's tail for barge-in. Keeping it synchronous means ConversationStore's
        SQLite connection is only ever used from the main thread.

        Known gap (Phase 1.5): since this blocks the main thread until the LLM
        finishes, a barge-in landing *during* generation isn't seen until
        generation completes -- only the playback tail past that point is
        interruptible. The unified duplex AudioSession (audio.py) will close this."""
        wav_in = session_dir / f"turn_{turn_index:03d}_in.wav"
        wav_out = session_dir / f"turn_{turn_index:03d}_out.wav"
        sf.write(str(wav_in), utterance.audio, utterance.sample_rate)
        duration_s = len(utterance.audio) / utterance.sample_rate

        collected: list[np.ndarray] = []
        combined_sr: int | None = None
        summary = None
        gen = pipeline.run_streaming(
            str(wav_in),
            on_stage_start=on_stage_start,
            on_result=on_result,
            # docs/barge-in-design.md stage 2: skip LLM/TTS/storage for a short
            # utterance whose ASR result is a bare backchannel word.
            should_continue_after_asr=lambda text: not is_backchannel(text, duration_s),
        )
        try:
            while True:
                try:
                    chunk = next(gen)
                except StopIteration as done:
                    summary = done.value
                    break
                if player.stop_requested():
                    gen.close()
                    break
                player.enqueue(chunk.samples, chunk.sample_rate)
                collected.append(chunk.samples)
                combined_sr = chunk.sample_rate
        finally:
            player.done()

        if summary is None:
            logger.info("[playback] reply interrupted")
            return
        if summary.get("skipped"):
            logger.info(f"[VAD] backchannel ignored: {summary['user_text']!r}")
            return

        logger.info(
            f"[timing] asr={summary['asr_ms']}ms llm={summary['llm_ms']}ms "
            f"tts={summary['tts_ms']}ms ttfa={summary['ttfa_ms']}ms"
        )
        # Persist the full reply audio (chunks concatenated) so the stored
        # reply_wav_path matches what was actually played.
        if collected and combined_sr is not None:
            sf.write(str(wav_out), np.concatenate(collected), combined_sr)
        store.log_turn(
            session_id,
            turn_index,
            summary["user_text"],
            summary["reply_text"],
            user_wav_path=str(wav_in),
            reply_wav_path=str(wav_out) if collected else None,
            asr_preset=presets["asr"],
            llm_preset=presets["llm"],
            tts_preset=presets["tts"],
            asr_ms=summary["asr_ms"],
            llm_ms=summary["llm_ms"],
            tts_ms=summary["tts_ms"],
        )

    logger.info("[TTS] synthesizing greeting...")
    logger.info(f"[nobody] {GREETING_TEXT}")
    greeting_audio, greeting_sr = pipeline.tts.synthesize_audio(GREETING_TEXT)
    greeting_player = make_player()
    greeting_player.enqueue(greeting_audio, greeting_sr)
    greeting_player.done()
    state["player"] = greeting_player

    turn_index = 0
    try:
        while True:
            # Sequential mode (--no-barge-in): finish playing the previous
            # reply/greeting BEFORE opening the mic, so a playback stream and a
            # mic stream are never open at once (avoids the macOS/CoreAudio
            # duplex conflict, err -50). Costs barge-in.
            if args.no_barge_in and state["player"] is not None:
                state["player"].join()

            logger.info("... listening ...")
            # Default (barge-in) mode: this starts while the previous reply is
            # still playing -- that overlap is what makes barge-in possible.
            # on_barge_in_confirmed clips the still-playing reply once confirmed.
            utterance = vad.listen_for_utterance(
                on_speech_start=on_speech_start,
                on_barge_in_confirmed=None if args.no_barge_in else on_barge_in_confirmed,
                turn_detector=turn_detector,
                frame_source=frame_source,
            )

            # Previous reply's player: barge-in (if any) already stopped it;
            # join so its worker is done before this turn starts a new one.
            if state["player"] is not None:
                state["player"].join()
                state["player"] = None

            if utterance.audio.size == 0:
                continue

            duration_s = len(utterance.audio) / utterance.sample_rate
            logger.info(f"[VAD] silence detected, recording ended ({duration_s:.1f}s captured)")

            turn_index += 1
            player = make_player()
            state["player"] = player
            produce(utterance, turn_index, player)
    except KeyboardInterrupt:
        pass
    finally:
        # Cut off whatever's still playing after Ctrl+C and join the player's
        # worker so it doesn't outlive the process (daemon anyway, but this
        # makes shutdown deterministic before memory extraction runs).
        if state["player"] is not None:
            state["player"].stop()
            state["player"].join(timeout=1.0)
        sd.stop()
        if audio_session is not None:
            audio_session.close()

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
                candidates = extract_memories(pipeline.llm, session_turns)
                # Mem0-style consolidation: instead of blindly saving every
                # candidate, diff against what's already stored and ADD/UPDATE/
                # NOOP each one (memory.py). Falls back to all-ADD if the model
                # output is unusable, so this can only tidy the table, never
                # lose a fact.
                existing = store.memories_for_consolidation()
                ops = consolidate_memories(pipeline.llm, existing, candidates)
                added = updated = skipped = 0
                for op in ops:
                    if op["op"] == "ADD":
                        m = op["memory"]
                        store.save_memory(session_id, m["category"], m["key"], m["value"], m["confidence"])
                        added += 1
                    elif op["op"] == "UPDATE":
                        m = op["memory"]
                        store.update_memory(op["target_id"], m["value"], m["confidence"])
                        updated += 1
                    else:
                        skipped += 1
                logger.info(f"[memory] added {added}, updated {updated}, skipped {skipped}")
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
