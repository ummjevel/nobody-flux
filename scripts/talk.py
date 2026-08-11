#!/usr/bin/env python3
"""Continuous voice conversation: microphone in, speaker out, no restart between turns.

Unlike run_pipeline.py (one wav in, one wav out, process exits), this keeps a
single STSPipeline alive for the whole session -- so the LLM's history carries
across turns and it is an actual multi-turn conversation rather than N
independent one-shot calls.

Everything about *when* to listen and *when* to stop talking lives in
src/nobody_flux/turn/controller.py, not here. This script is the composition
root: it parses flags, builds the pieces, and says what a turn means for
persistence and logging. That split is deliberate -- turn-taking is the part
worth testing without a microphone, and it cannot be if it lives in a CLI.

Turn-taking (see docs/barge-in-design.md, docs/FEATURES.md)
----------------------------------------------------------

Capture runs continuously on its own thread for the entire session, including
while a reply is being generated. That is Phase 4: an interruption during
generation is now *observed*, because the microphone is still being read then;
previously the whole generation window was a blind spot and speech in it was
simply lost.

Backchannel ("어", "응" -- frequent given this persona's casual tone) is told
apart from a real interruption in two stages. Stage 1 is duration: playback is
only cut once speech has continued past `barge_in_confirm_ms`, so most
backchannel never trips it. Stage 2 is lexical: a short utterance whose
transcript is a bare backchannel word skips the turn entirely -- no LLM, no
TTS, no stored turn.

Streaming ASR (--streaming-asr, Phase 3)
----------------------------------------

Off by default. When on, microphone frames are fed to a streaming Zipformer as
they arrive, so recognition finishes when speech does and the ASR stage drops
out of the turn's critical path entirely. The batch presets remain the default
because the live path is not yet microphone-validated.

Platform notes
--------------

Native Windows and native Linux both give real microphone access. WSL2 does
not reliably: capture goes through WSLg's audio bridge, and if sounddevice
cannot see an input device or capture hangs, that is the cause -- use the
Windows environment (.venv-win, scripts/setup_windows.ps1), the H100 server, or
fall back to run_pipeline.py with pre-recorded wavs.

Usage:
    uv run python scripts/talk.py [--asr PRESET] [--llm PRESET] [--tts PRESET]
                                  [--voice VOICE] [--aec MODE]
                                  [--endpoint-detect] [--streaming-asr]
                                  [--no-barge-in]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import soundfile as sf
from loguru import logger

from _cli import add_pipeline_args, build_pipeline_from_args
from src.nobody_flux import registry
from src.nobody_flux.audio.player import SessionPlayer, StreamPlayer
from src.nobody_flux.memory import consolidate_memories, extract_memories, format_recall_block
from src.nobody_flux.paths import PROJECT_ROOT
from src.nobody_flux.storage import ConversationStore
from src.nobody_flux.turn.backchannel import is_backchannel
from src.nobody_flux.turn.controller import CapturedTurn, TurnController, TurnState
from src.nobody_flux.turn.vad import VadEvent

SESSION_AUDIO_DIR = PROJECT_ROOT / "data" / "sessions"

# Longest the main thread ever blocks in one call. Every wait in this script is
# chunked to this so Ctrl+C is acted on promptly -- see ConversationSession.run
# for why an unbounded wait swallows it entirely on Windows. Short enough to
# feel immediate, long enough that idling costs nothing measurable.
WAIT_SLICE_S = 0.2

# Ceiling on waiting for a reply to finish sounding. Not a real limit -- this
# persona's replies run seconds, a confirmed barge-in ends playback immediately,
# and both players already abandon a wedged device on their own. It is here so
# that a bug somewhere below cannot strand the loop with the microphone open.
PLAYBACK_WAIT_CAP_S = 120.0

# Spoken once at session start, after the models finish loading. Audible
# confirmation that the session is ready and listening -- without it the first
# sign of life is silence until you happen to say something the VAD catches,
# which is indistinguishable from a broken microphone. Plain Korean, no
# numbers/English/emoji, same TTS-friendliness rules as persona.py.
GREETING_TEXT = "안녕, 나 지금 듣고 있어."

# --aec short names -> audio.session backend names.
AEC_BACKENDS = {
    "auto": "auto",
    "off": "off",
    "refgate": "shared-refgate",
    "speex": "shared-speex",
    "os": "os-echocancel",
    "vpio": "vpio",
}

STAGE_LABELS = {
    "asr": "[ASR] transcribing...",
    "llm": "[LLM] generating reply...",
    "tts": "[TTS] synthesizing...",
}

# The default loguru sink already carries a timestamp; this trims the rest to
# level + message. Module/line noise is not useful in what is effectively a
# live conversation transcript.
logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss.SSS}</green> | {message}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_pipeline_args(parser)
    parser.add_argument(
        "--endpoint-detect",
        action="store_true",
        help="use Smart Turn v3 endpoint detection so a mid-thought pause does not "
        "end the turn. Off by default -- the continuation loop is not yet "
        "microphone-validated.",
    )
    parser.add_argument(
        "--streaming-asr",
        action="store_true",
        help="Phase 3: recognize incrementally while the user speaks (streaming "
        "Zipformer + LocalAgreement) instead of transcribing the recorded "
        "utterance afterwards. Takes ASR off the turn's critical path. Off by "
        "default -- not yet microphone-validated; see configs/streaming_asr.yaml.",
    )
    parser.add_argument(
        "--no-barge-in",
        action="store_true",
        help="sequential mode: play each reply to completion before acting on any "
        "interruption. Capture still runs (so nothing is lost), but a confirmed "
        "barge-in no longer cancels the reply.",
    )
    parser.add_argument(
        "--aec",
        default="none",
        choices=["none", "auto", "off", "refgate", "speex", "os", "vpio"],
        help="route microphone and speaker through one duplex AudioSession with echo "
        "cancellation, instead of separate streams. 'none' (default) keeps them "
        "separate. A single duplex stream also resolves the macOS err -50 "
        "conflict. 'auto' picks per platform and installed libraries. See "
        "configs/audio.yaml.",
    )
    return parser.parse_args()


def build_turn_controller(args, audio_session) -> tuple[TurnController, Callable[[], None]]:
    """Assemble the controller from flags and config.

    Returns it together with a teardown callable for whatever devices this
    function opened -- nothing if the duplex session supplied the frames, the
    private input stream's close if it did not. The caller owns shutdown order,
    so it has to be handed the thing to shut down.

    Each optional component is built only when its flag asks for it: the turn
    detector loads an ONNX model plus a Whisper feature extractor, and the
    streaming transcriber loads a three-part transducer. Neither is cheap
    enough to construct speculatively.
    """
    turn_detector = registry.build_turn_detector() if args.endpoint_detect else None
    if turn_detector is not None:
        logger.info("[turn] Smart Turn v3 endpoint detection enabled")

    transcriber = registry.build_streaming_transcriber() if args.streaming_asr else None
    if transcriber is not None:
        logger.info("[turn] streaming ASR enabled (recognition overlaps speech)")

    # With a duplex session the frames are already echo-cancelled and come from
    # the same stream that plays the reply; without one, capture gets its own
    # input stream. Only the latter is ours to close -- the session is closed by
    # whoever started it.
    if audio_session is not None:
        frame_source, close_capture = audio_session.read_frame, lambda: None
    else:
        frame_source, close_capture = _mic_frame_source()

    def player_factory():
        # SessionPlayer routes playback through the duplex stream, which is what
        # supplies the echo canceller's reference. StreamPlayer owns a private
        # output stream for the non-AEC path. Both satisfy audio.player.ReplyPlayer,
        # so nothing downstream branches on which is in use.
        return SessionPlayer(audio_session) if audio_session is not None else StreamPlayer()

    def on_event(event: VadEvent, state: TurnState) -> None:
        # Runs on the capture thread -- keep it to logging. Anything heavier
        # delays the next frame read and, through that, every duration this
        # controller measures.
        if event is VadEvent.SPEECH_STARTED:
            logger.info("[VAD] speech detected, recording...")
        elif event is VadEvent.BARGE_IN_CONFIRMED and state is TurnState.RESPONDING:
            logger.info("[VAD] barge-in confirmed, stopping reply")

    controller = TurnController(
        vad=registry.build_vad(),
        frame_source=frame_source,
        player_factory=player_factory,
        turn_detector=turn_detector,
        transcriber=transcriber,
        allow_barge_in=not args.no_barge_in,
        on_event=on_event,
    )
    return controller, close_capture


def wait_for_playback(player, limit_s: float) -> None:
    """Block until the player finishes, but in slices, so Ctrl+C still works.

    ``player.join()`` with no timeout has the same problem as an unbounded
    ``queue.get()`` on Windows -- see ConversationSession.run. The bound is
    belt-and-braces: the players already time out a wedged device internally,
    so reaching it means something is wrong, and continuing is better than
    hanging the session on the greeting.
    """
    deadline = time.monotonic() + limit_s
    while player.is_active() and time.monotonic() < deadline:
        player.join(timeout=WAIT_SLICE_S)


def _mic_frame_source():
    """A frame source backed by a private input stream, for the non-AEC path.

    The duplex AudioSession is the better arrangement and supplies frames
    itself; this exists for `--aec none`, where capture and playback stay on
    separate streams.

    Returns ``(read, close)``. An earlier version returned only the reader and
    never closed the stream, on the reasoning that the process was about to exit
    anyway. That reasoning was wrong in a way that mattered: at shutdown the
    capture thread is parked inside PortAudio's blocking read, and tearing
    PortAudio down with a thread inside a live stream is exactly the shape of
    teardown that hangs. Closing it explicitly, before the interpreter starts
    unwinding, keeps exit under our control.

    Wrapped in skip_warmup for the same reason the duplex session is: opening
    an input stream emits a transient that the VAD reads as speech, and without
    dropping it the session's first act is answering a device click.
    """
    import sounddevice as sd

    from src.nobody_flux.audio.session import skip_warmup
    from src.nobody_flux.turn.vad import FRAME_SAMPLES, SAMPLE_RATE

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=FRAME_SAMPLES
    )
    stream.start()

    def read() -> np.ndarray:
        block, _overflowed = stream.read(FRAME_SAMPLES)
        # read() may reuse its internal buffer between calls, so copy before
        # this frame is retained past the current iteration.
        return block[:, 0].copy()

    def close() -> None:
        # abort(), not stop(): stop() drains, and there is nothing worth
        # draining from a capture stream we are done with. Errors are ignored --
        # this runs on the way out, and a device that is already gone is not a
        # problem to report.
        try:
            stream.abort()
            stream.close()
        except Exception:
            pass

    return skip_warmup(read), close


class ConversationSession:
    """One run of the conversation loop, plus everything it must persist.

    A class rather than a pile of closures because the turn handler needs the
    store, the session id, the resolved preset names and the audio directory,
    and threading five of those through nested functions is exactly how the
    previous version of this script became hard to follow.
    """

    def __init__(self, pipeline, presets: dict[str, str], controller: TurnController):
        self.pipeline = pipeline
        self.presets = presets
        self.controller = controller
        self.store = ConversationStore()
        self.session_id = self.store.start_session()
        self.audio_dir = SESSION_AUDIO_DIR / str(self.session_id)
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.turns_handled = 0

    # -- setup -------------------------------------------------------------

    def recall_previous_sessions(self) -> None:
        """Load memories from earlier sessions into the system prompt.

        Once, here, before the first turn -- nothing in *this* session can add
        to the memory table until it ends (extraction runs at shutdown), so
        re-fetching mid-session could not find anything new. On a fresh install
        format_recall_block returns "" and the LLM only appends a non-empty
        suffix, so this is a no-op rather than a special case.
        """
        block = format_recall_block(self.store.recent_memories())
        if not block:
            return
        self.pipeline.llm.system_prompt_suffix = block
        logger.info(f"[memory] recalled from previous sessions:\n{block}")

    def greet(self) -> None:
        """Speak the greeting so the user knows the session is live.

        Driven through the same begin/finish response cycle as a real reply, so
        the greeting is interruptible exactly like one and the controller's
        state stays honest. Blocking the main thread on join() is harmless
        here: capture runs on its own thread throughout, so anything said over
        the greeting is still captured and simply arrives as the first turn.
        """
        logger.info("[TTS] synthesizing greeting...")
        logger.info(f"[nobody] {GREETING_TEXT}")
        samples, sample_rate = self.pipeline.tts.synthesize_audio(GREETING_TEXT)
        player = self.controller.begin_response()
        try:
            player.enqueue(samples, sample_rate)
            player.done()
            wait_for_playback(player, limit_s=len(samples) / sample_rate + 5.0)
        finally:
            self.controller.finish_response()

    # -- the loop ----------------------------------------------------------

    def run(self) -> None:
        while True:
            # Waited for in short slices rather than indefinitely, so that
            # Ctrl+C works.
            #
            # A bare next_turn() blocks in queue.get() with no timeout, and on
            # Windows that is not interruptible: CPython can only run a signal
            # handler in the main thread between bytecodes, and an unbounded
            # lock wait never gets there -- so Ctrl+C sets its flag and nothing
            # happens until the user next speaks. (POSIX hides this, since the
            # underlying wait returns EINTR.) Returning every WAIT_SLICE_S
            # gives the interpreter a point at which to raise KeyboardInterrupt.
            turn = self.controller.next_turn(timeout=WAIT_SLICE_S)
            if turn is None or turn.utterance.audio.size == 0:
                continue
            logger.info(
                f"[VAD] turn captured ({turn.duration_s:.1f}s)"
                + (f' streamed="{turn.streamed_text}"' if turn.streamed_text else "")
            )
            player = self.controller.begin_response()
            try:
                self.handle_turn(turn, player)
                # Stay in RESPONDING until the audio has actually finished, not
                # merely until the last chunk was handed to the player.
                #
                # handle_turn returns as soon as everything is enqueued, which
                # for a multi-sentence reply is seconds before it stops
                # sounding. Leaving RESPONDING at that point means speech during
                # the tail is classified as the user taking their turn rather
                # than interrupting -- so the reply is not cut, and the new turn
                # is captured on top of it. Both talk at once, which is the
                # exact failure barge-in exists to prevent.
                wait_for_playback(player, limit_s=PLAYBACK_WAIT_CAP_S)
            finally:
                self.controller.finish_response()

    def handle_turn(self, turn: CapturedTurn, player) -> None:
        """Generate and speak one reply, then record what happened.

        Runs on the main thread throughout. That is not incidental: the
        conversation store is SQLite, whose connection has thread affinity, and
        keeping every write here means no lock is needed anywhere. The capture
        thread meanwhile keeps reading the microphone, which is what allows the
        barge-in poll below to fire during generation.
        """
        wav_in = self.audio_dir / f"turn_{turn.index:03d}_in.wav"
        wav_out = self.audio_dir / f"turn_{turn.index:03d}_out.wav"
        sf.write(str(wav_in), turn.utterance.audio, turn.utterance.sample_rate)

        chunks: list[np.ndarray] = []
        chunk_rate: int | None = None
        summary = None

        generator = self.pipeline.run_streaming(
            wav_in=str(wav_in),
            # Phase 3: when streaming ASR ran, the text already exists and the
            # ASR stage is skipped. The wav is still written, for the stored
            # record and for offline re-runs.
            pretranscribed=turn.streamed_text,
            on_stage_start=lambda stage: logger.info(STAGE_LABELS[stage]),
            on_result=self._log_stage_result,
            # Stage 2 backchannel filter: a short utterance that transcribes to
            # a bare "어"/"응" is not worth an LLM call and is not a turn.
            should_continue_after_asr=lambda text: not is_backchannel(text, turn.duration_s),
            # Phase 4: polled between LLM deltas, so an interruption lands
            # during generation rather than after it.
            should_cancel=lambda: self.controller.cancelled,
        )

        try:
            while True:
                try:
                    chunk = next(generator)
                except StopIteration as done:
                    summary = done.value
                    break
                if player.stop_requested():
                    generator.close()
                    break
                player.enqueue(chunk.samples, chunk.sample_rate)
                chunks.append(chunk.samples)
                chunk_rate = chunk.sample_rate
        finally:
            player.done()

        if summary is None or summary.get("cancelled"):
            logger.info("[playback] reply interrupted")
            return
        if summary.get("skipped"):
            logger.info(f"[VAD] backchannel ignored: {summary['user_text']!r}")
            return

        logger.info(
            f"[timing] asr={summary['asr_ms']}ms llm={summary['llm_ms']}ms "
            f"tts={summary['tts_ms']}ms ttfa={summary['ttfa_ms']}ms"
        )
        # Persist the reply exactly as played (chunks concatenated), so the
        # stored path matches what the user actually heard.
        if chunks and chunk_rate is not None:
            sf.write(str(wav_out), np.concatenate(chunks), chunk_rate)

        self.store.log_turn(
            self.session_id,
            turn.index,
            summary["user_text"],
            summary["reply_text"],
            user_wav_path=str(wav_in),
            reply_wav_path=str(wav_out) if chunks else None,
            asr_preset=self.presets["asr"],
            llm_preset=self.presets["llm"],
            tts_preset=self.presets["tts"],
            asr_ms=summary["asr_ms"],
            llm_ms=summary["llm_ms"],
            tts_ms=summary["tts_ms"],
        )
        self.turns_handled += 1

    @staticmethod
    def _log_stage_result(stage: str, text: str, elapsed_ms: int) -> None:
        """Print each stage's text the moment it exists, not after the whole
        turn completes -- otherwise the transcript is withheld until synthesis
        and playback have also finished."""
        if stage == "asr":
            logger.info(f"[user]   {text}  ({elapsed_ms}ms)")
        elif stage == "llm":
            logger.info(f"[nobody] {text}  ({elapsed_ms}ms)")

    # -- shutdown ----------------------------------------------------------

    def extract_memories_from_session(self) -> None:
        """Session-end memory extraction and consolidation.

        Timing is per docs/memory-design.md, which prefers this over per-turn
        extraction. Must run before pipeline.close(), since it still needs the
        LLM loaded. Skipped for a zero-turn session -- nothing was said, and it
        would otherwise burn a generation call on an empty transcript.

        Failures are logged and swallowed on purpose: memory is an enhancement,
        and losing it should not take the rest of shutdown down with it. A
        second Ctrl+C is treated the same way -- the user asking to quit again,
        during the one part of shutdown that is slow, means they want out now,
        not that they want the DB and the audio device left unreleased. Skipping
        this costs at most one session's memories.
        """
        if self.turns_handled == 0:
            return
        try:
            session_turns = [
                (user_text, reply_text)
                for _asr, _llm, _tts, user_text, reply_text in self.store.turns_for_session(
                    self.session_id
                )
            ]
            logger.info("[memory] extracting from this session...")
            candidates = extract_memories(self.pipeline.llm, session_turns)
            # Mem0-style consolidation: diff each candidate against what is
            # already stored and ADD/UPDATE/NOOP it, rather than blindly saving
            # everything. Falls back to all-ADD if the model output is unusable,
            # so this can only tidy the table, never lose a fact.
            existing = self.store.memories_for_consolidation()
            added = updated = skipped = 0
            for op in consolidate_memories(self.pipeline.llm, existing, candidates):
                if op["op"] == "ADD":
                    m = op["memory"]
                    self.store.save_memory(
                        self.session_id, m["category"], m["key"], m["value"], m["confidence"]
                    )
                    added += 1
                elif op["op"] == "UPDATE":
                    m = op["memory"]
                    self.store.update_memory(op["target_id"], m["value"], m["confidence"])
                    updated += 1
                else:
                    skipped += 1
            logger.info(f"[memory] added {added}, updated {updated}, skipped {skipped}")
        except KeyboardInterrupt:
            logger.info("[memory] 건너뜀 (Ctrl+C)")
        except Exception:
            logger.exception("[memory] extraction failed, skipping")

    def close(self) -> None:
        self.store.end_session(self.session_id)
        self.store.close()
        logger.info(f"Session {self.session_id} ended ({self.turns_handled} turns).")


def main() -> None:
    args = parse_args()

    logger.info("Loading models...")
    pipeline, presets = build_pipeline_from_args(args)

    audio_session = None
    if args.aec != "none":
        audio_session = registry.build_audio_session(AEC_BACKENDS[args.aec])
        audio_session.start()
        logger.info(f"[audio] duplex AEC session active (--aec {args.aec})")

    controller, close_capture = build_turn_controller(args, audio_session)
    session = ConversationSession(pipeline, presets, controller)
    session.recall_previous_sessions()

    logger.info(f"Session {session.session_id} started. Just start talking; Ctrl+C to end.")
    controller.start()
    try:
        session.greet()
        session.run()
    except KeyboardInterrupt:
        # Acknowledged immediately, because shutdown is not instant: memory
        # extraction below runs a generation pass, which takes seconds. Without
        # a line here that gap is indistinguishable from Ctrl+C having been
        # ignored, and the natural response is to press it again.
        logger.info("Ctrl+C -- 정리 중 (기억 추출까지 몇 초 걸려. 다시 누르면 건너뜀)")
    finally:
        # Stop capture first: it is the only thread that can still queue work,
        # and letting it run while the pipeline is torn down would race.
        controller.stop()
        # Then release the device itself. controller.stop() only asks the
        # capture thread to finish and gives up on it after a moment -- it
        # cannot interrupt a blocking device read, so that thread is very
        # likely still sitting inside one right now. Closing the stream is what
        # actually ends it, and doing so here rather than leaving it to
        # interpreter shutdown is what keeps exit from hanging in PortAudio.
        close_capture()
        if controller.barge_in_count:
            logger.info(f"[turn] {controller.barge_in_count} barge-in(s) this session")
        session.extract_memories_from_session()
        # Shut down any server-backed ASR/TTS subprocess. These stay alive
        # across turns for speed, so nothing else stops them.
        pipeline.close()
        if audio_session is not None:
            audio_session.close()
        session.close()


if __name__ == "__main__":
    main()
    # Exit without waiting for interpreter teardown.
    #
    # Everything above this line has been released explicitly and the session is
    # over, so there is nothing left that a normal exit would do for us. What it
    # can still do is hang: the native libraries this process loads (PortAudio,
    # onnxruntime, llama.cpp) each run their own threads and atexit teardown,
    # and a daemon capture thread may still be parked inside one of them. Ctrl+C
    # meaning "the process is gone" matters more here than a tidy unwind of
    # state that is already saved.
    #
    # Flush first -- os._exit does not, and the last log lines are the ones that
    # say the session closed cleanly.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
