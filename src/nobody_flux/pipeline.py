"""End-to-end local STS pipeline: wav in -> ASR -> LLM -> TTS -> wav out.

Prototype scope (see docs/output/ondevice_asr_llm_tts_research_20260716.md for
the model research this is built on). Which concrete ASR/LLM/TTS implementation
runs is decided by the caller (see registry.py) -- this class only orchestrates
whatever asr/llm/tts objects it's given and times each stage, it doesn't know
about presets.

This validates the pipeline shape on dev hardware (GPU available here) before
any on-device optimization work. Wakeword and full streaming ASR are out of
scope for this pass; turn-taking is a simple VAD utterance boundary (see vad.py),
not full-duplex barge-in.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Iterator

import numpy as np

from . import registry
from .stage.asr import NobodyASR
from .stage.llm import NobodyLLM
from .stage.tts import NobodyTTS
from .textchunk import SentenceChunker, sanitize_for_tts

# How many sentence chunks may sit queued for synthesis ahead of the TTS
# worker. Small on purpose: the queue exists so the LLM keeps decoding while a
# sentence synthesizes (code-review #2), not to synthesize a whole reply ahead
# of playback -- a deep queue would just turn a barge-in into wasted synthesis.
# 2 is enough to keep the worker fed across chunk boundaries.
TTS_QUEUE_DEPTH = 2

# Sentinel telling the synthesis worker there is no more text coming.
_TTS_DONE = object()


@dataclass
class AudioChunk:
    """One speakable piece of a streamed reply: the synthesized samples plus the
    text they came from and their 1-based order. Yielded by
    STSPipeline.run_streaming so talk.py can start playback on chunk 1 while the
    LLM is still generating (and TTS still synthesizing) later chunks."""

    samples: np.ndarray  # mono float32
    sample_rate: int
    index: int
    text: str


@dataclass
class STSPipeline:
    asr: NobodyASR = field(default_factory=registry.build_asr)
    llm: NobodyLLM = field(default_factory=registry.build_llm)
    tts: NobodyTTS = field(default_factory=registry.build_tts)

    def run(
        self,
        wav_in: str,
        wav_out: str,
        on_stage_start: Callable[[str], None] | None = None,
        on_result: Callable[[str, str, int], None] | None = None,
        should_continue_after_asr: Callable[[str], bool] | None = None,
    ) -> dict:
        """One turn: transcribe wav_in, generate a reply, synthesize it to wav_out.

        Returns a dict with the intermediate text plus per-stage latency (ms) so
        callers (this module's own CLI, scripts/talk.py, storage.py) can inspect
        what happened at each stage and compare presets over time.

        on_stage_start: optional callback invoked with "asr"/"llm"/"tts" right
        before that stage starts. Each stage can take seconds (see the *_ms
        fields this returns), and without any signal a caller has no way to
        tell "still working" apart from "hung" -- talk.py uses this to print
        progress.

        on_result: optional callback invoked with (stage, text, elapsed_ms)
        the moment that stage's output exists -- ("asr", user_text, asr_ms)
        right after transcription, ("llm", reply_text, llm_ms) right after
        the reply is generated -- rather than only after the whole turn
        (asr+llm+tts) completes. TTS has no comparable intermediate text to
        report, so it never fires for "tts". talk.py uses this to print the
        transcript (and how long it took) right after ASR instead of holding
        it until playback is about to start.

        should_continue_after_asr: optional callback invoked with user_text
        right after ASR; if it returns False, this returns immediately
        (llm_text/tts_ms/wav_out all None, result["skipped"] = True)
        without running the LLM or TTS stage at all. talk.py uses this for
        docs/barge-in-design.md's stage 2 (post-hoc lexical backchannel
        check) -- a short "어"/"응" shouldn't spend a full LLM+TTS call (and
        shouldn't get logged as a real conversation turn) just because ASR
        happened to transcribe it successfully.

        Neither callback is required; run_pipeline.py's single one-shot call
        doesn't need live progress, so all three stay optional rather than
        forcing every caller to care.
        """

        def stage_start(stage: str) -> None:
            if on_stage_start is not None:
                on_stage_start(stage)

        def result(stage: str, value: str, elapsed_ms: int) -> None:
            if on_result is not None:
                on_result(stage, value, elapsed_ms)

        t0 = time.perf_counter()
        stage_start("asr")
        user_text = self.asr.transcribe_file(wav_in)
        t1 = time.perf_counter()
        asr_ms = round((t1 - t0) * 1000)
        result("asr", user_text, asr_ms)

        if should_continue_after_asr is not None and not should_continue_after_asr(user_text):
            return {
                "user_text": user_text,
                "reply_text": None,
                "wav_out": None,
                "asr_ms": asr_ms,
                "llm_ms": None,
                "tts_ms": None,
                "skipped": True,
            }

        stage_start("llm")
        reply_text = self.llm.reply(user_text)
        t2 = time.perf_counter()
        llm_ms = round((t2 - t1) * 1000)
        result("llm", reply_text, llm_ms)

        stage_start("tts")
        self.tts.synthesize(reply_text, out_path=wav_out)
        t3 = time.perf_counter()
        tts_ms = round((t3 - t2) * 1000)
        return {
            "user_text": user_text,
            "reply_text": reply_text,
            "wav_out": wav_out,
            "asr_ms": asr_ms,
            "llm_ms": llm_ms,
            "tts_ms": tts_ms,
            "skipped": False,
        }

    def run_streaming(
        self,
        wav_in: str | None = None,
        on_stage_start: Callable[[str], None] | None = None,
        on_result: Callable[[str, str, int], None] | None = None,
        should_continue_after_asr: Callable[[str], bool] | None = None,
        should_cancel: Callable[[], bool] | None = None,
        pretranscribed: str | None = None,
    ) -> Iterator[AudioChunk]:
        """One turn, but pipelined: transcribe, then stream the LLM reply and
        synthesize+yield it sentence by sentence so the caller can begin
        playback on the first sentence instead of waiting for the whole reply
        to be generated and synthesized (Phase 1 latency lever -- see
        textchunk.SentenceChunker and docs/FEATURES.md).

        Yields AudioChunk objects in order. The final summary dict (same shape
        as run(), plus ttfa_ms = time-to-first-audio and n_chunks) is the
        generator's *return value* -- retrieve it from StopIteration.value:

            gen = pipeline.run_streaming(...)
            try:
                while True:
                    play(next(gen))
            except StopIteration as done:
                result = done.value

        Callbacks mirror run()'s: on_stage_start("asr"/"llm"/"tts"),
        on_result("asr", text, ms) / ("llm", full_reply, ms). should_continue_after_asr
        works exactly as in run() (backchannel skip) -- when it returns False,
        no chunks are yielded and the summary has skipped=True.

        should_cancel: optional predicate polled at every point where work could
        still be abandoned -- after ASR, after each LLM delta, and before each
        synthesized chunk is yielded. When it returns True this stops
        immediately and the summary has cancelled=True.

        This is Phase 4's half of production-window barge-in. Detecting an
        interruption is the turn controller's job; *acting* on one is this
        method's, and it can only act at the points where it holds control.
        Polling after each delta is what makes an interruption land during
        generation rather than at the end of it -- the difference between the
        reply stopping when the user speaks over it and stopping a second and a
        half later, once the model happens to finish.

        Cancellation is checked, not raised: an exception thrown through a
        generator that owns a partially-consumed LLM stream and a live TTS
        backend would leave both in an undefined state. Returning normally lets
        each stage unwind the way it does on any other completed turn.
        """

        def stage_start(stage: str) -> None:
            if on_stage_start is not None:
                on_stage_start(stage)

        def result(stage: str, value: str, elapsed_ms: int) -> None:
            if on_result is not None:
                on_result(stage, value, elapsed_ms)

        def cancelled() -> bool:
            return should_cancel is not None and should_cancel()

        def summary(**overrides) -> dict:
            """Build the summary dict, so every early exit reports the same
            keys. Callers read this by key; a path that omitted one would fail
            at the call site rather than here."""
            base = {
                "user_text": None,
                "reply_text": None,
                "asr_ms": None,
                "llm_ms": None,
                "tts_ms": None,
                "ttfa_ms": None,
                "n_chunks": 0,
                "skipped": False,
                "cancelled": False,
            }
            base.update(overrides)
            return base

        t0 = time.perf_counter()
        if pretranscribed is not None:
            # Phase 3: recognition already happened, incrementally, while the
            # user was still speaking (see stage/asr_stream.py and
            # turn/controller.py). There is nothing left to transcribe, so the
            # ASR stage is skipped entirely rather than re-decoding audio whose
            # text we are already holding.
            #
            # asr_ms is reported as 0 rather than None on purpose: it is a real
            # measurement, not a missing one. The recognition cost was paid
            # during the user's own speech, where it overlapped time that was
            # going to elapse anyway, so its contribution to *turn* latency
            # genuinely is zero. Recording it as None would break every
            # comparison against the batch path in storage and benchmarks.
            user_text = pretranscribed
            asr_ms = 0
        else:
            if wav_in is None:
                raise ValueError(
                    "run_streaming needs either wav_in (to transcribe) or "
                    "pretranscribed (text already recognized by the streaming ASR)."
                )
            stage_start("asr")
            user_text = self.asr.transcribe_file(wav_in)
            asr_ms = round((time.perf_counter() - t0) * 1000)
        result("asr", user_text, asr_ms)

        if should_continue_after_asr is not None and not should_continue_after_asr(user_text):
            return summary(user_text=user_text, asr_ms=asr_ms, skipped=True)

        # Checked before the LLM is touched at all: if the user has already
        # interrupted by the time recognition finished, generation is pure waste.
        if cancelled():
            return summary(user_text=user_text, asr_ms=asr_ms, cancelled=True)

        stage_start("llm")
        chunker = SentenceChunker()
        llm_t0 = time.perf_counter()
        reply_pieces: list[str] = []

        # -- TTS worker (code-review #2) --------------------------------------
        # Synthesis used to run synchronously inside this loop, and the default
        # GGUF stream only decodes when pulled -- so while a sentence was being
        # synthesized, token generation was zero and total response time was
        # sum(llm) + sum(tts) instead of their overlap. One worker thread keeps
        # the LLM decoding while TTS runs; a *single* worker, so chunk order is
        # preserved without any re-sequencing.
        tts_state = {"total_s": 0.0, "ttfa_ms": None, "index": 0, "started": False}
        work_q: queue.Queue = queue.Queue(maxsize=TTS_QUEUE_DEPTH)
        out_q: queue.Queue = queue.Queue()
        worker_done = False

        def synth_one(text: str) -> AudioChunk | None:
            """Synthesize one chunk, or None if it isn't speakable / produced no
            audio (see sanitize_for_tts). A None chunk costs no index/ttfa so an
            emoji-only fragment simply vanishes instead of becoming a silent,
            unplayable (0-sample, sr=0) chunk. Runs on the worker thread."""
            clean = sanitize_for_tts(text)
            if not clean:
                return None
            if not tts_state["started"]:
                # Before the first synthesis *begins*, not after it ends -- the
                # old placement made talk.py's "[TTS] synthesizing..." line
                # appear only once the first chunk was already done.
                tts_state["started"] = True
                stage_start("tts")
            ts = time.perf_counter()
            samples, sr = self.tts.synthesize_audio(clean)
            tts_state["total_s"] += time.perf_counter() - ts
            if samples.size == 0:
                return None
            if tts_state["ttfa_ms"] is None:
                tts_state["ttfa_ms"] = round((time.perf_counter() - t0) * 1000)
            tts_state["index"] += 1
            return AudioChunk(
                samples=samples, sample_rate=sr, index=tts_state["index"], text=clean
            )

        def tts_worker() -> None:
            while True:
                text = work_q.get()
                if text is _TTS_DONE:
                    out_q.put(_TTS_DONE)
                    return
                try:
                    out_q.put(synth_one(text))
                except Exception as exc:  # surfaced on the consuming thread
                    out_q.put(exc)
                    return

        def handle(item) -> AudioChunk | None:
            """Classify one out_q item; raises a worker exception here, on the
            consuming thread, exactly where the synchronous version raised."""
            nonlocal worker_done
            if item is _TTS_DONE:
                worker_done = True
                return None
            if isinstance(item, Exception):
                worker_done = True
                raise item
            return item

        def abort_worker() -> None:
            """Stop the worker and discard whatever it hadn't spoken yet. A
            sentence already inside synthesize_audio() finishes (it is not
            interruptible) and is thrown away. Idempotent -- also the finally
            path for GeneratorExit, when the caller close()s mid-reply."""
            nonlocal worker_done
            if not worker_done:
                while True:
                    try:
                        work_q.get_nowait()
                    except queue.Empty:
                        break
                work_q.put(_TTS_DONE)
                while not worker_done:
                    item = out_q.get()
                    if item is _TTS_DONE or isinstance(item, Exception):
                        # A worker that died while being aborted is not worth
                        # re-raising over -- the reply is being discarded anyway.
                        worker_done = True
            worker.join()

        worker = threading.Thread(target=tts_worker, name="tts-synth", daemon=True)
        worker.start()

        # Latched, not re-polled (code-review #8): the summary's `cancelled`
        # means "work was actually abandoned", so a barge-in that lands after
        # the last chunk was already yielded does not retroactively mark a
        # fully-delivered reply as cancelled (which used to drop it from
        # storage and memory extraction entirely).
        interrupted = False

        try:
            stream = self.llm.reply_stream(user_text)
            for piece in stream:
                reply_pieces.append(piece)
                # Polled once per delta -- the finest granularity available
                # without reaching inside the model's own generation loop, and
                # fine enough that a barge-in costs at most one token's worth
                # of extra work.
                if cancelled():
                    # close() propagates GeneratorExit into reply_stream, which
                    # stops the underlying generation and, importantly, lets it
                    # skip the history update -- an interrupted reply the user
                    # never heard should not become conversational context.
                    stream.close()
                    interrupted = True
                    break
                for chunk_text in chunker.push(piece):
                    while True:
                        try:
                            work_q.put_nowait(chunk_text)
                            break
                        except queue.Full:
                            # The worker is a couple of sentences behind. Keep
                            # the caller supplied with finished audio while
                            # waiting for space, and keep watching for barge-in
                            # so a full queue can't delay cancellation.
                            if cancelled():
                                stream.close()
                                interrupted = True
                                break
                            try:
                                chunk = handle(out_q.get(timeout=0.05))
                            except queue.Empty:
                                continue
                            if chunk is not None:
                                yield chunk
                    if interrupted:
                        break
                if interrupted:
                    break
                # Hand over whatever the worker finished during this delta.
                while not worker_done:
                    try:
                        item = out_q.get_nowait()
                    except queue.Empty:
                        break
                    chunk = handle(item)
                    if chunk is not None:
                        yield chunk

            if interrupted:
                abort_worker()
                return summary(
                    user_text=user_text,
                    reply_text="".join(reply_pieces).strip(),
                    asr_ms=asr_ms,
                    llm_ms=round((time.perf_counter() - llm_t0) * 1000),
                    tts_ms=round(tts_state["total_s"] * 1000),
                    ttfa_ms=tts_state["ttfa_ms"],
                    n_chunks=tts_state["index"],
                    cancelled=True,
                )

            llm_ms = round((time.perf_counter() - llm_t0) * 1000)
            reply_text = "".join(reply_pieces).strip()
            result("llm", reply_text, llm_ms)

            tail = chunker.flush()
            if tail and cancelled():
                # The tail will never be spoken -- that is abandoned work, so
                # this is a genuine cancellation (unlike the post-yield re-poll
                # this replaces).
                interrupted = True
                abort_worker()
                return summary(
                    user_text=user_text,
                    reply_text=reply_text,
                    asr_ms=asr_ms,
                    llm_ms=llm_ms,
                    tts_ms=round(tts_state["total_s"] * 1000),
                    ttfa_ms=tts_state["ttfa_ms"],
                    n_chunks=tts_state["index"],
                    cancelled=True,
                )
            if tail:
                work_q.put(tail)
            work_q.put(_TTS_DONE)
            while not worker_done:
                chunk = handle(out_q.get())
                if chunk is not None:
                    yield chunk
            worker.join()

            return summary(
                user_text=user_text,
                reply_text=reply_text,
                asr_ms=asr_ms,
                llm_ms=llm_ms,
                tts_ms=round(tts_state["total_s"] * 1000),
                ttfa_ms=tts_state["ttfa_ms"],
                n_chunks=tts_state["index"],
                cancelled=False,
            )
        finally:
            # Runs on every exit: normal return (worker already joined),
            # cancellation (already aborted), GeneratorExit from the caller
            # closing us mid-reply, or a raising stage. Guarantees no orphaned
            # synthesis thread survives the turn.
            abort_worker()

    def close(self) -> None:
        """Release persistent subprocess resources (server-backed ASR/TTS
        presets like VibeAsrBitnet/FreyaTtsKo). No-op for stages that don't
        hold one (in-process ASR/LLM, per-call-subprocess TTS) -- callers
        should call this unconditionally rather than checking which preset
        is active."""
        for stage in (self.asr, self.llm, self.tts):
            close = getattr(stage, "close", None)
            if close is not None:
                close()
