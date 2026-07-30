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

import time
from dataclasses import dataclass, field
from typing import Callable

from . import registry
from .asr import NobodyASR
from .llm import NobodyLLM
from .tts import NobodyTTS


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

        Neither callback is required; run_pipeline.py's single one-shot call
        doesn't need live progress, so both stay optional rather than forcing
        every caller to care.
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
        }

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
