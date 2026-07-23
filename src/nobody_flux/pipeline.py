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

from . import registry
from .asr import NobodyASR
from .llm import NobodyLLM
from .tts import NobodyTTS


@dataclass
class STSPipeline:
    asr: NobodyASR = field(default_factory=registry.build_asr)
    llm: NobodyLLM = field(default_factory=registry.build_llm)
    tts: NobodyTTS = field(default_factory=registry.build_tts)

    def run(self, wav_in: str, wav_out: str) -> dict:
        """One turn: transcribe wav_in, generate a reply, synthesize it to wav_out.

        Returns a dict with the intermediate text plus per-stage latency (ms) so
        callers (this module's own CLI, scripts/talk.py, storage.py) can inspect
        what happened at each stage and compare presets over time.
        """
        t0 = time.perf_counter()
        user_text = self.asr.transcribe_file(wav_in)
        t1 = time.perf_counter()
        reply_text = self.llm.reply(user_text)
        t2 = time.perf_counter()
        self.tts.synthesize(reply_text, out_path=wav_out)
        t3 = time.perf_counter()
        return {
            "user_text": user_text,
            "reply_text": reply_text,
            "wav_out": wav_out,
            "asr_ms": round((t1 - t0) * 1000),
            "llm_ms": round((t2 - t1) * 1000),
            "tts_ms": round((t3 - t2) * 1000),
        }
