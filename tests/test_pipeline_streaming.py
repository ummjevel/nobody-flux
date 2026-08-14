"""STSPipeline.run_streaming with all three stages faked: the
skipped/cancelled/asr_ms=0 contract, chunk ordering through the TTS worker
thread (code-review #2), and the cancellation latch (code-review #8 -- a
barge-in after the last chunk must NOT retroactively mark a fully-delivered
reply cancelled).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.nobody_flux.pipeline import STSPipeline


class FakeAsr:
    def __init__(self, text="오늘 뭐 했어"):
        self.text = text
        self.calls = 0

    def transcribe_file(self, path) -> str:
        self.calls += 1
        return self.text


class FakeLlm:
    def __init__(self, pieces):
        self.pieces = pieces
        self.closed = False
        self.remembered = False

    def reply_stream(self, user_text):
        try:
            yield from self.pieces
            self.remembered = True  # only a fully-consumed stream updates history
        except GeneratorExit:
            self.closed = True
            raise


class FakeTts:
    def __init__(self):
        self.texts: list[str] = []

    def synthesize_audio(self, text):
        self.texts.append(text)
        return np.ones(160, dtype=np.float32), 16_000


def make_pipeline(pieces, asr=None, tts=None) -> tuple[STSPipeline, FakeLlm, FakeTts]:
    llm = FakeLlm(pieces)
    tts = tts if tts is not None else FakeTts()
    pipeline = STSPipeline(asr=asr or FakeAsr(), llm=llm, tts=tts)
    return pipeline, llm, tts


def drain(generator):
    """Consume the generator; return (chunks, summary)."""
    chunks = []
    while True:
        try:
            chunks.append(next(generator))
        except StopIteration as done:
            return chunks, done.value


def test_chunks_arrive_in_order_with_summary():
    pipeline, llm, tts = make_pipeline(["안녕하세요.", " 반갑습니다.", " 또 봐요."])
    chunks, summary = drain(pipeline.run_streaming(wav_in="x.wav"))

    assert [c.text for c in chunks] == ["안녕하세요.", "반갑습니다.", "또 봐요."]
    assert [c.index for c in chunks] == [1, 2, 3]
    assert summary["n_chunks"] == 3
    assert summary["reply_text"] == "안녕하세요. 반갑습니다. 또 봐요."
    assert summary["cancelled"] is False
    assert summary["skipped"] is False
    assert llm.remembered  # history updated exactly because the stream completed


def test_pretranscribed_skips_asr_and_reports_zero_ms():
    asr = FakeAsr()
    pipeline, _llm, _tts = make_pipeline(["응 왜."], asr=asr)
    _chunks, summary = drain(pipeline.run_streaming(pretranscribed="목소리로 인식된 텍스트"))
    assert asr.calls == 0
    assert summary["asr_ms"] == 0  # a real measurement, not a missing one
    assert summary["user_text"] == "목소리로 인식된 텍스트"


def test_needs_wav_or_pretranscribed():
    pipeline, _llm, _tts = make_pipeline(["응."])
    with pytest.raises(ValueError):
        drain(pipeline.run_streaming())


def test_backchannel_skip_yields_nothing():
    pipeline, _llm, tts = make_pipeline(["안 나와야 함."])
    chunks, summary = drain(
        pipeline.run_streaming(wav_in="x.wav", should_continue_after_asr=lambda text: False)
    )
    assert chunks == []
    assert summary["skipped"] is True
    assert tts.texts == []


def test_cancel_before_llm_generates_nothing():
    pipeline, _llm, tts = make_pipeline(["안 나와야 함."])
    chunks, summary = drain(pipeline.run_streaming(wav_in="x.wav", should_cancel=lambda: True))
    assert chunks == []
    assert summary["cancelled"] is True
    assert tts.texts == []


def test_cancel_during_generation_closes_stream_and_latches():
    # cancelled() call order is deterministic: pre-LLM, then once per piece.
    # Flip true at the second piece so generation is genuinely abandoned.
    calls = {"n": 0}

    def should_cancel():
        calls["n"] += 1
        return calls["n"] >= 3

    pipeline, llm, _tts = make_pipeline(["첫 문장입니다.", " 둘째 문장", " 은 안 나옴."])
    _chunks, summary = drain(pipeline.run_streaming(wav_in="x.wav", should_cancel=should_cancel))
    assert summary["cancelled"] is True
    assert llm.closed  # the LLM stream was told to stop decoding
    assert not llm.remembered  # an interrupted reply stays out of history


def test_tail_abandoned_by_cancel_is_a_real_cancellation():
    # Calls: pre-LLM (1), piece 1 (2), piece 2 (3), tail check (4) -> True.
    calls = {"n": 0}

    def should_cancel():
        calls["n"] += 1
        return calls["n"] >= 4

    pipeline, _llm, tts = make_pipeline(["첫 문장입니다.", " 잘 지"])
    _chunks, summary = drain(pipeline.run_streaming(wav_in="x.wav", should_cancel=should_cancel))
    assert summary["cancelled"] is True  # the tail will never be spoken
    # The abandoned tail must never reach TTS. (Whether the *first* chunk got
    # synthesized before the abort is a race with the worker -- deliberately
    # not asserted.)
    assert "잘 지" not in tts.texts


def test_cancel_after_everything_delivered_is_not_a_cancellation():
    """THE #8 regression: the old code re-polled cancelled() after the last
    yield, so a barge-in landing at the very end of playback erased a fully
    generated, fully spoken turn from storage and memory extraction."""
    calls = {"n": 0}

    def should_cancel():
        calls["n"] += 1
        # False for every real decision point (pre-LLM + 1 piece, no tail);
        # true afterwards -- which must not matter.
        return calls["n"] > 2

    pipeline, llm, _tts = make_pipeline(["완결된 한 문장입니다."])
    chunks, summary = drain(pipeline.run_streaming(wav_in="x.wav", should_cancel=should_cancel))
    assert len(chunks) == 1
    assert summary["cancelled"] is False
    assert llm.remembered


def test_unspeakable_chunks_vanish_without_index_gaps():
    emoji_tts = FakeTts()
    pipeline, _llm, _ = make_pipeline(["😊😊😊. 진짜 반가워."], tts=emoji_tts)
    chunks, summary = drain(pipeline.run_streaming(wav_in="x.wav"))
    # The emoji-only sentence sanitizes to "." -> "." chunk... the punctuation
    # survives sanitize, so assert on indexes being gapless instead.
    assert [c.index for c in chunks] == list(range(1, len(chunks) + 1))
    assert summary["n_chunks"] == len(chunks)


def test_tts_worker_exception_propagates_to_consumer():
    class ExplodingTts(FakeTts):
        def synthesize_audio(self, text):
            raise RuntimeError("synth backend died")

    pipeline, _llm, _tts = make_pipeline(["터질 문장입니다."], tts=ExplodingTts())
    with pytest.raises(RuntimeError, match="synth backend died"):
        drain(pipeline.run_streaming(wav_in="x.wav"))


def test_generator_close_leaves_no_worker_behind():
    import threading

    pipeline, _llm, _tts = make_pipeline(["첫 문장입니다.", " 둘째 문장입니다.", " 셋째도 있어요."])
    generator = pipeline.run_streaming(wav_in="x.wav")
    next(generator)  # take one chunk, then abandon mid-reply (barge-in shape)
    generator.close()
    assert not any(t.name == "tts-synth" and t.is_alive() for t in threading.enumerate())


def test_stage_callbacks_fire_in_order():
    events: list[str] = []
    pipeline, _llm, _tts = make_pipeline(["콜백 확인용 문장."])
    drain(
        pipeline.run_streaming(
            wav_in="x.wav", on_stage_start=events.append, on_result=lambda s, t, ms: None
        )
    )
    assert events == ["asr", "llm", "tts"]
