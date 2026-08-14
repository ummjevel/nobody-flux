"""SentenceChunker + sanitize_for_tts: the critical path every reply's text
takes on its way to TTS. min_chars/max_chars interaction decides
time-to-first-audio; sanitize's "" signal decides whether unplayable chunks
reach the synthesizer.
"""

from __future__ import annotations

from src.nobody_flux.textchunk import SentenceChunker, sanitize_for_tts


# -- push/drain --------------------------------------------------------------


def test_single_delta_can_complete_multiple_chunks():
    chunker = SentenceChunker()
    chunks = chunker.push("안녕하세요. 반갑습니다. 또 봐")
    assert chunks == ["안녕하세요.", "반갑습니다."]
    assert chunker.flush() == "또 봐"


def test_chunks_accumulate_across_deltas():
    chunker = SentenceChunker()
    assert chunker.push("오늘 날씨") == []
    assert chunker.push("가 좋네요.") == ["오늘 날씨가 좋네요."]


def test_min_chars_suppresses_tiny_chunks():
    # "응." hits a primary boundary at 2 chars -- below min_chars, so it must
    # wait rather than become its own TTS call.
    chunker = SentenceChunker()
    assert chunker.push("응.") == []
    assert chunker.push(" 그런데 말이야.") == ["응. 그런데 말이야."]


def test_max_chars_forces_cut_at_secondary_boundary():
    chunker = SentenceChunker(min_chars=6, max_chars=80)
    text = "가" * 70 + "," + "나" * 29
    chunks = chunker.push(text)
    assert chunks == ["가" * 70 + ","]
    assert chunker.flush() == "나" * 29


def test_max_chars_cuts_flat_without_any_boundary():
    chunker = SentenceChunker(min_chars=6, max_chars=80)
    chunks = chunker.push("가" * 100)
    assert chunks == ["가" * 80]
    assert chunker.flush() == "가" * 20


def test_fullwidth_punctuation_is_a_primary_boundary():
    chunker = SentenceChunker()
    assert chunker.push("정말 그런가요？네！") == ["정말 그런가요？"]


def test_flush_resets_for_reuse():
    chunker = SentenceChunker()
    chunker.push("남은 텍스트")
    assert chunker.flush() == "남은 텍스트"
    assert chunker.flush() is None
    assert chunker.push("새 턴이에요.") == ["새 턴이에요."]


def test_flush_returns_none_for_whitespace_only():
    chunker = SentenceChunker()
    chunker.push("   ")
    assert chunker.flush() is None


# -- sanitize ----------------------------------------------------------------


def test_sanitize_strips_emoji_keeps_text():
    assert sanitize_for_tts("좋아요 😊 진짜") == "좋아요 진짜"


def test_sanitize_emoji_only_chunk_becomes_empty_signal():
    # "" is the caller's skip signal -- an emoji-only chunk must not reach TTS,
    # where Matcha returns zero samples for it.
    assert sanitize_for_tts("😊") == ""
    assert sanitize_for_tts("✨👍") == ""


def test_sanitize_collapses_whitespace():
    assert sanitize_for_tts("안녕   하세요\n반가워") == "안녕 하세요 반가워"


def test_sanitize_keeps_digits_and_punctuation():
    assert sanitize_for_tts("3시에 만나!") == "3시에 만나!"
