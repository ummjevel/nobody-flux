"""The turn verdict vocabulary.

This is a naming refactor, so the most important tests here are the ones proving
equivalence with what the three call sites did before -- if those pass, nothing
about the conversation changed.

Weights-free and device-free: the whole point of extracting these decisions is
that they can be checked without a microphone or an ONNX session.
"""

import pytest

from src.nobody_flux.turn.backchannel import is_backchannel, is_empty_transcript
from src.nobody_flux.turn.verdict import (
    TurnVerdict,
    judge_acoustic,
    judge_transcript,
    should_respond,
)


# ------------------------------------------------------------------ acoustic

def test_complete_segment_is_finished():
    assert judge_acoustic(is_complete=True, at_max_duration=False) is TurnVerdict.FINISHED


def test_incomplete_segment_is_unfinished():
    assert judge_acoustic(is_complete=False, at_max_duration=False) is TurnVerdict.UNFINISHED


def test_max_duration_overrides_an_incomplete_verdict():
    """max_speech_duration is a guarantee, not a preference.

    A detector that never says "complete" -- or a microphone stuck on -- must not
    be able to hold a turn open forever, so the cap wins over the model.
    """
    assert judge_acoustic(is_complete=False, at_max_duration=True) is TurnVerdict.FINISHED


def test_acoustic_never_returns_a_transcript_verdict():
    """WAIT and EMPTY require a transcript, which does not exist yet at this point."""
    for complete in (True, False):
        for at_max in (True, False):
            verdict = judge_acoustic(is_complete=complete, at_max_duration=at_max)
            assert verdict in (TurnVerdict.FINISHED, TurnVerdict.UNFINISHED)


# ---------------------------------------------------------------- transcript

def test_real_utterance_is_finished():
    assert judge_transcript("내일 날씨 어때?", 1.5) is TurnVerdict.FINISHED


def test_multisyllable_backchannel_is_wait():
    assert judge_transcript("그래", 0.5) is TurnVerdict.WAIT
    assert judge_transcript("맞아", 0.5) is TurnVerdict.WAIT
    assert judge_transcript("어어", 0.4) is TurnVerdict.WAIT


def test_single_syllable_backchannel_is_reported_empty_not_wait():
    """Documents a shadowing bug found while writing these tests.

    "응"/"네"/"어" are one character, so is_empty_transcript catches them before
    is_backchannel ever runs -- half of BACKCHANNEL_WORDS is unreachable. Both
    verdicts skip the turn, so nothing user-visible breaks today, but the
    diagnostic counter is wrong and the docstring in backchannel.py used to claim
    the opposite. Asserting the real behaviour so a future fix has to update this
    test deliberately rather than discovering it by accident.
    """
    assert judge_transcript("응", 0.4) is TurnVerdict.EMPTY
    assert judge_transcript("네", 0.4) is TurnVerdict.EMPTY


def test_empty_transcript_is_empty():
    """Recognizers signal silence with punctuation, not with an empty string --
    a live session produced turns transcribed as '.', '그.', '예.'."""
    assert judge_transcript(".", 0.7) is TurnVerdict.EMPTY
    assert judge_transcript("", 0.7) is TurnVerdict.EMPTY


def test_empty_is_checked_before_backchannel():
    """Otherwise a '.' could be reported as WAIT and the dead-microphone signal
    that talk.py counts would be lost."""
    assert judge_transcript(".", 0.1) is TurnVerdict.EMPTY


def test_a_long_utterance_is_not_backchannel_even_if_the_words_match():
    """Duration and lexis both have to agree -- "그래" held for two seconds is
    the start of a sentence, not an acknowledgment."""
    assert judge_transcript("그래", 2.0) is TurnVerdict.FINISHED


def test_short_unrecognized_utterance_of_two_syllables_is_a_real_turn():
    """Conservative on purpose: dropping something the user actually said is
    worse than answering a stray sound."""
    assert judge_transcript("누구", 0.3) is TurnVerdict.FINISHED


def test_one_syllable_real_question_is_wrongly_discarded():
    """Known issue, asserted so it cannot regress silently.

    "뭐?" and "왜?" are ordinary 반말 questions. The <= 1 length rule in
    is_empty_transcript cannot tell them from a stray fragment like "그", so the
    assistant ignores them -- and counts them toward the "microphone is probably
    dead" warning. Fixing it needs a lexical distinction, not a length tweak.
    """
    assert judge_transcript("뭐", 0.4) is TurnVerdict.EMPTY
    assert judge_transcript("왜", 0.4) is TurnVerdict.EMPTY


def test_transcript_never_returns_unfinished():
    """By the time there is a transcript the utterance is over, so "still
    talking" is no longer an available answer."""
    for text in ("응", ".", "내일 날씨 어때?", "뭐"):
        for dur in (0.1, 0.5, 2.0):
            assert judge_transcript(text, dur) is not TurnVerdict.UNFINISHED


def test_max_backchannel_threshold_is_overridable():
    """The constant is an estimate awaiting real labelled recordings
    (scripts/_calibrate_turn_params.py); a parameter is what lets a calibrated
    value land later without touching the function."""
    assert judge_transcript("그래", 1.0) is TurnVerdict.FINISHED
    assert judge_transcript("그래", 1.0, max_backchannel_s=1.5) is TurnVerdict.WAIT


# ------------------------------------------------------------ should_respond

def test_only_finished_earns_a_reply():
    assert should_respond(TurnVerdict.FINISHED)
    assert not should_respond(TurnVerdict.UNFINISHED)
    assert not should_respond(TurnVerdict.WAIT)
    assert not should_respond(TurnVerdict.EMPTY)


# ------------------------------------------------------- equivalence proofs

@pytest.mark.parametrize(
    "text,duration",
    [
        ("내일 날씨 어때?", 1.5),
        ("응", 0.4),
        ("그래", 0.5),
        ("그래", 2.0),
        (".", 0.7),
        ("", 0.7),
        ("그.", 0.8),
        ("뭐", 0.3),
        ("예", 0.55),
        ("오늘 산책 코스 추천해줘", 2.8),
    ],
)
def test_matches_the_old_inline_gate_exactly(text, duration):
    """The gate this replaced was, verbatim, talk.py:472:

        should_continue_after_asr=lambda text: not (
            is_empty_transcript(text) or is_backchannel(text, turn.speech_duration_s)
        )

    If this holds for every case, the refactor changed no behaviour.
    """
    old = not (is_empty_transcript(text) or is_backchannel(text, duration))
    new = should_respond(judge_transcript(text, duration))
    assert old == new


@pytest.mark.parametrize("is_complete", [True, False])
@pytest.mark.parametrize("at_max", [True, False])
def test_matches_the_old_vad_condition_exactly(is_complete, at_max):
    """The condition this replaced was, verbatim, vad.py:529:

        if is_complete or len(combined) >= self._max_samples:
    """
    old = is_complete or at_max
    new = judge_acoustic(is_complete=is_complete, at_max_duration=at_max) is TurnVerdict.FINISHED
    assert old == new
