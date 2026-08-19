"""Small pure functions with outsized blast radius: the backchannel gate, the
LocalAgreement prefix rule (ported from scripts/_smoke_turn.py so it runs
without weights), the stabilizer's monotonicity invariant, and the endpoint
grace clamp."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.nobody_flux.stage.asr_stream import LocalAgreementStabilizer, longest_common_prefix
from src.nobody_flux.turn.backchannel import (
    BACKCHANNEL_WORDS,
    ONE_SYLLABLE_WORDS,
    is_backchannel,
    is_empty_transcript,
)
from src.nobody_flux.turn.verdict import TurnVerdict, judge_transcript
from src.nobody_flux.turn.vad import VoiceActivityDetector


# -- backchannel gate ----------------------------------------------------------


def test_backchannel_word_within_duration():
    assert is_backchannel("응", 0.3) is True
    assert is_backchannel("그렇구나", 0.5) is True


def test_backchannel_duration_boundary_is_inclusive():
    assert is_backchannel("응", 0.6) is True
    assert is_backchannel("응", 0.61) is False


def test_backchannel_long_utterance_never_matches():
    # "그래" said as the start of "그래서..." but running long is a real turn.
    assert is_backchannel("그래", 2.0) is False


def test_backchannel_unknown_short_word_is_a_real_turn():
    # Conservative: dropping something the user actually said is worse than
    # answering a stray sound.
    assert is_backchannel("뭐", 0.3) is False


def test_backchannel_strips_ascii_and_fullwidth_punctuation():
    assert is_backchannel("응.", 0.3) is True
    assert is_backchannel("네!", 0.3) is True
    assert is_backchannel(" 어~ ", 0.3) is True
    assert is_backchannel("응。", 0.3) is True
    assert is_backchannel("네！", 0.3) is True


def test_empty_transcript_catches_punctuation_only_asr_output():
    assert is_empty_transcript(".") is True
    assert is_empty_transcript("그.") is True  # bare syllable fragment
    assert is_empty_transcript("") is True
    assert is_empty_transcript("   ") is True
    assert is_empty_transcript("네네") is False


def test_one_syllable_questions_are_not_silence():
    """"뭐?" and "왜?" are whole 반말 questions, not fragments.

    The old length-only rule discarded them, and worse, talk.py's dead-microphone
    warning counts discarded turns -- so asking "뭐?" three times accused the
    microphone of being broken.
    """
    assert is_empty_transcript("뭐") is False
    assert is_empty_transcript("뭐?") is False
    assert is_empty_transcript("왜") is False
    assert judge_transcript("뭐?", 0.4) is TurnVerdict.FINISHED


def test_one_character_backchannel_words_reach_the_backchannel_gate():
    """The shadowing bug: 네 응 음 ... are one character, so a length-only empty
    check returned True for them and is_backchannel never saw them at all --
    meaning the assistant could not hear the user say "네"."""
    for word in ("네", "응", "음", "어", "예", "오", "와", "헐", "아", "넵"):
        assert is_empty_transcript(word) is False, word
        assert judge_transcript(word, 0.3) is TurnVerdict.WAIT, word


def test_no_backchannel_word_is_shadowed_by_the_empty_check():
    """Every entry in the list must be reachable. Half of them were not."""
    unreachable = sorted(w for w in BACKCHANNEL_WORDS if is_empty_transcript(w))
    assert unreachable == []


def test_longer_one_character_acknowledgment_becomes_a_real_turn():
    """Past the backchannel duration it is answered rather than dropped.

    Not an accident: a drawn-out "네" is more likely an answer ("yes, do it")
    than an acknowledgment, and the prior behaviour -- counting it as a dead
    microphone -- was the worst of the three options.
    """
    assert judge_transcript("네", 0.3) is TurnVerdict.WAIT
    assert judge_transcript("네", 1.2) is TurnVerdict.FINISHED


def test_bare_fragments_are_still_discarded():
    """The <= 1 rule was written against real observed failures; promoting a
    fragment to a turn is the thing it exists to prevent."""
    for frag in ("그", "그.", "은", "를", "ㄱ", "…"):
        assert is_empty_transcript(frag) is True, frag
        assert judge_transcript(frag, 0.4) is TurnVerdict.EMPTY, frag


def test_the_two_word_lists_do_not_overlap():
    """An overlap would make a word's verdict depend on set iteration order in
    is_empty_transcript's fallthrough -- ONE_SYLLABLE_WORDS means FINISHED,
    BACKCHANNEL_WORDS means WAIT, and nothing may claim both."""
    assert ONE_SYLLABLE_WORDS & BACKCHANNEL_WORDS == set()


# -- LocalAgreement prefix (cases ported from _smoke_turn.py) --------------------


@pytest.mark.parametrize(
    ("hypotheses", "expected"),
    [
        (["그래서내가", "그래서내가말"], "그래서내가"),  # stable prefix, growing tail
        (["그래서내가", "그래서냈어"], "그래서"),  # tail revised -> only agreed part
        (["안녕", "안녕"], "안녕"),  # identical hypotheses commit fully
        (["가나다", "라마바"], ""),  # no agreement -> nothing commits
        ([], ""),  # no hypotheses at all
        (["하나"], "하나"),  # a single hypothesis is its own prefix
        (["짧다", "짧다길어짐", "짧"], "짧"),  # 3-way agreement bounded by shortest
    ],
)
def test_longest_common_prefix(hypotheses, expected):
    assert longest_common_prefix(hypotheses) == expected


# -- LocalAgreementStabilizer -----------------------------------------------------


def test_stabilizer_commits_only_after_agreement():
    stab = LocalAgreementStabilizer(agreement_n=2)
    stab.observe("그래서")
    assert stab.committed == ""  # one hypothesis is not agreement
    stab.observe("그래서내가")
    assert stab.committed == "그래서"


def test_stabilizer_committed_never_shrinks():
    """The invariant consumers rely on: text once committed stays committed,
    even through a decoder reset that empties the hypothesis."""
    stab = LocalAgreementStabilizer(agreement_n=2)
    stab.observe("그래서내가")
    stab.observe("그래서내가말")
    assert stab.committed == "그래서내가"
    stab.observe("")  # decoder hiccup
    stab.observe("그")
    assert stab.committed == "그래서내가"


def test_stabilizer_agreement_n_three_needs_three():
    stab = LocalAgreementStabilizer(agreement_n=3)
    stab.observe("안녕")
    stab.observe("안녕")
    assert stab.committed == ""
    stab.observe("안녕하")
    assert stab.committed == "안녕"


def test_stabilizer_hypothesis_tracks_latest():
    stab = LocalAgreementStabilizer(agreement_n=2)
    assert stab.hypothesis == ""
    stab.observe("안")
    stab.observe("안녕")
    assert stab.hypothesis == "안녕"


def test_stabilizer_force_commit_and_reset():
    stab = LocalAgreementStabilizer(agreement_n=2)
    stab.force_commit("최종 텍스트")
    assert stab.committed == "최종 텍스트"
    stab.reset()
    assert stab.committed == ""
    assert stab.hypothesis == ""


# -- endpoint grace clamp ----------------------------------------------------------


def _grace_config(**overrides) -> SimpleNamespace:
    # VoiceActivityDetector's __post_init__ builds a real sherpa VAD, which
    # needs model weights -- so the clamp logic is exercised unbound, on a bare
    # config namespace. (Making the detector a pure config+factory is
    # code-review #13's suggestion; this test gets simpler when that lands.)
    config = dict(adaptive_endpoint_grace=True, endpoint_grace_ms=800, endpoint_grace_min_ms=300)
    config.update(overrides)
    return SimpleNamespace(**config)


def grace_frames(config, prob: float) -> int:
    return VoiceActivityDetector.grace_frames_for_prob(config, prob)


def test_grace_clamps_low_probability_to_full_budget():
    assert grace_frames(_grace_config(), -0.5) == int(800 / 30)
    assert grace_frames(_grace_config(), 0.0) == int(800 / 30)


def test_grace_clamps_high_probability_to_minimum():
    assert grace_frames(_grace_config(), 1.0) == int(300 / 30)
    assert grace_frames(_grace_config(), 7.0) == int(300 / 30)


def test_grace_scales_linearly_between():
    assert grace_frames(_grace_config(), 0.5) == int((300 + 500 * 0.5) / 30)


def test_grace_fixed_when_adaptive_disabled():
    config = _grace_config(adaptive_endpoint_grace=False)
    assert grace_frames(config, 0.99) == int(800 / 30)


def test_grace_never_below_one_frame():
    config = _grace_config(endpoint_grace_ms=10, endpoint_grace_min_ms=0)
    assert grace_frames(config, 1.0) == 1
