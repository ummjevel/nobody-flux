"""Unit tests for src/nobody_flux/metrics.py.

Every case here is drawn from a transcript this project actually produced, so
the test suite documents the failure shapes as much as it checks the arithmetic.
"""

import unicodedata

import pytest

from src.nobody_flux.metrics import (
    ErrorCounts,
    align_counts,
    cer,
    cer_detail,
    is_effectively_empty,
    normalize_for_cer,
)


# --------------------------------------------------------------- normalization

def test_nfc_composes_conjoining_jamo():
    """Decomposed and precomposed Hangul must score as identical.

    '가' exists as U+AC00 and as U+1100 U+1161. Without NFC these compare as a
    1-vs-2 character mismatch and every syllable reads as an error.
    """
    precomposed = "가"
    decomposed = unicodedata.normalize("NFD", "가")
    assert precomposed != decomposed              # different code points...
    assert cer(precomposed, decomposed) == 0.0    # ...same syllable


def test_drop_space_collapses_spurious_eojeol_breaks():
    """SenseVoice inserts mid-eojeol spaces ('생각 을'); that is not a recognition error."""
    assert cer("생각을", "생각 을") == 0.0
    # ...but only because we asked it to be ignored.
    assert cer("생각을", "생각 을", drop_space=False) > 0.0


def test_drop_punct_ignores_unspoken_marks():
    assert cer("누구세요", "누구세요.") == 0.0
    assert cer("누구세요", "누구세요?") == 0.0


def test_keeping_space_normalizes_runs():
    assert normalize_for_cer("a,  b", drop_space=False) == "a b"


# ------------------------------------------------------------ the empty failure

def test_empty_hypothesis_is_all_deletions():
    """The streaming-zipformer-ko failure signature.

    Returning '' scores CER 1.0 made entirely of deletions. This must be
    distinguishable from a model that misheard every character, which scores the
    same 1.0 via substitutions -- that distinction is the whole reason the
    S/D/I breakdown exists.
    """
    counts = cer_detail("누구세요", "")
    assert counts == ErrorCounts(substitutions=0, deletions=4, insertions=0, ref_len=4)
    assert counts.rate == 1.0


def test_punctuation_only_hypothesis_is_also_empty():
    """A live session produced turns transcribed as '.' and '그.'."""
    assert is_effectively_empty(".")
    assert is_effectively_empty("  ")
    assert not is_effectively_empty("그.")   # '그' is a real character
    # '.' normalizes away entirely, so it scores as a total deletion, not a near-miss.
    assert cer_detail("누구세요", ".").deletions == 4


def test_fully_wrong_same_length_is_all_substitutions():
    counts = cer_detail("가나다", "마바사")
    assert counts == ErrorCounts(substitutions=3, deletions=0, insertions=0, ref_len=3)
    assert counts.rate == 1.0


# ------------------------------------------------------------------- arithmetic

def test_identical_is_zero():
    assert cer("오늘 산책 코스 추천해줘", "오늘 산책 코스 추천해줘") == 0.0


def test_single_substitution():
    counts = cer_detail("가나다", "가라다")
    assert (counts.substitutions, counts.deletions, counts.insertions) == (1, 0, 0)
    assert counts.rate == pytest.approx(1 / 3)


def test_deletion_of_leading_syllables():
    """The documented pre-roll loss: '산책 코스 추천해줘' arrived as '코 추천해 줘'."""
    counts = cer_detail("산책코스추천해줘", "코추천해줘")
    assert counts.deletions == 3
    assert counts.substitutions == 0
    assert counts.insertions == 0


def test_insertion_counted():
    counts = cer_detail("가나", "가나다")
    assert (counts.substitutions, counts.deletions, counts.insertions) == (0, 0, 1)


def test_rate_can_exceed_one_when_babbling():
    """A model that emits far more than it heard is worse than 100% wrong."""
    counts = cer_detail("가", "가나다라마바사")
    assert counts.rate > 1.0


def test_empty_reference_with_output_is_rate_one():
    """Nothing to be wrong about, but emitting speech into silence is still a failure."""
    assert cer_detail("", "무언가").rate == 1.0


def test_empty_reference_and_empty_hypothesis_is_zero():
    assert cer_detail("", "").rate == 0.0


def test_align_counts_is_symmetric_in_total_distance():
    """Levenshtein distance is symmetric even though D and I swap roles."""
    a, b = "가나다", "가다"
    assert align_counts(a, b).total == align_counts(b, a).total


@pytest.mark.parametrize(
    "ref,hyp,expected_total",
    [
        ("", "", 0),
        ("a", "", 1),
        ("", "a", 1),
        ("abc", "abc", 0),
        ("abc", "abd", 1),
        ("abc", "ab", 1),
        ("ab", "abc", 1),
        ("kitten", "sitting", 3),   # the textbook case
    ],
)
def test_edit_distance_matches_known_values(ref, hyp, expected_total):
    assert align_counts(ref, hyp).total == expected_total
