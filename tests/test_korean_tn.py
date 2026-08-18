"""Korean number/Latin expansion for the TTS input path.

Cases are drawn from three places: persona.py's own worked examples (the spec
the model is asked to follow), docs/FEATURES.md's recorded compliance failures,
and the scripts/_ab_tts.py measurements where both TTS presets got numbers wrong
on their own.

Pure text, no weights, no audio -- same contract as the rest of tests/.
"""

import pytest

from src.nobody_flux.korean_tn import (
    expand,
    has_unspeakable,
    native_attributive,
    native_cardinal,
    sino_cardinal,
    spell_latin,
)


# ------------------------------------------------------------- Sino cardinals

@pytest.mark.parametrize(
    "n,expected",
    [
        (0, "영"),
        (1, "일"),
        (10, "십"),          # not 일십
        (11, "십일"),
        (20, "이십"),
        (26, "이십육"),       # persona.py's own example
        (100, "백"),          # not 일백
        (101, "백일"),
        (1000, "천"),
        (1234, "천이백삼십사"),
    ],
)
def test_sino_below_10000(n, expected):
    assert sino_cardinal(n) == expected


@pytest.mark.parametrize(
    "n,expected",
    [
        (10000, "만"),            # not 일만
        (12000, "만 이천"),        # the case sherpa-matcha-ko read as "만 천"
        (100000, "십만"),
        (110000, "십일만"),
        (12345, "만 이천삼백사십오"),
        (100000000, "억"),
        (123456789, "억 이천삼백사십오만 육천칠백팔십구"),
    ],
)
def test_sino_ten_thousand_grouping(n, expected):
    """Korean groups by ten thousands, which is why a comma-driven reading of
    12,000 produces the wrong answer."""
    assert sino_cardinal(n) == expected


def test_sino_negative():
    assert sino_cardinal(-5) == "마이너스 오"


def test_absurdly_large_returns_empty_rather_than_guessing():
    assert sino_cardinal(10**25) == ""


# ----------------------------------------------------------- native numerals

@pytest.mark.parametrize(
    "n,expected",
    [(1, "하나"), (2, "둘"), (3, "셋"), (10, "열"), (20, "스물"), (23, "스물셋"), (99, "아흔아홉")],
)
def test_native_cardinal(n, expected):
    assert native_cardinal(n) == expected


@pytest.mark.parametrize(
    "n,expected",
    [
        (1, "한"),
        (2, "두"),
        (3, "세"),
        (4, "네"),
        (5, "다섯"),      # unchanged before a counter
        (20, "스무"),      # 스무 살, not 스물 살
        (21, "스물한"),
        (23, "스물세"),    # 스물세 살
        (30, "서른"),
    ],
)
def test_native_attributive(n, expected):
    assert native_attributive(n) == expected


def test_native_series_has_no_forms_above_99():
    assert native_cardinal(100) == ""
    assert native_attributive(100) == ""


# ------------------------------------------------------------------ counters

def test_native_counter_takes_native_numeral():
    assert expand("사과 3개 주세요") == "사과 세 개 주세요"
    assert expand("20살이야") == "스무 살이야"
    assert expand("23살이라고 했잖아") == "스물세 살이라고 했잖아"


def test_clock_hour_is_native_and_minute_is_sino():
    """persona.py's example: "3시" -> "세 시". Minutes stay Sino."""
    assert expand("지금 3시 20분이야") == "지금 세 시 이십 분이야"


def test_sino_counter_takes_sino_numeral():
    assert expand("3년 걸렸어") == "삼 년 걸렸어"
    assert expand("2개월 남았어") == "이 개월 남았어"


def test_native_counter_falls_back_to_sino_above_99():
    assert expand("100명 왔어") == "백 명 왔어"


def test_unknown_counter_falls_back_to_sino():
    """An unlisted counter is not recognized as a unit at all, so only the
    numeral is rewritten and it stays attached to whatever followed it.

    Sino is the right fallback: Sino with a native counter merely sounds stilted,
    while native with a Sino counter sounds wrong.
    """
    assert expand("5뭉치") == "오뭉치"


# -------------------------------------------------------- month irregulars

@pytest.mark.parametrize(
    "text,expected",
    [
        ("6월에 만나", "유월에 만나"),
        ("10월", "시월"),
        ("3월", "삼월"),
        ("12월", "십이월"),
    ],
)
def test_months(text, expected):
    """6월 is 유월 and 10월 is 시월 -- never 육월/십월. And a month is one word
    in Korean, so 삼월 rather than "삼 월".

    Asserted by equality, not membership. An earlier version of this test used
    `"유월" in expand("6월")` and passed while the function was returning
    "유월 월" -- the irregular form already contains 월 and the caller appended it
    again. A substring assertion cannot see that class of bug.
    """
    assert expand(text) == expected


# ------------------------------------------------------------------ decimals

def test_decimal_reads_fraction_digit_by_digit():
    assert expand("3.14") == "삼 점 일 사"


def test_decimal_zero_in_fraction():
    assert expand("1.05") == "일 점 영 오"


# -------------------------------------------------------- percent, currency

def test_percent():
    assert expand("30% 할인") == "삼십 퍼센트 할인"


def test_currency_with_thousands_comma():
    """The full failing case from scripts/_ab_tts.py."""
    assert expand("가격은 12,000원이고 30% 할인 중이야") == (
        "가격은 만 이천 원이고 삼십 퍼센트 할인 중이야"
    )


# --------------------------------------------------------- digit strings

def test_phone_number_is_read_digit_by_digit_with_gong():
    """0 is 공, not 영, when reading a number out digit by digit."""
    assert expand("010-1234-5678") == "공 일 공 일 이 삼 사 오 육 칠 팔"


def test_a_plain_number_is_not_read_digit_by_digit():
    assert expand("1234") == "천이백삼십사"


# ----------------------------------------------------------------- Latin

@pytest.mark.parametrize(
    "run,expected",
    [
        ("AI", "에이아이"),      # persona.py's example
        ("GPU", "지피유"),       # persona.py's example
        ("USB", "유에스비"),
        ("A", "에이"),
    ],
)
def test_acronyms_are_spelled_out(run, expected):
    assert spell_latin(run) == expected


def test_z_does_not_collide_with_g():
    assert spell_latin("Z") != spell_latin("G")


def test_words_are_left_alone_rather_than_spelled_out():
    """Spelling a word letter-by-letter is a worse failure than leaving it.

    "Starbucks" as 에스티에이알비유씨케이에스 would be unintelligible; handed to
    the TTS it at least produces something word-shaped.
    """
    assert spell_latin("Starbucks") == "Starbucks"
    assert spell_latin("hello") == "hello"


def test_long_all_caps_run_is_treated_as_a_word():
    assert spell_latin("ABCDEFGH") == "ABCDEFGH"


def test_loanword_table_beats_letter_spelling():
    assert expand("와이파이 비밀번호") == "와이파이 비밀번호"
    assert expand("wifi 비밀번호") == "와이파이 비밀번호"
    assert expand("AI 얘기") == "에이아이 얘기"


def test_loanword_match_is_case_insensitive_and_longest_first():
    assert expand("Wi-Fi") == "와이파이"
    assert expand("YouTube") == "유튜브"


# -------------------------------------------------------------- the sweep

def test_expand_is_idempotent_on_clean_korean():
    text = "오늘 산책 코스 추천해 줄까?"
    assert expand(text) == text


def test_has_unspeakable_flags_leftovers():
    assert not has_unspeakable(expand("지금 3시 20분이야"))
    # "Starbucks" is in the loanword table, so it does NOT survive -- use a word
    # that is not, since an unknown word is deliberately left alone and the flag
    # staying true is the signal, not a bug.
    assert not has_unspeakable(expand("Starbucks 갈까?"))
    assert has_unspeakable(expand("Kierkegaard 읽었어?"))


def test_mixed_sentence():
    got = expand("26살인데 3시에 AI 얘기 30분 했어")
    assert got == "스물여섯 살인데 세 시에 에이아이 얘기 삼십 분 했어"
