"""Korean text normalization for the TTS input path: digits and Latin to Hangul.

## Why this exists as code rather than as a prompt instruction

persona.py already asks the model for this, in detail -- write "이십육" not "26",
"에이아이" not "AI", "세 시" not "3시" -- and docs/FEATURES.md records the model
failing to comply ("20대" survived into a live reply). Prompt compliance degrades
exactly where this project lives: small quantized models, long outputs. Worse, it
is untestable. A guard on the input path is neither.

So the prompt instruction stays as layer one -- it is free, and the model's own
Korean number sense handles counter agreement better than any table -- and this is
layer two: a deterministic sweep for whatever got through.

docs/code-review-20260814.md already named the checkpoint: "숫자·로마자는 안 거른다
... 자연스러운 단일 체크포인트가 여기다", pointing at textchunk.sanitize_for_tts.
This module is what that checkpoint calls.

## Why hand-written rather than a dependency

Nothing off the shelf clears all four bars this project needs -- permissive
license, no JVM, no mandatory native extension, and an aarch64 wheel:

  KoG2P, KoNLPy, soynlp, hangul-utils   GPL
  num2words, kiwipiepy                  LGPL
  g2pK, g2pkc, g2pkk, g2pk2             Apache-2.0, but every one of them
                                        imports mecab in __init__ -- including
                                        the forks advertised as mecab-free,
                                        which only hid it behind a dynamic import
  KoNLPy                                additionally wants a JVM on a 4GB board
  NeMo-text-processing                  Apache-2.0 and has genuinely complete
                                        Korean TN/ITN grammars, but needs pynini,
                                        which publishes no aarch64 wheel

The Apache-2.0 ones are readable, so their numeral logic informed this; the
implementation is ours.

## Measured motivation

Both TTS presets get numbers wrong on their own, so this is not one preset's
defect (scripts/_ab_tts.py, 2026-08-18):

    "가격은 12,000원이고 30% 할인 중이야"
      sherpa-matcha-ko -> "가격은 만 천원이고 30%센 할인인 중이야"   (만 이천, not 만 천)
      supertonic-3-ko  -> "가격은 12000고 30 할인 중이야"            (not expanded at all)

## Verified how far, honestly

Measured 2026-08-18 by synthesizing each case twice -- raw, then through
`textchunk.sanitize_for_tts` -- and transcribing both with SenseVoice.

The Latin cases improve unambiguously, on both presets:

    "와이파이 비밀번호는 ABC야."
      sherpa-matcha-ko   before "와이파이 비밀번호는 엠비이야"  (ABC read as MBC)
                         after  "와이파이 비밀번호는 에이비시아"
      supertonic-3-ko    before "W파이 Premier번 on ABCia"    (unintelligible)
                         after  "와이파이 비밀번호는 에이비시야"

    "AI랑 GPU 얘기"
      sherpa-matcha-ko   before "얘들랑 디 비유"   after "에이아이랑 지피유"
      supertonic-3-ko    before "에엘랑쥐 피해"     after "베이아이랑 지피유"

Native-counter numbers improve too: "23살" was read "2십3 살" (matcha) and
"이샘살" (supertonic) before, and "스물세 살" / "23살" after.

**But this harness cannot validate Sino number expansion, and it is important not
to claim that it did.** SenseVoice applies inverse text normalization, so it
writes "20분" for a correctly-spoken "이십 분" -- the judge undoes precisely the
transformation under test, and a correct result is indistinguishable from an
untouched one. One case even reads as a possible regression that may only be an
ASR artifact: "만 이천 원" came back as "120 원". Whether the TTS said that, or
the recognizer misheard it, cannot be settled without listening.

So: Latin and native counters are measured. Sino numerals are unit-tested for the
text transformation and **unverified acoustically** -- that needs a human ear, and
belongs on docs/FEATURES.md's human-verification list rather than being counted
as done here.

## Scope, honestly

Handled: Sino-Korean cardinals with 만/억/조 grouping, native-Korean numerals
selected by counter, the 유월/시월 month irregulars, decimals, percent, currency,
clock times, hyphenated digit strings read digit-by-digit, and Latin acronyms.

Not handled, deliberately: transliterating lowercase or mixed-case Latin words.
Spelling "Starbucks" out as letter names would be far worse than leaving it, so
unknown words are left for the TTS and a small loanword table covers the common
ones. Also not handled: counter-specific irregulars beyond the common set, and
ordinals (첫째/둘째).
"""

from __future__ import annotations

import re

# --------------------------------------------------------------- Sino-Korean

_SINO_DIGITS = ("", "일", "이", "삼", "사", "오", "육", "칠", "팔", "구")
# Grouped by ten thousands, which is the Korean grouping -- not by thousands.
# This is why a naive comma-driven reading produces "만 천" for 12,000: the comma
# sits at the wrong boundary for the language.
_SINO_GROUPS = ("", "만", "억", "조", "경")
_SINO_PLACES = ((1000, "천"), (100, "백"), (10, "십"))


def _sino_below_10000(value: int) -> str:
    """Read 1..9999. A leading 1 in the 천/백/십 places is dropped (십, not 일십)
    but kept in the ones place (일)."""
    out = []
    for size, name in _SINO_PLACES:
        digit, value = divmod(value, size)
        if digit:
            out.append(("" if digit == 1 else _SINO_DIGITS[digit]) + name)
    if value:
        out.append(_SINO_DIGITS[value])
    return "".join(out)


def sino_cardinal(n: int) -> str:
    """Sino-Korean reading of a non-negative integer.

    Groups of four digits, so 12000 is 만 이천 -- one 만 plus 이천 -- rather than
    the twelve-thousand reading a Western grouping suggests. The unit multiplier
    is dropped when it is 1: 10000 is 만, not 일만.
    """
    if n < 0:
        return "마이너스 " + sino_cardinal(-n)
    if n == 0:
        return "영"

    groups: list[tuple[int, str]] = []
    index = 0
    while n > 0 and index < len(_SINO_GROUPS):
        n, remainder = divmod(n, 10000)
        if remainder:
            groups.append((remainder, _SINO_GROUPS[index]))
        index += 1
    if n:  # beyond 경 -- give up rather than mis-say it
        return ""

    parts = []
    for value, unit in reversed(groups):
        if unit and value == 1:
            parts.append(unit)
        else:
            parts.append(_sino_below_10000(value) + unit)
    return " ".join(parts)


# ------------------------------------------------------------ native Korean

_NATIVE_ONES = ("", "하나", "둘", "셋", "넷", "다섯", "여섯", "일곱", "여덟", "아홉")
_NATIVE_TENS = ("", "열", "스물", "서른", "마흔", "쉰", "예순", "일흔", "여든", "아흔")
# Before a counter, four of the ones change shape, and 20 changes entirely.
_ATTRIBUTIVE = {"하나": "한", "둘": "두", "셋": "세", "넷": "네"}

NATIVE_MAX = 99


def native_cardinal(n: int) -> str:
    """Native-Korean reading, 1..99. Empty string outside that range -- the
    native series has no productive forms above 아흔아홉, which is why Korean
    switches to Sino for larger counts even with native counters."""
    if not 1 <= n <= NATIVE_MAX:
        return ""
    tens, ones = divmod(n, 10)
    return _NATIVE_TENS[tens] + _NATIVE_ONES[ones]


def native_attributive(n: int) -> str:
    """Native reading in the form used before a counter: 한/두/세/네, and 스무
    for exactly 20 (스무 살, but 스물세 살 for 23)."""
    if not 1 <= n <= NATIVE_MAX:
        return ""
    if n == 20:
        return "스무"
    tens, ones = divmod(n, 10)
    tail = _NATIVE_ONES[ones]
    return _NATIVE_TENS[tens] + _ATTRIBUTIVE.get(tail, tail)


# ------------------------------------------------------------------ counters

# Counters that take native numerals. Not exhaustive -- the long tail of Korean
# counters is enormous -- but these are the ones that show up in conversation,
# and an unlisted counter falls back to Sino, which is the safer error: Sino with
# a native counter sounds stilted, while native with a Sino counter sounds wrong.
NATIVE_COUNTERS = frozenset(
    """개 명 살 시 시간 마리 권 장 번 켤레 벌 채 대 그릇 병 잔 조각 송이 자루 통 갑 판 줄
       가지 걸음 사람 살배기 달 군데 곳 판 척 그루 포기 알 톨 마디 뼘 아이""".split()
)

# Counters that take Sino numerals. Listed explicitly rather than inferred so a
# clash with the native set is a visible conflict rather than dictionary order.
SINO_COUNTERS = frozenset(
    """분 초 년 월 일 원 인분 층 호 번지 퍼센트 프로 도 미터 센티미터 킬로미터 킬로그램 그램
       리터 밀리리터 주 주일 개월 학년 회 등 위 점 페이지 쪽 배 인 명분 킬로 센티""".split()
)

# 6월 and 10월 are not 육월 and 십월. These are the whole reading, 월 included,
# which is why _read_month returns the complete word rather than a numeral for
# the caller to append 월 to -- doing that produced "유월 월".
_MONTH_IRREGULAR = {6: "유월", 10: "시월"}


def _read_month(n: int) -> str:
    """The complete month name. Months are one word in Korean (삼월, not 삼 월)."""
    if n in _MONTH_IRREGULAR:
        return _MONTH_IRREGULAR[n]
    return sino_cardinal(n) + "월"


# ------------------------------------------------------------------- Latin

_LETTER_NAMES = {
    "A": "에이", "B": "비", "C": "씨", "D": "디", "E": "이", "F": "에프",
    "G": "지", "H": "에이치", "I": "아이", "J": "제이", "K": "케이", "L": "엘",
    "M": "엠", "N": "엔", "O": "오", "P": "피", "Q": "큐", "R": "알",
    "S": "에스", "T": "티", "U": "유", "V": "브이", "W": "더블유", "X": "엑스",
    "Y": "와이",
    # 제트 rather than 지 so Z does not collide with G when spelled aloud.
    "Z": "제트",
}

# Words that would be mangled by letter-spelling and are common enough to be
# worth a table. Deliberately short: this is not a transliteration engine.
LOANWORDS = {
    "wifi": "와이파이", "wi-fi": "와이파이", "youtube": "유튜브",
    "instagram": "인스타그램", "kakaotalk": "카카오톡", "google": "구글",
    "netflix": "넷플릭스", "starbucks": "스타벅스", "iphone": "아이폰",
    "android": "안드로이드", "email": "이메일", "ok": "오케이",
    "tv": "티브이", "pc": "피씨", "ai": "에이아이", "gpu": "지피유",
    "cpu": "씨피유", "usb": "유에스비", "app": "앱", "sns": "에스엔에스",
}

# Longest-first so "wi-fi" wins over "wi".
_LOANWORD_RE = re.compile(
    "|".join(re.escape(w) for w in sorted(LOANWORDS, key=len, reverse=True)),
    re.IGNORECASE,
)
_LATIN_RUN_RE = re.compile(r"[A-Za-z]+")
_ACRONYM_MAX = 5


def spell_latin(run: str) -> str:
    """Letter-by-letter Hangul for an acronym; unchanged for a word.

    The split is on shape, not on a dictionary: an all-caps run of at most
    _ACRONYM_MAX letters is an acronym (AI, GPU, USB), and anything else is a
    word. Spelling a word out -- "Starbucks" as 에스티에이알비유씨케이에스 -- is a
    far worse failure than handing it to the TTS as-is, so words are left alone.
    """
    if len(run) == 1:
        return _LETTER_NAMES.get(run.upper(), run)
    if run.isupper() and len(run) <= _ACRONYM_MAX:
        return "".join(_LETTER_NAMES.get(ch, ch) for ch in run)
    return run


# --------------------------------------------------------------- the sweep

# A phone number or an ID: read digit-by-digit, and 0 is 공 rather than 영 in
# this context specifically.
_DIGIT_STRING_RE = re.compile(r"\b\d{2,4}-\d{3,4}-\d{4}\b")
_DIGIT_BY_DIGIT = {"0": "공", "1": "일", "2": "이", "3": "삼", "4": "사",
                   "5": "오", "6": "육", "7": "칠", "8": "팔", "9": "구"}

_PERCENT_UNITS = {"%": "퍼센트"}

_ALL_COUNTERS = sorted(NATIVE_COUNTERS | SINO_COUNTERS, key=len, reverse=True)
_NUMBER_RE = re.compile(
    r"(?P<int>\d[\d,]*)"
    r"(?:\.(?P<frac>\d+))?"
    r"\s*"
    r"(?P<unit>%|" + "|".join(re.escape(c) for c in _ALL_COUNTERS) + r")?"
)


def _read_digits(digits: str) -> str:
    return " ".join(_DIGIT_BY_DIGIT[d] for d in digits)


def _replace_number(match: re.Match) -> str:
    raw = match.group("int").replace(",", "")
    frac = match.group("frac")
    unit = match.group("unit") or ""
    unit = _PERCENT_UNITS.get(unit, unit)

    try:
        value = int(raw)
    except ValueError:  # pragma: no cover -- the regex cannot produce this
        return match.group(0)

    if frac is not None:
        # Decimals: integer part read normally, fraction digit-by-digit after 점.
        head = sino_cardinal(value)
        body = head + " 점 " + " ".join(_SINO_DIGITS[int(d)] or "영" for d in frac)
        return (body + " " + unit).strip() if unit else body

    if unit == "월":
        return _read_month(value)

    if unit in NATIVE_COUNTERS:
        native = native_attributive(value)
        if native:
            return native + " " + unit
        # Above 99 the native series runs out and Korean uses Sino even here.
        return sino_cardinal(value) + " " + unit

    spoken = sino_cardinal(value)
    if not spoken:
        return match.group(0)  # too large to say correctly; leave it
    return (spoken + " " + unit).strip() if unit else spoken


def expand(text: str) -> str:
    """Rewrite digits and Latin in `text` into speakable Hangul.

    Order matters. Hyphenated digit strings go first, because the number sweep
    would otherwise read each group of a phone number as a quantity. Loanwords go
    before the generic Latin pass so "wifi" becomes 와이파이 rather than being
    spelled out.
    """
    text = _DIGIT_STRING_RE.sub(lambda m: _read_digits(m.group(0).replace("-", "")), text)
    text = _LOANWORD_RE.sub(lambda m: LOANWORDS[m.group(0).lower()], text)
    text = _NUMBER_RE.sub(_replace_number, text)
    text = _LATIN_RUN_RE.sub(lambda m: spell_latin(m.group(0)), text)
    return re.sub(r"\s+", " ", text).strip()


def has_unspeakable(text: str) -> bool:
    """Whether anything a phoneme TTS would stumble on survives.

    Used for logging rather than control flow: if this is true after `expand`,
    the sweep has a gap worth knowing about, and silently shipping the text is
    how the gap stays invisible.
    """
    return bool(re.search(r"[0-9A-Za-z]", text))
