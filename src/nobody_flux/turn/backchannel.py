"""Stage 2 of docs/barge-in-design.md's two-stage barge-in/backchannel
disambiguation: a post-hoc lexical check against the ASR result, run once an
utterance has already been captured in full.

Stage 1 (vad.py's barge_in_confirm_ms/on_barge_in_confirmed) only sees audio
and decides in real time whether to cut off playback -- it can't know
whether a confirmed-real utterance was actually the word "어" or a whole
sentence, because that requires ASR, which only exists once the utterance is
over. This module is what talk.py calls after ASR to decide whether the
utterance should still become a full conversation turn (LLM + TTS + storage)
or get quietly skipped.
"""

from __future__ import annotations

# Exact-match set, not substring/regex -- a short casual reply like "어제
# 뭐했어?" starts with the same syllable as the backchannel "어" but is a
# real question, not an acknowledgment. Matching the whole (normalized)
# utterance instead of looking for these as a prefix/substring avoids that
# false positive. Not exhaustive -- docs/barge-in-design.md flags this list
# as a draft to be expanded from real usage, not a finished vocabulary.
BACKCHANNEL_WORDS = {
    "어", "어어", "응", "으응", "네", "넵", "예",
    "오", "오오", "와", "헐",
    "진짜", "정말", "그렇구나", "그래", "그니까", "맞아", "아하", "음", "아",
}

# docs/barge-in-design.md's initial estimate -- a real interruption that
# happens to be lexically ambiguous (e.g. just "그래" said as the start of
# "그래서 있잖아...") but runs longer than this isn't treated as backchannel,
# since duration is the stronger signal for a longer utterance and the word
# list is only meant to catch genuinely short acknowledgments.
BACKCHANNEL_MAX_DURATION_S = 0.6


def _normalize(text: str) -> str:
    """Strips whitespace and trailing punctuation ASR might emit ("어.",
    "응!", "어 ") so BACKCHANNEL_WORDS can stay a plain set of bare words
    instead of a pile of punctuation variants. Includes the CJK fullwidth
    forms ("。！？") -- Korean models emit both widths (see textchunk.PRIMARY)."""
    return text.strip().strip(".!?~,。！？、…")


def is_empty_transcript(text: str) -> bool:
    """True if ASR came back with nothing worth replying to.

    Recognizers do not return an empty string for silence; they return
    punctuation. A live session where the microphone had gone silent produced
    six turns transcribed as ``'.'``, ``'그.'``, ``'예.'`` -- and the LLM
    answered every one of them, repeating its previous reply verbatim because
    there was no new content to respond to. From the outside that reads as the
    assistant having lost its mind, when in fact it was being handed nothing.

    Guarding on the transcript rather than on the audio level is deliberate:
    "the recognizer found no words" is the condition that actually matters, and
    it holds whether the cause was a muted microphone, a stray noise, or speech
    too quiet to decode. An audio-level threshold would additionally have to be
    right for every microphone and room, and would still not catch a clear
    recording of a door closing.

    A single character is included because that is what a bare syllable
    fragment looks like ("그"), and one syllable carries no answerable content.

    CAUTION -- this docstring used to claim that "anything genuinely meant as a
    one-word reply ('네', '응') is caught by is_backchannel below". That is not
    true, and the two examples it named are the proof: both are one character, so
    this function returns True for them and they never reach is_backchannel at
    all. Half of BACKCHANNEL_WORDS is shadowed the same way -- 네 넵 아 어 예 오
    와 음 응 헐 are unreachable entries, while the two-syllable ones (그래, 맞아,
    어어, 진짜 ...) do work.

    Today that costs nothing user-visible, because EMPTY and WAIT both skip the
    turn. Two things it does cost:

      - Real one-syllable turns are discarded. "뭐?" and "왜?" are ordinary
        반말 questions and this treats them as silence.
      - The diagnostic lies. talk.py counts consecutive empty turns and warns
        that the microphone is probably dead; a user saying "뭐?" three times
        trips that warning.

    Not changed here, deliberately: separating "그" (a fragment) from "뭐" (a
    word) is a lexical judgement needing a word list, not a length tweak, and the
    <= 1 rule was itself written against an observed failure. See
    docs/FEATURES.md's human-verification list.
    """
    return len(_normalize(text)) <= 1


def is_backchannel(
    text: str, duration_s: float, max_duration_s: float = BACKCHANNEL_MAX_DURATION_S
) -> bool:
    """True if `text` (an ASR transcript) together with the utterance's
    `duration_s` looks like backchannel rather than a real conversational
    turn -- see module docstring for why both signals matter together (a
    long utterance that happens to start with a backchannel word is not
    backchannel; a short one that isn't in the word list also is not,
    conservatively -- unrecognized short utterances are treated as real
    turns rather than silently dropped, since dropping something the user
    actually said is worse than answering a stray sound).

    `max_duration_s` defaults to the module constant, so callers that don't pass
    it behave exactly as before. It exists as a parameter because the constant is
    an estimate with no config slot -- code-review #9 flagged that, and
    docs/FEATURES.md lists it as awaiting real labelled recordings via
    scripts/_calibrate_turn_params.py. A parameter is what lets that calibration
    land in a yaml later without touching this function again.
    """
    if duration_s > max_duration_s:
        return False
    return _normalize(text) in BACKCHANNEL_WORDS
