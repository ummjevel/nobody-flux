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
    instead of a pile of punctuation variants."""
    return text.strip().strip(".!?~,")


def is_backchannel(text: str, duration_s: float) -> bool:
    """True if `text` (an ASR transcript) together with the utterance's
    `duration_s` looks like backchannel rather than a real conversational
    turn -- see module docstring for why both signals matter together (a
    long utterance that happens to start with a backchannel word is not
    backchannel; a short one that isn't in the word list also is not,
    conservatively -- unrecognized short utterances are treated as real
    turns rather than silently dropped, since dropping something the user
    actually said is worse than answering a stray sound).
    """
    if duration_s > BACKCHANNEL_MAX_DURATION_S:
        return False
    return _normalize(text) in BACKCHANNEL_WORDS
