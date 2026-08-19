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

# Real one-syllable utterances, kept as a word list because that is what the
# problem actually needs: length is not a proxy for meaning. "그" is a fragment
# of a word that ASR cut short; "뭐" is a whole question. is_empty_transcript
# used to discard both, on length alone.
#
# Deliberately small. An entry here promotes something to a full conversational
# turn, so a wrong one means answering a fragment -- which is the exact failure
# the "<= 1 character" rule was written against. Only unambiguous standalone
# words belong: 나/너 ("me?"/"you?") are grammatical but far more often the
# leading syllable of a longer sentence, so they are left out until real
# recordings say otherwise (scripts/_calibrate_turn_params.py).
ONE_SYLLABLE_WORDS = {"뭐", "뭘", "왜"}

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

    One character is *usually* nothing: that is what a syllable fragment looks
    like when ASR cuts a word short ("그"). But it is not always nothing, and for
    a while this function acted as if it were.

    What the plain ``len <= 1`` rule got wrong (fixed 2026-08-19):

      - It discarded real one-syllable turns. "뭐?" and "왜?" are ordinary 반말
        questions, and they were treated as silence.
      - It shadowed half of BACKCHANNEL_WORDS. 네 넵 아 어 예 오 와 음 응 헐 are
        all one character, so they returned True here and never reached
        is_backchannel at all -- meaning the assistant could not hear the user
        say "네". Only the two-syllable entries (그래, 맞아, 어어, 진짜 ...) ever
        worked.
      - It made the diagnostic lie. talk.py counts consecutive empty turns and
        warns that the microphone is probably dead; a user saying "뭐?" three
        times tripped that warning.

    So the question asked here is "is this a fragment?", not "is this short?",
    and that needs a word list rather than a length tweak. A one-character
    transcript survives if it is a known word -- either a real utterance
    (ONE_SYLLABLE_WORDS, which becomes a full turn) or an acknowledgment
    (BACKCHANNEL_WORDS, which now actually reaches is_backchannel and becomes
    WAIT). Anything else one character or shorter is still discarded.

    What this deliberately does NOT settle: "네" is genuinely ambiguous between
    an acknowledgment ("mm-hmm, go on") and an answer ("yes, do it"), and text
    plus duration cannot tell them apart -- so it stays in BACKCHANNEL_WORDS and
    a short "네" is still not replied to. That is a policy question for real
    labelled recordings (scripts/_calibrate_turn_params.py), not something to
    guess at here. But it is strictly better than before, where a short "네" was
    counted as a dead microphone.
    """
    normalized = _normalize(text)
    if len(normalized) > 1:
        return False
    return normalized not in ONE_SYLLABLE_WORDS and normalized not in BACKCHANNEL_WORDS


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
