"""One vocabulary for "what did the user just do", shared by the three places
that decide it.

docs/voice-agent-oss-survey.md's action #3 was to adopt TEN Framework's
three-state turn model -- finished / unfinished / wait -- because a binary
end-of-turn cannot express "paused mid-sentence" or "just acknowledging you", and
this project handles both. Those cases were already handled, correctly, in three
different files with no shared name:

    vad.py:528       the detector says the segment is not a complete turn, so
                     carry it and extend the silence grace   -> UNFINISHED
    vad.py:529       ...unless max_speech_duration is hit, which ends the turn
                     regardless of what the model thinks     -> FINISHED
    talk.py:472      an inline lambda: skip the reply if the transcript is empty
                     or lexically a backchannel              -> EMPTY / WAIT

This module names them. That is the whole change -- **behaviour is identical**,
and the tests that pinned the old behaviour still pass unmodified. What it buys
is that the four outcomes are visible in one place, and that the lexical gate is
a tested function rather than a lambda inside a CLI script.

## Two phases, and why they cannot be collapsed

The tempting reading of a three-state model is one function that looks at a turn
and returns its state. That is not implementable here, and the reason is
structural rather than incidental:

  - FINISHED vs UNFINISHED is decided **from audio, before ASR runs**. It has to
    be: it is what determines whether to keep waiting, and waiting for a
    transcript first would defeat the purpose.
  - WAIT is decided **from the transcript, after ASR**. It has to be: telling
    "어" (an acknowledgment) from "어제 뭐 했어?" (a question) is a lexical
    judgement, and docs/barge-in-design.md records measuring Smart Turn as a
    backchannel classifier and finding it unsuitable -- "응" scored 0.723 against
    a real trailing-off sentence at 0.506.

So there are two entry points, `judge_acoustic` and `judge_transcript`, and the
verdict can be revised downward when the transcript arrives: a turn that looked
FINISHED acoustically becomes WAIT once we can read it. That revision is the
existing two-stage barge-in design (docs/barge-in-design.md), now named.

## The fourth state

TEN's model has three; this has four. EMPTY is not in TEN's vocabulary because
theirs is audio-native and never sees a transcript, whereas ours can get one back
that contains no speech -- recognizers signal silence with punctuation, and a
live session produced six turns transcribed as '.', '그.', '예.' with the LLM
answering every one.

EMPTY and WAIT produce the same *action* (do not reply, keep listening) but are
kept distinct because they mean different things diagnostically: consecutive
EMPTY verdicts are the signature of a dead microphone, which talk.py warns on,
while consecutive WAITs are just an agreeable user.
"""

from __future__ import annotations

from enum import Enum

from .backchannel import BACKCHANNEL_MAX_DURATION_S, is_backchannel, is_empty_transcript


class TurnVerdict(Enum):
    """What the user's last stretch of speech amounts to."""

    # A complete turn. Reply to it.
    FINISHED = "finished"
    # A pause inside a thought, not the end of one. Keep the audio, extend the
    # silence grace, wait for them to continue.
    UNFINISHED = "unfinished"
    # Real speech, but an acknowledgment rather than a turn ("응", "그래").
    # Stay listening; replying would talk over someone who is still listening.
    WAIT = "wait"
    # Nothing recognizable came back. Discard, and count it -- a run of these
    # means the microphone died, not that the user went quiet.
    EMPTY = "empty"


def judge_acoustic(*, is_complete: bool, at_max_duration: bool) -> TurnVerdict:
    """FINISHED or UNFINISHED, from audio alone, before ASR.

    ``is_complete`` is Smart Turn's verdict (``TurnDetector.predict`` returns it
    as ``prob >= complete_threshold``). It is passed in rather than re-derived
    from the probability here so the threshold has exactly one home.

    ``at_max_duration`` overrides the model, and must: ``max_speech_duration``
    exists so that a stuck microphone or a detector that never says "complete"
    cannot hold a turn open forever. A model verdict is a preference; this is a
    guarantee.
    """
    if is_complete or at_max_duration:
        return TurnVerdict.FINISHED
    return TurnVerdict.UNFINISHED


def judge_transcript(
    text: str,
    speech_duration_s: float,
    *,
    max_backchannel_s: float = BACKCHANNEL_MAX_DURATION_S,
) -> TurnVerdict:
    """FINISHED, WAIT or EMPTY, from the transcript, after ASR.

    Never returns UNFINISHED: by the time there is a transcript the utterance is
    already over, so "they are still talking" is no longer one of the available
    answers.

    Order matters. Empty is checked first because an empty transcript would
    otherwise normalize to something the backchannel matcher could not match
    anyway, and reporting it as WAIT would lose the dead-microphone signal.

    ``speech_duration_s`` must be the *speech* duration, excluding pre-roll.
    Passing the padded length is a bug this project already shipped once: the
    gate silently became dead code when pre_roll_ms rose from 300 to 500,
    because the padded duration never fell under the threshold (code-review #1).
    """
    if is_empty_transcript(text):
        return TurnVerdict.EMPTY
    if is_backchannel(text, speech_duration_s, max_duration_s=max_backchannel_s):
        return TurnVerdict.WAIT
    return TurnVerdict.FINISHED


def should_respond(verdict: TurnVerdict) -> bool:
    """Whether this verdict earns a full turn -- LLM, TTS and storage.

    Only FINISHED does. UNFINISHED is still in progress, and WAIT and EMPTY are
    both "stay listening", which is why replacing the old
    ``not (is_empty_transcript(...) or is_backchannel(...))`` with this is a pure
    rename: the same three inputs map to the same two outcomes.
    """
    return verdict is TurnVerdict.FINISHED
