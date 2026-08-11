"""Turn-taking: deciding when the user started talking, when they stopped, and
whether what they said was a turn at all.

This is the part of a voice agent that has no equivalent in a text chat, and
the part where a cascaded ASR->LLM->TTS design has to make explicit decisions
that an end-to-end duplex speech model makes implicitly. The modules split by
the *question* they answer:

``vad``
    "Is there speech right now, and where does this utterance begin and end?"
    TEN-VAD, frame by frame. Knows only audio energy patterns -- never text.

``detector``
    "Did they actually finish that sentence, or just pause mid-thought?"
    Smart Turn v3, an audio classifier over the whole captured utterance. Used
    to extend the endpoint past a natural pause that pure silence-timing would
    have cut. See ``docs/barge-in-design.md`` for why it is *not* used as a
    backchannel filter, despite that being why it was first added.

``backchannel``
    "Was that a real turn, or just 'mm-hm'?" A lexical check against the ASR
    text, applied after recognition. Cheap and preset-independent, but only
    catches words someone thought to list.

``controller``
    The state machine that composes the three with an audio session and the
    pipeline, and owns the rule that a conversation is a sequence of states,
    not a sequence of blocking calls.

Ordering matters and is easy to get wrong: ``vad`` decides *when to listen*,
``detector`` refines *when to stop listening*, ``backchannel`` decides *whether
to answer*. They are three different questions asked at three different points,
which is why they are three modules rather than one "turn logic" blob.
"""
