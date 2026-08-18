"""Character-level error rate for comparing transcripts.

This project had no accuracy metric at all before this module. Latency was
measured everywhere -- `pipeline.py` instruments every stage and `storage.py`
persists it -- but "did the recognizer get the words right" was only ever
answered by a human reading the log. That is fine for judging a preset once and
useless for the two questions the experiment tracks ask: *did chunking SenseVoice
make recognition worse*, and *is this TTS voice intelligible*.

CER rather than WER, deliberately. Korean is written without reliable inter-word
spacing, and the checkpoints here disagree about where the spaces go -- SenseVoice
inserts spurious mid-eojeol breaks ("생각 을"), and the streaming Zipformer emits
no inter-eojeol spaces at all (see `stage/asr_stream.py`, which computes its
LocalAgreement over characters for exactly this reason). Word error rate over
tokens split on whitespace would therefore measure the tokenizer's spacing habits
far more than it measures recognition, and would rank a model worse for spacing a
correct sentence differently. Characters are the unit both models actually agree
on.

The S/D/I breakdown is not decoration. The failure this project is chasing --
`streaming-zipformer-ko` returning `''` on real microphone captures -- produces
CER 1.0 composed entirely of *deletions*, which looks nothing like a model that
merely misheard (substitutions). A bare rate would collapse those two into the
same number and hide the diagnosis.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

# Recognizers do not return "" for silence; they return punctuation. A live
# session in this project produced six turns transcribed as '.', '그.', '예.'
# (see docs/FEATURES.md). Punctuation is also not spoken aloud, so scoring it
# would charge a model for a comma it had no way to hear.
_PUNCT_CATEGORIES = {"Pc", "Pd", "Pe", "Pf", "Pi", "Po", "Ps", "Sm", "Sk"}


def normalize_for_cer(text: str, *, drop_space: bool = True, drop_punct: bool = True) -> str:
    """Canonicalize a transcript before scoring.

    NFC first, because Hangul arrives both precomposed (U+AC00 '가') and as
    conjoining jamo (U+1100 U+1161) depending on which library produced it, and
    the two are the same syllable to a reader but different code points to an
    edit-distance function -- comparing them unnormalized reports every syllable
    as an error.
    """
    text = unicodedata.normalize("NFC", text)
    out = []
    for ch in text:
        if ch.isspace():
            if not drop_space:
                out.append(" ")
            continue
        if drop_punct and unicodedata.category(ch) in _PUNCT_CATEGORIES:
            continue
        out.append(ch)
    result = "".join(out)
    if not drop_space:
        # Collapse runs left by dropped punctuation so " a  b " scores as "a b".
        result = " ".join(result.split())
    return result


@dataclass(frozen=True)
class ErrorCounts:
    """Edit operations turning `ref` into `hyp`, plus the reference length."""

    substitutions: int
    deletions: int
    insertions: int
    ref_len: int

    @property
    def total(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def rate(self) -> float:
        """Errors per reference character.

        Can exceed 1.0 -- an insertion-heavy hypothesis (a model that babbles)
        has more errors than the reference has characters. Callers that want a
        bar chart should clamp; callers that want the truth should not.
        """
        if self.ref_len == 0:
            # No reference to be wrong about. Any output at all is pure insertion.
            return 0.0 if self.insertions == 0 else 1.0
        return self.total / self.ref_len


def align_counts(ref: str, hyp: str) -> ErrorCounts:
    """Levenshtein with operation counts, O(len(ref) x len(hyp)) time.

    Backtracking needs the full matrix, so this is not the rolling-two-rows
    version. Transcripts here are single utterances -- tens of characters -- so
    the matrix is trivially small; if that ever changes, the rate alone can be
    had from a rolling implementation and the breakdown dropped.
    """
    n, m = len(ref), len(hyp)
    if n == 0:
        return ErrorCounts(0, 0, m, 0)
    if m == 0:
        return ErrorCounts(0, n, 0, n)

    # d[i][j] = edit distance between ref[:i] and hyp[:j]
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = 1 + min(
                    d[i - 1][j - 1],  # substitute
                    d[i - 1][j],      # delete from ref
                    d[i][j - 1],      # insert into ref
                )

    # Walk back along one optimal path. Ties are broken toward substitution
    # then deletion; any optimal path has the same total, and the split only
    # shifts between equally-valid alignments.
    subs = dels = ins = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] and d[i][j] == d[i - 1][j - 1]:
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + 1:
            subs += 1
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            dels += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return ErrorCounts(subs, dels, ins, n)


def cer(
    ref: str,
    hyp: str,
    *,
    drop_space: bool = True,
    drop_punct: bool = True,
) -> float:
    """Character error rate after normalization. See `normalize_for_cer`."""
    return cer_detail(ref, hyp, drop_space=drop_space, drop_punct=drop_punct).rate


def cer_detail(
    ref: str,
    hyp: str,
    *,
    drop_space: bool = True,
    drop_punct: bool = True,
) -> ErrorCounts:
    """`cer` with the S/D/I breakdown retained."""
    r = normalize_for_cer(ref, drop_space=drop_space, drop_punct=drop_punct)
    h = normalize_for_cer(hyp, drop_space=drop_space, drop_punct=drop_punct)
    return align_counts(r, h)


def is_effectively_empty(text: str) -> bool:
    """True when a transcript carries no recognized speech.

    Mirrors the guard in `turn/backchannel.py:is_empty_transcript`, which uses
    length <= 1 on the normalized text because recognizers signal silence with
    punctuation rather than an empty string. Kept separate from that function on
    purpose: this one scores an experiment, that one gates a live turn, and
    coupling them would mean a scoring tweak could change conversation behaviour.
    """
    return len(normalize_for_cer(text)) == 0
