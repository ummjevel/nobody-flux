"""Turn a stream of LLM text deltas into speakable chunks, so TTS on the first
sentence can start while the rest of the reply is still being generated (see
pipeline.STSPipeline.run_streaming). This is the Phase 1 latency lever: the old
path synthesized the whole reply in one shot, so time-to-first-audio was the
full LLM reply plus the full TTS pass; chunking cuts it to first-sentence-LLM +
first-sentence-TTS.

Pure text, no audio/model deps -- unit-testable in isolation. The rules are
deliberately simple (punctuation + length bounds), not a real Korean sentence
segmenter: a wrong split just means a slightly awkward TTS phrase boundary, not
a correctness bug, so the cost of over-engineering here is higher than the cost
of an occasional clumsy cut.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

_WHITESPACE = re.compile(r"\s+")
# Zero-width joiner + variation selectors -- glue that turns codepoints into
# composite emoji; strip so they don't linger after the emoji itself is gone.
_EMOJI_GLUE = {"‍", "︎", "️"}


def sanitize_for_tts(text: str) -> str:
    """Drop characters a phoneme TTS can't speak (emoji/pictographs and other
    symbol-other codepoints) and collapse whitespace. Small local models emit
    emoji despite persona.py asking them not to (LFM2 especially -- see
    configs/models.yaml); left in, an emoji that lands alone in its own
    streamed chunk makes sherpa's Matcha return zero samples (it can't tokenize
    it), which is an unplayable chunk. Returns "" if nothing speakable remains,
    which the caller treats as "skip this chunk"."""
    kept = []
    for ch in text:
        if ch in _EMOJI_GLUE:
            continue
        # So = "Symbol, other" (emoji, dingbats, pictographs); Cs/Co/Cn =
        # surrogate/private/unassigned. Letters, digits, punctuation, and
        # ordinary math/currency symbols are kept.
        if unicodedata.category(ch) in ("So", "Cs", "Co", "Cn"):
            continue
        kept.append(ch)
    return _WHITESPACE.sub(" ", "".join(kept)).strip()

# Sentence-final marks: cut *after* one of these (keeping it, TTS prosody uses
# it). Covers ASCII + CJK fullwidth forms since a Korean 0.6B model emits both.
PRIMARY = frozenset(".!?…\n。！？")
# Clause-level marks: only used to force an earlier cut once the buffer is
# getting long (max_chars) so a comma-spliced run-on doesn't delay first audio.
SECONDARY = frozenset(",;:、，；·")


@dataclass
class SentenceChunker:
    # Don't emit a chunk shorter than this even at a boundary -- a bare "응."
    # or "어?" would otherwise become its own TTS call, which is more per-chunk
    # overhead than it saves. Set just below a typical short Korean sentence
    # ("밥 먹었어?" is 6) so real short sentences still stream, but 1-2 syllable
    # backchannel doesn't. Measured in characters, not bytes (Korean syllables
    # are one char each here). Tuning knob, like everything timing-related in
    # this repo -- expect to revisit against real replies.
    min_chars: int = 6
    # Force a cut once the buffer reaches this even with no primary boundary --
    # bounds worst-case first-audio latency for a model that rambles without
    # punctuation (LFM2 did exactly this, see configs/models.yaml).
    max_chars: int = 80
    primary: frozenset = PRIMARY
    secondary: frozenset = SECONDARY
    _buf: str = field(default="", init=False)

    def push(self, delta: str) -> list[str]:
        """Feed the next text delta; return zero or more completed chunks.

        A single delta can complete more than one chunk (the model may emit
        several tokens' worth at once), so this drains greedily until no
        further cut is available, leaving the remainder buffered for the next
        push()/flush().
        """
        self._buf += delta
        chunks: list[str] = []
        while True:
            cut = self._find_cut()
            if cut is None:
                break
            chunk = self._buf[:cut].strip()
            self._buf = self._buf[cut:]
            if chunk:
                chunks.append(chunk)
        return chunks

    def flush(self) -> str | None:
        """Return whatever's left (end of stream) as a final chunk, or None if
        only whitespace remains. Resets the buffer so the instance is reusable
        for the next turn."""
        leftover = self._buf.strip()
        self._buf = ""
        return leftover or None

    def _find_cut(self) -> int | None:
        """Index to slice the buffer at (exclusive), or None if no chunk is
        ready yet. Cuts at the first primary boundary once min_chars have
        accumulated; failing that, forces a cut near max_chars, preferring a
        secondary boundary so the split lands at a natural phrase edge."""
        buf = self._buf
        n = len(buf)

        for i, ch in enumerate(buf):
            if ch in self.primary and (i + 1) >= self.min_chars:
                return i + 1

        if n >= self.max_chars:
            # Walk back from the max toward min_chars looking for a clause mark
            # to cut on; if there's none, cut flat at max_chars.
            for i in range(min(n, self.max_chars) - 1, self.min_chars - 1, -1):
                if buf[i] in self.secondary:
                    return i + 1
            return self.max_chars

        return None
