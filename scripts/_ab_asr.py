#!/usr/bin/env python3
"""Compare ASR engines on the real captures, including the streaming behaviour.

The batch ASR presets are already comparable through scripts/benchmark.py, which
measures latency and leaves accuracy to a human reading transcripts. That is not
enough for the streaming question, which has three parts a latency table cannot
express: does the engine return anything at all on real microphone speech, is the
final text as good as the batch path, and does it actually produce usable
partials before the utterance ends.

The first of those is not hypothetical. `streaming-zipformer-ko` returns the
empty string on real captures while SenseVoice reads the same audio fine
(docs/FEATURES.md, and sherpa-onnx issue #2886). A tool that only reported CER
would have scored that as 1.0 and buried the diagnosis; this one counts empties
separately, because "said nothing" and "said the wrong thing" need different
responses.

## What it reports, and why each column exists

  CER-vs-batch   Character error rate against the batch reference. Not absolute
                 accuracy -- see "ground truth" below.
  empty          Transcripts with no recognizable content. The zipformer failure
                 signature. Compared against the reference's own empty count,
                 since four of our captures are genuinely silent.
  S / D / I      Substitutions, deletions, insertions. An engine returning ''
                 scores CER 1.0 entirely in deletions; one that mishears every
                 character scores 1.0 entirely in substitutions. Same rate,
                 completely different bug.
  first-commit   When the first non-empty prefix was committed, as a fraction of
                 the utterance. This is the streaming payoff, and it is where
                 chunked SenseVoice loses: its floor is min_decode_s + hop_s.
  pre%           How much of the final text was already committed before
                 finalize. 0% means the "streaming" engine gave the pipeline
                 nothing early, whatever its final accuracy.
  retract        Times the committed prefix stopped being a prefix of the next
                 one. LocalAgreement's monotonic guard only stops the prefix
                 from getting *shorter*; a full re-decode can change its reading
                 of audio it already saw ("오늘 산책 코스 추천" became
                 "오늘 산체코 추천해줘."). Any non-zero value here means
                 committed text is not safe to act on.
  amp            Audio decoded / audio in. 1.0 for a true streaming model;
                 chunked re-decoding is quadratic in utterance length and
                 measured 3.0x at the defaults.

## Ground truth

There is none, and the tool says so rather than pretending. The transcripts in
conversations.db came from SenseVoice and several are known wrong -- four are the
documented '.' artifacts and one is the recorded pre-roll corruption of "산책
코스". So the default reference is the *current* batch ASR, which makes this a
regression detector: it answers "is this engine as good as the batch path", not
"is this engine correct".

Absolute CER needs human labels. `scripts/_make_benchmark_set.py` writes
labels.json for exactly that; pass --use-labels to score only the rows a person
has marked verified, and the tool will tell you how many that is.

Usage:
    python scripts/_make_benchmark_set.py --db ../../todak-flux/data/conversations.db
    python scripts/_ab_asr.py
    python scripts/_ab_asr.py --engines chunked-sensevoice zipformer
    python scripts/_ab_asr.py --use-labels
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.nobody_flux import registry  # noqa: E402
from src.nobody_flux.metrics import (  # noqa: E402
    cer_detail,
    is_effectively_empty,
    normalize_for_cer,
)
from src.nobody_flux.paths import PROJECT_ROOT  # noqa: E402

BENCH_DIR = PROJECT_ROOT / "data" / "benchmark_wavs"
MANIFEST = BENCH_DIR / "manifest.json"
LABELS = BENCH_DIR / "labels.json"
FRAME_SAMPLES = 480  # 30ms at 16kHz, the capture path's frame size


@dataclass
class StreamTrace:
    """What the committed prefix did over the course of one utterance."""

    first_commit_s: float | None = None
    retractions: int = 0
    committed_at_end: str = ""
    final: str = ""

    @property
    def prefix_fraction(self) -> float:
        """Share of the final text that was committed before finalize."""
        final = normalize_for_cer(self.final)
        if not final:
            return 0.0
        return len(normalize_for_cer(self.committed_at_end)) / len(final)


def replay(transcriber, audio: np.ndarray, sample_rate: int) -> StreamTrace:
    """Feed an utterance frame by frame, recording what the prefix did.

    Frame-by-frame rather than calling transcribe_array: the point is to observe
    the intermediate state, which the convenience wrapper hides.
    """
    transcriber.reset()
    trace = StreamTrace()
    previous = ""
    for offset in range(0, len(audio), FRAME_SAMPLES):
        transcriber.accept_frame(audio[offset : offset + FRAME_SAMPLES], sample_rate)
        committed = transcriber.committed
        if committed == previous:
            continue
        elapsed = (offset + FRAME_SAMPLES) / sample_rate
        if trace.first_commit_s is None and normalize_for_cer(committed):
            trace.first_commit_s = elapsed
        if previous and not committed.startswith(previous):
            # Not merely growing -- the engine revised text it had already
            # promised. Monotonic-in-length is not monotonic-in-content.
            trace.retractions += 1
        previous = committed
    trace.committed_at_end = previous
    trace.final = transcriber.finalize()
    return trace


@dataclass
class EngineResult:
    engine: str
    errors: int = 0
    ref_chars: int = 0
    subs: int = 0
    dels: int = 0
    ins: int = 0
    empties: int = 0
    scored: int = 0
    first_commits: list[float] = field(default_factory=list)
    prefix_fractions: list[float] = field(default_factory=list)
    retractions: int = 0
    never_committed: int = 0
    wall_s: float = 0.0
    audio_s: float = 0.0
    decoded_s: float = 0.0
    rows: list[tuple[str, str, str]] = field(default_factory=list)

    @property
    def cer(self) -> float:
        return (self.errors / self.ref_chars) if self.ref_chars else 0.0

    @property
    def amplification(self) -> float:
        return (self.decoded_s / self.audio_s) if self.audio_s else 1.0

    @property
    def rtf(self) -> float:
        return (self.wall_s / self.audio_s) if self.audio_s else 0.0


def load_set(use_labels: bool) -> tuple[list[dict], dict]:
    if not MANIFEST.exists():
        raise SystemExit(
            "no %s -- run scripts/_make_benchmark_set.py first "
            "(and pass --db if this is a fresh worktree)" % MANIFEST
        )
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    labels = json.loads(LABELS.read_text(encoding="utf-8")) if LABELS.exists() else {}
    if use_labels:
        keep = [r for r in manifest if labels.get(r["id"], {}).get("verified")]
        if not keep:
            raise SystemExit(
                "--use-labels needs human ground truth, and none of the %d rows in "
                "%s is marked verified. Listen to the wavs, correct 'reference', "
                "set \"verified\": true." % (len(labels), LABELS.name)
            )
        return keep, labels
    return manifest, labels


def evaluate(engine: str, manifest: list[dict], labels: dict, use_labels: bool,
             reference_asr) -> EngineResult:
    transcriber = registry.build_streaming_transcriber(engine=engine)
    result = EngineResult(engine=engine)

    for row in manifest:
        wav = BENCH_DIR / row["wav"]
        audio, sample_rate = sf.read(str(wav), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        if use_labels:
            reference = labels[row["id"]]["reference"]
        else:
            reference = reference_asr.transcribe_file(str(wav))

        start = time.perf_counter()
        trace = replay(transcriber, audio, sample_rate)
        result.wall_s += time.perf_counter() - start
        result.audio_s += row["duration_s"]

        counts = cer_detail(reference, trace.final)
        result.errors += counts.total
        result.ref_chars += counts.ref_len
        result.subs += counts.substitutions
        result.dels += counts.deletions
        result.ins += counts.insertions
        result.scored += 1
        if is_effectively_empty(trace.final):
            result.empties += 1

        if trace.first_commit_s is None:
            result.never_committed += 1
        else:
            result.first_commits.append(trace.first_commit_s)
        result.prefix_fractions.append(trace.prefix_fraction)
        result.retractions += trace.retractions
        result.rows.append((row["id"], reference, trace.final))

    # Cost instrumentation, where the engine bothers to keep it. Deliberately
    # not required: a true streaming engine has nothing to report here.
    decoded = getattr(transcriber, "decoded_samples_total", None)
    if decoded is not None:
        result.decoded_s = decoded / 16_000
    else:
        result.decoded_s = result.audio_s
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--engines",
        nargs="+",
        default=["chunked-sensevoice", "zipformer"],
        help="streaming engines from configs/streaming_asr.yaml",
    )
    parser.add_argument(
        "--reference-asr",
        default=None,
        help="batch ASR preset used as the reference transcript (default: the default preset)",
    )
    parser.add_argument(
        "--use-labels",
        action="store_true",
        help="score against human-verified references in labels.json instead of the "
             "batch ASR, i.e. absolute accuracy rather than regression",
    )
    parser.add_argument("--verbose", action="store_true", help="print every transcript")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest, labels = load_set(args.use_labels)

    reference_asr = None
    if not args.use_labels:
        preset = args.reference_asr or registry.default_preset("asr")
        reference_asr = registry.build_asr(preset)
        print("reference : batch %s  (regression mode -- NOT ground truth)" % preset)
    else:
        print("reference : human labels, %d verified row(s)" % len(manifest))
    real = sum(1 for r in manifest if r["kind"] == "real-capture")
    print("set       : %d wav(s), %d real capture(s)" % (len(manifest), real))
    print("threads   : asr=%s  (NOBODY_CPU_BUDGET honoured)" % registry.stage_threads("asr"))
    print()

    results = []
    for engine in args.engines:
        try:
            result = evaluate(engine, manifest, labels, args.use_labels, reference_asr)
        except Exception as exc:
            print("  SKIP %-20s %s: %s" % (engine, type(exc).__name__, str(exc)[:110]))
            continue
        results.append(result)
        print("  %-20s CER %.3f  empty %d/%d" % (engine, result.cer, result.empties, result.scored))

    if not results:
        print("\nNo engine could be evaluated.")
        return 1

    print()
    print("=" * 100)
    print("%-20s %6s %5s %5s %5s %6s %8s %5s %7s %5s %5s"
          % ("engine", "CER", "S", "D", "I", "empty", "1stCmt", "pre%", "retract", "amp", "rtf"))
    print("-" * 100)
    for r in sorted(results, key=lambda x: (x.empties, x.cer)):
        first = ("%.2fs" % statistics.median(r.first_commits)) if r.first_commits else "never"
        pre = 100.0 * statistics.median(r.prefix_fractions) if r.prefix_fractions else 0.0
        print("%-20s %6.3f %5d %5d %5d %6s %8s %4.0f%% %7d %5.1f %5.2f"
              % (r.engine, r.cer, r.subs, r.dels, r.ins,
                 "%d/%d" % (r.empties, r.scored), first, pre,
                 r.retractions, r.amplification, r.rtf))
    print("-" * 100)

    print("CER is vs the batch reference unless --use-labels: a regression check, not accuracy.")
    print("empty: no recognizable content. Some captures are genuinely silent, so compare")
    print("       engines against each other rather than against zero.")
    print("D >> S means the engine is returning nothing, not mishearing -- a different bug.")
    print("1stCmt/pre%: the streaming payoff. pre% near 0 means partials arrived too late")
    print("       to be useful, whatever the final CER says.")
    print("retract > 0 means committed text was later revised. It is then NOT safe to act on.")
    print("amp: audio decoded / audio in. 1.0 = true streaming; higher = re-decoding cost.")

    for r in results:
        if r.never_committed:
            print()
            print("  %s: %d of %d utterance(s) committed nothing before finalize."
                  % (r.engine, r.never_committed, r.scored))

    if args.verbose:
        for r in results:
            print("\n=== %s ===" % r.engine)
            for rid, reference, got in r.rows:
                flag = "  <<" if normalize_for_cer(reference) != normalize_for_cer(got) else ""
                print("  %-9s ref=%-28s got=%-28s%s" % (rid, reference, got, flag))

    if not args.use_labels:
        print()
        print("For absolute accuracy, verify references in %s and rerun with --use-labels."
              % LABELS.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
