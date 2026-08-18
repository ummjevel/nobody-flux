#!/usr/bin/env python3
"""Compare TTS presets on intelligibility, speed, and number handling.

`scripts/benchmark.py` measures how long TTS takes and deliberately leaves
quality to a human ear, which is right as far as it goes -- but it means a new
TTS preset could halve the latency while being harder to understand and nothing
would catch it. This project had no accuracy metric of any kind before
`src/nobody_flux/metrics.py`; this script is the TTS half of putting that to use.

## How intelligibility is measured

By ASR round-trip: synthesize the text, transcribe the result with SenseVoice,
and compute CER between what we asked for and what came back. SenseVoice is
already a dependency, so this costs no new packages and no human time, and it
measures the thing that actually matters downstream -- whether the words survive.

**This is intelligibility, not naturalness.** A monotone robot that articulates
perfectly scores 0.000 here. Prosody, warmth, and whether anyone wants to listen
to this voice for ten minutes are not in this number and cannot be; that is why
`--keep-wavs` exists and why `docs/FEATURES.md` keeps a "사람이 직접 해야 할 검증"
list. Use this to rule out candidates, never to pick a winner.

## Why numbers are scored separately

SenseVoice applies inverse text normalization: give it a correctly-spoken
"세 시 이십 분" and it writes back "3 시 20 분". Scored as CER against the input
that is a 33% error rate for synthesis that was perfect. Measured on both
sherpa-matcha-ko and supertonic-3-ko -- they got the identical bogus penalty,
which is how the artifact was spotted.

So numeric inputs never enter the aggregate. They are printed with their raw
transcripts under a separate heading, for a human to read, and they come in two
flavours because they answer two different questions:

  hangul-numerals  "세 시 이십 분" -- what persona.py's prompt instruction is
                   supposed to make the LLM produce. Should sound right.
  digit-numerals   "3시 20분" -- what happens when that instruction fails, which
                   it will under quantization on long outputs. No TTS path in
                   this project expands digits, so this is undefined behaviour
                   and the transcript shows what the model actually did with it.

The second is the direct evidence for whether a deterministic Korean number
expander is needed on the TTS input path.

## Synthesis is stochastic, so one measurement is not a measurement

Measured, the hard way: the same preset on the same text produced CER 0.032,
0.043 and 0.074 across three separate runs of this script. That is not ASR
variance -- SenseVoice returns byte-identical transcripts for a given wav, checked
-- it is the TTS. Matcha is a flow-matching model with a `noise_scale`, and
sherpa-onnx seeds it per process, so the audio differs between runs (different
md5 *and* a different sample count for identical input). Within one process it is
reproducible; across processes it is not, and there is no seed parameter exposed
on `OfflineTtsConfig` or `generate()` to pin it.

So this script repeats each preset `--repeat` times and reports the median with
the observed min/max. Read the spread, not the median alone: on the default
8-sentence set the spread for a single preset was ~4 characters out of 94, i.e.
**CER resolution here is about ±0.04**. Two presets inside that band are not
distinguishable, however many decimal places get printed.

That band is wide enough to have misled earlier conclusions in this repo, so it is
worth being blunt: a 10-speaker "ranking" spanning 0.025 to 0.089 was mostly
noise, and "preset A ties preset B at 0.074" was two single samples from
overlapping distributions. RTF is a different story -- it varies by a few percent
and the gaps between these presets are 1.7x and 2.4x, far outside it.

To resolve small CER differences, raise --repeat and add texts; do not read the
existing numbers harder.

Usage:
    python scripts/_ab_tts.py
    python scripts/_ab_tts.py --tts sherpa-matcha-ko supertonic-3-ko
    python scripts/_ab_tts.py --tts supertonic-3-ko --sid-sweep
    python scripts/_ab_tts.py --keep-wavs data/tmp/ttscmp
"""

from __future__ import annotations

import argparse
import statistics
import sys
import tempfile
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.nobody_flux import registry  # noqa: E402
from src.nobody_flux.metrics import cer_detail, is_effectively_empty  # noqa: E402

# Ordinary conversational Korean in the register this project's persona actually
# speaks (반말, short turns). These carry the aggregate CER.
PLAIN_TEXTS = [
    "안녕. 오늘 날씨가 참 좋네.",
    "산책 코스 추천해 줄까?",
    "그건 나도 잘 모르겠는데, 왜?",
    "어제 뭐 했어? 재미있는 일 있었어?",
    "밥은 먹었어? 배고프면 같이 먹자.",
    "그 영화 진짜 재미없었어. 왜 그렇게 인기가 많지?",
    "조금만 기다려 줘. 금방 찾아볼게.",
    "잘 자. 좋은 꿈 꿔.",
]

# Never aggregated -- see the module docstring.
NUMERIC_TEXTS = [
    ("hangul-numerals", "지금 세 시 이십 분이야."),
    ("hangul-numerals", "스물세 살이라고 했잖아."),
    ("digit-numerals", "지금 3시 20분이야."),
    ("digit-numerals", "23살이라고 했잖아."),
    ("digit-numerals", "가격은 12,000원이고 30% 할인 중이야."),
    ("latin", "와이파이 비밀번호는 ABC123이야."),
]


@dataclass
class Result:
    label: str
    cer: float                       # median across repeats
    cers: list[float]                # one per repeat -- the spread is the point
    ref_chars: int
    errors: int
    empties: int
    rtf_median: float
    rtf_max: float
    sample_rate: int
    load_s: float
    numeric: list[tuple[str, str, str]] = field(default_factory=list)


def write_wav(path: Path, samples: np.ndarray, rate: int) -> None:
    """16-bit mono PCM.

    Clipped rather than normalized: a preset that returns samples outside
    [-1, 1] has a gain problem the listener would hear, and quietly rescaling
    here would hide it from both the CER and the human check.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes())


def evaluate(label: str, tts, asr, out_dir: Path, load_s: float, repeat: int = 3) -> Result:
    errors = ref_chars = empties = 0
    rtfs: list[float] = []
    cers: list[float] = []
    rate = 0

    # Repeat the whole set rather than each sentence: a run is the unit that
    # varies, and repeating per sentence would understate the spread by averaging
    # inside it.
    for attempt in range(max(1, repeat)):
        run_err = run_ref = 0
        for i, text in enumerate(PLAIN_TEXTS):
            t0 = time.perf_counter()
            samples, rate = tts.synthesize_audio(text)
            elapsed = time.perf_counter() - t0
            duration = len(samples) / rate if rate else 0.0
            if duration:
                rtfs.append(elapsed / duration)

            wav = out_dir / ("%s__r%d_plain%02d.wav" % (label.replace("=", ""), attempt, i))
            write_wav(wav, samples, rate)
            hyp = asr.transcribe_file(str(wav))
            if is_effectively_empty(hyp):
                # Either the synthesis produced nothing playable or it produced
                # something the recognizer could not read at all. Both are total
                # failures, but they are worth counting apart from the CER because
                # CER 1.0 from one bad utterance and CER 1.0 across the board look
                # identical in an aggregate.
                empties += 1
            counts = cer_detail(text, hyp)
            run_err += counts.total
            run_ref += counts.ref_len

        errors += run_err
        ref_chars += run_ref
        if run_ref:
            cers.append(run_err / run_ref)

    numeric: list[tuple[str, str, str]] = []
    for j, (kind, text) in enumerate(NUMERIC_TEXTS):
        samples, rate = tts.synthesize_audio(text)
        wav = out_dir / ("%s__num%02d.wav" % (label.replace("=", ""), j))
        write_wav(wav, samples, rate)
        numeric.append((kind, text, asr.transcribe_file(str(wav))))

    return Result(
        label=label,
        cer=statistics.median(cers) if cers else 0.0,
        cers=cers,
        ref_chars=ref_chars,
        errors=errors,
        empties=empties,
        rtf_median=statistics.median(rtfs) if rtfs else 0.0,
        rtf_max=max(rtfs) if rtfs else 0.0,
        sample_rate=rate,
        load_s=load_s,
        numeric=numeric,
    )


def build(preset: str, sid: int | None):
    overrides = {} if sid is None else {"speaker_id": sid}
    t0 = time.perf_counter()
    tts = registry.build_tts(preset, **overrides)
    return tts, time.perf_counter() - t0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--tts",
        nargs="+",
        default=None,
        help="TTS presets to compare. Default: every registered preset.",
    )
    p.add_argument("--asr", default=None, help="ASR preset used as the judge (default: the default)")
    p.add_argument(
        "--sid-sweep",
        action="store_true",
        help="for multi-speaker presets, score every speaker id separately -- "
             "speaker choice moved CER more than the choice of model did",
    )
    p.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="how many times to run the whole text set per preset. Synthesis is "
             "stochastic across processes (see module docstring), so 1 is not a "
             "measurement.",
    )
    p.add_argument(
        "--keep-wavs",
        type=Path,
        default=None,
        help="write the synthesized wavs here instead of a temp dir, so they can be listened to",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    presets = args.tts or registry.list_presets("tts")

    out_dir = args.keep_wavs or Path(tempfile.mkdtemp(prefix="ab_tts_"))
    asr_preset = args.asr or registry.default_preset("asr")
    asr = registry.build_asr(asr_preset)
    print("judge ASR : %s" % asr_preset)
    print("wavs      : %s" % out_dir)
    print("threads   : tts=%s asr=%s  (NOBODY_CPU_BUDGET honoured)"
          % (registry.stage_threads("tts"), registry.stage_threads("asr")))
    print()

    results: list[Result] = []
    for preset in presets:
        try:
            tts, load_s = build(preset, None)
        except Exception as exc:
            # A preset that needs a GPU, an isolated venv, or weights this
            # machine does not have should not stop the comparison -- that is
            # the normal state of this table, not an error.
            print("  SKIP %-22s %s: %s" % (preset, type(exc).__name__, str(exc)[:120]))
            continue

        sids = [None]
        if args.sid_sweep:
            n = getattr(getattr(tts, "tts", None), "num_speakers", 1) or 1
            if n > 1:
                sids = list(range(n))

        for sid in sids:
            if sid is None:
                label, inst, l_s = preset, tts, load_s
            else:
                label = "%s sid=%d" % (preset, sid)
                try:
                    inst, l_s = build(preset, sid)
                except Exception as exc:
                    print("  SKIP %-22s %s" % (label, type(exc).__name__))
                    continue
            try:
                r = evaluate(label, inst, asr, out_dir, l_s, args.repeat)
            except Exception as exc:
                print("  FAIL %-22s %s: %s" % (label, type(exc).__name__, str(exc)[:120]))
                continue
            results.append(r)
            print("  %-24s CER %.3f  RTF %.2f" % (r.label, r.cer, r.rtf_median))

    if not results:
        print("\nNo preset could be evaluated.")
        return 1

    print()
    print("=" * 78)
    print("%-24s %6s %13s %6s %6s %7s %6s"
          % ("preset", "CER~", "CER min-max", "rtf~", "rtfMx", "rate", "load"))
    print("-" * 78)
    for r in sorted(results, key=lambda x: (x.cer, x.rtf_median)):
        flag = "  <-- %d empty" % r.empties if r.empties else ""
        spread = ("%.3f-%.3f" % (min(r.cers), max(r.cers))) if r.cers else "-"
        print("%-24s %6.3f %13s %6.2f %6.2f %7d %6.2f%s"
              % (r.label, r.cer, spread, r.rtf_median, r.rtf_max,
                 r.sample_rate, r.load_s, flag))
    print("-" * 78)
    per_run = results[0].ref_chars // max(1, len(results[0].cers))
    print("CER~ is the median of %d run(s) over %d reference chars each; a 0.01 gap is"
          % (len(results[0].cers), per_run))
    print("~%d character(s). Synthesis is stochastic across runs -- if two presets'"
          % max(1, round(per_run * 0.01)))
    print("min-max ranges overlap, they are NOT distinguishable. Trust RTF.")
    print("RTF < 1.0 means faster than real time. rtfMx is the worst sentence,")
    print("which is what bounds TTFA on the first chunk of a reply.")

    print()
    print("=" * 78)
    print("Numbers and Latin -- NOT in the CER above. Read these.")
    print("SenseVoice writes digits for spoken numerals, so CER here would be a lie.")
    print("=" * 78)
    for r in results:
        print("\n%s" % r.label)
        for kind, text, hyp in r.numeric:
            print("  [%-16s] %-38s -> %s" % (kind, text, hyp or "(nothing)"))

    print()
    print("Listen to the wavs before trusting any of this: CER is intelligibility,")
    print("not naturalness. See docs/FEATURES.md for what only a human can check.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
