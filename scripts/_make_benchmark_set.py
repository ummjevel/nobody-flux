#!/usr/bin/env python3
"""Assemble data/benchmark_wavs/ from real captured turns.

`scripts/benchmark.py` has always wanted a `--wav-dir` of "a handful of
representative test utterances" and that directory has never existed, so the
script raises FileNotFoundError on a fresh checkout and nobody has run it. The
utterances it wants, though, already exist: every `talk.py` session writes the
captured turn audio under `data/sessions/<id>/`, and `data/conversations.db`
records the transcript, the preset, and the per-stage latency for each one.

So this script does not create test data -- it promotes data the project already
produced by being used. That matters for the streaming-ASR experiment: the wavs
here are exactly the microphone captures on which `streaming-zipformer-ko`
returns the empty string while SenseVoice reads them fine (docs/FEATURES.md).
A synthetic or studio-clean set would not reproduce the bug at all.

## What SenseVoice said is not what was said

The transcripts in the database came out of SenseVoice, and several are known to
be wrong. Four of the fifteen are the documented '.' silence artifacts, and
'오늘 산체코 추천해줘.' is the recorded pre-roll corruption of "오늘 산책 코스
추천해줘". Treating that column as ground truth would bake those errors into
every future measurement and quietly reward a model for reproducing them.

The set is therefore written as two files:

  manifest.json  generated, overwritten on every run. Audio properties plus
                 `sensevoice_baseline` -- what the shipped batch recognizer
                 produced at the time, useful as a *comparison* point.
  labels.json    human ground truth, `verified` false until a person has
                 listened. NEVER overwritten once it exists; new entries are
                 merged in and existing ones left alone, because the transcripts
                 are the one part of this that cannot be regenerated.

Consumers should compare against the baseline for regression detection, which
needs no labels, and report absolute accuracy only over verified rows.

Both files and the audio stay under `data/`, which is gitignored -- these are
personal voice recordings and their transcripts, and they are not committed.

Usage:
    python scripts/_make_benchmark_set.py            # build/refresh
    python scripts/_make_benchmark_set.py --list     # show what is there now

## Running this inside a git worktree

`data/` is deliberately per-worktree -- `storage.py` resolves it off PROJECT_ROOT
so two concurrent experiments cannot write the same SQLite file. The consequence
is that a fresh worktree has no `conversations.db`, and this script would find
nothing but the clean control. Point it at the checkout that has the sessions:

    python scripts/_make_benchmark_set.py --db ../../todak-flux/data/conversations.db

The audio needs no equivalent flag: `user_wav_path` is stored absolute, so the
rows already point at wherever the capture was written.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.nobody_flux.metrics import is_effectively_empty  # noqa: E402
from src.nobody_flux.paths import PROJECT_ROOT  # noqa: E402
from src.nobody_flux.storage import DEFAULT_DB_PATH  # noqa: E402

OUT_DIR = PROJECT_ROOT / "data" / "benchmark_wavs"
MANIFEST = OUT_DIR / "manifest.json"
LABELS = OUT_DIR / "labels.json"

# SenseVoice ships these with its release, so they are present on any machine
# that completed setup and are git-tracked -- the same reasoning
# `scripts/_smoke_turn.py` uses for picking ko.wav over a committed asset.
CLEAN_CONTROL = PROJECT_ROOT / "models" / "sense-voice" / "test_wavs" / "ko.wav"

NOTE_UNVERIFIED = "SenseVoice output, unverified -- listen and correct."
NOTE_UNVERIFIED_EMPTY = (
    "SenseVoice returned nothing usable here -- this is a documented ASR failure "
    "case. Transcribe what was actually said."
)


def wav_props(path: Path) -> dict:
    """Duration, rate and channel count, read from the header only.

    `soundfile` would do this too but pulls in libsndfile for a job the stdlib
    already does, and this script has to run before any model is loaded.
    """
    with wave.open(str(path), "rb") as w:
        frames, rate, channels = w.getnframes(), w.getframerate(), w.getnchannels()
    return {
        "duration_s": round(frames / rate, 3) if rate else None,
        "sample_rate": rate,
        "channels": channels,
    }


def collect_from_db(db_path: Path) -> list[dict]:
    if not db_path.exists():
        return []
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            """SELECT session_id, turn_index, user_text, user_wav_path,
                      asr_preset, asr_ms
               FROM turns
               WHERE user_wav_path IS NOT NULL
               ORDER BY session_id, turn_index"""
        ).fetchall()
    finally:
        con.close()

    out = []
    for r in rows:
        src = Path(r["user_wav_path"])
        if not src.exists():
            # Sessions get pruned by hand sometimes; a row without its audio is
            # not an error, just nothing we can measure.
            continue
        out.append(
            {
                "id": "s%dt%d" % (r["session_id"], r["turn_index"]),
                "source": "session %d turn %d" % (r["session_id"], r["turn_index"]),
                "kind": "real-capture",
                "src": src,
                "sensevoice_baseline": r["user_text"] or "",
                "baseline_preset": r["asr_preset"],
                "baseline_asr_ms": r["asr_ms"],
            }
        )
    return out


def build(force_relink: bool, db_path: Path) -> tuple[list[dict], dict, int]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    entries = collect_from_db(db_path)

    if CLEAN_CONTROL.exists():
        entries.append(
            {
                "id": "clean-ko",
                "source": "models/sense-voice/test_wavs/ko.wav (SenseVoice release)",
                "kind": "clean-control",
                "src": CLEAN_CONTROL,
                "sensevoice_baseline": "",
                "baseline_preset": None,
                "baseline_asr_ms": None,
            }
        )

    manifest = []
    for e in entries:
        dst = OUT_DIR / (e["id"] + ".wav")
        if force_relink and dst.exists():
            dst.unlink()
        if not dst.exists():
            # Hardlink so the set costs nothing and stays byte-identical to the
            # capture; fall back to a copy across volumes or on a filesystem
            # that refuses links.
            try:
                dst.hardlink_to(e["src"])
            except (OSError, NotImplementedError):
                dst.write_bytes(e["src"].read_bytes())
        row = {
            "id": e["id"],
            "wav": dst.name,
            "kind": e["kind"],
            "source": e["source"],
            "sensevoice_baseline": e["sensevoice_baseline"],
            "baseline_preset": e["baseline_preset"],
            "baseline_asr_ms": e["baseline_asr_ms"],
            "baseline_is_empty": is_effectively_empty(e["sensevoice_baseline"]),
        }
        row.update(wav_props(dst))
        manifest.append(row)

    MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # Merge, never clobber. Human transcripts are the one irreplaceable artifact
    # in this directory.
    labels = json.loads(LABELS.read_text(encoding="utf-8")) if LABELS.exists() else {}
    added = 0
    for row in manifest:
        if row["id"] in labels:
            continue
        labels[row["id"]] = {
            "reference": row["sensevoice_baseline"],
            "verified": False,
            "note": NOTE_UNVERIFIED_EMPTY if row["baseline_is_empty"] else NOTE_UNVERIFIED,
        }
        added += 1
    LABELS.write_text(
        json.dumps(labels, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest, labels, added


def show(manifest: list[dict], labels: dict) -> None:
    header = "%-9s %-13s %6s %6s %3s %5s  %s" % (
        "id", "kind", "dur", "rate", "ch", "ref?", "baseline",
    )
    print(header)
    print("-" * 88)
    for row in manifest:
        lab = labels.get(row["id"], {})
        mark = "ok" if lab.get("verified") else "--"
        base = row["sensevoice_baseline"] or "(none)"
        # Only a real capture can *fail* -- the clean control was never run
        # through talk.py, so it has no historical baseline by construction and
        # flagging it as an ASR failure would be a lie.
        if row["kind"] == "real-capture" and row["baseline_is_empty"]:
            base += "   << ASR RETURNED NOTHING"
        print(
            "%-9s %-13s %6s %6s %3s %5s  %s"
            % (
                row["id"], row["kind"], row["duration_s"],
                row["sample_rate"], row["channels"], mark, base,
            )
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--list", action="store_true", help="show the current set and exit")
    p.add_argument(
        "--force-relink",
        action="store_true",
        help="recreate the hardlinks (use after moving or re-recording a capture)",
    )
    p.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="conversations.db to harvest captures from. Defaults to this checkout's, "
             "which in a fresh worktree is empty -- point it at the main checkout.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.list:
        if not MANIFEST.exists():
            print("no manifest at %s -- run without --list first" % MANIFEST)
            return 1
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        labels = json.loads(LABELS.read_text(encoding="utf-8")) if LABELS.exists() else {}
        show(manifest, labels)
        return 0

    db = args.db.resolve()
    if not db.exists():
        print("no database at %s" % db)
        print("A fresh worktree has an empty data/ by design. Pass --db pointing at")
        print("the checkout that holds the sessions, e.g.")
        print("  --db ../../todak-flux/data/conversations.db")
        print()

    manifest, labels, added = build(args.force_relink, db)
    show(manifest, labels)

    total = len(labels)
    verified = sum(1 for v in labels.values() if v.get("verified"))
    empties = sum(
        1 for r in manifest if r["kind"] == "real-capture" and r["baseline_is_empty"]
    )
    real = sum(1 for r in manifest if r["kind"] == "real-capture")

    print()
    print("  %d wavs (%d real captures, %d control) -> %s"
          % (len(manifest), real, len(manifest) - real, OUT_DIR))
    line = "  labels: %d/%d verified" % (verified, total)
    if added:
        line += " (%d newly added)" % added
    print(line)
    if empties:
        print("  %d baseline transcript(s) empty/punctuation-only -- these are the" % empties)
        print("  documented ASR failures. Keep them: they are the regression cases.")
    if verified < total:
        print()
        print("  Absolute CER needs ground truth. Edit %s: fix each" % LABELS.name)
        print('  "reference" by listening to the wav, then set "verified": true.')
        print("  Until then, compare against the SenseVoice baseline only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
