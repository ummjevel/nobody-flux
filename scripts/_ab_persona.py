#!/usr/bin/env python3
"""Compare LLM presets on persona adherence and turn latency.

scripts/benchmark.py already measures per-stage latency across preset
combinations, and deliberately leaves quality to a human reading transcripts.
That works when the question is "is this fast enough". It does not work for the
question this script exists for: *does the model do what the persona asks*.

The persona's rules are specific and checkable -- 반말, no emoji, no parroting,
numbers written as Hangul -- and the 0.6B default broke three of them in a
single live session. Checking them mechanically turns "the replies feel off"
into a number that can be compared across models and across prompt changes.

Rule compliance is necessary and not sufficient. This is a companion, not an
assistant: the thing that decides whether it is worth building is whether
someone wants to talk to it *again*, and a model can break no rules while being
dull enough that nobody does. The companion-chatbot literature finds enjoyment,
not task success, is the leading reason people keep using one -- so alongside
the rule checks there are three cheap mechanical proxies for whether a reply
keeps a conversation alive.

What it measures, per preset:

  위반         how many replies broke at least one rule, out of runs x inputs.
               Sampling is stochastic, so each input runs several times and the
               failures are counted rather than judged from one sample.
  되묻기       fraction of replies that hand the turn back -- a question, an
               invitation to continue. A companion that only answers makes the
               user do all the work of keeping the conversation going.
  판박이       distinct replies as a fraction of all replies. The live session
               that started this work had one model emit the same sentence
               four turns running; a model that says the same thing regardless
               of input is finished as a conversational partner even if every
               sentence is individually fine.
  길이         median reply length in characters. Both directions are failures
               here: one-word replies are work to talk to, and a paragraph is
               worse, because every character is spoken aloud and the user
               cannot skim it.
  ttfb / total first-token and full-reply latency. Persona adherence is
               worthless if the model is too slow to hold a conversation, and
               these two trade against each other directly as models get
               bigger. Measured on CPU, because CPU is what the CM4 target has.

None of these say a model is *good company* -- that needs a person talking to
it. They say when it is definitely not, which is what a cheap automated pass
should do.

The inputs are the turns that actually failed in a live session, plus a couple
that exercise rules those did not reach. This is a regression set, not a
benchmark suite: it is small, it is specific, and every entry is there because
something went wrong on it.

    .venv-win\\Scripts\\python.exe scripts\\_ab_persona.py
    .venv-win\\Scripts\\python.exe scripts\\_ab_persona.py --presets qwen3-0.6b-gguf exaone-2.4b-gguf
    .venv-win\\Scripts\\python.exe scripts\\_ab_persona.py --runs 5 --verbose
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.nobody_flux import registry

# Every one of these came from a real failure, except the last two which cover
# rules the failing turns happened not to reach.
INPUTS = [
    "누구세요?",           # drew a 존댓말 reply -- the model mirrored the user's register
    "산책 코스 추천해줘",  # drew the request echoed straight back
    "내일 날씨 어때?",     # drew a confident invention, then an emoji
    "오늘 좀 피곤하다",    # not a question; drew 존댓말 sympathy
    "너 몇 살이야?",       # numbers, which the rules require spelled in Hangul
    "요즘 허리가 계속 아픈데 어떡하지?",  # health -- rules say defer to a professional
]

# 요/니다 endings, anchored to the end of a sentence so that a noun which merely
# contains 요 (요리, 요즘) does not count.
POLITE = re.compile(r"(요|니다|세요|십시오|습니다)\s*[.!?~]*\s*$")
EMOJI = re.compile("[\U0001f300-\U0001faff☀-➿]")
DIGITS = re.compile(r"[0-9]")
LATIN = re.compile(r"[A-Za-z]{2,}")


def hands_turn_back(reply: str) -> bool:
    """Does this reply give the user something to answer?

    A question mark, or one of the endings Korean uses to ask without one
    ("...어?" written flat, "...지", "...나"). Crude on purpose -- it is a rate
    compared across models, not a judgement of any single reply.
    """
    if "?" in reply:
        return True
    tail = reply.rstrip().rstrip(".!~ ")
    return tail.endswith(("어", "야", "지", "나", "까", "니", "래"))


def violations(user_text: str, reply: str) -> list[str]:
    """Which persona rules this reply broke. Empty list means clean."""
    found = []
    for sentence in filter(None, (s.strip() for s in re.split(r"[.!?\n]", reply))):
        if POLITE.search(sentence):
            found.append("존댓말")
            break
    if EMOJI.search(reply):
        found.append("이모지")
    if DIGITS.search(reply) or LATIN.search(reply):
        # The rules require these spelled out, because the TTS mispronounces
        # them -- this is a pronunciation defect, not a style preference.
        found.append("숫자/로마자")
    stripped = user_text.strip().rstrip("?.!")
    if len(stripped) >= 6 and stripped in reply:
        found.append("앵무새")
    return found


def evaluate(preset: str, runs: int, verbose: bool) -> dict:
    started = time.perf_counter()
    model = registry.build_llm(preset)
    load_s = time.perf_counter() - started

    bad = total = 0
    ttfb: list[float] = []
    totals: list[float] = []
    by_rule: dict[str, int] = {}
    replies: list[str] = []
    invites = 0

    for text in INPUTS:
        for _ in range(runs):
            model.reset()
            t0 = time.perf_counter()
            first = None
            pieces = []
            for piece in model.reply_stream(text):
                if first is None and piece.strip():
                    first = time.perf_counter() - t0
                pieces.append(piece)
            elapsed = time.perf_counter() - t0
            reply = "".join(pieces).strip()

            ttfb.append(first if first is not None else elapsed)
            totals.append(elapsed)
            replies.append(reply)
            invites += bool(hands_turn_back(reply))
            problems = violations(text, reply)
            for rule in problems:
                by_rule[rule] = by_rule.get(rule, 0) + 1
            bad += bool(problems)
            total += 1
            if verbose:
                flag = ",".join(problems) if problems else "ok"
                print(f"    [{flag:<18}] {text}  ->  {reply[:72]}")

    def median(values: list[float]) -> float:
        ordered = sorted(values)
        return ordered[len(ordered) // 2] if ordered else 0.0

    return {
        "preset": preset,
        "violations": bad,
        "total": total,
        "by_rule": by_rule,
        "invite_rate": invites / max(total, 1),
        # Distinct replies over all replies. Low means the model is not really
        # responding to what was said.
        "distinct_rate": len({r for r in replies if r}) / max(len(replies), 1),
        "median_len": int(median([float(len(r)) for r in replies])),
        "ttfb_s": median(ttfb),
        "total_s": median(totals),
        "load_s": load_s,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--presets", nargs="+", default=None,
                        help="LLM presets to compare (default: every preset in models.yaml)")
    parser.add_argument("--runs", type=int, default=3,
                        help="samples per input (default: 3)")
    parser.add_argument("--verbose", action="store_true", help="print every reply")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    presets = args.presets or registry.list_presets("llm")

    results = []
    for preset in presets:
        print(f"\n=== {preset} ===")
        try:
            result = evaluate(preset, args.runs, args.verbose)
        except Exception as exc:
            # One unavailable model must not cost the whole comparison -- these
            # runs take minutes each.
            print(f"  SKIPPED: {type(exc).__name__}: {str(exc)[:160]}")
            continue
        results.append(result)
        rules = ", ".join(f"{k} {v}" for k, v in sorted(result["by_rule"].items())) or "-"
        print(
            f"  위반 {result['violations']}/{result['total']}   되묻기 {result['invite_rate']:.0%}   "
            f"판박이 {result['distinct_rate']:.0%} distinct   {result['median_len']}자   "
            f"turn {result['total_s']:.2f}s   ({rules})"
        )

    if not results:
        print("\nNo preset could be evaluated.")
        return 1

    print(
        f"\n{'preset':<22}{'위반':>9}{'되묻기':>8}{'판박이':>8}{'길이':>7}{'turn':>8}  주요 위반"
    )
    print("-" * 88)
    for r in sorted(results, key=lambda r: (r["violations"] / max(r["total"], 1), r["total_s"])):
        rules = ", ".join(f"{k} {v}" for k, v in sorted(r["by_rule"].items(), key=lambda kv: -kv[1]))
        print(
            f"{r['preset']:<22}{r['violations']:>3}/{r['total']:<5}"
            f"{r['invite_rate']:>7.0%}{r['distinct_rate']:>8.0%}{r['median_len']:>6}자"
            f"{r['total_s']:>7.2f}s  {rules[:28]}"
        )
    print(
        "\nCPU 기준 (CM4 타깃과 같은 조건). 위반이 적고 되묻기가 높고 판박이가 100%에 가까울수록 좋다."
        "\n길이는 낮을수록 좋은 게 아니라 30~60자 근처가 적당하다 -- 전부 음성으로 나가기 때문."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
