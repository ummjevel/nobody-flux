"""Verify warm_up()'s disk-backed KV prefix cache actually restores state.

This is worth a script rather than a unit test because the failure mode is not
an exception. If the restored KV and llama-cpp-python's token bookkeeping
disagree, generation proceeds from a state that was never evaluated and the
model emits plausible-looking garbage -- so the check that matters is that a
reply produced after a *restored* prefix is identical to one produced after a
normal prefill, greedily decoded so the comparison is meaningful.

Runs each condition in a fresh process (see --phase), because llama.cpp's
in-process cache would otherwise hide whether the disk layer did anything.

Usage:
    python scripts/_verify_kv_prefix.py            # drives the phases
    python scripts/_verify_kv_prefix.py --phase X  # one phase (internal)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# Replies are Korean and this box's console is cp949.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

PROMPT = "오늘 날씨 어때?"


def _build():
    from nobody_flux.stage.llm import NobodyLLMGguf

    return NobodyLLMGguf(max_new_tokens=24)


def _reply_greedy(llm) -> str:
    # temperature=0 so the comparison between the cached and uncached paths is
    # about the KV state, not about sampling luck.
    out = llm._llm.create_completion(
        llm._build_prompt(PROMPT), max_tokens=24, temperature=0.0, stop=llm.stop
    )
    return out["choices"][0]["text"]


def phase(name: str) -> dict:
    llm = _build()
    prefix = llm._static_prefix_tokens()
    path = llm._kv_cache_path(prefix)

    t0 = time.perf_counter()
    llm.warm_up()
    warm_s = time.perf_counter() - t0

    return {
        "phase": name,
        "prefix_tokens": len(prefix),
        "cache_path": path.name,
        "cache_exists_after": path.exists(),
        "cache_bytes": path.stat().st_size if path.exists() else 0,
        "n_tokens_after_warmup": int(llm._llm.n_tokens),
        "warm_up_s": round(warm_s, 3),
        "reply": _reply_greedy(llm),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase")
    args = ap.parse_args()

    if args.phase:
        print("__RESULT__" + json.dumps(phase(args.phase), ensure_ascii=False))
        return 0

    # Driver: cold (no cache) -> warm (cache present) -> stale (key changed).
    llm = _build()
    cache = llm._kv_cache_path(llm._static_prefix_tokens())
    del llm
    if cache.exists():
        cache.unlink()

    results = []
    for name in ("cold", "warm"):
        p = subprocess.run(
            [sys.executable, __file__, "--phase", name],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        line = next((l for l in p.stdout.splitlines() if l.startswith("__RESULT__")), None)
        if line is None:
            print("phase %s produced no result (rc=%d)" % (name, p.returncode))
            print((p.stdout + p.stderr)[-1500:])
            return 1
        results.append(json.loads(line[len("__RESULT__"):]))

    cold, warm = results
    for r in results:
        print("%-5s prefix=%-5d n_tokens=%-5d warm_up=%5.2fs cache=%s (%d bytes)"
              % (r["phase"], r["prefix_tokens"], r["n_tokens_after_warmup"],
                 r["warm_up_s"], r["cache_exists_after"], r["cache_bytes"]))
    print()
    print("cold reply: %r" % cold["reply"])
    print("warm reply: %r" % warm["reply"])
    print()

    checks = [
        ("cold run created the snapshot", cold["cache_exists_after"] and cold["cache_bytes"] > 0),
        ("both runs agree on the prefix length", cold["prefix_tokens"] == warm["prefix_tokens"]),
        ("both runs agree on the cache key", cold["cache_path"] == warm["cache_path"]),
        ("warm run restored n_tokens == prefix length",
         warm["n_tokens_after_warmup"] == warm["prefix_tokens"]),
        ("cold run also left n_tokens >= prefix length",
         cold["n_tokens_after_warmup"] >= cold["prefix_tokens"]),
        ("restored state yields the SAME greedy reply", cold["reply"] == warm["reply"]),
        ("warm warm_up() is not slower than cold", warm["warm_up_s"] <= cold["warm_up_s"]),
    ]
    ok = True
    for label, passed in checks:
        print("  [%s] %s" % ("PASS" if passed else "FAIL", label))
        ok &= bool(passed)

    print()
    print("saved %.2fs of warm_up (%.2fs -> %.2fs)"
          % (cold["warm_up_s"] - warm["warm_up_s"], cold["warm_up_s"], warm["warm_up_s"]))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
