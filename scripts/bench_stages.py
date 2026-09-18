"""
Measure the cost of the deterministic detection stages (Stage 1 + Stage 3).

Runs the same documents through the scoring path twice - once with the
literal pre-filter enabled, once with it disabled - and reports the per
document cost of each.  Because `tests/test_equivalence.py` proves the two
paths produce identical output, the difference is pure overhead removed.

The ML stage (Stage 2) is excluded: it is optional, dominates the total when
present, and its cost is unaffected by this work.

Usage
-----
    python scripts/bench_stages.py
    python scripts/bench_stages.py --docs 5000 --repeat 5 --attack-rate 0.2
"""
from __future__ import annotations

import argparse
import os
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

os.environ["AF_DISABLE_ML"] = "true"

from agentforensics import _prefilter, classifier           # noqa: E402
from agentforensics.classifier import _run_heuristics       # noqa: E402
from agentforensics.semantic import instruction_boundary_score  # noqa: E402
from corpus import RULE_TRIGGERS, _WORDS                    # noqa: E402


def make_traffic(n_docs: int, attack_rate: float, seed: int = 11) -> list[str]:
    """
    Build a document mix resembling real agent traffic: mostly benign fetched
    content of varying length, with a minority carrying an injection payload.
    """
    rng = random.Random(seed)
    docs: list[str] = []
    for _ in range(n_docs):
        body = " ".join(rng.choice(_WORDS) for _ in range(rng.randint(40, 400)))
        if rng.random() < attack_rate:
            payload = rng.choice(RULE_TRIGGERS)
            cut = rng.randint(0, len(body))
            body = f"{body[:cut]} {payload} {body[cut:]}"
        docs.append(body)
    return docs


def legacy_check_h05(text: str) -> bool:
    """
    The pre-optimisation H05 implementation, kept here only so the benchmark
    can price the change.  `tests/test_equivalence.py` holds the copy that
    asserts the current implementation still agrees with it.
    """
    if not text:
        return False
    alpha_chars = [c for c in text if c.isalpha()]
    if len(alpha_chars) < 20:
        return False
    non_latin_count = sum(1 for c in alpha_chars if ord(c) > 0x024F)
    ratio = non_latin_count / len(alpha_chars)
    return 0.05 <= ratio <= 0.95


def time_pass(docs: list[str], repeat: int) -> float:
    """Return the best per-document time in microseconds."""
    runs = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        for text in docs:
            _run_heuristics(text)
            instruction_boundary_score(text)
        runs.append(time.perf_counter() - t0)
    return min(runs) / len(docs) * 1e6


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--docs", type=int, default=1000)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--attack-rate", type=float, default=0.10,
                    help="fraction of documents carrying an injection payload")
    args = ap.parse_args()

    docs = make_traffic(args.docs, args.attack_rate)
    mean_len = statistics.mean(len(d) for d in docs)

    print(f"{args.docs} documents, mean length {mean_len:.0f} chars, "
          f"{args.attack_rate:.0%} carrying a payload, "
          f"best of {args.repeat} runs\n")

    # Warm up import-time and branch-prediction effects before timing.
    time_pass(docs[:50], 1)

    # `_run_heuristics` resolves _check_h05 as a module global at call time,
    # so swapping it here reproduces the original code path exactly.
    fast_h05 = classifier._check_h05

    # Arm 1: both optimisations off - the original implementation.
    classifier._check_h05 = legacy_check_h05
    _prefilter.PREFILTER_ENABLED = False
    baseline = time_pass(docs, args.repeat)

    # Arm 2: H05 short-circuit only.
    classifier._check_h05 = fast_h05
    h05_only = time_pass(docs, args.repeat)

    # Arm 3: both optimisations on - what ships.
    _prefilter.PREFILTER_ENABLED = True
    current = time_pass(docs, args.repeat)

    print(f"  original (both off)      : {baseline:8.1f} us/doc")
    print(f"  + H05 ASCII short-circuit: {h05_only:8.1f} us/doc  "
          f"({baseline / h05_only:.2f}x)")
    print(f"  + literal pre-filter     : {current:8.1f} us/doc  "
          f"({h05_only / current:.2f}x)")
    print(f"\n  overall                  : {baseline / current:8.2f}x "
          f"({(1 - current / baseline) * 100:.1f}% of the cost removed)")
    print("\nOutput is identical in all three arms - see tests/test_equivalence.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
