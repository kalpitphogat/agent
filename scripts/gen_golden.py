"""
Record the detection output of the current working tree into a golden file.

The golden file is the contract that lets the scoring stages be optimised
safely: any change that alters a score, a matched rule ID or an evidence
snippet for any corpus text will fail `tests/test_equivalence.py`.

Usage
-----
    python scripts/gen_golden.py            # write tests/data/golden_scores.json
    python scripts/gen_golden.py --check    # verify the tree still matches

Regenerate the file only when a detection change is *intended*; the diff
then shows exactly which inputs changed verdict, and belongs in the PR.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# The golden file records detection logic only, so the optional ML stage is
# switched off - it is non-deterministic across machines and GPU/CPU builds.
os.environ["AF_DISABLE_ML"] = "true"

GOLDEN_PATH = ROOT / "tests" / "data" / "golden_scores.json"


def compute_records() -> list[dict]:
    """Score every corpus text through both deterministic stages."""
    from tests.corpus import build_corpus
    from agentforensics.classifier import _check_h05, _run_heuristics
    from agentforensics.semantic import instruction_boundary_score

    records: list[dict] = []
    for text in build_corpus():
        h_score, h_rules, h_evidence = _run_heuristics(text)
        ib_score, ib_rules, ib_evidence = instruction_boundary_score(text)
        records.append({
            "sha": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
            "len": len(text),
            "h_score": h_score,
            "h_rules": h_rules,
            "h_evidence": h_evidence,
            "h05": _check_h05(text),
            "ib_score": ib_score,
            "ib_rules": ib_rules,
            "ib_evidence": ib_evidence,
        })
    return records


def digest(records: list[dict]) -> str:
    blob = json.dumps(records, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="compare against the stored golden file instead of rewriting it")
    args = ap.parse_args()

    records = compute_records()
    payload = {"corpus_size": len(records), "digest": digest(records), "records": records}

    if args.check:
        if not GOLDEN_PATH.exists():
            print(f"FAIL: no golden file at {GOLDEN_PATH}")
            return 1
        stored = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
        if stored["digest"] == payload["digest"]:
            print(f"OK: {len(records)} corpus texts match the golden file "
                  f"(digest {payload['digest'][:16]}).")
            return 0
        changed = [
            (i, s, p) for i, (s, p) in enumerate(zip(stored["records"], records)) if s != p
        ]
        print(f"FAIL: detection output changed for {len(changed)} of {len(records)} texts.")
        for i, s, p in changed[:5]:
            print(f"  [{i}] sha={s['sha']}\n      was {s}\n      now {p}")
        return 1

    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    size_kb = GOLDEN_PATH.stat().st_size / 1024
    print(f"Wrote {GOLDEN_PATH.relative_to(ROOT)} - {len(records)} texts, "
          f"{size_kb:.0f} KB, digest {payload['digest'][:16]}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
