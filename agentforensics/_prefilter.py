"""
Literal pre-filtering for the regex detection stages.

Profiling the Stage 1 / Stage 3 hot path showed `re.Pattern.search` dominating
the per-document cost: every rule in the set is run against every external
message, even though the overwhelming majority of real traffic cannot match
any of them.  Regex scanning is O(len(text)) per rule, so a clean document
pays for all 18 rules in full.

Each rule therefore declares one or more **guard literals**.  A guard is a
lowercase substring that a match is *required* to contain - it is a necessary
condition derived from the pattern, not a heuristic approximation.  If none of
a rule's guards appear in the document, the pattern provably cannot match and
the regex is skipped.  `str.__contains__` is a C-level substring search and is
far cheaper than running the compiled pattern.

Two properties keep this safe:

* Guards are only ever *necessary* conditions, so skipping a rule can never
  suppress a match that the regex would have found.  Guards are allowed to be
  loose - a guard that lets non-matching documents through only costs time.
* Guard tests run against `str.lower()`, which reproduces `re.IGNORECASE`
  exactly for ASCII text but not for the full Unicode case-folding the regex
  engine applies (`İ`, `ſ`, `K` and friends).  Non-ASCII
  documents therefore bypass the pre-filter entirely and run every rule.

`tests/test_equivalence.py` asserts that enabling and disabling the pre-filter
produces byte-identical results across the whole corpus, so a guard that is
mistakenly *not* a necessary condition fails the build.
"""
from __future__ import annotations

# Flipped to False by the equivalence test to force the unoptimised path.
PREFILTER_ENABLED = True


def prefilter_key(text: str) -> str | None:
    """
    Return a lowercased copy of `text` for guard testing, or None when guards
    must not be applied (pre-filter disabled, or non-ASCII text whose case
    folding `str.lower()` cannot faithfully reproduce).
    """
    if not PREFILTER_ENABLED or not text.isascii():
        return None
    return text.lower()


def skip_rule(guards: tuple[str, ...], key: str | None) -> bool:
    """True when `guards` proves the rule cannot match and may be skipped."""
    if key is None or not guards:
        return False
    for guard in guards:
        if guard in key:
            return False
    return True
