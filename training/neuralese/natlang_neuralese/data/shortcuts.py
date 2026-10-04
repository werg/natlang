"""Shortcut checks for supervised decisions (stop heads, block lengths).

In the first S3 pilot every span was 16 tokens, so the stop head learned "stop after 16" and nothing flagged it.
These checks make such shortcuts visible before and during training:

- `length_profile` summarises a length distribution (distinct values, the most common value's share, entropy).
- `count_hazard` is the best predictor of the stop decision that sees only the count of vectors written so far:
  P(stop at c | not stopped before c), estimated from the lengths. Its BCE and accuracy on the same decisions say
  how much of the stop task a counter already solves. A stop head that does not beat it has learned nothing about
  content; a count baseline near zero BCE means the data leaves the head nothing else to learn.
- `audit_lengths` combines both with flags, which the trainer writes to `shortcuts.json` and logs.

The flags describe the data; whether a flagged phase is acceptable is a judgement the run report states, so the
trainer warns rather than refusing unless asked to.
"""

from __future__ import annotations

import math
from collections import Counter


def length_profile(lengths: list[int]) -> dict:
    counts = Counter(int(n) for n in lengths)
    total = sum(counts.values())
    if not total:
        return {"n": 0}
    top, top_count = counts.most_common(1)[0]
    entropy = -sum(c / total * math.log2(c / total) for c in counts.values())
    return {"n": total, "distinct": len(counts), "most_common": top, "most_common_share": top_count / total,
            "entropy_bits": entropy, "mean": sum(n * c for n, c in counts.items()) / total,
            "min": min(counts), "max": max(counts)}


def count_hazard(lengths: list[int]) -> dict[int, float]:
    """P(stop at count c | reached c) for c = 1..max, from the lengths themselves."""
    counts = Counter(int(n) for n in lengths)
    remaining, hazard = len(lengths), {}
    for c in range(1, max(counts, default=0) + 1):
        if remaining <= 0:
            break
        hazard[c] = counts.get(c, 0) / remaining
        remaining -= counts.get(c, 0)
    return hazard


def count_baseline(lengths: list[int], hazard: dict[int, float] | None = None, eps: float = 1e-6) -> dict:
    """BCE and accuracy of the count-only predictor over every stop decision of `lengths` (continue before the
    length, stop at it). With `hazard` from other data, this scores that fixed predictor on these lengths."""
    hazard = hazard if hazard is not None else count_hazard(lengths)
    total = correct = stops = caught = 0
    loss = 0.0
    for length in lengths:
        for c in range(1, int(length) + 1):
            p = min(1 - eps, max(eps, hazard.get(c, 1.0)))
            stop = c == length
            loss -= math.log(p if stop else 1 - p)
            correct += (p > 0.5) == stop
            total += 1
            stops += stop
            caught += stop and p > 0.5
    # Most decisions are "continue", so accuracy is high for any data; the share of stops a counter gets right is
    # what shows whether the count alone determines where blocks end.
    return {"decisions": total, "bce": loss / total if total else None, "accuracy": correct / total if total else None,
            "stop_recall": caught / stops if stops else None}


def audit_lengths(lengths: list[int], where: str) -> dict:
    profile = length_profile(lengths)
    baseline = count_baseline(lengths)
    flags = []
    if profile.get("n") and profile["most_common_share"] >= 0.9:
        flags.append(f"{where}: {profile['most_common_share']:.0%} of lengths are {profile['most_common']}; "
                     "a stop head can learn the count instead of the content")
    if baseline["stop_recall"] is not None and baseline["stop_recall"] >= 0.5:
        flags.append(f"{where}: counting alone places {baseline['stop_recall']:.0%} of block ends "
                     f"(count-only BCE {baseline['bce']:.3f})")
    return {"where": where, "profile": profile, "count_baseline": baseline, "flags": flags}
