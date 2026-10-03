"""Law-consistency hooks (S3 §6.8, S0 §4.2). Disabled as objectives in S3.

The real operators (`read`, `map`, `combine`, `split`, `zip`) arrive in S5. These hooks fix
the measurement interface now so the harness is ready: each law is measured through `read`
on concrete cases, never as vector equality. The operators are passed in as callables, so
tests use stand-ins.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

Read = Callable[[Any], Any]


@dataclass
class LawResult:
    law: str
    cases: int
    agreement: float  # share of cases where both sides read out equal


def _agreement(pairs) -> float:
    pairs = list(pairs)
    return sum(1 for a, b in pairs if a == b) / max(1, len(pairs))


def map_identity(values, read: Read, map_: Callable) -> LawResult:
    """read(map(v, x => x)) ≈ read(v)."""
    pairs = [(read(map_(v, lambda x: x)), read(v)) for v in values]
    return LawResult("map_identity", len(pairs), _agreement(pairs))


def read_map_commutation(values, read: Read, map_: Callable, f: Callable) -> LawResult:
    """read(map(v, f)) ≈ f(read(v))."""
    pairs = [(read(map_(v, f)), f(read(v))) for v in values]
    return LawResult("read_map_commutation", len(pairs), _agreement(pairs))


def map_fusion(values, read: Read, map_: Callable, f: Callable, g: Callable) -> LawResult:
    """map(map(v, g), f) ≈ map(v, f ∘ g)."""
    pairs = [(read(map_(map_(v, g), f)), read(map_(v, lambda x: f(g(x))))) for v in values]
    return LawResult("map_fusion", len(pairs), _agreement(pairs))


def combine_associativity(triples, read: Read, combine: Callable) -> LawResult:
    pairs = [(read(combine(combine(a, b), c)), read(combine(a, combine(b, c)))) for a, b, c in triples]
    return LawResult("combine_associativity", len(pairs), _agreement(pairs))


def combine_identity(values, read: Read, combine: Callable, empty: Callable) -> LawResult:
    pairs = [(read(combine(v, empty())), read(v)) for v in values]
    return LawResult("combine_identity", len(pairs), _agreement(pairs))


def split_zip(pairs_in, read: Read, zip_: Callable, split: Callable) -> LawResult:
    pairs = []
    for a, b in pairs_in:
        left, right = split(zip_(a, b))
        pairs.append(((read(left), read(right)), (read(a), read(b))))
    return LawResult("split_zip", len(pairs), _agreement(pairs))
