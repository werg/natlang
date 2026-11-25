"""Policy identity for on-policy purposes: the effective policy (the weights), not one file's bytes.

A published policy's checkpoint can be rebuilt (aei1 2026-11-25: retention deleted r0010's 4dec7ca9 checkpoint, the
rebuild from r0009 + r0010/sft/factors has the same ``effective_policy_sha256`` f005e7f6 and a new file sha cde55352).
Episodes collected under the old file sha are samples of the same weights. A lineage entry therefore carries
``equivalent_sha256``: file shas of the same effective policy, taken from its ``published.json`` ``rebuilt`` record
(``previous_checkpoint_sha256`` when its ``effective_policy_sha256`` equals the publication's). Everything that selects
collections or attests episodes by a policy sha matches the entry's ``sha256`` and its equivalents.

``NATLANG_EI_POLICY_EQUIVALENTS=0`` opts out (only the file sha names a policy).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ENV = 'NATLANG_EI_POLICY_EQUIVALENTS'


def enabled() -> bool:
    return os.environ.get(ENV, '1') != '0'


def published_equivalents(published: dict) -> list[str]:
    """File shas a ``natlang.published-policy`` record names as the same weights as its checkpoint: a ``rebuilt``
    record's ``previous_checkpoint_sha256`` (one record or a list) whose ``effective_policy_sha256`` equals the
    publication's (``merge.effective_policy_sha256``; the rebuilt record's own when the merge lacks one)."""
    current = published.get('checkpoint_sha256')
    effective = (published.get('merge') or {}).get('effective_policy_sha256')
    rebuilt = published.get('rebuilt') or []
    out = []
    for rec in (rebuilt if isinstance(rebuilt, list) else [rebuilt]):
        if not isinstance(rec, dict):
            continue
        prev, eff = rec.get('previous_checkpoint_sha256'), rec.get('effective_policy_sha256')
        if prev and prev != current and eff and (effective is None or eff == effective) and prev not in out:
            out.append(prev)
    return out


def published_path(entry: dict) -> Path | None:
    checkpoint = entry.get('checkpoint')
    return Path(checkpoint).parent / 'published.json' if checkpoint else None


def entry_equivalents(entry: dict) -> list[str]:
    """``entry``'s recorded ``equivalent_sha256`` plus those its ``published.json`` names (none with the opt-out)."""
    if not enabled():
        return []
    out = [s for s in entry.get('equivalent_sha256') or () if s != entry.get('sha256')]
    path = published_path(entry)
    try:
        published = json.loads(path.read_text()) if path is not None and path.is_file() else None
    except (OSError, ValueError):
        published = None
    if published and published.get('checkpoint_sha256') == entry.get('sha256'):
        out += [s for s in published_equivalents(published) if s not in out]
    return out


def entry_shas(entry: dict | None) -> list[str]:
    """Every file sha naming ``entry``'s weights: its ``sha256`` first, then its equivalents."""
    if not entry or not entry.get('sha256'):
        return []
    return [entry['sha256'], *entry_equivalents(entry)]


def with_equivalents(entry: dict) -> dict:
    """``entry`` with ``equivalent_sha256`` filled in (absent when it has none)."""
    eq = entry_equivalents(entry)
    out = {k: v for k, v in entry.items() if k != 'equivalent_sha256'}
    return {**out, 'equivalent_sha256': eq} if eq else out


def alias_map(entries) -> dict[str, str]:
    """{equivalent sha: canonical sha} over ``entries``."""
    return {alias: e['sha256'] for e in entries if e for alias in entry_shas(e)[1:]}
