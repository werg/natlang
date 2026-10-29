"""Bind one multi-identity scan to the namespace snapshot taken before it."""
from __future__ import annotations
from dataclasses import dataclass
from .retention_snapshot import snapshot,refresh_references


@dataclass(frozen=True)
class ReferenceEvidence:
    roots:tuple[str,...]
    digests:frozenset[str]
    snapshot:dict
    references:dict


def prepare_reference_evidence(roots,digests,scan):
    roots=tuple(map(str,roots));digests=frozenset(digests)
    if not digests:raise ValueError('reference evidence requires identities')
    before=snapshot(roots)
    refs=scan(sorted(digests),roots)
    after,refs=refresh_references(roots,before,refs,sorted(digests),scan,allow_busy=True)
    return ReferenceEvidence(roots,digests,after,refs)


def use_reference_evidence(evidence,roots,digests):
    if not isinstance(evidence,ReferenceEvidence) or tuple(map(str,roots))!=evidence.roots or not set(digests)<=evidence.digests:raise ValueError('reference evidence roots or identities differ')
    return evidence.snapshot,evidence.references
