"""Shared, versioned meanings for source decision labels.

The JSON contract is the sole source of taxonomy text. Callers keep the
original source row intact and use the returned criteria as a derived,
model-visible contract.
"""
from __future__ import annotations

import json
from pathlib import Path
from functools import lru_cache
from typing import Any

from natlang_neuralese.common.hashing import sha256_hex


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / 'training' / 'decision_task_contracts.json'


@lru_cache(maxsize=4)
def load_contracts(path: Path = CONTRACT_PATH) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    document = json.loads(raw)
    if document.get('schema') != 'natlang.decision-task-contracts/1':
        raise ValueError(f'{path}: unsupported decision-task-contracts schema')
    if not isinstance(document.get('contracts'), dict):
        raise ValueError(f'{path}: contracts must be an object')
    return document, sha256_hex(raw)


def _validate_criteria(criteria: Any, labels: list[str], where: str) -> dict[str, str]:
    if not isinstance(criteria, dict) or set(criteria) != set(labels):
        raise ValueError(f'{where}: criteria keys must exactly match the declared labels')
    if any(not isinstance(criteria[label], str) or not criteria[label].strip() for label in labels):
        raise ValueError(f'{where}: every criterion must be a nonempty string')
    return criteria


def criteria_for(*, family: str, source: str, kind: str, labels: list[str],
                 explicit: Any = None, where: str = 'decision case') -> tuple[dict[str, str], dict[str, Any] | None]:
    """Return explicit criteria when authored; otherwise derive a pinned family contract."""
    if not isinstance(labels, list) or not labels or any(not isinstance(label, str) or not label for label in labels):
        raise ValueError(f'{where}: labels must be nonempty strings')
    document, digest = load_contracts()
    contract = document['contracts'].get(family)
    if explicit is not None:
        return _validate_criteria(explicit, labels, where), {
            'source': 'explicit_source_criteria', 'contract_path': str(CONTRACT_PATH),
            'contract_sha256': digest, 'family_contract_applied': False,
        }
    if contract is None:
        return {label: label for label in labels}, None
    if contract.get('source') != source or contract.get('kind') != kind:
        raise ValueError(f'{where}: family source/kind does not match its shared task contract')
    if contract.get('label_order') != labels:
        raise ValueError(f'{where}: labels/order do not match the shared task contract')
    criteria = _validate_criteria(contract.get('criteria'), labels, where)
    return criteria, {
        'source': 'shared_family_task_contract', 'contract_path': str(CONTRACT_PATH),
        'contract_sha256': digest, 'family_contract_applied': True,
        'taxonomy_provenance': contract.get('provenance'),
    }
