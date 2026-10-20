"""Explicit source-bound CE mass continuation; no corpus or policy-RL rewriting."""
from __future__ import annotations

from dataclasses import replace, dataclass, field, fields
import json
import math
from pathlib import Path
import re

from ..common.hashing import sha256_file_hex

from ..data.native_graph_schema import NativeGraphExample


@dataclass
class NativeHeldGraphExample(NativeGraphExample):
    evaluation_weights: dict = field(default_factory=dict)
    evaluation_binding: dict = field(default_factory=dict)


def attach_held_measurements(example, weights, binding):
    """Optional evaluation sidecar; raw parser/index identities stay unchanged."""
    if example.split != 'test':
        raise ValueError('held CE measurements require protected test rows')
    return NativeHeldGraphExample(**{item.name:getattr(example,item.name) for item in fields(NativeGraphExample)},
        evaluation_weights=weights, evaluation_binding=binding)


SCHEMA = 'natlang.native-ce-supervision-policy/1'
ENCODINGS = {'ordinary-suffix-raw-v1', 'witnessed-pair-raw-v1'}


def load_policy(binding):
    """Read a bound declaration before restoring/training any parameters."""
    if binding is None:
        return None
    path = Path(binding['path'])
    digest = sha256_file_hex(path)
    if digest != binding.get('sha256') or ('bytes' in binding and path.stat().st_size != binding['bytes']):
        raise ValueError('CE supervision policy SHA-256 mismatch')
    value = json.loads(path.read_text())
    if sha256_file_hex(path) != digest:
        raise ValueError('CE supervision policy changed while reading')
    if value.get('schema') != SCHEMA or not isinstance(value.get('sources'), list) or not value['sources']:
        raise ValueError('CE supervision policy requires explicit source declarations')
    if type(value.get('require_eligible_sources', True)) is not bool:
        raise ValueError('CE policy eligible-source coverage must be explicitly boolean')
    if not isinstance(value.get('held_sources', []), list):
        raise ValueError('CE held sources must be explicit declarations')
    seen = set()
    for source in value['sources'] + value.get('held_sources', []):
        digest = source.get('sha256')
        weight = source.get('context_weight')
        if (not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest)
                or digest in seen or source.get('encoding') not in ENCODINGS
                or type(weight) not in (int, float) or not math.isfinite(weight)
                or not 0 <= weight < 1):
            raise ValueError('CE supervision source needs a unique SHA, exact raw encoding and context coefficient below one')
        seen.add(digest)
    return value


def apply_policy(example, row, *, source_sha256, policy):
    """Decode ONLY a declared immutable source encoding and retain node mass.

    Undeclared eligible sources fail when coverage is required; other sources
    retain their original objective. Already segmented whole trajectories and
    sampled RL are not inferred from magnitudes or reweighted. Raw row
    identities stay unchanged.
    """
    if policy is None:
        return example
    if example.split == 'test':
        return held_measurements(example, row, source_sha256=source_sha256, policy=policy)
    if not isinstance(source_sha256, str) or not re.fullmatch('[0-9a-f]{64}', source_sha256):
        raise ValueError('CE supervision requires an authenticated immutable source SHA')
    conversion = row.get('conversion') or {}
    declarations = [entry for entry in policy['sources'] if entry['sha256'] == source_sha256]
    if not declarations:
        eligible = (conversion.get('role_interpretation') == 'ordinary-text'
                    and 'source_suffix_start' in conversion or bool(row.get('source_pairing')))
        if eligible and not conversion.get('supervision_policy') and policy.get('require_eligible_sources', True):
            raise ValueError('eligible native CE source missing from explicit supervision policy: ' + source_sha256)
        return example
    declaration, = declarations
    return _apply_declaration(example, row, declaration)


def _apply_declaration(example, row, declaration):
    conversion = row.get('conversion') or {}
    coefficient = declaration['context_weight']
    kind = declaration['encoding']
    if conversion.get('supervision_policy'):
        raise ValueError('CE policy cannot reinterpret an already declared trajectory objective')
    nodes = row['nodes']
    if kind == 'ordinary-suffix-raw-v1':
        if (len(nodes) != 1 or conversion.get('role_interpretation') != 'ordinary-text'
                or conversion.get('context_weight') != coefficient
                or type(conversion.get('source_suffix_start')) is not int):
            raise ValueError('ordinary CE continuation requires its authenticated conversion suffix/coefficient')
        suffixes = [conversion['source_suffix_start']]
    else:
        provenance = row.get('source_pairing') or {}
        if (len(nodes) != 2 or [node['name'] for node in nodes] != ['producer', 'consumer']
                or not nodes[0].get('producing') or not nodes[1].get('imports')
                or any(edge['source'] != 'producer' for edge in nodes[1]['imports'])
                or not provenance.get('pair_id') or not provenance.get('payloads')
                or not provenance.get('consumed_occurrences') or not provenance.get('source_ids')):
            raise ValueError('legacy pair CE continuation requires witnessed producer/consumer provenance')
        # This is the explicitly selected producer's exact binary encoding,
        # not an approximate threshold applied to arbitrary graph weights.
        suffixes = [next((index for index, value in enumerate(node['target_weights']) if value == 1.),
                         len(node['token_ids'])) for node in nodes]
    import torch
    from .text_supervision import balanced_position_weights
    effective = []
    for node, suffix in zip(nodes, suffixes):
        raw = node['target_weights']
        if not 0 <= suffix <= len(raw):
            raise ValueError('CE source suffix is outside its exact token coordinates')
        for index, weight in enumerate(raw):
            if weight and weight != (coefficient if index < suffix else 1.):
                raise ValueError('CE source violates its declared exact raw-weight encoding')
        if coefficient == 0 or suffix == len(raw) or not any(raw):
            effective.extend(raw)
            continue
        span = torch.tensor([raw], dtype=torch.float32)
        balanced = balanced_position_weights(span, [suffix], context_weight=coefficient, exclude=span == 0)
        balanced *= span.sum() / balanced.sum()
        effective.extend(balanced[0].tolist())
    return replace(example, target_weights=tuple(effective))


def held_measurements(example, row, *, source_sha256, policy):
    """Attach evaluation-only weights after exact raw row/source verification."""
    if policy is None:
        return example
    if example.split != 'test':
        raise ValueError('held CE measurements require protected test rows')
    declarations = [item for item in policy.get('held_sources', []) if item['sha256'] == source_sha256]
    if not declarations:
        return example
    declaration, = declarations
    weighted = _apply_declaration(example, row, declaration)
    if declaration['encoding'] == 'ordinary-suffix-raw-v1':
        suffixes = [row['conversion']['source_suffix_start']]
    else:
        suffixes = [next((i for i, value in enumerate(node['target_weights']) if value == 1.),
                         len(node['token_ids'])) for node in row['nodes']]
    targets = tuple(float(weight > 0 and i >= suffix)
        for node, suffix in zip(row['nodes'], suffixes)
        for i, weight in enumerate(node['target_weights']))
    return attach_held_measurements(example, {'declared': weighted.target_weights, 'target': targets},
        {'source_sha256': source_sha256, **declaration})
