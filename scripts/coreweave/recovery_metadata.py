#!/usr/bin/env python3
"""Recover only byte-identical metadata from checkpoint trees and JSON captures.

Serialization candidates are accepted exclusively when the previously bound
SHA-256 matches. No source population, quality label or training history is
inferred, and no recovered artifact replaces an original path.
"""
from __future__ import annotations

import hashlib
import itertools
import copy
import json
from pathlib import Path

try:
    from .recovery_inputs import durable_json
except ImportError:
    from recovery_inputs import durable_json


def encoded_match(value, expected):
    for sort_keys, indent, newline in itertools.product((True, False), (2, 1, None), ('', '\n')):
        raw = (json.dumps(value, indent=indent, sort_keys=sort_keys) + newline).encode()
        if hashlib.sha256(raw).hexdigest() == expected:
            return raw
    return None


def source_manifest_bytes(receipt, model):
    children = receipt.get('sources', [])
    if not children:
        return None
    priorities = [(None,) * len(children), (4,) * len(children), (0,) * len(children)]
    if len(children) <= 3:
        priorities.extend(itertools.product((None, 0, 4), repeat=len(children)))
    for kinds in ('native', 'pairs-by-name'):
        for values in priorities:
            sources = []
            for child, priority in zip(children, values):
                pair = kinds == 'pairs-by-name' and not child.get('sources') and 'pairs' in child['path']
                value = {'kind': 'semantic-pairs' if pair else 'native-graphs',
                         'path': child['path'], 'sha256': child['sha256']}
                if pair:
                    value['model'] = model
                if priority is not None:
                    value['training_priority'] = priority
                sources.append(value)
            raw = encoded_match({'schema': 'natlang.native-graph-sources/1', 'sources': sources},
                                receipt['sha256'])
            if raw is not None:
                return raw
    return None


def publish_exact(directory, binding, raw, provenance):
    if hashlib.sha256(raw).hexdigest() != binding['sha256']:
        raise ValueError('recovered metadata differs from original binding')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / (binding['sha256'] + '.json')
    if target.exists():
        if target.read_bytes() != raw:
            raise ValueError('existing metadata recovery differs')
    else:
        # Atomic JSON serialization is unsuitable here: retain the exact bytes
        # that matched the historical digest, including original whitespace.
        import os
        pending = target.with_suffix('.pending')
        with pending.open('xb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        pending.replace(target)
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    return {'original': {'path': binding['path'], 'sha256': binding['sha256']},
            'local': str(target), 'bytes': len(raw), 'provenance': provenance,
            'exact_original_bytes': True, 'training_admission': False}


def recover_sources(metadata, model, directory):
    results, seen = [], set()

    def walk(receipt):
        key = (receipt['path'], receipt['sha256'])
        if key in seen:
            return
        seen.add(key)
        if receipt.get('sources'):
            raw = source_manifest_bytes(receipt, model)
            if raw is not None:
                results.append(publish_exact(directory, receipt, raw, 'saved checkpoint source receipt'))
            for child in receipt['sources']:
                walk(child)

    for receipt in metadata['training']['source_receipts']:
        walk(receipt)
    return results


def recover_capture(value, directory):
    """Recover complete values only; summarized/truncated captures never match."""
    results = []
    if isinstance(value, dict) and isinstance(value.get('stdout'), str):
        try:
            value = json.loads(value['stdout'])
        except ValueError:
            return results
    if not isinstance(value, dict):
        return results
    files = value.get('files', {})
    if not isinstance(files, dict):
        return results
    for item in files.values():
        if not isinstance(item, dict) or not isinstance(item.get('binding'), dict) or 'value' not in item:
            continue
        binding = item['binding']
        raw = encoded_match(item['value'], binding['sha256'])
        if raw is not None:
            results.append(publish_exact(directory, binding, raw, 'full bound value in local capture'))
    return results


def appended_recipe_bytes(prior_recipe, metadata, binding_prefix):
    """Replay the source-only append's JSON assembly, then require exact SHA."""
    options = metadata['training']['options']
    binding = json.loads(options['recipe_binding'])
    train = metadata['training']['source_receipts'][0]
    history = options.get('source_classification_history', [])
    if not train.get('sources') or not history:
        return None
    recipe = copy.deepcopy(prior_recipe)
    stage = next(s for s in recipe['stages'] if s['id'] == 'native_full_sft')
    changes = {
        'native_graphs': {'path': train['path'], 'sha256': train['sha256']},
        'continued_native_graphs': {k: train['sources'][0][k] for k in ('path', 'sha256')},
        'source_classification_migration': {'path': options['source_classification_migration'],
                                          'sha256': history[-1]['proof_sha256']},
        'ce_supervision_policy': options['ce_supervision_policy_binding']}
    for name, reference in changes.items():
        key = binding_prefix + name
        if key in recipe['input_bindings']:
            return None
        recipe['input_bindings'][key] = {k: reference[k] for k in ('path', 'sha256')}
        stage['inputs'][name] = key
    used = {name for s in recipe['stages'] for name in s.get('inputs', {}).values()}
    recipe['input_bindings'] = {name: ref for name, ref in recipe['input_bindings'].items() if name in used}
    return encoded_match(recipe, binding['sha256'])
