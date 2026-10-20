#!/usr/bin/env python3
"""Declare balanced CE for exact currently admitted native source containers.

This creates an objective declaration, never corpus admission or row credits.
Use the same source manifests the two production recipes actually consume.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from natlang_neuralese.common.hashing import sha256_file_hex_cached as sha256_file_hex
from natlang_neuralese.common.jsonio import write_json_durable
from natlang_neuralese.data.native_sources import _jsonl
from natlang_neuralese.train.native_supervision import SCHEMA, load_policy
from natlang_neuralese.train.recipe import native_source_dependencies


def build(sources, *, legacy_pair_context_weight, held_sources=(), expected_split='train'):
    leaves, roots = {}, []
    for source in sources:
        path = Path(source).resolve(strict=True)
        roots.append({'path':str(path),'sha256':sha256_file_hex(path)})
        dependencies = native_source_dependencies(path)
        for binding in dependencies or [roots[-1]]:
            child = Path(binding['path'])
            if child.suffix != '.json':
                leaves[str(child)] = binding
    declarations, skipped = {}, []
    for binding in leaves.values():
        path = Path(binding['path'])
        if sha256_file_hex(path) != binding['sha256']:
            raise ValueError('CE declaration source changed after inventory validation')
        row = next(_jsonl(path), None)
        if row is None:
            raise ValueError('CE declaration source is empty: ' + str(path))
        if row.get('schema') == 'natlang.native-semantic-pair/1':
            graphs = [view['graph'] for view in row['models'].values()]
        elif row.get('schema') == 'natlang.neuralese-native-graph/1':
            graphs = [row]
        else:
            raise ValueError('CE declaration input is not an admitted native graph/pair source')
        if row.get('split') != expected_split or any(graph.get('split') != expected_split for graph in graphs):
            raise ValueError('CE declaration source has wrong explicit split: ' + str(path))
        encodings = set()
        for graph in graphs:
            conversion = graph.get('conversion') or {}
            if conversion.get('supervision_policy'):
                encodings.add(('already-declared', None))
            elif conversion.get('role_interpretation') == 'ordinary-text' and 'source_suffix_start' in conversion:
                encodings.add(('ordinary-suffix-raw-v1', conversion.get('context_weight')))
            elif graph.get('source_pairing'):
                encodings.add(('witnessed-pair-raw-v1', legacy_pair_context_weight))
            else:
                raise ValueError('CE source lacks authenticated target intent: ' + str(path))
        if len(encodings) != 1:
            raise ValueError('CE source contains incompatible model-view supervision declarations')
        encoding, weight = encodings.pop()
        if encoding == 'already-declared':
            skipped.append({**binding, 'reason':'existing trajectory supervision retained exactly'})
            continue
        declaration = {'sha256':binding['sha256'],'encoding':encoding,'context_weight':weight}
        if declaration['sha256'] in declarations and declarations[declaration['sha256']] != declaration:
            raise ValueError('CE source digest has conflicting objective declarations')
        declarations[declaration['sha256']] = declaration
    value = {'schema':SCHEMA,'require_eligible_sources':True,
        'sources':[declarations[key] for key in sorted(declarations)],
        'inventory':roots,'retained_sources':skipped,
        'scope':'First-row source-intent classification; every consumed row validates the exact declared encoding.',
        'history':'Original graph bytes, IDs, row credits, node mass, optimizer/RNG and raw held CE retained.'}
    if held_sources:
        held = build(held_sources, legacy_pair_context_weight=legacy_pair_context_weight, expected_split='test')
        if set(item['sha256'] for item in value['sources']) & set(item['sha256'] for item in held['sources']):
            raise ValueError('training and held CE source declarations must be disjoint')
        value.update(held_sources=held['sources'], held_inventory=held['inventory'],
                     retained_held_sources=held['retained_sources'])
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', action='append', type=Path, required=True)
    parser.add_argument('--legacy-pair-context-weight', type=float, required=True,
        help='explicit original witnessed_pair_graph coefficient, never estimated from weights')
    parser.add_argument('--held-source', action='append', type=Path, default=[],
        help='exact protected test source; evaluation declaration only, never training exposure')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    value = build(args.source, legacy_pair_context_weight=args.legacy_pair_context_weight, held_sources=args.held_source)
    if args.out.exists():
        if json.loads(args.out.read_text()) != value:
            raise ValueError('existing CE policy differs; write a new explicit continuation artifact')
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_json_durable(args.out, value)
    binding = {'path':str(args.out.resolve()),'sha256':sha256_file_hex(args.out)}
    load_policy(binding)
    print(json.dumps({'policy':binding,'declared_sources':len(value['sources']),
                      'retained_sources':len(value['retained_sources'])}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
