#!/usr/bin/env python3
"""Stream an acquired graph census under the actual production loss objective.

Raw positive positions are not supervised-token counts. This audits effective
roles, emits bounded per-row metadata and keeps source annotations distinct from
verified execution. It neither admits sources nor changes their raw graphs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'training/neuralese'))
from natlang_neuralese.common.hashing import canonical_json_sha256_hex, sha256_file_hex
from natlang_neuralese.common.jsonio import open_jsonl_text, write_json_durable
from natlang_neuralese.train.own_output_supervision import CATEGORIES, graph_weights, policy_objective


def bind(path):
    return {'path': str(path.resolve(strict=True)), 'bytes': path.stat().st_size,
            'sha256': sha256_file_hex(path)}


def row_signal(row, objective):
    weights, categories = graph_weights(row, objective)
    mass, tokens = Counter(), Counter()
    for weight, category in zip(weights, categories):
        name = CATEGORIES[category]
        mass[name] += weight
        tokens[name] += 1
    if mass['context'] or mass['assistant_context']:
        raise ValueError('acquired source has effective context supervision')
    acquired = row.get('conversion', {}).get('acquired_session') or {}
    lineage = acquired.get('source_lineage') or {}
    notes = lineage.get('notes') or {}
    diagnosis = notes.get('source_execution_diagnosis')
    if diagnosis and (diagnosis.get('failed_actions_positive_targets') is not False
                      or diagnosis.get('source_labels_unchanged') is not True):
        raise ValueError('diagnosis lacks explicit non-imitation and original-label custody')
    return {'id': row['id'], 'graph_sha256': canonical_json_sha256_hex(row),
            'source_store': lineage.get('store'), 'source_groups': row.get('source_groups', []),
            'source_row_id': row.get('conversion', {}).get('source_row_id'),
            'source_curated': acquired.get('admission_class'),
            'oracle_verified': acquired.get('oracle_verified') is True,
            'native_execution_verified': acquired.get('native_execution_verified') is True,
            'source_label_diagnosis': bool(diagnosis),
            'forward_tokens': len(weights), 'effective_positive_tokens': sum(w > 0 for w in weights),
            'raw_positive_tokens': sum(w > 0 for n in row['nodes'] for w in n['target_weights']),
            'category_tokens': dict(tokens), 'category_mass': dict(mass),
            'eligible': bool(sum(weights))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--screen', type=Path, required=True)
    parser.add_argument('--ce-policy', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    screen_ref, policy_ref = bind(args.screen), bind(args.ce_policy)
    screen = json.loads(args.screen.read_text())
    if screen.get('schema') != 'natlang.acquired-current-protection-screen/1':
        raise ValueError('not a protected acquired-source screen')
    graph = Path(screen['output']['path'])
    if bind(graph) != {k: screen['output'][k] for k in ('path', 'bytes', 'sha256')}:
        raise ValueError('screened source bytes changed')
    objective = policy_objective(json.loads(args.ce_policy.read_text()))
    rows, counts, masses, stores, unknown, seen = 0, Counter(), Counter(), Counter(), Counter(), set()
    metadata = args.out / 'row-signals.jsonl'
    with open_jsonl_text(graph) as source, metadata.open('x') as target:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get('split') != 'train' or row['id'] in seen:
                raise ValueError('source split or unique graph identity changed')
            seen.add(row['id'])
            signal = row_signal(row, objective)
            target.write(json.dumps(signal, sort_keys=True) + '\n')
            rows += 1
            stores[signal['source_store'] or '<unknown>'] += 1
            if not signal['source_store']:
                unknown['source_store'] += 1
            for key in ('forward_tokens', 'raw_positive_tokens', 'effective_positive_tokens'):
                counts[key] += signal[key]
            counts['zero_effective_rows'] += not signal['eligible']
            counts['source_label_diagnosis_rows'] += signal['source_label_diagnosis']
            counts['oracle_verified_rows'] += signal['oracle_verified']
            masses.update(signal['category_mass'])
            if rows % 128 == 0:
                print(json.dumps({'rows': rows, 'expected': screen['output']['rows'], 'counts': dict(counts)}), flush=True)
    if rows != screen['output']['rows'] or bind(args.screen) != screen_ref or bind(args.ce_policy) != policy_ref:
        raise ValueError('source count or audit controls changed during census')
    result = {'schema': 'natlang.acquired-effective-learning-census/1',
              'status': 'census complete; source admission and behavioral qualification remain separate',
              'screen': screen_ref, 'ce_policy': policy_ref, 'builder': bind(Path(__file__)),
              'objective_sha256': canonical_json_sha256_hex(objective),
              'row_signals': bind(metadata), 'rows': rows, 'counts': dict(counts),
              'category_mass': dict(masses), 'source_stores': dict(stores),
              'unknown': dict(unknown), 'training_admission': False}
    write_json_durable(args.out / 'receipt.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
