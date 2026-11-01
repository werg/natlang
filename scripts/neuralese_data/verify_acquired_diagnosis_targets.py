#!/usr/bin/env python3
"""Verify every census-declared diagnosis target against retained source labels."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'training/neuralese')]
from natlang_neuralese.common.hashing import canonical_json_sha256_hex
from natlang_neuralese.common.jsonio import open_jsonl_text, write_json_durable
from natlang_neuralese.train.own_output_supervision import policy_objective, graph_weights, REPLY
from scripts.neuralese_data.build_acquired_streaming_successor import bind, checked, read


def verify(screen_path, census_path, policy_path, base, out):
    paths = [Path(p).resolve(strict=True) for p in (screen_path, census_path, policy_path)]
    references = [bind(p) for p in paths]
    screen, census, policy = [read(p) for p in paths]
    objective = policy_objective(policy)
    if (census.get('screen') != references[0] or census.get('ce_policy') != references[2]
            or census.get('objective_sha256') != canonical_json_sha256_hex(objective)):
        raise ValueError('diagnosis census is not bound to the exact screen and effective objective')
    signals_path = checked(census['row_signals'], 'diagnosis census signals')
    expected = {}
    with open_jsonl_text(signals_path) as source:
        for line in source:
            if not line.strip(): continue
            row = json.loads(line)
            if row.get('source_label_diagnosis'):
                if row['id'] in expected: raise ValueError('census repeats a diagnosis identity')
                expected[row['id']] = row
    if len(expected) != census['counts']['source_label_diagnosis_rows']:
        raise ValueError('diagnosis census metadata count differs')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True, trust_remote_code=False)
    graph_path = checked(screen['output'], 'screened diagnosis source')
    verified, seen, scanned = [], set(), 0
    with open_jsonl_text(graph_path) as source:
        for line in source:
            if not line.strip(): continue
            row = json.loads(line); scanned += 1
            if row['id'] not in expected: continue
            if row['id'] in seen or canonical_json_sha256_hex(row) != expected[row['id']]['graph_sha256']:
                raise ValueError('diagnosis graph differs from completed census')
            lineage = row['conversion']['acquired_session']['source_lineage']
            diagnosis = lineage['notes']['source_execution_diagnosis']; meta = diagnosis['source_metadata']
            store = lineage['store']
            if store != expected[row['id']]['source_store']:
                raise ValueError('diagnosis source ownership differs')
            if store == 'mcemri/MAST-Data':
                annotations = {'mast_annotation': meta['mast_annotation']} if 'mast_annotation' in meta else {
                    'round': meta.get('round'), 'annotations': meta['annotations']}
            elif store == 'mascope/mascope-bench':
                annotations = {key: meta[key] for key in ('mast_annotation', 'anomaly_type', 'anomaly_injected', 'note') if key in meta}
            else: raise ValueError('unrecognized diagnosis source ' + store)
            weights, categories = graph_weights(row, objective)
            ids = [token for node in row['nodes'] for token in node['token_ids']]
            text = tokenizer.decode([token for token, weight, category in zip(ids, weights, categories)
                if weight > 0 and category == REPLY and token not in {
                    objective['role_tokens']['think_open'], objective['role_tokens']['think_close']}], skip_special_tokens=True)
            if json.loads(text.strip()) != {'schema': 'natlang.source-execution-diagnosis/1', 'source_annotations': annotations}:
                raise ValueError('effective diagnosis reply changes source annotations: ' + row['id'])
            seen.add(row['id']); verified.append({'id': row['id'], 'source_store': store,
                'target_sha256': hashlib.sha256(text.encode()).hexdigest()})
            if len(seen) == len(expected): break
    if seen != set(expected): raise ValueError('source does not cover every census-declared diagnosis')
    if ([bind(p) for p in paths] != references or bind(signals_path) != census['row_signals']
            or bind(graph_path)['sha256'] != screen['output']['sha256']):
        raise ValueError('diagnosis inputs changed during verification')
    result = {'schema': 'natlang.acquired-source-diagnosis-target-verification/1',
        'screen': references[0], 'census': references[1], 'ce_policy': references[2],
        'verified_rows': len(verified), 'scanned_prefix_rows': scanned, 'rows': verified,
        'builder': bind(Path(__file__)),
        'scope': 'Complete census-declared subset: effective reply JSON equals retained source annotations; not local oracle verification or admission.'}
    write_json_durable(out, result)
    return {key: value for key, value in result.items() if key != 'rows'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('screen', 'census', 'ce-policy', 'base', 'out'): parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.screen, args.census, args.ce_policy, args.base, args.out)), flush=True)


if __name__ == '__main__': main()
