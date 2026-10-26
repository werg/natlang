#!/usr/bin/env python3
"""Create an exact, predicate-bound successor for ill-posed RL task rows.

The source packet and all of its original audit/held/protection evidence are
copied intact. Only the active train packet is filtered, by the fixed
source-semantic predicate in rl.task_source; arbitrary exclusion lists are
not accepted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from natlang_neuralese.common.hashing import sha256_file_hex
from natlang_neuralese.rl.task_source import _ill_posed_port_record_continuation, resolve_task_source


def canonical_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n').encode()


def build(source_manifest: Path, output: Path):
    resolved = resolve_task_source(source_manifest)
    source_dir = resolved['manifest'].parent
    output = output.resolve(strict=False)
    if output == source_dir or output.is_relative_to(source_dir):
        raise ValueError('correction output cannot be inside its immutable source directory')
    if output.exists():
        raise FileExistsError(f'output must be new; preserving existing evidence: {output}')
    parent_dir = output / 'parent-source'
    shutil.copytree(source_dir, parent_dir)
    source_manifest_copy = parent_dir / 'task-source-manifest.json'
    if not source_manifest_copy.exists():
        raise ValueError('source task packet has no standard task-source-manifest.json')
    parent_manifest_name = 'parent-source/task-source-manifest.json'

    source_cases_bytes = resolved['cases'].read_bytes()
    source_lines = source_cases_bytes.splitlines(keepends=True)
    kept, excluded = [], []
    count = 0
    for line_no, raw in enumerate(source_lines, 1):
        if not raw.strip():
            continue
        row = json.loads(raw)
        count += 1
        if _ill_posed_port_record_continuation(row):
            items = []
            for name, payload in sorted(row['semantics']['folder_files'].items()):
                if name.startswith('items/') and name.endswith('.json'):
                    item = json.loads(payload)
                    items.append({'path': name,
                                  'request_sha256': hashlib.sha256(item['request'].encode()).hexdigest()})
            excluded.append({
                'line': line_no, 'id': row['id'],
                'row_sha256': hashlib.sha256(json.dumps(row, ensure_ascii=False, sort_keys=True,
                    separators=(',', ':')).encode()).hexdigest(),
                'predicate': 'port-record-next-assistant-turn-unprovided-v1',
                'reason': 'asks for historical next assistant message without supplying that message as evidence',
                'items': items,
            })
        else:
            kept.append(raw)
    output_cases = output / 'cases.jsonl'
    output_cases.write_bytes(b''.join(kept))
    proof = {
        'schema': 'natlang.rl-task-source-correction-proof/1',
        'predicate': 'port-record-next-assistant-turn-unprovided-v1',
        'parent_manifest_sha256': resolved['manifest_sha256'],
        'parent_cases_sha256': resolved['cases_sha256'],
        'parent_rows': count,
        'retained_rows': len(kept),
        'excluded_rows': len(excluded),
        'excluded': excluded,
        'cases_sha256': sha256_file_hex(output_cases),
        'cases_rows': len(kept),
    }
    proof_path = output / 'source-correction-proof.json'
    proof_path.write_bytes(canonical_bytes(proof))
    manifest = {
        'schema': 'natlang.rl-task-source-correction/1',
        'id': resolved['manifest'].stem + '-exact-source-correction-v1',
        'split': json.loads((output / parent_manifest_name).read_text())['split'],
        'parent_manifest': parent_manifest_name,
        'parent_manifest_sha256': sha256_file_hex(output / parent_manifest_name),
        'cases': 'cases.jsonl',
        'cases_sha256': proof['cases_sha256'],
        'rows': len(kept),
        'correction_proof': proof_path.name,
        'correction_proof_sha256': sha256_file_hex(proof_path),
    }
    manifest_path = output / 'task-source-manifest.json'
    manifest_path.write_bytes(canonical_bytes(manifest))
    verified = resolve_task_source(manifest_path)
    return {'output': str(output), 'manifest': str(manifest_path),
            'manifest_sha256': verified['manifest_sha256'], 'rows': verified['rows'],
            'excluded_rows': verified['source_correction_excluded_rows'],
            'cases_sha256': verified['cases_sha256']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source_manifest, args.output), sort_keys=True))


if __name__ == '__main__':
    main()
