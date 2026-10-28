#!/usr/bin/env python3
"""Create an exact, predicate-bound successor for ill-posed RL task rows.

The successor references its registered source packet by relative path and
manifest SHA-256 (which binds all of the parent's audit/held/protection
evidence), so the parent stays intact in place and its bytes are not
duplicated. Only the active train packet is filtered, by the fixed
source-semantic predicate in rl.task_source; arbitrary exclusion lists are
not accepted.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from natlang_neuralese.common.hashing import sha256_file_hex
from natlang_neuralese.common.jsonio import canonical_json_bytes
from natlang_neuralese.rl.task_source import (PORT_CONTINUATION_PREDICATE, _ill_posed_port_record_continuation,
                                              correction_exclusion, resolve_task_source)


def build(source_manifest: Path, output: Path, *, correction_id: str | None = None):
    resolved = resolve_task_source(source_manifest, as_correction_parent=True)
    source_dir = resolved['manifest'].parent
    output = output.resolve(strict=False)
    if output == source_dir or output.is_relative_to(source_dir):
        raise ValueError('correction output cannot be inside its immutable source directory')
    if output.exists():
        raise FileExistsError(f'output must be new; preserving existing evidence: {output}')
    kept, excluded = [], []
    count = 0
    for line_no, raw in enumerate(resolved['cases'].read_bytes().splitlines(keepends=True), 1):
        if not raw.strip():
            continue
        row = json.loads(raw)
        count += 1
        if _ill_posed_port_record_continuation(row):
            excluded.append(correction_exclusion(line_no, row))
        else:
            kept.append(raw)
    if not excluded:
        raise ValueError('source has no ill-posed rows; it needs no correction successor')
    output.mkdir(parents=True)
    output_cases = output / 'cases.jsonl'
    output_cases.write_bytes(b''.join(kept))
    proof = {
        'schema': 'natlang.rl-task-source-correction-proof/1',
        'predicate': PORT_CONTINUATION_PREDICATE,
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
    proof_path.write_bytes(canonical_json_bytes(proof) + b'\n')
    manifest = {
        'schema': 'natlang.rl-task-source-correction/1',
        'id': correction_id or f'{source_dir.name}-source-correction-v1',
        'split': json.loads(resolved['manifest'].read_text())['split'],
        'parent_manifest': os.path.relpath(resolved['manifest'], output),
        'parent_manifest_sha256': resolved['manifest_sha256'],
        'cases': 'cases.jsonl',
        'cases_sha256': proof['cases_sha256'],
        'rows': len(kept),
        'correction_proof': proof_path.name,
        'correction_proof_sha256': sha256_file_hex(proof_path),
    }
    manifest_path = output / 'task-source-manifest.json'
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b'\n')
    verified = resolve_task_source(manifest_path)
    return {'output': str(output), 'manifest': str(manifest_path),
            'manifest_sha256': verified['manifest_sha256'], 'rows': verified['rows'],
            'excluded_rows': verified['source_correction_excluded_rows'],
            'cases_sha256': verified['cases_sha256']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--id', dest='correction_id')
    args = parser.parse_args()
    print(json.dumps(build(args.source_manifest, args.output, correction_id=args.correction_id), sort_keys=True))


if __name__ == '__main__':
    main()
