#!/usr/bin/env python3
"""Verify source-review claims against saved IR before changing admission policy.

This checks identity, not whether a source annotation is semantically correct.
Relative artifact paths are resolved against --root, never the audit directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(audit: Path, root: Path) -> dict:
    data = json.loads(audit.read_text())
    cases = data.get('new_cases')
    if not isinstance(cases, list) or not cases:
        raise ValueError('Audit must contain a nonempty new_cases list with artifact and source_id.')
    rows = []
    for case in cases:
        path = Path(case['artifact'])
        if not path.is_absolute():
            path = root / path
        path = path.resolve()
        saved = json.loads(path.read_text())
        ir = saved['task']['program_ir']
        ids = ir.get('source_ids', [])
        errors = []
        if ids != [case['source_id']]:
            errors.append('source_id_does_not_match_saved_ir')
        if 'program_id' in case and case['program_id'] != ir['id']:
            errors.append('program_id_does_not_match_saved_ir')
        if 'question' in case:
            semantics = ir['semantics']
            prompt = semantics['files'][semantics['root']]
            if case['question'] not in prompt.splitlines():
                errors.append('question_is_not_an_exact_saved_prompt_line')
        external_id = ir.get('external_source', {}).get('source_id')
        if external_id is not None and ids != [external_id]:
            errors.append('saved_external_source_identity_conflict')
        rows.append(dict(artifact=str(path), artifact_sha256=digest(path),
                         reported_source_id=case['source_id'], actual_source_ids=ids,
                         actual_program_id=ir['id'], errors=errors))
    return dict(version='natlang.source_audit_identity/1', audit=str(audit),
                audit_sha256=digest(audit), cases=len(rows),
                mismatched_cases=sum(bool(row['errors']) for row in rows), rows=rows,
                semantic_adjudication_performed=False, training_data_promoted=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audit', type=Path)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = verify(args.audit.resolve(), args.root.resolve())
    encoded = json.dumps(result, indent=2) + '\n'
    if args.output:
        # Immutable evidence: choose a new name for a corrected audit.
        with args.output.open('x') as stream:
            stream.write(encoded)
    print(json.dumps({key: result[key] for key in ('cases', 'mismatched_cases', 'audit_sha256')}))
    return 1 if result['mismatched_cases'] else 0


if __name__ == '__main__':
    sys.exit(main())
