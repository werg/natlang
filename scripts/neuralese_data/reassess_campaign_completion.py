#!/usr/bin/env python3
"""Reindex archived completion from bound semantic metadata; never rewrite evidence.

This does not repeat call/IR admission already performed by export/ingestion.
It verifies the manifest/verification/workspace bindings relevant to completion,
without loading large model-call trees or retaining them in memory.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from natlang_neuralese.common.hashing import sha256_file_hex
from live_teacher_sources import bound_file, semantic_goal_completed, has_training_manifest, verify_completed_reassessment


def reassess(archive, scope=None):
    archive = Path(archive)
    receipts, seeds, errors = {}, {}, []
    before, after = set(), set()
    classifications = Counter()
    for receipt_path in sorted(archive.glob('*/receipt.json')):
        original = json.loads(receipt_path.read_text())
        cases = {}
        for name, item in original.items():
            if scope is not None and name not in scope:
                continue
            if item.get('status') == 'verified':
                before.add(name)
            if not has_training_manifest(item):
                continue
            try:
                manifest = (receipt_path.parent / 'capture' / item['manifest']).resolve()
                if not manifest.is_relative_to((receipt_path.parent / 'capture').resolve()):
                    raise ValueError('manifest escapes its original capture')
                spec = json.loads(manifest.read_text())
                if scope is not None and (spec['id'] != scope[name]['taskId'] or
                        hashlib.sha256(json.dumps(spec['interaction']['goal'], ensure_ascii=False, separators=(',', ':')).encode()).hexdigest() != scope[name]['goalSha256']):
                    raise ValueError('original manifest does not match the current catalog task/goal')
                verification_path = bound_file(spec['verification'], manifest.parent)
                verification = json.loads(verification_path.read_text())
                if verification['task_id'] != spec['id'] or verification['world'] != spec['world']:
                    raise ValueError('verification identifies another original task/world')
                proofs, evidence, workspace_payloads = [], [], []
                for binding in verification['evidence']:
                    if binding.get('kind') != 'workspace':
                        continue
                    path = bound_file(binding, manifest.parent)
                    payload = json.loads(path.read_text())
                    workspace_payloads.append(payload)
                    proofs.append(payload.get('proof') or {})
                    evidence.append({'path': str(path), 'sha256': sha256_file_hex(path)})
                interaction = spec.get('interaction')
                if interaction and not any(proof.get('goal') == interaction['goal'] and isinstance(proof.get('passed'), bool) for proof in proofs):
                    raise ValueError('open goal lacks an authoritative goal-bound semantic proof')
                verify_completed_reassessment(verification, spec, manifest.parent)
                completed = semantic_goal_completed(interaction, proofs, verification,
                    workspace_evidence=workspace_payloads, reassessment_verified=verification.get('reassessment') is not None)
                if verification['status'] == 'partial' and any(proof.get('goal') == (interaction or {}).get('goal') and proof.get('passed') is True for proof in proofs):
                    classifications['partial_scope_with_matching_native_pass'] += 1
                row = {'task_completed': completed, 'task_id': spec['id'],
                       'manifest': {'path': str(manifest), 'sha256': sha256_file_hex(manifest)},
                       'verification': {'path': str(verification_path), 'sha256': sha256_file_hex(verification_path), 'status': verification['status']},
                       'workspace_evidence': evidence,
                       'method': 'bound user-goal effects and requested returns; root exit not required'}
                cases[name] = row
                classifications['completed' if completed else 'partial'] += 1
                if completed:
                    after.add(name)
                if (verification['status'] == 'passed') != completed:
                    classifications['verification_semantic_disagreement'] += 1
            except (OSError, ValueError, KeyError, TypeError) as error:
                errors.append({'receipt': str(receipt_path), 'case': name, 'error': str(error)})
        record = {'receipt_sha256': sha256_file_hex(receipt_path), 'original_receipt': str(receipt_path), 'cases': cases}
        if receipt_path.parent.name.isdigit():
            receipts[receipt_path.parent.name + '-result.json'] = record
        else:
            seeds[receipt_path.parent.name] = record
    return {'schema': 'natlang.campaign-completion-reassessment/1',
            'implementation_sha256': sha256_file_hex(__file__),
            'semantic_reader_sha256': sha256_file_hex(Path(__file__).with_name('live_teacher_sources.py')), 'archive': str(archive.resolve()),
            'summary': {'before_completed_goals': len(before), 'after_completed_goals': len(after),
                        'reopened_goals': sorted((before - after) - {row['case'] for row in errors}), 'newly_recognized_completed_goals': sorted(after - before),
                        'unresolved': len(errors), **classifications},
            'receipts': receipts, 'seed_receipts': seeds, 'errors': errors,
            'original_evidence_modified': False, 'call_admission_repeated': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--held', type=Path, required=True)
    args = parser.parse_args()
    from vibestudio_campaign import campaign_scope
    _, _, scope = campaign_scope(args.catalog, args.held)
    result = reassess(args.archive, scope)
    result['catalog'] = {'path': str(args.catalog), 'sha256': sha256_file_hex(args.catalog)}
    result['held'] = {'path': str(args.held), 'sha256': sha256_file_hex(args.held)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['summary']))
    raise SystemExit(bool(result['errors']))
