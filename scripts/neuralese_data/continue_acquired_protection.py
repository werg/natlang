"""Certify unchanged admitted numeric rows under a stricter connected-source closure."""
from __future__ import annotations
import argparse
from collections import Counter
import json
import os
import shutil
import re
from pathlib import Path

from natlang_neuralese.common.hashing import canonical_json_sha256_hex, sha256_file_hex
from natlang_neuralese.common.jsonio import write_json_durable
from scripts.neuralese_data.build_acquired_full_session_graphs import sessions
from scripts.neuralese_data.build_verified_teacher_trajectory_graphs import current_protection
from scripts.neuralese_data.prepare_acquired_session_admission import (
    bound, decision_contract, verify_global_source_closure)
from scripts.neuralese_data.decision_holds import load_decision_holds
from scripts.neuralese_data.conversion_output_lease import acquire_output_lease

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 'natlang.acquired-protection-continuation/1'


def _binding(path, base):
    path = Path(path).resolve(strict=True)
    return {'path': os.path.relpath(path, base), 'sha256': sha256_file_hex(path), 'bytes': path.stat().st_size}


def _resolved(ref, base):
    return bound(base / ref['path'], ref)


def verify_certificate(path, *, audit_sha256=None):
    """Verify a published eligibility receipt and all its immutable dependencies."""
    path = Path(path).resolve(strict=True)
    value = json.loads(path.read_text())
    if (value.get('schema') != SCHEMA or value.get('passed') is not True
            or value.get('eligibility_only') is not True or value.get('numeric_rerendered') is not False
            or not value.get('dependencies')
            or (audit_sha256 is not None and value.get('original_audit', {}).get('sha256') != audit_sha256)):
        raise ValueError('invalid acquired protection continuation certificate')
    for ref in value['dependencies']:
        _resolved(ref, path.parent)
    for key in ('original_audit', 'old_protection', 'new_protection', 'row_proof'):
        ref = value[key]
        if ref not in value['dependencies']:
            raise ValueError('continuation certificate lost a required dependency binding')
    audit = json.loads(_resolved(value['original_audit'], path.parent).read_text())
    if (audit.get('status') != 'passed'
            or value['old_protection']['sha256'] != audit.get('protection', {}).get('sha256')
            or value['row_proof']['sha256'] != audit.get('row_proof_sha256')):
        raise ValueError('continuation certificate does not bind original qualified audit')
    return value


def verify_protection_transition(old_path, new_path):
    """One exact eligibility transition for existing audits and first-time audits."""
    old_path, new_path = Path(old_path), Path(new_path)
    old, new = current_protection(old_path), current_protection(new_path)
    verify_protection_sets(old, new)
    return old, new, verify_global_source_closure(new_path, new)


def verify_protection_sets(old, new):
    if (not old['held'] <= new['held'] or old['quality'] != new['quality']
            or old['pending'] != new['pending']
            or old['proof'].get('source_review_sha256') != new['proof'].get('source_review_sha256')):
        raise ValueError('protection continuation released held groups or changed quality/pending review policy')


def copy_evidence(source, target):
    source, target = Path(source), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256_file_hex(target) != sha256_file_hex(source):
            raise ValueError('immutable protection evidence changed')
    else:
        shutil.copyfile(source, target)
        with target.open('rb') as stream: os.fsync(stream.fileno())
    return target


def projection_evidence(projection, directory):
    """Copy immutable protection files with their relative roles intact."""
    projection, directory = Path(projection), Path(directory)
    destination = directory / 'protection-continuation-evidence' / sha256_file_hex(projection)
    files = []
    manifest = _binding(copy_evidence(projection, destination / 'protection-manifest.json'), directory)
    for child in json.loads(projection.read_text())['files']:
        relative = Path(child['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('unsafe protection evidence path')
        source = bound(projection.parent / relative, child)
        files.append(_binding(copy_evidence(source, destination / relative), directory))
    return {'manifest': manifest, 'files': files}


def verify_session_transition(value, old, new, held_sources):
    from scripts.neuralese_data.build_acquired_full_session_graphs import session_exclusion
    groups = set(value['trajectory']['split_groups'])
    if (not groups or ('test' if groups & old['held'] else 'train') !=
            ('test' if groups & new['held'] else 'train')
            or session_exclusion(value, old, held_sources) != session_exclusion(value, new, held_sources)):
        raise ValueError('protection continuation affects an original source split or eligibility: ' + value['trajectory']['id'])


def certify(audit_path, old_path, new_path, output, *, decision_paths=None):
    with acquire_output_lease(Path(output).parent):
        return _certify(audit_path, old_path, new_path, output, decision_paths=decision_paths)


def _certify(audit_path, old_path, new_path, output, *, decision_paths=None):
    audit_path, old_path, new_path = (Path(p).resolve(strict=True) for p in (audit_path, old_path, new_path))
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('protection continuation output is immutable')
    audit = json.loads(audit_path.read_text())
    if (audit.get('schema') != 'natlang.acquired-session-admission-audit/1'
            or audit.get('status') != 'passed' or audit.get('train_held_group_intersections') != 0
            or audit.get('oracle_verified') is not False):
        raise ValueError('protection continuation requires an original passed source audit')
    bound(old_path, audit['protection'])
    old, new, closure = verify_protection_transition(old_path, new_path)
    if audit['pending_source_groups_sha256'] != canonical_json_sha256_hex(sorted(old['pending'])):
        raise ValueError('original audit differs from prior pending protection')
    decisions = sorted((ROOT / 'training/decision-holds').glob('*.jsonl')) if decision_paths is None else decision_paths
    _, decision_bindings = load_decision_holds(decisions)
    if decision_contract(decision_bindings) != decision_contract(audit['decision_holds']):
        raise ValueError('protection continuation changed decision holds')
    dependencies = {}
    def keep(path, expected=None):
        path = Path(path).resolve(strict=True)
        if expected is not None:
            bound(path, expected)
        ref = _binding(path, output.parent)
        dependencies[str(path)] = ref
        return ref
    audit_ref = keep(audit_path)
    projection_refs = []
    # Keep the actual projection files and all policy children, not merely labels.
    for projection in (old_path, new_path):
        bundle = projection_evidence(projection, output.parent)
        projection_refs.append(keep(output.parent / bundle['manifest']['path'], bundle['manifest']))
        for child in bundle['files']:
            keep(output.parent / child['path'], child)
    old_ref, new_ref = projection_refs
    for item in decision_bindings:
        source = bound(item['path'], item)
        keep(copy_evidence(source, output.parent / 'protection-continuation-evidence' / 'decisions' / item['sha256'] / source.name), item)
    row_path = audit_path.parent / 'row-proof.jsonl'
    row_ref = keep(row_path, {'sha256': audit['row_proof_sha256']})
    expected_counts = Counter()
    for conversion in audit['conversions']:
        receipt_path = Path(conversion['receipt']['path'])
        keep(receipt_path, conversion['receipt'])
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get('schema') != 'natlang.acquired-full-session-graphs/1'
                or receipt.get('protection') != conversion.get('conversion_protection', audit['protection'])
                or decision_contract(receipt['decision_holds']) != decision_contract(audit['decision_holds'])
                or len(receipt['outputs']) != len(conversion['outputs'])):
            raise ValueError('original conversion receipt differs from qualified audit')
        keep(receipt_path.parent / 'resume-state.json', {'sha256': receipt['resume_state_sha256']})
        keep(receipt_path.parent / 'omissions.jsonl', {'sha256': receipt['omissions_sha256']})
        for original, qualified in zip(receipt['outputs'], conversion['outputs']):
            if any(qualified.get(key) != value for key, value in original.items()):
                raise ValueError('original aggregate binding differs from qualified audit')
            source = Path(original['path'])
            if qualified['split'] not in {'train', 'test'} or source.parent.name != qualified['split']:
                raise ValueError('original aggregate split differs from qualified audit')
            keep(source, original)
            expected_counts[(conversion['model'], qualified['split'])] += original['rows']
    seen = set(); counts = Counter()
    for row in sessions(row_path):
        model, split, identifier = row['model'], row['split'], row['id']
        groups = row.get('source_groups')
        if (split not in {'train', 'test'} or not isinstance(identifier, str) or not identifier
                or not isinstance(groups, list) or not groups or len(groups) != len(set(groups))
                or any(not isinstance(group, str) or not group for group in groups)
                or not isinstance(row.get('graph_sha256'), str) or not re.fullmatch(r'[0-9a-f]{64}', row['graph_sha256'])
                or (model, identifier) in seen):
            raise ValueError('original row proof has invalid or repeated row identities')
        groups = set(groups)
        if (groups & (new['quality'] | new['pending'])
                or ('test' if groups & old['held'] else 'train') != split
                or ('test' if groups & new['held'] else 'train') != split):
            raise ValueError('protection continuation affects an admitted row split or eligibility: ' + identifier)
        seen.add((model, identifier)); counts[(model, split)] += 1
    if counts != +expected_counts or not counts:
        raise ValueError('original row proof does not cover complete qualified aggregate inventory')
    if (load_decision_holds(decisions)[1] != decision_bindings
            or current_protection(old_path) != old or current_protection(new_path) != new):
        raise ValueError('protection changed during continuation verification')
    value = {'schema': SCHEMA, 'passed': True, 'eligibility_only': True, 'numeric_rerendered': False,
        'native_execution_verified': False, 'exposure_granted': False,
        'original_audit': audit_ref, 'old_protection': old_ref, 'new_protection': new_ref,
        'row_proof': row_ref, 'new_global_source_closure': closure,
        'row_counts': {model: {split: counts[(model, split)] for split in ('train', 'test')}
                       for model in sorted({key[0] for key in counts})},
        'new_protection_proof': {**new['proof'], 'path': new_ref['path']},
        'held_groups_added': len(new['held'] - old['held']),
        'dependencies': sorted(dependencies.values(), key=lambda ref: ref['path']),
        'verifier_sha256': sha256_file_hex(__file__)}
    for directory in sorted({output.parent, *(Path(output.parent / ref['path']).parent for ref in dependencies.values())}, key=str):
        descriptor = os.open(directory, os.O_DIRECTORY)
        try: os.fsync(descriptor)
        finally: os.close(descriptor)
    write_json_durable(output, value)
    verify_certificate(output)
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('audit', 'old-protection', 'new-protection', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args(argv)
    value = certify(args.audit, args.old_protection, args.new_protection, args.output)
    print(json.dumps({'path': str(args.output), 'sha256': sha256_file_hex(args.output), 'row_counts': value['row_counts']}))


if __name__ == '__main__': main()
