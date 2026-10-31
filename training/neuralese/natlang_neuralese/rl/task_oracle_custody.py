"""Verify declared grouped task-oracle evidence against the actual collector runtime."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from ..common.hashing import sha256_file_hex


def _reference(path):
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'sha256': sha256_file_hex(path)}


def verify_task_oracle_runtime(task_source, runtime, cases_sha, *, _dependency_cache=None):
    """Bind an explicitly repaired oracle to the collector actually used for RL."""
    manifest_path = Path(task_source).resolve(strict=True)
    if _dependency_cache is None:
        _dependency_cache = set()
    manifest = json.loads(manifest_path.read_text())
    components = manifest.get('oracle_components')
    if components is not None:
        if not isinstance(components, list) or not components or manifest.get('oracle_validation') is not None:
            raise ValueError('task oracle union needs explicit component proofs')
        retained = bytearray()
        for component in components:
            path = (manifest_path.parent / component['path']).resolve(strict=True)
            if not path.is_relative_to(manifest_path.parent) or _reference(path)['sha256'] != component['sha256']:
                raise ValueError('task oracle component changed or escaped its source')
            if path == manifest_path:
                raise ValueError('task oracle component cannot reference its own union')
            declaration = json.loads(path.read_text())
            if declaration.get('oracle_validation') is None and declaration.get('oracle_components') is None:
                raise ValueError('task oracle component lacks its own oracle proof')
            from .task_source import resolve_task_source
            admitted = resolve_task_source(path, _skip_oracle_validation=True)
            verify_task_oracle_runtime(path, runtime, admitted['cases_sha256'], _dependency_cache=_dependency_cache)
            retained.extend(admitted['cases'].read_bytes())
        union_path = (manifest_path.parent / manifest['cases']).resolve(strict=True)
        if (not union_path.is_relative_to(manifest_path.parent) or _reference(union_path)['sha256'] != cases_sha
                or union_path.read_bytes() != bytes(retained)):
            raise ValueError('task oracle union differs from its exact ordered components')
        return
    validation = manifest.get('oracle_validation')
    if validation is None:
        raise ValueError('RL task source lacks its actual adversarial oracle proof')
    path = (manifest_path.parent / validation['path']).resolve(strict=True)
    if not path.is_relative_to(manifest_path.parent) or _reference(path)['sha256'] != validation['sha256']:
        raise ValueError('task oracle proof changed or escaped its source')
    proof = json.loads(path.read_text())
    if proof.get('schema') not in {'natlang.lfm3-derived-skill-oracle-validation/1',
                                   'natlang.lfm3-effect-oracle-validation/1',
                                   'natlang.lfm3-world-oracle-validation/1'}:
        raise ValueError('task oracle proof has an unsupported schema')
    if not proof.get('group_identity_sha256'):
        raise ValueError('task oracle proof lacks its dependency-bound complete-group evidence')
    if (proof.get('source', {}).get('sha256') != cases_sha
            or proof.get('runtime') != manifest.get('runtime_requirement')
            or not isinstance(proof.get('runtime'), dict)
            or set(proof['runtime']) not in ({'collector', 'required_actions'},
                                             {'collector', 'required_actions', 'service_receipt'})):
        raise ValueError('task oracle proof does not bind the exact source and runtime')
    runtime = Path(runtime).resolve(strict=True)
    if proof.get('group_identity_sha256'):
        identity_path = manifest_path.parent / 'verification-identity.json'
        journal_path = manifest_path.parent / 'oracle-groups.jsonl'
        if (_reference(identity_path)['sha256'] != proof['group_identity_sha256']
                or _reference(journal_path)['sha256'] != proof.get('group_journal_sha256')):
            raise ValueError('task oracle group identity or journal changed')
        identity = json.loads(identity_path.read_text())
        node = subprocess.run(['node', '--version'], check=True, capture_output=True, text=True).stdout.strip()
        if identity.get('node') != node:
            raise ValueError('task oracle Node version differs from its group verification')
        dependency_sha = proof.get('runtime_dependency_inventory_sha256')
        if dependency_sha is None:
            raise ValueError('task oracle grouped proof lacks its installed runtime dependency inventory')
        if dependency_sha is not None:
            dependency_path = manifest_path.parent / 'runtime-inventory.json'
            entries = [item for item in identity['inputs'] if item['path'].endswith('/runtime-inventory.json')]
            if (len(entries) != 1 or entries[0]['sha256'] != dependency_sha
                    or _reference(dependency_path)['sha256'] != dependency_sha):
                raise ValueError('task oracle dependency inventory changed')
            receipt = json.loads(dependency_path.read_text())
            if receipt.get('schema') != 'natlang.rl-oracle-runtime-inventory/1':
                raise ValueError('task oracle dependency inventory has an unsupported schema')
            for label in ('preflight', 'postflight'):
                check_path = manifest_path.parent / f'runtime-{label}.json'
                if _reference(check_path)['sha256'] != proof.get(f'runtime_{label}_sha256'):
                    raise ValueError('task oracle dependency boundary receipt changed')
                check = json.loads(check_path.read_text())
                if (check.get('matched') is not True or
                        (label == 'postflight' and check.get('inventory_sha256') != dependency_sha)):
                    raise ValueError('task oracle dependency boundary was not verified')
            cache_key = (str(runtime), dependency_sha)
            if cache_key not in _dependency_cache:
                from ..common.runtime_inventory import task_runtime_inventory
                current, expected = task_runtime_inventory(runtime), receipt['inventory']
                ignored = {'@dependency-source-links'}
                if (set(current) - ignored != set(expected) - ignored
                        or any(current[key] != value for key, value in expected.items() if key not in ignored)):
                    raise ValueError('task oracle installed runtime dependencies differ from their verified inventory')
                _dependency_cache.add(cache_key)
        for name in ('groups.jsonl', 'derivation.json'):
            entries = [item for item in identity['inputs'] if item['path'].endswith('/' + name)]
            if len(entries) != 1 or _reference(manifest_path.parent / name)['sha256'] != entries[0]['sha256']:
                raise ValueError('task oracle group source or derivation changed')
        events = [json.loads(line) for line in journal_path.read_text().splitlines() if line.strip()]
        passed = [result for event in events if event.get('passed') is True and event.get('failure') is None
                  for result in event.get('results', [])]
        if passed != proof.get('results'):
            raise ValueError('task oracle proof differs from its closed passing groups')
        runtime_root = identity.get('runtime_root', '/home/werg/natlang/ts-host')
        if not isinstance(runtime_root, str) or not Path(runtime_root).is_absolute():
            raise ValueError('task oracle runtime source root is invalid')
        prefix = runtime_root.rstrip('/') + '/dist/'
        inventory = {item['path'][len(prefix):]: item['sha256'] for item in identity['inputs']
                     if item['path'].startswith(prefix)}
        actual_files = {str(p.relative_to(runtime / 'dist')) for p in (runtime / 'dist').rglob('*') if p.is_file()}
        if not inventory or set(inventory) != actual_files or any(
                _reference(runtime / 'dist' / name)['sha256'] != digest for name, digest in inventory.items()):
            raise ValueError('task oracle collector runtime inventory differs from its group verification')
        lock = next((item for item in identity['inputs'] if item['path'] == runtime_root.rstrip('/') + '/package-lock.json'), None)
        if lock is None or _reference(runtime / 'package-lock.json')['sha256'] != lock['sha256']:
            raise ValueError('task oracle runtime dependency lock differs from its group verification')
    for name, digest in proof['runtime'].items():
        actual = runtime / 'dist' / 'teacher' / (name.replace('_', '-') + '.js')
        if _reference(actual)['sha256'] != digest:
            raise ValueError('task oracle runtime differs from its adversarial verification')
    effect = proof.get('schema') == 'natlang.lfm3-effect-oracle-validation/1'
    world = proof.get('schema') == 'natlang.lfm3-world-oracle-validation/1'
    if world and 'service_receipt' not in proof['runtime']:
        raise ValueError('world oracle lacks its exact service receipt runtime')
    if not proof.get('results') or any(not row.get('reference', {}).get('ok')
            or (not world and row.get('wrong_answer_accepted') is not False)
            or (effect and (row.get('effect_check') not in {'effects', 'files'}
                            or row.get('missing_required_effect_accepted') is not False
                            or row.get('missing_required_effect_check') is not False))
            or (world and ((row.get('control_kind') == 'delivery' and
                            (row.get('wrong_answer_accepted') is not False
                             or row.get('missing_required_effect_accepted') is not False
                             or row.get('missing_required_effect_check') is not False
                             or row.get('unsafe_route_accepted') is not False
                             or row.get('unsafe_world_check') is not False))
                           or (row.get('control_kind') == 'blocked' and
                               (row.get('false_success_accepted') is not False
                                or row.get('expected_status_check') is not False))
                           or row.get('control_kind') not in {'delivery', 'blocked'}))
            or (not effect and not world and row.get('missing_required_read_accepted') is not False)
            for row in proof['results']):
        raise ValueError('task oracle lacks complete reference and adversarial evidence')
    cases_path = (manifest_path.parent / manifest['cases']).resolve(strict=True)
    if not cases_path.is_relative_to(manifest_path.parent) or _reference(cases_path)['sha256'] != cases_sha:
        raise ValueError('task oracle cases differ from their source')
    ids = [json.loads(line)['id'] for line in cases_path.read_text().splitlines() if line.strip()]
    if ids != [row.get('id') for row in proof['results']]:
        raise ValueError('task oracle evidence does not cover the exact ordered task rows')

