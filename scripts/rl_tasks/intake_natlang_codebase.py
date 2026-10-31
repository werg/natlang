#!/usr/bin/env python3
"""Package a deep Natlang application and its reviewed invocation worlds as RL cases.

The codebase snapshot is the first-class program under test. Every world names
an actual entry operation in that package; related invocation worlds stay on a
single split group. Exact and independently judged oracles remain distinct.
The result still has to pass the ordinary RL task verifier and reviewed
trajectory collection path before it can be considered for admission.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.rl_tasks.common import jsonl_write, slug, text_digest
from scripts.rl_tasks.audit_natlang_packages import audit
from scripts.rl_tasks.codebase_services import load_external_services


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _artifact_digest(path: Path) -> str:
    path = path.expanduser().resolve(strict=True)
    if path.is_file():
        return _sha(path.read_bytes())
    if not path.is_dir():
        raise ValueError(f'dataset artifact is not a file or directory: {path}')
    entries = []
    for item in sorted(path.rglob('*')):
        if item.is_symlink():
            raise ValueError(f'dataset artifact contains a symlink: {item}')
        if item.is_file() and '.git' not in item.relative_to(path).parts:
            rel = item.relative_to(path).as_posix()
            entries.append([rel, _sha(item.read_bytes())])
    return _sha(json.dumps(entries, separators=(',', ':'), ensure_ascii=False).encode())


def _bound_dataset_paths(metadata: dict, package_parent: Path) -> dict:
    bound = json.loads(json.dumps(metadata))
    dataset = bound.get('dataset')
    if isinstance(dataset, dict):
        for key in ('artifact_path', 'provenance_path', 'license_evidence_path'):
            if isinstance(dataset.get(key), str):
                path = Path(dataset[key]).expanduser()
                if not path.is_absolute():
                    path = package_parent / path
                dataset[key] = str(path.resolve(strict=True))
    return bound


def _load_json(path: Path, *, required: bool = True) -> dict[str, Any]:
    if not path.exists():
        if required:
            raise ValueError(f'missing required file: {path}')
        return {}
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'{path} must contain an object')
    return value


def _project(package: Path, metadata: dict | None = None) -> dict[str, str]:
    project = {}
    for path in sorted(package.rglob('*')):
        if path.is_symlink():
            raise ValueError(f'symlink in package: {path}')
        if not path.is_file() or path.name.endswith('.d.nl.ts') or path.suffix not in {'.nl', '.ts', '.mjs'} and path.name != 'natlang.json':
            continue
        rel = path.relative_to(package).as_posix()
        if rel.startswith('../') or '\\' in rel or any(part in {'.', '..'} for part in rel.split('/')):
            raise ValueError(f'unsafe package path: {rel}')
        project[rel] = path.read_text(encoding='utf-8')
    for rel in (metadata or {}).get('package_assets', []):
        if not isinstance(rel, str) or not rel or rel.startswith('/') or '\\' in rel or any(part in {'.', '..'} for part in rel.split('/')):
            raise ValueError(f'unsafe metadata.package_assets entry: {rel!r}')
        if any(part.lower() in {'worlds', 'tests', 'test', 'fixtures', 'held', 'confirmation', 'gold', 'expected'}
               for part in Path(rel).parts):
            raise ValueError(f'worlds, tests and expected answers cannot be packaged: {rel}')
        asset = package / rel
        if asset.is_symlink() or not asset.is_file():
            raise ValueError(f'metadata.package_assets entry is not a regular file: {rel}')
        project[rel] = asset.read_text(encoding='utf-8')
    if not project:
        raise ValueError(f'no loadable source files under {package}')
    return project


def _split_for(row: dict, source_split: str, metadata: dict, group: str) -> str:
    split_table = metadata.get('split', {})
    if not isinstance(split_table, dict):
        raise ValueError('metadata.split must be an object mapping group IDs to split names')
    declared = {name: set(split_table.get(f'{name}_groups', [])) for name in ('train', 'dev', 'confirmation')}
    matches = [name for name, groups in declared.items() if group in groups]
    if len(matches) > 1:
        raise ValueError(f'world group {group!r} is assigned to multiple splits')
    if matches and matches[0] != source_split:
        raise ValueError(f'world group {group!r} file disagrees with metadata.split')
    split = matches[0] if matches else source_split
    if split not in {'train', 'dev', 'confirmation'}:
        raise ValueError(f'unsupported split {split!r}')
    return split


def _worlds(root: Path, metadata: dict) -> list[dict]:
    worlds = root.parent / 'worlds'
    if not worlds.is_dir():
        raise ValueError(f'missing world fixtures directory: {worlds}')
    rows, seen_ids, group_splits = [], set(), {}
    for filename, split in (('train.jsonl', 'train'), ('dev.jsonl', 'dev'),
                            ('confirmation.jsonl', 'confirmation')):
        path = worlds / filename
        if not path.exists():
            continue
        for line_no, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f'{path}:{line_no}: expected a JSON object')
            required = ('id', 'group_id', 'task_family', 'operation', 'task', 'inputs', 'expected', 'oracle')
            missing = [key for key in required if key not in row]
            if missing:
                raise ValueError(f'{path}:{line_no}: missing {missing}')
            if not all(isinstance(row[k], str) and row[k].strip() for k in ('id', 'group_id', 'task_family', 'operation', 'task')):
                raise ValueError(f'{path}:{line_no}: identity, family, operation and task must be nonempty strings')
            if row['id'] in seen_ids:
                raise ValueError(f'duplicate world id: {row["id"]}')
            seen_ids.add(row['id'])
            if not isinstance(row['inputs'], dict):
                raise ValueError(f'{path}:{line_no}: inputs must be an object')
            source_groups = row.get('source_groups')
            if (not isinstance(source_groups, list) or not source_groups or
                    any(not isinstance(group, str) or not group.strip() for group in source_groups) or
                    len(set(source_groups)) != len(source_groups)):
                raise ValueError(f'{path}:{line_no}: source_groups must contain complete upstream namespaced world IDs')
            oracle = row['oracle']
            if oracle == 'exact':
                pass
            elif isinstance(oracle, dict) and oracle.get('level') == 'judged' and isinstance(oracle.get('rubric'), str) and oracle['rubric'].strip():
                pass
            else:
                raise ValueError(f'{path}:{line_no}: oracle must be exact or a judged object with an independent rubric')
            reference_action = row.get('reference_action')
            if reference_action is not None:
                if (not isinstance(reference_action, dict) or reference_action.get('tool') != 'eval' or
                        not isinstance(reference_action.get('arguments'), dict) or
                        not isinstance(reference_action['arguments'].get('code'), str) or
                        reference_action['arguments'].get('finish') is not True):
                    raise ValueError(f'{path}:{line_no}: reference_action must be a finishing eval tool action')
            wrong_code = row.get('wrong_code', [])
            if (not isinstance(wrong_code, list) or
                    any(not isinstance(code, str) or not code.strip() for code in wrong_code)):
                raise ValueError(f'{path}:{line_no}: wrong_code must be a list of nonempty eval programs')
            assigned = _split_for(row, split, metadata, row['group_id'])
            previous = group_splits.setdefault(row['group_id'], assigned)
            if previous != assigned:
                raise ValueError(f'related world group {row["group_id"]!r} crosses splits')
            rows.append({**row, 'split': assigned})
    if not rows:
        raise ValueError('no fixture worlds found')
    return rows


def _make_cases(program_id: str, package_revision: str, package: dict[str, str], metadata: dict,
                worlds: list[dict], external_services: dict[str, str],
                service_bindings: dict[str, dict[str, str]]) -> list[dict]:
    dataset = metadata.get('dataset')
    required_dataset = ('identity', 'revision', 'license', 'source_uri', 'artifact_path', 'artifact_sha256',
                        'provenance_path', 'provenance_sha256', 'license_evidence_path', 'license_evidence_sha256')
    if not isinstance(dataset, dict) or not all(isinstance(dataset.get(k), str) and dataset[k].strip()
                                                for k in required_dataset):
        raise ValueError(f'metadata.dataset must bind {", ".join(required_dataset)}')
    for field in ('artifact_sha256', 'provenance_sha256'):
        if not re.fullmatch(r'[0-9a-f]{64}', dataset[field]):
            raise ValueError(f'metadata.dataset.{field} must be a full lowercase SHA-256')
    artifact_path = Path(dataset['artifact_path']).expanduser()
    actual_artifact_sha = _artifact_digest(artifact_path)
    if actual_artifact_sha != dataset['artifact_sha256']:
        raise ValueError('dataset artifact bytes do not match the pinned artifact_sha256')
    provenance_path = Path(dataset['provenance_path']).expanduser().resolve(strict=True)
    if _sha(provenance_path.read_bytes()) != dataset['provenance_sha256']:
        raise ValueError('dataset provenance bytes do not match the pinned provenance_sha256')
    license_path = Path(dataset['license_evidence_path']).expanduser().resolve(strict=True)
    if not re.fullmatch(r'[0-9a-f]{64}', dataset['license_evidence_sha256']) or _sha(license_path.read_bytes()) != dataset['license_evidence_sha256']:
        raise ValueError('license evidence is absent or differs from its pin')
    if dataset.get('license_status') not in {'cleared', 'hold'}:
        raise ValueError('metadata.dataset.license_status must be cleared or hold')
    if dataset['license'].lower() in {'unknown', 'not-stated', 'not stated'} and dataset['license_status'] != 'hold':
        raise ValueError('unstated license must remain on rights hold')
    entity_model = metadata.get('entity_model')
    workflows = metadata.get('workflows')
    if not isinstance(entity_model, dict) or not entity_model.get('entities') or not entity_model.get('relations'):
        raise ValueError('metadata.entity_model must declare entities and relations for this codebase')
    if not isinstance(workflows, list) or not workflows:
        raise ValueError('metadata.workflows must name the codebase workflows')
    curriculum_domain = metadata.get('curriculum_domain', 'other')
    if curriculum_domain not in {'logic', 'relational', 'actor', 'other', 'writing'}:
        raise ValueError('metadata.curriculum_domain must be a supported collector domain')
    split_policy = metadata.get('split_policy', 'codebase')
    if split_policy not in {'codebase', 'world'}:
        raise ValueError("metadata.split_policy must be 'codebase' or explicit 'world'")
    package_splits = {row['split'] for row in worlds}
    if split_policy == 'codebase' and len(package_splits) != 1:
        raise ValueError('whole-codebase split policy requires all worlds for one package to share one split')
    source_id = f'{program_id}@{package_revision}'
    cases = []
    for row in worlds:
        root = row['operation']
        if root not in package or not root.endswith('.nl'):
            raise ValueError(f'{row["id"]}: operation must name a package .nl entrypoint: {root}')
        case_files = dict(package)
        header_end = case_files[root].find('\n---', 4)
        if not case_files[root].startswith('---\n') or header_end < 0:
            raise ValueError(f'{root}: expected a natlang frontmatter signature')
        case_files[root] = (case_files[root][:header_end + 5] +
                            f'For this invocation: {row["task"].strip()}\n\n' +
                            case_files[root][header_end + 5:])
        groups = set(row['source_groups'])
        groups.add(f'world:{program_id}:{row["group_id"]}')
        if split_policy == 'codebase':
            groups.add(f'codebase:{program_id}:{package_revision}')
        groups = sorted(groups)
        oracle = row['oracle']
        case_id = f'codebase:{slug(program_id, 50)}:{slug(row["id"], 70)}:{text_digest(source_id + row["id"])[:10]}'
        case = {
            'version': 'natlang.program/2', 'id': case_id,
            'kind': 'lambda_source', 'family': f'natlang_codebase_{slug(program_id, 48)}_{slug(row["task_family"], 32)}',
            'source': f'external-natlang-codebase:{program_id}', 'split': row['split'],
            'source_ids': [f'{dataset["identity"]}@{dataset["revision"]}', f'{source_id}:world:{row["id"]}'],
            'source_groups': groups, 'source_revisions': [f'{dataset["identity"]}@{dataset["revision"]}', source_id],
            'license': dataset['license'], 'gold_sources': row.get('gold_sources', ['independent-fixture']),
            'generation': {'generator': 'natlang-codebase-intake/1', 'teacher_generation': False,
                           'method': 'authored Natlang application operation evaluated on a pinned world fixture; no sampled trajectory'},
            'curriculum': {
                'version': 'natlang.inline_curriculum/1', 'family': f'ext_natlang_codebase_{slug(program_id, 36)}_{slug(row["task_family"], 30)}',
                'family_version': 1, 'shape': f'codebase-{slug(row["task_family"], 44)}', 'variant': row.get('variant', 'base'),
                'pair_group': row.get('pair_group'), 'split_group': (f'codebase:{program_id}:{package_revision}'
                    if split_policy == 'codebase' else f'world:{program_id}:{row["group_id"]}'),
                'slice': 'nested_scoped', 'domain': curriculum_domain,
                'mode': row.get('mode', 'multi_call' if row.get('invocation_index', 0) else 'single_call'),
                'inline': 'optional', 'evidence': {'world': [], 'retrieved': [], 'background': []},
                'assumptions': [], 'decisive': [], 'plausible_actions': [], 'minimum_sequence': [],
                'reference': {'root': [[row['reference_action']['tool'], row['reference_action']['arguments']]]
                              if row.get('reference_action') else
                              [['return_result', {'status': 'success', 'value': row['expected']}]]},
                'rl': {'level': int(row.get('level', 2)), 'family': f'ext_natlang_codebase_{slug(program_id, 36)}_{slug(row["task_family"], 30)}',
                       'basis': 'independent-codebase-world-fixture'},
                'external': {'source': 'natlang-codebase-intake', 'upstream': dataset['identity'],
                             'upstream_id': row['id'], 'upstream_split': row['split'],
                             'unit': f'{program_id}:{row["group_id"]}', 'shape': f'codebase-{slug(row["task_family"], 44)}',
                             'task_family': row['task_family'], 'failure_mode': row.get('failure_mode'),
                             'authored_domain': row.get('domain') or metadata.get('domain'),
                             'operation': root, 'program_id': program_id, 'program_revision': package_revision,
                             'invocation_index': row.get('invocation_index'), 'entity_model_sha256': _sha(json.dumps(entity_model, sort_keys=True).encode()),
                             'workflow_ids': row.get('workflow_ids', []), 'instruction_sha256': text_digest(row['task']),
                             'external_services': service_bindings,
                             'checks': {'wrong_code': row.get('wrong_code', [])},
                             'oracle_kind': 'exact' if oracle == 'exact' else 'judged'},
            },
            'semantics': {'root': root, 'files': case_files, 'inputs': row['inputs'], 'expected': row['expected'],
                          'oracle': 'exact' if oracle == 'exact' else oracle},
        }
        if external_services:
            case['semantics']['services'] = external_services
        cases.append(case)
    return cases


def intake(package_root: Path, *, output: Path, diversity_output: Path | None = None,
           judged_output: Path | None = None) -> dict:
    package_root = package_root.resolve(strict=True)
    if package_root.name != 'package':
        raise ValueError('package root must be a directory named package/ under a codebase folder')
    metadata = _load_json(package_root.parent / 'metadata.json')
    for key in ('program_id', 'source_revision', 'domain', 'user_goal', 'author'):
        if not metadata.get(key):
            raise ValueError(f'metadata.{key} is required')
    if not isinstance(metadata.get('split'), dict):
        raise ValueError('metadata.split is required, including held-world protections')
    metadata = _bound_dataset_paths(metadata, package_root.parent)
    files = _project(package_root, metadata)
    external_services, service_bindings = load_external_services(package_root, metadata)
    world_rows = _worlds(package_root, metadata)
    service_set_sha = _sha(json.dumps(service_bindings, sort_keys=True, separators=(',', ':')).encode())
    package_revision = metadata['source_revision'] + (f'+services:{service_set_sha}' if service_bindings else '')
    cases = _make_cases(metadata['program_id'], package_revision, files, metadata, world_rows,
                        external_services, service_bindings)
    output.parent.mkdir(parents=True, exist_ok=True)
    exact_cases = [case for case in cases if case['semantics']['oracle'] == 'exact']
    judged_cases = [case for case in cases if isinstance(case['semantics']['oracle'], dict)]
    jsonl_write(output, exact_cases)
    judged_output = judged_output or output.with_name(f'{output.stem}.judged{output.suffix}')
    if judged_cases:
        judged_output.parent.mkdir(parents=True, exist_ok=True)
        jsonl_write(judged_output, judged_cases)
    worlds_dir = package_root.parent / 'worlds'
    world_bindings = []
    for filename in ('train.jsonl', 'dev.jsonl', 'confirmation.jsonl'):
        path = worlds_dir / filename
        if path.is_file():
            world_bindings.append({'path': f'worlds/{filename}', 'sha256': _sha(path.read_bytes())})
    report = audit([package_root], max_near_partners=0)
    record = {
        'schema': 'natlang.codebase_intake_receipt/1', 'program_id': metadata['program_id'],
        'source_revision': package_revision, 'package_source_revision': metadata['source_revision'],
        'dataset': metadata.get('dataset'),
        'package_path': str(package_root),
        'package_sha256': _sha(json.dumps({'files': files, 'services': external_services}, sort_keys=True, ensure_ascii=False).encode()),
        'external_services': service_bindings,
        'world_count': len(world_rows), 'case_count': len(cases),
        'exact_case_count': len(exact_cases), 'judged_case_count': len(judged_cases),
        'exact_cases_path': str(output), 'judged_cases_path': str(judged_output) if judged_cases else None,
        'exact_cases_sha256': _sha(output.read_bytes()),
        'judged_cases_sha256': _sha(judged_output.read_bytes()) if judged_cases else None,
        'world_bindings': world_bindings,
        'split_counts': {split: sum(1 for row in world_rows if row['split'] == split) for split in ('train', 'dev', 'confirmation')},
        'distinct_world_groups': len({row['group_id'] for row in world_rows}),
        'distinct_task_families': len({row['task_family'] for row in world_rows}),
        'distinct_operations': len({row['operation'] for row in world_rows}),
        'split_policy': metadata.get('split_policy', 'codebase'),
        'rights_status': metadata['dataset']['license_status'],
        'admission_eligible': metadata['dataset']['license_status'] == 'cleared',
        'oracle_counts': {'exact': sum(1 for row in world_rows if row['oracle'] == 'exact'),
                          'judged': sum(1 for row in world_rows if isinstance(row['oracle'], dict))},
        'semantic_status': 'fixture-backed, not yet runtime-verified or live-model behaviorally evaluated',
        'diversity': report,
    }
    if diversity_output:
        diversity_output.parent.mkdir(parents=True, exist_ok=True)
        with diversity_output.open('x', encoding='utf-8') as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--diversity-output', type=Path)
    parser.add_argument('--judged-output', type=Path,
                        help='separate source file for rows that require an independent judge; exact output remains --output')
    args = parser.parse_args()
    receipt = intake(args.package, output=args.output, diversity_output=args.diversity_output,
                     judged_output=args.judged_output)
    with args.receipt.open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({k: receipt[k] for k in ('program_id', 'case_count', 'split_counts', 'oracle_counts', 'semantic_status')}, sort_keys=True))


if __name__ == '__main__':
    main()
