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
from scripts.rl_tasks.codebase_services import load_external_services, bind_world_services
from scripts.rl_tasks.execution_adapter import reject_uninvoked_execution_adapter, verify_runtime_trees


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


_NL_CALL_REFERENCE = re.compile(r"\b(?:call|invoke|run)\s+([A-Za-z_$][A-Za-z0-9_$]*)(?:\.[A-Za-z_$][A-Za-z0-9_$]*)?\b", re.IGNORECASE)


def _declared_uses(source: str) -> list[str]:
    """Read simple `uses` entries from NL frontmatter for callable-scope checks."""
    if not source.startswith('---\n'):
        return []
    end = source.find('\n---', 4)
    if end < 0:
        return []
    header = source[4:end]
    match = re.search(r'(?m)^uses:\s*(.*)$', header)
    if not match:
        return []
    inline = match.group(1).strip()
    if inline.startswith('[') and inline.endswith(']'):
        return [item.strip().strip("'\"") for item in inline[1:-1].split(',') if item.strip()]
    uses: list[str] = []
    lines = header.splitlines()
    start = next((i for i, line in enumerate(lines) if re.match(r'^uses:\s*$', line)), None)
    if start is not None:
        for line in lines[start + 1:]:
            item = re.match(r'^\s+-\s*([^\s#]+)', line)
            if not item:
                if line.strip() and not line.lstrip().startswith('#'):
                    break
                continue
            uses.append(item.group(1).strip("'\""))
    return uses


def _validate_natlang_child_call_visibility(files: dict[str, str]) -> dict:
    """Fail intake when instructions name a package child outside that NL node's tree.

    NL children are loaded from a callable's companion directory, while `uses`
    attaches an explicitly named package item. A prose instruction to call a
    function elsewhere in the package does not make that sibling visible. This
    small source inventory catches such mistakes before a teacher request; it
    does not claim that the model follows the instructions or that the workflow
    itself is behaviorally correct.
    """
    nl_paths = sorted(path for path in files if path.endswith('.nl'))
    callable_paths = {path for path in files if path.endswith(('.nl', '.ts'))}
    names: dict[str, list[str]] = {}
    for path in callable_paths:
        name = Path(path).stem
        names.setdefault(name, []).append(path)

    checked = []
    violations = []
    for path in nl_paths:
        source = files[path]
        parent = Path(path).parent
        companion = (parent / Path(path).stem).as_posix()
        visible = {
            candidate[len(companion) + 1:].split('/', 1)[0].rsplit('.', 1)[0]
            for candidate in callable_paths
            if candidate.startswith(companion + '/') and
            '/' not in candidate[len(companion) + 1:]
        }
        for item in _declared_uses(source):
            normalized = item.removesuffix('.nl').removesuffix('.ts').strip('/')
            if normalized:
                visible.add(Path(normalized).name)
        references = []
        _, _, instructions = source.partition('\n---')
        for match in _NL_CALL_REFERENCE.finditer(instructions):
            name = match.group(1)
            if name not in names:
                continue
            references.append(name)
            if name not in visible:
                violations.append({'path': path, 'callable': name,
                                   'defined_at': names[name],
                                   'reason': 'referenced child is not in this callable’s companion folder or declared uses'})
        checked.append({'path': path, 'visible_children': sorted(visible),
                        'internal_call_references': sorted(set(references))})
    if violations:
        details = '; '.join(f"{item['path']} calls {item['callable']} but it is only defined at {', '.join(item['defined_at'])}"
                            for item in violations)
        raise ValueError(f'Natlang child-call visibility check failed: {details}')
    return {'schema': 'natlang.child-call-visibility/1', 'checked_functions': checked,
            'call_reference_count': sum(len(row['internal_call_references']) for row in checked)}


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


def _validate_json_semantic_files_oracle(row: dict, path: Path, line_no: int) -> None:
    spec = row.get('files_oracle')
    if spec is None or not isinstance(spec, dict) or spec.get('compare') != 'json-semantic-structured':
        return
    comparison = spec.get('comparison_spec')
    reports, contexts = spec.get('json_report_paths'), spec.get('rationale_context_paths')
    if (not isinstance(comparison, dict) or comparison.get('schema_version') != 'semantic-structured-comparison/1' or
            any(not isinstance(comparison.get(field), list) for field in ('unordered_paths', 'ordered_paths', 'prose_paths')) or
            not comparison['prose_paths'] or not isinstance(comparison.get('prose_policy'), str) or
            not comparison['prose_policy'].strip()):
        raise ValueError(f'{path}:{line_no}: json-semantic-structured requires a complete semantic-structured-comparison/1 contract')
    pointers = [*comparison['unordered_paths'], *comparison['ordered_paths'], *comparison['prose_paths']]
    if (any(not isinstance(pointer, str) or not pointer.startswith('/') or '\x00' in pointer or
            re.search(r'~(?![01])', pointer) for pointer in pointers) or len(set(pointers)) != len(pointers)):
        raise ValueError(f'{path}:{line_no}: JSON comparison paths must be unique valid JSON pointers')
    if (not isinstance(reports, list) or not reports or
            any(not isinstance(name, str) for name in reports) or
            len(set(reports)) != len(reports) or
            any(not name or name.startswith('/') or '\\' in name or '\x00' in name or
                re.match(r'^[A-Za-z]:', name) or any(part in {'', '.', '..'} for part in name.split('/')) or
                not name.lower().endswith('.json') for name in reports)):
        raise ValueError(f'{path}:{line_no}: json_report_paths must be unique safe relative JSON output paths')
    expected = row.get('expected_files')
    folder = row.get('folder_files')
    if (not isinstance(expected, dict) or not isinstance(folder, dict) or
            any(name not in expected or name in folder for name in reports)):
        raise ValueError(f'{path}:{line_no}: every json_report_path must name an expected new report file')
    if (not isinstance(contexts, list) or not contexts or
            any(not isinstance(name, str) for name in contexts) or
            len(set(contexts)) != len(contexts) or
            any(not name or name.startswith('/') or '\\' in name or '\x00' in name or
                re.match(r'^[A-Za-z]:', name) or any(part in {'', '.', '..'} for part in name.split('/')) or
                name not in folder for name in contexts)):
        raise ValueError(f'{path}:{line_no}: rationale_context_paths must name existing source files in folder_files')


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
            for field in ('folder_files', 'expected_files'):
                if field not in row:
                    continue
                files = row[field]
                if not isinstance(files, dict) or any(
                    not isinstance(name, str) or not name or name.startswith('/') or '\\' in name or '\x00' in name
                    or re.match(r'^[A-Za-z]:', name)
                    or any(part in {'', '.', '..'} for part in name.split('/'))
                    or not isinstance(value, str) for name, value in files.items()
                ):
                    raise ValueError(f'{path}:{line_no}: {field} must map safe relative paths to text')
            if 'expected_files' in row and 'folder_files' not in row:
                raise ValueError(f'{path}:{line_no}: expected_files requires folder_files')
            _validate_json_semantic_files_oracle(row, path, line_no)
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
            elif isinstance(oracle, dict) and oracle.get('level') == 'normalized' and oracle.get('normalization') == 'semantic-structured':
                comparison = oracle.get('comparison_spec')
                if (not isinstance(comparison, dict) or comparison.get('schema_version') != 'semantic-structured-comparison/1' or
                        any(not isinstance(comparison.get(field), list) for field in ('unordered_paths', 'ordered_paths', 'prose_paths')) or
                        not comparison['prose_paths'] or not isinstance(comparison.get('prose_policy'), str) or
                        not comparison['prose_policy'].strip()):
                    raise ValueError(f'{path}:{line_no}: semantic-structured oracle requires complete comparison_spec')
                pointers = [*comparison['unordered_paths'], *comparison['ordered_paths'], *comparison['prose_paths']]
                if (any(not isinstance(pointer, str) or not pointer.startswith('/') or '\x00' in pointer or
                        re.search(r'~(?![01])', pointer) for pointer in pointers) or len(set(pointers)) != len(pointers)):
                    raise ValueError(f'{path}:{line_no}: semantic-structured oracle paths must be unique valid JSON pointers')
                context = oracle.get('context')
                context_files = context.get('source_files') if isinstance(context, dict) else None
                folder_files = row.get('folder_files', {})
                if (not isinstance(context_files, dict) or not context_files or
                        any(not isinstance(name, str) or not isinstance(content, str) or
                            folder_files.get(name) != content for name, content in context_files.items())):
                    raise ValueError(f'{path}:{line_no}: semantic-structured oracle context must bind source text from folder_files')
            else:
                raise ValueError(f'{path}:{line_no}: oracle must be exact, independently judged, or a validated semantic-structured oracle')
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
                service_bindings: dict[str, dict[str, str]], world_services: dict[str, dict[str, str]] | None = None) -> list[dict]:
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
                              ([['return_result', {'status': 'success', 'value': row['expected']}]]
                               if oracle == 'exact' else [])},
                'rl': {'level': int(row.get('level', 2)), 'family': f'ext_natlang_codebase_{slug(program_id, 36)}_{slug(row["task_family"], 30)}',
                       'basis': 'independent-codebase-world-fixture'},
                'external': {'source': 'natlang-codebase-intake', 'upstream': dataset['identity'],
                             'upstream_id': row['id'], 'upstream_split': row['split'],
                             'unit': f'{program_id}:{row["group_id"]}', 'shape': f'codebase-{slug(row["task_family"], 44)}',
                             'task_family': row['task_family'], 'failure_mode': row.get('failure_mode'),
                             'authored_domain': row.get('domain') or metadata.get('domain'),
                             'quality_status': row.get('quality_status', 'ready'),
                             'quality_reason': row.get('quality_reason'),
                             'operation': root, 'program_id': program_id, 'program_revision': package_revision,
                             'invocation_index': row.get('invocation_index'), 'entity_model_sha256': _sha(json.dumps(entity_model, sort_keys=True).encode()),
                             'workflow_ids': row.get('workflow_ids', []), 'instruction_sha256': text_digest(row['task']),
                             'external_services': service_bindings,
                             'checks': {'wrong_code': row['wrong_code']} if row.get('wrong_code') else {},
                             'oracle_kind': ('exact' if oracle == 'exact' else
                                 'semantic-structured' if isinstance(oracle, dict) and
                                 oracle.get('normalization') == 'semantic-structured' else 'judged')},
            },
            'semantics': {'root': root, 'files': case_files, 'inputs': row['inputs'], 'expected': row['expected'],
                          'oracle': 'exact' if oracle == 'exact' else oracle},
        }
        if external_services:
            case['semantics']['services'] = {**external_services, **(world_services or {}).get(row['id'], {})}
        # Optional source-bound capability partitions are copied verbatim into the runtime contract.  The TS
        # collector validates that each alias only names exports from its bound external source.
        for contract_field in ('service_aliases', 'service_scopes', 'report_completeness'):
            if contract_field in metadata:
                case['semantics'][contract_field] = metadata[contract_field]
        if metadata.get('collector_controller'):
            controller = metadata['collector_controller']
            root_dir = root[:-3] if root.endswith('.nl') else ''
            if root_dir + '/' + controller['operations'] + '.nl' not in package:
                raise ValueError(f'{row["id"]}: collector controller operations are not a child of {root}')
            frontmatter = case_files[root].split('---', 2)[1]
            if not re.search(rf'\buses\s*:\s*\[[^]]*\b{re.escape(controller["module"])}\b[^]]*\]', frontmatter):
                raise ValueError(f'{row["id"]}: operation root must expose controller module with uses:')
            case['semantics']['application_controller'] = controller
        for field in ('folder_files', 'expected_files'):
            if field in row:
                case['semantics'][field] = row[field]
        # Preserve the file-specific oracle contract from an authored world.
        # In particular, judged rewrites need their own rubric so the runtime
        # can judge newly created output files against the bound source folder.
        if 'files_oracle' in row:
            case['semantics']['files_oracle'] = row['files_oracle']
        cases.append(case)
    return cases


def _validate_runtime_contracts(package: dict[str, str], metadata: dict[str, Any],
                                external_services: dict[str, str]) -> None:
    if metadata.get('execution_adapter') is not None:
        raise ValueError('metadata.execution_adapter is not invoked by the batch collector; declare collector_controller instead')
    controller = metadata.get('collector_controller')
    if controller is not None:
        if (not isinstance(controller, dict) or controller.get('schema') != 'natlang.application-controller/1' or
                any(not isinstance(controller.get(field), str) or
                    not re.fullmatch(r'[A-Za-z_$][A-Za-z0-9_$]*', controller[field])
                    for field in ('module', 'export', 'operations')) or
                not isinstance(controller.get('input_arguments'), list) or not controller['input_arguments'] or
                any(not isinstance(name, str) or not re.fullmatch(r'[A-Za-z_$][A-Za-z0-9_$]*', name)
                    for name in controller['input_arguments']) or
                len(set(controller['input_arguments'])) != len(controller['input_arguments'])):
            raise ValueError('metadata.collector_controller must satisfy natlang.application-controller/1')
        module_path = controller['module'] + '.ts'
        if module_path not in package or not re.search(
                rf'\bexport\s+(?:async\s+)?function\s+{re.escape(controller["export"])}\s*\(', package[module_path]):
            raise ValueError('collector_controller must name an exported function in a packaged TypeScript module')
    aliases = metadata.get('service_aliases')
    if aliases is not None:
        if not isinstance(aliases, dict) or not aliases:
            raise ValueError('metadata.service_aliases must be a non-empty namespace map')
        alias_names = set()
        source_methods: dict[str, set[str]] = {}
        for alias, spec in aliases.items():
            if not isinstance(alias, str) or not re.fullmatch(r'[A-Za-z_$][A-Za-z0-9_$]*', alias) or not isinstance(spec, dict):
                raise ValueError(f'invalid service alias entry: {alias!r}')
            source, methods = spec.get('source'), spec.get('methods')
            if source not in external_services or not isinstance(methods, list) or not methods or \
                    any(not isinstance(method, str) or not re.fullmatch(r'[A-Za-z_$][A-Za-z0-9_$]*', method) for method in methods) or \
                    len(set(methods)) != len(methods):
                raise ValueError(f'invalid source or method list for service alias {alias!r}')
            source_methods.setdefault(source, set()).update(methods)
            alias_names.add(alias)
        for source, methods in source_methods.items():
            for method in methods:
                if not re.search(rf'\bexport\s+(?:async\s+)?function\s+{re.escape(method)}\s*\(', external_services[source]):
                    raise ValueError(f'service alias method {source}.{method} is not an exported host function')
    else:
        alias_names = set(external_services)

    scopes = metadata.get('service_scopes')
    if scopes is not None:
        if not isinstance(scopes, dict):
            raise ValueError('metadata.service_scopes must map service namespaces to callable paths')
        for service, paths in scopes.items():
            if service not in alias_names or not isinstance(paths, list) or not paths or \
                    any(not isinstance(path, str) for path in paths) or len(set(paths)) != len(paths):
                raise ValueError(f'invalid service scope for namespace {service!r}')
            for path in paths:
                if not isinstance(path, str) or not path.endswith('.nl') or path not in package:
                    raise ValueError(f'service scope {service!r} names a missing Natlang function: {path!r}')

    gate = metadata.get('report_completeness')
    if gate is not None:
        required = ('mutating_methods', 'report_effects_path', 'checkpoint_effects_path',
                    'report_diagnosis_path', 'diagnosis_function')
        if not isinstance(gate, dict) or gate.get('schema') not in {'natlang.report-completeness/1', 'natlang.report-completeness/2'} or \
                any(field not in gate for field in required):
            raise ValueError('metadata.report_completeness must satisfy a supported natlang.report-completeness contract')
        if gate['schema'] == 'natlang.report-completeness/2':
            receipt_path = gate.get('effect_receipt_path')
            if not isinstance(receipt_path, str) or not re.fullmatch(r'[A-Za-z_$][A-Za-z0-9_$]*(?:\.[A-Za-z_$][A-Za-z0-9_$]*)*', receipt_path):
                raise ValueError('report_completeness v2 requires a dotted effect_receipt_path')
        methods = gate['mutating_methods']
        if not isinstance(methods, list) or any(not isinstance(method, str) or not method for method in methods) or \
                len(set(methods)) != len(methods):
            raise ValueError('report_completeness.mutating_methods must be unique non-empty strings')
        exposed = {method for spec in aliases.values() for method in spec['methods']} if aliases else set()
        if not set(methods).issubset(exposed):
            raise ValueError('report_completeness methods must be exposed through the declared aliases')
        if gate['diagnosis_function'] not in package or not gate['diagnosis_function'].endswith('.nl'):
            raise ValueError('report_completeness.diagnosis_function must name a package Natlang function')
        for field in ('report_effects_path', 'checkpoint_effects_path', 'report_diagnosis_path'):
            value = gate[field]
            if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z_$][A-Za-z0-9_$]*(?:\.[A-Za-z_$][A-Za-z0-9_$]*)*', value):
                raise ValueError(f'report_completeness.{field} must be a dotted value path')


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
    reject_uninvoked_execution_adapter(metadata)
    runtime_dependency_trees = verify_runtime_trees(metadata, package_root.parent)
    files = _project(package_root, metadata)
    child_call_visibility = _validate_natlang_child_call_visibility(files)
    external_services, service_bindings = load_external_services(package_root, metadata)
    _validate_runtime_contracts(files, metadata, external_services)
    world_rows = _worlds(package_root, metadata)
    controller = metadata.get('collector_controller')
    controller_frontend_receipts = []
    world_services = bind_world_services(package_root, metadata, external_services, service_bindings)
    fixture_aliases = {alias for alias, binding in service_bindings.items() if 'fixtures_path' in binding}
    if fixture_aliases and (set(world_services) != {row['id'] for row in world_rows} or
                           any(set(bound) != fixture_aliases for bound in world_services.values())):
        raise ValueError('fixture-bound services must cover exactly all admitted worlds')
    service_set_sha = _sha(json.dumps(service_bindings, sort_keys=True, separators=(',', ':')).encode())
    package_revision = metadata['source_revision'] + (f'+services:{service_set_sha}' if service_bindings else '')
    cases = _make_cases(metadata['program_id'], package_revision, files, metadata, world_rows,
                        external_services, service_bindings, world_services)
    if controller is not None:
        from scripts.rl_tasks.application_controller_frontend import validate_controller_frontend
        roots = sorted({row['operation'] for row in world_rows})
        if not roots:
            raise ValueError('collector_controller requires at least one reviewed invocation root')
        for root in roots:
            controller_frontend_receipts.append(validate_controller_frontend(files, root, controller))
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
        'natlang_child_call_visibility': child_call_visibility,
        **({'controller_frontend_preflight': controller_frontend_receipts}
           if controller_frontend_receipts else {}),
        'runtime_dependency_trees': runtime_dependency_trees,
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
                          'judged': sum(1 for row in world_rows if isinstance(row['oracle'], dict) and
                                        row['oracle'].get('level') == 'judged'),
                          'semantic_structured': sum(1 for row in world_rows if isinstance(row['oracle'], dict) and
                                        row['oracle'].get('normalization') == 'semantic-structured')},
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
