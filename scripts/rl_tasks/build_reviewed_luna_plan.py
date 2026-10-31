#!/usr/bin/env python3
"""Prepare, but never launch, a reviewed six-slot Luna plan from verified cases.

The planner admits only train rows already present in a verifier-produced
cases.jsonl, rechecks each candidate against current ProtectedWorlds, samples
balanced codebase × operation × upstream-world cells, writes the exact row inventory,
and emits an unapproved plan draft. A parent must review and approve it before
the reviewed dispatcher can start any provider request.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / 'scripts') not in sys.path:
    sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'training' / 'neuralese'))

from freeze_training_runtime import tree_identity  # noqa: E402
from luna_source_inventory import inventory  # noqa: E402
from natlang_neuralese.common.custody import ProtectedWorlds, REGISTRY  # noqa: E402
from natlang_neuralese.common.hashing import sha256_file_hex as sha_file  # noqa: E402
from scripts.rl_tasks.execution_adapter import verify_runtime_tree  # noqa: E402
import run_bonsai_queue  # noqa: E402
import run_reviewed_luna_dispatcher as dispatcher  # noqa: E402


def _sha(path: Path) -> str:
    return sha_file(path)


def _rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f'{path}:{line_no}: expected an object')
        rows.append(row)
    return rows


def _verify_rows(source: Path, verification_path: Path, *, intake_receipt: dict | None = None) -> tuple[list[dict], dict]:
    rows = _rows(source)
    if not rows:
        raise ValueError('verified source has no cases')
    verification = json.loads(verification_path.read_text(encoding='utf-8'))
    levels = [row.get('semantics', {}).get('oracle') for row in rows]
    judged_only = all(value == 'judged' or isinstance(value, dict) and value.get('level') == 'judged'
                      for value in levels)
    if verification.get('schema') == 'natlang.judged-source-qualification/1':
        if (not judged_only or not isinstance(intake_receipt, dict) or
                verification.get('cases_sha256') != _sha(source) or
                verification.get('judged_cases_path') != str(source.resolve()) or
                verification.get('judged_cases_sha256') != intake_receipt.get('judged_cases_sha256') or
                verification.get('intake_receipt_sha256') != hashlib.sha256(
                    json.dumps(intake_receipt, sort_keys=True, separators=(',', ':')).encode()).hexdigest() or
                verification.get('behaviorally_verified') is not False or
                intake_receipt.get('judged_cases_path') != str(source.resolve()) or
                intake_receipt.get('judged_cases_sha256') != _sha(source) or
                intake_receipt.get('admission_eligible') is not True):
            raise ValueError('judged source qualification does not bind its source and cleared intake receipt')
        return rows, verification
    if verification.get('schema') != 'natlang.broad-rl-case-verification/1':
        raise ValueError('unsupported source verification receipt')
    if verification.get('cases_sha256') != _sha(source):
        raise ValueError('verification receipt does not bind this cases.jsonl')
    by_id = {result.get('id'): result for result in verification.get('results', [])}
    for row in rows:
        result = by_id.get(row.get('id'))
        if not result or result.get('ok') is not True:
            raise ValueError(f'case is not an independently verified kept result: {row.get("id")}')
    return rows, verification


def _primary_world_group(row: dict) -> str:
    groups = row.get('source_groups')
    if not isinstance(groups, list) or not groups or not all(isinstance(x, str) and x for x in groups):
        raise ValueError(f'{row.get("id")}: complete source_groups are required')
    # The exact upstream group is kept in the row and plan inventory. This key
    # is only used to balance selection; it never rewrites group identities.
    return next((group for group in groups if not group.startswith(('codebase:', 'world:'))), groups[0])


def _intake_execution_adapter_bindings(intake_receipt: dict | None) -> list[tuple[Path, str]]:
    """Return and revalidate the host files authenticated by codebase intake.

    Intake-time validation alone is insufficient: the executable adapter can
    change between intake and campaign planning. These paths become ordinary
    reviewed-plan artifact pins, so dispatcher validation checks them again
    immediately before launch.
    """
    if intake_receipt is None:
        return []
    bindings = intake_receipt.get('execution_adapter_bindings', [])
    if not isinstance(bindings, list):
        raise ValueError('intake execution_adapter_bindings must be a list')
    checked: list[tuple[Path, str]] = []
    seen: set[Path] = set()
    for binding in bindings:
        if not isinstance(binding, dict):
            raise ValueError('invalid intake execution adapter binding')
        path_value, expected = binding.get('path'), binding.get('sha256')
        if not isinstance(path_value, str) or not isinstance(expected, str):
            raise ValueError('incomplete intake execution adapter binding')
        path = Path(path_value).expanduser().resolve(strict=True)
        if path in seen or not path.is_file() or _sha(path) != expected:
            raise ValueError(f'Execution adapter changed since intake: {path}')
        seen.add(path)
        checked.append((path, expected))
    return checked


def _intake_execution_adapter_trees(intake_receipt: dict | None) -> list[dict]:
    if intake_receipt is None:
        return []
    trees = intake_receipt.get('execution_adapter_trees', [])
    if not isinstance(trees, list):
        raise ValueError('intake execution_adapter_trees must be a list')
    checked = []
    for record in trees:
        if not isinstance(record, dict):
            raise ValueError('invalid intake execution adapter tree binding')
        actual = verify_runtime_tree(root=Path(record.get('root', '')),
                                     manifest_path=Path(record.get('manifest_path', '')),
                                     manifest_sha256=record.get('manifest_sha256', ''))
        for field in ('tree_sha256', 'file_count', 'total_bytes', 'files'):
            if actual[field] != record.get(field):
                raise ValueError(f'Execution adapter runtime tree changed since intake: {record.get("root")}')
        checked.append({key: record[key] for key in
                        ('name', 'root', 'manifest_path', 'manifest_sha256', 'tree_sha256', 'file_count', 'total_bytes')})
    return checked


def _program_id(row: dict) -> str:
    identifier = (row.get('curriculum', {}).get('external') or {}).get('program_id')
    return identifier if isinstance(identifier, str) and identifier else '__unknown_codebase__'


def select_balanced_train(rows: list[dict], *, max_cases: int, seed: int,
                          protected: ProtectedWorlds,
                          include_judged: bool = False,
                          case_ids: set[str] | None = None,
                          quality_exclusions_out: list[dict] | None = None) -> tuple[list[dict], list[dict]]:
    candidates, receipts, seen_ids, quality_exclusions = [], [], set(), {}
    for row in rows:
        if row.get('split') != 'train':
            continue
        oracle = row.get('semantics', {}).get('oracle', 'exact')
        level = oracle if isinstance(oracle, str) else oracle.get('level') if isinstance(oracle, dict) else None
        if level == 'judged' and not include_judged:
            raise ValueError(f'judged row is mixed into the exact source: {row.get("id")}')
        if row.get('id') in seen_ids:
            raise ValueError(f'duplicate training case identity: {row.get("id")}')
        seen_ids.add(row.get('id'))
        receipt = protected.check(row)
        external = (row.get('curriculum', {}).get('external') or {})
        quality_status = external.get('quality_status', 'ready')
        if quality_status != 'ready':
            exclusion = {'case_id': row['id'], 'quality_status': quality_status,
                         'quality_reason': external.get('quality_reason'),
                         'source_groups': row['source_groups'], 'receipt': receipt}
            quality_exclusions[row['id']] = exclusion
            if quality_exclusions_out is not None:
                quality_exclusions_out.append(exclusion)
            receipts.append({**exclusion, 'excluded': True})
            continue
        receipts.append({'case_id': row['id'], 'source_groups': row['source_groups'], 'receipt': receipt,
                         'quality_status': 'ready', 'excluded': False})
        operation = (row.get('curriculum', {}).get('external') or {}).get('operation')
        if not isinstance(operation, str) or not operation.endswith('.nl'):
            raise ValueError(f'{row.get("id")}: actual operation path is required')
        candidates.append((_program_id(row), operation, _primary_world_group(row), row))
    if not candidates:
        raise ValueError('verified source has no train rows after split filtering')
    if case_ids is not None:
        pending_requested = sorted(case_ids & quality_exclusions.keys())
        if pending_requested:
            reasons = [quality_exclusions[item] for item in pending_requested]
            raise ValueError(f'explicit case-id selection includes quality-pending rows: {reasons}')
        by_id = {row['id']: (program, operation, group, row)
                 for program, operation, group, row in candidates}
        missing = sorted(case_ids - by_id.keys())
        if missing:
            raise ValueError(f'case-id selector did not match public TRAIN rows: {missing}')
        candidates = [by_id[case_id] for case_id in sorted(case_ids)]
    if max_cases < 1:
        raise ValueError('max_cases must be positive')
    max_cases = min(max_cases, len(candidates))
    buckets: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for program, operation, group, row in candidates:
        buckets[(program, operation, group)].append(row)
    for key, bucket in buckets.items():
        bucket.sort(key=lambda row: hashlib.sha256(f'{seed}|{key}|{row["id"]}'.encode()).hexdigest())
    programs = sorted({key[0] for key in buckets})
    operations = {program: deque(sorted({key[1] for key in buckets if key[0] == program}))
                  for program in programs}
    groups = {(program, operation): deque(sorted(key[2] for key in buckets
              if key[:2] == (program, operation)))
              for program in programs for operation in operations[program]}
    positions: Counter[tuple[str, str, str]] = Counter()
    selected = []
    # First rotate codebases, then each codebase's operations and upstream worlds.
    # A package with many public roots must not drown out a different application.
    while len(selected) < max_cases:
        advanced = False
        for program in programs:
            found = False
            for _ in range(len(operations[program])):
                operation = operations[program][0]
                operations[program].rotate(-1)
                worlds = groups[(program, operation)]
                for _ in range(len(worlds)):
                    group = worlds[0]
                    worlds.rotate(-1)
                    key = (program, operation, group)
                    index = positions[key]
                    if index < len(buckets[key]):
                        selected.append(buckets[key][index])
                        positions[key] += 1
                        advanced = found = True
                        break
                if found:
                    break
            if len(selected) == max_cases:
                break
        if not advanced:
            break
    # Every app/world candidate was custody-checked above. Repeated invocations
    # share their exact upstream source groups, so no synthetic alias is created.
    selected_ids = {row['id'] for row in selected}
    selected_receipts = [receipt for receipt in receipts if receipt['case_id'] in selected_ids]
    return selected, selected_receipts


def build_plan(*, source: Path, verification: Path, runtime: Path, authority: Path,
               campaign_id: str, campaign_root: Path, output: Path, inventory_output: Path,
               selected_source: Path, protection_receipt: Path, max_cases: int = 42,
               seed: int = 20261031, intake_receipt: Path | None = None,
               diversity_manifest: Path | None = None,
               case_ids: set[str] | None = None,
               judge_model: dict[str, str] | None = None,
               protected: ProtectedWorlds | None = None) -> dict:
    source = source.resolve(strict=True)
    verification = verification.resolve(strict=True)
    runtime = runtime.resolve(strict=True)
    runtime_receipt = runtime / 'frozen-runtime.json'
    if not runtime_receipt.is_file():
        raise ValueError(f'frozen runtime receipt missing: {runtime_receipt}')
    runtime_manifest = json.loads(runtime_receipt.read_text(encoding='utf-8'))
    if tree_identity(runtime) != runtime_manifest.get('files'):
        raise ValueError('frozen runtime tree differs from its receipt')
    authority = authority.expanduser().resolve(strict=True)
    campaign_root = campaign_root.expanduser().resolve()
    storage_root = dispatcher._campaign_storage_root(campaign_root)
    if campaign_root.exists():
        raise ValueError(f'campaign root already exists: {campaign_root}')
    intake_record = json.loads(intake_receipt.read_text(encoding='utf-8')) if intake_receipt else None
    rows, verify_record = _verify_rows(source, verification, intake_receipt=intake_record)
    if intake_record and not intake_record.get('admission_eligible'):
        raise ValueError('codebase dataset has an unresolved rights hold')
    protected = protected or ProtectedWorlds.load()
    judged_train = [row for row in rows if row.get('split') == 'train' and
                    (row.get('curriculum', {}).get('external') or {}).get('quality_status', 'ready') == 'ready' and
                    (row.get('semantics', {}).get('oracle') == 'judged' or
                     isinstance(row.get('semantics', {}).get('oracle'), dict) and
                     row['semantics']['oracle'].get('level') == 'judged')]
    if judged_train and judge_model is None:
        raise ValueError('judged train rows require an explicit independent judge_model')
    if judge_model is not None:
        run_bonsai_queue.judge_arguments({'judge_model': judge_model}, 'gpt-6-luna')
    quality_exclusions = []
    selected, selected_receipts = select_balanced_train(
        rows, max_cases=max_cases, seed=seed, protected=protected,
        include_judged=judge_model is not None, case_ids=case_ids,
        quality_exclusions_out=quality_exclusions)
    selected_source.parent.mkdir(parents=True, exist_ok=True)
    selected_source.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in selected), encoding='utf-8')
    cases = [{'index': index, 'seed': int(hashlib.sha256(f'{campaign_id}|{seed}|{row["id"]}'.encode()).hexdigest()[:8], 16)}
             for index, row in enumerate(selected)]
    inventory_output.parent.mkdir(parents=True, exist_ok=True)
    inventory_data = inventory(selected_source, cases)
    inventory_output.write_text(json.dumps(inventory_data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    protection_record = {
        'schema': 'natlang.luna-codebase-protection-preflight/1',
        'observed_at_source': str(REGISTRY), 'protected_worlds_digest': protected.digest,
        'registry_sha256': _sha(ROOT / REGISTRY), 'checked_train_rows': len(rows) - sum(1 for row in rows if row.get('split') != 'train'),
        'selected_train_rows': len(selected), 'selected_receipts': selected_receipts,
        'quality_excluded_train_rows': quality_exclusions,
        'all_candidate_train_rows_checked_before_balanced_selection': True,
    }
    protection_receipt.parent.mkdir(parents=True, exist_ok=True)
    protection_receipt.write_text(json.dumps(protection_record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    dispatcher_path = dispatcher.DISPATCHER.resolve()
    supervisor_path = run_bonsai_queue.__file__ and Path(run_bonsai_queue.__file__).resolve()
    artifacts = [dispatcher_path, supervisor_path, source, verification, selected_source, inventory_output,
                 runtime_receipt, ROOT / REGISTRY, ROOT / 'training/neuralese/natlang_neuralese/common/custody.py',
                 Path(__file__).resolve(), Path(__file__).with_name('intake_natlang_codebase.py'),
                 Path(__file__).with_name('audit_natlang_packages.py'), verification.parent / 'verification.json',
                 protection_receipt]
    if intake_receipt:
        artifacts.append(intake_receipt.resolve(strict=True))
        adapter_bindings = _intake_execution_adapter_bindings(intake_record)
        artifacts.extend(path for path, _ in adapter_bindings)
        adapter_trees = _intake_execution_adapter_trees(intake_record)
        artifacts.extend(Path(tree['manifest_path']) for tree in adapter_trees)
    if diversity_manifest:
        artifacts.append(diversity_manifest.resolve(strict=True))
    artifact_hashes = {str(path.resolve(strict=True)): _sha(path.resolve(strict=True))
                       for path in artifacts if path.is_file()}
    if intake_record:
        for adapter_path, expected_sha in _intake_execution_adapter_bindings(intake_record):
            if artifact_hashes.get(str(adapter_path)) != expected_sha:
                raise ValueError(f'Execution adapter changed while the reviewed plan was being assembled: {adapter_path}')
    plan = {
        'schema': 'natlang.reviewed_luna_dispatch_plan/2', 'root_approved': False,
        'campaign_id': campaign_id, 'cwd': str(ROOT), 'supervisor': str(supervisor_path),
        'supervisor_sha256': artifact_hashes[str(supervisor_path)],
        'dispatcher_sha256': artifact_hashes[str(dispatcher_path)],
        'source': str(selected_source.resolve()), 'source_sha256': artifact_hashes[str(selected_source.resolve())],
        'runtime': str(runtime), 'runtime_receipt_sha256': artifact_hashes[str(runtime_receipt.resolve(strict=True))],
        'artifact_hashes': artifact_hashes,
        'execution_adapter_trees': _intake_execution_adapter_trees(intake_record),
        'source_inventory': {'path': str(inventory_output.resolve()), 'sha256': artifact_hashes[str(inventory_output.resolve())]},
        'campaign_root': str(campaign_root), 'claim_ledger': str(campaign_root / 'dispatch/claims.jsonl'),
        'storage_root': str(storage_root),
        'launch_record': str(campaign_root / 'dispatch/launch.json'), 'authority': str(authority),
        'provider': 'openai-codex', 'model_id': 'gpt-6-luna', 'model_concurrency': 1,
        'slots': 6, 'execution_plans': True, 'reasoning_effort': 'low',
        'resource_policy': {
            'schema': 'natlang.teacher_execution_resource_policy/1',
            'resumable': False, 'backfill': False,
            'partial_execution_is_resumable': False,
            'partial_execution_disposition': 'durable evidence only; no graph or simulator-state continuation',
            'campaign_resume': 'explicit only for unclaimed cases; uncertain claims are terminalized',
        },
        'prompt_rendering': 'full', 'text_neuralese_emulation': True,
        'max_transport_retries': 1, 'max_turns': 60, 'max_model_requests': 384,
        # Hosted teachers author text-backed Neuralese. Native student KV is
        # reconstructed during graph conversion/training, never claimed here.
        'context_tokens': 32768, 'case_seconds': 1800, 'min_free_mib': 40960,
        'cases': cases,
    }
    if judge_model is not None:
        plan['judge_model'] = judge_model
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, ensure_ascii=False, sort_keys=True, indent=2) + '\n', encoding='utf-8')
    plan_sha = _sha(output)
    counts_program = Counter(_program_id(row) for row in selected)
    counts_operation = Counter((row['curriculum']['external']['operation'] for row in selected))
    counts_world = Counter(_primary_world_group(row) for row in selected)
    return {
        'schema': 'natlang.luna-six-slot-plan-preparation/1', 'status': 'prepared_unapproved',
        'plan': str(output.resolve()), 'plan_sha256': plan_sha,
        'case_source': str(selected_source.resolve()), 'case_source_sha256': _sha(selected_source),
        'verification': {'path': str(verification), 'sha256': _sha(verification),
                         'source_sha256': verify_record['cases_sha256'],
                         'mode': verify_record.get('schema'),
                         'behaviorally_verified': verify_record.get('behaviorally_verified', True)},
        'protected_worlds_digest': protected.digest, 'checked_train_rows': protection_record['checked_train_rows'],
        'selected_rows': len(selected), 'program_counts': dict(counts_program),
        'rows_with_unknown_codebase': counts_program.get('__unknown_codebase__', 0),
        'operation_counts': dict(counts_operation),
        'world_group_counts': dict(counts_world), 'source_groups_preserved': True,
        'quality_excluded_rows': sum(1 for row in rows if row.get('split') == 'train' and
                                     (row.get('curriculum', {}).get('external') or {}).get('quality_status', 'ready') != 'ready'),
        'prompt_rendering': 'full', 'slots': 6, 'root_approved': False,
        'dispatch_command_after_review': (
            f'python3 scripts/memory_ledger.py run --unit {campaign_id} --budget-gb 2 --class collection '
            f'--priority low --no-resumable -- python3 scripts/run_reviewed_luna_dispatcher.py '
            f'{output.resolve()} --sha256 {plan_sha}'
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='verifier-produced cases.jsonl')
    parser.add_argument('--verification', type=Path, required=True, help='matching verification.json')
    parser.add_argument('--runtime', type=Path, required=True, help='frozen Node runtime directory')
    parser.add_argument('--authority', type=Path, required=True, help='current shared Luna authority file')
    parser.add_argument('--campaign-id', required=True)
    parser.add_argument('--campaign-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='unapproved plan draft')
    parser.add_argument('--inventory-output', type=Path, required=True)
    parser.add_argument('--selected-source', type=Path, required=True)
    parser.add_argument('--protection-receipt', type=Path, required=True)
    parser.add_argument('--intake-receipt', type=Path)
    parser.add_argument('--diversity-manifest', type=Path)
    parser.add_argument('--judge-model-id', help='distinct independent judge for judged train rows')
    judge_transport = parser.add_mutually_exclusive_group()
    judge_transport.add_argument('--judge-provider')
    judge_transport.add_argument('--judge-server')
    parser.add_argument('--max-cases', type=int, default=42)
    parser.add_argument('--case-id', action='append', dest='case_ids',
                        help='select an exact authenticated public TRAIN row ID; repeat for multiple rows')
    parser.add_argument('--seed', type=int, default=20261031)
    args = parser.parse_args()
    if (args.judge_provider or args.judge_server) and not args.judge_model_id:
        parser.error('--judge-provider/--judge-server require --judge-model-id')
    if args.judge_model_id and not (args.judge_provider or args.judge_server):
        parser.error('--judge-model-id requires exactly one judge transport')
    receipt = build_plan(source=args.source, verification=args.verification, runtime=args.runtime,
                         authority=args.authority, campaign_id=args.campaign_id,
                         campaign_root=args.campaign_root, output=args.output,
                         inventory_output=args.inventory_output, selected_source=args.selected_source,
                         protection_receipt=args.protection_receipt, max_cases=args.max_cases,
                         seed=args.seed, intake_receipt=args.intake_receipt,
                         diversity_manifest=args.diversity_manifest,
                         case_ids=set(args.case_ids) if args.case_ids else None,
                         judge_model=({'model_id': args.judge_model_id,
                                       **({'provider': args.judge_provider} if args.judge_provider else {}),
                                       **({'server': args.judge_server} if args.judge_server else {})}
                                      if args.judge_model_id else None))
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))


if __name__ == '__main__':
    main()
