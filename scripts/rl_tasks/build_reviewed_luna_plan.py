#!/usr/bin/env python3
"""Prepare, but never launch, a reviewed six-slot Luna plan from verified cases.

The planner admits only train rows already present in a verifier-produced
cases.jsonl, rechecks each candidate against current ProtectedWorlds, samples
balanced operation × upstream-world cells, writes the exact row inventory,
and emits an unapproved plan draft. A parent must review and approve it before
the reviewed dispatcher can start any provider request.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
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


def _verify_rows(source: Path, verification_path: Path) -> tuple[list[dict], dict]:
    rows = _rows(source)
    if not rows:
        raise ValueError('verified source has no cases')
    verification = json.loads(verification_path.read_text(encoding='utf-8'))
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
    return groups[0]


def select_balanced_train(rows: list[dict], *, max_cases: int, seed: int,
                          protected: ProtectedWorlds) -> tuple[list[dict], list[dict]]:
    candidates, receipts = [], []
    for row in rows:
        if row.get('split') != 'train':
            continue
        oracle = row.get('semantics', {}).get('oracle', 'exact')
        level = oracle if isinstance(oracle, str) else oracle.get('level') if isinstance(oracle, dict) else None
        if level == 'judged':
            raise ValueError(f'judged row is mixed into the exact source: {row.get("id")}')
        receipt = protected.check(row)
        receipts.append({'case_id': row['id'], 'source_groups': row['source_groups'], 'receipt': receipt})
        operation = (row.get('curriculum', {}).get('external') or {}).get('operation')
        if not isinstance(operation, str) or not operation.endswith('.nl'):
            raise ValueError(f'{row.get("id")}: actual operation path is required')
        candidates.append((operation, _primary_world_group(row), row))
    if not candidates:
        raise ValueError('verified source has no train rows after split filtering')
    if max_cases < 1:
        raise ValueError('max_cases must be positive')
    max_cases = min(max_cases, len(candidates))
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for operation, group, row in candidates:
        buckets[(operation, group)].append(row)
    for key, bucket in buckets.items():
        bucket.sort(key=lambda row: hashlib.sha256(f'{seed}|{key}|{row["id"]}'.encode()).hexdigest())
    operations = sorted({key[0] for key in buckets})
    groups_by_operation = {op: sorted(group for candidate_op, group in buckets if candidate_op == op)
                           for op in operations}
    selected = []
    group_positions = {operation: 0 for operation in operations}
    bucket_positions: Counter[tuple[str, str]] = Counter()
    while len(selected) < max_cases:
        advanced = False
        for operation in operations:
            groups = groups_by_operation[operation]
            start = group_positions[operation]
            for offset in range(len(groups)):
                position = (start + offset) % len(groups)
                group = groups[position]
                key = (operation, group)
                bucket = buckets[key]
                candidate_index = bucket_positions[key]
                if candidate_index < len(bucket):
                    selected.append(bucket[candidate_index])
                    bucket_positions[key] += 1
                    group_positions[operation] = (position + 1) % len(groups)
                    advanced = True
                    if len(selected) == max_cases:
                        break
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
    if ROOT / 'runs' not in campaign_root.parents:
        raise ValueError('campaign root must be below this checkout runs/')
    if campaign_root.exists():
        raise ValueError(f'campaign root already exists: {campaign_root}')
    rows, verify_record = _verify_rows(source, verification)
    program_ids = {(row.get('curriculum', {}).get('external') or {}).get('program_id') for row in rows}
    program_ids.discard(None)
    if len(program_ids) > 1:
        raise ValueError('one reviewed campaign source must name one first-class codebase')
    intake_record = json.loads(intake_receipt.read_text(encoding='utf-8')) if intake_receipt else None
    if intake_record and not intake_record.get('admission_eligible'):
        raise ValueError('codebase dataset has an unresolved rights hold')
    protected = protected or ProtectedWorlds.load()
    selected, selected_receipts = select_balanced_train(rows, max_cases=max_cases, seed=seed, protected=protected)
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
    if diversity_manifest:
        artifacts.append(diversity_manifest.resolve(strict=True))
    artifact_hashes = {str(path.resolve(strict=True)): _sha(path.resolve(strict=True))
                       for path in artifacts if path.is_file()}
    plan = {
        'schema': 'natlang.reviewed_luna_dispatch_plan/2', 'root_approved': False,
        'campaign_id': campaign_id, 'cwd': str(ROOT), 'supervisor': str(supervisor_path),
        'supervisor_sha256': artifact_hashes[str(supervisor_path)],
        'dispatcher_sha256': artifact_hashes[str(dispatcher_path)],
        'source': str(selected_source.resolve()), 'source_sha256': artifact_hashes[str(selected_source.resolve())],
        'runtime': str(runtime), 'runtime_receipt_sha256': artifact_hashes[str(runtime_receipt.resolve(strict=True))],
        'artifact_hashes': artifact_hashes,
        'source_inventory': {'path': str(inventory_output.resolve()), 'sha256': artifact_hashes[str(inventory_output.resolve())]},
        'campaign_root': str(campaign_root), 'claim_ledger': str(campaign_root / 'dispatch/claims.jsonl'),
        'launch_record': str(campaign_root / 'dispatch/launch.json'), 'authority': str(authority),
        'provider': 'openai-codex', 'model_id': 'gpt-6-luna', 'model_concurrency': 1,
        'slots': 6, 'execution_plans': True, 'reasoning_effort': 'low',
        'prompt_rendering': 'full', 'text_neuralese_emulation': False,
        'max_transport_retries': 1, 'max_turns': 60, 'max_model_requests': 384,
        'context_tokens': 32768, 'case_seconds': 1800, 'min_free_mib': 1024,
        'cases': cases,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, ensure_ascii=False, sort_keys=True, indent=2) + '\n', encoding='utf-8')
    plan_sha = _sha(output)
    counts_operation = Counter((row['curriculum']['external']['operation'] for row in selected))
    counts_world = Counter(_primary_world_group(row) for row in selected)
    return {
        'schema': 'natlang.luna-six-slot-plan-preparation/1', 'status': 'prepared_unapproved',
        'plan': str(output.resolve()), 'plan_sha256': plan_sha,
        'case_source': str(selected_source.resolve()), 'case_source_sha256': _sha(selected_source),
        'verification': {'path': str(verification), 'sha256': _sha(verification), 'source_sha256': verify_record['cases_sha256']},
        'protected_worlds_digest': protected.digest, 'checked_train_rows': protection_record['checked_train_rows'],
        'selected_rows': len(selected), 'operation_counts': dict(counts_operation),
        'world_group_counts': dict(counts_world), 'source_groups_preserved': True,
        'prompt_rendering': 'full', 'slots': 6, 'root_approved': False,
        'dispatch_command_after_review': f'python3 scripts/run_reviewed_luna_dispatcher.py {output.resolve()} --sha256 {plan_sha}',
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
    parser.add_argument('--max-cases', type=int, default=42)
    parser.add_argument('--seed', type=int, default=20261031)
    args = parser.parse_args()
    receipt = build_plan(source=args.source, verification=args.verification, runtime=args.runtime,
                         authority=args.authority, campaign_id=args.campaign_id,
                         campaign_root=args.campaign_root, output=args.output,
                         inventory_output=args.inventory_output, selected_source=args.selected_source,
                         protection_receipt=args.protection_receipt, max_cases=args.max_cases,
                         seed=args.seed, intake_receipt=args.intake_receipt,
                         diversity_manifest=args.diversity_manifest)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))


if __name__ == '__main__':
    main()
