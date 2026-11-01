"""Read-only continuity checks for completed pre-resource-policy Luna campaigns.

This path authenticates finished evidence for conversion only. It cannot create
or authorize a dispatcher plan, and it never fills absent launch-policy fields.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from natlang_neuralese.common.custody import ProtectedWorlds, REGISTRY
from natlang_neuralese.common.hashing import sha256_file_hex

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / 'scripts') not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / 'scripts'))
from scripts.run_reviewed_luna_dispatcher import (
    _verify_ledger_events, _verify_ledger_identity, digest,
    tree_identity, verify_luna_source_inventory,
)
from scripts.neuralese_data.portable_paths import node_executable

SCHEMA = 'natlang.historical-provider-source-continuity/1'


def _binding(path: Path) -> dict:
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha256_file_hex(path)}


def _same_binding(actual: dict, declared: dict, label: str) -> None:
    if (not isinstance(declared, dict) or declared.get('path') != actual['path']
            or declared.get('bytes') != actual['bytes'] or declared.get('sha256') != actual['sha256']):
        raise ValueError(f'{label} differs from its bound historical artifact')


def _historical_git_object(repo: Path, path: Path, expected: str) -> dict:
    try:
        relative = str(path.resolve(strict=False).relative_to(repo.resolve(strict=True)))
    except ValueError as exc:
        raise ValueError(f'changed non-repository historical artifact is unavailable: {path}') from exc
    commits = subprocess.run(['git', '-C', str(repo), 'log', '--all', '--format=%H', '--', relative],
                             capture_output=True, text=True, check=True).stdout.splitlines()
    for commit in commits:
        result = subprocess.run(['git', '-C', str(repo), 'show', f'{commit}:{relative}'],
                                capture_output=True)
        if result.returncode == 0 and hashlib.sha256(result.stdout).hexdigest() == expected:
            return {'mode': 'git-object', 'path': str(path.resolve(strict=False)),
                    'git_path': relative, 'git_commit': commit, 'sha256': expected}
    raise ValueError(f'no immutable Git object matches the historical plan pin: {path}')


def _native_row_digest(runtime: Path, runtime_manifest: dict, row: dict) -> tuple[str, str]:
    """Use the historical runtime's own TS row digest; Python JSON hashing differs."""
    relative = 'dist/teacher/native-materializer.js'
    module = runtime / relative
    expected = (runtime_manifest.get('files') or {}).get(relative)
    actual = sha256_file_hex(module)
    if not isinstance(expected, str) or actual != expected:
        raise ValueError('historical runtime materializer differs from its frozen file inventory')
    node = node_executable()
    code = ('import fs from "node:fs";'
            'const {nativeRowDigest}=await import(process.argv[1]);'
            'process.stdout.write(nativeRowDigest(JSON.parse(fs.readFileSync(0,"utf8"))));')
    completed = subprocess.run([node, '--input-type=module', '-e', code, module.as_uri()],
        input=json.dumps(row, ensure_ascii=False), text=True, capture_output=True,
        check=True, timeout=30)
    digest_value = completed.stdout.strip()
    if len(digest_value) != 64 or any(character not in '0123456789abcdef' for character in digest_value):
        raise ValueError('historical frozen materializer returned an invalid source-row digest')
    return digest_value, actual


def _artifact_evidence(plan: dict, repo: Path) -> list[dict]:
    pins = plan.get('artifact_hashes')
    if not isinstance(pins, dict) or not pins:
        raise ValueError('historical plan has no artifact hash closure')
    result = []
    for name, expected in sorted(pins.items()):
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f'invalid historical artifact digest for {name}')
        path = Path(name).expanduser().resolve(strict=False)
        if path.is_file() and sha256_file_hex(path) == expected:
            result.append({'mode': 'present-file', 'path': str(path), 'sha256': expected,
                           'bytes': path.stat().st_size})
        else:
            result.append(_historical_git_object(repo, path, expected))
    return result


def derive(provider_source_path: Path, *, repo: Path | None = None) -> dict:
    provider_source_path = Path(provider_source_path).resolve(strict=True)
    repo = Path(repo).resolve(strict=True) if repo is not None else Path(__file__).resolve().parents[2]
    declaration = json.loads(provider_source_path.read_text())
    if (declaration.get('schema') != 'natlang.provider-campaign-source/1'
            or declaration.get('training_admission') is not False
            or declaration.get('training_exposure') is not False
            or declaration.get('oracle_regrades') not in (None, [])
            or declaration.get('model') != 'gpt-6-luna'
            or declaration.get('provider') != 'openai-codex'):
        raise ValueError('historical import requires an unadmitted exact Luna provider-source declaration')
    source = Path(declaration['source']['path']).resolve(strict=True)
    _same_binding(_binding(source), declaration['source'], 'raw provider trajectories')
    plan_path = Path(declaration['plan']['path']).resolve(strict=True)
    _same_binding(_binding(plan_path), declaration['plan'], 'historical reviewed plan')
    plan = json.loads(plan_path.read_text())
    # Record the absence as history. No policy is supplied to the dispatcher or
    # to the imported plan object.
    if ('resource_policy' in plan or plan.get('schema') != 'natlang.reviewed_luna_dispatch_plan/2'
            or plan.get('root_approved') is not True
            or plan.get('model_id') != 'gpt-6-luna'
            or plan.get('provider') != 'openai-codex'
            or plan.get('execution_plans') is not True):
        raise ValueError('historical read-only import supports only the exact pre-resource-policy Luna plan shape')
    source_path = Path(plan['source']).resolve(strict=True)
    if (source_path != source and sha256_file_hex(source_path) != plan.get('source_sha256')):
        raise ValueError('historical plan source bytes do not match its declared source digest')
    if sha256_file_hex(source_path) != plan.get('source_sha256'):
        raise ValueError('historical plan source digest changed')
    runtime = Path(plan['runtime']).resolve(strict=True)
    runtime_receipt = runtime / 'frozen-runtime.json'
    if sha256_file_hex(runtime_receipt) != plan.get('runtime_receipt_sha256'):
        raise ValueError('historical frozen runtime receipt changed')
    runtime_manifest = json.loads(runtime_receipt.read_text())
    if tree_identity(runtime) != runtime_manifest.get('files'):
        raise ValueError('historical frozen runtime tree differs from its exact receipt')
    artifact_evidence = _artifact_evidence(plan, repo)
    if plan.get('runtime_receipt_sha256') != plan.get('artifact_hashes', {}).get(str(runtime_receipt)):
        raise ValueError('historical runtime receipt is not pinned by the original plan')

    cases = [json.loads(line) for line in source_path.read_text().splitlines() if line.strip()]
    inventory_ref = plan.get('source_inventory')
    if not isinstance(inventory_ref, dict):
        raise ValueError('historical plan lacks its source inventory')
    inventory_path = Path(inventory_ref['path']).resolve(strict=True)
    if (sha256_file_hex(inventory_path) != inventory_ref.get('sha256')
            or plan['artifact_hashes'].get(str(inventory_path)) != inventory_ref['sha256']):
        raise ValueError('historical source inventory changed or was not plan-pinned')
    inventory = verify_luna_source_inventory(inventory_path, source_path, plan['cases'])
    inventory_by_index = {item['index']: item for item in inventory['entries']}

    ledger_path = Path(declaration['claim_ledger']['path']).resolve(strict=True)
    _same_binding(_binding(ledger_path), declaration['claim_ledger'], 'closed claim ledger')
    events = [json.loads(line) for line in ledger_path.read_text().splitlines() if line.strip()]
    headers = [event for event in events if event.get('event') == 'campaign_open']
    if len(headers) != 1 or not isinstance(headers[0].get('identity'), dict):
        raise ValueError('historical claim ledger must have exactly one immutable campaign header')
    identity = headers[0]['identity']
    identity_hash = identity.get('identity_sha256')
    identity_body = {key: value for key, value in identity.items() if key != 'identity_sha256'}
    if hashlib.sha256(json.dumps(identity_body, sort_keys=True, separators=(',', ':')).encode()).hexdigest() != identity_hash:
        raise ValueError('historical campaign identity digest is invalid')
    expected_identity = {
        'campaign_id': plan['campaign_id'], 'campaign_root': plan['campaign_root'],
        'plan_path': str(plan_path), 'plan_sha256': digest(plan_path),
        'source': str(source_path), 'source_sha256': plan['source_sha256'],
        'runtime': str(runtime), 'runtime_receipt_sha256': plan['runtime_receipt_sha256'],
        'dispatcher_sha256': plan['dispatcher_sha256'], 'supervisor': plan['supervisor'],
        'supervisor_sha256': plan['supervisor_sha256'], 'authority': plan['authority'],
        'slots': plan['slots'],
    }
    if any(identity.get(key) != value for key, value in expected_identity.items()):
        raise ValueError('historical campaign header differs from the exact original plan identity')
    if 'resource_policy' in identity:
        raise ValueError('historical campaign header unexpectedly contains a resource policy')
    _verify_ledger_identity(events, identity)
    _verify_ledger_events(events, identity)

    raw_rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    if len(raw_rows) != len(declaration.get('results', [])) or not raw_rows:
        raise ValueError('historical provider snapshot does not enumerate every included trajectory')
    claims = {event['claim_id']: event for event in events if event.get('event') == 'claim'}
    terminals = {event['claim_id']: event for event in events if event.get('event') == 'terminal'}
    results = []
    checked_programs = []
    current_worlds = ProtectedWorlds.load()
    for row, entry in zip(raw_rows, declaration['results']):
        claim = claims.get(entry.get('claim_id'))
        terminal = terminals.get(entry.get('claim_id'))
        if not claim or not terminal or terminal.get('exit_code') != 0:
            raise ValueError('included historical output lacks a closed successful claim')
        queue = Path(entry['queue']['path']).resolve(strict=True)
        _same_binding(_binding(queue), entry['queue'], 'historical queue')
        queue_rows = [json.loads(line) for line in queue.read_text().splitlines() if line.strip()]
        if len(queue_rows) != 1 or queue_rows[0].get('index') != claim.get('index'):
            raise ValueError('historical queue does not contain its exact claimed source index')
        index = claim['index']
        if index not in inventory_by_index or cases[index] != row.get('task', {}).get('program_ir'):
            raise ValueError('historical provider task differs from its indexed source program')
        inventoried = inventory_by_index[index]
        if (inventoried.get('program_id') != row.get('task', {}).get('program_ir', {}).get('id')
                or inventoried.get('source_groups') != row.get('task', {}).get('program_ir', {}).get('source_groups')):
            raise ValueError('historical provider task differs from its authenticated source inventory')
        result_path = Path(entry['result']['path']).resolve(strict=True)
        manifest_path = Path(entry['manifest']['path']).resolve(strict=True)
        _same_binding(_binding(result_path), entry['result'], 'historical provider result')
        _same_binding(_binding(manifest_path), entry['manifest'], 'historical collector manifest')
        result_rows = [json.loads(line) for line in result_path.read_text().splitlines() if line.strip()]
        result_manifest = json.loads(manifest_path.read_text())
        accounting = terminal.get('output_accounting') or {}
        if (len(result_rows) != 1 or result_rows[0] != row
                or accounting.get('complete') is not True
                or accounting.get('output_sha256') != entry['result']['sha256']
                or result_manifest.get('completed') != 1
                or result_manifest.get('output_sha256') != entry['result']['sha256']
                or result_manifest.get('source_sha256') != plan['source_sha256']
                or result_manifest.get('model') != 'gpt-6-luna'):
            raise ValueError('historical output or collector accounting differs from the sealed snapshot')
        program = row.get('task', {}).get('program_ir') or {}
        outcome, provenance = row.get('outcome') or {}, row.get('provenance') or {}
        if (outcome.get('status') != 'done' or outcome.get('accepted') is not True
                or outcome.get('oracle', {}).get('accepted') is not True
                or outcome.get('oracle', {}).get('level') != 'exact'
                or provenance.get('model') != 'gpt-6-luna'
                or provenance.get('provider') != 'openai-codex'
                or provenance.get('collection_role') != 'teacher'
                or provenance.get('transport') != 'pi-provider'):
            raise ValueError('historical trajectory is not an accepted exact Luna teacher result')
        custody = current_worlds.check(program)
        checked_programs.append({'trajectory_id': row['id'], 'program_id': program.get('id'),
            'source_groups': program.get('source_groups'), 'current_protected_worlds_receipt': custody})
        source_row_sha256, row_digest_materializer_sha256 = _native_row_digest(runtime, runtime_manifest, row)
        results.append({'claim_id': entry['claim_id'], 'queue': _binding(queue),
            'result': _binding(result_path), 'collector_manifest': _binding(manifest_path),
            'trajectory_id': row['id'], 'source_row_sha256': source_row_sha256,
            'row_digest_materializer_sha256': row_digest_materializer_sha256})

    return {
        'schema': SCHEMA, 'scope': 'read-only historical source continuity for graph conversion',
        'dispatch_authorized': False, 'training_admission': False, 'training_exposure': False,
        'provider_source_manifest': _binding(provider_source_path),
        'raw_trajectory_source': _binding(source), 'original_plan': _binding(plan_path),
        'claim_ledger': _binding(ledger_path),
        'historical_plan_policy_fields': {'resource_policy_present': False,
            'note': 'absence preserved exactly; no modern launch policy synthesized'},
        'historical_provider_source_fields': {'oracle_regrades_present': 'oracle_regrades' in declaration,
            'note': 'legacy field absence is preserved; no default regrade list synthesized'},
        'historical_artifact_evidence': artifact_evidence,
        'historical_runtime': {'path': str(runtime), 'receipt_sha256': plan['runtime_receipt_sha256'],
            'tree_files_sha256': hashlib.sha256(json.dumps(runtime_manifest['files'], sort_keys=True,
                separators=(',', ':')).encode()).hexdigest(), 'model': 'gpt-6-luna'},
        'campaign_identity': identity,
        'included_accepted_results': results,
        'current_protection': {'registry_path': str(Path(REGISTRY).resolve()),
            'registry_sha256': sha256_file_hex(REGISTRY), 'identity_digest': current_worlds.digest,
            'checked_programs': checked_programs},
        'resource_limits_as_recorded': {key: plan[key] for key in
            ('case_seconds', 'max_turns', 'max_model_requests', 'slots', 'model_concurrency',
             'max_transport_retries') if key in plan},
    }


def verify(provider_source_path: Path, receipt_path: Path, *, repo: Path | None = None) -> tuple[dict, dict]:
    recorded = json.loads(Path(receipt_path).read_text())
    expected = derive(provider_source_path, repo=repo)
    if recorded != expected:
        raise ValueError('historical source continuity receipt differs from live immutable evidence')
    plan = json.loads(Path(expected['original_plan']['path']).read_text())
    return plan, expected['campaign_identity']


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--provider-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.verify:
        plan, identity = verify(args.provider_source, args.output)
        result = {'verified': True, 'campaign_id': plan['campaign_id'],
                  'trajectory_count': len(identity['cases']), 'dispatch_authorized': False}
    else:
        if args.output.exists():
            raise ValueError('historical continuity receipt is immutable; choose a fresh output')
        record = derive(args.provider_source)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2) + '\n')
        result = {'receipt': str(args.output), 'sha256': sha256_file_hex(args.output),
                  'included_results': len(record['included_accepted_results']),
                  'dispatch_authorized': False}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
