#!/usr/bin/env python3
"""Check or materialize root-approved v9 reviewed_single_pool artifacts.

This helper is check-only by default.  Materialization requires a root-written
approval JSON bound to the exact prepared packet and current check result.  It
never starts or authorizes a provider process by itself.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import sys

ROOT = Path('/home/werg/natlang')
BASE = ROOT / 'runs/dgx-qwen36-current-train-refresh-20261003'
CAMPAIGN_NAMES = {'pool-v9-alternates-v2', 'pool-v9-alternates-v3'}
V7 = BASE / 'pool-v7-r7'
V8 = BASE / 'pool-v8-alternates-v3'
V9V2 = BASE / 'pool-v9-alternates-v2'
QWEN_HISTORY = BASE / 'pool-v6/qwen-history-v1.json'
ARM_RUNTIME_MANIFEST_SHA = '5aa7470c294c7a9979912a2f9b50925a4e3711c6d17527793e36b08a45930b19'
X64_RUNTIME_MANIFEST_SHA = 'd47d4f8946ed1cc7066d7d83c25cd1b6807822b89c2acb7d50d57b58bd0ab03e'
CHAT_SHA = '7762d0da0da3f526e9fe13c9c423937d4e94f4ef526b1290e1ce352d863a5472'


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_rows(path: Path):
    with path.open(encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                try:
                    yield line_no, json.loads(line)
                except Exception as exc:
                    raise ValueError(f'invalid JSONL {path}:{line_no}: {exc}') from exc


def canonical(value):
    if isinstance(value, list):
        return '[' + ','.join(canonical(x) for x in value) + ']'
    if isinstance(value, dict):
        return '{' + ','.join(json.dumps(k, ensure_ascii=False) + ':' + canonical(value[k]) for k in sorted(value)) + '}'
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def payload_signature(ir):
    sem = ir.get('semantics') or {}
    payload = {
        'source_ids': ir.get('source_ids') or [],
        'source_groups': ir.get('source_groups') or [],
        'source_revisions': ir.get('source_revisions') or [],
        'license': ir.get('license'),
        'inputs': sem.get('inputs'),
        'expected': sem.get('expected'),
        'folder_files': sem.get('folder_files'),
        'expected_files': sem.get('expected_files'),
    }
    return sha_bytes(canonical(payload).encode())


def json_file(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def assignment_draft_path(campaign: Path) -> Path:
    return campaign / ('assignment-draft-v2.json' if campaign.name.endswith('-v2') else 'assignment-draft.json')


def check_campaign(campaign: Path):
    campaign = campaign.resolve()
    if campaign.parent != BASE.resolve() or campaign.name not in CAMPAIGN_NAMES:
        raise ValueError('campaign must be one of the two pinned v9 candidate directories')
    name = campaign.name
    v3 = name.endswith('-v3')
    bundle = campaign / 'bundle'
    packet_path = campaign / 'root-review-packet.json'
    # v9-v2 has a preserved corrected v2 draft; the original draft points at
    # an obsolete x64 runtime directory. v9-v3's draft already carries r1.
    assignment_draft_path = campaign / ('assignment-draft-v2.json' if name.endswith('-v2') else 'assignment-draft.json')
    native_draft_path = bundle / 'native-review-draft-v2.json'
    inputs = {
        'packet': packet_path, 'assignment_draft': assignment_draft_path,
        'native_review_draft': native_draft_path,
        'native_review_correction': bundle / 'native-review-draft-correction-v1.json',
        'ir': bundle / 'cases.ir.jsonl', 'selection_proof': bundle / 'selection-proof.jsonl',
        'native_rows': bundle / 'native-reference-rows.jsonl',
        'native_report': bundle / 'native-reference-report.json',
        'native_join': bundle / 'native-runner-join-check.json',
        'visibility': campaign / 'component-visibility-audit.json',
        'closure': campaign / 'full-v13-test-alias-closure.json',
        'arm_manifest': bundle / 'arm64-frozen-runtime.json',
        'chat_config': bundle / 'chat-request-config.json',
        'runner': campaign / 'runner/reviewed_single_pool.py',
        'paths_helper': campaign / 'runner/reviewed_pool_paths.py',
        'accounting_helper': campaign / 'runner/run_bonsai_queue.py',
    }
    missing = [str(p) for p in inputs.values() if not p.is_file()]
    if missing:
        raise ValueError('missing required candidate artifacts: ' + ', '.join(missing))
    hashes = {k: sha(v) for k, v in inputs.items()}
    packet, assignment, native, report = (json_file(inputs[k]) for k in ('packet', 'assignment_draft', 'native_review_draft', 'native_report'))
    correction = json_file(inputs['native_review_correction'])
    join, visibility, closure = (json_file(inputs[k]) for k in ('native_join', 'visibility', 'closure'))
    count = 1024
    errors = []
    expected_ir_sha = hashes['ir']
    expected_proof_sha = hashes['selection_proof']
    expected_rows_sha = hashes['native_rows']
    expected_runtime_sha = ARM_RUNTIME_MANIFEST_SHA

    if packet.get('schema') != 'dgx.qwen-current-train-reviewed-pool-packet/1':
        errors.append('unexpected root-review packet schema')
    allowed_packet_status = {'prepared_not_root_reviewed_not_launched', 'prepared_not_root_reviewed_not_authorized_not_launched'}
    if packet.get('status') not in allowed_packet_status or packet.get('launch_authorized') is not False:
        errors.append('candidate packet is no longer in a prepared, unlaunched state')
    packet_ir = packet.get('selected_ir', {})
    packet_proof = packet.get('selection_proof', {})
    packet_native = packet.get('native_reference', {})
    packet_closure = packet.get('v13_closure', {})
    packet_visibility = packet.get('visible_input_audit' if v3 else 'visibility', {})
    packet_count = packet.get('cases', packet.get('case_count'))
    if packet_count != count:
        errors.append(f'expected 1024 cases, packet says {packet_count!r}')
    if packet_ir.get('sha256') != expected_ir_sha:
        errors.append('packet selected IR SHA differs from exact bytes')
    if packet_proof.get('sha256') != expected_proof_sha:
        errors.append('packet selection proof SHA differs from exact bytes')
    if packet_native.get('rows_sha256') != expected_rows_sha:
        errors.append('packet native rows SHA differs from exact bytes')
    if packet_native.get('runtime_manifest_sha256') != expected_runtime_sha:
        errors.append('packet ARM runtime pin differs from reviewed runtime')
    if packet_closure.get('sha256') != hashes['closure'] or packet_visibility.get('sha256') != hashes['visibility']:
        errors.append('packet source closure or visible-input SHA is stale')
    if sha(inputs['arm_manifest']) != expected_runtime_sha:
        errors.append('actual ARM runtime manifest differs from reviewed v41 pin')
    if sha(Path(assignment.get('local_import_runtime', '')) / 'frozen-runtime.json') != X64_RUNTIME_MANIFEST_SHA:
        errors.append('actual local x64 import runtime manifest differs from reviewed v41 pin')
    if hashes['chat_config'] != CHAT_SHA or assignment.get('chat_request_config_sha256') != CHAT_SHA:
        errors.append('chat-request config differs from reviewed hash')

    # The assignment draft contains inherited v39 launcher fields; only its
    # execution settings are used.  Runtime/node identity is derived from the
    # pinned v41 ARM manifest when materializing, and never from that stale path.
    if assignment.get('remote_runtime_manifest_sha256') != expected_runtime_sha:
        errors.append('assignment draft ARM manifest pin mismatch')
    if assignment.get('local_import_runtime_manifest_sha256') != X64_RUNTIME_MANIFEST_SHA:
        errors.append('assignment draft x64 import manifest pin mismatch')
    if assignment.get('selected_ir_sha256') != expected_ir_sha or assignment.get('selection_proof_sha256') != expected_proof_sha:
        errors.append('assignment draft selected IR or selection proof is stale')
    if assignment.get('cases') != count or assignment.get('launch_authorized') is not False:
        errors.append('assignment draft count or authorization state mismatch')

    if report.get('schema') != 'natlang.arm_native_reference_rows_proof/1' or report.get('input_sha256') != expected_ir_sha or report.get('rows_sha256') != expected_rows_sha or report.get('runtime_manifest_sha256') != expected_runtime_sha or report.get('model_calls') != 0:
        errors.append('native reference report does not bind the exact rows/IR/runtime or is not model-free')
    report_counts = report.get('counts') or {}
    native_count = packet_native.get('counts') or {}
    if report_counts != native_count or report_counts.get('cases') != count or report_counts.get('admitted') != count or report_counts.get('unlinked') != 0 or report_counts.get('denied') != 0 or report_counts.get('held_decisions') != 0:
        errors.append('native reference report and packet counts differ or include rejects/holds')
    if report_counts.get('materialized') != report_counts.get('approved_decisions'):
        errors.append('native materialized and approved-decision counts differ')
    if join.get('schema') != 'natlang.reviewed_single_pool_native_join_check/1' or join.get('status') != 'passed' or join.get('ir_sha256') != expected_ir_sha or join.get('native_rows_sha256') != expected_rows_sha or join.get('runtime_manifest_sha256') != expected_runtime_sha:
        errors.append('native runner join does not pass for exact candidate inputs')
    if join.get('counts', {}).get('cases') != count or join.get('counts', {}).get('matched') != count or join.get('counts', {}).get('unlinked') != 0 or join.get('mismatches') or join.get('holds') or join.get('conversion'):
        errors.append('native runner join contains mismatches, holds, or conversion errors')
    if visibility.get('status') != 'passed' or visibility.get('cases_sha256') != expected_ir_sha or visibility.get('checks', {}).get('bad_components') != 0:
        errors.append('visible-input/source-component audit did not pass exact candidate')
    if closure.get('status') != 'passed' or closure.get('selected_ir_sha256') != expected_ir_sha:
        errors.append('full v13 closure does not pass for exact candidate')
    if closure.get('combined_identity_overlap_count', closure.get('combined_identity_overlap')) != 0 or closure.get('selected_groups_not_train_count', 0) != 0:
        errors.append('full v13 protected identity/group overlap is nonzero')
    if closure.get('prepared_test_rows', closure.get('test_rows', closure.get('protected_test_rows'))) != 5432:
        errors.append('full v13 protected test row scan is not the exact 5,432-row corpus')
    if closure.get('protected_counts', {}).get('identity_aliases', closure.get('protected_identity_aliases')) != 10338:
        errors.append('full v13 protected alias union is not the independently reviewed 10,338 aliases')
    if native.get('version') != 'natlang.reviewed_single_pool_native_review/1' or native.get('status') != 'prepared_not_approved_root_review_required' or native.get('ir_sha256') != expected_ir_sha or native.get('runtime_manifest_sha256') != expected_runtime_sha or native.get('range') != {'start': 0, 'count': count}:
        errors.append('native review draft is stale or not awaiting approval')
    if native.get('provider_calls') != 0 or native.get('model_calls') != 0 or native.get('native_reference_rows') != count or native.get('current_admitted') != count:
        errors.append('native draft counts/provider-call status are invalid')
    if native.get('materialized_decisions') != report_counts.get('materialized'):
        errors.append('native draft decision count differs from actual materialized report')
    original_native_draft_path = bundle / 'native-review-draft.json'
    if correction.get('schema') != 'dgx.qwen-v9-native-review-field-correction/1' or correction.get('source_draft_sha256') != sha(original_native_draft_path) or correction.get('derived_draft_sha256') != hashes['native_review_draft']:
        errors.append('corrected native review draft does not bind its immutable original and derived bytes')
    if (correction.get('original_materializer_accepted') != report_counts.get('materialized') or
            correction.get('derived_materializer_accepted') != count or
            correction.get('materialized_decisions_preserved') != report_counts.get('materialized')):
        errors.append('native review count-field correction does not preserve case and decision cardinalities explicitly')

    # Bind every proof row to the exact selected IR row, and selected roots to
    # their proven base payload fields; do not rely on a prior count alone.
    cases = [r for _, r in read_rows(inputs['ir'])]
    proofs = [r for _, r in read_rows(inputs['selection_proof'])]
    native_rows = [r for _, r in read_rows(inputs['native_rows'])]
    if not (len(cases) == len(proofs) == len(native_rows) == count):
        errors.append(f'candidate row cardinality mismatch: IR={len(cases)} proof={len(proofs)} native={len(native_rows)}')
    ids, signatures, duplicate_ids, duplicate_payloads = set(), set(), [], []
    for index, case in enumerate(cases):
        signature = payload_signature(case)
        if case.get('id') in ids:
            duplicate_ids.append(case.get('id'))
        if signature in signatures:
            duplicate_payloads.append(case.get('id'))
        ids.add(case.get('id'))
        signatures.add(signature)
        proof = proofs[index] if index < len(proofs) else {}
        if proof.get('id') != case.get('id') or proof.get('source_ids') != case.get('source_ids') or proof.get('source_groups') != case.get('source_groups'):
            errors.append(f'selection proof lineage mismatch at row {index}')
            break
        if proof.get('program_sha256') != sha_bytes(json.dumps(case, ensure_ascii=False, separators=(',', ':')).encode()):
            errors.append(f'selection proof program digest mismatch at row {index}')
            break
        native_row = native_rows[index] if index < len(native_rows) else {}
        record = native_row.get('task', {}).get('program_ir', {})
        # Native provenance uses the TypeScript runtime's localeCompare key
        # ordering.  The independently produced join receipt validates those
        # runtime digests; here compare the full parsed IR record structurally,
        # avoiding a subtly different Python reimplementation of its digest.
        if record != case or native_row.get('outcome', {}).get('accepted') is not True:
            errors.append(f'native exact-row identity/admission mismatch at row {index}')
            break
    if duplicate_ids or duplicate_payloads:
        errors.append(f'candidate itself has duplicate IDs={len(duplicate_ids)} or payloads={len(duplicate_payloads)}')

    # Historical payloads + currently queued pools are checked by exact task
    # payload signature, not by source ID alone.  Same-source comparison is an
    # explicit property of these candidate pools.
    prior_payloads, prior_ids, unreadable = set(), set(), 0
    history_rows = 0
    if not QWEN_HISTORY.is_file():
        errors.append('Qwen v6 history index is missing')
        hist = {'ledgers': []}
    else:
        hist = json_file(QWEN_HISTORY)
    for ledger in hist.get('ledgers', []):
        ledger_path = Path(ledger['path'])
        if not ledger_path.is_file():
            unreadable += 1
            continue
        for _, entry in read_rows(ledger_path):
            history_rows += 1
            artifact = Path(entry.get('path', ''))
            if not artifact.is_file():
                unreadable += 1
                continue
            try:
                saved = json_file(artifact)
                old_ir = saved.get('task', {}).get('program_ir')
                if old_ir:
                    prior_payloads.add(payload_signature(old_ir))
                    prior_ids.add(old_ir.get('id'))
            except Exception:
                unreadable += 1
    # Exact rows from predecessor/static imports that may not be in the older
    # history ledger yet (including the currently queued v8 pool).
    for extra_path in [V7 / 'bundle/cases.ir.jsonl', V8 / 'bundle/cases.ir.jsonl'] + ([V9V2 / 'bundle/cases.ir.jsonl'] if v3 else []):
        if not extra_path.is_file():
            errors.append(f'missing predecessor/current-queue IR: {extra_path}')
            continue
        for _, old_ir in read_rows(extra_path):
            prior_payloads.add(payload_signature(old_ir))
            prior_ids.add(old_ir.get('id'))
    for import_ledger in [V7 / 'import-ledger.jsonl', V8 / 'import-ledger.jsonl'] + ([V9V2 / 'import-ledger.jsonl'] if v3 else []):
        # A not-yet-launched predecessor has no import ledger. The authoritative
        # exact IR above remains the queued-identity source in that state.
        if not import_ledger.is_file():
            continue
        for _, entry in read_rows(import_ledger):
            artifact = Path(entry.get('path', ''))
            if not artifact.is_file():
                unreadable += 1
                continue
            try:
                old_ir = json_file(artifact).get('task', {}).get('program_ir')
                if old_ir:
                    prior_payloads.add(payload_signature(old_ir))
                    prior_ids.add(old_ir.get('id'))
            except Exception:
                unreadable += 1
    collisions = [case.get('id') for case in cases if payload_signature(case) in prior_payloads]
    prior_id_collisions = [case.get('id') for case in cases if case.get('id') in prior_ids]
    if unreadable:
        errors.append(f'{unreadable} historical/import result artifacts were unreadable')
    if collisions or prior_id_collisions:
        errors.append(f'prior/current queue collision: payloads={len(collisions)} IDs={len(prior_id_collisions)}')

    # Check frozen source-review implementation pins from the ARM manifest and
    # local x64 import runtime. Those are the policy gates used by the runner.
    arm_manifest = json_file(inputs['arm_manifest'])
    arm_files = arm_manifest.get('files') or {}
    source_ts = arm_files.get('src/teacher/source-review.ts')
    source_js = arm_files.get('dist/teacher/source-review.js')
    if not source_ts or not source_js:
        errors.append('ARM manifest omits current source-review implementation pins')
    x64_runtime = Path(assignment.get('local_import_runtime', ''))
    for rel in ['src/teacher/source-review.ts', 'dist/teacher/source-review.js']:
        p = x64_runtime / rel
        if not p.is_file():
            errors.append(f'x64 import runtime missing {rel}')
    report_result = {
        'schema': 'dgx.qwen-v9-approval-helper-check/1',
        'status': 'passed' if not errors else 'failed',
        'campaign': str(campaign), 'variant': 'v3' if v3 else 'v2',
        'packet_sha256': hashes['packet'], 'ir_sha256': expected_ir_sha,
        'selection_proof_sha256': expected_proof_sha, 'native_rows_sha256': expected_rows_sha,
        'native_draft_sha256': hashes['native_review_draft'],
        'native_draft_correction_sha256': hashes['native_review_correction'],
        'native_report_sha256': hashes['native_report'], 'native_join_sha256': hashes['native_join'],
        'visibility_sha256': hashes['visibility'], 'closure_sha256': hashes['closure'],
        'arm_runtime_manifest_sha256': hashes['arm_manifest'], 'x64_import_manifest_sha256': sha(Path(assignment['local_import_runtime']) / 'frozen-runtime.json'),
        'chat_request_config_sha256': hashes['chat_config'],
        'runner_sha256': hashes['runner'], 'pool_paths_helper_sha256': hashes['paths_helper'],
        'output_accounting_helper_sha256': hashes['accounting_helper'],
        'source_review_arm': {'source_sha256': source_ts, 'compiled_sha256': source_js},
        'history': {'v6_ledger_rows_scanned': history_rows, 'unreadable_artifacts': unreadable,
                    'unique_prior_payload_signatures': len(prior_payloads), 'unique_prior_program_ids': len(prior_ids),
                    'candidate_payload_collisions': len(collisions), 'candidate_id_collisions': len(prior_id_collisions),
                    'exact_queued_ir_sets_checked': [
                        {'name': 'v7', 'ir_sha256': sha(V7 / 'bundle/cases.ir.jsonl')},
                        {'name': 'v8-current-predecessor', 'ir_sha256': sha(V8 / 'bundle/cases.ir.jsonl')},
                    ] + ([{'name': 'v9-v2', 'ir_sha256': sha(V9V2 / 'bundle/cases.ir.jsonl')}] if v3 else [])},
        'selected': {'cases': len(cases), 'unique_ids': len(ids), 'unique_payloads': len(signatures),
                     'approved_decisions': report_counts.get('approved_decisions'),
                     'purpose': packet.get('purpose')},
        'native_join': {'host': packet_native.get('host'), 'arch': packet_native.get('arch'),
                        'node': packet_native.get('node'), 'cases': report_counts.get('cases'),
                        'admitted': report_counts.get('admitted'), 'materialized': report_counts.get('materialized'),
                        'unlinked': report_counts.get('unlinked'), 'provider_calls': native.get('provider_calls'),
                        'model_calls': native.get('model_calls')},
        'v13_closure': {'test_rows': closure.get('prepared_test_rows', closure.get('test_rows', closure.get('protected_test_rows'))),
                        'protected_aliases': closure.get('protected_counts', {}).get('identity_aliases', closure.get('protected_identity_aliases')),
                        'overlap_count': closure.get('combined_identity_overlap_count', closure.get('combined_identity_overlap')),
                        'selected_group_count': packet.get('unique_source_groups')},
        'errors': errors,
    }
    return report_result


def materialize(campaign: Path, approval_path: Path, check: dict):
    approval = json_file(approval_path)
    expected = {
        'schema': 'dgx.qwen-v9-pool-root-approval/1',
        'status': 'approved',
        'authorized_action': 'materialize_pool_approval_artifacts_no_launch',
        'campaign': str(campaign.resolve()),
        'check_sha256': sha_bytes((json.dumps(check, indent=2) + '\n').encode()),
        'packet_sha256': check['packet_sha256'], 'ir_sha256': check['ir_sha256'],
        'selection_proof_sha256': check['selection_proof_sha256'],
        'native_rows_sha256': check['native_rows_sha256'],
        'native_draft_sha256': check['native_draft_sha256'],
        'native_draft_correction_sha256': check['native_draft_correction_sha256'],
        'native_report_sha256': check['native_report_sha256'],
        'native_join_sha256': check['native_join_sha256'],
        'visibility_sha256': check['visibility_sha256'], 'closure_sha256': check['closure_sha256'],
        'arm_runtime_manifest_sha256': check['arm_runtime_manifest_sha256'],
        'x64_import_manifest_sha256': check['x64_import_manifest_sha256'],
        'chat_request_config_sha256': check['chat_request_config_sha256'],
    }
    mismatches = [k for k, v in expected.items() if approval.get(k) != v]
    if mismatches:
        raise ValueError('root approval missing/mismatched exact binding fields: ' + ', '.join(mismatches))
    if not approval.get('approved_by') or not approval.get('approval_text'):
        raise ValueError('root approval must name approver and explain approval')
    bundle = campaign / 'bundle'
    assignment = json_file(assignment_draft_path(campaign))
    native_review = json_file(bundle / 'native-review-draft-v2.json')
    remote = assignment['remote_directory']
    runtime = assignment['remote_runtime']
    manifest = json_file(bundle / 'arm64-frozen-runtime.json')
    node_rel = next((p for p in manifest.get('files', {}) if p.endswith('/bin/node') and '/node-v24.0.1-linux-arm64/' in p), None)
    if not node_rel:
        raise ValueError('pinned ARM runtime manifest has no Node 24 executable')
    node_hash = manifest['files'][node_rel]
    node_binary = runtime + '/' + node_rel
    assignment.update({
        'status': 'approved_pending_launch', 'launch_authorized': True,
        'node_binary': node_binary, 'node_binary_sha256': node_hash,
        'launch_path_prefix': str(Path(node_binary).parent) + ':/usr/local/bin:/usr/bin:/bin',
        'native_review_sha256': '', 'root_approval_sha256': sha(approval_path),
    })
    native_review.update({
        'status': 'approved',
        # `materializer_accepted` is the runner's case-level gate; decision
        # cardinality is carried separately in materialized_decisions.
        'materializer_accepted': assignment['cases'],
        'evidence': str(native_review.get('evidence', '')) +
                    f" Root approval {approval_path} sha256 {sha(approval_path)}.",
        'root_approval_path': str(approval_path), 'root_approval_sha256': sha(approval_path),
    })
    # Runner-required native fields are checked against the actual draft above;
    # preserve the candidate's fresh v13 alias metrics rather than v8 prose.
    created = {
        bundle / 'root-native-review.json': (json.dumps(native_review, indent=2) + '\n').encode(),
    }
    assignment['native_review_sha256'] = sha_bytes(created[bundle / 'root-native-review.json'])
    assignment['native_review_status'] = 'approved'
    created[campaign / 'assignment.json'] = (json.dumps(assignment, indent=2) + '\n').encode()

    status = remote + '/worker-status.json'
    jobs = remote + '/jobs'
    output = remote + '/results.jsonl'
    control = remote + '/control'
    auth = {
        'version': 'natlang.reviewed_single_pool_authorization/1', 'status': 'approved',
        'ir_sha256': check['ir_sha256'], 'runtime_manifest_sha256': assignment['remote_runtime_manifest_sha256'],
        'start': 0, 'count': assignment['cases'], 'root_seed': assignment['root_seed'],
        'model_id': assignment['model'], 'workers': assignment['workers'],
        'model_concurrency': assignment['model_concurrency'], 'max_model_requests': assignment['max_model_requests_per_case'],
        'max_turns': assignment['max_turns_per_case'], 'chat_request_config_sha256': assignment['chat_request_config_sha256'],
        'transport_retries': assignment['transport_retries'], 'retry_delay_ms': assignment['retry_delay_ms'],
        'pool_runner_sha256': check['runner_sha256'], 'final_export_only': True,
        'pool_paths_helper_sha256': check['pool_paths_helper_sha256'],
        'native_review_sha256': assignment['native_review_sha256'],
        'output_accounting_helper_sha256': check['output_accounting_helper_sha256'],
        'jobs_path': jobs, 'output_path': output, 'control_dir': control,
        'worker_status_path': status, 'supervisor_journal_path': remote + '/journal-pool.jsonl',
        'case_events_path': remote + '/journal-cases.jsonl', 'hard_abort_file': remote + '/hard-abort.request',
        'server': assignment['endpoint'], 'reasoning_effort': assignment['reasoning_effort'],
        'file_tools': assignment['file_tools'], 'execution_plans': assignment['execution_plans'],
        'heap_mib': assignment['heap_mib'], 'kv_tokens': assignment['kv_tokens'],
        'context_tokens': assignment['context_tokens'], 'wall_seconds': assignment['resource_wall_seconds'],
        'min_free_mib': assignment['minimum_free_mib'], 'min_available_memory_mib': assignment['minimum_available_memory_mib'],
        'status_interval': 10,
        'authorization_path': remote + '/bundle/root-pool-authorization.json',
        'native_review_path': remote + '/bundle/root-native-review.json',
        'runtime_path': runtime, 'runtime_manifest_path': runtime + '/frozen-runtime.json',
        'chat_request_config_path': remote + '/bundle/chat-request-config.json', 'pool_status_path': control + '/pool-status.json',
        'source_review_source_sha256': check['source_review_arm']['source_sha256'],
        'source_review_compiled_sha256': check['source_review_arm']['compiled_sha256'],
        'authorization': approval['approval_text'], 'root_approval_artifact_sha256': sha(approval_path),
    }
    created[bundle / 'root-pool-authorization.json'] = (json.dumps(auth, indent=2) + '\n').encode()
    runner = remote + '/runner/reviewed_single_pool.py'
    command = [
        '/usr/bin/python3', runner,
        '--ir', remote + '/bundle/cases.ir.jsonl', '--ir-sha256', check['ir_sha256'],
        '--start', '0', '--count', str(assignment['cases']), '--runtime', runtime,
        '--runtime-manifest', runtime + '/frozen-runtime.json', '--runtime-manifest-sha256', assignment['remote_runtime_manifest_sha256'],
        '--authorization', remote + '/bundle/root-pool-authorization.json', '--native-review', remote + '/bundle/root-native-review.json',
        '--jobs', jobs, '--output', output, '--control-dir', control,
        '--model-id', assignment['model'], '--server', assignment['endpoint'],
        '--chat-request-config', remote + '/bundle/chat-request-config.json', '--chat-request-config-sha256', assignment['chat_request_config_sha256'],
        '--root-seed', str(assignment['root_seed']), '--workers', str(assignment['workers']),
        '--model-concurrency', str(assignment['model_concurrency']), '--max-model-requests', str(assignment['max_model_requests_per_case']),
        '--max-turns', str(assignment['max_turns_per_case']), '--heap-mib', str(assignment['heap_mib']), '--kv-tokens', str(assignment['kv_tokens']),
        '--context-tokens', str(assignment['context_tokens']), '--transport-retries', str(assignment['transport_retries']),
        '--retry-delay-ms', str(assignment['retry_delay_ms']), '--reasoning-effort', assignment['reasoning_effort'],
        '--file-tools', assignment['file_tools'], '--execution-plans', '--wall-seconds', str(assignment['resource_wall_seconds']),
        '--min-free-mib', str(assignment['minimum_free_mib']), '--min-available-memory-mib', str(assignment['minimum_available_memory_mib']),
        '--status-interval', '10', '--hard-abort-file', remote + '/hard-abort.request',
        '--status-file', status, '--journal', remote + '/journal-pool.jsonl', '--case-events-file', remote + '/journal-cases.jsonl',
    ]
    launch = ('#!/bin/sh\nset -eu\nexport PATH=' + shlex.quote(str(Path(node_binary).parent) + ':/usr/local/bin:/usr/bin:/bin') +
              '\nexec ' + ' '.join(shlex.quote(arg) for arg in command) + ' "$@"\n').encode()
    created[campaign / 'root-launch.sh'] = launch
    # All materialized files are new, never overwrite a draft or prior result.
    collisions = [str(path) for path in created if path.exists()]
    if collisions:
        raise FileExistsError('refusing to overwrite existing approval artifacts: ' + ', '.join(collisions))
    written = []
    try:
        for path, data in created.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as f:
                f.write(data)
            if path.name == 'root-launch.sh':
                path.chmod(0o755)
            written.append(path)
    except Exception:
        for path in written:
            path.unlink(missing_ok=True)
        raise
    receipt = {
        'schema': 'dgx.qwen-v9-root-approval-artifact-receipt/1',
        'status': 'approved_artifacts_prepared_no_launch',
        'approval_path': str(approval_path), 'approval_sha256': sha(approval_path),
        'approval_action': 'materialize pool auth/native review/assignment/launch wrapper only',
        'candidate_check': check,
        'created_files': {str(p): sha(p) for p in written},
        'node_binary': node_binary, 'node_binary_sha256': node_hash,
        'launch_performed': False,
        'prelaunch_required': ['sync exact prepared files to DGX', 'run remote --preflight-only', 'root separately approves launch'],
    }
    receipt_path = campaign / 'root-approval-artifacts-receipt.json'
    with receipt_path.open('x', encoding='utf-8') as f:
        f.write(json.dumps(receipt, indent=2) + '\n')
    return receipt_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', type=Path, required=True, help='exact v9-v2 or v9-v3 candidate directory')
    parser.add_argument('--apply-root-approval', type=Path, help='root-authored exact approval JSON; no launch is performed')
    parser.add_argument('--check-only', action='store_true', help='recheck inputs and write only a check receipt (default)')
    args = parser.parse_args()
    if args.apply_root_approval and args.check_only:
        parser.error('choose either --check-only or --apply-root-approval')
    try:
        result = check_campaign(args.campaign)
        # Write only a versioned check receipt. This does not alter packet, proof,
        # assignment drafts, native drafts, or any live authority.
        check_bytes = (json.dumps(result, indent=2) + '\n').encode()
        check_path = args.campaign.resolve() / ('approval-helper-check-' + sha_bytes(check_bytes)[:16] + '.json')
        if check_path.exists():
            if check_path.read_bytes() != check_bytes:
                raise FileExistsError(f'check receipt path exists with different bytes: {check_path}')
        else:
            with check_path.open('xb') as f:
                f.write(check_bytes)
        print(json.dumps({'check': str(check_path), 'check_sha256': sha(check_path), **result}, indent=2))
        if result['status'] != 'passed':
            return 2
        if args.apply_root_approval:
            approval = json_file(args.apply_root_approval)
            expected_check_sha = sha(check_path)
            if approval.get('check_sha256') != expected_check_sha:
                raise ValueError('root approval is not bound to this exact recomputed check report')
            receipt = materialize(args.campaign.resolve(), args.apply_root_approval.resolve(), result)
            print(json.dumps({'status': 'approved_artifacts_prepared_no_launch', 'receipt': str(receipt), 'receipt_sha256': sha(receipt)}, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'}, indent=2), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
