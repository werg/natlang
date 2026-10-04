#!/usr/bin/env python3
"""Fail-closed readiness gate and optional root-approved DGX handoff controller.

Without ``--start`` this is read-only. With ``--start`` it requires a
root-approved, hash-pinned config and starts/synchronizes only after exact
predecessor accounting, imported-artifact checks, live authority checks, and
successor approval pins pass. ``--watch --start`` keeps checking until that
exact gate passes, then launches once.
"""
import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from generation_authority import authority_lock


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def cfg_successor(cfg):
    return cfg['successor']


def cfg_successor_server(cfg):
    # The actual runner authorization and assignment are checked separately;
    # this is a lightweight model endpoint identity probe for wait/preflight.
    c = cfg_successor(cfg)
    auth_path = Path(c.get('authorization', ''))
    if auth_path.is_file():
        auth = read_json(auth_path)
        if isinstance(auth.get('server'), str):
            return auth['server']
    return c.get('server') or 'http://127.0.0.1:8082'


def cfg_successor_model(cfg):
    return cfg_successor(cfg).get('model')


def validate_remote_environment(remote, predecessor_service=None):
    blockers = []
    services_exit = remote.get('active_qwen_services_exit')
    services = remote.get('active_qwen_services')
    if services_exit != 0 or not isinstance(services, list):
        blockers.append('remote_qwen_pool_service_inventory_unavailable')
    else:
        # The inventory is from `systemctl --user list-units --state=active`,
        # the same user-systemd scope used by reviewed pool launches.
        active = [x.get('unit') for x in services if isinstance(x, dict) and x.get('active') == 'active']
        if active:
            blockers.append('another_qwen_user_service_active:' + ','.join(active))
    server = remote.get('server_probe')
    if not isinstance(server, dict) or server.get('ok') is not True or server.get('matches_expected') is not True:
        blockers.append('qwen_server_model_identity_unverified')
    disk = remote.get('disk') or {}
    if not isinstance(disk.get('free_mib'), int) or disk.get('free_mib') < disk.get('minimum_free_mib', 0):
        blockers.append('remote_disk_floor_not_met_or_unverified')
    memory = remote.get('memory') or {}
    if not isinstance(memory.get('available_mib'), int) or memory.get('available_mib') < memory.get('minimum_available_memory_mib', 0):
        blockers.append('remote_memory_floor_not_met_or_unverified')
    return blockers


def atomic_json(path, value):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + f'.tmp-{os.getpid()}-{time.time_ns()}')
    with temp.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, target)


def rows(path):
    with Path(path).open() as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                yield line_no, json.loads(line)


def remote_snapshot(host, campaign, service, launch_script=None, launch_script_sha256=None,
                   server=None, model_id=None, min_free_mib=0, min_available_memory_mib=0,
                   include_events=True):
    # Pass paths as positional arguments, not interpolated Python source.
    code = r'''import json, shutil, subprocess, sys, urllib.request
from pathlib import Path
p=Path(sys.argv[1]); unit=sys.argv[2]; server=sys.argv[4] if len(sys.argv)>4 else None
model_id=sys.argv[5] if len(sys.argv)>5 else None
min_free=int(sys.argv[6]) if len(sys.argv)>6 else 0
min_available=int(sys.argv[7]) if len(sys.argv)>7 else 0
include_events=(sys.argv[8]!='0') if len(sys.argv)>8 else True
def obj(name):
 f=p/name
 return json.loads(f.read_text()) if f.is_file() else None
events=[]
f=p/'journal-pool.jsonl'
if include_events and f.is_file():
 for line in f.read_text().splitlines():
  if line.strip(): events.append(json.loads(line))
r=subprocess.run(['systemctl','--user','is-active',unit],capture_output=True,text=True)
units=subprocess.run(['systemctl','--user','list-units','--all','--state=active','--type=service','--plain','--no-legend','natlang-qwen36-*'],capture_output=True,text=True)
active=[]
if units.returncode==0:
 for line in units.stdout.splitlines():
  fields=line.split(None,4)
  if len(fields)>=4 and fields[0].endswith('.service'):
   active.append({'unit':fields[0],'load':fields[1],'active':fields[2],'sub':fields[3]})
server_probe=None
if server:
 try:
  url=server.rstrip('/')+'/v1/models'
  with urllib.request.urlopen(url,timeout=8) as response:
   body=json.loads(response.read().decode())
  ids=sorted(x.get('id') for x in body.get('data',[]) if isinstance(x,dict) and isinstance(x.get('id'),str))
  server_probe={'ok':True,'url':url,'model_ids':ids,'expected_model_id':model_id,'matches_expected':model_id in ids}
 except Exception as exc:
  server_probe={'ok':False,'url':server.rstrip('/')+'/v1/models','error':type(exc).__name__+': '+str(exc),'expected_model_id':model_id,'matches_expected':False}
disk=None
try:
 usage=shutil.disk_usage(p)
 disk={'path':str(p),'free_mib':usage.free//(1024*1024),'minimum_free_mib':min_free}
except Exception as exc:
 disk={'path':str(p),'error':type(exc).__name__+': '+str(exc),'free_mib':None,'minimum_free_mib':min_free}
available=None
try:
 for line in Path('/proc/meminfo').read_text().splitlines():
  if line.startswith('MemAvailable:'):
   available=int(line.split()[1])//1024; break
 memory={'available_mib':available,'minimum_available_memory_mib':min_available}
except Exception as exc:
 memory={'available_mib':None,'minimum_available_memory_mib':min_available,'error':type(exc).__name__+': '+str(exc)}
launch=None
if len(sys.argv)>3 and sys.argv[3]:
 q=subprocess.run(['sha256sum',sys.argv[3]],capture_output=True,text=True)
 launch={'exit':q.returncode,'sha256':q.stdout.split()[0] if q.returncode==0 and q.stdout.split() else None}
print(json.dumps({'status':obj('worker-status.json'),'pool_status':obj('control/pool-status.json'),
 'events':events,'unit_active_exit':r.returncode,'unit_active_text':r.stdout.strip(),
 'active_qwen_services_exit':units.returncode,'active_qwen_services':active,
 'server_probe':server_probe,'disk':disk,'memory':memory,'launch_script':launch}))'''
    remote_command = 'python3 -c ' + shlex.quote(code) + ' ' + shlex.quote(campaign) + ' ' + shlex.quote(service)
    if launch_script:
        remote_command += ' ' + shlex.quote(launch_script)
    else:
        remote_command += ' ' + shlex.quote('')
    remote_command += ' ' + shlex.quote(server or '') + ' ' + shlex.quote(model_id or '') + ' ' + shlex.quote(str(min_free_mib)) + ' ' + shlex.quote(str(min_available_memory_mib))
    remote_command += ' ' + ('1' if include_events else '0')
    command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', host, remote_command]
    result = subprocess.run(command, capture_output=True, text=True, timeout=45)
    if result.returncode:
        raise RuntimeError(f'remote status read failed (ssh exit {result.returncode})')
    return json.loads(result.stdout)


def validate_predecessor(cfg, authority):
    p = cfg['predecessor']
    blockers = []
    assignment_path = Path(p['assignment'])
    if not assignment_path.is_file() or sha(assignment_path) != p['assignment_sha256']:
        blockers.append('predecessor_assignment_hash_mismatch')
        return blockers
    assignment = read_json(assignment_path)
    if assignment.get('cases') != p['cases'] or assignment.get('selected_ir_sha256') != p['ir_sha256']:
        blockers.append('predecessor_assignment_scope_mismatch')
    teacher = authority.get('additional_teachers', {}).get(p['authority_key'])
    if not isinstance(teacher, dict):
        blockers.append('current_authority_binding_missing')
    elif (teacher.get('assignment') != str(assignment_path) or
          teacher.get('assignment_sha256') != p['assignment_sha256'] or
          teacher.get('campaign') != p['local_campaign']):
        blockers.append('current_authority_does_not_name_exact_predecessor')

    remote = remote_snapshot(p['host'], p['remote_directory'], p['service'],
        server=cfg_successor_server(cfg), model_id=cfg_successor_model(cfg),
        min_free_mib=cfg_successor(cfg).get('minimum_free_mib', 0),
        min_available_memory_mib=cfg_successor(cfg).get('minimum_available_memory_mib', 0),
        include_events=False)
    status = remote.get('status')
    if not isinstance(status, dict):
        blockers.append('remote_worker_status_missing')
        return blockers
    if (status.get('state') != 'finished' or status.get('status') != 'complete' or
            status.get('disposition') != 'complete' or remote.get('unit_active_exit') == 0):
        # Avoid hashing thousands of imported artifacts on every poll while
        # the predecessor is plainly still running. The final full validation
        # below is still required after it reaches an inactive terminal state.
        blockers.append('predecessor_still_running')
        return blockers
    remote = remote_snapshot(p['host'], p['remote_directory'], p['service'],
        server=cfg_successor_server(cfg), model_id=cfg_successor_model(cfg),
        min_free_mib=cfg_successor(cfg).get('minimum_free_mib', 0),
        min_available_memory_mib=cfg_successor(cfg).get('minimum_available_memory_mib', 0),
        include_events=True)
    status = remote.get('status')
    if not isinstance(status, dict) or status.get('state') != 'finished' or status.get('status') != 'complete' or status.get('disposition') != 'complete':
        blockers.append('predecessor_terminal_status_changed_during_recheck')
        return blockers
    # systemctl is-active returns 3 for inactive and 4 for absent. Everything
    # else is either active or an inability to prove inactivity.
    if remote.get('unit_active_exit') not in (3, 4):
        blockers.append('remote_service_not_proven_inactive')
    blockers.extend(validate_remote_environment(remote))
    events = remote.get('events') or []
    finishes = [e for e in events if e.get('event') == 'pool_finish']
    if not finishes:
        blockers.append('remote_pool_finish_event_missing')
    else:
        finish = finishes[-1]
        accounting = finish.get('output_accounting') or {}
        if (finish.get('status') != 'complete' or finish.get('exit_code') != 0 or
            accounting.get('complete') is not True or
            accounting.get('disposition') != 'all_exact_results_exported' or
            accounting.get('expected_jobs') != p['cases'] or
            accounting.get('exact_result_rows') != p['cases'] or
            accounting.get('output_rows') != p['cases'] or
            accounting.get('manifest_valid') is not True or
            accounting.get('explicitly_generation_held') != 0):
            blockers.append('remote_exact_output_accounting_incomplete')

    # Reconcile the imported ledger by this assignment SHA only. Importer's
    # new_rows is a per-poll delta and must not be mistaken for campaign total.
    import_path = Path(p['import_status'])
    sync_path = Path(p['sync_status'])
    ledger_path = Path(p['import_ledger'])
    if not import_path.is_file() or not ledger_path.is_file() or not sync_path.is_file():
        blockers.append('local_sync_import_receipt_or_ledger_missing')
        return blockers
    imported = read_json(import_path)
    sync = read_json(sync_path)
    if sync.get('transfer_exit_code') != 0 or sync.get('import_exit_code') != 0:
        blockers.append('latest_import_sync_failed')
    assignment_digest = p['assignment_sha256']
    if p.get('import_assignment'):
        # Preserve old ledger identities when a reviewed assignment changed only
        # its native-proof attestation, never its executable scope or controls.
        imported_assignment_path = Path(p['import_assignment'])
        imported_digest = p.get('import_assignment_sha256')
        if (not imported_assignment_path.is_file() or sha(imported_assignment_path) != imported_digest or
                assignment.get('supersedes_assignment_sha256') != imported_digest):
            return blockers + ['predecessor_import_assignment_binding_invalid']
        previous = read_json(imported_assignment_path)
        ignored = {'native_review_sha256', 'supersedes_assignment_sha256'}
        projection = lambda data: {key: value for key, value in data.items() if key not in ignored}
        if projection(previous) != projection(assignment):
            return blockers + ['predecessor_import_assignment_scope_changed']
        assignment_digest = imported_digest
    ir_ids = [r.get('id') for _, r in rows(p['ir'])]
    if len(ir_ids) != p['cases'] or len(set(ir_ids)) != p['cases']:
        blockers.append('predecessor_ir_identity_count_mismatch')
        return blockers
    imported_rows = [r for _, r in rows(ledger_path) if r.get('assignment_sha256') == assignment_digest]
    imported_ids = [r.get('program_id') for r in imported_rows]
    if (len(imported_ids) != p['cases'] or len(set(imported_ids)) != p['cases'] or
        set(imported_ids) != set(ir_ids)):
        blockers.append('imported_ledger_does_not_reconcile_every_assigned_case')
    for row in imported_rows:
        artifact = Path(row.get('path', ''))
        if not artifact.is_file() or sha(artifact) != row.get('sha256'):
            blockers.append('imported_artifact_missing_or_hash_mismatch')
            break
    if imported.get('total_unique_artifacts') != p['cases']:
        blockers.append('importer_unique_artifact_total_mismatch')
    return blockers


def validate_candidate(cfg):
    c = cfg['successor']
    blockers = []
    assignment_path = Path(c['assignment'])
    auth_path = Path(c['authorization'])
    native_path = Path(c['native_review'])
    for label, path in [('successor_assignment', assignment_path), ('successor_authorization', auth_path),
                        ('successor_native_review', native_path)]:
        if not path.is_file():
            blockers.append(f'{label}_not_root_approved_or_not_created')
    if blockers:
        return blockers
    assignment = read_json(assignment_path)
    auth = read_json(auth_path)
    native = read_json(native_path)
    for pinned_path, expected in cfg.get('artifact_hashes', {}).items():
        target = Path(pinned_path)
        if not target.is_file() or sha(target) != expected:
            blockers.append(f'controller_helper_pin_mismatch:{pinned_path}')
    controller_path = Path(__file__).resolve()
    if cfg.get('artifact_hashes', {}).get(str(controller_path)) != sha(controller_path):
        blockers.append('controller_implementation_not_hash_pinned')
    ir_path = Path(c['cases_ir'])
    proof_path = Path(c['selection_proof'])
    if not ir_path.is_file() or sha(ir_path) != c['ir_sha256']:
        blockers.append('successor_ir_missing_or_hash_mismatch')
    if not proof_path.is_file() or sha(proof_path) != c['selection_proof_sha256']:
        blockers.append('successor_selection_proof_hash_mismatch')
    arm_manifest = Path(c['arm_runtime_manifest'])
    if not arm_manifest.is_file() or sha(arm_manifest) != c['runtime_manifest_sha256']:
        blockers.append('successor_arm_runtime_manifest_hash_mismatch')
    if sha(assignment_path) != c['assignment_sha256']:
        blockers.append('successor_assignment_hash_mismatch')
    if assignment.get('launch_authorized') is not True or assignment.get('status') not in {'approved_pending_launch', 'approved'}:
        blockers.append('successor_assignment_not_explicitly_root_approved')
    if assignment.get('cases') != c['cases'] or assignment.get('selected_ir_sha256') != c['ir_sha256']:
        blockers.append('successor_assignment_scope_mismatch')
    if sha(auth_path) != c['authorization_sha256'] or auth.get('status') != 'approved':
        blockers.append('successor_authorization_not_root_approved_or_hash_mismatch')
    if (auth.get('ir_sha256') != c['ir_sha256'] or auth.get('count') != c['cases'] or
        auth.get('runtime_manifest_sha256') != c['runtime_manifest_sha256']):
        blockers.append('successor_authorization_scope_mismatch')
    expected_auth_fields = {
        'model_id': assignment.get('model'), 'workers': assignment.get('workers'),
        'model_concurrency': assignment.get('model_concurrency'),
        'max_model_requests': assignment.get('max_model_requests_per_case'),
        'max_turns': assignment.get('max_turns_per_case'),
        'chat_request_config_sha256': assignment.get('chat_request_config_sha256'),
        'transport_retries': assignment.get('transport_retries'),
        'retry_delay_ms': assignment.get('retry_delay_ms'),
        'final_export_only': assignment.get('final_export_only'),
        'execution_plans': assignment.get('execution_plans'),
    }
    if any(value is not None and auth.get(key) != value for key, value in expected_auth_fields.items()):
        blockers.append('successor_authorization_runtime_controls_differ_from_assignment')
    if sha(native_path) != c['native_review_sha256'] or native.get('status') != 'approved':
        blockers.append('successor_native_review_not_root_approved_or_hash_mismatch')
    if auth.get('native_review_sha256') != c['native_review_sha256']:
        blockers.append('successor_authorization_native_review_hash_mismatch')
    campaign = Path(c['local_campaign'])
    for key, path in (('authorization_path', auth_path), ('native_review_path', native_path)):
        expected_remote = str(Path(c['remote_directory']) / path.relative_to(campaign))
        if auth.get(key) != expected_remote:
            blockers.append(f'successor_authorization_receipt_path_mismatch:{key}')
    if (native.get('ir_sha256') != c['ir_sha256'] or native.get('runtime_manifest_sha256') != c['runtime_manifest_sha256'] or
        native.get('range') != {'start': 0, 'count': c['cases']} or
        native.get('current_admitted') != c['cases'] or native.get('materializer_accepted') != c['cases'] or
        native.get('unlinked') != 0 or native.get('model_calls') != 0 or native.get('provider_calls') != 0):
        blockers.append('successor_native_proof_scope_or_counts_invalid')
    if (native.get('source_conversion_valid') != c['cases'] or
        native.get('all_selected_rows_pass_runner_identity_join') is not True or
        native.get('all_selected_rows_current_policy_unheld') is not True or
        native.get('actual_execution_arch') not in {'aarch64', 'arm64'} or
        'arm64' not in str(native.get('actual_node', ''))):
        blockers.append('successor_native_source_or_identity_gate_invalid')
    native_rows = native_path.parent / native.get('native_reference_rows_path', '')
    native_report = native_path.parent / native.get('proof_report_path', '')
    if (not native_rows.is_file() or sha(native_rows) != native.get('native_reference_rows_sha256') or
        not native_report.is_file() or sha(native_report) != native.get('proof_report_sha256')):
        blockers.append('successor_native_proof_child_artifact_missing_or_hash_mismatch')
    if cfg.get('root_approved') is True:
        approval_path = Path(cfg.get('controller_approval_receipt', ''))
        if (not approval_path.is_file() or sha(approval_path) != cfg.get('controller_approval_receipt_sha256')):
            blockers.append('root_controller_approval_receipt_missing_or_hash_mismatch')
        else:
            approval = read_json(approval_path)
            if (approval.get('status') != 'approved' or
                approval.get('assignment_sha256') != c['assignment_sha256'] or
                approval.get('authorization_sha256') != c['authorization_sha256'] or
                approval.get('native_review_sha256') != c['native_review_sha256']):
                blockers.append('controller_approval_receipt_scope_mismatch')
        script = c.get('remote_launch_script')
        expected_script_sha = c.get('remote_launch_script_sha256')
        if (not script or not isinstance(expected_script_sha, str) or
            not re.fullmatch(r'[0-9a-f]{64}', expected_script_sha) or
            not re.fullmatch(r'[A-Za-z0-9_.@-]+', str(c.get('service', '')))):
            blockers.append('root_approved_controller_missing_remote_launch_script_pin')
        else:
            remote = remote_snapshot(cfg['predecessor']['host'], c['remote_directory'], c['service'], script,
                server=auth.get('server'), model_id=assignment.get('model'),
                min_free_mib=assignment.get('minimum_free_mib', auth.get('min_free_mib', 0)),
                min_available_memory_mib=assignment.get('minimum_available_memory_mib', auth.get('min_available_memory_mib', 0)))
            actual = (remote.get('launch_script') or {}).get('sha256')
            if actual != expected_script_sha:
                blockers.append('remote_launch_script_hash_mismatch')
            if remote.get('unit_active_exit') not in (3, 4):
                blockers.append('successor_remote_service_already_active_or_unproven')
            if remote.get('status') is not None or remote.get('events'):
                blockers.append('successor_remote_campaign_already_has_run_evidence')
            blockers.extend(validate_remote_environment(remote))
        if not c.get('sync_helper') or not Path(c['sync_helper']).is_file() or sha(c['sync_helper']) != c.get('sync_helper_sha256'):
            blockers.append('local_sync_helper_pin_missing_or_changed')
    return blockers


def validate_payload_separation(cfg):
    proof_paths = [Path(p) for p in cfg['global_payload_proofs']]
    current = Path(cfg['successor']['selection_proof'])
    if not current.is_file():
        return ['successor_selection_proof_missing']
    seen = set()
    current_ir = None
    for path in proof_paths + [current]:
        if not path.is_file():
            return [f'global_payload_proof_missing:{path}']
        for _, row in rows(path):
            signature = row.get('payload_signature') or row.get('task_payload_signature_sha256')
            if signature is None and path == current:
                # Source-preservation proofs can omit a redundant payload digest.
                # Derive it from the exact pinned IR, with an explicit helper pin
                # and a one-to-one proof identity join; never skip separation.
                from approve_dgx_v9_pool import payload_signature
                helper = Path(__file__).with_name('approve_dgx_v9_pool.py').resolve()
                if cfg.get('artifact_hashes', {}).get(str(helper)) != sha(helper):
                    return ['payload_signature_derivation_helper_not_pinned']
                ir_path = Path(cfg['successor']['cases_ir'])
                if not ir_path.is_file() or sha(ir_path) != cfg['successor']['ir_sha256']:
                    return ['payload_signature_derivation_ir_hash_mismatch']
                if current_ir is None:
                    current_ir = [r for _, r in rows(ir_path)]
                index = row.get('index')
                if (type(index) is not int or not 0 <= index < len(current_ir)
                        or row.get('id') != current_ir[index].get('id')):
                    return ['payload_signature_derivation_proof_identity_mismatch']
                signature = payload_signature(current_ir[index])
            if not isinstance(signature, str) or not re.fullmatch(r'[0-9a-f]{64}', signature):
                return [f'payload_signature_missing_or_invalid:{path}']
            if path == current:
                if signature in seen:
                    return ['successor_contains_duplicate_payload_signature']
                seen.add(signature)
            elif signature in seen:
                return [f'cross_campaign_payload_collision:{path}:{signature}']
            else:
                seen.add(signature)
    return []


def validate_preparation_screen(cfg):
    path = Path(cfg['successor']['preparation'])
    if not path.is_file() or sha(path) != cfg['successor']['preparation_sha256']:
        return ['successor_preparation_screen_missing_or_hash_mismatch']
    data = read_json(path)
    screen = data.get('prior_screen') or {}
    if cfg['stage'] == 'v9-v2':
        # v2 split its history receipt across preparation v1 and v2. The latter
        # binds the v1 parent SHA and explicitly records the 409+512 screens.
        parent = Path(data.get('parent_preparation_path', ''))
        parent_data = read_json(parent) if parent.is_file() else {}
        if (not parent.is_file() or
            data.get('status') != 'prepared_for_root_review_not_authorized_not_launched' or
            data.get('parent_preparation_sha256') != sha(parent)):
            return ['v9-v2_parent_preparation_binding_invalid']
        if (parent_data.get('prior_history_ledger_rows') != 6055 or
            parent_data.get('unreadable_history_artifacts') != 0 or
            parent_data.get('v8_cases') != 512 or
            parent_data.get('selected_cases') != 1024 or
            'v7 409-case import ledger' not in data.get('important_history_note', '')):
            return ['v9-v2_history_scope_does_not_match_reviewed_preparation']
        return []
    if (screen.get('unreadable_artifacts') != 0 or screen.get('all exact prior/generated collisions') != 0):
        return ['successor_global_history_screen_not_clean']
    if cfg['stage'] == 'v9-v3' and (screen.get('v9v2_cases') != 1024 or screen.get('v8_cases') != 512):
        return ['v9-v3_history_scope_does_not_include_v8_and_v9-v2']
    return []


def check(cfg):
    authority = read_json(cfg['authority'])
    blockers = validate_predecessor(cfg, authority)
    if 'predecessor_still_running' in blockers:
        return {'schema': 'natlang.dgx_conditional_handoff_check/1', 'stage': cfg['stage'],
                'checked_predecessor': cfg['predecessor']['label'], 'successor': cfg['successor']['label'],
                'ready': False, 'blockers': blockers, 'heavy_import_artifact_hashes_skipped': True,
                'launch_performed': False, 'approval_written': False,
                'authority_mutated': False, 'sync_performed': False,
                'next_step': 'Wait for the pinned predecessor user service to finish; the full artifact/import checks run after terminal status.'}
    blockers.extend(validate_payload_separation(cfg))
    blockers.extend(validate_preparation_screen(cfg))
    blockers.extend(validate_candidate(cfg))
    return {'schema': 'natlang.dgx_conditional_handoff_check/1', 'stage': cfg['stage'],
            'checked_predecessor': cfg['predecessor']['label'], 'successor': cfg['successor']['label'],
            'ready': not blockers, 'blockers': blockers,
            'launch_performed': False, 'approval_written': False,
            'authority_mutated': False, 'sync_performed': False,
            'next_step': 'Root must create a new approval-bound immutable controller config; only then may --start perform the gated launch and sync handoff.'}


def launch_after_approval(cfg, config_path, expected_sha):
    config_path = Path(config_path).resolve()
    if sha(config_path) != expected_sha:
        raise ValueError('root-reviewed controller config SHA mismatch')
    cfg = read_json(config_path)
    if cfg.get('root_approved') is not True:
        raise ValueError('controller requires explicit root approval')
    result = check(cfg)
    if not result['ready']:
        raise ValueError('handoff gate blocked: ' + ', '.join(result['blockers']))
    p, c = cfg['predecessor'], cfg['successor']
    record = Path(c['launch_record'])
    record.parent.mkdir(parents=True, exist_ok=True)
    lock_path = record.with_suffix(record.suffix + '.lock')
    with lock_path.open('a') as lock:
        import fcntl
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if record.exists():
            raise ValueError('handoff launch record already exists; refuse duplicate launch')
        # Re-read authority under its lock immediately before any remote start.
        authority_path = Path(cfg['authority'])
        with authority_lock(authority_path):
            if sha(config_path) != expected_sha:
                raise ValueError('root-reviewed controller config changed while acquiring authority lock')
            cfg = read_json(config_path)
            p, c = cfg['predecessor'], cfg['successor']
            authority = read_json(authority_path)
            teacher = authority.get('additional_teachers', {}).get(p['authority_key'], {})
            if (teacher.get('assignment') != p['assignment'] or
                teacher.get('assignment_sha256') != p['assignment_sha256']):
                raise ValueError('predecessor authority changed before handoff')
            recheck = check(cfg)
            if not recheck['ready']:
                raise ValueError('handoff gate changed while acquiring locks: ' + ', '.join(recheck['blockers']))
            atomic_json(record, {'status': 'claimed', 'config_sha256': expected_sha,
                                 'predecessor_assignment_sha256': p['assignment_sha256'],
                                 'successor_assignment_sha256': c['assignment_sha256'],
                                 'started_at': time.time()})
            remote_command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', p['host'],
                'systemd-run --user --unit=' + shlex.quote(c['service']) + ' --collect '
                '--property=KillMode=mixed --property=TimeoutStopSec=600 '
                + shlex.quote(c['remote_launch_script'])]
            try:
                started = subprocess.run(remote_command, capture_output=True, text=True, timeout=45)
            except Exception as exc:
                atomic_json(record, {'status': 'start_unknown', 'config_sha256': expected_sha,
                    'error': f'{type(exc).__name__}: {exc}', 'launch_attempted_at': time.time(),
                    'unit': c['service'], 'launch_performed': None})
                raise
            # systemd-run may report a transport error after the remote unit was
            # accepted. Probe the actual user-unit state before calling this a
            # failure; never invite a blind duplicate retry after ambiguity.
            try:
                probe = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', p['host'],
                    'systemctl --user is-active ' + shlex.quote(c['service'])],
                    capture_output=True, text=True, timeout=30)
            except Exception as exc:
                atomic_json(record, {'status': 'start_unknown', 'config_sha256': expected_sha,
                    'systemd_run_exit_code': started.returncode,
                    'systemd_run_stdout': started.stdout[-1000:],
                    'error': f'Post-start probe {type(exc).__name__}: {exc}',
                    'unit': c['service'], 'launch_performed': None,
                    'launch_attempted_at': time.time()})
                raise
            is_active = probe.returncode == 0 and probe.stdout.strip() in {'active', 'activating'}
            proven_inactive = probe.returncode in (3, 4) and probe.stdout.strip() in {'inactive', 'unknown', 'not-found'}
            if not is_active:
                state = 'start_failed' if started.returncode and proven_inactive else 'start_unknown'
                atomic_json(record, {'status': state, 'config_sha256': expected_sha,
                    'exit_code': started.returncode, 'stderr': started.stderr[-1000:],
                    'probe_exit_code': probe.returncode, 'probe_text': probe.stdout.strip(),
                    'unit': c['service'], 'launch_performed': False if state == 'start_failed' else None,
                    'launch_attempted_at': time.time()})
                raise RuntimeError('remote systemd start not confirmed (' + state + ')')
            atomic_json(record, {'status': 'remote_started', 'config_sha256': expected_sha,
                'systemd_run_exit_code': started.returncode, 'systemd_run_stdout': started.stdout[-1000:],
                'unit': c['service'], 'launch_performed': True, 'remote_started_at': time.time()})
            try:
                previous = dict(teacher)
                authority.setdefault('completed_teacher_assignments', []).append({
                    **previous, 'completion_state': 'complete', 'handoff_record': str(record),
                    'completed_at': time.time()})
                teacher.update({'host': 'dgx', 'model': c['model'], 'model_revision': c['model_revision'],
                    'campaign': c['local_campaign'], 'assignment': c['assignment'],
                    'assignment_sha256': c['assignment_sha256'], 'service': c['service'],
                    'cases': c['cases'], 'state': 'starting',
                    'actual_runtime_manifest_sha256': c['runtime_manifest_sha256'],
                    'import_runtime_manifest_sha256': c['import_runtime_manifest_sha256'],
                    'model_concurrency': c['model_concurrency'], 'native_review_sha256': c['native_review_sha256'],
                    'remote_directory': c['remote_directory'], 'sync_status': c['sync_status'],
                    'import_status': c['import_status'], 'import_ledger': c['import_ledger'],
                    'last_handoff_from': previous.get('assignment')})
                authority['additional_teachers'][p['authority_key']] = teacher
                atomic_json(authority_path, authority)
                Path(c['sync_log']).parent.mkdir(parents=True, exist_ok=True)
                with Path(c['sync_log']).open('ab') as sync_log:
                    sync = subprocess.Popen([sys.executable, c['sync_helper'], c['local_campaign'], '--loop',
                        '--authority', str(authority_path), '--teacher-key', p['authority_key']],
                        cwd=cfg['repo_root'], stdin=subprocess.DEVNULL,
                        stdout=sync_log, stderr=subprocess.STDOUT, start_new_session=True)
                atomic_json(record, {'status': 'running', 'config_sha256': expected_sha,
                    'sync_pid': sync.pid, 'started_at': time.time(), 'unit': c['service'], 'launch_performed': True})
            except Exception as exc:
                atomic_json(record, {'status': 'started_handoff_failed', 'config_sha256': expected_sha,
                    'error': f'{type(exc).__name__}: {exc}', 'unit': c['service'],
                    'launch_performed': True, 'remote_started_at': time.time()})
                raise
    return {'status': 'launched', 'launch_performed': True, 'launch_record': str(record), 'sync_pid': sync.pid}


def launch_state(cfg):
    record_path = Path(cfg.get('successor', {}).get('launch_record', ''))
    if not record_path.is_file():
        return {'launch_state': 'not_started', 'launch_performed': False}
    try:
        record = read_json(record_path)
    except Exception as exc:
        return {'launch_state': 'record_unreadable', 'launch_performed': None,
                'launch_record_error': f'{type(exc).__name__}: {exc}'}
    state = record.get('status')
    if state in {'remote_started', 'running', 'started_handoff_failed', 'importer_finished', 'importer_failed'}:
        performed = True
    elif state in {'start_unknown', 'claimed'}:
        performed = None
    else:
        performed = False
    return {'launch_state': state, 'launch_performed': performed, 'launch_record': str(record_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--watch', action='store_true', help='poll checks until ready; combined with --start, launch once when the exact gate passes')
    parser.add_argument('--start', action='store_true', help='start only under a separately root-approved, hash-pinned config')
    parser.add_argument('--sha256', help='required exact root-approved config digest for --start')
    parser.add_argument('--interval', type=int, default=30)
    args = parser.parse_args()
    if args.interval < 5:
        parser.error('--interval must be at least 5 seconds')
    config_path = args.config.resolve()
    config = read_json(config_path)
    if args.start:
        if not args.sha256:
            parser.error('--start requires --sha256')
        while True:
            try:
                if sha(config_path) != args.sha256:
                    raise ValueError('root-reviewed controller config SHA mismatch during wait')
                config = read_json(config_path)
                prior_record = Path(config.get('successor', {}).get('launch_record', ''))
                if prior_record.exists():
                    raise ValueError('handoff launch record already exists; refusing to wait or launch twice')
                if args.watch:
                    result = check(config)
                    print(json.dumps(result, sort_keys=True), flush=True)
                    if not result.get('ready'):
                        time.sleep(args.interval)
                        continue
                launched = launch_after_approval(config, config_path, args.sha256)
                print(json.dumps(launched, sort_keys=True), flush=True)
                # The importer is our child in this service's systemd cgroup.
                # Retain its owner until it exits, outside all launch locks.
                _, status = os.waitpid(launched['sync_pid'], 0)
                exit_code = os.waitstatus_to_exitcode(status)
                record_path = Path(launched['launch_record'])
                record = read_json(record_path)
                record.update(status='importer_finished' if exit_code == 0 else 'importer_failed',
                              sync_exit_code=exit_code, importer_exited_at=time.time(), launch_performed=True)
                atomic_json(record_path, record)
                return 0 if exit_code == 0 else 2
            except Exception as exc:
                state = launch_state(config)
                print(json.dumps({'status': 'blocked', 'error': f'{type(exc).__name__}: {exc}',
                                  **state}), file=sys.stderr, flush=True)
                return 2
    while True:
        try:
            result = check(config)
        except Exception as exc:
            result = {'schema': 'natlang.dgx_conditional_handoff_check/1', 'stage': config.get('stage'),
                      'ready': False, 'blockers': [f'fail_closed:{type(exc).__name__}:{exc}'],
                      'launch_performed': False, 'approval_written': False,
                      'authority_mutated': False, 'sync_performed': False}
        print(json.dumps(result, sort_keys=True), flush=True)
        if result.get('ready') or not args.watch:
            return 0 if result.get('ready') else 2
        time.sleep(args.interval)


if __name__ == '__main__':
    sys.exit(main())
