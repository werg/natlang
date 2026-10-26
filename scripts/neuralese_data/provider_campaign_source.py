"""Authenticate closed provider trajectories without borrowing a sandbox actor.

The original per-case result, collector manifest, reviewed dispatch plan and
closed claim remain authority. This is source custody, not decision approval,
corpus registration, readiness or a claim of training exposure.
"""
from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path

from natlang_neuralese.common.hashing import canonical_json_sha256_hex, sha256_file_hex

SCHEMA = 'natlang.provider-campaign-source/1'

# The existing dispatcher is also a standalone scripts/ entrypoint. Its shared
# authority helpers must resolve identically when used by a module invocation.
SCRIPT_ROOT = str(Path(__file__).resolve().parents[1])
if SCRIPT_ROOT not in sys.path:
    sys.path.insert(0, SCRIPT_ROOT)


def bound(reference):
    path = Path(reference['path']).resolve(strict=True)
    if (path.stat().st_size != reference['bytes']
            or sha256_file_hex(path) != reference['sha256']):
        raise ValueError('provider source artifact changed: ' + str(path))
    return path


def validated_dispatch_plan(path, expected_sha256):
    from scripts.run_reviewed_luna_dispatcher import validate_plan
    plan, identity = validate_plan(path, expected_sha256)
    identity['identity_sha256'] = hashlib.sha256(json.dumps(
        identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return plan, identity


def source_binding(args, *, protection_reader):
    from scripts.run_reviewed_luna_dispatcher import _verify_ledger_identity, _verify_ledger_events

    manifest_path = Path(args.provider_source_manifest).resolve(strict=True)
    declaration = json.loads(manifest_path.read_text())
    if declaration.get('schema') != SCHEMA or declaration.get('training_admission') is not False:
        raise ValueError('provider source needs an unadmitted exact custody declaration')
    source = bound(declaration['source'])
    if source != Path(args.source).resolve(strict=True):
        raise ValueError('provider declaration names another combined source')
    plan_path = bound(declaration['plan'])
    plan, identity = validated_dispatch_plan(plan_path, declaration['plan']['sha256'])
    ledger_path = bound(declaration['claim_ledger'])
    events = [json.loads(line) for line in ledger_path.read_text().splitlines() if line.strip()]
    _verify_ledger_identity(events, identity)
    _verify_ledger_events(events, identity)
    claims = {event['claim_id']: event for event in events if event['event'] == 'claim'}
    terminals = {event['claim_id']: event for event in events if event['event'] == 'terminal'}
    programs = {index: json.loads(line) for index, line in
                enumerate(Path(plan['source']).read_text().splitlines()) if line.strip()}
    current = protection_reader(Path(args.protection_manifest))
    proof_paths = {'provider_source': manifest_path, 'source': source,
                   'dispatch_plan': plan_path, 'claim_ledger': ledger_path,
                   'transfer': manifest_path}
    source_files, selected_ids, groups, packet_programs = {}, set(), set(), {}
    collection_key = 'provider-campaign:' + declaration['plan']['sha256']
    combined = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    entries = declaration.get('results')
    if not isinstance(entries, list) or not entries or len(entries) != len(combined):
        raise ValueError('provider source must enumerate every complete original trajectory')
    for number, (entry, row) in enumerate(zip(entries, combined)):
        claim = claims.get(entry.get('claim_id'))
        terminal = terminals.get(entry.get('claim_id'))
        if not claim or not terminal or terminal.get('exit_code') != 0:
            raise ValueError('provider trajectory has no exact successfully closed claim')
        queue_path = bound(entry['queue'])
        if str(queue_path) != claim['queue'] or entry['queue']['sha256'] != claim['queue_sha256']:
            raise ValueError('provider trajectory queue differs from its actual claim')
        queue = [json.loads(line) for line in queue_path.read_text().splitlines() if line.strip()]
        if len(queue) != 1 or queue[0]['index'] != claim['index']:
            raise ValueError('provider source claim must name exactly one original case')
        result_path, receipt_path = bound(entry['result']), bound(entry['manifest'])
        if str(result_path) != queue[0]['output'] or receipt_path != Path(str(result_path)+'.manifest.json'):
            raise ValueError('provider result paths differ from the claimed collector output')
        actual = [json.loads(line) for line in result_path.read_text().splitlines() if line.strip()]
        receipt = json.loads(receipt_path.read_text())
        accounting = terminal.get('output_accounting') or {}
        if (len(actual) != 1 or actual[0] != row or not accounting.get('complete')
                or accounting.get('output_sha256') != entry['result']['sha256']
                or receipt.get('version') != 'natlang.teacher_batch.native/1'
                or receipt.get('source_sha256') != plan['source_sha256']
                or receipt.get('range') != {'start':claim['index'],'count':1}
                or receipt.get('completed') != 1 or receipt.get('missing')
                or receipt.get('output_sha256') != entry['result']['sha256']
                or receipt.get('model') != plan['model_id']):
            raise ValueError('provider result lacks complete exact collector accounting')
        program = programs[claim['index']]
        trace = row.get('provenance') or {}
        outcome = row.get('outcome') or {}
        if (row.get('version') != 'natlang.teacher_trajectory.native/1'
                or row.get('task',{}).get('program_ir') != program
                or trace.get('program_ir_sha256') != canonical_json_sha256_hex(program)
                or trace.get('model') != plan['model_id'] or trace.get('provider') != plan['provider']
                or not row.get('execution_identity',{}).get('execution_run_id')
                or not row.get('trajectory') or program.get('split') != 'train'
                or outcome.get('status') != 'done' or outcome.get('accepted') is not True
                or outcome.get('oracle',{}).get('accepted') is not True
                or not outcome.get('checks') or any(v is not True for v in outcome['checks'].values())):
            raise ValueError('provider trajectory differs from its actual actor/task/success')
        if row['id'] in source_files or program['id'] in selected_ids:
            raise ValueError('provider source repeats an execution or selected task')
        if set(program['source_groups']) & (current['held'] | current['quality'] | current['pending']):
            raise ValueError('provider source intersects current protected source groups')
        selected_ids.add(program['id']); groups.update(program['source_groups'])
        packet_programs[program['id']] = canonical_json_sha256_hex(program)
        source_files[row['id']] = {**entry['result'],'row_id':row['id'],
                                  'collection_source':collection_key,'source':str(result_path)}
        proof_paths.update({f'case:{number}:result':result_path,
                            f'case:{number}:manifest':receipt_path,
                            f'case:{number}:queue':queue_path})
    packet = {'selected_ids':selected_ids,'program_sha256':packet_programs,
              'train_groups':groups,'held_groups':set(),
              'proofs':{'selection':declaration['plan'],
                        'global_protection_join':{'path':str(args.protection_manifest),
                                                 'sha256':current['proof']['sha256']}}}
    return {'source_sha256':declaration['source']['sha256'],
            'transfer':{'schema':SCHEMA,'training_admission':False,
                        'combined':declaration['source'],'files':list(source_files.values())},
            'teacher':{'teacher':plan['model_id'],
                       'revision':'provider-managed; exact weight revision unavailable',
                       'provider':plan['provider'],'runtime':plan['runtime_receipt_sha256']},
            'evidence':{},'source_files':source_files,'selected_ids':selected_ids,
            'train_groups':groups,'held_groups':set(),'proof_paths':proof_paths,
            'proof_sha256':{key:sha256_file_hex(path)for key,path in proof_paths.items()},
            'collections':{collection_key:{**declaration['plan'],'runtime_model':plan['model_id']}},
            'packets':{collection_key:packet},'current':current}
