"""Authenticate closed provider trajectories without borrowing a sandbox actor.

The original per-case result, collector manifest, reviewed dispatch plan and
closed claim remain authority. This is source custody, not decision approval,
corpus registration, readiness or a claim of training exposure.
"""
from __future__ import annotations

import json
import os
import shutil
import hashlib
import subprocess
import sys
from pathlib import Path

from natlang_neuralese.common.hashing import canonical_json_sha256_hex, sha256_file_hex
from scripts.neuralese_data.teacher_outcome_contract import exact_honest_root_block
from scripts.neuralese_data.provider_judged_regrade import project as project_judged_outcome, validate_projection

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
    from scripts.run_reviewed_luna_dispatcher import digest as original_digest, validate_plan
    try:
        plan, identity = validate_plan(path, expected_sha256)
    except ValueError as error:
        # Old closed campaigns pin the exact dispatcher and queue code used at
        # collection. Verify those bytes from Git history while still applying
        # the current plan validator to the frozen plan, case list, runtime and
        # output ledger. This is an explicit continuation; never rewrite the
        # historical plan or treat current source files as its execution code.
        attestation_path = Path(path).with_name('dispatch-plan-history-attestation.json')
        if (not any(marker in str(error) for marker in
                    ('Missing or changed reviewed artifact pin:', 'Reviewed artifact changed:'))
                or not attestation_path.is_file()):
            raise
        attestation = json.loads(attestation_path.read_text())
        plan_sha = sha256_file_hex(path)
        commit = attestation.get('git_commit')
        artifacts = attestation.get('historical_artifacts')
        if (attestation.get('schema') != 'natlang.reviewed-dispatch-history-attestation/1'
                or attestation.get('plan_path') != str(Path(path).resolve())
                or attestation.get('plan_sha256') != plan_sha
                or not isinstance(commit, str) or len(commit) != 40
                or not isinstance(artifacts, dict) or not artifacts):
            raise ValueError('historical dispatcher attestation does not bind this frozen plan') from error
        repo = Path(__file__).resolve().parents[2]
        expected_pins = json.loads(Path(path).read_text()).get('artifact_hashes', {})
        if set(artifacts) - set(expected_pins):
            raise ValueError('historical dispatcher attestation names unpinned artifacts') from error
        historical = {}
        for absolute, binding in artifacts.items():
            target = Path(absolute).resolve(strict=False)
            if target != repo and repo not in target.parents:
                raise ValueError('historical dispatcher artifact is outside the repository') from error
            relative = str(target.relative_to(repo))
            if (not isinstance(binding, dict) or binding.get('git_path') != relative
                    or binding.get('sha256') != expected_pins[absolute]
                    or binding.get('git_commit') != commit):
                raise ValueError('historical dispatcher artifact binding differs from plan pins') from error
            result = subprocess.run(['git', '-C', str(repo), 'show', f'{commit}:{relative}'],
                                    capture_output=True, check=True)
            if hashlib.sha256(result.stdout).hexdigest() != binding['sha256']:
                raise ValueError('historical dispatcher artifact differs from its Git object') from error
            historical[str(target)] = binding['sha256']
        def bound_digest(target):
            absolute = str(Path(target).resolve(strict=False))
            if absolute in historical:
                return historical[absolute]
            return original_digest(target)
        plan, identity = validate_plan(path, expected_sha256, digest_fn=bound_digest)
        identity['dispatch_plan_history_attestation'] = str(attestation_path.resolve())
    identity['identity_sha256'] = hashlib.sha256(json.dumps(
        identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return plan, identity


def ledger_dispatch_identity(identity):
    """Recover the exact immutable ledger identity for historically pinned plans.

    The attestation is a later custody artifact and was not part of the claim
    header. Keep it in conversion provenance, but verify claims against the
    original dispatch identity reconstructed without that later annotation.
    """
    if not identity.get('dispatch_plan_history_attestation'):
        return identity
    ledger = {key: value for key, value in identity.items()
              if key not in {'dispatch_plan_history_attestation', 'identity_sha256'}}
    ledger['identity_sha256'] = hashlib.sha256(json.dumps(
        ledger, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return ledger


def collector_program_digests(runtime, programs):
    """Use the authenticated collector's wire digest, not Python key ordering.

    The collector sorts keys with JavaScript localeCompare. Python canonical
    JSON sorts Unicode code points, which differs for real source-file keys.
    The caller has already verified this frozen runtime against its receipt.
    """
    module = (Path(runtime) / 'dist/teacher/collector.js').resolve(strict=True)
    node = os.environ.get('NATLANG_NODE') or shutil.which(
        'node', path=str(Path.home() / '.local/bin') + os.pathsep + os.environ.get('PATH', ''))
    if not node:
        raise ValueError('Node is required to verify the original collector program digests')
    code = ('import fs from "node:fs";'
            'const {recordDigest}=await import(process.argv[1]);'
            'process.stdout.write(JSON.stringify(JSON.parse(fs.readFileSync(0,"utf8")).map(recordDigest)));')
    result = subprocess.run([node, '--input-type=module', '-e', code, module.as_uri()],
        input=json.dumps(programs, ensure_ascii=False), text=True, capture_output=True, check=True, timeout=60)
    digests = json.loads(result.stdout)
    if (not isinstance(digests, list) or len(digests) != len(programs)
            or any(not isinstance(item, str) or len(item) != 64
                   or any(char not in '0123456789abcdef' for char in item) for item in digests)):
        raise ValueError('frozen collector returned malformed program digests')
    return digests


def source_binding(args, *, protection_reader):
    from scripts.run_reviewed_luna_dispatcher import _verify_ledger_identity, _verify_ledger_events
    from natlang_neuralese.common.custody import ProtectedWorlds, REGISTRY as protected_worlds_registry

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
    ledger_identity = ledger_dispatch_identity(identity)
    _verify_ledger_identity(events, ledger_identity)
    _verify_ledger_events(events, ledger_identity)
    claims = {event['claim_id']: event for event in events if event['event'] == 'claim'}
    terminals = {event['claim_id']: event for event in events if event['event'] == 'terminal'}
    programs = {index: json.loads(line) for index, line in
                enumerate(Path(plan['source']).read_text().splitlines()) if line.strip()}
    current = protection_reader(Path(args.protection_manifest))
    current_worlds = ProtectedWorlds.load()
    current_worlds_sha = sha256_file_hex(protected_worlds_registry)
    proof_paths = {'provider_source': manifest_path, 'source': source,
                   'dispatch_plan': plan_path, 'claim_ledger': ledger_path,
                   'transfer': manifest_path, 'current_protected_worlds': protected_worlds_registry}
    history = identity.get('dispatch_plan_history_attestation')
    if history:
        proof_paths['dispatch_plan_history_attestation'] = history
    source_files, selected_ids, groups, packet_programs = {}, set(), set(), {}
    declaration_regrades = declaration.get('oracle_regrades', [])
    if not isinstance(declaration_regrades, list):
        raise ValueError('provider source oracle regrades must be an exact list')
    regrades_by_case = {}
    for item in declaration_regrades:
        if (not isinstance(item, dict) or not isinstance(item.get('case_id'), str)
                or item['case_id'] in regrades_by_case):
            raise ValueError('provider source oracle regrades contain duplicate or malformed case bindings')
        regrades_by_case[item['case_id']] = item
    validated_regrades = set()
    collection_key = 'provider-campaign:' + declaration['plan']['sha256']
    combined = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    wire_digests = collector_program_digests(plan['runtime'], [row.get('task', {}).get('program_ir') for row in combined])
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
        if (len(actual) != 1 or not accounting.get('complete')
                or accounting.get('output_sha256') != entry['result']['sha256']
                or receipt.get('version') != 'natlang.teacher_batch.native/1'
                or receipt.get('source_sha256') != plan['source_sha256']
                or receipt.get('range') != {'start':claim['index'],'count':1}
                or receipt.get('completed') != 1 or receipt.get('missing')
                or receipt.get('output_sha256') != entry['result']['sha256']
                or receipt.get('model') != plan['model_id']):
            raise ValueError('provider result lacks complete exact collector accounting')
        program = programs[claim['index']]
        row_regrade = row.get('source_regrade')
        regrade_projection = row_regrade.get('projection') if isinstance(row_regrade, dict) else None
        regrade_ref = row_regrade.get('receipt') if isinstance(row_regrade, dict) else None
        if regrade_projection is not None:
            if (not isinstance(regrade_ref, dict) or entry.get('source_regrade') != regrade_ref
                    or row_regrade.get('schema') != 'natlang.provider-judged-outcome-regrade/1'):
                raise ValueError('derived provider row lacks its exact regrade receipt reference')
            declared_regrade = regrades_by_case.get(row.get('id'))
            if (not isinstance(declared_regrade, dict) or declared_regrade.get('projection') != regrade_ref
                    or declared_regrade.get('proof', {}).get('path') != regrade_projection.get('qualification_path')
                    or declared_regrade.get('proof', {}).get('sha256') != regrade_projection.get('qualification_sha256')):
                raise ValueError('provider source declaration omits or changes this row regrade binding')
            validated_regrades.add(row['id'])
            regrade_path = bound(regrade_ref)
            stored_projection = json.loads(regrade_path.read_text())
            if stored_projection != regrade_projection:
                raise ValueError('derived row regrade projection differs from its sealed proof file')
            qualified_row, verified_projection = project_judged_outcome(actual[0], program,
                Path(regrade_projection['source_worlds_path']), Path(regrade_projection['qualification_path']),
                plan_path=plan_path, plan_sha256=declaration['plan']['sha256'], claim_id=entry['claim_id'],
                claim_index=claim['index'],
                queue_path=queue_path, result_path=result_path, result_manifest_path=receipt_path,
                runtime=Path(plan['runtime']))
            if verified_projection != regrade_projection:
                raise ValueError('provider regrade projection does not revalidate against original source artifacts')
            qualified_row['source_regrade'] = row_regrade
            if canonical_json_sha256_hex(qualified_row) != canonical_json_sha256_hex(row):
                raise ValueError('provider derived row differs from its exact regrade projection')
            validate_projection(actual[0], row, regrade_projection)
            proof_paths[f'case:{number}:outcome_regrade'] = regrade_path
            proof_paths[f'case:{number}:qualification'] = Path(regrade_projection['qualification_path']).resolve(strict=True)
            proof_paths[f'case:{number}:sol_grade'] = Path(regrade_projection['sol_regrade_path']).resolve(strict=True)
        elif actual[0] != row:
            raise ValueError('provider trajectory differs from the exact collector result without a regrade')
        trace = row.get('provenance') or {}
        outcome = row.get('outcome') or {}
        accepted_block = exact_honest_root_block(row, program)
        if (row.get('version') != 'natlang.teacher_trajectory.native/1'
                or row.get('task',{}).get('program_ir') != program
                or trace.get('program_ir_sha256') != wire_digests[number]
                or trace.get('model') != plan['model_id'] or trace.get('provider') != plan['provider']
                or not row.get('execution_identity',{}).get('execution_run_id')
                or not row.get('trajectory') or program.get('split') != 'train'
                or (outcome.get('status') != 'done' and not accepted_block)
                or outcome.get('accepted') is not True
                or outcome.get('oracle',{}).get('accepted') is not True
                or not outcome.get('checks') or any(v is not True for v in outcome['checks'].values())):
            raise ValueError('provider trajectory differs from its actual actor/task/success')
        if row['id'] in source_files or program['id'] in selected_ids:
            raise ValueError('provider source repeats an execution or selected task')
        if set(program['source_groups']) & (current['held'] | current['quality'] | current['pending']):
            raise ValueError('provider source intersects current protected source groups')
        try:
            current_worlds.check(program)
        except ValueError as error:
            raise ValueError('provider source intersects current ProtectedWorlds custody: ' + str(error)) from error
        selected_ids.add(program['id']); groups.update(program['source_groups'])
        packet_programs[program['id']] = canonical_json_sha256_hex(program)
        source_files[row['id']] = {**entry['result'],'row_id':row['id'],
                                  'collection_source':collection_key,'source':str(result_path),
                                  **({'source_regrade': regrade_ref} if regrade_projection is not None else {})}
        proof_paths.update({f'case:{number}:result':result_path,
                            f'case:{number}:manifest':receipt_path,
                            f'case:{number}:queue':queue_path})
    if validated_regrades != set(regrades_by_case):
        raise ValueError('provider source declares unused or missing outcome regrades')
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
            'packets':{collection_key:packet},'current':current,
            'current_protected_worlds':{'path':str(protected_worlds_registry),
                'sha256':current_worlds_sha,'identity_digest':current_worlds.digest}}
