#!/usr/bin/env python3
"""Independently verify an exported full-state receipt without its original paths.

This validates immutable bytes and checkpoint custody; the isolated restore
qualification bound by the plan is a separate, stronger runtime exercise.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from natlang_neuralese.train.checkpoint_offload import (RecoveryExportQueue, file_sha256,
    recovery_closure_identity, validate_full_recovery_checkpoint, validate_recovery_plan_shape, _plan_matches_checkpoint)


def verify_proof_chain(plan, lookup, seen=None):
    from natlang_neuralese.train.native_resident import digest
    validate_recovery_plan_shape(plan);ref=plan['restore_qualification'];seen=set() if seen is None else set(seen)
    if ref['sha256'] in seen or len(seen)>=128:raise ValueError('export proof chain is cyclic or too long')
    seen.add(ref['sha256']);proof=lookup(ref);checks=proof.get('checks') or {}
    if (proof.get('schema')!='natlang.local-recovery-restore-qualification/1' or proof.get('status')!='passed'
        or proof.get('inputs')!=plan['inputs'] or proof.get('restore')!=plan.get('restore')
        or proof.get('additional_inputs',[])!=plan.get('additional_inputs',[]) or proof.get('controller_config')!=plan['controller']['config']
        or proof.get('require_static_contract')!=plan.get('require_static_contract') or proof.get('sft_static_contract')!=plan.get('sft_static_contract')
        or proof.get('control_snapshot')!=plan.get('control_snapshot') or proof.get('closure_identity')!=recovery_closure_identity(plan)):
        raise ValueError('export restore qualification differs')
    if proof.get('mode')=='resident-append/1':
        from natlang_neuralese.train.native_recovery_append import validate_committed_record
        predecessor=lookup(proof['predecessor_plan']);evidence=lookup(proof['transition_evidence']);draft=lookup(evidence['successor_draft'])
        staged_intent=lookup(evidence['staged_intent']);staged=lookup(evidence['staged_closure']);staged_draft=lookup(evidence['staged_draft'])
        intent=evidence.get('intent') or {};committed=evidence.get('committed') or {};payload=committed.get('payload') or {}
        validate_committed_record(committed,payload,intent.get('generation'),evidence.get('actual_global_step'))
        committed_snapshot=lookup(evidence['resident_current'])
        roles_payload=((payload.get('plan') or {}).get('stage_inputs') or {}).get(intent.get('stage_id'),{})
        sha_of=lambda value:value.get('sha256') if isinstance(value,dict) else None
        fixed={'runtime','environment','system-python','system-libraries','node','interpreter','base','tokenizer','task-assets'}
        roles=lambda p:{(x['role'],x['path'],x['sha256']) for x in p['closure'] if x['role'] in fixed}
        prior_foundation=predecessor['inputs']['foundation'];next_foundation=draft['inputs']['foundation']
        retained={(x['kind'],x['path'],x['sha256']) for x in draft.get('additional_inputs',[])}
        if (committed_snapshot!=committed or evidence.get('schema')!='natlang.local-recovery-resident-transition/1'
            or staged_intent.get('schema')!='natlang.local-recovery-resident-append-intent/1'
            or staged.get('schema')!='natlang.local-recovery-staged-closure/1'
            or staged_draft!=draft or staged_intent.get('predecessor_plan')!=proof['predecessor_plan']
            or staged_intent.get('successor_draft')!=evidence['staged_draft'] or staged_intent.get('staged_closure')!=evidence['staged_closure']
            or staged_intent.get('generation')!=digest(payload) or staged_intent.get('predecessor_generation')!=payload.get('predecessor')
            or intent.get('generation')!=digest(payload) or intent.get('predecessor_plan_sha256')!=RecoveryExportQueue._plan_digest(predecessor)
            or intent.get('successor_draft_sha256')!=RecoveryExportQueue._plan_digest(draft) or intent.get('inputs')!=plan['inputs']
            or payload.get('stage_id')!=intent.get('stage_id') or payload.get('predecessor')!=intent.get('predecessor_generation')
            or sha_of(payload.get('prior_train'))!=sha_of(predecessor['inputs']['train']) or sha_of(payload.get('prior_held'))!=sha_of(predecessor['inputs']['held'])
            or sha_of(payload.get('train'))!=sha_of(draft['inputs']['train']) or sha_of(payload.get('held'))!=sha_of(draft['inputs']['held'])
            or any(sha_of(roles_payload.get(role))!=sha_of(draft['inputs'][target]) for role,target in
                   (('native_graphs','train'),('foundation_native_graphs','foundation'),('held_native_graphs','held'),('ce_supervision_policy','ce')))
            or staged.get('plan_sha256')!=RecoveryExportQueue._plan_digest(draft)
            or {x['sha256'] for x in staged.get('objects',[])}!={x['sha256'] for x in draft['closure']}
            or any(draft.get(k)!=plan.get(k) for k in ('inputs','restore','controller','control_snapshot','additional_inputs','session','destination','journal_directory'))
            or not {(x['role'],x['path'],x['sha256']) for x in draft['closure']} <= {(x['role'],x['path'],x['sha256']) for x in plan['closure']}
            or roles(predecessor)!=roles(draft) or predecessor['restore']!=draft['restore'] or predecessor['control_snapshot']!=draft['control_snapshot']
            or predecessor['controller']!=draft['controller'] or predecessor['inputs']['held']!=draft['inputs']['held']
            or (next_foundation!=prior_foundation and not (prior_foundation==predecessor['inputs']['train'] and next_foundation==draft['inputs']['train']))
            or any(ref and ref!=draft['inputs'][role] and (('ce' if role=='ce' else 'source'),ref['path'],ref['sha256']) not in retained
                   for role,ref in ((role,predecessor['inputs'][role]) for role in ('train','held','ce')))
            or committed.get('generation')!=digest(payload) or not committed.get('validation')
            or evidence.get('actual_static_contract')!=plan['sft_static_contract'] or committed.get('at_global_step')!=evidence.get('actual_global_step')
            or set(checks)!={'predecessor_restore_anchor','resident_commit_validated','successor_metadata_validated'} or not all(x is True for x in checks.values())):
            raise ValueError('export resident transition proof differs')
        verify_proof_chain(predecessor,lookup,seen)
    else:
        checkpoint=proof.get('checkpoint') or {};custody=checkpoint.get('training_custody') or {}
        required=('all_payloads_reread','source_closure_resolved','environment_imports','tokenizer_loaded','checkpoint_full_state_loaded','checkpoint_plan_custody','node_collector_imported','task_assets_read')
        if (not isinstance(checkpoint.get('sha256'),str) or len(checkpoint['sha256'])!=64 or custody.get('source_roots')!=[plan['inputs']['train'],plan['inputs']['held']]
            or custody.get('ce')!=plan['inputs']['ce'] or not all(checks.get(key) is True for key in required)):
            raise ValueError('export full restore qualification differs')
        evidence=proof.get('evidence') or {}
        if set(evidence)!=set(required):raise ValueError('export full restore evidence differs')
        for reference in evidence.values():lookup(reference)


def verify(path, expected_sha256):
    path=Path(path).resolve(strict=True)
    if file_sha256(path)!=expected_sha256: raise ValueError('export receipt SHA differs')
    receipt=json.loads(path.read_text()); plan=receipt.get('plan')
    validate_recovery_plan_shape(plan)
    if (receipt.get('schema')!='natlang.local-full-recovery-export/1'
        or receipt.get('plan_sha256')!=RecoveryExportQueue._plan_digest(plan)
        or receipt.get('session')!=plan['session'] or receipt.get('inputs')!=plan['inputs']
        or receipt.get('full_state') is not True or receipt.get('independent_readback') is not True):
        raise ValueError('export receipt identity differs')
    observed=receipt.get('closure'); controller=receipt.get('controller_state') or {}
    expected={(x['role'],x['path'],x['sha256']) for x in plan['closure']}
    expected.add(('controller-state',controller.get('path'),controller.get('sha256')))
    if (not isinstance(observed,list) or len(observed)!=len(expected)
        or {(x.get('role'),x.get('declared_path'),x.get('sha256')) for x in observed}!=expected):
        raise ValueError('export closure bindings differ')
    for item in observed:
        if file_sha256(Path(item['path']).resolve(strict=True))!=item['sha256']:
            raise ValueError('export closure bytes differ')
    checkpoint=receipt.get('checkpoint') or {}; checkpoint_path=Path(checkpoint.get('path','')).resolve(strict=True)
    if (checkpoint_path.stat().st_size!=checkpoint.get('bytes')
        or file_sha256(checkpoint_path)!=checkpoint.get('sha256')):
        raise ValueError('export checkpoint bytes differ')
    state=validate_full_recovery_checkpoint(checkpoint_path,receipt.get('training_custody'))
    if not _plan_matches_checkpoint(plan,state):raise ValueError('export checkpoint source or static SFT contract differs')
    exported={(item['declared_path'],item['sha256']):item['path'] for item in observed}
    def lookup(reference):
        if not isinstance(reference,dict) or set(reference)!={'path','sha256'}:raise ValueError('invalid exported proof reference')
        key=(reference['path'],reference['sha256'])
        if key not in exported:raise ValueError('proof dependency missing from exported closure')
        return json.loads(Path(exported[key]).read_text())
    verify_proof_chain(plan,lookup)
    return {'receipt_sha256':expected_sha256,'checkpoint_sha256':checkpoint['sha256'],
            'global_step':receipt['global_step'],'closure_entries':len(observed),
            'scope':'exported bytes and full-state custody verified; prior isolated restore qualification checked'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--receipt',required=True);parser.add_argument('--sha256',required=True)
    args=parser.parse_args();print(json.dumps(verify(args.receipt,args.sha256)),flush=True)

if __name__=='__main__':main()
