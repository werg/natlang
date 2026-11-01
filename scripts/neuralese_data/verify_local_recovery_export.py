#!/usr/bin/env python3
"""Independently verify an exported full-state receipt without its original paths.

This validates immutable bytes and checkpoint custody; the isolated restore
qualification bound by the plan is a separate, stronger runtime exercise.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from natlang_neuralese.train.checkpoint_offload import (RecoveryExportQueue, file_sha256,
    recovery_closure_identity, validate_full_recovery_checkpoint, validate_recovery_plan_shape)


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
    validate_full_recovery_checkpoint(checkpoint_path,receipt.get('training_custody'))
    proof_ref=plan['restore_qualification']
    proof_entries=[item for item in observed if item['role']=='controls'
                   and item['declared_path']==proof_ref['path'] and item['sha256']==proof_ref['sha256']]
    if len(proof_entries)!=1:raise ValueError('restore qualification not in exported closure')
    proof=json.loads(Path(proof_entries[0]['path']).read_text())
    qualified_checkpoint=proof.get('checkpoint') or {}; qualified_custody=qualified_checkpoint.get('training_custody') or {}
    if (proof.get('status')!='passed' or proof.get('inputs')!=plan['inputs']
        or proof.get('restore')!=plan.get('restore') or proof.get('additional_inputs',[])!=plan.get('additional_inputs',[])
        or proof.get('controller_config')!=plan['controller']['config']
        or not isinstance(qualified_checkpoint.get('sha256'),str) or len(qualified_checkpoint['sha256'])!=64
        or qualified_custody.get('source_roots')!=[plan['inputs']['train'],plan['inputs']['held']]
        or qualified_custody.get('ce')!=plan['inputs']['ce']
        or proof.get('closure_identity')!=recovery_closure_identity(plan)
        or not all(proof.get('checks',{}).get(key) is True for key in
                   ('all_payloads_reread','source_closure_resolved','environment_imports','tokenizer_loaded',
                    'checkpoint_full_state_loaded','checkpoint_plan_custody','node_collector_imported','task_assets_read'))):
        raise ValueError('export restore qualification differs')
    return {'receipt_sha256':expected_sha256,'checkpoint_sha256':checkpoint['sha256'],
            'global_step':receipt['global_step'],'closure_entries':len(observed),
            'scope':'exported bytes and full-state custody verified; prior isolated restore qualification checked'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--receipt',required=True);parser.add_argument('--sha256',required=True)
    args=parser.parse_args();print(json.dumps(verify(args.receipt,args.sha256)),flush=True)

if __name__=='__main__':main()
