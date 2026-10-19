#!/usr/bin/env python3
"""Bind a future role controller to real reviewed inputs after a later-runtime handoff.

Pure metadata preparation. The existing binder independently checks the actual
live owner, source plan, continuation, geometry, task runtime and six-role maps.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'training/neuralese'))
from natlang_neuralese.common.hashing import sha256_file_hex
from natlang_neuralese.common.jsonio import write_json_durable
from compose_native_actor_runtime import bound
from verify_native_actor_runtime import verify_files

REQUIRED = {'entry_root','bootstrap','root_review','provider_readiness','publication_configuration','frozen_refs'}


def prepare(composition,role,inputs,out):
    composition=Path(composition).resolve(strict=True);record=verify_files(composition)
    material=json.loads(bound(record['role_materials'][role]).read_bytes())
    inputs=json.loads(Path(inputs).read_bytes())
    if set(inputs)-REQUIRED-{'task_runtime'} or not REQUIRED<=set(inputs):
        raise ValueError('future role binder requires its real owner, bootstrap, review and provider inputs')
    if not Path(inputs['entry_root']).is_absolute():
        raise ValueError('future owner requires its explicit absolute entry root')
    for name in ('bootstrap','root_review','provider_readiness'):
        bound(inputs[name])
    for reference in inputs['frozen_refs']:
        bound(reference)
    bootstrap=json.loads(bound(inputs['bootstrap']).read_bytes())
    if (bootstrap.get('schema')!='natlang.native-raw-rl-bootstrap/1'
            or bootstrap.get('owned_models')!=[role]
            or role not in (bootstrap.get('role_exchange') or {}).get('roles',{})):
        raise ValueError('future bootstrap belongs to another source-host role')
    review=json.loads(bound(inputs['root_review']).read_bytes())
    if (review.get('passed') is not True or any(
            (review.get(name) or {}).get(key)!=material[name][key]
            for name in ('controller','binder') for key in ('path','sha256'))):
        raise ValueError('future root review does not cover the actual composed controller and binder')
    for name in ('controller','binder'):
        bound(review[name])
    config={**inputs,**{name:material[name] for name in ('model','runtime','runtime_code_sha256',
        'actor_runtime','controller','control_loader','control_manifest','controls','role_mechanism_authorities')}}
    # Existing controller repair proofs encode their own path+SHA pair. Keep
    # that producer contract while retaining/verifying bytes in frozen_refs.
    config['controller']={key:material['controller'][key] for key in ('path','sha256')}
    config['frozen_refs']=[*inputs['frozen_refs'],record['role_materials'][role],material['binder'],
        material['controller'],material['control_loader'],material['control_manifest'],
        material['role_mechanism_authorities'],*material['controls'].values()]
    out=Path(out)
    if out.exists() and json.loads(out.read_bytes())!=config:
        raise ValueError('future binder config is already bound to different inputs')
    write_json_durable(out,config)
    return {'path':str(out.resolve()),'sha256':sha256_file_hex(out),'bytes':out.stat().st_size}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--composition',type=Path,required=True)
    parser.add_argument('--role',choices=['lfm','mellum'],required=True)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps({'binder_config':prepare(args.composition,args.role,args.inputs,args.out),
                      'cloud_mutations':0,'owner_started':False,'readiness_granted':False}))


if __name__=='__main__':main()
