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
from natlang_neuralese.common.hashing import sha256_file_hex,canonical_json_sha256_hex
from natlang_neuralese.common.jsonio import write_json_durable
from compose_native_actor_runtime import bound
from verify_native_actor_runtime import verify_files

REQUIRED = {'entry_root','bootstrap','root_review','provider_readiness','publication_configuration','frozen_refs'}


def prepare(composition,role,inputs,out):
    composition=Path(composition).resolve(strict=True);record=verify_files(composition)
    material=json.loads(bound(record['role_materials'][role]).read_bytes())
    inputs=json.loads(Path(inputs).read_bytes())
    if set(inputs)-REQUIRED-{'task_runtime','historical_controls'} or not REQUIRED<=set(inputs):
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
    if material.get('publisher_loader'):
        out=Path(out).resolve()
        if out != Path(material['binder']['path']).parent/'binder-config.json':
            raise ValueError('binder config must live at its physical controller root')
        config.update({name:material[name] for name in ('publisher_loader','publisher_manifest')})
        historical=inputs.get('historical_controls') or {}
        if not isinstance(historical,dict) or not set(historical)<={'native_interrupted_update','native_interrupted_resume'}:
            raise ValueError('only exact interrupted actors can preserve historical source identity')
        config['controls']=dict(material['controls'])
        for name,reference in historical.items():
            path=bound(reference)
            reviewed=material['controls'].get(name) or {}
            if (Path(reference['path']).is_symlink() or str(path)!=reference['path']
                    or any(reference.get(key)!=reviewed.get(key) for key in ('sha256','bytes'))):
                raise ValueError('historical actor must retain its exact actual reviewed source and byte count')
            config['controls'][name]=dict(reference)
        if historical:
            manifest=json.loads(bound(material['publisher_manifest']).read_bytes())
            manifest['controls']=config['controls']
            path=out.parent/('publishing-manifest-source-actors-'+canonical_json_sha256_hex(manifest)+'.json')
            if path.exists() and json.loads(path.read_bytes())!=manifest:
                raise ValueError('immutable historical actor publishing manifest changed')
            write_json_durable(path,manifest)
            config['publisher_manifest']={'path':str(path),'sha256':sha256_file_hex(path),'bytes':path.stat().st_size}
    elif inputs.get('historical_controls'):
        raise ValueError('historical actors require the complete reviewed publisher/controller closure')
    # Existing controller repair proofs encode their own path+SHA pair. Keep
    # that producer contract while retaining/verifying bytes in frozen_refs.
    config['controller']={key:material['controller'][key] for key in ('path','sha256')}
    config['frozen_refs']=[*inputs['frozen_refs'],record['role_materials'][role],material['binder'],
        material['controller'],material['control_loader'],material['control_manifest'],
        material['role_mechanism_authorities'],*material['controls'].values()]
    if material.get('publisher_loader'):
        config['frozen_refs'] += [config['publisher_loader'],config['publisher_manifest'],
                                  *config['controls'].values()]
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
