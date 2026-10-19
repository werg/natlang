#!/usr/bin/env python3
"""Verify an exact locally composed actor/controller tree without model allocation."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'training/neuralese'))
from natlang_neuralese.common.hashing import canonical_json_sha256_hex,sha256_file_hex
from natlang_neuralese.common.jsonio import write_json_durable
from natlang_neuralese.train.native_resident import code_inventory


def verify_files(root):
    root=Path(root).resolve(strict=True)
    record=json.loads((root/'construction.json').read_bytes())
    if record.get('schema')!='natlang.native-actor-coordinator-composition/1':
        raise ValueError('unsupported native actor composition')
    for name,reference in record['files'].items():
        path=root/name
        if (Path(name).is_absolute() or '..' in Path(name).parts or path.is_symlink()
                or not path.resolve(strict=True).is_relative_to(root)
                or sha256_file_hex(path)!=reference['sha256'] or path.stat().st_size!=reference['bytes']):
            raise ValueError('composed source file changed: '+name)
    inventory=code_inventory(root/'training/neuralese/natlang_neuralese')
    if inventory!=record['inventory'] or canonical_json_sha256_hex(inventory)!=record['code']:
        raise ValueError('composed native package changed')
    return record


CONTROL_PROBE = r'''
import ast,inspect,json,pathlib,sys
root=pathlib.Path(sys.argv[1]).resolve();role=sys.argv[2]
record=json.loads((root/'construction.json').read_bytes())
from natlang_neuralese.train.checkpoint_policy import CheckpointPolicy
CheckpointPolicy.load=staticmethod(lambda *a,**k:(_ for _ in()).throw(AssertionError('control preflight forbids checkpoint decode')))
loader=record['publisher_loader'];path=pathlib.Path(loader['path'])
scope={'__file__':str(path),'__name__':'qualified_role_publisher'}
exec(compile(path.read_bytes(),str(path),'exec'),scope)
evidence=scope['load'](record['role_controls'][role]['publisher_manifest'])
from natlang_neuralese.rl import native_role_release, native_raw_rounds, native_rounds, production, native_supervised_boundary
from natlang_neuralese.common.runtime_inventory import python_runtime_binding
from natlang_neuralese.train.native_resident import code_inventory
from scripts.coreweave import teacher_engine_guard
assert pathlib.Path(teacher_engine_guard.__file__).resolve()==root/'scripts/coreweave/teacher_engine_guard.py'
assert 'session_only' in inspect.signature(teacher_engine_guard.cleanup_session).parameters
assert (root/'scripts/run_bonsai_queue.py').is_file()
assert native_role_release.mechanism()==record['role_mechanism_authorities'][role]
assert pathlib.Path(production.__file__).resolve()==root/'training/neuralese/natlang_neuralese/rl/production.py'
# Inspect the actual function code loaded by the external publisher, rather than
# the shadowed installed file's source or a stale declaration/config note.
source=pathlib.Path(native_raw_rounds.configure_model.__code__.co_filename)
assert str(source)==record['role_controls'][role]['controls']['native_raw_rounds']['path']
tree=ast.parse(source.read_bytes())
function=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='configure_model')
assert any(isinstance(node,ast.keyword) and node.arg=='actor_runtime' for node in ast.walk(function))
assert 'actor_runtime' in production.configure_native.__code__.co_consts
assert 'actor_runtime' in native_rounds.bootstrap_configs.__code__.co_consts
assert hasattr(native_supervised_boundary,'_task_runtime')
assert code_inventory(root/'training/neuralese/natlang_neuralese')==record['inventory']
if record.get('role_materials'):
    material=json.loads(pathlib.Path(record['role_materials'][role]['path']).read_bytes())
    binder=pathlib.Path(material['binder']['path'])
    tree=ast.parse(binder.read_bytes())
    guard=next(node for node in ast.walk(tree) if isinstance(node,ast.Assert)
        and 'root_review' in ast.unparse(node.test))
    from natlang_neuralese.common.hashing import sha256_file_hex
    def binding(path):return {'path':str(path),'sha256':sha256_file_hex(path)}
    for with_bytes in (False,True):
        review={'passed':True,**{name:dict(material[name]) if with_bytes else
            {key:material[name][key] for key in ('path','sha256')} for name in ('controller','binder')}}
        config={'controller':{key:material['controller'][key] for key in ('path','sha256')}}
        scope={'root_review':review,'config':config,'binding':binding,'__file__':str(binder)}
        expression=compile(ast.Expression(guard.test),str(binder),'eval')
        assert eval(expression,scope)
        review['binder']['sha256']='0'*64
        assert not eval(expression,scope)
print(json.dumps({'role':role,'load':evidence,'active_raw_rounds_source':str(source),
 'python_runtime':python_runtime_binding(),'source_keyword_verified':True,
 'checkpoint_decodes':0,'model_allocations':0,'readiness_granted':False}))
'''


def verify(root):
    root=Path(root).resolve(strict=True);record=verify_files(root)
    proofs={}
    for role in ('lfm','mellum'):
        environment={**os.environ,'PYTHONPATH':str(root/'training/neuralese')+':'+str(root),
            'NATLANG_REPO':str(root),'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'}
        result=subprocess.run([sys.executable,'-c',CONTROL_PROBE,str(root),role],
            cwd=root,env=environment,text=True,capture_output=True)
        if result.returncode:
            raise ValueError('actual composed publisher failed for '+role+': '+result.stderr[-4000:])
        proofs[role]=json.loads(result.stdout)
    verify_files(root)
    return {'schema':'natlang.native-actor-coordinator-preflight/1','code':record['code'],
        'roles':proofs,'source_files':len(record['files']),'model_allocations':0,
        'cloud_mutations':0,'qualification_credit':False,'readiness_granted':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--composition',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();proof=verify(args.composition)
    write_json_durable(args.out,proof)
    print(json.dumps({'passed':True,'code':proof['code'],'roles':list(proof['roles']),
                      'model_allocations':0,'cloud_mutations':0,'receipt':str(args.out)}))


if __name__=='__main__':main()
