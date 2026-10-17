"""Rebind only the capacity proof of the same queued, uncredited append."""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

from . import native_resident as resident
from ..common.hashing import sha256_hex
from ..common.jsonio import write_bytes_durable


@contextmanager
def _cache_root(root):
    """Scope the existing index verifier to this isolated CPU helper call."""
    name='NATLANG_ROW_INDEX_CACHE';previous=os.environ.get(name)
    os.environ[name]=str(root)
    try:yield
    finally:
        if previous is None:os.environ.pop(name,None)
        else:os.environ[name]=previous


def _relocate_indexes(indexes, old_root):
    from ..common import row_index
    actual=row_index.cache_dir().resolve();old_root=Path(old_root).resolve()
    with _cache_root(old_root):resident._check_indexes(indexes)
    actual.mkdir(parents=True,exist_ok=True)
    relocated=[];joins=[]
    for witness in indexes:
        source=resident.bound(witness['cache'])
        target=row_index.entry_path('native-graphs',resident.digest(witness['key'])).resolve()
        if not target.exists():
            fd,name=tempfile.mkstemp(prefix=target.name+'.',suffix='.tmp',dir=actual)
            temporary=Path(name)
            try:
                with os.fdopen(fd,'wb') as out,source.open('rb') as stream:
                    shutil.copyfileobj(stream,out,length=1024*1024);out.flush();os.fsync(out.fileno())
                if resident.sha256_file_hex(temporary)!=witness['cache']['sha256']:
                    raise ValueError('resident index changed while copying verified bytes')
                # Only this immutable new copy is linked for no-clobber atomic
                # publication. The original cache is never hardlinked.
                try:os.link(temporary,target)
                except FileExistsError:pass
            finally:temporary.unlink(missing_ok=True)
        replacement={**witness,'cache':resident.binding(target)}
        resident._check_indexes([replacement])
        with target.open('rb') as stream:os.fsync(stream.fileno())
        relocated.append(replacement)
        joins.append({'old':witness['cache'],'new':replacement['cache'],
            'key':witness['key'],'metadata_sha256':witness['metadata_sha256']})
    fd=os.open(actual,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    resident._check_indexes(relocated)
    return relocated,{'old_root':str(old_root),'learner_root':str(actual),'witnesses':joins}


def refresh(directory, request, *, runtime_package=None, old_cache_root=None, publish=True):
    if type(publish) is not bool:
        raise ValueError('resident geometry refresh publish must be an explicit boolean')
    directory=Path(directory).resolve()
    package=Path(runtime_package or Path(resident.__file__).resolve().parents[1])
    with resident.locked(directory):
        state=resident.current(directory)
        plan=(state['payload']['plan'] if state else
            json.loads((directory.parent/'recipe-plan.json').read_text()))
        predecessor=state['generation'] if state else resident.digest(plan)
        initial_ref=resident.binding(directory/'initial.json')
        initial_bytes=resident.bound(initial_ref).read_bytes()
        if sha256_hex(initial_bytes)!=initial_ref['sha256']:
            raise ValueError('resident initial changed while observing geometry authority')
        initial=json.loads(initial_bytes)
        pointer=json.loads((directory/'pending.json').read_text())
    value=json.loads(resident.bound(request).read_text())
    payload=value.get('payload')
    if value.get('schema')!=resident.SCHEMA or resident.digest(payload)!=value.get('generation'):
        raise ValueError('resident geometry refresh requires an intact prepared request')
    stage=payload['stage_id']
    if (payload['predecessor']!=predecessor or initial['stage_id']!=stage
            or not state and initial['plan_sha256']!=resident.digest(plan)):
        raise ValueError('resident geometry refresh current stage/generation differs')
    if resident.code_inventory(package)!=plan['code']:
        raise ValueError('resident geometry refresh runtime differs')
    resident._contract(plan,payload['plan'],stage)
    if payload['geometry_identity']!=initial['geometry_identity']:
        raise ValueError('resident geometry refresh requires the same actual geometry identity')
    for role,old_role in (('train','native_graphs'),('held','held_native_graphs')):
        expected=payload['plan']['stage_inputs'][stage][old_role]
        if resident.binding(resident.bound(payload[role]))!={k:expected[k] for k in ('path','sha256')}:
            raise ValueError('resident geometry refresh source differs from prepared plan')
        prior=plan['stage_inputs'][stage][old_role]
        if resident.binding(resident.bound(payload['prior_'+role]))!={k:prior[k] for k in ('path','sha256')}:
            raise ValueError('resident geometry refresh predecessor source differs')
    from .recipe import load_recipe,native_recipe_binding,resolve_stage_inputs
    authority=state['payload']['recipe'] if state else initial['recipe']
    if (native_recipe_binding(authority['path'],plan['recipe_sha256'])!=authority
            or load_recipe(Path(authority['path']))!=plan['recipe']):
        raise ValueError('resident geometry refresh current recipe authority changed')
    recipe=payload['recipe']
    resident.bound({'path':recipe['path'],'sha256':recipe['file_sha256']})
    if (native_recipe_binding(recipe['path'],plan['recipe_sha256'])!=recipe
            or load_recipe(Path(recipe['path']))!=payload['plan']['recipe']):
        raise ValueError('resident geometry refresh prepared recipe changed')
    resolved={s['id']:({} if s.get('admitted') is False else
        resolve_stage_inputs(payload['plan']['recipe'],s,{'records':None,'pieces':None,'text_data':None}))
        for s in payload['plan']['recipe']['stages']}
    if resolved!=payload['plan']['stage_inputs']:
        raise ValueError('resident geometry refresh resolved inputs changed')
    for reference in (payload['geometry_qualification'],initial['geometry_qualification']):
        limits=resident.geometry_capacity(reference,identity=initial['geometry_identity'],code=plan['code'])
        if (set(payload['geometry'])!=set(limits)
                or any(type(v) is not int or v<0 for v in payload['geometry'].values())
                or any(payload['geometry'][key]>limits[key] for key in limits)):
            raise ValueError('resident geometry refresh exceeds qualified whole-graph geometry')
    if old_cache_root is None:
        resident._check_indexes(payload['indexes']);indexes=payload['indexes'];relocation=None
    else:indexes,relocation=_relocate_indexes(payload['indexes'],old_cache_root)
    replacement=copy.deepcopy(value)
    replacement['payload']['geometry_qualification']=initial['geometry_qualification']
    replacement['payload']['indexes']=indexes
    replacement['generation']=resident.digest(replacement['payload'])
    requests=directory/'requests'
    new_path=requests/(replacement['generation']+'.json')
    proof_path=requests/(replacement['generation']+'-'+request['sha256']+'-geometry-refresh.json')
    with resident.locked(directory):
        actual=resident.current(directory)
        generation=actual['generation'] if actual else resident.digest(json.loads(
            (directory.parent/'recipe-plan.json').read_text()))
        if generation!=predecessor or resident.binding(directory/'initial.json')!=initial_ref:
            raise ValueError('resident geometry refresh authority changed during verification')
        # A retry may finish only this exact replacement of this exact request.
        actual_pointer=json.loads((directory/'pending.json').read_text())
        if actual_pointer!=pointer:
            raise ValueError('resident pending obligation changed during geometry refresh')
        if pointer!=request:
            if not new_path.exists() or pointer!=resident.binding(new_path):
                raise ValueError('resident geometry refresh cannot supersede another pending obligation')
        if new_path.exists():
            if json.loads(new_path.read_text())!=replacement:
                raise ValueError('resident geometry refresh immutable request differs')
        else:resident.write_json(new_path,replacement)
        new_ref=resident.binding(new_path)
        proof={'schema':'natlang.native-resident-geometry-refresh/1','request':request,
            'replacement':new_ref,'predecessor':predecessor,
            'old_geometry_qualification':payload['geometry_qualification'],
            'new_geometry_qualification':initial['geometry_qualification'],
            'payload_changes':['geometry_qualification']+(['indexes'] if indexes!=payload['indexes'] else []),
            'scope':'same prepared append; no earned exposure or numerical requalification'}
        if relocation:proof['index_cache_relocation']=relocation
        if proof_path.exists():
            previous=json.loads(proof_path.read_text())
            if {k:v for k,v in previous.items() if k not in ('initial','original_initial')}!=proof:
                raise ValueError('resident geometry refresh immutable receipt differs')
            historical=resident.bound(previous['initial']).read_bytes()
            if (sha256_hex(historical)!=previous['original_initial']['sha256']
                    or previous['original_initial']['path']!=initial_ref['path']):
                raise ValueError('resident geometry refresh historical authority snapshot differs')
            historical=json.loads(historical)
            if any(historical[key]!=initial[key] for key in
                    ('recipe','stage_id','plan_sha256','geometry_identity','geometry_qualification')):
                raise ValueError('resident geometry refresh historical authority contract differs')
        else:
            snapshot=requests/('geometry-initial-'+initial_ref['sha256']+'.json')
            if snapshot.exists():
                if snapshot.read_bytes()!=initial_bytes:
                    raise ValueError('resident geometry refresh immutable authority snapshot differs')
            else:write_bytes_durable(snapshot,initial_bytes)
            resident.write_json(proof_path,{**proof,'initial':resident.binding(snapshot),
                'original_initial':initial_ref})
        if publish:resident.write_json(directory/'pending.json',new_ref)
        return {'request':new_ref,'receipt':resident.binding(proof_path),'published':publish}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--request-sha256',required=True)
    parser.add_argument('--runtime-package',type=Path)
    parser.add_argument('--old-cache-root',type=Path)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args(argv)
    print(json.dumps(refresh(args.directory,{'path':str(args.request.resolve()),'sha256':args.request_sha256},
        runtime_package=args.runtime_package,old_cache_root=args.old_cache_root,publish=not args.prepare_only)))
    return 0


if __name__=='__main__':raise SystemExit(main())
