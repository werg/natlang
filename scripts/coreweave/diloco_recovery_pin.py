#!/usr/bin/env python3
"""Create immutable DiLoCo worker and coordinator recovery pins without changing live state.

Every file under the coordinator's recovery root is pinned while a complete inventory is stable.
This is a topology component, not a consolidated global model or a claim of recovered local updates.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
import time

from natlang_neuralese.common.hashing import canonical_json_sha256_hex, sha256_file_hex
from natlang_neuralese.train.native_resident import code_inventory, locked, current
from natlang_neuralese.train.checkpoint_policy import CheckpointPolicy
from natlang_neuralese.train.native_state_sft import TRAINING_SCHEMA
from natlang_neuralese.model.native_model import CHECKPOINT_SCHEMA
from natlang_neuralese.train.diloco.coordinator import read_manifest, _metadata


def _hash(path): return sha256_file_hex(path, drop_cache=True)


def worker_pin(checkpoint, destination, worker, expected_fingerprint, package):
    checkpoint, destination = Path(checkpoint), Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists(): raise ValueError('worker pin already exists')
    os.link(checkpoint,destination)  # CheckpointPolicy publishes via atomic replacement; link the exact inode.
    try:
        state=CheckpointPolicy.load(destination)
        training=state.get('training') or {}; dl=training.get('diloco') or {}
        if (state.get('schema')!=CHECKPOINT_SCHEMA or training.get('schema')!=TRAINING_SCHEMA
                or dl.get('worker')!=worker or dl.get('fingerprint')!=expected_fingerprint
                or not training.get('optimizer') or not training.get('rng') or not training.get('exposure_counts')
                or not training.get('source_receipts') or type(training.get('global_step')) is not int):
            raise ValueError('worker checkpoint lacks exact full-state DiLoCo custody')
        version=dl.get('versions'); sequences=dl.get('sequences')
        if (not isinstance(version,list) or not isinstance(sequences,list) or len(version)!=len(sequences)
                or not all(type(x) is int and x>=0 for x in [*version,*sequences])):
            raise ValueError('worker checkpoint has malformed fragment progress')
        from scripts.coreweave.recovery_readiness import required_bindings
        dependencies=required_bindings({'training':training,'model_config':state['model_config']})
        return {'path':str(destination),'sha256':_hash(destination),'bytes':destination.stat().st_size,
                'global_step':training['global_step'],'worker':worker,'fingerprint':dl['fingerprint'],
                'versions':version,'sequences':sequences,'pending_fragments':sorted(dl.get('pending',{})),
                'exposure_sha256':canonical_json_sha256_hex(training['exposure_counts']),
                'source_hash':training.get('source_hash'),
                'required_dependencies':dependencies,
                'resident_generation':(training.get('options',{}).get('resident_adoption') or {}).get('generation'),
                'recipe_binding':training.get('options',{}).get('recipe_binding'),
                'ce_supervision_policy_binding':training.get('options',{}).get('ce_supervision_policy_binding'),
                'runtime_code_sha256':canonical_json_sha256_hex(code_inventory(package))}
    except BaseException:
        destination.unlink(missing_ok=True)
        raise



def resident_pin(directory, archive):
    """Pin committed resident control under its writer lock, separate from worker weights."""
    directory, archive = Path(directory), Path(archive)
    if archive.exists(): raise ValueError('resident pin already exists')
    archive.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.resident-pin-', dir=archive.parent))
    try:
        with locked(directory):
            state = current(directory)
            initial = directory / 'initial.json'
            plan = directory.parent / 'recipe-plan.json'
            if not initial.is_file() or not plan.is_file():
                raise ValueError('resident initial state or active plan missing')
            for source, rel in [(plan, Path('recipe-plan.json')), *[(p, Path('resident-data') / p.relative_to(directory)) for p in sorted(directory.rglob('*')) if p.is_file() and p.name != 'lock']]:
                if source.is_symlink(): raise ValueError('resident control contains a symlink')
                target = staging / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                os.link(source, target)
            initial_state = json.loads((staging / 'resident-data/initial.json').read_text())
            active_plan = json.loads((staging / 'recipe-plan.json').read_text())
            if state and state['payload']['plan'] != active_plan:
                raise ValueError('resident committed generation and active plan differ')
            if not state and initial_state['plan_sha256'] != canonical_json_sha256_hex(active_plan):
                raise ValueError('resident initial generation and active plan differ')
        temp = archive.with_name(archive.name + '.tmp')
        with tarfile.open(temp, 'x', dereference=True) as tar: tar.add(staging, arcname='diloco-resident')
        os.link(temp, archive); temp.unlink()
        return {'path': str(archive), 'sha256': _hash(archive), 'bytes': archive.stat().st_size,
                'generation': state['generation'] if state else initial_state['plan_sha256'],
                'at_global_step': state['at_global_step'] if state else None,
                'captured_at': time.time()}
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        archive.with_name(archive.name + '.tmp').unlink(missing_ok=True)

def _inventory(root):
    root=Path(root);result={}
    names=['manifest.json','seed.safetensors','progress.json','fragments','public','pending']
    for name in names:
        entry=root/name
        if not entry.exists():
            if name in ('manifest.json','seed.safetensors','progress.json','fragments'):raise ValueError('missing coordinator '+name)
            continue
        files=sorted(entry.rglob('*')) if entry.is_dir() else [entry]
        for path in files:
            if path.is_symlink():raise ValueError('coordinator state contains a symlink')
            if not path.is_file() or path.name.startswith('.'):continue
            stat=path.stat();result[str(path.relative_to(root))]=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns)
    return result


def coordinator_pin(root, archive, package, *, attempts=5):
    root,archive=Path(root),Path(archive)
    manifest=read_manifest(root)
    if archive.exists():raise ValueError('coordinator pin already exists')
    archive.parent.mkdir(parents=True,exist_ok=True)
    for _ in range(attempts):
        before=_inventory(root); staging=Path(tempfile.mkdtemp(prefix='.coordinator-pin-',dir=archive.parent))
        try:
            for name in before:
                target=staging/name;target.parent.mkdir(parents=True,exist_ok=True);os.link(root/name,target)
            after=_inventory(root)
            if before!=after:
                continue
            versions=[];ledgers=[]
            for fragment in manifest['fragments']:
                index=fragment['index'];meta=_metadata(staging/'fragments'/f'{index:05d}.safetensors')
                versions.append(int(meta['version']));ledgers.append(json.loads(meta.get('last_seq','{}')))
            temp=archive.with_name(archive.name+'.tmp')
            with tarfile.open(temp,'x',dereference=True) as tar:tar.add(staging,arcname='diloco-coordinator')
            os.link(temp,archive);temp.unlink()
            return {'path':str(archive),'sha256':_hash(archive),'bytes':archive.stat().st_size,
                    'fingerprint':manifest['fingerprint'],'versions':versions,'last_seq':ledgers,
                    'manifest':manifest,'captured_at':time.time(),
                    'runtime_code_sha256':canonical_json_sha256_hex(code_inventory(package)),
                    'scope':'stable coordinator files; separate worker checkpoints retain local optimizer and pending updates'}
        finally:
            shutil.rmtree(staging,ignore_errors=True)
            archive.with_name(archive.name+'.tmp').unlink(missing_ok=True)
    raise RuntimeError('coordinator changed during each recovery pin attempt')


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    w=sub.add_parser('worker');w.add_argument('checkpoint');w.add_argument('destination');w.add_argument('worker',type=int)
    w.add_argument('fingerprint');w.add_argument('package')
    c=sub.add_parser('coordinator');c.add_argument('root');c.add_argument('archive');c.add_argument('package')
    r=sub.add_parser('resident');r.add_argument('directory');r.add_argument('archive')
    a=p.parse_args()
    value=(worker_pin(a.checkpoint,a.destination,a.worker,a.fingerprint,a.package) if a.action=='worker' else
           coordinator_pin(a.root,a.archive,a.package) if a.action=='coordinator' else resident_pin(a.directory,a.archive))
    print(json.dumps(value,sort_keys=True))


if __name__=='__main__':main()
