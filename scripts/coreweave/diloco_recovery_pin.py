#!/usr/bin/env python3
"""Create immutable DiLoCo worker and coordinator recovery pins without changing live state.

Every file under the coordinator's recovery root is pinned while a complete inventory is stable.
This is a topology component, not a consolidated global model or a claim of recovered local updates.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

from natlang_neuralese.common.hashing import canonical_json_sha256_hex, sha256_file_hex
from natlang_neuralese.train.native_resident import code_inventory, locked, current
from natlang_neuralese.train.checkpoint_policy import CheckpointPolicy
from natlang_neuralese.train.native_state_sft import TRAINING_SCHEMA
from natlang_neuralese.model.native_model import CHECKPOINT_SCHEMA
from natlang_neuralese.train.diloco.coordinator import read_manifest, _metadata
from scripts.coreweave.cache_bounded_archive import open_recovery_tar


def _hash(path): return sha256_file_hex(path, drop_cache=True)


def _delta_hash(tensor):
    value=tensor.detach().float().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


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
        pending={}
        for key, entry in (dl.get('pending') or {}).items():
            index=int(key);payload=(dl.get('pending_payloads') or {}).get(str(index))
            if index<0 or index>=len(sequences) or entry.get('sequence')!=sequences[index] or payload is None:
                raise ValueError('worker pending upload lacks exact checkpoint payload and sequence')
            pending[str(index)]={'sequence':entry['sequence'],'delta_sha256':_delta_hash(payload)}
        awaiting=[int(index) for index in dl.get('awaiting_global',[])]
        if any(index<0 or index>=len(sequences) for index in awaiting):
            raise ValueError('worker awaiting-global fragment is invalid')
        from scripts.coreweave.recovery_readiness import required_bindings
        dependencies=required_bindings({'training':training,'model_config':state['model_config']})
        return {'path':str(destination),'sha256':_hash(destination),'bytes':destination.stat().st_size,
                'global_step':training['global_step'],'worker':worker,'fingerprint':dl['fingerprint'],
                'versions':version,'sequences':sequences,'pending':pending,'awaiting_global':awaiting,
                'exposure_sha256':canonical_json_sha256_hex(training['exposure_counts']),
                'source_hash':training.get('source_hash'),
                'required_dependencies':dependencies,
                'resident_generation':(training.get('options',{}).get('resident_adoption') or {}).get('generation'),
                'source_receipt_sha256': [x.get('sha256') for x in training['source_receipts'][:2]],
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
        initial_binding = None
        if not state:
            try:
                initial_binding = {'plan_sha256': initial_state['plan_sha256'],
                    'recipe_sha256': active_plan['recipe_sha256'],
                    'runtime_code_sha256': canonical_json_sha256_hex(active_plan['code']),
                    'roles': [{'train': x.get('native_graphs', {}).get('sha256'),
                               'held': x.get('held_native_graphs', {}).get('sha256')}
                              for x in active_plan['stage_inputs'].values()]}
            except (KeyError, TypeError, AttributeError) as error:
                raise ValueError('resident initial plan lacks recovery binding metadata') from error
        temp = archive.with_name(archive.name + '.tmp')
        with open_recovery_tar(temp) as tar: tar.add(staging, arcname='diloco-resident')
        os.link(temp, archive); temp.unlink()
        return {'path': str(archive), 'sha256': _hash(archive), 'bytes': archive.stat().st_size,
                'generation': state['generation'] if state else initial_state['plan_sha256'],
                'binding_kind': 'adopted' if state else 'initial-only',
                'initial_binding': initial_binding,
                'at_global_step': state['at_global_step'] if state else None,
                'captured_at': time.time()}
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        archive.with_name(archive.name + '.tmp').unlink(missing_ok=True)

def _inventory(root, *, include_public=False):
    root=Path(root);result={}
    # public/ is a derived serving cache; Coordinator._sync_public rebuilds it
    # from authoritative fragment theta/version on every coordinator startup.
    names=['manifest.json','seed.safetensors','progress.json','fragments','pending']
    if include_public:names.append('public')
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


def coordinator_pin(root, archive, package, *, attempts=5, tree_only=False):
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
            from safetensors.torch import load_file
            versions=[];ledgers=[];queued=[]
            for fragment in manifest['fragments']:
                index=fragment['index'];meta=_metadata(staging/'fragments'/f'{index:05d}.safetensors')
                versions.append(int(meta['version']));ledgers.append(json.loads(meta.get('last_seq','{}')))
                items={}
                folder=staging/'pending'/f'{index:05d}'
                if folder.is_dir():
                    for path in sorted(folder.glob('*.safetensors')):
                        entry=_metadata(path);worker=str(int(entry['worker']))
                        if worker in items:raise ValueError('duplicate coordinator pending worker')
                        items[worker]={'sequence':int(entry['sequence']), 'base_version':int(entry['base_version']),
                                       'steps':int(entry['steps']),'tokens':int(entry['tokens']),
                                       'delta_sha256':_delta_hash(load_file(path)['delta'])}
                queued.append(items)
            if tree_only:
                for directory in sorted([staging, *[p for p in staging.rglob('*') if p.is_dir()]], key=lambda p:len(p.parts), reverse=True):
                    fd=os.open(directory,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
                os.rename(staging,archive)
                fd=os.open(archive.parent,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
                return {'path':str(archive),'binding_kind':'immutable-hardlink-tree','file_identity':before,
                        'fingerprint':manifest['fingerprint'],'versions':versions,'last_seq':ledgers,'queued':queued,
                        'manifest':manifest,'captured_at':time.time(),
                        'runtime_code_sha256':canonical_json_sha256_hex(code_inventory(package)),
                        'scope':'stable coordinator hardlinks; separate worker checkpoints retain local optimizer and pending updates'}
            temp=archive.with_name(archive.name+'.tmp')
            with open_recovery_tar(temp) as tar:tar.add(staging,arcname='diloco-coordinator')
            os.link(temp,archive);temp.unlink()
            return {'path':str(archive),'sha256':_hash(archive),'bytes':archive.stat().st_size,
                    'fingerprint':manifest['fingerprint'],'versions':versions,'last_seq':ledgers,'queued':queued,
                    'manifest':manifest,'captured_at':time.time(),
                    'runtime_code_sha256':canonical_json_sha256_hex(code_inventory(package)),
                    'scope':'stable coordinator files; separate worker checkpoints retain local optimizer and pending updates'}
        finally:
            shutil.rmtree(staging,ignore_errors=True)
            archive.with_name(archive.name+'.tmp').unlink(missing_ok=True)
    raise RuntimeError('coordinator changed during each recovery pin attempt')



def archive_coordinator_tree(tree, archive, identity):
    tree,archive=Path(tree),Path(archive)
    expected={k:tuple(v) for k,v in json.loads(Path(identity).read_text()).items()}
    # Already pinned trees may use the older inventory that included public/.
    include_public=any(k.startswith('public/') for k in expected)
    if _inventory(tree,include_public=include_public)!=expected:raise ValueError('coordinator hardlink tree changed before archive')
    seal=archive.with_name(archive.name+'.seal.json')
    identity_sha=sha256_file_hex(identity)
    if archive.exists() and seal.exists():
        receipt=json.loads(seal.read_text())
        if (receipt.get('identity_sha256')!=identity_sha or receipt.get('bytes')!=archive.stat().st_size
                or receipt.get('sha256')!=_hash(archive)):
            raise ValueError('coordinator archive retry differs from its durable seal')
        return {k:receipt[k] for k in ('path','sha256','bytes')}
    if archive.exists():archive.unlink()  # unsealed interrupted preparation was never published
    temp=archive.with_name(archive.name+'.tmp');temp.unlink(missing_ok=True)
    try:
        with open_recovery_tar(temp) as tar:tar.add(tree,arcname='diloco-coordinator')
        if _inventory(tree,include_public=include_public)!=expected:raise ValueError('coordinator hardlink tree changed during archive')
        os.link(temp,archive);temp.unlink()
        result={'path':str(archive),'sha256':_hash(archive),'bytes':archive.stat().st_size}
        from natlang_neuralese.common.jsonio import write_json_durable
        write_json_durable(seal,{**result,'identity_sha256':identity_sha})
        return result
    finally:temp.unlink(missing_ok=True)

def release_pin(primary, expected_sha, expected_bytes, *, recovery_root, fenced_source=None, fence_root=None):
    """Remove only a verified recovery pin, never the learner's live checkpoint."""
    primary=Path(primary);recovery_root=Path(recovery_root)
    if primary.is_symlink() or not primary.resolve().is_relative_to(recovery_root.resolve()) or '..' in primary.parts:
        raise ValueError('recovery pin is outside its exact cycle root')
    secondary=Path(fenced_source) if fenced_source else None
    if secondary is not None:
        if fence_root is None or secondary.is_symlink() or not secondary.resolve().is_relative_to(Path(fence_root).resolve()) or '..' in secondary.parts:
            raise ValueError('fence pin is outside its exact fence root')
    paired=False
    if primary.exists():
        if not primary.is_file() or primary.stat().st_size!=expected_bytes:
            raise ValueError('recovery pin byte count changed')
        paired=(secondary is not None and secondary.is_file() and
                (primary.stat().st_dev,primary.stat().st_ino)==(secondary.stat().st_dev,secondary.stat().st_ino))
        if _hash(primary)!=expected_sha:raise ValueError('recovery pin hash changed')
        primary.unlink()
    if secondary is not None and secondary.exists():
        if not secondary.is_file() or secondary.stat().st_size!=expected_bytes or (not paired and _hash(secondary)!=expected_sha):
            raise ValueError('fence pin differs from exported bytes')
        secondary.unlink()
    return {'released':True}

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    w=sub.add_parser('worker');w.add_argument('checkpoint');w.add_argument('destination');w.add_argument('worker',type=int)
    w.add_argument('fingerprint');w.add_argument('package')
    c=sub.add_parser('coordinator');c.add_argument('root');c.add_argument('archive');c.add_argument('package')
    ct=sub.add_parser('coordinator-tree');ct.add_argument('root');ct.add_argument('archive');ct.add_argument('package')
    ca=sub.add_parser('coordinator-tree-archive');ca.add_argument('tree');ca.add_argument('archive');ca.add_argument('identity')
    r=sub.add_parser('resident');r.add_argument('directory');r.add_argument('archive')
    d=sub.add_parser('release');d.add_argument('primary');d.add_argument('sha256');d.add_argument('bytes',type=int)
    d.add_argument('recovery_root');d.add_argument('--fenced-source');d.add_argument('--fence-root')
    a=p.parse_args()
    value=(archive_coordinator_tree(a.tree,a.archive,a.identity) if a.action=='coordinator-tree-archive' else
           coordinator_pin(a.root,a.archive,a.package,tree_only=True) if a.action=='coordinator-tree' else
           release_pin(a.primary,a.sha256,a.bytes,recovery_root=a.recovery_root,fenced_source=a.fenced_source,fence_root=a.fence_root) if a.action=='release' else
           worker_pin(a.checkpoint,a.destination,a.worker,a.fingerprint,a.package) if a.action=='worker' else
           coordinator_pin(a.root,a.archive,a.package) if a.action=='coordinator' else resident_pin(a.directory,a.archive))
    print(json.dumps(value,sort_keys=True))


if __name__=='__main__':main()
