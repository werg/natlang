"""Prepare a fresh on-policy controller from one exact supervisor-drained native parent.

This only verifies inputs and writes a configuration.  The role owner performs
any subsequent actor/learner launch and retains independent checkpoint export.
"""
from __future__ import annotations
import argparse,json,re
from pathlib import Path
from types import SimpleNamespace
from ..common.hashing import sha256_file_hex_cached
from ..common.jsonio import read_jsonl
from ..common.custody import ProtectedWorlds


def checked_parent(receipt, *, sandbox_id, boot_id, min_step, source_sha256):
    import torch
    pin=json.loads(Path(receipt).read_text())
    if (pin.get('schema')!='natlang.lfm3-final-standalone-drain/1' or pin.get('phase')!='pinned'
            or pin.get('ready') is not True or pin.get('immutable_pin_verified') is not True
            or pin.get('full_state_verified') is not True or pin.get('optimizer_verified') is not True
            or pin.get('rng_verified') is not True or pin.get('sandbox_id')!=sandbox_id
            or pin.get('boot_id')!=boot_id or pin.get('train_source_sha256')!=source_sha256
            or type(pin.get('step')) is not int or pin['step']<min_step or pin.get('global_step')!=pin['step']
            or not re.fullmatch('[0-9a-f]{64}',str(pin.get('sha256','')))):
        raise ValueError('on-policy parent is not the expected exact drained full-state pin')
    path=Path(pin['path']).resolve(strict=True);before=path.stat()
    if before.st_size!=pin.get('bytes') or sha256_file_hex_cached(path)!=pin['sha256']:
        raise ValueError('drained checkpoint bytes differ from the owner receipt')
    state=torch.load(path,map_location='cpu',weights_only=False,mmap=True);training=state['training']
    if (training.get('global_step')!=pin['step'] or not training.get('optimizer') or not training.get('rng')
            or not training.get('sft_progress') or training.get('exposure_counts') is None
            or (training.get('source_receipts') or [{}])[0].get('sha256')!=source_sha256):
        raise ValueError('drained parent lacks exact source, optimizer, RNG, exposure or SFT frame')
    after=path.stat()
    if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns):
        raise ValueError('drained checkpoint changed during verification')
    return path,pin


def checked_fresh_parent(checkpoint, *, expected_sha256, source_sha256):
    """Bind a real zero-update initializer; never manufacture a drain receipt."""
    import torch
    path=Path(checkpoint).resolve(strict=True)
    before=path.stat()
    if not re.fullmatch('[0-9a-f]{64}',expected_sha256) or sha256_file_hex_cached(path)!=expected_sha256:
        raise ValueError('fresh initialization checkpoint digest mismatch')
    state=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    training=state['training']; init=(training.get('options') or {}).get('fresh_initialization') or {}
    optimizer=training.get('optimizer') or {}
    if (init.get('schema')!='natlang.fresh-native-initialization/1' or init.get('parent_checkpoint') is not None
            or init.get('optimizer_updates')!=0 or init.get('earned_exposure')!=0
            or training.get('step')!=0 or training.get('global_step')!=0
            or any((training.get('exposure_counts') or {}).values()) or not training.get('rng')
            or not training.get('sft_progress') or optimizer.get('state')
            or (training.get('source_receipts') or [{}])[0].get('sha256')!=source_sha256):
        raise ValueError('fresh checkpoint has inherited training, exposure, or wrong source')
    after=path.stat()
    if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns):
        raise ValueError('fresh checkpoint changed during verification')
    return path


def prepare(args):
    from .production import configure_native
    source=Path(args.source).resolve(strict=True);cases=Path(args.cases).resolve(strict=True)
    held=Path(args.held_native_graphs).resolve(strict=True);recipe=Path(args.production_recipe).resolve(strict=True)
    for path,digest in ((source,args.source_sha256),(cases,args.cases_sha256),(held,args.held_sha256),(recipe,args.recipe_sha256)):
        if sha256_file_hex_cached(path)!=digest:raise ValueError('JOLT input differs from its declared exact digest')
    proof=json.loads(Path(args.protection_proof).read_text())
    if (proof.get('schema')!='natlang.prompt-variant-protection-union/1' or proof.get('status')!='passed'
            or proof.get('train_sha256')!=args.source_sha256 or proof.get('cases_sha256')!=args.cases_sha256
            or proof.get('conflicts')!=[]):raise ValueError('JOLT cases lack current source/protection-union proof')
    custody=ProtectedWorlds.load()
    if custody.digest!=proof.get('world_protection_sha256'):raise ValueError('JOLT protected-world inventory changed')
    rows=read_jsonl(cases)
    if not rows or len(rows)!=proof.get('rows'):raise ValueError('JOLT train-case count changed')
    for row in rows:custody.check(row)
    if args.fresh_checkpoint:
        if args.pin_receipt or not args.fresh_checkpoint_sha256:
            raise ValueError('fresh initialization requires its digest and no drained parent')
        parent=checked_fresh_parent(args.fresh_checkpoint,expected_sha256=args.fresh_checkpoint_sha256,
            source_sha256=args.source_sha256)
    else:
        if not args.pin_receipt or not args.sandbox_id or not args.boot_id or args.min_step is None:
            raise ValueError('earned continuation requires exact drained-parent identity')
        parent,_=checked_parent(args.pin_receipt,sandbox_id=args.sandbox_id,boot_id=args.boot_id,
            min_step=args.min_step,source_sha256=args.source_sha256)
    if Path(args.out).exists() and any(Path(args.out).iterdir()):raise ValueError('JOLT controller output must be a new namespace')
    config=Path(args.out)/'config.json'
    eval_spec=json.loads(Path(args.task_eval).read_text())
    parameters=SimpleNamespace(model='lfm',checkpoint=parent,base=Path(args.base),cases=cases,
        runtime=Path(args.runtime)/'ts-host',actor_runtime=Path(args.runtime),out=Path(args.out),config=config,
        native_participant=None,coefficient_policy=None,coefficient_optimizer=None,coefficient_task_eval=None,
        semantic_adapter=None,semantic_refresh_config=None,seed_native_graphs=source,held_native_graphs=held,
        production_recipe=recipe,production_stage=args.production_stage,port=args.port,rl_policy_loss=None,
        task_eval=eval_spec,task_eval_every=args.task_eval_every)
    return configure_native(parameters)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','source-sha256','cases','cases-sha256',
                 'held-native-graphs','held-sha256','production-recipe','recipe-sha256','protection-proof',
                 'base','runtime','out','task-eval'):
        parser.add_argument('--'+name,required=True)
    for name in ('pin-receipt','sandbox-id','boot-id','fresh-checkpoint','fresh-checkpoint-sha256'):
        parser.add_argument('--'+name)
    parser.add_argument('--min-step',type=int)
    parser.add_argument('--production-stage',default='native_full_sft')
    parser.add_argument('--task-eval-every',type=int,default=25)
    parser.add_argument('--port',type=int,default=8097)
    prepare(parser.parse_args())
if __name__=='__main__':main()
