"""Prune superseded private checkpoint exports under an explicit owner policy.

CLI defaults to a dry run. Model/source data, unknown names and unverified newer
uploads are outside this checkpoint-only mechanism.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'training' / 'neuralese'))
from natlang_neuralese.common.jsonio import write_json_durable
from natlang_neuralese.common.hashing import sha256_hex


@contextmanager
def run_authority(run, policy_path, directory):
    """One explicit policy owns each run across all local journal directories."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    key = sha256_hex(run.encode('utf-8'))
    with (root / (key + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        authority = root / (key + '.json')
        expected = {'run': run, 'policy': str(Path(policy_path).resolve(strict=True))}
        if authority.exists():
            if json.loads(authority.read_text()) != expected:
                raise ValueError('storage run already belongs to another explicit retention policy')
        def claim():
            if not authority.exists():
                write_json_durable(authority, expected)
        yield claim


def candidates(role, state, files, protected, *, kind='checkpoint'):
    receipt = state['receipt']
    identity = receipt['identity']
    fetched = receipt['independent_verification']
    if (receipt.get('private') is not True or
            not re.fullmatch('[0-9a-f]{64}', identity['sha256']) or
            type(identity['bytes']) is not int or identity['bytes'] <= 0 or
            any(fetched.get(k) != identity[k] for k in ('sha256', 'bytes'))):
        raise ValueError('retention requires independently verified full private recovery')
    if role not in ('lfm', 'mellum'):
        raise ValueError('unknown checkpoint export role')
    if kind not in ('checkpoint','inference'):
        raise ValueError('unknown retained native artifact kind')
    latest = re.fullmatch(rf'{role}/([0-9a-f]{{64}})/{role}-native-{kind}-step-(\d+)\.pt', receipt['name'])
    if (not latest or latest[1] != identity['sha256'] or
            type(state['last_verified_step']) is not int or
            int(latest[2]) != state['last_verified_step']):
        raise ValueError('latest export name differs from verified checkpoint identity')
    files = list(files)
    present = {f.name: f for f in files}
    if receipt['name'] not in present or present[receipt['name']].size != identity['bytes']:
        raise ValueError('latest verified recovery file is missing or has changed size')
    keep = {receipt['name'], *protected}
    found = []
    for f in files:
        spelling = '(?:native-)?checkpoint' if kind=='checkpoint' else 'native-inference'
        match = re.fullmatch(rf'{role}/[0-9a-f]{{64}}/{role}-{spelling}-step-(\d+)\.pt', f.name)
        if match and int(match[1]) < state['last_verified_step'] and f.name not in keep:
            found.append(f)
    return found


def inference_dependencies(role, declaration):
    """Read existing transfer/request custody without loading any model bytes."""
    protected=set(declaration['protected_files'])
    def protect_receipt(receipt):
        identity=receipt['identity'];proof=receipt['independent_verification']
        match=re.fullmatch(rf'{role}/([0-9a-f]{{64}})/{role}-native-(?:inference|checkpoint)-step-(\d+)\.pt',receipt['name'])
        if (receipt.get('private')is not True or receipt.get('run')!=declaration['run']
                or type(identity.get('bytes'))is not int or identity['bytes']<=0
                or any(proof.get(k)!=identity[k]for k in ('sha256','bytes'))
                or not match or match[1]!=identity['sha256']):
            raise ValueError('native retention dependency has no verified same-run artifact identity')
        protected.add(receipt['name'])
        return int(match[2])
    def protect_parent(parent):
        if (parent.get('schema')!='natlang.native-inference-source/1'
                or not re.fullmatch('[0-9a-f]{64}',parent.get('training_checkpoint_sha256',''))
                or type(parent.get('step'))is not int or parent['step']<0
                or type(parent.get('training_checkpoint_bytes'))is not int or parent['training_checkpoint_bytes']<=0):
            raise ValueError('native inference dependency has no exact full parent')
        protected.add(f"{role}/{parent['training_checkpoint_sha256']}/{role}-native-checkpoint-step-{parent['step']}.pt")
    for directory in declaration.get('inference_ingesters',[]):
        directory=Path(directory)
        for path in (directory/'incoming').glob('*.receipt.json'):
            protect_receipt(json.loads(path.read_text()))
        state=directory/'native-async-state.json'
        active=json.loads(state.read_text()).get('active')if state.exists()else None
        if active:
            request=json.loads((Path(active['request'])/'request.json').read_text())
            evidence=request['evaluation_provenance']
            if evidence['source_checkpoint_sha256']!=active['checkpoint_sha256']:
                raise ValueError('active inference request changed its source identity')
            receipt=evidence['source_export'];step=protect_receipt(receipt)
            if receipt['identity']['sha256']!=active['checkpoint_sha256']or step!=active['step']:
                raise ValueError('active inference export receipt differs from the sealed source')
            if evidence.get('inference_source')is not None:protect_parent(evidence['inference_source'])
    view_path=declaration.get('inference_state')
    view=json.loads(Path(view_path).read_text())if view_path and Path(view_path).exists()else None
    if view is not None:
        protect_receipt(view['receipt']);protect_parent(view['inference_source'])
        if view['inference_source']['step']!=view['last_verified_step']:
            raise ValueError('inference pointer differs from full-parent step')
        if view['inference_source']['training_checkpoint_bytes']<=view['receipt']['identity']['bytes']:
            raise ValueError('inference pointer is not a pruned full-parent view')
    for path in declaration.get('inference_pending',[]):
        if Path(path).exists():
            pending=json.loads(Path(path).read_text());receipt=pending['full_private_receipt']
            step=protect_receipt(receipt)
            parent=pending['full_training_checkpoint']
            if (parent!={'step':step,**receipt['identity']}or pending['step']!=step):
                raise ValueError('pending inference full receipt differs from its declared parent')
            protect_parent({'schema':'natlang.native-inference-source/1','step':parent['step'],
                'training_checkpoint_sha256':parent['sha256'],'training_checkpoint_bytes':parent['bytes']})
            identity=pending['identity']
            if not re.fullmatch('[0-9a-f]{64}',identity['sha256']):raise ValueError('pending inference identity is invalid')
            protected.add(f"{role}/{identity['sha256']}/{role}-native-inference-step-{pending['step']}.pt")
    return protected,view


def tick(policy_path, journal_directory, *, apply=False, api=None, ownership_directory=None):
    if api is None:
        import wandb
        api = wandb.Api(timeout=45)
    policy_bytes = Path(policy_path).read_bytes()
    policy = json.loads(policy_bytes)
    policy_binding = {'path': str(Path(policy_path).resolve()), 'sha256': sha256_hex(policy_bytes),
                      'bytes': len(policy_bytes)}
    root = Path(journal_directory)
    root.mkdir(parents=True, exist_ok=True)
    summaries = []
    owners = ownership_directory or Path.home() / '.local/state/natlang/coreweave/checkpoint-retention'
    for role, declaration in policy['models'].items():
        with run_authority(declaration['run'], policy_path, owners) as claim:
            summary = _prune_run(role, declaration, root, apply, api, policy_binding, claim)
        summaries.append(summary)
    return summaries


def _prune_run(role, declaration, root, apply, api, policy_binding, claim):
    state = json.loads(Path(declaration['state']).read_text())
    if state['receipt']['run'] != declaration['run']:
        raise ValueError('retention state belongs to a different storage run')
    files = list(api.run(declaration['run']).files(per_page=1000))
    protected,view = inference_dependencies(role,declaration)
    # During a download the existing incoming receipt knows the physical view
    # and saved step; full-parent provenance is sealed only after loading it.
    # Preserve same-step full parents across that gap, even if a newer pointer
    # arrives. Unknown same-step aliases are kept until that consumer drains.
    view_steps={int(match[1])for name in protected if (match:=re.fullmatch(
        rf'{role}/[0-9a-f]{{64}}/{role}-native-inference-step-(\d+)\.pt',name))}
    for file in files:
        match=re.fullmatch(rf'{role}/[0-9a-f]{{64}}/{role}-(?:native-)?checkpoint-step-(\d+)\.pt',file.name)
        if match and int(match[1])in view_steps:protected.add(file.name)
    selected = candidates(role, state, files, protected)
    if view is not None:
        if view['last_verified_step']>state['last_verified_step']:
            raise ValueError('inference pointer is ahead of verified full recovery')
        if (view['last_verified_step']==state['last_verified_step'] and
                view['inference_source']['training_checkpoint_sha256']!=state['receipt']['identity']['sha256']):
            raise ValueError('inference pointer and full recovery disagree at the same step')
        selected += candidates(role,view,files,protected,kind='inference')
    if apply:
        claim()  # Only valid recovery under an applying policy may acquire ownership.
    intent = {'at': time.time(), 'role': role, 'run': declaration['run'],
              'verified_latest': state['receipt']['name'], 'apply': apply,
              'verified_export_state': state, 'policy': policy_binding,
              'protected_files': sorted(protected),
              **({'verified_inference_export_state':view}if view is not None else {}),
              'files': [{'name': f.name, 'bytes': f.size, 'md5': f.md5} for f in selected]}
    path = root / f'{role}-rolling-intent-{time.time_ns()}.json'
    if selected:
        write_json_durable(path, intent)
    if apply and selected:
        with (root / 'rolling-deletions.jsonl').open('a', buffering=1) as log:
            for f in selected:
                f.delete()
                log.write(json.dumps({'at': time.time(), 'run': declaration['run'],
                                      'name': f.name, 'bytes': f.size, 'intent': str(path)}) + '\n')
                log.flush()
                os.fsync(log.fileno())
        remaining = {f.name for f in api.run(declaration['run']).files(per_page=1000)}
        if any(f.name in remaining for f in selected):
            raise RuntimeError('deleted checkpoint export is still listed')
    summary = {'role': role, 'apply': apply, 'candidates': len(selected),
               'bytes': sum(f.size for f in selected), 'verified_latest_step': state['last_verified_step']}
    print(json.dumps(summary), flush=True)
    return summary

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--journal-dir', type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--every', type=int)
    args = parser.parse_args(argv)
    if args.every is not None and args.every <= 0:
        parser.error('--every must be positive')
    try:
        from . import cwctl
    except ImportError:
        import cwctl
    cwctl.load_env_file()
    args.journal_dir.mkdir(parents=True, exist_ok=True)
    with (args.journal_dir / 'rolling-retention.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            try:
                tick(args.policy, args.journal_dir, apply=args.apply)
            except Exception as error:
                # Storage exceptions may contain signed URLs or credentials.
                print(json.dumps({'at': time.time(), 'error': type(error).__name__,
                                  'scope': 'retention failed closed; no replacement authority granted'}), flush=True)
                if args.every is None:
                    raise
            if args.every is None:
                break
            time.sleep(args.every)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
