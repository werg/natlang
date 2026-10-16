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


def candidates(role, state, files, protected):
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
    latest = re.fullmatch(rf'{role}/([0-9a-f]{{64}})/{role}-native-checkpoint-step-(\d+)\.pt', receipt['name'])
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
        match = re.fullmatch(rf'{role}/[0-9a-f]{{64}}/{role}-(?:native-)?checkpoint-step-(\d+)\.pt', f.name)
        if match and int(match[1]) < state['last_verified_step'] and f.name not in keep:
            found.append(f)
    return found


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
    selected = candidates(role, state, files, declaration['protected_files'])
    if apply:
        claim()  # Only valid recovery under an applying policy may acquire ownership.
    intent = {'at': time.time(), 'role': role, 'run': declaration['run'],
              'verified_latest': state['receipt']['name'], 'apply': apply,
              'verified_export_state': state, 'policy': policy_binding,
              'protected_files': declaration['protected_files'],
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
