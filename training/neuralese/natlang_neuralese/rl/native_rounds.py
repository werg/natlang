"""Parallel sealed native RL rounds followed by one joint semantic refresh.

Learners update different participants of ONE immutable parent concurrently.
This is synchronous policy-lag-zero execution; no stale/off-policy KV reuse.
Commands in the existing per-model controller may run locally or through the
qualified sandbox execution bridge. No cloud resources are allocated here.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
from pathlib import Path

from ..common.hashing import sha256_file_hex_cached
from ..common.jsonio import write_json_durable
from .production import run, run_command


def apply_deployment(config, deployment):
    """Idempotent publication bookkeeping, even after a crash between actors."""
    out = Path(config['out']); out.mkdir(parents=True, exist_ok=True)
    path = out/'state.json'
    if not path.exists():
        return
    with (out/'controller.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads(path.read_text())
        name = config['native_participant']
        pending = state.get('pending_native_checkpoint')
        if pending:
            accepted = deployment.get('updated_participants', {}).get(name)
            if accepted and accepted['sha256'] == sha256_file_hex_cached(pending):
                state.update(round=state['round'] + 1, trained_rounds=state['trained_rounds'] + 1,
                             collection_attempt=0, phase='collect')
                state.pop('pending_native_checkpoint', None)
                state.pop('pending_native_round_dir', None)
            else:
                # Pending learner belongs to the current sealed parent and
                # must still join the refresh, not be replaced by another actor.
                if state['checkpoint'] != deployment['checkpoint']['path']:
                    raise ValueError('pending native RL update was not merged into the published joint checkpoint')
                return
        if state.get('checkpoint') != deployment['checkpoint']['path']:
            state['group_stats'] = []
        state.update(checkpoint=deployment['checkpoint']['path'],
                     semantic_adapter=deployment['semantic_adapter']['path'])
        write_json_durable(path, state)


def run_rounds(config_paths, out, *, rounds=None, device='cuda'):
    from .native_refresh import refresh
    from ..train.recipe import resolve_recipe_data
    out = Path(out).resolve(); out.mkdir(parents=True, exist_ok=True)
    with (out/'joint-controller.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        latest = out/'latest.json'
        completed = 0
        while rounds is None or completed < rounds:
            configs = [resolve_recipe_data(path) for path in config_paths]
            names = [config.get('native_participant') for config in configs]
            if (len(configs) < 2 or len(set(names)) != len(names)
                    or not all(config.get('native_state') for config in configs)):
                raise ValueError('joint RL requires distinct native participant configurations')
            if latest.exists():
                transaction = json.loads(latest.read_text())
                deployment, cycle = transaction['deployment'], transaction['next_round']
                for config in configs:
                    apply_deployment(config, deployment)
                adapter = deployment['semantic_adapter']['path']
            else:
                cycle = 0
                adapters = {str(Path(config['semantic_adapter']).resolve()) for config in configs}
                parents = {sha256_file_hex_cached(config['checkpoint']) for config in configs}
                if len(adapters) != 1 or len(parents) != 1:
                    raise ValueError('parallel native learners must start from the same immutable joint parent')
                adapter = next(iter(adapters))
            semantic_configs = {str(Path(config['semantic_refresh_config']).resolve()) for config in configs}
            if len(semantic_configs) != 1:
                raise ValueError('both learners must share one actual semantic refresh curriculum')
            # Each existing controller durably seals its packet and finishes
            # its complete shared-trainer epoch before returning its update.
            with ThreadPoolExecutor(max_workers=len(configs)) as pool:
                futures = [pool.submit(run, path, rounds=1, defer_native_refresh=True) for path in config_paths]
                states, failures = [], []
                for future in futures:
                    try:
                        states.append(future.result())
                    except Exception as error:
                        failures.append(error)
                        states.append(None)
            if failures:
                raise failures[0]  # successful sibling checkpoints remain resumable/pending
            updates = {name: state['pending_native_checkpoint'] for name, state in zip(names, states)
                       if state.get('pending_native_checkpoint')}
            if not updates:
                return {'phase': 'await_useful_rollouts_or_data', 'states': states, 'production_complete': False}
            directory = out/f'round-{cycle:06d}'
            deployment = refresh(adapter=adapter, updates=updates, config=next(iter(semantic_configs)),
                                 out=directory/'semantic-refresh', device=device)
            # All external actors receive the whole joint publication, not
            # only the model which happened to finish its learner first.
            for config in configs:
                if config.get('after_update'):
                    run_command(config['after_update'], {**config,
                        'checkpoint': deployment['checkpoint']['path'],
                        'policy_id': deployment['checkpoint']['sha256'],
                        'semantic_adapter': deployment['semantic_adapter']['path'],
                        'semantic_adapter_sha256': deployment['semantic_adapter']['sha256'],
                        'deployment': str(directory/'semantic-refresh/deployment.json')})
            write_json_durable(latest, {'schema': 'natlang.native-rl-joint-controller/1',
                                      'deployment': deployment, 'next_round': cycle + 1})
            for config in configs:
                apply_deployment(config, deployment)
            completed += 1
        return {'phase': 'collect', 'deployment': deployment, 'production_complete': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', action='append', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--rounds', type=int)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args(argv)
    return run_rounds(args.config, args.out, rounds=args.rounds, device=args.device)


if __name__ == '__main__':
    main()
