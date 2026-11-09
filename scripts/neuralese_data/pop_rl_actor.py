#!/usr/bin/env python3
"""Serve the live RL policy from the Pop laptop as a second rollout actor (controller hooks).

    pop_rl_actor.py start --checkpoint CK --policy-id SHA --runtime R [--port 18531] [--bind-host IP]
    pop_rl_actor.py stop
    pop_rl_actor.py status

``start`` is the RL controller's ``after_actor_ready`` hook and ``stop`` its ``after_collection`` hook
(rl/production.py, ``remote_endpoints``). Both always exit 0: when Pop is unreachable, busy (an evaluation holds
its one GPU) or fails to attest, the controller's own attestation poll drops the endpoint and the round collects on
the managed DGX actor alone.

start reuses the Pop behavioural-evaluation machinery (pop_behavior_eval.py): installs the policy's frozen runtime on
Pop, prepares the model-only view of the policy checkpoint on the DGX under that runtime (scope ``rl-controller-pin``;
the diagnostic scope is refused as an RL policy), sends it as an exact delta, and launches pop_dev16_driver.py in
``rl-actor`` mode under Pop's memory ledger: the evaluation's container image, limits and serving flags, bound to the
Pop tailscale address, with a block store so the collector fetches the stored RL blocks over HTTP. The actor attests
the training checkpoint SHA (the transport receipt binds it) and binds ``actor_export`` in every capture: arithmetic on
another GPU differs from the learner's, so the learner guards the engine mismatch per episode and corrects it with
the loss's importance weights instead of applying the exact native replay gate.

While the actor runs, ``<pop-behavior-eval root>/rl-actor-active.json`` keeps the evaluation watcher off Pop's GPU.
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# The controller's frozen NATLANG_REPO would redirect the DGX helper paths (venv, ledger) into the frozen tree.
os.environ.pop('NATLANG_REPO', None)
os.environ.pop('PYTHONPATH', None)
REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / 'training/neuralese'), str(REPO)]

from scripts.neuralese_data import pop_behavior_eval as pbe  # noqa: E402

RL_PORT = 18531
STATE = pbe.ROOT / 'rl-actor.json'
WAITER = pbe.ROOT / 'rl-actor-waiter.json'   # a deferred start waiting for the dev16 evaluation to release Pop
EXPORT_FORMAT = 'native-pop-rtx4060-bf16'
RL_MEMORY_GB = 8   # container cap and ledger claim; 4 GiB was OOM-killed after 4 h at 3 workers (anon 4.2 GiB, 2026-11-09)


def log(**event):
    pbe.log(hook='pop_rl_actor', **event)


def pop_host_address():
    """The address the actor binds and the DGX dials: the SSH host's own address (Pop's tailscale IP)."""
    out = subprocess.run(['ssh', '-G', pbe.POP], capture_output=True, text=True, check=True).stdout
    return next(line.split()[1] for line in out.splitlines() if line.startswith('hostname '))


def attested(endpoint, policy_id):
    try:
        with urllib.request.urlopen(endpoint + '/v1/neuralese/info', timeout=5) as f:
            info = json.load(f)
    except OSError:
        return None
    binding = info.get('rl_policy_binding') or {}
    if binding.get('checkpoint_sha256') == policy_id and binding.get('actor_export'):
        return info
    return None


def pop_containers():
    return pbe.ssh("docker ps --format '{{.Names}}'", check=True).split()


def release():
    pbe.RL_ACTOR_MARKER.unlink(missing_ok=True)


def stop_unit(state):
    unit = state.get('unit')
    if unit:
        pbe.ssh(shlex.join(['systemctl', '--user', 'stop', unit + '.service']), check=False, timeout=120)
        for _ in range(30):
            if pbe.pop_unit_state(unit).get('ActiveState') in ('inactive', 'failed', None):
                break
            time.sleep(2)


def eval_pending():
    return pbe.eval_pending(pbe.read_json(pbe.ROOT / 'state.json') or {'candidates': {}, 'controllers': {},
                                                                         'parent': {'sha256': ''}})


def start(args):
    # A previous actor of this lineage that still serves the same policy under the same runtime (the controller
    # restarted mid-collection) keeps its block store. One under another runtime is replaced: the controller refuses
    # replicas whose Python runtimes disagree, and retained-partial custody rehydrates a fresh actor's store.
    previous = pbe.read_json(STATE)
    runtime = str(Path(args.runtime).resolve(strict=True))
    if (previous and previous['policy_id'] == args.policy_id and previous.get('runtime') == runtime
            and attested(previous['endpoint'], args.policy_id)):
        pbe.write_json(pbe.RL_ACTOR_MARKER, {**pbe.read_json(pbe.RL_ACTOR_MARKER, {}), 'started': time.time()})
        log(event='Pop RL actor already serves the policy', endpoint=previous['endpoint'])
        return previous
    if previous:
        stop_unit(previous)
    if args.defer_to_eval and eval_pending():
        if not args.after_eval:
            # The round starts on the DGX now; a waiter starts this actor once the evaluation releases Pop, and the
            # controller's slots for it join the round when it attests the policy (rl.production late_endpoints).
            unit = f'pop-rl-actor-late-{int(time.time())}'
            command = [sys.executable, str(Path(__file__).resolve()), 'start', '--checkpoint', args.checkpoint,
                       '--policy-id', args.policy_id, '--runtime', args.runtime, '--port', str(args.port),
                       '--ready-seconds', str(args.ready_seconds), '--prepare-wait', str(args.prepare_wait),
                       '--defer-to-eval', '--after-eval', *(['--bind-host', args.bind_host] if args.bind_host else [])]
            subprocess.run(['systemd-run', '--user', '--quiet', '--collect', '--unit', unit,
                            f'--setenv=POP_RL_WAITER_UNIT={unit}', '--', *command], check=True, timeout=60)
            pbe.write_json(WAITER, {'unit': unit, 'policy_id': args.policy_id, 'started': time.time()})
            log(event='Pop RL actor deferred until the dev16 evaluation releases Pop', unit=unit)
            return {'deferred': unit}
        while eval_pending():
            time.sleep(30)
        log(event='dev16 evaluation released Pop; starting the deferred RL actor')
    host = args.bind_host or pop_host_address()
    endpoint = f'http://{host}:{args.port}'
    pbe.ssh('true', timeout=30)
    busy = [name for name in pop_containers() if not name.startswith('poprl-')]
    if busy:
        raise RuntimeError(f'Pop GPU is held by {busy} (an evaluation owns it until it finishes)')
    pbe.write_json(pbe.RL_ACTOR_MARKER, {'started': time.time(), 'policy_id': args.policy_id, 'endpoint': endpoint,
                                         'unit': None})
    started = time.time()
    closure = pbe.ensure_pop_runtime(runtime)
    prep = pbe.prepare('rl', args.checkpoint, runtime, args.policy_id, wait=args.prepare_wait)
    state = pbe.read_json(pbe.ROOT / 'state.json') or {'parent': {'sha256': ''}}
    pbe.make_pop_room(state, prep['view']['bytes'], {prep['source']['sha256']})
    pbe.send(prep)
    source = prep['source']['sha256']
    if source != args.policy_id:
        raise RuntimeError('prepared view binds another source than the controller policy')
    py = prep['python_runtime']['sha256']
    work = f'{pbe.POP_EVAL_ROOT}/rl-actor-{prep["step"]}-{source[:12]}-py{py[:12]}'
    name = f'poprl-{source[:12]}'
    transport_json = json.loads(Path(prep['transport']['path']).read_text())
    job = {'mode': 'rl-actor', 'arm': 'rl', 'name': name, 'port': args.port, 'host': host, 'runtime': runtime,
           'runtime_closure_sha256': closure, 'base': pbe.POP_BASE, 'image': pbe.IMAGE, 'step': prep['step'],
           'source_sha256': source, 'stage': f'{pbe.POP_TRANSPORT_ROOT}/{source}', 'view': prep['view']['path'],
           'view_sha256': prep['view']['sha256'], 'work': work, 'min_free_bytes': pbe.POP_MIN_FREE,
           'actor_export_format': EXPORT_FORMAT, 'memory': f'{RL_MEMORY_GB}g',
           'transport': {'name': Path(prep['transport']['path']).name, 'sha256': prep['transport']['sha256'],
                         'view_receipt_name': prep['view_receipt']['sha256'] + '.json',
                         'view_receipt_sha256': prep['view_receipt']['sha256'], 'scope': transport_json['scope']}}
    # A restart under the same policy reuses this work dir (and its block store); the previous driver copy came from a
    # sealed read-only runtime file, so it is replaced rather than overwritten.
    pbe.ssh(shlex.join(['mkdir', '-p', work]) + ' && ' + shlex.join(['rm', '-f', f'{work}/driver.py']))
    subprocess.run(['scp', '-q', str(REPO / 'scripts/neuralese_data/pop_dev16_driver.py'), f'{pbe.POP}:{work}/driver.py'],
                   check=True, timeout=120)
    pbe._pop_upload_json(job, f'{work}/job.json')
    unit = f'{name}-{int(time.time())}'
    pbe.ssh(shlex.join(['/usr/bin/python3', pbe.POP_REPO + '/scripts/memory_ledger.py', 'run', '--unit', unit,
                        '--budget-gb', str(RL_MEMORY_GB), '--reserve-gb', str(pbe.POP_RESERVE_GB),
                        '--class', 'experiment', '--no-resumable', '--container', name, '--workdir', work, '--',
                        '/usr/bin/python3', f'{work}/driver.py', f'{work}/job.json']))
    result = {'endpoint': endpoint, 'unit': unit, 'container': name, 'policy_id': source, 'work': work, 'runtime': runtime,
              'prepared_seconds': round(time.time() - started, 1)}
    pbe.write_json(STATE, result)
    pbe.write_json(pbe.RL_ACTOR_MARKER, {'started': time.time(), 'policy_id': source, 'endpoint': endpoint, 'unit': unit})
    try:
        pbe.prune_dgx(state)   # older derived DGX views (rl-views included); the one just sent stays as the next delta base
    except Exception as error:
        log(event='DGX view pruning failed', error=repr(error)[:300])
    deadline = time.monotonic() + args.ready_seconds
    while time.monotonic() < deadline:
        if pbe.pop_unit_state(unit).get('ActiveState') in ('inactive', 'failed'):
            tail = pbe.ssh(shlex.join(['journalctl', '--user', '-u', unit + '.service', '-n', '20', '--no-pager']), check=False)
            raise RuntimeError('Pop RL actor unit exited before readiness: ' + tail[-1200:])
        if attested(endpoint, source):
            result['ready_seconds'] = round(time.time() - started, 1)
            pbe.write_json(STATE, result)
            log(event='Pop RL actor ready', **result)
            return result
        time.sleep(5)
    raise RuntimeError(f'Pop RL actor did not attest {source[:12]} at {endpoint} within {args.ready_seconds} s')


def stop(_args=None):
    waiter = pbe.read_json(WAITER)
    if waiter:
        # A round that ends (or a controller that stops) before the evaluation released Pop cancels its waiter.
        if waiter['unit'] != os.environ.get('POP_RL_WAITER_UNIT'):
            subprocess.run(['systemctl', '--user', 'stop', waiter['unit']], check=False, timeout=60)
        WAITER.unlink(missing_ok=True)
    state = pbe.read_json(STATE)
    if state:
        stop_unit(state)
        pbe.ssh(shlex.join(['docker', 'rm', '-f', state['container']]), check=False, timeout=60)
        # The store stays: this hook also runs when a controller stops or crashes mid-round, before the collector has
        # bundled the round's blocks (2026-11-08: deleting it lost 22 native states of Pop-sampled rollouts). Only
        # compile caches are removed; the container wrote them as root, so from a throwaway container.
        work = state['work']
        pbe.ssh(shlex.join(['docker', 'run', '--rm', '-v', f'{work}:/work', 'alpine:3.21', 'sh', '-c',
                            'rm -rf ' + ' '.join(f'/work/{name}' for name in ('inductor', 'triton', 'hf', 'cuda'))]),
                check=False, timeout=120)
        STATE.unlink(missing_ok=True)
        log(event='Pop RL actor stopped', unit=state['unit'])
    release()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest='action', required=True)
    s = commands.add_parser('start')
    s.add_argument('--checkpoint', required=True)
    s.add_argument('--policy-id', required=True)
    s.add_argument('--runtime', required=True, help='the frozen runtime that serves the policy (also installed on Pop)')
    s.add_argument('--port', type=int, default=RL_PORT)
    s.add_argument('--bind-host')
    s.add_argument('--defer-to-eval', action='store_true',
                   help='while the evaluation watcher has a queued/running/undiscovered candidate, return at once and '
                        'start the actor from a background waiter when Pop is released; the round joins it then '
                        '(default: rollouts take Pop; the watcher then waits for the collection to end)')
    s.add_argument('--after-eval', action='store_true', help=argparse.SUPPRESS)
    s.add_argument('--ready-seconds', type=int, default=600)
    s.add_argument('--prepare-wait', type=int, default=900, help='ledger admission bound of the DGX view preparation')
    commands.add_parser('stop')
    commands.add_parser('status')
    args = parser.parse_args(argv)
    if args.action == 'status':
        print(json.dumps({'state': pbe.read_json(STATE), 'marker': pbe.read_json(pbe.RL_ACTOR_MARKER)}))
        return 0
    try:
        if args.action == 'start':
            print(json.dumps(start(args)), flush=True)
        else:
            stop()
    except BaseException as error:   # never fail the controller's round: its attestation poll drops the endpoint
        log(event=f'Pop RL actor {args.action} failed; the round proceeds without it', error=repr(error)[:2000])
        if args.action == 'start':
            try:
                stop()
            except Exception as cleanup:
                log(event='Pop RL actor cleanup failed', error=repr(cleanup)[:500])
        if isinstance(error, KeyboardInterrupt):
            raise
    return 0


if __name__ == '__main__':
    sys.exit(main())
