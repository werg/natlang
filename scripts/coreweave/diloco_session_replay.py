"""Replay an elastic DiLoCo session start on the DGX: CPU containers that see only each worker's shipped closure.

Each named worker lineage is staged exactly as ``rl_cloud seed`` stages it (the prepared session control relocated as
window 0 of ``ROOT/<lineage>``, its ``sft_window closure``, the historical checkpoints its seed would ship, including
``historical_from``), but nothing is uploaded: the lineage tree is written under ``--work`` and mounted at its lineage
path. Every worker runs in its own container (closure_replay mounts) on one internal docker network, with its own
``/work/diloco`` state root and ``$NATLANG_CW_SECRETS`` (a replay-only bearer token; non-cohosts reach the cohost's
coordinator through their own relay, as in the cloud). Each container runs the real controller
(``natlang_neuralese.rl.production run --rounds 1``): join.py waits for the secrets, starts the relay, the cohost
initializes the coordinator from its parent and serves it, every worker computes and admits its role, and the trainer
runs its CPU preflight until it needs CUDA.

Then each container runs the exchange probe (diloco_exchange_probe.py, under the worker's frozen runtime): a
DiLoCo learner over the worker's parent masters, with its admitted role, installs the seed fragments, uploads fragment 0
after the session's local steps, and waits until every probe has installed the merged fragment 0 version 1 through
its relay. The replay passes when every controller reached the CUDA model load (or later) after ``role_admitted`` and
every probe reports the merged install.

Usage:
  diloco_session_replay.py --run RUN.json --worker LINEAGE ... --work DIR [--unit NAME] [--build-only]

Run under the memory ledger (``memory_ledger.py run --unit NAME --container NAME-w<K> ...``). Containers are named
``NAME-w<K>``; the network ``NAME``. Everything is removed afterwards unless --keep.
"""
import argparse
import json
import os
import secrets as pysecrets
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts/coreweave'))
sys.path.insert(0, str(REPO / 'training/neuralese'))

import closure_replay  # noqa: E402
import rl_cloud  # noqa: E402

PROBE = Path(__file__).resolve().parent / 'diloco_exchange_probe.py'


def stage(run, name, work):
    """The lineage tree and closure ``rl_cloud seed`` would build for a prepared session control, under ``work``."""
    spec = run['lineages'][name]
    boundary = rl_cloud.prepared_boundary(spec)
    L = rl_cloud.lineage_dir(run, name)
    copy = work / 'source' / name
    if copy.exists():
        shutil.rmtree(copy)
    window, parent_path = L / 'w000000', L / 'parent' / 'checkpoint.pt'
    source = json.loads(Path(spec['controller']).read_text())
    config = rl_cloud.relocate_control(source, window, parent_path, f'lfm3-cloud-{name}-w000000')
    for key, value in spec.get('config', {}).items():
        if value is None:
            config.pop(key, None)
        else:
            config[key] = value
    (copy / 'w000000').mkdir(parents=True)
    (copy / 'w000000' / 'dgx-prepared-config.json').write_text(json.dumps(source, indent=1) + '\n')
    (copy / 'w000000' / 'config.json').write_text(json.dumps(config, indent=1) + '\n')
    (copy / 'lineage.json').write_text(json.dumps({'schema': rl_cloud.sandbox.LINEAGE_SCHEMA, 'origin': boundary,
                                                   'windows': [{'index': 0, 'out': str(window),
                                                                'config': str(window / 'config.json')}]}, indent=1))
    # The closure tool reads the control at its lineage path: stage it there only for the closure computation.
    closure_path = work / f'closure-{name}.json'
    probe_dir = work / 'closure-input' / name
    if probe_dir.exists():
        shutil.rmtree(probe_dir)
    shutil.copytree(copy, probe_dir)
    text = (probe_dir / 'w000000' / 'config.json').read_text().replace(str(L), str(probe_dir))
    (probe_dir / 'w000000' / 'config.json').write_text(text)
    rl_cloud.runtime_tool(boundary['runtime'], 'closure', '--config', str(probe_dir / 'w000000' / 'config.json'),
                          '-o', str(closure_path), *rl_cloud.observed_reads(spec))
    closure = json.loads(closure_path.read_text())
    # Paths of the staged probe tree are the lineage's own paths in the sandbox.
    closure['files'] = [f for f in closure['files'] if not f['path'].startswith(str(probe_dir))]
    missing = [m['path'] for m in closure['missing'] if not m['path'].startswith(str(probe_dir))
               and m['path'] not in set(spec.get('allow_missing', []))]
    if missing:
        raise SystemExit(f'{name}: inputs the control names are absent on the DGX: {missing[:20]}')
    historical, skipped, links = rl_cloud.historical_checkpoint_blobs(run, closure, parent_path)
    for origin in spec.get('historical_from', []):
        seed = rl_cloud.read_json(rl_cloud.hub(run) / 'exports' / origin / 'seed.json')
        origin_parent = str(rl_cloud.lineage_dir(run, origin) / 'parent' / 'checkpoint.pt')
        have = {b['path'] for b in historical}
        historical += [b for b in seed['blobs'] if b.get('kind') == 'checkpoint' and b['path'] != origin_parent
                       and b['path'] not in have]
        links += [x for x in seed.get('links', []) if x not in links]
    return {'name': name, 'spec': spec, 'boundary': boundary, 'source': copy, 'closure': closure,
            'historical': historical, 'skipped': skipped, 'links': links, 'config': window / 'config.json',
            'worker': spec['diloco_session']['worker'], 'session': spec['diloco_session']['session']}


def container_argv(run, item, work, unit, network, token, cohost_alias):
    """closure_replay mounts for one staged worker, plus its secrets, DiLoCo state root and network."""
    name, K = item['name'], item['worker']
    wdir = work / f'w{K}'
    wdir.mkdir(parents=True, exist_ok=True)
    closure = dict(item['closure'])
    out, runtime, summary = closure_replay.mounts(run, name, closure, Path(item['boundary']['parent']), wdir, [],
                                                  source=item['source'], historical=(item['historical'], item['links']))
    secrets = wdir / 'secrets'
    if secrets.exists():
        shutil.rmtree(secrets)
    secrets.mkdir()
    (secrets / 'diloco-token').write_text(token + '\n')
    if cohost_alias is not None:  # a non-cohost reaches the cohost's coordinator as it would through a sandbox URL
        (secrets / 'diloco-url').write_text(f'http://{cohost_alias}:8080\n')
        (secrets / 'diloco-gateway').write_text('')
    state = wdir / 'diloco-state'
    state.mkdir(exist_ok=True)
    out += ['-v', f'{secrets}:/work/.coreweave/secrets', '-v', f'{state}:/work/diloco', '-v', f'{PROBE}:/replay-probe.py:ro']
    env = closure_replay.environment(runtime, Path(item['config']).parent)
    env['NATLANG_CW_SECRETS'] = '/work/.coreweave/secrets'
    for k, v in env.items():
        out += ['-e', f'{k}={v}']
    controller = (f'{closure_replay.VENV}/bin/python -u -m natlang_neuralese.rl.production run --config '
                  f'{item["config"]} --rounds 1 > /replay-state/controller.log 2>&1; '
                  f'echo "controller_exit=$?" >> /replay-state/controller.log; exec sleep infinity')
    return ['docker', 'run', '-d', '--name', f'{unit}-w{K}', '--network', network, '--network-alias', f'w{K}',
            '--user', f'{os.getuid()}:{os.getgid()}', '--shm-size', '8g', *out, '-w', runtime,
            closure_replay.IMAGE, 'bash', '-c', controller], runtime, summary


def joined(item, work):
    """The join log events of one worker (its window is under closure_replay's writable lineage copy)."""
    path = work / f'w{item["worker"]}' / item['name'] / 'w000000' / 'learner' / 'diloco-join.jsonl'
    return [json.loads(x) for x in path.read_text().splitlines()] if path.is_file() else []


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', required=True)
    ap.add_argument('--worker', action='append', required=True, help='lineage name with a diloco_session')
    ap.add_argument('--work', required=True)
    ap.add_argument('--unit', default='diloco-replay')
    ap.add_argument('--build-only', action='store_true')
    ap.add_argument('--keep', action='store_true')
    ap.add_argument('--timeout', type=float, default=3 * 3600)
    a = ap.parse_args(argv)
    run = json.loads(Path(a.run).read_text())
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    items = [stage(run, name, work) for name in a.worker]
    sessions = {item['session'] for item in items}
    if len(sessions) != 1:
        raise SystemExit(f'workers name different sessions: {sessions}')
    session = json.loads(Path(sessions.pop()).read_text())
    cohost = [item for item in items if item['worker'] == session['cohost_worker']]
    if len(cohost) != 1:
        raise SystemExit(f'exactly one replayed worker must be the session cohost (worker {session["cohost_worker"]})')
    token = pysecrets.token_hex(32)
    commands, report = [], {'workers': {}}
    for item in items:
        argv, runtime, summary = container_argv(run, item, work, a.unit, a.unit, token,
                                                None if item is cohost[0] else f'w{session["cohost_worker"]}')
        commands.append(argv)
        report['workers'][item['name']] = {'worker': item['worker'], 'skipped_historical': item['skipped'],
                                           'historical': [b['path'] for b in item['historical']], **summary}
        (work / f'w{item["worker"]}' / 'docker.argv').write_text('\n'.join(argv) + '\n')
    print(json.dumps(report, indent=1, default=str), flush=True)
    if a.build_only:
        return 0
    subprocess.run(['docker', 'network', 'create', '--internal', a.unit], check=True, capture_output=True)
    names = [f'{a.unit}-w{item["worker"]}' for item in items]
    try:
        for argv in commands:
            subprocess.run(argv, check=True, capture_output=True)
        deadline = time.monotonic() + a.timeout
        result = {}
        while time.monotonic() < deadline:
            events = {item['name']: [e['event'] for e in joined(item, work)] for item in items}
            logs = {item['name']: (work / f'w{item["worker"]}' / 'state' / 'controller.log') for item in items}
            done = {n: p.is_file() and 'controller_exit=' in p.read_text(errors='replace') for n, p in logs.items()}
            if all(done.values()):
                result['events'] = events
                break
            time.sleep(20)
        else:
            raise SystemExit(f'replay timed out: {events}')
        failed = [n for n, ev in result['events'].items() if 'exec_trainer' not in ev]
        print(json.dumps({'event': 'controllers_done', 'join_events': result['events'], 'not_admitted': failed}),
              flush=True)
        if failed:
            return 1
        probes = []
        for item, name in zip(items, names):
            parent = rl_cloud.lineage_dir(run, item['name']) / 'parent' / 'checkpoint.pt'
            log = open(work / f'w{item["worker"]}' / 'probe.log', 'wb')
            probes.append(subprocess.Popen(['docker', 'exec', name, f'{closure_replay.VENV}/bin/python', '-u',
                                            '/replay-probe.py', '--session', item['session'], '--worker',
                                            str(item['worker']), '--parent', str(parent), '--expect', str(len(items))],
                                           stdout=log, stderr=subprocess.STDOUT))
        codes = [p.wait() for p in probes]
        print(json.dumps({'event': 'probes_done', 'exit_codes': codes}), flush=True)
        return 0 if not any(codes) else 1
    finally:
        if not a.keep:
            for name in names:
                subprocess.run(['docker', 'rm', '-f', name], capture_output=True)
            subprocess.run(['docker', 'network', 'rm', a.unit], capture_output=True)


if __name__ == '__main__':
    sys.exit(main())
