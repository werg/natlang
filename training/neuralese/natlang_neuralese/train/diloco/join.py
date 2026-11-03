"""Join one learner to an elastic DiLoCo session: the per-worker role contract and the join path.

An elastic session (``natlang.diloco-elastic-session/1``) has N worker slots, a quorum of 1 and one session window
control prepared once from an earned boundary. Every worker runs that control relocated to its own lineage (its
window directory and parent copy differ; everything else, the parent bytes included, is identical). The control's
``seed_command`` runs this wrapper in front of the unchanged trainer command:

    python join.py --session SESSION.json --worker K -- PYTHON -m natlang_neuralese.train.native_state_sft ARGS...

Before exec'ing the trainer it

1. waits for the DiLoCo secrets (``diloco-token``; ``diloco-url``/``diloco-gateway`` name the current coordinator),
2. ensures this sandbox's relay (relay.py) serves ``127.0.0.1:relay_port``, the coordinator URL every worker's
   ``--diloco`` block names (a failover rewrites the two secret files; no trainer argument changes),
3. computes this worker's exact role exactly as the trainer will select it: the effective argv
   (``role_contract.resident_argv`` over the committed resident plan, as ``native_state_sft`` does), the identity
   (``native_diloco.identity``) and the exact file bindings ``role_contract.validate`` requires, then validates them
   against the local files,
4. on the session's cohost worker only: initializes the coordinator root from this worker's parent (the session
   seed) with this role as the namespace template and serves it (detached, with the bundle's own code), and
5. admits the role at the coordinator (``POST /v1/admit-role``; idempotent).

It imports only modules the frozen trainer runtime already has (runtime v44 and later) and the standard library;
the coordinator runs the bundle code named by the session (``coordinator.code``).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

SESSION_SCHEMA = 'natlang.diloco-elastic-session/1'
REQUIRED_FLAGS = ('--native-graphs', '--held-native-graphs')
OPTIONAL_FLAGS = ('--foundation-native-graphs', '--continued-native-graphs', '--continued-held-native-graphs',
                  '--pending-focus', '--ce-supervision-policy')
TRAINER_MODULE = 'natlang_neuralese.train.native_state_sft'


def flag(argv, name, required=True):
    found = [argv[i + 1] for i, x in enumerate(argv[:-1]) if x == name]
    if len(found) > 1 or (required and not found):
        raise SystemExit(f'join: the trainer argv must carry exactly one {name}')
    return found[0] if found else None


class Log:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, event, **fields):
        record = {'event': event, 'at': time.time(), **fields}
        with self.path.open('a') as stream:
            stream.write(json.dumps(record, sort_keys=True) + '\n')
        print(json.dumps({'diloco_join': record}, sort_keys=True), flush=True)


def wait_for(path, log, what, seconds=6 * 3600):
    deadline = time.monotonic() + seconds
    announced = False
    while not (Path(path).is_file() and Path(path).read_text().strip()):
        if not announced:
            log('await', what=what, path=str(path))
            announced = True
        if time.monotonic() > deadline:
            raise SystemExit(f'join: {what} never arrived at {path}')
        time.sleep(10)


def healthy(url, timeout=3.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read())
    except (OSError, ValueError):
        return None


def spawn(argv, log_path, env=None):
    """A detached, self-restarting daemon in its own session: the trainer's exit or a controller kill does not take it
    down, and a crashed server restarts on its durable state (the coordinator's commit records make replay exact)."""
    import shlex
    loop = f'while true; do {shlex.join(argv)}; echo "{{\\"event\\":\\"daemon_exit\\",\\"rc\\":$?}}"; sleep 5; done'
    with open(log_path, 'ab') as out:
        process = subprocess.Popen(['bash', '-c', loop], stdin=subprocess.DEVNULL, stdout=out, stderr=out, env=env,
                                   start_new_session=True, close_fds=True)
    return process.pid


def ensure_relay(session, state_dir, secrets, log):
    port = session['relay_port']
    if (healthy(f'http://127.0.0.1:{port}/relay-health') or {}).get('ok') is not None:
        return
    pid = spawn([sys.executable, session['relay'], '--port', str(port), '--secrets', secrets],
                state_dir / 'relay.log')
    for _ in range(100):
        if healthy(f'http://127.0.0.1:{port}/relay-health') is not None:
            log('relay_started', pid=pid, port=port)
            return
        time.sleep(0.1)
    raise SystemExit('join: the relay did not start (see relay.log)')


def compute_role(trainer, workers):
    """The worker's role exactly as native_state_sft selects it (effective argv, identity, exact bindings)."""
    from natlang_neuralese.common.hashing import sha256_file_hex_cached
    from natlang_neuralese.train.diloco.role_contract import SCHEMA, resident_argv, validate
    from natlang_neuralese.train.diloco.worker import DilocoSettings
    from natlang_neuralese.train.native_diloco import identity
    settings = DilocoSettings.from_dict(json.loads(flag(trainer, '--diloco')))
    effective = list(trainer)
    resident_directory = flag(trainer, '--resident-data-directory', required=False)
    if resident_directory:
        from natlang_neuralese.train.native_resident import current
        state = current(resident_directory)
        if state is not None:  # native_state_sft: effective_argv = resident_argv(...) whenever a resident plan exists
            effective = resident_argv(trainer, state['payload'], flag(trainer, '--stage-id'))
    parent = Path(flag(effective, '--continue-from'))
    ident = identity(parent, effective, settings)
    argv = ident['trainer_argv']
    binding = json.loads(flag(argv, '--recipe-binding'))
    paths = [binding['path'], *(flag(argv, name) for name in REQUIRED_FLAGS)]
    paths += [value for value in (flag(argv, name, required=False) for name in OPTIONAL_FLAGS) if value]
    refs, seen = [], set()
    for path in paths:
        path = os.path.abspath(path)
        if path not in seen:
            seen.add(path)
            refs.append({'path': path, 'sha256': sha256_file_hex_cached(Path(path))})
    item = {'kind': 'sft', 'identity': ident, 'dependencies': refs}
    contract = {'schema': SCHEMA, 'baseline_sha256': ident['parent_checkpoint_sha256'],
                'recipe_lineage_sha256': binding['lineage_sha256'], 'diloco': settings.identity(),
                'roles': {str(i): [item] for i in range(workers)}}
    validate(contract, workers, verify_files=True)
    namespace = {'out': os.path.normpath(os.path.dirname(flag(argv, '--out'))), 'parent': os.path.normpath(str(parent))}
    return item, contract, namespace, settings


def ensure_coordinator(session, contract, item, namespace, settings, parent, secrets, state_dir, log):
    """Cohost only: initialize the coordinator from the session seed (this worker's parent) once, then serve it."""
    from natlang_neuralese.common.hashing import canonical_json_sha256_hex
    spec = session['coordinator']
    root = Path(spec['root'])
    code = spec['code']
    env = {**os.environ, 'PYTHONPATH': f'{code}/training/neuralese:{code}', 'NATLANG_REPO': code,
           'NATLANG_CW_SECRETS': secrets}
    if not (root / 'manifest.json').is_file():
        declared = {**contract, 'namespace_template': {'worker': session['cohost_worker'], 'namespace': namespace,
                                                       'role_sha256': canonical_json_sha256_hex(item)}}
        path = state_dir / 'role-contract.json'
        path.write_text(json.dumps(declared, sort_keys=True, indent=1) + '\n')
        block = {k: v for k, v in json.loads(json.dumps(settings.__dict__)).items() if k not in ('coordinator_url', 'worker_id')}
        if root.exists() and any(root.iterdir()):
            raise SystemExit(f'join: {root} holds a partial coordinator without a manifest; inspect before retrying')
        log('coordinator_init', root=str(root), parent=str(parent))
        started = time.monotonic()
        subprocess.run([spec['python'], '-m', 'natlang_neuralese.train.native_diloco', 'init', '--root', str(root),
                        '--parent', str(parent), '--trainer-argv', '[]', '--settings', json.dumps(block),
                        '--role-contract', str(path), *spec.get('init', [])], env=env, check=True)
        log('coordinator_initialized', seconds=round(time.monotonic() - started, 1))
    pid_path = state_dir / 'coordinator.pid'
    alive = False
    if pid_path.is_file():
        pid = int(pid_path.read_text().split()[0])
        alive = Path(f'/proc/{pid}').exists() and 'natlang_neuralese.train.diloco' in Path(f'/proc/{pid}/cmdline').read_text(errors='replace')
    if not alive:
        pid = spawn([spec['python'], '-m', 'natlang_neuralese.train.diloco', 'serve', '--root', str(root),
                     '--host', '0.0.0.0', '--port', str(spec['port'])], state_dir / 'coordinator.log', env=env)
        pid_path.write_text(f'{pid}\n')
        log('coordinator_serving', pid=pid, port=spec['port'])


def admit(session, worker, item, namespace, log):
    from natlang_neuralese.train.diloco.coordinator import TOKEN_ENV, TOKEN_SECRET, credential
    from natlang_neuralese.train.diloco.worker import Transport, patient
    transport = Transport(f'http://127.0.0.1:{session["relay_port"]}', credential(TOKEN_ENV, TOKEN_SECRET), 600.0)
    result = patient(lambda: transport.json('POST', '/v1/admit-role', {'worker': worker, 'contract': item,
                                                                      'namespace': namespace}), what='admit-role')
    log('role_admitted', worker=worker, **result)
    return result


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if '--' not in argv:
        raise SystemExit('usage: join.py --session S --worker K -- TRAINER COMMAND')
    split = argv.index('--')
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--session', required=True)
    parser.add_argument('--worker', type=int, required=True)
    args = parser.parse_args(argv[:split])
    command = argv[split + 1:]
    if TRAINER_MODULE not in command:
        raise SystemExit(f'join: the wrapped command must run {TRAINER_MODULE}')
    trainer = command[command.index(TRAINER_MODULE) + 1:]
    session = json.loads(Path(args.session).read_text())
    if session.get('schema') != SESSION_SCHEMA or not 0 <= args.worker < session['workers']:
        raise SystemExit('join: not an elastic DiLoCo session or worker outside its slots')
    block = json.loads(flag(trainer, '--diloco'))
    if 'worker_id' in block or block.get('run_id') != session['run_id'] or block.get('workers') != session['workers'] \
            or block.get('coordinator_url') != f'http://127.0.0.1:{session["relay_port"]}':
        raise SystemExit('join: the trainer --diloco block is not this session\'s common block')
    # The worker slot is the only per-worker setting; it is not part of the coordinator identity.
    os.environ['NATLANG_DILOCO_WORKER_ID'] = str(args.worker)
    secrets = os.environ.get('NATLANG_CW_SECRETS')
    if not secrets:
        raise SystemExit('join: $NATLANG_CW_SECRETS is not set')
    out = Path(flag(trainer, '--out'))
    state_dir = Path(session.get('state_root', '/work/diloco')) / session['run_id']
    state_dir.mkdir(parents=True, exist_ok=True)
    log = Log(out / 'diloco-join.jsonl')
    log('join_start', worker=args.worker, run_id=session['run_id'], pid=os.getpid())
    wait_for(Path(secrets) / 'diloco-token', log, 'bearer token')
    cohost = args.worker == session['cohost_worker']
    if cohost and not (Path(secrets) / 'diloco-url').is_file():
        (Path(secrets) / 'diloco-url').write_text(f'http://127.0.0.1:{session["coordinator"]["port"]}\n')
        (Path(secrets) / 'diloco-gateway').write_text('')
    wait_for(Path(secrets) / 'diloco-url', log, 'coordinator URL')
    ensure_relay(session, state_dir, secrets, log)
    started = time.monotonic()
    item, contract, namespace, settings = compute_role(trainer, session['workers'])
    log('role_computed', seconds=round(time.monotonic() - started, 1), namespace=namespace,
        parent_sha256=item['identity']['parent_checkpoint_sha256'])
    if cohost:
        ensure_coordinator(session, contract, item, namespace, settings, Path(flag(trainer, '--continue-from')),
                           secrets, state_dir, log)
    admit(session, args.worker, item, namespace, log)
    log('exec_trainer')
    os.execv(command[0], command)


if __name__ == '__main__':
    main()
