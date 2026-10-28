#!/usr/bin/env python3
"""Launch one DiLoCo stage: by default the coordinator shares worker 0's GPU sandbox.

Every role is an ordinary cwctl job (job.py, transport "api"). The run file names two job templates:

    {"schema": "natlang.coreweave-diloco/1", "id": "lfm-diloco-20261028",
     "settings": {"run_id": "lfm-diloco-20261028", "workers": 4, "local_steps": 64},
     "coordinator": {"job": "JOB.json", "bundle_file": "BUNDLE.json", "parent": "/work/in/parent.pt",
                     "args": ["--quorum", "3", "--outer-lr", "0.7"]},
     "learners": {"job": "JOB.json", "bundle_file": "BUNDLE.json"}}

The learner template's command is the ordinary trainer, ``python -m natlang_neuralese.train.native_state_sft
ARGS``, continuing from the same parent. Each learner gets ``--diloco`` with the coordinator URL and its
learner ID (NATLANG_DILOCO_WORKER_ID). The coordinator template supplies the image, resources and the bundle
holding the parent checkpoint; its command is replaced by ``native_diloco init`` and ``diloco serve``. A fresh
bearer token and the coordinator's share token reach the sandboxes only as secret files.

    diloco.py launch RUN.json     start the coordinator, then the learners (stops all on any failure)
    diloco.py status RUN_ID       the coordinator's merge status
    diloco.py attach-eval RUN_ID JOB.json BUNDLE.json   an eval sandbox with the coordinator URL and secrets

Cohosted mode uses the learner bundle for both roles; it must include the exact coordinator parent and code.
Worker 0 uses loopback, all others use the share-token gateway. Its coordinator remains live after worker 0 finishes.
Collect the worker outputs and coordinator root before stopping the shared sandbox once; the run journal under
``~/.local/state/natlang/coreweave/diloco/RUN_ID.json`` records each sandbox's owned roles and current launch state. Replacing
worker 0 also replaces the coordinator: restore its durable root and worker 0's full state deliberately; this launcher
does not implement automatic failover. If an SDK launch fails before returning its sandbox ID, reconcile the cwctl
launch event journal before any replacement; the per-run journal cannot yet contain that ID. To opt into a separate CPU sandbox, set ``"placement": "separate"``.
For either placement, collect all learners and the coordinator root while the sandbox is live; then stop it and run
``python -m natlang_neuralese.train.native_diloco export`` with every learner checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import secrets
import shlex
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cwctl  # noqa: E402
import job as jobs  # noqa: E402

SCHEMA = 'natlang.coreweave-diloco/1'
PORT = 8080
TRAINER = ['python', '-m', 'natlang_neuralese.train.native_state_sft']
RUNS = cwctl.STATE / 'diloco'


def validate(run: dict) -> dict:
    if run.get('schema') != SCHEMA:
        raise cwctl.ControllerError(f'a DiLoCo run file has "schema": "{SCHEMA}"')
    for key in ('id', 'settings', 'coordinator', 'learners'):
        if key not in run:
            raise cwctl.ControllerError(f'run file lacks {key!r}')
    if not {'job', 'bundle_file', 'parent'} <= set(run['coordinator']) or not {'job', 'bundle_file'} <= set(run['learners']):
        raise cwctl.ControllerError('coordinator needs job, bundle_file and parent; learners need job and bundle_file')
    if run.get('placement', 'cohosted') not in ('cohosted', 'separate'):
        raise cwctl.ControllerError('placement must be cohosted or separate')
    prepare_parent = run['coordinator'].get('prepare_parent')
    if prepare_parent is not None:
        if run.get('placement', 'cohosted') != 'cohosted':
            raise cwctl.ControllerError('prepare_parent requires cohosted placement')
        if not isinstance(prepare_parent, list) or not prepare_parent or not all(isinstance(x, str) and x for x in prepare_parent):
            raise cwctl.ControllerError('prepare_parent must be a nonempty argv list')
    run = dict(run)
    resident = run.setdefault('resident', {'directory': '/work/diloco-resident/resident-data',
                                           'geometry_qualification': '/work/diloco-resident/geometry.json'})
    if resident != 'off' and (not isinstance(resident, dict) or set(resident) != {'directory', 'geometry_qualification'} or
            not all(isinstance(v, str) and v.startswith('/') for v in resident.values())):
        raise cwctl.ControllerError('resident needs absolute directory and geometry_qualification paths')
    if 'coordinator_url' in run['settings']:
        raise cwctl.ControllerError('settings.coordinator_url is filled in by the launch')
    return run


def _template(path: str) -> dict:
    return jobs.validate(json.loads(Path(path).read_text()))


def coordinator_job(run: dict, trainer_args: list[str]) -> dict:
    job = copy.deepcopy(_template(run['coordinator']['job']))
    root = f"{job['scratch']['path']}/diloco-coordinator"
    init = ['python', '-m', 'natlang_neuralese.train.native_diloco', 'init', '--root', root,
            '--parent', run['coordinator']['parent'], '--trainer-argv', json.dumps(trainer_args),
            '--settings', json.dumps(run['settings']), *run['coordinator'].get('args', [])]
    serve = ['python', '-m', 'natlang_neuralese.train.diloco', 'serve', '--root', root, '--port', str(PORT)]
    job.update(id=f"{run['id']}-coordinator", service={'port': PORT}, outputs=[root],
               command=['bash', '-c', f'{shlex.join(init)} && exec {shlex.join(serve)}'])
    return jobs.validate(job)


def trainer_command(run: dict, job: dict) -> list[str]:
    """Actual trainer command before the restored-volume verification wrapper."""
    command = list(job['command'])
    if isinstance(run.get('resident'), dict):
        if any(flag in command for flag in ('--resident-data-directory', '--resident-geometry-qualification')):
            raise cwctl.ControllerError('declare resident paths in the run, not twice in the trainer command')
        command += ['--resident-data-directory', run['resident']['directory'],
                    '--resident-geometry-qualification', run['resident']['geometry_qualification']]
    return command


def learner_job(run: dict, worker: int, url: str) -> dict:
    job = copy.deepcopy(_template(run['learners']['job']))
    settings = {**run['settings'], 'coordinator_url': url}
    ident = f"{run['id']}-learner-{worker}"
    environment = {**job['env'], 'NATLANG_DILOCO_WORKER_ID': str(worker)}
    if job['scratch'].get('restore_from_snapshot_id') or run.get('parent_transfer'):
        environment['NATLANG_CW_INSTANCE'] = ident
    command = [*trainer_command(run, job), '--diloco', json.dumps(settings, sort_keys=True)]
    expected = environment.get('NATLANG_DILOCO_SEED_SHA256')
    if (job['scratch'].get('restore_from_snapshot_id') or run.get('parent_transfer')) and expected:
        import re
        if not re.fullmatch('[0-9a-f]{64}', expected):
            raise cwctl.ControllerError('restored seed needs its exact SHA-256')
        parent = run['coordinator']['parent']
        roots = []
        if '--out' in command:
            roots.append(command[command.index('--out') + 1])
        if isinstance(run.get('resident'), dict):
            roots.append(run['resident']['directory'])
        work = job['scratch']['path'].rstrip('/')
        if any(not path.startswith(work + '/') or os.path.normpath(path) != path for path in roots):
            raise cwctl.ControllerError('restored worker state directories must be canonical scratch paths')
        verify = r'''import json,os,pathlib,sys
from natlang_neuralese.common.hashing import sha256_file_hex
p,want,instance,roots=sys.argv[1:]; roots=json.loads(roots)
r=json.load(open(p+'.seed.json'))
actual=sha256_file_hex(p)
assert actual==want==r['sha256'], 'restored seed hash mismatch'
assert r['global_step']>=r['requested_stop']>0 and r['full_optimizer_rng_exposure']
control=pathlib.Path(os.environ.get('NATLANG_CW_WORK','/work'))/'.coreweave/instances'/instance
marker=control/'clone-sanitized.json'; expected={'roots':roots,'seed_sha256':want}
if marker.exists():
 assert json.loads(marker.read_text())==expected, 'restored state quarantine identity changed'
else:
 quarantine=control/'inherited'; quarantine.mkdir(parents=True,exist_ok=True)
 for i,name in enumerate(roots):
  source=pathlib.Path(name); target=quarantine/str(i)
  if source.is_symlink(): raise ValueError('inherited state root is a symlink')
  if source.exists():
   if target.exists(): raise ValueError('inherited state quarantine already exists')
   os.rename(source,target)
 with marker.open('x') as f:
  json.dump(expected,f,sort_keys=True); f.flush(); os.fsync(f.fileno())
'''
        fetch = shlex.join(['python3','-m','scripts.coreweave.diloco_fetch_parent',parent]) + ' && ' if run.get('parent_transfer') else ''
        command = ['bash', '-c', f'{fetch}{shlex.join(["python3", "-c", verify, parent, expected, ident, json.dumps(roots)])} && exec {shlex.join(command)}']
    job.update(id=ident, env=environment, command=command)
    return jobs.validate(job)


def cohosted_job(run: dict, trainer_args: list[str]) -> dict:
    """Worker 0 owns the GPU; the coordinator owns CPU and its own root on the same sandbox."""
    job = learner_job(run, 0, f'http://127.0.0.1:{PORT}')
    root = f"{job['scratch']['path']}/diloco-coordinator"
    init = ['python', '-m', 'natlang_neuralese.train.native_diloco', 'init', '--root', root,
            '--parent', run['coordinator']['parent'], '--trainer-argv', json.dumps(trainer_args),
            '--settings', json.dumps(run['settings']), *run['coordinator'].get('args', [])]
    serve = ['python', '-m', 'natlang_neuralese.train.diloco', 'serve', '--root', root, '--port', str(PORT)]
    logs = f"{job['scratch']['path']}/diloco-cohost-logs"
    # The authenticated status probe confirms HTTP readiness, not merely a listening socket.
    probe = ("import urllib.request; from natlang_neuralese.train.diloco.coordinator import credential, TOKEN_ENV, TOKEN_SECRET; "
             f"r=urllib.request.Request('http://127.0.0.1:{PORT}/v1/status', "
             "headers={'Authorization':'Bearer '+credential(TOKEN_ENV,TOKEN_SECRET)}); "
             "urllib.request.urlopen(r,timeout=2).read()")
    prepare = run['coordinator'].get('prepare_parent')
    fetch = (shlex.join(['python3','-m','scripts.coreweave.diloco_fetch_parent',run['coordinator']['parent']]) + '\n') if run.get('parent_transfer') else ''
    seed = (shlex.join(prepare) + ' >' + shlex.quote(logs + '/prepare-parent.log') + ' 2>&1 &\n'
            "seed_pid=$!\nwait \"$seed_pid\"\nseed_pid=''") if prepare else ':'
    script = f"""set -e
mkdir -p {shlex.quote(logs)}
seed_pid=''
coord=''
worker=''
cleanup() {{
  if [ -n "$seed_pid" ]; then kill "$seed_pid" 2>/dev/null || true; wait "$seed_pid" 2>/dev/null || true; fi
  if [ -n "$worker" ]; then kill "$worker" 2>/dev/null || true; wait "$worker" 2>/dev/null || true; fi
  if [ -n "$coord" ]; then kill "$coord" 2>/dev/null || true; wait "$coord" 2>/dev/null || true; fi
}}
trap cleanup EXIT
trap 'exit 143' TERM INT
{fetch}{seed}
CUDA_VISIBLE_DEVICES='' {shlex.join(init)} >{shlex.quote(logs + '/init.log')} 2>&1
CUDA_VISIBLE_DEVICES='' {shlex.join(serve)} >{shlex.quote(logs + '/server.log')} 2>&1 &
coord=$!
ready=0
for i in $(seq 1 300); do
  if python -c {shlex.quote(probe)} >/dev/null 2>&1; then ready=1; break; fi
  kill -0 "$coord" 2>/dev/null || {{ echo 'coordinator exited before readiness' >&2; exit 1; }}
  sleep 1
done
[ "$ready" = 1 ] || {{ echo 'coordinator HTTP readiness timed out' >&2; exit 1; }}
{shlex.join(job['command'])} >{shlex.quote(logs + '/worker-0.log')} 2>&1 &
worker=$!
set +e
wait -n -p finished "$coord" "$worker"
result=$?
if [ "$finished" = "$coord" ]; then echo 'coordinator exited while worker ran' >&2; exit 1; fi
worker=''
[ "$result" = 0 ] || exit "$result"
echo 'worker 0 complete; coordinator remains available for other learners'
wait "$coord"
exit $?
"""
    seed_outputs = ([run['coordinator']['parent'], run['coordinator']['parent'] + '.seed.json'] if prepare else [])
    resources=copy.deepcopy(run['coordinator'].get('cohost_resources',job['resources']))
    if not isinstance(resources,dict) or resources.get('gpu')!=1 or set(resources)-{'gpu','cpu','memory'}:
        raise cwctl.ControllerError('cohost resources must request one GPU and optional CPU/memory only')
    job.update(id=f"{run['id']}-learner-0", resources=resources,
               service={'port': PORT}, outputs=list(dict.fromkeys([*job.get('outputs', []), root, logs, *seed_outputs])),
               command=['bash', '-c', script])
    return jobs.validate(job)


def _keep(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, 'w') as stream:
        stream.write(value)


def wait_coordinator(url: str, token: str, gateway: bytes, timeout: float = 600, sandbox_id=None) -> None:
    """Wait for authenticated coordinator HTTP readiness before starting other workers."""
    deadline = time.monotonic() + timeout
    error = None
    next_status = 0.0
    while time.monotonic() < deadline:
        try:
            request = urllib.request.Request(url.rstrip('/') + '/v1/status', headers={
                'Authorization': 'Bearer ' + token, 'X-Sandbox-Share-Token': gateway.decode()})
            with urllib.request.urlopen(request, timeout=10) as response:
                status = json.load(response)
            if not isinstance(status, dict):
                raise ValueError('coordinator status is not an object')
            return
        except (OSError, ValueError) as exc:
            error = exc
            if sandbox_id and time.monotonic() >= next_status:
                try:
                    sandbox = cwctl.read_sandbox(sandbox_id, patience=1)
                    state = str(getattr(sandbox.status, 'value', sandbox.status)).lower()
                    if state in cwctl.TERMINAL:
                        raise cwctl.ControllerError(f'coordinator sandbox {sandbox_id} ended during startup: {state}')
                except cwctl.ControllerError:
                    raise
                except Exception:
                    pass  # transient provider status is not evidence that the sandbox stopped
                next_status = time.monotonic() + 30
            time.sleep(2)
    raise cwctl.ControllerError(f'coordinator HTTP readiness timed out: {error}')


# Learners boot concurrently: placed one at a time with --wait, sandboxes came up 3-6 minutes apart and learner 0 ran
# its shard alone before a quorum existed. Creates stay serialized by cwctl.throttled_create (file lock), a few
# seconds apart unless NATLANG_CW_CREATE_INTERVAL says otherwise; input pushes and boots overlap.
LEARNER_CREATE_INTERVAL_SECONDS = 5.0



def parent_read_secret(run):
    """Issue a fresh scoped GET only for an independently verified exact private parent."""
    transfer = run.get('parent_transfer')
    if not transfer: return {}
    from private_export_guard import validate_receipt
    import requests, wandb
    from urllib.parse import urlparse
    import hashlib
    pin_path=Path(transfer['pin']);receipt_path=Path(transfer['receipt'])
    if (hashlib.sha256(pin_path.read_bytes()).hexdigest()!=transfer['pin_sha256']
            or hashlib.sha256(receipt_path.read_bytes()).hexdigest()!=transfer['receipt_sha256']):
        raise cwctl.ControllerError('private parent pin or receipt binding changed')
    pin=json.loads(pin_path.read_text());receipt=json.loads(receipt_path.read_text())
    identity={'sha256':pin['sha256'],'bytes':pin['bytes']}
    validate_receipt(receipt,identity)
    expected=_template(run['learners']['job'])['env'].get('NATLANG_DILOCO_SEED_SHA256')
    if (pin['path']!=run['coordinator']['parent'] or pin['sha256']!=expected
            or not pin.get('full_optimizer_rng_exposure') or not 0<pin['requested_stop']<=pin['global_step']):
        raise cwctl.ControllerError('private parent pin does not match run parent')
    remote=wandb.Api().run(receipt['run']).file(receipt['name'])
    if remote.size!=identity['bytes']:raise cwctl.ControllerError('private parent object size changed')
    url=remote.url
    if urlparse(url).hostname=='api.wandb.ai':
        response=requests.get(url,auth=('api',os.environ['WANDB_API_KEY']),stream=True,timeout=60)
        try:
            response.raise_for_status();url=response.url
        finally:response.close()
    parsed=urlparse(url)
    if parsed.scheme!='https' or parsed.hostname!='storage.googleapis.com' or parsed.username or parsed.password:
        raise cwctl.ControllerError('private parent scoped read capability is unavailable')
    cap={**pin,'url':url,**({'base':transfer['base']} if transfer.get('base') else {})}
    return {'parent-read':json.dumps(cap).encode()}

def launch_learners(run: dict, url: str, token: str, gateway: bytes, started: list[str], *, first_worker=0, on_started=None) -> list[str]:
    """Create every learner sandbox in its own thread; any failure stops all of them (the caller stops ``started``)."""
    import threading
    if 'NATLANG_CW_CREATE_INTERVAL' not in os.environ:
        cwctl.CREATE_INTERVAL_SECONDS = LEARNER_CREATE_INTERVAL_SECONDS
    workers = run['settings']['workers']
    results: dict[int, str] = {}
    errors: list[BaseException] = []
    lock = threading.Lock()

    def one(worker: int) -> None:
        try:
            entry = cwctl.launch(learner_job(run, worker, url), False, True, None, Path(run['learners']['bundle_file']),
                                 secrets={'diloco-token': token.encode(), 'diloco-gateway': gateway, **parent_read_secret(run)})
            with lock:
                started.append(entry['sandbox_id'])
                if on_started is not None:
                    on_started(worker, entry['sandbox_id'])
                if 'wait_error' in entry or 'push' not in entry:
                    raise cwctl.ControllerError(f'learner {worker} did not start: {entry}')
                results[worker] = entry['sandbox_id']
        except BaseException as error:  # noqa: BLE001 - reported and re-raised by the caller
            with lock:
                errors.append(error)

    threads = [threading.Thread(target=one, args=(worker,), name=f'learner-{worker}') for worker in range(first_worker, workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]
    return [results[worker] for worker in range(first_worker, workers)]


def launch(run: dict, *, coordinator_only=False) -> dict:
    run = validate(run)
    if coordinator_only and run.get('placement', 'cohosted') != 'cohosted':
        raise cwctl.ControllerError('coordinator-only pause requires cohosted placement')
    record_path = RUNS / f"{run['id']}.json"
    if record_path.exists():
        raise cwctl.ControllerError(f'{record_path} exists: a DiLoCo run id is never reused')
    learner_template = _template(run['learners']['job'])
    if learner_template['command'][:len(TRAINER)] != TRAINER:
        raise cwctl.ControllerError(f'the learner command must start with {shlex.join(TRAINER)}')
    trainer_args = trainer_command(run, learner_template)[len(TRAINER):]
    token = secrets.token_urlsafe(32)
    _keep(RUNS / run['id'] / 'token', token)
    started: list[str] = []
    placement = run.get('placement', 'cohosted')
    record = {'run': run, 'placement': placement, 'state': 'launching', 'coordinator': None,
              'learners': [None] * run['settings']['workers'], 'ownership': {}}

    def save_record():
        tmp = record_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(record, indent=1) + '\n')
        os.replace(tmp, record_path)

    def assigned(worker, sandbox_id):
        record['learners'][worker] = sandbox_id
        record['ownership'][sandbox_id] = [f'learner-{worker}']
        save_record()

    save_record()
    try:
        cohosted = placement == 'cohosted'
        job = cohosted_job(run, trainer_args) if cohosted else coordinator_job(run, trainer_args)
        bundle = run['learners']['bundle_file'] if cohosted else run['coordinator']['bundle_file']
        coordinator = cwctl.launch(job, False, True, None, Path(bundle), secrets={'diloco-token': token.encode(), **parent_read_secret(run)})
        started.append(coordinator['sandbox_id'])
        record['coordinator'] = coordinator['sandbox_id']
        record['ownership'][coordinator['sandbox_id']] = ['coordinator', 'learner-0'] if cohosted else ['coordinator']
        if cohosted:
            record['learners'][0] = coordinator['sandbox_id']
        save_record()
        if 'service_url' not in coordinator or 'wait_error' in coordinator or 'push' not in coordinator:
            raise cwctl.ControllerError(f"coordinator {coordinator['sandbox_id']} did not start: {coordinator}")
        record['url'] = coordinator['service_url']
        save_record()
        gateway = cwctl.share_token(coordinator['sandbox_id'])
        wait_coordinator(coordinator['service_url'], token, gateway,
                         timeout=21600 if run['coordinator'].get('prepare_parent') else 600,
                         sandbox_id=coordinator['sandbox_id'])
        record['state'] = 'coordinator-ready'
        save_record()
        if coordinator_only:
            if not cohosted:
                raise cwctl.ControllerError('coordinator-only pause requires cohosted placement')
            return record
        launch_learners(run, coordinator['service_url'], token, gateway, started,
                        first_worker=1 if cohosted else 0, on_started=assigned)
    except BaseException as error:
        record['state'] = 'failed-stopping'
        record['error'] = f'{type(error).__name__}: {error}'
        save_record()
        stop_errors = {}
        for sandbox_id in dict.fromkeys(started):
            try:
                result = cwctl.passthrough(['stop', sandbox_id])
                if result not in (None, 0):
                    stop_errors[sandbox_id] = f'stop returned {result}'
            except Exception as stop_error:
                stop_errors[sandbox_id] = f'{type(stop_error).__name__}: {stop_error}'
            record['stop_requests'] = {**record.get('stop_requests', {}), sandbox_id: stop_errors.get(sandbox_id, 'requested')}
            save_record()
        record['state'] = 'failed-stop-incomplete' if stop_errors else 'failed-stop-requested'
        save_record()
        raise
    record['state'] = 'running'
    save_record()
    cwctl.record({'event': 'diloco_launched', 'run': run['id'], 'coordinator': record['coordinator'],
                  'learners': record['learners'], 'placement': placement, 'ownership': record['ownership']})
    return record


def launch_remaining(run_id: str) -> dict:
    """Resume an explicitly paused cohosted launch after the seed workspace was secured."""
    path = RUNS / f'{run_id}.json'
    record = json.loads(path.read_text())
    run = record['run']
    if record.get('state') != 'coordinator-ready' or record.get('placement') != 'cohosted':
        raise cwctl.ControllerError('run is not a paused cohosted coordinator-ready launch')
    if any(record['learners'][1:]):
        raise cwctl.ControllerError('remaining learners already assigned; reconcile their actual sandboxes')
    token = (RUNS / run_id / 'token').read_text()
    gateway = cwctl.share_token(record['coordinator'])
    wait_coordinator(record['url'], token, gateway, sandbox_id=record['coordinator'])
    started = []
    def save():
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(record, indent=1) + '\n')
        os.replace(tmp, path)
    def assigned(worker, sandbox_id):
        record['learners'][worker] = sandbox_id
        record['ownership'][sandbox_id] = [f'learner-{worker}']
        save()
    try:
        launch_learners(run, record['url'], token, gateway, started, first_worker=1, on_started=assigned)
    except BaseException as error:
        record['state'] = 'remaining-launch-failed'
        record['error'] = f'{type(error).__name__}: {error}'
        save()
        raise
    record['state'] = 'running'
    save()
    cwctl.record({'event': 'diloco_learners_started', 'run': run_id, 'learners': record['learners']})
    return record


def attach_eval(run_id: str, job_path: str, bundle_file: str) -> dict:
    """Attach an evaluation sandbox to a running stage: it receives the coordinator URL (NATLANG_DILOCO_URL) and the
    learners' secrets, so it pulls global versions over the coordinator API instead of through the controller."""
    record_path = RUNS / f'{run_id}.json'
    record = json.loads(record_path.read_text())
    job = copy.deepcopy(_template(job_path))
    job['env'] = {**job.get('env', {}), 'NATLANG_DILOCO_URL': record['url']}
    entry = cwctl.launch(jobs.validate(job), False, True, None, Path(bundle_file),
                         secrets={'diloco-token': (RUNS / run_id / 'token').read_bytes(),
                                  'diloco-gateway': cwctl.share_token(record['coordinator'])})
    record.setdefault('evals', []).append({'sandbox_id': entry['sandbox_id'], 'job': job['id']})
    record_path.write_text(json.dumps(record, indent=1) + '\n')
    cwctl.record({'event': 'diloco_eval_attached', 'run': run_id, 'sandbox_id': entry['sandbox_id']})
    return entry


def status(run_id: str) -> dict:
    record = json.loads((RUNS / f'{run_id}.json').read_text())
    request = urllib.request.Request(record['url'].rstrip('/') + '/v1/status', headers={
        'Authorization': 'Bearer ' + (RUNS / run_id / 'token').read_text(),
        'X-Sandbox-Share-Token': cwctl.share_token(record['coordinator']).decode()})
    with urllib.request.urlopen(request, timeout=60) as response:
        return {'coordinator': record['coordinator'], 'learners': record['learners'], **json.loads(response.read())}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    launch_cmd = commands.add_parser('launch')
    launch_cmd.add_argument('run', type=Path)
    launch_cmd.add_argument('--coordinator-only', action='store_true')
    commands.add_parser('launch-remaining').add_argument('run_id')
    commands.add_parser('status').add_argument('run_id')
    attach = commands.add_parser('attach-eval')
    attach.add_argument('run_id')
    attach.add_argument('job')
    attach.add_argument('bundle_file')
    args = parser.parse_args(argv)
    cwctl.load_env_file()
    if args.command == 'launch':
        print(json.dumps(launch(json.loads(args.run.read_text()), coordinator_only=args.coordinator_only), indent=1))
    elif args.command == 'launch-remaining':
        print(json.dumps(launch_remaining(args.run_id), indent=1))
    elif args.command == 'attach-eval':
        print(json.dumps(attach_eval(args.run_id, args.job, args.bundle_file), indent=1))
    else:
        print(json.dumps(status(args.run_id), indent=1))
    return 0


if __name__ == '__main__':
    if not os.environ.get('NATLANG_CW_NO_REEXEC'):
        cwctl.reexec_in_venv()
    try:
        sys.exit(main())
    except (cwctl.ControllerError, jobs.JobError) as error:
        print(f'diloco: {error}', file=sys.stderr)
        sys.exit(2)
