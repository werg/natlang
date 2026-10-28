#!/usr/bin/env python3
"""Launch one DiLoCo stage on CoreWeave: a CPU coordinator sandbox and N one-GPU learner sandboxes.

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

After the learners finish, collect them (cwctl.py collect), stop the coordinator with a graceful SIGTERM
(``cwctl.py stop ID``: it merges what arrived last), collect its root and run
``python -m natlang_neuralese.train.native_diloco export`` to produce the continuation checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import secrets
import shlex
import sys
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


def learner_job(run: dict, worker: int, url: str) -> dict:
    job = copy.deepcopy(_template(run['learners']['job']))
    settings = {**run['settings'], 'coordinator_url': url}
    job.update(id=f"{run['id']}-learner-{worker}", env={**job['env'], 'NATLANG_DILOCO_WORKER_ID': str(worker)},
               command=[*job['command'], '--diloco', json.dumps(settings, sort_keys=True)])
    return jobs.validate(job)


def _keep(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, 'w') as stream:
        stream.write(value)


# Learners boot concurrently: placed one at a time with --wait, sandboxes came up 3-6 minutes apart and learner 0 ran
# its shard alone before a quorum existed. Creates stay serialized by cwctl.throttled_create (file lock), a few
# seconds apart unless NATLANG_CW_CREATE_INTERVAL says otherwise; input pushes and boots overlap.
LEARNER_CREATE_INTERVAL_SECONDS = 5.0


def launch_learners(run: dict, url: str, token: str, gateway: bytes, started: list[str]) -> list[str]:
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
                                 secrets={'diloco-token': token.encode(), 'diloco-gateway': gateway})
            with lock:
                started.append(entry['sandbox_id'])
                if 'wait_error' in entry or 'push' not in entry:
                    raise cwctl.ControllerError(f'learner {worker} did not start: {entry}')
                results[worker] = entry['sandbox_id']
        except BaseException as error:  # noqa: BLE001 - reported and re-raised by the caller
            with lock:
                errors.append(error)

    threads = [threading.Thread(target=one, args=(worker,), name=f'learner-{worker}') for worker in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]
    return [results[worker] for worker in range(workers)]


def launch(run: dict) -> dict:
    run = validate(run)
    record_path = RUNS / f"{run['id']}.json"
    if record_path.exists():
        raise cwctl.ControllerError(f'{record_path} exists: a DiLoCo run id is never reused')
    learner_template = _template(run['learners']['job'])
    if learner_template['command'][:len(TRAINER)] != TRAINER:
        raise cwctl.ControllerError(f'the learner command must start with {shlex.join(TRAINER)}')
    trainer_args = learner_template['command'][len(TRAINER):]
    token = secrets.token_urlsafe(32)
    _keep(RUNS / run['id'] / 'token', token)
    started: list[str] = []
    try:
        coordinator = cwctl.launch(coordinator_job(run, trainer_args), False, True, None,
                                   Path(run['coordinator']['bundle_file']), secrets={'diloco-token': token.encode()})
        started.append(coordinator['sandbox_id'])
        if 'service_url' not in coordinator:
            raise cwctl.ControllerError(f"coordinator {coordinator['sandbox_id']} did not start: {coordinator}")
        gateway = cwctl.share_token(coordinator['sandbox_id'])
        learners = launch_learners(run, coordinator['service_url'], token, gateway, started)
    except BaseException:
        for sandbox_id in started:
            cwctl.passthrough(['stop', sandbox_id])
        raise
    record = {'run': run, 'coordinator': coordinator['sandbox_id'], 'url': coordinator['service_url'],
              'learners': learners}
    record_path.write_text(json.dumps(record, indent=1) + '\n')
    cwctl.record({'event': 'diloco_launched', 'run': run['id'], 'coordinator': record['coordinator'],
                  'learners': learners})
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
    commands.add_parser('launch').add_argument('run', type=Path)
    commands.add_parser('status').add_argument('run_id')
    attach = commands.add_parser('attach-eval')
    attach.add_argument('run_id')
    attach.add_argument('job')
    attach.add_argument('bundle_file')
    args = parser.parse_args(argv)
    cwctl.load_env_file()
    if args.command == 'launch':
        print(json.dumps(launch(json.loads(args.run.read_text())), indent=1))
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
