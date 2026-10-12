#!/usr/bin/env python3
"""natlang controller for CoreWeave serverless sandboxes (plans/COREWEAVE_SANDBOXES.md).

Crisp mechanism only: it launches a declared job (scripts/coreweave/job.py), records every sandbox it starts, and
passes list/get/exec/logs/stop/files/snapshots through to the official `cwsandbox` CLI so there is one client
implementation. Which job goes to CoreWeave is the placement policy's decision, recorded in the job file.

  cwctl.py doctor                         client, credentials (names only), bucket settings; no API call
  cwctl.py plan JOB.json [--gpu-hour-price P]   validated request + cost bound; no API call (the dry run)
  cwctl.py launch JOB.json [--bundle K] [--resume] [--dry-run] [--wait]   start the sandbox, record it
                                          (--resume: continue from <output prefix><dir>/latest.json)
  cwctl.py smoke [--gpu] [--dry-run]      a 10-minute sandbox that prints nvidia-smi / torch device facts
  cwctl.py records                        sandboxes this machine launched (local state file)
  cwctl.py status-line                    one line for this machine's coord status page
  cwctl.py ls|get|exec|logs|stop|files|snapshots ...   passed to `cwsandbox` (tag filter: --tag natlang)

Credentials: CWSANDBOX_API_KEY (the documented name for a CoreWeave API access token) or COREWEAVE_API_TOKEN (the
name `cwic` reads), taken from the environment or from NATLANG_CW_ENV_FILE (default ~/.config/natlang/coreweave.env,
mode 600, never in the repo). Never printed, never passed into a sandbox. The client runs from the venv made by
scripts/coreweave/setup_client.sh (cwsandbox >= 1.14.2, boto3 >= 1.42.5); this script re-executes itself there.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shlex
import stat
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import job as jobs  # noqa: E402

VENV = Path(os.environ.get('NATLANG_CW_VENV', Path.home() / '.local/share/natlang/cw-venv'))
STATE = Path(os.environ.get('NATLANG_CW_STATE', Path.home() / '.local/state/natlang/coreweave'))
ENV_FILE = Path(os.environ.get('NATLANG_CW_ENV_FILE', Path.home() / '.config/natlang/coreweave.env'))
KEY_NAMES = ('CWSANDBOX_API_KEY', 'COREWEAVE_API_TOKEN')
PASSTHROUGH = ('ls', 'get', 'exec', 'logs', 'stop', 'files', 'snapshots', 'delete', 'sh')
SMOKE_CHECK = ("import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_arch_list()); "
               "print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), "
               "torch.cuda.mem_get_info()); x = torch.ones(1 << 20, device='cuda'); print((x + x).sum().item())")


class ControllerError(RuntimeError):
    pass


def load_env_file(path: Path = ENV_FILE) -> list[str]:
    """KEY=VALUE lines into os.environ (existing variables win). Refuses a group/world-readable file."""
    if not path.is_file():
        return []
    mode = path.stat().st_mode
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise ControllerError(f'{path} is readable by others; chmod 600 {path}')
    loaded = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        name, value = line.removeprefix('export ').split('=', 1)
        name, value = name.strip(), value.strip().strip('"').strip("'")
        if name not in os.environ:
            os.environ[name] = value
            loaded.append(name)
    return loaded


def api_key_name() -> str | None:
    """Name of the credential in use; COREWEAVE_API_TOKEN alone is copied to CWSANDBOX_API_KEY for the SDK."""
    if os.environ.get('CWSANDBOX_API_KEY'):
        return 'CWSANDBOX_API_KEY'
    if os.environ.get('COREWEAVE_API_TOKEN'):
        os.environ['CWSANDBOX_API_KEY'] = os.environ['COREWEAVE_API_TOKEN']
        return 'COREWEAVE_API_TOKEN'
    return None


def require_key() -> None:
    if api_key_name() is None:
        raise ControllerError('no credential: export CWSANDBOX_API_KEY (CoreWeave API access token with the Sandbox '
                              f'User role), or put it in {ENV_FILE} (chmod 600)')


def in_venv() -> bool:
    return Path(sys.prefix).resolve() == VENV.resolve()


def reexec_in_venv() -> None:
    python = VENV / 'bin' / 'python'
    if not in_venv() and python.is_file() and not os.environ.get('NATLANG_CW_NO_REEXEC'):
        os.execv(str(python), [str(python), *sys.argv])


def bootstrap_text() -> str:
    return (HERE / 'bootstrap.sh').read_text()


def load_job(path: Path, bundle: str | None = None, bucket: str | None = None, resume: bool = False) -> dict:
    """The job file, with the bundle key from the command line and bucket/region from the environment if unset."""
    try:
        job = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ControllerError(f'{path}: not JSON ({error})')
    if bundle:
        job['bundle'] = bundle
    if resume:
        job['resume'] = True
    job.setdefault('bucket', bucket or os.environ.get('NATLANG_CW_BUCKET'))
    if job['bucket'] is None:
        job.pop('bucket')
    region = os.environ.get('NATLANG_CW_REGION') or os.environ.get('AWS_REGION')
    if region and 'region' not in job:
        job['region'] = region
    return jobs.validate(job)


def describe_request(request: dict) -> dict:
    """The request as printed: the mounted bootstrap is summarized by size and sha256."""
    shown = json.loads(json.dumps(request))
    shown['mounted_files'] = {path: {'sha256': hashlib.sha256(content.encode()).hexdigest(),
                                     'bytes': len(content.encode())}
                              for path, content in request.get('mounted_files', {}).items()}
    return shown


def wire_request(request: dict) -> dict | str:
    """The CreateSandboxRequest the installed client would send, built offline (no network, no credential).

    Uses the client's private request builder, so it can break on a client upgrade; it exists because the docs and
    the client disagree in places (mounted_files keys, max_timeout_seconds), and only the client's own encoding is
    authoritative. Mounted file contents are replaced by their sha256.
    """
    try:
        from cwsandbox import Sandbox
        from google.protobuf.json_format import MessageToDict
        sandbox = Sandbox(**request)
        wire = MessageToDict(sandbox._build_create_request(request_id='dry-run', start_kwargs=dict(sandbox._start_kwargs)))
    except ImportError:
        return 'cwsandbox not installed: run scripts/coreweave/setup_client.sh'
    except Exception as error:  # a request the client itself rejects is the finding
        return f'client rejected the request: {type(error).__name__}: {error}'
    for container in wire.get('sandbox', {}).get('spec', {}).get('containers', []):
        for item in container.get('files', []):
            item['content'] = 'sha256:' + hashlib.sha256(base64.b64decode(item['content'])).hexdigest()
    return wire


def record(event: dict) -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / 'sandboxes.jsonl').open('a') as stream:
        stream.write(json.dumps({'at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **event}) + '\n')


def records() -> list[dict]:
    path = STATE / 'sandboxes.jsonl'
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def launch(job: dict, dry_run: bool, wait: bool, gpu_hour_price: float | None) -> dict:
    request = jobs.sandbox_request(job, bootstrap_text())
    summary = {'job': job['id'], 'kind': job['kind'], 'placement': job['placement'],
               'cost': jobs.cost_estimate(job, gpu_hour_price), 'request': describe_request(request)}
    if dry_run:
        return {'dry_run': True, **summary, 'wire_request': wire_request(request)}
    require_key()
    from cwsandbox import AuthStrategy, Sandbox
    sandbox = Sandbox.run(auth=AuthStrategy.COREWEAVE_API_KEY, **request)
    entry = {'event': 'launched', 'sandbox_id': sandbox.sandbox_id, 'job': job['id'], 'kind': job['kind'],
             'lifetime_hours': job['lifetime_hours'], 'resources': job['resources'], 'bucket': job.get('bucket'),
             'bundle': job.get('bundle'), 'reason': job['placement']['reason']}
    record(entry)
    if wait:
        sandbox.wait()
        entry['status'] = str(sandbox.status)
    return {**entry, 'cost': summary['cost']}


def smoke_job(gpu: bool) -> dict:
    command = ['python', '-c', SMOKE_CHECK] if gpu else ['sh', '-c', 'uname -a; nproc; free -g; df -h /work']
    return {'schema': jobs.SCHEMA, 'id': 'smoke-gpu' if gpu else 'smoke-cpu', 'kind': 'smoke',
            'placement': {'reason': 'bring-up smoke: credentials, placement, image pull, GPU visibility'},
            'image': 'nvcr.io/nvidia/pytorch:26.09-py3' if gpu else 'python:3.11',
            'resources': {'cpu': '4', 'memory': '16Gi', 'gpu': 1} if gpu else {'cpu': '1', 'memory': '2Gi'},
            'lifetime_hours': 0.25, 'scratch': {'path': '/work', 'size': '20Gi'},
            'env': {'NATLANG_CW_IMAGE_READY': '1'}, 'command': command}


def doctor() -> dict:
    facts = {'venv': str(VENV), 'venv_present': (VENV / 'bin' / 'python').is_file(), 'running_in_venv': in_venv(),
             'env_file': str(ENV_FILE), 'env_file_present': ENV_FILE.is_file(),
             'credential': api_key_name(), 'credential_names_checked': list(KEY_NAMES),
             'bucket': os.environ.get('NATLANG_CW_BUCKET'), 'region': os.environ.get('AWS_REGION')
             or os.environ.get('NATLANG_CW_REGION'), 'state': str(STATE)}
    for module in ('cwsandbox', 'boto3'):
        try:
            imported = __import__(module)
            facts[module] = getattr(imported, '__version__', 'installed')
        except ImportError:
            facts[module] = None
    return facts


def status_line() -> str:
    launched = [r for r in records() if r.get('event') == 'launched']
    if not launched:
        return 'coreweave: no sandboxes launched from this machine'
    last = launched[-1]
    return (f'coreweave: {len(launched)} sandboxes launched; last {last["sandbox_id"]} job {last["job"]} '
            f'({last["kind"]}, {last["lifetime_hours"]} h max) at {last["at"]}; live list: cwctl.py ls --tag natlang')


def passthrough(argv: list[str]) -> int:
    require_key()
    cli = VENV / 'bin' / 'cwsandbox'
    if not cli.is_file():
        raise ControllerError('cwsandbox CLI missing: run scripts/coreweave/setup_client.sh')
    if argv[0] == 'stop' and not any(a.startswith('--graceful-shutdown-seconds') for a in argv):
        # The trainers write their final state on SIGTERM; 10 s (the client default) is far too short for 47 GB.
        argv = [*argv, '--graceful-shutdown-seconds', os.environ.get('NATLANG_CW_STOP_GRACE', '600')]
    if argv[0] == 'stop':
        record({'event': 'stop', 'args': argv[1:]})
    return subprocess.call([str(cli), *argv])


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    load_env_file()
    if argv and argv[0] in PASSTHROUGH:
        return passthrough(argv)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('doctor')
    for name in ('plan', 'launch'):
        p = sub.add_parser(name)
        p.add_argument('job', type=Path)
        p.add_argument('--gpu-hour-price', type=float, help='USD per GPU-hour once CoreWeave quotes it')
        p.add_argument('--bundle', help='bundle manifest key, bundles/<id>.json (from bundle.py plan + upload)')
        p.add_argument('--bucket', help='AI Object Storage bucket (default NATLANG_CW_BUCKET)')
        p.add_argument('--resume', action='store_true',
                       help='continue this job id from its latest published outputs (after expiry, stop or failure)')
        if name == 'launch':
            p.add_argument('--dry-run', action='store_true')
            p.add_argument('--wait', action='store_true', help='block until the sandbox is RUNNING')
    p = sub.add_parser('smoke')
    p.add_argument('--gpu', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    sub.add_parser('records')
    sub.add_parser('status-line')
    args = parser.parse_args(argv)
    if args.command == 'doctor':
        print(json.dumps(doctor(), indent=1))
    elif args.command == 'plan':
        print(json.dumps(launch(load_job(args.job, args.bundle, args.bucket, args.resume), True, False, args.gpu_hour_price), indent=1))
    elif args.command == 'launch':
        print(json.dumps(launch(load_job(args.job, args.bundle, args.bucket, args.resume), args.dry_run, args.wait, args.gpu_hour_price), indent=1))
    elif args.command == 'smoke':
        result = launch(jobs.validate(smoke_job(args.gpu)), args.dry_run, not args.dry_run, None)
        print(json.dumps(result, indent=1))
        if not args.dry_run:
            print(f'logs: {shlex.quote(sys.argv[0])} logs {result["sandbox_id"]} --follow', file=sys.stderr)
    elif args.command == 'records':
        print(json.dumps(records(), indent=1))
    elif args.command == 'status-line':
        print(status_line())
    return 0


if __name__ == '__main__':
    if not os.environ.get('NATLANG_CW_NO_REEXEC'):
        reexec_in_venv()
    try:
        sys.exit(main())
    except (ControllerError, jobs.JobError) as error:
        print(f'cwctl: {error}', file=sys.stderr)
        sys.exit(2)
