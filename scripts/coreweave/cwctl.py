#!/usr/bin/env python3
"""natlang controller for CoreWeave serverless sandboxes (plans/COREWEAVE_SANDBOXES.md).

Crisp mechanism only: it launches a declared job (scripts/coreweave/job.py), records every sandbox it starts, and
passes list/get/exec/logs/stop/files/snapshots through to the official `cwsandbox` CLI so there is one client
implementation. Which job goes to CoreWeave is the placement policy's decision, recorded in the job file.

  cwctl.py doctor                         client, credentials (names only), bucket settings; no API call
  cwctl.py plan JOB.json [--gpu-hour-price P]   validated request + cost bound; no API call (the dry run)
  cwctl.py launch JOB.json [--bundle K] [--resume] [--dry-run] [--wait]   start the sandbox, record it
                                          (--resume: continue from <output prefix><dir>/latest.json)
  cwctl.py place JOB.json [--mode crisp|nl|shadow] [--write]   placement policy (applications/ops-placement)
  cwctl.py smoke [--gpu] [--dry-run]      a 15-minute sandbox that prints device facts; waits and checks the log
                                          (a passed GPU smoke is recorded and makes CoreWeave "ready" for placement)
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
KEY_NAMES = ('CWSANDBOX_API_KEY', 'COREWEAVE_API_TOKEN', 'WANDB_API_KEY')
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


def auth_strategy() -> str:
    """`coreweave_api_key` or `wandb` (CWSANDBOX_AUTH, the name the cwsandbox CLI reads; --auth sets it).

    Unset: a CoreWeave token if one is present, else W&B if WANDB_API_KEY is. W&B auth serves serverless sandboxes
    (billed through Forge); runner discovery and other admin calls need a CoreWeave token.
    """
    chosen = os.environ.get('CWSANDBOX_AUTH')
    if chosen:
        if chosen not in ('coreweave_api_key', 'wandb'):
            raise ControllerError(f'CWSANDBOX_AUTH is coreweave_api_key or wandb, got {chosen!r}')
        return chosen
    if os.environ.get('CWSANDBOX_API_KEY') or os.environ.get('COREWEAVE_API_TOKEN'):
        return 'coreweave_api_key'
    return 'wandb' if os.environ.get('WANDB_API_KEY') else 'coreweave_api_key'


def sdk_auth():
    from cwsandbox import AuthStrategy
    return AuthStrategy(auth_strategy())


def api_key_name() -> str | None:
    """Name of the credential in use; COREWEAVE_API_TOKEN alone is copied to CWSANDBOX_API_KEY for the SDK."""
    if auth_strategy() == 'wandb':
        return 'WANDB_API_KEY' if os.environ.get('WANDB_API_KEY') else None
    if os.environ.get('CWSANDBOX_API_KEY'):
        return 'CWSANDBOX_API_KEY'
    if os.environ.get('COREWEAVE_API_TOKEN'):
        os.environ['CWSANDBOX_API_KEY'] = os.environ['COREWEAVE_API_TOKEN']
        return 'COREWEAVE_API_TOKEN'
    return None


def require_key() -> None:
    if api_key_name() is None:
        raise ControllerError(f'no credential for auth {auth_strategy()}: export CWSANDBOX_API_KEY (CoreWeave API access '
                              'token with the Sandbox User role) or WANDB_API_KEY with CWSANDBOX_AUTH=wandb, or put '
                              f'it in {ENV_FILE} (chmod 600)')


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


def local_docker_command(request: dict, name: str, files_dir: Path, extra_env: dict[str, str] | None = None,
                         image: str | None = None) -> list[str]:
    """The `docker run` that stands in for a sandbox in the local rehearsal (rehearse.py): same image, command,
    environment and mounted files as `request`; container filesystem as scratch; host network for the local S3.

    Writes the mounted files under ``files_dir`` (mounted read-only). ``extra_env`` carries what CoreWeave would
    inject (the object-storage endpoint and credentials; fake ones for the stand-in).
    """
    command = ['docker', 'run', '--rm', '--name', name, '--network', 'host', '--ipc', 'host']
    resources = request.get('resources', {})
    if 'cpu' in resources:
        command += ['--cpus', str(jobs.cpu_cores(resources['cpu']))]
    if 'memory' in resources:
        command += ['--memory', f"{int(jobs.memory_gib(resources['memory']) * 1024)}m"]
    if resources.get('gpu'):
        command += ['--gpus', 'all']
    for index, (path, content) in enumerate(sorted(request.get('mounted_files', {}).items())):
        source = files_dir / f'mounted-{index}'
        source.write_text(content)
        command += ['-v', f'{source}:{path}:ro']
    for key, value in sorted({**request.get('environment_variables', {}), **(extra_env or {})}.items()):
        command += ['-e', f'{key}={value}']
    command += ['-e', f'CW_SANDBOX_ID=local-{name}', image or request['container_image'], request['command'],
                *request.get('args', [])]
    return command


def record(event: dict) -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / 'sandboxes.jsonl').open('a') as stream:
        stream.write(json.dumps({'at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **event}) + '\n')


def records() -> list[dict]:
    path = STATE / 'sandboxes.jsonl'
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# Create throttle (owner 2026-10-12: never hit CoreWeave's create rate limits; creates only for real jobs).
CREATE_INTERVAL_SECONDS = float(os.environ.get('NATLANG_CW_CREATE_INTERVAL', '45'))
CREATE_ATTEMPTS = int(os.environ.get('NATLANG_CW_CREATE_ATTEMPTS', '5'))
BACKOFF_BASE_SECONDS = 60.0
BACKOFF_CAP_SECONDS = 1800.0
RETRYABLE_REASONS = {'CWSANDBOX_SERVERLESS_CREATE_THROTTLED', 'CWSANDBOX_OCCUPANCY_QUOTA_EXCEEDED',
                     'CWSANDBOX_RUNNER_CAPACITY_EXHAUSTED', 'CWSANDBOX_RUNNER_OVERLOADED', 'CWSANDBOX_RUNNER_UNAVAILABLE',
                     'CWSANDBOX_BACKEND_UNAVAILABLE'}


def error_facts(error: BaseException) -> dict:
    """reason / metadata / retry delay of a cwsandbox error, and whether a create may be retried after backoff."""
    reason = getattr(error, 'reason', None)
    delay = getattr(error, 'retry_delay', None)
    text = f'{type(error).__name__}: {error}'
    retryable = (reason in RETRYABLE_REASONS or type(error).__name__ == 'SandboxResourceExhaustedError'
                 or 'RESOURCE_EXHAUSTED' in text or '429' in text)
    return {'reason': reason, 'metadata': dict(getattr(error, 'metadata', {}) or {}),
            'retry_delay_seconds': delay.total_seconds() if delay is not None else None, 'error': text[:500],
            'retryable': retryable}


def backoff_seconds(attempt: int, hint: float | None, rand=None) -> float:
    """Exponential backoff with full jitter, never below the server's retry hint nor the create interval."""
    import random
    rand = rand or random.random
    ceiling = min(BACKOFF_CAP_SECONDS, BACKOFF_BASE_SECONDS * 2 ** attempt)
    return max(hint or 0.0, CREATE_INTERVAL_SECONDS, ceiling * (0.5 + 0.5 * rand()))


def throttled_create(create, *, sleep=time.sleep, clock=time.time, attempts: int | None = None):
    """Run ``create()`` serialized across processes (a file lock), at least CREATE_INTERVAL_SECONDS after the
    previous create from this machine, retrying only retryable errors with backoff. Every refusal is recorded
    (an occupancy-quota error tells us the real concurrent limit: learned passively, never probed)."""
    import fcntl
    STATE.mkdir(parents=True, exist_ok=True)
    attempts = attempts or CREATE_ATTEMPTS
    with (STATE / 'create.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        stamp = STATE / 'last-create'
        for attempt in range(attempts):
            last = float(stamp.read_text()) if stamp.is_file() else 0.0
            wait = last + CREATE_INTERVAL_SECONDS - clock()
            if wait > 0:
                sleep(wait)
            stamp.write_text(str(clock()))
            try:
                return create()
            except Exception as error:
                facts = error_facts(error)
                record({'event': 'create_refused', 'attempt': attempt, **facts})
                if facts['reason'] == 'CWSANDBOX_OCCUPANCY_QUOTA_EXCEEDED':
                    record({'event': 'quota_observed', 'limit': facts['metadata'].get('limit'),
                            'current': facts['metadata'].get('current')})
                if not facts['retryable'] or attempt == attempts - 1:
                    raise
                sleep(backoff_seconds(attempt, facts['retry_delay_seconds']))
    raise AssertionError('unreachable')


def run_arguments(request: dict) -> list[str]:
    """`Sandbox.run(*args)` takes the command and its arguments positionally (the constructor takes command=/args=)."""
    return [request['command'], *request.get('args', [])]


def run_keywords(request: dict) -> dict:
    return {k: v for k, v in request.items() if k not in ('command', 'args')}


def launch(job: dict, dry_run: bool, wait: bool, gpu_hour_price: float | None, bundle_file: Path | None = None) -> dict:
    request = jobs.sandbox_request(job, bootstrap_text())
    summary = {'job': job['id'], 'kind': job['kind'], 'placement': job['placement'],
               'cost': jobs.cost_estimate(job, gpu_hour_price), 'request': describe_request(request)}
    if dry_run:
        return {'dry_run': True, **summary, 'wire_request': wire_request(request)}
    where = job['placement'].get('where')
    if where not in (None, 'coreweave'):
        raise ControllerError(f'the placement decision put this job on {where!r} ({job["placement"]["reason"]}); '
                              'run `cwctl.py place` again when the fleet changes')
    require_key()
    from cwsandbox import Sandbox
    auth = sdk_auth()
    created = time.time()
    sandbox = throttled_create(lambda: Sandbox.run(*run_arguments(request), auth=auth, **run_keywords(request)))
    created = max(created, time.time() - 5)  # the throttle may have waited before the create itself
    entry = {'event': 'launched', 'sandbox_id': sandbox.sandbox_id, 'job': job['id'], 'kind': job['kind'],
             'lifetime_hours': job['lifetime_hours'], 'resources': job['resources'], 'bucket': job.get('bucket'),
             'bundle': job.get('bundle'), 'reason': job['placement']['reason'], 'cost': summary['cost']}
    record(entry)
    if wait:
        # GPU sandboxes attach the GPU to a VM and pull the image (NGC PyTorch is ~20 GB): far beyond the client's
        # 300 s default. The first measured GPU start is in plans/COREWEAVE_SANDBOXES.md.
        try:
            sandbox.wait(timeout=float(os.environ.get('NATLANG_CW_START_TIMEOUT', '5400')))
        except Exception as error:  # a transient poll failure must not lose the record of a created sandbox
            entry['wait_error'] = f'{type(error).__name__}: {error}'
            record({'event': 'wait_failed', 'sandbox_id': entry['sandbox_id'], 'error': entry['wait_error']})
            return {**entry, 'cost': summary['cost']}
        entry['status'] = str(sandbox.status)
        entry['seconds_to_running'] = round(time.time() - created, 1)
        record({'event': 'running', 'sandbox_id': entry['sandbox_id'], 'seconds_to_running': entry['seconds_to_running']})
    if job['transport'] == 'api' and bundle_file is not None:
        if not wait:
            sandbox.wait(timeout=float(os.environ.get('NATLANG_CW_START_TIMEOUT', '5400')))
        entry['push'] = push_inputs(sandbox, job, bundle_file)
        record({'event': 'inputs_pushed', 'sandbox_id': entry['sandbox_id'], **entry['push']})
    return {**entry, 'cost': summary['cost']}


def place(job_path: Path, mode: str | None, model: str | None, write: bool) -> dict:
    """Run the placement policy for one job file; with ``write``, record the served decision in its placement."""
    import placement
    config = placement.settings()
    raw = json.loads(job_path.read_text())
    job = jobs.validate(raw)
    needs = placement.job_needs(job, config)
    fleet = {**placement.dgx_facts(job['kind']),
             **placement.coreweave_facts(config, records(), api_key_name() is not None,
                                         bool(job.get('bucket') or os.environ.get('NATLANG_CW_BUCKET')))}
    item = {'id': job['id'], 'job': needs, 'fleet': fleet, 'crisp': placement.crisp_decision(needs, fleet)}
    served = placement.decide([item], placement.mode_of(config, mode), config, model)[0]
    if write:
        raw['placement'] = {'where': served['where'], 'reason': served['reason'], 'served': served['served'],
                            'at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
        job_path.write_text(json.dumps(raw, indent=1) + '\n')
    return {'decision': served, 'needs': needs, 'fleet': fleet, 'crisp': item['crisp'], 'written': write}


SMOKE_PROBES = ('echo "== nvidia-smi"; nvidia-smi; nvidia-smi --query-gpu=name,memory.total,driver_version,'
                'compute_cap --format=csv; echo "== disk"; df -h /work; nproc; free -g; ')
# Egress in Python (urllib), so the probe needs nothing beyond the image's python.
SMOKE_EGRESS = ("import urllib.request as u\n"
                "for url in ['https://pypi.org/simple/boto3/', 'https://nodejs.org/dist/index.json',\n"
                "            'https://registry.npmjs.org/', 'https://huggingface.co', 'https://github.com',\n"
                "            'https://cwobject.com', 'https://api.wandb.ai']:\n"
                "    try:\n"
                "        print('egress', url, u.urlopen(url, timeout=15).status)\n"
                "    except Exception as e:\n"
                "        print('egress', url, type(e).__name__, getattr(e, 'code', ''), str(e)[:80])\n")


PUSH_CHUNK = 8 * 1024 * 1024


def _chunks(path: Path):
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(PUSH_CHUNK), b''):
            yield block


def push_inputs(sandbox, job: dict, manifest_path: Path, log=print) -> dict:
    """File-API transport: push bundle.json, the code snapshot and every entry without a public URL into the
    sandbox's inbox (named by sha256), then write the inputs-ready marker the bootstrap waits for."""
    manifest = json.loads(manifest_path.read_text())
    sources = json.loads(manifest_path.with_suffix('.sources.json').read_text())
    inbox = f"{job['scratch']['path']}/.coreweave/inbox"
    pushed, bytes_pushed = 0, 0
    items = [i for i in manifest['files'] if not i.get('url')] + ([manifest['code']] if manifest.get('code') else [])
    seen = set()
    for item in items:
        if item['sha256'] in seen:
            continue
        seen.add(item['sha256'])
        source = Path(sources[item['key']])
        started = time.time()
        sandbox.write_file_streaming(f"{inbox}/{item['sha256']}", _chunks(source), timeout_seconds=7200).result()
        pushed += 1
        bytes_pushed += item['bytes']
        log(json.dumps({'event': 'pushed', 'path': item.get('path', 'code'), 'bytes': item['bytes'],
                        'seconds': round(time.time() - started, 1)}))
    sandbox.write_file(f"{job['scratch']['path']}/.coreweave/bundle.json", manifest_path.read_bytes()).result()
    sandbox.write_file(f"{job['scratch']['path']}/.coreweave/inputs-ready", b'').result()
    return {'pushed': pushed, 'bytes': bytes_pushed}


def collect(sandbox_id: str, scratch: str = '/work', mapping: list[tuple[str, str]] = (), timeout: float = 86400,
            log=print) -> dict:
    """File-API transport: wait for the bootstrap's outputs-ready, pull every output file, verify its sha256 against
    the manifest the sandbox wrote, place it (``mapping`` remaps path prefixes), then release the sandbox."""
    import bundle
    from cwsandbox import Sandbox
    sandbox = Sandbox.from_id(sandbox_id, auth=sdk_auth()).result()
    base = f'{scratch}/.coreweave'
    deadline = time.time() + timeout
    while sandbox.exec(['test', '-f', f'{base}/outputs-ready']).result().returncode != 0:
        if time.time() > deadline:
            raise ControllerError(f'{sandbox_id}: no outputs-ready within {timeout} s')
        time.sleep(30)
    names = sandbox.exec(['sh', '-c', f'ls {base}/outputs/ 2>/dev/null; cat {base}/exit-status']).result().stdout.split()
    status = int(names.pop()) if names and names[-1].lstrip('-').isdigit() else None
    results = []
    for name in names:
        manifest = json.loads(sandbox.read_file(f'{base}/outputs/{name}').result())
        files = 0
        for item in manifest['files']:
            target = bundle.target_path(item['path'], Path('/'), list(mapping))
            target.parent.mkdir(parents=True, exist_ok=True)
            temp = target.with_name(f'.cw-fetch-{target.name}')
            with temp.open('wb') as out:
                for block in sandbox.read_file_streaming(item['path'], timeout_seconds=7200):
                    out.write(block)
            if bundle.sha256_file_hex(temp) != item['sha256']:
                temp.unlink()
                raise ControllerError(f'{item["path"]}: pulled bytes do not match the sandbox manifest')
            temp.replace(target)
            files += 1
        results.append({'directory': manifest.get('directory'), 'files': files, 'bytes': manifest['bytes'],
                        'bundle': manifest['id']})
        log(json.dumps({'event': 'collected', **results[-1]}))
    sandbox.write_file(f'{base}/outputs-collected', b'').result()
    entry = {'event': 'collected', 'sandbox_id': sandbox_id, 'status': status, 'outputs': results}
    record(entry)
    return entry


def smoke_job(gpu: bool, image: str | None = None) -> dict:
    command = (['sh', '-c', SMOKE_PROBES + 'python -c "$0"; python -c "$1"', SMOKE_EGRESS, SMOKE_CHECK] if gpu
               else ['sh', '-c', 'uname -a; nproc; free -g; df -h /work'])
    return {'schema': jobs.SCHEMA, 'id': 'smoke-gpu' if gpu else 'smoke-cpu', 'kind': 'smoke',
            'placement': {'reason': 'bring-up smoke: credentials, placement, image pull, GPU visibility'},
            'image': image or ('nvcr.io/nvidia/pytorch:26.09-py3' if gpu else 'python:3.11'),
            'resources': {'cpu': '4', 'memory': '16Gi', 'gpu': 1} if gpu else {'cpu': '1', 'memory': '2Gi'},
            'lifetime_hours': 2, 'scratch': {'path': '/work', 'size': '20Gi'},
            'env': {'NATLANG_CW_IMAGE_READY': '1'}, 'command': command}


def doctor() -> dict:
    facts = {'venv': str(VENV), 'venv_present': (VENV / 'bin' / 'python').is_file(), 'running_in_venv': in_venv(),
             'env_file': str(ENV_FILE), 'env_file_present': ENV_FILE.is_file(),
             'auth': auth_strategy(), 'credential': api_key_name(),
             'credential_names_checked': list(KEY_NAMES),
             'bucket': os.environ.get('NATLANG_CW_BUCKET'), 'region': os.environ.get('AWS_REGION')
             or os.environ.get('NATLANG_CW_REGION'), 'state': str(STATE)}
    for module in ('cwsandbox', 'boto3'):
        try:
            imported = __import__(module)
            facts[module] = getattr(imported, '__version__', 'installed')
        except ImportError:
            facts[module] = None
    return facts


def read_sandbox(sandbox_id: str, patience: float = 900):
    """Sandbox.from_id, retrying transient control-plane answers ("being reconciled with its runner", deadline
    exceeded) every 30 s for up to ``patience`` seconds. Reads only; creates never retry here."""
    from cwsandbox import Sandbox
    deadline = time.time() + patience
    while True:
        try:
            return Sandbox.from_id(sandbox_id, auth=sdk_auth()).result()
        except Exception as error:
            transient = type(error).__name__ in ('SandboxUnavailableError', 'SandboxRequestTimeoutError')
            if not transient or time.time() > deadline:
                raise
            time.sleep(30)


TERMINAL = ('completed', 'failed', 'terminated')


def check_smoke(sandbox_id: str, gpu: bool, timeout: float = 5400) -> dict:
    """Poll the smoke until it ends, then check its log; a passed GPU smoke is what makes CoreWeave "ready".

    Records each status change with its time, which is the start-up measurement (creating -> running)."""
    started = time.time()
    last = None
    transitions = []
    while True:
        sandbox = read_sandbox(sandbox_id)
        status = str(getattr(sandbox.status, 'value', sandbox.status))
        if status != last:
            transitions.append({'status': status, 'at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                                'runner_id': sandbox.runner_id})
            record({'event': 'status', 'sandbox_id': sandbox_id, **transitions[-1]})
            last = status
        if status in TERMINAL or time.time() - started > timeout:
            break
        time.sleep(30)
    completed = status in TERMINAL
    try:
        text = ''.join(sandbox.stream_logs())
    except Exception as error:
        text = f'(logs unavailable: {type(error).__name__}: {error})'
    ok = ('sm_120' in text and '"status":0' in text) if gpu else '"status":0' in text
    result = {'event': 'gpu_smoke_ok' if gpu and ok else 'smoke_ok' if ok else 'smoke_failed',
              'sandbox_id': sandbox_id, 'completed': completed, 'status': status, 'transitions': transitions,
              'bootstrap_starts': text.count('"bootstrap":"start"'), 'log': text[-12000:]}
    if not completed:
        sandbox.stop(graceful_shutdown_seconds=10, missing_ok=True).result()
        record({'event': 'stop', 'args': [sandbox_id], 'why': 'smoke did not complete within its timeout'})
    record({k: v for k, v in result.items() if k != 'log'})
    return result


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
    # The CLI reads CWSANDBOX_AUTH and the key from this environment; nothing is put on its command line.
    os.environ['CWSANDBOX_AUTH'] = auth_strategy()
    return subprocess.call([str(cli), *argv])


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ['--auth'] and len(argv) > 1:
        os.environ['CWSANDBOX_AUTH'], argv = argv[1], argv[2:]
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
        p.add_argument('--bundle-file', type=Path, help='local manifest from bundle.py plan: pushed through the file '
                       'API (transport "api", no bucket)')
        p.add_argument('--bucket', help='AI Object Storage bucket (default NATLANG_CW_BUCKET)')
        p.add_argument('--resume', action='store_true',
                       help='continue this job id from its latest published outputs (after expiry, stop or failure)')
        if name == 'launch':
            p.add_argument('--dry-run', action='store_true')
            p.add_argument('--wait', action='store_true', help='block until the sandbox is RUNNING')
    p = sub.add_parser('place', help='decide where the job runs (applications/ops-placement); --write records it')
    p.add_argument('job', type=Path)
    p.add_argument('--mode', choices=('crisp', 'nl', 'shadow'))
    p.add_argument('--model')
    p.add_argument('--write', action='store_true')
    p = sub.add_parser('smoke')
    p.add_argument('--gpu', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--attach', metavar='SANDBOX_ID', help='check an already launched smoke instead of creating one')
    p.add_argument('--image', help='image for the smoke (default NGC pytorch 26.09; a smaller one separates image pull '
                   'time from placement)')
    p = sub.add_parser('collect', help='pull and verify a file-API job\'s outputs, then release its sandbox')
    p.add_argument('sandbox_id')
    p.add_argument('--scratch', default='/work')
    p.add_argument('--map', action='append', default=[], help='OLD=NEW path prefix for where outputs land')
    sub.add_parser('records')
    sub.add_parser('status-line')
    args = parser.parse_args(argv)
    if args.command == 'doctor':
        print(json.dumps(doctor(), indent=1))
    elif args.command == 'plan':
        print(json.dumps(launch(load_job(args.job, args.bundle, args.bucket, args.resume), True, False, args.gpu_hour_price), indent=1))
    elif args.command == 'launch':
        print(json.dumps(launch(load_job(args.job, args.bundle, args.bucket, args.resume), args.dry_run, args.wait, args.gpu_hour_price, args.bundle_file), indent=1))
    elif args.command == 'collect':
        mapping = [tuple(m.split('=', 1)) for m in args.map]
        print(json.dumps(collect(args.sandbox_id, args.scratch, mapping), indent=1))
    elif args.command == 'place':
        print(json.dumps(place(args.job, args.mode, args.model, args.write), indent=1))
    elif args.command == 'smoke' and args.attach:
        print(json.dumps(check_smoke(args.attach, args.gpu), indent=1))
    elif args.command == 'smoke':
        result = launch(jobs.validate(smoke_job(args.gpu, args.image)), args.dry_run, not args.dry_run, None)
        print(json.dumps(result, indent=1))
        if not args.dry_run:
            print(json.dumps(check_smoke(result['sandbox_id'], args.gpu), indent=1))
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
