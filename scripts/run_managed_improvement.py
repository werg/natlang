#!/usr/bin/env python3
"""Run a student-improvement recipe with sequential, owned model services."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_training_pipeline import atomic_json, digest_file, process_identity


def recipe_digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def ready(url):
    try:
        with urlopen(url, timeout=2) as response:
            return 200 <= response.status < 300
    except (HTTPError, URLError, TimeoutError, OSError):
        return False


def validate_spec(spec, config, run):
    if spec.get('version') != 'natlang.model_swap/1':
        raise ValueError('unsupported model-swap config')
    if spec.get('pipeline_sha256') != recipe_digest(config) or Path(spec.get('run_directory', '')).resolve() != run:
        raise ValueError('model-swap config does not match this recipe and run')
    for role in ('student', 'teacher'):
        service = spec.get(role)
        if not isinstance(service, dict) or service.get('endpoint') != config.get('collection', {}).get(f'{role}_server'):
            raise ValueError(f'{role} service endpoint differs from the round recipe')
        if not isinstance(service.get('ready_url'), str) or not service['ready_url'].startswith(service['endpoint']):
            raise ValueError(f'{role} ready URL must be under its endpoint')
        if not isinstance(service.get('start'), list) or not service['start'] or not all(isinstance(x, str) for x in service['start']):
            raise ValueError(f'{role} start must be an argv array')
        if service.get('stop') is not None and (not isinstance(service['stop'], list) or
                not service['stop'] or not all(isinstance(x, str) for x in service['stop'])):
            raise ValueError(f'{role} stop must be an argv array')
        if not isinstance(service.get('env', {}), dict) or any(not isinstance(k, str) or not isinstance(v, str)
                                                               for k, v in service.get('env', {}).items()):
            raise ValueError(f'{role} env must map strings to strings')
        for path, wanted in service.get('artifacts', {}).items():
            if not Path(path).is_file() or digest_file(path) != wanted:
                raise ValueError(f'{role} serving artifact changed or is missing: {path}')


def stage_complete(run, name):
    path = run / 'pipeline-state.json'
    if not path.exists():
        return False
    return json.loads(path.read_text()).get('stages', {}).get(name, {}).get('status') == 'complete'


def run_pipeline_phase(recipe, run, until=None):
    command = [sys.executable, str(Path(__file__).resolve().parent / 'run_training_pipeline.py'),
               str(recipe), str(run), *(['--until', until] if until else [])]
    child = subprocess.Popen(command, start_new_session=True)
    try:
        return child.wait()
    except BaseException:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            child.wait()  # The pipeline/trainer decides when its emergency checkpoint is safe.
        raise


def stop_owned(run, spec, *, child=None, timeout=45):
    marker = run / 'model-swap-service.json'
    if not marker.exists():
        return
    ownership = json.loads(marker.read_text())
    if (ownership.get('pipeline_sha256') != spec['pipeline_sha256'] or
            ownership.get('service_plan_sha256') != recipe_digest(spec)):
        raise ValueError('owned-service journal belongs to a different recipe or service plan')
    role = ownership['role']
    service = spec[role]
    pid = ownership.get('pid')
    if service.get('stop'):
        subprocess.run(service['stop'], timeout=timeout, check=False,
                       env={**os.environ, **service.get('env', {})})
    if pid and process_identity(pid) == ownership.get('process_start'):
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    if child is not None and child.poll() is None:
        try:
            child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            raise RuntimeError(f'{role} service did not stop; journal retained')
    deadline = time.monotonic() + timeout
    while ready(service['ready_url']) and time.monotonic() < deadline:
        time.sleep(0.5)
    if ready(service['ready_url']):
        raise RuntimeError(f'{role} endpoint remains live; journal retained')
    marker.unlink()


def run_managed(recipe, run, spec_path):
    recipe, run, spec_path = Path(recipe).resolve(), Path(run).resolve(), Path(spec_path).resolve()
    config = json.loads(recipe.read_text())
    spec = json.loads(spec_path.read_text())
    validate_spec(spec, config, run)
    run.mkdir(parents=True, exist_ok=True)
    with (run / '.model-swap.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (run / 'model-swap-service.json').exists():
            stop_owned(run, spec)
        for role, boundary in (('student', 'build-hard-states'), ('teacher', 'combine-verified')):
            service = spec[role]
            if stage_complete(run, boundary):
                code = run_pipeline_phase(recipe, run, boundary)
                if code:
                    return code
                continue
            if ready(service['ready_url']):
                raise RuntimeError(f'{role} endpoint is already live but not owned by this run; refusing to stop or reuse it')
            log_path = run / f'{role}-service.log'
            atomic_json(run / 'model-swap-service.json', {'version': 'natlang.owned_service/1',
                        'pipeline_sha256': spec['pipeline_sha256'],
                        'service_plan_sha256': recipe_digest(spec), 'role': role, 'status': 'starting',
                        'start_argv': service['start'], 'stop_argv': service.get('stop'), 'log': str(log_path)})
            child = None
            try:
                with log_path.open('ab', buffering=0) as log:
                    child = subprocess.Popen(service['start'], cwd=config['repository'],
                                             env={**os.environ, **service.get('env', {})},
                                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                atomic_json(run / 'model-swap-service.json', {'version': 'natlang.owned_service/1',
                            'pipeline_sha256': spec['pipeline_sha256'],
                            'service_plan_sha256': recipe_digest(spec), 'role': role, 'status': 'running',
                            'pid': child.pid, 'process_start': process_identity(child.pid),
                            'start_argv': service['start'], 'stop_argv': service.get('stop'), 'log': str(log_path)})
                deadline = time.monotonic() + int(service.get('startup_timeout_seconds', 600))
                while not ready(service['ready_url']):
                    if child.poll() is not None:
                        raise RuntimeError(f'{role} service exited before readiness; see {log_path}')
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f'{role} service did not become ready; see {log_path}')
                    time.sleep(0.5)
                code = run_pipeline_phase(recipe, run, boundary)
            finally:
                stop_owned(run, spec, child=child)
            if code:
                return code
        return run_pipeline_phase(recipe, run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recipe', type=Path)
    parser.add_argument('run', type=Path)
    parser.add_argument('--services', required=True, type=Path)
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise InterruptedError(f'received signal {signum}')
    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        return run_managed(args.recipe, args.run, args.services)
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (KeyboardInterrupt, InterruptedError):
        sys.exit(130)
