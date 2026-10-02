#!/usr/bin/env python3
"""Run a reviewed standalone generation bundle on its assigned remote host."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    root = args.plan.resolve().parent
    with (root / 'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def status(state, **extra):
            row = {'state': state, 'updated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   'pid': os.getpid(), 'model': plan['model'], 'model_revision': plan['model_revision'], **extra}
            tmp = root / 'worker-status.json.tmp'
            tmp.write_text(json.dumps(row, indent=2) + '\n')
            tmp.replace(root / 'worker-status.json')
        def verify():
            if plan.get('approved') is not True:
                raise ValueError('generation bundle lacks root approval')
            for path, sha in plan['pins'].items():
                if hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha:
                    raise ValueError('reviewed artifact changed: ' + path)
        verify()
        runtime = Path(plan['runtime'])
        frozen = json.loads((runtime / 'frozen-runtime.json').read_text())
        for path, sha in frozen['files'].items():
            if hashlib.sha256((runtime / path).read_bytes()).hexdigest() != sha:
                raise ValueError('frozen runtime changed: ' + path)
        while True:
            result = subprocess.run(['docker', 'inspect', plan['download_container'], '--format',
                                     '{{.State.Status}} {{.State.ExitCode}}'], capture_output=True, text=True)
            if result.returncode:
                status('paused_missing_download_container')
                return
            state, code = result.stdout.strip().split()
            if state == 'exited':
                if code != '0':
                    status('paused_download_failed', download_exit_code=int(code))
                    return
                break
            status('waiting_for_model_download', download_container=plan['download_container'])
            time.sleep(30)
        model_path = Path(plan['model_path'])
        index = json.loads((model_path / 'model.safetensors.index.json').read_text())
        for name in set(index['weight_map'].values()):
            if not (model_path / name).is_file():
                status('paused_incomplete_model', missing_file=name)
                return
        expected_files = json.loads(Path(plan['model_metadata']).read_text())['files']
        for item in expected_files:
            name = item['rfilename']
            if not name.endswith('.safetensors'):
                continue
            file = model_path / name
            status('verifying_model_files', file=name)
            if not file.is_file() or file.stat().st_size != item['size']:
                status('paused_model_size_mismatch', file=name)
                return
            expected_sha = (item.get('lfs') or {}).get('sha256')
            if expected_sha:
                digest = hashlib.sha256()
                with file.open('rb') as source:
                    for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
                        digest.update(block)
                if digest.hexdigest() != expected_sha:
                    status('paused_model_hash_mismatch', file=name)
                    return
        status('starting_model_server')
        existing = subprocess.run(['docker', 'inspect', plan['server_container'], '--format', '{{.State.Status}}'],
                                  capture_output=True, text=True)
        if existing.returncode == 0:
            if existing.stdout.strip() != 'running':
                status('paused_existing_stopped_server', container=plan['server_container'])
                return
        else:
            with (root / 'server-launch.log').open('a') as log:
                launched = subprocess.run(plan['server_command'], stdout=log, stderr=subprocess.STDOUT)
            if launched.returncode:
                status('paused_server_launch_failed', exit_code=launched.returncode)
                return
        while True:
            try:
                with urllib.request.urlopen(plan['endpoint'] + '/health', timeout=10) as response:
                    if response.status == 200:
                        break
            except Exception:
                pass
            info = subprocess.run(['docker', 'inspect', plan['server_container'], '--format', '{{.State.Status}}'],
                                  capture_output=True, text=True)
            if info.returncode or info.stdout.strip() != 'running':
                status('paused_server_exited', container=plan['server_container'])
                return
            status('waiting_for_model_server', container=plan['server_container'])
            time.sleep(15)
        verify()
        if plan.get('template_preflight'):
            # Tokenization exercises the actual deployed chat template without inference.
            # Include synthetic tool histories, not only a single plain user message.
            payload = Path(plan['template_preflight']).read_bytes()
            request = urllib.request.Request(plan['endpoint'] + '/tokenize', data=payload,
                                             headers={'Content-Type': 'application/json'})
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    rendered = json.load(response)
                if not isinstance(rendered.get('count'), int) or rendered['count'] < 1:
                    raise ValueError('tokenization did not return a positive token count')
            except Exception as error:
                status('paused_template_preflight_failed', error_type=type(error).__name__)
                return
            (root / 'template-preflight-receipt.json').write_text(json.dumps({
                'checked_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'request_sha256': hashlib.sha256(payload).hexdigest(),
                'token_count': rendered['count'], 'model_calls': 0,
            }, indent=2) + '\n')
        children = []
        with (root / 'supervisor.log').open('a') as log:
            for command in plan.get('worker_commands', [plan.get('worker_command')]):
                children.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                    env={**os.environ, 'PATH': plan['node_directory'] + ':' + os.environ.get('PATH', '')}))
            def stop(signum, frame):
                for child in children:
                    if child.poll() is None:
                        child.terminate()
            signal.signal(signal.SIGTERM, stop)
            signal.signal(signal.SIGINT, stop)
            while any(child.poll() is None for child in children):
                failed = [child for child in children if child.poll() not in (None, 0)]
                if failed:
                    stop(None, None)
                    for child in children:
                        child.wait()
                    status('paused_supervisor_failed', supervisors=[{'pid': child.pid, 'exit_code': child.returncode} for child in children])
                    return
                server = subprocess.run(['docker', 'inspect', plan['server_container'], '--format', '{{.State.Status}}'],
                                        capture_output=True, text=True)
                if server.returncode or server.stdout.strip() != 'running':
                    stop(None, None)
                    for child in children:
                        child.wait()
                    status('paused_server_exited', supervisor_pids=[child.pid for child in children])
                    return
                status('running', supervisors=[{'pid': child.pid, 'exit_code': child.poll()} for child in children],
                       cases=plan['cases'], model_concurrency=plan['model_concurrency'])
                time.sleep(15)
            paused = []
            for name in plan.get('journals', [plan.get('journal')]):
                journal = Path(name)
                last = json.loads(journal.read_text().splitlines()[-1]) if journal.exists() and journal.stat().st_size else {}
                if last.get('event') == 'pause':
                    paused.append({'journal': name, 'reason': last.get('reason')})
            codes = [child.returncode for child in children]
            state = 'paused' if paused else ('finished' if all(code == 0 for code in codes) else 'stopped')
            status(state, supervisor_pids=[child.pid for child in children], exit_codes=codes, pauses=paused)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
