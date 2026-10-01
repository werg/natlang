#!/usr/bin/env python3
"""Launch one reviewed OpenRouter queue. Keys remain outside plans, arguments and logs."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import urllib.error
import urllib.request

MODEL = 'stealth/space-bunny-alpha'

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def credential(path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError('credential file must be a regular file owned by this user with mode 600')
    lines = [line.strip() for line in path.read_text().splitlines()
             if line.strip() and not line.lstrip().startswith('#')]
    if len(lines) != 1 or not lines[0].startswith('OPENROUTER_API_KEY='):
        raise ValueError('expected exactly one OPENROUTER_API_KEY entry')
    value = lines[0].split('=', 1)[1].strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1]
    if not value or any(c.isspace() for c in value):
        raise ValueError('credential missing or malformed')
    return value

def get_json(url, key=None):
    headers = {'Authorization': 'Bearer ' + key} if key else {}
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        raise ValueError(f'OpenRouter metadata/authentication returned HTTP {error.code}') from None
    except Exception:
        raise ValueError('OpenRouter metadata/authentication unavailable') from None

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path)
    parser.add_argument('--key-file', type=Path, default=Path.home() / '.config/natlang/openrouter.env')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    root = args.plan.resolve().parent
    with (root / 'worker.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('this OpenRouter worker is already running') from None
        def status(state, **extra):
            value = {'state': state, 'model': MODEL, 'updated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                     'launcher_pid': os.getpid(), **extra}
            tmp = root / 'worker-status.json.tmp'
            tmp.write_text(json.dumps(value, indent=2) + '\n')
            tmp.replace(root / 'worker-status.json')
        if plan.get('approved') is not True or plan.get('model') != MODEL or plan.get('model_concurrency') != 1:
            raise ValueError('expected a reviewed single-request Space Bunny plan')
        for path, sha in plan['pins'].items():
            if digest(path) != sha:
                raise ValueError('reviewed artifact hash changed: ' + path)
        runtime = Path(plan['runtime'])
        frozen = json.loads((runtime / 'frozen-runtime.json').read_text())
        for path, sha in frozen['files'].items():
            if digest(runtime / path) != sha:
                raise ValueError('frozen runtime hash changed: ' + path)
        dependencies = json.loads(Path(plan['provider_dependency_pins']).read_text())
        for path, sha in dependencies['files'].items():
            if digest(path) != sha:
                raise ValueError('provider dependency changed: ' + path)
        if datetime.datetime.now(datetime.timezone.utc).date() >= datetime.date(2026, 10, 5):
            status('paused_model_expired')
            return
        key = credential(args.key_file)
        account = get_json('https://openrouter.ai/api/v1/key', key)['data']
        if account.get('limit_remaining') is not None and account['limit_remaining'] < 0:
            status('paused_account_limit')
            return
        endpoint = get_json('https://openrouter.ai/api/v1/models/' + MODEL + '/endpoints')['data']
        eligible = [item for item in endpoint['endpoints'] if item.get('tag') == 'stealth'
                    and float(item['pricing']['prompt']) == 0 and float(item['pricing']['completion']) == 0
                    and item.get('supports_tool_choice', {}).get('auto') is True]
        if not eligible:
            status('paused_no_compatible_free_endpoint')
            return
        controls = json.loads(Path(plan['provider_request_config']).read_text())
        routing = controls['piPayload']['provider']
        if (routing.get('only') != ['stealth'] or routing.get('allow_fallbacks') is not False or
                routing.get('max_price') != {'prompt': 0, 'completion': 0} or
                'enforce_distillable_text' in routing or controls['piPayload'].get('tool_choice') != 'auto' or
                controls.get('omitPayloadKeys') != ['seed']):
            raise ValueError('request controls differ from the approved routing and compatibility policy')
        command = [sys.executable, plan['supervisor'], plan['queue'], plan['journal'],
                   '--runtime', str(runtime), '--provider', 'openrouter', '--model-id', MODEL,
                   '--model-concurrency', '1', '--case-seconds', '1200', '--min-free-mib', '1024',
                   '--provider-request-config', plan['provider_request_config']]
        env = dict(os.environ, OPENROUTER_API_KEY=key)
        key = None
        with (root / 'worker.log').open('a') as output:
            child = subprocess.Popen(command, env=env, stdout=output, stderr=subprocess.STDOUT)
            env.pop('OPENROUTER_API_KEY', None)
            status('running', supervisor_pid=child.pid, queue=plan['queue'], journal=plan['journal'],
                   request_policy='free-stealth-only; no distillation flag; automatic tools; no wire seed')
            def stop(signum, frame):
                child.terminate()
            signal.signal(signal.SIGINT, stop)
            signal.signal(signal.SIGTERM, stop)
            code = child.wait()
            journal = Path(plan['journal'])
            last = json.loads(journal.read_text().splitlines()[-1]) if journal.exists() and journal.stat().st_size else {}
            state = 'paused' if last.get('event') == 'pause' else ('finished' if code == 0 else 'stopped')
            status(state, pause_reason=last.get('reason'), supervisor_pid=child.pid, exit_code=code,
                   queue=plan['queue'], journal=plan['journal'])

if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        # Provider response bodies and credential values never enter diagnostics.
        if isinstance(error, ValueError):
            print(str(error), file=sys.stderr)
        else:
            print('OpenRouter worker setup failed: ' + type(error).__name__, file=sys.stderr)
        sys.exit(1)
