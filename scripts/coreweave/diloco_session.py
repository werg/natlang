#!/usr/bin/env python3
"""Operator side of an elastic DiLoCo session on CoreWeave learner sandboxes (DGX; run with the cw venv).

The sandbox side is training/neuralese/natlang_neuralese/train/diloco/{session,join,relay}.py. These commands move
the few small files a session needs into a running sandbox through the gateway exec plane (no relaunch, no new
bundle) and read its state back:

  diloco_session.py token RUN.json SESSION_ID          create/print the path of the session bearer token (DGX, 0600)
  diloco_session.py secrets SANDBOX --token FILE (--url URL --gateway-of SANDBOX | --local PORT)
                                                       write diloco-token, diloco-url, diloco-gateway into the
                                                       sandbox's $NATLANG_CW_SECRETS (the relay re-reads url/gateway
                                                       per request: this is also the failover switch)
  diloco_session.py push SANDBOX LOCAL REMOTE          one file (atomic replace; sha256 verified)
  diloco_session.py request SANDBOX LINEAGE_DIR SESSION_DIR --worker K [-- PREPARE ARGS]
                                                       LINEAGE_DIR/diloco-session-request.json: the next window
                                                       boundary builds the session in place (sft_window.py next)
  diloco_session.py refresh SANDBOX                    SIGUSR2 to the lineage's native_state_sft learner: a
                                                       semantic-refresh window boundary at its next completed update
  diloco_session.py fetch SANDBOX REMOTE_DIR LOCAL_DIR copy a small directory back (tar, sha256 per file)
  diloco_session.py status SANDBOX [--relay-port 18080]
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import shlex
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cwctl  # noqa: E402

CHUNK = 256 * 1024


def run(sandbox, script, check=True):
    result = cwctl.read_sandbox(sandbox).exec(['bash', '-c', script]).result()
    if check and result.returncode != 0:
        raise SystemExit(f'{sandbox}: exec failed rc={result.returncode}: {result.stderr[-2000:]} {result.stdout[-2000:]}')
    return result.stdout


def put_bytes(sandbox, data, remote, mode='0644'):
    remote = str(remote)
    temp = f'{remote}.push-{os.getpid()}'
    run(sandbox, f'mkdir -p {shlex.quote(os.path.dirname(remote))} && : > {shlex.quote(temp)}')
    for start in range(0, len(data), CHUNK):
        encoded = base64.b64encode(data[start:start + CHUNK]).decode()
        run(sandbox, f'printf %s {encoded} | base64 -d >> {shlex.quote(temp)}')
    digest = hashlib.sha256(data).hexdigest()
    run(sandbox, f'test "$(sha256sum {shlex.quote(temp)} | cut -d" " -f1)" = {digest} && chmod {mode} '
                 f'{shlex.quote(temp)} && mv -f {shlex.quote(temp)} {shlex.quote(remote)}')
    return {'remote': remote, 'sha256': digest, 'bytes': len(data)}


def secrets_dir(sandbox):
    path = run(sandbox, 'echo "${NATLANG_CW_SECRETS:-/work/.coreweave/secrets}"').strip()
    return path or '/work/.coreweave/secrets'


def cmd_token(args):
    run_file = json.loads(Path(args.run).read_text())
    path = Path(run_file['root']) / '.cloud' / 'diloco' / f'{args.session}.token'
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        import secrets as source
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, 'w') as stream:
            stream.write(source.token_urlsafe(32))
    print(path)


def cmd_secrets(args):
    directory = secrets_dir(args.sandbox)
    token = Path(args.token).read_bytes().strip()
    put_bytes(args.sandbox, token, f'{directory}/diloco-token', '0600')
    if args.local is not None:
        url, gateway = f'http://127.0.0.1:{args.local}'.encode(), b''
    else:
        url = args.url.encode()
        gateway = cwctl.share_token(args.gateway_of) if args.gateway_of else b''
    put_bytes(args.sandbox, gateway, f'{directory}/diloco-gateway', '0600')
    put_bytes(args.sandbox, url, f'{directory}/diloco-url', '0600')
    print(json.dumps({'sandbox': args.sandbox, 'secrets': directory, 'url': url.decode(), 'gateway': bool(gateway)}))


def cmd_push(args):
    print(json.dumps(put_bytes(args.sandbox, Path(args.local).read_bytes(), args.remote)))


def cmd_request(args):
    value = {'session_dir': args.session_dir, 'worker': args.worker, 'prepare_args': args.prepare_args,
             'id_prefix': args.id_prefix}
    path = f'{args.lineage_dir.rstrip("/")}/diloco-session-request.json'
    print(json.dumps(put_bytes(args.sandbox, (json.dumps(value, indent=1) + '\n').encode(), path)))


def cmd_refresh(args):
    # The learner is the python child of the lineage's rl.production controller (never this exec's own shells).
    script = r'''for p in /proc/[0-9]*; do c=$(tr "\0" " " < $p/cmdline 2>/dev/null); pp=$(awk '{print $4}' $p/stat 2>/dev/null);
case "$c" in *python*" -m natlang_neuralese.train.native_state_sft "*)
  case "$(tr "\0" " " < /proc/$pp/cmdline 2>/dev/null)" in *"-m natlang_neuralese.rl.production run"*) echo "${p#/proc/}";; esac;; esac; done'''
    pids = [x for x in run(args.sandbox, script).split() if x.isdigit()]
    if len(pids) != 1:
        raise SystemExit(f'expected exactly one native_state_sft learner, found {pids}')
    run(args.sandbox, f'kill -USR2 {pids[0]}')
    print(json.dumps({'sandbox': args.sandbox, 'pid': int(pids[0]), 'signal': 'SIGUSR2'}))


def cmd_fetch(args):
    encoded = run(args.sandbox, f'tar -C {shlex.quote(args.remote)} -cf - . | base64 -w0')
    local = Path(args.local)
    if local.exists() and any(local.iterdir()):
        raise SystemExit(f'{local} is not empty')
    local.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(encoded))) as archive:
        archive.extractall(local, filter='data')
    files = sorted(str(p.relative_to(local)) for p in local.rglob('*') if p.is_file())
    print(json.dumps({'local': str(local), 'files': len(files)}))


def cmd_status(args):
    script = (f'curl -s -m 5 http://127.0.0.1:{args.relay_port}/relay-health; echo; '
              f'curl -s -m 10 -H "Authorization: Bearer $(cat ${{NATLANG_CW_SECRETS:-/work/.coreweave/secrets}}/diloco-token)" '
              f'http://127.0.0.1:{args.relay_port}/v1/status | head -c 4000; echo; '
              'ls /work/diloco 2>/dev/null; free -g | head -2; cat /sys/fs/cgroup/memory.peak 2>/dev/null')
    print(run(args.sandbox, script, check=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('token'); p.add_argument('run'); p.add_argument('session')
    p = sub.add_parser('secrets'); p.add_argument('sandbox'); p.add_argument('--token', required=True)
    p.add_argument('--url'); p.add_argument('--gateway-of'); p.add_argument('--local', type=int)
    p = sub.add_parser('push'); p.add_argument('sandbox'); p.add_argument('local'); p.add_argument('remote')
    p = sub.add_parser('request'); p.add_argument('sandbox'); p.add_argument('lineage_dir'); p.add_argument('session_dir')
    p.add_argument('--worker', type=int, required=True); p.add_argument('--id-prefix', default='lfm3-cloud')
    p.add_argument('prepare_args', nargs=argparse.REMAINDER)
    p = sub.add_parser('refresh'); p.add_argument('sandbox')
    p = sub.add_parser('fetch'); p.add_argument('sandbox'); p.add_argument('remote'); p.add_argument('local')
    p = sub.add_parser('status'); p.add_argument('sandbox'); p.add_argument('--relay-port', type=int, default=18080)
    args = parser.parse_args(argv)
    if getattr(args, 'prepare_args', None) and args.prepare_args[:1] == ['--']:
        args.prepare_args = args.prepare_args[1:]
    if args.command == 'secrets' and (args.local is None) == (args.url is None):
        raise SystemExit('secrets: give --url (with --gateway-of) or --local PORT')
    cwctl.load_env_file() if hasattr(cwctl, 'load_env_file') else None
    {'token': cmd_token, 'secrets': cmd_secrets, 'push': cmd_push, 'request': cmd_request, 'refresh': cmd_refresh,
     'fetch': cmd_fetch, 'status': cmd_status}[args.command](args)


if __name__ == '__main__':
    main()
