#!/usr/bin/env python3
"""Cross-account bring-up test: one minimal CPU sandbox per account, an HTTPS echo behind share token + bearer, and a
reachability matrix (every account's sandbox calls every other account's endpoint; the DGX calls all of them).

Run only once the owner says the accounts are live (it creates one 1-CPU/2Gi sandbox per account for minutes):

    /home/werg/.local/share/natlang/cw-venv/bin/python scripts/coreweave/account_bringup.py run --accounts A,B,C,D
    account_bringup.py plan --accounts A,B,C,D     the exact requests; no API call

Each sandbox is created in its own account (explicit per-call auth from that account's profile) and bound to it, so
every later call uses the owner account. What enters a sandbox: its own bearer token, and for the probe the peers'
URLs, share tokens and bearer tokens. Share tokens are per-sandbox gateway capabilities, exactly what DiLoCo learners
in one account receive for a coordinator in another; no account API key ever enters a sandbox (checked). The report
(STATE/bringup/<stamp>.json) holds the matrix with status codes, plus negative checks: without the share token the
gateway must refuse, and without the bearer the echo must refuse. All sandboxes are stopped at the end.
"""
from __future__ import annotations

import argparse
import json
import os
import secrets as tokens
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cwctl  # noqa: E402
import job as jobs  # noqa: E402

PORT = 8080
IMAGE = 'python:3.11'
LIFETIME_SECONDS = 2 * 3600
DIR = '/tmp/natlang-bringup'
SERVER = r'''
import http.server, json, os, pathlib, time
d = pathlib.Path(DIR)
d.mkdir(parents=True, exist_ok=True)
while not (d / 'token').exists():
    time.sleep(0.5)
token = (d / 'token').read_text().strip()
class Echo(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        ok = self.headers.get('Authorization') == 'Bearer ' + token
        body = json.dumps({'account': ACCOUNT, 'echo': self.path, 'authorized': ok,
                           'sandbox': os.environ.get('CW_SANDBOX_ID')}).encode()
        self.send_response(200 if ok else 401)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *args):
        pass
http.server.ThreadingHTTPServer(('0.0.0.0', PORT), Echo).serve_forever()
'''
PROBE = r'''
import json, sys, urllib.error, urllib.request
def call(url, headers):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as r:
            return {'status': r.status, 'body': json.loads(r.read() or b'{}')}
    except urllib.error.HTTPError as e:
        return {'status': e.code}
    except Exception as e:
        return {'status': None, 'error': type(e).__name__ + ': ' + str(e)[:200]}
out = {}
for peer in json.load(open(sys.argv[1])):
    url = peer['url'].rstrip('/') + '/echo'
    full = {'Authorization': 'Bearer ' + peer['bearer'], 'X-Sandbox-Share-Token': peer['share']}
    out[peer['account']] = {'full': call(url, full),
                            'no_share_token': call(url, {'Authorization': full['Authorization']}),
                            'no_bearer': call(url, {'X-Sandbox-Share-Token': peer['share']})}
print(json.dumps(out))
'''


def request(account: str) -> tuple[list[str], dict]:
    """(run arguments, keywords) of the account's bring-up sandbox (the same service shape as the coordinator)."""
    program = f'DIR={DIR!r}\nACCOUNT={account!r}\nPORT={PORT}\n' + SERVER
    compile(program, '<bringup-server>', 'exec')
    compile(PROBE, '<bringup-probe>', 'exec')
    return ['python3', '-u', '-c', program], {
        'container_image': IMAGE, 'resources': {'cpu': '1', 'memory': '2Gi'},
        'max_lifetime_seconds': LIFETIME_SECONDS, 'tags': ['natlang', f'job-account-bringup-{account}', 'kind-smoke'],
        'placement_mode': 'serverless', 'data_plane_mode': 'gateway',
        'services': [{'port': PORT, 'visibility': 'public', 'endpoint': {
            'kind': 'https', 'auth': 'share_token', 'request_timeout_seconds': jobs.SERVICE_TIMEOUT_SECONDS}}]}


def dgx_call(url: str, bearer: str, share: str, opener=urllib.request.urlopen) -> dict:
    try:
        with opener(urllib.request.Request(url.rstrip('/') + '/echo', headers={
                'Authorization': 'Bearer ' + bearer, 'X-Sandbox-Share-Token': share}), timeout=30) as response:
            return {'status': response.status, 'body': json.loads(response.read() or b'{}')}
    except urllib.error.HTTPError as error:
        return {'status': error.code}
    except Exception as error:  # noqa: BLE001 - reported in the matrix
        return {'status': None, 'error': f'{type(error).__name__}: {error}'[:200]}


def verdict(matrix: dict) -> dict:
    """Cross-account traffic works when every full call answered 200 from the right account and both negative checks
    were refused."""
    problems = []
    for source, row in matrix.items():
        for target, cell in row.items():
            full = cell.get('full', cell)
            if full.get('status') != 200 or (full.get('body') or {}).get('account') != target:
                problems.append(f'{source}->{target}: {full}')
            for check in ('no_share_token', 'no_bearer'):
                if check in cell and cell[check].get('status') in (200, None):
                    problems.append(f'{source}->{target} {check}: not refused ({cell[check]})')
    return {'cross_account_ok': not problems, 'problems': problems}


def run(account_names: list[str], *, create=None, read=None, stop=None, opener=urllib.request.urlopen,
        wait_ready=None) -> dict:
    create = create or (lambda args, keywords, account, entry: cwctl.create_recorded(args, keywords, account, entry))
    read = read or (lambda sandbox_id: cwctl.read_sandbox(sandbox_id, patience=120))
    stop = stop or (lambda sandbox: sandbox.stop(graceful_shutdown_seconds=10, missing_ok=True).result(timeout=120))
    nodes, report = {}, {'accounts': account_names, 'started': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    try:
        for account in account_names:
            args, keywords = request(account)
            sandbox, _ = create(args, keywords, account, {
                'job': f'account-bringup-{account}', 'kind': 'smoke', 'lifetime_hours': LIFETIME_SECONDS / 3600,
                'resources': keywords['resources'], 'reason': 'multi-account bring-up reachability test',
                'exports': []})
            nodes[account] = {'sandbox': sandbox, 'bearer': tokens.token_urlsafe(32)}
        for account, node in nodes.items():
            sandbox = node['sandbox']
            (wait_ready or (lambda box: box.wait(timeout=1800)))(sandbox)
            node['share'] = cwctl.keep_share_token(sandbox).read_text()
            node['url'] = cwctl.service_url(sandbox)
            payload = {'token': node['bearer'].encode()}
            cwctl.cwaccounts.refuse_credentials(payload)
            box = read(sandbox.sandbox_id)
            box.exec(['mkdir', '-p', DIR]).result(timeout=60)
            box.write_file(f'{DIR}/token', payload['token']).result(timeout=60)
        time.sleep(0 if wait_ready else 10)  # the echo server starts once its token file appears
        matrix = {'dgx': {a: dgx_call(n['url'], n['bearer'], n['share'], opener) for a, n in nodes.items()}}
        for account, node in nodes.items():
            peers = [{'account': a, 'url': n['url'], 'share': n['share'], 'bearer': n['bearer']}
                     for a, n in nodes.items() if a != account]
            data = json.dumps(peers).encode()
            cwctl.cwaccounts.refuse_credentials({'peers': data})
            box = read(node['sandbox'].sandbox_id)
            box.write_file(f'{DIR}/peers.json', data).result(timeout=60)
            result = box.exec(['python3', '-c', PROBE, f'{DIR}/peers.json'], timeout_seconds=300).result(timeout=330)
            matrix[account] = (json.loads(result.stdout) if result.returncode == 0
                               else {'error': result.stderr[-500:]})
        report['matrix'] = matrix
        report['sandboxes'] = {a: n['sandbox'].sandbox_id for a, n in nodes.items()}
        report.update(verdict({s: r for s, r in matrix.items() if 'error' not in r}))
        if any('error' in r for r in matrix.values()):
            report['cross_account_ok'] = False
            report['problems'] += [f'{s}: probe failed' for s, r in matrix.items() if 'error' in r]
    finally:
        for account, node in nodes.items():
            try:
                stop(node['sandbox'])
                cwctl.record({'event': 'stop', 'account': account, 'args': [node['sandbox'].sandbox_id],
                              'why': 'bring-up test finished'})
            except Exception as error:  # noqa: BLE001 - reported; the operator stops it by hand
                report.setdefault('stop_errors', {})[account] = f'{type(error).__name__}: {error}'
    out = cwctl.STATE / 'bringup' / f"{report['started'].replace(':', '')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + '\n')
    report['report'] = str(out)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=('plan', 'run'))
    parser.add_argument('--accounts', required=True, help='comma-separated profile names, at least two')
    args = parser.parse_args(argv)
    names = [cwctl.cwaccounts.check_name(n) for n in args.accounts.split(',') if n]
    if len(set(names)) < 2:
        parser.error('at least two distinct accounts')
    cwctl.load_env_file()
    if args.command == 'plan':
        missing = [n for n in names if not cwctl.cwaccounts.has_credential(n)]
        print(json.dumps({'requests': {n: request(n)[1] for n in names}, 'missing_credentials': missing}, indent=1))
        return 0
    for name in names:
        if not cwctl.cwaccounts.has_credential(name):
            raise cwctl.ControllerError(f'account {name!r} has no credential profile')
    report = run(names)
    print(json.dumps({k: v for k, v in report.items() if k != 'matrix'}, indent=1))
    print(json.dumps(report.get('matrix'), indent=1))
    return 0 if report.get('cross_account_ok') else 1


if __name__ == '__main__':
    if not os.environ.get('NATLANG_CW_NO_REEXEC'):
        cwctl.reexec_in_venv()
    sys.exit(main())
