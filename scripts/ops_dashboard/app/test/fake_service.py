#!/usr/bin/env python3
"""A stand-in for data_service.py for the app's tests and local end-to-end runs: same JSON-lines protocol, canned answers.
ops.snapshot returns a small v3 snapshot that changes every call's second; edits.apply records the edit in FAKE_SERVICE_LOG."""
import json
import os
import sys
import time

LOG = os.environ.get('FAKE_SERVICE_LOG')
DASH = os.environ.get('FAKE_SERVICE_DASH', '/tmp')


def answer(q, inp):
    if q == 'ops.paths':
        return {'telemetry': DASH + '/page-telemetry.jsonl'}
    if q == 'ops.snapshot':
        return {'v': 3, 'ts': int(time.time() * 1000) // 30000 * 30000, 'machines': {}, 'processes': [], 'alerts': [],
                'ei': {'run': 'fake', 'rounds': [{'id': 'r0001', 'blob': 'x' * 20000}]}, 'runs': []}
    if q.startswith('data.'):
        return {'q': q, 'input': inp, 'rows': list(range(10))}
    if q == 'edits.apply':
        if LOG:
            with open(LOG, 'a') as f:
                f.write(json.dumps(inp) + '\n')
        return {'status': 'applied', 'detail': 'fake service applied it', 'files': ''}
    raise KeyError(q)


for line in sys.stdin:
    req = json.loads(line)
    try:
        out = {'id': req['id'], 'ok': True, 'value': answer(req['q'], req.get('input'))}
    except Exception as e:  # noqa: BLE001
        out = {'id': req['id'], 'ok': False, 'error': str(e), 'code': 'error'}
    print(json.dumps(out), flush=True)
