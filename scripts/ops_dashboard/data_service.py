#!/usr/bin/env python3
"""natlang ops data service: answers the ops page's on-demand queries and applies its edits.

Runs as a child of the native participant (app/ops-peer.mjs) speaking JSON lines on stdin/stdout:
    request  {"id": n, "q": "data.item", "input": {...}}
    reply    {"id": n, "ok": true, "value": ...} | {"id": n, "ok": false, "error": "...", "code": "..."}
The same handlers serve the local layout preview (preview_server.py). Everything is read lazily and bounded: no handler
loads a whole multi-GB file; large JSONL files are read through byte-offset indexes cached under the dashboard dir.

Query families: ops.* (exporter state), data.* (training data catalog, items, runs; data_catalog.py / data_items.py),
edits.* (apply page edits to the edit layers; data_edits.py), inbox.* (questions from the page).
"""
import json
import os
import sys
import time
import traceback

if __name__ == '__main__':      # one module instance: data_items etc. `import data_service` and must register here
    sys.modules['data_service'] = sys.modules[__name__]
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
CONFIG = json.load(open(os.path.join(HERE, 'config.json')))
DATA = os.path.expanduser(CONFIG['data_dir'])
REPO = os.path.expanduser(CONFIG['repo'])
DASH = DATA + '/ops/dashboard'
OUT = DASH + '/out'

HANDLERS = {}


class QueryError(Exception):
    def __init__(self, msg, code='invalid'):
        super().__init__(msg)
        self.code = code


def handler(name):
    def wrap(fn):
        HANDLERS[name] = fn
        return fn
    return wrap


def jsonl_since(path, offset=0, max_bytes=4 << 20, keep=None):
    """Records of a growing JSONL file after byte `offset` (complete lines only), at most `max_bytes` read from the end
    when offset is 0. Returns (records, new_offset)."""
    try:
        size = os.path.getsize(path)
    except OSError:
        return [], 0
    if offset > size:
        offset = 0
    start = offset if offset else max(0, size - max_bytes)
    with open(path, 'rb') as f:
        f.seek(start)
        data = f.read(size - start)
    skip = data.find(b'\n') + 1 if start and not offset else 0
    body = data[skip:]
    end = body.rfind(b'\n') + 1
    recs = []
    for raw in body[:end].splitlines():
        try:
            r = json.loads(raw)
        except ValueError:
            continue
        if keep is None or keep(r):
            recs.append(r)
    return recs, start + skip + end


def _mtime_ms(path):
    try:
        return int(os.path.getmtime(path) * 1000)
    except OSError:
        return None


# ---------- ops: the exporter's state (export.py writes these files every 30 s) ----------

@handler('ops.tick')
def ops_tick(_):
    return {'ts': int(time.time() * 1000), 'snapshot': _mtime_ms(OUT + '/latest.json'),
            'metrics': _mtime_ms(OUT + '/metrics.jsonl'), 'machines': _mtime_ms(OUT + '/machines.jsonl'),
            'events': _mtime_ms(OUT + '/events.jsonl'), 'edits': _mtime_ms(DATA + '/ops/data-edits/log.jsonl')}


@handler('ops.paths')
def ops_paths(_):
    return {'telemetry': DASH + '/page-telemetry.jsonl'}


@handler('ops.snapshot')
def ops_snapshot(_):
    try:
        rec = json.loads(open(OUT + '/latest.json').read())
    except (OSError, ValueError) as e:
        raise QueryError(f'no exporter snapshot: {e}', 'unavailable')
    return rec.get('data')


def _tail(name, inp, max_bytes, max_age_s):
    off = int((inp or {}).get('offset') or 0)
    cutoff = (time.time() - max_age_s) * 1000
    recs, new = jsonl_since(f'{OUT}/{name}', off, max_bytes, keep=lambda r: (r.get('ts') or 0) >= cutoff)
    return {'records': recs, 'offset': new}


@handler('ops.metrics')
def ops_metrics(inp):
    return _tail('metrics.jsonl', inp, 6 << 20, 30 * 86400)


@handler('ops.machines')
def ops_machines(inp):
    return _tail('machines.jsonl', inp, 2 << 20, 36 * 3600)


@handler('ops.events')
def ops_events(inp):
    return _tail('events.jsonl', inp, 1 << 20, 7 * 86400)


def _load_modules():
    """Data and edit handlers live in their own modules; they register into HANDLERS on import."""
    for mod in ('data_catalog', 'data_items', 'data_edits'):
        try:
            __import__(mod)
        except ImportError as e:
            if e.name != mod:
                raise


def answer(q, inp):
    fn = HANDLERS.get(q)
    if not fn:
        raise QueryError(f'unknown query {q}', 'unknown')
    return fn(inp)


def serve():
    _load_modules()
    out = sys.stdout
    for line in sys.stdin:
        try:
            req = json.loads(line)
        except ValueError:
            continue
        rid = req.get('id')
        try:
            val = answer(req.get('q'), req.get('input'))
            msg = {'id': rid, 'ok': True, 'value': val}
        except QueryError as e:
            msg = {'id': rid, 'ok': False, 'error': str(e), 'code': e.code}
        except Exception as e:      # one bad query must not stop the service
            traceback.print_exc(file=sys.stderr)
            msg = {'id': rid, 'ok': False, 'error': f'{type(e).__name__}: {e}', 'code': 'error'}
        out.write(json.dumps(msg, ensure_ascii=False, separators=(',', ':'), default=str) + '\n')
        out.flush()


if __name__ == '__main__':
    if '--serve' in sys.argv:
        serve()
    else:       # one-shot: data_service.py QUERY [JSON]
        _load_modules()
        print(json.dumps(answer(sys.argv[1], json.loads(sys.argv[2]) if len(sys.argv) > 2 else None), indent=1, default=str)[:20000])
