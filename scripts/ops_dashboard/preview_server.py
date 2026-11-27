#!/usr/bin/env python3
"""Local layout preview of the natlang ops page with real data: serves a built page graph and answers the page's queries
with the same handlers the agent program uses (data_service.py). It is not the application session: commands
(edits, asks) are kept in this process's memory, shown as previews and never applied. Loopback only.

usage: (cd app && node pages/ops/build.mjs --preview pages/ops/preview) && preview_server.py [--page DIR] [--port 8765]
(run under the memory ledger; see README.md)
"""
import argparse
import http.server
import json
import os
import sys
import threading
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_service  # noqa: E402

MEMORY = {'edits': [], 'applications': [], 'asks': [], 'answers': []}
LOCK = threading.Lock()


class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write('%s %s\n' % (time.strftime('%H:%M:%S'), fmt % args))

    def end_headers(self):
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()

    def _json(self, code, value):
        body = json.dumps(value, ensure_ascii=False, default=str).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith('/__preview/ping'):
            return self._json(200, {'preview': True})
        return super().do_GET()

    def do_POST(self):
        n = int(self.headers.get('Content-Length') or 0)
        body = json.loads(self.rfile.read(n) or b'{}')
        if self.path == '/__preview/query':
            q = body.get('q') or ''
            if q.startswith(('edits.', 'inbox.')):
                return self._json(403, {'error': 'native-only operation', 'code': 'forbidden'})
            try:
                with LOCK:          # the data service is single-threaded by design (the agent program serializes it too)
                    return self._json(200, data_service.answer(q, body.get('input')))
            except data_service.QueryError as e:
                return self._json(400, {'error': str(e), 'code': e.code})
            except Exception as e:
                return self._json(500, {'error': f'{type(e).__name__}: {e}', 'code': 'error'})
        if self.path == '/__preview/command':
            name, inp, op = body.get('name'), body.get('input') or {}, str(uuid.uuid4())
            if name == 'proposeEdit':
                if (inp.get('target') or {}).get('kind', '').startswith('held'):
                    return self._json(200, {'operationId': op, 'status': 'applied', 'outcome': 'rejected', 'rejection': 'HELD_DATA_IS_VIEW_ONLY'})
                MEMORY['edits'].append({**inp, 'operationId': op, 'author': 'preview'})
                MEMORY['applications'].append({'edit': op, 'status': 'applied', 'detail': 'Preview only: not applied to any data layer.', 'files': '', 'author': 'preview'})
            elif name == 'ask':
                MEMORY['asks'].append({**inp, 'operationId': op, 'author': 'preview'})
            elif name == 'answer':           # lets a preview session play the supervising agent (curl)
                MEMORY['answers'].append({**inp, 'operationId': op, 'author': 'preview-agent'})
            return self._json(200, {'operationId': op, 'status': 'applied', 'outcome': 'accepted'})
        if self.path == '/__preview/state':
            return self._json(200, {'edits': {'edits': MEMORY['edits'], 'applications': MEMORY['applications']},
                                    'inbox': {'asks': MEMORY['asks'], 'answers': MEMORY['answers']}})
        return self._json(404, {'error': 'unknown'})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--page', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app', 'pages', 'ops', 'preview'))
    ap.add_argument('--port', type=int, default=8765)
    args = ap.parse_args()
    data_service._load_modules()
    import functools       # the page directory is rebuilt in place (build.mjs --preview): resolve it per request, never chdir
    srv = http.server.ThreadingHTTPServer(('127.0.0.1', args.port), functools.partial(Handler, directory=args.page))
    print(f'preview: http://127.0.0.1:{args.port}/', flush=True)
    srv.serve_forever()


if __name__ == '__main__':
    main()
