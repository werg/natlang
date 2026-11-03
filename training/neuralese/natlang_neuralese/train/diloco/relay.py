"""Local forwarding relay from a learner to the current DiLoCo coordinator.

Every learner's ``--diloco`` block names ``http://127.0.0.1:PORT`` (identical in every worker control). This relay,
one per sandbox, forwards each request to the coordinator named by the secret files ``diloco-url`` and
``diloco-gateway`` under ``$NATLANG_CW_SECRETS`` (or ``--secrets``), re-read on every request. A primary failover
is therefore a rewrite of those two files on each learner sandbox: learners see a short transient outage (waited out
by ``worker.patient`` or counted as offline steps), never a changed trainer argument or identity.

The bearer token passes through unchanged. The gateway share token of the current target replaces whatever the
learner sent; an empty ``diloco-gateway`` (a coordinator on this host) sends none. Bodies stream in bounded blocks in
both directions (fragment uploads are hundreds of MB). Standard library only: it runs beside any frozen runtime.

    python3 relay.py --port 18080 [--secrets DIR] [--bind 127.0.0.1]
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import sys
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

BLOCK = 4 << 20
HOP = {'connection', 'keep-alive', 'proxy-authenticate', 'proxy-authorization', 'te', 'trailers',
       'transfer-encoding', 'upgrade', 'host', 'x-sandbox-share-token', 'content-length'}
TIMEOUT = 900.0


def target(secrets):
    """(scheme, host, port, base path, gateway token) of the current coordinator."""
    url = (Path(secrets) / 'diloco-url').read_text().strip()
    gateway_file = Path(secrets) / 'diloco-gateway'
    gateway = gateway_file.read_text().strip() if gateway_file.is_file() else ''
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in ('http', 'https') or not parsed.hostname:
        raise ValueError('diloco-url is not an absolute HTTP(S) URL')
    port = parsed.port or (443 if parsed.scheme == 'https' else 80)
    return parsed.scheme, parsed.hostname, port, parsed.path.rstrip('/'), gateway


def make_handler(secrets, counters):
    class Relay(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, format, *args):
            pass  # never log authorization headers

        def _error(self, status, message):
            body = json.dumps({'error': f'relay: {message}'}).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _forward(self):
            if self.path == '/relay-health':
                try:
                    scheme, host, port, _, gateway = target(secrets)
                    value = {'ok': True, 'target': f'{scheme}://{host}:{port}', 'gateway': bool(gateway), **counters}
                except (OSError, ValueError) as error:
                    value = {'ok': False, 'error': str(error), **counters}
                body = json.dumps(value).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            try:
                scheme, host, port, base, gateway = target(secrets)
            except (OSError, ValueError) as error:
                return self._error(503, f'no coordinator target: {error}')
            length = int(self.headers.get('Content-Length') or 0)
            headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
            if gateway:
                headers['X-Sandbox-Share-Token'] = gateway
            headers['Host'] = host if port in (80, 443) else f'{host}:{port}'
            headers['Content-Length'] = str(length)
            factory = http.client.HTTPSConnection if scheme == 'https' else http.client.HTTPConnection
            connection = factory(host, port, timeout=TIMEOUT)
            try:
                try:
                    connection.putrequest(self.command, base + self.path, skip_host=True, skip_accept_encoding=True)
                    for key, value in headers.items():
                        connection.putheader(key, value)
                    connection.endheaders()
                    remaining = length
                    while remaining:
                        block = self.rfile.read(min(BLOCK, remaining))
                        if not block:
                            raise ConnectionError('learner closed the request body early')
                        connection.send(block)
                        remaining -= len(block)
                    response = connection.getresponse()
                except (OSError, http.client.HTTPException) as error:
                    counters['upstream_errors'] += 1
                    return self._error(502, f'{type(error).__name__}: {error}')
                size = response.getheader('Content-Length')
                body = None if size is not None else response.read()
                self.send_response(response.status, response.reason)
                for key, value in response.getheaders():
                    if key.lower() not in HOP:
                        self.send_header(key, value)
                self.send_header('Content-Length', str(len(body) if body is not None else int(size)))
                self.end_headers()
                if body is not None:
                    self.wfile.write(body)
                else:
                    remaining = int(size)
                    while remaining:
                        block = response.read(min(BLOCK, remaining))
                        if not block:
                            raise ConnectionError('coordinator closed the response early')
                        self.wfile.write(block)
                        remaining -= len(block)
                counters['forwarded'] += 1
            finally:
                connection.close()

        do_GET = do_POST = do_PUT = do_DELETE = _forward

    return Relay


def serve(port, secrets, bind='127.0.0.1', stop=None):
    counters = {'forwarded': 0, 'upstream_errors': 0}
    server = ThreadingHTTPServer((bind, port), make_handler(secrets, counters))
    server.daemon_threads = True
    if stop is not None:
        threading.Thread(target=lambda: (stop.wait(), server.shutdown()), daemon=True).start()
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        server.server_close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--secrets', default=os.environ.get('NATLANG_CW_SECRETS'))
    parser.add_argument('--bind', default='127.0.0.1')
    args = parser.parse_args(argv)
    if not args.secrets:
        raise SystemExit('relay needs --secrets or $NATLANG_CW_SECRETS')
    print(json.dumps({'event': 'relay_start', 'port': args.port, 'pid': os.getpid()}), flush=True)
    serve(args.port, args.secrets, args.bind)


if __name__ == '__main__':
    sys.exit(main())
