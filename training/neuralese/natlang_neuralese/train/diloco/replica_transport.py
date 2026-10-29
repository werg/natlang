"""Authenticated streaming transfer of ordered replica journal transactions."""
from __future__ import annotations
import base64
import hmac
import http.client
import json
import os
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
from .replica_journal import CHUNK


def serve(journal, host, port, token, *, max_bytes, stop):
    guard = threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_PUT(self):
            if not hmac.compare_digest(self.headers.get('Authorization', ''), 'Bearer '+token):
                self.send_error(403); return
            if self.path != '/v1/transaction':
                self.send_error(404); return
            temp = None
            try:
                length = int(self.headers['Content-Length'])
                if length < 0 or length > max_bytes: raise ValueError('invalid transaction length')
                header = json.loads(base64.b64decode(self.headers['X-Transaction'], validate=True))
                if set(header) != {'sequence','previous','event'}: raise ValueError('invalid transaction metadata')
                fd, name = tempfile.mkstemp(prefix='.incoming-', dir=journal.root); temp = Path(name)
                with os.fdopen(fd, 'wb') as out:
                    left = length
                    while left:
                        block = self.rfile.read(min(CHUNK,left))
                        if not block: raise ValueError('truncated transaction')
                        out.write(block);left -= len(block)
                    out.flush();os.fsync(out.fileno())
                with guard:
                    ack = journal.append(header['event'],temp,expected_sequence=header['sequence'],expected_previous=header['previous'])
                payload=json.dumps(ack,sort_keys=True).encode();self.send_response(200);self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)
            except (KeyError,ValueError,OSError,TypeError):
                self.close_connection=True;self.send_error(400)
            finally:
                if temp is not None:temp.unlink(missing_ok=True)
    server=ThreadingHTTPServer((host,port),Handler);server.timeout=.2
    try:
        while not stop.is_set():server.handle_request()
    finally:server.server_close()


def transmit(url, token, *, sequence, previous, event, payload, timeout=30, allow_http=False, gateway=None):
    address=urlsplit(url)
    if address.scheme not in ('https','http') or (address.scheme=='http' and not allow_http) or not address.hostname or address.path not in ('','/'):
        raise ValueError('replica endpoint must be an HTTPS origin')
    header=base64.b64encode(json.dumps({'sequence':sequence,'previous':previous,'event':event},sort_keys=True,separators=(',',':')).encode()).decode()
    if len(header)>65536:raise ValueError('replica event metadata too large')
    path=Path(payload);size=path.stat().st_size
    cls=http.client.HTTPSConnection if address.scheme=='https' else http.client.HTTPConnection
    conn=cls(address.hostname,address.port,timeout=timeout)
    try:
        conn.putrequest('PUT','/v1/transaction');conn.putheader('Authorization','Bearer '+token);conn.putheader('Content-Length',str(size));conn.putheader('X-Transaction',header)
        if gateway:conn.putheader('X-Sandbox-Share-Token',gateway)
        conn.endheaders()
        with path.open('rb') as source:
            while block:=source.read(CHUNK):conn.send(block)
        response=conn.getresponse();data=response.read(65537)
        if response.status!=200 or len(data)>65536:raise ValueError('replica did not durably acknowledge transaction')
        ack=json.loads(data)
        if set(ack)!={'sequence','head'} or ack['sequence']!=sequence:raise ValueError('replica acknowledgement differs')
        return ack
    finally:conn.close()
