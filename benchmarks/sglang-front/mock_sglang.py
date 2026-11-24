"""A stand-in SGLang /generate for front benchmarks (run.sh): replays recorded replies (keyed by the prompt's last 64 ids) as
--incremental-streaming-output chunks of 8 tokens with k=1 natlang_top_k records, at --rate tokens/s per stream."""
import base64, json, struct, sys, threading, time, random
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
KEY = 64
episodes = json.load(open(sys.argv[1]))
port, rate = int(sys.argv[2]), float(sys.argv[3])
table = {}
for e in episodes:
    for t in e:
        table[tuple(t['key'])] = t
replies = list(table.values())
stats = {'hit': 0, 'miss': 0, 'tokens': 0}
lock = threading.Lock()

def record(token, logp):
    return base64.b64encode(struct.pack('<f', logp) + struct.pack('<i', token) + struct.pack('<e', max(logp, -60000.0))).decode()

class H(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    def log_message(self, *a): pass
    def _send(self, value):
        data = json.dumps(value).encode()
        self.send_response(200); self.send_header('content-type', 'application/json')
        self.send_header('content-length', str(len(data))); self.end_headers(); self.wfile.write(data)
    def do_GET(self):
        if self.path.startswith('/stats'):
            return self._send(stats)
        if self.path in ('/server_info', '/get_server_info'):
            return self._send({'enable_deterministic_inference': False})
        if self.path == '/model_info':
            return self._send({'model_path': None, 'weight_version': None})
        self._send({})
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['content-length'])))
        ids = body['input_ids']
        params = body.get('sampling_params') or {}
        t = table.get(tuple(ids[-KEY:]))
        with lock:
            stats['hit' if t is not None else 'miss'] += 1
        if t is None:
            t = random.choice(replies)
        tokens, logps = t['tokens'], t['logps']
        stops = set(params.get('stop_token_ids') or [])
        n = min(len(tokens), int(params.get('max_new_tokens') or len(tokens)))
        out = []
        for i in range(n):
            out.append(i)
            if tokens[i] in stops:
                break
        if not body.get('stream'):
            with lock: stats['tokens'] += len(out)
            time.sleep(len(out) / rate)
            return self._send({'output_ids': [tokens[i] for i in out],
                               'meta_info': {'natlang_top_k': [record(tokens[i], logps[i] or 0.0) for i in out],
                                             'prompt_tokens': len(ids), 'cached_tokens': len(ids) - 64}})
        self.send_response(200); self.send_header('content-type', 'text/event-stream')
        self.send_header('transfer-encoding', 'chunked'); self.end_headers()
        def chunk(data):
            self.wfile.write(f'{len(data):x}\r\n'.encode() + data + b'\r\n'); self.wfile.flush()
        start = time.monotonic()
        try:
            for at in range(0, len(out), 8):
                part = out[at:at + 8]
                due = start + (at + len(part)) / rate
                delay = due - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                value = {'output_ids': [tokens[i] for i in part],
                         'meta_info': {'natlang_top_k': [record(tokens[i], logps[i] or 0.0) for i in part],
                                       'prompt_tokens': len(ids), 'cached_tokens': len(ids) - 64}}
                chunk(b'data: ' + json.dumps(value).encode() + b'\n\n')
                with lock: stats['tokens'] += len(part)
            chunk(b'data: [DONE]\n\n'); chunk(b'')
        except (BrokenPipeError, ConnectionResetError):
            pass

class S(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 1024
print('mock sglang', len(table), 'replies on', port, flush=True)
S(('127.0.0.1', port), H).serve_forever()
