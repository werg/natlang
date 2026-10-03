"""HTTP surface of the reference server (spec/NEURALESE_PORT.md, "Wire protocol").

| Endpoint | Meaning |
| --- | --- |
| `POST /v1/chat/completions` | OpenAI-style chat completion with Neuralese parts. With `"stream": true`, answers server-sent `chat.completion.chunk` events: text deltas, a `[{"type": "neuralese", "id": …}]` content delta per written block, and a final chunk whose `x_natlang_message` is the complete parsed message. |
| `GET /v1/models`, `GET /health` | Model listing and liveness. |
| `GET /v1/neuralese/info` | Dialect, width, dtype, maximum block length, `grad` availability. |
| `PUT /v1/neuralese/blocks/{id}` | Store a block (safetensors body); the ID is checked against the content. |
| `GET /v1/neuralese/blocks/{id}` | Fetch a block (safetensors body). |
| `GET /v1/neuralese/blocks/{id}/meta` | Block metadata as JSON. |
| `POST /v1/neuralese/blocks/{id}/pin`, `…/unpin` | Keep or release a block through collection. |
| `POST /v1/neuralese/collect` | Drop unpinned blocks not in `{"referenced": […]}`. |
| `POST /v1/neuralese/grad` | Gradient replay sessions: 501 until implemented. |

Request fields beyond OpenAI's: `neuralese_temperature` (default 0, deterministic), `neuralese_max_length` (capped
by the server's hard maximum), and the test hook `x_natlang_forced`.
"""

from __future__ import annotations

import json
import queue
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .chat import RequestError
from .engine import Engine, GenerationRequest
from .store import decode_block, encode_block

_BLOCK = re.compile(r"^/v1/neuralese/blocks/(nz1_[a-z2-7]+)(/meta|/pin|/unpin)?$")


def make_handler(engine: Engine):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format, *args):  # quiet by default
            pass

        def _send(self, status: int, body: bytes, content_type: str = "application/json"):
            self.send_response(status)
            self.send_header("content-type", content_type)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, value):
            self._send(status, json.dumps(value).encode())

        def _error(self, status: int, code: str, detail: str):
            self._json(status, {"error": {"code": code, "message": detail}})

        def _body(self) -> bytes:
            return self.rfile.read(int(self.headers.get("content-length") or 0))

        def do_GET(self):
            if self.path in ("/health", "/v1/health"):
                return self._json(200, {"status": "ok"})
            if self.path == "/v1/models":
                return self._json(200, {"object": "list", "data": [{"id": engine.model_name, "object": "model"}]})
            if self.path == "/v1/neuralese/info":
                return self._json(200, {"dialects": [engine.dialect], "width": engine.width, "dtype": "f32",
                                        "max_block_length": engine.max_block, "grad": False,
                                        "cutoff": engine.heads.cutoff})
            match = _BLOCK.match(self.path)
            if match and match.group(2) in (None, "/meta"):
                block = engine.store.get(match.group(1))
                if block is None:
                    return self._error(404, "neuralese-unknown-block", match.group(1))
                if match.group(2) == "/meta":
                    return self._json(200, block.meta())
                return self._send(200, encode_block(block), "application/octet-stream")
            self._error(404, "not-found", self.path)

        def do_PUT(self):
            match = _BLOCK.match(self.path)
            if not match or match.group(2):
                return self._error(404, "not-found", self.path)
            try:
                block = decode_block(self._body())
            except ValueError as error:
                return self._error(400, "neuralese-bad-block", str(error))
            if block.id != match.group(1):
                return self._error(400, "neuralese-id-mismatch", f"content hashes to {block.id}")
            stored = engine.store.put(block)
            self._json(201, stored.meta())

        def do_POST(self):
            try:
                if self.path == "/v1/chat/completions":
                    return self._chat(json.loads(self._body() or b"{}"))
                if self.path == "/v1/neuralese/collect":
                    referenced = set(json.loads(self._body() or b"{}").get("referenced") or [])
                    return self._json(200, {"removed": engine.store.collect(referenced)})
                if self.path == "/v1/neuralese/grad":
                    return self._error(501, "neuralese-grad-unavailable", "gradient sessions are not implemented")
                match = _BLOCK.match(self.path)
                if match and match.group(2) in ("/pin", "/unpin"):
                    if match.group(2) == "/pin":
                        engine.store.pin(match.group(1))
                    else:
                        engine.store.unpin(match.group(1))
                    return self._json(200, {"ok": True})
                self._error(404, "not-found", self.path)
            except KeyError as error:
                self._error(404, "neuralese-unknown-block", str(error))
            except RequestError as error:
                self._error(400, error.code, str(error))
            except json.JSONDecodeError as error:
                self._error(400, "bad-json", str(error))

        def _chat(self, body: dict):
            request = GenerationRequest(
                messages=body.get("messages") or [], tools=body.get("tools"),
                max_tokens=int(body.get("max_tokens") or body.get("max_completion_tokens") or 512),
                temperature=float(body.get("temperature") or 0.0), seed=body.get("seed"),
                neuralese_temperature=float(body.get("neuralese_temperature") or 0.0),
                neuralese_max_length=body.get("neuralese_max_length"), forced=body.get("x_natlang_forced"))
            if body.get("stream"):
                return self._stream(request)
            try:
                response = engine.submit(request).result()
            except RequestError as error:
                return self._error(400, error.code, str(error))
            self._json(200, response)

        def _chunk(self, data: bytes):
            self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
            self.wfile.flush()

        def _event(self, value):
            self._chunk(b"data: " + (value if isinstance(value, bytes) else json.dumps(value).encode()) + b"\n\n")

        def _stream(self, request: GenerationRequest):
            deltas: queue.Queue = queue.Queue()
            request.on_delta = deltas.put
            future = engine.submit(request)
            future.add_done_callback(lambda _: deltas.put(None))
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("cache-control", "no-cache")
            self.send_header("transfer-encoding", "chunked")
            self.end_headers()
            created = int(time.time())

            def chunk(delta, finish=None, extra=None):
                value = {"id": request.request_id, "object": "chat.completion.chunk", "created": created,
                         "model": engine.model_name,
                         "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
                value.update(extra or {})
                return value

            self._event(chunk({"role": "assistant"}))
            while True:
                item = deltas.get()
                if item is None:
                    break
                if "text" in item:
                    self._event(chunk({"content": item["text"]}))
                else:
                    self._event(chunk({"content": [{"type": "neuralese", "id": item["neuralese"]["id"]}]},
                                      extra={"neuralese": {"block": item["neuralese"]}}))
            try:
                response = future.result()
                choice = response["choices"][0]
                self._event(chunk({}, choice["finish_reason"], {"x_natlang_message": choice["message"],
                                                                "usage": response["usage"],
                                                                "neuralese": response["neuralese"]}))
            except RequestError as error:
                self._event({"error": {"code": error.code, "message": str(error)}})
            except Exception as error:  # noqa: BLE001 - reported to the client
                self._event({"error": {"code": "server-error", "message": str(error)}})
            self._event(b"[DONE]")
            self._chunk(b"")

    return Handler


def serve(engine: Engine, host: str = "127.0.0.1", port: int = 0) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), make_handler(engine))
    server.daemon_threads = True
    return server
