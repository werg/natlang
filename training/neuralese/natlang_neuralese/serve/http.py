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
| `POST /v1/neuralese/grad` | Gradient replay session (`grad.GradSession`): loss, per-term losses, gradient block IDs. |
| `POST /v1/neuralese/decide` | Decision readout: `{"messages", "options"}` → `{"log_probs", "tokens"}`, each option scored as the whole assistant reply after one prompt pass. |
| `POST /v1/neuralese/optim` | One SGD or Adam step on parameter blocks; returns new parameter and optimiser-state blocks. |
| `POST /v1/neuralese/adapters` | A zero adapter block for this backbone (`{"kind", "rank", "u", "layers", "targets", "seed"}`) → block metadata; see `model/tiny_adapters.py`. |
| `POST /v1/neuralese/embed` | A block initialised from text (token embeddings): `{"text", "type"}` → block metadata. |
| `POST /v1/neuralese/guidance/check` | The first rejection of `{"reply", "guidance"}` checked prefix by prefix as during generation (serve/guidance.py), for conformance. |
| `POST /v1/neuralese/render` | The rendered prompt of `{"messages", "tools"?}` (blocks as `<block>`), for conformance with the llama.cpp fork. |
| `POST /v1/neuralese/encode` | A block encoding text in one forward pass through the port (supplied-input write, one vector per token): `{"text", "type", "context"?}` → block metadata. |
| `POST /v1/neuralese/write` | The write procedure at a write site: `{"messages", "prefix"?, "tools"?, "neuralese_temperature"?, "length"?, "passes"?}` → the written block's metadata. The reply is forced to `prefix` and then the open marker; the stop head decides the length unless `length` hints it (`passes`: write it block-wise). |
| `POST /v1/neuralese/digest` | The digest operator (`digest.py`): `{"name", "type", "value", "instructions", "system"?, "window"?}` → the digest block's metadata and `parts` (1 unless the value exceeds the write site's window, by default the model's context, and is digested in chunks). `system` is the digest instructions, as text or parts (their soft form). |

Request fields beyond OpenAI's: `neuralese_temperature` (default 0, deterministic), `neuralese_max_length` (capped
by the server's hard maximum), `neuralese_length` (an optional size hint: write exactly that many vectors, no stop
decision) and `neuralese_passes` (with a hint, write the block in that many parallel passes; exact when ≥ the length), `x_natlang_adapters` (`[{"id", "scale"}]`: adapter blocks active for the whole
request, or `{"code", "projection", "scale"}`: a Neuralese block decoded into an adapter by a served projection;
batches may mix requests with different adapters), `neuralese_template` (template readout: the reply is forced to
a call, `{"call", "arguments"?, "argument"?, "value": "write" | "decode"}`, cut from the model's own rendering of that
call; "write" makes the argument a written block and closes the call, "decode" decodes the value and the rest), and
`guidance` (serve/guidance.py: `true` or `{"require_call"?, "repeat"?, "syntax"?, "retries"?}`: the reply opens a
tool call, call names are checked, eval code is checked line by line for repetition and TypeScript syntax, and a
rejected line is rolled back and resampled; the response reports `x_natlang_guidance.rejections`), and the test hook
`x_natlang_forced`. `decide` and
`grad` bodies take `adapters` in the same form.
"""

from __future__ import annotations

import json
import queue
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .chat import RequestError
from .engine import Engine, GenerationRequest
from .guidance import Settings
from .grad import GradSession, decide, embed_text, encode_text, new_adapter, optim_step
from .store import decode_block, encode_block

_BLOCK = re.compile(r"^/v1/neuralese/blocks/(nz1_[a-z2-7]+)(/meta|/pin|/unpin)?$")
_LORA = re.compile(r"^/v1/neuralese/adapters/(nz1_[a-z2-7]+)/lora$")


def adapter_lora(engine: Engine, block_id: str) -> bytes:
    """A stored adapter as a GGUF LoRA (export/adapters.py), for servers that apply adapters as LoRAs (the fork)."""
    import tempfile
    from pathlib import Path

    from ..export.adapters import export_lora_gguf, export_peft
    from ..model.lfm2_port import resolve_base
    from ..model.tiny_adapters import AdapterSpec, is_adapter_dialect

    cache = engine.__dict__.setdefault("_loras", {})
    if block_id not in cache:
        block = engine.store.get(block_id)
        if block is None or not is_adapter_dialect(block.dialect):
            raise RequestError("neuralese-unknown-block", f"{block_id} is not a stored adapter")
        spec = AdapterSpec.parse(block.dialect)
        with tempfile.TemporaryDirectory() as tmp:
            peft = export_peft(engine.adapter_bank, spec, block.payload.float(), Path(tmp) / "peft")
            cache[block_id] = export_lora_gguf(peft, resolve_base(getattr(engine, "base_dir", None)),
                                               Path(tmp) / "lora.gguf").read_bytes()
    return cache[block_id]


def make_handler(engine: Engine):
    import threading

    grad_lock = threading.Lock()

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
                                        "max_block_length": engine.max_block, "grad": True, "grad_order": 2,
                                        "cutoff": engine.heads.cutoff, "adapters": ["xs", "tiny"],
                                        "projections": {name: {"source": p.source_dialect, "target": p.target, "identity": p.identity()}
                                                        for name, p in engine.projections.items()}})
            lora = _LORA.match(self.path)
            if lora:
                try:
                    return self._send(200, adapter_lora(engine, lora.group(1)), "application/octet-stream")
                except RequestError as error:
                    return self._error(404, error.code, str(error))
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
                    body = json.loads(self._body() or b"{}")
                    with grad_lock:
                        return self._json(200, GradSession(engine).run(body))
                if self.path == "/v1/neuralese/decide":
                    body = json.loads(self._body() or b"{}")
                    with grad_lock:
                        return self._json(200, decide(engine, body))
                if self.path == "/v1/neuralese/optim":
                    body = json.loads(self._body() or b"{}")
                    with grad_lock:
                        return self._json(200, optim_step(engine, body))
                if self.path == "/v1/neuralese/adapters":
                    body = json.loads(self._body() or b"{}")
                    with grad_lock:
                        return self._json(201, new_adapter(engine, body).meta())
                if self.path == "/v1/neuralese/embed":
                    body = json.loads(self._body() or b"{}")
                    return self._json(201, embed_text(engine, body.get("text") or "", body.get("type")).meta())
                if self.path == "/v1/neuralese/write":
                    body = json.loads(self._body() or b"{}")
                    prefix = body.get("prefix") or ""
                    request = GenerationRequest(messages=body.get("messages") or [], tools=body.get("tools"),
                                                max_tokens=engine.max_block + len(engine._tokens(prefix)) + 8,
                                                neuralese_temperature=float(body.get("neuralese_temperature") or 0.0),
                                                neuralese_length=body.get("length"), neuralese_passes=body.get("passes"),
                                                forced=([prefix] if prefix else []) + [{"neuralese": "write"}])
                    blocks = (engine.submit(request).result().get("neuralese") or {}).get("blocks") or []
                    if not blocks:
                        return self._error(500, "neuralese-write", "the write produced no block")
                    return self._json(201, blocks[0])
                if self.path == "/v1/neuralese/digest":
                    from ..digest import INSTRUCTIONS, PREFIX, window_of, write_digest

                    body = json.loads(self._body() or b"{}")

                    def write(messages):
                        request = GenerationRequest(messages=messages, max_tokens=engine.max_block + 32,
                                                    forced=[PREFIX, {"neuralese": "write"}])
                        blocks = (engine.submit(request).result().get("neuralese") or {}).get("blocks") or []
                        if not blocks:
                            raise RequestError("neuralese-digest", "a digest write produced no block")
                        return blocks[0]["id"]

                    instructions = body.get("instructions") or ""
                    window = window_of(engine, instructions, body.get("window"))
                    block, parts = write_digest(write, body.get("system") or INSTRUCTIONS, body.get("name") or "value",
                                                body.get("type") or "unknown", body.get("value") or "", instructions,
                                                engine.tokenizer, window)
                    return self._json(201, {**engine.lookup(block).meta(), "parts": parts, "window": window})
                if self.path == "/v1/neuralese/guidance/check":
                    from .guidance import Guide

                    body = json.loads(self._body() or b"{}")
                    g = body.get("guidance") or {}
                    guide = Guide(Settings(tools=g.get("tools") or None, repeat=int(g.get("repeat", 3)),
                                           syntax=bool(g.get("syntax", True)), run=int(g.get("run", 4))))
                    reply = body.get("reply") or ""
                    for end in range(1, len(reply) + 1):
                        verdict = guide.check(reply[:end])
                        if verdict:
                            return self._json(201, {"reason": verdict[0], "offset": verdict[1], "end": end})
                    return self._json(201, {"reason": None})
                if self.path == "/v1/neuralese/render":
                    from .chat import render_messages, split_escaped

                    body = json.loads(self._body() or b"{}")
                    rendered = render_messages(body.get("messages") or [], body.get("tools"), engine._template, engine.specials)
                    prompt = "".join("<block>" if isinstance(segment, int) else
                                     "".join(run for run, _ in split_escaped(segment, rendered.escape_nonce))
                                     for segment in rendered.segments)
                    return self._json(201, {"prompt": prompt})
                if self.path == "/v1/neuralese/encode":
                    body = json.loads(self._body() or b"{}")
                    with grad_lock:
                        block = encode_text(engine, body.get("text") or "", body.get("type"), body.get("context"))
                    return self._json(201, block.meta())
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
                neuralese_max_length=body.get("neuralese_max_length"), forced=body.get("x_natlang_forced"),
                neuralese_length=body.get("neuralese_length"), neuralese_passes=body.get("neuralese_passes"),
                template=body.get("neuralese_template"), adapters=body.get("x_natlang_adapters"),
                guidance=Settings.of(body.get("guidance", getattr(engine, "default_guidance", None)), body.get("tools"),
                                     body.get("tool_choice")))
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
            # Text streams as content deltas until a tool call opens; the call markup is held back and the parsed
            # calls arrive as `tool_calls` deltas at the end (OpenAI streaming), so clients that only read deltas see
            # the same message as a non-streaming request.
            marker, streamed, pending, in_call = "<|tool_call_start|>", "", "", False
            while True:
                item = deltas.get()
                if item is None:
                    break
                if in_call:
                    continue
                if "text" in item:
                    pending += item["text"]
                    at = pending.find(marker)
                    if at >= 0:
                        out, pending, in_call = pending[:at], "", True
                    else:
                        keep = max((n for n in range(1, len(marker)) if pending.endswith(marker[:n])), default=0)
                        out, pending = pending[:len(pending) - keep], pending[len(pending) - keep:]
                    if out:
                        streamed += out
                        self._event(chunk({"content": out}))
                else:
                    self._event(chunk({"content": [{"type": "neuralese", "id": item["neuralese"]["id"]}]},
                                      extra={"neuralese": {"block": item["neuralese"]}}))
            try:
                response = future.result()
                choice = response["choices"][0]
                message = choice["message"]
                content = message.get("content")
                if isinstance(content, str) and content.startswith(streamed.strip()) and len(content) > len(streamed.strip()):
                    self._event(chunk({"content": content[len(streamed.strip()):]}))
                elif not in_call and pending:
                    self._event(chunk({"content": pending}))
                calls = message.get("tool_calls") or []
                if calls:
                    self._event(chunk({"tool_calls": [{"index": i, "id": c["id"], "type": "function",
                                                       "function": {"name": c["function"]["name"],
                                                                    "arguments": c["function"]["arguments"] if isinstance(
                                                                        c["function"]["arguments"], str) else json.dumps(
                                                                        c["function"]["arguments"])}}
                                                      for i, c in enumerate(calls)]}))
                self._event(chunk({}, choice["finish_reason"], {"x_natlang_message": message,
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
