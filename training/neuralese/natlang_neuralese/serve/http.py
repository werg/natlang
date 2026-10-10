"""HTTP surface of the reference server (spec/NEURALESE_PORT.md, "Wire protocol").

| Endpoint | Meaning |
| --- | --- |
| `POST /v1/chat/completions` | OpenAI-style chat completion with Neuralese parts. With `"stream": true` (as the fork), answers server-sent `chat.completion.chunk` events: text deltas until a tool call opens (`<|tool_call_start|>` or `<tool_call>`; the markup is held back), a `[{"type": "neuralese", "id": …}]` content delta per written block (with `neuralese.block`, its metadata), the parsed calls as one `tool_calls` delta, and a final chunk whose `x_natlang_message` is the complete parsed message (with `usage`, `neuralese` and, with guidance, `x_natlang_guidance`). |
| `GET /v1/models`, `GET /health`, `GET /v1/health` | Model listing and liveness. |
| `GET /v1/neuralese/info` | `dialects`, `width`, `dtype`, `max_block_length`, `cutoff`, `grad` (true), `grad_order` (2), `adapters` (kinds applied directly), `projections` (adapter-code decoders) and `stream` (true: chat completions stream; clients stream only to servers that declare it). |
| `PUT /v1/neuralese/blocks/{id}` | Store a block (safetensors body); the ID is checked against the content. |
| `GET /v1/neuralese/blocks/{id}` | Fetch a block (safetensors body). |
| `GET /v1/neuralese/blocks/{id}/meta` | Block metadata as JSON. |
| `POST /v1/neuralese/blocks/{id}/pin`, `…/unpin` | Keep or release a block through every collection (pins count per owner). |
| `POST /v1/neuralese/collect` | `{"referenced": […]}`: with an owner, the owner now holds exactly the referenced blocks (and its pinned ones); blocks it released are dropped unless another owner holds or pins them. Without an owner, drops the blocks no owner holds or pins that are not referenced. Answers `{"removed": […]}`. |
| `POST /v1/neuralese/grad` | Gradient replay session (`grad.GradSession`): loss, per-term losses, gradient block IDs. |
| `POST /v1/neuralese/decide` | Decision readout: `{"messages", "options", "tools"?, "adapters"?}` → `{"log_probs", "tokens"}`, each option scored as the whole assistant reply after one prompt pass. |
| `POST /v1/neuralese/decide_many`, `POST /v1/natlang/score` | Batched decisions: `{"items": [{"messages", "options" or "continuations", "tools"?, "adapters"?}], "adapters"?}` → `{"results": [{"log_probs", "tokens"} or {"error"}]}`; each item as `decide` alone, a failing item fails alone. |
| `POST /v1/neuralese/optim` | One SGD or Adam step on parameter blocks; returns new parameter and optimiser-state blocks. |
| `POST /v1/neuralese/adapters` | A zero adapter block for this backbone (`{"kind", "rank", "u", "layers", "targets", "seed"}`) → block metadata; see `model/tiny_adapters.py`. |
| `GET /v1/neuralese/adapters/{id}/lora` | A stored adapter exported as a GGUF LoRA, for the fork (which loads it with `PUT …/lora`). |
| `POST /v1/neuralese/embed` | A block initialised from text (token embeddings): `{"text", "type"?}` → block metadata. |
| `POST /v1/neuralese/guidance/check` | The first rejection of `{"reply", "guidance"}` checked prefix by prefix as during generation (serve/guidance.py), for conformance. |
| `POST /v1/neuralese/render` | The rendered prompt of `{"messages", "tools"?}` (blocks as `<block>`), for conformance with the llama.cpp fork. |
| `POST /v1/neuralese/encode` | A block encoding text in one forward pass through the port (supplied-input write, one vector per token): `{"text", "type"?, "context"?}` → block metadata. |
| `POST /v1/neuralese/write` | The write procedure at a write site: `{"messages", "prefix"?, "tools"?, "neuralese_temperature"?, "length"?, "passes"?}` → the written block's metadata. The reply is forced to `prefix` and then the open marker; the stop head decides the length unless `length` hints it (`passes`: write it block-wise). |
| `POST /v1/neuralese/view` | The Neuralese instance of the builtin `view(value, instructions?)` (`view.py`): `{"value", "instructions"?, "system"?, "window"?}` → the view block's metadata, `parts` (1 unless the value exceeds the write site's window, by default the model's context, and is viewed in chunks) and `window`. Every write is the template write of view's body (the reply forced to `return_result`, its value written). Without `instructions` the view is faithful. `system` is view's body, as text or parts (its soft form). |

Owners (serve/store.py `TensorStore`): a client names itself (a session or runtime ID) in the `x-natlang-owner`
header. That owner holds every block it uploads and every block ID the server names to it in a response, and its pins
and collections are its own: one session's collection never drops another's blocks. With `--store-dir` blocks, holds
and pins outlive restarts; a client that still gets `neuralese-unknown-block` re-uploads the block from its archive.

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
`x_natlang_forced`. `neuralese_template` also takes `value_type` (`"string"` or `"unknown"`: the written value sits
unquoted, in native value syntax) and `argument_path` (keys into nested arguments), as the fork does. `decide`,
`decide_many` and `grad` bodies take `adapters` in the same form.
"""

from __future__ import annotations

import json
import queue
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .chat import RequestError, arguments_text
from .engine import Engine, GenerationRequest
from .guidance import Settings
from .grad import GradSession, decide, decide_many, embed_text, encode_text, new_adapter, optim_step
from .store import decode_block, encode_block

_BLOCK = re.compile(r"^/v1/neuralese/blocks/(nz1_[a-z2-7]+)(/meta|/pin|/unpin)?$")
_BLOCK_ID = re.compile(r"^nz1_[a-z2-7]+$")
OWNER_HEADER = "x-natlang-owner"


def block_ids(value) -> set[str]:
    """Every block ID a JSON value names (as a whole string anywhere in it)."""
    if isinstance(value, str):
        return {value} if _BLOCK_ID.match(value) else set()
    if isinstance(value, dict):
        return set().union(*map(block_ids, value.values())) if value else set()
    if isinstance(value, (list, tuple)):
        return set().union(*map(block_ids, value)) if value else set()
    return set()
_LORA = re.compile(r"^/v1/neuralese/adapters/(nz1_[a-z2-7]+)/lora$")

# What this server serves (spec/NEURALESE_PORT.md "Capabilities"): every runtime reports its list in
# /v1/neuralese/info and answers a capability it lacks with 501 `neuralese-<capability>-unavailable`.
CAPABILITIES = sorted([
    "adapters.create", "adapters.direct", "adapters.lora-export", "adapters.projection", "chat", "chat.stream",
    "decide", "embed", "encode", "grad", "grad.order2", "guidance.check", "optim", "parts.value-type",
    "render", "score", "store", "store.owners", "template", "template.argument-path", "template.value-type", "view", "write"])


def unavailable_code(capability: str) -> str:
    return "neuralese-" + capability.replace(".", "-") + "-unavailable"


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

        @property
        def _owner(self) -> str | None:
            return self.headers.get(OWNER_HEADER) or None

        def _hold(self, value):
            """The requesting owner holds every block the server names to it."""
            if self._owner is not None:
                ids = block_ids(value)
                if ids:
                    engine.store.hold(self._owner, ids)
            return value

        def _json(self, status: int, value):
            self._send(status, json.dumps(self._hold(value) if status < 400 else value).encode())

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
                                        "store": {"owners": True, "persistent": engine.store.directory is not None},
                                        "max_block_length": engine.max_block, "context": engine.context, "grad": True, "grad_order": 2, "stream": True,
                                        "cutoff": engine.heads.cutoff, "adapters": ["xs", "tiny"],
                                        "projections": {name: {"source": p.source_dialect, "target": p.target, "identity": p.identity()}
                                                        for name, p in engine.projections.items()},
                                        "server": "reference",
                                        # Owner-scoped holds only once the store implements them.
                                        "capabilities": [c for c in CAPABILITIES
                                                         if c != "store.owners" or hasattr(engine.store, "hold")]})
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
                self._hold(block.id)
                return self._send(200, encode_block(block), "application/octet-stream")
            self._error(404, "not-found", self.path)

        def _unavailable(self, capability: str):
            self._error(501, unavailable_code(capability),
                        f"{capability} is not served by this server (see capabilities in /v1/neuralese/info)")

        def _method_not_allowed(self):
            self._error(405, "method-not-allowed", self.command)

        do_DELETE = do_PATCH = do_HEAD = do_OPTIONS = _method_not_allowed

        def __getattr__(self, name: str):
            # Any other method (BaseHTTPRequestHandler looks up `do_<METHOD>`) answers 405 as well, not Python's 501.
            if name.startswith("do_"):
                return self._method_not_allowed
            raise AttributeError(name)

        def _object(self) -> dict:
            """The JSON object body; anything else is 400 `bad-json`, as the fork answers."""
            body = json.loads(self._body() or b"{}")
            if not isinstance(body, dict):
                raise json.JSONDecodeError("request body is not a JSON object", "", 0)
            return body

        def do_PUT(self):
            if _LORA.match(self.path):
                # Adapters apply directly here; loading a LoRA is the fork's way of applying them.
                return self._unavailable("adapters.lora-load")
            match = _BLOCK.match(self.path)
            if not match or match.group(2):
                return self._error(404, "not-found", self.path)
            try:
                block = decode_block(self._body())
            except ValueError as error:
                return self._error(400, "neuralese-bad-block", str(error))
            if block.id != match.group(1):
                return self._error(400, "neuralese-id-mismatch", f"content hashes to {block.id}")
            stored = engine.store.put(block, self._owner)
            self._json(201, stored.meta())

        def do_POST(self):
            try:
                if self.path == "/v1/chat/completions":
                    return self._chat(self._object())
                if self.path == "/v1/neuralese/collect":
                    referenced = self._object().get("referenced")
                    if referenced is None:
                        referenced = []
                    if not isinstance(referenced, list) or not all(isinstance(i, str) for i in referenced):
                        return self._error(400, "bad-json", "referenced must be a list of block IDs")
                    referenced = set(referenced)
                    return self._send(200, json.dumps({"removed": engine.store.collect(referenced, self._owner)}).encode())
                if self.path == "/v1/neuralese/grad":
                    body = self._object()
                    with grad_lock:
                        return self._json(200, GradSession(engine).run(body))
                if self.path == "/v1/neuralese/decide":
                    body = self._object()
                    with grad_lock:
                        return self._json(200, decide(engine, body))
                if self.path in ("/v1/natlang/score", "/v1/neuralese/decide_many"):
                    body = self._object()
                    with grad_lock:
                        return self._json(200, decide_many(engine, body))
                if self.path == "/v1/neuralese/optim":
                    body = self._object()
                    with grad_lock:
                        return self._json(200, optim_step(engine, body))
                if self.path == "/v1/neuralese/adapters":
                    body = self._object()
                    with grad_lock:
                        return self._json(201, new_adapter(engine, body).meta())
                if self.path == "/v1/neuralese/embed":
                    body = self._object()
                    return self._json(201, embed_text(engine, body.get("text") or "", body.get("type")).meta())
                if self.path == "/v1/neuralese/write":
                    body = self._object()
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
                if self.path == "/v1/neuralese/view":
                    from ..view import INSTRUCTIONS, REPLY_TOKENS, TEMPLATE, TOOLS, window_of, write_view

                    body = self._object()

                    def write(messages):
                        # The template write of view's body: the reply forced to return_result, its value written.
                        request = GenerationRequest(messages=messages, tools=TOOLS, template=dict(TEMPLATE),
                                                    max_tokens=engine.max_block + REPLY_TOKENS)
                        blocks = (engine.submit(request).result().get("neuralese") or {}).get("blocks") or []
                        if not blocks:
                            raise RequestError("neuralese-view", "a view write produced no block")
                        return blocks[0]["id"]

                    instructions = body.get("instructions") or None
                    system = body.get("system") or INSTRUCTIONS
                    window = window_of(engine, instructions, body.get("window"), system)
                    block, parts = write_view(write, system, body.get("value") or "", instructions, engine.tokenizer,
                                              window)
                    return self._json(201, {**engine.lookup(block).meta(), "parts": parts, "window": window})
                if self.path == "/v1/neuralese/guidance/check":
                    from .guidance import Guide

                    body = self._object()
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

                    body = self._object()
                    # As prefill renders it: a stored block typed Neuralese<unknown> defaults to value_type "unknown".
                    rendered = render_messages(body.get("messages") or [], body.get("tools"), engine._template, engine.specials,
                                               block_type=engine.block_value_type)
                    prompt = "".join("<block>" if isinstance(segment, int) else
                                     "".join(run for run, _ in split_escaped(segment, rendered.escape_nonce))
                                     for segment in rendered.segments)
                    return self._json(201, {"prompt": prompt})
                if self.path == "/v1/neuralese/encode":
                    body = self._object()
                    with grad_lock:
                        block = encode_text(engine, body.get("text") or "", body.get("type"), body.get("context"))
                    return self._json(201, block.meta())
                match = _BLOCK.match(self.path)
                if match and match.group(2) in ("/pin", "/unpin"):
                    if match.group(2) == "/pin":
                        engine.store.pin(match.group(1), self._owner)
                    else:
                        engine.store.unpin(match.group(1), self._owner)
                    return self._json(200, {"ok": True})
                self._error(404, "not-found", self.path)
            except KeyError as error:
                self._error(404, "neuralese-unknown-block", str(error))
            except RequestError as error:
                self._error(400, error.code, str(error))
            except json.JSONDecodeError as error:
                self._error(400, "bad-json", str(error))
            except Exception as error:  # noqa: BLE001 - every failure answers the error envelope, as the fork's
                self._error(500, "internal", f"{type(error).__name__}: {error}")

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
            self._chunk(b"data: " + (value if isinstance(value, bytes) else json.dumps(self._hold(value)).encode()) + b"\n\n")

        def _client_gone(self) -> bool:
            """Whether the client closed its connection (readable with nothing to read): a streamed reply checks this
            while it has nothing to send, as the fork's sink.is_writable()."""
            import select
            import socket

            try:
                readable, _, _ = select.select([self.connection], [], [], 0)
                return bool(readable) and self.connection.recv(1, socket.MSG_PEEK) == b""
            except OSError:
                return True

        def _stream(self, request: GenerationRequest):
            """A streamed chat completion. A client that leaves (a failed write, or a closed connection while the
            stream has nothing to send) cancels the request: the engine drops it at its next round."""
            try:
                self._stream_events(request)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                request.cancelled = True
                self.close_connection = True

        def _stream_events(self, request: GenerationRequest):
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
            # Call markup is LFM2's Pythonic `<|tool_call_start|>` or the Qwen family's `<tool_call>` (chat.build_message).
            markers, streamed, pending, in_call = ("<|tool_call_start|>", "<tool_call>"), "", "", False
            while True:
                try:
                    item = deltas.get(timeout=0.25)
                except queue.Empty:
                    if self._client_gone():
                        raise BrokenPipeError("the client went away")
                    continue
                if item is None:
                    break
                if in_call:
                    continue
                if "text" in item:
                    pending += item["text"]
                    at = min((i for i in (pending.find(m) for m in markers) if i >= 0), default=-1)
                    if at >= 0:
                        out, pending, in_call = pending[:at], "", True
                    else:
                        keep = max((n for m in markers for n in range(1, len(m)) if pending.endswith(m[:n])), default=0)
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
                                                                        c["function"]["arguments"], str) else arguments_text(
                                                                        c["function"]["arguments"])}}
                                                      for i, c in enumerate(calls)]}))
                self._event(chunk({}, choice["finish_reason"], {"x_natlang_message": message,
                                                                "usage": response["usage"],
                                                                "neuralese": response["neuralese"],
                                                                **({"x_natlang_guidance": response["x_natlang_guidance"]}
                                                                   if "x_natlang_guidance" in response else {})}))
            except RequestError as error:
                self._event({"error": {"code": error.code, "message": str(error)}})
            except Exception as error:  # noqa: BLE001 - reported to the client, as every unexpected failure
                self._event({"error": {"code": "internal", "message": f"{type(error).__name__}: {error}"}})
            self._event(b"[DONE]")
            self._chunk(b"")

    return Handler


def serve(engine: Engine, host: str = "127.0.0.1", port: int = 0) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), make_handler(engine))
    server.daemon_threads = True
    return server
