"""HTTP surface for native shared-state serving.

Supported operations are ordinary text chat over a native checkpoint, typed
native-state writes and imports, authenticated graph gradients, decision
scoring, optimizer steps, adapter creation, guidance, view, and native artifact
custody. The former projected-block routes and GradSession API are retired.
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
from .store import decode_native_state, encode_native_state

_NATIVE_STATE = re.compile(r"^/v1/neuralese/native-states/(nzs1_[a-z2-7]+)(/meta|/pin|/unpin)?$")
_NATIVE_TENSOR = re.compile(r"^/v1/neuralese/native-tensor-artifacts/(nnt1_[a-z2-7]+)(/meta|/pin|/unpin)?$")
_BLOCK_ID = re.compile(r"^(?:nzs1_|nnt1_)[a-z2-7]+$")
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

# What this server serves (spec/NEURALESE_PORT.md "Capabilities"), reported in /v1/neuralese/info. Clients choose
# paths by this list and fail a request that needs an unlisted capability before sending it
# (`neuralese-<capability>-unavailable`); this server serves every capability it lists.
NATIVE_CAPABILITIES = sorted([
    "chat", "chat.stream", "decide", "score", "grad", "grad.order2", "optim", "native.adapters", "adapters.create", "guidance", "guidance.check", "native.text-write", "native.init-body", "parts.value-type", "render", "store", "store.native-state", "store.native-tensor-artifact",
    "native.protocol-recovery", "store.owners", "template", "template.argument-path", "template.value-type", "view"])


def completion_limit(body: dict) -> int:
    """A chat completion's allowance (SERVER_HARMONIZATION #20, the fork's `engine::completion_limit`): `max_tokens`,
    else `max_completion_tokens`, absent or null 512. A non-negative integer is honoured as given; 0 generates nothing
    (empty reply, finish_reason "length"). Anything else is a 400 `bad-max-tokens`, not a silent default."""
    for key in ("max_tokens", "max_completion_tokens"):
        value = body.get(key)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RequestError("bad-max-tokens", f"{key} must be a non-negative integer, got {json.dumps(value)}")
        return value
    return 512


def _grad_request(run):
    """Recorded-objective and derivative-program requests: a request that fails validation is a 400
    `neuralese-native-grad-invalid`, as for graph requests (native_grad.native_graph_gradient)."""
    try:
        return run()
    except RequestError:
        raise
    except (ValueError, KeyError, TypeError) as error:
        raise RequestError('neuralese-native-grad-invalid', str(error)) from error


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

        def _guarded(self, handler):
            """Every failure answers the error envelope, for GET and PUT as for POST."""
            try:
                return handler()
            except RequestError as error:
                self._error(400, error.code, str(error))
            except Exception as error:  # noqa: BLE001
                self._error(500, "internal", f"{type(error).__name__}: {error}")

        def do_GET(self):
            return self._guarded(self._get)

        def _get(self):
            if self.path == "/metrics":
                body = ''.join(f'{name} {value}\n' for name, value in engine.progress_metrics().items())
                return self._send(200, body.encode(), "text/plain; version=0.0.4")
            if self.path in ("/health", "/v1/health"):
                return self._json(200, {"status": "ok"})
            if self.path == "/v1/models":
                return self._json(200, {"object": "list", "data": [{"id": engine.model_name, "object": "model"}]})
            if self.path == "/v1/neuralese/info":
                from ..data.prompt_rendering import prompt_rendering_identity
                prompt_identity = prompt_rendering_identity(engine.prompt_rendering)
                return self._json(200, {"dialects": [engine.dialect], "width": engine.width,
                                        "dtype": str(engine.backbone.embedding_weight.dtype).replace("torch.", ""),
                                        "store": {"owners": True, "persistent": engine.store.directory is not None},
                                        "max_block_length": engine.max_block, "context": engine.context,
                                        "grad": True, "grad_order": 2, "stream": True,
                                        "native_state": {"enabled": True, "format":
                                                              ('neuralese.semantic-state/1' if getattr(engine, 'semantic_channel', None)
                                                               else 'neuralese.native-state/1'),
                                                              **({'semantic_adapter_sha256': engine.semantic_channel.adapter_sha256}
                                                                 if getattr(engine, 'semantic_channel', None) else {}),
                                                              "consumption_modes": ["exact-native-state",
                                                                                     "same-base-adapter-state",
                                                                                     "best-effort-text-retokenization"] +
                                                              (['learned-semantic-state-deltas'] if getattr(engine, 'semantic_channel', None) else []),
                                                              "architecture": engine.model_config.get("architecture"),
                                                              "checkpoint_sha256": engine.checkpoint_sha256,
                                                              "tokenizer_fingerprint": engine.model_config.get("tokenizer_fingerprint"),
                                                              "embedding_weight_sha256": engine.embedding_weight_sha256,
                                                              "role_marker_ids": list(engine.role_tokens.ids)},
                                        "server": "reference",
                                        **({"rl_policy_binding": engine.rl_policy_binding}
                                           if getattr(engine, "rl_policy_binding", None) else {}),
                                        "prompt_rendering": engine.prompt_rendering,
                                        "prompt_rendering_identity_sha256": prompt_identity["identity_sha256"],
                                        "capabilities": [c for c in NATIVE_CAPABILITIES
                                                         if c != "store.owners" or hasattr(engine.store, "hold")]})
            native = _NATIVE_STATE.match(self.path)
            if native and native.group(2) in (None, "/meta"):
                try:
                    record = engine.store.get_native_state(native.group(1))
                except ValueError as error:
                    return self._error(409, "neuralese-native-state-kind", str(error))
                if record is None:
                    return self._error(404, "neuralese-unknown-native-state", native.group(1))
                if native.group(2) == "/meta":
                    return self._json(200, record.meta())
                self._hold(record.id)
                return self._send(200, encode_native_state(record.block), "application/octet-stream")
            tensor = _NATIVE_TENSOR.match(self.path)
            if tensor and tensor.group(2) in (None, "/meta"):
                try:
                    record = engine.store.get_native_tensor_artifact(tensor.group(1))
                except ValueError as error:
                    return self._error(409, "neuralese-native-artifact-kind", str(error))
                if record is None:
                    return self._error(404, "neuralese-unknown-native-artifact", tensor.group(1))
                if tensor.group(2) == "/meta":
                    return self._json(200, record.meta())
                from ..model.native_artifact import encode_native_tensor_artifact
                self._hold(record.id)
                return self._send(200, encode_native_tensor_artifact(record.artifact), "application/octet-stream")
            self._error(404, "not-found", self.path)

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
            return self._guarded(self._put)

        def _put(self):
            native = _NATIVE_STATE.match(self.path)
            if native and native.group(2) is None:
                return self._put_native_state(native.group(1))
            tensor = _NATIVE_TENSOR.match(self.path)
            if tensor and tensor.group(2) is None:
                return self._put_native_tensor_artifact(tensor.group(1))
            return self._error(404, "not-found", self.path)

        def _put_native_state(self, expected_id):
            try:
                block, actual_id = decode_native_state(self._body())
            except ValueError as error:
                return self._error(400, "neuralese-bad-native-state", str(error))
            if actual_id != expected_id:
                return self._error(400, "neuralese-id-mismatch", f"content hashes to {actual_id}")
            try:
                stored = engine.store.put_native_state(block, self._owner)
            except ValueError as error:  # a failed derivation check or a content collision with a stored state
                return self._error(400, "neuralese-bad-native-state", str(error))
            return self._json(201, stored.meta())

        def _put_native_tensor_artifact(self, expected_id):
            from ..model.native_artifact import decode_native_tensor_artifact
            try:
                artifact, actual_id = decode_native_tensor_artifact(self._body())
            except ValueError as error:
                return self._error(400, "neuralese-bad-native-artifact", str(error))
            if actual_id != expected_id:
                return self._error(400, "neuralese-id-mismatch", f"content hashes to {actual_id}")
            try:
                stored = engine.store.put_native_tensor_artifact(artifact, self._owner)
            except (ValueError, TypeError) as error:  # a failed gradient-origin check or a content collision
                return self._error(400, "neuralese-bad-native-artifact", str(error))
            return self._json(201, stored.meta())

        def do_POST(self):
            try:
                if self.path == "/v1/neuralese/native-states/write-text":
                    body = self._object()
                    return self._json(201, engine.native_write_text(body.get("text"), value_type=body.get("type"), adapters=body.get('adapters')))
                if self.path == "/v1/neuralese/native-states/init-body":
                    body = self._object()
                    return self._json(201, engine.native_init_body(body.get("messages") or [], body.get("tools"),
                        body.get("placeholder"), body.get("text"), value_type=body.get("type"),
                        placeholder_occurrence=body.get("placeholder_occurrence"), adapters=body.get('adapters'), native_state_modes=body.get('native_state_modes')))
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
                    from .native_grad import native_graph_gradient, native_recorded_gradient
                    from .native_derivative import execute_derivative_program
                    with grad_lock:
                        if 'native_recorded_objective' in body:
                            return self._json(200, _grad_request(lambda: native_recorded_gradient(
                                engine, body['native_recorded_objective'], self._owner)))
                        if body.get('schema') == 'natlang.native-recorded-objective-request/1':
                            return self._json(200, _grad_request(lambda: native_recorded_gradient(engine, body, self._owner)))
                        if body.get('schema') == 'natlang.native-derivative-program/1':
                            return self._json(200, _grad_request(lambda: execute_derivative_program(engine, body, self._owner)))
                        if isinstance(body.get('native_derivative_program'), dict):
                            return self._json(200, _grad_request(lambda: execute_derivative_program(
                                engine, body['native_derivative_program'], self._owner)))
                        return self._json(200, native_graph_gradient(engine, body, self._owner))
                if self.path == "/v1/neuralese/decide":
                    body = self._object()
                    from .native_learning import decide_native
                    return self._json(200, decide_native(engine, body))
                if self.path in ("/v1/natlang/score", "/v1/neuralese/decide_many"):
                    body = self._object()
                    from .native_learning import decide_many_native
                    return self._json(200, decide_many_native(engine, body))
                if self.path == "/v1/neuralese/optim":
                    body = self._object()
                    from .native_optim import native_optimizer_step
                    with grad_lock:
                        return self._json(200, native_optimizer_step(engine, body, self._owner))
                if self.path == "/v1/neuralese/adapters":
                    body = self._object()
                    with grad_lock, engine.native_execution_lock:
                        from ..model.native_adapters import create_adapter
                        try:
                            return self._json(201, create_adapter(engine, body, self._owner).meta())
                        except (ValueError, TypeError, RuntimeError) as error:
                            raise RequestError('neuralese-native-adapter-invalid', str(error)) from error
                if self.path == "/v1/neuralese/view":
                    from ..view import INSTRUCTIONS, REPLY_TOKENS, PART_MARGIN, TEMPLATE, TOOLS, view_site, write_view

                    body = self._object()

                    class ViewModelFailure(Exception):
                        pass
                    source_turns = []
                    def write(messages):
                        # The template write of view's body: the reply forced to return_result, its value written.
                        request = GenerationRequest(messages=messages, tools=TOOLS, template=dict(TEMPLATE),
                                                    max_tokens=engine.max_block + REPLY_TOKENS,
                                                    temperature=1.0, rl_capture=True,
                                                    native_protocol_recovery=body.get('x_natlang_native_protocol_recovery') is True,
                                                    native_store_owner=self._owner)
                        response = engine.submit(request).result()
                        receipt = response.get("x_natlang_rl_capture")
                        if not isinstance(receipt, dict) or receipt.get("schema") not in {"natlang.native-rl-sampling/2", "natlang.native-rl-sampling/3", "natlang.native-rl-sampling/4"}:
                            raise RequestError("neuralese-view-source", "native view must retain its actual template sampling receipt")
                        source_turns.append({"messages": messages, "tools": TOOLS, "template": dict(TEMPLATE),
                                             "response": response})
                        if response.get("x_natlang_native_failure") is not None:
                            raise ViewModelFailure()
                        blocks = (response.get("native_state") or {}).get("blocks") or []
                        if not blocks:
                            raise RequestError("neuralese-view", "a view write produced no block")
                        return blocks[0]["id"]

                    instructions = body.get("instructions") or None
                    system = body.get("system") or INSTRUCTIONS
                    reserve = engine.max_block + REPLY_TOKENS + PART_MARGIN
                    available = engine.context - engine.prompt_positions(view_site(system, "", instructions), TOOLS) - reserve
                    requested = body.get("window")
                    if requested is not None and (type(requested) is not int or requested <= 0):
                        raise RequestError("neuralese-view-window", "view window must be a positive token count")
                    if available < 1:
                        raise RequestError("context-length", "view site and output reserve do not fit this model context")
                    window = min(available, requested) if requested is not None else available
                    fits = lambda messages: engine.prompt_positions(messages, TOOLS) + reserve < engine.context
                    try:
                        block, parts = write_view(write, system, body.get("value") or "", instructions, engine.tokenizer,
                                                  window, fits=fits)
                    except ViewModelFailure:
                        return self._json(200, {'x_natlang_native_failure':source_turns[-1]['response']['x_natlang_native_failure'],
                            'source_turns':source_turns,'window':window})
                    record = engine.store.get_native_state(block)
                    return self._json(201, {**record.meta(), "parts": parts, "window": window,
                                           "source_turns": source_turns})
                if self.path == "/v1/neuralese/guidance/check":
                    from .guidance import Guide

                    body = self._object()
                    g = body.get("guidance") or {}
                    guide = Guide(Settings(tools=g.get("tools") or None, repeat=int(g.get("repeat", 3)),
                                           syntax=bool(g.get("syntax", True)), run=int(g.get("run", 4)),
                                           call_format="xml" if getattr(engine, "tool_call_format", None) == "xml" else "pythonic"))
                    reply = body.get("reply") or ""
                    for end in range(1, len(reply) + 1):
                        verdict = guide.check(reply[:end])
                        if verdict:
                            return self._json(201, {"reason": verdict[0], "offset": verdict[1], "end": end})
                    return self._json(201, {"reason": None})
                if self.path == "/v1/neuralese/render":
                    from .chat import split_escaped

                    body = self._object()
                    rendered = engine.render_request(body.get("messages") or [], body.get("tools"))
                    prompt = "".join("<block>" if isinstance(segment, int) else
                                     "".join(run for run, _ in split_escaped(segment, rendered.escape_nonce))
                                     for segment in rendered.segments)
                    return self._json(201, {"prompt": prompt})
                native = _NATIVE_STATE.match(self.path)
                tensor = _NATIVE_TENSOR.match(self.path)
                pin_match = native or tensor
                if pin_match and pin_match.group(2) in ("/pin", "/unpin"):
                    if pin_match.group(2) == "/pin":
                        try:
                            engine.store.pin(pin_match.group(1), self._owner)
                        except KeyError:
                            return self._error(404, "neuralese-unknown-native-state" if native else
                                               "neuralese-unknown-native-artifact", pin_match.group(1))
                    else:
                        engine.store.unpin(pin_match.group(1), self._owner)
                    return self._json(200, {"ok": True})
                self._error(404, "not-found", self.path)
            except RequestError as error:
                self._error(400, error.code, str(error))
            except json.JSONDecodeError as error:
                self._error(400, "bad-json", str(error))
            except Exception as error:  # noqa: BLE001 - every failure answers the error envelope, as the fork's
                self._error(500, "internal", f"{type(error).__name__}: {error}")

        def _chat(self, body: dict):
            retired_fields = {'x_natlang_forced', 'neuralese_temperature', 'neuralese_max_length',
                              'neuralese_length', 'neuralese_passes'}
            supplied = retired_fields.intersection(body)
            if supplied:
                raise RequestError('unsupported-request-field',
                    'remove retired projected-write fields: ' + ', '.join(sorted(supplied)))
            if 'x_natlang_native_protocol_recovery' in body and type(body['x_natlang_native_protocol_recovery']) is not bool:
                raise RequestError('bad-request', 'x_natlang_native_protocol_recovery must be boolean')
            if 'x_natlang_forced_message' in body and (not isinstance(body['x_natlang_forced_message'],dict)
                    or body.get('x_natlang_native_recording') is not True):
                raise RequestError('bad-request','x_natlang_forced_message requires a bound teacher turn and native structural recording')
            # Captured sampled rollouts record the sampler's top-k log-probs by default (score-centered RL);
            # x_natlang_rl_top_k: 0 opts out.
            from ..common.sampling import MAX_BEHAVIOR_TOP_K, DEFAULT_BEHAVIOR_TOP_K
            rl_top_k=body.get('x_natlang_rl_top_k',
                              DEFAULT_BEHAVIOR_TOP_K if body.get('x_natlang_rl_capture') is True else None)
            if rl_top_k == 0 and type(rl_top_k) is int:
                rl_top_k = None
            if rl_top_k is not None:
                if type(rl_top_k) is not int or not 1 <= rl_top_k <= MAX_BEHAVIOR_TOP_K:
                    raise RequestError('bad-request', f'x_natlang_rl_top_k must be 0 or an integer in [1, {MAX_BEHAVIOR_TOP_K}]')
                if body.get('x_natlang_rl_capture') is not True:
                    raise RequestError('bad-request', 'x_natlang_rl_top_k records sampler top-k log-probs inside x_natlang_rl_capture; request both')
            privileged_context=body.get('x_natlang_privileged_context')
            if privileged_context is not None and body.get('x_natlang_rl_capture') is not True:
                raise RequestError('bad-request', 'x_natlang_privileged_context renders a captured JOLT+ teacher '
                                   'attempt; request x_natlang_rl_capture too')
            request = GenerationRequest(
                messages=body.get("messages") or [], tools=body.get("tools"),
                max_tokens=completion_limit(body),
                temperature=float(body.get("temperature") or 0.0), seed=body.get("seed"),
                template=body.get("neuralese_template"),
                forced_message=body.get('x_natlang_forced_message'),
                native_protocol_recovery=body.get('x_natlang_native_protocol_recovery') is True,
                native_store_owner=self._owner,
                rl_capture=bool(body.get('x_natlang_rl_capture', False) or body.get('x_natlang_native_recording', False)),
                rl_top_k=rl_top_k,
                privileged_context=privileged_context,
                native_recording=bool(body.get('x_natlang_native_recording', False)),
                native_state_modes=body.get('x_natlang_native_state_modes'),
                # Resolve this exact binding in Engine._native_run. Passing
                # even an empty list is meaningful: HTTP has already applied
                # the declared policy/defaults and must not inherit local
                # native_default_adapters a second time.
                adapters=body.get('x_natlang_adapters'),
                guidance=Settings.of(body.get("guidance", getattr(engine, "default_guidance", None)), body.get("tools"),
                                     body.get("tool_choice"), getattr(engine, "tool_call_format", None)))
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
            # Call markup is LFM2's Pythonic `<|tool_call_start|>` or `<tool_call>` (Qwen/Mellum JSON, LFM3/Qwen3.5 XML;
            # chat.build_message parses both forms after the stream ends).
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
                    block = item["native_state"]
                    self._event(chunk({"content": [{"type": "native_state", "id": block["id"]}]},
                                      extra={"native_state": {"block": block}}))
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
                extra = {"x_natlang_message": message, "usage": response["usage"]}
                if 'x_natlang_native_failure' in response:
                    extra['x_natlang_native_failure']=response['x_natlang_native_failure']
                if "native_state" in response:
                    extra["native_state"] = response["native_state"]
                self._event(chunk({}, choice["finish_reason"], {**extra,
                                                                **({'x_natlang_rl_capture': response['x_natlang_rl_capture']}
                                                                   if 'x_natlang_rl_capture' in response else {}),
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
