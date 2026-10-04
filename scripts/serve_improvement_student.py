#!/usr/bin/env python3
"""Serve a pinned CPU student and optional LoRA checkpoint without altering other model servers."""

import argparse
import contextlib
import json
import select
import socket
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer

from projection_scoring import completion_ids, score_completion, validate_sampling_config

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

if __package__:
    from .render_training_corpus import _call_template, RENDERER_VERSION
    from .model_response import parse_response, tool_calls
    from .student_serving import (
        assistant_end_token_id,
        strip_final_assistant_terminator,
        response_finish_reason,
        bounded_output_limit,
        sampling_options,
        checkpoint_file_hashes,
        tokenize_chat_prompt,
    )
else:
    from render_training_corpus import _call_template, RENDERER_VERSION
    from model_response import parse_response, tool_calls
    from student_serving import (
        assistant_end_token_id,
        strip_final_assistant_terminator,
        response_finish_reason,
        bounded_output_limit,
        sampling_options,
        checkpoint_file_hashes,
        tokenize_chat_prompt,
    )


def _event(event, request_id, started_at, **fields):
    """Emit a small lifecycle record; never include prompts, bodies, or credentials."""
    record = {
        "event": event,
        "request_id": request_id,
        "started_unix_ms": started_at,
        "observed_unix_ms": int(time.time() * 1000),
        "elapsed_ms": max(0, int((time.monotonic() - _request_monotonic[request_id]) * 1000)),
    }
    record.update(fields)
    print(json.dumps(record, separators=(",", ":")), file=sys.stderr, flush=True)


_request_monotonic = {}


def _begin_request():
    request_id = uuid.uuid4().hex
    started_at = int(time.time() * 1000)
    _request_monotonic[request_id] = time.monotonic()
    return request_id, started_at


def _finish_request(request_id):
    _request_monotonic.pop(request_id, None)


class ClientDisconnectStoppingCriteria(StoppingCriteria):
    """Poll for a closed HTTP peer without reading or consuming socket bytes."""

    def __init__(self, connection, poll_every_tokens=4):
        self.connection = connection
        self.poll_every_tokens = max(1, int(poll_every_tokens))
        self.disconnected = False
        self._last_polled_length = -1
        self._prompt_length = 0

    def poll(self):
        if self.disconnected:
            return True
        try:
            readable, _, exceptional = select.select([self.connection], [], [self.connection], 0)
            if exceptional:
                self.disconnected = True
                return True
            if not readable:
                return False
            # A zero-length peek means FIN. Buffered bytes are left untouched;
            # this criterion only observes closure and never consumes a request.
            pending = self.connection.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT)
            self.disconnected = pending == b""
            return self.disconnected
        except (BlockingIOError, InterruptedError):
            return self.disconnected
        except (OSError, ValueError):
            self.disconnected = True
            return True

    def __call__(self, input_ids, scores, **kwargs):
        current_length = int(input_ids.shape[-1])
        if current_length != self._last_polled_length:
            generated = max(0, current_length - self._prompt_length)
            if generated == 0 or generated % self.poll_every_tokens == 0:
                self.poll()
            self._last_polled_length = current_length
        return torch.full(
            (input_ids.shape[0],), self.disconnected, dtype=torch.bool, device=input_ids.device
        )

    def set_prompt_length(self, prompt_length):
        self._prompt_length = int(prompt_length)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="LiquidAI/LFM2.5-350M")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--checkpoint-map")
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--max-context", type=int, default=8192)
    parser.add_argument("--max-output-tokens", type=int, default=2048,
                        help="hard server-side cap on generated tokens, regardless of request")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--cuda-memory-fraction", type=float, default=1.0)
    parser.add_argument("--host", default="127.0.0.1",
                        help="bind address; explicitly opt in to non-loopback serving")
    args = parser.parse_args()

    adapter_hashes = checkpoint_file_hashes(args.adapter)
    torch.set_num_threads(args.threads)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, local_files_only=True)
    assistant_eos_id = assistant_end_token_id(tokenizer)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else assistant_eos_id
    if args.load_in_4bit and args.device != "cuda":
        raise ValueError("4-bit loading requires CUDA")
    if args.device == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction)
    options = {"dtype": torch.bfloat16 if args.device == "cuda" else torch.float32}
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        options.update(device_map={"": 0}, quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        ))
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, local_files_only=True, **options
    )
    if not args.load_in_4bit:
        model = model.to(args.device)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    selected = {"name": "starting" if not args.adapter else "default"}
    checkpoints = {}
    if args.checkpoint_map:
        with open(args.checkpoint_map) as handle:
            checkpoints = json.load(handle)
        from peft import PeftModel
        adapters = [(name, path) for name, path in checkpoints.items() if path]
        for index, (name, path) in enumerate(adapters):
            if index == 0 and not args.adapter:
                model = PeftModel.from_pretrained(model, path, adapter_name=name)
            else:
                model.load_adapter(path, adapter_name=name)
        selected["name"] = "starting"
    model.eval()
    if checkpoint_file_hashes(args.adapter) != adapter_hashes:
        raise ValueError("adapter changed while loading")
    loaded_identity = {
        "base_model": args.model,
        "revision": args.revision,
        "adapter": args.adapter,
        "weight_pins": adapter_hashes,
        "checkpoint_map": bool(args.checkpoint_map),
        "prompt_tokenization": "chat-template-single-bos/1",
        "projection_scoring": "closed-assistant-temperature-logprob/1",
        "assistant_terminator_id": assistant_eos_id,
    }

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def send(self, status, data):
            payload = json.dumps(data).encode()
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                self.wfile.flush()
                return True
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                return False

        def do_GET(self):
            if self.path == "/natlang/student-identity":
                self.send(200, loaded_identity)
            else:
                self.send(200, {"data": [{"id": args.adapter or args.model, "object": "model"}]})

        def do_POST(self):
            request_id, started_at = _begin_request()
            disconnect_criteria = ClientDisconnectStoppingCriteria(self.connection)
            request = None
            prompt_length = None
            output_limit = None
            completion_token_count = None
            finish_reason = None
            try:
                body_length = int(self.headers["Content-Length"])
                request = json.loads(self.rfile.read(body_length))
                _event("request_received", request_id, started_at,
                       body_bytes=body_length,
                       endpoint="checkpoint_select" if self.path == "/select" else "completion")
                if disconnect_criteria.poll():
                    _event("client_disconnected", request_id, started_at,
                           request_kind="checkpoint_select" if self.path == "/select" else "completion",
                           response_sent=False, cancellation="before_prompt_render")
                    return
                if self.path == "/select":
                    name = request["checkpoint"]
                    if name not in checkpoints:
                        raise ValueError("checkpoint is not in the frozen experiment map")
                    if checkpoints[name]:
                        model.set_adapter(name)
                    selected["name"] = name
                    response_sent = self.send(200, {"checkpoint": name})
                    if not response_sent:
                        disconnect_criteria.disconnected = True
                    _event("client_disconnected" if disconnect_criteria.disconnected else "request_completed", request_id, started_at,
                           request_kind="checkpoint_select", response_sent=response_sent,
                           client_disconnected=disconnect_criteria.disconnected)
                    return

                if self.path == "/natlang/score":
                    if checkpoints:
                        raise ValueError("projection scoring requires one immutable adapter")
                    messages, tools = request["messages"], request.get("tools") or []
                    ids = request.get("completion_token_ids")
                    if ids is None:
                        ids = completion_ids(tokenizer, messages, tools, request["assistant"], assistant_eos_id)
                    if len(ids) > args.max_output_tokens:
                        raise ValueError("completion exceeds projection output budget")
                    score = score_completion(model, tokenizer, messages, tools, ids,
                                             request.get("score_temperature", 1.0),
                                             args.max_context, assistant_eos_id)
                    response_sent = self.send(200, {**score, "completion_token_ids": ids})
                    _event("request_completed", request_id, started_at, request_kind="score",
                           scored_tokens=len(ids), response_sent=response_sent)
                    return
                if self.path != "/v1/chat/completions":
                    raise ValueError("unsupported POST endpoint")
                if request.get("natlang_projection") and (checkpoints or request.get("temperature", 0) <= 0):
                    raise ValueError("projection requires a fixed adapter and stochastic sampling")
                if request.get("natlang_projection"):
                    validate_sampling_config(model.generation_config)
                messages = request["messages"]
                prompt = _call_template(tokenizer, messages, request.get("tools") or [], True)
                inputs = tokenize_chat_prompt(tokenizer, prompt, return_tensors="pt").to(model.device)
                prompt_length = int(inputs.input_ids.shape[1])
                output_limit = bounded_output_limit(request, args.max_output_tokens)
                context_remaining = max(0, int(args.max_context) - prompt_length)
                if prompt_length + output_limit > args.max_context:
                    _event("request_rejected", request_id, started_at,
                           request_kind="completion", prompt_tokens=prompt_length,
                           output_token_limit=output_limit, context_limit=args.max_context,
                           context_remaining=context_remaining, input_truncated=False,
                           reason="context_allowance_exceeded")
                    raise ValueError("context allowance exceeded; input was not truncated")
                if disconnect_criteria.poll():
                    _event("client_disconnected", request_id, started_at,
                           request_kind="completion", prompt_tokens=prompt_length,
                           output_token_limit=output_limit, context_limit=args.max_context,
                           context_remaining=context_remaining, input_truncated=False,
                           response_sent=False, cancellation="before_generation")
                    return

                sampling, seed = sampling_options(request)
                devices = [model.device.index or 0] if model.device.type == "cuda" else []
                disconnect_criteria.set_prompt_length(prompt_length)
                generation_started = time.monotonic()
                with (torch.random.fork_rng(devices=devices) if seed is not None else contextlib.nullcontext()), \
                        torch.inference_mode(), \
                        (model.disable_adapter() if checkpoints and selected["name"] == "starting" else contextlib.nullcontext()):
                    if seed is not None:
                        torch.manual_seed(seed)
                    output = model.generate(
                        **inputs,
                        max_new_tokens=output_limit,
                        stopping_criteria=StoppingCriteriaList([disconnect_criteria]),
                        **sampling,
                        repetition_penalty=1.0,
                        pad_token_id=pad_token_id,
                        eos_token_id=assistant_eos_id,
                    )
                generate_ms = max(0, int((time.monotonic() - generation_started) * 1000))
                tokens = output[0, prompt_length:]
                completion_token_count = int(len(tokens))
                if disconnect_criteria.disconnected:
                    _event("client_disconnected", request_id, started_at,
                           request_kind="completion", prompt_tokens=prompt_length,
                           generated_tokens=completion_token_count, output_token_limit=output_limit,
                           context_limit=args.max_context, context_remaining=context_remaining,
                           input_truncated=False, generate_ms=generate_ms,
                           prefill_timing_available=False, response_sent=False,
                           cancellation="stopping_criteria")
                    return

                response_tokens, terminated = strip_final_assistant_terminator(tokens, assistant_eos_id)
                text = tokenizer.decode(response_tokens, skip_special_tokens=False)
                try:
                    decoded = parse_response(text, request.get("tools") or [])
                except (ValueError, SyntaxError):
                    decoded = {"content": text, "tool_calls": [], "reasoning_content": None}
                calls = decoded["tool_calls"]
                content = decoded["content"]
                response_id = "student_" + request_id
                finish_reason = response_finish_reason(
                    terminated=terminated,
                    token_count=completion_token_count,
                    output_limit=output_limit,
                    has_tool_calls=bool(calls),
                )
                response_sent = self.send(200, {
                    "id": response_id,
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": args.adapter or args.model,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": content or None,
                            **({"reasoning_content": decoded["reasoning_content"]} if decoded["reasoning_content"] else {}),
                            **({"tool_calls": calls} if calls else {}),
                        },
                        "finish_reason": finish_reason,
                    }],
                    **({"natlang_projection": {
                        "completion_token_ids": tokens.tolist(),
                        "terminated": terminated,
                        "temperature": request.get("temperature"),
                        "sampling": "temperature-only-top-k-0-top-p-1",
                    }} if request.get("natlang_projection") else {}),
                    "usage": {
                        "prompt_tokens": prompt_length,
                        "completion_tokens": completion_token_count,
                        "total_tokens": prompt_length + completion_token_count,
                    },
                })
                if not response_sent:
                    disconnect_criteria.disconnected = True
                _event("client_disconnected" if disconnect_criteria.disconnected else "request_completed",
                       request_id, started_at, request_kind="completion", response_id=response_id,
                       prompt_tokens=prompt_length, completion_tokens=completion_token_count,
                       total_tokens=prompt_length + completion_token_count, finish_reason=finish_reason,
                       output_token_limit=output_limit, context_limit=args.max_context,
                       context_remaining=context_remaining, input_truncated=False,
                       generate_ms=generate_ms, prefill_timing_available=False,
                       response_sent=response_sent, client_disconnected=disconnect_criteria.disconnected)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError) as error:
                disconnect_criteria.disconnected = True
                _event("client_disconnected", request_id, started_at,
                       request_kind="completion", prompt_tokens=prompt_length,
                       completion_tokens=completion_token_count, output_token_limit=output_limit,
                       response_sent=False, error_type=type(error).__name__)
            except Exception as error:
                if disconnect_criteria.disconnected or self.send(400, {"error": {"message": str(error)}}) is False:
                    _event("client_disconnected", request_id, started_at,
                           request_kind="completion", prompt_tokens=prompt_length,
                           completion_tokens=completion_token_count, output_token_limit=output_limit,
                           response_sent=False, error_type=type(error).__name__)
                else:
                    _event("request_failed", request_id, started_at,
                           request_kind="completion", prompt_tokens=prompt_length,
                           output_token_limit=output_limit, response_sent=True,
                           error_type=type(error).__name__)
            finally:
                _finish_request(request_id)

    print(json.dumps({
        "ready": True,
        "renderer": RENDERER_VERSION,
        "port": args.port,
        "model": args.model,
        "revision": args.revision,
        "adapter": args.adapter,
        "max_output_tokens": args.max_output_tokens,
    }), flush=True)
    # Keep the established single-thread CPU serving policy: one expensive
    # generation owns the model until it finishes or the peer disconnects.
    HTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
