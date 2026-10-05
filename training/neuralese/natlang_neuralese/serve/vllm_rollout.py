"""Rollouts on vLLM with Neuralese (S4 §7, first form): text on vLLM, writes on the reference port.

High-throughput generation for S5 and S7: vLLM decodes text for many sequences at once. Reads use vLLM's prompt
embeddings: a prompt with blocks is rendered exactly as the reference server renders it and embedded the same way
(token embeddings, and for each block the open marker, the port's read interface over the payload, the close marker:
`GradSession._embed_items`). Writes are rare, so they run on the reference port: vLLM stops a sequence at the
`<|neuralese|>` token, the write procedure runs on the exact context (prompt items and generated tokens, then the
open marker; `train.execution.unroll_write`, the same procedure the reference writer uses), the block is stored, and
the sequence continues on vLLM with the block in its prompt. A model-runner hook that writes inside vLLM (§7's
second form) would remove the re-prefill after each write.

Requests are dicts with `messages`, `tools`, `max_tokens`, `temperature`, `seed`, `neuralese_temperature`, and
optionally `forced`: a plan of text and `{"neuralese": "write"}` items forced at the start of the reply (as the
reference server's `x_natlang_forced`), for tests and template readout. Responses have the reference server's shape
(`choices[0].message` with block parts, `neuralese.blocks`).

vLLM serves the base weights: a LoRA or phase-F adapter must be merged into an HF checkpoint first (`base`).
Run inside the DGX's vLLM image (`vllm-node`), with this package on the path.
"""

from __future__ import annotations

import time
import uuid

import torch

from .chat import build_message, placeholder, render_messages
from .grad import GradSession
from .engine import derive_seed
from .store import make_block


class VllmNeuralese:
    def __init__(self, engine, llm):
        self.engine, self.llm = engine, llm
        self.session = GradSession(engine)
        self.open_id = engine.backbone.controls.open_id

    @classmethod
    def load(cls, base: str | None = None, heads: str | None = None, gpu_memory_utilization: float = 0.08,
             dtype: str = "float32", max_model_len: int = 8192, port_dtype: torch.dtype | None = None):
        from vllm import LLM

        from . import load_engine
        from ..model.lfm2_port import resolve_base

        engine = load_engine(base, heads_checkpoint=heads, device="cuda", dtype=port_dtype)
        llm = LLM(model=resolve_base(base), enable_prompt_embeds=True, dtype=dtype, max_model_len=max_model_len,
                  gpu_memory_utilization=gpu_memory_utilization, enforce_eager=False)
        return cls(engine, llm)

    # Prompts ---------------------------------------------------------------------------------
    def _prompt_items(self, request: dict) -> list:
        engine = self.engine
        rendered = render_messages(request.get("messages") or [], request.get("tools"), engine._template, engine.specials)
        return self.session._items(rendered.segments, rendered.blocks)

    def _vllm_prompt(self, items: list):
        """Token IDs when the context holds no block; otherwise its embeddings, as the reference reads them."""
        if all(kind == "tok" for kind, _ in items):
            return {"prompt_token_ids": [value for _, value in items]}
        with torch.no_grad():
            embeds = self.session._embed_items(items, {})[0]
        return {"prompt_embeds": embeds.to(self.llm.llm_engine.model_config.dtype).cpu()}

    # Writes ----------------------------------------------------------------------------------
    def _write(self, items: list, tau: float, seed: int, request_id: str, index: int):
        """The reference write procedure (`write.write_block`) after `items` and the open marker; returns the stored
        block."""
        from ..write import Opened, write_block

        backbone, heads = self.engine.backbone, self.engine.heads
        with torch.no_grad():
            embeds = self.session._embed_items(items + [("tok", self.open_id)], {})
            out = backbone.forward_embeds(embeds, cutoff=heads.cutoff, logits=False)
            opened = Opened(cache=out["cache"], h_cut=out["h_cut"][:, -1], logits=None)
            written = write_block(backbone, heads, opened, max_length=self.engine.max_block,
                                  generator=torch.Generator().manual_seed(seed), temperature=tau)
        n = int(written.lengths[0])
        record = {"kind": "write", "request": request_id, "index": index, "cutoff": heads.cutoff, "temperature": tau,
                  "seed": seed, "engine": "vllm-rollout"}
        return self.engine.store.put(make_block(written.payload[0, :n], self.engine.dialect, producer=record,
                                                truncated=bool(written.truncated[0])))

    # Generation ------------------------------------------------------------------------------
    def generate(self, requests: list[dict], max_writes: int = 16) -> list[dict]:
        from vllm import SamplingParams

        tokenizer = self.engine.tokenizer
        states = []
        for request in requests:
            request_id = request.get("id") or f"chatcmpl-{uuid.uuid4().hex[:12]}"
            items = self._prompt_items(request)
            state = {"request": request, "id": request_id, "items": items, "prompt": len(items), "out": [],
                     "blocks": [], "budget": int(request.get("max_tokens") or 256), "done": False, "finish": "length",
                     "forced": list(request.get("forced") or [])}
            states.append(state)
        for _ in range(max_writes + 1):
            # Forced plans first: forced text is appended as tokens, a forced write is written here.
            for state in states:
                while not state["done"] and state["forced"]:
                    step = state["forced"].pop(0)
                    if isinstance(step, str):
                        ids = tokenizer.encode(step, add_special_tokens=False)
                        state["items"] += [("tok", i) for i in ids]
                        state["out"] += [("tok", i) for i in ids]
                        state["budget"] -= len(ids)
                    else:
                        self._append_write(state)
            active = [s for s in states if not s["done"] and s["budget"] > 0]
            if not active:
                break
            params = []
            for state in active:
                request = state["request"]
                seed = request.get("seed")
                params.append(SamplingParams(max_tokens=state["budget"], temperature=float(request.get("temperature") or 0.0),
                                             top_p=float(request.get("top_p") or 1.0), stop_token_ids=[self.open_id],
                                             seed=None if seed is None else derive_seed("vllm", seed, len(state["blocks"])),
                                             skip_special_tokens=False))
            outputs = self.llm.generate([self._vllm_prompt(s["items"]) for s in active], params, use_tqdm=False)
            for state, output in zip(active, outputs):
                completion = output.outputs[0]
                ids = [t for t in completion.token_ids]
                opened = completion.stop_reason == self.open_id or (ids and ids[-1] == self.open_id)
                if ids and ids[-1] == self.open_id:
                    ids = ids[:-1]
                state["items"] += [("tok", i) for i in ids]
                state["out"] += [("tok", i) for i in ids]
                state["budget"] -= len(ids) + (1 if opened else 0)
                if opened and state["budget"] > 0:
                    self._append_write(state)
                else:
                    state["done"] = True
                    state["finish"] = "length" if completion.finish_reason == "length" or state["budget"] <= 0 else "stop"
        return [self._response(state) for state in states]

    def _append_write(self, state: dict):
        request = state["request"]
        index = len(state["blocks"])
        seed = derive_seed("neuralese", request.get("seed") if request.get("seed") is not None else state["id"], index)
        block = self._write(state["items"], float(request.get("neuralese_temperature") or 0.0), seed, state["id"], index)
        state["blocks"].append(block)
        state["items"].append(("block", block.id))
        state["out"].append(("block", block.id))
        state["budget"] -= block.length + 2

    def _response(self, state: dict) -> dict:
        tokenizer = self.engine.tokenizer
        text, run, ids = [], [], []
        for kind, value in state["out"]:
            if kind == "block":
                if run:
                    text.append(tokenizer.decode(run, skip_special_tokens=False))
                    run = []
                text.append(placeholder(len(ids)))
                ids.append(value)
            else:
                run.append(value)
        if run:
            text.append(tokenizer.decode(run, skip_special_tokens=False))
        message = build_message("".join(text), ids, call_prefix=state["id"])
        finish = state["finish"]
        if finish == "stop" and message.get("tool_calls"):
            finish = "tool_calls"
        completion = sum(1 if kind == "tok" else self.engine.lookup(value).length + 2 for kind, value in state["out"])
        return {"id": state["id"], "object": "chat.completion", "created": int(time.time()), "model": self.engine.model_name,
                "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                "usage": {"prompt_tokens": state["prompt"], "completion_tokens": completion,
                          "total_tokens": state["prompt"] + completion},
                "neuralese": {"dialect": self.engine.dialect, "blocks": [b.meta() for b in state["blocks"]]}}
