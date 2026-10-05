"""Rollouts on vLLM with Neuralese (S4 §7, first form): text on vLLM, writes on the reference port.

High-throughput generation for S5 and S7: vLLM decodes text for many sequences at once. Reads use vLLM's prompt
embeddings: a prompt with blocks is rendered exactly as the reference server renders it and embedded the same way
(token embeddings, and for each block the open marker, the port's read interface over the payload, the close marker:
`GradSession._embed_items`). Writes are rare, so they run on the reference port: vLLM stops a sequence at the
`<|neuralese|>` token, the write procedure runs on the exact context (prompt items and generated tokens, then the
open marker; `train.execution.unroll_write`, the same procedure the reference writer uses), the block is stored, and
the sequence continues on vLLM with the block in its prompt. A model-runner hook that writes inside vLLM (§7's
second form) would remove the re-prefill after each write.

Requests are dicts with `messages`, `tools`, `max_tokens`, `temperature`, `seed`, `neuralese_temperature`, `adapters`, and
optionally `forced`: a plan of text and `{"neuralese": "write"}` items forced at the start of the reply (as the
reference server's `x_natlang_forced`), for tests and template readout. Responses have the reference server's shape
(`choices[0].message` with block parts, `neuralese.blocks`).

Weight adapters (`adapters: [{"id", "scale"}]`, as the reference server's `x_natlang_adapters`): the adapters a
request binds become one LoRA (each `tiny`/`xs` adapter is exactly a rank-r LoRA; several concatenate into one of
the summed rank, scales folded in), exported once as a PEFT directory and served by vLLM's LoRA path; writes run on
the reference port under the same adapters. A phase-F adapter on the backbone must be merged into an HF checkpoint
first (`base`). Run inside the DGX's vLLM image (`vllm-node`), with this package on the path.
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
    def __init__(self, engine, llm, lora_dir: str | None = None):
        self.engine, self.llm = engine, llm
        self.session = GradSession(engine)
        self.open_id = engine.backbone.controls.open_id
        self.lora_dir = lora_dir
        self._loras: dict[str, object] = {}

    @classmethod
    def load(cls, base: str | None = None, heads: str | None = None, gpu_memory_utilization: float = 0.08,
             dtype: str = "float32", max_model_len: int = 8192, port_dtype: torch.dtype | None = None,
             max_lora_rank: int = 0, lora_dir: str | None = None):
        """`max_lora_rank` > 0 enables adapters (the summed rank of the adapters one request binds, at most)."""
        import tempfile

        from vllm import LLM

        from . import load_engine
        from ..model.lfm2_port import resolve_base

        engine = load_engine(base, heads_checkpoint=heads, device="cuda", dtype=port_dtype)
        lora = {"enable_lora": True, "max_lora_rank": max_lora_rank, "max_loras": 4} if max_lora_rank else {}
        llm = LLM(model=resolve_base(base), enable_prompt_embeds=True, dtype=dtype, max_model_len=max_model_len,
                  gpu_memory_utilization=gpu_memory_utilization, enforce_eager=False, **lora)
        return cls(engine, llm, lora_dir or (tempfile.mkdtemp(prefix="natlang-vllm-loras-") if max_lora_rank else None))

    # Adapters --------------------------------------------------------------------------------
    def _lora_request(self, adapters):
        """The request's adapters as one vLLM LoRA (exported once per set of adapters and scales), or None."""
        if not adapters:
            return None
        if self.lora_dir is None:
            raise ValueError("adapters need VllmNeuralese.load(max_lora_rank=...)")
        import json

        key = json.dumps(adapters, sort_keys=True)
        if key not in self._loras:
            from pathlib import Path

            from safetensors.torch import save_file
            from vllm.lora.request import LoRARequest

            bank = self.engine.adapter_bank
            factors: dict[str, list] = {}
            for spec, coefficients, scale in self.engine.resolve_adapters(list(adapters)):
                for module, (a, b) in bank.lora(spec, coefficients.detach().float()).items():
                    factors.setdefault(module, []).append((a.cpu(), (b.float() * scale).cpu()))
            rank = max(sum(a.shape[0] for a, _ in pairs) for pairs in factors.values())
            tensors = {}
            for module, pairs in factors.items():
                # Concatenated factors: Σ B_i A_i = [B_1 … B_k] [A_1; …; A_k]; zero-padded to one rank for all modules.
                a = torch.cat([a for a, _ in pairs], 0)
                b = torch.cat([b for _, b in pairs], 1)
                pad = rank - a.shape[0]
                tensors[f"base_model.model.{module}.lora_A.weight"] = torch.nn.functional.pad(a, (0, 0, 0, pad)).contiguous()
                tensors[f"base_model.model.{module}.lora_B.weight"] = torch.nn.functional.pad(b, (0, pad)).contiguous()
            out = Path(self.lora_dir) / f"set-{len(self._loras) + 1}"
            out.mkdir(parents=True, exist_ok=True)
            save_file(tensors, str(out / "adapter_model.safetensors"))
            (out / "adapter_config.json").write_text(json.dumps({
                "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": rank, "lora_alpha": rank, "lora_dropout": 0.0,
                "bias": "none", "target_modules": sorted({m.rsplit(".", 1)[-1] for m in factors})}))
            self._loras[key] = LoRARequest(f"natlang-{len(self._loras) + 1}", len(self._loras) + 1, str(out))
        return self._loras[key]

    # Prompts ---------------------------------------------------------------------------------
    def _prompt_items(self, request: dict) -> list:
        engine = self.engine
        rendered = render_messages(request.get("messages") or [], request.get("tools"), engine._template, engine.specials)
        return self.session._items(rendered.segments, rendered.blocks, rendered.escape_nonce)

    def _vllm_prompt(self, items: list):
        """Token IDs when the context holds no block; otherwise its embeddings, as the reference reads them."""
        if all(kind == "tok" for kind, _ in items):
            return {"prompt_token_ids": [value for _, value in items]}
        with torch.no_grad():
            embeds = self.session._embed_items(items, {})[0]
        return {"prompt_embeds": embeds.to(self.llm.llm_engine.model_config.dtype).cpu()}

    # Writes ----------------------------------------------------------------------------------
    def _write(self, items: list, tau: float, seed: int, request_id: str, index: int, adapters=None,
               max_length: int | None = None):
        """The reference write procedure (`write.write_block`) after `items` and the open marker, under the request's
        adapters; returns the stored block."""
        from ..write import Opened, write_block

        backbone, heads = self.engine.backbone, self.engine.heads
        with torch.no_grad(), self.session._adapted(adapters, {}):
            embeds = self.session._embed_items(items + [("tok", self.open_id)], {})
            out = backbone.forward_embeds(embeds, cutoff=heads.cutoff, logits=False)
            opened = Opened(cache=out["cache"], h_cut=out["h_cut"][:, -1], logits=None)
            written = write_block(backbone, heads, opened, max_length=max_length or self.engine.max_block,
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
                    elif not self._append_write(state):
                        state["done"], state["finish"] = True, "length"
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
            loras = [self._lora_request(s["request"].get("adapters")) for s in active]
            outputs = self.llm.generate([self._vllm_prompt(s["items"]) for s in active], params, use_tqdm=False,
                                        **({"lora_request": loras} if any(loras) else {}))
            for state, output in zip(active, outputs):
                completion = output.outputs[0]
                ids = [t for t in completion.token_ids]
                opened = completion.stop_reason == self.open_id or (ids and ids[-1] == self.open_id)
                if ids and ids[-1] == self.open_id:
                    ids = ids[:-1]
                state["items"] += [("tok", i) for i in ids]
                state["out"] += [("tok", i) for i in ids]
                state["budget"] -= len(ids)  # the open marker is counted with its block
                if opened and self._append_write(state):
                    continue
                state["done"] = True
                state["finish"] = "length" if opened or completion.finish_reason == "length" or state["budget"] <= 0 else "stop"
        return [self._response(state) for state in states]

    def _append_write(self, state: dict) -> bool:
        """Write a block at the sequence's end; False when the budget cannot hold open, one vector and close (the
        reference engine then finishes the reply at "length"). The block is capped as the reference caps it."""
        request = state["request"]
        if state["budget"] < 3:
            return False
        index = len(state["blocks"])
        seed = derive_seed("neuralese", request.get("seed") if request.get("seed") is not None else state["id"], index)
        limit = min(int(request.get("neuralese_max_length") or self.engine.max_block), self.engine.max_block, state["budget"] - 2)
        block = self._write(state["items"], float(request.get("neuralese_temperature") or 0.0), seed, state["id"], index,
                            request.get("adapters"), max_length=limit)
        state["blocks"].append(block)
        state["items"].append(("block", block.id))
        state["out"].append(("block", block.id))
        state["budget"] -= block.length + 2
        return True

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
