"""Decoding with the Neuralese ports, one state machine per sequence (S4 §4.2).

A sequence alternates between ordinary text decoding and writing blocks. When the LM head picks `<|neuralese|>`,
the sequence runs the single write procedure (S3 §3.1; `write.write_block`) one step at a time: the stop head
decides before each new vector (masked before the first), the feedback projection gives the next sketch input, the
shallow layers advance, and the runtime's hard maximum forces closure and records truncation. On closure the upper
layers complete the block from the block-start cache, the content projection gives the payload distribution, the
payload is sampled at the Neuralese temperature (deterministic at 0), and read back: every layer is restored to the
block start and the payload plus `<|/neuralese|>` are prefilled before text decoding resumes.

The scheduler steps every active sequence in turn, so requests are admitted while others decode and any of them can
be in the middle of a block. Each step is one batch-1 forward; batching rows of different lengths into one tensor
needs a padded cache and is not done yet.

`StepWriter` mirrors `write.write_block` exactly; tests check they agree.
"""

from __future__ import annotations

import hashlib
import itertools
import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field

import torch

from ..model.heads import PortHeads, sample_payload
from ..model.lfm2_port import PortBackbone, PortCache
from ..write import Opened, read_back
from .chat import RequestError, build_message, placeholder, render_messages
from .store import Block, TensorStore, make_block

WRITE = object()  # a forced-plan item: run the write procedure here


def derive_seed(*parts) -> int:
    digest = hashlib.sha256(repr(parts).encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**31)


class StepWriter:
    """`write.write_block` for one row, advanced one decision at a time."""

    def __init__(self, backbone: PortBackbone, heads: PortHeads, opened: Opened, max_length: int,
                 allow_empty: bool = False):
        self.backbone, self.heads, self.opened = backbone, heads, opened
        self.max_length = min(max_length, heads.max_length)
        self.allow_empty = allow_empty
        self.cache = opened.cache
        self.state = opened.h_cut
        self.count = 0
        self.sketches, self.shallow, self.stop_logits = [], [], []
        self.done = False
        self.truncated = False

    def step(self) -> bool:
        """One decision; True once the block is closed."""
        if self.done:
            return True
        device = self.state.device
        if self.count > 0 or self.allow_empty:
            logit = self.heads.stop(self.state, torch.full((1,), self.count, device=device, dtype=torch.long))
            if self.count > 0:
                self.stop_logits.append(float(logit[0]))
            if bool(logit[0] > 0):
                self.done = True
                return True
        if self.count == self.max_length:
            self.done, self.truncated = True, True
            return True
        sketch = self.heads.feedback(self.state)
        h, self.cache = self.backbone.run_layers(sketch[:, None], range(0, self.heads.cutoff), self.cache)
        self.sketches.append(sketch)
        self.shallow.append(h[:, 0])
        self.state = h[:, 0]
        self.count += 1
        return False

    def complete(self, temperature: float, generator: torch.Generator | None):
        """Upper layers, content distribution, payload sample. Returns (payload, mu, log_sigma) as [1, L, d]."""
        if not self.count:
            empty = self.opened.h_cut.new_zeros(1, 0, self.opened.h_cut.shape[-1])
            return empty, empty, empty.float()
        sketches = torch.stack(self.sketches, 1)
        shallow = torch.stack(self.shallow, 1)
        final, _ = self.backbone.run_layers(shallow, range(self.heads.cutoff, self.backbone.num_layers),
                                            self.opened.cache)
        mu, log_sigma = self.heads.content.distribution(sketches, final)
        sampled = sample_payload(mu, log_sigma, temperature, generator)
        return sampled.payload, mu, log_sigma


@dataclass
class GenerationRequest:
    messages: list
    tools: list | None = None
    max_tokens: int = 512
    temperature: float = 0.0
    seed: int | None = None
    neuralese_temperature: float = 0.0
    neuralese_max_length: int | None = None
    forced: list | None = None  # test hook: text strings and {"neuralese": "write"} items
    request_id: str = ""


@dataclass
class Sequence:
    request: GenerationRequest
    future: Future
    phase: str = "prefill"
    cache: PortCache | None = None
    logits: torch.Tensor | None = None
    items: list = field(default_factory=list)  # generated token IDs and Block objects
    writer: StepWriter | None = None
    blocks: list = field(default_factory=list)  # written blocks with their write records
    prompt_positions: int = 0
    generated_positions: int = 0
    forced: list | None = None
    finish_reason: str = "stop"
    rng: torch.Generator | None = None
    started: float = field(default_factory=time.perf_counter)


class Engine:
    def __init__(self, backbone: PortBackbone, heads: PortHeads, tokenizer, store: TensorStore, dialect: str,
                 max_block: int = 64, model_name: str = "natlang-neuralese", device: str = "cpu"):
        self.backbone, self.heads, self.tokenizer, self.store = backbone, heads, tokenizer, store
        self.dialect, self.max_block, self.model_name, self.device = dialect, max_block, model_name, device
        self.width = backbone.config.hidden_size
        self.stop_ids = {i for i in (tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.eos_token_id) if i is not None}
        self._incoming: queue.Queue = queue.Queue()
        self._active: list[Sequence] = []
        self._thread: threading.Thread | None = None
        self._stopping = False
        self._ids = itertools.count()

    # Lifecycle -----------------------------------------------------------------------------
    def start(self):
        self._thread = threading.Thread(target=self._loop, name="neuralese-engine", daemon=True)
        self._thread.start()

    def stop(self):
        self._stopping = True
        self._incoming.put(None)
        if self._thread:
            self._thread.join(timeout=10)

    def submit(self, request: GenerationRequest) -> Future:
        future: Future = Future()
        request.request_id = request.request_id or f"chatcmpl-{next(self._ids)}"
        self._incoming.put(Sequence(request, future))
        return future

    def generate(self, request: GenerationRequest) -> dict:
        """Run one request to completion on the calling thread (no scheduler)."""
        sequence = Sequence(request, Future())
        with torch.inference_mode():
            while not sequence.future.done():
                self._step(sequence)
        return sequence.future.result()

    def _loop(self):
        with torch.inference_mode():
            while not self._stopping:
                block = not self._active
                try:
                    while True:
                        item = self._incoming.get(block=block)
                        block = False
                        if item is None:
                            if self._stopping:
                                return
                            continue
                        self._active.append(item)
                except queue.Empty:
                    pass
                for sequence in list(self._active):
                    try:
                        self._step(sequence)
                    except Exception as error:  # a failing request must not stop the others
                        if not sequence.future.done():
                            sequence.future.set_exception(error)
                    if sequence.future.done():
                        self._active.remove(sequence)

    # Input ---------------------------------------------------------------------------------
    def _template(self, messages, tools):
        return self.tokenizer.apply_chat_template(messages, tools=tools or None, tokenize=False,
                                                  add_generation_prompt=True)

    def _tokens(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def lookup(self, block_id: str) -> Block:
        block = self.store.get(block_id)
        if block is None:
            raise RequestError("neuralese-unknown-block", f"{block_id} is not in this server's store; PUT it first")
        if block.dialect != self.dialect:
            raise RequestError("neuralese-dialect-mismatch",
                               f"{block_id} is in {block.dialect}, this server speaks {self.dialect}")
        return block

    def prompt_embeddings(self, messages, tools) -> torch.Tensor:
        rendered = render_messages(messages, tools, self._template)
        blocks = [self.lookup(i) for i in rendered.blocks]
        dtype = self.backbone.embedding_weight.dtype
        pieces = []
        for segment in rendered.segments:
            if isinstance(segment, str):
                ids = self._tokens(segment)
                if ids:
                    pieces.append(self.backbone.embed(torch.tensor([ids], device=self.device)))
            else:
                payload = blocks[segment].payload.to(self.device, dtype)
                pieces.append(self.backbone.embed(torch.tensor([[self.backbone.controls.open_id]], device=self.device)))
                pieces.append(self.heads.interface(payload)[None].to(dtype))
                pieces.append(self.backbone.embed(torch.tensor([[self.backbone.controls.close_id]], device=self.device)))
        return torch.cat(pieces, 1)

    def _forced_plan(self, forced) -> list:
        plan = []
        for item in forced:
            if isinstance(item, str):
                plan.extend(self._tokens(item))
            elif isinstance(item, dict) and item.get("neuralese") == "write":
                plan.append(WRITE)
            else:
                raise RequestError("forced-plan", f"unsupported forced item {item!r}")
        return plan

    # Decoding ------------------------------------------------------------------------------
    def _step(self, seq: Sequence):
        if seq.phase == "prefill":
            request = seq.request
            embeds = self.prompt_embeddings(request.messages, request.tools)
            out = self.backbone.forward_embeds(embeds)
            seq.cache, seq.logits = out["cache"], out["logits"][:, -1]
            seq.prompt_positions = int(embeds.shape[1])
            seq.forced = self._forced_plan(request.forced) if request.forced is not None else None
            seq.rng = torch.Generator().manual_seed(
                derive_seed("text", request.seed if request.seed is not None else request.request_id))
            seq.phase = "text"
            return
        if seq.phase == "text":
            self._text_step(seq)
            return
        if seq.phase == "sketch":
            if seq.writer.step():
                self._close_block(seq)
            return

    def _next_token(self, seq: Sequence):
        if seq.forced is not None:
            if not seq.forced:
                return None
            item = seq.forced.pop(0)
            return self.backbone.controls.open_id if item is WRITE else item
        logits = seq.logits[0].float()
        if seq.request.temperature and seq.request.temperature > 0:
            probs = torch.softmax(logits / seq.request.temperature, -1).cpu()
            return int(torch.multinomial(probs, 1, generator=seq.rng))
        return int(logits.argmax())

    def _finish(self, seq: Sequence, reason: str):
        seq.finish_reason = reason
        seq.phase = "done"
        seq.future.set_result(self._response(seq))

    def _text_step(self, seq: Sequence):
        request = seq.request
        if seq.generated_positions >= request.max_tokens:
            return self._finish(seq, "length")
        token = self._next_token(seq)
        if token is None or token in self.stop_ids:
            return self._finish(seq, "stop")
        ids = torch.tensor([[token]], device=self.device)
        if token == self.backbone.controls.open_id:
            out = self.backbone.forward_ids(ids, cache=seq.cache, cutoff=self.heads.cutoff)
            opened = Opened(cache=out["cache"], h_cut=out["h_cut"][:, -1], logits=out["logits"][:, -1])
            limit = request.neuralese_max_length or self.max_block
            seq.writer = StepWriter(self.backbone, self.heads, opened, min(limit, self.max_block))
            seq.generated_positions += 1
            seq.phase = "sketch"
            return
        out = self.backbone.forward_ids(ids, cache=seq.cache)
        seq.cache, seq.logits = out["cache"], out["logits"][:, -1]
        seq.items.append(token)
        seq.generated_positions += 1

    def _close_block(self, seq: Sequence):
        writer, request = seq.writer, seq.request
        index = len(seq.blocks)
        tau = float(request.neuralese_temperature or 0.0)
        seed = derive_seed("neuralese", request.seed if request.seed is not None else request.request_id, index)
        generator = torch.Generator().manual_seed(seed)
        payload, mu, log_sigma = writer.complete(tau, generator)
        back = read_back(self.backbone, self.heads, writer.opened.cache, payload)
        seq.cache, seq.logits = back["cache"], back["logits"]
        record = {"kind": "write", "request": request.request_id, "index": index, "cutoff": self.heads.cutoff,
                  "temperature": tau, "seed": seed, "stop_logits": writer.stop_logits}
        if writer.count:
            mean = self.store.put(make_block(mu[0], self.dialect, producer={"kind": "payload-mean"}))
            scale = self.store.put(make_block(log_sigma[0], self.dialect, producer={"kind": "payload-log-sigma"}))
            record.update(mean=mean.id, log_sigma=scale.id)
        block = self.store.put(make_block(payload[0], self.dialect, producer=record, truncated=writer.truncated))
        seq.blocks.append(block)
        seq.items.append(block)
        seq.generated_positions += writer.count + 1
        seq.writer = None
        seq.phase = "text"
        # A forced plan's own close marker, if any, is written by the procedure instead.
        if seq.forced and seq.forced[0] == self.backbone.controls.close_id:
            seq.forced.pop(0)

    # Output --------------------------------------------------------------------------------
    def _response(self, seq: Sequence) -> dict:
        text, run, ids = [], [], []
        for item in seq.items:
            if isinstance(item, Block):
                if run:
                    text.append(self.tokenizer.decode(run, skip_special_tokens=False))
                    run = []
                text.append(placeholder(len(ids)))
                ids.append(item.id)
            else:
                run.append(item)
        if run:
            text.append(self.tokenizer.decode(run, skip_special_tokens=False))
        message = build_message("".join(text), ids, call_prefix=seq.request.request_id)
        finish = seq.finish_reason
        if finish == "stop" and message.get("tool_calls"):
            finish = "tool_calls"
        return {
            "id": seq.request.request_id, "object": "chat.completion", "created": int(time.time()),
            "model": self.model_name,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": seq.prompt_positions, "completion_tokens": seq.generated_positions,
                      "total_tokens": seq.prompt_positions + seq.generated_positions},
            "neuralese": {"dialect": self.dialect, "blocks": [b.meta() for b in seq.blocks]},
        }
