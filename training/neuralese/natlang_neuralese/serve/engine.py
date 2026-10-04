"""Decoding with the Neuralese ports, one state machine per sequence (S4 §4.2).

A sequence alternates between ordinary text decoding and writing blocks. When the LM head picks `<|neuralese|>`,
the sequence runs the single write procedure (S3 §3.1; `write.write_block`) one step at a time: the stop head
decides before each new vector (masked before the first), the feedback projection gives the next sketch input, the
shallow layers advance, and the runtime's hard maximum forces closure and records truncation. On closure the upper
layers complete the block from the block-start cache, the content projection gives the payload distribution, the
payload is sampled at the Neuralese temperature (deterministic at 0), and read back: every layer is restored to the
block start and the payload plus `<|/neuralese|>` are prefilled before text decoding resumes.

The scheduler runs rounds: in each round every active sequence advances one position. Text steps of all sequences
run as one batched forward, and so do the shallow sketch steps of all sequences inside a block (`batch.step_rows`:
projections, convolutions and feed-forward batched, attention per row against each row's own cache, so rows of
different lengths need no padding). Prefill, block opening, completion and readback run per sequence.

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
from .batch import split_rows, step_rows
from .chat import RequestError, build_message, placeholder, render_messages, split_escaped
from .store import Block, TensorStore, make_block

WRITE = object()  # a forced-plan item: run the write procedure here


def derive_seed(*parts) -> int:
    digest = hashlib.sha256(repr(parts).encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**31)


class StepWriter:
    """`write.write_block` for one row, advanced one decision at a time."""

    def __init__(self, backbone: PortBackbone, heads: PortHeads, opened: Opened, max_length: int,
                 allow_empty: bool = False, lookahead: int = 4):
        self.backbone, self.heads, self.opened = backbone, heads, opened
        # Final stop source: decide every `lookahead` positions on completed states (see write.write_block).
        self.lookahead, self.upper, self.checked = max(1, lookahead), opened.cache, 0
        self.max_length = min(max_length, heads.max_length)
        self.allow_empty = allow_empty
        self.cache = opened.cache
        self.state = opened.h_cut
        self.count = 0
        self.sketches, self.shallow, self.stop_logits = [], [], []
        self.done = False
        self.truncated = False

    def decide(self) -> torch.Tensor | None:
        """The next decision: None once the block is closed, else the next sketch input ([1, d])."""
        if self.done:
            return None
        device = self.state.device
        if self.heads.stop_source == "final":
            return self._decide_lookahead()
        if self.count > 0 or self.allow_empty:
            logit = self.heads.stop(self.state, torch.full((1,), self.count, device=device, dtype=torch.long))
            if self.count > 0:
                self.stop_logits.append(float(logit[0]))
            if bool(logit[0] > 0):
                self.done = True
                return None
        if self.count == self.max_length:
            self.done, self.truncated = True, True
            return None
        return self.heads.feedback(self.state)

    def _decide_lookahead(self) -> torch.Tensor | None:
        if self.count and (self.count - self.checked >= self.lookahead or self.count == self.max_length):
            pending = torch.stack(self.shallow[self.checked:self.count], 1)
            final, self.upper = self.backbone.run_layers(pending, range(self.heads.cutoff, self.backbone.num_layers),
                                                         self.upper)
            counts = torch.arange(self.checked + 1, self.count + 1, device=pending.device)[None]
            logits = self.heads.stop(final, counts)[0]
            for j, c in enumerate(range(self.checked + 1, self.count + 1)):
                if c == self.max_length:
                    break
                self.stop_logits.append(float(logits[j]))
                if bool(logits[j] > 0):  # stop after c vectors: drop the positions written ahead
                    self.sketches, self.shallow, self.count = self.sketches[:c], self.shallow[:c], c
                    self.done = True
                    return None
            self.checked = self.count
        if self.count == self.max_length:
            self.done, self.truncated = True, True
            return None
        return self.heads.feedback(self.state)

    def advance(self, sketch: torch.Tensor, shallow: torch.Tensor, cache: PortCache):
        """Record one sketch position after its shallow layers ran (`shallow` is [1, d])."""
        self.sketches.append(sketch)
        self.shallow.append(shallow)
        self.state = shallow
        self.cache = cache
        self.count += 1

    def step(self) -> bool:
        """One decision; True once the block is closed."""
        sketch = self.decide()
        if sketch is None:
            return True
        h, cache = self.backbone.run_layers(sketch[:, None], range(0, self.heads.cutoff), self.cache)
        self.advance(sketch, h[:, 0], cache)
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
    on_delta: object = None  # streaming listener: called with {"text": …} or {"neuralese": meta}
    adapters: list | None = None  # [{"id": adapter block, "scale": 1.0}], active for every forward of the request


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
    pending: list = field(default_factory=list)  # undecoded token IDs while streaming
    adapters: list = field(default_factory=list)  # resolved (spec, coefficients, scale) of the request's adapters
    started: float = field(default_factory=time.perf_counter)


class Engine:
    def __init__(self, backbone: PortBackbone, heads: PortHeads, tokenizer, store: TensorStore, dialect: str,
                 max_block: int = 64, model_name: str = "natlang-neuralese", device: str = "cpu",
                 prefill_tokens: int = 8192, prefill_padding: bool = False):
        self.backbone, self.heads, self.tokenizer, self.store = backbone, heads, tokenizer, store
        # Padded-token budget of one batched prefill (new requests that arrive in the same round).
        self.prefill_tokens = prefill_tokens
        # Equal-length prompts batch exactly (same bits as alone, so block IDs reproduce). Left-padded packing of
        # different lengths is faster but not bit-identical; it is opt-in.
        self.prefill_padding = prefill_padding
        self.dialect, self.max_block, self.model_name, self.device = dialect, max_block, model_name, device
        self.width = backbone.config.hidden_size
        self.stop_ids = {i for i in (tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.eos_token_id) if i is not None}
        self._incoming: queue.Queue = queue.Queue()
        self._active: list[Sequence] = []
        self._thread: threading.Thread | None = None
        self._stopping = False
        self._ids = itertools.count()
        self._adapter_bank = None
        self.projections: dict = {}  # name → AdapterProjection (model/projections.py), from --projection

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
                self._round([sequence])
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
                self._round(list(self._active))
                self._active = [s for s in self._active if not s.future.done()]

    # Input ---------------------------------------------------------------------------------
    def _template(self, messages, tools):
        return self.tokenizer.apply_chat_template(messages, tools=tools or None, tokenize=False,
                                                  add_generation_prompt=True)

    def _tokens(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def _template_tokens(self, text: str, escape_nonce: str = "") -> list[int]:
        """Template text: structure as special tokens, escaped content runs as plain text (spec §3.3)."""
        ids: list[int] = []
        for run, escaped in split_escaped(text, escape_nonce):
            ids.extend(self.tokenizer(run, add_special_tokens=False, split_special_tokens=escaped)["input_ids"])
        return ids

    @property
    def specials(self) -> tuple:
        found = getattr(self, "_specials", None)
        if found is None:
            names = set(self.tokenizer.all_special_tokens)
            names.update(t.content for t in self.tokenizer.added_tokens_decoder.values() if t.special)
            names.update(("<|neuralese|>", "<|/neuralese|>"))
            found = self._specials = tuple(sorted(n for n in names if n))
        return found

    def lookup(self, block_id: str) -> Block:
        block = self.store.get(block_id)
        if block is None:
            raise RequestError("neuralese-unknown-block", f"{block_id} is not in this server's store; PUT it first")
        if block.dialect != self.dialect:
            raise RequestError("neuralese-dialect-mismatch",
                               f"{block_id} is in {block.dialect}, this server speaks {self.dialect}")
        return block

    # Adapters --------------------------------------------------------------------------------
    @property
    def adapter_bank(self):
        if self._adapter_bank is None:
            from ..model.tiny_adapters import AdapterBank

            self._adapter_bank = AdapterBank(self.backbone)
        return self._adapter_bank

    def lookup_adapter(self, block_id: str):
        """(spec, coefficients) of a stored adapter block for this backbone; installs its hooks."""
        from ..model.tiny_adapters import AdapterSpec, is_adapter_dialect

        block = self.store.get(block_id)
        if block is None:
            raise RequestError("neuralese-unknown-block", f"{block_id} is not in this server's store; PUT it first")
        if not is_adapter_dialect(block.dialect) or "#" in block.dialect:
            raise RequestError("neuralese-adapter", f"{block_id} is a {block.dialect} block, not an adapter")
        spec = AdapterSpec.parse(block.dialect)
        try:
            self.adapter_bank.check(spec, block.payload)
        except ValueError as error:
            raise RequestError("neuralese-adapter", str(error)) from error
        self.adapter_bank.install(spec)
        return spec, block.payload

    def resolve_adapters(self, adapters, leaves: dict | None = None) -> list:
        """[(spec, coefficients, scale)] for request entries `{"id", "scale"}` (or bare IDs); coefficient tensors are
        taken from `leaves` when given (gradient sessions)."""
        out = []
        for entry in adapters or []:
            if isinstance(entry, dict) and entry.get("code"):
                out.append(self._decoded_adapter(entry, leaves))
                continue
            block_id, scale = (entry, 1.0) if isinstance(entry, str) else (entry.get("id"), float(entry.get("scale", 1.0)))
            spec, coefficients = self.lookup_adapter(block_id)
            if leaves and block_id in leaves:
                coefficients = leaves[block_id]
            out.append((spec, coefficients.to(self.device), scale))
        return out

    def _decoded_adapter(self, entry: dict, leaves: dict | None):
        """An adapter given as a Neuralese code: `{"code": block, "projection": name, "scale"}` decodes through the
        named projection P (LEARNING_CONTINUUM §6.4). A code that is a gradient leaf gets its gradient through P."""
        from ..model.tiny_adapters import AdapterSpec

        name = entry.get("projection")
        projection = self.projections.get(name)
        if projection is None:
            raise RequestError("neuralese-projection", f"this server has no projection {name!r}; known: {sorted(self.projections)}")
        code = self.lookup(entry["code"])
        if code.dialect != projection.source_dialect:
            raise RequestError("neuralese-projection", f"projection {name} reads {projection.source_dialect}, not {code.dialect}")
        spec = AdapterSpec.parse(projection.target)
        if spec.base != self.adapter_bank.base_hash():
            raise RequestError("neuralese-projection", f"projection {name} decodes adapters for base {spec.base}")
        self.adapter_bank.install(spec)
        payload = leaves[entry["code"]] if leaves and entry["code"] in leaves else code.payload.to(self.device)
        coefficients = projection(payload.float())
        self.adapter_bank.check(spec, coefficients)
        return spec, coefficients, float(entry.get("scale", 1.0))

    def using(self, seqs):
        """Adapters of these sequences active for one batched forward (one row each, in order)."""
        from ..model.tiny_adapters import active

        return active([seq.adapters for seq in seqs])

    def prompt_embeddings(self, messages, tools) -> torch.Tensor:
        rendered = render_messages(messages, tools, self._template, self.specials)
        blocks = [self.lookup(i) for i in rendered.blocks]
        dtype = self.backbone.embedding_weight.dtype
        pieces = []
        for segment in rendered.segments:
            if isinstance(segment, str):
                ids = self._template_tokens(segment, rendered.escape_nonce)
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
    def _round(self, sequences: list[Sequence]):
        """One scheduler round: every sequence advances one position; text and sketch steps run batched."""
        phases = [(seq, seq.phase) for seq in sequences]
        text_rows, sketch_rows = [], []
        fresh = [seq for seq, phase in phases if phase == "prefill"]
        if fresh:
            self._prefill_all(fresh)
        for seq, phase in phases:
            try:
                if phase == "prefill":
                    continue
                elif phase == "text":
                    token = self._choose(seq)
                    if token is not None:
                        text_rows.append((seq, token))
                elif phase == "sketch":
                    with self.using([seq]):
                        sketch = seq.writer.decide()
                    if sketch is None:
                        self._close_block(seq)
                    else:
                        sketch_rows.append((seq, sketch))
            except Exception as error:  # a failing request must not stop the others
                self._fail(seq, error)
        if text_rows:
            self._batched(text_rows, self._text_batch)
        if sketch_rows:
            self._batched(sketch_rows, self._sketch_batch)

    def _fail(self, seq: Sequence, error: Exception):
        seq.phase = "done"
        if not seq.future.done():
            seq.future.set_exception(error)

    def _batched(self, rows, run):
        try:
            run(rows)
        except Exception as error:
            if len(rows) == 1:
                return self._fail(rows[0][0], error)
            for row in rows:  # isolate the failing request
                self._batched([row], run)

    def _text_batch(self, rows):
        ids = torch.tensor([[token] for _, token in rows], device=self.device)
        with self.using([seq for seq, _ in rows]):
            h, caches = step_rows(self.backbone, self.backbone.embed(ids), [seq.cache for seq, _ in rows],
                                  range(0, self.backbone.num_layers))
            logits = self.backbone.logits(h)[:, -1]
        for index, (seq, token) in enumerate(rows):
            seq.cache, seq.logits = caches[index], logits[index:index + 1]
            seq.items.append(token)
            seq.generated_positions += 1
            self._emit(seq, token=token)

    def _sketch_batch(self, rows):
        sketches = torch.cat([sketch for _, sketch in rows], 0)[:, None]
        with self.using([seq for seq, _ in rows]):
            h, caches = step_rows(self.backbone, sketches, [seq.writer.cache for seq, _ in rows],
                                  range(0, self.heads.cutoff))
        for index, (seq, sketch) in enumerate(rows):
            seq.writer.advance(sketch, h[index:index + 1, 0], caches[index])

    def _prefill_all(self, seqs: list[Sequence]):
        """Prefill new requests together: left-padded batches by length under a padded-token budget."""
        ready = []
        for seq in seqs:
            try:
                seq.adapters = self.resolve_adapters(seq.request.adapters)
                ready.append((seq, self.prompt_embeddings(seq.request.messages, seq.request.tools)))
            except Exception as error:
                self._fail(seq, error)
        ready.sort(key=lambda row: row[1].shape[1])
        group: list = []
        for row in ready:
            width = max([row[1].shape[1]] + [e.shape[1] for _, e in group])
            mixed = bool(group) and not self.prefill_padding and row[1].shape[1] != group[0][1].shape[1]
            if group and (mixed or width * (len(group) + 1) > self.prefill_tokens):
                self._prefill_group(group)
                group = []
            group.append(row)
        if group:
            self._prefill_group(group)

    def _prefill_group(self, group):
        if len(group) == 1:
            seq, embeds = group[0]
            try:
                with self.using([seq]):
                    out = self.backbone.forward_embeds(embeds)
                self._prefilled(seq, out["cache"], out["logits"][:, -1], int(embeds.shape[1]))
            except Exception as error:
                self._fail(seq, error)
            return
        try:
            width = max(e.shape[1] for _, e in group)
            pad = torch.tensor([width - e.shape[1] for _, e in group], device=self.device)
            embeds = torch.cat([torch.cat([e.new_zeros(1, width - e.shape[1], e.shape[2]), e], 1) for _, e in group], 0)
            padded = bool(pad.any())
            with self.using([seq for seq, _ in group]):
                out = self.backbone.forward_embeds(embeds, left_pad=pad if padded else None, logits=False)
                logits = self.backbone.logits(out["h_final"][:, -1:])[:, -1]
            caches = split_rows(out["cache"], pad)
        except Exception:
            for row in group:  # isolate a failing request
                self._prefill_group([row])
            return
        for index, (seq, e) in enumerate(group):
            try:
                self._prefilled(seq, caches[index], logits[index:index + 1], int(e.shape[1]))
            except Exception as error:
                self._fail(seq, error)

    def _prefill(self, seq: Sequence):
        self._prefill_all([seq])

    def _prefilled(self, seq: Sequence, cache, logits, positions: int):
        request = seq.request
        seq.cache, seq.logits = cache, logits
        seq.prompt_positions = positions
        seq.forced = self._forced_plan(request.forced) if request.forced is not None else None
        seq.rng = torch.Generator().manual_seed(
            derive_seed("text", request.seed if request.seed is not None else request.request_id))
        seq.phase = "text"

    def _emit(self, seq: Sequence, token: int | None = None, block: Block | None = None):
        """Streaming: hand new output to the request's listener, if any."""
        listener = seq.request.on_delta
        if listener is None:
            return
        if token is not None:
            text = self.tokenizer.decode(seq.pending + [token], skip_special_tokens=False)
            if "\ufffd" in text:  # an incomplete UTF-8 sequence; wait for the next token
                seq.pending.append(token)
                return
            seq.pending = []
            listener({"text": text})
        elif block is not None:
            listener({"neuralese": block.meta()})

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
        if seq.request.on_delta is not None and seq.pending:
            seq.request.on_delta({"text": self.tokenizer.decode(seq.pending, skip_special_tokens=False)})
            seq.pending = []
        seq.finish_reason = reason
        seq.phase = "done"
        seq.future.set_result(self._response(seq))

    def _choose(self, seq: Sequence) -> int | None:
        """Pick the next text token. Returns it for the batched forward, or None if the sequence finished or opened
        a block (opening runs here, on its own)."""
        request = seq.request
        if seq.generated_positions >= request.max_tokens:
            self._finish(seq, "length")
            return None
        token = self._next_token(seq)
        if token is None or token in self.stop_ids:
            self._finish(seq, "stop")
            return None
        if token == self.backbone.controls.open_id:
            remaining = request.max_tokens - seq.generated_positions
            # A nonempty block needs open + at least one vector + close. Do not
            # emit a dangling opener when the completion allowance cannot fit it.
            if remaining < 3:
                self._finish(seq, "length")
                return None
            ids = torch.tensor([[token]], device=self.device)
            with self.using([seq]):
                out = self.backbone.forward_ids(ids, cache=seq.cache, cutoff=self.heads.cutoff)
            opened = Opened(cache=out["cache"], h_cut=out["h_cut"][:, -1], logits=out["logits"][:, -1])
            limit = request.neuralese_max_length or self.max_block
            seq.writer = StepWriter(self.backbone, self.heads, opened, min(limit, self.max_block, remaining - 2))
            seq.generated_positions += 1
            seq.phase = "sketch"
            return None
        return token

    def _close_block(self, seq: Sequence):
        writer, request = seq.writer, seq.request
        index = len(seq.blocks)
        tau = float(request.neuralese_temperature or 0.0)
        seed = derive_seed("neuralese", request.seed if request.seed is not None else request.request_id, index)
        generator = torch.Generator().manual_seed(seed)
        with self.using([seq]):
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
        self._emit(seq, block=block)
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
