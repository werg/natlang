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
import json
import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field


def decode_prefix_tokens(tokenize, prefix: str) -> list[int]:
    """Force only the token prefix shared with JSON continuations.

    Cutting text at ``value=`` can manufacture an out-of-distribution token: the
    trained continuation starts with a merged ``=[`` or ``={``. Leave that final
    boundary token (and any differing suffix) to normal decoding instead.
    """
    tokens = tokenize(prefix)
    continuations = ['[]', '{}', '\"x\"', "'x'", 'true', 'false', 'null', '-1'] + list('0123456789')
    for continuation in continuations:
        other = tokenize(prefix + continuation)
        n = 0
        while n < min(len(tokens), len(other)) and tokens[n] == other[n]:
            n += 1
        tokens = tokens[:n]
    if not tokens:
        raise ValueError('chat template has no stable decode prefix')
    return tokens

import torch

from ..model.heads import PortHeads, sample_payload
from ..model.lfm2_port import PortBackbone, PortCache
from ..write import Opened, read_back
from .batch import split_rows, step_rows
from .chat import RequestError, build_message, placeholder, render_messages, split_escaped
from .store import Block, TensorStore, make_block
from .native_protocol import capture_enabled

WRITE = object()  # a forced-plan item: run the write procedure here


def derive_seed(*parts) -> int:
    digest = hashlib.sha256(repr(parts).encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**31)


class StepWriter:
    """`write.write_block` for one row, advanced one decision at a time."""

    def __init__(self, backbone: PortBackbone, heads: PortHeads, opened: Opened, max_length: int,
                 allow_empty: bool = False, lookahead: int = 4, length: int | None = None):
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
        # A length hint: write exactly this many vectors, no stop decisions (never required; the stop head decides
        # without one).
        self.length = None if length is None else max(1, min(length, self.max_length))
        self.passes = 0  # block-wise passes run (0: written position by position)

    def decide(self) -> torch.Tensor | None:
        """The next decision: None once the block is closed, else the next sketch input ([1, d])."""
        if self.done:
            return None
        device = self.state.device
        if self.length is not None:
            if self.count == self.length:
                self.done = True
                return None
            return self.heads.feedback(self.state)
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

    def blockwise(self, passes: int | None = None, tol: float = 0.0):
        """Write the hinted length at once (execution.blockwise_sketch) instead of position by position."""
        from ..train.execution import blockwise_sketch

        inputs, shallow, self.passes = blockwise_sketch(self.backbone, self.heads, self.opened.h_cut, self.opened.cache,
                                                        self.length, passes, tol)
        self.sketches = list(inputs.unbind(1))
        self.shallow = list(shallow.unbind(1))
        self.count, self.done = self.length, True

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
        mu, log_sigma = self.heads.content.distribution(sketches, self.heads.payload_states(final, self.opened.top))
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
    # Optional size hint: write exactly this many vectors (capped by the maximum). With `neuralese_passes` the block is
    # written block-wise in that many parallel passes (exact when passes >= length) instead of position by position.
    neuralese_length: int | None = None
    neuralese_passes: int | None = None
    forced: list | None = None  # test hook: text strings and {"neuralese": "write"} items
    forced_message: dict | None = None  # exact assistant target from an authenticated teacher turn
    # Template readout: the reply is forced to a call of `call` with `arguments`, up to `argument` (default
    # "value") or its nested `argument_path`; with "value": "write" that exact value is a written block, with
    # "decode" the value and the rest of the reply are decoded.
    template: dict | None = None
    guidance: object = None  # serve.guidance.Settings: envelope and line checks with backtracking, or None
    request_id: str = ""
    on_delta: object = None  # streaming listener: called with {"text": …} or {"neuralese": meta}
    adapters: list | None = None  # [{"id": adapter block, "scale": 1.0}], active for every forward of the request
    # Set by the transport when the client has gone (a streamed reply's connection closed): the engine drops the
    # sequence at its next round and fails its future with RequestError("cancelled").
    cancelled: bool = False
    native_protocol_recovery: bool = False  # negotiated typed failed-turn response
    native_store_owner: str | None = None  # existing HTTP owner protects trace dependencies before reply
    rl_capture: bool = False  # optional exact rollout sampling receipt
    native_recording: bool = False  # structural receipt also permits greedy, without a sampled density
    native_state_modes: dict | None = None  # explicit input-ID keyed transport declarations


@dataclass
class Sequence:
    request: GenerationRequest
    future: Future
    phase: str = "prefill"
    prompt_plan: list | None = None  # host-only token IDs/block IDs, prepared before queueing
    cache: PortCache | None = None
    logits: torch.Tensor | None = None
    cut_state: torch.Tensor | None = None
    token_history: torch.Tensor | None = None  # current target segment, bounded TCP receptive field
    top_state: torch.Tensor | None = None  # top-layer state at the last position (raw writes: autoregressive payload 0)
    items: list = field(default_factory=list)  # generated token IDs and Block objects
    writer: StepWriter | None = None
    blocks: list = field(default_factory=list)  # written blocks with their write records
    prompt_positions: int = 0
    generated_positions: int = 0
    forced: list | None = None
    prefilled_forced: list = field(default_factory=list)  # the forced plan's leading tokens, prefilled with the prompt
    free_after_forced: bool = False  # template readout that decodes the value: sample once the forced plan is spent
    guide: object = None  # serve.guidance.Guide over this reply
    pieces: list = field(default_factory=list)  # decoded text of each item ("__nz" for a block), for the guide
    snapshots: list = field(default_factory=list)  # (item count, cache, logits, generated positions, cut state, top state)
    bans: dict = field(default_factory=dict)  # item index -> tokens rejected there
    rl_samples: list = field(default_factory=list)
    retries: dict = field(default_factory=dict)  # reply offset -> rollbacks to it
    finish_reason: str = "stop"
    rng: torch.Generator | None = None
    pending: list = field(default_factory=list)  # undecoded token IDs while streaming
    adapters: list = field(default_factory=list)  # resolved (spec, coefficients, scale) of the request's adapters
    started: float = field(default_factory=time.perf_counter)
    native_trace_enabled: bool = False
    native_protocol_failure: dict | None = None
    native_capture_chunks: list = field(default_factory=list)
    native_capture_ids: list[int] = field(default_factory=list)
    native_capture_open: bool = False
    native_imports: list = field(default_factory=list)
    native_rl_prompt: list = field(default_factory=list)
    native_rl_inputs: list = field(default_factory=list)
    native_rl_writes: list = field(default_factory=list)
    native_rl_policy: dict | None = None
    native_rl_adapter_binding: dict | None = None
    native_write_start_sample: int | None = None
    native_write_close_sample: int | None = None
    native_write_start_own: int | None = None
    native_write_open_sample: int | None = None
    native_own_position: int = 0
    native_sampling_seed: int | None = None
    native_forced_tokens: list[int] = field(default_factory=list)
    native_forced_replay: bool = False
    native_forced_source: dict | None = None
    native_forced_replay_prompt_sha256: str | None = None
    native_template_suffix: list[int] = field(default_factory=list)
    native_template_value_type: str | None = None
    native_item_positions: list[int] = field(default_factory=list)
    native_guidance_anchor: dict | None = None
    native_guidance_flushing: bool = False


class Engine:
    def __init__(self, backbone: PortBackbone, heads: PortHeads | None, tokenizer, store: TensorStore, dialect: str,
                 max_block: int = 64, model_name: str = "natlang-neuralese", device: str = "cpu",
                 prefill_tokens: int = 8192, prefill_padding: bool = False, context: int | None = None,
                 prompt_rendering: str = "full", *, shared_model=None, role_tokens=None, model_config=None,
                 checkpoint_sha256: str | None = None, embedding_weight_sha256: str | None = None):
        from .chat import bind_history_reasoning
        self.model_config = dict(model_config or {})
        # Explicit checkpoint profiles override the backbone default; old
        # checkpoints without this field retain their historical rendering.
        tokenizer = bind_history_reasoning(tokenizer,
            policy=self.model_config.get('history_reasoning_policy'),
            model_type=getattr(getattr(backbone, "config", None), "model_type", None))
        self.backbone, self.heads, self.tokenizer, self.store = backbone, heads, tokenizer, store
        self.shared_model, self.role_tokens = shared_model, role_tokens
        self.native_state_enabled = shared_model is not None
        self.native_execution_lock = threading.RLock()
        self.checkpoint_sha256 = checkpoint_sha256
        self.embedding_weight_sha256 = embedding_weight_sha256
        if self.native_state_enabled:
            self.native_identity = {'checkpoint_sha256': checkpoint_sha256,
                'tokenizer_fingerprint': self.model_config.get('tokenizer_fingerprint'),
                'embedding_weight_sha256': embedding_weight_sha256,
                'architecture': self.model_config.get('architecture') or self.backbone.config.model_type}
        if self.native_state_enabled:
            if heads is not None or role_tokens is None or not checkpoint_sha256 or not embedding_weight_sha256:
                raise ValueError("native serving requires role tokens and exact checkpoint/embedding identities, with no PortHeads")
        # Padded-token budget of one batched prefill (new requests that arrive in the same round).
        self.prefill_tokens = prefill_tokens
        # Equal-length prompts share positions. Different matrix batch shapes can
        # change low bits and content IDs; compare payloads numerically. Left-padded
        # packing additionally changes mask layout and remains opt-in.
        self.prefill_padding = prefill_padding
        self.dialect, self.max_block, self.model_name, self.device = dialect, max_block, model_name, device
        from ..data.prompt_rendering import PROMPT_RENDERINGS
        if prompt_rendering not in PROMPT_RENDERINGS:
            raise ValueError(f"unknown prompt_rendering {prompt_rendering!r}; expected one of {PROMPT_RENDERINGS}")
        self.prompt_rendering = prompt_rendering
        self.width = backbone.config.hidden_size
        # The served context (spec: info `context`): positions one request's prompt and reply may span. Default the
        # model's training context (max_position_embeddings); the fork's equivalent is min(n_ctx_train, -c). View plans
        # its write sites against it.
        config = getattr(getattr(backbone, "hf", None), "config", None) or backbone.config
        trained = int(getattr(config, "max_position_embeddings", None) or 32768)
        self.context = min(trained, int(context)) if context else trained
        self.stop_ids = {i for i in (tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.eos_token_id) if i is not None}
        self._incoming: queue.Queue = queue.Queue()
        self._active: list[Sequence] = []
        self._thread: threading.Thread | None = None
        self._stopping = False
        self._ids = itertools.count()
        self._native_decoded_tokens = 0  # completed generated-token forwards, including private spans
        self._adapter_bank = None
        self.projections: dict = {}  # name → AdapterProjection (model/projections.py), from --projection

    def progress_metrics(self):
        """Host-only scheduler observations; never block on actor execution/GPU."""
        active = tuple(self._active)
        return {'natlang_native_decode_tokens_total': self._native_decoded_tokens,
                'natlang_active_sequences': sum(seq.phase != 'done' for seq in active),
                'natlang_native_open_produce_spans': sum(
                    seq.phase != 'done' and seq.native_capture_open for seq in active),
                'natlang_queued_requests': self._incoming.qsize()}

    def token_logits(self, hidden, segments=None, *, last_only=False):
        from ..model.neuralese_projection import token_choice_logits
        return token_choice_logits(self.backbone, hidden, getattr(self, 'token_choice_projection', None),
                                   segments, last_only=last_only)

    def advance_token_readout(self, history, hidden, *, last_only=False):
        """Score new target states and retain only the current segment's causal suffix.

        A new reply seeds this with its last prefix state. Caller contexts are
        never silently included in the token-choice convolution's history.
        """
        projection = getattr(self, 'token_choice_projection', None)
        if projection is None:
            return self.token_logits(hidden, last_only=last_only), None
        from ..model.neuralese_projection import causal_conv_receptive_field
        states = hidden if history is None else torch.cat([history, hidden], 1)
        segments = torch.zeros(states.shape[:2], dtype=torch.long, device=states.device)
        logits = self.token_logits(states, segments, last_only=last_only)
        if not last_only:
            logits = logits[:, -hidden.shape[1]:]
        return logits, states[:, -causal_conv_receptive_field(projection):]

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
        # HTTP request threads can prepare CPU text while the scheduler advances
        # active GPU sequences. No embeddings, adapters or CUDA state cross here.
        try:
            plan = self._prompt_plan(request.messages, request.tools)
        except Exception as error:
            future.set_exception(error)
            return future
        self._incoming.put(Sequence(request, future, prompt_plan=plan))
        return future

    def generate(self, request: GenerationRequest) -> dict:
        """Run one request to completion on the calling thread (no scheduler)."""
        request.request_id = request.request_id or f"chatcmpl-{next(self._ids)}"
        sequence = Sequence(request, Future(), prompt_plan=self._prompt_plan(request.messages, request.tools))
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

    def render_request(self, messages, tools=None, template=None):
        """Render one request with the same generic/program header policy used by direct SFT."""
        from ..data.prompt_rendering import apply_prompt_rendering
        from .chat import render_messages

        # In program-header mode the natlang runtime sends no generic tools. A separate
        # client (for example an app or harness) can still supply its own tool schemas.
        messages, tools = apply_prompt_rendering(messages, tools, self.prompt_rendering,
                                                 keep_supplied_tools=True)
        return render_messages(messages, tools, template or self._template, self.specials,
                               block_type=self.block_value_type,
                               native_state=getattr(self, "native_state_enabled", False))

    def _tokens(self, text: str) -> list[int]:
        cache = getattr(self, '_token_cache', None)
        if cache is not None and cache.tokenizer is self.tokenizer:
            return cache.tokens(text)
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def _template_tokens(self, text: str, escape_nonce: str = "") -> list[int]:
        """Template text: structure as special tokens, escaped content runs as plain text (spec §3.3)."""
        ids: list[int] = []
        for run, escaped in split_escaped(text, escape_nonce):
            cache = getattr(self, '_token_cache', None)
            ids.extend(cache.tokens(run, escaped=escaped) if cache is not None and cache.tokenizer is self.tokenizer
                       else self.tokenizer(run, add_special_tokens=False, split_special_tokens=escaped)["input_ids"])
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
        if block.width != self.width:
            # As the fork: a block of another width cannot be read (or decoded, or differentiated) here.
            raise RequestError("neuralese-bad-block", f"{block_id} has width {block.width}, this server reads {self.width}")
        return block

    # Adapters --------------------------------------------------------------------------------
    @property
    def adapter_bank(self):
        if self._adapter_bank is None:
            from ..model.tiny_adapters import AdapterBank

            self._adapter_bank = AdapterBank.of(self.backbone)
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

    def _prompt_plan(self, messages, tools) -> list:
        """Render and tokenize on the caller's CPU thread; preserve segment boundaries."""
        rendered = self.render_request(messages, tools)
        plan = []
        for segment in rendered.segments:
            if isinstance(segment, str):
                plan.append(("tokens", self._template_tokens(segment, rendered.escape_nonce)))
            else:
                kind = (getattr(rendered, 'block_kinds', None) or
                        ["neuralese"] * len(rendered.blocks))[segment]
                if kind=='native_state' and rendered.block_stop_gradients and rendered.block_stop_gradients[segment]:
                    kind='native_state_constant'
                value = rendered.blocks[segment]
                if kind in ('native_state','native_state_constant') and rendered.block_source_occurrences and rendered.block_source_occurrences[segment]:
                    value = {'id': value, 'source_occurrence': rendered.block_source_occurrences[segment]}
                plan.append((kind if kind in ('native_state','native_state_constant') else 'block', value))
        return plan

    def prompt_positions(self, messages, tools) -> int:
        """Positions the rendered prompt takes in the cache (its `usage.prompt_tokens`), without running the model:
        text tokens, and per block its vectors (with the read markers when the heads add them)."""
        if getattr(self, 'native_state_enabled', False):
            positions = 0
            for kind, value in self._prompt_plan(messages, tools):
                if kind == 'tokens':
                    positions += len(value)
                elif kind in ('native_state','native_state_constant'):
                    record = self.store.get_native_state(value['id'] if isinstance(value,dict) else value)
                    if record is None:
                        raise RequestError('neuralese-unknown-native-state', f'{value} is not in this server store; restore it first')
                    positions += 2 + (len(record.block.token_ids) if self._native_exact(record.block)
                                      else len(self._tokens(record.block.text)))
                else:
                    raise RequestError('neuralese-native-state-unavailable', 'native prompts cannot consume projected blocks')
            return positions
        markers = 2 if self.heads.read_markers else 0
        return sum(len(value) if kind == "tokens" else int(self.lookup(value).length) + markers
                   for kind, value in self._prompt_plan(messages, tools))

    def prompt_embeddings(self, messages, tools, *, block_mode="port", prepared=None) -> torch.Tensor:
        if block_mode not in {"port", "transparent"}:
            raise RequestError("neuralese-read-mode", "block mode must be port or transparent")
        plan = self._prompt_plan(messages, tools) if prepared is None else prepared
        # Resolve every block before doing any GPU work, as in the unprepared path.
        blocks = {value: self.lookup(value) for kind, value in plan if kind == "block"}
        dtype = self.backbone.embedding_weight.dtype
        pieces = []
        for kind, value in plan:
            if kind == "tokens":
                if value:
                    pieces.append(self.backbone.embed(torch.tensor([value], device=self.device)))
            else:
                payload = blocks[value].payload.to(self.device, dtype)
                if block_mode == "transparent":
                    pieces.append(payload[None].to(dtype))
                else:
                    pieces.append(self.heads.read_embeddings(self.backbone, payload[None]))
        return torch.cat(pieces, 1)

    def block_value_type(self, block_id):
        block = self.store.get(block_id)
        if block is not None and block.type == "Neuralese<unknown>":
            return "unknown"
        return None

    def _template_plan(self, template: dict) -> tuple[list, bool]:
        """A template readout's forced plan (chat.call_reply), and whether decoding follows it."""
        from .chat import call_reply, write_reply

        mode = template.get("value")
        if not isinstance(template.get("call"), str) or mode not in ("write", "decode"):
            raise RequestError("neuralese-template", "a template needs a call name and value 'write' or 'decode'")
        apply = lambda messages, generation: self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=generation)
        argument_path = template.get("argument_path")
        if argument_path is not None and (not isinstance(argument_path, list) or
                any(not isinstance(part, str) and type(part) is not int for part in argument_path)):
            raise RequestError("neuralese-template", "argument_path must be a list of string or integer keys")
        prefix, suffix = call_reply(apply, template["call"], template.get("arguments") or {},
                                    template.get("argument") or "value", quoted=mode == "write",
                                    argument_path=argument_path)
        if mode == "decode":
            return decode_prefix_tokens(self._tokens, prefix), True
        prefix, suffix = write_reply(apply, template["call"], template.get("arguments") or {},
                                     template.get("argument") or "value", template.get("value_type", "string"),
                                     argument_path=argument_path)
        return self._forced_plan([prefix, {"neuralese": "write"}, suffix]), False

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

    def _native_run(self, callback, adapters=None):
        from ..model.native_adapters import run_native, NativeAdapterError
        try:
            return run_native(self, callback, adapters)
        except NativeAdapterError as error:
            raise RequestError('neuralese-native-adapter-invalid', str(error)) from error

    @property
    def execution_native_identity(self):
        from ..model.native_adapters import execution_identity
        return execution_identity(self)

    def _native_forward(self, ids, cache, *, producing=False):
        return self._native_run(lambda: self._native_forward_impl(ids, cache, producing=producing))

    def _native_forward_impl(self, ids, cache, *, producing=False):
        from ..model.shared_state import native_step
        token_ids = torch.tensor([ids], dtype=torch.long, device=self.backbone.embedding_weight.device)
        return native_step(self.shared_model, token_ids, cache, producing=producing)

    def _native_prefill_forward(self, ids, cache):
        """Bound ordinary prefill workspace while preserving every positional state.

        Only the final token's hidden state is needed for next-token decoding.
        Producer captures use the original forward path and keep their full span.
        """
        if not ids:
            raise ValueError('native prefill needs at least one token')
        if type(self.prefill_tokens) is not int or self.prefill_tokens <= 0:
            raise ValueError('native prefill_tokens must be a positive integer')
        for start in range(0, len(ids), self.prefill_tokens):
            output = self._native_forward(ids[start:start+self.prefill_tokens], cache)
            cache = output['cache']
            if start + self.prefill_tokens < len(ids):
                # Do not retain the preceding chunk's raw captures/hidden state
                # while allocating the next chunk's layer workspace.
                del output
        return output

    def _native_logits(self, hidden):
        return self._native_run(lambda: self.shared_model.logits(hidden[:, -1]))

    def _native_exact(self, block):
        architecture = self.model_config.get("architecture") or self.backbone.config.model_type
        layer_kinds = tuple("attention" if self.backbone.is_attention(index) else "finite_convolution"
                             for index in range(self.backbone.num_layers))
        try:
            block.require_native(checkpoint_sha256=self.execution_native_identity["checkpoint_sha256"],
                                 tokenizer_fingerprint=self.model_config["tokenizer_fingerprint"],
                                 embedding_weight_sha256=self.embedding_weight_sha256,
                                 architecture=architecture, layer_kinds=layer_kinds)
            return True
        except ValueError:
            return False

    def _native_consume(self, seq, block_id, *, stop_gradient=False, source_occurrence=None):
        def consume():
            begin,len_imports=len(seq.native_rl_prompt),len(seq.native_imports)
            result=self._native_consume_impl(seq,block_id,stop_gradient=stop_gradient)
            use = {**({'stop_gradient': True} if stop_gradient else {}), **({'source_occurrence': source_occurrence} if source_occurrence else {})}
            for segment in seq.native_rl_prompt[begin:]:
                if segment['kind']=='native_state':segment.update(use)
            for receipt in seq.native_imports[len_imports:]:receipt.update(use)
            return result
        return self._native_run(consume, seq.request.adapters)

    def _native_consume_impl(self, seq, block_id, *, stop_gradient=False):
        from ..model.shared_state import splice_block
        record = self.store.get_native_state(block_id)
        if record is None:
            raise RequestError("neuralese-unknown-native-state", f"{block_id} is not in this server's store; PUT it first")
        block = record.block
        if block.derivation is not None:
            from ..model.native_state_derivation import verify_state_derivation
            verify_state_derivation(self.store,block)
        exact = self._native_exact(block)
        adaptation = None
        state_receipt = None
        declared_mode = (seq.request.native_state_modes or {}).get(block_id)
        if declared_mode is not None:
            from ..model.native_adapters import adapter_state_input_receipt, SAME_BASE_STATE_MODE
            if declared_mode != SAME_BASE_STATE_MODE:
                raise RequestError('neuralese-native-state-mode', 'use same-base-adapter-state only for exact-base raw native state exchange')
            try:
                state_receipt = adapter_state_input_receipt(self, block, block_id,
                    target_position_start=(int(seq.cache.lengths[0]) if seq.cache is not None else 0) + 1)
            except ValueError as error:
                raise RequestError('neuralese-native-state-mode', str(error)) from error
        observed_id = block_id if exact else None
        open_out = self._native_forward([self.role_tokens.consume_open], seq.cache)
        seq.cache = open_out["cache"]
        if capture_enabled(seq):
            seq.native_rl_prompt.append({"kind": "tokens", "token_ids": [self.role_tokens.consume_open],
                                         "producing": [False]})
        seq.native_own_position += 1
        if state_receipt is not None:
            # Validate and splice with the authenticated SOURCE identity. The
            # target actor stays unchanged; the source artifact is never relabelled.
            source = state_receipt['source_native_identity']
            seq.cache = splice_block(self.shared_model, seq.cache, block,
                checkpoint_sha256=source['checkpoint_sha256'],
                tokenizer_fingerprint=source['tokenizer_fingerprint'],
                embedding_weight_sha256=source['embedding_weight_sha256'], stop_gradient=stop_gradient)
            mode = 'same-base-adapter-state'
            observed_id = block_id
            if capture_enabled(seq):
                seq.native_rl_prompt.append({'kind': 'native_state', 'id': block_id,
                    'native_identity': source, 'adapter_state_consumption': state_receipt,
                    **({'state_derivation':block.derivation} if block.derivation is not None else {})})
        elif exact:
            seq.cache = splice_block(self.shared_model, seq.cache, block,
                                     checkpoint_sha256=self.execution_native_identity["checkpoint_sha256"],
                                     tokenizer_fingerprint=self.model_config["tokenizer_fingerprint"],
                                     embedding_weight_sha256=self.embedding_weight_sha256, stop_gradient=stop_gradient)
            mode = "exact-native-state"
            if capture_enabled(seq):
                seq.native_rl_prompt.append({"kind": "native_state", "id": block_id,
                    "native_identity": self._native_block_identity(block),
                    **({'state_derivation':block.derivation} if block.derivation is not None else {})})
        elif (getattr(self, 'semantic_channel', None) is not None and
              self.execution_native_identity in self.semantic_channel.bindings.values() and
              self._native_block_identity(block) in self.semantic_channel.bindings.values()):
            try:
                adapted, adaptation = self.semantic_channel.adapt(block, target_name=self.semantic_participant,
                    target_model=self.shared_model, target_tokenizer=self.tokenizer, target_identity=self.execution_native_identity,
                    allow_nearest=True,source_state_id=block_id)
            except ValueError as error:
                raise RequestError('neuralese-semantic-adapter-mismatch',
                    f'{error}; restore the matching model/adapter export or write the text with the target native writer') from error
            seq.cache = splice_block(self.shared_model, seq.cache, adapted,
                                     checkpoint_sha256=self.execution_native_identity["checkpoint_sha256"],
                                     tokenizer_fingerprint=self.model_config['tokenizer_fingerprint'],
                                     embedding_weight_sha256=self.embedding_weight_sha256, stop_gradient=stop_gradient)
            mode = 'learned-semantic-state-deltas'
            if capture_enabled(seq):
                adapted_record = self.store.put_native_state(adapted, owner=seq.request.native_store_owner)
                observed_id = adapted_record.id
                seq.native_rl_prompt.append({'kind': 'native_state', 'id': adapted_record.id,
                    'source_state_id': block_id, 'semantic_adaptation': adaptation,
                    'native_identity': self._native_block_identity(adapted),
                    **({'state_derivation':adapted.derivation} if adapted.derivation is not None else {})})
        else:
            # Until joint semantic adaptation is trained, foreign state is
            # best-effort only: discard its tensors and rebuild target KV from text.
            fallback_ids = self._tokens(block.text)
            if not fallback_ids and len(block.token_ids):
                raise RequestError("neuralese-native-state-empty-text", f"{block_id} has no text for target reencoding")
            if fallback_ids:
                output = self._native_prefill_forward(fallback_ids, seq.cache)
                seq.cache = output["cache"]
            mode = "best-effort-text-retokenization"
            if capture_enabled(seq):
                seq.native_rl_prompt.append({"kind": "tokens", "token_ids": fallback_ids,
                                             "producing": [False] * len(fallback_ids)})
            seq.native_own_position += len(fallback_ids)
        close_out = self._native_forward([self.role_tokens.consume_close], seq.cache)
        seq.cache = close_out["cache"]
        if capture_enabled(seq):
            seq.native_rl_prompt.append({"kind": "tokens", "token_ids": [self.role_tokens.consume_close],
                                         "producing": [False]})
        seq.native_own_position += 1
        seq.logits = self._native_logits(close_out["h_final"])
        seq.native_imports.append({"id": block_id, "mode": mode,
                                   **({'state_derivation':block.derivation} if block.derivation is not None else {}),
                                   **({'observed_native_state_id': observed_id} if observed_id is not None else {}),
                                   **({'semantic_adaptation': adaptation} if adaptation is not None else {}),
                                   **({'adapter_state_consumption': state_receipt} if state_receipt is not None else {})})

    @staticmethod
    def _native_block_identity(block):
        return {name: getattr(block, name) for name in
                ("checkpoint_sha256", "tokenizer_fingerprint", "embedding_weight_sha256", "architecture")}

    @staticmethod
    def _native_record_input(seq, token, *, producing, sample_index=None):
        """Record one actual own-token input; imported KV slots never advance this coordinate."""
        if capture_enabled(seq) or seq.guide is not None:
            seq.native_rl_inputs.append({'token_id': int(token),
                'input_role': 'produce' if producing else 'ordinary', 'sample_index': sample_index})
        seq.native_own_position += 1

    def _native_prefill(self, seq):
        return self._native_run(lambda: self._native_prefill_impl(seq), seq.request.adapters)

    @staticmethod
    def _validate_native_state_modes(modes, input_ids):
        if modes is not None and (not isinstance(modes, dict) or
                any(not isinstance(key, str) or key not in input_ids or value != 'same-base-adapter-state'
                    for key, value in modes.items())):
            raise RequestError('neuralese-native-state-mode',
                'native_state_modes must map actual native input IDs to same-base-adapter-state')

    def _native_prefill_impl(self, seq):
        seq.native_trace_enabled = seq.request.native_protocol_recovery
        self._validate_native_state_modes(seq.request.native_state_modes,
            {value['id'] if isinstance(value,dict) else value for kind, value in seq.prompt_plan or () if kind in ('native_state','native_state_constant')})
        if capture_enabled(seq):
            from ..model.native_adapters import execution_binding, SCHEMA
            seq.native_rl_policy = dict(self.execution_native_identity)
            bindings = execution_binding(self)
            seq.native_rl_adapter_binding = ({'schema': SCHEMA, 'base_native_identity': dict(self.native_identity),
                'adapters': bindings} if bindings else None)
        if seq.request.forced or seq.request.neuralese_length is not None or seq.request.neuralese_passes is not None:
            raise RequestError("native-write-controls-unsupported", "native shared-state serving uses sampled role markers, not legacy forced writes")
        if seq.request.neuralese_max_length is not None or seq.request.neuralese_temperature:
            raise RequestError("native-write-controls-unsupported", "native shared-state spans have no legacy write length or temperature control")
        if seq.request.rl_capture and seq.request.temperature <= 0 and not seq.request.native_recording:
            raise RequestError("native-rl-sampling-law", "native rollout capture requires positive temperature")
        seq.cache = None
        seq.native_sampling_seed = (int(seq.request.seed) if seq.request.seed is not None else
                                    derive_seed("native-chat", seq.request.request_id))
        seq.rng = torch.Generator(device="cpu").manual_seed(seq.native_sampling_seed)
        template_prefix = template_suffix = None
        if seq.request.template:
            from .chat import call_reply, write_reply
            template = seq.request.template
            mode = template.get('value')
            if not isinstance(template.get('call'), str) or mode not in ('write', 'decode'):
                raise RequestError('neuralese-template', "a template needs a call name and value 'write' or 'decode'")
            argument_path = template.get('argument_path')
            if argument_path is not None and (not isinstance(argument_path, list) or
                    any(not isinstance(part, str) and type(part) is not int for part in argument_path)):
                raise RequestError('neuralese-template', 'argument_path must contain string or integer keys')
            apply = lambda messages, generation: self.tokenizer.apply_chat_template(
                messages, tools=seq.request.tools or None, tokenize=False, add_generation_prompt=generation)
            arguments = template.get('arguments') or {}
            argument = template.get('argument') or 'value'
            if mode == 'decode':
                prefix, suffix = call_reply(apply, template['call'], arguments, argument,
                    quoted=False, argument_path=argument_path)
                template_prefix, template_suffix = self._tokens(prefix), []
            else:
                value_type = template.get('value_type', 'string')
                prefix, suffix = write_reply(apply, template['call'], arguments, argument, value_type,
                                             argument_path=argument_path)
                template_prefix, template_suffix = self._tokens(prefix), self._tokens(suffix)
                seq.native_template_value_type = value_type
        for kind, value in seq.prompt_plan or ():
            if kind == "tokens":
                if capture_enabled(seq) and value:
                    seq.native_rl_prompt.append({"kind": "tokens", "token_ids": list(value),
                                                 "producing": [False] * len(value)})
                seq.native_own_position += len(value)
                if value:
                    output = self._native_prefill_forward(value, seq.cache)
                    seq.cache = output["cache"]
                    seq.logits = self._native_logits(output["h_final"])
            elif kind in ("native_state","native_state_constant"):
                self._native_consume(seq, value['id'] if isinstance(value,dict) else value,stop_gradient=kind=='native_state_constant',
                    source_occurrence=value.get('source_occurrence') if isinstance(value,dict) else None)
            else:
                raise RequestError("neuralese-native-state-unavailable", "native serving accepts native_state parts, not projected blocks")
        if seq.cache is None or seq.logits is None:
            raise RequestError("empty-prompt", "native serving needs at least one prompt token")
        seq.prompt_positions = seq.cache.lengths[0]
        if seq.prompt_positions >= self.context:
            raise RequestError("context-length", f"prompt uses {seq.prompt_positions} positions, context is {self.context}")
        if template_prefix is not None:
            needed = len(template_prefix) + (1 if seq.request.template.get('value') == 'write' else 0)
            if seq.prompt_positions + seq.generated_positions + needed >= self.context:
                raise RequestError('context-length', 'templated native reply prefix exceeds remaining context')
            for token in template_prefix:
                output = self._native_forward([token], seq.cache)
                seq.cache, seq.logits = output['cache'], self._native_logits(output['h_final'])
                seq.items.append(token)
                seq.native_item_positions.append(seq.native_own_position)
                seq.generated_positions += 1
                self._native_record_input(seq, token, producing=False)
            if seq.request.template.get('value') == 'write':
                seq.native_write_start_own = seq.native_own_position
                seq.native_write_open_sample = None
                opened = self._native_forward([self.role_tokens.produce_open], seq.cache)
                seq.cache, seq.logits = opened['cache'], self._native_logits(opened['h_final'])
                seq.generated_positions += 1
                seq.native_capture_open = True
                seq.native_capture_chunks = []
                seq.native_capture_ids = []
                seq.native_template_suffix = template_suffix
                self._native_record_input(seq, self.role_tokens.produce_open, producing=False)
        if seq.request.guidance is not None:
            from .native_guidance import initialize
            initialize(self, seq)
        if seq.request.forced_message is not None:
            if not capture_enabled(seq) or seq.request.native_recording is not True:
                raise RequestError('forced-message','forced teacher replay requires native structural recording')
            seq.native_forced_tokens=self._forced_message_plan(seq.request.forced_message,seq.request.tools,
                opening=seq.request.messages)
            if len(seq.native_forced_tokens)>seq.request.max_tokens:
                raise RequestError('forced-message','exact teacher reply exceeds the request token allowance')
            seq.native_forced_replay=True
            import copy
            seq.native_forced_source=copy.deepcopy(seq.request.forced_message['source_receipt'])
            seq.native_forced_replay_prompt_sha256=hashlib.sha256(json.dumps(
                {'messages':seq.request.messages,'tools':seq.request.tools or []},ensure_ascii=False,
                sort_keys=True,separators=(',',':')).encode('utf-8')).hexdigest()
        seq.phase = "text"

    def _native_token(self, seq):
        from ..common.sampling import masked_log_distribution
        log_probs = masked_log_distribution(seq.logits[0], seq.request.temperature or 1.0,
            banned=sorted(seq.bans.get(seq.native_own_position, ())))
        if seq.request.temperature and seq.request.temperature > 0:
            token = int(torch.multinomial(log_probs.exp().cpu(), 1, generator=seq.rng))
            return token, float(log_probs[token])
        return int(log_probs.argmax()), None

    def _native_guide(self, seq):
        if seq.guide is not None:
            from .native_guidance import check
            return check(self, seq)
        return False

    def _native_finish_block(self, seq):
        if not seq.native_capture_chunks:
            from ..model.shared_state import native_step
            seq.native_capture_chunks.append(native_step(self.shared_model,
                torch.empty((1, 0), dtype=torch.long, device=self.device), seq.cache, producing=True))
        block, record = self._native_store_capture(
            seq.native_capture_ids, seq.native_capture_chunks,
            text=self.tokenizer.decode(seq.native_capture_ids, skip_special_tokens=False),
            owner=seq.request.native_store_owner)
        seq.blocks.append(record)
        seq.items.append(record)
        seq.native_item_positions.append(seq.native_write_start_own)
        seq.native_capture_chunks.clear()
        seq.native_capture_ids.clear()
        seq.native_capture_open = False
        if seq.native_template_suffix:
            seq.native_forced_tokens = seq.native_template_suffix
            seq.native_template_suffix = []
        if capture_enabled(seq):
            seq.native_rl_writes.append({"state_id": record.id,
                **({'raw_state_id': self.store.put_native_state(block, owner=seq.request.native_store_owner).id}
                   if record.block.metadata()['schema'] == 'neuralese.semantic-state/1' else {}),
                "begin_sample": seq.native_write_start_sample,
                "end_sample": (seq.native_write_close_sample + 1
                               if seq.native_write_close_sample is not None else None),
                "own_begin": seq.native_write_start_own,
                "own_end": seq.native_own_position,
                "payload_own_begin": (seq.native_write_start_own + 1
                                       if seq.native_write_start_own is not None else None),
                "payload_own_end": seq.native_own_position - 1,
                "native_identity": self._native_block_identity(block)})
        seq.native_write_start_sample = None
        seq.native_write_close_sample = None
        seq.native_write_start_own = None
        seq.native_write_open_sample = None
        if self._native_guide(seq):
            return
        if seq.request.on_delta is not None and seq.guide is None:
            seq.request.on_delta({"native_state": record.meta()})

    def _native_store_capture(self, token_ids, chunks, *, text, owner=None):
        """Store one actual producer span through the common versioned native-state path."""
        from ..model.state_block import NativeStateBlock
        from ..model.shared_state import concatenate_captures
        fields = concatenate_captures(chunks)
        expected = torch.as_tensor(token_ids, dtype=torch.int64, device=fields["token_ids"].device)
        if not torch.equal(fields["token_ids"], expected):
            raise ValueError("captured producer token IDs differ from the declared source span")
        from ..model.native_adapters import execution_binding, SCHEMA as ADAPTER_SCHEMA
        bindings = execution_binding(self)
        block = NativeStateBlock(text=text, tokenizer_fingerprint=self.model_config["tokenizer_fingerprint"],
                                 checkpoint_sha256=self.execution_native_identity["checkpoint_sha256"],
                                 embedding_weight_sha256=self.embedding_weight_sha256,
                                 architecture=self.model_config.get("architecture") or self.backbone.config.model_type,
                                 adapter_binding=({'schema': ADAPTER_SCHEMA, 'base_native_identity': self.native_identity,
                                                   'adapters': bindings} if bindings else None), **fields)
        block.validate()
        channel = getattr(self, 'semantic_channel', None)
        stored = channel.message(block) if channel is not None and self.execution_native_identity in channel.bindings.values() else block
        return block, self.store.put_native_state(stored, owner=owner)

    def native_write_text(self, text: str, *, value_type: str | None = None, adapters=None):
        return self._native_run(lambda: self._native_write_text_impl(text, value_type=value_type), adapters)

    def _forced_message_plan(self,forced,tools,*,opening):
        """Render an exact assistant response against its actual prompt, forcing text spans through native roles."""
        if (not isinstance(forced,dict) or forced.get('schema')!='natlang.native-teacher-forced-turn/1'
                or not isinstance(forced.get('assistant'),dict) or not isinstance(forced.get('source_receipt'),dict)):
            raise RequestError('forced-message','forced replay needs the bound teacher-turn source receipt and assistant target')
        message=forced['assistant'];source=forced['source_receipt'];actor=source.get('actor')
        valid_hash=lambda value:isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)
        if (source.get('schema')!='natlang.teacher-turn-source/1' or not isinstance(actor,dict)
                or not isinstance(source.get('source_path'),str) or not source['source_path']
                or not isinstance(source.get('source_view'),str) or source['source_view'] not in {'unhinted','privileged-hint'}
                or not isinstance(source.get('source_verification_path'),str) or not source['source_verification_path']
                or not isinstance(source.get('transfer_receipt_path'),str) or not source['transfer_receipt_path']
                or not isinstance(source.get('invocation_id'),str) or not source['invocation_id']
                or type(source.get('ordinal')) is not int or source['ordinal']<0
                or any(not valid_hash(source.get(key)) for key in ('turn_sha256','prompt_sha256','task_sha256','source_file_sha256'))
                or actor.get('model')!='nvidia/Qwen3.6-35B-A3B-NVFP4'
                or actor.get('revision')!='1355db6a052410cfd62085d94b58866fd0f2c3c5'
                or not isinstance(actor.get('actor_source_path'),str) or not actor['actor_source_path']
                or not isinstance(actor.get('execution_identity_path'),str) or not actor['execution_identity_path']
                or any(not valid_hash(actor.get(key)) for key in ('actor_source_sha256','execution_identity_sha256'))):
            raise RequestError('forced-message','teacher turn source binding is incomplete or has the wrong teacher actor')
        if not isinstance(message,dict) or message.get('role')!='assistant':
            raise RequestError('forced-message','the replay target must be one assistant message')
        import copy,re
        target=copy.deepcopy(message);markers={}
        vocab=getattr(self.tokenizer,'get_vocab',lambda:{})()
        def marker_for(body):
            supplied=json.dumps({'target':target,'opening':opening,'tools':tools},ensure_ascii=False,sort_keys=True)
            special_ids=set(getattr(self.tokenizer,'all_special_ids',()))
            candidates=sorted(vocab.items(),key=lambda item:(item[1],item[0]))
            for token,token_id in candidates:
                if (not isinstance(token,str) or not re.fullmatch(r'[A-Za-z0-9_]{2,}',token) or
                        token in self.specials or token_id in special_ids or token_id in self.stop_ids or
                        token in supplied or token in body or token in markers):continue
                ids=self.tokenizer(token,add_special_tokens=False,split_special_tokens=True)['input_ids']
                if ids==[token_id]:
                    markers[token]=body
                    return token
            raise RequestError('forced-message','could not allocate a collision-free exact teacher-span marker')
        spans=forced.get('native_text_spans')
        if not isinstance(spans,list):raise RequestError('forced-message','teacher native spans need explicit typed occurrence paths')
        seen_spans=set();parsed_arguments={}
        for span in spans:
            if (not isinstance(span,dict) or not isinstance(span.get('path'),list) or
                    not isinstance(span.get('marker'),str) or not span['marker'] or
                    not isinstance(span.get('text'),str) or not span['text']):
                raise RequestError('forced-message','teacher native span path, marker, or typed body is malformed')
            path=span['path'];marker=span['marker'];body=span['text']
            if marker in seen_spans or json.dumps(target,ensure_ascii=False).count(marker)!=1:
                raise RequestError('forced-message','teacher span marker is duplicate or ambiguous in its exact assistant target')
            seen_spans.add(marker)
            if path==['content']:
                value=target.get('content')
                if not isinstance(value,str) or value.count(marker)!=1:
                    raise RequestError('forced-message','teacher content span does not resolve to one exact typed occurrence')
                target['content']=value.replace(marker,marker_for(body),1)
                continue
            if len(path)<4 or path[0]!='tool_calls' or type(path[1]) is not int or path[2]!='arguments':
                raise RequestError('forced-message','teacher tool span must name an exact tool argument field')
            index=path[1];calls=target.get('tool_calls') or []
            if index<0 or index>=len(calls):raise RequestError('forced-message','teacher tool span call ordinal is outside the target')
            function=calls[index].get('function') or {};name=function.get('name')
            argument_path=path[3:]
            if not ((name=='eval' and argument_path and argument_path[0]=='code') or
                    (name=='return_result' and argument_path and argument_path[0]=='value')):
                raise RequestError('forced-message','typed teacher span is not at a supported eval-code or typed return-result occurrence')
            if index not in parsed_arguments:
                try:parsed_arguments[index]=json.loads(function.get('arguments'))
                except (TypeError,json.JSONDecodeError) as error:
                    raise RequestError('forced-message','teacher tool arguments are not exact structured JSON') from error
            value=parsed_arguments[index]
            for part in argument_path[:-1]:
                if isinstance(value,dict) and isinstance(part,str) and part in value:value=value[part]
                elif isinstance(value,list) and type(part) is int and 0<=part<len(value):value=value[part]
                else:raise RequestError('forced-message','teacher typed span path is missing in structured tool arguments')
            leaf=argument_path[-1]
            if isinstance(value,dict) and isinstance(leaf,str) and isinstance(value.get(leaf),str) and value[leaf].count(marker)==1:
                value[leaf]=value[leaf].replace(marker,marker_for(body),1)
            elif isinstance(value,list) and type(leaf) is int and 0<=leaf<len(value) and isinstance(value[leaf],str) and value[leaf].count(marker)==1:
                value[leaf]=value[leaf].replace(marker,marker_for(body),1)
            else:
                raise RequestError('forced-message','teacher tool span does not resolve to one exact typed string occurrence')
        for index,arguments in parsed_arguments.items():
            target['tool_calls'][index]['function']['arguments']=json.dumps(arguments,ensure_ascii=False,separators=(',',':'))
        from .chat import assistant_reply_segments,normalize_tool_call_arguments
        target=normalize_tool_call_arguments([target])[0][0]
        apply=lambda messages,generation:self.tokenizer.apply_chat_template(messages,tools=tools or None,
            tokenize=False,add_generation_prompt=generation)
        segments=assistant_reply_segments(apply,target,self.specials,opening=opening)
        if segments is None:raise RequestError('forced-message','assistant response does not continue this exact actual prompt')
        plan=[];found={marker:0 for marker in markers}
        def append_text(text,escaped):
            if text:
                plan.extend(self.tokenizer(text,add_special_tokens=False,split_special_tokens=True)['input_ids']
                    if escaped else self._template_tokens(text))
        for text,escaped in segments:
            cursor=0
            while cursor<len(text):
                positions=[(text.find(marker,cursor),marker,body) for marker,body in markers.items()
                    if text.find(marker,cursor)>=0]
                if not positions:
                    append_text(text[cursor:],escaped);break
                start,marker,body=min(positions,key=lambda item:item[0])
                append_text(text[cursor:start],escaped)
                body_ids=self.tokenizer(body,add_special_tokens=False,split_special_tokens=True)['input_ids']
                if not body_ids or any(token in self.stop_ids for token in body_ids):
                    raise RequestError('forced-message','native teacher text is empty or contains an exact model stop token')
                plan.extend([self.role_tokens.produce_open,*body_ids,self.role_tokens.produce_close])
                found[marker]+=1;cursor=start+len(marker)
        if any(count!=1 for count in found.values()):
            raise RequestError('forced-message','native teacher text marker does not resolve to one exact token aligned occurrence')
        return plan

    def _native_write_text_impl(self, text: str, *, value_type: str | None = None):
        """Force supplied text through this model's actual producer transform and archive its native state."""
        if not getattr(self, "native_state_enabled", False):
            raise RequestError("neuralese-native-state-unavailable", "native text writes require shared-state serving")
        if not isinstance(text, str):
            raise RequestError("neuralese-native-state-text", "native text writes require text")
        from .chat import escape_specials
        nonce = hashlib.sha256((text + str(time.time_ns())).encode()).hexdigest()
        ids = self._template_tokens(escape_specials(text, self.specials, nonce), nonce)
        if len(ids) + 2 >= self.context:
            raise RequestError("context-length", "forced native text and producer markers exceed this model context")
        from ..model.shared_state import native_step
        with torch.inference_mode():
            opened = native_step(self.shared_model, torch.tensor([[self.role_tokens.produce_open]],
                dtype=torch.long, device=self.device))
            capture = native_step(self.shared_model, torch.tensor([ids], dtype=torch.long, device=self.device),
                                  opened["cache"], producing=True)
            native_step(self.shared_model, torch.tensor([[self.role_tokens.produce_close]],
                dtype=torch.long, device=self.device), capture["cache"])
        block, record = self._native_store_capture(ids, [capture], text=text)
        meta = record.meta()
        if value_type is not None:
            meta["type"] = value_type
        result = {"id": record.id, "native_state": meta}
        result["source_capture"] = {"policy": self._native_block_identity(block), "segments": [
            {"kind": "tokens", "token_ids": [self.role_tokens.produce_open], "producing": [False]},
            {"kind": "tokens", "token_ids": ids, "producing": [True] * len(ids)},
            {"kind": "tokens", "token_ids": [self.role_tokens.produce_close], "producing": [False]}],
            "payload_start": 1, "payload_stop": len(ids) + 1}
        result["source_text_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return result

    def native_init_body(self, messages, tools, placeholder: str, text: str, *, value_type: str | None = None, placeholder_occurrence: int | None = None, adapters=None, native_state_modes=None):
        return self._native_run(lambda: self._native_init_body_impl(messages, tools, placeholder, text,
            value_type=value_type, placeholder_occurrence=placeholder_occurrence, native_state_modes=native_state_modes), adapters)

    def _native_init_body_impl(self, messages, tools, placeholder: str, text: str, *, value_type: str | None = None, placeholder_occurrence: int | None = None, native_state_modes=None):
        """Capture supplied body tokens at their actual rendered causal position in a soft call."""
        if not getattr(self, "native_state_enabled", False):
            raise RequestError("neuralese-native-state-unavailable", "native body initialization requires shared-state serving")
        if not isinstance(placeholder, str) or not placeholder.startswith("nzs1_") or not isinstance(text, str):
            raise RequestError("neuralese-native-state-text", "native body initialization needs a native placeholder and text")
        if self.store.get_native_state(placeholder) is None:
            raise RequestError("neuralese-unknown-native-state", f"{placeholder} is not in this server's store; write it first")
        marker = f"\ue000NATIVE_BODY_{hashlib.sha256((placeholder + text).encode()).hexdigest()}\ue001"
        if placeholder_occurrence is not None and (type(placeholder_occurrence) is not int or placeholder_occurrence < 0):
            raise RequestError('neuralese-native-state-placeholder', 'placeholder_occurrence must be a nonnegative occurrence index')
        found = 0
        def replace(value):
            nonlocal found
            if isinstance(value, list):
                return [replace(item) for item in value]
            if isinstance(value, dict):
                if value.get("type") == "native_state" and value.get("id") == placeholder:
                    current = found
                    found += 1
                    return {"type": "text", "text": marker} if placeholder_occurrence is None or current == placeholder_occurrence else value
                return {key: replace(item) for key, item in value.items()}
            if isinstance(value, str):
                before = found
                try:
                    parsed = json.loads(value)
                except (TypeError, json.JSONDecodeError):
                    return value
                replaced = replace(parsed)
                return json.dumps(replaced, ensure_ascii=False, separators=(",", ":")) if found != before else value
            return value
        rendered = self.render_request(replace(messages), tools)
        self._validate_native_state_modes(native_state_modes, {block_id for block_id, kind in
            zip(rendered.blocks, rendered.block_kinds or ['neuralese'] * len(rendered.blocks)) if kind == 'native_state'})
        if (placeholder_occurrence is None and found != 1) or (placeholder_occurrence is not None and found <= placeholder_occurrence):
            raise RequestError("neuralese-native-state-placeholder", f"found {found} matching parts; pass placeholder_occurrence to select the exact body occurrence when its content ID is also imported")
        matches = [(index, segment) for index, segment in enumerate(rendered.segments)
                   if isinstance(segment, str) and segment.count(marker) == 1]
        if len(matches) != 1:
            raise RequestError("neuralese-native-state-text", "rendered body marker must occur exactly once in text")
        body_segment, source = matches[0]
        before, after = source.split(marker)
        from .chat import escape_specials
        escaped_body = escape_specials(text, self.specials, rendered.escape_nonce)
        left = self._template_tokens(before, rendered.escape_nonce)
        body = self._template_tokens(escaped_body, rendered.escape_nonce)
        right = self._template_tokens(after, rendered.escape_nonce)
        if self.prompt_positions(messages, tools) >= self.context:
            raise RequestError("context-length", "contextual native body source exceeds this model context")
        from ..model.shared_state import native_step
        cache = None
        captured = None
        source_segments = []
        source_imports = []
        source_own_position = 0
        payload_start = None
        payload_stop = None
        def token_segment(ids, producing=False):
            nonlocal source_own_position
            if ids:
                source_segments.append({"kind": "tokens", "token_ids": list(ids),
                                        "producing": [bool(producing)] * len(ids)})
                source_own_position += len(ids)
        with torch.inference_mode():
            for index, segment in enumerate(rendered.segments):
                if isinstance(segment, int):
                    kinds = rendered.block_kinds or ["neuralese"] * len(rendered.blocks)
                    block_id = rendered.blocks[segment]
                    if kinds[segment] != "native_state":
                        raise RequestError("neuralese-native-state-text", "legacy projected blocks cannot initialize native state")
                    temp = Sequence(GenerationRequest(messages=[], rl_capture=True, native_state_modes=native_state_modes), Future(), cache=cache)
                    self._native_consume(temp, block_id,stop_gradient=bool(rendered.block_stop_gradients and rendered.block_stop_gradients[segment]),
                        source_occurrence=rendered.block_source_occurrences[segment] if rendered.block_source_occurrences else None)
                    cache = temp.cache
                    source_segments.extend(temp.native_rl_prompt)
                    source_imports.extend(temp.native_imports)
                    source_own_position += temp.native_own_position
                elif index == body_segment:
                    if left:
                        cache = native_step(self.shared_model, torch.tensor([left], dtype=torch.long, device=self.device), cache)["cache"]
                        token_segment(left)
                    open_marker = [self.role_tokens.produce_open]
                    cache = native_step(self.shared_model, torch.tensor([open_marker], dtype=torch.long, device=self.device),
                                        cache)["cache"]
                    token_segment(open_marker)
                    payload_start = source_own_position
                    captured = native_step(self.shared_model, torch.tensor([body], dtype=torch.long, device=self.device),
                                           cache, producing=True)
                    cache = captured["cache"]
                    token_segment(body, producing=True)
                    payload_stop = source_own_position
                    close_marker = [self.role_tokens.produce_close]
                    cache = native_step(self.shared_model, torch.tensor([close_marker], dtype=torch.long, device=self.device),
                                        cache)["cache"]
                    token_segment(close_marker)
                    if right:
                        cache = native_step(self.shared_model, torch.tensor([right], dtype=torch.long, device=self.device), cache)["cache"]
                        token_segment(right)
                else:
                    ids = self._template_tokens(segment, rendered.escape_nonce)
                    if ids:
                        cache = native_step(self.shared_model, torch.tensor([ids], dtype=torch.long, device=self.device), cache)["cache"]
                        token_segment(ids)
        if captured is None:
            raise RequestError("neuralese-native-state-text", "rendered call did not contain the body span")
        block, record = self._native_store_capture(body, [captured], text=text)
        meta = record.meta()
        if value_type is not None:
            meta["type"] = value_type
        result = {"id": record.id, "native_state": meta}
        result["init"] = {"schema": "natlang.native-state-text-init/1", "tokens": len(body),
                          "exact": False, "aligned": True, "exact_transport": True, "contextual": True,
                          "source_position_start": captured["cache"].lengths[0] - len(body)}
        result["gate"] = None
        result["source_capture"] = {"policy": self._native_block_identity(block), "segments": source_segments,
                                    "imports": source_imports,
                                    "payload_start": payload_start, "payload_stop": payload_stop}
        result["source_text_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return result

    def _native_round(self, sequences):
        for seq in sequences:
            try:
                self._native_run(lambda: self._native_round_impl([seq]), seq.request.adapters)
            except Exception as error:
                self._fail(seq, error)

    def _native_round_impl(self, sequences):
        """Native marker-token generation through the same scheduler and TensorStore."""
        for seq in sequences:
            if seq.request.cancelled:
                self._fail(seq, RequestError("cancelled", "the client went away"))
                continue
            if seq.phase == "done":
                continue
            try:
                if (capture_enabled(seq) and seq.native_rl_policy is not None
                        and self.execution_native_identity != seq.native_rl_policy):
                    raise RequestError('neuralese-native-actor-changed',
                        'keep the exact adapter binding unchanged until this invocation finishes')
                if seq.phase == "prefill":
                    self._native_prefill(seq)
                    continue
                if seq.native_forced_tokens and not seq.native_forced_replay:
                    token = seq.native_forced_tokens.pop(0)
                    if token in self.stop_ids:
                        self._finish(seq, 'stop')
                        continue
                    output = self._native_forward([token], seq.cache)
                    seq.cache, seq.logits = output['cache'], self._native_logits(output['h_final'])
                    seq.generated_positions += 1
                    self._native_decoded_tokens += 1
                    self._native_record_input(seq, token,
                        producing=seq.native_capture_open and token not in (
                            self.role_tokens.produce_open, self.role_tokens.produce_close))
                    seq.items.append(token)
                    seq.native_item_positions.append(seq.native_own_position - 1)
                    if not self._native_guide(seq):
                        self._emit(seq, token=token)
                    continue
                if seq.native_forced_replay and not seq.native_forced_tokens:
                    if seq.native_capture_open:
                        raise RequestError('neuralese-native-state-open','forced teacher reply ended inside a producer span')
                    self._finish(seq,'stop')
                    continue
                if seq.generated_positions >= seq.request.max_tokens:
                    if seq.native_capture_open:
                        raise RequestError("neuralese-native-state-open", "generation ended inside a produce span")
                    self._finish(seq, "length")
                    continue
                forced=seq.native_forced_replay
                token, sampled_logp = ((seq.native_forced_tokens.pop(0),None) if forced else self._native_token(seq))
                sample_index = len(seq.rl_samples)
                # The next-token query is the output position of the last own
                # input; imported KV slots do not contribute to this index.
                query_own_position = seq.native_own_position - 1
                terminal = token in self.stop_ids
                producing_input = seq.native_capture_open and token not in (
                    self.role_tokens.produce_open, self.role_tokens.produce_close)
                if capture_enabled(seq) or seq.guide is not None:
                    seq.rl_samples.append({"index": sample_index, "token_id": token, "logp": sampled_logp,
                                           **({'forced':True} if forced else {}),
                                           **({'banned_token_ids': sorted(seq.bans.get(seq.native_own_position, ()))}
                                              if seq.guide is not None else {}),
                                           "input_role": "produce" if producing_input else "ordinary",
                                           "terminal": terminal,
                                           "query_own_position": query_own_position})
                if terminal:
                    if seq.native_capture_open:
                        raise RequestError("neuralese-native-state-open", "generation stopped inside a produce span")
                    self._finish(seq, "stop")
                    continue
                marker = self.role_tokens
                if token == marker.produce_close and not seq.native_capture_open:
                    raise RequestError("neuralese-native-state-close", "generation emitted a produce end marker without an open span")
                if token in (marker.consume_open, marker.consume_close):
                    raise RequestError("neuralese-native-state-consume-marker",
                                       "consume markers are inserted only around an imported native-state part")
                if seq.native_capture_open and token == marker.produce_close:
                    seq.native_write_close_sample = sample_index
                    output = self._native_forward([token], seq.cache)
                    seq.cache, seq.logits = output["cache"], self._native_logits(output["h_final"])
                    seq.generated_positions += 1
                    self._native_decoded_tokens += 1
                    self._native_record_input(seq, token, producing=False, sample_index=sample_index)
                    self._native_finish_block(seq)
                    continue
                producing = seq.native_capture_open and token != marker.produce_open
                if token == marker.produce_open:
                    seq.native_write_start_own = seq.native_own_position
                    seq.native_write_open_sample = sample_index
                output = self._native_forward([token], seq.cache, producing=producing)
                seq.cache, seq.logits = output["cache"], self._native_logits(output["h_final"])
                seq.generated_positions += 1
                self._native_decoded_tokens += 1
                self._native_record_input(seq, token, producing=producing, sample_index=sample_index)
                if token == marker.produce_open:
                    if seq.native_capture_open:
                        raise RequestError("neuralese-native-state-nested", "produce spans cannot nest")
                    seq.native_capture_chunks = []
                    seq.native_capture_ids = []
                    seq.native_capture_open = True
                    seq.native_write_start_sample = sample_index
                elif producing:
                    seq.native_capture_chunks.append(output)
                    seq.native_capture_ids.append(token)
                else:
                    seq.items.append(token)
                    seq.native_item_positions.append(seq.native_own_position - 1)
                    if not self._native_guide(seq):
                        self._emit(seq, token=token)
            except Exception as error:  # one malformed sequence must not stop other requests
                self._native_protocol_fail(seq, error)

    # Decoding ------------------------------------------------------------------------------
    def _round(self, sequences: list[Sequence]):
        """One scheduler round: every sequence advances one position; text and sketch steps run batched."""
        if getattr(self, 'native_state_enabled', False):
            return self._native_round(sequences)
        for seq in sequences:
            if seq.request.cancelled:
                self._fail(seq, RequestError("cancelled", "the client went away"))
        sequences = [seq for seq in sequences if seq.phase != "done"]
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

    def _native_protocol_fail(self, seq, error):
        from .native_protocol import FAILURE_CODES,failure_descriptor
        if (not isinstance(error,RequestError) or error.code not in FAILURE_CODES
                or not seq.request.native_protocol_recovery or seq.native_rl_policy is None):
            return self._fail(seq,error)
        seq.native_protocol_failure=failure_descriptor(self,seq,error)
        seq.finish_reason='error';seq.phase='done'
        if not seq.future.done():seq.future.set_result(self._native_response(seq))

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
            if self.heads.read_markers:
                h, caches = step_rows(self.backbone, self.backbone.embed(ids), [seq.cache for seq, _ in rows],
                                      range(0, self.backbone.num_layers))
            else:
                cut, caches = step_rows(self.backbone, self.backbone.embed(ids), [seq.cache for seq, _ in rows],
                                        range(0, self.heads.cutoff))
                h, caches = step_rows(self.backbone, cut, caches, range(self.heads.cutoff, self.backbone.num_layers))
            logits = self.backbone.logits(h)[:, -1] if getattr(self, 'token_choice_projection', None) is None else None
        for index, (seq, token) in enumerate(rows):
            if logits is None:
                with self.using([seq]):
                    score, seq.token_history = self.advance_token_readout(seq.token_history, h[index:index + 1], last_only=True)
                seq.cache, seq.logits = caches[index], score[:, -1]
            else:
                seq.cache, seq.logits = caches[index], logits[index:index + 1]
            if not self.heads.read_markers:
                seq.cut_state = cut[index:index + 1, -1]
                seq.top_state = h[index:index + 1, -1]
            seq.items.append(token)
            seq.generated_positions += 1
            if seq.guide is not None:
                seq.pieces.append(self.tokenizer.decode([token], skip_special_tokens=False))
                self._guide(seq)
            self._emit(seq, token=token)

    def _guide(self, seq: Sequence):
        """Run the reply checks after a token; on a rejection roll back to the offending line and ban its first token.
        Snapshots are kept at line starts (the last few); the tokens between a snapshot and the rollback point are
        recomputed."""
        reply = "".join(seq.pieces)
        verdict = seq.guide.check(reply)
        starts, offset = [], 0
        for piece in seq.pieces:
            starts.append(offset)
            offset += len(piece)
        if verdict is None:
            # A line that starts where the reply now ends: snapshot here (cheap restore for the next rejection).
            if reply.endswith("\\n") or reply.endswith("\n") or reply.endswith("["):
                seq.snapshots = (seq.snapshots + [(len(seq.items), seq.cache, seq.logits, seq.generated_positions, seq.cut_state, seq.top_state, seq.token_history)])[-4:]
            return
        reason, at = verdict
        target = max(i for i, start in enumerate(starts) if start <= at)  # the item holding the line's first character
        tries = seq.retries.get(at, 0)
        usable = [snap for snap in seq.snapshots if snap[0] <= target]
        between = seq.items[usable[-1][0]:target] if usable else []
        if tries >= seq.guide.settings.retries or not usable or any(not isinstance(i, int) for i in between):
            seq.guide.accept(at)
            seq.guide.rejections.append({"reason": reason, "offset": at, "accepted": True})
            return
        seq.retries[at] = tries + 1
        seq.guide.rejections.append({"reason": reason, "offset": at})
        seq.bans.setdefault(target, set()).add(seq.items[target])
        count, cache, logits, generated, cut_state, top_state, token_history = usable[-1]
        if between:
            with self.using([seq]):
                out = self.backbone.forward_ids(torch.tensor([between], device=self.device), cache=cache,
                                                cutoff=self.heads.cutoff if not self.heads.read_markers else None)
            cache, logits = out["cache"], out["logits"][:, -1]
            if getattr(self, 'token_choice_projection', None) is not None:
                with self.using([seq]):
                    score, token_history = self.advance_token_readout(token_history, out['h_final'], last_only=True)
                logits = score[:, -1]
            if not self.heads.read_markers:
                cut_state, top_state = out['h_cut'][:, -1], out['h_final'][:, -1]
            generated += len(between)
        seq.cache, seq.logits, seq.generated_positions = cache, logits, generated
        seq.cut_state, seq.top_state, seq.token_history = cut_state, top_state, token_history
        if seq.request.rl_capture:
            seq.rl_samples = [sample for sample in seq.rl_samples if sample['item'] < target]
        seq.guide.rewind(starts[target])
        del seq.items[target:], seq.pieces[target:]
        seq.snapshots = [snap for snap in seq.snapshots if snap[0] <= target]
        for index in [i for i in seq.bans if i > target]:
            del seq.bans[index]

    def _sketch_batch(self, rows):
        # A direct writer reads its next input as its feedback says (token embeddings under direct v1); a sketch
        # writer reads its sketch.
        sketches = torch.cat([seq.writer.next_input(sketch) if hasattr(seq.writer, 'next_input') else sketch
                              for seq, sketch in rows], 0)[:, None]
        depth = self.backbone.num_layers if getattr(self, 'neuralese_projection', None) is not None else self.heads.cutoff
        with self.using([seq for seq, _ in rows]):
            h, caches = step_rows(self.backbone, sketches, [seq.writer.cache for seq, _ in rows],
                                  range(0, depth))
        for index, (seq, sketch) in enumerate(rows):
            seq.writer.advance(sketch, h[index:index + 1, 0], caches[index])

    def _prefill_all(self, seqs: list[Sequence]):
        """Prefill new requests together: left-padded batches by length under a padded-token budget."""
        ready = []
        for seq in seqs:
            try:
                seq.adapters = self.resolve_adapters(seq.request.adapters)
                if seq.request.rl_capture and seq.prompt_plan is None:
                    seq.prompt_plan = self._prompt_plan(seq.request.messages, seq.request.tools)
                embeds = self.prompt_embeddings(seq.request.messages, seq.request.tools, prepared=seq.prompt_plan)
                self._plan(seq)
                if seq.prefilled_forced:
                    # The forced reply's leading tokens are prefilled with the prompt, as training and the causal
                    # reference run them: one pass, not one decode step per token (a BF16 near-tie can flip there).
                    embeds = torch.cat([embeds, self.backbone.embed(
                        torch.tensor([seq.prefilled_forced], device=self.device)).to(embeds)], 1)
                ready.append((seq, embeds))
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
                    out = self.backbone.forward_embeds(embeds, logits=False, cutoff=self.heads.cutoff if not self.heads.read_markers else None)
                    start = out['h_final'].shape[1] - len(seq.prefilled_forced) - 1
                    score, token_history = self.advance_token_readout(None, out['h_final'][:, start:], last_only=True)
                    logits = score[:, -1]
                self._prefilled(seq, out["cache"], logits, int(embeds.shape[1]),
                                out['h_cut'][:, -1] if not self.heads.read_markers else None,
                                out['h_final'][:, -1] if not self.heads.read_markers else None, token_history)
            except Exception as error:
                self._fail(seq, error)
            return
        try:
            width = max(e.shape[1] for _, e in group)
            pad_lengths = [width - e.shape[1] for _, e in group]
            pad = torch.tensor(pad_lengths, device=self.device)
            embeds = torch.cat([torch.cat([e.new_zeros(1, width - e.shape[1], e.shape[2]), e], 1) for _, e in group], 0)
            padded = any(pad_lengths)
            with self.using([seq for seq, _ in group]):
                out = self.backbone.forward_embeds(embeds, left_pad=pad if padded else None, logits=False,
                                                   cutoff=self.heads.cutoff if not self.heads.read_markers else None)
                logits = self.backbone.logits(out["h_final"][:, -1:])[:, -1]
            caches = split_rows(out["cache"], pad)
        except Exception:
            for row in group:  # isolate a failing request
                self._prefill_group([row])
            return
        for index, (seq, e) in enumerate(group):
            try:
                token_history = None
                score = logits[index:index + 1]
                if getattr(self, 'token_choice_projection', None) is not None:
                    with self.using([seq]):
                        start = out['h_final'].shape[1] - len(seq.prefilled_forced) - 1
                        scores, token_history = self.advance_token_readout(None, out['h_final'][index:index + 1, start:], last_only=True)
                        score = scores[:, -1]
                self._prefilled(seq, caches[index], score, int(e.shape[1]),
                                out['h_cut'][index:index + 1, -1] if not self.heads.read_markers else None,
                                out['h_final'][index:index + 1, -1] if not self.heads.read_markers else None, token_history)
            except Exception as error:
                self._fail(seq, error)

    def _prefill(self, seq: Sequence):
        self._prefill_all([seq])

    def _plan(self, seq: Sequence):
        """The request's forced plan, and its leading text tokens that prefill with the prompt (not under guidance,
        whose backtracking snapshots start after the prompt)."""
        request = seq.request
        seq.forced = self._forced_plan(request.forced) if request.forced is not None else None
        if request.template is not None and request.forced is None:  # the test hook's plan replaces the template
            seq.forced, seq.free_after_forced = self._template_plan(request.template)
        lead = []
        if seq.forced and request.guidance is None:
            # Only what the decode steps would have fed: within max_tokens, up to a stop or block opening.
            while (len(lead) < min(len(seq.forced), request.max_tokens) and type(seq.forced[len(lead)]) is int
                   and seq.forced[len(lead)] not in self.stop_ids
                   and seq.forced[len(lead)] != self.backbone.controls.open_id):
                lead.append(seq.forced[len(lead)])
            del seq.forced[:len(lead)]
        seq.prefilled_forced = lead

    def _prefilled(self, seq: Sequence, cache, logits, positions: int, cut_state=None, top_state=None, token_history=None):
        request = seq.request
        seq.cache, seq.logits = cache, logits
        seq.cut_state, seq.top_state, seq.token_history = cut_state, top_state, token_history
        lead = seq.prefilled_forced
        seq.prompt_positions = positions - len(lead)
        if request.guidance is not None:
            from .guidance import Guide

            seq.guide = Guide(request.guidance)
            prefix = seq.guide.forced_prefix()
            if prefix and seq.forced is None:
                seq.forced, seq.free_after_forced = self._tokens(prefix), True
            seq.snapshots = [(0, seq.cache, seq.logits, 0, seq.cut_state, seq.top_state, seq.token_history)]
        for token in lead:  # already in the cache: emitted as the decode steps would have
            seq.items.append(token)
            seq.generated_positions += 1
            self._emit(seq, token)
        seq.rng = torch.Generator().manual_seed(
            derive_seed("text", request.seed if request.seed is not None else request.request_id))
        seq.phase = "text"

    def _emit(self, seq: Sequence, token: int | None = None, block: Block | None = None):
        """Streaming: hand new output to the request's listener, if any."""
        if getattr(self, 'native_state_enabled', False) and seq.guide is not None and not seq.native_guidance_flushing:
            return
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
        if seq.forced is not None and not seq.forced and seq.free_after_forced:
            seq.forced = None
        if seq.forced is not None:
            if not seq.forced:
                return None
            item = seq.forced.pop(0)
            return self.backbone.controls.open_id if item is WRITE else item
        logits = seq.logits[0].float()
        banned = seq.bans.get(len(seq.items))
        if banned:
            logits = logits.clone()
            logits[list(banned)] = float("-inf")
        if seq.request.temperature and seq.request.temperature > 0:
            probs = torch.softmax(logits / seq.request.temperature, -1).cpu()
            token = int(torch.multinomial(probs, 1, generator=seq.rng))
            if seq.request.rl_capture:
                seq.rl_samples.append({'item': len(seq.items), 'token': token,
                    'logp': float(torch.log_softmax(logits / seq.request.temperature, -1)[token]),
                    'banned': sorted(banned or [])})
            return token
        return int(logits.argmax())

    def _finish(self, seq: Sequence, reason: str):
        if getattr(self, 'native_state_enabled', False) and seq.guide is not None:
            from .native_guidance import flush
            flush(self, seq)
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
        # The close control is trained as the boundary of a text/document span.
        # A generated close outside a write cannot close a Neuralese block: the
        # writer closes its own block in _close_block. Never expose it as text.
        if token is None or token in self.stop_ids or token == self.backbone.controls.close_id:
            self._finish(seq, "stop")
            return None
        if token == self.backbone.controls.open_id:
            remaining = request.max_tokens - seq.generated_positions
            # A nonempty block needs open + at least one vector + close. Do not
            # emit a dangling opener when the completion allowance cannot fit it.
            if remaining < 3:
                self._finish(seq, "length")
                return None
            if self.heads.read_markers:
                ids = torch.tensor([[token]], device=self.device)
                with self.using([seq]):
                    out = self.backbone.forward_ids(ids, cache=seq.cache, cutoff=self.heads.cutoff)
                opened = Opened(cache=out["cache"], h_cut=out["h_cut"][:, -1], logits=out["logits"][:, -1])
            else:
                if seq.cut_state is None:
                    raise RuntimeError('raw write missing causal prefix state')
                opened = Opened(cache=seq.cache, h_cut=seq.cut_state, logits=seq.logits, top=seq.top_state)
            limit = request.neuralese_max_length or self.max_block
            direct = getattr(self, 'neuralese_projection', None)
            if direct is not None:
                from .direct_write import DirectWriter
                seq.writer = DirectWriter(self.backbone, direct, opened, min(limit, self.max_block, remaining - 2),
                                          self.stop_ids, length=request.neuralese_length,
                                          token_choice_projection=getattr(self, 'token_choice_projection', None))
            else:
                seq.writer = StepWriter(self.backbone, self.heads, opened, min(limit, self.max_block, remaining - 2),
                                        length=request.neuralese_length)
            if request.neuralese_length and request.neuralese_passes:
                with self.using([seq]):
                    if direct is not None:
                        # Direct v1 has no shallow sketch passes. The full-depth
                        # writer remains the sole implementation for this artifact.
                        while not seq.writer.step():
                            pass
                    else:
                        seq.writer.blockwise(request.neuralese_passes)
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
            back = read_back(self.backbone, self.heads, writer.opened.cache, payload, all_logits=False,
                             final_states=getattr(self, 'token_choice_projection', None) is not None)
            if getattr(self, 'token_choice_projection', None) is not None:
                scores, seq.token_history = self.advance_token_readout(seq.token_history, back['final_states'], last_only=True)
                back['logits'] = scores[:, -1]
        seq.cache, seq.logits = back["cache"], back["logits"]
        if not self.heads.read_markers:
            seq.cut_state, seq.top_state = back['h_cut'], back['h_final']
        record = {"kind": "write", "request": request.request_id, "index": index, "cutoff": self.heads.cutoff,
                  "temperature": tau, "seed": seed, "stop_logits": writer.stop_logits}
        if getattr(self, 'neuralese_projection', None) is not None:
            record.update(algorithm='neuralese-direct-v1', cutoff=self.backbone.num_layers)
        if writer.length is not None:
            record.update(length_hint=writer.length, passes=writer.passes)
        parts = []
        if writer.count:
            mean = make_block(mu[0], self.dialect, producer={"kind": "payload-mean"})
            scale = make_block(log_sigma[0], self.dialect, producer={"kind": "payload-log-sigma"})
            record.update(mean=mean.id, log_sigma=scale.id)
            parts = [mean, scale]
        # The payload first: at temperature 0 it is its mean, and the store keeps the first block of an ID, which must
        # be the one carrying the write record.
        value_type = (seq.request.template or {}).get("value_type")
        block = self.store.put(make_block(payload[0], self.dialect,
            type="Neuralese<unknown>" if value_type == "unknown" else None, producer=record, truncated=writer.truncated))
        for part in parts:
            self.store.put(part)
        seq.blocks.append(block)
        seq.items.append(block)
        if seq.guide is not None:
            seq.pieces.append("__nz")
        self._emit(seq, block=block)
        seq.generated_positions += writer.count + 1
        seq.writer = None
        seq.phase = "text"
        # A forced plan's own close marker, if any, is written by the procedure instead.
        if seq.forced and seq.forced[0] == self.backbone.controls.close_id:
            seq.forced.pop(0)

    # Output --------------------------------------------------------------------------------
    def _response(self, seq: Sequence) -> dict:
        if getattr(self, 'native_state_enabled', False):
            return self._native_response(seq)
        text, run, ids = [], [], []
        for item in seq.items:
            if isinstance(item, Block):
                if run:
                    text.append(self.tokenizer.decode(run, skip_special_tokens=False))
                    run = []
                wire = placeholder(len(ids))
                template = seq.request.template or {}
                if template.get("value") == "write" and template.get("value_type") == "unknown":
                    import json
                    wire = json.dumps(wire)
                text.append(wire)
                ids.append(item.id)
            else:
                run.append(item)
        if run:
            text.append(self.tokenizer.decode(run, skip_special_tokens=False))
        message = build_message("".join(text), ids, call_prefix=seq.request.request_id)
        template = seq.request.template or {}
        if template.get("value") == "write" and template.get("value_type") == "unknown":
            import json
            from .chat import _at_value_path, arguments_text
            def mark_unknown(value):
                if isinstance(value, list):
                    for item in value:mark_unknown(item)
                elif isinstance(value, dict):
                    if value.get("type") == "neuralese":value["value_type"] = "unknown"
                    else:
                        for item in value.values():mark_unknown(item)
            for call in message.get("tool_calls") or []:
                arguments = json.loads(call["function"]["arguments"])
                argument = template.get("argument") or "value"
                path = template.get("argument_path")
                value = arguments.get(argument) if path is None else _at_value_path(arguments, path)
                mark_unknown(value)
                call["function"]["arguments"] = arguments_text(arguments)
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
            **({'x_natlang_rl_capture': {
                'schema': 'natlang.rl-sampling/1', 'temperature': seq.request.temperature,
                'guided': seq.request.guidance is not None,
                'prompt': seq.prompt_plan,
                'items': [('block', item.id) if isinstance(item, Block) else ('tok', item) for item in seq.items],
                'samples': seq.rl_samples,
                'finish_reason': seq.finish_reason,
            }} if seq.request.rl_capture else {}),
            **({"x_natlang_guidance": {"rejections": seq.guide.rejections}} if seq.guide is not None else {}),
        }

    def _native_response(self, seq: Sequence) -> dict:
        from .chat import arguments_text, build_message
        text, run, block_ids = [], [], []
        for item in seq.items:
            if hasattr(item, "block") and hasattr(item, "meta"):
                if run:
                    text.append(self.tokenizer.decode(run, skip_special_tokens=False))
                    run = []
                text.append(placeholder(len(block_ids)))
                block_ids.append(item.id)
            else:
                run.append(item)
        if run:
            text.append(self.tokenizer.decode(run, skip_special_tokens=False))
        message = build_message("".join(text), block_ids, call_prefix=seq.request.request_id)

        def native_parts(value):
            if isinstance(value, list):
                return [native_parts(item) for item in value]
            if isinstance(value, dict):
                if value.get("type") == "neuralese" and value.get("id", "").startswith("nzs1_"):
                    return {"type": "native_state", "id": value["id"]}
                return {key: native_parts(item) for key, item in value.items()}
            return value

        message = native_parts(message)
        for call in message.get("tool_calls") or []:
            try:
                args = json.loads(call["function"]["arguments"])
                call["function"]["arguments"] = arguments_text(native_parts(args))
            except (KeyError, json.JSONDecodeError):
                continue
        if seq.native_template_value_type is not None:
            for call in message.get('tool_calls') or []:
                try:
                    arguments = json.loads(call['function']['arguments'])
                    template = seq.request.template or {}
                    argument = template.get('argument') or 'value'
                    path = template.get('argument_path') or [argument]
                    value = arguments
                    for part in path:
                        value = value[part]
                    def mark_value(item):
                        if isinstance(item, list):
                            for nested in item: mark_value(nested)
                        elif isinstance(item, dict) and item.get('type') == 'native_state':
                            item['value_type'] = seq.native_template_value_type
                    mark_value(value)
                    call['function']['arguments'] = arguments_text(arguments)
                except (KeyError, TypeError, json.JSONDecodeError):
                    continue
        if seq.native_protocol_failure is not None:
            from .native_protocol import generated_tokens
            # No action from a malformed turn executes, including an earlier
            # parseable tool call. Exact original tokens remain in the receipt.
            message={'role':'assistant','content':self.tokenizer.decode(generated_tokens(seq),skip_special_tokens=False)}
        records = [item for item in seq.items if hasattr(item, "block") and hasattr(item, "meta")]
        response = {"id": seq.request.request_id, "object": "chat.completion", "created": int(time.time()),
                "model": self.model_name,
                "choices": [{"index": 0, "message": message, "finish_reason": seq.finish_reason}],
                "usage": {"prompt_tokens": seq.prompt_positions, "completion_tokens": seq.generated_positions,
                          "total_tokens": seq.prompt_positions + seq.generated_positions},
                "native_state": {"blocks": [record.meta() for record in records], "imports": seq.native_imports}}
        if seq.guide is not None:
            response['x_natlang_guidance'] = {'rejections': seq.guide.rejections}
        if seq.request.rl_capture or seq.native_protocol_failure is not None:
            response["x_natlang_rl_capture"] = {"schema": ("natlang.native-teacher-forced-capture/1"
                    if seq.native_forced_replay else "natlang.native-structural-capture/1"
                    if seq.request.temperature <= 0 else "natlang.native-rl-sampling/3"
                    if seq.native_rl_adapter_binding else "natlang.native-rl-sampling/2"),
                "policy": dict(seq.native_rl_policy),
                **({'teacher_source_receipt':seq.native_forced_source} if seq.native_forced_source is not None else {}),
                **({'replay_prompt_sha256':seq.native_forced_replay_prompt_sha256} if seq.native_forced_replay_prompt_sha256 else {}),
                **({'adapter_binding': seq.native_rl_adapter_binding} if seq.native_rl_adapter_binding else {}),
                "prompt": seq.native_rl_prompt, "samples": seq.rl_samples,
                "generated_inputs": seq.native_rl_inputs, "writes": seq.native_rl_writes,
                "finish_reason": seq.finish_reason,
                "sample_range_convention": "half-open",
                "sampling_law": {"temperature": seq.request.temperature, "seed": seq.native_sampling_seed,
                                 "guidance": seq.guide is not None, "probabilities": ("exact-forced-sequence"
                                     if seq.native_forced_replay else "greedy-argmax"
                                     if seq.request.temperature <= 0 else "post-processing-normalized")}}
        if seq.native_protocol_failure is not None:
            failure=seq.native_protocol_failure
            response['x_natlang_native_failure']=failure
            capture=response['x_natlang_rl_capture']
            capture['schema']=('natlang.native-structural-capture/2' if seq.request.temperature<=0
                               else 'natlang.native-rl-sampling/4')
            capture['protocol_failure']=failure
            response['usage']['completion_tokens']=len(seq.rl_samples)+sum(
                value.get('sample_index') is None for value in seq.native_rl_inputs)
            response['usage']['total_tokens']=response['usage']['prompt_tokens']+response['usage']['completion_tokens']
        return response
