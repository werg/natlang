"""Serving for native shared-state checkpoints.

The scheduler generates ordinary text with the model's native token head and
captures typed producer spans as positional KV and hybrid state. Imported
native-state parts are spliced with their exact source identity and recorded
positions. There is no projected Neuralese serving path in this module.
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

import torch

from .chat import RequestError, build_message, placeholder, render_messages, split_escaped
from .native_protocol import capture_enabled
from ..common.sampling import behavior_top_k_declaration
from .store import TensorStore


def derive_seed(*parts) -> int:
    digest = hashlib.sha256(repr(parts).encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**31)


def _insert_privileged(plan, block):
    """Split the prompt plan's token segments at the block's own-token index (never at a native-state import)."""
    at, tokens = block.get('insert_at'), block.get('token_ids')
    if (type(at) is not int or at < 0 or not isinstance(tokens, list) or not tokens
            or any(type(token) is not int or token < 0 for token in tokens)):
        raise RequestError('bad-request', 'a privileged block needs its own-token insert_at and token_ids')
    result, own, inserted = [], 0, False
    for number, (kind, value) in enumerate(plan):
        if kind != 'tokens':
            if not inserted and own == at:
                raise RequestError('bad-request', 'a privileged block cannot share a native-state import boundary')
            result.append((kind, value))
            continue
        ids = list(value)
        if not inserted and own <= at <= own + len(ids):
            cut = at - own
            following = plan[number+1][0] if number+1 < len(plan) else None
            if cut == len(ids) and following not in (None, 'tokens'):
                raise RequestError('bad-request', 'a privileged block cannot share a native-state import boundary')
            result.extend(piece for piece in (('tokens', ids[:cut]), ('privileged', tokens), ('tokens', ids[cut:]))
                          if piece[1])
            inserted = True
        else:
            result.append((kind, value))
        own += len(ids)
    if not inserted:
        raise RequestError('bad-request', 'privileged insertion point is outside the rendered prompt')
    return result


@dataclass
class GenerationRequest:
    messages: list
    tools: list | None = None
    max_tokens: int = 512
    temperature: float = 0.0
    seed: int | None = None
    forced_message: dict | None = None
    template: dict | None = None
    guidance: object = None
    request_id: str = ""
    on_delta: object = None
    adapters: list | None = None
    cancelled: bool = False
    native_protocol_recovery: bool = False
    native_store_owner: str | None = None
    rl_capture: bool = False
    # Score-centered RL: also record the sampler's top-k log-probs per sampled
    # position (common.sampling behaviour top-k receipt). None records none.
    rl_top_k: int | None = None
    # JOLT+ teacher attempt: {'schema': 'natlang.privileged-request/1', 'custody', 'facts', 'invocation': {name,
    # role}}. The engine renders this invocation's rl.privileged_context block from its own prompt (the learner's
    # rendering and selection), inserts it at its declared own-token index and marks it 'privileged' in the capture.
    privileged_context: dict | None = None
    native_recording: bool = False
    native_state_modes: dict | None = None


@dataclass
class Sequence:
    request: GenerationRequest
    future: Future | None
    phase: str = "prefill"
    prompt_plan: list | None = None
    cache: object = None
    logits: torch.Tensor | None = None
    items: list = field(default_factory=list)
    blocks: list = field(default_factory=list)
    prompt_positions: int = 0
    generated_positions: int = 0
    guide: object = None
    pieces: list = field(default_factory=list)
    bans: dict = field(default_factory=dict)
    rl_samples: list = field(default_factory=list)
    retries: dict = field(default_factory=dict)
    finish_reason: str = "stop"
    rng: torch.Generator | None = None
    pending: list = field(default_factory=list)
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
    native_sample_top_k: dict | None = None
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
    def __init__(self, backbone, tokenizer, store: TensorStore, dialect: str,
                 max_block: int = 64, model_name: str = "natlang-native", device: str = "cpu",
                 prefill_tokens: int = 8192, context: int | None = None,
                 prompt_rendering: str = "full", *, shared_model, role_tokens, model_config,
                 checkpoint_sha256: str, embedding_weight_sha256: str):
        from .chat import bind_history_reasoning
        from ..data.prompt_rendering import PROMPT_RENDERINGS
        if shared_model is None or role_tokens is None or not checkpoint_sha256 or not embedding_weight_sha256:
            raise ValueError("native serving requires a shared-state model and exact checkpoint/embedding identities")
        if prompt_rendering not in PROMPT_RENDERINGS:
            raise ValueError(f"unknown prompt_rendering {prompt_rendering!r}; expected one of {PROMPT_RENDERINGS}")
        self.model_config = dict(model_config)
        tokenizer = bind_history_reasoning(tokenizer, policy=self.model_config.get('history_reasoning_policy'),
            model_type=getattr(getattr(backbone, "config", None), "model_type", None))
        self.backbone, self.tokenizer, self.store = backbone, tokenizer, store
        self.shared_model, self.role_tokens = shared_model, role_tokens
        self.native_state_enabled = True
        self.native_execution_lock = threading.RLock()
        self.checkpoint_sha256 = checkpoint_sha256
        self.embedding_weight_sha256 = embedding_weight_sha256
        from .chat import tool_call_format
        # Declared by the backbone's model_type (chat.TOOL_CALL_FORMATS), never inferred from its vocabulary.
        self.tool_call_format = tool_call_format(getattr(getattr(backbone, "config", None), "model_type", None))
        self.native_identity = {'checkpoint_sha256': checkpoint_sha256,
            'tokenizer_fingerprint': self.model_config.get('tokenizer_fingerprint'),
            'embedding_weight_sha256': embedding_weight_sha256,
            'architecture': self.model_config.get('architecture') or self.backbone.config.model_type}
        self.prefill_tokens = prefill_tokens
        # Optional exact prompt-prefix reuse (serve/prefix_cache.py); None keeps single-chunk prefill.
        self.prefix_cache = None
        self.dialect, self.max_block, self.model_name, self.device = dialect, max_block, model_name, device
        self.prompt_rendering = prompt_rendering
        self.width = backbone.config.hidden_size
        config = getattr(getattr(backbone, "hf", None), "config", None) or backbone.config
        trained = int(getattr(config, "max_position_embeddings", None) or 32768)
        self.context = min(trained, int(context)) if context else trained
        self.stop_ids = {i for i in (tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.eos_token_id) if i is not None}
        self._incoming: queue.Queue = queue.Queue()
        self._active: list[Sequence] = []
        self._thread: threading.Thread | None = None
        self._stopping = False
        self._ids = itertools.count()
        self._native_decoded_tokens = 0

    def progress_metrics(self):
        """Host-only scheduler observations; never block on actor execution/GPU."""
        active = tuple(self._active)
        return {'natlang_native_decode_tokens_total': self._native_decoded_tokens,
                'natlang_native_batched_decode_rounds_total': getattr(self, '_native_batched_decode_rounds', 0),
                'natlang_native_batched_decode_tokens_total': getattr(self, '_native_batched_decode_tokens', 0),
                'natlang_active_sequences': sum(seq.phase != 'done' for seq in active),
                'natlang_native_open_produce_spans': sum(
                    seq.phase != 'done' and seq.native_capture_open for seq in active),
                'natlang_queued_requests': self._incoming.qsize(),
                **(self.prefix_cache.metrics() if self.prefix_cache is not None else {})}

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

    def _template(self, messages, tools):
        return self.tokenizer.apply_chat_template(messages, tools=tools or None, tokenize=False,
                                                  add_generation_prompt=True)

    def render_request(self, messages, tools=None, template=None):
        """Render an ordinary text or native-state prompt under the declared chat-template policy."""
        from ..data.prompt_rendering import apply_prompt_rendering
        from .chat import render_messages

        # In program-header mode the natlang runtime sends no generic tools. A separate
        # client (for example an app or harness) can still supply its own tool schemas.
        try:
            messages, tools = apply_prompt_rendering(messages, tools, self.prompt_rendering,
                                                     keep_supplied_tools=True)
        except ValueError as error:  # the messages do not fit the declared prompt rendering
            raise RequestError("prompt-rendering", str(error)) from error
        return render_messages(messages, tools, template or self._template, self.specials,
                               call_format=getattr(self, 'tool_call_format', None))

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
            found = self._specials = tuple(sorted(n for n in names if n))
        return found

    def _privileged_rendering(self):
        rendering = getattr(self, '_privileged_rendering_cache', None)
        if rendering is None:
            from types import SimpleNamespace
            from ..data.render import Renderer
            from ..rl.privileged_context import PrivilegedRendering
            from .chat import tokenizer_model_type
            renderer = Renderer(self.tokenizer, SimpleNamespace(open_id=self.role_tokens.produce_open,
                                                                close_id=self.role_tokens.produce_close))
            rendering = PrivilegedRendering.from_renderer(renderer, template=tokenizer_model_type(self.tokenizer),
                tokenizer_fingerprint=self.native_identity['tokenizer_fingerprint'])
            self._privileged_rendering_cache = rendering
        return rendering

    def _privileged_block(self, request, plan):
        """This invocation's JOLT+ privileged block, rendered exactly as the learner rebuilds it (None: no fact fits)."""
        from ..rl.privileged_context import Fact, render_block
        context = request.privileged_context
        invocation = context.get('invocation') if isinstance(context, dict) else None
        if (not isinstance(context, dict) or set(context) != {'schema', 'custody', 'facts', 'invocation'}
                or context['schema'] != 'natlang.privileged-request/1' or not isinstance(context['custody'], dict)
                or not isinstance(context['facts'], list) or not isinstance(invocation, dict)
                or set(invocation) != {'name', 'role'} or not isinstance(invocation['name'], str)
                or invocation['role'] not in ('root', 'child')):
            raise RequestError('bad-request', 'x_natlang_privileged_context needs schema, custody, facts and the '
                               'invocation {name, role}')
        try:
            facts = tuple(Fact(item['id'], item['kind'], item['text'], item.get('invocation'),
                               tuple(item.get('match') or ())) for item in context['facts'])
        except (KeyError, TypeError) as error:
            raise RequestError('bad-request', 'privileged facts are {id, kind, text, invocation?, match?}') from error
        own = [token for kind, value in plan if kind == 'tokens' for token in value]
        text = '\n'.join(message['content'] for message in request.messages or []
                         if isinstance(message, dict) and isinstance(message.get('content'), str))
        block = render_block(self._privileged_rendering(), invocation={**invocation, 'text': text}, token_ids=own,
                             facts=facts, custody_receipt=context['custody'])
        return {'insert_at': block.insert_at, 'token_ids': list(block.token_ids)} if block.token_ids else None

    def _prompt_plan(self, messages, tools) -> list:
        """Render and tokenize on the caller's CPU thread; preserve segment boundaries."""
        rendered = self.render_request(messages, tools)
        plan = []
        for segment in rendered.segments:
            if isinstance(segment, str):
                plan.append(("tokens", self._template_tokens(segment, rendered.escape_nonce)))
            else:
                kind = 'native_state'
                if rendered.block_stop_gradients and rendered.block_stop_gradients[segment]:
                    kind='native_state_constant'
                value = rendered.blocks[segment]
                if rendered.block_source_occurrences and rendered.block_source_occurrences[segment]:
                    value = {'id': value, 'source_occurrence': rendered.block_source_occurrences[segment]}
                plan.append((kind, value))
        return plan

    def prompt_positions(self, messages, tools) -> int:
        """Count text tokens and native-state spans without executing the model."""
        positions = 0
        for kind, value in self._prompt_plan(messages, tools):
            if kind == 'tokens':
                positions += len(value)
            else:
                record = self.store.get_native_state(value['id'] if isinstance(value,dict) else value)
                if record is None:
                    raise RequestError('neuralese-unknown-native-state', f'{value} is not in this server store; restore it first')
                positions += 2 + (len(record.block.token_ids) if self._native_exact(record.block)
                                  else len(self._tokens(record.block.text)))
        return positions

    def _native_run(self, callback, adapters=None):
        from ..model.native_adapters import run_native, NativeAdapterError
        # In-process task evaluation and guided serving share the same declared
        # actor policy as the HTTP client.  HTTP requests already carry their
        # resolved defaults explicitly; local callers inherit them here.
        from ..model.native_adapters import execution_binding
        if adapters is None and not execution_binding(self):
            adapters = getattr(self, 'native_default_adapters', None)
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
        from ..model.native_adapters import live_adapter_leaves
        if cache is None and self.prefix_cache is not None and not torch.is_grad_enabled() and not live_adapter_leaves():
            return self._native_prefix_prefill(ids)
        for start in range(0, len(ids), self.prefill_tokens):
            output = self._native_forward(ids[start:start+self.prefill_tokens], cache)
            cache = output['cache']
            if start + self.prefill_tokens < len(ids):
                # Do not retain the preceding chunk's raw captures/hidden state
                # while allocating the next chunk's layer workspace.
                del output
        return output

    def _native_prefix_prefill(self, ids):
        """A fresh prompt prefilled on the prefix cache's block grid, resuming from its longest stored prefix."""
        from ..model.native_adapters import execution_binding
        prefix = self.prefix_cache
        digests = prefix.digests({'adapters': execution_binding(self), 'identity': self.native_identity}, ids)
        start, cache, output = 0, None, None
        hit = prefix.lookup(digests)
        if hit is not None:
            start, cache, h_last = hit
            output = {'cache': cache, 'h_final': h_last}
        for begin in range(start, len(ids), prefix.block):
            end = min(begin + prefix.block, len(ids))
            output = self._native_forward(ids[begin:end], cache)
            cache = output['cache']
            prefix.store(digests[end], cache, output['h_final'][:, -1:])
        return output

    def _native_logits(self, hidden):
        return self._native_run(lambda: self.shared_model.logits(hidden[:, -1]))

    def _native_exact(self, block):
        architecture = self.model_config.get("architecture") or self.backbone.config.model_type
        layer_kinds = tuple("attention" if self.backbone.is_attention(index) else
                            "gated_delta" if self.backbone.layer_types[index] == 'linear_attention' else
                            "finite_convolution" for index in range(self.backbone.num_layers))
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
                    quoted=False, argument_path=argument_path, call_format=getattr(self, "tool_call_format", None))
                template_prefix, template_suffix = self._tokens(prefix), []
            else:
                value_type = template.get('value_type', 'string')
                prefix, suffix = write_reply(apply, template['call'], arguments, argument, value_type,
                                             argument_path=argument_path, call_format=getattr(self, "tool_call_format", None))
                template_prefix, template_suffix = self._tokens(prefix), self._tokens(suffix)
                seq.native_template_value_type = value_type
        plan = list(seq.prompt_plan or ())
        if seq.request.privileged_context is not None:
            block = self._privileged_block(seq.request, plan)
            if block is not None:
                plan = _insert_privileged(plan, block)
        for kind, value in plan:
            privileged = kind == 'privileged'
            if privileged:
                kind = 'tokens'
            if kind == "tokens":
                if capture_enabled(seq) and value:
                    seq.native_rl_prompt.append({"kind": "tokens", "token_ids": list(value),
                                                 "producing": [False] * len(value),
                                                 **({'privileged': True} if privileged else {})})
                seq.native_own_position += len(value)
                if value:
                    output = self._native_prefill_forward(value, seq.cache)
                    seq.cache = output["cache"]
                    seq.logits = self._native_logits(output["h_final"])
            elif kind in ("native_state","native_state_constant"):
                self._native_consume(seq, value['id'] if isinstance(value,dict) else value,stop_gradient=kind=='native_state_constant',
                    source_occurrence=value.get('source_occurrence') if isinstance(value,dict) else None)
            else:
                raise ValueError(f"unsupported native prompt segment {kind!r}")
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
            if seq.request.rl_top_k is not None:
                from ..common.sampling import encode_top_k
                seq.native_sample_top_k = encode_top_k(log_probs, seq.request.rl_top_k)
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
        segments=assistant_reply_segments(apply,target,self.specials,opening=opening,
            call_format=getattr(self,"tool_call_format",None),tools=tools)
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
        self._validate_native_state_modes(native_state_modes, set(rendered.blocks))
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
                    block_id = rendered.blocks[segment]
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

    def _native_step_logits(self, output):
        return output['logits'] if 'logits' in output else self._native_logits(output['h_final'])

    def _native_round(self, sequences):
        # Exact adapter declarations define disjoint model executions. An absent
        # declaration inherits defaults; an explicit empty one does not.
        from ..common.hashing import canonical_json_sha256_hex
        groups = {}
        for seq in sequences:
            try:
                key = canonical_json_sha256_hex(seq.request.adapters)
                groups.setdefault(key, []).append(seq)
            except Exception as error:
                self._fail(seq, error)
        for group in groups.values():
            try:
                self._native_run(lambda: self._native_round_group(group), group[0].request.adapters)
            except Exception as error:
                for seq in group:
                    if not seq.future.done(): self._native_protocol_fail(seq, error)

    def _native_round_group(self, sequences):
        from ..model.native_decode_batch import native_step_batch, validate_decode_cache
        pending = []
        for seq in sequences:
            step = self._native_sequence_step(seq)
            try:
                request = next(step)
                validate_decode_cache(self.backbone, request[1])
                pending.append((seq, step, request))
            except StopIteration:
                pass
            except Exception as error:
                step.close()
                self._native_protocol_fail(seq, error)
        if not pending: return
        if len(pending) == 1:
            _, _, (ids, cache, producing) = pending[0]
            outputs = [self._native_forward(ids, cache, producing=producing)]
        else:
            ids = torch.tensor([request[0] for _, _, request in pending], dtype=torch.long,
                               device=self.backbone.embedding_weight.device)
            outputs = native_step_batch(self.shared_model, ids,
                [request[1] for _, _, request in pending],
                producing=[request[2] for _, _, request in pending])
            logits = self.shared_model.logits(torch.cat([out['h_final'][:, -1] for out in outputs], dim=0))
            for index, output in enumerate(outputs): output['logits'] = logits[index:index+1]
            self._native_batched_decode_rounds = getattr(self, '_native_batched_decode_rounds', 0) + 1
            self._native_batched_decode_tokens = getattr(self, '_native_batched_decode_tokens', 0) + len(pending)
        for (seq, step, _), output in zip(pending, outputs):
            try:
                step.send(output)
            except StopIteration:
                pass
            except Exception as error:
                self._native_protocol_fail(seq, error)
            else:
                step.close()
                self._fail(seq, RuntimeError('one native decode round yielded multiple model steps'))

    def _native_round_impl(self, sequences):
        """Independent cached reference for the same single-token state machine."""
        for seq in sequences:
            step = self._native_sequence_step(seq)
            try:
                ids, cache, producing = next(step)
                step.send(self._native_forward(ids, cache, producing=producing))
            except StopIteration:
                pass
            finally:
                step.close()

    def _native_sequence_step(self, seq):
        """Preserve per-request sampling, protocol, forced turns and capture."""
        if seq.request.cancelled:
            self._fail(seq, RequestError("cancelled", "the client went away"))
            return
        if seq.phase == "done":
            return
        try:
            if (capture_enabled(seq) and seq.native_rl_policy is not None
                    and self.execution_native_identity != seq.native_rl_policy):
                raise RequestError('neuralese-native-actor-changed',
                    'keep the exact adapter binding unchanged until this invocation finishes')
            if seq.phase == "prefill":
                self._native_prefill(seq)
                return
            if seq.native_forced_tokens and not seq.native_forced_replay:
                token = seq.native_forced_tokens.pop(0)
                if token in self.stop_ids:
                    self._finish(seq, 'stop')
                    return
                output = yield ([token], seq.cache, False)
                seq.cache, seq.logits = output['cache'], self._native_step_logits(output)
                seq.generated_positions += 1
                self._native_decoded_tokens += 1
                self._native_record_input(seq, token,
                    producing=seq.native_capture_open and token not in (
                        self.role_tokens.produce_open, self.role_tokens.produce_close))
                seq.items.append(token)
                seq.native_item_positions.append(seq.native_own_position - 1)
                if not self._native_guide(seq):
                    self._emit(seq, token=token)
                return
            if seq.native_forced_replay and not seq.native_forced_tokens:
                if seq.native_capture_open:
                    raise RequestError('neuralese-native-state-open','forced teacher reply ended inside a producer span')
                self._finish(seq,'stop')
                return
            if seq.generated_positions >= seq.request.max_tokens:
                if seq.native_capture_open:
                    raise RequestError("neuralese-native-state-open", "generation ended inside a produce span")
                self._finish(seq, "length")
                return
            forced=seq.native_forced_replay
            seq.native_sample_top_k = None
            token, sampled_logp = ((seq.native_forced_tokens.pop(0),None) if forced else self._native_token(seq))
            top_k_receipt = None
            if seq.request.rl_top_k is not None and not forced and seq.request.temperature > 0:
                top_k_receipt, seq.native_sample_top_k = seq.native_sample_top_k, None
                if top_k_receipt is None:
                    # A sampler that did not record its distribution: the same normalized law over these logits.
                    from ..common.sampling import encode_top_k, masked_log_distribution
                    top_k_receipt = encode_top_k(masked_log_distribution(seq.logits[0], seq.request.temperature,
                        banned=sorted(seq.bans.get(seq.native_own_position, ()))), seq.request.rl_top_k)
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
                                       **({'top_k': top_k_receipt} if top_k_receipt is not None else {}),
                                       **({'banned_token_ids': sorted(seq.bans.get(seq.native_own_position, ()))}
                                          if seq.guide is not None else {}),
                                       "input_role": "produce" if producing_input else "ordinary",
                                       "terminal": terminal,
                                       "query_own_position": query_own_position})
            if terminal:
                if seq.native_capture_open:
                    raise RequestError("neuralese-native-state-open", "generation stopped inside a produce span")
                self._finish(seq, "stop")
                return
            marker = self.role_tokens
            if token == marker.produce_close and not seq.native_capture_open:
                raise RequestError("neuralese-native-state-close", "generation emitted a produce end marker without an open span")
            if token in (marker.consume_open, marker.consume_close):
                raise RequestError("neuralese-native-state-consume-marker",
                                   "consume markers are inserted only around an imported native-state part")
            if seq.native_capture_open and token == marker.produce_close:
                seq.native_write_close_sample = sample_index
                output = yield ([token], seq.cache, False)
                seq.cache, seq.logits = output["cache"], self._native_step_logits(output)
                seq.generated_positions += 1
                self._native_decoded_tokens += 1
                self._native_record_input(seq, token, producing=False, sample_index=sample_index)
                self._native_finish_block(seq)
                return
            producing = seq.native_capture_open and token != marker.produce_open
            if token == marker.produce_open:
                seq.native_write_start_own = seq.native_own_position
                seq.native_write_open_sample = sample_index
            output = yield ([token], seq.cache, producing)
            seq.cache, seq.logits = output["cache"], self._native_step_logits(output)
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

    def _round(self, sequences):
        self._native_round(sequences)

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

    def _finish(self, seq: Sequence, reason: str):
        if seq.guide is not None:
            from .native_guidance import flush
            flush(self, seq)
        if seq.request.on_delta is not None and seq.pending:
            seq.request.on_delta({"text": self.tokenizer.decode(seq.pending, skip_special_tokens=False)})
            seq.pending = []
        seq.finish_reason = reason
        seq.phase = "done"
        if not seq.future.done():
            seq.future.set_result(self._native_response(seq))

    def _emit(self, seq: Sequence, token: int):
        """Streaming: hand new output to the request's listener, if any."""
        if seq.guide is not None and not seq.native_guidance_flushing:
            return
        listener = seq.request.on_delta
        if listener is None:
            return
        text = self.tokenizer.decode(seq.pending + [token], skip_special_tokens=False)
        if "\ufffd" in text:  # an incomplete UTF-8 sequence; wait for the next token
            seq.pending.append(token)
            return
        seq.pending = []
        listener({"text": text})

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
        message = build_message("".join(text), block_ids, call_prefix=seq.request.request_id,
                                tools=seq.request.tools)

        for call in message.get("tool_calls") or []:
            try:
                args = json.loads(call["function"]["arguments"])
                call["function"]["arguments"] = arguments_text(args)
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
                **({'behavior_top_k': behavior_top_k_declaration(seq.request.rl_top_k)}
                   if seq.request.rl_top_k is not None and seq.request.temperature > 0 and not seq.native_forced_replay else {}),
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
