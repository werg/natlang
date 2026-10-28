"""The optimizer's native computation on the executor's base with the optimizer's adapters (plan "Native states the
writer can reach", "Weight arrays", "Evidence and output computation").

One incremental native computation (`native_step`, `splice_block`, the same layer math as graph training) runs:

1. the rendered optimizer request (program view, compact evidence, current bundle), splicing every native part it
   imports: current instruction/skill states, evidence digests and the bank view's component keys;
2. for each binding being rewritten, **contextual re-encoding**: the binding's current text tokens between producer
   markers, plus declared forced extra slots, captured as the new state. Positions and length are the target's, so
   state targets align slot for slot and gradients reach the writer without passing through sampled tokens;
3. for each scope, its component-token decision (`mixture`): the scope label, then (with ``cue=True``) the keys of
   the view's AVAILABLE components, each re-imported as forced context. The final hidden state is the decision
   position: the mixture head reads it for the component-token logits and the group gains. The hidden state at the
   end of each key is returned too (`mixture.MixtureHead.initialize_rows`).

With gradients enabled every output is a live function of the writer adapter's leaves and any live imported states
(trainable keys, digests); the mixture head is applied by the caller. Run it inside `run_native` with the writer's
entries. The computation is deterministic: no token is sampled.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import torch

from . import LearnedOptimizerError

WRITER_SCHEMA = 'natlang.lopt.writer/2'


@dataclass
class StateWrite:
    key: str             # bundle key: binding (instruction) or `binding#target` (skill)
    kind: str            # 'instruction' | 'skill'
    text: str            # the binding's current text: re-encoded in context
    extra_slots: int = 0


@dataclass
class ArrayWrite:
    scope: str
    available: tuple
    keys: tuple          # per slot: key state id (cue), or None without cues


@dataclass
class WriterResult:
    states: dict = field(default_factory=dict)          # key -> live NativeStateBlock
    decision_hidden: dict = field(default_factory=dict)  # scope -> (width,)
    key_hidden: dict = field(default_factory=dict)       # scope -> (available slots, width) at each key's end
    receipt: dict = field(default_factory=dict)


class Writer:
    def __init__(self, engine, *, cue=True, extra_token=None):
        self.engine, self.cue = engine, cue
        self.model = engine.shared_model
        self.device = engine.device
        self.roles = engine.role_tokens
        self.extra_token = extra_token if extra_token is not None else self._text(',')[-1]

    # -- primitives ------------------------------------------------------------------------------------------------
    def _step(self, ids, cache, producing=False):
        from ...model.shared_state import native_step
        return native_step(self.model, torch.tensor([ids], dtype=torch.long, device=self.device), cache, producing=producing)

    def _splice(self, cache, state_id, overrides, imports):
        """Consume one native state between consume markers; identity checked as exact or same-base exchange."""
        from ...model.shared_state import splice_block
        from ...model.native_adapters import execution_identity
        block = overrides.get(state_id) if overrides else None
        if block is None:
            record = self.engine.store.get_native_state(state_id)
            if record is None:
                raise LearnedOptimizerError(f'the optimizer request imports unknown state {state_id}')
            block = record.block
        current = execution_identity(self.engine)
        own = {'checkpoint_sha256': block.checkpoint_sha256, 'tokenizer_fingerprint': block.tokenizer_fingerprint,
               'embedding_weight_sha256': block.embedding_weight_sha256}
        if block.checkpoint_sha256 == current['checkpoint_sha256']:
            mode = 'exact-native-state'
        else:
            base = (block.adapter_binding or {}).get('base_native_identity') or {k: getattr(block, k) for k in (*own, 'architecture')}
            if base != self.engine.native_identity:
                raise LearnedOptimizerError(f'imported state {state_id} was written on another base')
            mode = 'same-base-adapter-state'
        cache = self._step([self.roles.consume_open], cache)['cache']
        start = cache.lengths[0]
        cache = splice_block(self.model, cache, block, **own)
        imports.append({'id': state_id, 'mode': mode, 'position_start': start, 'tokens': len(block.token_ids),
                        'live': bool(overrides and state_id in overrides)})
        return self._step([self.roles.consume_close], cache)

    def _text(self, text, escape=''):
        return self.engine._template_tokens(text, escape)

    def _capture_block(self, captured, text):
        from ...model.state_block import NativeStateBlock
        from ...model.shared_state import concatenate_captures
        from ...model.native_adapters import execution_binding, execution_identity, SCHEMA as ADAPTER_SCHEMA
        fields = concatenate_captures(captured)
        bindings = execution_binding(self.engine)
        engine = self.engine
        block = NativeStateBlock(text=text, tokenizer_fingerprint=engine.model_config['tokenizer_fingerprint'],
                                 checkpoint_sha256=execution_identity(engine)['checkpoint_sha256'],
                                 embedding_weight_sha256=engine.embedding_weight_sha256,
                                 architecture=engine.model_config.get('architecture') or engine.backbone.config.model_type,
                                 adapter_binding=({'schema': ADAPTER_SCHEMA, 'base_native_identity': engine.native_identity,
                                                   'adapters': bindings} if bindings else None), **fields)
        block.validate()
        return block

    # -- the computation -------------------------------------------------------------------------------------------
    def run(self, messages, writes, arrays, *, state_overrides=None):
        engine = self.engine
        result = WriterResult()
        imports = []
        rendered = engine.render_request(messages, [])
        cache, last = None, None
        for segment in rendered.segments:
            if isinstance(segment, int):
                out = self._splice(cache, rendered.blocks[segment], state_overrides, imports)
            else:
                ids = self._text(segment, rendered.escape_nonce)
                if not ids:
                    continue
                out = self._step(ids, cache)
            cache, last = out['cache'], out
        if last is None:
            raise LearnedOptimizerError('the optimizer request rendered no tokens')
        from ...serve.chat import escape_specials
        for write in writes:
            if write.kind not in ('instruction', 'skill') or type(write.extra_slots) is not int or write.extra_slots < 0:
                raise LearnedOptimizerError('a state write is an instruction or skill with nonnegative extra slots')
            label = self._text(f' {write.key}=')
            cache = self._step(label + [self.roles.produce_open], cache)['cache']
            nonce = hashlib.sha256(write.text.encode()).hexdigest()
            body = self._text(escape_specials(write.text, engine.specials, nonce), nonce)
            produced = self._step(body + [self.extra_token] * write.extra_slots, cache, producing=True)
            cache = self._step([self.roles.produce_close], produced['cache'])['cache']
            result.states[write.key] = self._capture_block([produced], write.text)
        for array in arrays:
            cache = self._array(array, cache, result, imports, state_overrides)
        result.receipt.update({'schema': WRITER_SCHEMA, 'cue': self.cue, 'imports': imports,
                               'writes': [{'key': w.key, 'kind': w.kind, 'extra_slots': w.extra_slots,
                                           'text_sha256': hashlib.sha256(w.text.encode()).hexdigest()} for w in writes]})
        return result

    def _array(self, array, cache, result, imports, overrides):
        if self.cue and len(array.keys) != len(array.available):
            raise LearnedOptimizerError('cued decisions need one key per slot')
        out = self._step(self._text(f' {array.scope}='), cache)
        cache, keys = out['cache'], []
        for slot, free in enumerate(array.available):
            if free and self.cue:
                out = self._splice(cache, array.keys[slot], overrides, imports)
                cache = out['cache']
                keys.append(out['h_final'][0, -1])
        result.decision_hidden[array.scope] = out['h_final'][0, -1]
        result.key_hidden[array.scope] = torch.stack(keys) if keys else out['h_final'][0, :0]
        return cache
