"""The optimizer's native computation on the executor's base with the optimizer's adapters (plan "Native states the
writer can reach", "Weight arrays", "Evidence and output computation").

One incremental native computation (`native_step`, `splice_block`, the same layer math as graph training) runs:

1. the rendered optimizer request (program view, compact evidence, current bundle), splicing every native part it
   imports: current instruction/skill states, evidence digests and the bank view's component keys;
2. for each binding being rewritten, **contextual re-encoding**: the binding's current text tokens between producer
   markers, plus declared forced extra slots, captured as the new state. Positions and length are the target's, so
   state targets align slot for slot and gradients reach the writer without passing through sampled tokens;
3. for each scope, its weight array in the forced grid template (`numbers.layout`). With ``cue=True`` each slot's
   component key is re-imported as forced context before its sign token. The decision position (the cue's close
   marker, or the separator/open bracket without cues) gives the sign logits and the coefficient head's input; the
   position after the sign gives the digit logits. ``branch=True`` takes one extra step with the other sign token so
   `numbers.expectation` is exact.

With gradients enabled every output is a live function of the writer adapter's leaves, the coefficient head (applied
by the caller) and any live imported states (trainable keys, digests). Run it inside `run_native` with the writer's
entries. Sampling uses the grammar-restricted law and records each free token's log probability; forced tokens have
no sampling score. Without targets or sampling the arrays are decoded greedily: a declared deterministic reference run,
not a policy sample.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import torch

from . import LearnedOptimizerError
from ...common.stable_softmax import log_softmax
from .numbers import layout

WRITER_SCHEMA = 'natlang.lopt.writer/1'


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
    targets: tuple | None = None  # teacher-forced grid integers


@dataclass
class WriterResult:
    states: dict = field(default_factory=dict)          # key -> live NativeStateBlock
    decision_hidden: dict = field(default_factory=dict)  # scope -> (K, width)
    sign_logits: dict = field(default_factory=dict)      # scope -> (K, V)
    digit_logits: dict = field(default_factory=dict)     # scope -> (K, V) realized branch
    other_digit_logits: dict = field(default_factory=dict)
    integers: dict = field(default_factory=dict)         # scope -> list[int]
    logprobs: dict = field(default_factory=dict)         # scope -> (K,) grammar-restricted log p of free tokens
    tokens: dict = field(default_factory=dict)           # scope -> generated template IDs
    receipt: dict = field(default_factory=dict)


class Writer:
    def __init__(self, engine, grid, *, cue=True, extra_token=None, temperature=1.):
        self.engine, self.grid, self.cue, self.temperature = engine, grid, cue, float(temperature)
        self.model = engine.shared_model
        self.device = engine.device
        self.roles = engine.role_tokens
        self.extra_token = extra_token if extra_token is not None else grid.separator

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
    def run(self, messages, writes, arrays, *, state_overrides=None, sample=False, generator=None, branch=True):
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
            self._array(array, cache, result, imports, state_overrides, sample=sample, generator=generator, branch=branch)
            cache = result.receipt.pop('_cache')
        result.receipt.update({'schema': WRITER_SCHEMA, 'cue': self.cue, 'imports': imports, 'sampled': sample,
                               'temperature': self.temperature, 'branch': branch,
                               'writes': [{'key': w.key, 'kind': w.kind, 'extra_slots': w.extra_slots,
                                           'text_sha256': hashlib.sha256(w.text.encode()).hexdigest()} for w in writes]})
        return result

    def _array(self, array, cache, result, imports, overrides, *, sample, generator, branch):
        grid = self.grid
        K = len(array.available)
        greedy = array.targets is None and not sample  # a deterministic reference run, not a policy sample
        if self.cue and len(array.keys) != K:
            raise LearnedOptimizerError('cued decoding needs one key per slot')
        out = self._step(self._text(f' {array.scope}=') + [grid.open], cache)
        cache = out['cache']
        hidden, signs, digits, others, logps, integers = [], [], [], [], [], []
        tokens = [grid.open]
        for slot, free in enumerate(array.available):
            if slot:
                out = self._step([grid.separator], cache)
                cache = out['cache']
                tokens.append(grid.separator)
            if self.cue:
                out = self._splice(cache, array.keys[slot], overrides, imports)
                cache = out['cache']
            decision = out['h_final'][0, -1]
            sign_logits = self.model.logits(decision[None])[0]
            target = None if array.targets is None else int(array.targets[slot])
            if not free:
                value = 0
                if target not in (None, 0):
                    raise LearnedOptimizerError(f'teacher-forced target of unavailable slot {slot} must be 0')
            elif target is not None and not sample:
                value = target
            else:
                value = None
            sign_lp = log_softmax(torch.stack([sign_logits[grid.positive], sign_logits[grid.negative]]).float() / self.temperature, 0)
            if value is None and greedy:
                negative = bool(sign_lp[1] > sign_lp[0])
            elif value is None:
                negative = bool(torch.multinomial(sign_lp.exp().detach().cpu(), 1, generator=generator).item() == 1)
            else:
                negative = value < 0
            sign = grid.negative if negative else grid.positive
            after_sign = self._step([sign], cache)
            digit_logits = self.model.logits(after_sign['h_final'][0, -1][None])[0]
            index = torch.tensor(grid.digits, device=digit_logits.device)
            digit_lp = log_softmax(digit_logits[index].float() / self.temperature, 0)
            if negative:
                digit_lp = log_softmax(torch.cat([digit_lp.new_full((1,), -torch.inf), digit_logits[index[1:]].float() / self.temperature]), 0)
            if value is None:
                magnitude = (int(digit_lp.argmax()) if greedy else
                             int(torch.multinomial(digit_lp.exp().detach().cpu(), 1, generator=generator).item()))
                value = -magnitude if negative else magnitude
            magnitude = abs(value)
            if magnitude >= len(grid.digits):
                raise LearnedOptimizerError(f'grid value {value} is outside the digits')
            cache = self._step([grid.digits[magnitude]], after_sign['cache'])['cache']
            tokens.extend([sign, grid.digits[magnitude]])
            integers.append(value)
            hidden.append(decision)
            signs.append(sign_logits)
            digits.append(digit_logits)
            if free:
                logps.append(sign_lp[1 if negative else 0] + digit_lp[magnitude])
                if branch:
                    other = self._step([grid.positive if negative else grid.negative], out['cache'])
                    others.append(self.model.logits(other['h_final'][0, -1][None])[0])
            else:
                logps.append(sign_logits.new_zeros((), dtype=torch.float32))  # forced: no sampling score
                if branch:
                    others.append(digit_logits)
        out = self._step([grid.close], cache)
        tokens.append(grid.close)
        expected = [token for token, _, _ in layout(grid, integers, array.available)]
        if tokens != expected:
            raise LearnedOptimizerError('generated weight array deviates from its forced template')
        result.decision_hidden[array.scope] = torch.stack(hidden)
        result.sign_logits[array.scope] = torch.stack(signs)
        result.digit_logits[array.scope] = torch.stack(digits)
        if branch:
            result.other_digit_logits[array.scope] = torch.stack(others)
        result.integers[array.scope] = integers
        result.logprobs[array.scope] = torch.stack(logps)
        result.tokens[array.scope] = tokens
        result.receipt['_cache'] = out['cache']
