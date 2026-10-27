"""The outer learner-training operation (plan "The outer learner-training operation", "One continuous training
operation", "Gradients for native states, numerical tokens and the shared bank").

``theta`` = the writer adapter on the executor's base, the coefficient head, the component keys (native states) and,
when the recipe trains them, the bank's component factors and basis. One logical update over a batch of complete
target tuples runs three phases against ONE parameter snapshot (see DECOMPOSITION.md, "Gradient path"):

A. writer forward under the writer adapter without gradients; states, decision hidden states and decision logits
   are detached into leaves;
B. executor objectives with those leaves: distillation against the target bundle on its training-role histories,
   state/decoded-weight/head auxiliaries. Composed cores are live functions of the head outputs (realized integers,
   or `token-expectation-v1`) and the bank leaves; backward yields the leaves' cotangents;
C. writer forward again under the writer adapter with gradients, ``backward(outputs, cotangents)`` plus the
   writer-local mixture-token CE. Exact cotangent replay: the plan's sanctioned split form.

Then every active group updates once: the writer adapter through `optimize_adapter` (immutable lineage, continued
moments), keys through `optimize_state`, head and bank through AdamW with FP32 masters. Bank/basis changes publish a
new snapshot only every ``publish_every`` updates; labels are then marked stale for refitting.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import torch

from ...common.hashing import canonical_json_sha256_hex
from . import LearnedOptimizerError
from . import objectives as O
from .head import CoefficientHead, executed, expected, refinement, modulation, shares_from_norms
from .numbers import expectation
from .views import BankView
from .writer import ArrayWrite, StateWrite, Writer

STATE_SCHEMA = 'natlang.lopt.trainer-state/1'


@dataclass
class OuterRecipe:
    steps: int = 1
    batch_targets: int = 2
    seed: int = 0
    writer: dict = field(default_factory=lambda: {'kind': 'free-lora', 'rank': 8, 'alpha': 8.})
    writer_optimizer: dict = field(default_factory=lambda: {'name': 'adam', 'learning_rate': 1e-4})
    head_optimizer: dict = field(default_factory=lambda: {'learning_rate': 1e-3, 'weight_decay': 0.})
    key_optimizer: dict = field(default_factory=lambda: {'name': 'adam', 'learning_rate': 1e-4})
    bank_optimizer: dict = field(default_factory=lambda: {'learning_rate': 1e-4, 'weight_decay': 0.})
    train_keys: bool = True
    train_bank: bool = False
    train_basis: bool = False
    publish_every: int = 50
    weights: dict = field(default_factory=lambda: {'distill': 1., 'downstream': 0., 'instruction_state': .1,
                                                   'skill_state': .1, 'mixture_tokens': .2, 'core': .05, 'head': .05})
    coefficients: str = 'realized'        # 'realized' | 'token-expectation-v1'
    head_mode: str = 'head'               # 'head' | 'head-off'
    beta: float = .5
    cue: bool = True
    zero_share: float = .5

    @classmethod
    def from_json(cls, value):
        unknown = set(value) - set(cls.__dataclass_fields__)
        if unknown:
            raise LearnedOptimizerError(f'unknown outer recipe fields {sorted(unknown)}')
        recipe = cls(**value)
        if recipe.coefficients not in ('realized', 'token-expectation-v1') or recipe.head_mode not in ('head', 'head-off'):
            raise LearnedOptimizerError('coefficients is realized|token-expectation-v1; head_mode is head|head-off')
        if set(recipe.weights) - set(O.TERMS):
            raise LearnedOptimizerError('objective weights name unknown terms')
        return recipe

    def as_json(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def _hex(tensor):
    return [float(x).hex() for x in tensor.detach().double().reshape(-1).tolist()]


class OuterTrainer:
    """Resumable outer trainer over admitted target tuples (`targets.target_record` plus their training context)."""

    def __init__(self, engine, bank, grid, recipe, targets, *, state=None, owner=None, extra_token=None):
        self.engine, self.bank, self.grid, self.recipe, self.owner = engine, bank, grid, recipe, owner
        self.targets = list(targets)  # empty when the trainer only applies the optimizer (`episode.propose`)
        for target in self.targets:
            if target['bank'] != bank.snapshot:
                raise LearnedOptimizerError(f'target {target["id"]} was labeled against another bank snapshot; refit it')
        width = engine.shared_model.port.embedding_weight.shape[1]
        self.head = CoefficientHead(width, len(bank.group_labels), beta=recipe.beta)
        self.writer = Writer(engine, grid, cue=recipe.cue, extra_token=extra_token)
        self.state = state or self._initial_state()
        if self.state['schema'] != STATE_SCHEMA or self.state['bank'] != bank.snapshot:
            raise LearnedOptimizerError('trainer state belongs to another bank snapshot or schema')
        self._restore()

    # -- state ---------------------------------------------------------------------------------------------------
    def _initial_state(self):
        from ...model.native_adapters import create_adapter, target_parameters
        names = sorted(target_parameters(self.engine.shared_model))
        spec = self.recipe.writer
        writer = create_adapter(self.engine, {'kind': spec['kind'], 'rank': spec['rank'], 'seed': self.recipe.seed,
                                              'alpha': float(spec.get('alpha', spec['rank'])), 'parameter_names': names},
                                self.owner)
        return {'schema': STATE_SCHEMA, 'bank': self.bank.snapshot, 'step': 0, 'epoch': 0, 'position': 0,
                'writer': writer.id, 'writer_optimizer': None, 'writer_history': [writer.id],
                'head': None, 'head_optimizer': None, 'keys': {c.id: c.key for c in self.bank.components},
                'key_optimizers': {}, 'bank_tensors': None, 'bank_optimizer': None, 'labels_stale': False,
                'exposure': {}, 'losses': [], 'recipe': self.recipe.as_json()}

    def _restore(self):
        store = self.engine.store
        if self.state['head']:
            tensors = store.get_native_tensor_artifact(self.state['head']).artifact.tensors
            self.head.load_state_dict({'readout.weight': tensors['weight'], 'readout.bias': tensors['bias']})
        self.head_optimizer = torch.optim.AdamW(self.head.parameters(), lr=self.recipe.head_optimizer['learning_rate'],
                                                weight_decay=self.recipe.head_optimizer.get('weight_decay', 0.))
        if self.state['head_optimizer']:
            self.head_optimizer.load_state_dict(_load_optimizer(store, self.state['head_optimizer'], self.head_optimizer))
        self.bank_leaves = None
        if self.recipe.train_bank or self.recipe.train_basis:
            self.bank_leaves = _bank_leaves(self.bank, self.recipe)
            params = [v for v in self.bank_leaves['factors'].values() for v in v] + list(self.bank_leaves['basis'].values())
            self.bank_optimizer = torch.optim.AdamW([p for p in params if p.requires_grad],
                                                    lr=self.recipe.bank_optimizer['learning_rate'],
                                                    weight_decay=self.recipe.bank_optimizer.get('weight_decay', 0.))
            if self.state['bank_optimizer']:
                self.bank_optimizer.load_state_dict(_load_optimizer(store, self.state['bank_optimizer'], self.bank_optimizer))

    # -- batches -------------------------------------------------------------------------------------------------
    def _next_batch(self):
        if not self.targets:
            raise LearnedOptimizerError('supervised outer training needs admitted target tuples')
        size = min(self.recipe.batch_targets, len(self.targets))
        batch, epoch, position = [], self.state['epoch'], self.state['position']
        order = torch.randperm(len(self.targets), generator=torch.Generator().manual_seed(self.recipe.seed * 7919 + epoch)).tolist()
        while len(batch) < size:
            if position == len(order):
                epoch, position = epoch + 1, 0
                order = torch.randperm(len(self.targets), generator=torch.Generator().manual_seed(self.recipe.seed * 7919 + epoch)).tolist()
            batch.append(order[position])
            position += 1
        return batch, epoch, position

    def _writer_inputs(self, target):
        writes = [StateWrite(w['key'], w['kind'], w['text'], w.get('extra_slots', 0)) for w in target['writes']]
        arrays = []
        for item in target['arrays']:
            view = item['view']
            keys = tuple(self.state['keys'][component] for component in view['componentIds']) if self.recipe.cue else (None,) * len(view['componentIds'])
            arrays.append(ArrayWrite(view['scope'], tuple(view['available']), keys, tuple(item['integers'])))
        return writes, arrays

    def _key_overrides(self, live_keys):
        return {state: block for state, (block, _) in live_keys.items()}

    # -- one logical update -------------------------------------------------------------------------------------
    def step(self):
        from ...model.native_adapters import run_native
        from ...serve.native_grad import _leaf_block
        engine, recipe, store = self.engine, self.recipe, self.engine.store
        batch, epoch, position = self._next_batch()
        writer_values = {k: v.detach().clone().requires_grad_(True)
                         for k, v in store.get_native_tensor_artifact(self.state['writer']).artifact.tensors.items()}
        writer_entries = [{'id': self.state['writer'], 'scale': 1.}]
        live_keys = {}
        if recipe.train_keys and recipe.cue:
            for state in sorted(set(self.state['keys'].values()) - {None}):
                live_keys[state] = _leaf_block(store.get_native_state(state).block)
        shares_all = shares_from_norms(self.bank.group_norms())
        records = []
        for index in batch:
            target = self.targets[index]
            writes, arrays = self._writer_inputs(target)
            # A: detached writer outputs.
            with torch.no_grad():
                out = run_native(engine, lambda: self.writer.run(target['messages'], writes, arrays,
                                 state_overrides=self._key_overrides(live_keys) or None, branch=recipe.coefficients != 'realized'),
                                 writer_entries)
            leaves = {'states': {}, 'hidden': {}, 'sign': {}, 'digit': {}, 'other': {}}
            for key, block in out.states.items():
                leaves['states'][key] = _leaf_block(block)
            for scope in out.integers:
                leaves['hidden'][scope] = out.decision_hidden[scope].detach().float().clone().requires_grad_(True)
                leaves['sign'][scope] = out.sign_logits[scope].detach().clone().requires_grad_(True)
                leaves['digit'][scope] = out.digit_logits[scope].detach().clone().requires_grad_(True)
                if scope in out.other_digit_logits:
                    leaves['other'][scope] = out.other_digit_logits[scope].detach().clone().requires_grad_(True)
            # B: executor objectives on the leaves.
            terms, coefficients = self._executor_terms(target, leaves, shares_all, len(batch))
            local = {k: v for k, v in terms.items() if v.requires_grad and recipe.weights.get(k, 0)}
            if local:
                (O.combine(local, recipe.weights) / len(batch)).backward()
            routes = {kind: math.sqrt(sum(float(leaf.grad.float().pow(2).sum()) for leaf in leaves[kind].values()
                                          if leaf.grad is not None)) for kind in ('hidden', 'sign', 'digit', 'other')}
            routes['states'] = math.sqrt(sum(float(v.grad.float().pow(2).sum()) for _, values in leaves['states'].values()
                                             for v in values.values() if v.grad is not None))
            # C: replay the writer with gradients and push the cotangents into its producers.
            def replay():
                live = self.writer.run(target['messages'], writes, arrays, state_overrides=self._key_overrides(live_keys) or None,
                                       branch=recipe.coefficients != 'realized')
                outputs, grads = [], []
                for key, block in live.states.items():
                    for (_, values) in [leaves['states'][key]]:
                        for name, leaf in values.items():
                            if leaf.grad is not None:
                                layer, field_name = name.split('.')[1:]
                                outputs.append(block.layers[int(layer)][field_name]); grads.append(leaf.grad)
                for scope in live.integers:
                    for kind, tensor in (('hidden', live.decision_hidden[scope].float()), ('sign', live.sign_logits[scope]),
                                         ('digit', live.digit_logits[scope]), ('other', live.other_digit_logits.get(scope))):
                        leaf = leaves[kind].get(scope)
                        if leaf is not None and leaf.grad is not None and tensor is not None:
                            outputs.append(tensor); grads.append(leaf.grad)
                local = sum((O.mixture_token_loss(self.grid, self.bank.number_format, live.sign_logits[a.scope], live.digit_logits[a.scope],
                                                  a.targets, a.available, zero_share=recipe.zero_share)[0] for a in arrays),
                            torch.zeros(()))
                weight = float(recipe.weights.get('mixture_tokens', 0.)) / len(batch)
                if weight:
                    outputs.append(local * weight); grads.append(torch.ones(()))
                if outputs:
                    torch.autograd.backward(outputs, grads)
                return float(local.detach())
            mixture = run_native(engine, replay, writer_entries, leaves={self.state['writer']: writer_values}, gradient_precision='fp32')
            records.append({'target': target['id'], **{k: float(v.detach()) for k, v in terms.items()}, 'mixture_tokens': mixture,
                            'cotangents': routes})
        return self._apply(batch, epoch, position, writer_values, live_keys, records)

    def _executor_terms(self, target, leaves, shares_all, batch_size):
        from ...model.native_adapters import run_native, adapter_state_input_receipt
        from ...data.native_graph_schema import parse_graph
        recipe, bank, engine = self.recipe, self.bank, self.engine
        cores_by_scope, terms, coefficients = {}, {}, {}
        rho_delta = {}
        for item in target['arrays']:
            scope, view = item['view']['scope'], item['view']
            slots = [bank.index(c) for c in view['componentIds']]
            shares = shares_all[slots]
            u, v = self.head(leaves['hidden'][scope])
            head_on = recipe.head_mode == 'head'
            if recipe.coefficients == 'realized':
                coefficient = executed(item['integers'], u, v, shares, step=bank.number_format.step, beta=recipe.beta, head=head_on)
            else:
                means, signed = [], []
                for slot in range(len(slots)):
                    negative = item['integers'][slot] < 0
                    mean, sign_diff, _ = expectation(self.grid, bank.number_format, leaves['sign'][scope][slot],
                                                     leaves['digit'][scope][slot], negative, leaves['other'][scope][slot])
                    means.append(mean); signed.append(sign_diff)
                available = torch.tensor(view['available'], dtype=torch.float32)
                coefficient = expected(torch.stack(means) * available, torch.stack(signed) * available, u, v, shares,
                                       step=bank.number_format.step, beta=recipe.beta, head=head_on)
            coefficients[scope] = coefficient
            factors = None
            if self.bank_leaves is not None:
                factors = {k: self.bank_leaves['factors_by_component'][k] for k in slots}
            cores_by_scope[scope] = bank.composed_cores(slots, coefficient, factors=factors)
            rho_delta[scope] = (refinement(u), modulation(v, shares, recipe.beta), item)
        placeholders = target['placeholders']
        state_blocks = {key: (block, None) for key, (block, _) in leaves['states'].items()}
        distill = []
        for row in target['rows']:
            example = parse_graph(row['graph'], tokenizer_fingerprint=engine.model_config['tokenizer_fingerprint'])
            scales = target['manifest'].get('scales', {})
            ancestry = next(b['ancestry'] for b in target['manifest']['bindings'] if b['id'] == row['binding'])
            scopes = [s for s in ancestry if s in placeholders and scales.get(s, 1) != 0]
            entries = [{'id': placeholders[s], 'scale': float(scales.get(s, 1))} for s in scopes]
            target_entries = [{'id': target['target_bundle']['adapters'][s], 'scale': float(scales.get(s, 1))}
                              for s in ancestry if s in target['target_bundle']['adapters'] and scales.get(s, 1) != 0]
            soft = row.get('soft') or {}
            target_blocks = {name: engine.store.get_native_state(target['target_bundle']['states'][key]).block for name, key in soft.items()}
            with torch.no_grad():
                target_logp, weights = run_native(engine, lambda: O.supervised_logprobs(engine.shared_model, example,
                                                  external_blocks=target_blocks or None, external_identity=_identities(engine, target_blocks)),
                                                  target_entries)
            live_blocks = {name: leaves['states'][key][0] for name, key in soft.items() if key in leaves['states']}
            live_blocks.update({name: target_blocks[name] for name in soft if name not in live_blocks})
            def execute():
                student, w = O.supervised_logprobs(engine.shared_model, example, external_blocks=live_blocks or None,
                                                   external_identity=_identities(engine, live_blocks))
                return O.distill_loss(target_logp, student, w)
            row_leaves = {placeholders[s]: bank.adapter_tensors(cores_by_scope[s]) for s in scopes if s in cores_by_scope}
            if self.bank_leaves is not None and recipe.train_basis:
                row_leaves[bank.basis_artifact] = self.bank_leaves['basis']
            weight = float(recipe.weights.get('distill', 0.)) / (batch_size * len(target['rows']))
            distill.append(_backward_inside(engine, execute, entries, row_leaves, weight))
        terms['distill'] = torch.stack(distill).mean()  # already backpropagated, detached for the record
        state_terms = {'instruction': [], 'skill': []}
        for write in target['writes']:
            if write.get('target_state') and write['key'] in leaves['states']:
                block = leaves['states'][write['key']][0]
                target_block = engine.store.get_native_state(write['target_state']).block
                loss, _ = O.state_loss(block, target_block, target['state_scales'])
                state_terms[write['kind']].append(loss)
        if state_terms['instruction']:
            terms['instruction_state'] = torch.stack(state_terms['instruction']).mean()
        if state_terms['skill']:
            terms['skill_state'] = torch.stack(state_terms['skill']).mean()
        core_terms, head_terms = [], []
        for scope, cores in cores_by_scope.items():
            factors = target.get('target_factors', {}).get(scope)
            if factors:
                core_terms.append(O.decoded_weight_loss(bank, cores, factors))
            rho, delta, item = rho_delta[scope]
            if recipe.head_mode == 'head' and 'rho' in item:
                head_terms.append(O.head_regression(rho, delta, torch.tensor(item['rho']), torch.tensor(item['delta']), item['view']['available']))
        if core_terms:
            terms['core'] = torch.stack(core_terms).mean()
        if head_terms:
            terms['head'] = torch.stack(head_terms).mean()
        return terms, coefficients

    def _apply(self, batch, epoch, position, writer_values, live_keys, records):
        from ...model.native_adapters import optimize_adapter
        from ...model.native_artifact import NativeTensorArtifact
        from ...model.native_state_derivation import optimize_state
        engine, store, recipe = self.engine, self.engine.store, self.recipe
        updated = {}
        grads = {k: v.grad for k, v in writer_values.items() if v.grad is not None}
        if grads:
            gradient = NativeTensorArtifact('adapter-gradient', engine.native_identity,
                                            {k: g.detach().cpu() for k, g in grads.items()}, self.state['writer'])
            result = optimize_adapter(engine, gradient, dict(recipe.writer_optimizer), self.state['writer_optimizer'], self.owner)
            self.state['writer'], self.state['writer_optimizer'] = result['adapter']['id'], result['optimizer_state_id']
            self.state['writer_history'].append(self.state['writer'])
            updated['writer'] = True
        if any(p.grad is not None for p in self.head.parameters()):
            self.head_optimizer.step()
            self.head_optimizer.zero_grad(set_to_none=True)
            updated['head'] = True
        key_moves = {}
        for state, (block, values) in live_keys.items():
            key_grads = {name: value.grad for name, value in values.items() if value.grad is not None}
            if not key_grads:
                continue
            gradient = store.put_native_tensor_artifact(NativeTensorArtifact('state-gradient', engine.native_identity,
                {name: g.detach().cpu() for name, g in key_grads.items()}, state,
                {'scope': 'natlang.lopt.component-key/1', 'step': self.state['step'] + 1}), self.owner)
            previous = self.state['key_optimizers'].get(state)
            result, _, _ = optimize_state(engine, gradient, dict(recipe.key_optimizer), previous, self.owner)
            key_moves[state] = (result['native_state_id'], result['optimizer_state_id'])
        for component, state in list(self.state['keys'].items()):
            if state in key_moves:
                new, optimizer_state = key_moves[state]
                self.state['keys'][component] = new
                self.state['key_optimizers'][new] = optimizer_state
        if key_moves:
            updated['keys'] = len(key_moves)
        if self.bank_leaves is not None and any(p.grad is not None for group in self.bank_optimizer.param_groups for p in group['params']):
            self.bank_optimizer.step()
            self.bank_optimizer.zero_grad(set_to_none=True)
            updated['bank'] = True
        self.state['step'] += 1
        self.state.update(epoch=epoch, position=position)
        for index in batch:
            target = self.targets[index]['id']
            self.state['exposure'][target] = self.state['exposure'].get(target, 0) + 1
        self.state['head'] = store.put_native_tensor_artifact(NativeTensorArtifact('parameter-set', engine.native_identity,
            {'weight': self.head.readout.weight.detach().cpu().clone(), 'bias': self.head.readout.bias.detach().cpu().clone()},
            None, {'scope': 'natlang.lopt.coefficient-head/1', 'config': self.head.config(), 'step': self.state['step']}), self.owner).id
        self.state['head_optimizer'] = _save_optimizer(engine, self.head_optimizer, 'head', self.owner)
        if self.bank_leaves is not None:
            self.state['bank_optimizer'] = _save_optimizer(engine, self.bank_optimizer, 'bank', self.owner)
            if self.state['step'] % recipe.publish_every == 0:
                self.publish_bank()
        record = {'step': self.state['step'], 'targets': records, 'updated': updated, 'head': self.head.identity()}
        self.state['losses'].append(record)
        return record

    def policy_step(self, records, *, clip_low=.2, clip_high=.28):
        """`joint-ratio-v1` on recorded proposals. Each record: ``context`` (assembled, with the executed query action
        rows as ``rows``), ``behavior`` (from `episode.propose`), ``old_executor_logprob`` and a context-only
        ``advantage``. The surrogate depends on theta only through ``log r = (log pi - log pi_old) + (log p_exec -
        log p_exec_old)``, so its gradient is ``g * grad log r`` with the scalar ``g = d loss / d log r``: the executor
        factor is regenerated from the same recorded tokens under current parameters (head, keys, writer, bank)."""
        from ...model.native_adapters import run_native
        from ...serve.native_grad import _leaf_block
        from ...data.native_graph_schema import parse_graph
        from .episode import _labels
        from .meta_rl import check_replay
        engine, recipe, store, bank = self.engine, self.recipe, self.engine.store, self.bank
        writer_values = {k: v.detach().clone().requires_grad_(True)
                         for k, v in store.get_native_tensor_artifact(self.state['writer']).artifact.tensors.items()}
        writer_entries = [{'id': self.state['writer'], 'scale': 1.}]
        live_keys = {}
        if recipe.train_keys and recipe.cue:
            for state in sorted(set(self.state['keys'].values()) - {None}):
                live_keys[state] = _leaf_block(store.get_native_state(state).block)
        shares_all = shares_from_norms(bank.group_norms())
        summary = []
        for record in records:
            context, behavior = record['context'], record['behavior']
            if behavior['bank'] != bank.snapshot:
                raise LearnedOptimizerError('the recorded proposal used another bank snapshot; collect fresh rollouts')
            for item in context['arrays']:
                check_replay({'view': behavior['arrays'][item['view']['scope']]['view']}, {'view': item['view']})
            writes = [StateWrite(w['key'], w['kind'], w['text'], w.get('extra_slots', 0)) for w in context['writes']]
            arrays = [ArrayWrite(item['view']['scope'], tuple(item['view']['available']),
                                 tuple(self.state['keys'][c] for c in item['view']['componentIds']) if recipe.cue else (None,) * len(item['view']['componentIds']),
                                 tuple(behavior['arrays'][item['view']['scope']]['integers'])) for item in context['arrays']]
            with torch.no_grad():
                out = run_native(engine, lambda: self.writer.run(context['messages'], writes, arrays,
                                 state_overrides=self._key_overrides(live_keys) or None, branch=False), writer_entries)
            for scope, tokens in out.tokens.items():
                if tokens != behavior['arrays'][scope]['tokens']:
                    raise LearnedOptimizerError('replayed optimizer tokens differ from the recorded spelling')
            old_pi = sum(float.fromhex(x) for a in behavior['arrays'].values() for x in a['logprobs'])
            new_pi = float(sum(out.logprobs[scope].sum() for scope in out.logprobs))
            state_leaves = {key: _leaf_block(block) for key, block in out.states.items()}
            hidden = {scope: out.decision_hidden[scope].detach().float().clone().requires_grad_(True) for scope in out.integers}
            def cores_for():
                cores = {}
                for item in context['arrays']:
                    scope = item['view']['scope']
                    slots = [bank.index(c) for c in item['view']['componentIds']]
                    u, v = self.head(hidden[scope])
                    coefficient = executed(behavior['arrays'][scope]['integers'], u, v, shares_all[slots],
                                           step=bank.number_format.step, beta=recipe.beta, head=recipe.head_mode == 'head')
                    factors = {k: self.bank_leaves['factors_by_component'][k] for k in slots} if self.bank_leaves is not None else None
                    cores[scope] = bank.composed_cores(slots, coefficient, factors=factors)
                return cores
            def run_rows(weight):
                total = 0.
                for row in context['rows']:
                    example = parse_graph(row['graph'], tokenizer_fingerprint=engine.model_config['tokenizer_fingerprint'])
                    scales = context['manifest'].get('scales', {})
                    ancestry = next(b['ancestry'] for b in context['manifest']['bindings'] if b['id'] == row['binding'])
                    scopes = [s for s in ancestry if s in context['placeholders'] and scales.get(s, 1) != 0]
                    entries = [{'id': context['placeholders'][s], 'scale': float(scales.get(s, 1))} for s in scopes]
                    soft = row.get('soft') or {}
                    blocks = {name: state_leaves[key][0] for name, key in soft.items() if key in state_leaves}
                    labels = _labels(example)
                    cores = cores_for()
                    leaves = {context['placeholders'][s]: bank.adapter_tensors(cores[s]) for s in scopes if s in cores}
                    def execute():
                        logp, w = O.supervised_logprobs(engine.shared_model, example, external_blocks=blocks or None,
                                                        external_identity=_identities(engine, blocks))
                        return logp.gather(-1, labels.to(logp.device)[:, None])[:, 0].sum()
                    total += float(_backward_inside(engine, execute, entries, leaves, weight))
                return total
            new_exec = run_rows(0.)
            log_ratio = (new_pi - old_pi) + (new_exec - record['old_executor_logprob'])
            ratio = math.exp(max(min(log_ratio, 20.), -20.))
            advantage = float(record['advantage'])
            clipped = (advantage > 0 and ratio > 1 + clip_high) or (advantage < 0 and ratio < 1 - clip_low)
            g = 0. if clipped else -advantage * ratio / len(records)
            if g:
                run_rows(g)
                def replay():
                    live = self.writer.run(context['messages'], writes, arrays, state_overrides=self._key_overrides(live_keys) or None,
                                           branch=False)
                    outputs, grads = [], []
                    for scope in live.logprobs:
                        outputs.append(live.logprobs[scope].sum()); grads.append(torch.tensor(g))
                        if hidden[scope].grad is not None:
                            outputs.append(live.decision_hidden[scope].float()); grads.append(hidden[scope].grad)
                    for key, (_, values) in state_leaves.items():
                        for name, leaf in values.items():
                            if leaf.grad is not None:
                                layer, field_name = name.split('.')[1:]
                                outputs.append(live.states[key].layers[int(layer)][field_name]); grads.append(leaf.grad)
                    torch.autograd.backward(outputs, grads)
                run_native(engine, replay, writer_entries, leaves={self.state['writer']: writer_values}, gradient_precision='fp32')
            summary.append({'logRatio': log_ratio, 'ratio': ratio, 'clipped': clipped, 'advantage': advantage,
                            'optimizerLogRatio': new_pi - old_pi, 'executorLogRatio': new_exec - record['old_executor_logprob']})
        result = self._apply([], self.state['epoch'], self.state['position'], writer_values, live_keys,
                             [{'policy': item} for item in summary])
        result['estimator'] = 'joint-ratio-v1'
        return result

    def publish_bank(self):
        """Publish the trained bank/basis as a new snapshot at an update boundary; labels become stale."""
        from dataclasses import replace
        from .bank import SharedBasis, component_id
        leaves = self.bank_leaves
        components = []
        for k, component in enumerate(self.bank.components):
            factors = {name: (P.detach().clone(), Q.detach().clone()) for name, (P, Q) in leaves['factors_by_component'][k].items()}
            components.append(replace(component, factors=factors, key=self.state['keys'].get(component.id, component.key),
                                      id=component_id(component.source, factors)))
        bank = self.bank
        if self.recipe.train_basis:
            from .bank import orthonormalize
            left, right = {}, {}
            for name in bank.basis.names:
                left[name], _ = orthonormalize(leaves['basis']['left.' + name].detach())
                right[name], _ = orthonormalize(leaves['basis']['right.' + name].detach())
            basis = SharedBasis(bank.basis.native_identity, bank.basis.rank, left, right,
                                {**bank.basis.method, 'refined_from': bank.basis_artifact})
            record = basis.to_artifact(self.engine, self.owner)
            bank = replace(bank, basis=basis, basis_artifact=record.id)
        published = bank.with_components(components, reason='outer-training publication')
        self.state['published'] = self.state.get('published', []) + [published.snapshot]
        self.state['labels_stale'] = True
        return published

    def run(self, steps=None):
        return [self.step() for _ in range((self.recipe.steps - self.state['step']) if steps is None else steps)]

    def receipt(self):
        return {'schema': 'natlang.lopt.training-receipt/1', 'step': self.state['step'], 'bank': self.state['bank'],
                'writer': self.state['writer'], 'head': self.state['head'], 'keys': self.state['keys'],
                'exposure': {'targets_seen': sum(self.state['exposure'].values()), 'distinct_targets': len(self.state['exposure'])},
                'recipe': self.recipe.as_json(), 'labels_stale': self.state['labels_stale'],
                'published': self.state.get('published', []), 'losses': self.state['losses']}


def _identities(engine, blocks):
    """Each imported state's own source identity (the same-base adapter-state exchange never relabels a writer)."""
    if not blocks:
        return None
    return {name: {key: getattr(block, key) for key in ('checkpoint_sha256', 'tokenizer_fingerprint', 'embedding_weight_sha256',
                                                         'architecture')} for name, block in blocks.items()}


def _backward_inside(engine, execute, entries, leaves, weight):
    """Run an executor objective inside the bundle's native execution and backward it THERE (activation-checkpoint
    replay and precision carriers must see the same effective weights). The composition graph built outside is
    retained for the other rows. Returns the detached loss."""
    from ...model.native_adapters import run_native
    def body():
        loss = execute()
        if loss.requires_grad and weight:
            (loss * weight).backward(retain_graph=True)
        return loss.detach()
    return run_native(engine, body, entries, leaves=leaves, gradient_precision='fp32')


def _bank_leaves(bank, recipe):
    factors_by_component = {}
    for k, component in enumerate(bank.components):
        factors_by_component[k] = {name: (P.detach().clone().float().requires_grad_(recipe.train_bank),
                                          Q.detach().clone().float().requires_grad_(recipe.train_bank))
                                   for name, (P, Q) in component.factors.items()}
    basis = {f'{side}.{name}': getattr(bank.basis, side)[name].detach().clone().requires_grad_(recipe.train_basis)
             for name in bank.basis.names for side in ('left', 'right')}
    return {'factors_by_component': factors_by_component, 'factors': factors_by_component and
            {k: [t for pair in v.values() for t in pair] for k, v in factors_by_component.items()}, 'basis': basis}


def _save_optimizer(engine, optimizer, scope, owner):
    from ...model.native_artifact import NativeTensorArtifact
    saved = optimizer.state_dict()
    tensors = {}
    for index, item in saved['state'].items():
        for key, value in item.items():
            tensors[f'{index}.{key}'] = torch.as_tensor(value).detach().cpu().clone().float() if key != 'step' else torch.as_tensor(value, dtype=torch.float64).reshape(1)
    if not tensors:
        return None
    return engine.store.put_native_tensor_artifact(NativeTensorArtifact('optimizer-state', engine.native_identity, tensors, None,
        {'scope': f'natlang.lopt.{scope}-optimizer/1', 'optimizer': 'adamw', 'param_groups': json.loads(json.dumps(saved['param_groups'], default=str))}),
        owner).id


def _load_optimizer(store, artifact_id, optimizer):
    record = store.get_native_tensor_artifact(artifact_id).artifact
    state = {}
    for key, value in record.tensors.items():
        index, name = key.split('.', 1)
        state.setdefault(int(index), {})[name] = value.reshape(()) if name == 'step' else value
    groups = optimizer.state_dict()['param_groups']
    return {'state': state, 'param_groups': groups}
