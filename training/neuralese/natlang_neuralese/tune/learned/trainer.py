"""The outer learner-training operation (plan "The outer learner-training operation", "One continuous training
operation", "Gradients for native states, component tokens and the shared bank").

``theta`` = the writer adapter on the executor's base, the mixture head (component rows and group-gain readout,
`mixture.MixtureHead`), the component keys (native states) and, when the recipe trains them, the bank's component
factors and basis. One logical update over a batch of complete target tuples runs three phases against ONE parameter
snapshot (see DECOMPOSITION.md, "Gradient path"):

A. writer forward under the writer adapter without gradients; states and decision hidden states are detached into
   leaves (base dropout drops each available component with the recipe's probability, recorded per step);
B. executor objectives with those leaves: the mixture head turns each decision leaf into component-token logits and
   gains; composed cores are live functions of the truncated distribution, the gains and the bank leaves.
   Distillation against the target bundle on its training-role histories, the state/decoded-weight auxiliaries, the
   KL to the fitted distribution and gain labels, and the batch load balance; backward yields the leaves' cotangents
   and the head's gradients;
C. writer forward again under the writer adapter with gradients and ``backward(outputs, cotangents)``. Exact
   cotangent replay: the plan's sanctioned split form.

Then every active group updates once: the writer adapter through `optimize_adapter` (immutable lineage, continued
moments), keys through `optimize_state`, the mixture head through `AdamWSR` (BF16 rows with stochastic rounding, FP32
gain readout) and the bank through AdamW with FP32 masters. Bank/basis changes publish a new snapshot only every
``publish_every`` updates and only when they changed; labels are then marked stale for refitting.

`policy_step` is the exact expected-score update (`expected-score-v1`) over scored finite token sets.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import torch

from . import LearnedOptimizerError
from . import mixture as M
from . import objectives as O
from .writer import ArrayWrite, StateWrite, Writer

STATE_SCHEMA = 'natlang.lopt.trainer-state/2'


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
    train_bank: bool = True
    train_basis: bool = True
    publish_every: int = 50
    weights: dict = field(default_factory=lambda: {'distill': 1., 'downstream': 0., 'instruction_state': .1,
                                                   'skill_state': .1, 'mixture': .2, 'gain': .05, 'core': .05,
                                                   'balance': .01})
    cue: bool = True
    dropout: float = .1            # base dropout of available components per training step (0 disables)
    row_init: float = 4.           # initial component-row logit range against the decision state

    @classmethod
    def from_json(cls, value):
        unknown = set(value) - set(cls.__dataclass_fields__)
        if unknown:
            raise LearnedOptimizerError(f'unknown outer recipe fields {sorted(unknown)}')
        recipe = cls(**value)
        if set(recipe.weights) - set(O.TERMS):
            raise LearnedOptimizerError('objective weights name unknown terms')
        if not 0 <= recipe.dropout < 1:
            raise LearnedOptimizerError('dropout lies in [0, 1)')
        return recipe

    def as_json(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def _hex(tensor):
    return [float(x).hex() for x in tensor.detach().double().reshape(-1).tolist()]


class OuterTrainer:
    """Resumable outer trainer over admitted joint target records (`assemble.training_target`, any tier)."""

    def __init__(self, engine, bank, recipe, targets, *, state=None, owner=None, extra_token=None):
        self.engine, self.bank, self.recipe, self.owner = engine, bank, recipe, owner
        self.targets = list(targets)  # empty when the trainer only applies the optimizer (`episode.propose`)
        for target in self.targets:
            if target['bank'] != bank.snapshot:
                raise LearnedOptimizerError(f'target {target["id"]} was labeled against another bank snapshot; refit it')
            if target.get('tier') is None:
                raise LearnedOptimizerError(f'{target["id"]} is a request context without labels, not a target')
        self._factors = {}
        width = engine.shared_model.port.embedding_weight.shape[1]
        self.head = M.MixtureHead(width, len(bank.group_labels), bank.ids, bank.mixture_format, row_init=recipe.row_init)
        self.writer = Writer(engine, cue=recipe.cue, extra_token=extra_token)
        self.state = state or self._initial_state()
        if self.state['schema'] != STATE_SCHEMA or self.state['bank'] != bank.snapshot:
            raise LearnedOptimizerError('trainer state belongs to another bank snapshot or schema')
        previous_recipe = self.state.get('recipe', {})
        self.bank_transition = {key: (bool(previous_recipe.get(key, False)), getattr(recipe, key))
                                for key in ('train_bank', 'train_basis')
                                if bool(previous_recipe.get(key, False)) != getattr(recipe, key)}
        self._restore()
        if self.bank_transition:
            previous_bank_optimizer = self.state.get('bank_optimizer')
            self.state['bank_optimizer'] = (_save_optimizer(self.engine, self.bank_optimizer, 'bank', self.owner)
                                            if self.bank_optimizer is not None else None)
            self.state.setdefault('recipe_transitions', []).append({
                'schema': 'natlang.lopt.bank-trainability-transition/1', 'step': self.state['step'],
                'changes': {key: {'from': old, 'to': new} for key, (old, new) in self.bank_transition.items()},
                'previous_bank_optimizer': previous_bank_optimizer,
                'optimizer_history': 'preserved-for-unchanged-parameters;new-parameters-start-without-moments'})
            self.state['recipe'] = self.recipe.as_json()
        if self.state.get('rng') is not None:
            _restore_rng(self.state['rng'])

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
                'head': None, 'head_components': list(self.bank.ids), 'head_optimizer': None,
                'keys': {c.id: c.key for c in self.bank.components},
                'key_optimizers': {}, 'bank_tensors': None, 'bank_optimizer': None, 'bank_dirty': False,
                'labels_stale': False, 'exposure': {}, 'losses': [], 'recipe': self.recipe.as_json(), 'rng': _capture_rng()}

    def _restore(self):
        store = self.engine.store
        if self.state.get('head_components', self.bank.ids) != self.bank.ids:
            raise LearnedOptimizerError('the saved mixture head belongs to another component list; adopt the bank first')
        if self.state['head']:
            self.head.load(store.get_native_tensor_artifact(self.state['head']).artifact.tensors)
        self.head_optimizer = _head_optimizer(self.head, self.recipe)
        if self.state['head_optimizer']:
            self.head_optimizer.load_state_dict(_load_optimizer(store, self.state['head_optimizer'], self.head_optimizer))
        self.bank_leaves = None
        self.bank_optimizer = None
        if self.recipe.train_bank or self.recipe.train_basis or self.state.get('bank_tensors'):
            self.bank_leaves = _bank_leaves(self.bank, self.recipe)
            if self.state.get('bank_tensors'):
                _restore_bank_leaves(self.engine, self.bank, self.bank_leaves, self.state['bank_tensors'], self.state['step'])
            elif self.state.get('bank_optimizer'):
                raise LearnedOptimizerError('bank optimizer history has no saved parameter snapshot; exact resume is unavailable')
            params = [v for v in self.bank_leaves['factors'].values() for v in v] + list(self.bank_leaves['basis'].values())
            trainable = [p for p in params if p.requires_grad]
            if trainable:
                self.bank_optimizer = torch.optim.AdamW(trainable, lr=self.recipe.bank_optimizer['learning_rate'],
                                                        weight_decay=self.recipe.bank_optimizer.get('weight_decay', 0.))
            if self.state['bank_optimizer'] and self.bank_optimizer is not None:
                saved = _load_optimizer(store, self.state['bank_optimizer'], self.bank_optimizer)
                if self.bank_transition:
                    old = self.state.get('recipe', {})
                    factor_count = sum(len(v) for v in self.bank_leaves['factors'].values())
                    basis_count = len(self.bank_leaves['basis'])
                    old_names = ([('factor', i) for i in range(factor_count)] if old.get('train_bank', False) else []) + \
                                ([('basis', i) for i in range(basis_count)] if old.get('train_basis', False) else [])
                    new_names = ([('factor', i) for i in range(factor_count)] if self.recipe.train_bank else []) + \
                                ([('basis', i) for i in range(basis_count)] if self.recipe.train_basis else [])
                    if any(index >= len(old_names) for index in saved['state']):
                        raise LearnedOptimizerError('saved bank optimizer parameter inventory differs from its recorded recipe')
                    positions = {name: index for index, name in enumerate(new_names)}
                    saved['state'] = {positions[old_names[index]]: value for index, value in saved['state'].items()
                                      if old_names[index] in positions}
                self.bank_optimizer.load_state_dict(saved)

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

    def _views(self, items, dropout_seed=None):
        """Each array's view with base dropout applied to its availability (``dropout_seed`` None: none)."""
        views = {}
        for number, item in enumerate(items):
            view = dict(item['view'])
            if dropout_seed is not None and self.recipe.dropout:
                generator = torch.Generator().manual_seed(dropout_seed * 1009 + number)
                retained = (torch.rand(len(view['available']), generator=generator) >= self.recipe.dropout).tolist()
                view['available'] = [bool(a and r) for a, r in zip(view['available'], retained)]
            views[view['scope']] = view
        return views

    def _writer_inputs(self, target, views):
        writes = [StateWrite(w['key'], w['kind'], w['text'], w.get('extra_slots', 0)) for w in target['writes']]
        arrays = []
        for scope, view in views.items():
            keys = tuple(self.state['keys'][c] for c in view['componentIds']) if self.recipe.cue else (None,) * len(view['componentIds'])
            arrays.append(ArrayWrite(scope, tuple(view['available']), keys))
        return writes, arrays

    def _target_factors(self, adapter):
        """Exact effective update factors of a target adapter (any basis), loaded once."""
        if adapter not in self._factors:
            from ..trainer import adapter_factors
            self._factors[adapter] = {name: value for name, value in adapter_factors(self.engine, adapter).items()
                                      if name in self.bank.basis.left}
        return self._factors[adapter]

    def _key_overrides(self, live_keys):
        return {state: block for state, (block, _) in live_keys.items()}

    def _live(self):
        store = self.engine.store
        writer_values = {k: v.detach().clone().requires_grad_(True)
                         for k, v in store.get_native_tensor_artifact(self.state['writer']).artifact.tensors.items()}
        live_keys = {}
        if self.recipe.train_keys and self.recipe.cue:
            from ...serve.native_grad import _leaf_block
            for state in sorted(set(self.state['keys'].values()) - {None}):
                live_keys[state] = _leaf_block(store.get_native_state(state).block)
        return writer_values, [{'id': self.state['writer'], 'scale': 1.}], live_keys

    def _decide(self, hidden, view):
        logits = self.head.logits(hidden, view['componentIds'], view['available'])
        probs, keep = M.distribution(logits, self.bank.mixture_format.top_p)
        return logits, probs, keep, self.head.gains(hidden)

    # -- one logical update -------------------------------------------------------------------------------------
    def step(self):
        from ...model.native_adapters import run_native
        from ...serve.native_grad import _leaf_block
        if self.state.get('labels_stale'):
            raise LearnedOptimizerError('published bank needs freshly refitted labels before another update')
        engine, recipe = self.engine, self.recipe
        batch, epoch, position = self._next_batch()
        writer_values, writer_entries, live_keys = self._live()
        items, usage = [], []
        for number, index in enumerate(batch):
            target = self.targets[index]
            views = self._views(target['arrays'], self.state['step'] * 7919 + number + 1)
            writes, arrays = self._writer_inputs(target, views)
            # A: detached writer outputs.
            with torch.no_grad():
                out = run_native(engine, lambda: self.writer.run(target['messages'], writes, arrays,
                                 state_overrides=self._key_overrides(live_keys) or None), writer_entries)
                if recipe.cue:
                    for scope, view in views.items():
                        self.head.initialize_rows([c for c, a in zip(view['componentIds'], view['available']) if a],
                                                  out.key_hidden[scope], out.decision_hidden[scope])
            leaves = {'states': {key: _leaf_block(block) for key, block in out.states.items()},
                      'hidden': {scope: out.decision_hidden[scope].detach().float().clone().requires_grad_(True)
                                 for scope in out.decision_hidden}}
            # B: executor objectives on the leaves.
            terms, used = self._executor_terms(target, views, leaves, len(batch))
            usage.extend(used)
            local = {k: v for k, v in terms.items() if v.requires_grad and recipe.weights.get(k, 0)}
            if local:
                (O.combine(local, recipe.weights) / len(batch)).backward(retain_graph=True)
            items.append((target, writes, arrays, leaves, terms))
        balance = M.balance(usage) if recipe.weights.get('balance', 0) else None
        if balance is not None:
            (float(recipe.weights['balance']) * balance).backward()
        records = []
        for target, writes, arrays, leaves, terms in items:
            routes = {'hidden': math.sqrt(sum(float(leaf.grad.float().pow(2).sum()) for leaf in leaves['hidden'].values()
                                              if leaf.grad is not None)),
                      'states': math.sqrt(sum(float(v.grad.float().pow(2).sum()) for _, values in leaves['states'].values()
                                              for v in values.values() if v.grad is not None))}
            # C: replay the writer with gradients and push the cotangents into its producers.
            def replay(target=target, writes=writes, arrays=arrays, leaves=leaves):
                live = self.writer.run(target['messages'], writes, arrays, state_overrides=self._key_overrides(live_keys) or None)
                outputs, grads = [], []
                for key, block in live.states.items():
                    for name, leaf in leaves['states'][key][1].items():
                        if leaf.grad is not None:
                            layer, field_name = name.split('.')[1:]
                            outputs.append(block.layers[int(layer)][field_name]); grads.append(leaf.grad)
                for scope, hidden in live.decision_hidden.items():
                    leaf = leaves['hidden'][scope]
                    if leaf.grad is not None:
                        outputs.append(hidden.float()); grads.append(leaf.grad)
                if outputs:
                    torch.autograd.backward(outputs, grads)
            run_native(engine, replay, writer_entries, leaves={self.state['writer']: writer_values}, gradient_precision='fp32')
            records.append({'target': target['id'], **{k: float(v.detach()) for k, v in terms.items()}, 'cotangents': routes})
        if balance is not None:
            records.append({'balance': float(balance.detach())})
        return self._apply(batch, epoch, position, writer_values, live_keys, records)

    def _executor_terms(self, target, views, leaves, batch_size):
        from ...model.native_adapters import run_native
        from ...data.native_graph_schema import parse_graph
        recipe, bank, engine, fmt = self.recipe, self.bank, self.engine, self.bank.mixture_format
        cores_by_scope, terms, used = {}, {}, []
        mixture_terms, gain_terms = [], []
        for item in target['arrays']:
            scope = item['view']['scope']
            view = views[scope]
            slots = [bank.index(c) for c in view['componentIds']]
            logits, probs, _, gains = self._decide(leaves['hidden'][scope], view)
            used.append((['null', *view['componentIds']], probs))
            factors = {k: self.bank_leaves['factors_by_component'][k] for k in slots} if self.bank_leaves is not None else None
            cores_by_scope[scope] = bank.composed_cores(slots, M.coefficients(probs, gains), factors=factors,
                                                        offset=bank.incoming(engine, target, scope))
            if 'distribution' in item:
                label = torch.tensor(item['distribution']) * torch.tensor([True, *view['available']])
                if float(label.sum()) > 0:  # dropout can remove the whole fitted support: no mixture label then
                    mixture_terms.append(M.mixture_kl(label / label.sum(), logits))
                gain_terms.append(M.gain_loss(item['gain'], gains, fmt))
        placeholders = target['placeholders']
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
        core_terms = [O.decoded_weight_loss(bank, cores, self._target_factors(adapter))
                      for scope, cores in cores_by_scope.items()
                      for adapter in [(target.get('target_factors') or {}).get(scope)] if adapter]
        if core_terms:
            terms['core'] = torch.stack(core_terms).mean()
        if mixture_terms:
            terms['mixture'] = torch.stack(mixture_terms).mean()
        if gain_terms:
            terms['gain'] = torch.stack(gain_terms).mean()
        return terms, used

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
        if self.bank_optimizer is not None and any(p.grad is not None for group in self.bank_optimizer.param_groups for p in group['params']):
            self.bank_optimizer.step()
            self.bank_optimizer.zero_grad(set_to_none=True)
            self.state['bank_dirty'] = True
            updated['bank'] = True
        self.state['step'] += 1
        self.state.update(epoch=epoch, position=position)
        for index in batch:
            target = self.targets[index]['id']
            self.state['exposure'][target] = self.state['exposure'].get(target, 0) + 1
        self.state['head'] = store.put_native_tensor_artifact(NativeTensorArtifact('parameter-set', engine.native_identity,
            self.head.tensors(), None, {'scope': 'natlang.lopt.mixture-head/1', 'config': self.head.config(),
                                        'step': self.state['step']}), self.owner).id
        self.state['head_optimizer'] = _save_optimizer(engine, self.head_optimizer, 'head', self.owner)
        if self.bank_leaves is not None:
            self.state['bank_tensors'] = _save_bank_leaves(engine, self.bank, self.bank_leaves, self.state['step'], self.owner)
            self.state['bank_optimizer'] = (_save_optimizer(engine, self.bank_optimizer, 'bank', self.owner)
                                            if self.bank_optimizer is not None else None)
            if self.state['step'] % recipe.publish_every == 0 and self.state.get('bank_dirty'):
                self.publish_bank()
        record = {'step': self.state['step'], 'targets': records, 'updated': updated, 'head': self.head.identity()}
        self.state['losses'].append(record)
        self.state['rng'] = _capture_rng()
        return record

    def policy_step(self, records):
        """`expected-score-v1` on scored candidates. Each record: ``context`` (assembled), ``behavior`` (the mixture
        proposal of `episode.propose`), ``candidate`` ``{scope, token}`` (a kept token executed alone, the other scopes
        at their mixture) and ``reward`` (its paired total). Per context and scope the update is the exact gradient of
        ``sum_k p_k R_k`` over the scored tokens, with ``p`` recomputed under current parameters and renormalized over
        the scored set (`mixture.expected_score_loss`); records without a candidate are reported only."""
        from ...model.native_adapters import run_native
        from .meta_rl import check_replay
        if self.state.get('labels_stale'):
            raise LearnedOptimizerError('published bank needs freshly refitted labels before another update')
        groups = {}
        for record in records:
            if record.get('candidate') is None:
                continue
            context, behavior = record['context'], record['behavior']
            if behavior['bank'] != self.bank.snapshot:
                raise LearnedOptimizerError('the recorded proposal used another bank snapshot; collect fresh rollouts')
            group = groups.setdefault(context['id'], {'context': context, 'behavior': behavior, 'scores': {}})
            if group['behavior']['arrays'] != behavior['arrays']:
                raise LearnedOptimizerError('candidates of one context come from different mixture proposals')
            candidate = record['candidate']
            group['scores'].setdefault(candidate['scope'], {})[json.dumps(candidate['token'])] = float(record['reward']['total'])
        writer_values, writer_entries, live_keys = self._live()
        summary = []
        for group in groups.values():
            context, behavior = group['context'], group['behavior']
            views = {}
            for item in context['arrays']:
                check_replay({'view': behavior['arrays'][item['view']['scope']]['view']}, {'view': item['view']})
                views[item['view']['scope']] = item['view']
            writes, arrays = self._writer_inputs(context, views)
            def update(context=context, behavior=behavior, views=views, writes=writes, arrays=arrays, group=group):
                live = self.writer.run(context['messages'], writes, arrays, state_overrides=self._key_overrides(live_keys) or None)
                losses, report = [], {}
                for scope, scores in group['scores'].items():
                    view = views[scope]
                    logits = self.head.logits(live.decision_hidden[scope], view['componentIds'], view['available'])
                    tokens = [M.NULL, *range(len(view['componentIds']))]
                    scored = [json.dumps(token) in scores for token in tokens]
                    probs, _ = M.distribution(logits, 1., keep=scored)
                    rewards = [scores.get(json.dumps(token), 0.) for token in tokens]
                    losses.append(M.expected_score_loss(probs, rewards))
                    report[scope] = {'scored': sum(scored), 'kept': sum(behavior['arrays'][scope]['kept']),
                                     'expectedScore': float((probs.detach() * torch.tensor(rewards)).sum())}
                loss = torch.stack(losses).sum() / len(groups)
                loss.backward()
                return report
            report = run_native(self.engine, update, writer_entries, leaves={self.state['writer']: writer_values},
                                gradient_precision='fp32')
            summary.append({'context': context['id'], 'scopes': report})
        result = self._apply([], self.state['epoch'], self.state['position'], writer_values, live_keys,
                             [{'policy': item} for item in summary])
        result['estimator'] = 'expected-score-v1'
        return result

    def publish_bank(self):
        """Publish the trained bank/basis as a new snapshot at an update boundary; labels become stale."""
        from dataclasses import replace
        from .bank import SharedBasis, component_id
        leaves = self.bank_leaves
        bank = self.bank
        basis = None
        transforms = {}
        if self.recipe.train_basis:
            from .bank import orthonormalize
            left, right = {}, {}
            for name in bank.basis.names:
                left[name], r_left = orthonormalize(leaves['basis']['left.' + name].detach())
                right[name], r_right = orthonormalize(leaves['basis']['right.' + name].detach())
                transforms[name] = (r_left, r_right)
            basis = SharedBasis(bank.basis.native_identity, bank.basis.rank, left, right,
                                {**bank.basis.method, 'refined_from': bank.basis_artifact})
        components = []
        for k, component in enumerate(bank.components):
            factors = {}
            for name, (P, Q) in leaves['factors_by_component'][k].items():
                P, Q = P.detach().clone(), Q.detach().clone()
                if name in transforms:
                    # U P Q^T V^T = Q_u (R_u P) (R_v Q)^T Q_v^T.
                    # Publication changes coordinates, never the learned update.
                    r_left, r_right = transforms[name]
                    P, Q = r_left @ P, r_right @ Q
                factors[name] = (P, Q)
            components.append(replace(component, factors=factors, key=self.state['keys'].get(component.id, component.key),
                                      id=component_id(component.source, factors)))
        published = bank.with_components(components, reason='outer-training publication')
        if basis is not None:
            record = basis.to_artifact(self.engine, self.owner)
            published = replace(published, basis=basis, basis_artifact=record.id)
        from .bank import save_bank, load_bank
        manifest = save_bank(self.engine, published, self.owner)
        if load_bank(self.engine, manifest).snapshot != published.snapshot:
            raise LearnedOptimizerError('published bank failed its manifest readback')
        self.state['published_banks'] = self.state.get('published_banks', []) + [manifest]
        self.state['published'] = self.state.get('published', []) + [published.snapshot]
        self.state['labels_stale'] = True
        self.state['bank_dirty'] = False
        return published

    def run(self, steps=None):
        return [self.step() for _ in range((self.recipe.steps - self.state['step']) if steps is None else steps)]

    def receipt(self):
        return {'schema': 'natlang.lopt.training-receipt/1', 'step': self.state['step'], 'bank': self.state['bank'],
                'writer': self.state['writer'], 'head': self.state['head'], 'keys': self.state['keys'],
                'exposure': {'targets_seen': sum(self.state['exposure'].values()), 'distinct_targets': len(self.state['exposure'])},
                'recipe': self.recipe.as_json(), 'labels_stale': self.state['labels_stale'],
                'published': self.state.get('published', []), 'published_banks': self.state.get('published_banks', []),
                'losses': self.state['losses']}


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


def _bank_tensor_map(leaves):
    tensors = {'basis.' + key: value for key, value in leaves['basis'].items()}
    for index, factors in leaves['factors_by_component'].items():
        for name, (p, q) in factors.items():
            tensors[f'component.{index}.P.{name}'] = p
            tensors[f'component.{index}.Q.{name}'] = q
    return tensors


def _save_bank_leaves(engine, bank, leaves, step, owner):
    from ...model.native_artifact import NativeTensorArtifact
    tensors = {name: value.detach().cpu().contiguous().clone() for name, value in _bank_tensor_map(leaves).items()}
    return engine.store.put_native_tensor_artifact(NativeTensorArtifact(
        'parameter-set', engine.native_identity, tensors, bank.basis_artifact,
        {'scope': 'natlang.lopt.bank-parameters/1', 'bank': bank.snapshot, 'step': step}), owner).id


def _restore_bank_leaves(engine, bank, leaves, artifact_id, step):
    record = engine.store.get_native_tensor_artifact(artifact_id)
    if record is None:
        raise LearnedOptimizerError('saved bank parameter snapshot is unavailable')
    artifact = record.artifact
    if (artifact.kind != 'parameter-set' or artifact.native_identity != engine.native_identity
            or artifact.source_artifact_id != bank.basis_artifact
            or artifact.provenance != {'scope': 'natlang.lopt.bank-parameters/1', 'bank': bank.snapshot, 'step': step}):
        raise LearnedOptimizerError('saved bank parameter snapshot has a different identity, basis or step')
    expected = _bank_tensor_map(leaves)
    if set(expected) != set(artifact.tensors):
        raise LearnedOptimizerError('saved bank parameter inventory differs from the bank')
    with torch.no_grad():
        for name, value in expected.items():
            saved = artifact.tensors[name]
            if saved.shape != value.shape or saved.dtype != value.dtype:
                raise LearnedOptimizerError('saved bank parameter shape or dtype differs from the bank')
            value.copy_(saved.to(value.device))


ADOPTION_SCHEMA = 'natlang.lopt.bank-adoption/1'


def adopt_published_bank(engine, state, recipe, owner=None):
    """The explicit transition that continues outer training on its latest published bank snapshot.

    Publication leaves the trainer stale (`labels_stale`): its targets were labeled against the previous snapshot. After
    the operation refits the labels against the published bank, training continues from this transition: the state
    binds the published snapshot, every component's live key moves to its published component ID, and the bank
    parameters restart from the published tensors (the same composed updates; with ``train_basis`` the basis was
    re-orthonormalized and factors re-expressed). When publication kept the coordinates (no basis training) the bank
    optimizer moments continue exactly; otherwise the bank optimizer starts an honestly new history (saved moments are
    never reinterpreted through the QR change of coordinates) and the old history is retained as transition provenance.
    Component rows of the mixture head follow their components by position like the keys. Writer, head and key
    optimizer histories, exposure, losses and RNG are unchanged. Returns ``(state, bank)``."""
    from .bank import load_bank
    if not state.get('labels_stale') or not state.get('published_banks'):
        raise LearnedOptimizerError('no published bank awaits adoption')
    published = load_bank(engine, state['published_banks'][-1])
    old_snapshot = state['bank']
    if published.parent != old_snapshot:
        raise LearnedOptimizerError('the latest published bank does not descend from the trainer\'s bank snapshot')
    # Component order is preserved by publication; keys follow their components by position.
    old_ids = list(state['keys'])
    if len(old_ids) != len(published.components):
        raise LearnedOptimizerError('published bank changed the number of components; adopt through bank growth instead')
    keys = {component.id: component.key for component in published.components}
    # Without basis training, publication copies the live factors unchanged (`publish_bank`): same coordinates.
    same_coordinates = not recipe.train_basis
    transition = {'schema': ADOPTION_SCHEMA, 'step': state['step'], 'from': old_snapshot, 'to': published.snapshot,
                  'basis_changed': not same_coordinates, 'old_bank_tensors': state.get('bank_tensors'),
                  'old_bank_optimizer': state.get('bank_optimizer'),
                  'bank_optimizer': 'continued' if same_coordinates else 'new-history-after-coordinate-change'}
    adopted = {**state, 'bank': published.snapshot, 'keys': keys, 'labels_stale': False,
               'head_components': [component.id for component in published.components],
               'adoptions': [*state.get('adoptions', []), transition]}
    if recipe.train_bank or recipe.train_basis:
        leaves = _bank_leaves(published, recipe)
        adopted['bank_tensors'] = _save_bank_leaves(engine, published, leaves, state['step'], owner)
        if not same_coordinates:
            adopted['bank_optimizer'] = None
    return adopted, published


def _capture_rng():
    return {'cpu': bytes(torch.get_rng_state().tolist()).hex(),
            'cuda': [bytes(value.tolist()).hex() for value in torch.cuda.get_rng_state_all()]
                    if torch.cuda.is_initialized() else None}


def _restore_rng(state):
    torch.set_rng_state(torch.tensor(list(bytes.fromhex(state['cpu'])), dtype=torch.uint8))
    if state.get('cuda') is not None:
        if not torch.cuda.is_available() or torch.cuda.device_count() != len(state['cuda']):
            raise LearnedOptimizerError('saved CUDA RNG requires the same available device count')
        torch.cuda.set_rng_state_all([torch.tensor(list(bytes.fromhex(value)), dtype=torch.uint8) for value in state['cuda']])


def _head_optimizer(head, recipe):
    from ...train.optim import AdamWSR
    return AdamWSR(head.parameters(), lr=recipe.head_optimizer['learning_rate'],
                   weight_decay=recipe.head_optimizer.get('weight_decay', 0.))


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
        {'scope': f'natlang.lopt.{scope}-optimizer/1', 'optimizer': type(optimizer).__name__.lower(), 'param_groups': json.loads(json.dumps(saved['param_groups'], default=str))}),
        owner).id


def _load_optimizer(store, artifact_id, optimizer):
    record = store.get_native_tensor_artifact(artifact_id).artifact
    state = {}
    for key, value in record.tensors.items():
        index, name = key.split('.', 1)
        state.setdefault(int(index), {})[name] = value.reshape(()) if name == 'step' else value
    groups = optimizer.state_dict()['param_groups']
    return {'state': state, 'param_groups': groups}
