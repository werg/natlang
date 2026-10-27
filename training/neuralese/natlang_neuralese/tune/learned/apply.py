"""Applying the trained learned optimizer: the adaptation episode's services and the crisp episode.

`python -m natlang_neuralese.tune.learned.apply --plan plan.json --out DIR` has the same plan and `result.json`
contract as `tune.build`, so `natlang tune --learned` validates, activates and rolls back its release through the
ordinary pipeline. The plan is a `natlang.tune-plan/v1` plus ``learned``: ``{"bank": bank.json, "state": optimizer
state.json, "store": operation artifact store, "scopes": {level: view size}, "steps": K, "patience": P, "seed": S}``.

Services (one implementation, used by the crisp episode here and by the Natlang `adapt` program through
``python -m natlang_neuralese.tune.learned service``): ``evidence`` (the permitted view), ``propose`` (a complete
learned bundle), ``evaluate`` (exact support loss of a bundle), ``refine`` (local in-bank refinement from a bundle),
``retrieve`` (gradient-free mixing over the bank view, the retrieval baseline). The crisp operator policy (the
pluggable default) proposes, deterministically first and then by sampling, keeps the better bundle on the support
evidence and stops after ``patience`` steps without improvement or after ``K`` steps (structural measure).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from . import LearnedOptimizerError
from ..build import import_artifacts, import_states, load_plan, persist_artifacts, _write


class Context:
    """One program's prepared adaptation context in an output directory (resumable: soft initialization and the
    assembled context are written once)."""

    def __init__(self, plan_path, out):
        self.plan, self.manifest, self.turns, _ = load_plan(plan_path)
        self.learned = self.plan.get('learned')
        if not isinstance(self.learned, dict):
            raise LearnedOptimizerError('the plan carries no learned optimizer settings')
        from ...serve import load_native_engine
        from ...serve.store import TensorStore
        config = self.plan.get('engine') or {'checkpoint': self.plan['student']['checkpoint'], **self.plan['student'].get('engine', {})}
        self.config = config
        self.engine = load_native_engine(config['checkpoint'], device=config.get('device', 'cpu'),
                                         dtype=getattr(torch, config.get('dtype', 'float32')))
        self.engine.store.close()
        self.engine.store = TensorStore(self.learned['store'])
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        if (self.out / 'artifacts').is_dir():
            import_artifacts(self.engine, self.out / 'artifacts')
        if (self.out / 'states').is_dir():
            import_states(self.engine, self.out / 'states')
        self._prepare()

    def close(self):
        self.engine.store.close()

    def _prepare(self):
        from .assemble import training_target
        from .bank import load_bank
        from .evidence import build_view, demonstration_facts
        from .fit import SupportSet, placeholder_adapters
        from .trainer import OuterRecipe, OuterTrainer
        from .views import select_view
        from .__main__ import _grid
        from ..rows import lower_rows
        engine, learned = self.engine, self.learned
        self.bank = load_bank(engine, json.loads(Path(learned['bank']).read_text())['bank'])
        state = json.loads(Path(learned['state']).read_text())
        self.trainer = OuterTrainer(engine, self.bank, _grid(engine, {**self.config, **learned}),
                                    OuterRecipe.from_json(state['recipe']), [], state=state)
        init = self.out / 'soft-init.json'
        if init.exists():
            self.soft = json.loads(init.read_text())
        elif self.plan.get('softStates'):
            from ..soft import initialize_soft_states
            self.soft = initialize_soft_states(engine, self.turns)
            persist_artifacts(engine, [record['state'] for record in self.soft.values()], self.out / 'artifacts', self.out / 'states')
            _write(init, self.soft)
        else:
            self.soft = {}
        self.rows = lower_rows(engine, self.turns, prompt_weight=0., soft={k: r['state'] for k, r in self.soft.items()})
        self.support = SupportSet(engine, self.manifest, self.rows)
        levels = {node['id']: node['level'] for node in self.manifest['tree']}
        self.scopes = sorted(node for node, level in levels.items() if level in learned['scopes'])
        self.placeholders = placeholder_adapters(engine, self.bank, self.scopes)
        sizes = {scope: learned['scopes'][levels[scope]] for scope in self.scopes}
        self.views = {scope: select_view(self.bank, scope, levels[scope], sizes[scope], policy='full-bank-v1')
                      if len(self.bank.components) <= sizes[scope] else
                      select_view(self.bank, scope, levels[scope], sizes[scope], policy='key-similarity-v1',
                                  key_scores=[0.] * len(self.bank.components)) for scope in self.scopes}
        items = []
        from .. import TuneMathError
        from ..rows import TURN_SCHEMA
        for turn in self.turns:
            if turn.get('schema') != TURN_SCHEMA:
                continue  # already-lowered graphs carry no demonstration record
            try:
                items.append(demonstration_facts(turn, 'support'))
            except (LearnedOptimizerError, TuneMathError):
                continue  # unverified turns are context, never demonstrations
        self.evidence = build_view(self.manifest['program']['id'], 'incoming', items)
        contract = json.dumps({'program': self.manifest['program'], 'bindings': [b['id'] for b in self.manifest['bindings']]})
        texts = {key: engine.store.get_native_state(record['state']).block.text for key, record in self.soft.items()}
        self.context = training_target(engine, self.bank, manifest=self.manifest, rows=self.rows, contract=contract,
                                       view=self.evidence, soft_texts=texts,
                                       incoming_states={k: r['state'] for k, r in self.soft.items()}, bank_views=self.views,
                                       target_integers={s: [0] * len(v.slots) for s, v in self.views.items()}, target_adapters={},
                                       target_states={}, placeholders=self.placeholders)
        self.context['program'] = self.manifest['program']

    # -- services ------------------------------------------------------------------------------------------------
    def _states_of(self, bundle):
        out = {}
        for key, record in self.soft.items():
            binding, kind, target = json.loads(key)
            state = record['state'] if bundle is None else (
                bundle['instructions'].get(binding, record['state']) if kind == 'instructions'
                else bundle['skills'].get(f'{binding}#{target}', record['state']))
            out[key] = (self.engine.store.get_native_state(state).block, state)
        return out

    def evaluate(self, bundle):
        adapters = dict(self.placeholders) if bundle is None else {s: bundle['adapters'][s] for s in self.scopes}
        with torch.no_grad():
            return self.support.accumulate(adapters, lambda: {}, self._states_of(bundle), backward=False)

    def propose(self, parent, *, sample, seed, digests=(), focus=None):
        from .episode import propose
        from .request import request_messages
        context = dict(self.context, parent=parent)
        if digests or focus is not None:
            # The Natlang analysis' native digests and focus enter the request; everything else is unchanged.
            content = context['messages'][0]['content'] + [{'type': 'text', 'text': '\n'}]
            if focus:
                content.append({'type': 'text', 'text': 'Revise: ' + ', '.join(focus) + '\n'})
            for binding, state in digests:
                content += [{'type': 'text', 'text': f'Missing behavior at {binding}: '}, {'type': 'native_state', 'id': state},
                            {'type': 'text', 'text': '\n'}]
            context['messages'] = [{'role': 'user', 'content': content}]
        return propose(self.trainer, context, sample=sample, seed=seed)

    def refine(self, parent, *, steps=10):
        """Local in-bank refinement starting from ``parent``'s mixtures and states (a gradient operator inside the
        episode); returns a complete bundle composed from the refined labels."""
        from .bundle import make_bundle
        from .fit import ScopeArray, fit_in_bank
        from .views import mixture_encoding
        from .head import receipt as head_receipt
        arrays = [ScopeArray(scope, self.placeholders[scope], list(self.views[scope].slots), list(self.views[scope].available))
                  for scope in self.scopes]
        init = None
        if parent is not None:
            init = {scope: torch.tensor([int(w) * self.bank.number_format.step for w in parent['loraMixtures'][scope]['weights']])
                    for scope in self.scopes}
        soft = {key: state for key, (_, state) in self._states_of(parent).items()}
        result = fit_in_bank(self.engine, self.bank, self.support, arrays, soft_states=soft, steps=steps,
                             refine_steps=max(1, steps // 2), init=init)
        adapters, mixtures = {}, {}
        for scope in self.scopes:
            fitted = result['arrays'][scope]
            adapters[scope] = self.bank.compose_adapter(self.engine, self.bank.composed_cores(arrays[self.scopes.index(scope)].slots,
                                                                                              fitted['executed'])).id
            mixtures[scope] = mixture_encoding(self.views[scope], fitted['integers'],
                                               head_receipt(None, fitted['integers'], fitted['u'], fitted['v'], fitted['executed'], mode='head'),
                                               fitted['executed'])
        instructions, skills = dict((parent or {}).get('instructions', {})), dict((parent or {}).get('skills', {}))
        for key, block in result['states'].items():
            binding, kind, target = json.loads(key)
            state = self.engine.store.put_native_state(block).id
            if kind == 'instructions':
                instructions[binding] = state
            else:
                skills[f'{binding}#{target}'] = state
        return make_bundle(program=self.manifest['program'], base=self.engine.native_identity, bank=self.bank,
                           parent=None if parent is None else parent['id'], instructions=instructions, skills=skills,
                           adapters=adapters, mixtures=mixtures, provenance={'actor': 'in-bank-refinement', 'loss': result['loss']['final']})

    def retrieve(self, parent, *, rounds=1):
        """Gradient-free mixing over each scope's view (the retrieval baseline): coordinate search over grid integers
        on the exact support loss, from the parent's integers."""
        from .bundle import make_bundle
        from .head import executed, shares_from_norms
        from .views import mixture_encoding
        shares = shares_from_norms(self.bank.group_norms())
        current = {scope: list((parent or {}).get('loraMixtures', {}).get(scope, {}).get('weights', [0] * len(self.views[scope].slots)))
                   for scope in self.scopes}
        def coefficients_for(values, scope):
            slots = list(self.views[scope].slots)
            return executed(values[scope], torch.zeros(len(slots)), torch.zeros(len(slots), len(self.bank.group_labels)),
                            shares[slots], step=self.bank.number_format.step, beta=.5, head=False)
        def loss_of(values):
            # Trials run through live leaves on the placeholder adapters; only the selected mixture is stored.
            leaves = {self.placeholders[scope]: self.bank.adapter_tensors(self.bank.composed_cores(list(self.views[scope].slots),
                      coefficients_for(values, scope))) for scope in self.scopes}
            with torch.no_grad():
                return self.support.accumulate(dict(self.placeholders), lambda: leaves, self._states_of(parent), backward=False)
        best_loss = loss_of(current)
        for _ in range(rounds):
            for scope in self.scopes:
                for slot, free in enumerate(self.views[scope].available):
                    if not free:
                        continue
                    for delta in (1, -1):
                        trial = {s: list(v) for s, v in current.items()}
                        trial[scope][slot] = max(-self.bank.number_format.max_int, min(self.bank.number_format.max_int, trial[scope][slot] + delta))
                        loss = loss_of(trial)
                        if loss < best_loss:
                            best_loss, current = loss, trial
        adapters, mixtures = {}, {}
        for scope in self.scopes:
            coefficients = coefficients_for(current, scope)
            adapters[scope] = self.bank.compose_adapter(self.engine, self.bank.composed_cores(list(self.views[scope].slots), coefficients)).id
            mixtures[scope] = mixture_encoding(self.views[scope], current[scope], None, coefficients)
        best = make_bundle(program=self.manifest['program'], base=self.engine.native_identity, bank=self.bank,
                           parent=None if parent is None else parent['id'],
                           instructions=dict((parent or {}).get('instructions', {})), skills=dict((parent or {}).get('skills', {})),
                           adapters=adapters, mixtures=mixtures, provenance={'actor': 'gradient-free-mixing'})
        return best, best_loss


def episode(context, *, steps, patience, seed):
    """The crisp operator policy over the services; returns ``(incumbent, incumbent loss, history)``."""
    incumbent, incumbent_loss, history, stale = None, context.evaluate(None), [], 0
    for step in range(steps):
        bundle, proposal, behavior = context.propose(incumbent, sample=step > 0, seed=seed + step)
        loss = context.evaluate(bundle)
        accepted = loss < incumbent_loss
        history.append({'step': step, 'proposal': proposal['id'], 'bundle': bundle['id'], 'supportLoss': loss,
                        'incumbentLoss': incumbent_loss, 'accepted': accepted, 'behavior': behavior})
        if accepted:
            incumbent, incumbent_loss, stale = bundle, loss, 0
        else:
            stale += 1
            if stale >= patience:
                break
    return incumbent, incumbent_loss, history


def release_fragment(context, incumbent):
    """`tune.build`'s release fragment for the selected bundle: composed scope adapters and writer states."""
    artifacts, soft_states = [], []
    if incumbent is None:
        return artifacts, soft_states
    parent_of = {node['id']: node['parent'] for node in context.manifest['tree']}
    for scope in context.scopes:
        requires, parent = [], parent_of[scope]
        while parent is not None:
            if parent in incumbent['adapters']:
                requires.append(incumbent['adapters'][parent])
            parent = parent_of[parent]
        artifacts.append({'node': scope, 'artifact': incumbent['adapters'][scope], 'rank': context.bank.basis.rank,
                          'parameterization': 'native-subspace', 'requires': requires[::-1]})
    for key, (block, state) in sorted(context._states_of(incumbent).items()):
        binding, kind, target = json.loads(key)
        soft_states.append({'binding': binding, 'kind': kind, 'target': target, 'state': state,
                            'textSha256': context.soft[key]['textSha256'], 'tokens': len(block.token_ids)})
    return artifacts, soft_states


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--plan', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    context = Context(args.plan, args.out)
    try:
        learned = context.learned
        incoming_loss = context.evaluate(None)
        incumbent, _, history = episode(context, steps=learned.get('steps', 4), patience=learned.get('patience', 2),
                                        seed=learned.get('seed', 0))
        artifacts, soft_states = release_fragment(context, incumbent)
        if incumbent is not None:
            persist_artifacts(context.engine, [item['artifact'] for item in artifacts] + [s['state'] for s in soft_states],
                              context.out / 'artifacts', context.out / 'states')
            if 'gguf' in context.plan.get('exports', []):
                export_bank_nodes(context.engine, context.bank, context.plan, context.out, artifacts)
        receipt = {'schema': 'natlang.lopt.apply-receipt/1', 'bank': context.bank.snapshot, 'writer': context.trainer.state['writer'],
                   'head': context.trainer.state['head'], 'incomingLoss': incoming_loss,
                   'selected': None if incumbent is None else incumbent['id'], 'history': history,
                   'operatorPolicy': 'crisp-greedy-then-sample-v1'}
        _write(context.out / 'result.json', {'schema': 'natlang.tune.build-result/1', 'artifacts': artifacts,
                                             'families': [dict(item) for item in context.manifest.get('families', [])],
                                             'memberships': [dict(item) for item in context.manifest.get('memberships', [])],
                                             'softStates': soft_states,
                                             'softStateFiles': {item['state']: f"states/{item['state']}.nzs" for item in soft_states},
                                             'receipt': receipt})
    finally:
        context.close()
    return 0


def service(spec):
    """One service call of the Natlang `adapt` program: ``{"plan", "out", "op", ...}``; bundles travel as JSON."""
    context = Context(spec['plan'], spec['out'])
    try:
        op = spec['op']
        parent = spec.get('parent')
        if op == 'evidence':
            return {'view': context.evidence, 'scopes': context.scopes,
                    'incoming': {k: r['state'] for k, r in context.soft.items()}, 'incomingLoss': context.evaluate(None)}
        if op == 'evaluate':
            return {'supportLoss': context.evaluate(parent)}
        if op == 'propose':
            bundle, proposal, behavior = context.propose(parent, sample=spec.get('sample', False), seed=spec.get('seed', 0),
                                                         digests=[tuple(item) for item in spec.get('digests', [])],
                                                         focus=spec.get('focus'))
            persist_artifacts(context.engine, list(bundle['adapters'].values()) + list(bundle['instructions'].values())
                              + list(bundle['skills'].values()), context.out / 'artifacts', context.out / 'states')
            return {'bundle': bundle, 'proposal': proposal, 'behavior': behavior, 'supportLoss': context.evaluate(bundle)}
        if op == 'refine':
            bundle = context.refine(parent, steps=spec.get('steps', 10))
            persist_artifacts(context.engine, list(bundle['adapters'].values()) + list(bundle['instructions'].values())
                              + list(bundle['skills'].values()), context.out / 'artifacts', context.out / 'states')
            return {'bundle': bundle, 'supportLoss': context.evaluate(bundle)}
        if op == 'retrieve':
            bundle, loss = context.retrieve(parent, rounds=spec.get('rounds', 1))
            persist_artifacts(context.engine, list(bundle['adapters'].values()), context.out / 'artifacts', context.out / 'states')
            return {'bundle': bundle, 'supportLoss': loss}
        if op == 'release':
            artifacts, soft_states = release_fragment(context, parent)
            return {'artifacts': artifacts, 'softStates': soft_states}
        raise LearnedOptimizerError(f'unknown adaptation service {op}')
    finally:
        context.close()


def export_bank_nodes(engine, bank, plan, out, artifacts):
    """One unscaled GGUF per scope node's composed bank adapter (exact factors ``B = U C``, ``A = V^T``)."""
    from ...common.hashing import sha256_file_hex
    from ...model.native_adapters import target_parameters
    from ..bases import CanonicalBasis, tensor_digest
    from ..export import export_gguf, export_peft
    from ..parameterization import SubspaceDelta
    from ..stack import StackMember, stack_factors
    from ..targets import module_map
    student = plan['student']
    if not student.get('hfConfigDir') or not student.get('baseGgufSha256'):
        raise LearnedOptimizerError('GGUF export needs student.hfConfigDir and student.baseGgufSha256')
    shapes = {name: tuple(value.shape) for name, value in target_parameters(engine.shared_model).items() if name in bank.basis.left}
    targets, _ = module_map(sorted(shapes))
    R = bank.basis.rank
    bases = {f'{bank.basis_artifact}:{name}': CanonicalBasis(f'{bank.basis_artifact}:{name}', name, R, bank.basis.left[name],
             bank.basis.right[name], tensor_digest(bank.basis.left[name], bank.basis.right[name])) for name in bank.basis.names}
    for artifact in artifacts:
        tensors = engine.store.get_native_tensor_artifact(artifact['artifact']).artifact.tensors
        deltas = {name: SubspaceDelta(f'{bank.basis_artifact}:{name}', R, value.reshape(*value.shape[:-1], R, R))
                  for name, value in tensors.items()}
        file = f"exports/{artifact['artifact']}.gguf"
        target = out / file
        if not target.exists():
            peft = out / 'exports' / (artifact['artifact'] + '.peft')
            export_peft(stack_factors([StackMember(artifact['node'], 'lambda', 1., deltas)], bases, shapes),
                        {name: targets[name] for name in shapes}, peft, base_identity=dict(engine.native_identity),
                        base_model_name_or_path=student['hfConfigDir'], stack_identity=artifact['artifact'])
            export_gguf(peft, student['hfConfigDir'], target)
        artifact['exports'] = {'gguf': {'file': file, 'sha256': sha256_file_hex(target), 'baseGgufSha256': student['baseGgufSha256']}}


if __name__ == '__main__':
    raise SystemExit(main())
