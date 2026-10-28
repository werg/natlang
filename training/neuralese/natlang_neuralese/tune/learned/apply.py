"""Applying the trained learned optimizer: the adaptation episode's services and the crisp episode.

`python -m natlang_neuralese.tune.learned.apply --plan plan.json --out DIR` has the same plan and `result.json`
contract as `tune.build`, so `natlang tune --learned` validates, activates and rolls back its release through the
ordinary pipeline. The plan is a `natlang.tune-plan/v1` plus ``learned``: ``{"bank": bank.json, "state": optimizer
state.json, "store": operation artifact store, "scopes": {level: view size}, "steps": K, "patience": P, "seed": S}``.

Services (one implementation, used by the crisp episode here and by the Natlang `adapt` program through
``python -m natlang_neuralese.tune.learned service``): ``evidence`` (the permitted view), ``propose`` (a complete
learned bundle), ``evaluate`` (exact support loss of a bundle), ``refine`` (local in-bank refinement from a bundle),
``retrieve`` (the best single component token per scope on the support loss, the retrieval baseline). The crisp operator policy (the
pluggable default) proposes, deterministically first and then by sampling, keeps the better bundle on the support
evidence and stops after ``patience`` steps without improvement or after ``K`` steps (structural measure).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from . import LearnedOptimizerError
from . import mixture as M
from ..build import export_release_nodes, load_plan, persist_artifacts, _write


class Context:
    """One program's prepared adaptation context in an output directory (`program.ProgramContext` with the trained
    optimizer). ``learned.incoming`` optionally names a tuner build whose tuned bundle is the incoming one: the
    optimizer then revises that tuned program (its adapters must slice the bank basis)."""

    def __init__(self, plan_path, out):
        from ...serve import load_native_engine
        from ...serve.store import TensorStore
        from .bank import load_bank
        from .program import ProgramContext, read_build
        from .trainer import OuterRecipe, OuterTrainer
        self.plan, _, _, _ = load_plan(plan_path)
        self.plan_path = str(Path(plan_path).resolve())
        self.learned = self.plan.get('learned')
        if not isinstance(self.learned, dict):
            raise LearnedOptimizerError('the plan carries no learned optimizer settings')
        config = self.plan.get('engine') or {'checkpoint': self.plan['student']['checkpoint'], **self.plan['student'].get('engine', {})}
        self.config = config
        self.engine = load_native_engine(config['checkpoint'], device=config.get('device', 'cpu'),
                                         dtype=getattr(torch, config.get('dtype', 'float32')))
        self.engine.store.close()
        self.engine.store = TensorStore(self.learned['store'])
        self.out = Path(out)
        learned, engine = self.learned, self.engine
        self.bank = load_bank(engine, json.loads(Path(learned['bank']).read_text())['bank'])
        state = json.loads(Path(learned['state']).read_text())
        self.trainer = OuterTrainer(engine, self.bank, OuterRecipe.from_json(state['recipe']), [], state=state)
        incoming = learned.get('incoming')
        self.pc = ProgramContext(engine, self.bank, plan_path, out=out, scopes=learned['scopes'], build=incoming)
        pc = self.pc
        self.manifest, self.soft, self.rows, self.support = pc.manifest, pc.soft, pc.rows, pc.support
        self.scopes, self.placeholders, self.views, self.evidence = pc.scopes, pc.placeholders, pc.views, pc.evidence
        self.incoming_states = {key: record['state'] for key, record in self.soft.items()}
        self.incoming_adapters = {}
        if incoming:
            tuned = read_build(incoming)['state']
            self.incoming_states.update(tuned.get('soft_states', {}))
            self.incoming_adapters = {s: a for s, a in tuned['node_artifacts'].items() if s in self.scopes}
        self.context = pc.request(incoming_states=self.incoming_states, incoming_adapters=self.incoming_adapters)

    def close(self):
        self.engine.store.close()

    # -- services ------------------------------------------------------------------------------------------------
    def _states_of(self, bundle):
        out = {}
        for key, record in self.soft.items():
            binding, kind, target = json.loads(key)
            incoming = self.incoming_states.get(key, record['state'])
            state = incoming if bundle is None else (
                bundle['instructions'].get(binding, incoming) if kind == 'instructions'
                else bundle['skills'].get(f'{binding}#{target}', incoming))
            out[key] = (self.engine.store.get_native_state(state).block, state)
        return out

    def evaluate(self, bundle):
        adapters = {s: self.incoming_adapters.get(s, self.placeholders[s]) for s in self.scopes} if bundle is None \
            else {s: bundle['adapters'][s] for s in self.scopes}
        with torch.no_grad():
            return self.support.accumulate(adapters, lambda: {}, self._states_of(bundle), backward=False)

    # -- what one revision may change ----------------------------------------------------------------------------
    OPERATIONS = ('reencode', 'replace-skill', 'weights', 'keep')

    def selection(self, operations=None):
        """``(soft keys, scopes)`` a revision may change. ``operations`` maps bindings to `OPERATIONS` (the Natlang
        `chooseRevision` plan): "reencode" rewrites the binding's instruction state, "replace-skill" its skill states,
        "weights" the LoRA mixtures of every scope on its path, "keep" nothing; unnamed bindings keep everything. None
        selects the whole bundle. Every operator (proposal, refinement, retrieval) changes exactly the selection and
        carries every other entry over from its parent unchanged."""
        if operations is None:
            return set(self.soft), set(self.scopes)
        bindings = {binding['id']: binding for binding in self.manifest['bindings']}
        keys, scopes = set(), set()
        for binding, operation in operations.items():
            if binding not in bindings:
                raise LearnedOptimizerError(f'the revision plan names {binding}, which is not a binding of this program; '
                                            f'use one of {sorted(bindings)}')
            if operation not in self.OPERATIONS:
                raise LearnedOptimizerError(f'revision operation {operation!r} is not one of {list(self.OPERATIONS)}')
            kind = {'reencode': 'instructions', 'replace-skill': 'skill'}.get(operation)
            if kind:
                own = {key for key in self.soft if json.loads(key)[:2] == [binding, kind]}
                if not own:
                    raise LearnedOptimizerError(f'{binding} has no {"skill" if kind == "skill" else "instruction"} state '
                                                f'to revise; choose "weights" or "keep" for it')
                keys |= own
            elif operation == 'weights':
                scopes |= set(bindings[binding]['ancestry']) & set(self.scopes)
        if not keys and not scopes:
            raise LearnedOptimizerError('the revision plan selects nothing to change; choose the operator "keep"')
        return keys, scopes

    def _parent_adapters(self, parent):
        """Every scope's adapter of ``parent`` (None: the incoming bundle)."""
        return {s: self.incoming_adapters.get(s, self.placeholders[s]) for s in self.scopes} if parent is None \
            else dict(parent['adapters'])

    def _carry(self, parent, adapters, mixtures, states, *, actor, extra=None):
        """A complete bundle: the selection's new entries over ``parent``'s (or the incoming) others."""
        from .bundle import bundle_slot, make_bundle
        instructions, skills = dict((parent or {}).get('instructions', {})), dict((parent or {}).get('skills', {}))
        for key, state in states.items():
            field, slot = bundle_slot(key)
            (instructions if field == 'instructions' else skills)[slot] = state
        all_adapters = {**self._parent_adapters(parent), **adapters}
        all_mixtures = {**(parent or {}).get('loraMixtures', {}), **mixtures}
        for scope in self.scopes:
            if scope not in all_mixtures:  # an unrevised scope of the incoming bundle: the null token, no change
                view = self.views[scope]
                all_mixtures[scope] = self._encoding(scope, M.one_hot(len(view.slots), M.NULL),
                                                     torch.zeros(len(self.bank.group_labels)), chosen=M.NULL)
                all_adapters[scope] = self.bank.compose_adapter(self.engine, self.bank.composed_cores(
                    list(view.slots), torch.zeros(len(view.slots), len(self.bank.group_labels)),
                    offset=self.bank.incoming(self.engine, self.context, scope))).id
        return make_bundle(program=self.manifest['program'], base=self.engine.native_identity, bank=self.bank,
                           parent=None if parent is None else parent['id'], instructions=instructions, skills=skills,
                           adapters=all_adapters, mixtures=all_mixtures, provenance={'actor': actor, **(extra or {})})

    def incoming_bundle(self):
        """The incoming adaptation as a complete bundle (its every scope composed exactly), the parent of a first
        revision."""
        if getattr(self, '_incoming', None) is None:
            self._incoming = self._carry(None, {}, {}, {}, actor='incoming')
        return self._incoming

    def propose(self, parent, *, sample, seed, digests=(), focus=None, key_evidence=None, operations=None):
        """The learned writer's revision of the selection; the request carries the analysis' focus, key evidence and
        native digests through the one request builder."""
        from .episode import propose
        keys, scopes = self.selection(operations)
        context = self.pc.request(incoming_states=self.incoming_states, incoming_adapters=self.incoming_adapters,
                                  focus=focus, digests=digests, key_evidence=key_evidence)
        context = dict(context, parent=parent or self.incoming_bundle(), program=context['program'],
                       writes=[w for w in context['writes'] if w['key'] in keys],
                       arrays=[a for a in context['arrays'] if a['view']['scope'] in scopes])
        return propose(self.trainer, context, choose='sample' if sample else None, seed=seed)

    def _encoding(self, scope, probs, gains, *, chosen=None):
        """The bundle mixture of one scope executing ``probs`` (null first) with ``gains``."""
        from .views import mixture_encoding
        executed = M.coefficients(probs, gains)
        receipt = M.receipt(None, list(self.views[scope].component_ids), None, probs, probs > 0, gains, executed, chosen=chosen)
        return mixture_encoding(self.views[scope], receipt, executed, self.incoming_adapters.get(scope))

    def refine(self, parent, *, steps=10, operations=None):
        """Local in-bank refinement of the selection from ``parent``'s mixtures and states (a gradient operator inside
        the episode); everything else runs at its parent value."""
        from .fit import fit_in_bank
        keys, scopes = self.selection(operations)
        parents = self._parent_adapters(parent)
        arrays = [array for array in self.pc.scope_arrays(self.incoming_adapters) if array.scope in scopes]
        init = None
        if parent is not None:
            init = {array.scope: {'distribution': parent['loraMixtures'][array.scope]['distribution'],
                                  'gain': parent['loraMixtures'][array.scope]['gains']} for array in arrays}
        soft = {key: state for key, (_, state) in self._states_of(parent).items()}
        result = fit_in_bank(self.engine, self.bank, self.support, arrays, soft_states=soft, steps=steps,
                             refine_steps=max(1, steps // 2), init=init, fixed_states=set(soft) - keys,
                             fixed_adapters={s: a for s, a in parents.items() if s not in scopes})
        adapters, mixtures = {}, {}
        for array in arrays:
            fitted = result['arrays'][array.scope]
            adapters[array.scope] = self.bank.compose_adapter(self.engine, self.bank.composed_cores(
                array.slots, fitted['executed'], offset=array.offset)).id
            mixtures[array.scope] = self._encoding(array.scope, fitted['distribution'], fitted['gain'])
        states = {key: self.engine.store.put_native_state(block).id for key, block in result['states'].items()}
        return self._carry(parent, adapters, mixtures, states, actor='in-bank-refinement', extra={'loss': result['loss']['final']})

    def retrieve(self, parent, *, rounds=1, operations=None):
        """The retrieval baseline without gradients: per selected scope, the single best component token (or the
        null token) at its parent's gains (or the format's initial gain) on the exact support loss; ``rounds``
        coordinate passes over the scopes; other scopes run at their parent adapters."""
        _, scopes = self.selection(operations)
        selected = [scope for scope in self.scopes if scope in scopes]
        parents = self._parent_adapters(parent)
        fmt, G = self.bank.mixture_format, len(self.bank.group_labels)
        gains = {scope: torch.tensor((parent or {}).get('loraMixtures', {}).get(scope, {}).get('gains')
                                     or [fmt.gain_max * fmt.gain_init] * G) for scope in selected}
        current = {scope: M.NULL for scope in selected}
        def cores(scope, token):
            view = self.views[scope]
            return self.bank.composed_cores(list(view.slots), M.coefficients(M.one_hot(len(view.slots), token), gains[scope]),
                                            offset=self.bank.incoming(self.engine, self.context, scope))
        def loss_of(choice):
            # Trials run through live leaves on the placeholder adapters; only the selected mixture is stored.
            leaves = {self.placeholders[scope]: self.bank.adapter_tensors(cores(scope, choice[scope])) for scope in selected}
            adapters = {scope: self.placeholders[scope] if scope in scopes else parents[scope] for scope in self.scopes}
            with torch.no_grad():
                return self.support.accumulate(adapters, lambda: leaves, self._states_of(parent), backward=False)
        best_loss = loss_of(current)
        for _ in range(rounds):
            for scope in selected:
                for slot, free in enumerate(self.views[scope].available):
                    if free:
                        trial = {**current, scope: slot}
                        loss = loss_of(trial)
                        if loss < best_loss:
                            best_loss, current = loss, trial
        adapters, mixtures = {}, {}
        for scope in selected:
            adapters[scope] = self.bank.compose_adapter(self.engine, cores(scope, current[scope])).id
            mixtures[scope] = self._encoding(scope, M.one_hot(len(self.views[scope].slots), current[scope]), gains[scope],
                                             chosen=current[scope])
        return self._carry(parent, adapters, mixtures, {}, actor='single-component-retrieval'), best_loss


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


def bundle_release(engine, manifest, adapters, states, text_shas):
    """`tune.build`'s release fragment of one bundle: its scope adapters (tree order, each with its actual rank and
    the adapters of its ancestors) and its writer states (``states``: soft key -> state id; ``text_shas``: soft key ->
    the binding text's hash)."""
    parent_of = {node['id']: node['parent'] for node in manifest['tree']}
    artifacts, soft_states = [], []
    for scope in (node['id'] for node in manifest['tree']):
        if scope not in adapters:
            continue
        requires, parent = [], parent_of[scope]
        while parent is not None:
            if parent in adapters:
                requires.append(adapters[parent])
            parent = parent_of[parent]
        rank = engine.store.get_native_tensor_artifact(adapters[scope]).artifact.provenance['adapter_spec']['rank']
        artifacts.append({'node': scope, 'artifact': adapters[scope], 'rank': rank,
                          'parameterization': 'native-subspace', 'requires': requires[::-1]})
    for key, state in sorted(states.items()):
        binding, kind, target = json.loads(key)
        soft_states.append({'binding': binding, 'kind': kind, 'target': target, 'state': state, 'textSha256': text_shas[key],
                            'tokens': len(engine.store.get_native_state(state).block.token_ids)})
    return artifacts, soft_states


def write_release(engine, manifest, out, artifacts, soft_states, receipt):
    """A release fragment as a `tune.build` output directory (``result.json``, ``artifacts/``, ``states/``), which
    `natlang tune` seals into a release exactly as it does a fit's output."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    persist_artifacts(engine, [item['artifact'] for item in artifacts] + [item['state'] for item in soft_states],
                      out / 'artifacts', out / 'states')
    _write(out / 'result.json', {'schema': 'natlang.tune.build-result/1', 'artifacts': artifacts,
                                 'families': [dict(item) for item in manifest.get('families', [])],
                                 'memberships': [dict(item) for item in manifest.get('memberships', [])],
                                 'softStates': soft_states,
                                 'softStateFiles': {item['state']: f"states/{item['state']}.nzs" for item in soft_states},
                                 'receipt': receipt})


def release_fragment(context, incumbent):
    """The release fragment of the selected bundle (or, when nothing improved on a tuned incoming bundle, that
    incoming bundle); nothing when there are no adapters to release."""
    adapters = incumbent['adapters'] if incumbent is not None else context.incoming_adapters
    if not adapters:
        return [], []
    states = {key: state for key, (_, state) in context._states_of(incumbent).items()}
    return bundle_release(context.engine, context.manifest, {s: a for s, a in adapters.items() if s in context.scopes}, states,
                          {key: record['textSha256'] for key, record in context.soft.items()})


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
        receipt = {'schema': 'natlang.lopt.apply-receipt/1', 'bank': context.bank.snapshot, 'writer': context.trainer.state['writer'],
                   'head': context.trainer.state['head'], 'incomingLoss': incoming_loss,
                   'selected': None if incumbent is None else incumbent['id'], 'history': history,
                   'operatorPolicy': 'crisp-greedy-then-sample-v1'}
        write_release(context.engine, context.manifest, context.out, artifacts, soft_states, receipt)
        if artifacts and 'gguf' in context.plan.get('exports', []):
            export_release_nodes(context.engine, context.plan, context.out, artifacts)
    finally:
        context.close()
    return 0


def handle(context, request):
    """One adaptation service call on a prepared context; bundles travel as JSON. ``operations`` (the revision plan)
    restricts what an operator may change (`Context.selection`)."""
    op, parent, operations = request['op'], request.get('parent'), request.get('operations')

    def persist(bundle, loss):
        # Artifacts for the release, and the measured bundle under its id: a root's verifier checks the bundle an
        # episode reports against what the services actually measured (`tune.reward.mjs`).
        persist_artifacts(context.engine, [*bundle['adapters'].values(), *bundle['instructions'].values(),
                                           *bundle['skills'].values()], context.out / 'artifacts', context.out / 'states')
        (context.out / 'bundles').mkdir(parents=True, exist_ok=True)
        _write(context.out / 'bundles' / f"{bundle['id']}.json", {'bundle': bundle, 'supportLoss': loss})
        return {'bundle': bundle, 'supportLoss': loss}
    if op == 'evidence':
        return {'view': context.evidence, 'scopes': context.scopes,
                'bindings': [binding['id'] for binding in context.manifest['bindings']],
                'incoming': dict(context.incoming_states), 'incomingLoss': context.evaluate(None)}
    if op == 'evaluate':
        return {'supportLoss': context.evaluate(parent)}
    if op == 'propose':
        bundle, proposal, behavior = context.propose(parent, sample=request.get('sample', False), seed=request.get('seed', 0),
                                                     digests=[tuple(item) for item in request.get('digests', [])],
                                                     focus=request.get('focus'), key_evidence=request.get('keyEvidence'),
                                                     operations=operations)
        return {**persist(bundle, context.evaluate(bundle)), 'proposal': proposal, 'behavior': behavior}
    if op == 'refine':
        bundle = context.refine(parent, steps=request.get('steps', 10), operations=operations)
        return persist(bundle, context.evaluate(bundle))
    if op == 'retrieve':
        return persist(*context.retrieve(parent, rounds=request.get('rounds', 1), operations=operations))
    if op == 'release':
        artifacts, soft_states = release_fragment(context, parent)
        return {'artifacts': artifacts, 'softStates': soft_states}
    if op == 'finish':
        controller = request.get('controller')
        if not isinstance(controller, dict) or not controller.get('release'):
            raise LearnedOptimizerError('a controller release identity is required')
        artifacts, soft_states = release_fragment(context, parent)
        receipt = {'schema': 'natlang.lopt.apply-receipt/1', 'bank': context.bank.snapshot,
                   'writer': context.trainer.state['writer'], 'head': context.trainer.state['head'],
                   'incomingLoss': context.evaluate(None), 'selected': None if parent is None else parent['id'],
                   'history': request.get('history', []), 'operatorPolicy': 'natlang-controller-v1',
                   'controller': controller}
        write_release(context.engine, context.manifest, context.out, artifacts, soft_states, receipt)
        if artifacts and 'gguf' in context.plan.get('exports', []):
            export_release_nodes(context.engine, context.plan, context.out, artifacts)
        return receipt
    raise LearnedOptimizerError(f'unknown adaptation service {op}')


def service(spec):
    """One service call of the Natlang `adapt` program: ``{"plan", "out", "op", ...}`` (a fresh context per call)."""
    context = Context(spec['plan'], spec['out'])
    try:
        return handle(context, spec)
    finally:
        context.close()


def serve(stream_in, stream_out, *, commands=None, limit=8):
    """The persistent service: one JSON request per input line, one JSON reply per output line (``{"id", "result"}``
    or ``{"id", "error"}``). A request is an adaptation service call (``{"id", "plan", "out", "op", ...}``, as
    `service`) or, with ``commands``, any command of the package (``{"id", "command", "spec"}``). Adaptation contexts
    stay loaded per (plan, out) while the plan, bank and optimizer state files are unchanged, at most ``limit`` at a time (least recently used first out), so an episode's calls and
    many episodes of one program pay the model and context preparation once. Ends at end of input."""
    import contextlib
    import sys
    contexts = {}
    for line in stream_in:
        if not line.strip():
            continue
        request = json.loads(line)
        reply = {'id': request.get('id')}
        try:
            with contextlib.redirect_stdout(sys.stderr):  # the replies own the output stream
                if 'command' in request:
                    if commands is None:
                        raise LearnedOptimizerError('this service runs adaptation calls only')
                    reply['result'] = commands(request['command'], request['spec'])
                else:
                    key = _context_key(request)
                    for stale in [k for k in contexts if k[:2] == key[:2] and k != key]:
                        contexts.pop(stale).close()
                    context = contexts.pop(key, None) or Context(request['plan'], request['out'])
                    contexts[key] = context
                    while len(contexts) > limit:
                        contexts.pop(next(iter(contexts))).close()
                    reply['result'] = handle(context, request)
        except LearnedOptimizerError as error:
            reply['error'] = str(error)
        except Exception as error:  # an infrastructure fault is reported to the caller, which fails the call
            reply['error'] = f'{type(error).__name__}: {error}'
            print(f'learned optimizer service fault: {reply["error"]}', file=sys.stderr, flush=True)
        stream_out.write(json.dumps(reply, sort_keys=True, default=_tensor_json) + '\n')
        stream_out.flush()
    for context in contexts.values():
        context.close()


def _context_key(request):
    """A loaded context's identity: its plan and output, and the files it read (a retrained optimizer, a published
    bank or a changed plan loads afresh)."""
    plan = Path(request['plan']).resolve()
    learned = json.loads(plan.read_text()).get('learned') or {}
    files = [plan, *(Path(learned[name]).resolve() for name in ('bank', 'state') if learned.get(name))]
    return (str(Path(request['out']).resolve()), str(plan),
            *((str(path), path.stat().st_mtime_ns, path.stat().st_size) for path in files))


def _tensor_json(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f'not JSON serializable: {type(value).__name__}')


if __name__ == '__main__':
    raise SystemExit(main())
