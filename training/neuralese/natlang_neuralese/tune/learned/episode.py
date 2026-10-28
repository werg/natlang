"""Crisp services of the adaptation episode (plan "The adaptation episode", "Reusing rollouts correctly").

`propose` runs the trained optimizer once on a program context: the writer under its adapter re-encodes the bindings
under revision and reaches each scope's decision position; the mixture head gives the component-token distribution and
group gains (`mixture`); the bank composes each scope's adapter; the writer's captures are stored as actual writer
states (never relabelled). ``choose`` selects what executes per scope: the truncated mixture (default), one token
sampled from it (``'sample'``), or a given token (view position or `mixture.NULL`; the finite candidate set the
expected-score update scores). It returns the complete bundle and a behavior receipt holding everything a replay
needs: view and mask, the distribution, kept set and gains, head identity, executed coefficients, writer/key/bank
identities and the seed.

`action_logprobs` scores recorded executor actions (native graphs whose positive weights mark the sampled action
tokens) under a bundle: the shaping term's log-likelihoods.
"""
from __future__ import annotations

import torch

from ...common.hashing import canonical_json_sha256_hex
from . import LearnedOptimizerError
from .bundle import bundle_slot, make_bundle, make_proposal
from .mixture import NULL, coefficients, distribution, one_hot, receipt as mixture_receipt
from .views import mixture_encoding, validate_distribution
from .writer import ArrayWrite, StateWrite

BEHAVIOR_SCHEMA = 'natlang.lopt.behavior/2'


def decide(trainer, out, views):
    """Per scope ``(logits, probs, keep, gains)`` from the writer's decision states under the current head (rows of
    components first seen here are initialized from their keys in this request)."""
    head, fmt = trainer.head, trainer.bank.mixture_format
    decisions = {}
    for scope, hidden in out.decision_hidden.items():
        view = views[scope]
        available = [c for c, free in zip(view['componentIds'], view['available']) if free]
        if trainer.recipe.cue:
            head.initialize_rows(available, out.key_hidden[scope], hidden)
        logits = head.logits(hidden, view['componentIds'], view['available'])
        probs, keep = distribution(logits, fmt.top_p)
        decisions[scope] = (logits, probs, keep, head.gains(hidden))
    return decisions


def propose(trainer, context, *, choose=None, seed=0, owner=None):
    """``context``: an assembled target-shaped dict (messages, writes, arrays with views, manifest, program, parent
    bundle, incoming states) without target labels. ``choose``: None (the mixture), ``'sample'`` or ``{scope: token}``.
    Returns ``(bundle, proposal, behavior)``."""
    from ...model.native_adapters import run_native
    engine, bank, recipe = trainer.engine, trainer.bank, trainer.recipe
    writes = [StateWrite(w['key'], w['kind'], w['text'], w.get('extra_slots', 0)) for w in context['writes']]
    arrays, views = [], {}
    for item in context['arrays']:
        view = item['view']
        views[view['scope']] = view
        keys = tuple(trainer.state['keys'][c] for c in view['componentIds']) if recipe.cue else (None,) * len(view['componentIds'])
        arrays.append(ArrayWrite(view['scope'], tuple(view['available']), keys))
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        out = run_native(engine, lambda: trainer.writer.run(context['messages'], writes, arrays),
                         [{'id': trainer.state['writer'], 'scale': 1.}])
        decisions = decide(trainer, out, views)
    adapters, mixtures, behavior_arrays = {}, {}, {}
    for scope, (logits, probs, keep, gains) in decisions.items():
        view = views[scope]
        slots = [bank.index(c) for c in view['componentIds']]
        chosen = choose.get(scope) if isinstance(choose, dict) else None
        if choose == 'sample':
            index = int(torch.multinomial(probs.detach().cpu(), 1, generator=generator).item())
            chosen = NULL if index == 0 else index - 1
        executed_probs = probs if chosen is None else one_hot(len(slots), chosen).to(probs)
        if chosen is not None and float(probs[0 if chosen == NULL else 1 + int(chosen)]) == 0:
            raise LearnedOptimizerError(f'scope {scope}: the chosen token {chosen} is outside the kept set')
        with torch.no_grad():
            executed = coefficients(executed_probs, gains)
        record = bank.compose_adapter(engine, bank.composed_cores(slots, executed, offset=bank.incoming(engine, context, scope)), owner,
                                      {'scope': scope, 'writer': trainer.state['writer']})
        adapters[scope] = record.id
        receipt = mixture_receipt(trainer.head, view['componentIds'], logits, executed_probs, keep, gains, executed, chosen=chosen)
        validate_distribution(_view(view), bank, [float.fromhex(x) for x in receipt['distribution']])
        mixtures[scope] = mixture_encoding(_view(view), receipt, executed, (context.get('incoming_adapters') or {}).get(scope))
        behavior_arrays[scope] = {'view': view, 'distribution': [float(x).hex() for x in probs.tolist()],
                                  'kept': [bool(x) for x in keep.tolist()], 'gains': [float(x).hex() for x in gains.tolist()],
                                  'chosen': chosen}
    states = {}
    for key, block in out.states.items():
        states[key] = engine.store.put_native_state(block, owner).id
    parent = context.get('parent')
    instructions = dict((parent or {}).get('instructions', {}))
    skills = dict((parent or {}).get('skills', {}))
    for write in context['writes']:
        field, key = bundle_slot(write['key'])
        (instructions if field == 'instructions' else skills)[key] = states[write['key']]
    all_adapters = {**(parent or {}).get('adapters', {}), **adapters}
    all_mixtures = {**(parent or {}).get('loraMixtures', {}), **mixtures}
    behavior = {'schema': BEHAVIOR_SCHEMA, 'writer': trainer.state['writer'], 'head': trainer.head.identity(),
                'keys': {c: trainer.state['keys'][c] for v in views.values() for c in v['componentIds']},
                # The proposal's adapters execute this published snapshot; when the trainer has since updated its
                # live bank/basis, the expected-score update recomputes the distribution under current parameters.
                'bank': bank.snapshot, 'liveBankParameters': trainer.state.get('bank_tensors'), 'seed': seed,
                'choose': choose if not isinstance(choose, dict) else 'given', 'arrays': behavior_arrays,
                'writerReceipt': out.receipt}
    bundle = make_bundle(program=context['program'], base=engine.native_identity, bank=bank,
                         parent=None if parent is None else parent['id'], instructions=instructions, skills=skills,
                         adapters=all_adapters, mixtures=all_mixtures, family_gates=(parent or {}).get('familyGates'),
                         provenance={'actor': 'learned-optimizer', 'writer': trainer.state['writer'],
                                     'behavior': canonical_json_sha256_hex(behavior)})
    return bundle, make_proposal(parent, bundle, behavior), behavior


def candidates(behavior, limit=None):
    """The finite token set of each scope a proposal's expected-score update scores: every kept token (null
    included), most probable first; ``limit`` caps each scope (the update then covers the scored part only and is
    recorded as such)."""
    out = {}
    for scope, item in behavior['arrays'].items():
        probs = [float.fromhex(x) for x in item['distribution']]
        kept = [index for index, flag in enumerate(item['kept']) if flag]
        kept.sort(key=lambda index: -probs[index])
        out[scope] = [NULL if index == 0 else index - 1 for index in kept[:limit]]
    return out


def _view(item):
    from .views import BankView
    return BankView(item['snapshot'], item['scope'], item['level'], tuple(item['slots']), tuple(item['componentIds']),
                    tuple(item['eligible']), tuple(item['dropoutMask']), None if item['probe'] is None else tuple(item['probe']),
                    item['selection'], item['mask'])


def action_logprobs(engine, bundle, manifest, rows):
    """Summed log-probability of each row's weighted action tokens under the bundle (no gradient)."""
    from ...data.native_graph_schema import parse_graph
    from ...model.native_adapters import run_native
    from .bundle import binding_entries, soft_states_for
    from .objectives import supervised_logprobs
    out = []
    for row in rows:
        example = parse_graph(row['graph'], tokenizer_fingerprint=engine.model_config['tokenizer_fingerprint'])
        entries = binding_entries(bundle, manifest, row['binding'])
        soft = soft_states_for(bundle, row['binding'])
        blocks = {name: engine.store.get_native_state(soft[key]).block for name, key in (row.get('soft') or {}).items()}
        identity = {name: {k: getattr(b, k) for k in ('checkpoint_sha256', 'tokenizer_fingerprint', 'embedding_weight_sha256', 'architecture')}
                    for name, b in blocks.items()} or None
        with torch.no_grad():
            logp, weights = run_native(engine, lambda: supervised_logprobs(engine.shared_model, example,
                                       external_blocks=blocks or None, external_identity=identity), entries)
        labels = _labels(example)
        picked = logp.gather(-1, labels.to(logp.device)[:, None])[:, 0]
        out.append(float((picked * (weights > 0)).sum()))
    return out


def _labels(example):
    labels = []
    for node, offset in zip(example.plan.invocations, example.plan.offsets):
        for local in range(1, len(node.token_ids)):
            if example.target_weights[offset + local] > 0:
                labels.append(int(node.token_ids[local]))
    return torch.tensor(labels, dtype=torch.long)


def validate_proposal(bank, bundle, views):
    """Mixture contract of every scope against its bound view (exact length, snapshot, zero unavailable slots)."""
    for scope, mixture in bundle['loraMixtures'].items():
        if scope in views:
            validate_distribution(views[scope], bank, mixture['distribution'])
    return bundle
