"""Crisp services of the adaptation episode (plan "The adaptation episode", "Reusing rollouts correctly").

`propose` runs the trained optimizer once on a program context: the writer under its adapter samples (or
deterministically decodes) every scope's grid array and re-encodes the bindings under revision; the coefficient head
refines the visible integers; the bank composes each scope's adapter; the writer's captures are stored as actual
writer states (never relabelled). It returns the complete bundle and a behavior receipt holding everything a replay
needs: view and mask, token spellings, grammar-restricted log probabilities, head identity and outputs, executed
coefficients, writer/key/bank identities and the RNG seed.

`action_logprobs` scores recorded executor actions (native graphs whose positive weights mark the sampled action
tokens) under a bundle: the executor factor of the joint likelihood and the shaping term's log-likelihoods.
"""
from __future__ import annotations

import torch

from ...common.hashing import canonical_json_sha256_hex
from .bundle import bundle_slot, make_bundle, make_proposal
from .head import executed, receipt as head_receipt, shares_from_norms
from .views import mixture_encoding, validate_array
from .writer import ArrayWrite, StateWrite

BEHAVIOR_SCHEMA = 'natlang.lopt.behavior/1'


def propose(trainer, context, *, sample=True, seed=0, temperature=None, owner=None):
    """``context``: an assembled target-shaped dict (messages, writes, arrays with views, manifest, program, parent
    bundle, incoming states) without target labels. Returns ``(bundle, proposal, behavior)``."""
    from ...model.native_adapters import run_native
    engine, bank, recipe = trainer.engine, trainer.bank, trainer.recipe
    if temperature is not None:
        trainer.writer.temperature = float(temperature)
    writes = [StateWrite(w['key'], w['kind'], w['text'], w.get('extra_slots', 0)) for w in context['writes']]
    arrays = []
    views = {}
    for item in context['arrays']:
        view = item['view']
        views[view['scope']] = view
        keys = tuple(trainer.state['keys'][c] for c in view['componentIds']) if recipe.cue else (None,) * len(view['componentIds'])
        arrays.append(ArrayWrite(view['scope'], tuple(view['available']), keys, None))
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        out = run_native(engine, lambda: trainer.writer.run(context['messages'], writes, arrays, sample=sample,
                                                            generator=generator, branch=False),
                         [{'id': trainer.state['writer'], 'scale': 1.}])
    shares_all = shares_from_norms(bank.group_norms())
    adapters, mixtures, behavior_arrays = {}, {}, {}
    for scope, integers in out.integers.items():
        view = views[scope]
        slots = [bank.index(c) for c in view['componentIds']]
        with torch.no_grad():
            u, v = trainer.head(out.decision_hidden[scope])
            coefficients = executed(integers, u, v, shares_all[slots], step=bank.number_format.step, beta=recipe.beta,
                                    head=recipe.head_mode == 'head')
        record = bank.compose_adapter(engine, bank.composed_cores(slots, coefficients, offset=bank.incoming(engine, context, scope)), owner,
                                      {'scope': scope, 'writer': trainer.state['writer']})
        adapters[scope] = record.id
        receipt = head_receipt(trainer.head, integers, u, v, coefficients, mode=recipe.head_mode)
        mixtures[scope] = mixture_encoding(_view(view), integers, receipt, coefficients,
                                           (context.get('incoming_adapters') or {}).get(scope))
        behavior_arrays[scope] = {'tokens': out.tokens[scope], 'integers': integers,
                                  'logprobs': [float(x).hex() for x in out.logprobs[scope].tolist()], 'view': view}
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
                # live bank/basis, those saved parameters differ and the executor ratio of `policy_step` covers it.
                'bank': bank.snapshot, 'liveBankParameters': trainer.state.get('bank_tensors'), 'seed': seed, 'sampled': sample, 'temperature': trainer.writer.temperature,
                'arrays': behavior_arrays, 'writerReceipt': out.receipt,
                'optimizerLogprob': float(sum(sum(float.fromhex(x) for x in a['logprobs']) for a in behavior_arrays.values())).hex()}
    bundle = make_bundle(program=context['program'], base=engine.native_identity, bank=bank,
                         parent=None if parent is None else parent['id'], instructions=instructions, skills=skills,
                         adapters=all_adapters, mixtures=all_mixtures, family_gates=(parent or {}).get('familyGates'),
                         provenance={'actor': 'learned-optimizer', 'writer': trainer.state['writer'],
                                     'behavior': canonical_json_sha256_hex(behavior)})
    return bundle, make_proposal(parent, bundle, behavior), behavior


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
    """Array contract of every scope against its bound view (exact length, snapshot, zero unavailable slots)."""
    for scope, mixture in bundle['loraMixtures'].items():
        if scope in views:
            validate_array(views[scope], bank, mixture['weights'])
    return bundle
