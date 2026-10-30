"""Hindsight reconstruction objective (rl/hindsight.py): the hinted packed graph, the leave-one-out score-function and
pathwise terms against a float64 transcription, and the channel KL entering the loss."""
import torch
import pytest

from natlang_neuralese.model.import_channel import ImportChannel, import_channel, measure_scales
from natlang_neuralese.model.native_graph_plan import ImportSpan, Invocation, compile_graph
from natlang_neuralese.model.shared_state import SharedStateModel
from natlang_neuralese.rl.hindsight import (Hint, hindsight_declaration, hindsight_terms, hinted_graph, group_loss,
                                            leave_one_out)
from test_jolt_privileged_context import BOS, IM_START, IM_END, encode, load, rendering, turn

DECLARATION = hindsight_declaration({'k': 2, 'lambda_len': .1, 'lambda_explore': .01, 'lambda_kl': .002, 'eta': .5})


def tau(roles):
    """A two-node tau*: a child whose answer the root imports, each with assistant actions after its prompt."""
    child = [BOS, *turn('system', 'You count.'), *turn('user', 'Count "A1"'), IM_START, *encode('assistant\n'),
             roles.produce_open, 5, 6, roles.produce_close]
    root = [BOS, *turn('system', 'You run programs.'), IM_START, *encode('user\nprogram'), roles.consume_open,
            roles.consume_close, IM_END, *encode('\n'), IM_START, *encode('assistant\n'), 11, 13, 2]
    at = root.index(roles.consume_open) + 1
    produce = child.index(roles.produce_open) + 1
    nodes = [Invocation('child', tuple(child), producing=((produce, produce + 2),)),
             Invocation('root', tuple(root), imports=(ImportSpan(at, 'child', produce, produce + 2),))]
    actions = [{'turn': 'child', 'local_query': len(child) - 4 + i, 'token_id': child[len(child) - 3 + i]} for i in range(3)]
    actions += [{'turn': 'root', 'local_query': len(root) - 4 + i, 'token_id': root[len(root) - 3 + i]} for i in range(3)]
    return nodes, actions


def writer_nodes():
    """Two writer episodes (one turn each) whose sampled tokens carry the score-function term."""
    nodes, actions = [], []
    for number in range(2):
        tokens = [BOS, *turn('user', f'write hint {number}'), IM_START, *encode('assistant\n'), 20 + number, 21, 22]
        nodes.append(Invocation(f'writer{number}', tuple(tokens)))
        actions.append([{'turn': f'writer{number}', 'query': None, 'local_query': len(tokens) - 4 + i,
                         'token_id': tokens[len(tokens) - 3 + i], 'temperature': .8, 'allowed_token_ids': None}
                        for i in range(3)])
    return nodes, actions


def test_hinted_graph_inserts_each_nodes_import_at_the_declared_place():
    model, roles, identity = load('lfm')
    nodes, actions = tau(roles)
    members = [{'root': Hint('root', (3, 4, 7)), 'child': Hint('child', (8,))}, {'root': Hint('root', (9, 10))}]
    plan, member_actions, hint_names = hinted_graph(nodes, actions, members, rendering(), roles)
    assert hint_names == [['child#m0#hint', 'root#m0#hint'], ['root#m1#hint']]
    by_name = dict(zip((node.name for node in plan.invocations), plan.invocations))
    spec = rendering()
    tokens, offset = spec.state_turn()
    for member, hints in enumerate(members):
        for node in nodes:
            copy = by_name[node.name + f'#m{member}']
            if node.name in hints:
                at, _ = spec.insertion_point(node.token_ids)
                assert copy.token_ids == node.token_ids[:at] + tuple(tokens) + node.token_ids[at:]
                edge = next(edge for edge in copy.imports if edge.source.endswith('#hint'))
                assert (edge.at, edge.source, edge.start, edge.stop) == (
                    at + offset, f'{node.name}#m{member}#hint', 1, 1 + len(hints[node.name].token_ids))
            else:
                assert copy.token_ids == node.token_ids
    flat = [t for node in plan.invocations for t in node.token_ids]
    for member in member_actions:
        assert [flat[action['query'] + 1] for action in member] == [action['token_id'] for action in actions]
    with pytest.raises(ValueError, match='no tau'):
        hinted_graph(nodes, actions, [{'nobody': Hint('nobody', (3,))}], rendering(), roles)


def literal(model, hidden, member_actions, writer_actions, kl, *, hint_tokens, explore_tokens, outcomes):
    d = DECLARATION
    log = lambda rows, temperature: torch.log_softmax(model.logits(hidden[rows]).double() / temperature, -1)
    nll = torch.stack([-log(torch.tensor([a['query'] for a in actions]), 1.)[
        torch.arange(len(actions)), torch.tensor([a['token_id'] for a in actions])].sum() for actions in member_actions])
    writer = torch.stack([log(torch.tensor([a['query'] for a in actions]), .8)[
        torch.arange(len(actions)), torch.tensor([a['token_id'] for a in actions])].sum() for actions in writer_actions])
    rewards = [-float(nll[i]) - d['lambda_len'] * hint_tokens[i] - d['lambda_explore'] * explore_tokens[i]
               - d['lambda_kl'] * float(kl[i]) + d['eta'] * outcomes[i] for i in range(2)]
    advantages = torch.tensor([rewards[0] - rewards[1], rewards[1] - rewards[0]], dtype=torch.float64)
    return (nll + d['lambda_kl'] * kl.double() - advantages * writer).sum() / (2 * len(member_actions[0]))


def test_group_loss_matches_float64_transcription_and_reaches_the_writer_through_the_import():
    model, roles, identity = load('lfm')
    nodes, actions = tau(roles)
    prefix, writer_actions = writer_nodes()
    members = [{'root': Hint('root', (3, 4, 7)), 'child': Hint('child', (8, 9))}, {'root': Hint('root', (9, 10))}]
    plan, member_actions, hint_names = hinted_graph(nodes, actions, members, rendering(), roles, prefix=prefix)
    offsets = dict(zip((node.name for node in plan.invocations), plan.offsets))
    writer_actions = [[{**a, 'query': offsets[a['turn']] + a['local_query']} for a in actions_] for actions_ in writer_actions]
    sources = tuple(name for names in hint_names for name in names)
    channel = ImportChannel(sources, .5, 1., measure_scales(model, plan), seed=3)
    with import_channel(channel):
        result = model(plan)
    from natlang_neuralese.rl.native_replay import token_action_logps
    hidden = result['h_final'][0]
    writer_logp = torch.stack([token_action_logps(model, hidden, a).sum() for a in writer_actions])
    options = dict(hint_tokens=[5, 2], explore_tokens=[7, 3], outcomes=[1., 0.])
    loss, details = group_loss(model, result, member_actions, hint_names, writer_logp, declaration=DECLARATION, **options)
    kl = torch.stack([result['import_channel']['kl'][:2].sum(), result['import_channel']['kl'][2]])
    assert kl.min() > 0 and details['kl'] == pytest.approx([float(v) for v in kl], rel=1e-6)
    expected = literal(model, hidden, member_actions, writer_actions, kl, **options)
    assert float(loss) == pytest.approx(float(expected), rel=1e-4, abs=1e-6)
    assert details['advantages'][0] == pytest.approx(-details['advantages'][1])
    parameters = [p for p in model.parameters() if p.requires_grad]
    actual = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    reference = torch.autograd.grad(expected, parameters, retain_graph=True, allow_unused=True)
    for a, b in zip(actual, reference):
        if a is None:
            assert b is None or b.abs().max() == 0
        else:
            torch.testing.assert_close(a.double(), b.double(), rtol=2e-3, atol=1e-7)
    # The pathwise reconstruction term alone reaches the hint producers' rows (through the consumer import).
    ids = torch.tensor([t for node in plan.invocations for t in node.token_ids])
    embeddings = model.embed(ids[None]).detach().requires_grad_(True)
    with import_channel(channel):
        again = model(plan, embeddings=embeddings)
    from natlang_neuralese.rl.hindsight import reconstruction_nll
    nll = sum(reconstruction_nll(model, again['h_final'][0], acts) for acts in member_actions)
    grad = torch.autograd.grad(nll, embeddings)[0][0]
    hint_rows = torch.tensor([offsets[name] + 1 for name in sources])
    assert grad[hint_rows].abs().amax() > 0


def test_node_classes_give_the_root_and_long_children_a_writer_pass():
    from natlang_neuralese.rl.hindsight import node_classes
    declaration = hindsight_declaration({'child_hint_min_tokens': 100})
    nodes = {'r': {'role': 'root', 'action_tokens': 10}, 'a': {'role': 'child', 'action_tokens': 99},
             'b': {'role': 'child', 'action_tokens': 100}}
    assert node_classes(nodes, declaration) == {'r': 'hint', 'a': 'sft', 'b': 'hint'}
    with pytest.raises(ValueError, match='one root'):
        node_classes({'a': nodes['a']}, declaration)


def test_terms_and_declaration_contract():
    assert leave_one_out([1., 2., 6.]) == [-3., -1.5, 4.5]
    with pytest.raises(ValueError):
        leave_one_out([1.])
    for bad in ({'k': 1}, {'lambda_len': -1}, {'gamma': 1}, {'child_hint_min_tokens': 0}):
        with pytest.raises(ValueError):
            hindsight_declaration(bad)
    assert hindsight_declaration()['trainable'] == 'writer'
    nll = torch.tensor([3., 5.], requires_grad=True)
    logp = torch.tensor([-1., -2.], requires_grad=True)
    loss, details = hindsight_terms(nll, torch.zeros(2), logp, hint_tokens=[1, 1], explore_tokens=[0, 0],
                                    outcomes=[0, 0], declaration=hindsight_declaration({'k': 2, 'lambda_kl': 0.}),
                                    target_tokens=4)
    loss.backward()
    assert details['advantages'] == [2., -2.]
    torch.testing.assert_close(nll.grad, torch.tensor([.125, .125]))
    torch.testing.assert_close(logp.grad, torch.tensor([-.25, .25]))
