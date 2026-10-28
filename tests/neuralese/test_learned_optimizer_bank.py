"""Joint learned optimizer: component-token mixture law, shared basis/component bank and gradient probe
(plans/neuralese/JOINT_LEARNED_OPTIMIZER.md) on the real native engine."""
import math

import pytest
import torch

from test_native_serving_adapters import engine_fixture
from test_tune_trainer import rows_for
from natlang_neuralese.model.native_adapters import run_native, target_parameters
from natlang_neuralese.model.shared_state import Invocation, compile_graph
from natlang_neuralese.tune.learned import LearnedOptimizerError
from natlang_neuralese.tune.learned import mixture as M
from natlang_neuralese.tune.learned.bank import (Component, component_id, core_norms, load_bank, new_bank, normalize, orthonormalize,
                                                 save_bank, transform_core, weight_svd_basis)
from natlang_neuralese.tune.learned.subspace import pooled_basis, probe, source_gradients


def test_masked_top_p_law_executes_gains_times_probabilities():
    head = M.MixtureHead(6, 2, ['a', 'b', 'c'], M.MixtureFormat(top_p=.8), dtype=torch.float32)
    with torch.no_grad():
        head.rows[1:] = torch.tensor([[3., 0, 0, 0, 0, 0], [0, 2., 0, 0, 0, 0], [0, 0, 9., 0, 0, 0]])
    hidden = torch.tensor([1., 1.4, 1., 0, 0, 0])
    logits = head.logits(hidden, ['a', 'b', 'c'], (True, True, False))
    assert logits[3] == -math.inf  # the unavailable component is masked before the softmax
    probs, keep = M.distribution(logits, .8)
    full = torch.softmax(logits, -1)
    assert keep.tolist() == [False, True, True, False] and float(full[1] + full[2]) >= .8 > float(full[1])
    torch.testing.assert_close(probs, torch.where(keep, full, torch.zeros_like(full)) / full[keep].sum())
    gains = head.gains(hidden)
    torch.testing.assert_close(gains, torch.full((2,), 2. * .25))  # zero weight, bias at logit(gain_init)
    torch.testing.assert_close(M.coefficients(probs, gains), probs[1:, None] * gains[None, :])
    assert M.one_hot(3, M.NULL).tolist() == [1., 0, 0, 0] and M.one_hot(3, 1).tolist() == [0, 0, 1., 0]
    with pytest.raises(LearnedOptimizerError):
        M.MixtureFormat(top_p=0.)


def test_rows_initialize_from_keys_follow_components_and_round_trip():
    fmt = M.MixtureFormat()
    head = M.MixtureHead(4, 3, ['a', 'b'], fmt)
    assert head.rows.dtype == torch.bfloat16 and head.initialized.tolist() == [True, False, False]
    decision = torch.tensor([2., 0, 0, 0])
    assert head.initialize_rows(['b'], torch.tensor([[0., 5, 0, 0]]), decision) == ['b']
    assert head.initialize_rows(['b'], torch.tensor([[1., 0, 0, 0]]), decision) == []  # initialized once
    torch.testing.assert_close(head.rows[2].float(), torch.tensor([0., 4. / 2, 0, 0]))
    grown = head.resized(['c', 'b'])
    torch.testing.assert_close(grown.rows[2], head.rows[2])
    assert grown.initialized.tolist() == [True, False, True]
    again = M.MixtureHead(4, 3, ['a', 'b'], fmt)
    again.load(head.tensors())
    assert again.identity() == head.identity()
    assert M.MixtureFormat.from_json(fmt.as_json()) == fmt


def test_labels_balance_and_expected_score_are_exact():
    logits = torch.tensor([0., 1., -1., -math.inf], requires_grad=True)
    target = torch.tensor([.2, .8, 0., 0.])
    probs = torch.softmax(logits, -1)
    torch.testing.assert_close(M.mixture_kl(target, logits), (target[:2] * (target[:2] / probs[:2]).log()).sum())
    a, b = torch.tensor([.5, .5, 0.]), torch.tensor([.5, 0., .5])
    value = M.balance([(['null', 'x', 'y'], a), (['null', 'x', 'z'], b)])
    mean = torch.tensor([.5, .25, 0., .25])
    assert abs(float(value) - (math.log(4) + float((mean[mean > 0] * mean[mean > 0].log()).sum()))) < 1e-6
    assert M.balance([(['null'], torch.ones(1))]) is None
    rewards = [1., 3., 0., 0.]
    probs, _ = M.distribution(logits, 1.)
    loss = M.expected_score_loss(probs, rewards)
    loss.backward()
    # The exact gradient of -sum_k p_k R_k: -p_k (R_k - E[R]).
    p = torch.softmax(logits.detach(), -1)
    expected = -p * (torch.tensor(rewards) - (p * torch.tensor(rewards)).sum())
    torch.testing.assert_close(logits.grad[:3], expected[:3])
    gains = torch.tensor([.5, 1.5], requires_grad=True)
    assert float(M.gain_loss([.5, 1.5], gains, M.MixtureFormat())) == 0


def _bank(engine, components=(), rank=4):
    names = sorted(target_parameters(engine.shared_model))[:6]
    return new_bank(engine, weight_svd_basis(engine, names, rank), M.MixtureFormat(), components=components)


def _component(bank, seed, rank=2):
    generator = torch.Generator().manual_seed(seed)
    factors = {name: (torch.randn(*bank.basis.left[name].shape[:-2], bank.basis.rank, rank, generator=generator),
                      torch.randn(*bank.basis.left[name].shape[:-2], bank.basis.rank, rank, generator=generator))
               for name in bank.basis.names[seed % 2::2]}
    source = {'kind': 'planted', 'seed': seed}
    component, _ = normalize(Component(component_id(source, factors), factors, None, f'planted {seed}', source), bank.q)
    return component


def test_composed_bank_adapter_executes_its_exact_dense_update(tmp_path):
    engine = engine_fixture(tmp_path, 'lfm')
    bank = _bank(engine)
    bank = bank.with_components([_component(bank, 1), _component(bank, 2)], reason='planted')
    assert math.isclose(float(sum(core_norms(bank.components[0], bank.q).values())), 1., rel_tol=1e-5)
    coefficients = torch.tensor([[.7] * len(bank.group_labels), [-1.3] * len(bank.group_labels)])
    cores = bank.composed_cores([0, 1], coefficients)
    adapter = bank.compose_adapter(engine, cores)
    plan = compile_graph([Invocation('answer', (4, 5, 6, 7))])
    model = engine.shared_model
    adapted = run_native(engine, lambda: model.logits(model(plan)['h_final']).detach(), [{'id': adapter.id, 'scale': 1.}])
    params = target_parameters(model)
    saved = {name: params[name].detach().clone() for name in bank.basis.names}
    try:
        with torch.no_grad():
            for name in bank.basis.names:
                params[name] += bank.dense_update(name, cores[name]).to(params[name])
        reference = model.logits(model(plan)['h_final']).detach()
    finally:
        with torch.no_grad():
            for name, value in saved.items():
                params[name].copy_(value)
    torch.testing.assert_close(adapted, reference, rtol=1e-4, atol=1e-5)
    # Weight-space projection recovers a planted mixture exactly.
    a, residual, target = bank.fit_mixture(cores)
    torch.testing.assert_close(a, torch.tensor([.7, -1.3]), rtol=1e-4, atol=1e-4)
    assert residual < 1e-6 * target
    # Snapshot round trip; any component change is a new snapshot.
    manifest = save_bank(engine, bank)
    assert load_bank(engine, manifest).snapshot == bank.snapshot
    changed = bank.with_components(bank.components[:1], reason='retire')
    assert changed.snapshot != bank.snapshot and changed.parent == bank.snapshot


def test_live_bank_basis_leaves_receive_gradients(tmp_path):
    engine = engine_fixture(tmp_path, 'lfm')
    bank = _bank(engine)
    bank = bank.with_components([_component(bank, 1)], reason='planted')
    cores = bank.composed_cores([0], torch.ones(1, len(bank.group_labels)))
    adapter = bank.compose_adapter(engine, cores)
    basis = {f'{side}.{name}': getattr(bank.basis, side)[name].clone().requires_grad_(True)
             for name in bank.basis.names for side in ('left', 'right')}
    plan = compile_graph([Invocation('answer', (4, 5, 6, 7))])
    model = engine.shared_model
    def execute():
        loss = model.logits(model(plan)['h_final']).float().pow(2).mean()
        return torch.autograd.grad(loss, list(basis.values()), allow_unused=True)
    grads = run_native(engine, execute, [{'id': adapter.id, 'scale': 1.}],
                       leaves={adapter.id: bank.adapter_tensors(cores), bank.basis_artifact: basis}, gradient_precision='fp32')
    # dL/dU = G V C^T: matrices without a component core correctly receive none.
    covered = set(bank.components[0].factors)
    for key, g in zip(basis, grads):
        name = key.split('.', 1)[1]
        assert (g is not None and float(g.abs().sum()) > 0) == (name in covered), key


def test_orthonormalize_and_core_transform_preserve_the_update():
    generator = torch.Generator().manual_seed(3)
    U, V, C = (torch.randn(12, 4, generator=generator), torch.randn(9, 4, generator=generator), torch.randn(4, 4, generator=generator))
    qu, ru = orthonormalize(U)
    qv, rv = orthonormalize(V)
    torch.testing.assert_close(qu @ transform_core(C, ru, rv) @ qv.T, U @ C @ V.T, rtol=1e-4, atol=1e-5)


def test_gradient_probe_matches_finite_differences_and_subspaces_pool(tmp_path):
    engine = engine_fixture(tmp_path, 'lfm')
    bank = _bank(engine)
    bank = bank.with_components([_component(bank, 1), _component(bank, 2), _component(bank, 3)], reason='planted')
    rows = rows_for(engine, 'bx', 3)
    from natlang_neuralese.data.native_graph import graph_loss
    from natlang_neuralese.data.native_graph_schema import parse_graph
    examples = [parse_graph(row['graph'], tokenizer_fingerprint=engine.model_config['tokenizer_fingerprint']) for row in rows]
    support = lambda: sum(graph_loss(engine.shared_model, example, checkpoint_layers=False)[0] for example in examples)
    base = torch.tensor([[.03] * len(bank.group_labels), [0.] * len(bank.group_labels), [-.02] * len(bank.group_labels)])
    cores = bank.composed_cores([0, 1, 2], base)
    adapter = bank.compose_adapter(engine, cores)
    values, _ = probe(engine, bank, [0, 1, 2], adapter.id, cores, support)
    # Exact reference: differentiate the support loss through the composition into one coefficient per slot.
    scalar = torch.zeros(3, requires_grad=True)
    def execute():
        live = bank.composed_cores([0, 1, 2], base + scalar[:, None])
        loss = support()
        return torch.autograd.grad(loss, [scalar])[0]
    live_cores = bank.composed_cores([0, 1, 2], base + scalar[:, None])
    reference = run_native(engine, execute, [{'id': adapter.id, 'scale': 1.}],
                           leaves={adapter.id: bank.adapter_tensors(live_cores)}, gradient_precision='fp32')
    torch.testing.assert_close(torch.tensor(values, dtype=torch.float32), reference, rtol=1e-3, atol=1e-6)
    eps = .05
    for k in range(3):
        losses = []
        for sign in (1, -1):
            shifted = base.clone(); shifted[k] += sign * eps
            other = bank.compose_adapter(engine, bank.composed_cores([0, 1, 2], shifted))
            with torch.no_grad():
                losses.append(float(run_native(engine, support, [{'id': other.id, 'scale': 1.}])))
        assert math.isclose(values[k], (losses[0] - losses[1]) / (2 * eps), rel_tol=.1, abs_tol=2e-4)
    gradients = source_gradients(engine, rows, bank.basis.names)
    pooled = pooled_basis(engine.native_identity, [gradients], 4, per_source_rank=2,
                          weight_svd=weight_svd_basis(engine, bank.basis.names, 4))
    assert pooled.rank == 4 and pooled.method['kind'] == 'gradient-subspace'
