"""Joint learned optimizer: number grid, coefficient head, shared basis/component bank and gradient probe
(plans/neuralese/JOINT_LEARNED_OPTIMIZER.md) on the real native engine."""
import glob
import math

import pytest
import torch

from test_native_serving_adapters import engine_fixture
from test_tune_trainer import rows_for
from natlang_neuralese.model.native_adapters import run_native, target_parameters
from natlang_neuralese.model.shared_state import Invocation, compile_graph
from natlang_neuralese.tune.learned import LearnedOptimizerError
from natlang_neuralese.tune.learned import head as H
from natlang_neuralese.tune.learned import numbers as N
from natlang_neuralese.tune.learned.bank import (Component, component_id, core_norms, load_bank, new_bank, normalize, orthonormalize,
                                                 save_bank, transform_core, weight_svd_basis)
from natlang_neuralese.tune.learned.subspace import pooled_basis, probe, source_gradients


class ToyEncoder:
    """A tokenizer with the students' grid spelling: one sign-bearing space token and one digit per value."""
    def __call__(self, text):
        out = []
        index = 0
        while index < len(text):
            char = text[index]
            if char == ' ' and text[index + 1:index + 2] == '-':
                out.append(11); index += 2
            elif char == ' ':
                out.append(10); index += 1
            elif char.isdigit():
                out.append(int(char)); index += 1
            else:
                out.append({'[': 12, ']': 13, ',': 14}[char]); index += 1
        return out


TOKENS = N.GridTokens.from_encoder(ToyEncoder())


def test_real_student_tokenizer_spells_the_grid_as_sign_then_digit():
    paths = glob.glob('/home/werg/.cache/huggingface/hub/models--LiquidAI--LFM2.5-350M/snapshots/*/tokenizer.json')
    if not paths:
        pytest.skip('LFM2.5-350M tokenizer is not cached here')
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(paths[0])
    tokens = N.GridTokens.from_encoder(lambda text: tokenizer.encode(text, add_special_tokens=False).ids)
    assert tokenizer.encode(N.text([3, 0, -2]), add_special_tokens=False).ids == tokens.spell([3, 0, -2])


def test_layout_parse_round_trip_and_contract_failures():
    values, available = [3, 0, -2, 0], [True, True, True, False]
    items = N.layout(TOKENS, values, available)
    ids = [token for token, _, _ in items]
    assert N.parse(TOKENS, ids, available) == values
    assert [kind for _, kind, slot in items if slot == 3] == ['forced-sign', 'forced-digit']
    with pytest.raises(LearnedOptimizerError, match='negative zero'):
        N.parse(TOKENS, TOKENS.spell([0]) [:1] + [TOKENS.negative, TOKENS.digits[0], TOKENS.close], [True])
    with pytest.raises(LearnedOptimizerError, match='must be zero'):
        N.parse(TOKENS, TOKENS.spell([1]), [False])
    with pytest.raises(LearnedOptimizerError, match='template'):
        N.parse(TOKENS, TOKENS.spell([1, 2]), [True])
    assert N.text([3, 0, -2]) == '[ 3, 0, -2]'


def test_token_expectation_matches_exact_enumeration():
    fmt = N.NumberFormat(.25)
    generator = torch.Generator().manual_seed(0)
    sign, pos, neg = (torch.randn(20, generator=generator) for _ in range(3))
    logp = N.value_logprobs(TOKENS, fmt, sign, pos, neg)
    assert math.isclose(float(logp.exp().sum()), 1., rel_tol=1e-6)
    # Enumerate the grammar directly.
    ps = torch.softmax(torch.stack([sign[TOKENS.positive], sign[TOKENS.negative]]), 0)
    dp = torch.softmax(pos[list(TOKENS.digits)], 0)
    dn = torch.softmax(neg[list(TOKENS.digits[1:])], 0)
    mean = sum(ps[0] * dp[m] * m for m in range(10)) - sum(ps[1] * dn[m - 1] * m for m in range(1, 10))
    signed = ps[0] * (1 - dp[0]) - ps[1]
    got_mean, got_signed, estimator = N.expectation(TOKENS, fmt, sign, pos, False, neg)
    assert estimator == N.EXACT_ESTIMATOR
    torch.testing.assert_close(got_mean, mean)
    torch.testing.assert_close(got_signed, signed)
    assert N.expectation(TOKENS, fmt, sign, pos, False)[2] == N.APPROX_ESTIMATOR


def test_quantize_and_saturation():
    fmt = N.NumberFormat(.1)
    assert N.quantize(.04, fmt) == (0, 0., False)
    n, rho, saturated = N.quantize(-.33, fmt)
    assert n == -3 and math.isclose(rho, .3, abs_tol=1e-6) and not saturated
    assert N.quantize(5., fmt)[2] and N.quantize(5., fmt)[0] == 9


def test_coefficient_head_invariants_and_finite_difference_gradient():
    torch.manual_seed(0)
    head = H.CoefficientHead(8, 3)
    shares = H.shares_from_norms(torch.tensor([[1., 2., 1.], [0., 0., 0.], [3., 0., 1.]]))
    hidden = torch.randn(3, 8)
    u, v = head(hidden)
    integers = [2, 0, -4]
    zero = H.executed(integers, u, v, shares, step=.5, beta=.5)
    torch.testing.assert_close(zero, torch.tensor([[1.] * 3, [0.] * 3, [-2.] * 3]))
    with torch.no_grad():
        head.readout.weight.normal_()
        head.readout.bias.normal_()
    u, v = head(hidden)
    c = H.executed(integers, u, v, shares, step=.5, beta=.5)
    assert torch.equal(c[1], torch.zeros(3))
    assert bool((c[0] > 0).all()) and bool((c[2] < 0).all())
    mean = (c * shares).sum(-1) / .5
    assert [round(float(x)) for x in mean.detach()] == integers
    off = H.executed(integers, u, v, shares, step=.5, beta=.5, head=False)
    torch.testing.assert_close(off, zero)
    # Finite differences of the head's bias.
    bias = head.readout.bias
    loss = lambda: (H.executed(integers, *head(hidden), shares, step=.5, beta=.5) ** 2).sum()
    (grad,) = torch.autograd.grad(loss(), [bias])
    eps = 1e-3
    for index in range(bias.numel()):
        with torch.no_grad():
            bias[index] += eps; up = float(loss())
            bias[index] -= 2 * eps; down = float(loss())
            bias[index] += eps
        assert math.isclose(float(grad[index]), (up - down) / (2 * eps), rel_tol=1e-2, abs_tol=1e-3)
    with pytest.raises(LearnedOptimizerError):
        H.CoefficientHead(8, 3, beta=.9)


def _bank(engine, components=(), rank=4):
    names = sorted(target_parameters(engine.shared_model))[:6]
    return new_bank(engine, weight_svd_basis(engine, names, rank), N.NumberFormat(.5), components=components)


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
