"""In-bank fits through the exact composition, gradient parity of the composition and the supervised objectives."""
import torch

from test_native_serving_adapters import engine_fixture
from test_tune_trainer import manifest_for, rows_for
from test_learned_optimizer_bank import _bank, _component
from natlang_neuralese.model.native_adapters import run_native
from natlang_neuralese.tune.learned import mixture as M
from natlang_neuralese.tune.learned import objectives as O
from natlang_neuralese.tune.learned.controls import planted_mixture, planted_rows
from natlang_neuralese.tune.learned.fit import ScopeArray, SupportSet, fit_in_bank, focus_labels, placeholder_adapters
from natlang_neuralese.tune.learned.views import select_view, with_dropout


def planted(tmp_path, seed):
    """A bank, a view and support rows labeled by a planted component-token mixture's exact greedy continuations."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    engine = engine_fixture(tmp_path, 'lfm')
    bank = _bank(engine)
    bank = bank.with_components([_component(bank, s) for s in (1, 2, 3, 4)], reason='planted')
    full = select_view(bank, 'lambda:x', 'lambda', 4, policy='full-bank-v1')
    # A recorded dropout mask that removes exactly one component (the fit must leave it at zero).
    view = next(v for v in (with_dropout(full, .25, seed=k) for k in range(100)) if sum(v.available) == 3)
    plant = planted_mixture(view, rng_seed=seed, components=2)
    gains = torch.full((len(bank.group_labels),), 1.5)
    adapter = bank.compose_adapter(engine, bank.composed_cores(list(view.slots), M.coefficients(torch.tensor(plant), gains))).id
    prompts = [row['graph']['nodes'][0]['token_ids'][:6] for row in rows_for(engine, 'bx', 4)]
    support = SupportSet(engine, manifest_for(engine), planted_rows(engine, adapter, prompts, binding='bx'))
    adapters = placeholder_adapters(engine, bank, ['lambda:x'])
    array = ScopeArray('lambda:x', adapters['lambda:x'], list(view.slots), list(view.available))
    with torch.no_grad():
        reference = support.accumulate(adapters, lambda: {adapters['lambda:x']: bank.adapter_tensors(
            bank.composed_cores(array.slots, M.coefficients(torch.tensor(plant), gains)))}, backward=False)
    return engine, bank, view, support, adapters, array, reference


def test_in_bank_fits_reach_low_support_loss_and_are_exactly_representable(tmp_path):
    for seed in (0, 1):  # two planted programs
        engine, bank, view, support, adapters, array, reference = planted(tmp_path / str(seed), seed)
        with torch.no_grad():
            start = support.accumulate(adapters, lambda: {adapters['lambda:x']: bank.adapter_tensors(
                bank.composed_cores(array.slots, torch.zeros(4, len(bank.group_labels))))}, backward=False)
        result = fit_in_bank(engine, bank, support, [array], steps=30, refine_steps=10, learning_rate=.1)
        fitted = result['arrays']['lambda:x']
        # The tiny fixture model is near uniform, so absolute losses barely move; the planted mixture's own loss is the
        # reachable reference, and the fit must recover at least nine tenths of its improvement over the zero bundle.
        assert start - result['loss']['final'] >= .9 * (start - reference) > 0, (seed, start, reference, result['loss'])
        # Unavailable slots have no mass and execute exact zeros; the label executes exactly as gains times its law.
        for slot, free in enumerate(view.available):
            if not free:
                assert float(fitted['distribution'][1 + slot]) == 0
                assert torch.equal(fitted['executed'][slot], torch.zeros(len(bank.group_labels)))
        torch.testing.assert_close(M.coefficients(fitted['distribution'], fitted['gain']), fitted['executed'])
        assert abs(float(fitted['distribution'].sum()) - 1) < 1e-6 and fitted['support'] >= 1
        engine.store.close()
    labels = focus_labels(lambda excluded: .3 if excluded == 'bx' else 1., ['bx', 'by'], full_gain=1.)
    assert labels['bx']['focus'] and not labels['by']['focus']


def test_gradient_through_the_composition_matches_finite_differences(tmp_path):
    engine, bank, view, support, adapters, array, _ = planted(tmp_path, 0)
    fmt = bank.mixture_format
    mask = torch.tensor([True, *view.available])
    def loss_at(logits, gain_logits, backward):
        def leaves():  # rebuilt per row: each row's backward owns its composition graph
            masked = torch.where(mask, logits, torch.full_like(logits, -torch.inf))
            executed = M.coefficients(torch.softmax(masked, -1), fmt.gain_max * torch.sigmoid(gain_logits))
            return {adapters['lambda:x']: bank.adapter_tensors(bank.composed_cores(array.slots, executed))}
        return support.accumulate(adapters, leaves, backward=backward)
    logits = torch.tensor([0., .3, -.2, .1, .4]).requires_grad_(True)
    gain_logits = torch.zeros(len(bank.group_labels)).requires_grad_(True)
    loss_at(logits, gain_logits, True)
    eps = 1e-2
    free = [1 + slot for slot, flag in enumerate(view.available) if flag]
    for index in free[:2]:
        step = torch.zeros_like(logits)
        step[index] = eps
        with torch.no_grad():
            numeric = (loss_at(logits.detach() + step, gain_logits.detach(), False)
                       - loss_at(logits.detach() - step, gain_logits.detach(), False)) / (2 * eps)
        assert abs(float(logits.grad[index]) - numeric) <= .05 * abs(numeric) + 1e-4, (index, float(logits.grad[index]), numeric)
    step = torch.zeros_like(gain_logits)
    group = int(torch.argmax(gain_logits.grad.abs()))
    step[group] = eps
    with torch.no_grad():
        numeric = (loss_at(logits.detach(), gain_logits.detach() + step, False)
                   - loss_at(logits.detach(), gain_logits.detach() - step, False)) / (2 * eps)
    assert abs(float(gain_logits.grad[group]) - numeric) <= .05 * abs(numeric) + 1e-4
    engine.store.close()


def test_distillation_objective_pulls_toward_the_target(tmp_path):
    engine = engine_fixture(tmp_path, 'lfm')
    bank = _bank(engine)
    bank = bank.with_components([_component(bank, 1), _component(bank, 2)], reason='planted')
    rows = rows_for(engine, 'bx', 1)
    from natlang_neuralese.data.native_graph_schema import parse_graph
    example = parse_graph(rows[0]['graph'], tokenizer_fingerprint=engine.model_config['tokenizer_fingerprint'])
    target_adapter = bank.compose_adapter(engine, bank.composed_cores([0, 1], torch.tensor([[.05] * 12, [-.04] * 12])[:, :len(bank.group_labels)]))
    with torch.no_grad():
        target, weights = run_native(engine, lambda: O.supervised_logprobs(engine.shared_model, example), [{'id': target_adapter.id, 'scale': 1.}])
    coefficient = torch.zeros(2, len(bank.group_labels), requires_grad=True)
    placeholder = placeholder_adapters(engine, bank, ['s'])['s']
    def execute():
        student, w = O.supervised_logprobs(engine.shared_model, example)
        loss = O.distill_loss(target, student, w)
        loss.backward()
        return float(loss)
    leaves = {placeholder: bank.adapter_tensors(bank.composed_cores([0, 1], coefficient))}
    value = run_native(engine, execute, [{'id': placeholder, 'scale': 1.}], leaves=leaves, gradient_precision='fp32')
    assert value > 0 and float(coefficient.grad.abs().sum()) > 0
    # Distillation pulls toward the target: one gradient step reduces the KL along the planted direction.
    assert float(coefficient.grad[0].sum()) < 0 < float(coefficient.grad[1].sum())
