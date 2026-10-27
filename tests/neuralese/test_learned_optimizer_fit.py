"""In-bank fits through the exact composition and the supervised objectives."""
import dataclasses

import torch

from test_native_serving_adapters import engine_fixture
from test_tune_trainer import manifest_for, rows_for
from test_learned_optimizer_bank import _bank, _component
from natlang_neuralese.model.native_adapters import run_native
from natlang_neuralese.tune.learned import head as H
from natlang_neuralese.tune.learned import numbers as N
from natlang_neuralese.tune.learned import objectives as O
from natlang_neuralese.tune.learned.fit import ScopeArray, SupportSet, fit_in_bank, focus_labels, placeholder_adapters
from natlang_neuralese.tune.learned.views import select_view, with_dropout
from natlang_neuralese.tune.learned.writer import Writer, ArrayWrite
from test_learned_optimizer_writer import GRID


def test_in_bank_fit_lowers_support_loss_and_is_exactly_representable(tmp_path):
    engine = engine_fixture(tmp_path, 'lfm')
    bank = _bank(engine)
    bank = bank.with_components([_component(bank, seed) for seed in (1, 2, 3, 4)], reason='planted')
    bank = dataclasses.replace(bank, number_format=N.NumberFormat(.08))
    manifest = manifest_for(engine)
    support = SupportSet(engine, manifest, rows_for(engine, 'bx', 4))
    view = with_dropout(select_view(bank, 'lambda:x', 'lambda', 4, policy='full-bank-v1'), .25, seed=3, all_disabled='resample')
    adapters = placeholder_adapters(engine, bank, ['lambda:x'])
    array = ScopeArray('lambda:x', adapters['lambda:x'], list(view.slots), list(view.available))
    with torch.no_grad():
        start = support.accumulate(adapters, lambda: {adapters['lambda:x']: bank.adapter_tensors(
            bank.composed_cores(array.slots, torch.zeros(4, len(bank.group_labels))))}, backward=False)
    result = fit_in_bank(engine, bank, support, [array], steps=12, refine_steps=6, learning_rate=.05)
    fitted = result['arrays']['lambda:x']
    assert result['loss']['continuous'][-1] < result['loss']['continuous'][0]
    assert result['loss']['final'] < start
    step, saturation = N.grid_step_for(result['arrays']['lambda:x']['means'].tolist())
    assert step > 0 and 0 <= saturation <= .1
    # Unavailable slots stay exact zeros; the label executes exactly through the head's functions.
    for slot, free in enumerate(view.available):
        if not free:
            assert fitted['integers'][slot] == 0 and torch.equal(fitted['executed'][slot], torch.zeros(len(bank.group_labels)))
    shares = H.shares_from_norms(bank.group_norms())[array.slots]
    again = H.executed(fitted['integers'], fitted['u'], fitted['v'], shares, step=bank.number_format.step, beta=.5)
    torch.testing.assert_close(again, fitted['executed'])
    labels = focus_labels(lambda excluded: .3 if excluded == 'bx' else 1., ['bx', 'by'], full_gain=1.)
    assert labels['bx']['focus'] and not labels['by']['focus']


def test_distillation_and_mixture_token_objectives(tmp_path):
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
    writer = Writer(engine, GRID, cue=False)
    with torch.no_grad():
        out = writer.run([{'role': 'user', 'content': 't4 t5'}], [], [ArrayWrite('s', (True, True, False), (None,) * 3, targets=(2, 0, 0))])
    loss, info = O.mixture_token_loss(GRID, bank.number_format, out.sign_logits['s'], out.digit_logits['s'], [2, 0, 0], (True, True, False))
    assert float(loss) > 0 and info == {'slots': 2, 'zero_targets': 1}
