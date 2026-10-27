"""Publication must preserve learned weight updates after a basis coordinate change."""
import dataclasses
import pytest
import torch

from test_native_serving_adapters import engine_fixture
from test_learned_optimizer_bank import _bank, _component
from natlang_neuralese.tune.learned.trainer import OuterRecipe, OuterTrainer, _bank_leaves
from natlang_neuralese.tune.learned.bank import component_id


@pytest.mark.parametrize('family', ['lfm', 'mellum'])
@pytest.mark.parametrize('train_bank', [False, True])
def test_publication_reexpresses_components_without_changing_updates(tmp_path, family, train_bank):
    engine = engine_fixture(tmp_path, family)
    from natlang_neuralese.model.native_adapters import target_parameters
    from natlang_neuralese.tune.targets import module_map
    from natlang_neuralese.tune.learned.bank import new_bank, weight_svd_basis
    from natlang_neuralese.tune.learned.numbers import NumberFormat
    names, _ = module_map(sorted(target_parameters(engine.shared_model)))
    bank = new_bank(engine, weight_svd_basis(engine, sorted(names)[:6], 4), NumberFormat(.5))
    bank = bank.with_components([_component(bank, seed) for seed in (1, 2)], reason='fixture')
    trainer = object.__new__(OuterTrainer)
    trainer.bank, trainer.engine, trainer.owner = bank, engine, None
    trainer.recipe = OuterRecipe(train_basis=True, train_bank=train_bank)
    trainer.state = {'keys': {}}
    trainer.bank_leaves = _bank_leaves(bank, trainer.recipe)
    generator = torch.Generator().manual_seed(817)
    with torch.no_grad():
        for tensor in trainer.bank_leaves['basis'].values():
            tensor.mul_(1.4).add_(torch.randn(tensor.shape, generator=generator), alpha=.13)
        if train_bank:
            for factors in trainer.bank_leaves['factors_by_component'].values():
                for P, Q in factors.values():
                    P.add_(.02); Q.mul_(.9)
    expected = {}
    before = {key: value.clone() for key, value in trainer.bank_leaves['basis'].items()}
    for index, factors in trainer.bank_leaves['factors_by_component'].items():
        for name, (P, Q) in factors.items():
            U, V = before['left.' + name], before['right.' + name]
            expected[index, name] = (U @ P @ Q.transpose(-2, -1) @ V.transpose(-2, -1)).detach()
    published = trainer.publish_bank()
    published.basis.check()
    assert published.parent == bank.snapshot
    assert published.history[-1]['parent'] == bank.snapshot
    assert trainer.state['labels_stale'] is True
    assert trainer.state['published'] == [published.snapshot]
    for key, value in trainer.bank_leaves['basis'].items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    for index, component in enumerate(published.components):
        assert component.id == component_id(component.source, component.factors)
        for name, (P, Q) in component.factors.items():
            U, V = published.basis.left[name], published.basis.right[name]
            actual = U @ P @ Q.transpose(-2, -1) @ V.transpose(-2, -1)
            torch.testing.assert_close(actual, expected[index, name], rtol=2e-5, atol=2e-6)
    from natlang_neuralese.model.native_adapters import run_native
    from natlang_neuralese.model.shared_state import compile_graph, Invocation
    coefficients = torch.tensor([[.07] * len(bank.group_labels), [-.03] * len(bank.group_labels)])
    original = bank.compose_adapter(engine, bank.composed_cores([0, 1], coefficients,
        factors=trainer.bank_leaves['factors_by_component']))
    exported = published.compose_adapter(engine, published.composed_cores([0, 1], coefficients))
    plan = compile_graph([Invocation('answer', (4, 5, 6, 7))])
    model = engine.shared_model
    execute = lambda: model.logits(model(plan)['h_final']).detach()
    before_logits = run_native(engine, execute, [{'id': original.id, 'scale': 1.}],
        leaves={bank.basis_artifact: trainer.bank_leaves['basis']})
    after_logits = run_native(engine, execute, [{'id': exported.id, 'scale': 1.}])
    torch.testing.assert_close(after_logits, before_logits, rtol=1e-4, atol=1e-5)
    def coefficient_gradient(bank_value, artifact, basis_leaves=None, factors=None):
        live = coefficients.clone().requires_grad_(True)
        cores = bank_value.composed_cores([0, 1], live, factors=factors)
        leaves = {artifact.id: bank_value.adapter_tensors(cores)}
        if basis_leaves is not None:
            leaves[bank_value.basis_artifact] = basis_leaves
        def backward():
            objective = model.logits(model(plan)['h_final']).float().square().mean()
            return torch.autograd.grad(objective, live)[0].detach()
        return run_native(engine, backward, [{'id': artifact.id, 'scale': 1.}], leaves=leaves)
    original_gradient = coefficient_gradient(bank, original, trainer.bank_leaves['basis'],
        trainer.bank_leaves['factors_by_component'])
    published_gradient = coefficient_gradient(published, exported)
    assert original_gradient.abs().max() > 0
    torch.testing.assert_close(published_gradient, original_gradient, rtol=1e-4, atol=1e-5)
    engine.store.close()
