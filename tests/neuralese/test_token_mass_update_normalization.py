from types import SimpleNamespace

import pytest
import torch

from natlang_neuralese.train import native_state_sft as trainer


def test_token_mass_is_additive_across_packing_and_update_boundaries(monkeypatch):
    class Model(torch.nn.Module):
        native_cotangent_accumulator_backend = 'plain'
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

    class Optimizer(torch.optim.SGD):
        latent = None

    def fake_loss(model, batch, **kwargs):
        return model.weight * sum(item.value for item in batch) / len(batch), {}

    monkeypatch.setattr(trainer, 'packed_graph_loss', fake_loss)
    monkeypatch.setattr(trainer, 'merge_update_details', lambda parts, scored: {})
    monkeypatch.setattr(trainer, 'sparse_row_updates', lambda model, before: {})
    def example(value, mode='token'):
        return SimpleNamespace(value=value, loss_normalization=mode, target_weights=(1,), decisions=())
    def run(groups):
        model = Model(); optimizer = Optimizer(model.parameters(), lr=0.0)
        trainer.native_training_update(model, optimizer, groups[-1], accumulate=groups[:-1])
        return model.weight.grad.item()
    rows = [example(1), example(2), example(3)]
    assert run([rows]) == pytest.approx(6 / 64)
    assert run([rows[:1], rows[1:]]) == pytest.approx(6 / 64)
    assert run([rows[:1]]) + run([rows[1:]]) == pytest.approx(6 / 64)
    legacy = [example(1, 'graph'), example(2, 'graph'), example(3, 'graph')]
    assert run([legacy[:1], legacy[1:]]) == pytest.approx(2.0)
    with pytest.raises(ValueError, match='cannot mix'):
        run([[rows[0], legacy[0]]])
