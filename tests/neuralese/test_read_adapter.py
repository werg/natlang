"""Reader-side Neuralese input adaptation (owner 2026-10-09): identity at creation, trained by readers, carried in the
port config so engines restore it."""
import torch
import pytest


def test_read_adapter_starts_as_identity_and_learns(loaded):
    from natlang_neuralese.model.heads import PortHeads

    torch.manual_seed(2)
    heads = PortHeads(loaded[2], cutoff=6, max_length=8)
    payload = torch.randn(1, 5, loaded[2].embedding_weight.shape[1])
    before = heads.read_in(payload)
    adapter = heads.add_read_adapter()
    assert torch.equal(heads.read_in(payload), before)
    heads.read_in(payload).square().sum().backward()
    assert adapter.proj.weight.grad is not None and adapter.proj.weight.grad.abs().sum() > 0
    assert heads.port_config()["read_adapter"] == "full-residual-v1"
    assert heads.port_config()["read_adapter_norm_eps"] == adapter.norm.eps
    assert any(k.startswith("read_adapter.") for k in heads.state_dict())


def test_read_adapter_eps_is_restorable():
    from natlang_neuralese.model.heads import NeuraleseReadAdapter

    adapter = NeuraleseReadAdapter(4, eps=2e-6)
    assert adapter.norm.eps == 2e-6


def test_old_fixed_epsilon_checkpoint_restores_the_same_nonzero_transform():
    from natlang_neuralese.model.heads import NeuraleseReadAdapter, read_adapter_checkpoint_config

    torch.manual_seed(11)
    original = NeuraleseReadAdapter(4)
    with torch.no_grad():
        original.proj.weight.normal_(std=0.05)
        original.proj.bias.normal_(std=0.05)
    saved = {f"read_adapter.{name}": value.clone() for name, value in original.state_dict().items()}
    variant, eps = read_adapter_checkpoint_config({"read_adapter": "full-residual-v1"}, saved)
    restored = NeuraleseReadAdapter(4, eps=eps)
    restored.load_state_dict({name.removeprefix("read_adapter."): value for name, value in saved.items()})
    values = torch.randn(3, 4)
    assert variant == "full-residual-v1" and eps == 1e-5
    assert torch.equal(original(values), restored(values))


@pytest.mark.parametrize("metadata,head_state", [
    ({"read_adapter": "unknown"}, {
        "read_adapter.norm.weight": torch.ones(4),
        "read_adapter.proj.weight": torch.zeros(4, 4),
        "read_adapter.proj.bias": torch.zeros(4),
    }),
    ({"read_adapter": "full-residual-v1", "read_adapter_norm_eps": float("nan")}, {
        "read_adapter.norm.weight": torch.ones(4),
        "read_adapter.proj.weight": torch.zeros(4, 4),
        "read_adapter.proj.bias": torch.zeros(4),
    }),
    ({"read_adapter": "full-residual-v1"}, {"read_adapter.norm.weight": torch.ones(4)}),
    ({"read_adapter_norm_eps": 1e-5}, {}),
])
def test_reader_adapter_checkpoint_metadata_is_strict(metadata, head_state):
    from natlang_neuralese.model.heads import read_adapter_checkpoint_config

    with pytest.raises(ValueError):
        read_adapter_checkpoint_config(metadata, head_state)


def test_projection_anchor_schedule_decays_from_the_stage_origin():
    weight, steps, origin = 1.0, 100, 50
    schedule = lambda step: weight * max(0.0, 1.0 - (step - origin) / steps)
    assert schedule(50) == 1.0 and schedule(100) == 0.5 and schedule(150) == 0.0 and schedule(400) == 0.0
