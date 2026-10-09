"""Reader-side Neuralese input adaptation (owner 2026-10-09): identity at creation, trained by readers, carried in the
port config so engines restore it."""
import torch


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
    assert any(k.startswith("read_adapter.") for k in heads.state_dict())


def test_projection_anchor_schedule_decays_from_the_stage_origin():
    weight, steps, origin = 1.0, 100, 50
    schedule = lambda step: weight * max(0.0, 1.0 - (step - origin) / steps)
    assert schedule(50) == 1.0 and schedule(100) == 0.5 and schedule(150) == 0.0 and schedule(400) == 0.0
