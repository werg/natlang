"""shared_parametrized_weights: Maple QAT weights built once per pass give the same loss and gradients as building
them at every use, through checkpointed isolated-sequence layers and one backward per pass."""
import torch

from natlang_neuralese.maple.maple_port import MaplePortBackbone
from natlang_neuralese.model.lfm2_port import ControlTokens, PortCache
from natlang_neuralese.train.backbone_policy import shared_parametrized_weights

from test_maple_model import WINDOW, _reference, _save  # noqa: F401  (WINDOW keeps the fixture module importable)


def _port(tmp_path):
    from natlang_neuralese.maple.model import load_maple
    from natlang_neuralese.maple.nested_train import Member, setup
    from natlang_neuralese.train.adapters import install_maple_qat, maple_qat_parameters

    reference, config = _reference()
    model = load_maple(_save(reference, config, tmp_path / "ckpt"), dtype=torch.float32)
    setup(model, [Member.parse("2x3", 4)], rank=2, private_rank=2, learn_scales=True, expert_scales=True)
    for p in model.parameters():
        p.requires_grad_(False)
    port = MaplePortBackbone(model, ControlTokens(open_id=94, close_id=95))
    install_maple_qat(port)
    named = maple_qat_parameters(port)
    torch.manual_seed(1)
    with torch.no_grad():  # move the latents off zero so the ternarization depends on them
        for name, p in named:
            if name.endswith(".dense"):
                p.add_(torch.randn_like(p) * 0.05)
    port.checkpoint_layers = True
    return port, named


def test_shared_weights_match_per_use(tmp_path):
    torch.set_num_threads(1)
    port, named = _port(tmp_path)
    dim = port.embedding_weight.shape[1]
    torch.manual_seed(2)
    prefix = torch.randn(1, 3, dim)
    fixed, replacements = torch.randn(1, 6, dim), torch.randn(1, 6, dim)

    def passes(next_pass=lambda: None):
        losses = []
        for scale in (1.0, 0.5):
            _, cache = port.run_layers(prefix, range(port.num_layers), PortCache.empty(port.num_layers))
            out = port.isolated_sequence(fixed * scale, replacements, cache, cutoff=2)
            loss = out["final"].square().mean() + out["history_shallow"].square().mean()
            loss.backward()
            next_pass()
            losses.append(float(loss))
        return losses, {n: p.grad.clone() for n, p in named if p.grad is not None}

    plain_losses, plain = passes()
    for _, p in named:
        p.grad = None
    with shared_parametrized_weights(port.hf) as next_pass:
        shared_losses, shared = passes(next_pass)
    assert plain_losses == shared_losses
    assert plain.keys() == shared.keys() and any(n.endswith(".dense") for n in plain)
    for name in plain:
        torch.testing.assert_close(shared[name], plain[name], atol=1e-6, rtol=1e-5, msg=name)
