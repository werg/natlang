"""LionSR latents stepped inside one combined backward: the same update as separate backwards plus one step."""
import json

import pytest
import torch
from torch import nn
from torch.nn.utils import parametrize
from torch.utils.checkpoint import checkpoint

from natlang_neuralese.train import optim
from natlang_neuralese.train.optim import LionSR, PortMuonAdamW
from natlang_neuralese.train.trajectory_state import clip_finite_gradients, gradient_norm, trajectory_optimizer

needs_muon = pytest.mark.skipif(not hasattr(torch.optim, 'Muon'), reason='needs torch.optim.Muon')


def tiny_student():
    """The text warm-up tests' tiny LFM student (tests/neuralese/test_text_warmup.py), built here: importing a test
    module from another one re-registers its conftest and loses the session fixtures."""
    try:
        from transformers import Lfm2Config, Lfm2ForCausalLM
    except ImportError:
        pytest.skip('installed transformers does not provide the tiny LFM2 model')
    from natlang_neuralese.model.heads import PortHeads
    from natlang_neuralese.model.lfm2_port import ControlTokens, PortBackbone
    torch.manual_seed(21)
    config = Lfm2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                        num_attention_heads=4, num_key_value_heads=2, block_multiple_of=8,
                        block_auto_adjust_ff_dim=False,
                        layer_types=['conv', 'full_attention', 'conv', 'full_attention'])
    model = Lfm2ForCausalLM(config).eval()
    backbone = PortBackbone(model, ControlTokens(62, 63), fast=False)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    heads = PortHeads(backbone, cutoff=2, max_length=8, profile='latent-sketch-v2').eval()
    with torch.no_grad():
        heads.feedback.correction.weight.normal_(0, .02)
    return backbone, heads


def deterministic_rounding(monkeypatch):
    """Stochastic rounding draws from the global RNG in stepping order, which differs between the paths (parameter
    order vs backward order). Fix the rounding offset so both paths write identical BF16 values."""
    real = torch.randint_like

    def rows(p, g, m, scale, lr, beta1, beta2, decay):
        torch.randint_like = lambda t, low, high: torch.full_like(t, 1 << 15)
        try:
            return optim._lion_rows(p, g, m, scale, lr, beta1, beta2, decay)
        finally:
            torch.randint_like = real
    monkeypatch.setattr(optim, '_fused_lion_rows', rows)


class Twice(nn.Module):
    """A stand-in for a precision point's parametrization (PrecisionSTE): a differentiable function of the latent."""
    def forward(self, weight):
        return weight * 1.5


class Tiny(nn.Module):
    def __init__(self, dtype):
        super().__init__()
        torch.manual_seed(3)
        self.embed = nn.Parameter((torch.randn(24, 16) * .5).to(dtype))  # used at the input and the tied readout
        self.layers = nn.ModuleList(nn.Linear(16, 16, bias=False).to(dtype) for _ in range(3))
        parametrize.register_parametrization(self.layers[1], 'weight', Twice(), unsafe=True)
        self.norm = nn.Parameter(torch.ones(16, dtype=dtype))
        self.head = nn.Parameter((torch.randn(16, 16) * .3).to(dtype))

    def forward(self, ids, *, checkpointed=True):
        h = self.embed[ids]
        for layer in self.layers:  # non-reentrant per-layer checkpointing, as the backbones run it
            h = checkpoint(lambda x, layer=layer: x + torch.tanh(layer(x)), h, use_reentrant=False) if checkpointed \
                else h + torch.tanh(layer(h))
        return (h * self.norm) @ self.head @ self.embed.t()

    def named(self):
        latents = [('backbone.embed', self.embed), ('backbone.layers.0.weight', self.layers[0].weight),
                   ('backbone.layers.1.parametrizations.weight.original',
                    self.layers[1].parametrizations.weight.original),
                   ('backbone.layers.2.weight', self.layers[2].weight)]
        return [('heads.head', self.head), ('backbone.norm', self.norm)], latents


def terms(model, step):
    """Three weighted objective passes of one update (two sequence passes and a preserve-like stream)."""
    g = torch.Generator().manual_seed(step)
    a, b = torch.randint(0, 24, (2, 9), generator=g), torch.randint(0, 24, (1, 12), generator=g)
    first = nn.functional.cross_entropy(model(a[:, :-1]).flatten(0, 1).float(), a[:, 1:].flatten())
    second = nn.functional.cross_entropy(model(a[:, 1:]).flatten(0, 1).float(), a[:, :-1].flatten())
    third = model(b).float().log_softmax(-1).mean().neg()
    return [first / 2, second / 2, .5 * third]


def build(dtype, in_backward):
    model = Tiny(dtype)
    named, latents = model.named()
    optimizer = PortMuonAdamW(named, lr=1e-2, vocab_size=24,
                              latent=[(n, q, 3e-3, torch.full((q.shape[0], 1), .7)) for n, q in latents])
    if in_backward:
        optimizer.latent.step_in_backward(gated=True)
    return model, optimizer, [q for _, q in named + latents]


def run(dtype, in_backward, steps, scale):
    model, optimizer, parameters = build(dtype, in_backward)
    norms = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        losses = [scale * t for t in terms(model, step)]
        if in_backward:
            total = losses[0] + losses[1] + losses[2]
            with optimizer.latent.in_backward() as armed:
                total.backward()
            assert armed.started and len(armed.stepped) == 4
            assert all(q.grad is None for _, q, *_ in [(n, q) for n, q in model.named()[1]])  # no latent gradient
            consumed = armed.norm()
            norms.append(float(clip_finite_gradients(parameters, consumed_norm=consumed)))
        else:
            for loss in losses:
                loss.backward()
            norms.append(float(clip_finite_gradients(parameters)))
        optimizer.step()
    state = {n: q.detach().float().clone() for n, q in model.named()[0] + model.named()[1]}
    momentum = {n: optimizer.latent.state[q]['momentum'].float() for n, q in model.named()[1]}
    return state, momentum, norms


@needs_muon
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_one_combined_backward_with_in_backward_steps_matches_separate_backwards_and_one_step(dtype, monkeypatch):
    deterministic_rounding(monkeypatch)
    # Gradient norm below the clip threshold: the two paths are the same update.
    old, old_momentum, old_norms = run(dtype, False, 5, scale=.02)
    new, new_momentum, new_norms = run(dtype, True, 5, scale=.02)
    exact = dtype == torch.float32
    # The engine sums a latent's gradient contributions in its own order; in BF16 that rounds differently.
    assert max(old_norms) < 1 and old_norms == pytest.approx(new_norms, rel=1e-3 if exact else 1e-2)
    for name in old:
        if name.startswith('heads.') or name == 'backbone.norm':
            # Muon/AdamW step after the backward from gradients summed in another order: float rounding only.
            torch.testing.assert_close(new[name], old[name], rtol=1e-4 if exact else 2e-2, atol=1e-6 if exact else 1e-2)
        elif exact:
            torch.testing.assert_close(new[name], old[name], rtol=0, atol=0)  # the same sign steps
        else:
            # BF16: a gradient element within rounding of zero may take the other sign; one step is 3e-3 x 0.7.
            differ = (new[name] - old[name]).abs()
            assert float((differ > 0).float().mean()) <= .02 and float(differ.max()) <= 2 * 5 * 3e-3 * .7 + 1e-2
    for key in old_momentum:
        torch.testing.assert_close(new_momentum[key], old_momentum[key], rtol=2e-2 if exact else 1e-1, atol=1e-6 if exact else 1e-3)


@needs_muon
def test_clipping_counts_stepped_latents_in_the_global_norm_and_scales_only_the_rest(monkeypatch):
    deterministic_rounding(monkeypatch)
    # Above the threshold: heads and norms get the same clipped step; the latents take Lion's sign step of the
    # unclipped gradient (identical on a first step, whose momentum is zero).
    old, _, old_norms = run(torch.float32, False, 1, scale=50.)
    new, _, new_norms = run(torch.float32, True, 1, scale=50.)
    assert old_norms[0] > 1 and new_norms[0] == pytest.approx(old_norms[0], rel=1e-4)
    for name in old:
        torch.testing.assert_close(new[name], old[name], rtol=1e-4, atol=1e-6)


@needs_muon
def test_a_second_backward_inside_one_update_refuses_and_ungated_backwards_accumulate(monkeypatch):
    deterministic_rounding(monkeypatch)
    model, optimizer, _ = build(torch.float32, True)
    first, second, _ = terms(model, 0)
    first.backward()  # outside in_backward: an ordinary accumulation for step()
    assert model.embed.grad is not None
    optimizer.zero_grad(set_to_none=True)
    with optimizer.latent.in_backward() as armed:
        terms(model, 0)[0].backward()
        # A graph built before the first step is caught by autograd's version check; one built after it, here.
        with pytest.raises(RuntimeError, match='second gradient'):
            terms(model, 1)[0].backward()
    assert armed.started
    with pytest.raises(RuntimeError, match='needs step_in_backward'):
        LionSR([nn.Parameter(torch.zeros(2, 2))], lr=1.).in_backward()


@needs_muon
def test_in_backward_hooks_follow_staged_learning_rates_and_reloaded_groups(monkeypatch):
    deterministic_rounding(monkeypatch)
    model, optimizer, _ = build(torch.float32, True)
    again = PortMuonAdamW(model.named()[0], lr=1e-2, vocab_size=24,
                          latent=[(n, q, 3e-3, torch.full((q.shape[0], 1), .7)) for n, q in model.named()[1]])
    optimizer.load_state_dict(again.state_dict())  # replaces the LionSR group dictionaries
    for group in optimizer.param_groups:
        group['lr'] = 0.
    before = model.embed.detach().clone()
    total = sum(terms(model, 0))
    with optimizer.latent.in_backward():
        total.backward()
    assert torch.equal(model.embed.detach(), before)  # the hook read the staged lr of the live group


@needs_muon
def test_trajectory_optimizer_routes_the_latent_partition_to_lionsr():
    model = Tiny(torch.bfloat16)
    _, latents = model.named()
    names = [n.removeprefix('backbone.') for n, _ in latents] + ['norm']
    lora = [q for _, q in latents] + [model.norm]
    lion = optim.latent_partition(latents + [('backbone.norm', model.norm)], 3e-4)
    assert [n for n, *_ in lion] == [n for n, _ in latents]  # weight matrices only; norms stay on AdamW
    params = {'soft': nn.Parameter(torch.zeros(3, 16))}
    optimizer = trajectory_optimizer('muon', params, lora, [model.head], vocab_size=24, lr=1e-3, lora_lr=1e-5,
                                     heads_lr=1e-4, lora_names=names, lion=lion)
    kinds = {row['name']: row['optimizer'] for row in optimizer.schema}
    assert {kinds[n] for n, _ in latents} == {'lion'} and kinds['backbone.norm'] == 'adamw'
    with pytest.raises(ValueError, match='muon policy'):
        trajectory_optimizer('adamw', params, lora, [model.head], vocab_size=24, lr=1e-3, lora_lr=1e-5,
                             heads_lr=1e-4, lora_names=names, lion=lion)


def test_in_backward_memory_floor_has_no_latent_gradient_buffer_and_geometry_sums_the_passes():
    from types import SimpleNamespace
    from natlang_neuralese.train.text_warmup import _warmup_memory_kind, _warmup_update_floor_bytes
    from natlang_neuralese.train.memory_policy import text_warmup_update_geometry_bytes
    big, small, head = (nn.Parameter(torch.zeros(64, 32, dtype=torch.bfloat16)),
                        nn.Parameter(torch.zeros(8, 32, dtype=torch.bfloat16)), nn.Parameter(torch.zeros(4, 4)))
    lion = LionSR([{'params': [big]}, {'params': [small]}], lr=1.)
    optimizer = SimpleNamespace(muon=None, auxiliary=None, latent=lion)
    named = [('backbone.big', big), ('backbone.small', small), ('heads.content.proj.w', head)]
    separate = _warmup_update_floor_bytes(named, optimizer, bootstrap=False)
    combined = _warmup_update_floor_bytes(named, optimizer, bootstrap=False, in_backward=True)
    momentum = (big.numel() + small.numel()) * 2
    assert separate == (big.numel() + small.numel()) * 2 + head.numel() * 4 + momentum
    assert combined == 4 * big.numel() * 2 + head.numel() * 4 + momentum
    assert _warmup_memory_kind(1, 2, single_backward=True).endswith(':single')
    layout = dict(layers=4, width=32, intermediate=64, kv_width=16, dtype_bytes=2, checkpointed=True)
    one = text_warmup_update_geometry_bytes(8, 6, 2, 1, layout, {**layout, 'layers': 2}, cutoff=2, vocab_size=64)
    both = text_warmup_update_geometry_bytes(8, 6, 2, 1, layout, {**layout, 'layers': 2}, cutoff=2,
                                             vocab_size=64, single_backward=True)
    single = text_warmup_update_geometry_bytes(8, 6, 1, 1, layout, {**layout, 'layers': 2}, cutoff=2, vocab_size=64)
    assert both > one >= single  # both passes' graphs live together instead of one after the other


@needs_muon
def test_warmup_main_in_backward_update_matches_separate_backwards(tmp_path, monkeypatch):
    """The text warm-up end to end on a tiny LFM student (latent policy, LionSR latents, multi-pass backbone phase):
    --latent-step-in-backward writes the same parameters as the separate-backward path."""
    from types import SimpleNamespace
    from natlang_neuralese.train import text_warmup
    deterministic_rounding(monkeypatch)
    real_clip = text_warmup.clip_finite_gradients
    # The paths differ only in clipping the latents; keep the clip inactive so the comparison is of the restructure.
    monkeypatch.setattr(text_warmup, 'clip_finite_gradients',
                        lambda parameters, maximum=1., consumed_norm=None: real_clip(parameters, 1e9, consumed_norm))

    def load(*_args):
        backbone, heads = tiny_student()
        return SimpleNamespace(backbone=backbone, heads=heads, tokenizer=None, _tokens=lambda _text: [9, 3, 5, 8]), None
    monkeypatch.setattr(text_warmup, 'load_initial', load)
    heads_path = tmp_path / 'heads.pt'
    torch.save({}, heads_path)
    records = tmp_path / 'records.jsonl'
    records.write_text('')
    text = tmp_path / 'text.jsonl'
    text.write_text('\n'.join(json.dumps({'text': s, 'split': split, 'source_groups': [s]})
                              for s, split in [('train', 'train'), ('held', 'test')]) + '\n')

    def run(out, *extra):
        torch.manual_seed(0)
        text_warmup.main(['--heads', str(heads_path), '--records', str(records), '--text-data', str(text),
                          '--out', str(tmp_path / out), '--device', 'cpu', '--steps', '4', '--tokens', '8',
                          '--prefix-tokens', '2', '--batch', '1', '--eval-batch', '1', '--held-documents', '1',
                          '--eval-every', '1', '--checkpoint-every', '1', '--optimizer', 'muon',
                          '--backbone-training', 'latent', '--latent-optimizer', 'lionsr', '--qat-latent-lr', '1e-3',
                          '--projection-patience', '1', '--projection-min-evals', '2',
                          '--projection-min-improvement', '1', '--backbone-ramp-evals', '1', '--pass-ramp-evals', '1',
                          *extra])
        rows = [json.loads(line) for line in (tmp_path / out / 'train.jsonl').read_text().splitlines()]
        return torch.load(tmp_path / out / 'checkpoint.pt', weights_only=False), rows
    old, old_rows = run('separate')
    new, new_rows = run('single', '--latent-step-in-backward')
    assert [r['phase'] for r in new_rows] == [r['phase'] for r in old_rows]
    trained = [r for r in new_rows if r['phase'] != 'projection_only']
    assert trained
    assert [r['schedule']['sequence_passes'] for r in new_rows] == [r['schedule']['sequence_passes'] for r in old_rows]
    assert max(r['schedule']['sequence_passes'] for r in trained) > 1  # several passes in one backward
    assert all(r['updates']['backbone'] for r in trained)
    assert all(r['latents_stepped_in_backward'] > 0 for r in trained) and 'latents_stepped_in_backward' not in old_rows[-1]
    assert all(r['latents_stepped_in_backward'] == 0 for r in new_rows if r['phase'] == 'projection_only')
    assert [r['backbone_gradient_norm'] for r in new_rows] == pytest.approx(
        [r['backbone_gradient_norm'] for r in old_rows], rel=1e-4)
    moved = 0
    for name, value in old['student_parameters'].items():
        if name.startswith('backbone.') and value.ndim >= 2:
            torch.testing.assert_close(new['student_parameters'][name], value, rtol=0, atol=0)
        else:
            torch.testing.assert_close(new['student_parameters'][name], value, rtol=1e-4, atol=1e-6)
        moved += name.startswith('backbone.')
    assert moved


@needs_muon
def test_layer_staging_steps_each_layer_once_from_the_top_and_matches_separate_backwards(monkeypatch):
    """Two passes through a real port (tiny LFM): under LayerStaging the latents step layer by layer, deepest first,
    none during the losses' own backward (so no layer's partial gradient waits for another pass), and the result
    equals separate backwards plus one step."""
    from natlang_neuralese.model.layer_staging import LayerStaging
    deterministic_rounding(monkeypatch)

    def setup():
        backbone, _ = tiny_student()
        named = [('backbone.' + n, q) for n, q in backbone.hf.named_parameters() if n.startswith('model.layers.')]
        for _, q in named:
            q.requires_grad_(True)
        lion = optim.latent_partition(named, 1e-2)
        optimizer = LionSR([{'params': [q], 'lr': lr, 'row_scale': s, 'weight_decay': 0., 'name': n}
                            for n, q, lr, s in lion], lr=1e-2)
        return backbone, optimizer, named

    def losses(backbone, step):
        g = torch.Generator().manual_seed(step)
        ids = torch.randint(0, 60, (2, 7), generator=g)
        first = backbone.forward_ids(ids, logits=False)['h_final'].square().mean()
        second = backbone.forward_embeds(backbone.embed(ids.flip(1)), logits=False)['h_final'].tanh().sum() * .01
        return [first, second]

    reference, reference_optimizer, reference_named = setup()
    for step in range(3):
        for loss in losses(reference, step):
            loss.backward()
        reference_optimizer.step()
        reference_optimizer.zero_grad(set_to_none=True)

    backbone, optimizer, named = setup()
    optimizer.step_in_backward(gated=True)
    order, phase = [], ['loss']
    real_update = optimizer._update
    monkeypatch.setattr(optimizer, '_update', lambda p, group: (order.append((phase[0], group['name'])),
                                                                 real_update(p, group)))
    for step in range(3):
        order.clear()
        with LayerStaging() as tape:
            total = sum(losses(backbone, step))
            with optimizer.in_backward() as armed:
                total.backward()
                phase[0] = 'layers'
                tape.backward()
                phase[0] = 'loss'
        assert len(armed.stepped) == sum(len(g['params']) for g in optimizer.param_groups) == 26
        assert {p for p, _ in order} == {'layers'}
        layers = [int(name.split('.')[3]) for _, name in order]
        assert layers == sorted(layers, reverse=True)  # deepest layer first, each layer's latents together
    for (name, value), (_, expected) in zip(named, reference_named):
        torch.testing.assert_close(value.detach(), expected.detach(), rtol=0, atol=0, msg=name)
