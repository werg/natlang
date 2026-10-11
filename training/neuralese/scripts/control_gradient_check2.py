"""Gradient checks, part 2 (plans/mellum-port.md "Behaviour gate bug hunt"):

1. ``router``: central finite differences of the preserve loss along -g on FP32 routers (layers 5 and 20) at relative
   steps 1e-4 .. 1e-6 (below routing flips), against autograd's g·d.
2. ``descent``: away from the optimum. The student is perturbed with Gaussian noise (relative 11%, held KL ~0.4); on
   one record, one plain SGD step at lr 1e-4 .. 1e-1 must lower that record's preserve loss, and LionSR / MuonSR steps
   at small lr too.

    python control_gradient_check2.py --out gradcheck2.jsonl
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from natlang_neuralese.common.paths import resolve  # noqa: E402

MODEL = str(resolve('models', 'mellum21-12b-a2.5b-thinking'))
SUFFIXES = ('.self_attn.q_proj.weight', '.self_attn.k_proj.weight', '.self_attn.v_proj.weight',
            '.self_attn.o_proj.weight', '.mlp.experts.gate_up', '.mlp.experts.down', '.mlp.gate.weight')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--recipe', default='recipes/raw-recurrence-mellum-v12.json')
    p.add_argument('--record', type=int, default=7)
    p.add_argument('--noise', type=float, default=0.11)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    from natlang_neuralese.maple.model import set_dense_moe_kernel
    from natlang_neuralese.maple.qat_convert import row_scale
    from natlang_neuralese.maple.student import load_student
    from natlang_neuralese.train.optim import LionSR, MuonSR
    from natlang_neuralese.train.quantization import Preserve
    from natlang_neuralese.train.recipe import resolve_recipe_data

    spec = resolve_recipe_data(a.recipe)['quantization']
    model, _ = load_student(MODEL, precision='bf16', device='cuda')
    set_dense_moe_kernel('grouped')
    for layer in model.model.layers:
        layer.mlp.experts.make_latent(quantize=False)
    preserve = Preserve(spec['preserve'], 'core_text_warmup')
    log = open(a.out, 'a')
    params = dict(model.named_parameters())
    latents = {n: q for n, q in params.items() if n.startswith('model.layers.') and n.endswith(SUFFIXES)}

    def emit(row):
        log.write(json.dumps(row) + '\n'); log.flush()
        print(json.dumps(row), flush=True)

    def loss(grad=False):
        with torch.set_grad_enabled(grad):
            return preserve.loss(model, a.record, checkpoint_layers=True)[0]

    # 1. FP32 routers at small steps.
    for name in ('model.layers.5.mlp.gate.weight', 'model.layers.20.mlp.gate.weight'):
        for n, q in params.items():
            q.requires_grad_(n == name)
        q = params[name]
        q.grad = None
        loss(True).backward()
        g = q.grad.detach().clone()
        q.grad = None
        base = q.detach().clone()
        with torch.no_grad():
            centre = float(loss())
            for relative in (1e-4, 1e-5, 1e-6):
                d = -g * (relative * base.norm() / g.norm())
                q.copy_(base + d); plus = float(loss())
                q.copy_(base - d); minus = float(loss())
                q.copy_(base)
                emit({'control': 'router', 'tensor': name, 'relative_step': relative,
                      'predicted_change': float((g * d).sum()), 'finite_difference_change': (plus - minus) / 2,
                      'curvature_change': (plus + minus) / 2 - centre, 'loss': centre})

    # 2. Descent away from the optimum.
    generator = torch.Generator(device='cuda').manual_seed(0)
    with torch.no_grad():
        for q in latents.values():
            w = q.float()
            noise = torch.randn(w.shape, device=w.device, generator=generator)
            q.copy_((w + noise * (a.noise * w.norm() / noise.norm())).to(q.dtype))
    perturbed = {n: q.detach().clone() for n, q in latents.items()}
    for n, q in params.items():
        q.requires_grad_(n in latents)

    def restore():
        with torch.no_grad():
            for n, q in latents.items():
                q.copy_(perturbed[n])

    for name, lr in (('sgd', 1e-4), ('sgd', 1e-3), ('sgd', 1e-2), ('sgd', 1e-1), ('lionsr', 3e-6),
                     ('lionsr', 3e-5), ('muonsr', 7.5e-7), ('muonsr', 7.5e-6)):
        restore()
        before = float(loss())
        if name == 'sgd':
            loss(True).backward()
            with torch.no_grad():
                for q in latents.values():
                    q.copy_((q.float() - lr * q.grad.float()).to(q.dtype))
                    q.grad = None
        else:
            optimizer = (LionSR([{'params': [q], 'lr': lr, 'row_scale': row_scale(q), 'weight_decay': 0.0}
                                 for q in latents.values()], lr=lr) if name == 'lionsr'
                         else MuonSR([{'params': [q]} for q in latents.values()], lr=lr))
            optimizer.step_in_backward(gated=True)
            with optimizer.in_backward():
                loss(True).backward()
            for hook in optimizer._hooks:
                hook.remove()
        after = float(loss())
        change = sum(float((q.float() - perturbed[n].float()).norm() / perturbed[n].float().norm())
                     for n, q in latents.items()) / len(latents)
        emit({'control': 'descent_perturbed', 'noise': a.noise, 'optimizer': name, 'lr': lr,
              'loss_before': before, 'loss_after': after, 'weight_change_mean': change})


if __name__ == '__main__':
    main()
