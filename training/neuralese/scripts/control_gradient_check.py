"""Gradient checks of the preserve loss on BF16 Mellum (plans/mellum-port.md "Behaviour gate bug hunt"): every
optimizer step of the null tests raised the loss of the very record it stepped on, so test the gradient itself.

1. ``repeat``: the same record's loss twice (determinism), with and without grad, with and without layer checkpointing.
2. ``paths``: the gradient of a few tensors with layer checkpointing on and off, and with the loop and grouped MoE.
3. ``directional``: for single tensors (a router in FP32, an attention and an expert matrix in BF16), central finite
   differences of the loss along the negative gradient, against autograd's g·d.

    python control_gradient_check.py --out gradcheck.jsonl
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL = '/home/werg/data/models/mellum21-12b-a2.5b-thinking'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--recipe', default='recipes/raw-recurrence-mellum-v12.json')
    p.add_argument('--record', type=int, default=7)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    from natlang_neuralese.maple.model import set_dense_moe_kernel
    from natlang_neuralese.maple.student import load_student
    from natlang_neuralese.train.quantization import Preserve
    from natlang_neuralese.train.recipe import resolve_recipe_data

    spec = resolve_recipe_data(a.recipe)['quantization']
    model, _ = load_student(MODEL, precision='bf16', device='cuda')
    for layer in model.model.layers:
        layer.mlp.experts.make_latent(quantize=False)
    preserve = Preserve(spec['preserve'], 'core_text_warmup')
    log = open(a.out, 'a')
    probes = ['model.layers.5.mlp.gate.weight', 'model.layers.5.self_attn.q_proj.weight',
              'model.layers.5.mlp.experts.down', 'model.layers.20.self_attn.o_proj.weight']
    params = dict(model.named_parameters())
    for name, q in params.items():
        q.requires_grad_(name in probes)

    def emit(row):
        log.write(json.dumps(row) + '\n'); log.flush()
        print(json.dumps(row), flush=True)

    def loss(checkpoint=True, grad=False):
        with torch.set_grad_enabled(grad):
            return preserve.loss(model, a.record, checkpoint_layers=checkpoint)[0]

    def gradients(checkpoint, kernel):
        set_dense_moe_kernel(kernel)
        for name in probes:
            params[name].grad = None
        loss(checkpoint, True).backward()
        return {name: params[name].grad.detach().float().clone() for name in probes}

    set_dense_moe_kernel('grouped')
    emit({'control': 'repeat', 'no_grad_ckpt': [float(loss()) for _ in range(2)],
          'grad_ckpt': float(loss(True, True)), 'no_grad_plain': float(loss(False)),
          'grad_plain': float(loss(False, True))})
    reference = gradients(True, 'grouped')
    for checkpoint, kernel in ((False, 'grouped'), (True, 'loop'), (True, 'grouped')):
        other = gradients(checkpoint, kernel)
        emit({'control': 'paths', 'checkpoint': checkpoint, 'kernel': kernel,
              **{name: {'relative_difference': float((other[name] - reference[name]).norm() /
                                                     reference[name].norm().clamp_min(1e-30)),
                        'norm': float(other[name].norm())} for name in probes}})
    set_dense_moe_kernel('grouped')
    with torch.no_grad():
        for name in probes:
            q, g = params[name], reference[name]
            base = q.detach().clone()
            for relative in (1e-3, 1e-2):
                step = relative * base.float().norm() / g.norm()
                direction = -g * step  # a descent step along the negative gradient
                q.copy_((base.float() + direction).to(q.dtype))
                plus = float(loss())
                q.copy_((base.float() - direction).to(q.dtype))
                minus = float(loss())
                q.copy_(base)
                centre = float(loss())
                applied_plus = None
                emit({'control': 'directional', 'tensor': name, 'dtype': str(q.dtype), 'relative_step': relative,
                      'predicted_change': float((g * direction).sum()), 'finite_difference_change': (plus - minus) / 2,
                      'loss_descent_step': plus, 'loss_ascent_step': minus, 'loss': centre})


if __name__ == '__main__':
    main()
