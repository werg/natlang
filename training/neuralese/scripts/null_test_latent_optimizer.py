"""Null tests of the backbone-latent update path (plans/mellum-port.md "Behaviour gate bug hunt"): BF16 Mellum's
attention, expert and router matrices as latents stepped by the trainer's optimizer (LionSR in backward, per-row
ternary-scale step, stochastic rounding: train/text_warmup.py --latent-optimizer lionsr), driven by

- ``zero``: the preserve loss times 0 (any drift is a mechanical bug: optimizer, SR, latent path);
- ``preserve``: the preserve stream alone (self-distillation to BF16 Mellum's own top-64; KL must stay ~0).

Every ``--eval-every`` updates: held KL to the BF16 teacher (the behaviour gate's held rows, prompt/response split)
and the latents' relative change from the start.

    python null_test_latent_optimizer.py --mode preserve --steps 100 --lr 3e-4 --out preserve.jsonl
    python null_test_latent_optimizer.py --mode preserve --optimizer muonsr --lr 7.5e-6 --out muon.jsonl
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from natlang_neuralese.common.paths import resolve  # noqa: E402

MODEL = str(resolve('models', 'mellum21-12b-a2.5b-thinking'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--recipe', default='recipes/raw-recurrence-mellum-v12.json')
    p.add_argument('--mode', choices=('zero', 'preserve'), required=True)
    p.add_argument('--optimizer', choices=('lionsr', 'muonsr', 'adamw'), default='lionsr')
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--eval-every', type=int, default=10)
    p.add_argument('--held-rows', type=int, default=16)
    p.add_argument('--sweep', help='optimizer:lr,... run one after another from the same start')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    from natlang_neuralese.maple.model import set_dense_moe_kernel
    from natlang_neuralese.maple.qat_convert import row_scale, weighted_topk_kl
    from natlang_neuralese.maple.student import load_student
    from natlang_neuralese.train.optim import LionSR, MuonSR
    from natlang_neuralese.train.quantization import Preserve
    from natlang_neuralese.train.recipe import resolve_recipe_data

    spec = resolve_recipe_data(a.recipe)['quantization']
    model, _ = load_student(MODEL, precision='bf16', device='cuda')
    set_dense_moe_kernel('grouped')
    model.model.checkpoint_layers = True
    latents = []
    for layer in model.model.layers:
        layer.mlp.experts.make_latent(quantize=False)
    suffixes = ('.self_attn.q_proj.weight', '.self_attn.k_proj.weight', '.self_attn.v_proj.weight',
                '.self_attn.o_proj.weight', '.mlp.experts.gate_up', '.mlp.experts.down', '.mlp.gate.weight')
    for name, q in model.named_parameters():
        selected = name.startswith('model.layers.') and name.endswith(suffixes)
        q.requires_grad_(selected)
        if selected:
            latents.append((name, q))
    start = {name: q.detach().clone() for name, q in latents}
    preserve = Preserve(spec['preserve'], 'core_text_warmup')
    held = preserve.held
    log = open(a.out, 'a')
    runs = [(a.optimizer, a.lr)] if not a.sweep else [
        (item.split(':')[0], float(item.split(':')[1])) for item in a.sweep.split(',')]

    def make(name, lr):
        if name == 'lionsr':
            optimizer = LionSR([{'params': [q], 'lr': lr, 'row_scale': row_scale(q), 'weight_decay': 0.0}
                                for _, q in latents], lr=lr)
        elif name == 'muonsr':
            optimizer = MuonSR([{'params': [q]} for _, q in latents], lr=lr)
        else:
            return torch.optim.AdamW([q for _, q in latents], lr=lr, weight_decay=0.0)
        optimizer.step_in_backward(gated=True)
        return optimizer

    for name, lr in runs:
        with torch.no_grad():
            for n, q in latents:
                q.copy_(start[n])
        optimizer = make(name, lr)

        def drift(step, extra):
            sums, rows = {}, range(min(a.held_rows, len(held['ids'])))
            with torch.no_grad():
                for i in rows:
                    scores = weighted_topk_kl(model, held['ids'][i][None].long().cuda(), held['top_ids'][i],
                                              held['top_logp'][i], held['prompt_len'][i], 0.25)
                    for key, value in scores.items():
                        sums[key] = sums.get(key, 0.0) + float(value)
                change = [float((q.float() - start[n].float()).norm() / start[n].float().norm()) for n, q in latents]
            row = {'step': step, 'mode': a.mode, 'optimizer': name, 'lr': lr,
                   **{'held_' + k.replace('loss_', ''): v / len(rows) for k, v in sums.items()},
                   'weight_change_mean': sum(change) / len(change), 'weight_change_max': max(change), **extra}
            log.write(json.dumps(row) + '\n'); log.flush()
            print(json.dumps(row), flush=True)

        drift(0, {})
        started, update_seconds = time.time(), 0.0
        for step in range(1, a.steps + 1):
            loss, metrics = preserve.loss(model, step, checkpoint_layers=True)
            if a.mode == 'zero':
                loss = loss * 0.0
            torch.cuda.synchronize()
            tick = time.time()
            if name in ('lionsr', 'muonsr'):
                with optimizer.in_backward() as armed:
                    loss.backward()
                norm = float(armed.norm())
            else:
                loss.backward()
                norm = float(torch.sqrt(sum(q.grad.float().square().sum() for _, q in latents if q.grad is not None)))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            update_seconds += time.time() - tick
            if step % a.eval_every == 0:
                drift(step, {'preserve_kl': metrics['preserve_kl'], 'grad_norm': norm,
                             'seconds_per_step': (time.time() - started) / step,
                             'backward_and_step_seconds': update_seconds / step})
        for hook in getattr(optimizer, '_hooks', []):
            hook.remove()
        del optimizer
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
