"""Controls for the behaviour-drift null tests (plans/mellum-port.md "Behaviour gate bug hunt"), on BF16 Mellum:

1. ``noise``: isotropic Gaussian perturbations with a given per-tensor relative norm ‖ΔW‖/‖W‖ (all latents, or one
   group: attention, experts, router), then the held KL the null tests measure. Random noise of the size a run moved
   the weights tells whether its drift is noise-like or structured against the objective.
2. ``align``: on preserve training records, the base model's greedy agreement with the teacher's top-1 at position
   offsets -1, 0, +1 (the teacher rows must line up at offset 0) and the KL there.
3. ``descent``: one fixed preserve record, one optimizer step (LionSR, MuonSR, plain SGD) at a small lr: the same
   record's preserve loss must go down.

    python control_drift_sensitivity.py --out controls.jsonl
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from natlang_neuralese.common.paths import resolve  # noqa: E402

MODEL = str(resolve('models', 'mellum21-12b-a2.5b-thinking'))
SUFFIXES = {'attention': ('.self_attn.q_proj.weight', '.self_attn.k_proj.weight', '.self_attn.v_proj.weight',
                          '.self_attn.o_proj.weight'),
            'experts': ('.mlp.experts.gate_up', '.mlp.experts.down'), 'router': ('.mlp.gate.weight',)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--recipe', default='recipes/raw-recurrence-mellum-v12.json')
    p.add_argument('--held-rows', type=int, default=16)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    from natlang_neuralese.maple.generation_gate import held_kl
    from natlang_neuralese.maple.model import set_dense_moe_kernel
    from natlang_neuralese.maple.qat_convert import row_scale, weighted_topk_kl
    from natlang_neuralese.maple.student import load_student
    from natlang_neuralese.train.optim import LionSR, MuonSR
    from natlang_neuralese.train.quantization import Preserve
    from natlang_neuralese.train.recipe import resolve_recipe_data

    spec = resolve_recipe_data(a.recipe)['quantization']
    model, _ = load_student(MODEL, precision='bf16', device='cuda')
    set_dense_moe_kernel('grouped')
    for layer in model.model.layers:
        layer.mlp.experts.make_latent(quantize=False)
    latents = {name: q for name, q in model.named_parameters()
               if name.startswith('model.layers.') and name.endswith(sum(SUFFIXES.values(), ()))}
    original = {name: q.detach().clone() for name, q in latents.items()}
    preserve = Preserve(spec['preserve'], 'core_text_warmup')
    log = open(a.out, 'a')

    def emit(row):
        log.write(json.dumps(row) + '\n'); log.flush()
        print(json.dumps(row), flush=True)

    def restore():
        with torch.no_grad():
            for name, q in latents.items():
                q.copy_(original[name])

    # 1. Equal-norm random perturbations.
    emit({'control': 'noise', 'group': 'none', 'relative': 0.0, **held_kl(model, preserve.held, a.held_rows)})
    generator = torch.Generator(device='cuda').manual_seed(0)
    for group, relative in (('all', 0.001), ('all', 0.0037), ('all', 0.11), ('attention', 0.0037),
                            ('experts', 0.0037), ('router', 0.0037)):
        with torch.no_grad():
            for name, q in latents.items():
                if group != 'all' and not name.endswith(SUFFIXES[group]):
                    continue
                w = original[name].float()
                noise = torch.randn(w.shape, device=w.device, generator=generator)
                q.copy_((w + noise * (relative * w.norm() / noise.norm())).to(q.dtype))
        emit({'control': 'noise', 'group': group, 'relative': relative,
              **held_kl(model, preserve.held, a.held_rows)})
        restore()

    # 2. Teacher rows against student positions on training records.
    with torch.no_grad():
        for index in range(3):
            record = preserve.train
            ids = record['ids'][index][:preserve.max_tokens].long().cuda()
            top = record['top_ids'][index][:len(ids) - 1].long().cuda()
            logits = model.lm_head(model.model(input_ids=ids[None]).last_hidden_state[0, :-1]).float()
            greedy = logits.argmax(-1)
            row = {'control': 'align', 'record': index, 'tokens': int(len(ids)),
                   'prompt_len': int(record['prompt_len'][index])}
            for offset in (-1, 0, 1):
                if offset < 0:
                    match = (greedy[-offset:] == top[:offset, 0]).float().mean()
                elif offset > 0:
                    match = (greedy[:-offset] == top[offset:, 0]).float().mean()
                else:
                    match = (greedy == top[:, 0]).float().mean()
                row[f'top1_agreement_offset_{offset}'] = float(match)
            row['teacher_top1_is_next_token'] = float((top[:, 0] == ids[1:]).float().mean())
            emit(row)

    # 3. One step on one fixed record must lower that record's preserve loss.
    trainable = list(latents.values())
    for q in trainable:
        q.requires_grad_(True)
    step = 7

    def loss_value():
        with torch.no_grad():
            return float(preserve.loss(model, step, checkpoint_layers=True)[0])

    for name, lr in (('sgd', 1.0), ('sgd', 10.0), ('lionsr', 3e-6), ('lionsr', 3e-5), ('lionsr', 3e-4),
                     ('muonsr', 7.5e-7), ('muonsr', 7.5e-6)):
        restore()
        before = loss_value()
        loss, _ = preserve.loss(model, step, checkpoint_layers=True)
        if name == 'sgd':
            loss.backward()
            with torch.no_grad():
                for q in trainable:
                    q.add_(q.grad.float().to(q.dtype) * -lr)
                    q.grad = None
        else:
            optimizer = (LionSR([{'params': [q], 'lr': lr, 'row_scale': row_scale(q), 'weight_decay': 0.0}
                                 for q in trainable], lr=lr) if name == 'lionsr'
                         else MuonSR([{'params': [q]} for q in trainable], lr=lr))
            optimizer.step_in_backward(gated=True)
            with optimizer.in_backward():
                loss.backward()
            for hook in optimizer._hooks:
                hook.remove()
        after = loss_value()
        change = sum(float((q.float() - original[n].float()).norm() / original[n].float().norm())
                     for n, q in latents.items()) / len(latents)
        emit({'control': 'descent', 'optimizer': name, 'lr': lr, 'loss_before': before, 'loss_after': after,
              'weight_change_mean': change})
    restore()


if __name__ == '__main__':
    main()
