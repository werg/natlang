"""SFT positive test of the backbone-latent optimizers (plans/mellum-port.md "Behaviour gate bug hunt"): plain
whole-trajectory SFT (response CE at weight 1, prompt at ``--prompt-weight``) of BF16 Mellum's attention, expert and
router matrices on an exact-scored task, with the latent optimizers of the text warm-up (LionSR, MuonSR: one BF16
momentum, stochastic rounding, stepping in backward). Per configuration, from the same start:

- every ``--eval-every`` updates: held task accuracy (greedy, exact match) and answer CE, held KL to BF16 Mellum on
  the behaviour gate's held rows, per-group ‖ΔW‖/‖W‖, mean update seconds;
- at the end: the full behaviour gate (24 probes, texts kept, and held KL).

The base model's task accuracy and gate are measured once first.

    python sft_positive_test.py --task-dir DIR --sweep lionsr:3e-6,lionsr:1e-5,muonsr:2.5e-6:3 --out sft.jsonl
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from natlang_neuralese.common.paths import resolve  # noqa: E402

MODEL = str(resolve('models', 'mellum21-12b-a2.5b-thinking'))
GROUPS = {'attention': ('.self_attn.q_proj.weight', '.self_attn.k_proj.weight', '.self_attn.v_proj.weight',
                        '.self_attn.o_proj.weight'),
          'experts': ('.mlp.experts.gate_up', '.mlp.experts.down'), 'router': ('.mlp.gate.weight',)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--recipe', default='recipes/raw-recurrence-mellum-v12.json')
    p.add_argument('--task-dir', required=True)
    p.add_argument('--sweep', required=True, help='optimizer:lr[:ns_steps],...')
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--eval-every', type=int, default=50)
    p.add_argument('--prompt-weight', type=float, default=0.25)
    p.add_argument('--drift-rows', type=int, default=16)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    from natlang_neuralese.maple.distill_data import render
    from natlang_neuralese.maple.generation_gate import behaviour_report, greedy, held_kl
    from natlang_neuralese.maple.model import set_dense_moe_kernel
    from natlang_neuralese.maple.qat_convert import row_scale
    from natlang_neuralese.maple.student import load_student
    from natlang_neuralese.train.optim import LionSR, MuonSR
    from natlang_neuralese.train.quantization import Behaviour, teacher_shards
    from natlang_neuralese.common.paths import root
    from natlang_neuralese.train.recipe import resolve_recipe_data

    spec = resolve_recipe_data(a.recipe)['quantization']
    model, tokenizer = load_student(MODEL, precision='bf16', device='cuda')
    set_dense_moe_kernel('grouped')
    model.model.checkpoint_layers = True
    for layer in model.model.layers:
        layer.mlp.experts.make_latent(quantize=False)
    suffixes = sum(GROUPS.values(), ())
    latents = []
    for name, q in model.named_parameters():
        selected = name.startswith('model.layers.') and name.endswith(suffixes)
        q.requires_grad_(selected)
        if selected:
            latents.append((name, q))
    start = {name: q.detach().clone() for name, q in latents}
    _, held, _ = teacher_shards(spec['preserve']['teacher']['artifact'])
    behaviour = Behaviour(spec['gate'], tokenizer, held, root=root('repo'))
    eos = tokenizer.convert_tokens_to_ids('<|im_end|>')
    train = [json.loads(l) for l in open(Path(a.task_dir) / 'train.jsonl')]
    test = [json.loads(l) for l in open(Path(a.task_dir) / 'test.jsonl')]

    def encode(item):
        # The generation prompt (non-thinking: an empty think block) plus the answer and <|im_end|>: exactly what
        # greedy decoding continues (the template's own rendering of a final assistant turn drops the think block).
        prompt = render(tokenizer, {'messages': item['messages']})
        return prompt, prompt + tokenizer.encode(item['answer'], add_special_tokens=False) + [eos]
    encoded_train = [encode(item) for item in train]
    encoded_test = [encode(item) for item in test]
    log = open(a.out, 'a')

    def emit(row):
        log.write(json.dumps(row) + '\n'); log.flush()
        print(json.dumps({k: v for k, v in row.items() if k != 'gate'}), flush=True)

    def sequence_loss(prompt, full, prompt_weight):
        ids = torch.tensor([full], device='cuda')
        h = model.model(input_ids=ids).last_hidden_state[0, :-1]
        logp = torch.log_softmax(model.lm_head(h).float(), -1)
        ce = -logp.gather(-1, ids[0, 1:, None])[:, 0]
        weight = torch.ones_like(ce)
        weight[:len(prompt) - 1] = prompt_weight
        return (weight * ce).sum() / weight.sum(), ce[len(prompt) - 1:].mean()

    def task_eval():
        correct, by_kind, ce_sum = 0, {}, 0.0
        with torch.no_grad():
            for item, (prompt, full) in zip(test, encoded_test):
                tokens = greedy(model, prompt, 8, eos)
                text = tokenizer.decode(tokens, skip_special_tokens=True).strip().strip('"\'.').lower()
                ok = text == item['answer'].lower()
                correct += ok
                kind = by_kind.setdefault(item['kind'], [0, 0])
                kind[0] += ok; kind[1] += 1
                ce_sum += float(sequence_loss(prompt, full, 0.0)[1])
        return {'task_accuracy': correct / len(test), 'task_accuracy_by_kind': {k: v[0] / v[1] for k, v in by_kind.items()},
                'task_answer_ce': ce_sum / len(test)}

    def change():
        out = {}
        with torch.no_grad():
            for group, ends in GROUPS.items():
                values = [float((q.float() - start[n].float()).norm() / start[n].float().norm())
                          for n, q in latents if n.endswith(ends)]
                out[group] = sum(values) / len(values)
        return out

    def gate():
        report = behaviour_report(model, tokenizer, behaviour.probes, render, eos, held,
                                  spec['gate'].get('held_rows', 48), spec['gate'].get('prompt_weight', 0.25))
        return {'gate_pass': report['gate_pass'], 'gate_checks': report['gate_checks'],
                'gate_kinds': report['gate_kinds'], 'held_kl_48': report.get('held_kl'), 'gate': report}

    emit({'config': 'base', 'step': 0, **task_eval(), **held_kl(model, held, a.drift_rows), **gate()})
    for item in a.sweep.split(','):
        parts = item.split(':')
        name, lr = parts[0], float(parts[1])
        ns = int(parts[2]) if len(parts) > 2 else 5
        with torch.no_grad():
            for n, q in latents:
                q.copy_(start[n])
        if name == 'lionsr':
            optimizer = LionSR([{'params': [q], 'lr': lr, 'row_scale': row_scale(q), 'weight_decay': 0.0}
                                for _, q in latents], lr=lr)
        else:
            optimizer = MuonSR([{'params': [q]} for _, q in latents], lr=lr, ns_steps=ns)
        optimizer.step_in_backward(gated=True)
        config = f'{name}:{lr:g}' + (f':ns{ns}' if name == 'muonsr' else '')
        order = random.Random(0)
        seconds, losses = 0.0, []
        for step in range(1, a.steps + 1):
            batch = [encoded_train[order.randrange(len(encoded_train))] for _ in range(a.batch)]
            torch.cuda.synchronize()
            tick = time.time()
            with optimizer.in_backward():
                total = sum(sequence_loss(prompt, full, a.prompt_weight)[0] for prompt, full in batch) / a.batch
                total.backward()
            torch.cuda.synchronize()
            seconds += time.time() - tick
            losses.append(float(total))
            if step % a.eval_every == 0 or step == a.steps:
                row = {'config': config, 'step': step, 'train_loss': sum(losses) / len(losses),
                       'seconds_per_update': seconds / step, 'weight_change': change(), **task_eval(),
                       **held_kl(model, held, a.drift_rows)}
                if step == a.steps:
                    row.update(gate())
                emit(row)
                losses = []
        for hook in optimizer._hooks:
            hook.remove()
        del optimizer
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
