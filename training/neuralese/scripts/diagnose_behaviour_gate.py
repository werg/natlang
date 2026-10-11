"""Behaviour-gate diagnosis (plans/mellum-port.md "Behaviour gate bug hunt"): the exact gate of a recipe's quantization
component (train/quantization.Behaviour: same probes, render, greedy decode and held KL) on BF16 Mellum, optionally
with backbone weights from a trained state, keeping every generation's text.

    python diagnose_behaviour_gate.py --out report.json                       # published BF16 Mellum
    python diagnose_behaviour_gate.py --weights run-v9/core_text_warmup/checkpoint.pt --out step4096.json
    python diagnose_behaviour_gate.py --weights <qat_convert best-weights.pt> --out v3.json

``--weights`` reads ``student_parameters`` (text warm-up full state) or ``latents`` (qat_convert) by plain name.
``--point`` evaluates a recipe precision point at deploy mix (lambda = 1) instead of BF16.
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from natlang_neuralese.common.paths import root  # noqa: E402

MODEL = '/home/werg/data/models/mellum21-12b-a2.5b-thinking'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--recipe', default='recipes/raw-recurrence-mellum-v12.json')
    p.add_argument('--model', default=MODEL)
    p.add_argument('--weights')
    p.add_argument('--point', default='bf16')
    p.add_argument('--held-rows', type=int)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    from natlang_neuralese.maple.distill_data import render
    from natlang_neuralese.maple.generation_gate import behaviour_report, deployed_cache, run_gate
    from natlang_neuralese.maple.model import set_dense_moe_kernel
    from natlang_neuralese.maple.student import load_student
    from natlang_neuralese.train.backbone_policy import plain_named_tensors
    from natlang_neuralese.train.quantization import Behaviour, Quantization, teacher_shards
    from natlang_neuralese.train.recipe import resolve_recipe_data

    recipe = resolve_recipe_data(a.recipe)
    spec = recipe['quantization']
    started = time.time()
    model, tokenizer = load_student(a.model, precision='bf16', device='cuda')
    set_dense_moe_kernel('grouped')
    loaded = {}
    if a.weights:
        state = torch.load(a.weights, map_location='cpu', mmap=True, weights_only=False)
        values = state.get('student_parameters') or state.get('latents')
        tensors = plain_named_tensors(model)
        from natlang_neuralese.train.quantization import plain_parameter_name
        missing = []
        with torch.no_grad():
            for name, value in values.items():
                plain = plain_parameter_name(name)
                for prefix in ('backbone.hf.', 'backbone.'):  # text warm-up states name the port's backbone
                    if plain.startswith(prefix):
                        plain = plain[len(prefix):]
                        break
                if plain not in tensors or tensors[plain].shape != value.shape:
                    missing.append(name)
                    continue
                before = tensors[plain].float()
                delta = (value.to(tensors[plain]).float() - before).norm() / before.norm().clamp_min(1e-12)
                tensors[plain].copy_(value.to(tensors[plain]))
                loaded[plain] = float(delta)
        loaded_names = sorted(loaded)
        print(json.dumps({'event': 'weights_loaded', 'loaded': len(loaded), 'skipped': missing[:20],
                          'skipped_count': len(missing), 'step': state.get('step')}), flush=True)
        del state, values
    gate = spec['gate']
    _, held, _ = teacher_shards(spec['preserve']['teacher']['artifact'])
    behaviour = Behaviour(gate, tokenizer, held, root=root('repo'))
    groups = None
    if a.point != 'bf16':
        component = {**spec, 'stage': {'id': 'core_text_warmup', 'progress': [0.0, 0.35], 'start': 0, 'end': 4096}}
        groups = Quantization(component, root=root('repo')).resolve(a.point, 1.0)
    from natlang_neuralese.maple.ternary import active_precision
    with active_precision(groups, None if a.point == 'bf16' else a.point), torch.no_grad(), deployed_cache():
        gate_report = run_gate(model, tokenizer, behaviour.probes, render, behaviour.eos, keep_text=True)
        held_report = behaviour_report(model, tokenizer, behaviour.probes[:1], render, behaviour.eos, held,
                                       a.held_rows or gate.get('held_rows', 48), gate.get('prompt_weight', 0.25))
    report = {'weights': a.weights, 'point': a.point, 'gate_pass': gate_report['gate_pass'],
              'gate_checks': gate_report['gate_checks'], 'gate_kinds': gate_report['gate_kinds'],
              **{k: v for k, v in held_report.items() if k.startswith('held_')},
              'results': gate_report['gate_results'], 'seconds': time.time() - started}
    if loaded:
        report['weight_relative_change'] = {'count': len(loaded), 'max': max(loaded.values()),
                                            'mean': sum(loaded.values()) / len(loaded),
                                            'top': sorted(loaded.items(), key=lambda kv: -kv[1])[:12], 'all': loaded}
    with open(a.out, 'w') as handle:
        json.dump(report, handle, indent=1)
    print(json.dumps({k: v for k, v in report.items() if k not in ('results', 'weight_relative_change')}), flush=True)


if __name__ == '__main__':
    main()
