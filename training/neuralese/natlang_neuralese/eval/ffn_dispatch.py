"""Isolated FFN dispatch diagnostic; synthetic weights, never channel qualification."""
import argparse
import json
import time
from pathlib import Path
from typing import Tuple

import torch
from peft import LoraConfig
from peft.tuners.lora.layer import Linear

Weights = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]


def lora(value: torch.Tensor, weights: Weights) -> torch.Tensor:
    base, a, b, scale = weights
    result = torch.nn.functional.linear(value, base)
    delta = torch.nn.functional.linear(torch.nn.functional.linear(value.to(a.dtype), a), b) * scale
    return result + delta.to(result.dtype)


def functional(value: torch.Tensor, w1: Weights, w2: Weights, w3: Weights) -> torch.Tensor:
    return lora(torch.nn.functional.silu(lora(value, w1)) * lora(value, w3), w2)


def pack(module):
    return (module.base_layer.weight, module.lora_A['neuralese'].weight,
            module.lora_B['neuralese'].weight, module.scaling['neuralese'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--width', type=int, default=1024)
    parser.add_argument('--intermediate', type=int, default=4608)
    parser.add_argument('--tokens', default='1,1024')
    parser.add_argument('--rows', default='1,3')
    parser.add_argument('--repeats', type=int, default=16)
    args = parser.parse_args()
    tokens = [int(n) for n in args.tokens.split(',')]
    rows = [int(n) for n in args.rows.split(',')]
    if min(args.width, args.intermediate, args.repeats, *tokens, *rows) < 1:
        raise ValueError('positive bounds required')
    if args.out.exists():
        raise ValueError('fresh diagnostic output required')
    args.out.mkdir(parents=True)
    torch.manual_seed(1942)
    torch.set_num_threads(1)
    config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0)

    def module(input_width, output_width):
        result = Linear(torch.nn.Linear(input_width, output_width, bias=False).to('cuda', torch.bfloat16),
                        'neuralese', config=config, r=16, lora_alpha=32)
        result.to(device='cuda', dtype=torch.bfloat16)
        result.base_layer.weight.requires_grad_(False)
        with torch.no_grad():
            result.lora_B['neuralese'].weight.normal_(0, .01)
        return result

    w1, w2, w3 = module(args.width, args.intermediate), module(args.intermediate, args.width), module(args.width, args.intermediate)
    weights = pack(w1), pack(w2), pack(w3)
    scripted = torch.jit.script(functional)
    functions = {
        'peft-eager': lambda x: w2(torch.nn.functional.silu(w1(x)) * w3(x)),
        'functional-eager': lambda x: functional(x, *weights),
        'functional-torchscript': lambda x: scripted(x, *weights),
    }
    parameters = [p for m in (w1, w2, w3) for p in m.parameters() if p.requires_grad]
    results = []
    for batch in rows:
        for length in tokens:
            x = torch.randn(batch, length, args.width, device='cuda', dtype=torch.bfloat16, requires_grad=True)
            targets = [x, *parameters]
            output = functions['peft-eager'](x)
            reference = output.detach().clone()
            adjoints = torch.autograd.grad(output.float().square().mean(), targets)
            del output
            for name, forward in functions.items():
                output = forward(x)
                gradients = torch.autograd.grad(output.float().square().mean(), targets)
                row = {'method': name, 'rows': batch, 'tokens': length,
                       'output_equal': torch.equal(output.detach(), reference),
                       'max_abs_output': float((output.detach() - reference).abs().max()),
                       'gradients_equal': all(torch.equal(a, b) for a, b in zip(gradients, adjoints)),
                       'max_abs_gradient': max(float((a - b).abs().max()) for a, b in zip(gradients, adjoints))}
                del output, gradients
                for _ in range(3):
                    torch.autograd.grad(forward(x).float().square().mean(), targets)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                for _ in range(args.repeats):
                    torch.autograd.grad(forward(x).float().square().mean(), targets)
                torch.cuda.synchronize()
                row.update(wall_ms_per_forward_backward=(time.perf_counter() - started) * 1000 / args.repeats,
                           peak_allocated_bytes=torch.cuda.max_memory_allocated())
                results.append(row)
                print(json.dumps(row), flush=True)
            del x, targets, reference, adjoints
    (args.out / 'summary.json').write_text(json.dumps({
        'scope': 'Synthetic standalone BF16 rank16 FFN at declared geometry; paired forward/adjoints and wall time including dispatch. No full model, norm, attention, recurrence, optimizer, replay, or channel qualification; no deployment.',
        'device': torch.cuda.get_device_name(), 'torch': torch.__version__,
        'geometry': {'width': args.width, 'intermediate': args.intermediate},
        'results': results}, indent=2) + '\n')


if __name__ == '__main__':
    main()
