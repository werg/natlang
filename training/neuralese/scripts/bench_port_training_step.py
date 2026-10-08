"""Training-step throughput of a Maple-family student (Maple or Mellum) through our port, on one real token window.

The same code path the Neuralese trainers use (``load_maple`` + ``MaplePortBackbone``), so Maple and Mellum compare
like for like. Trainable: attention projections, norms and router (as under QAT), and, where experts are dense BF16
(Mellum), the experts too (``--train-experts``). Forward-only and forward+backward CE (no optimizer step).

    python scripts/bench_port_training_step.py --model DIR --text-data T [--tokens 8192 --train-experts]
"""
import argparse
import json
import time

import torch


def timed(fn, reps):
    fn()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) / reps


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--text-data', required=True)
    p.add_argument('--tokens', type=int, default=8192)
    p.add_argument('--train-experts', action='store_true')
    p.add_argument('--cache', default=None, help='converted-checkpoint cache (Maple)')
    p.add_argument('--reps', type=int, default=3)
    a = p.parse_args()
    from transformers import AutoTokenizer
    from natlang_neuralese.maple.model import load_maple
    from natlang_neuralese.train.joint_kd import chunked_ce_kl
    tokenizer = AutoTokenizer.from_pretrained(a.model)
    model = load_maple(a.model, device='cuda', dtype=torch.bfloat16, ternary_attention=False, cache=a.cache)
    text = next(row['text'] for row in map(json.loads, open(a.text_data)) if len(row['text']) > 8 * a.tokens)
    ids = tokenizer(text, return_tensors='pt').input_ids[:, :a.tokens].cuda()
    labels = torch.cat([ids[:, 1:], torch.full_like(ids[:, :1], -100)], 1)
    trainable = 0
    for name, parameter in model.named_parameters():
        train = parameter.is_floating_point() and (
            '.self_attn.' in name or 'norm' in name or '.mlp.gate' in name or (a.train_experts and '.experts.' in name))
        parameter.requires_grad_(train)
        trainable += parameter.numel() if train else 0
    head = model.get_output_embeddings().weight

    def loss():
        hidden = model.model(input_ids=ids).last_hidden_state
        return chunked_ce_kl(hidden, head, labels, chunk=1024)[0]

    results = {'model': a.model, 'model_type': model.config.model_type, 'tokens': ids.shape[1],
               'parameters': sum(q.numel() for q in model.parameters()), 'trainable': trainable,
               'train_experts': a.train_experts}
    with torch.no_grad():
        results['forward_seconds'] = timed(loss, a.reps)
        results['ce'] = float(loss())
    torch.cuda.reset_peak_memory_stats()

    def step():
        model.zero_grad(set_to_none=True)
        loss().backward()
    try:
        seconds = timed(step, a.reps)
        results['train_step'] = {'seconds': seconds, 'tokens_per_second': ids.shape[1] / seconds,
                                 'peak_gb': torch.cuda.max_memory_allocated() / 2**30}
    except torch.OutOfMemoryError as error:
        results['train_step'] = {'error': str(error)[:200]}
    print(json.dumps(results, indent=1))


if __name__ == '__main__':
    main()
