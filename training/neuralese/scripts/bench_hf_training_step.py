"""Training-step throughput of a native transformers causal LM on one real token window (GB10).

Forward-only, and forward+backward with every parameter trainable (no optimizer step), optionally with activation
checkpointing, on the first ``--tokens`` tokens of a text-corpus document. A first speed read for a candidate
student before any port work: the native implementation is unoptimized for our trainer, and no QAT is applied.

    python scripts/bench_hf_training_step.py --model DIR --text-data T [--tokens 8192 --experts grouped_mm]
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
    p.add_argument('--experts', default='grouped_mm', help='transformers experts implementation (eager, grouped_mm, ...)')
    p.add_argument('--reps', type=int, default=3)
    a = p.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(a.model)
    kwargs = {'experts_implementation': a.experts} if a.experts else {}
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map='cuda', **kwargs)
    text = next(row['text'] for row in map(json.loads, open(a.text_data)) if len(row['text']) > 8 * a.tokens)
    ids = tokenizer(text, return_tensors='pt').input_ids[:, :a.tokens].cuda()
    results = {'model': a.model, 'tokens': ids.shape[1], 'experts_implementation': a.experts,
               'parameters': sum(q.numel() for q in model.parameters())}
    model.eval()
    with torch.no_grad():
        results['forward_seconds'] = timed(lambda: model(input_ids=ids, labels=ids), a.reps)
        results['forward_ce'] = float(model(input_ids=ids, labels=ids).loss)
    model.train()
    for checkpointing in (False, True):
        if checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        torch.cuda.reset_peak_memory_stats()
        def step():
            model.zero_grad(set_to_none=True)
            model(input_ids=ids, labels=ids).loss.backward()
        try:
            seconds = timed(step, a.reps)
            results['train_step' + ('_checkpointed' if checkpointing else '')] = {
                'seconds': seconds, 'tokens_per_second': ids.shape[1] / seconds,
                'peak_gb': torch.cuda.max_memory_allocated() / 2**30}
        except torch.OutOfMemoryError as error:
            results['train_step' + ('_checkpointed' if checkpointing else '')] = {'error': str(error)[:200]}
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
    print(json.dumps(results, indent=1))


if __name__ == '__main__':
    main()
