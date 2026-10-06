"""Measure CPU preparation overlap and native decode batching on exact learned weights."""
import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import json
from pathlib import Path
import time

import torch

from ..serve.engine import GenerationRequest, Sequence
from ..serve.recurrence_checkpoint import load_recurrence_checkpoint
from ..train.output_embedding_projection import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--requests', type=int, default=16)
    parser.add_argument('--max-tokens', type=int, default=64)
    args = parser.parse_args()
    if args.out.exists() or args.requests < 1 or args.max_tokens < 1:
        raise ValueError('fresh output and positive bounds required')
    args.out.mkdir(parents=True)
    torch.set_num_threads(2)
    engine, _ = load_recurrence_checkpoint(args.checkpoint, device='cuda', dtype=torch.bfloat16)
    engine.start()
    rows = []
    try:
        for words in (128, 1024):
            messages = [{'role': 'user', 'content': 'Read these records and summarize their repeated observation.\n' +
                         ('The package arrived at the sorting center.\n' * words)}]
            start = time.perf_counter()
            for _ in range(8):
                engine._prompt_plan(messages, None)
            host_ms = (time.perf_counter() - start) * 1000 / 8
            for concurrency in (1, 4, 8):
                for prepared in (False, True):
                    def run(index):
                        request = GenerationRequest(messages=messages, max_tokens=args.max_tokens, seed=73,
                                                    request_id=f'bench-{index}')
                        begin = time.perf_counter()
                        if prepared:
                            future = engine.submit(request)
                        else:
                            # Old scheduler path: no host preparation before queueing.
                            future = Future()
                            engine._incoming.put(Sequence(request, future))
                        result = future.result()
                        return time.perf_counter() - begin, result['usage']['completion_tokens']
                    # Same warm-up for both modes; excludes model loading and CUDA setup.
                    run(-1)
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    begin = time.perf_counter()
                    with ThreadPoolExecutor(max_workers=concurrency) as pool:
                        results = list(pool.map(run, range(args.requests)))
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - begin
                    row = dict(words=words, concurrency=concurrency, host_prepared=prepared,
                               requests=args.requests, elapsed_s=elapsed, host_preparation_ms=host_ms,
                               completion_tokens=sum(r[1] for r in results),
                               mean_request_s=sum(r[0] for r in results)/len(results),
                               peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                               peak_reserved_bytes=torch.cuda.max_memory_reserved())
                    row['completion_tokens_per_s'] = row['completion_tokens']/elapsed
                    rows.append(row)
                    with (args.out/'metrics.jsonl').open('a') as stream:
                        stream.write(json.dumps(row)+'\n')
                    print(json.dumps(row), flush=True)
    finally:
        engine.stop()
    (args.out/'summary.json').write_text(json.dumps(dict(
        schema='natlang.scheduler-throughput/1', pins={str(args.checkpoint):sha(args.checkpoint)},
        rows=rows, synthetic_prompts=True, training_publication=False,
        scope='Whole server schedule walltime, synthetic crisp prompts; no task accuracy or recurrence training speedup claim.'),indent=2)+'\n')


if __name__ == '__main__':
    main()
