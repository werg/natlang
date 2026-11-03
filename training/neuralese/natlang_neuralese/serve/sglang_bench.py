"""SGLang actor throughput and actor/learner mismatch on real RL turns (natlang SGLang fork, serve/sglang_actor.py).

Workload: JSONL turns with the exact rendered prompt token ids an actor received and the tokens it sampled
(``workload`` builds it from collector partials). Steps, each writing one JSON report under ``--out``:

  workload   pilot jobs -> turns.jsonl (ordinary-token turns; prompt ids, native samples and their captured logps)
  serve      HF export of the checkpoint (export/native_hf.py) and the fork's server with the actor's launch flags
  sample     T=1 SGLang sampling at each concurrency with the compact top-k capture: output tok/s, cached prompt
             tokens; sampled tokens and logps are kept for ``rescore``
  score      SGLang prefill log-probabilities of the native actor's own sampled continuation vs its captured logps
  rescore    the learner's arithmetic (native prefill + one native forward over the sampled tokens) on every SGLang
             sample: the per-token |dlogp| and per-turn max/k3 distribution the engine-mismatch guard bounds
             (rl/native_replay.py DEFAULT_ENGINE_GUARD)
  all        serve, score, sample, stop the server, rescore

Numbers are measurements of this host and GPU; a shared GPU makes them lower bounds.
"""
from __future__ import annotations

import argparse
import base64
import glob
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


def _post(url, body, timeout=7200):
    request = urllib.request.Request(url + '/generate', data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def _quantiles(values):
    a = np.abs(np.asarray([v for v in values if v is not None], dtype=np.float64))
    if not len(a):
        return {}
    return {'n': int(len(a)), 'mean': float(a.mean()), 'p50': float(np.quantile(a, .5)),
            'p90': float(np.quantile(a, .9)), 'p99': float(np.quantile(a, .99)),
            'p999': float(np.quantile(a, .999)), 'max': float(a.max())}


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.partial')
    tmp.write_text(json.dumps(value, indent=1) + '\n')
    os.replace(tmp, path)
    print(json.dumps({k: v for k, v in value.items() if k not in ('per_turn',)})[:4000], flush=True)


def build_workload(roots, out):
    rows, seen = [], set()
    for root in roots:
        for path in sorted(glob.glob(os.path.join(root, '**/jobs/*.json'), recursive=True)):
            try:
                data = json.load(open(path))
            except (OSError, ValueError):
                continue
            for index, turn in enumerate(data.get('turns') or []):
                capture = ((turn.get('response') or {}).get('raw_response') or {}).get('x_natlang_rl_capture')
                if not capture or capture.get('writes'):
                    continue
                prompt = []
                for segment in capture['prompt']:
                    if (segment['kind'] != 'tokens' or any(segment.get('producing') or [])
                            or segment.get('privileged')):
                        prompt = None
                        break
                    prompt += segment['token_ids']
                samples = capture['samples']
                if not prompt or any(s.get('forced') or s.get('input_role') != 'ordinary' for s in samples):
                    continue
                key = hashlib.sha256(json.dumps([prompt, [s['token_id'] for s in samples]]).encode()).hexdigest()
                if key in seen:
                    continue
                seen.add(key)
                rows.append({'source': path, 'turn': index, 'policy': capture['policy']['checkpoint_sha256'],
                             'prompt': prompt, 'tokens': [s['token_id'] for s in samples],
                             'logps': [s['logp'] for s in samples],
                             'terminal': [bool(s.get('terminal')) for s in samples]})
    Path(out).write_text(''.join(json.dumps(row) + '\n' for row in rows))
    return {'turns': len(rows), 'policies': sorted({r['policy'] for r in rows}),
            'prompt_tokens': sum(len(r['prompt']) for r in rows), 'sampled_tokens': sum(len(r['tokens']) for r in rows),
            'stop_ids': sorted({r['tokens'][i] for r in rows for i, t in enumerate(r['terminal']) if t})}


def serve(args):
    """Export (cached by checkpoint) and start the fork's server with serve_native_checkpoint's actor flags."""
    root = Path(__file__).resolve().parents[4]
    fork = json.loads((root / 'training/neuralese/sglang-fork.json').read_text())
    python = args.sglang_python or fork['python']
    export_dir = Path(args.export_dir)
    if not (export_dir / 'natlang-export.json').is_file():
        subprocess.run([sys.executable, '-m', 'natlang_neuralese.export.native_hf', str(args.checkpoint),
                        str(export_dir), '--base', str(args.base)], check=True)
    run = args.max_running
    command = [python, '-m', 'sglang.launch_server', '--model-path', str(export_dir), '--host', '127.0.0.1',
               '--port', str(args.port), '--dtype', 'bfloat16', '--mem-fraction-static', str(args.mem_fraction),
               '--context-length', '32768', '--max-total-tokens', str(args.max_total_tokens),
               '--max-running-requests', str(run), '--max-mamba-cache-size', str(8 * run),
               '--cuda-graph-max-bs-decode', str(run), '--incremental-streaming-output', '--stream-interval', '8',
               '--skip-tokenizer-init', '--trust-remote-code']
    cuda = os.environ.get('CUDA_HOME', '/usr/local/cuda')
    env = {**os.environ, 'PYTHONPATH': str(root / 'training/neuralese'), 'CUDA_HOME': cuda,
           'PATH': os.pathsep.join([str(Path(python).parent), cuda + '/bin', os.environ.get('PATH', '')]),
           'NATLANG_ADAPTER_SLOTS': '4', 'NATLANG_ADAPTER_RANK': '32'}
    log = open(Path(args.out) / 'sglang-server.log', 'ab')
    started = time.monotonic()
    process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    url = f'http://127.0.0.1:{args.port}'
    while True:
        if process.poll() is not None:
            raise RuntimeError(f'SGLang server exited with {process.returncode}')
        try:
            with urllib.request.urlopen(url + '/health_generate', timeout=60) as response:
                if response.status == 200:
                    break
        except Exception:
            pass
        if time.monotonic() - started > 1800:
            os.killpg(process.pid, signal.SIGKILL)
            raise RuntimeError('SGLang server not healthy after 30 min')
        time.sleep(5)
    export = json.loads((export_dir / 'natlang-export.json').read_text())
    _write(Path(args.out) / 'serve.json', {'ready_seconds': time.monotonic() - started, 'command': command,
        'fork': {k: fork[k] for k in ('repository', 'commit')}, 'checkpoint_sha256': export['checkpoint_sha256'],
        'hf_export_weights_sha256': export['weights_sha256'], 'gpu': _gpu()})
    return process, url


def _gpu():
    try:
        return subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'],
                              capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return None


def score(url, rows, concurrency):
    def one(row):
        body = {'input_ids': row['prompt'] + row['tokens'], 'return_logprob': True,
                'logprob_start_len': len(row['prompt']) - 1, 'sampling_params': {'temperature': 0.0, 'max_new_tokens': 1}}
        values = [item[0] for item in _post(url, body)['meta_info']['input_token_logprobs']][1:]
        return values[:len(row['tokens'])]
    started = time.monotonic()
    with ThreadPoolExecutor(concurrency) as pool:
        results = list(pool.map(one, rows))
    deltas, turns = [], []
    for row, values in zip(rows, results):
        delta = [s - n for s, n in zip(values, row['logps']) if s is not None]
        deltas += delta
        if delta:
            turns.append({'tokens': len(delta), 'max_abs': float(np.max(np.abs(delta))),
                          'mean_abs': float(np.mean(np.abs(delta)))})
    return {'step': 'score', 'what': 'SGLang prefill logp of the native actor sampled tokens minus native capture logp',
            'seconds': time.monotonic() - started, 'token_abs_delta': _quantiles(deltas),
            'turn_max_abs': _quantiles([t['max_abs'] for t in turns]), 'per_turn': turns}


def sample(url, rows, concurrency, requests, *, max_tokens, min_tokens, stop_ids, out):
    work = [rows[i % len(rows)] for i in range(requests)]

    def one(row):
        limit = max(min_tokens, min(max_tokens, len(row['tokens'])))
        body = {'input_ids': row['prompt'], 'sampling_params': {'temperature': 1.0, 'max_new_tokens': limit,
                'stop_token_ids': stop_ids, 'skip_special_tokens': False, 'no_stop_trim': True,
                'custom_params': {'natlang_top_k': 128}}}
        started = time.monotonic()
        result = _post(url, body)
        records = result['meta_info'].get('natlang_top_k') or []
        logps = [float(np.frombuffer(base64.b64decode(r)[:4], dtype='<f4')[0]) for r in records]
        meta = {k: v for k, v in result['meta_info'].items() if not isinstance(v, (list, dict))}
        return {'prompt': row['prompt'], 'tokens': result['output_ids'], 'logps': logps,
                'seconds': time.monotonic() - started, 'meta': meta}
    started = time.monotonic()
    with ThreadPoolExecutor(concurrency) as pool:
        results = list(pool.map(one, work))
    wall = time.monotonic() - started
    with open(out, 'w') as handle:
        for result in results:
            handle.write(json.dumps(result) + '\n')
    tokens = sum(len(r['tokens']) for r in results)
    return {'step': 'sample', 'concurrency': concurrency, 'requests': len(results), 'wall_seconds': wall,
            'output_tokens': tokens, 'output_tok_s': tokens / wall,
            'prompt_tokens': sum(len(r['prompt']) for r in results),
            'cached_prompt_tokens': sum(int(r['meta'].get('cached_tokens', 0)) for r in results),
            'per_request_tok_s_median': float(np.median([len(r['tokens']) / r['seconds'] for r in results])),
            'records_cover_tokens': all(len(r['logps']) == len(r['tokens']) for r in results)}


def rescore(args, sample_files):
    import torch
    from . import load_native_engine, configure_native_prefix_cache
    from ..model.shared_state import native_step
    engine = load_native_engine(str(args.checkpoint), base=str(args.base), device='cuda', prompt_rendering='full')
    configure_native_prefix_cache(engine, 2.0, None)
    model = engine.shared_model
    rows = [json.loads(line) for name in sample_files for line in open(name)]
    deltas, turns = [], []
    started = time.monotonic()
    with torch.inference_mode():
        for row in rows:
            tokens = row['tokens']
            if not tokens or len(row['logps']) != len(tokens):
                continue
            prefill = engine._native_prefill_forward(row['prompt'], None)
            values = [float(torch.log_softmax(model.logits(prefill['h_final'][:, -1]).float(), -1)[0, tokens[0]])]
            if len(tokens) > 1:
                hidden = native_step(model, torch.tensor([tokens[:-1]], device='cuda'), prefill['cache'])['h_final'][0]
                for start in range(0, hidden.shape[0], 256):
                    lp = torch.log_softmax(model.logits(hidden[start:start + 256]).float(), -1)
                    index = torch.tensor(tokens[1 + start:1 + start + lp.shape[0]], device='cuda')
                    values += lp.gather(1, index[:, None])[:, 0].tolist()
            delta = [a - b for a, b in zip(values, row['logps'])]
            deltas += delta
            turns.append({'tokens': len(delta), 'max_abs': float(np.max(np.abs(delta))),
                          'mean_abs': float(np.mean(np.abs(delta))), 'mean_delta': float(np.mean(delta)),
                          'mean_k3': float(np.mean([np.exp(min(x, 20.)) - 1 - x for x in delta]))})
    return {'step': 'rescore', 'what': 'native learner-arithmetic logp minus SGLang sampled logp, SGLang samples',
            'seconds': time.monotonic() - started, 'turns': len(turns), 'token_abs_delta': _quantiles(deltas),
            'mean_delta': float(np.mean(deltas)) if deltas else None,
            'turn_max_abs': _quantiles([t['max_abs'] for t in turns]),
            'turn_mean_k3': _quantiles([t['mean_k3'] for t in turns]), 'per_turn': turns}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('step', choices=('workload', 'serve-measure', 'rescore', 'all'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--workload', type=Path)
    p.add_argument('--roots', nargs='*', default=[])
    p.add_argument('--checkpoint', type=Path); p.add_argument('--base', type=Path)
    p.add_argument('--export-dir', type=Path); p.add_argument('--sglang-python')
    p.add_argument('--port', type=int, default=9310); p.add_argument('--max-running', type=int, default=32)
    p.add_argument('--mem-fraction', type=float, default=0.3); p.add_argument('--max-total-tokens', type=int, default=131072)
    p.add_argument('--concurrency', type=int, nargs='*', default=[1, 8, 32])
    p.add_argument('--max-tokens', type=int, default=1024); p.add_argument('--min-tokens', type=int, default=64)
    p.add_argument('--stop-ids', type=int, nargs='*', default=[124900])
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.step == 'workload':
        _write(args.out / 'workload.json', build_workload(args.roots, args.workload))
        return
    if args.step in ('serve-measure', 'all'):
        rows = [json.loads(line) for line in open(args.workload)]
        process, url = serve(args)
        try:
            _write(args.out / 'score.json', score(url, rows, 8))
            for c in args.concurrency:
                _write(args.out / f'sample-c{c}.json', sample(url, rows, c, 4 if c < 8 else 2 * c,
                    max_tokens=args.max_tokens, min_tokens=args.min_tokens, stop_ids=args.stop_ids,
                    out=args.out / f'samples-c{c}.jsonl'))
        finally:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
    if args.step in ('rescore', 'all'):
        files = sorted(str(f) for f in args.out.glob('samples-c*.jsonl'))
        _write(args.out / 'rescore.json', rescore(args, files))


if __name__ == '__main__':
    main()
