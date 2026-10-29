#!/usr/bin/env python3
"""LFM3 logits on natlang's SGLang fork (training/neuralese/sglang-fork.json) against the vendored HF reference.

usage: check_sglang_lfm3_parity.py --model DIR --requests REQ.jsonl --out OUT.json [--n 6] [--gen 48] [--k 20]

REQ.jsonl rows hold pre-tokenized prompts (``{"prompt_ids": [...]}``, as
``runs/lfm3-engine-bench-20261028/inputs/make_requests.py`` writes from real natlang episode turns). Phase 1, in the
fork's Python (subprocess): SGLang prefills each prompt with input logprobs from position 0 and top-k (every prefill
position: chunked prefill across ``--chunk``, the 256-token sliding window and the width-7 convolution state carried
across chunks) and decodes ``gen`` greedy tokens with top-k (cached decode), then samples at temperatures 1.0 and 0.7
with the fork's compact capture (``custom_params.natlang_top_k``). Phase 2, in this Python: the HF reference
(natlang_neuralese.third_party.lfm3) runs prompt + generated tokens in one full forward in FP32 (the reference) and in
BF16 (the noise floor: a BF16 forward is ~0.1 nats from FP32 on low-probability prompt tokens). Both sides are raw
log-softmax (SGLANG_RETURN_ORIGINAL_LOGPROB). Compared per position: the logprob of the actual next token, the top-1
token and the top-k set, separately for prefill and decode positions. Fails when SGLang is further from FP32 than HF's
own BF16 forward (mean |delta| above 1.25x the floor + 0.005, or top-1 agreement more than 0.01 below it), or when
the compact capture's sampled logp exceeds TOLERANCE. Also records whether the fork runs LFM3's CuTe DSL gated conv.
"""
import argparse, json, os, subprocess, sys
from pathlib import Path

TOLERANCE = {'max_abs_logprob': 0.6, 'mean_abs_logprob': 0.05}  # compact capture vs the FP32 reference


def sglang_phase(a):
    os.environ.setdefault('SGLANG_RETURN_ORIGINAL_LOGPROB', '1')  # raw log-softmax, as the vLLM side's raw_logprobs
    import sglang as sgl
    rows = [json.loads(l) for l in open(a.requests)][:a.n]
    ids = [r.get('prompt_token_ids') or r['prompt_ids'] for r in rows]
    kw = dict(model_path=a.model, dtype='bfloat16', mem_fraction_static=a.mem_fraction, context_length=16384,
              chunked_prefill_size=a.chunk, random_seed=0, log_level='warning', trust_remote_code=True)
    if a.disable_cuda_graph:
        kw['disable_cuda_graph'] = True
    kw.update(json.loads(a.engine_kw))
    from sglang.srt.models import lfm3 as sglang_lfm3
    import torch
    eng = sgl.Engine(**kw)
    outs = eng.generate(input_ids=ids, sampling_params={'temperature': 0.0, 'max_new_tokens': a.gen,
                                                         'ignore_eos': True},
                        return_logprob=True, logprob_start_len=0, top_logprobs_num=a.k)
    res = []
    for p, o in zip(ids, outs):
        m = o['meta_info']
        top_in = m['input_top_logprobs']
        tok_in = m['input_token_logprobs']
        prompt = [None] + [{str(t): lp for lp, t, _ in (top_in[i] or [])} | {str(tok_in[i][1]): tok_in[i][0]}
                           for i in range(1, len(p))]
        gen = [t for _, t, _ in m['output_token_logprobs']]
        dec = [{str(t): lp for lp, t, _ in top} | {str(t0): lp0}
               for top, (lp0, t0, _) in zip(m['output_top_logprobs'], m['output_token_logprobs'])]
        res.append({'prompt_token_ids': p, 'generated': gen, 'prompt_logprobs': prompt, 'logprobs': dec})
    # The fork's compact capture (custom_params.natlang_top_k): the exact post-temperature law of sampled tokens.
    sampled = []
    for temperature in (1.0, 0.7):
        outs = eng.generate(input_ids=ids, sampling_params={'temperature': temperature, 'max_new_tokens': a.gen,
                                                             'ignore_eos': True, 'custom_params': {'natlang_top_k': a.k}})
        for p, o in zip(ids, outs):
            sampled.append({'prompt_token_ids': p, 'temperature': temperature, 'generated': o['output_ids'],
                            'records': o['meta_info']['natlang_top_k']})
    eng.shutdown()
    Path(a.engine_out).write_text(json.dumps({'rows': res, 'sampled': sampled, 'cute_gated_conv': bool(sglang_lfm3._use_cute_conv),
        'device_capability': list(torch.cuda.get_device_capability()), 'sglang': sgl.__version__}))


def hf_phase(a):
    """Reference: the HF model in FP32. The HF model in BF16 against the same reference is the noise floor a BF16
    engine is held to (a BF16 forward differs from FP32 by ~0.1 nats on low-probability prompt tokens)."""
    import torch
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / 'training/neuralese'))
    from natlang_neuralese.third_party import lfm3
    lfm3.register()
    from transformers import AutoModelForCausalLM

    def load(dtype):
        return AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype, trust_remote_code=False,
                                                    attn_implementation='sdpa').to('cuda').eval()
    model, noisy = load(torch.float32), load(torch.bfloat16)
    engine = json.loads(Path(a.engine_out).read_text())
    rows = engine['rows']
    stats = {'prefill': [], 'decode': []}
    floor = {'prefill': [], 'decode': []}
    for row in rows:
        ids = row['prompt_token_ids'] + row['generated']
        with torch.no_grad():
            logp = torch.log_softmax(model(torch.tensor([ids], device='cuda')).logits[0].float(), -1)
            bf16 = torch.log_softmax(noisy(torch.tensor([ids], device='cuda')).logits[0].float(), -1)
        n = len(row['prompt_token_ids'])
        pos = [('prefill', i, row['prompt_logprobs'][i], ids[i]) for i in range(1, n)]
        pos += [('decode', n + j, row['logprobs'][j], ids[n + j]) for j in range(len(row['generated']))]
        for phase, i, theirs, actual in pos:
            ref = logp[i - 1]
            top = torch.topk(ref, a.k).indices.tolist()
            theirs = {int(t): v for t, v in theirs.items()}
            mine = sorted(theirs, key=lambda t: -theirs[t])[:a.k]
            common = [t for t in theirs if t in set(top)]
            stats[phase].append({'actual': abs(theirs[actual] - float(ref[actual])) if actual in theirs else None,
                                 'top1': mine[0] == top[0], 'overlap': len(set(mine) & set(top)) / a.k,
                                 'max_common': max((abs(theirs[t] - float(ref[t])) for t in common), default=0.0)})
            other = bf16[i - 1]
            floor[phase].append({'actual': abs(float(other[actual]) - float(ref[actual])),
                                 'top1': int(other.argmax()) == top[0]})
    import base64
    import numpy as np
    law = []
    for row in engine['sampled']:
        ids = row['prompt_token_ids'] + row['generated']
        with torch.no_grad():
            logits = model(torch.tensor([ids], device='cuda')).logits[0].float()
        n = len(row['prompt_token_ids'])
        logp = torch.log_softmax(logits[n - 1:n - 1 + len(row['generated'])] / row['temperature'], -1)
        for j, (token, record) in enumerate(zip(row['generated'], row['records'])):
            raw = base64.b64decode(record)
            width = (len(raw) - 4) // 6
            mine_logp = float(np.frombuffer(raw[:4], '<f4')[0])
            mine_ids = np.frombuffer(raw[4:4 + 4 * width], '<i4').tolist()
            mine_lps = np.frombuffer(raw[4 + 4 * width:], '<f2').astype(np.float32)
            ref = logp[j]
            top = torch.topk(ref, width).indices.tolist()
            law.append({'sampled': abs(mine_logp - float(ref[token])), 'top1': mine_ids[0] == top[0],
                        'overlap': len(set(mine_ids) & set(top)) / width,
                        'topk_max': float(max(abs(v - float(ref[t])) for t, v in zip(mine_ids, mine_lps))),
                        'sorted': bool(np.all(np.diff(mine_lps) <= 0))})
    stats_law = {'positions': len(law), 'max_abs_sampled_logprob': max(r['sampled'] for r in law),
                 'mean_abs_sampled_logprob': sum(r['sampled'] for r in law) / len(law),
                 'top1_agreement': sum(r['top1'] for r in law) / len(law),
                 'topk_overlap': sum(r['overlap'] for r in law) / len(law),
                 'max_abs_topk_logprob': max(r['topk_max'] for r in law),
                 'sorted': all(r['sorted'] for r in law)}
    report = {'compact_capture': stats_law}
    noise = {phase: {'mean_abs_logprob': sum(r['actual'] for r in rs) / max(len(rs), 1),
                     'max_abs_logprob': max((r['actual'] for r in rs), default=0.0),
                     'top1_agreement': sum(r['top1'] for r in rs) / max(len(rs), 1)} for phase, rs in floor.items()}
    report['hf_bf16_noise_floor'] = noise
    for phase, rs in stats.items():
        act = [r['actual'] for r in rs if r['actual'] is not None]
        report[phase] = {'positions': len(rs), 'max_abs_logprob': max(act, default=0.0),
                         'mean_abs_logprob': sum(act) / max(len(act), 1),
                         'max_abs_topk_logprob': max((r['max_common'] for r in rs), default=0.0),
                         'top1_agreement': sum(r['top1'] for r in rs) / max(len(rs), 1),
                         'topk_overlap': sum(r['overlap'] for r in rs) / max(len(rs), 1)}
    # A BF16 engine passes when it is no further from FP32 than HF's own BF16 forward (with a small margin).
    fails = [f'{p}.mean_abs_logprob' for p in ('prefill', 'decode')
             if report[p]['mean_abs_logprob'] > 1.25 * noise[p]['mean_abs_logprob'] + 0.005]
    fails += [f'{p}.top1_agreement' for p in ('prefill', 'decode')
              if report[p]['top1_agreement'] < noise[p]['top1_agreement'] - 0.01]
    if (stats_law['max_abs_sampled_logprob'] > TOLERANCE['max_abs_logprob']
            or stats_law['mean_abs_sampled_logprob'] > TOLERANCE['mean_abs_logprob'] or not stats_law['sorted']):
        fails.append('compact_capture')
    out = {'model': a.model, 'engine_out': a.engine_out, 'prompts': len(rows), 'k': a.k, 'gen': a.gen, 'chunk': a.chunk,
           **{key: engine[key] for key in ('cute_gated_conv', 'device_capability', 'sglang')},
           'prompt_tokens': [len(r['prompt_token_ids']) for r in rows], 'tolerance': TOLERANCE,
           'report': report, 'failures': fails}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--phase', choices=['all', 'sglang', 'hf'], default='all')
    p.add_argument('--model', required=True)
    p.add_argument('--requests')
    p.add_argument('--out')
    p.add_argument('--engine-out')
    p.add_argument('--sglang-python', help="default: training/neuralese/sglang-fork.json python")
    p.add_argument('--n', type=int, default=6)
    p.add_argument('--gen', type=int, default=48)
    p.add_argument('--k', type=int, default=20)
    p.add_argument('--chunk', type=int, default=2048)
    p.add_argument('--mem-fraction', type=float, default=0.15)
    p.add_argument('--disable-cuda-graph', action='store_true')
    p.add_argument('--engine-kw', default='{}', help='extra sglang.Engine keyword arguments (JSON)')
    a = p.parse_args()
    a.engine_out = a.engine_out or str(Path(a.out).with_suffix('.sglang.json'))
    if a.phase == 'sglang':
        sglang_phase(a)
        sys.exit(0)
    if a.phase == 'all':
        fork = json.loads((Path(__file__).resolve().parents[2] / 'training/neuralese/sglang-fork.json').read_text())
        argv = [arg for arg in sys.argv[1:] if not arg.startswith('--phase')]
        python = a.sglang_python or fork['python']
        # FlashInfer JIT-compiles kernels: the fork venv's ninja and the CUDA toolkit's nvcc on PATH.
        cuda = os.environ.get('CUDA_HOME', '/usr/local/cuda')
        path = os.pathsep.join([str(Path(python).parent), cuda + '/bin', os.environ.get('PATH', '')])
        subprocess.run([python, __file__, '--phase', 'sglang', '--engine-out', a.engine_out, *argv], check=True,
                       env={**os.environ, 'PYTHONPATH': '', 'PATH': path, 'CUDA_HOME': cuda})
    sys.exit(1 if hf_phase(a)['failures'] else 0)
