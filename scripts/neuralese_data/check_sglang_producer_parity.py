#!/usr/bin/env python3
"""Producer spans on natlang's SGLang fork against the native engine's own math (native_step with a producing mask).

usage: check_sglang_producer_parity.py --checkpoint CK --base DIR --requests REQ.jsonl --out OUT.json [--n 4]

The checkpoint's producer adapter is replaced in memory by a seeded random one (``--update-ratio``: its update norm
relative to the embedding norm), because trained lineages may still hold the identity adapter (``up`` = 0) and a
parity check of an identity adapter would pass with the adapter ignored. The perturbed model is exported with
``export.native_hf.export_loaded`` (producer table and markers included) and served by the fork. Cases are real natlang
turns (REQ.jsonl ``prompt_ids``) followed by spans:

- forced: prompt + text + open + span text + close + text, prefilled with input logprobs (extend-side mapping);
- decode: prompts ending right after an open marker, inside a span, and after a closed span, sampled at T=1 with the
  compact capture (decode-side mapping from the span state stored per request slot, through CUDA graphs); and inside a
  span with a produce_close logit bias (the reference adds the same bias), so decode crosses sampled close markers;
- a forced sequence split across prefill chunks (``--chunk``), so a span crosses a chunk boundary.

The reference runs each whole sequence through ``native_step`` with the producing mask the native engine applies
(tokens after an open marker until its close marker, markers themselves ordinary), in FP32; the same in BF16 is the
noise floor. A control reference without the producing mask shows the check would catch an ignored adapter. Fails when
SGLang is further from FP32 than native BF16 (mean above 1.25 x floor + 0.005, top-1 more than 0.01 below), or when the
control is not clearly separated (its mean below 4 x SGLang's on producing positions).
"""
import argparse, base64, json, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEED = 20261030
CLOSE_BIAS = (10.0, 14.0, 18.0, 22.0)  # an untrained close row is far below; one of these lands mid-range


def producing_mask(ids, opening, closing):
    inside, mask = False, []
    for token in ids:
        mask.append(inside and token not in (opening, closing))
        if token == opening:
            inside = True
        elif token == closing:
            inside = False
    return mask


def load(a, dtype):
    """The native model with the seeded perturbed producer adapter (drawn once, from the BF16 load, then reused)."""
    import torch
    from natlang_neuralese.model.native_model import load_native_checkpoint
    model, tokenizer, roles, config, _ = load_native_checkpoint(a.checkpoint, base=a.base, device='cuda', dtype=dtype,
                                                                allow_weights_only=True)
    saved = Path(a.work) / 'adapter-up.pt'
    with torch.no_grad():
        adapter = model.adapter
        if saved.exists():
            adapter.up.weight.copy_(torch.load(saved).to('cuda', adapter.up.weight.dtype))
            return model, tokenizer, roles, config
        generator = torch.Generator().manual_seed(SEED)
        up = torch.randn(adapter.up.weight.shape, generator=generator, dtype=torch.float32)
        # Scale against the actual embedding: E[|up(down(e))|] = ratio * E[|e|] over a sample of rows.
        sample = model.embed(torch.arange(0, model.port.embedding_weight.shape[0], 97, device='cuda')[None]).float()
        down = sample @ adapter.down.weight.float().T
        update = down @ up.to('cuda').T
        scale = a.update_ratio * sample.norm(dim=-1).mean() / update.norm(dim=-1).mean()
        adapter.up.weight.copy_((up.to('cuda') * scale).to(adapter.up.weight.dtype))
        torch.save(adapter.up.weight.detach().float().cpu(), saved)
    return model, tokenizer, roles, config


def native_phase(a):
    import torch
    from natlang_neuralese.export.native_hf import export_loaded
    model, tokenizer, roles, config = load(a, torch.bfloat16)
    export = Path(a.work) / 'export'
    export_loaded(model, tokenizer, roles, config, checkpoint=a.checkpoint,
                  checkpoint_sha256=f'perturbed-producer-{SEED}-{a.update_ratio}:{Path(a.checkpoint).name}',
                  out_dir=export, dtype=torch.bfloat16)
    opening, closing = int(roles.produce_open), int(roles.produce_close)
    text = lambda s: tokenizer.encode(s, add_special_tokens=False)
    span = text(' greenhouse humidity averaged 61.2 percent across the north sensors; the south bank read 58.7 '
                'percent and two sensors reported no data for the last six hours of the window.')
    prompts = [json.loads(l)['prompt_ids'] for l in open(a.requests)][:a.n]
    forced, decode = [], []
    body = text(' Let me record the readings.') + [opening] + span + [closing] + text(' The totals are recorded.')
    # One span crossing a prefill chunk boundary.
    forced.append({'prompt': prompts[0][:a.chunk - 20], 'continuation': body})
    for i, prompt in enumerate(prompts):
        forced.append({'prompt': prompt, 'continuation': body})
        head = prompt + text(' Recording:') + [opening]
        decode.append({'prompt': head})                                   # first decode input: the open marker
        decode.append({'prompt': head + span[:3]})                        # decode inside the span
        decode.append({'prompt': head + span + [closing] + text(' Next:')})  # decode after a closed span
        # Inside the span with a close-marker logit bias: decode crosses produce_close (and later stray closes).
        decode.append({'prompt': head + span[:3], 'logit_bias': {str(closing): CLOSE_BIAS[i % len(CLOSE_BIAS)]}})
    cases = {'opening': opening, 'closing': closing, 'export': str(export), 'forced': forced, 'decode': decode}
    (Path(a.work) / 'cases.json').write_text(json.dumps(cases))
    for stale in [*Path(a.work).glob('reference-*.json'), *Path(a.work).glob('native-*.safetensors')]:
        stale.unlink()  # reference caches belong to the previous cases


def sglang_phase(a):
    os.environ.setdefault('SGLANG_RETURN_ORIGINAL_LOGPROB', '1')
    import sglang as sgl
    cases = json.loads((Path(a.work) / 'cases.json').read_text())
    eng = sgl.Engine(model_path=cases['export'], dtype='bfloat16', context_length=16384, chunked_prefill_size=a.chunk,
                     random_seed=0, log_level='warning', trust_remote_code=True, mem_fraction_static=0.3,
                     max_running_requests=8, max_total_tokens=65536, max_mamba_cache_size=16,
                     cuda_graph_max_bs_decode=8)
    forced = []
    for case in cases['forced']:
        ids = case['prompt'] + case['continuation']
        out = eng.generate(input_ids=ids, sampling_params={'temperature': 0.0, 'max_new_tokens': 1},
                           return_logprob=True, logprob_start_len=len(case['prompt']) - 1)
        forced.append([lp for lp, _, _ in out['meta_info']['input_token_logprobs']][1:])
    # Native-state export of each forced span (custom_params.natlang_capture, as serve/sglang_actor.py requests it),
    # after the prefill above cached the whole sequence: the logprob_start_len cap makes SGLang recompute the span.
    opening, closing = cases['opening'], cases['closing']
    for i, case in enumerate(cases['forced']):
        ids = case['prompt'] + case['continuation']
        start = ids.index(opening, len(case['prompt'])) + 1
        end = ids.index(closing, start)
        spans = [(start, end)] + ([(start, start)] if i == 0 else [])  # and one empty span (layout only)
        for j, (lo, hi) in enumerate(spans):
            path = str(Path(a.work) / f'capture-{i}-{j}.safetensors')
            if os.path.exists(path):
                os.unlink(path)
            eng.generate(input_ids=ids[:max(hi, lo)], return_logprob=True, logprob_start_len=lo - 1,
                         sampling_params={'temperature': 0.0, 'max_new_tokens': 1,
                                          'custom_params': {'natlang_capture': {'path': path, 'start': lo, 'end': hi}}})
    decode = []
    prompts = [case['prompt'] for case in cases['decode']]
    outs = eng.generate(input_ids=prompts, sampling_params=[{'temperature': 1.0, 'max_new_tokens': a.gen,
                        'ignore_eos': True, 'sampling_seed': SEED + i, 'custom_params': {'natlang_top_k': 20},
                        **({'logit_bias': case['logit_bias']} if 'logit_bias' in case else {})}
                        for i, case in enumerate(cases['decode'])])
    for out in outs:
        decode.append({'generated': out['output_ids'], 'records': out['meta_info']['natlang_top_k']})
    eng.shutdown()
    (Path(a.work) / 'sglang.json').write_text(json.dumps({'forced': forced, 'decode': decode}))


def export_report(a, cases, failures):
    """The fork's span export against native_step's capture of the same span (FP32 reference, BF16 floor)."""
    import torch
    from safetensors.torch import load_file
    from natlang_neuralese.model.shared_state import native_step
    opening, closing = cases['opening'], cases['closing']
    errors = {'engine': {}, 'floor': {}}
    shapes_ok, empty_ok = True, None
    from safetensors.torch import save_file
    for name, dtype in (('fp32', torch.float32), ('bf16', torch.bfloat16)):
        # Cached per dtype and span, so a preempted run resumes without recomputing.
        if all((Path(a.work) / f'native-{name}-{i}.safetensors').exists() for i in range(len(cases['forced']))):
            continue
        model = load(a, dtype)[0]
        for i, case in enumerate(cases['forced']):
            ids = case['prompt'] + case['continuation']
            start = ids.index(opening, len(case['prompt'])) + 1
            end = ids.index(closing, start)
            with torch.no_grad():
                out = native_step(model, torch.tensor([ids[:end]], device='cuda'),
                                  producing=torch.tensor([producing_mask(ids[:end], opening, closing)], device='cuda'))
            tensors = {'embeddings': out['embeddings'][start:end].float().cpu()}
            for index, layer in enumerate(out['layers']):
                for key, value in layer.items():
                    if key != 'kind':
                        tensors[f'layers.{index}.{key}'] = value[start:end].float().cpu().contiguous()
            save_file(tensors, str(Path(a.work) / f'native-{name}-{i}.safetensors'))
        del model
        torch.cuda.empty_cache()
    for i, case in enumerate(cases['forced']):
        ids = case['prompt'] + case['continuation']
        start = ids.index(opening, len(case['prompt'])) + 1
        end = ids.index(closing, start)
        mine = load_file(str(Path(a.work) / f'capture-{i}-0.safetensors'))
        native = {name: load_file(str(Path(a.work) / f'native-{name}-{i}.safetensors')) for name in ('fp32', 'bf16')}
        shapes_ok &= (torch.equal(mine['positions'], torch.arange(start, end))
                      and mine['token_ids'].tolist() == ids[start:end]
                      and set(native['fp32']) == set(mine) - {'positions', 'token_ids'})
        for key, ref in native['fp32'].items():
            kind = key.rsplit('.', 1)[-1]
            scale = ref.norm().item() or 1.0
            errors['engine'].setdefault(kind, []).append((mine[key].float() - ref).norm().item() / scale)
            errors['floor'].setdefault(kind, []).append((native['bf16'][key] - ref).norm().item() / scale)
            shapes_ok &= mine[key].shape == ref.shape
        if i == 0:
            empty = load_file(str(Path(a.work) / f'capture-{i}-1.safetensors'))
            empty_ok = (set(empty) == set(mine) and all(empty[key].shape[0] == 0 and empty[key].shape[1:] == mine[key].shape[1:]
                                                          for key in mine))
    report = {kind: {'relative_l2_mean': sum(v) / len(v), 'relative_l2_max': max(v),
                     'native_bf16_floor_mean': sum(errors['floor'][kind]) / len(v),
                     'native_bf16_floor_max': max(errors['floor'][kind])} for kind, v in errors['engine'].items()}
    for kind, r in report.items():
        if r['relative_l2_mean'] > 1.5 * r['native_bf16_floor_mean'] + 1e-3:
            failures.append(f'export.{kind}')
    if not shapes_ok:
        failures.append('export.layout')
    if not empty_ok:
        failures.append('export.empty_span')
    return {'spans': len(cases['forced']), 'layout_ok': shapes_ok, 'empty_span_ok': empty_ok, 'tensors': report}


def reference_phase(a):
    import numpy as np
    import torch
    from natlang_neuralese.model.shared_state import native_step
    cases = json.loads((Path(a.work) / 'cases.json').read_text())
    engine = json.loads((Path(a.work) / 'sglang.json').read_text())
    opening, closing = cases['opening'], cases['closing']
    sequences = []  # (ids, first scored position, sglang logps of ids[first:])
    for case, logps in zip(cases['forced'], engine['forced']):
        ids = case['prompt'] + case['continuation']
        sequences.append((ids, len(case['prompt']), logps, 'forced', None))
    for case, row in zip(cases['decode'], engine['decode']):
        ids = case['prompt'] + row['generated']
        sampled = [float(np.frombuffer(base64.b64decode(r)[:4], '<f4')[0]) for r in row['records']]
        sequences.append((ids, len(case['prompt']), sampled, 'decode', case.get('logit_bias')))

    def logprobs(model, ids, mask, bias, first):
        """Row r: the log-law predicting ids[first + r] (only scored positions reach the vocabulary head)."""
        with torch.no_grad():
            tensor = torch.tensor([ids], device='cuda')
            out = native_step(model, tensor, producing=torch.tensor([mask], device='cuda'))
            logits = model.logits(out['h_final'][:, first - 1:-1])[0].float()
            for token, value in (bias or {}).items():
                logits[:, int(token)] += value
            return torch.log_softmax(logits, -1)

    results = {}
    for name, dtype in (('fp32', torch.float32), ('bf16', torch.bfloat16)):
        cached = Path(a.work) / f'reference-{name}.json'
        if cached.exists():
            results[name] = json.loads(cached.read_text())
            continue
        model, *_ = load(a, dtype)
        rows = []
        for ids, first, _, _, bias in sequences:
            mask = producing_mask(ids, opening, closing)
            ref = logprobs(model, ids, mask, bias, first)
            control = logprobs(model, ids, [False] * len(ids), bias, first) if name == 'fp32' else None
            rows.append({'ref': [float(ref[i - first, ids[i]]) for i in range(first, len(ids))],
                         'control': [float(control[i - first, ids[i]]) for i in range(first, len(ids))] if control is not None else None})
        results[name] = rows
        cached.write_text(json.dumps(rows))
        del model
        torch.cuda.empty_cache()
    report = {}
    for kind in ('forced', 'decode'):
        for subset in ('producing', 'ordinary'):
            engine_d, floor_d, control_d = [], [], []
            for (ids, first, theirs, k, _), fp32, bf16 in zip(sequences, results['fp32'], results['bf16']):
                if k != kind:
                    continue
                mask = producing_mask(ids, opening, closing)
                # The position predicting ids[i] consumed ids[i-1]: classify by whether that input was producing.
                for j, i in enumerate(range(first, len(ids))):
                    if mask[i - 1] != (subset == 'producing'):
                        continue
                    engine_d.append(abs(theirs[j] - fp32['ref'][j]))
                    floor_d.append(abs(bf16['ref'][j] - fp32['ref'][j]))
                    control_d.append(abs(fp32['control'][j] - fp32['ref'][j]))
            if engine_d:
                report[f'{kind}.{subset}'] = {
                    'positions': len(engine_d), 'mean_abs_logprob': float(np.mean(engine_d)),
                    'max_abs_logprob': float(np.max(engine_d)), 'native_bf16_floor_mean': float(np.mean(floor_d)),
                    'native_bf16_floor_max': float(np.max(floor_d)), 'no_adapter_control_mean': float(np.mean(control_d))}
    # Decode steps that crossed a sampled produce_close (the span flag updated on the GPU between graph replays).
    closes = sum(1 for ids, first, _, k, _ in sequences if k == 'decode'
                 for i in range(first, len(ids)) if ids[i] == closing and producing_mask(ids, opening, closing)[i - 1])
    report['decode.sampled_span_closes'] = closes
    failures = []
    report['export'] = export_report(a, cases, failures)
    if not closes:
        failures.append('decode.no_sampled_span_close')
    for key, r in report.items():
        if not isinstance(r, dict) or 'mean_abs_logprob' not in r:
            continue
        if r['mean_abs_logprob'] > 1.25 * r['native_bf16_floor_mean'] + 0.005:
            failures.append(f'{key}.mean_abs_logprob')
        if key.endswith('.producing') and r['no_adapter_control_mean'] < 4 * r['mean_abs_logprob']:
            failures.append(f'{key}.control_not_separated')
    for kind in ('forced', 'decode'):
        if f'{kind}.producing' not in report:
            failures.append(f'{kind}.no_producing_positions')
    out = {'checkpoint': a.checkpoint, 'update_ratio': a.update_ratio, 'seed': SEED, 'chunk': a.chunk,
           'prompt_tokens': [len(c['prompt']) for c in cases['forced']], 'report': report, 'failures': failures}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--phase', choices=['all', 'native', 'sglang', 'reference'], default='all')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--base')
    p.add_argument('--requests', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--work')
    p.add_argument('--n', type=int, default=4)
    p.add_argument('--gen', type=int, default=64)
    p.add_argument('--chunk', type=int, default=2048)
    p.add_argument('--update-ratio', type=float, default=0.5)
    p.add_argument('--sglang-pythonpath', default='', help='PYTHONPATH for the SGLang phase (a fork worktree)')
    a = p.parse_args()
    a.work = a.work or str(Path(a.out).with_suffix('.work'))
    if a.phase != 'sglang':
        sys.path.insert(0, str(ROOT / 'training/neuralese'))
    Path(a.work).mkdir(parents=True, exist_ok=True)
    if a.phase == 'sglang':
        sglang_phase(a)
        sys.exit(0)
    if a.phase == 'native':
        native_phase(a)
    if a.phase == 'all':
        # Each phase in its own process, so no model stays resident while the next one loads.
        argv = [arg for arg in sys.argv[1:] if not arg.startswith('--phase')]
        subprocess.run([sys.executable, __file__, '--phase', 'native', '--work', a.work, *argv], check=True)
        fork = json.loads((ROOT / 'training/neuralese/sglang-fork.json').read_text())
        python = fork['python']
        cuda = os.environ.get('CUDA_HOME', '/usr/local/cuda')
        path = os.pathsep.join([str(Path(python).parent), cuda + '/bin', os.environ.get('PATH', '')])
        subprocess.run([python, __file__, '--phase', 'sglang', '--work', a.work, *argv], check=True,
                       env={**os.environ, 'PYTHONPATH': a.sglang_pythonpath, 'PATH': path, 'CUDA_HOME': cuda})
    if a.phase in ('all', 'reference'):
        sys.exit(1 if reference_phase(a)['failures'] else 0)
