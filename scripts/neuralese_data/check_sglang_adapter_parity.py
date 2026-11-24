"""Native adapters on the SGLang actor against the native engine's adapted forward (run_native + native_step), BF16.

usage: check_sglang_adapter_parity.py CHECKPOINT BASE WORK_DIR PORT [--release DIR --rows ROWS.jsonl [--prompts N]]
       [--adapter-rank R] [--kv-gb G]      (NATLANG_SGLANG_PYTHONPATH: a fork worktree under test)

One in-process SglangEngine front (its native model stays resident for the reference) with the fork's adapter slots,
launched with serve_native_checkpoint.py's own flags and defaults (`build_parser`), so the check runs the server's
configuration. Stacks: without --release, three random stacks (xs + free-lora on attention, conv and MLP matrices of
several layers); with --release, the program fit's own stack (`tune.build` output) on prompts of its rows. Adapted and
unadapted requests run concurrently through 2 slots (slot reuse), sampling at T=1 with behaviour capture, and once
greedily (T=0). Each sampled token's logp is compared with the native engine's log-softmax of the same token after the
same tokens under the same stack; top-1 agreement compares SGLang's top-k head with the native argmax at every sampled
position; greedy runs are compared token by token with the native greedy continuation of the same prompt ids. Control:
the native unadapted forward on the adapted samples (the adapter's effect, which parity must resolve). A stack whose
summed rank per module exceeds the slot rank falls back to the native engine; the report counts fallbacks."""
import argparse, base64, json, sys, types
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'training/neuralese')); sys.path.insert(0, str(ROOT / 'scripts/neuralese_data'))
import numpy as np
import torch
from natlang_neuralese.common.hashing import sha256_file_hex_cached
from natlang_neuralese.serve import load_native_engine
from natlang_neuralese.serve.engine import GenerationRequest
from natlang_neuralese.serve.sglang_actor import SglangEngine
from natlang_neuralese.serve.store import TensorStore
from natlang_neuralese.model.native_adapters import create_adapter, target_parameters
from natlang_neuralese.model.shared_state import native_step
from serve_native_checkpoint import build_parser, launch_sglang

cli = argparse.ArgumentParser()
cli.add_argument('checkpoint', type=Path)
cli.add_argument('base')
cli.add_argument('work', type=Path)
cli.add_argument('port', type=int)
cli.add_argument('--release', type=Path)
cli.add_argument('--rows', type=Path)
cli.add_argument('--prompts', type=int, default=6)
cli.add_argument('--tokens', type=int, default=48)
cli.add_argument('--adapter-rank', type=int, help="slot rank (default: serve_native_checkpoint.py's, from the release)")
cli.add_argument('--kv-gb', type=float, default=2.0)
cli.add_argument('--max-running', type=int, default=8)
cli.add_argument('--concurrency', type=int, default=8, help='requests in flight at once (the front threads)')
cli.add_argument('--merged', type=Path,
                 help='with --phase reference: score the adapted captures against this checkpoint unadapted (the release '
                      'merged into the weights, `program_tune.py merge`) instead of the native stack; report-merged.json')
cli.add_argument('--adapted-only', action='store_true', help='no unadapted requests (as an adapter arm serves)')
cli.add_argument('--server-arg', action='append', default=[],
                 help="extra serve_native_checkpoint.py argument (repeat; e.g. --server-arg=--sglang-kv-cache-dtype=fp8_e4m3)")
cli.add_argument('--sglang-python', help="the fork's Python (default: training/neuralese/sglang-fork.json's)")
cli.add_argument('--phase', choices=('both', 'collect', 'reference'), default='both',
                 help='with --release: collect runs SGLang alone (the front releases its native model once exported, as '
                      'serve_native_checkpoint.py serves) and saves the captures; reference loads the native engine in '
                      'a later process and scores them. Two small peaks instead of one large one on a shared host')
opts = cli.parse_args()
ck, base, work, port = opts.checkpoint, opts.base, opts.work, opts.port
work.mkdir(parents=True, exist_ok=True)
if opts.phase != 'both' and not opts.release:
    raise SystemExit('--phase collect/reference needs --release')
if opts.phase == 'reference':
    saved = json.loads((work / 'captures.json').read_text())
    from natlang_neuralese.tune.build import import_archive
    front = load_native_engine(str(opts.merged or ck), base=base, device='cuda', dtype=torch.bfloat16,
                               store=TensorStore(work / 'reference-store'))
    import_archive(front, [opts.release / 'artifacts'], [])
    kept = [j for j in saved['jobs'] if j['stack'] or not opts.merged]   # merged: the adapted captures only
    jobs = [(j['index'], j['stack'], j['temperature']) for j in kept]
    captures = [j['capture'] for j in kept]
else:
    # collect: the front loads on the CPU only to export (as Pop's actor does) and releases the model
    front = load_native_engine(str(ck), base=base, device='cpu' if opts.phase == 'collect' else 'cuda', dtype=torch.bfloat16,
                               engine_class=SglangEngine, store=TensorStore(work / 'store'))
if opts.phase != 'reference':
    server_argv = ['--checkpoint', str(ck), '--checkpoint-sha256', sha256_file_hex_cached(ck), '--port', str(port),
                   '--store-dir', str(work), '--hf-export-dir', str(work / 'export'), '--sglang-kv-gb', str(opts.kv_gb),
                   '--sglang-max-running', str(opts.max_running), '--sglang-context', '16384', '--sglang-adapter-slots', '2',
                   *(['--adapter-release', str(opts.release)] if opts.release else []),  # sizes the slot rank as served
                   *(['--sglang-python', opts.sglang_python] if opts.sglang_python else []),
                   *(['--sglang-adapter-rank', str(opts.adapter_rank)] if opts.adapter_rank else []), *opts.server_arg]
    args = build_parser().parse_args(server_argv)
    # collect: the front releases its native model once exported (serve_native_checkpoint.py's serving); a turn SGLang
    # cannot serve reloads it
    process = launch_sglang(front, args, ck, release={'checkpoint': str(ck), 'base': base, 'device': 'cuda', 'participant': None}
                            if opts.phase == 'collect' else None)
    front.start()
    messages_of = lambda text: {'messages': [{'role': 'user', 'content': text}], 'tools': None}
    if opts.release:
        from natlang_neuralese.tune.build import import_archive
        import_archive(front, [opts.release / 'artifacts'], [])
        result = json.loads((opts.release / 'result.json').read_text())
        stacks = [[{'id': item['artifact'], 'scale': 1.0} for item in result['artifacts']]]
        rows = [json.loads(line) for line in opts.rows.read_text().splitlines() if line.strip()]
        picks = sorted({round(i * (len(rows) - 1) / max(opts.prompts - 1, 1)) for i in range(opts.prompts)})
        prompts = [{'messages': rows[i]['messages'], 'tools': rows[i]['tools']} for i in picks]
    else:
        available = target_parameters(front.shared_model)
        names = [n for n in available if any(f'.layers.{i}.' in n for i in (0, 2, 4, 9))]
        stacks = []
        for seed in range(3):
            made = []
            for kind in ('xs', 'free-lora'):
                artifact = create_adapter(front, {'kind': kind, 'rank': 4, 'seed': seed * 2 + (kind == 'xs'), 'parameter_names': names}).artifact
                g = torch.Generator().manual_seed(100 + seed)
                values = {}
                for key, value in artifact.tensors.items():
                    scale = 0.6 if kind == 'xs' else (0.02 if key.startswith('lora_B.') else 1.0)
                    values[key] = torch.randn(value.shape, generator=g) * scale if kind == 'xs' or key.startswith('lora_B.') else value
                made.append(front.store.put_native_tensor_artifact(replace(artifact, tensors=values)).id)
            stacks.append([{'id': made[0], 'scale': 1.0}, {'id': made[1], 'scale': 0.5 + seed}])
        prompts = [messages_of(text) for text in (
            'Explain in two sentences why greenhouses need ventilation.', 'List three prime numbers and add them.',
            'Write a TypeScript function that returns the largest element of an array.', 'What does a conv layer do?')]
    jobs = [(i, s, t) for i in range(len(prompts)) for s in (*stacks, *(() if opts.adapted_only else (None,)))
            for t in (1.0, 0.0)]


    def run(job):
        index, stack, temperature = job
        request = GenerationRequest(messages=prompts[index]['messages'], tools=prompts[index]['tools'], temperature=temperature,
                                    max_tokens=opts.tokens, rl_capture=True, rl_top_k=20, native_recording=True, seed=7,
                                    adapters=stack if stack else [])
        return front.generate(request)['x_natlang_rl_capture']


    with ThreadPoolExecutor(opts.concurrency) as pool:
        captures = list(pool.map(run, jobs))
    if opts.phase == 'collect':
        (work / 'captures.json').write_text(json.dumps({
            'metrics': {k: v for k, v in front.progress_metrics().items() if 'sglang' in k or 'fallback' in k},
            'slot_rank': front.sglang_adapter_rank, 'stack_ranks': [front._sglang_adapter_file(stack)[1] for stack in stacks],
            'jobs': [{'index': i, 'stack': st, 'temperature': t, 'capture': c} for (i, st, t), c in zip(jobs, captures)]}))
        front.stop()
        import os, signal
        os.killpg(process.pid, signal.SIGTERM)
        print('collected', len(captures), flush=True)
        raise SystemExit(0)


def full(ids, stack, first=1):
    if opts.merged:
        stack = None   # the merged weights carry the release
    """log-softmax rows predicting ids[first:] and the token after ids, under the stack (the native engine's exact
    application); only those rows go through the head (a long prompt's full logits are GBs at LFM3's vocabulary)."""
    with torch.no_grad():
        out = front._native_run(lambda: front.shared_model.logits(
            native_step(front.shared_model, torch.tensor([ids], device='cuda'))['h_final'][:, first - 1:])[0].float(),
            stack if stack else [])
    return torch.log_softmax(out, -1)


def reference(ids, stack, first):
    return full(ids, stack, first)[:-1]


def native_greedy(ids, stack, steps):
    ids = list(ids)
    for _ in range(steps):
        ids.append(int(full(ids, stack, len(ids))[-1].argmax()))
    return ids


rows = {'adapted': [], 'unadapted': [], 'control': []}
top1 = {'adapted': [], 'unadapted': []}
greedy = {'adapted': [], 'unadapted': []}
for (index, stack, temperature), capture in zip(jobs, captures):
    key = 'adapted' if stack else 'unadapted'
    assert (capture.get('adapter_binding') is not None) == bool(stack)
    prompt_ids = [t for seg in capture['prompt'] for t in seg.get('token_ids', [])]
    fed = [x['token_id'] for x in capture['generated_inputs']]
    samples = [x for x in capture['samples'] if not x['terminal']][:len(fed)]
    if temperature == 0:
        got = [x['token_id'] for x in capture['samples']]
        want = native_greedy(prompt_ids, stack, len(got))[len(prompt_ids):]
        same = next((j for j, (a, b) in enumerate(zip(got, want)) if a != b), len(got))
        greedy[key].append({'prompt': index, 'agree': same / max(len(got), 1), 'tokens': len(got)})
        continue
    ids = prompt_ids + fed
    ref = reference(ids, stack, len(prompt_ids))
    rows[key] += [abs(s['logp'] - float(ref[j, s['token_id']])) for j, s in enumerate(samples)]
    for j, s in enumerate(samples):
        if 'top_k' in s:
            head = int(np.frombuffer(base64.b64decode(s['top_k']['ids']), '<i4')[0])
            top1[key].append(head == int(ref[j].argmax()))
    if stack:
        plain = reference(ids, None, len(prompt_ids))
        rows['control'] += [abs(float(plain[j, s['token_id']]) - float(ref[j, s['token_id']])) for j, s in enumerate(samples)]
mean = lambda v: sum(v) / max(len(v), 1)
report = {k: {'positions': len(v), 'mean_abs_logprob': mean(v), 'max_abs_logprob': max(v, default=0.)} for k, v in rows.items()}
for key in top1:
    report[key]['top1_agreement'] = mean(top1[key])
    report[key]['greedy'] = greedy[key]
    report[key]['greedy_agreement'] = mean([g['agree'] for g in greedy[key]])
if opts.phase == 'reference':
    metrics = saved['metrics']
    report['slot_rank'], report['stack_ranks'] = saved['slot_rank'], saved['stack_ranks']
else:
    metrics = front.progress_metrics()
    report['slot_rank'] = front.sglang_adapter_rank
    report['stack_ranks'] = [front._sglang_adapter_file(stack)[1] for stack in stacks]
report['metrics'] = {k: v for k, v in metrics.items() if 'sglang' in k or 'fallback' in k}
report['failures'] = ([] if report['adapted']['mean_abs_logprob'] <= 1.5 * report['unadapted']['mean_abs_logprob'] + 0.005 else ['adapted'])
report['failures'] += [] if report['control']['mean_abs_logprob'] > 4 * report['adapted']['mean_abs_logprob'] else ['control_not_separated']
report['failures'] += [] if metrics.get('natlang_native_fallback_requests_total', 0) == 0 else ['fallbacks']
(work / ('report-merged.json' if opts.merged else 'report.json')).write_text(json.dumps(report, indent=1))
print(json.dumps(report, indent=1), flush=True)
if opts.phase == 'both':
    front.stop()
    import os, signal
    os.killpg(process.pid, signal.SIGTERM)
