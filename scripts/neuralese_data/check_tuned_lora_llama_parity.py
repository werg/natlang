"""Tuned LoRA on llama.cpp (and the page) equals the native adapted model.

usage: check_tuned_lora_llama_parity.py CHECKPOINT OUT_DIR LLAMA_SERVER RELEASE [--base HF_DIR] [--rows ROWS.jsonl]
           [--prompts N] [--tokens T] [--outtypes f32,q8_0] [--ngl N] [--threads N] [--skip-llama] [--page FILE ...]

RELEASE (a `tune.build` output: result.json, artifacts/) is served as fitted: the fit's own adapter stack, every target
it trains (on LFM3 the attention gate_proj and o_proj included), through the tuner's exact export
(`tune.export.export_peft` -> `export_gguf`, the vendored `convert_lora_to_gguf.py`).

Phases (each cached in OUT_DIR):
1. Base GGUF: the native checkpoint's f32 export (`export.native_gguf`); other outtypes are quantized from it with the
   server's own `llama-quantize` (`q8_0`: the file the page can hold).
2. PEFT + GGUF LoRA of the stack (no model load: the release's artifacts are decoded with `tune.stack_export`).
3. Prompts: with ROWS (`natlang.tune-turn/v1`), N rows rendered with the serving template (`serve.targets.target_items`,
   the tuner's own rendering) as token ids, else one fixed sentence; `prompts.json`.
4. llama-server, launched as the managed server launches tuned runs (`--lora-init-without-apply --lora-per-sequence
   --lora LORA`), per outtype: each prompt greedily for T tokens through `/v1/completions` with `logprobs`, the
   adapter off (`lora` scale 0) and on (scale 1): `candidates/llama-<outtype>.json`.
5. Score (the native model loaded once, at the dtype the release binds: BF16 for LFM3 fits): for every candidate file (llama's and each `--page` file written by
   `ts-host/scripts/wllama-lora-smoke.mjs --parity`), one native forward over the prompt and the candidate's own
   tokens, with the release applied exactly (`run_native`) for adapted runs. Per position: |candidate logprob - native
   logprob| of the candidate's token; greedy agreement = the candidate's token is the native argmax (the first
   disagreement is reported with the native top-2 margin there, so a near-tie flip is told apart from a defect).
   Adapter-applied check: native adapted vs native base on the same tokens (the effect the candidate must reproduce)
   and whether the candidate's adapted continuation differs from its base continuation where the native ones do.
   With RELEASE, the export itself is checked: the PEFT's dense deltas added to the weights against `run_native`.
"""
import argparse, json, math, subprocess, sys, time, urllib.request
from pathlib import Path

import torch

parser = argparse.ArgumentParser()
parser.add_argument('checkpoint', type=Path)
parser.add_argument('out', type=Path)
parser.add_argument('server')
parser.add_argument('release', type=Path)
parser.add_argument('--base', help='HF base directory (LFM3: the pinned snapshot)')
parser.add_argument('--rows', type=Path, help='natlang.tune-turn/v1 rows whose prompts are compared')
parser.add_argument('--prompts', type=int, default=6)
parser.add_argument('--tokens', type=int, default=24)
parser.add_argument('--outtypes', default='f32', help='comma-separated base GGUF types (f32, q8_0, ...)')
parser.add_argument('--ngl', default='0', help='llama-server GPU layers (0: CPU)')
parser.add_argument('--threads', default='8')
parser.add_argument('--top', type=int, default=5, help='top logprobs requested per position')
parser.add_argument('--skip-llama', action='store_true', help='reuse candidates/llama-*.json')
parser.add_argument('--page', type=Path, action='append', default=[], help='wllama-lora-smoke.mjs --parity output')
parser.add_argument('--device', default='cuda')
parser.add_argument('--native-greedy', action=argparse.BooleanOptionalAction, default=True,
                    help="also decode the native reference greedily (2 x prompts x tokens forwards; the candidates' "
                         'agreement is scored on their own paths either way)')
parser.add_argument('--native-dtype', default='bfloat16',
                    help="the reference engine's dtype: the release binds the identity it was fit at (tune.build "
                         "engine.dtype, bfloat16 on LFM3); another dtype is refused as a different live checkpoint")
cli = parser.parse_args()
out = cli.out
out.mkdir(parents=True, exist_ok=True)
(out / 'candidates').mkdir(exist_ok=True)
outtypes = [item for item in cli.outtypes.split(',') if item]


def base_gguf():
    from natlang_neuralese.export.native_gguf import export as export_native_gguf
    if not ((out / 'base' / 'model-f32.gguf').exists() and (out / 'base' / 'hf-config').exists()):
        export_native_gguf(cli.checkpoint, out / 'base', outtypes=('f32',), base=cli.base)
    for outtype in outtypes:
        target = out / 'base' / f'model-{outtype}.gguf'
        if not target.exists():
            quantize = Path(cli.server).with_name('llama-quantize')
            subprocess.run([str(quantize), str(out / 'base' / 'model-f32.gguf'), str(target), outtype.upper()], check=True,
                           stdout=subprocess.DEVNULL)


def stack_and_targets():
    """(stack factors, {native name: ModuleTarget}, {native name: class}, base identity) without loading a model."""
    from natlang_neuralese.tune.targets import module_map
    from natlang_neuralese.tune.stack import stack_factors
    from natlang_neuralese.tune.stack_export import load_artifacts, members_from_artifacts
    result = json.loads((cli.release / 'result.json').read_text())
    artifacts = load_artifacts(cli.release / 'artifacts')
    members, bases, shapes = members_from_artifacts(
        artifacts, [{'id': item['artifact'], 'scale': 1., 'node': item['node']} for item in result['artifacts']])
    targets, kinds = module_map(sorted(shapes))
    identity = result['receipt']['base_native_identity']
    return stack_factors(members, bases, shapes), targets, kinds, identity, result


def export_lora(stack, targets):
    from natlang_neuralese.tune.export import export_peft, export_gguf
    peft = out / 'lora.peft'
    if not (out / 'lora.gguf').exists() and not (cli.skip_llama and (peft / 'adapter_config.json').exists()):
        export_peft(stack, {name: targets[name] for name in stack if stack[name].rank}, peft, base_identity=identity,
                    base_model_name_or_path=str(out / 'base' / 'hf-config'))
        export_gguf(peft, out / 'base' / 'hf-config', out / 'lora.gguf')
    return peft


def logprob_items(reply):
    """[(token id, logprob, {id: logprob} top)] of a /v1/completions or /completion reply (llama.cpp's shapes)."""
    choice = (reply.get('choices') or [{}])[0]
    items = ((choice.get('logprobs') or {}).get('content')) or reply.get('completion_probabilities') or []
    rows = []
    for item in items:
        top = item.get('top_logprobs') or item.get('top_probs') or []
        rows.append((int(item['id']), float(item['logprob']) if 'logprob' in item else math.log(item['prob']),
                     {int(t['id']): float(t['logprob']) if 'logprob' in t else math.log(t['prob']) for t in top}))
    return rows


def llama_candidates(prompts, outtype):
    port = 18977
    args = [cli.server, '-m', str(out / 'base' / f'model-{outtype}.gguf'), '--port', str(port), '-c', str(max(
        len(p['ids']) for p in prompts) + cli.tokens + 64), '-ngl', cli.ngl, '-t', cli.threads, '--no-warmup', '-np', '1',
            '--lora-init-without-apply', '--lora-per-sequence', '--lora', str(out / 'lora.gguf')]
    log = open(out / f'llama-{outtype}.log', 'w')
    process = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
    try:
        for _ in range(600):
            try:
                urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=1)
                break
            except Exception:
                if process.poll() is not None:
                    raise SystemExit(f'llama-server exited {process.returncode}; see {log.name}')
                time.sleep(0.5)
        runs = []
        for prompt in prompts:
            for scale in (0., 1.):
                body = json.dumps({'prompt': prompt['ids'], 'max_tokens': cli.tokens, 'temperature': 0, 'logprobs': cli.top,
                                   'cache_prompt': False, 'lora': [{'id': 0, 'scale': scale}]}).encode()
                started = time.time()
                reply = json.load(urllib.request.urlopen(urllib.request.Request(
                    f'http://127.0.0.1:{port}/v1/completions', body, {'content-type': 'application/json'}), timeout=1800))
                items = logprob_items(reply)
                runs.append({'prompt': prompt['index'], 'adapted': bool(scale), 'tokens': [t for t, _, _ in items],
                             'logprobs': [lp for _, lp, _ in items], 'top': [{str(k): v for k, v in top.items()} for *_, top in items],
                             'seconds': time.time() - started})
                print(json.dumps({'engine': f'llama-{outtype}', 'prompt': prompt['index'], 'adapted': bool(scale),
                                  'tokens': len(items), 'seconds': round(time.time() - started, 1)}), flush=True)
        return {'engine': f'llama-server {outtype}', 'server': cli.server, 'base_gguf': f'model-{outtype}.gguf', 'runs': runs}
    finally:
        process.terminate()
        process.wait(30)


stack, targets, kinds, identity, result = stack_and_targets()
if not cli.skip_llama:  # scoring alone (another host: the candidates, prompts and PEFT shipped) needs no base GGUF
    base_gguf()
peft = export_lora(stack, targets)
print('targets', {kind: sum(1 for name in stack if stack[name].rank and kinds[name] == kind) for kind in sorted(set(kinds.values()))},
      flush=True)

# Prompts: rendered once with the native engine's tokenizer and template (the engine is needed for scoring anyway).
from natlang_neuralese.serve import load_native_engine
from natlang_neuralese.model.shared_state import native_step
engine = None
prompts_path = out / 'prompts.json'
if not prompts_path.exists():
    engine = load_native_engine(str(cli.checkpoint), base=cli.base, device=cli.device, dtype=getattr(torch, cli.native_dtype))
    from natlang_neuralese.serve.targets import target_items
    from natlang_neuralese.tune.rows import reply_message
    rows = [json.loads(line) for line in cli.rows.read_text().splitlines() if line.strip()] if cli.rows else []
    picks = sorted({round(i * (len(rows) - 1) / max(cli.prompts - 1, 1)) for i in range(cli.prompts)}) if rows else []
    prompts = []
    for index in picks:
        row = rows[index]
        before, target = target_items(engine, row['messages'], row['tools'], reply_message(row['reply']))
        if any(kind != 'tok' for kind, _ in before + target):
            continue
        prompts.append({'index': index, 'root': row['root'], 'ids': [v for _, v in before], 'target': [v for _, v in target]})
    if not prompts:
        prompts = [{'index': 0, 'root': 'sentence', 'target': [],
                    'ids': engine.tokenizer.encode('Write one sentence about the sea and ships:', add_special_tokens=False)}]
    prompts_path.write_text(json.dumps({'checkpoint': str(cli.checkpoint), 'tokens': cli.tokens, 'prompts': prompts}))
prompts_file = json.loads(prompts_path.read_text())
if any('text' not in p for p in prompts_file['prompts']):
    # The page's wllama takes a text prompt only: each prompt's ids as text (special tokens written out). That the
    # engines' tokenizer reads it back as the same ids is checked against llama-server's /tokenize (`text_tokens`).
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(out / 'base' / 'hf-config' / 'tokenizer.json'))
    for p in prompts_file['prompts']:
        p['text'] = tokenizer.decode(p['ids'], skip_special_tokens=False)
    prompts_path.write_text(json.dumps(prompts_file))
prompts = prompts_file['prompts']
print('prompts', [(p['index'], len(p['ids'])) for p in prompts], flush=True)


def text_tokens():
    """{prompt index: ids llama.cpp's tokenizer reads from the prompt's text} (completion requests parse special tokens
    and add BOS only when the GGUF says so; /tokenize with add_special and parse_special is that tokenization)."""
    path = out / 'text-tokens.json'
    if path.exists():
        return {int(k): v for k, v in json.loads(path.read_text()).items()}
    port = 18978
    process = subprocess.Popen([cli.server, '-m', str(out / 'base' / f'model-{outtypes[-1]}.gguf'), '--port', str(port),
                                '-c', '512', '-t', '2', '--no-warmup'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(600):
            try:
                urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=1)
                break
            except Exception:
                time.sleep(0.5)
        found = {}
        for p in prompts:
            body = json.dumps({'content': p['text'], 'add_special': True, 'parse_special': True}).encode()
            found[p['index']] = json.load(urllib.request.urlopen(urllib.request.Request(
                f'http://127.0.0.1:{port}/tokenize', body, {'content-type': 'application/json'}), timeout=120))['tokens']
    finally:
        process.terminate()
        process.wait(30)
    path.write_text(json.dumps(found))
    return found

if not cli.skip_llama:
    if engine is not None:  # free the model while llama-server runs
        del engine
        torch.cuda.empty_cache()
        engine = None
    for outtype in outtypes:  # each outtype's candidates are kept once written (a preempted run resumes)
        if not (out / 'candidates' / f'llama-{outtype}.json').exists():
            (out / 'candidates' / f'llama-{outtype}.json').write_text(json.dumps(llama_candidates(prompts, outtype)))

if engine is None:
    engine = load_native_engine(str(cli.checkpoint), base=cli.base, device=cli.device, dtype=getattr(torch, cli.native_dtype))
from natlang_neuralese.tune.build import import_archive
import_archive(engine, [cli.release / 'artifacts'], [])
adapters = [{'id': item['artifact'], 'scale': 1.0} for item in result['artifacts']]
model = engine.shared_model
device = next(model.parameters()).device


def native_logprobs(ids, stack_entries, start=0):
    """log-softmax rows predicting ids[start + 1:] (float32), under the release (exact native application) or none. Only
    those rows go through the head (a 7k-token prompt's full logits are 3.5 GB at LFM3's vocabulary)."""
    def forward():
        with torch.no_grad():
            hidden = native_step(model, torch.tensor([ids], device=device))['h_final'][:, start:-1]
            return torch.log_softmax(model.logits(hidden)[0].float(), -1)
    return engine._native_run(forward, stack_entries or [])


def next_logprobs(ids, stack_entries):
    """The log-softmax row after the last of ids."""
    def forward():
        with torch.no_grad():
            hidden = native_step(model, torch.tensor([ids], device=device))['h_final'][:, -1:]
            return torch.log_softmax(model.logits(hidden)[0].float(), -1)[-1]
    return engine._native_run(forward, stack_entries or [])


def score(candidate):
    by_prompt = {p['index']: p for p in prompts}
    rows, effects, agreement, first, applied = [], [], [], [], []
    plain = {}
    for run in candidate['runs']:
        prompt = by_prompt[run['prompt']]
        # A text prompt (the page) is scored on the ids the engines' tokenizer reads from it.
        prompt_ids = text_tokens()[run['prompt']] if run.get('prompt_kind') == 'text' else prompt['ids']
        n = len(prompt_ids)
        ids = prompt_ids + run['tokens']
        tuned = native_logprobs(ids + [0], adapters if run['adapted'] else None, n - 1)[:len(run['tokens'])]
        diffs, agree = [], 0
        mismatch = None
        for j, (token, lp) in enumerate(zip(run['tokens'], run['logprobs'])):
            diffs.append(abs(lp - float(tuned[j, token])))
            top2 = torch.topk(tuned[j], 2)
            if mismatch is None:
                if int(top2.indices[0]) == token:
                    agree += 1
                else:
                    mismatch = {'position': j, 'native_margin': float(top2.values[0] - top2.values[1]),
                                'candidate_token_native_gap': float(top2.values[0] - tuned[j, token])}
        rows += diffs
        agreement.append(agree / max(len(run['tokens']), 1))
        first.append(mismatch)
        if run['adapted']:
            base = native_logprobs(ids + [0], None, n - 1)[:len(run['tokens'])]
            effects += [abs(float(tuned[j, t]) - float(base[j, t])) for j, t in enumerate(run['tokens'])]
        plain.setdefault(run['prompt'], {})[run['adapted']] = run['tokens']
    for index, pair in plain.items():
        applied.append(pair.get(True) != pair.get(False))
    mean = lambda v: sum(v) / max(len(v), 1)
    texts = sorted({run['prompt'] for run in candidate['runs'] if run.get('prompt_kind') == 'text'})
    return {'engine': candidate['engine'], 'positions': len(rows), 'max_abs_dlogprob': max(rows, default=0.),
            'text_prompts_tokenize_to_ids': all(text_tokens()[i] == by_prompt[i]['ids'] for i in texts) if texts else None,
            'mean_abs_dlogprob': mean(rows), 'greedy_agreement': mean(agreement),
            'all_greedy_equal': all(m is None for m in first), 'first_disagreements': [m for m in first if m],
            'adapted_effect_native_mean_abs_dlogprob': mean(effects),
            'adapter_changes_candidate_output': f'{sum(applied)}/{len(applied)}',
            'by_run': [{'prompt': r['prompt'], 'adapted': r['adapted'], 'agree': a, 'first_disagreement': m}
                       for r, a, m in zip(candidate['runs'], agreement, first)]}


report = {'release': str(cli.release), 'base_identity': identity, 'prompts': [(p['index'], len(p['ids'])) for p in prompts],
          'targets': {kind: sum(1 for name in stack if stack[name].rank and kinds[name] == kind) for kind in sorted(set(kinds.values()))}}

# Native continuation: greedy of base and adapted (the adapter must change what the reference says), and the fit target
# under both (the fit's own tokens: the adapter must raise their likelihood).
native_runs = []
for prompt in (prompts if cli.native_greedy else []):
    for adapted in (False, True):
        ids = list(prompt['ids'])
        for _ in range(cli.tokens):
            ids.append(int(next_logprobs(ids, adapters if adapted else None).argmax()))
        native_runs.append({'prompt': prompt['index'], 'adapted': adapted, 'tokens': ids[len(prompt['ids']):]})
pairs = {}
for run in native_runs:
    pairs.setdefault(run['prompt'], {})[run['adapted']] = run['tokens']
if pairs:
    report['native_adapter_changes_greedy'] = f"{sum(p[True] != p[False] for p in pairs.values())}/{len(pairs)}"
target_nll = {'base': [], 'adapted': []}
for prompt in prompts:
    if not prompt['target']:
        continue
    n, ids = len(prompt['ids']), prompt['ids'] + prompt['target']
    for key, entries in (('base', None), ('adapted', adapters)):
        lp = native_logprobs(ids, entries, n - 1)
        target_nll[key].append(-float(lp.gather(1, torch.tensor(prompt['target'], device=lp.device)[:, None]).mean()))
report['native_target_mean_nll'] = {k: sum(v) / max(len(v), 1) for k, v in target_nll.items()}
(out / 'candidates' / 'native.json').write_text(json.dumps({'engine': 'native', 'runs': native_runs}))

# The export: the PEFT's dense deltas (as PEFT and llama.cpp apply them) equal the native stack's own low-rank deltas
# (`lowrank_factors` of the resolved stack, what `run_native` adds), in float64 on every exported target.
from natlang_neuralese.tune.export import load_peft_dense
from natlang_neuralese.model.native_adapters import lowrank_factors, resolve_adapters
dense = load_peft_dense(peft)
native = lowrank_factors(resolve_adapters(engine, adapters))
gaps = []
for name, target in targets.items():
    if not stack[name].rank:
        continue
    factor_b, factor_a = native[name]
    want = factor_b.double().cpu() @ factor_a.double().cpu()
    got = dense[target.hf_module] if target.orientation == 'out_in' else dense[target.hf_module].T
    gaps.append(float((got - want).abs().max() / want.abs().max()))
report['export_dense_vs_native_max_rel_gap'] = max(gaps)
report['export_targets_checked'] = len(gaps)

report['engines'] = []
(out / 'scores').mkdir(exist_ok=True)
for path in sorted((out / 'candidates').glob('llama-*.json')) + list(cli.page):
    cached = out / 'scores' / f'{Path(path).stem}.json'   # per engine: an interrupted scoring run resumes
    if not cached.exists():
        cached.write_text(json.dumps(score(json.loads(Path(path).read_text()))))
    report['engines'].append(json.loads(cached.read_text()))
(out / 'report.json').write_text(json.dumps(report, indent=1))
print(json.dumps({k: v for k, v in report.items() if k != 'engines'}, indent=1))
for item in report['engines']:
    print(json.dumps({k: v for k, v in item.items() if k != 'by_run'}), flush=True)
