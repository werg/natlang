#!/usr/bin/env python3
"""Differential check of the SGLang fork's LFM3 serving across two fork trees (e.g. full-length KV for every attention
layer vs hybrid SWA KV, where the sliding layers keep only their window).

usage: check_sglang_hybrid_swa.py pair --model EXPORT --work DIR [--old 70b1991e6] [--new ace8d6868] [--dtypes fp8_e4m3,auto]
       check_sglang_hybrid_swa.py run --model EXPORT --fork-path FORK/python|'' --out OUT.json [--kv-cache-dtype fp8_e4m3]
       check_sglang_hybrid_swa.py compare A.json B.json

`pair` (one ledger unit, ~9 GB measured 12-13 GB peak, GPU): detached worktrees of the fork checkout at both commits
(the Rust extensions copied from the checkout), `run` on each per KV dtype, `compare` into DIR/compare-<dtype>.json;
finished runs are kept, so a stopped pair resumes. Pass when every deterministic section (greedy, prefix_cold,
prefix_hit, neuralese_*_run*) shows equal greedy tokens and max |delta| within the BF16 floor (FP8, 2026-11-13: all exact).

Run it with the fork venv's python (training/neuralese/sglang-fork.json). `run` launches `sglang.launch_server` from
the given fork tree (PYTHONPATH; '' = the venv's editable install) with the actor's serving flags (extra_buffer mamba
radix cache, CUDA graphs, vendor projection) and a small pool, and records, from real natlang turns
(runs/lfm3-engine-bench-20261028/out/capture.jsonl rendered with the export's chat template) cut to 300, 1.2k, 5k and
14k tokens (chunked prefill at 2048):

- greedy: each prompt prefilled with input logprobs from position 0 and 48 greedy tokens with top-5, alone and then all
  prompts concurrently (batched decode through CUDA graphs);
- prefix: each prompt + its greedy output + a tail, again: a radix/conv-state hit (cached_tokens recorded), logprobs of
  the tail and 32 greedy tokens;
- neuralese: spans of 600 and 100 tokens exported (custom_params.natlang_capture) and imported
  (custom_params.natlang_import) into another prompt's prefix, a continuation scored and 32 greedy tokens, twice (the
  second reads the placeholders' KV from the radix cache);
- retract: 8 concurrent 160-token greedy generations; mid-run pause_generation(retract), update_weights_from_disk (same
  weights, flush_cache) and continue_generation; outputs recorded with the run's server log pool lines.

`compare` pairs two runs: greedy token agreement and max/mean |delta| of every recorded logprob, per section. Retracted
generations are compared against the same prompts' undisturbed greedy decode of the same run (a recomputed prefix is
not bitwise equal to cached decode, so near-ties may flip) and the disagreement rate of both runs is reported.
"""
import argparse, hashlib, json, os, re, subprocess, sys, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TURNS = ROOT / 'runs/lfm3-engine-bench-20261028/out/capture.jsonl'
LENGTHS = (300, 1200, 5000, 14000)


def post(url, body, timeout=600):
    req = urllib.request.Request(url, json.dumps(body).encode(), {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read()
    try:
        return json.loads(body or b'null')
    except ValueError:   # /flush_cache and /continue_generation answer plain text
        return body.decode(errors='replace')


def prompts(model):
    """Real turns rendered with the export's template; concatenated where one turn is shorter than the cut."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    stream = []
    for line in open(TURNS):
        body = json.loads(line)['request']
        for message in body['messages']:
            for call in message.get('tool_calls') or []:
                if isinstance(call.get('function', {}).get('arguments'), str):
                    call['function']['arguments'] = json.loads(call['function']['arguments'])
        text = tok.apply_chat_template(body['messages'], tools=body.get('tools'), add_generation_prompt=True,
                                       tokenize=False)
        stream.append(tok.encode(text, add_special_tokens=False))
        if len(stream) >= 12:
            break
    out = []
    for i, n in enumerate(LENGTHS):
        ids = []
        j = i
        while len(ids) < n:
            ids += stream[j % len(stream)]
            j += 3
        out.append(ids[len(ids) - n:])  # the end of a turn: ends with the generation prompt
    return out, tok


def placeholders(path, count, vocab):
    base = 2 * vocab
    return [base + int.from_bytes(hashlib.sha256(f'{path}:{i}'.encode()).digest()[:4], 'little') % (2 ** 31 - 1 - base)
            for i in range(count)]


def generate(url, ids, **kw):
    params = {'temperature': 0.0, 'max_new_tokens': kw.pop('max_new_tokens', 1), 'ignore_eos': True,
              **kw.pop('params', {})}
    body = {'input_ids': ids, 'sampling_params': params, **kw}
    out = post(url + '/generate', body)
    m = out['meta_info']
    rec = {'output_ids': out['output_ids'], 'cached_tokens': m.get('cached_tokens')}
    if body.get('return_logprob'):
        rec['input_logprobs'] = [lp for lp, _, _ in m['input_token_logprobs']]
        rec['output_logprobs'] = [lp for lp, _, _ in m['output_token_logprobs']]
        if m.get('output_top_logprobs'):
            rec['output_top'] = [[[t, lp] for lp, t, _ in top] for top in m['output_top_logprobs']]
    return rec


def run(a):
    config = json.loads((Path(a.model) / 'config.json').read_text())
    vocab = config['vocab_size']
    roles = config['natlang_producer']
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    ps, tok = prompts(a.model)
    python = sys.executable
    env = {**os.environ, 'PYTHONPATH': os.pathsep.join(filter(None, [str(ROOT / 'training/neuralese'), a.fork_path])),
           'NATLANG_SGLANG_PROJECTION': a.projection, 'PATH': str(Path(python).parent) + os.pathsep + os.environ['PATH']}
    url = f'http://127.0.0.1:{a.port}'
    log = open(work / 'server.log', 'w')
    command = [python, '-m', 'sglang.launch_server', '--model-path', a.model, '--host', '127.0.0.1', '--port', str(a.port),
               '--dtype', 'bfloat16', '--mem-fraction-static', str(a.mem_fraction), '--context-length', '32768',
               '--max-total-tokens', str(a.max_total_tokens), '--max-running-requests', '16',
               '--kv-cache-dtype', a.kv_cache_dtype, '--mamba-radix-cache-strategy', 'extra_buffer',
               '--max-mamba-cache-size', '128', '--cuda-graph-max-bs-decode', '16', '--chunked-prefill-size', '2048',
               '--disable-overlap-schedule', '--skip-tokenizer-init', '--trust-remote-code', '--random-seed', '0',
               *a.sglang_arg]
    server = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    result = {'fork_path': a.fork_path, 'kv_cache_dtype': a.kv_cache_dtype, 'command': command}
    try:
        for _ in range(900):
            if server.poll() is not None:
                raise RuntimeError(f'server exited {server.returncode}; see {work / "server.log"}')
            try:
                urllib.request.urlopen(url + '/health', timeout=5)
                break
            except Exception:
                time.sleep(2)
        info = json.loads(urllib.request.urlopen(url + '/get_server_info').read())
        result['server_info'] = {k: info.get(k) for k in ('max_total_num_tokens', 'max_running_requests')}
        result['lengths'] = [len(p) for p in ps]
        flush = lambda: post(url + '/flush_cache', {})
        # greedy, one at a time from an empty cache, then concurrently
        result['greedy'] = []
        for p in ps:
            flush()
            result['greedy'].append(generate(url, p, max_new_tokens=48, return_logprob=True, logprob_start_len=0,
                                             top_logprobs_num=5))
        flush()
        with ThreadPoolExecutor(len(ps)) as pool:
            result['greedy_batched'] = list(pool.map(lambda p: generate(
                url, p, max_new_tokens=48, return_logprob=True, logprob_start_len=len(p) - 1), ps))
        # prefix hits: prompt + greedy output + a tail, cold (empty cache) and after the prompt's own greedy request
        # cached it (a radix and conv-state hit); the tail's logprobs and 32 greedy tokens
        tail = ps[1][:200]
        result['prefix_cold'], result['prefix'] = [], []
        for p, g in zip(ps, result['greedy']):
            ids = p + g['output_ids'] + tail
            for name in ('prefix_cold', 'prefix'):
                flush()
                if name == 'prefix':
                    generate(url, p, max_new_tokens=48)
                result[name].append(generate(url, ids, max_new_tokens=32, return_logprob=True,
                                             logprob_start_len=len(ids) - len(tail) - 1))
        # Neuralese export (cold) and import: spans longer and shorter than the 256-token window; the import cold and
        # then again (the placeholders' KV read from the radix cache)
        result['neuralese'] = []
        source = ps[3]
        after = tok.encode(' Using the recorded readings above, the north bank is warmer than the south bank.',
                           add_special_tokens=False)
        for length in (600, 100):
            lo = 4000
            hi = lo + length
            path = str(work / f'span-{length}.safetensors')
            if os.path.exists(path):
                os.unlink(path)
            flush()
            generate(url, source[:hi], return_logprob=True, logprob_start_len=lo - 1,
                     params={'custom_params': {'natlang_capture': {'path': path, 'start': lo, 'end': hi}}})
            prefix = ps[2][:1500]
            ids = prefix + [roles['consume_open']] + placeholders(path, length, vocab) + [roles['consume_close']] + after
            spec = {'path': path, 'start': len(prefix) + 1, 'count': length}
            runs = []
            flush()
            for _ in range(2):
                runs.append(generate(url, ids, max_new_tokens=32, return_logprob=True,
                                     logprob_start_len=len(ids) - len(after) - 1,
                                     params={'custom_params': {'natlang_import': [spec]}}))
            result['neuralese'].append({'length': length, 'runs': runs})
        # retract + weight update mid-generation
        post(url + '/flush_cache', {})
        gen = ps * 2
        outs = [None] * len(gen)

        def one(i):
            outs[i] = generate(url, gen[i], max_new_tokens=160, return_logprob=True, logprob_start_len=len(gen[i]) - 1)
        threads = [threading.Thread(target=one, args=(i,)) for i in range(len(gen))]
        [t.start() for t in threads]
        time.sleep(a.retract_after)
        t0 = time.time()
        post(url + '/pause_generation', {'mode': 'retract'})
        update = post(url + '/update_weights_from_disk', {'model_path': a.model, 'flush_cache': True})
        post(url + '/continue_generation', {})
        result['retract'] = {'update': update, 'pause_update_s': round(time.time() - t0, 2)}
        [t.join() for t in threads]
        result['retract']['outputs'] = outs
        # undisturbed reference for the retract phase (same run)
        result['retract']['reference'] = [generate(url, p, max_new_tokens=160, return_logprob=True,
                                                   logprob_start_len=len(p) - 1) for p in ps]
    finally:
        server.terminate()
        try:
            server.wait(60)
        except subprocess.TimeoutExpired:
            server.kill()
        log.close()
    text = (work / 'server.log').read_text(errors='replace')
    result['pool_log'] = [l for l in text.splitlines() if re.search(
        r'KV Cache is allocated|sliding window memory pool|max_total_num_tokens=|Tree cache initialized|Mamba Cache is '
        r'allocated|retract', l, re.I)][:40]
    Path(a.out).write_text(json.dumps(result))
    print(json.dumps({'out': a.out, 'server_info': result['server_info'], 'pool_log': result['pool_log'][:8]}, indent=1))


def diff(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    d = [abs(x - y) for x, y in pairs]
    return {'n': len(d), 'max': max(d, default=0.0), 'mean': sum(d) / len(d) if d else 0.0,
            'exact': sum(1 for v in d if v == 0.0)}


def compare(a):
    A, B = (json.loads(Path(p).read_text()) for p in (a.a, a.b))
    report = {}

    def section(name, xs, ys):
        tokens = [sum(1 for s, t in zip(x['output_ids'], y['output_ids']) if s == t) for x, y in zip(xs, ys)]
        total = [len(x['output_ids']) for x in xs]
        first = [next((i for i, (s, t) in enumerate(zip(x['output_ids'], y['output_ids'])) if s != t), None)
                 for x, y in zip(xs, ys)]
        lp_in = diff(sum((x.get('input_logprobs', []) for x in xs), []), sum((y.get('input_logprobs', []) for y in ys), []))
        # output logprobs only up to the first divergence of the greedy tokens
        ox, oy = [], []
        for x, y, f in zip(xs, ys, first):
            n = len(x['output_ids']) if f is None else f
            ox += x.get('output_logprobs', [])[:n]
            oy += y.get('output_logprobs', [])[:n]
        report[name] = {'greedy_tokens_equal': f'{sum(tokens)}/{sum(total)}', 'first_divergence': first,
                        'input_logprobs': lp_in, 'output_logprobs': diff(ox, oy),
                        'cached_tokens': [[x.get('cached_tokens'), y.get('cached_tokens')] for x, y in zip(xs, ys)]}
    section('greedy', A['greedy'], B['greedy'])
    section('greedy_batched', A['greedy_batched'], B['greedy_batched'])
    section('prefix_cold', A['prefix_cold'], B['prefix_cold'])
    section('prefix_hit', A['prefix'], B['prefix'])
    for name, R in (('A', A), ('B', B)):
        section(f'prefix_hit_vs_cold_{name}', R['prefix'], R['prefix_cold'])
        section(f'batched_vs_sequential_{name}', R['greedy_batched'], R['greedy'])
    for i, (na, nb) in enumerate(zip(A['neuralese'], B['neuralese'])):
        for j in range(2):
            section(f"neuralese_{na['length']}_run{j}", [na['runs'][j]], [nb['runs'][j]])
        # within one run: the radix-cached second import equals the first
        section(f"neuralese_{na['length']}_cached_vs_fresh_A", [na['runs'][0]], [na['runs'][1]])
        section(f"neuralese_{nb['length']}_cached_vs_fresh_B", [nb['runs'][0]], [nb['runs'][1]])
    for name, R in (('A', A), ('B', B)):
        ref = R['retract']['reference'] * 2
        section(f'retract_vs_undisturbed_{name}', R['retract']['outputs'], ref)
        report[f'retract_vs_undisturbed_{name}']['update'] = R['retract']['update']
    report['server_info'] = {'A': A['server_info'], 'B': B['server_info']}
    report['pool_log'] = {'A': A['pool_log'], 'B': B['pool_log']}
    print(json.dumps(report, indent=1))


def pair(a):
    fork = Path(json.loads((ROOT / 'training/neuralese/sglang-fork.json').read_text())['path'])
    fork = (ROOT / fork).resolve() if not fork.is_absolute() else fork
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    trees = {}
    for side, commit in (('old', a.old), ('new', a.new)):
        tree = work / f'fork-{side}'
        if not tree.exists():
            subprocess.run(['git', '-C', str(fork), 'worktree', 'add', '--detach', str(tree), commit], check=True)
            for so in (fork / 'python').rglob('*.so'):   # the Rust extensions are build products, not tracked
                target = tree / 'python' / so.relative_to(fork / 'python')
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(so.read_bytes())
        trees[side] = tree / 'python'
    try:
        for dtype in a.dtypes.split(','):
            for side in ('new', 'old'):
                out = work / f'{side}-{dtype}.json'
                if not out.exists():
                    subprocess.run([sys.executable, __file__, 'run', '--model', a.model, '--fork-path', str(trees[side]),
                                    '--out', str(out), '--work', str(work / f'{side}-{dtype}'), '--kv-cache-dtype', dtype,
                                    '--mem-fraction', '0.27', '--max-total-tokens', '40000',
                                    '--sglang-arg=--weight-loader-drop-cache-after-load'], check=True)
            with open(work / f'compare-{dtype}.json', 'w') as handle:
                subprocess.run([sys.executable, __file__, 'compare', str(work / f'old-{dtype}.json'),
                                str(work / f'new-{dtype}.json')], check=True, stdout=handle)
    finally:
        for tree in trees.values():
            subprocess.run(['git', '-C', str(fork), 'worktree', 'remove', '--force', str(tree.parent)])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--model', required=True)
    r.add_argument('--fork-path', default='', help="the fork's python/ directory ('' = the venv's installed sglang)")
    r.add_argument('--out', required=True)
    r.add_argument('--work', required=True)
    r.add_argument('--port', type=int, default=8391)
    r.add_argument('--kv-cache-dtype', default='fp8_e4m3')
    r.add_argument('--projection', default='vendor')
    r.add_argument('--mem-fraction', type=float, default=0.15)
    r.add_argument('--max-total-tokens', type=int, default=200000)
    r.add_argument('--retract-after', type=float, default=6.0)
    r.add_argument('--sglang-arg', action='append', default=[])
    q = sub.add_parser('pair')
    q.add_argument('--model', required=True)
    q.add_argument('--work', required=True)
    q.add_argument('--old', default='70b1991e665f9bb5baec4684fad66d13a41ca999')
    q.add_argument('--new', default='ace8d68681475f6e6485e285a734e336d9d06d02')
    q.add_argument('--dtypes', default='fp8_e4m3,auto')
    c = sub.add_parser('compare')
    c.add_argument('a')
    c.add_argument('b')
    a = p.parse_args()
    {'run': run, 'compare': compare, 'pair': pair}[a.cmd](a)


if __name__ == '__main__':
    main()
