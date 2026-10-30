#!/usr/bin/env python3
"""Decision-grade behavioral eval of LFM3 checkpoints against the base: all 128 cases of the protected dev packet
(16 shapes x 8 worlds), compared case by case with paired statistics.

  lfm3_checkpoint_gate.py MODEL [MODEL ...] --out DIR [--serving batched|deterministic] [--repeat-base]

MODEL is a native checkpoint (.pt, exported to BF16 GGUF under DIR) or an existing GGUF export directory. Each model's
report is DIR/<label>/decision.json; the base runs are cached under --eval-root keyed by everything that determines
them. All arms execute concurrently, each with its own llama-server.

Noise (root cause 2026-10-29, plans/neuralese/LFM3_DECISION_EVAL_20261029.md). The old gate's 33/38 vs 26/38 on one
base GGUF came from (a) a ts-host/dist rebuild between the runs (different system prompt and tool surface), (b) a
3600 s wall budget that ended long cases depending on server speed, and (c) batch-dependent greedy decoding:
llama.cpp picks the BF16 matmul kernel by the number of tokens in a batch (MMVF for one column, MMF for a few, cuBLAS
beyond) and reuses prompt-cache KV computed under other batch shapes, so byte-identical requests gave different
greedy replies (7 of 7 measured). Every mode here therefore
  1. bounds cases only by counts (64 model requests; successful cases have used at most 51), never wall time;
  2. executes a frozen copy of the TS runtime identified by its complete inventory digest, the same for all arms, and
     refuses a run whose rows carry more than one harness identity;
  3. applies the export's serving settings (chat-template kwargs and logit bias) to the trained models and the base.
Serving modes:
  deterministic  one sequence at a time without prompt-cache reuse (-np 1 --no-cache-prompt): identical requests give
                 identical replies (verified bitwise, also under concurrent GPU load); about 60 tok/s alone on the GB10
                 and a few tok/s while training shares the GPU, so a 128-case run takes many hours.
  batched        16 slots with continuous batching (default): ~4-5x the throughput, replies vary with load; use
                 --repeat-base to measure the base-vs-base flips, which the McNemar test tolerates as noise.

Decision rule (recommendation): count paired flips on the same cases, b = base fails / trained succeeds and
c = base succeeds / trained fails; exact two-sided McNemar p on (b, c). A checkpoint materially improves task
behavior when b > c with p < 0.05 and no shape loses net 3 or more of its 8 worlds; it regresses when c > b with
p < 0.05; otherwise there is no significant change. With batched serving the verdict must hold against both base
executions (the report gives both comparisons).
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import fcntl
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'training' / 'neuralese'))
from natlang_neuralese.common import paths  # noqa: E402
from natlang_neuralese.common.hashing import sha256_file_hex  # noqa: E402

DATA = paths.root('repo')  # protected packets and the built ts-host collector live in the canonical checkout
# The canonical venv on DGX; a sandbox (no such venv) runs the probe with this interpreter.
PY = os.environ.get('NATLANG_GATE_PYTHON') or (str(DATA / '.venv-neuralese/bin/python')
                                               if (DATA / '.venv-neuralese/bin/python').exists() else sys.executable)
PACKET = 'runs/natlang-dev-task-packet-v2-20261028/cases.jsonl'
PACKET_SHA256 = '6fa6ca6c2e3a46c4b050da535e1a246c8f34d70f355ea047bcfe19a0653aa560'
SERVING = {'deterministic': {'flags': ('-np', '1', '-c', '32768', '--no-cache-prompt'), 'workers': 8},
           'batched': {'flags': ('-np', '16', '-c', str(16 * 32768)), 'workers': 16}}
BUDGETS = {'max_model_requests': 64, 'max_turns': 64, 'case_seconds': 172800}
MODEL_CONCURRENCY = 2
SHAPE_NET_LOSS_LIMIT = 3
ALPHA = 0.05
SCHEMA = 'natlang.lfm3-decision-eval/1'


def decision_cases(packet):
    """Every packet row index."""
    return [index for index, line in enumerate(packet.read_text().splitlines()) if line.strip()]


def runtime_files(ts_host):
    """Content digests of the complete TS runtime closure (dist, package files, dependency files)."""
    from natlang_neuralese.common.runtime_inventory import task_runtime_inventory
    inventory = task_runtime_inventory(Path(ts_host))
    files = {key: value for key, value in inventory.items() if not key.startswith('@')}
    files.update({f'@dependency/{key}': value.get('sha256') for key, value in inventory['@dependency-files'].items()})
    return files


def runtime_digest(files):
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def snapshot_runtime(root, live=None):
    """Freeze the live ts-host runtime under root/<digest>/ts-host (dist, prelude and package files copied;
    the node_modules trees, vendor/ and the read-only training/ inputs linked) and verify the copy has the live digest."""
    live = Path(live or DATA / 'ts-host').resolve()
    files = runtime_files(live)
    digest = runtime_digest(files)
    target = Path(root) / digest[:16]
    runtime = target / 'ts-host'
    if not (target / 'verified').exists():
        if target.exists():
            shutil.rmtree(target)
        runtime.mkdir(parents=True)
        shutil.copytree(live / 'dist', runtime / 'dist', symlinks=True)
        for name in ('prelude.js', 'package.json'):
            shutil.copy2(live / name, runtime / name)
        for name in ('node_modules', 'vendor'):
            (runtime / name).symlink_to(live / name)
        for name in ('package.json', 'package-lock.json'):
            shutil.copy2(live.parent / name, target / name)
        for name in ('node_modules', 'training'):
            (target / name).symlink_to(live.parent / name)
        copied, after = runtime_files(runtime), runtime_files(live)
        if copied != files or after != files:
            shutil.rmtree(target)
            differing = sorted(key for key in set(files) | set(copied) | set(after)
                               if not files.get(key) == copied.get(key) == after.get(key))
            raise RuntimeError(f'frozen ts-host differs from the live runtime in {len(differing)} files: {differing[:5]}')
        (target / 'verified').write_text(digest + '\n')
    return runtime, digest


def server_args(binary, gguf, serving, port, mode='deterministic'):
    args = [str(binary), '-m', str(gguf), '--jinja', '-ngl', '999', *SERVING[mode]['flags'], '--host', '127.0.0.1',
            '--port', str(port), '--chat-template-kwargs', json.dumps(serving.get('chat_template_kwargs') or {})]
    for token, bias in sorted((serving.get('logit_bias') or {}).items()):
        args += ['--logit-bias', f'{int(token)}{int(bias):+d}']
    return args


def serve(binary, gguf, serving, port, log, mode):
    process = subprocess.Popen(server_args(binary, gguf, serving, port, mode), stdout=log, stderr=subprocess.STDOUT)
    for _ in range(300):
        try:
            urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=2)
            return process
        except OSError:
            if process.poll() is not None:
                raise RuntimeError(f'llama-server exited with {process.returncode}')
            time.sleep(2)
    process.terminate()
    raise RuntimeError('llama-server did not become healthy')


def execute(binary, gguf, serving, out, port, cases, runtime, mode, servers=1, prompt_rendering='full'):
    """Run the cases against one GGUF (once; a completed run in `out` is reused, and a concurrent gate executing the
    same run is waited for). Returns the run record."""
    out.mkdir(parents=True, exist_ok=True)
    with (out / '.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (out / 'run.json').exists():
            return json.loads((out / 'run.json').read_text())
        return _execute(binary, gguf, serving, out, port, cases, runtime, mode, servers, prompt_rendering)


def _execute(binary, gguf, serving, out, port, cases, runtime, mode, servers=1, prompt_rendering='full'):
    """servers > 1 shards the cases (interleaved) over that many servers of the same GGUF, one probe each: a case's
    actions do not depend on which server or which other cases run (deterministic serving computes every request
    alone), so this only changes speed. Shard k writes probe-k/ (one server writes probe/)."""
    shards = [('probe' if servers == 1 else f'probe-{k}', cases[k::servers]) for k in range(servers)]
    workers = max(1, SERVING[mode]['workers'] // servers)
    started = time.time()
    processes, logs = [], []
    try:
        for k, (name, shard) in enumerate(shards):
            log = (out / ('server.log' if servers == 1 else f'server-{k}.log')).open('w')
            logs.append(log)
            processes.append(serve(binary, gguf, serving, port + k, log, mode))

        def probe(k):
            name, shard = shards[k]
            spec = {'async': True, 'cases': shard, 'min_successes': 0, 'noise_successes': 0,
                    'reference_successes': None, 'workers': workers, 'runtime': str(runtime), **BUDGETS,
                    'prompt_rendering': prompt_rendering,
                    'packet': {'path': str(DATA / PACKET), 'sha256': PACKET_SHA256}}
            with (out / f'{name}.log').open('w') as probe_log:
                subprocess.run([PY, str(DATA / 'scripts/held_task_budget_probe.py'), '--spec', json.dumps(spec),
                                '--server', f'http://127.0.0.1:{port + k}', '--out', str(out / name),
                                '--case-seconds', str(BUDGETS['case_seconds']),
                                '--max-model-requests', str(BUDGETS['max_model_requests']),
                                '--max-turns', str(BUDGETS['max_turns']), '--workers', str(workers),
                                '--model-concurrency', str(MODEL_CONCURRENCY)],
                               stdout=probe_log, stderr=subprocess.STDOUT, check=True, cwd=DATA)
        with concurrent.futures.ThreadPoolExecutor(servers) as pool:
            for future in [pool.submit(probe, k) for k in range(servers)]:
                future.result()
    finally:
        for process in processes:
            process.terminate()
            process.wait(timeout=60)
        for log in logs:
            log.close()
    record = read_run(out)
    if sorted(int(case) for case in record['cases']) != sorted(cases):
        raise RuntimeError(f'shards did not cover exactly the requested cases: {out}')
    record['wall_seconds'] = round(time.time() - started, 1)
    record['serving'] = mode
    record['servers'] = servers
    record['prompt_rendering'] = prompt_rendering
    (out / 'run.json').write_text(json.dumps(record, indent=1) + '\n')
    return record


def read_run(out):
    """Per-case outcomes of a completed probe in `out` (probe/ or the shards probe-*/); refuses partial runs and
    mixed harness identities."""
    dirs = sorted(path for path in out.glob('probe*') if path.is_dir() and (path.name == 'probe' or
                                                                            path.name[6:].isdigit()))
    reports = [json.loads((path / 'budget-probe.json').read_text()) for path in dirs]
    if not reports:
        raise RuntimeError(f'no probe report in {out}')
    counts = collections.Counter()
    for report in reports:
        counts.update(report.get('outcome_counts') or {})
    probe = {'tasks': [task for report in reports for task in report['tasks']],
             'task_successes': sum(report['task_successes'] for report in reports)}
    counts = dict(counts)
    broken = counts.get('unresolved', 0) + counts.get('infrastructure_error', 0)
    wall = [task['case_index'] for task in probe['tasks'] if task.get('resource_limit_reason') == 'case_wall_budget']
    if broken or wall or sum(counts.values()) != len(probe['tasks']):
        raise RuntimeError(f'{broken} unresolved/infrastructure cases, wall-budget cases {wall}: not a measurement: {out}')
    identities = collections.Counter()
    for path in (p for d in dirs for p in d.glob('task-eval/probe/worker-*/case-*/results.jsonl')):
        for line in path.read_text().splitlines():
            if line.strip():
                provenance = json.loads(line)['provenance']
                identities[(provenance['system_prompt_sha256'], provenance['tool_surface_sha256'],
                            provenance['runtime_contract_version'], provenance['temperature'])] += 1
    if len(identities) > 1:
        raise RuntimeError(f'mixed harness identities within one run: {dict(identities)}')
    return {'cases': {str(task['case_index']): {'family': task['family'], 'outcome': task['outcome'],
                                                'requests': task['model_requests'],
                                                'completion_tokens': task['completion_tokens']}
                      for task in probe['tasks']},
            'successes': probe['task_successes'], 'outcome_counts': counts,
            'harness': list(next(iter(identities))) if identities else None,
            'completion_tokens': sum(task['completion_tokens'] or 0 for task in probe['tasks'])}


def mcnemar_p(b, c):
    """Exact two-sided McNemar p-value: binomial test of b successes in b + c discordant pairs at 1/2."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def compare(trained, base):
    """Paired comparison of two runs on the same cases."""
    if set(trained['cases']) != set(base['cases']):
        raise RuntimeError('runs cover different cases')
    if trained['harness'] != base['harness']:
        raise RuntimeError(f'runs used different harnesses: {trained["harness"]} vs {base["harness"]}')
    shapes = collections.defaultdict(lambda: {'cases': 0, 'trained': 0, 'base': 0, 'gained': 0, 'lost': 0})
    gained, lost = [], []
    for case, row in sorted(trained['cases'].items(), key=lambda item: int(item[0])):
        ok, base_ok = row['outcome'] == 'task_success', base['cases'][case]['outcome'] == 'task_success'
        shape = shapes[row['family']]
        shape['cases'] += 1
        shape['trained'] += ok
        shape['base'] += base_ok
        if ok and not base_ok:
            shape['gained'] += 1
            gained.append(int(case))
        elif base_ok and not ok:
            shape['lost'] += 1
            lost.append(int(case))
    b, c = len(gained), len(lost)
    p = mcnemar_p(b, c)
    shape_regressions = sorted(name for name, shape in shapes.items()
                               if shape['lost'] - shape['gained'] >= SHAPE_NET_LOSS_LIMIT)
    if b > c and p < ALPHA:
        verdict = 'improved' if not shape_regressions else 'improved-with-shape-regressions'
    elif c > b and p < ALPHA:
        verdict = 'regressed'
    else:
        verdict = 'no-significant-change'
    return {'successes': sum(s['trained'] for s in shapes.values()), 'base_successes': sum(s['base'] for s in shapes.values()),
            'cases': len(trained['cases']), 'gained': gained, 'lost': lost, 'mcnemar_p': p,
            'shape_regressions': shape_regressions, 'verdict': verdict, 'shapes': dict(sorted(shapes.items()))}


def base_key(binary, base_gguf, serving, runtime_sha, cases, mode, prompt_rendering='full'):
    """Everything that determines a base run's actions; a cached base run is reused only under the same key."""
    material = {'binary': sha256_file_hex(binary), 'gguf': sha256_file_hex(base_gguf), 'serving': serving,
                'mode': SERVING[mode], 'budgets': BUDGETS, 'model_concurrency': MODEL_CONCURRENCY,
                'runtime': runtime_sha, 'packet': PACKET_SHA256, 'cases': cases}
    if prompt_rendering != 'full':  # full keeps its historical key, so existing full base runs stay reusable
        material['prompt_rendering'] = prompt_rendering
    return hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()[:16]


def export(model, out, base):
    """GGUF export directory and manifest of a checkpoint (exported once) or of an existing export directory."""
    gguf_dir = model if model.is_dir() else out / 'gguf'
    if not (gguf_dir / 'manifest.json').exists():
        subprocess.run([PY, '-m', 'natlang_neuralese.export.native_gguf', str(model), str(gguf_dir),
                        '--outtype', 'bf16', '--base', str(base)], check=True, cwd=REPO)
    return gguf_dir, json.loads((gguf_dir / 'manifest.json').read_text())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('models', type=Path, nargs='+',
                        help='native checkpoints (.pt, exported here) or existing GGUF export directories')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--serving', choices=sorted(SERVING), default='batched')
    parser.add_argument('--base', type=Path, default=Path.home() / '.cache/huggingface/hub/models--LiquidAI--'
                        'LFM3-1.7B/snapshots/40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd')
    parser.add_argument('--base-gguf', type=Path, default=paths.resolve('data_nvme', 'lfm3-eval/LFM3-1.7B-BF16.gguf'))
    parser.add_argument('--eval-root', type=Path, default=paths.resolve('data_nvme', 'lfm3-eval/decision'),
                        help='holds frozen runtimes and cached base runs')
    parser.add_argument('--runtime', type=Path, help='an existing frozen ts-host under --eval-root/runtimes '
                        '(default: freeze the live checkout)')
    parser.add_argument('--llama-server', type=Path, default=paths.resolve('data_nvme',
        'dgxops/claude-recovery-20261028/data/llama-actor-worktree/build-cuda-121/bin/llama-server'))
    parser.add_argument('--port', type=int, default=8170, help='first server port; each arm uses the next one')
    parser.add_argument('--prompt-rendering', choices=('full', 'program-header'), default='full',
                        help='native prompt the cases run with: full (serving default) or program-header, the short '
                             'header without tool schemas that the SFT rows were rendered with')
    parser.add_argument('--repeat-base', action='store_true', help='also re-execute the base and report its flips')
    parser.add_argument('--servers', type=int, default=1, help='llama-servers per arm (deterministic serving: '
                        'one sequence per server, so several per arm use a large GPU; results are unchanged)')
    args = parser.parse_args(argv)
    cases = decision_cases(DATA / PACKET)
    args.out.mkdir(parents=True, exist_ok=True)
    exports = [export(model, args.out / f'model-{index}', args.base) for index, model in enumerate(args.models)]
    servings = {json.dumps({'chat_template_kwargs': manifest['serving']['chat_template_kwargs'],
                            'logit_bias': manifest['serving'].get('logit_bias') or {}}, sort_keys=True)
                for _, manifest in exports}
    if len(servings) != 1:
        raise SystemExit(f'the models need different serving settings, so no common base run: {servings}')
    serving = json.loads(servings.pop())
    if args.runtime:
        runtime = args.runtime
        runtime_sha = runtime_digest(runtime_files(runtime))
    else:
        runtime, runtime_sha = snapshot_runtime(args.eval_root / 'runtimes')
    base_dir = args.eval_root / f'base-{base_key(args.llama_server, args.base_gguf, serving, runtime_sha, cases, args.serving, args.prompt_rendering)}'
    jobs = {'base': (args.base_gguf, base_dir / 'run-1')}
    if args.repeat_base:
        jobs['base_repeat'] = (args.base_gguf, base_dir / 'run-2')
    labels = []
    for index, (gguf_dir, manifest) in enumerate(exports):
        label = f"step{manifest.get('step')}" if manifest.get('step') is not None else f'model-{index}'
        labels.append(label)
        suffix = '' if args.prompt_rendering == 'full' else f'-{args.prompt_rendering}'
        jobs[label] = (gguf_dir / manifest['files'][0]['file'], args.out / label / f'eval-{args.serving}{suffix}')
    with concurrent.futures.ThreadPoolExecutor(len(jobs)) as pool:
        futures = {name: pool.submit(execute, args.llama_server, model, serving, out, args.port + offset * args.servers,
                                     cases, runtime, args.serving, args.servers, args.prompt_rendering)
                   for offset, (name, (model, out)) in enumerate(jobs.items())}
        runs = {name: future.result() for name, future in futures.items()}
    summary = {'schema': SCHEMA, 'serving': args.serving, 'prompt_rendering': args.prompt_rendering, 'flags': list(SERVING[args.serving]['flags']),
               'budgets': BUDGETS, 'runtime_sha256': runtime_sha, 'base_dir': str(base_dir), 'serving_settings': serving,
               'wall_seconds': {name: run.get('wall_seconds') for name, run in runs.items()}, 'models': {}}
    if 'base_repeat' in runs:
        noise = compare(runs['base_repeat'], runs['base'])
        summary['base_noise'] = {'successes': [runs['base']['successes'], runs['base_repeat']['successes']],
                                 'flips': len(noise['gained']) + len(noise['lost']), 'mcnemar_p': noise['mcnemar_p']}
    regressed = False
    for label, model in zip(labels, args.models):
        report = {'checkpoint': str(model), **compare(runs[label], runs['base'])}
        if 'base_repeat' in runs:
            report['against_base_repeat'] = {key: value for key, value in compare(runs[label], runs['base_repeat']).items()
                                             if key != 'shapes'}
        (args.out / label / ('decision.json' if args.prompt_rendering == 'full' else f'decision-{args.prompt_rendering}.json')).write_text(json.dumps(report, indent=1) + '\n')
        summary['models'][label] = {key: report[key] for key in ('successes', 'base_successes', 'gained', 'lost',
                                                                 'mcnemar_p', 'shape_regressions', 'verdict')}
        if 'against_base_repeat' in report:
            summary['models'][label]['verdict_against_base_repeat'] = report['against_base_repeat']['verdict']
        regressed |= report['verdict'] == 'regressed'
    name = 'decision.json' if args.prompt_rendering == 'full' else f'decision-{args.prompt_rendering}.json'
    (args.out / name).write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps(summary))
    return 1 if regressed else 0


if __name__ == '__main__':
    sys.exit(main())
