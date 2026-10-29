#!/usr/bin/env python3
"""Decision-grade behavioral eval of an LFM3 checkpoint against the base: all 128 cases of the protected dev packet
(16 shapes x 8 worlds), executed deterministically, compared case by case.

  lfm3_checkpoint_gate.py CHECKPOINT.pt|GGUF_EXPORT_DIR --out DIR [--port 8170] [--repeat-base]

Determinism (root cause 2026-10-29, plans/neuralese/LFM3_DECISION_EVAL_20261029.md). The old gate ran 8 llama-server
slots with continuous batching. llama.cpp picks the BF16 matmul kernel by the number of tokens in a batch (MMVF for one
column, MMF for a few, cuBLAS beyond), so a reply depends on what the other slots were doing: byte-identical requests
gave different greedy replies (7 of 7 measured). Its 3600 s wall budget also made the outcome of long cases depend on
server speed, and a ts-host/dist rebuild between runs changed the system prompt and tool surface. This eval therefore
  1. serves one sequence at a time without prompt-cache reuse (-np 1 --no-cache-prompt), so every reply is computed
     from scratch with batch-size-1 decode: identical requests give identical replies (verified bitwise);
  2. bounds cases only by counts (64 model requests; successful base/trained cases used at most 51), never wall time;
  3. executes a frozen copy of the TS runtime identified by its complete inventory digest, the same for both arms;
  4. applies the export's serving settings (chat-template kwargs and logit bias) to both the trained and base model.
Separate server processes (base and trained in parallel, other GPU jobs) do not affect the numerics, only the speed.

Decision rule (recommendation): count paired flips on the same cases, b = base fails / trained succeeds and
c = base succeeds / trained fails; exact two-sided McNemar p on (b, c). The checkpoint materially improves task
behavior when b > c with p < 0.05 and no shape loses net 3 or more of its 8 worlds; it regresses when c > b with
p < 0.05; otherwise there is no significant change. Exact determinism makes base-vs-base noise zero; --repeat-base
re-executes the base to verify that on the full packet.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import hashlib
import json
import math
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
PY = str(DATA / '.venv-neuralese/bin/python')
PACKET = 'runs/natlang-dev-task-packet-v2-20261028/cases.jsonl'
PACKET_SHA256 = '6fa6ca6c2e3a46c4b050da535e1a246c8f34d70f355ea047bcfe19a0653aa560'
SERVER_FLAGS = ('-np', '1', '-c', '32768', '--no-cache-prompt')
BUDGETS = {'max_model_requests': 64, 'max_turns': 64, 'case_seconds': 172800}
WORKERS, MODEL_CONCURRENCY = 8, 2
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


def server_args(binary, gguf, serving, port):
    args = [str(binary), '-m', str(gguf), '--jinja', '-ngl', '999', *SERVER_FLAGS, '--host', '127.0.0.1',
            '--port', str(port), '--chat-template-kwargs', json.dumps(serving.get('chat_template_kwargs') or {})]
    for token, bias in sorted((serving.get('logit_bias') or {}).items()):
        args += ['--logit-bias', f'{int(token)}{int(bias):+d}']
    return args


def serve(binary, gguf, serving, port, log):
    process = subprocess.Popen(server_args(binary, gguf, serving, port), stdout=log, stderr=subprocess.STDOUT)
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


def execute(binary, gguf, serving, out, port, cases, runtime):
    """Run the cases against one GGUF (once; a completed run in `out` is reused). Returns the run record."""
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'run.json').exists():
        return json.loads((out / 'run.json').read_text())
    spec = {'async': True, 'cases': cases, 'min_successes': 0, 'noise_successes': 0, 'reference_successes': None,
            'workers': WORKERS, 'runtime': str(runtime), **BUDGETS,
            'packet': {'path': str(DATA / PACKET), 'sha256': PACKET_SHA256}}
    started = time.time()
    with (out / 'server.log').open('w') as log:
        server = serve(binary, gguf, serving, port, log)
        try:
            with (out / 'probe.log').open('w') as probe_log:
                subprocess.run([PY, str(DATA / 'scripts/held_task_budget_probe.py'), '--spec', json.dumps(spec),
                                '--server', f'http://127.0.0.1:{port}', '--out', str(out / 'probe'),
                                '--case-seconds', str(BUDGETS['case_seconds']),
                                '--max-model-requests', str(BUDGETS['max_model_requests']),
                                '--max-turns', str(BUDGETS['max_turns']), '--workers', str(WORKERS),
                                '--model-concurrency', str(MODEL_CONCURRENCY)],
                               stdout=probe_log, stderr=subprocess.STDOUT, check=True, cwd=DATA)
        finally:
            server.terminate()
            server.wait(timeout=60)
    record = read_run(out)
    record['wall_seconds'] = round(time.time() - started, 1)
    (out / 'run.json').write_text(json.dumps(record, indent=1) + '\n')
    return record


def read_run(out):
    """Per-case outcomes of a completed probe in `out`; refuses partial runs and mixed harness identities."""
    probe = json.loads((out / 'probe' / 'budget-probe.json').read_text())
    counts = probe.get('outcome_counts') or {}
    broken = counts.get('unresolved', 0) + counts.get('infrastructure_error', 0)
    wall = [task['case_index'] for task in probe['tasks'] if task.get('resource_limit_reason') == 'case_wall_budget']
    if broken or wall or sum(counts.values()) != len(probe['tasks']):
        raise RuntimeError(f'{broken} unresolved/infrastructure cases, wall-budget cases {wall}: not a measurement: {out}')
    identities = collections.Counter()
    for path in (out / 'probe').glob('task-eval/probe/worker-*/case-*/results.jsonl'):
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


def base_key(binary, base_gguf, serving, runtime_sha, cases):
    """Everything that determines a base run's actions; a cached base run is reused only under the same key."""
    material = {'binary': sha256_file_hex(binary), 'gguf': sha256_file_hex(base_gguf), 'serving': serving,
                'flags': SERVER_FLAGS, 'budgets': BUDGETS, 'workers': WORKERS, 'model_concurrency': MODEL_CONCURRENCY,
                'runtime': runtime_sha, 'packet': PACKET_SHA256, 'cases': cases}
    return hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()[:16]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('model', type=Path, help='native checkpoint (.pt, exported here) or an existing GGUF export dir')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--base', type=Path, default=Path.home() / '.cache/huggingface/hub/models--LiquidAI--'
                        'LFM3-1.7B/snapshots/40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd')
    parser.add_argument('--base-gguf', type=Path, default=paths.resolve('data_nvme', 'lfm3-eval/LFM3-1.7B-BF16.gguf'))
    parser.add_argument('--eval-root', type=Path, default=paths.resolve('data_nvme', 'lfm3-eval/decision'),
                        help='holds frozen runtimes and cached base runs (keyed by everything that determines them)')
    parser.add_argument('--llama-server', type=Path, default=paths.resolve('data_nvme',
        'dgxops/claude-recovery-20261028/data/llama-actor-worktree/build-cuda-121/bin/llama-server'))
    parser.add_argument('--port', type=int, default=8170, help='trained server port; the base uses port+1/+2')
    parser.add_argument('--repeat-base', action='store_true', help='also re-execute the base and report its flips')
    args = parser.parse_args(argv)
    cases = decision_cases(DATA / PACKET)
    args.out.mkdir(parents=True, exist_ok=True)
    gguf_dir = args.model if args.model.is_dir() else args.out / 'gguf'
    if not (gguf_dir / 'manifest.json').exists():
        subprocess.run([PY, '-m', 'natlang_neuralese.export.native_gguf', str(args.model), str(gguf_dir),
                        '--outtype', 'bf16', '--base', str(args.base)], check=True, cwd=REPO)
    manifest = json.loads((gguf_dir / 'manifest.json').read_text())
    serving = {'chat_template_kwargs': manifest['serving']['chat_template_kwargs'],
               'logit_bias': manifest['serving'].get('logit_bias') or {}}
    runtime, runtime_sha = snapshot_runtime(args.eval_root / 'runtimes')
    base_dir = args.eval_root / f'base-{base_key(args.llama_server, args.base_gguf, serving, runtime_sha, cases)}'
    gguf = gguf_dir / manifest['files'][0]['file']
    jobs = {'trained': (gguf, args.out / 'eval', args.port), 'base': (args.base_gguf, base_dir / 'run-1', args.port + 1)}
    if args.repeat_base:
        jobs['base_repeat'] = (args.base_gguf, base_dir / 'run-2', args.port + 2)
    with concurrent.futures.ThreadPoolExecutor(len(jobs)) as pool:
        futures = {name: pool.submit(execute, args.llama_server, model, serving, out, port, cases, runtime)
                   for name, (model, out, port) in jobs.items()}
        runs = {name: future.result() for name, future in futures.items()}
    report = {'schema': SCHEMA, 'checkpoint': str(args.model), 'step': manifest.get('step'),
              'runtime_sha256': runtime_sha, 'server_flags': list(SERVER_FLAGS), 'budgets': BUDGETS,
              'serving': serving, 'base_dir': str(base_dir),
              'wall_seconds': {name: run.get('wall_seconds') for name, run in runs.items()},
              **compare(runs['trained'], runs['base'])}
    if 'base_repeat' in runs:
        noise = compare(runs['base_repeat'], runs['base'])
        report['base_repeat_flips'] = len(noise['gained']) + len(noise['lost'])
    (args.out / 'decision.json').write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps({key: report.get(key) for key in ('step', 'successes', 'base_successes', 'gained', 'lost',
                                                         'mcnemar_p', 'shape_regressions', 'verdict',
                                                         'base_repeat_flips', 'wall_seconds')}))
    return 1 if report['verdict'] == 'regressed' else 0


if __name__ == '__main__':
    sys.exit(main())
