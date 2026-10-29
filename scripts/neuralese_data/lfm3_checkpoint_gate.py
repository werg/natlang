#!/usr/bin/env python3
"""Behavioral gate for a training checkpoint: actual dev task execution against the base, plus weight coherence.

The step-8341 regression ran ~8k updates before anyone executed tasks (6/8 base -> 1/8). Run this on every new
checkpoint of a lineage (e.g. every 100-200 accumulated updates):
  1. export the checkpoint to a BF16 GGUF (natlang_neuralese.export.native_gguf);
  2. serve it with the qualified llama-server and the export's chat-template kwargs;
  3. execute the gate cases of the protected dev packet (all 8 graph-search worlds and the first 2 worlds of every
     other shape) with capped budgets (scripts/held_task_budget_probe.py);
  4. compare with the base model on the same cases (measured once, cached in --base-dir);
  5. with --previous checkpoints, report weight coherence (scripts/neuralese_data/weight_coherence.py).
The verdict is a regression when successes fall below the base minus --noise. It is evidence for the supervisor,
not a capability certificate.

  lfm3_checkpoint_gate.py CHECKPOINT --out DIR [--previous CKPT ...] [--port 8170]
"""
from __future__ import annotations

import argparse
import collections
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'training' / 'neuralese'))
from natlang_neuralese.common import paths  # noqa: E402

DATA = paths.root('repo')  # protected packets and the built ts-host collector live in the canonical checkout
PY = str(DATA / '.venv-neuralese/bin/python')
PACKET = 'runs/natlang-dev-task-packet-v2-20261028/cases.jsonl'
PACKET_SHA256 = '6fa6ca6c2e3a46c4b050da535e1a246c8f34d70f355ea047bcfe19a0653aa560'
FRONTIER = 'curriculum_iterate_frontier'


def gate_cases(packet):
    """All frontier worlds, then the first two worlds of every other shape, as packet row indices."""
    rows = [json.loads(line) for line in packet.read_text().splitlines() if line.strip()]
    taken, cases = collections.Counter(), []
    for index, row in enumerate(rows):
        shape = row['family']
        if shape == FRONTIER or taken[shape] < 2:
            cases.append(index)
            taken[shape] += 1
    return cases


def serve(binary, gguf, kwargs, port, log):
    process = subprocess.Popen([str(binary), '-m', str(gguf), '--jinja', '-ngl', '999', '-np', '8', '-c', '262144',
                                '--host', '127.0.0.1', '--port', str(port),
                                '--chat-template-kwargs', json.dumps(kwargs)], stdout=log, stderr=subprocess.STDOUT)
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


def execute(binary, gguf, kwargs, out, port, cases):
    """Run the gate cases against one GGUF; returns the probe's summary dictionary."""
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / 'summary.json'
    if summary_path.exists():
        return json.loads(summary_path.read_text())
    if (out / 'probe' / 'budget-probe.json').exists():
        try:
            return finish(out)  # a completed run whose summary was not cached by an older gate
        except RuntimeError:
            pass
    spec = {'async': True, 'cases': cases, 'case_seconds': 3600, 'max_model_requests': 150, 'max_turns': 150,
            'min_successes': 0, 'noise_successes': 1, 'reference_successes': None, 'workers': 8,
            'packet': {'path': str(DATA / PACKET), 'sha256': PACKET_SHA256}}
    with (out / 'server.log').open('w') as log:
        server = serve(binary, gguf, kwargs, port, log)
        try:
            with (out / 'probe.log').open('w') as probe_log:
                subprocess.run([PY, str(DATA / 'scripts/held_task_budget_probe.py'), '--spec', json.dumps(spec),
                                '--server', f'http://127.0.0.1:{port}', '--out', str(out / 'probe'),
                                '--case-seconds', '3600', '--max-model-requests', '150', '--max-turns', '150',
                                '--workers', '8'], stdout=probe_log, stderr=subprocess.STDOUT, check=True, cwd=DATA)
        finally:
            server.terminate()
            server.wait(timeout=60)
    return finish(out)


def finish(out):
    """Validate a completed probe run in `out` and cache its summary."""
    summary_path = out / 'summary.json'
    lines = [line for line in (out / 'probe.log').read_text().splitlines() if line.startswith('{"task_successes"')]
    if not lines:
        raise RuntimeError(f'probe printed no summary: {out / "probe.log"}')
    summary = json.loads(lines[-1])
    probe = json.loads((out / 'probe' / 'budget-probe.json').read_text())
    # A nonzero collector exit is also how a case that exhausted its model budget ends; that is a classified
    # outcome. Cases that never executed (e.g. ts-host/dist rebuilt mid-run) are unresolved or infrastructure
    # errors, and then the count is not a measurement.
    counts = summary.get('outcome_counts') or {}
    broken = counts.get('unresolved', 0) + counts.get('infrastructure_error', 0)
    if broken or sum(counts.values()) != len(probe.get('collector_exit_codes', [])):
        raise RuntimeError(f'{broken} cases unresolved or infrastructure errors, not caching a partial result: {out / "probe"}')
    summary_path.write_text(json.dumps(summary, indent=1) + '\n')
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--previous', type=Path, nargs='*', default=[],
                        help='earlier checkpoints of the same lineage, in order, for weight coherence')
    parser.add_argument('--base', type=Path, default=Path.home() / '.cache/huggingface/hub/models--LiquidAI--'
                        'LFM3-1.7B/snapshots/40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd')
    parser.add_argument('--base-gguf', type=Path, default=paths.resolve('data_nvme','lfm3-eval/LFM3-1.7B-BF16.gguf'))
    parser.add_argument('--base-dir', type=Path, default=paths.resolve('data_nvme','lfm3-eval/gate-base-v1'),
                        help='cached base execution of the gate cases')
    parser.add_argument('--llama-server', type=Path, default=paths.resolve('data_nvme',
        'dgxops/claude-recovery-20261028/data/llama-actor-worktree/build-cuda-121/bin/llama-server'))
    parser.add_argument('--port', type=int, default=8170)
    parser.add_argument('--noise', type=int, default=2, help='successes below base tolerated as noise')
    args = parser.parse_args(argv)
    cases = gate_cases(DATA / PACKET)
    args.out.mkdir(parents=True, exist_ok=True)
    gguf_dir = args.out / 'gguf'
    if not (gguf_dir / 'manifest.json').exists():
        subprocess.run([PY, '-m', 'natlang_neuralese.export.native_gguf', str(args.checkpoint), str(gguf_dir),
                        '--outtype', 'bf16', '--base', str(args.base)], check=True, cwd=REPO)
    manifest = json.loads((gguf_dir / 'manifest.json').read_text())
    kwargs = manifest['serving']['chat_template_kwargs']
    trained = execute(args.llama_server, gguf_dir / manifest['files'][0]['file'], kwargs, args.out / 'eval',
                      args.port, cases)
    base = execute(args.llama_server, args.base_gguf, kwargs, args.base_dir, args.port, cases)
    report = {'schema': 'natlang.checkpoint-behavior-gate/1', 'checkpoint': str(args.checkpoint),
              'step': manifest.get('step'), 'cases': cases, 'successes': trained['task_successes'],
              'base_successes': base['task_successes'], 'outcomes': trained['outcome_counts'],
              'base_outcomes': base['outcome_counts'],
              'regression': trained['task_successes'] < base['task_successes'] - args.noise}
    if args.previous:
        coherence = args.out / 'coherence.json'
        subprocess.run([PY, str(REPO / 'scripts/neuralese_data/weight_coherence.py'), '--base', str(args.base),
                        *map(str, args.previous), str(args.checkpoint), '--out', str(coherence)], check=True,
                       cwd=REPO, stdout=subprocess.DEVNULL)
        report['coherence'] = json.loads(coherence.read_text())
    (args.out / 'gate.json').write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps({key: report[key] for key in ('step', 'successes', 'base_successes', 'regression')}))
    return 1 if report['regression'] else 0


if __name__ == '__main__':
    sys.exit(main())
