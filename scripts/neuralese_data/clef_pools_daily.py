#!/usr/bin/env python3
"""Historical Clef corpus registration and offline rendering.

External teacher generation was retired by the owner on 2026-10-15. The old
``run`` entry fails before provider calls or file writes. ``register`` and
``render(day)`` remain available for already-produced source data; they do not
obtain new model labels. Continuous generation belongs to the self-hosted
Qwen3.8-Flash-Next teacher on the existing CoreWeave generation sandbox.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from register_corpora import register as register_entries, registered_ids  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TS = ROOT / 'ts-host'
CURRICULUM = TS / 'scripts/inline-curriculum'
CORPORA = ROOT / 'data/neuralese/corpora'
EPOCH = dt.date(2026, 10, 11)
SEED_BASE = 8100  # seeds 8100+day; 71/72 (clef-new), 6145 (dataset expansion), 102/900 (evaluation) stay unused
DGX_SHARE = 800_000
# systemd units see /usr/bin/node (v18), which lacks the runtime's crypto.randomUUID path: use the interactive node.
NODE = os.environ.get('NATLANG_NODE') or shutil.which('node', path=f"{Path.home()}/.local/bin:{os.environ.get('PATH', '')}") or 'node'


VERSION = os.environ.get('CLEF_POOLS_VERSION', 'v1')  # a rebuilt batch (fixed template, same answers) is a new id


def corpus_id(day: str) -> str:
    return f'clef-pools-{day}-{VERSION}'


def batch_dir(day: str) -> Path:
    return CORPORA / corpus_id(day)


def seed_for(day: str) -> int:
    return SEED_BASE + (dt.datetime.strptime(day, '%Y%m%d').date() - EPOCH).days


def node(*args: str) -> str:
    # One retry: a concurrent ts-host dist rebuild by another agent can fail a run transiently.
    for attempt in range(2):
        out = subprocess.run([NODE, *args], cwd=TS, capture_output=True, text=True)
        sys.stderr.write(out.stderr)
        if out.returncode == 0:
            return out.stdout.strip().splitlines()[-1] if out.stdout.strip() else ''
        sys.stderr.write(f'node {args[0]} exited {out.returncode} (attempt {attempt + 1})\n')
    raise SystemExit(f'node {args[0]} failed twice')


def spent(answers: Path) -> int:
    if not answers.exists():
        return 0
    return sum(json.loads(line).get('usage', {}).get('input_tokens', 0) for line in answers.open() if line.strip())


def count_lines(path: Path) -> int:
    return sum(1 for _ in path.open()) if path.exists() else 0


def registered(day: str) -> bool:
    return corpus_id(day) in registered_ids()


def run(day: str, budget: int, count: int) -> None:
    """Keep historical invocations explicit; never restart the retired provider."""
    raise RuntimeError(
        'External teacher generation is retired by AGENTS.md ownership policy '
        '(2026-10-15). Use the self-hosted Qwen teacher on the existing CoreWeave '
        'generation sandbox. Use register for already-produced Clef corpora.')


HELD_TEXT = ['runs/neuralese-foundation-text-input-4962-root-adopted-union-20261009-v1/text-data.jsonl',
             'data/neuralese/corpora/harness-bench-swe-rebench-openhands-pi-text-lfm25-350m-20261010-v2/text.jsonl']


def render(day: str) -> None:
    """Queue the shared three-rendering twin build (both tokenizers) as a low-priority resumable ledger unit."""
    out = render_dir(corpus_id(day))
    if (out / 'receipt.json').exists():
        return
    turns = batch_dir(day) / f'clef-pools-{day}.turns.jsonl'
    subprocess.run([sys.executable, str(ROOT / 'scripts/memory_ledger.py'), 'run', '--unit', f'render-turns-clef-pools-{day}',
                    '--budget-gb', '5', '--priority', 'low', '--resumable', '--wait', '43200', '--',
                    'env', 'MEM=5g', 'CPUS=2', 'scripts/neuralese_data/render_native_turn_corpus.sh', str(out),
                    '--corpus-id', corpus_id(day), '--source', str(turns.relative_to(ROOT)), '--workers', '2',
                    '--shards', '8', '--held-text', *HELD_TEXT], cwd=ROOT, check=False)


def render_dir(cid: str) -> Path:
    return ROOT / f'runs/natlang-native-turns-{cid}-render'


def register(day: str | None = None) -> None:
    """Register DAY's batch (if built) and every finished render whose batch is registered, through the worktree
    helper. Idempotent: registered ids are skipped (a published manifest is immutable)."""
    known = registered_ids()
    if day and (batch_dir(day) / 'registry-entry.json').exists() and corpus_id(day) not in known:
        entry = json.loads((batch_dir(day) / 'registry-entry.json').read_text())
        register_entries([entry], f"Register Clef pools batch {entry['id']}")
        known.add(entry['id'])
    sys.path.insert(0, str(ROOT / 'scripts/neuralese_data'))
    import register_native_turn_render as rntr
    batches = sorted(json.loads(f.read_text())['id'] for f in CORPORA.glob('clef-pools-*/registry-entry.json'))
    for cid in batches:
        if any(other.rsplit('-', 1)[0] == cid.rsplit('-', 1)[0] and other > cid for other in batches):
            continue  # superseded by a rebuilt version of the same day
        out, rid = render_dir(cid), f'natlang-native-turns-{cid}-render'
        if cid not in known or rid in known or not (out / 'receipt.json').exists():
            continue
        a = rntr.parse(['--id', rid, '--render-dir', str(out), '--source-corpus', cid])
        register_entries([rntr.make_entry(a)], f'Register {rid}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('action', choices=['run', 'register'])
    parser.add_argument('--day', default=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d'))
    parser.add_argument('--budget', type=int, default=DGX_SHARE)
    parser.add_argument('--count', type=int, default=700, help='skeletons per day (more than the budget answers)')
    args = parser.parse_args()
    run(args.day, args.budget, args.count) if args.action == 'run' else register(args.day)


if __name__ == '__main__':
    main()
