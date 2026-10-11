#!/usr/bin/env python3
"""Render a large zstd-compressed native-turn corpus in parts with the shared renderer, within a bounded disk peak.

One implementation for every generated corpus (static curriculum batches, port-record call corpora, ...): the shared
renderer (render_native_turn_corpus.py) reads plain JSONL and keeps a working copy of its selection, so a whole corpus
at once needs ~2x its decompressed size (~0.4 MB a decision): 70 GB for one 94k-decision static batch (2026-10-12).
Here the turn stream (the given .turns.jsonl.zst files, in order) is cut into parts of about PART_DECISIONS decisions
at program boundaries (a program's decisions stay in one part), each part is decompressed only while it renders, and
the part receipts are merged into one receipt for the corpus.

  OUT/parts.json            {"cuts": [0, n1, ..., total], "part_decisions": N}  (decision indexes of the stream)
  OUT/parts/pKK/            the shared renderer's output for part KK (receipt.json, program-header/, whole-trajectory/,
                            full/ = the 5% warm-up sample, zstd)
  OUT/receipt.json          merged receipt (documents and per-split counts summed over parts)

Parts render independently: exact-duplicate removal and the internal train/test group-overlap check are per part
(held-text decontamination against other corpora is not). Resumable: finished parts are skipped.

  render_parts.py render --corpus-id ID --out OUT --source F.turns.jsonl.zst [F ...] [--register ENTRY.json]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PART_DECISIONS = 12000
PROTECTED = 'data/neuralese/corpora/protected-bgkit-20261003/bgkit-benchmarks.protected.json'
HELD_TEXT = [
    'runs/neuralese-foundation-text-input-4962-root-adopted-union-20261009-v1/text-data.jsonl',
    'data/neuralese/corpora/harness-bench-swe-rebench-openhands-pi-text-lfm25-350m-20261010-v2/text.jsonl',
]
DISK_MIN_GB = int(os.environ.get('NATLANG_DISK_MIN_GB', '100'))
PROGRAM = re.compile(rb'"program_ir_id":"((?:[^"\\]|\\.)*)"')


def load(path: Path):
    return json.loads(path.read_text())


def save(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=1, sort_keys=True) + '\n')
    tmp.replace(path)


def wait_for_disk(path: Path) -> None:
    """Pause while the NVMe has less than DISK_MIN_GB free (same rule as scripts/disk_guard.sh)."""
    said = False
    while shutil.disk_usage(path).free < DISK_MIN_GB * 2**30:
        if not said:
            print(json.dumps({'disk_guard': 'pausing', 'free_gb': shutil.disk_usage(path).free // 2**30}), flush=True)
            said = True
        time.sleep(120)


def stream(sources: list[Path]):
    """The decisions of all sources in order, one raw line each."""
    for source in sources:
        proc = subprocess.Popen(['zstd', '-dcq', '--long=27', str(source)], stdout=subprocess.PIPE)
        yield from proc.stdout
        if proc.wait():
            raise RuntimeError(f'zstd failed on {source}')


def program_of(line: bytes) -> bytes:
    found = PROGRAM.search(line, 0, 4096) or PROGRAM.search(line)
    if not found:
        raise ValueError('turn without source_ref.program_ir_id')
    return found.group(1)


def plan(sources: list[Path], out: Path, part_decisions: int = PART_DECISIONS) -> list[int]:
    path = out / 'parts.json'
    if path.exists():
        return load(path)['cuts']
    out.mkdir(parents=True, exist_ok=True)
    cuts, n, last = [0], 0, None
    for line in stream(sources):
        program = program_of(line)
        if n - cuts[-1] >= part_decisions and program != last:
            cuts.append(n)
        last, n = program, n + 1
    save(path, {'cuts': cuts + [n], 'part_decisions': part_decisions,
                'sources': [str(s) for s in sources]})
    return load(path)['cuts']


def render_part(sources: list[Path], out: Path, k: int, start: int, stop: int, corpus_id: str, unit_prefix: str,
                cpus: int, budget_gb: int, docker_gb: int) -> None:
    part = out / 'parts' / f'p{k:02d}'
    if (part / 'receipt.json').exists():
        return
    wait_for_disk(out)
    part.parent.mkdir(parents=True, exist_ok=True)
    # A part unit left from an interrupted run (preempted and requeued by the ledger) is waited for, not duplicated.
    prefix = f'{unit_prefix}-p{k:02d}-'
    while pending_units(prefix) and not (part / 'receipt.json').exists():
        time.sleep(60)
    if (part / 'receipt.json').exists():
        return
    plain = out / f'.p{k:02d}.turns.jsonl'
    with open(plain, 'wb') as f:
        for i, line in enumerate(stream(sources)):
            if i >= stop:
                break
            if i >= start:
                f.write(line)
    unit = f'{prefix}{int(time.time()) % 100000}'
    cmd = ['python3', str(REPO / 'scripts/memory_ledger.py'), 'run', '--unit', unit, '--budget-gb', str(budget_gb),
           '--priority', 'low', '--resumable', '--wait', '86400', '--workdir', str(REPO), '--',
           'env', f'MEM={docker_gb}g', f'CPUS={cpus}', 'nice', '-n', '10',
           'scripts/neuralese_data/render_native_turn_corpus.sh', str(part), '--corpus-id', corpus_id,
           '--source', str(plain), '--workers', str(cpus), '--shards', '32', '--full-sample', '0.05',
           '--protected', str(REPO / PROTECTED), '--reserved', str(REPO / 'training/source-split-reservations.json'),
           '--held-text', *[str(REPO / p) for p in HELD_TEXT]]
    try:
        subprocess.run(cmd, check=True)
        # The ledger unit runs asynchronously: wait for it to finish.
        while subprocess.run(['systemctl', '--user', 'is-active', '-q', f'{unit}.service']).returncode == 0:
            time.sleep(30)
    finally:
        plain.unlink(missing_ok=True)
    if not (part / 'receipt.json').exists():
        raise RuntimeError(f'render of {corpus_id} part {k} failed; see journalctl --user -u {unit}')
    # The full rendering is the 5% warm-up sample (--full-sample, shared rule); it is still stored compressed (~50x).
    for path in sorted((part / 'full').glob('*/*.jsonl')):
        subprocess.run(['zstd', '-q', '-3', '--long=27', '-T4', '--rm', '-f', str(path)], check=True)


def pending_units(prefix: str) -> list[str]:
    listed = subprocess.run(['systemctl', '--user', 'list-units', '--state=active,activating', '--plain', '--no-legend',
                             f'{prefix}*'], capture_output=True, text=True).stdout.split()
    try:
        status = json.loads(subprocess.run(['python3', str(REPO / 'scripts/memory_ledger.py'), 'status'],
                                           capture_output=True, text=True).stdout)
        queued = [w['unit'] for w in status.get('waiting', [])]
    except (ValueError, KeyError, TypeError):
        queued = []
    return [u for u in listed + queued if u.startswith(prefix)]


def merge_receipts(receipts: list[dict], cuts: list[int]) -> dict:
    """One corpus-level receipt over its parts: per-rendering documents and per-split counts summed, maxima kept."""
    renders: dict = {}
    for receipt in receipts:
        for key, info in receipt['renders'].items():
            agg = renders.setdefault(key, {'documents': 0, 'by_split': {}, 'max_document_tokens': 0})
            agg['documents'] += info.get('documents', 0)
            agg['max_document_tokens'] = max(agg['max_document_tokens'], info.get('max_document_tokens', 0))
            for split, counts in info.get('by_split', {}).items():
                into = agg['by_split'].setdefault(split, {})
                for field, value in counts.items():
                    into[field] = into.get(field, 0) + value
    return {'schema': 'natlang.render-parts/1', 'parts': len(receipts), 'cuts': cuts, 'renders': renders,
            'note': 'parts render independently: exact-duplicate removal and internal group-overlap checks are per part'}


def render(sources: list[Path], out: Path, corpus_id: str, unit_prefix: str, cpus: int = 6, budget_gb: int = 6,
           docker_gb: int = 8, part_decisions: int = PART_DECISIONS) -> dict:
    if (out / 'receipt.json').exists():
        return load(out / 'receipt.json')
    cuts = plan(sources, out, part_decisions)
    receipts = []
    for k, (start, stop) in enumerate(zip(cuts, cuts[1:])):
        render_part(sources, out, k, start, stop, corpus_id, unit_prefix, cpus, budget_gb, docker_gb)
        receipts.append(load(out / 'parts' / f'p{k:02d}' / 'receipt.json'))
    receipt = merge_receipts(receipts, cuts)
    save(out / 'receipt.json', receipt)
    return receipt


INCLUDE = ['receipt.json', 'parts.json', 'parts/*/receipt.json', 'parts/*/*.jsonl', 'parts/*/*/*/text-data.jsonl',
           'parts/*/*/*/text-data.jsonl.zst', 'parts/*/*/*/omissions.jsonl', 'parts/*/*/*/omissions.jsonl.zst',
           'parts/*/render-omissions/*/omissions.jsonl']


def render_summary(receipt: dict) -> dict:
    """The registry's `renders` block: per rendering and tokenizer, documents, per-split counts and the maximum."""
    out: dict = {}
    for key, info in receipt['renders'].items():
        mode, tok = key.split('/')
        out.setdefault(mode, {})[tok] = {k: v for k, v in info.items()
                                         if k in ('documents', 'by_split', 'max_document_tokens')}
    return out


def render_entry(source_id: str, rel_path: str, receipt: dict, **extra) -> dict:
    """Registry entry of a parts render of the admitted corpus SOURCE_ID (stored at the repo-relative REL_PATH)."""
    entry = {
        'id': f'{source_id}-render', 'owner': 'dgx', 'path': rel_path, 'include': INCLUDE,
        'source_class': 'rendered-natlang-native-teacher-turn-text',
        'split': 'per document, inherited from the source rows (validation/dev rendered as test); protected bgkit '
                 'benchmark overlaps and explicit test reservations dropped; train documents sharing a source group '
                 'with held text (4962 union, harness bench) dropped',
        'admission': f'renders of {source_id} in parts (scripts/neuralese_data/render_parts.py): program-header and '
                     'whole-trajectory for every program, full only for the 5% warm-up sample (zstd); LFM2.5 and '
                     'Mellum2.1 twins; no new example admission',
        'training_admission': True,
        'training_admission_reason': 'owner 2026-10-11: maximize data; renders of a training-admitted corpus',
        'derived_from': [source_id], 'license': 'facts from the source rows', 'renders': render_summary(receipt),
    }
    entry.update(extra)
    return entry


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('render')
    p.add_argument('--corpus-id', required=True, help='the admitted source corpus id')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--source', type=Path, nargs='+', required=True, help='.turns.jsonl.zst files, in order')
    p.add_argument('--unit-prefix', help='ledger unit name prefix of the part renders (default render-<corpus>)')
    p.add_argument('--cpus', type=int, default=6)
    p.add_argument('--budget-gb', type=int, default=6)
    p.add_argument('--docker-gb', type=int, default=8)
    p.add_argument('--register', type=Path, metavar='REL_PATH',
                   help='register the render (repo-relative path of OUT) through register_corpora.py')
    p.add_argument('--link', action='store_true',
                   help='with --register: create the REL_PATH symlink to OUT in the shared checkout')
    args = ap.parse_args()
    receipt = render(args.source, args.out, args.corpus_id, args.unit_prefix or f'render-{args.corpus_id}'[:60],
                     args.cpus, args.budget_gb, args.docker_gb)
    print(json.dumps({'rendered': args.corpus_id, 'parts': receipt['parts']}), flush=True)
    if args.register:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import register_corpora
        entry = render_entry(args.corpus_id, str(args.register), receipt)
        links = {str(args.register): str(args.out.resolve())} if args.link else None
        register_corpora.register([entry], f'Register {entry["id"]} (parts render)', links=links)
        print(json.dumps({'registered': entry['id']}), flush=True)


if __name__ == '__main__':
    main()
