#!/usr/bin/env python3
"""Text twins of the bulk harness-bench records (training/neuralese/scripts/harness_bench_bulk.sh), per tokenizer.

Whole-trajectory documents (coordinator 2026-10-11): one document per maximal decision chain of a trajectory, every
assistant turn supervised (tool output at the shared feedback weight), instead of one document per decision, which
repeats the whole context for each of a trajectory's ~8 targets (~4.8B Mellum tokens for ~1% supervised). Each
finished records shard (a set of whole repositories, so its split is the whole corpus's) is rendered by the shared
renderer (scripts/neuralese_data/render_native_turn_corpus.py --renderings whole-trajectory, both tokenizers in one
pass) into TWIN_ROOT/render/NN; once every shard is rendered, each tokenizer's text is concatenated in shard order into
TWIN_ROOT/<name>/text.jsonl and the per-shard receipts are summed into TWIN_ROOT/<name>/receipt.json. Resumable: a
shard with a receipt is done; an unfinished one resumes from the renderer's own work dir.

  swe_bulk_twins.py --bulk BULK_OUT --twin-root DIR --tokenizer lfm25=SNAPSHOT --tokenizer mellum21=SNAPSHOT [--parallel 2]
     [--wait]   (keep polling until the records run has finished every shard)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = ROOT / '.venv-neuralese/bin/python'
# The records' system prompt is the pi harness's own prompt, not natlang's generic runtime prose: keep it whole.
HEADER = ['--program-header-as-full']


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def render(shard: Path, tokenizers: dict, root: Path, workers: int) -> str:
    out = root / 'render' / shard.name
    if (out / 'receipt.json').exists():
        return f'{shard.name} kept'
    out.parent.mkdir(parents=True, exist_ok=True)
    log = root / 'render' / f'{shard.name}.log'
    cmd = ['nice', '-n', '10', str(PY), str(ROOT / 'scripts/neuralese_data/render_native_turn_corpus.py'),
           '--corpus-id', f'swe-rebench-bulk-{shard.name}', '--source', str(shard / 'records.jsonl'),
           '--pieces', str(shard / 'pieces.jsonl'), '--out-dir', str(out), '--renderings', 'whole-trajectory',
           '--full-sample', '0', '--workers', str(workers), '--shards', '16', *HEADER]
    for name, path in tokenizers.items():
        cmd += ['--tokenizer', f'{name}={path}']
    with open(log, 'a') as handle:
        code = subprocess.call(cmd, stdout=handle, stderr=subprocess.STDOUT)
    return f'{shard.name} exit {code}'


def assemble(root: Path, name: str, shards: list[Path]) -> None:
    target = root / name
    if (target / 'receipt.json').exists():
        return
    totals: dict = {}
    parts = []
    tmp = target / 'text.jsonl.tmp'
    target.mkdir(parents=True, exist_ok=True)
    with open(tmp, 'wb') as dst:
        for shard in shards:
            part = root / 'render' / shard.name
            receipt = json.loads((part / 'receipt.json').read_text())
            render = receipt['renders'][f'whole-trajectory/{name}']
            with open(part / 'whole-trajectory' / name / 'text-data.jsonl', 'rb') as src:
                shutil.copyfileobj(src, dst, 1 << 24)
            for split, counts in render['by_split'].items():
                into = totals.setdefault(split, {})
                for key, value in counts.items():
                    into[key] = into.get(key, 0) + value
            totals.setdefault('max_document_tokens', 0)
            totals['max_document_tokens'] = max(totals['max_document_tokens'], render['max_document_tokens'])
            keep = target / 'shards' / shard.name
            keep.mkdir(parents=True, exist_ok=True)
            shutil.copy(part / 'receipt.json', keep / 'receipt.json')
            shutil.copy(part / 'whole-trajectory' / name / 'omissions.jsonl', keep / 'omissions.jsonl')
            shutil.copy(part / 'render-omissions' / name / 'omissions.jsonl', keep / 'render-omissions.jsonl')
            parts.append({'shard': shard.name, 'receipt_sha256': file_sha(part / 'receipt.json'),
                          'text_data_sha256': render['text_data_sha256'], 'tokens': render['by_split'],
                          'merge_omissions': render['merge_omissions'], 'selection': receipt['selection']})
    tmp.rename(target / 'text.jsonl')
    first = json.loads((root / 'render' / shards[0].name / 'receipt.json').read_text())
    summary = {'schema': 'natlang.harness-bench-bulk-twin/2',
               'builder': 'scripts/neuralese_data/render_native_turn_corpus.py --renderings whole-trajectory per shard',
               'rendering': 'whole-trajectory', 'renderer_code': first['renderer_code'],
               'tokenizer': first['tokenizers'][name], 'whole_trajectory': first['whole_trajectory'],
               'text_jsonl_sha256': file_sha(target / 'text.jsonl'), 'tokens': totals, 'shards': parts,
               'concatenation': 'shard order 00..NN; each shard is a set of whole repositories'}
    (target / 'receipt.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    print(json.dumps({name: totals}), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bulk', type=Path, required=True)
    parser.add_argument('--twin-root', type=Path, required=True)
    parser.add_argument('--tokenizer', action='append', required=True, help='NAME=SNAPSHOT')
    parser.add_argument('--parallel', type=int, default=2, help='shards rendered at once')
    parser.add_argument('--workers', type=int, default=3, help='renderer worker processes per shard')
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    tokenizers = dict(item.split('=', 1) for item in args.tokenizer)
    shards = sorted(p for p in (args.bulk / 'shards').iterdir() if p.is_dir())
    while True:
        ready = [s for s in shards if (s / 'summary.json').exists()]
        todo = [s for s in ready if not (args.twin_root / 'render' / s.name / 'receipt.json').exists()]
        with ThreadPoolExecutor(args.parallel) as pool:
            for line in pool.map(lambda shard: render(shard, tokenizers, args.twin_root, args.workers), todo):
                print(line, flush=True)
        if len(ready) == len(shards):
            failed = [s.name for s in shards if not (args.twin_root / 'render' / s.name / 'receipt.json').exists()]
            if failed:
                print(json.dumps({'failed': failed}), flush=True)
                return 1
            for name in tokenizers:
                assemble(args.twin_root, name, shards)
            return 0
        if not args.wait:
            return 0
        time.sleep(300)


if __name__ == '__main__':
    sys.exit(main())
