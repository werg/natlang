#!/usr/bin/env python3
"""Text twins of the bulk harness-bench records (training/neuralese/scripts/harness_bench_bulk.sh), per tokenizer.

Each finished records shard (a set of whole repositories, so its train/test split is the whole corpus's) is rendered
by the shared packet builder (scripts/build_neuralese_text_packet.py: gold_text_rows, the backbone's declared history
reasoning) into TWIN_ROOT/<name>/shards/NN; its text is appended to TWIN_ROOT/<name>/text.jsonl in shard order once
every shard is rendered, and the per-shard receipts are summed into receipt.json. Resumable: a shard with a receipt
is done; a partial shard directory is removed and rendered again.

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


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def render(shard: Path, name: str, tokenizer: str, root: Path) -> str:
    out = root / name / 'shards' / shard.name
    if (out / 'receipt.json').exists():
        return f'{name}/{shard.name} kept'
    if out.exists():
        shutil.rmtree(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    log = root / name / 'shards' / f'{shard.name}.log'
    with open(log, 'w') as handle:
        code = subprocess.call(['nice', '-n', '10', str(PY), str(ROOT / 'scripts/build_neuralese_text_packet.py'),
                                '--records', str(shard / 'records.jsonl'), '--pieces', str(shard / 'pieces.jsonl'),
                                '--tokenizer', tokenizer, '--out', str(out)], stdout=handle, stderr=subprocess.STDOUT)
    return f'{name}/{shard.name} exit {code}'


def assemble(root: Path, name: str, shards: list[Path]) -> None:
    target = root / name
    if (target / 'receipt.json').exists():
        return
    totals: dict = {}
    parts = []
    tmp = target / 'text.jsonl.tmp'
    with open(tmp, 'wb') as dst:
        for shard in shards:
            part = target / 'shards' / shard.name
            receipt = json.loads((part / 'receipt.json').read_text())
            with open(part / 'text.jsonl', 'rb') as src:
                shutil.copyfileobj(src, dst, 1 << 24)
            for split, counts in receipt['tokens'].items():
                into = totals.setdefault(split, {'documents': 0, 'tokens': 0, 'suffix_tokens': 0, 'max_document_tokens': 0})
                for key in ('documents', 'tokens', 'suffix_tokens'):
                    into[key] += counts[key]
                into['max_document_tokens'] = max(into['max_document_tokens'], counts['max_document_tokens'])
            parts.append({'shard': shard.name, 'receipt_sha256': file_sha(part / 'receipt.json'),
                          'text_jsonl_sha256': receipt['text_jsonl_sha256'], 'source_files': receipt['source_files'],
                          'omitted_records': receipt.get('omitted_records'), 'tokens': receipt['tokens']})
    tmp.rename(target / 'text.jsonl')
    first = json.loads((target / 'shards' / shards[0].name / 'receipt.json').read_text())
    summary = {'schema': 'natlang.harness-bench-bulk-twin/1', 'builder': 'scripts/build_neuralese_text_packet.py per shard',
               'renderer_code': first['renderer_code'], 'tokenizer': first['tokenizer'],
               'history_reasoning': first.get('history_reasoning'), 'rendering': first.get('rendering'),
               'text_jsonl_sha256': file_sha(target / 'text.jsonl'), 'tokens': totals, 'shards': parts,
               'concatenation': 'shard order 00..NN; each shard is a set of whole repositories'}
    (target / 'receipt.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    for shard in shards:  # the text lives once, in text.jsonl; each shard keeps receipt, provenance and omissions
        (target / 'shards' / shard.name / 'text.jsonl').unlink()
    print(json.dumps({name: totals}), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bulk', type=Path, required=True)
    parser.add_argument('--twin-root', type=Path, required=True)
    parser.add_argument('--tokenizer', action='append', required=True, help='NAME=SNAPSHOT')
    parser.add_argument('--parallel', type=int, default=2)
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    tokenizers = dict(item.split('=', 1) for item in args.tokenizer)
    shards = sorted(p for p in (args.bulk / 'shards').iterdir() if p.is_dir())
    while True:
        ready = [s for s in shards if (s / 'summary.json').exists()]
        todo = [(s, n, t) for s in ready for n, t in tokenizers.items()
                if not (args.twin_root / n / 'shards' / s.name / 'receipt.json').exists()]
        with ThreadPoolExecutor(args.parallel) as pool:
            for line in pool.map(lambda job: render(job[0], job[1], job[2], args.twin_root), todo):
                print(line, flush=True)
        if len(ready) == len(shards):
            failed = [(s.name, n) for s in shards for n in tokenizers
                      if not (args.twin_root / n / 'shards' / s.name / 'receipt.json').exists()]
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
