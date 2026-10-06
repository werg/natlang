"""bgkit task stores → plain JSONL for the inline-curriculum generators (node cannot read parquet).

Data-only reuse (owner decision): content, the consumer's need and the gold target; bgkit's harness (framing,
sentinels, chat markup) stays behind. One file per store under OUT, rows {store, index, tool_name, tool_args,
instruction, prompt, context, target, split, meta} with meta decoded. A store's rows keep their bgkit order, so an
index names a row across exports.

    .venv-neuralese/bin/python scripts/neuralese_data/export_bgkit_jsonl.py tool_digest_v2 repo_qa_file_v3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from .bgkit import TASKS_DIR

OUT = Path('/home/werg/data/bgkit-export')
COLUMNS = ['tool_name', 'tool_args', 'instruction', 'prompt', 'context', 'target', 'split', 'meta']


def export(store: str, out: Path, limit: int | None) -> int:
    source = pq.ParquetFile(TASKS_DIR / f'{store}.parquet')
    columns = [name for name in COLUMNS if name in source.schema_arrow.names]
    path = out / f'{store}.jsonl'
    written = 0
    with path.with_suffix('.jsonl.tmp').open('w') as handle:
        for batch in source.iter_batches(batch_size=512, columns=columns):
            for row in batch.to_pylist():
                try:
                    row['meta'] = json.loads(row.get('meta') or '{}')
                except json.JSONDecodeError:
                    row['meta'] = {'raw': row.get('meta')}
                handle.write(json.dumps({'store': store, 'index': written, **row}, ensure_ascii=False) + '\n')
                written += 1
                if limit is not None and written >= limit:
                    break
            if limit is not None and written >= limit:
                break
    path.with_suffix('.jsonl.tmp').rename(path)
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stores', nargs='+')
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--limit', type=int, help='first N rows of each store')
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    for store in args.stores:
        print(json.dumps({'store': store, 'rows': export(store, args.out, args.limit)}), flush=True)


if __name__ == '__main__':
    main()
