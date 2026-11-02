"""Bounded ordered parallel indexing using the production native graph parser.

Workers receive raw JSON lines, run the unchanged NativeGraphSource parser and
emit only the same identity metadata plus measured packed-slot count. The
caller authenticates file bytes and stores the resulting cache/geometry with
an explicit builder receipt; no token arrays or plans are retained across rows.
"""
from __future__ import annotations

import json
import multiprocessing
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from natlang_neuralese.common.jsonio import open_jsonl_text
from natlang_neuralese.data.native_sources import NativeGraphSource
from natlang_neuralese.data.native_graph_schema import communication_classification

_WORKER_SOURCE = None
_WORKER_AUDIT = None


def _init(tokenizer, roles, audit=None):
    global _WORKER_SOURCE, _WORKER_AUDIT
    _WORKER_AUDIT=audit
    source = NativeGraphSource.__new__(NativeGraphSource)
    source.kind = 'native-graphs'
    source.pair_model = None
    source.role_ids = tuple(roles)
    source.tokenizer_fingerprint = tokenizer
    source.required_split = 'train'
    _WORKER_SOURCE = source


def _parse_line(line):
    row = json.loads(line)
    example, binding = _WORKER_SOURCE._parse(row)
    if example.split != 'train':
        raise ValueError(f'expected train row, got {example.split!r} in {example.id}')
    entry = {'id': example.id, 'groups': example.source_groups, 'binding': binding,
             'supervised_tokens': sum(weight > 0 for weight in example.target_weights),
             'forward_tokens': example.plan.token_count, **communication_classification(example)}
    base=(entry,len(example.plan.slot_sources),len(example.decisions))
    if _WORKER_AUDIT is None:return base
    from scripts.neuralese_data.audit_acquired_learning_signal import row_signal
    from scripts.neuralese_data.acquired_content_review import content_review
    markers = ((125019,125020) if _WORKER_SOURCE.tokenizer_fingerprint ==
        '133504d493a77e3121f62cc8c83699cb222ee993c46959d604c4b683a2ffdaf3' else None)
    return (*base,row_signal(row,_WORKER_AUDIT,tool_call_markers=markers),content_review(row,_WORKER_AUDIT))


def parallel_rows(path, tokenizer, roles, *, workers=2, max_pending=4, row_limit=None, audit=None):
    """Yield exact native index rows in source order with a bounded raw-line queue."""
    if type(workers) is not int or not 1 <= workers <= 4 or type(max_pending) is not int or not workers <= max_pending <= 8:
        raise ValueError('parallel index needs 1..4 workers and a bounded queue of at most 8')
    if row_limit is not None and (type(row_limit) is not int or row_limit < 1):
        raise ValueError('row limit must be positive')
    seen = set()
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=_init, initargs=(tokenizer, tuple(roles),audit)) as pool:
        pending = deque()
        def result(future):
            value=future.result()
            entry, slots, decisions = value[:3]
            if entry['id'] in seen:
                raise ValueError(f"duplicate example id {entry['id']!r}")
            if slots < 1:
                raise ValueError('packed geometry requires positive KV slots')
            seen.add(entry['id'])
            return value
        with open_jsonl_text(Path(path)) as source:
            for line in source:
                if not line.strip():
                    continue
                pending.append(pool.submit(_parse_line, line))
                if len(pending) >= max_pending:
                    yield result(pending.popleft())
                if row_limit is not None and len(seen) + len(pending) >= row_limit:
                    break
            while pending:
                yield result(pending.popleft())
