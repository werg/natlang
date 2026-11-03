#!/usr/bin/env python3
"""`extract` operator cases from broad graph cases (plans/neuralese/COMPACTION_OPERATOR_CONVERTERS.md section 6).

One extractor site per document (Neuralese-prompted, shared task statement), feeding a parent that answers from all
extracts without seeing the documents. Cases are the `natlang.broad-graph-case/1` rows of ``broad_graph_cases.py build``
(upstream splits, S1 group closure and source-review holds are already applied there); the documents are the
extractor's sources, the dataset gold answer is the target. Each document's view seeds its extractor's producer span
(``meta.operator_seed``, owner correction 2026-11-03): dataset supporting facts/paragraphs, answer-span sentences,
highlighted cells, relevant spans and the dataset's "no relevant facts" line for distractors are ``gold``; the
code-computed greedy ROUGE oracle against the Multi-News reference summary is ``checked``.

Selection under the caps is deterministic: eligible cases sorted by ``stable_rank(case id)``. Train takes at most
``--train-cap`` per source and ``--total-cap`` overall; when the per-source maxima exceed the total the largest sources
are lowered to a common level (water-filling), so small sources such as QMSum keep every eligible case. Test takes at
most ``--test-cap`` per source. Protected-world custody and the protection union are applied by the shared renderer.

    operator_cases_broad_extract.py [--cases CASES.jsonl] [--out-root DIR] [--limit N] [--sources a,b]
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neuralese_data.operator_cases_common import OUT_ROOT, RULES, CaseSink, stable_rank  # noqa: E402
from neuralese_data.records import TARGET_KINDS, VERSION, license_, seal, source  # noqa: E402

from natlang_neuralese.common.paths import resolve as data_path  # noqa: E402

CASES = data_path('data_hdd', 'natlang-compaction-operators-20261103', 'broad-cases', 'cases.jsonl')
SOURCES = ('hotpotqa', '2wiki', 'musique', 'multi_news', 'qmsum', 'fetaqa', 'squad')
TASK_KINDS = {'evidence', 'passage', 'meeting', 'news', 'table'}  # short text / summary answers; sql, calls, logs are not
CASE_SCHEMA = 'natlang.broad-graph-case/1'
WRITER_PREFIX = 'Extract what this task needs from the document. Task: '
RESULT_TYPE = 'Neuralese<string>'


def sink_name(source_name: str) -> str:
    return f'broad-{source_name}'


def ineligible(case: dict) -> str | None:
    """Reason a case cannot be an extract case, from the case alone (no record is built)."""
    if case.get('schema') != CASE_SCHEMA:
        return 'unknown-case-schema'
    if case.get('task_kind') not in TASK_KINDS:
        return 'task-kind-not-text-answer'
    documents = case.get('documents') or []
    low, high = RULES['extract']['sources']
    if not low <= len(documents) <= high:
        return 'document-count'
    lengths = [len(doc.get('text') or '') for doc in documents]
    if min(lengths) < RULES['extract']['min_chars']:
        return 'short-document'
    if max(lengths) > RULES['extract']['max_chars']:
        return 'long-document'
    request = (case.get('request') or {}).get('text')
    if not isinstance(request, str) or not request.strip():
        return 'missing-request'
    if not isinstance(case.get('answer'), str) or not case['answer'].strip():
        return 'missing-answer'
    if case.get('split') not in ('train', 'test'):
        return 'unknown-split'
    return None


def document_seed(document: dict) -> dict | None:
    """The gold/checked seed of one broad-case document from its view, or None (the shared mechanical seed then)."""
    view, origin = document.get('view'), document.get('view_origin')
    if not isinstance(view, str) or not view.strip() or not isinstance(origin, str):
        return None
    provenance = 'checked' if origin.startswith('code-computed:') else 'gold'
    return {'text': view, 'provenance': provenance, 'origin': f'broad-case-view:{origin}'}


def to_record(case: dict) -> dict:
    """The port record of one eligible case (section 6 mapping)."""
    oracle = case['oracle']
    level, alternates = (oracle['level'], oracle.get('alternates') or []) if isinstance(oracle, dict) else (oracle, [])
    kind = case.get('answer_kind') or 'text'
    target = {'kind': kind if kind in TARGET_KINDS else 'text', 'value': case['answer']}
    if alternates:
        target['alternatives'] = list(alternates)
    request = case['request']
    record_id = 'broad-extract:' + ''.join(str(case['id']).split())
    return seal({
        'version': VERSION, 'id': record_id, 'family': 'extract_' + case['task_kind'], 'task': 'consume',
        'sources': [source('document', doc['text'], title=doc.get('title'),
                           meta=({'operator_seed': document_seed(doc)} if document_seed(doc) else None))
                    for doc in case['documents']],
        'writer': {'instructions': WRITER_PREFIX + request['text'], 'result_type': RESULT_TYPE, 'context': []},
        'consumer': {'context': [{'role': 'user',
                                  'content': case['consumer']['instructions'] + '\n\n' + request['name'] + ': '
                                  + request['text']}],
                     'withheld': ['sources']},
        'target': target,
        'outcome': {'label': 'gold', 'checked': level},
        'lineage': {'project': 'upstream', 'store': case['dataset'], 'store_version': None, 'row': case['upstream_id'],
                    'upstream': case['dataset'], 'upstream_id': case['upstream_id'],
                    'upstream_revision': case['revision'], 'teacher': None, 'converter': Path(__file__).name,
                    'sha256': '', 'notes': {'broad_case': case['id'], 'upstream_split': case['upstream_split'],
                                            'answer_origin': case.get('answer_origin'), 'domain': case.get('domain')}},
        'license': license_(case['license']['spdx'], notes=case['license'].get('notes', '')),
        'split': case['split'], 'split_groups': list(case['groups'])})


def water_fill(sizes: dict, total: int, per_source: int) -> dict:
    """Per-source train quotas: min(size, per_source), lowered to a common level when their sum exceeds ``total``."""
    quota = {name: min(size, per_source) for name, size in sizes.items()}
    if sum(quota.values()) <= total:
        return quota
    low, high = 0, per_source
    while low < high:  # largest level whose capped sum fits the total
        mid = (low + high + 1) // 2
        if sum(min(size, mid) for size in quota.values()) <= total:
            low = mid
        else:
            high = mid - 1
    quota = {name: min(size, low) for name, size in quota.items()}
    spare = total - sum(quota.values())  # distribute the remainder one case at a time, sources by name
    for name in sorted(quota):
        if spare and quota[name] < min(sizes[name], per_source):
            quota[name] += 1
            spare -= 1
    return quota


def scan(cases_path: Path, sources, limit: int | None):
    """Pass 1 (streaming): per source and split, eligible (rank, offset) pairs and per-reason ineligible ids."""
    eligible = {s: {'train': [], 'test': []} for s in sources}
    refused, seen = {s: [] for s in sources}, Counter()
    with open(cases_path, 'rb') as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            if not line.strip():
                continue
            case = json.loads(line)
            name = case.get('source')
            if name not in eligible:
                continue
            seen[name] += 1
            if limit is not None and seen[name] > limit:
                continue
            reason = ineligible(case)
            if reason:
                refused[name].append((case.get('id'), reason))
            else:
                eligible[name][case['split']].append((stable_rank(case['id'], sink_name(name)), offset, case['id']))
    return eligible, refused, seen


def read_case(stream, offset: int) -> dict:
    stream.seek(offset)
    return json.loads(stream.readline())


def convert(cases_path: Path, out_root: Path, sources=SOURCES, *, train_cap: int = 8000, total_cap: int = 30000,
            test_cap: int = 300, limit: int | None = None) -> dict:
    eligible, refused, seen = scan(cases_path, sources, limit)
    for split in eligible.values():
        for items in split.values():
            items.sort()
    quota = water_fill({s: len(eligible[s]['train']) for s in sources}, total_cap, train_cap)
    summary = {}
    with contextlib.ExitStack() as stack, open(cases_path, 'rb') as stream:
        for name in sources:
            sink = stack.enter_context(CaseSink(
                'extract', sink_name(name), converter=__file__, inputs=[cases_path], out_root=out_root,
                caps={'train_per_source': train_cap, 'train_total': total_cap, 'test_per_source': test_cap,
                      'train_quota': quota[name]}))
            for case_id, reason in refused[name]:
                sink.reject(case_id, reason)
            omitted = {}
            for split, cap in (('train', quota[name]), ('test', test_cap)):
                taken = 0
                for position, (_, offset, _case_id) in enumerate(eligible[name][split]):
                    if taken >= cap:
                        omitted[split + '-cap'] = len(eligible[name][split]) - position
                        break
                    taken += sink.offer(to_record(read_case(stream, offset)))
            sink.notes = json.dumps({'broad_cases_seen': seen[name], 'eligible': {k: len(v) for k, v in
                                                                                  eligible[name].items()},
                                     'omitted_by_cap': omitted, 'selection': 'stable_rank(case id, source name)',
                                     'limit': limit}, sort_keys=True)
            summary[name] = sink
    return {name: {'counts': dict(sink.counts), 'rejects': dict(sink.rejects), 'quota': quota[name]}
            for name, sink in summary.items()}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', type=Path, default=CASES)
    parser.add_argument('--out-root', type=Path, default=OUT_ROOT)
    parser.add_argument('--sources', default=','.join(SOURCES))
    parser.add_argument('--limit', type=int, help='smoke: read only the first N cases of each source')
    parser.add_argument('--train-cap', type=int, default=8000)
    parser.add_argument('--total-cap', type=int, default=30000)
    parser.add_argument('--test-cap', type=int, default=300)
    args = parser.parse_args(argv)
    report = convert(args.cases, args.out_root, tuple(args.sources.split(',')), train_cap=args.train_cap,
                     total_cap=args.total_cap, test_cap=args.test_cap, limit=args.limit)
    print(json.dumps(report, indent=1, sort_keys=True))


if __name__ == '__main__':
    main()
