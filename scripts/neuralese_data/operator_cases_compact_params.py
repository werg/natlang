"""compact-params cases from the closed S1 long-document QA families (COMPACTION_OPERATOR_CONVERTERS.md section 4).

A call receives a large document argument; the learned operator compacts that argument's listing and the call
answers its instruction from the compacted state alone. Records are selected from the already split-closed S1
`final/` families, never reconverted: the single large source stays the operator's source, the consumer question
and gold target stay exactly as in S1, and only `writer.instructions` is rewritten into the listing purpose.

Selection is two pass and streaming. Pass 1 keeps, per family and split, the cheapest `stable_rank` candidates as
file offsets (bounded heap, no record bodies); pass 2 seeks to those offsets and offers them to the CaseSink in
rank order until the split cap is met (dedupe rejects are replaced by the next-ranked candidate).
"""
from __future__ import annotations

import argparse
import heapq
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neuralese_data.operator_cases_common import (  # noqa: E402
    IMITATION_LABELS, OUT_ROOT, RULES, CaseSink, source_text, stable_rank)
from neuralese_data.records import seal  # noqa: E402
from natlang_neuralese.common.paths import resolve as data_path  # noqa: E402

S1 = Path(data_path('data_hdd', 'natlang-development-data', 'data', 'neuralese', 'port-records', 'full-20261003',
                    'resume-20261003T1948-review1', 'final'))
OPERATOR = 'compact-params'
TEST_CAP = 300
# (S1 family, output source name, train cap) -- spec section 4.
FAMILIES = (('repo_qa', 's1-repo-qa', 9000), ('qa_narrative', 's1-qa-narrative', 5000),
            ('qa_narrative_long', 's1-qa-narrative-long', 4000), ('qa_long_context', 's1-qa-long-context', 4000),
            ('qa_mcq', 's1-qa-mcq', 2129), ('qa_multihop', 's1-qa-multihop', 3000),
            ('qa_trivia', 's1-qa-trivia', 3000))
OVERSAMPLE = 1.2   # candidates kept beyond the cap so dedupe rejects can be replaced by the next-ranked one


def consumer_question(record: dict) -> str | None:
    for message in (record.get('consumer') or {}).get('context') or []:
        if message.get('role') == 'user' and isinstance(message.get('content'), str) and message['content'].strip():
            return message['content']
    return None


def eligibility(record: dict) -> str | None:
    """None when the record is a compact-params candidate, else the drop reason."""
    sources = record.get('sources') or []
    if len(sources) != 1:
        return 'multi-source'
    if (record.get('outcome') or {}).get('label') not in IMITATION_LABELS:
        return 'label'
    rules = RULES[OPERATOR]
    if sources[0].get('role') not in rules['roles']:
        return 'source-role'
    length = len(source_text(sources[0]))
    if length < rules['min_chars']:
        return 'below-4096-chars'
    if length > rules['max_chars']:
        return 'above-120000-chars'
    if consumer_question(record) is None:
        return 'no-consumer-question'
    return None


def to_case(record: dict) -> dict:
    """Rewrite writer.instructions into the listing purpose; everything else stays as in S1 (re-sealed)."""
    role = record['sources'][0]['role']
    record['writer']['instructions'] = f'The call that receives this {role} answers: {consumer_question(record)}'
    return seal(record)


def split_of(record: dict) -> str:
    return 'test' if record['split'] in ('test', 'validation') else 'train'


def select(path: Path, caps: dict, sink: CaseSink, family: str, limit: int | None):
    """Pass 1: {split: [(rank, offset)]} of the best-ranked eligible records, plus drop counts."""
    heaps = {'train': [], 'test': []}
    seen = Counter()
    offset = 0
    with open(path, 'rb') as stream:
        for count, line in enumerate(stream):
            if limit is not None and count >= limit:
                break
            start, offset = offset, offset + len(line)
            if not line.strip():
                continue
            record = json.loads(line)
            seen[split_of(record)] += 1
            reason = eligibility(record)
            if reason:
                sink.reject(record.get('id'), f'{reason}: {family}')
                seen['drop:' + reason] += 1
                continue
            split = split_of(record)
            heap = heaps[split]
            item = (-stable_rank(record['id'], sink.source), start)
            keep = int(caps[split] * OVERSAMPLE) + 1
            if len(heap) < keep:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)
            seen['eligible-' + split] += 1
    return {split: sorted((-negative, start) for negative, start in heap) for split, heap in heaps.items()}, seen


def convert_family(family: str, name: str, train_cap: int, out_root: Path, limit: int | None) -> dict:
    path = S1 / f'{family}.port-records.jsonl'
    caps = {'train': train_cap, 'test': TEST_CAP}
    notes = Counter()
    with CaseSink(OPERATOR, name, converter=__file__, inputs=[path], out_root=out_root, caps=caps) as sink:
        chosen, seen = select(path, caps, sink, family, limit)
        offered = {'train': 0, 'test': 0}
        with open(path, 'rb') as stream:
            for split in ('train', 'test'):
                for _, start in chosen[split]:
                    if sink.counts[split] >= caps[split]:
                        notes[f'over-cap-{split}'] += 1
                        continue
                    stream.seek(start)
                    record = json.loads(stream.readline())
                    offered[split] += 1
                    sink.offer(to_case(record))
        notes.update(seen)
        sink.notes = (f'S1 family {family}; cap by stable_rank({name!r}) per split, test+validation share one test cap; '
                      f'selection {json.dumps(dict(sorted(notes.items())))}')
        summary = {'source': name, 'family': family, 'counts': dict(sink.counts), 'seen': dict(sorted(notes.items()))}
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--limit', type=int, default=None, help='smoke: read only the first N lines of each family')
    parser.add_argument('--out-root', default=str(OUT_ROOT))
    parser.add_argument('--families', nargs='*', help='S1 family names (default: all)')
    args = parser.parse_args(argv)
    for family, name, cap in FAMILIES:
        if args.families and family not in args.families:
            continue
        print(json.dumps(convert_family(family, name, cap, Path(args.out_root), args.limit)), flush=True)


if __name__ == '__main__':
    main()
