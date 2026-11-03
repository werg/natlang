#!/usr/bin/env python3
"""`compact-value` operator cases from bgkit tool digests (plans/neuralese/COMPACTION_OPERATOR_CONVERTERS.md section 2).

Inputs: the closed S1 ``tool_digest`` records and the bgkit digest stores that S1 did not convert
(``bgkit.TOOL_DIGEST_EXTRA_STORES``). The digests themselves are teacher-written and are not imitation targets: the
sources, writer and split groups are kept and the consumer is re-targeted to an exact, code-derived recall task over
the tool result the operator compacted:

* ``:refs``  the exact file paths, path:line locations and URLs of the result, in the order they first appear
  (``records.exact_refs_from``, at most 24 listed);
* ``:lines`` the line count and the first and last non-empty lines, for results with at least 40 lines.

Split custody: the upstream split decides (test/validation stay test). Rows whose repository is on a bgkit held-out
list go to test, and a group component that contains any test row (union over ``split_groups``) is test throughout, so
related worlds never straddle the split. Selection under the caps is by ``stable_rank`` (all ``:refs`` first, then
``:lines`` fill the remainder).

    operator_cases_tool_digest.py [--out-root DIR] [--limit N] [--train-cap N] [--test-cap N]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from natlang_neuralese.common import paths  # noqa: E402
from neuralese_data import bgkit  # noqa: E402
from neuralese_data.common import Reject, parquet_rows  # noqa: E402
from neuralese_data.operator_cases_common import OUT_ROOT, RULES, CaseSink, iter_jsonl, stable_rank  # noqa: E402
from neuralese_data.records import exact_refs_from, group_key, leakage, seal, text_hash  # noqa: E402

S1_TOOL_DIGEST = paths.resolve('data_hdd', 'natlang-development-data/data/neuralese/port-records/full-20261003/'
                               'resume-20261003T1948-review1/final/tool_digest.port-records.jsonl')
OPERATOR, SOURCE = 'compact-value', 'tool-digest'
CONVERTER = 'scripts/neuralese_data/operator_cases_tool_digest.py@2'
MAX_LISTED_REFS = 24
MIN_REFS = 3
MIN_LINES = 40
MAX_BOUNDARY_LINE_CHARS = 300  # first/last line targets beyond this are not a usable recall answer
MIN_CHARS, MAX_CHARS = RULES[OPERATOR]['min_chars'], RULES[OPERATOR]['max_chars']


def split_lines(text: str) -> list[str]:
    lines = text.split('\n')
    return lines[:-1] if lines and lines[-1] == '' else lines


def tool_and_purpose(record: dict) -> tuple[str, str]:
    source = record['sources'][0]
    tool = (source.get('meta') or {}).get('tool') or 'tool'
    instructions = record['writer']['instructions']
    prefix = f'Read this {tool} result for: '
    if instructions.startswith(prefix):
        purpose = instructions[len(prefix):]
    else:
        purpose = instructions.partition('for: ')[2]
    purpose = purpose.strip()
    if not purpose:
        raise Reject('no-purpose')
    return tool, purpose


def ordered_refs(text: str) -> tuple[list[str], int]:
    """The refs (first 64 as exact_refs_from finds them) in first-appearance order, and how many were found."""
    refs = exact_refs_from(text, limit=64)
    order = sorted(range(len(refs)), key=lambda i: (text.find(refs[i]['text']), i))
    return [refs[i]['text'] for i in order], len(refs)


def line_stats(text: str) -> dict | None:
    lines = split_lines(text)
    if len(lines) < MIN_LINES:
        return None
    non_empty = [line.strip() for line in lines if line.strip()]
    if not non_empty:
        return None
    return {'lines': len(lines), 'first': non_empty[0], 'last': non_empty[-1]}


def retarget(record: dict, kind: str) -> dict:
    """The ``kind`` (``refs`` or ``lines``) case of a teacher digest record; raises Reject when it does not apply."""
    text = record['sources'][0]['text']
    tool, purpose = tool_and_purpose(record)
    if kind == 'refs':
        refs, total = ordered_refs(text)
        if total < MIN_REFS:
            raise Reject('no-exact-refs')
        question = (f'From the {tool} result read for "{purpose}", list the exact file paths, locations (path:line) '
                    f'and URLs it contains, in the order they first appear, as a JSON array of strings.')
        if total > MAX_LISTED_REFS:
            question += f' List at most the first {MAX_LISTED_REFS}.'
        value, checked, details = refs[:MAX_LISTED_REFS], 'code:exact_refs_from', {'refs_total': total}
        seed_text = '\n'.join(value)
        family = 'tool_digest_refs'
    else:
        stats = line_stats(text)
        if stats is None:
            raise Reject('too-few-lines')
        if max(len(stats['first']), len(stats['last'])) > MAX_BOUNDARY_LINE_CHARS:
            raise Reject('boundary-line-too-long')
        question = (f'How many lines does the {tool} result read for "{purpose}" have, and what are its first and last '
                    f'non-empty lines? Answer as JSON {{"lines": n, "first": "...", "last": "..."}}.')
        value, checked, details = stats, 'code:line-stats', {}
        seed_text = f'{stats["lines"]} lines\nfirst: {stats["first"]}\nlast: {stats["last"]}'
        family = 'tool_digest_lines'
    lineage = dict(record['lineage'])
    notes = dict(lineage.get('notes') or {})
    notes['digest_teacher'] = lineage.get('teacher')
    lineage.update(teacher=None, notes=notes, converter=CONVERTER)
    out = {key: record[key] for key in ('version', 'task', 'writer', 'license', 'split', 'split_groups')}
    # Checked seed (spec section 2 step 5): the producer span starts as the exact code-derived answer text.
    seeded = dict(record['sources'][0])
    seeded['meta'] = {**(seeded.get('meta') or {}),
                      'operator_seed': {'text': seed_text, 'provenance': 'checked', 'origin': checked}}
    out.update(
        sources=[seeded],
        id=f'{record["id"]}:{kind}', family=family,
        consumer={'context': [{'role': 'user', 'content': question}], 'withheld': ['sources']},
        target={'kind': 'json', 'value': value, 'alternatives': []},
        outcome={'label': 'checked', 'checked': checked,
                 'details': {'retargeted_from': 'teacher-digest', **details}},
        lineage=lineage)
    out = seal(out)
    if leakage(out):
        raise Reject('target-in-writer')
    return out


def heldout_groups(tasks_dir: Path = bgkit.TASKS_DIR) -> set:
    """Repo group keys on bgkit's held-out lists: eval_repos_heldout.txt and the repos of every *_heldout task file."""
    repos = set(bgkit.load_heldout_repos(tasks_dir))
    for path in sorted(tasks_dir.glob('repo_*_tasks_heldout.jsonl')):
        with open(path, encoding='utf-8') as stream:
            for line in stream:
                if line.strip() and json.loads(line).get('repo'):
                    repos.add(json.loads(line)['repo'])
    return {group_key('repo', repo) for repo in repos}


def base_records(limit, s1_path, tasks_dir, counters):
    """Yield (origin, record, reject) for the S1 rows, then the unconverted bgkit stores (deduped by source text)."""
    seen = set()
    for record in iter_jsonl(s1_path, limit):
        seen.add(text_hash(record['sources'][0]['text']))
        yield 'S1', record, None
    counters['s1_texts'] = len(seen)
    task_repos = bgkit.load_task_repos(tasks_dir)
    for store in bgkit.TOOL_DIGEST_EXTRA_STORES:
        convert = bgkit.converter_for(store, task_repos, set())
        for row_index, row in parquet_rows(tasks_dir / f'{store}.parquet', limit=limit):
            try:
                record = convert(row_index, row)
                key = text_hash(record['sources'][0]['text'])
                if key in seen:
                    raise Reject('duplicate-source')
                seen.add(key)
            except Reject as reject:
                yield store, None, (f'bgkit:{store}:{row_index}', str(reject))
                continue
            yield store, record, None


class Components:
    def __init__(self):
        self.parent = {}

    def find(self, key):
        self.parent.setdefault(key, key)
        while self.parent[key] != key:
            self.parent[key] = self.parent[self.parent[key]]
            key = self.parent[key]
        return key

    def union(self, keys):
        roots = [self.find(key) for key in keys]
        for root in roots[1:]:
            self.parent[root] = roots[0]


def plan(limit, s1_path, tasks_dir, train_cap, test_cap):
    """Pass 1: decide which cases exist, their final splits, and which survive the caps."""
    held, counters, rejected = heldout_groups(tasks_dir), Counter(), []
    components, test_seeds, entries, seen = Components(), set(), [], set()
    for origin, record, reject in base_records(limit, s1_path, tasks_dir, counters):
        if reject:
            rejected.append((reject[0], f'{reject[1].split(":")[0]}: {origin}'))
            continue
        groups = record['split_groups']
        components.union(groups)
        if record['split'] != 'train' or held.intersection(groups):
            test_seeds.update(groups)
        text = record['sources'][0]['text']
        if not MIN_CHARS <= len(text) <= MAX_CHARS:
            rejected.append((record['id'], f'source-size: {origin} {len(text)} chars'))
            continue
        kinds = []
        for kind in ('refs', 'lines'):
            try:
                retarget(record, kind)
            except Reject as reject:
                rejected.append((f'{record["id"]}:{kind}', f'{reject}: {origin}'))
                continue
            key = (text_hash(text), kind)  # the same result text with another purpose has the same exact answer
            if key in seen:
                rejected.append((f'{record["id"]}:{kind}', f'duplicate: {origin} same source text and answer kind'))
                continue
            seen.add(key)
            kinds.append(kind)
        entries.append((origin, record['id'], record['split'], list(groups), kinds))
    test_roots = {components.find(group) for group in test_seeds}
    chosen, by_split = {}, {'train': {'refs': [], 'lines': []}, 'test': {'refs': [], 'lines': []}}
    for origin, record_id, split, groups, kinds in entries:
        final = 'test' if split != 'train' or components.find(groups[0]) in test_roots else 'train'
        if final == 'test' and split == 'train':
            counters[f'moved-to-test:{origin}'] += 1
        for kind in kinds:
            by_split[final][kind].append((stable_rank(f'{record_id}:{kind}', SOURCE), record_id, kind, origin, split))
    for final, cap in (('train', train_cap), ('test', test_cap)):
        left = cap
        for kind in ('refs', 'lines'):
            ranked = sorted(by_split[final][kind])
            for _, record_id, kind_, origin, split in ranked[:left]:
                chosen[f'{record_id}:{kind_}'] = (final, split, origin)
            for _, record_id, kind_, origin, _ in ranked[left:]:
                rejected.append((f'{record_id}:{kind_}', f'over-cap: {final} {origin}'))
            counters[f'eligible:{final}:{kind}'] = len(ranked)
            left = max(0, left - len(ranked))
    return chosen, counters, rejected


def convert(out_root: Path, limit=None, train_cap=30_000, test_cap=2_000, s1_path=S1_TOOL_DIGEST,
            tasks_dir=bgkit.TASKS_DIR):
    chosen, counters, rejected = plan(limit, s1_path, tasks_dir, train_cap, test_cap)
    inputs = [s1_path, *(tasks_dir / f'{store}.parquet' for store in bgkit.TOOL_DIGEST_EXTRA_STORES)]
    selected = Counter(f'{origin}:{final}:{case_id.rsplit(":", 1)[1]}' for case_id, (final, _, origin) in chosen.items())
    with CaseSink(OPERATOR, SOURCE, converter=__file__, inputs=inputs, out_root=out_root,
                  caps={'train': train_cap, 'test': test_cap}) as sink:
        for case_id, reason in rejected:
            sink.reject(case_id, reason)
        for origin, record, reject in base_records(limit, s1_path, tasks_dir, Counter()):
            if reject:
                continue
            for kind in ('refs', 'lines'):
                case_id = f'{record["id"]}:{kind}'
                if case_id not in chosen:
                    continue
                final, split, _ = chosen[case_id]
                case = retarget(record, kind)
                case['split'] = 'validation' if split == 'validation' else final
                sink.offer(case)
        sink.notes = json.dumps({'pipeline': dict(sorted(counters.items())), 'selected': dict(sorted(selected.items()))},
                                sort_keys=True)
    return json.loads(sink.manifest_path.read_text())


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-root', type=Path, default=OUT_ROOT)
    parser.add_argument('--limit', type=int, help='smoke: first N rows of each input')
    parser.add_argument('--train-cap', type=int, default=30_000)
    parser.add_argument('--test-cap', type=int, default=2_000)
    args = parser.parse_args(argv)
    manifest = convert(args.out_root, args.limit, args.train_cap, args.test_cap)
    print(json.dumps({key: manifest[key] for key in ('counts', 'rejects', 'output')}, indent=1))
    print(manifest['notes'])


if __name__ == '__main__':
    main()
