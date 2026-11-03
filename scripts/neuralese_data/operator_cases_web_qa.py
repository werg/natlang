#!/usr/bin/env python3
"""`compact-value` operator cases from bgkit web-research trajectories: tool results -> exact-match gold answer.

Question-driven cases (COMPACTION_OPERATOR_CONVERTERS.md section 3b). Each bgkit ``web_*`` store holds agent turns of
web-research trajectories; the ``final`` row of a trajectory ends in ``ANSWER: <answer>``. Its tool results (search
result pages and opened pages, in order, without the learned-compaction slots) are the operator's source (one
``tool_output``), the consumer answers the research question from the state alone, and the target is the answer that
the dataset's gold answers confirm by exact match and that occurs verbatim in the tool results (the state can carry it).
The seed is checked: the sentences that contain a gold answer, best question overlap first, then more
question-overlap sentences (``operator_cases_common.question_selected_spans``).

Diversity: at most one case per question world over all stores (several trajectories of one question differ only in
the pages they retrieved), and a question with any upstream-eval row is test throughout (all stores). A source text that
the compact-value pool already trains on (other converters' case files) is rejected as ``duplicate-in-pool``.

    operator_cases_web_qa.py [--out-root DIR] [--limit N] [--train-cap N] [--test-cap N] [--stores a b]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neuralese_data import bgkit  # noqa: E402
from neuralese_data.common import Reject, parquet_rows  # noqa: E402
from neuralese_data.operator_cases_common import (  # noqa: E402
    OUT_ROOT, RULES, CaseSink, pool_text_hashes, question_selected_spans, stable_rank, text_key)
from neuralese_data.operator_cases_web_search_r1 import (  # noqa: E402
    alternatives_of, answer_of, checked_exact_match, gold_aliases, json_list)
from neuralese_data.records import VERSION, group_key, license_, seal, source, strip_markup  # noqa: E402
from natlang_neuralese.data.operator_seeds import budget_chars  # noqa: E402

OPERATOR = 'compact-value'
CONVERTER = 'scripts/neuralese_data/operator_cases_web_qa.py@1'
# store -> output source name. The v3 stores supersede the v2 ones (same trajectories, compaction-chained slots).
STORES = {'web_openresearcher_v3': 'web-openresearcher', 'web_openseeker_v3': 'web-openseeker',
          'web_webshaper_v3': 'web-webshaper', 'web_sds_v3': 'web-sds', 'web_miroverse_nc_v2': 'web-miroverse'}
MIN_CHARS, MAX_CHARS = RULES[OPERATOR]['min_chars'], RULES[OPERATOR]['max_chars']
MAX_ANSWER_CHARS = 200
COLUMNS = ['context', 'prompt', 'target', 'split', 'meta']


def question_group(question: str) -> str:
    return group_key('web-q', hashlib.sha256(' '.join(question.split()).casefold().encode()).hexdigest()[:20])


def clean_answer(answer: str) -> str:
    answer = answer.strip().strip('*`').strip()
    if '\n' in answer:
        raise Reject('multi-line-answer')
    if not answer or len(answer) > MAX_ANSWER_CHARS:
        raise Reject('bad-answer-length')
    return answer


def tool_pages(row: dict) -> tuple[list[str], list[str]]:
    """The tool-result texts (no learned-compaction slots) and the calls that produced them."""
    slots = json_list(row.get('context'), 'context')
    prompts = json_list(row.get('prompt'), 'prompt')
    if len(slots) != len(prompts):
        raise Reject('slot-prompt-mismatch')
    pages, calls = [], []
    for slot, prompt in zip(slots, prompts):
        if not isinstance(slot, str) or not isinstance(prompt, str):
            raise Reject('bad-context')
        if prompt.startswith('compaction:') or slot.lstrip().startswith('{"compaction"'):
            continue
        text = strip_markup(slot).strip()
        if text:
            pages.append(text)
            calls.append(prompt.split(': ', 1)[0][:200])
    if not pages:
        raise Reject('no-tool-results')
    return pages, calls


def convert_row(store: str, row_index: int, row: dict) -> dict:
    meta = bgkit._meta(row)
    if not meta.get('final'):
        raise Reject('not-final')
    question = ' '.join(str(meta.get('question') or '').split())
    if not question:
        raise Reject('no-question')
    target = json_list(row.get('target'), 'target')
    checked = 'dataset-answer-em'
    try:
        answer = clean_answer(answer_of(target))
    except Reject as reject:
        # MiroVerse finals are written reports without an ANSWER line; the report is flagged exact-match to the gold
        # answer, which is then the short answer the consumer must give (a dataset gold answer, not a teacher's).
        aliases = gold_aliases(meta)
        if str(reject) != 'no-final-answer' or meta.get('answer_match') != 'em' or not aliases:
            raise
        answer, checked = clean_answer(aliases[0]), 'dataset-gold-answer-em-report'
    if not checked_exact_match(meta, answer):
        raise Reject('answer-not-em')
    pages, calls = tool_pages(row)
    text = '\n\n'.join(pages)
    if not MIN_CHARS <= len(text) <= MAX_CHARS:
        raise Reject(f'source-size: {len(text)} chars')
    aliases = [answer, *gold_aliases(meta)]
    seed_text = question_selected_spans(text, question, aliases, budget_chars(len(text)))
    if seed_text is None:
        raise Reject('answer-not-in-results')
    group = question_group(question)
    record = {
        'version': VERSION, 'id': f'bgkit:{store}:{row_index}', 'family': 'web_research_answer', 'task': 'consume',
        'sources': [source('tool_output', text, meta={
            'tool': 'web', 'calls': calls[:40],
            'operator_seed': {'text': seed_text, 'provenance': 'checked', 'origin': 'code:question-selected-spans'}})],
        'writer': {'instructions': f'Read these web results for: {question}',
                   'result_type': 'Neuralese<ToolResult>', 'context': []},
        'consumer': {'context': [{'role': 'user', 'content': f'{question}\nAnswer with a short phrase.'}],
                     'withheld': ['sources']},
        'target': {'kind': 'text', 'value': answer, 'alternatives': alternatives_of(meta, answer)},
        'outcome': {'label': 'gold', 'checked': checked,
                    'details': {'answer_match': meta.get('answer_match'), 'dataset': meta.get('dataset'),
                                'answer_in_results': True}},
        'lineage': {'project': 'bgkit', 'store': store, 'store_version': bgkit.store_version(store), 'row': row_index,
                    'upstream': meta.get('dataset'), 'upstream_id': meta.get('trajectory_id'),
                    'upstream_revision': None, 'teacher': None, 'converter': CONVERTER, 'sha256': ''},
        'license': license_('LicenseRef-unverified', 'NON-COMMERCIAL' in str(meta.get('license')).upper(),
                            str(meta.get('license') or 'unspecified on the dataset card')),
        'split': 'train', 'split_groups': [group],
    }
    return seal(record)


def plan(stores, limit, train_cap, test_cap, tasks_dir, out_root):
    """Pass 1: eligible candidates per store with ranks and the global test question set; returns the chosen ids."""
    test_groups, candidates, reasons = set(), {}, []
    for store in stores:
        path = tasks_dir / f'{store}.parquet'
        for row_index, row in parquet_rows(path, columns=COLUMNS, limit=limit):
            meta = bgkit._meta(row)
            question = ' '.join(str(meta.get('question') or '').split())
            if question and bgkit.SPLIT_MAP.get(str(row.get('split') or '').strip(), 'train') != 'train':
                test_groups.add(question_group(question))
            record_id = f'bgkit:{store}:{row_index}'
            try:
                record = convert_row(store, row_index, row)
            except Reject as reject:
                if str(reject) != 'not-final':
                    reasons.append((record_id, str(reject), store))
                continue
            candidates.setdefault(record['split_groups'][0], []).append(
                (stable_rank(record['id'], 'web-qa'), record['id'], store, text_key(record['sources'][0]['text'])))
    pool = pool_text_hashes(OPERATOR, exclude=set(STORES.values()), out_root=out_root)
    chosen, by_split, eligible = {}, {'train': [], 'test': []}, Counter()
    for group, items in candidates.items():
        items.sort()
        for _, record_id, store, key in items[1:]:
            reasons.append((record_id, 'duplicate: another trajectory of the same question world', store))
        _, record_id, store, key = items[0]
        if key in pool:
            reasons.append((record_id, 'duplicate-in-pool: source text already in the compact-value pool', store))
            continue
        split = 'test' if group in test_groups else 'train'
        by_split[split].append((items[0][0], record_id, store))
        eligible[f'{store}:{split}'] += 1
    kept = Counter()
    for split, cap in (('train', train_cap), ('test', test_cap)):
        for _, record_id, store in sorted(by_split[split]):
            if kept[(store, split)] >= cap:
                reasons.append((record_id, f'over-cap: {split}', store))
                continue
            kept[(store, split)] += 1
            chosen[record_id] = split
    return chosen, reasons, eligible


def convert(out_root: Path, stores, limit=None, train_cap=4000, test_cap=300, tasks_dir=bgkit.TASKS_DIR):
    chosen, reasons, eligible = plan(stores, limit, train_cap, test_cap, tasks_dir, out_root)
    manifests = {}
    for store in stores:
        name = STORES[store]
        path = tasks_dir / f'{store}.parquet'
        with CaseSink(OPERATOR, name, converter=__file__, inputs=[path], out_root=out_root,
                      caps={'train': train_cap, 'test': test_cap}) as sink:
            for record_id, reason, reason_store in reasons:
                if reason_store == store:
                    sink.reject(record_id, reason)
            for row_index, row in parquet_rows(path, columns=COLUMNS, limit=limit):
                split = chosen.get(f'bgkit:{store}:{row_index}')
                if split is None:
                    continue
                record = convert_row(store, row_index, row)
                record['split'] = split
                sink.offer(record)
            sink.notes = json.dumps({'eligible': {k: v for k, v in eligible.items() if k.startswith(store + ':')}},
                                    sort_keys=True)
        manifests[name] = json.loads(sink.manifest_path.read_text())
    return manifests


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out-root', type=Path, default=OUT_ROOT)
    parser.add_argument('--limit', type=int, help='smoke: first N rows of each store')
    parser.add_argument('--train-cap', type=int, default=4000, help='per store')
    parser.add_argument('--test-cap', type=int, default=300, help='per store')
    parser.add_argument('--stores', nargs='*', default=list(STORES))
    args = parser.parse_args(argv)
    for name, manifest in convert(args.out_root, args.stores, args.limit, args.train_cap, args.test_cap).items():
        print(json.dumps({'source': name, **{k: manifest[k] for k in ('counts', 'rejects', 'seed_provenance')}}),
              flush=True)


if __name__ == '__main__':
    main()
