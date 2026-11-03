#!/usr/bin/env python3
"""`compact-value` operator cases from bgkit ``web_search_r1_v3`` (COMPACTION_OPERATOR_CONVERTERS.md section 3).

One case per row: the retrieved result pages are the operator's source (one ``tool_output``), the consumer answers the
question from the state alone, and the target is the dataset's gold answer (Search-R1 NQ/HotpotQA train questions with
exact-match checked answers). The store is all upstream train; test is carved by question world
(``stable_rank(question_group) % 100 < 3``), so every row of one question lands on the same side.

    operator_cases_web_search_r1.py [--out-root DIR] [--limit N] [--train-cap N] [--test-cap N]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import string
import sys
import unicodedata
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neuralese_data import bgkit  # noqa: E402
from neuralese_data.common import Reject, parquet_rows  # noqa: E402
from neuralese_data.operator_cases_common import OUT_ROOT, RULES, CaseSink, stable_rank  # noqa: E402
from neuralese_data.records import VERSION, group_key, license_, seal, source, strip_answer_prefix, strip_markup, text_hash  # noqa: E402

STORE = 'web_search_r1_v3'
PATH = bgkit.TASKS_DIR / f'{STORE}.parquet'
OPERATOR, SOURCE = 'compact-value', 'web-search-r1'
CONVERTER = 'scripts/neuralese_data/operator_cases_web_search_r1.py@1'
TEST_PERCENT = 3
MIN_CHARS, MAX_CHARS = RULES[OPERATOR]['min_chars'], RULES[OPERATOR]['max_chars']
_ANSWER_PREFIX = re.compile(r'^\s*ANSWER:', re.IGNORECASE)
_PAGE_TITLE = re.compile(r'^\[[^\]\n]*\]\s*([^\n]+)')


def json_list(raw, what: str) -> list:
    try:
        value = json.loads(raw) if isinstance(raw, str) else list(raw)
    except (json.JSONDecodeError, TypeError) as error:
        raise Reject(f'bad-{what}: {error}') from error
    if not isinstance(value, list) or not value:
        raise Reject(f'bad-{what}: not a non-empty list')
    return value


def question_of(prompts: list, meta: dict) -> str:
    """The question after the first ``: `` of the first prompt; the store's own ``meta.question`` wins when the
    prompt ends with it (a search query that itself contains ``: `` would otherwise cut the question short)."""
    first = prompts[0]
    if not isinstance(first, str) or ': ' not in first:
        raise Reject('no-question')
    question = strip_markup(first.split(': ', 1)[1]).strip()
    declared = meta.get('question')
    if isinstance(declared, str) and declared.strip() and first.endswith(': ' + declared.strip()):
        question = declared.strip()
    if not question:
        raise Reject('no-question')
    return question


def answer_of(target: list) -> str:
    messages = [m for m in target if isinstance(m, dict) and m.get('role') == 'assistant']
    content = messages[-1].get('content') if messages else None
    if not isinstance(content, str) or not _ANSWER_PREFIX.match(content):
        raise Reject('no-final-answer')
    answer = strip_answer_prefix(strip_markup(content)).strip()
    if not answer:
        raise Reject('empty-answer')
    return answer


def normalize_answer(text: str) -> str:
    """SQuAD-style exact-match normalisation: case, punctuation, articles and spacing."""
    text = unicodedata.normalize('NFKC', text).casefold()
    text = ''.join(ch for ch in text if ch not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', text).split())


def gold_aliases(meta: dict) -> list:
    value = meta.get('gold')
    value = [value] if isinstance(value, str) else value
    return [item.strip() for item in value if isinstance(item, str) and item.strip()] if isinstance(value, list) else []


def alternatives_of(meta: dict, answer: str) -> list:
    out = []
    for item in gold_aliases(meta):
        if item != answer and item not in out:
            out.append(item)
    return out


def checked_exact_match(meta: dict, answer: str) -> bool:
    """The answer is exact-match to a dataset gold answer (or, with no gold list, the store flagged it ``em``)."""
    aliases = gold_aliases(meta)
    if aliases:
        return normalize_answer(answer) in {normalize_answer(item) for item in aliases}
    return meta.get('answer_match') == 'em'


def convert_row(row_index: int, row: dict) -> dict:
    meta = bgkit._meta(row)
    pages = [strip_markup(page) for page in json_list(row.get('context'), 'context')]
    if not all(isinstance(page, str) and page.strip() for page in pages):
        raise Reject('bad-context: empty page')
    prompts = json_list(row.get('prompt'), 'prompt')
    question = question_of(prompts, meta)
    answer = answer_of(json_list(row.get('target'), 'target'))
    if not checked_exact_match(meta, answer):
        raise Reject('answer-not-em')
    text = '\n\n'.join(pages)
    if answer.casefold() not in text.casefold():
        raise Reject('answer-not-in-results')
    queries = [p.split(': ', 1)[0].removeprefix('web search ').strip() for p in prompts if isinstance(p, str)]
    question_group = group_key('search-r1-q', hashlib.sha256(question.encode()).hexdigest()[:20])
    groups = [question_group]
    for page in pages:
        match = _PAGE_TITLE.match(page)
        if match:
            groups.append(group_key('wiki', match.group(1)))
    record = {
        'version': VERSION, 'id': f'bgkit:{STORE}:{row_index}', 'family': 'web_search_answer', 'task': 'consume',
        'sources': [source('tool_output', text, meta={'tool': 'web_search', 'queries': queries})],
        'writer': {'instructions': f'Read these web_search results for: {question}',
                   'result_type': 'Neuralese<ToolResult>', 'context': []},
        'consumer': {'context': [{'role': 'user', 'content': f'{question}\nAnswer with a short phrase.'}],
                     'withheld': ['sources']},
        'target': {'kind': 'text', 'value': answer, 'alternatives': alternatives_of(meta, answer)},
        'outcome': {'label': 'gold', 'checked': 'dataset-answer-em',
                    'details': {'answer_match': meta.get('answer_match'), 'dataset': meta.get('dataset')}},
        'lineage': {'project': 'bgkit', 'store': STORE, 'store_version': 'v3', 'row': row_index,
                    'upstream': meta.get('dataset') or 'ThornZ/Search-R1-SFT', 'upstream_id': None,
                    'upstream_revision': None, 'teacher': None, 'converter': CONVERTER, 'sha256': ''},
        'license': license_('LicenseRef-unverified', False, str(meta.get('license') or 'unspecified on the dataset card')),
        'split': 'test' if stable_rank(question_group, SOURCE) % 100 < TEST_PERCENT else 'train',
        'split_groups': list(dict.fromkeys(groups)),
    }
    return seal(record)


def convert(out_root: Path, limit=None, train_cap=15_000, test_cap=1_000, path=PATH):
    """Two passes: candidate ids with their ranks, then the capped selection is written."""
    reasons, candidates, seen = [], {'train': [], 'test': []}, set()
    for row_index, row in parquet_rows(path, limit=limit):
        try:
            record = convert_row(row_index, row)
        except Reject as reject:
            reasons.append((f'bgkit:{STORE}:{row_index}', str(reject)))
            continue
        length = len(record['sources'][0]['text'])
        if not MIN_CHARS <= length <= MAX_CHARS:
            reasons.append((record['id'], f'source-size: {length} chars'))
            continue
        key = (text_hash(record['sources'][0]['text']), record['target']['value'])
        if key in seen:
            reasons.append((record['id'], 'duplicate: same sources and target'))
            continue
        seen.add(key)
        candidates[record['split']].append((stable_rank(record['id'], SOURCE), record['id']))
    keep = set()
    for split, cap in (('train', train_cap), ('test', test_cap)):
        ranked = sorted(candidates[split])
        keep.update(record_id for _, record_id in ranked[:cap])
        reasons.extend((record_id, f'over-cap: {split}') for _, record_id in ranked[cap:])
    with CaseSink(OPERATOR, SOURCE, converter=__file__, inputs=[path], out_root=out_root,
                  caps={'train': train_cap, 'test': test_cap}) as sink:
        for record_id, reason in reasons:
            sink.reject(record_id, reason)
        for row_index, row in parquet_rows(path, limit=limit):
            if f'bgkit:{STORE}:{row_index}' not in keep:
                continue
            sink.offer(convert_row(row_index, row))
        sink.notes = json.dumps({'eligible': {split: len(items) for split, items in candidates.items()}}, sort_keys=True)
    return json.loads(sink.manifest_path.read_text())


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-root', type=Path, default=OUT_ROOT)
    parser.add_argument('--limit', type=int, help='smoke: first N rows')
    parser.add_argument('--train-cap', type=int, default=15_000)
    parser.add_argument('--test-cap', type=int, default=1_000)
    args = parser.parse_args(argv)
    manifest = convert(args.out_root, args.limit, args.train_cap, args.test_cap)
    print(json.dumps({key: manifest[key] for key in ('counts', 'rejects', 'output', 'notes')}, indent=1))


if __name__ == '__main__':
    main()
