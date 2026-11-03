#!/usr/bin/env python3
"""`compact-value` operator cases from bgkit ``agent_tool_slots_v8`` repository-question turns.

Question-driven cases (COMPACTION_OPERATOR_CONVERTERS.md section 3b). Final turns of the bgkit repo-QA agent tasks end in
``ANSWER: <exact text>`` after the agent read files, listed directories or grepped. The one tool result that contains the
answer (at least the compact-value floor, ``RULES``) is the operator's source (``tool_output``), the consumer answers the
repository question from the state alone and the target is the exact answer the task's static analysis produced. The seed
is checked: the result's sentences/lines containing the answer, best question overlap first, then more question-overlap
lines (``operator_cases_common.question_selected_spans``).

Split custody follows the tool-digest converter: upstream eval rows and repositories on bgkit's held-out lists are test,
and a group component (repo, task) containing a test row is test throughout. One case per task. A source text that the
compact-value pool already trains on is rejected as ``duplicate-in-pool``; overlap with the compact-params S1 repo-QA files
(same repository files asked about again) is only counted in the manifest notes.

    operator_cases_repo_tool_qa.py [--out-root DIR] [--limit N] [--train-cap N] [--test-cap N]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neuralese_data import bgkit  # noqa: E402
from neuralese_data.common import Reject, parquet_rows  # noqa: E402
from neuralese_data.operator_cases_common import (  # noqa: E402
    OUT_ROOT, RULES, CaseSink, iter_jsonl, pool_text_hashes, question_selected_spans, stable_rank, text_key)
from neuralese_data.operator_cases_tool_digest import Components, heldout_groups  # noqa: E402
from neuralese_data.records import VERSION, group_key, license_, seal, source, strip_markup  # noqa: E402
from natlang_neuralese.data.operator_seeds import budget_chars  # noqa: E402

OPERATOR, SOURCE, STORE = 'compact-value', 'agent-tool-slots-repo-qa', 'agent_tool_slots_v8'
CONVERTER = 'scripts/neuralese_data/operator_cases_repo_tool_qa.py@1'
MIN_CHARS, MAX_CHARS = RULES[OPERATOR]['min_chars'], RULES[OPERATOR]['max_chars']
MAX_ANSWER_CHARS = 300
MAX_TEMPLATE_SHARE = 0.3  # no question template may exceed this share of a split's cap (repetition bound)
TRIVIAL_ANSWERS = {'none', 'true', 'false', 'null', 'yes', 'no', 'nothing'}
COLUMNS = ['context', 'prompt', 'instruction', 'target', 'split', 'meta']


def question_of(row: dict) -> str:
    for message in json.loads(row['instruction']):
        if message.get('role') == 'user' and isinstance(message.get('content'), str) and message['content'].strip():
            text = strip_markup(message['content']).strip()
            if text.startswith('Repository overview'):  # the compressed file tree that preceded the question is not text
                text = text.rsplit('\n\n', 1)[-1].strip()
            return text
    raise Reject('no-question')


def answer_of(row: dict) -> str:
    messages = [m for m in json.loads(row['target']) if isinstance(m, dict) and m.get('role') == 'assistant']
    content = messages[-1].get('content') if messages else None
    if not isinstance(content, str) or 'ANSWER:' not in content:
        raise Reject('no-final-answer')
    answer = strip_markup(content).rsplit('ANSWER:', 1)[1].strip()
    if not answer or len(answer) > MAX_ANSWER_CHARS:
        raise Reject('bad-answer-length')
    if answer.casefold() in TRIVIAL_ANSWERS:
        raise Reject('trivial-answer')
    return answer


def question_template(question: str) -> str:
    """The question with its quoted identifiers/paths blanked: the phrasing family it instantiates."""
    return re.sub(r'`[^`]*`', '`_`', question.split('Answer ')[0]).strip()


def convert_row(row_index: int, row: dict, task_repos: dict) -> dict:
    meta = bgkit._meta(row)
    task_id = str(meta.get('task_id') or '')
    if not task_id.startswith('repo_qa:'):
        raise Reject('not-repo-qa')
    question, answer = question_of(row), answer_of(row)
    slots, prompts = json.loads(row['context']), json.loads(row['prompt'])
    if len(slots) != len(prompts):
        raise Reject('slot-prompt-mismatch')
    lower, found = answer.lower(), []
    for slot, prompt in zip(slots, prompts):
        if not isinstance(slot, str) or not prompt or str(prompt).startswith('compaction') or lower not in slot.lower():
            continue  # slot 0 (empty prompt) is the repo-overview Neuralese marker and 'compaction' slots are notes, not tool results
        found.append((strip_markup(slot).strip(), str(prompt)))
    if not found:
        raise Reject('answer-not-in-tool-results')
    sized = [(text, call) for text, call in found if MIN_CHARS <= len(text) <= MAX_CHARS]
    if not sized:
        raise Reject('source-size')
    text, call = max(sized, key=lambda item: len(item[0]))
    call = call[:-len(question) - 2] if call.endswith(': ' + question) else call.split(': ', 1)[0]
    tool = call.split(' ', 1)[0] if call else 'tool'
    seed_text = question_selected_spans(text, question, [answer], budget_chars(len(text)))
    if seed_text is None:
        raise Reject('answer-not-in-results')
    task = task_repos.get(task_id) or {}
    repo = meta.get('repo') or task.get('repo')
    groups = [group_key('bgkit-task', task_id)] + ([group_key('repo', repo)] if repo else [])
    record = {
        'version': VERSION, 'id': f'bgkit:{STORE}:{row_index}', 'family': 'repo_tool_answer', 'task': 'consume',
        'sources': [source('tool_output', text, meta={
            'tool': tool, 'call': call[:300],
            'operator_seed': {'text': seed_text, 'provenance': 'checked', 'origin': 'code:question-selected-spans'}})],
        'writer': {'instructions': f'Read this {tool} result for: {question}',
                   'result_type': 'Neuralese<ToolResult>', 'context': []},
        'consumer': {'context': [{'role': 'user', 'content': question}], 'withheld': ['sources']},
        'target': {'kind': 'text', 'value': answer, 'alternatives': []},
        'outcome': {'label': 'gold', 'checked': (f"static-analysis:{task['verifier']}" if task.get('verifier')
                                                 else 'static-analysis'),
                    'details': {'answer_in_result': True}},
        'lineage': {'project': 'bgkit', 'store': STORE, 'store_version': bgkit.store_version(STORE), 'row': row_index,
                    'upstream': f'github:{repo}' if repo else None, 'upstream_id': task_id,
                    'upstream_revision': task.get('commit'), 'teacher': None, 'converter': CONVERTER, 'sha256': ''},
        'license': license_('LicenseRef-repository-content', False,
                            'Tool output from public repositories under their own licences.'),
        'split': 'test' if bgkit._split(row) != 'train' else 'train', 'split_groups': groups,
    }
    return seal(record)


def plan(path, limit, train_cap, test_cap, tasks_dir, out_root):
    task_repos = bgkit.load_task_repos(tasks_dir)
    held, components, test_seeds, entries, reasons = heldout_groups(tasks_dir), Components(), set(), {}, []
    for row_index, row in parquet_rows(path, columns=COLUMNS, limit=limit):
        record_id = f'bgkit:{STORE}:{row_index}'
        try:
            record = convert_row(row_index, row, task_repos)
        except Reject as reject:
            if str(reject) not in ('not-repo-qa', 'no-final-answer'):
                reasons.append((record_id, str(reject)))
            continue
        groups = record['split_groups']
        components.union(groups)
        if record['split'] != 'train' or held.intersection(groups):
            test_seeds.update(groups)
        entries.setdefault(groups[0], []).append((stable_rank(record['id'], SOURCE), record['id'], groups,
                                                  text_key(record['sources'][0]['text']),
                                                  question_template(record['consumer']['context'][0]['content'])))
    pool = pool_text_hashes(OPERATOR, exclude={SOURCE}, out_root=out_root)
    test_roots = {components.find(group) for group in test_seeds}
    by_split, chosen, eligible, templates = {'train': [], 'test': []}, {}, Counter(), {}
    for task_group, items in entries.items():
        items.sort()
        for _, record_id, _, _, _ in items[1:]:
            reasons.append((record_id, 'duplicate: another turn of the same task'))
        rank, record_id, groups, key, template = items[0]
        if key in pool:
            reasons.append((record_id, 'duplicate-in-pool: source text already in the compact-value pool'))
            continue
        split = 'test' if components.find(groups[0]) in test_roots else 'train'
        by_split[split].append((rank, record_id, template))
        eligible[split] += 1
    for split, cap in (('train', train_cap), ('test', test_cap)):
        per_template, kept = Counter(), 0
        for _, record_id, template in sorted(by_split[split]):
            if per_template[template] >= max(1, int(cap * MAX_TEMPLATE_SHARE)):
                reasons.append((record_id, f'over-template-cap: {split}'))
            elif kept >= cap:
                reasons.append((record_id, f'over-cap: {split}'))
            else:
                chosen[record_id] = split
                per_template[template] += 1
                kept += 1
        templates[split] = dict(per_template.most_common(12))
    return chosen, reasons, eligible, templates


def compact_params_overlap(out_root: Path, texts: set) -> int:
    """How many chosen source texts the compact-params repo-QA case files also use (counted, not rejected)."""
    path = Path(out_root) / 'compact-params' / 's1-repo-qa.port-records.jsonl'
    if not path.exists():
        return -1
    return sum(1 for record in iter_jsonl(path) if text_key(record['sources'][0]['text']) in texts)


def convert(out_root: Path, limit=None, train_cap=8000, test_cap=300, tasks_dir=bgkit.TASKS_DIR):
    path = tasks_dir / f'{STORE}.parquet'
    chosen, reasons, eligible, templates = plan(path, limit, train_cap, test_cap, tasks_dir, out_root)
    task_repos = bgkit.load_task_repos(tasks_dir)
    texts = set()
    with CaseSink(OPERATOR, SOURCE, converter=__file__, inputs=[path], out_root=out_root,
                  caps={'train': train_cap, 'test': test_cap}) as sink:
        for record_id, reason in reasons:
            sink.reject(record_id, reason)
        for row_index, row in parquet_rows(path, columns=COLUMNS, limit=limit):
            split = chosen.get(f'bgkit:{STORE}:{row_index}')
            if split is None:
                continue
            record = convert_row(row_index, row, task_repos)
            record['split'] = split
            if sink.offer(record):
                texts.add(text_key(record['sources'][0]['text']))
        sink.notes = json.dumps({'eligible': dict(eligible), 'question_templates_kept': templates,
                                 'source_text_overlap_with_compact_params_s1_repo_qa': compact_params_overlap(out_root, texts)},
                                sort_keys=True)
    return json.loads(sink.manifest_path.read_text())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out-root', type=Path, default=OUT_ROOT)
    parser.add_argument('--limit', type=int, help='smoke: first N rows')
    parser.add_argument('--train-cap', type=int, default=8000)
    parser.add_argument('--test-cap', type=int, default=300)
    args = parser.parse_args(argv)
    manifest = convert(args.out_root, args.limit, args.train_cap, args.test_cap)
    print(json.dumps({key: manifest[key] for key in ('counts', 'rejects', 'seed_provenance', 'notes')}))


if __name__ == '__main__':
    main()
