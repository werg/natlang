"""Exact-number problems: GSM8K and MATH (numeric final answers only) as word problems computed in eval; FinQA as
numeric questions over a financial report's text and table (relative tolerance 0.5%, percent/fraction equivalent;
items whose upstream answer string and executed program value disagree are dropped as noisy)."""
from __future__ import annotations

import json
import random
import re

from ..common import SOURCES_DIR, make_case, pick, text_digest

BASE = SOURCES_DIR

WORD_INSTRUCTIONS = [
    'Solve problem. Compute every quantity exactly in eval rather than by mental arithmetic, and return the final '
    'numeric answer as a number.',
    'problem is a word problem with one numeric answer. Work out the steps, do the arithmetic in code, and return the '
    'answer as a plain number (no units).',
    'Return the number that answers problem. Calculate it with eval and check that it answers what was asked.',
]


def _plain_number(text):
    t = text.strip().replace(',', '').replace('$', '').rstrip('.')
    if re.fullmatch(r'-?\d+(\.\d+)?', t):
        value = float(t)
        return int(value) if value.is_integer() else value
    return None


def gsm8k(cap_train=1500, cap_eval=150, seed=20261031):
    out = []
    for split, cap in (('train', cap_train), ('test', cap_eval * 2)):
        rows = [json.loads(line) for line in open(f'{BASE}/gsm8k/{split}.jsonl')]
        random.Random(seed).shuffle(rows)
        for r in rows[:cap]:
            gold = _plain_number(r['answer'].split('####')[-1])
            if gold is None:
                continue
            steps = r['answer'].count('<<')
            unit = f'gsm8k:{text_digest(r["question"])[:16]}'
            out.append(make_case(
                source='gsm8k', family='word_problem', shape=f'gsm8k-{min(steps, 6)}steps', upstream='openai/gsm8k',
                upstream_id=unit.split(':', 1)[1], upstream_split='validation' if split == 'test' else 'train', unit=unit,
                groups=[unit], root_name='solve_problem', args={'problem': 'string'}, returns='number',
                instruction=pick(WORD_INSTRUCTIONS, unit), inputs={'problem': r['question']}, expected=gold,
                oracle='normalized', level=min(3, max(1, steps // 2)), license_='MIT', domain='other',
                slice_='inline_placement', wrong=[gold + 1, gold * 2 + 3], aliases=[float(gold)],
                revision=''))
    return out


def _boxed(solution):
    start = solution.rfind('\\boxed{')
    if start < 0:
        return None
    depth, i = 0, start + len('\\boxed{')
    for j in range(i, len(solution)):
        if solution[j] == '{':
            depth += 1
        elif solution[j] == '}':
            if depth == 0:
                return solution[i:j]
            depth -= 1
    return None


def math(cap_train=1200, cap_eval=150, seed=20261031):
    import glob
    out = []
    for split, cap in (('train', cap_train), ('test', cap_eval * 2)):
        rows = [json.loads(line) for path in sorted(glob.glob(f'{BASE}/hendrycks_math/*_{split}.jsonl')) for line in open(path)]
        random.Random(seed).shuffle(rows)
        n = 0
        for r in rows:
            boxed = _boxed(r['solution'])
            gold = _plain_number(boxed.replace('\\!', '').replace('{,}', ',')) if boxed else None
            if gold is None:
                continue
            level = int(re.sub(r'\D', '', r['level']) or 3)
            unit = f'math:{text_digest(r["problem"])[:16]}'
            out.append(make_case(
                source='math', family=f'competition_{r["type"].lower().replace(" ", "_").replace("&", "and")}',
                shape=f'math-level{level}', upstream='EleutherAI/hendrycks_math', upstream_id=unit.split(':', 1)[1],
                upstream_split='validation' if split == 'test' else 'train', unit=unit, groups=[unit],
                root_name='solve_problem', args={'problem': 'string'}, returns='number',
                instruction=pick(WORD_INSTRUCTIONS, unit), inputs={'problem': r['problem']}, expected=gold,
                oracle='normalized', level=min(3, max(0, (level - 1) // 1 - 1)), license_='MIT', domain='logic',
                slice_='inline_placement', wrong=[gold + 1, -gold if gold else 2], aliases=[float(gold)], revision='',
                difficulty=r['level']))
            n += 1
            if n >= cap:
                break
    return out


FINQA_INSTRUCTIONS = [
    'Answer question from the financial report excerpt: text holds its sentences and table its table rows. Compute '
    'the answer exactly in eval and return it as a number (a percentage as its number, e.g. 12.5 for 12.5%).',
    'question asks for a figure derived from report text and table. Find the inputs, calculate in code, and return '
    'the resulting number.',
]


def finqa(cap_train=1500, cap_eval=150, seed=20261031):
    out = []
    for split, cap in (('train', cap_train), ('dev', cap_eval), ('test', cap_eval)):
        rows = [json.loads(line) for line in open(f'{BASE}/finqa/{split}.jsonl')]
        random.Random(seed).shuffle(rows)
        n = 0
        for r in rows:
            qa = r['qa']
            value = qa.get('exe_ans')
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            stated = _plain_number(str(qa.get('answer', '')).replace('%', ''))
            if stated is None:
                continue
            close = lambda a, b: abs(a - b) <= 5e-3 * max(abs(a), abs(b), 1e-9)
            if not (close(stated, value) or close(stated, value * 100) or close(stated * 100, value)):
                continue
            value = round(value, 6)
            unit = f'finqa-page:{r["filename"]}'
            text = [s for s in r['pre_text'] + r['post_text'] if s.strip() not in ('', '.')]
            out.append(make_case(
                source='finqa', family='report_arithmetic', shape=f'finqa-{len(qa.get("steps") or [])}ops',
                upstream='czyssrs/FinQA', upstream_id=r['id'], upstream_split='validation' if split != 'train' else 'train',
                unit=unit, groups=[unit], root_name='answer_from_report',
                args={'question': 'string', 'text': 'string[]', 'table': 'string[][]'}, returns='number',
                instruction=pick(FINQA_INSTRUCTIONS, r['id']),
                inputs={'question': qa['question'], 'text': text, 'table': r['table_ori']}, expected=value,
                oracle=({'level': 'normalized', 'alternates': [stated]} if float(value).is_integer() and
                        isinstance(stated, int) and stated == value else
                        {'level': 'normalized', 'normalization': 'numeric-tolerance', 'tolerance': 5e-3,
                         'percent_equivalent': True, 'alternates': [stated]}),
                level=min(3, len(qa.get('steps') or [])), license_='MIT', domain='relational',
                wrong=[value * 1.5 + 1, -value - 1], aliases=[stated], revision=''))
            n += 1
            if n >= cap:
                break
    return out


TATQA_INSTRUCTIONS = [
    'Answer question from the annual-report excerpt: table holds its table rows and paragraphs its text. Compute '
    'arithmetic answers exactly in eval. Give numbers in the units the report uses (for example 12.5 for 12.5%, or 840.4 '
    'for $840.4 million when amounts are in millions), and text answers as the words from the report.',
    'question is about a company report given as table and paragraphs. Find or calculate the answer in code and '
    'return it: a number in the report\'s own units, the exact text span, or a list of spans when several items '
    'are asked for.',
]


def _tatqa_number(text):
    t = str(text).replace(',', '').replace('$', '').replace('%', '').strip()
    if t.startswith('(') and t.endswith(')'):
        t = '-' + t[1:-1]
    return _plain_number(t)


def tatqa(cap_train=1500, cap_eval=150, seed=20261031):
    import ast
    out = []
    for split, path in (('train', 'train'), ('validation', 'dev'), ('validation', 'test_gold')):
        contexts = [json.loads(line) for line in open(f'{BASE}/tatqa/{path}.jsonl')]
        random.Random(seed).shuffle(contexts)
        n, cap = 0, cap_train if split == 'train' else cap_eval
        for ctx in contexts:
            unit = f'tatqa:context:{ctx["table"]["uid"]}'
            table = ctx['table']['table']
            paragraphs = [p['text'] for p in sorted(ctx['paragraphs'], key=lambda p: int(p['order']))]
            for q in ctx['questions'][:3]:
                answer, kind = q['answer'], q['answer_type']
                if isinstance(answer, str) and answer.startswith('['):
                    answer = ast.literal_eval(answer)
                if kind in ('arithmetic', 'count') or (kind == 'span' and isinstance(answer, list) and len(answer) == 1
                                                       and _tatqa_number(answer[0]) is not None):
                    value = _tatqa_number(answer[0] if isinstance(answer, list) else answer)
                    if value is None:
                        continue
                    if isinstance(value, float):
                        value = round(value, 4)
                    returns, expected = 'number', value
                    oracle = {'level': 'normalized', 'normalization': 'numeric-tolerance', 'tolerance': 1e-6,
                              'abs_tolerance': 0.006}
                    wrong, aliases = [value * 1.1 + 1, -value - 1], [float(value)]
                elif kind == 'span' and isinstance(answer, list) and len(answer) == 1 and len(answer[0].split()) <= 8:
                    returns, expected = 'string', answer[0]
                    oracle = {'level': 'span', 'normalization': 'qa', 'threshold': 0.8}
                    others = [p for p in paragraphs if p and answer[0] not in p][:1]
                    wrong, aliases = ([others[0][:80]] if others else []) + ['none'], [answer[0].lower()]
                elif kind == 'multi-span' and isinstance(answer, list) and len(answer) > 1:
                    returns, expected = 'string[]', answer
                    oracle = {'level': 'normalized', 'normalization': 'answer-list'}
                    wrong, aliases = [answer[:-1], answer + ['Total']], [list(reversed(answer))]
                else:
                    continue
                out.append(make_case(
                    source='tatqa', family=f'report_qa_{kind.replace("-", "_")}', shape=f'tatqa-{kind}-{q["answer_from"]}',
                    upstream='NExTplusplus/TAT-QA', upstream_id=q['uid'], upstream_split=split, unit=unit, groups=[unit],
                    root_name='answer_from_report', args={'question': 'string', 'table': 'string[][]', 'paragraphs': 'string[]'},
                    returns=returns, instruction=pick(TATQA_INSTRUCTIONS, q['uid']),
                    inputs={'question': q['question'], 'table': table, 'paragraphs': paragraphs}, expected=expected,
                    oracle=oracle, level=2 if kind == 'arithmetic' else 1, license_='MIT', domain='relational',
                    wrong=wrong, aliases=aliases, revision='', difficulty=q.get('scale') or None))
                n += 1
            if n >= cap:
                break
    return out
