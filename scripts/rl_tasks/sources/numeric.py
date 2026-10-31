"""Exact-number problems: GSM8K and MATH (numeric final answers only) as word problems computed in eval; FinQA as
numeric questions over a financial report's text and table (relative tolerance 0.5%, percent/fraction equivalent;
items whose upstream answer string and executed program value disagree are dropped as noisy)."""
from __future__ import annotations

import json
import random
import re

from ..common import make_case, pick, sha

BASE = '/home/werg/data/external-rl-sources'

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
            unit = f'gsm8k:{sha(r["question"])[:16]}'
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
            unit = f'math:{sha(r["problem"])[:16]}'
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
                oracle={'level': 'normalized', 'normalization': 'numeric-tolerance', 'tolerance': 5e-3,
                        'percent_equivalent': True, 'alternates': [stated]},
                level=min(3, len(qa.get('steps') or [])), license_='MIT', domain='relational',
                wrong=[value * 1.5 + 1, -value - 1], aliases=[stated], revision=''))
            n += 1
            if n >= cap:
                break
    return out
