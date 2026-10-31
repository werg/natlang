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


# --- v3 additions: ConvFinQA and MultiHiertt ------------------------------------------------------------------------

def _finqa_page_splits():
    """FinQA's own split of each report page; ConvFinQA reuses FinQA pages, so a page keeps FinQA's split unit
    (`finqa-page:<filename>`) and split class."""
    splits = {}
    for split in ('train', 'dev', 'test'):
        with open(f'{BASE}/finqa/{split}.jsonl') as handle:
            for line in handle:
                splits[json.loads(line)['filename']] = 'train' if split == 'train' else 'validation'
    return splits


CONVFINQA_INSTRUCTIONS = [
    'question continues a conversation about a financial report excerpt (text and table). history holds the earlier '
    'questions with their answers; question may refer to them ("that", "the change"). Compute the answer exactly in '
    'eval and return it as a number.',
    'Answer the next question in this conversation about report text and table. Earlier questions and the numbers '
    'they were answered with are in history. Calculate in code and return the number.',
]
FIN_TOLERANCE = {'level': 'normalized', 'normalization': 'numeric-tolerance', 'tolerance': 5e-3,
                 'percent_equivalent': True}


def convfinqa(cap_train=1500, cap_eval=150, seed=20261031):
    pages = _finqa_page_splits()
    out = []
    for split, cap in (('train', cap_train), ('dev', cap_eval * 2)):
        rows = json.load(open(f'{BASE}/convfinqa/files/{split}_turn.json'))
        random.Random(seed).shuffle(rows)
        n = 0
        for r in rows:
            a = r['annotation']
            turn = int(a['turn_ind'])
            questions = list(a['dialogue_break'])
            answers = __import__('ast').literal_eval(a['exe_ans_list']) if isinstance(a['exe_ans_list'], str) else a['exe_ans_list']
            if turn < 1 or turn >= len(questions) or len(answers) != len(questions):
                continue
            value = answers[turn]
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            upstream_split = pages.get(r['filename'])
            if upstream_split is None:
                continue  # a page FinQA does not list: no split unit to share
            value = round(float(value), 6)
            value = int(value) if value.is_integer() else value
            unit = f'finqa-page:{r["filename"]}'
            history = [{'question': q, 'answer': v} for q, v in zip(questions[:turn], answers[:turn])]
            text = [s for s in r['pre_text'] + r['post_text'] if s.strip() not in ('', '.')]
            if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in answers[:turn]):
                continue  # a yes/no turn in the history: the history type holds numbers only
            wrong = [w for w in [answers[turn - 1], value * 1.5 + 1] if round(float(w), 4) != round(float(value), 4)]
            out.append(make_case(
                source='convfinqa', family='report_conversation', shape=f'convfinqa-turn{min(turn, 4)}',
                upstream='czyssrs/ConvFinQA', upstream_id=r['id'], upstream_split=upstream_split, unit=unit,
                groups=[unit, f'convfinqa-dialogue:{r["id"].rsplit("_", 1)[0]}'], root_name='answer_next_question',
                args={'question': 'string', 'history': '{ question: string; answer: number }[]', 'text': 'string[]',
                      'table': 'string[][]'}, returns='number', instruction=pick(CONVFINQA_INSTRUCTIONS, r['id']),
                inputs={'question': questions[turn], 'history': history, 'text': text, 'table': r['table_ori']},
                expected=value, oracle=FIN_TOLERANCE, level=min(3, 1 + turn // 2), license_='MIT', domain='relational',
                wrong=wrong, aliases=[float(value)], revision=''))
            n += 1
            if n >= cap:
                break
    return out


def html_table_rows(html):
    from html.parser import HTMLParser

    class Rows(HTMLParser):
        def __init__(self):
            super().__init__()
            self.rows, self.row, self.cell, self.span = [], None, None, 1

        def handle_starttag(self, tag, attrs):
            if tag == 'tr':
                self.row = []
            elif tag in ('td', 'th'):
                self.cell, self.span = [], int(dict(attrs).get('colspan') or 1)

        def handle_endtag(self, tag):
            if tag in ('td', 'th') and self.row is not None and self.cell is not None:
                self.row.extend([' '.join(''.join(self.cell).split())] * self.span)
                self.cell = None
            elif tag == 'tr' and self.row is not None:
                self.rows.append(self.row)
                self.row = None

        def handle_data(self, data):
            if self.cell is not None:
                self.cell.append(data)

    parser = Rows()
    parser.feed(html)
    return parser.rows


def _sentence_keys(sentences):
    return {text_digest(' '.join(s.lower().split()))[:16] for s in sentences if len(s) > 60}


MULTIHIERTT_INSTRUCTIONS = [
    'Answer question from a financial report: paragraphs holds its text and tables its tables (each a list of rows). '
    'The figures may sit in different tables. Compute arithmetic exactly in eval and return a number, or for a '
    'question that asks which item or year, return its text as written in the report.',
    'question is about a report with several hierarchical tables and paragraphs. Find the rows and columns it needs '
    'across tables, calculate in code, and return the answer.',
]
MULTIHIERTT_NEURALESE = {
    'helper': 'read_table', 'args': {'question': 'string', 'table': 'string[][]'}, 'returns': 'Neuralese<string>',
    'instruction': 'Read table for the figures question needs and write a private note of the relevant rows, '
                   'columns and values, or that the table has none of them.',
    'root_note': 'read_table gives a private note on one table. You may read each table with it and combine the '
                 'notes before computing the answer in eval.',
}


def multihiertt(cap_train=1500, cap_eval=150, seed=20261031):
    """MultiHiertt (train and dev; test answers are withheld upstream). Report pages that share a long sentence with
    any FinQA/ConvFinQA or TAT-QA page are dropped, so one report never sits on two splits across sources."""
    import ast
    seen = set()
    for split in ('train', 'dev', 'test'):
        with open(f'{BASE}/finqa/{split}.jsonl') as handle:
            for line in handle:
                r = json.loads(line)
                seen |= _sentence_keys(r['pre_text'] + r['post_text'])
    for path in ('train', 'dev', 'test_gold'):
        with open(f'{BASE}/tatqa/{path}.jsonl') as handle:
            for line in handle:
                seen |= _sentence_keys(p['text'] for p in json.loads(line)['paragraphs'])
    parse = lambda v: ast.literal_eval(v) if isinstance(v, str) else v
    out, overlapped = [], 0
    for split, cap in (('train', cap_train), ('dev', cap_eval * 2)):
        n = 0
        rows, rng = [], random.Random(seed)
        with open(f'{BASE}/multihiertt/{split}.jsonl') as handle:  # streamed: seeded reservoir of 3x the cap
            for count, line in enumerate(handle):
                if len(rows) < cap * 3:
                    rows.append(line)
                elif (j := rng.randrange(count + 1)) < cap * 3:
                    rows[j] = line
        rng.shuffle(rows)
        for line in rows:
            r = json.loads(line)
            qa, paragraphs = parse(r['qa']), [p for p in parse(r['paragraphs']) if p.strip()]
            if _sentence_keys(paragraphs) & seen:
                overlapped += 1
                continue
            tables = [html_table_rows(t) for t in parse(r['tables'])]
            if len(json.dumps(tables)) + len(json.dumps(paragraphs)) > 60000:
                continue
            answer = qa['answer']
            if qa['question_type'] == 'arithmetic':
                value = round(float(answer), 6)
                value = int(value) if value.is_integer() else value
                returns, expected, oracle = 'number', value, FIN_TOLERANCE
                wrong, aliases = [value * 1.5 + 1, -value - 1], [float(value)]
            else:
                text = str(answer).strip()
                number = _plain_number(text)
                if number is not None:
                    returns, expected = 'number | string', number
                    oracle = {'level': 'normalized', 'normalization': 'numeric-tolerance', 'tolerance': 1e-6}
                    wrong, aliases = [number + 1], [text]
                elif 0 < len(text.split()) <= 8:
                    returns, expected = 'string', text
                    oracle = {'level': 'span', 'normalization': 'qa', 'threshold': 0.8}
                    wrong, aliases = ['none', paragraphs[0][:60]], [text.lower()]
                else:
                    continue
            unit = f'multihiertt:{r["uid"]}'
            out.append(make_case(
                source='multihiertt', family=f'multi_table_{qa["question_type"]}',
                shape=f'multihiertt-{qa["question_type"]}-{min(len(tables), 4)}tables', upstream='psunlpgroup/MultiHiertt',
                upstream_id=r['uid'], upstream_split='train' if split == 'train' else 'validation', unit=unit,
                groups=[unit], root_name='answer_from_report',
                args={'question': 'string', 'paragraphs': 'string[]', 'tables': 'string[][][]'}, returns=returns,
                instruction=pick(MULTIHIERTT_INSTRUCTIONS, r['uid']),
                inputs={'question': qa['question'], 'paragraphs': paragraphs, 'tables': tables}, expected=expected,
                oracle=oracle, level=2 + (len(tables) > 2), license_='MIT', domain='relational', wrong=wrong,
                aliases=aliases, revision='', neuralese_room='per-table notes fanned in to the computation',
                neuralese=MULTIHIERTT_NEURALESE))
            n += 1
            if n >= cap:
                break
    print(json.dumps({'multihiertt_dropped_shared_report_sentences': overlapped}))
    return out
