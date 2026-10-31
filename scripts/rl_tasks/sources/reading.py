"""Reading comprehension and reasoning over supplied text: SQuAD (several questions per article section), DROP
(discrete reasoning over a passage), QuALITY (long-article multiple choice), bAbI (fact lists), CLUTRR (kinship
stories), LogiQA 2.0 (logic multiple choice)."""
from __future__ import annotations

import json
import random
import re
from collections import defaultdict

from ..common import SOURCES_DIR, distinct_wrong, make_case, pick, text_digest

BASE = SOURCES_DIR


def _rows(path):
    return [json.loads(line) for line in open(path) if line.strip()]


# ---------------------------------------------------------------- SQuAD
SQUAD_INSTRUCTIONS = [
    'Answer every question in questions from article. Each answer is a short span copied from article. Return a record '
    'mapping each question id to its answer.',
    'article is a section of an encyclopedia article and questions lists several questions about it. For each one, '
    'find the words in article that answer it and return a record from question id to that answer text.',
    'Return a record that answers each entry of questions (keyed by its id) with the shortest phrase from article '
    'that answers it.',
]
SQUAD_NEURALESE = {
    'helper': 'find_answer', 'args': {'article': 'string', 'question': 'string'}, 'returns': 'Neuralese<string>',
    'instruction': 'Find where article answers question and write a private note quoting the answering words and the '
                   'sentence they come from.',
    'root_note': 'find_answer gives a private note locating the answer to one question; use one per question, then '
                 'return each answer as the exact words from article.',
}


def squad(cap_train=700, cap_eval=80, per_case=4, seed=20261031):
    out = []
    for split, path, cap in (('train', 'train', cap_train), ('validation', 'validation', cap_eval * 2)):
        by_context = defaultdict(list)
        title_of = {}
        for r in _rows(f'{BASE}/squad_v1.1/{path}.jsonl'):
            by_context[r['context']].append(r)
            title_of[r['context']] = r['title']
        contexts = sorted(by_context, key=lambda c: text_digest(c))
        random.Random(seed).shuffle(contexts)
        pool = [r['answers']['text'][0] for c in contexts[:3000] for r in by_context[c]]
        for context in contexts[:cap]:
            qs = by_context[context][:per_case]
            if len(qs) < 2:
                continue
            title = title_of[context]
            questions = [{'id': f'q{i + 1}', 'question': q['question']} for i, q in enumerate(qs)]
            gold = {f'q{i + 1}': sorted(set(q['answers']['text']), key=q['answers']['text'].index) for i, q in enumerate(qs)}
            first = {k: v[0] for k, v in gold.items()}
            swapped = dict(first)
            keys = list(first)
            swapped[keys[0]], swapped[keys[1]] = first[keys[1]], first[keys[0]]
            other = dict(first)
            other[keys[-1]] = distinct_wrong(first[keys[-1]], pool[int(text_digest(context)[:6], 16) % 1000:], 1)[0]
            wrong = [w for w in (swapped, other) if w != first]
            aliases = [first]
            alt = {k: (v[1] if len(v) > 1 else v[0]) for k, v in gold.items()}
            if alt != first:
                aliases.append(alt)
            unit = f'squad-ctx:{text_digest(context)[:16]}'
            out.append(make_case(
                source='squad', family='article_multi_qa', shape=f'squad-{len(qs)}q', upstream='rajpurkar/squad',
                upstream_id=','.join(q['id'] for q in qs), upstream_split=split, unit=unit,
                groups=[f'wiki:{title.lower()}', unit] + [f'squad-q:{q["id"]}' for q in qs], root_name='answer_questions',
                args={'article': 'string', 'questions': '{ id: string; question: string }[]'},
                returns='Record<string, string>', instruction=pick(SQUAD_INSTRUCTIONS, unit),
                inputs={'article': context, 'questions': questions}, expected=gold,
                oracle={'level': 'normalized', 'normalization': 'qa-alias-map', 'threshold': 0.8}, level=1,
                license_='CC-BY-SA-4.0', domain='relational', wrong=wrong, aliases=aliases,
                reference_root=[['return_result', {'status': 'success', 'value': first}]],
                neuralese_room='one child per question returns located evidence; parent assembles the record',
                revision='', neuralese=SQUAD_NEURALESE))
            out[-1]['semantics']['expected'] = gold
    return out


# ---------------------------------------------------------------- DROP
DROP_INSTRUCTIONS = [
    'Answer question from passage. It may need counting, adding, subtracting, comparing or sorting facts in passage; '
    'compute exactly in eval when numbers are involved. Return only the answer: a number, a date, or a short span.',
    'question asks about passage and often needs arithmetic over the numbers it mentions. Work it out exactly and '
    'return just the answer (a number or a short phrase from passage).',
]


def _number(text):
    t = text.replace(',', '').strip()
    return float(t) if re.fullmatch(r'-?\d+(\.\d+)?', t) else None


def drop(cap_train=1500, cap_eval=150, seed=20261031):
    out = []
    for split, cap in (('train', cap_train), ('validation', cap_eval * 2)):
        rows = _rows(f'{BASE}/drop/{split}.jsonl')
        random.Random(seed).shuffle(rows)
        pool = [r['answers_spans']['spans'][0] for r in rows[:3000]]
        n = 0
        for r in rows:
            spans = list(dict.fromkeys(r['answers_spans']['spans']))
            types = r['answers_spans']['types']
            if not spans or (split == 'train' and len(spans) != 1):
                continue
            if split == 'validation' and len({s.lower() for s in spans}) > 1 and len(set(types)) > 1:
                continue
            gold = spans[0]
            number = _number(gold)
            if number is not None and all(_number(s) == number for s in spans):
                value = int(number) if number.is_integer() else number
                oracle, returns, expected = 'normalized', 'number', value
                wrong = [value + 1, value * 2 + 3]
                aliases = [str(value)]
                shape = 'drop-number'
            else:
                oracle = {'level': 'span', 'normalization': 'qa', 'threshold': 0.8, 'alternates': spans[1:]} \
                    if len(spans) > 1 else {'level': 'span', 'normalization': 'qa', 'threshold': 0.8}
                returns, expected = 'string', gold
                wrong = distinct_wrong(gold, pool[n:] + pool[:n], 2)
                aliases = ['The ' + gold] + spans[1:2]
                shape = f'drop-{types[0]}'
            unit = f'drop-passage:{r["section_id"]}'
            out.append(make_case(
                source='drop', family='passage_reasoning', shape=shape, upstream='ucinlp/drop', upstream_id=r['query_id'],
                upstream_split=split, unit=unit, groups=[unit], root_name='answer_from_passage',
                args={'passage': 'string', 'question': 'string'}, returns=returns,
                instruction=pick(DROP_INSTRUCTIONS, r['query_id']), inputs={'passage': r['passage'], 'question': r['question']},
                expected=expected, oracle=oracle, level=2 if returns == 'number' else 1, license_='CC-BY-SA-4.0',
                domain='relational', slice_='inline_placement', wrong=wrong, aliases=aliases, revision=''))
            n += 1
            if n >= cap:
                break
    return out


# ---------------------------------------------------------------- QuALITY
LETTERS = 'ABCD'
QUALITY_INSTRUCTIONS = [
    'article is a long story or essay. Answer question by choosing the option that article best supports. Return its '
    'letter.',
    'Read article and decide which of options answers question correctly. The answer may depend on details spread '
    'across the whole article. Return the letter of that option.',
]
QUALITY_NEURALESE = {
    'helper': 'read_part', 'args': {'part': 'string', 'question': 'string', 'options': 'Record<string, string>'},
    'returns': 'Neuralese<string>',
    'instruction': 'Read part, one consecutive part of a longer article, and write a private note of everything in it '
                   'that supports or rules out any of options as the answer to question.',
    'root_note': 'read_part gives a private evidence note for one part of article. You may split article into parts '
                 '(for example by paragraph groups), collect the notes, and decide from them.',
}


def quality(cap_train=1200, cap_eval=150, seed=20261031):
    out = []
    for split, cap in (('train', cap_train), ('validation', cap_eval * 2)):
        rows = list({(text_digest(r['article']), r['question'].strip()): r for r in _rows(f'{BASE}/quality/{split}.jsonl')}.values())
        random.Random(seed).shuffle(rows)
        for r in rows[:cap]:
            options = {LETTERS[i]: o for i, o in enumerate(r['options'])}
            gold = LETTERS[r['answer']]
            unit = f'quality-article:{text_digest(r["article"])[:16]}'
            qid = f'{unit}:{text_digest(r["question"])[:10]}'
            out.append(make_case(
                source='quality', family='long_article_mcq', shape=f'quality-{"hard" if r["hard"] else "easy"}',
                upstream='emozilla/quality', upstream_id=qid, upstream_split=split, unit=unit, groups=[unit],
                root_name='choose_answer', args={'article': 'string', 'question': 'string', 'options': 'Record<string, string>'},
                returns='"A" | "B" | "C" | "D"', instruction=pick(QUALITY_INSTRUCTIONS, qid),
                inputs={'article': r['article'], 'question': r['question'].strip(), 'options': options}, expected=gold,
                oracle='exact', level=3 if r['hard'] else 2, license_='CC-BY-4.0', domain='relational',
                wrong=[l for l in LETTERS if l != gold][:2], aliases=[], revision='7142df0',
                neuralese_room='long article split into parts read by children returning evidence notes',
                neuralese=QUALITY_NEURALESE, difficulty='hard' if r['hard'] else 'easy'))
    return out


# ---------------------------------------------------------------- bAbI
BABI_INSTRUCTIONS = [
    'facts is a numbered story, in order. Answer question from it, taking into account that later facts can change '
    'earlier ones. Return only the answer word or words (comma-separated when there are several).',
    'Answer question using the story in facts (in time order). Return just the answer, as a single word, or words '
    'separated by commas when the answer has several parts.',
]
BABI_HISTORY_NEURALESE = {
    'helper': 'track_query_history',
    'args': {'facts': 'string[]', 'question': 'string'},
    'returns': 'Neuralese<string>',
    'instruction': 'Read the complete numbered event history and question. Keep a concise, ordered timeline only for '
                   'the people, objects, places, and transitions needed to answer. Preserve each relevant fact id. '
                   'For a "before X" query, state the value immediately before event X; for an "after X" query, '
                   'state the value after it. Distinguish a person moving from an object changing holder or location. '
                   'Never omit a repeated move or infer a transition absent from the facts.',
    'root_note': 'Use the whole numbered history, even when it is long. Call `const timeline = await '
                 'track_query_history(facts, question);` once with the full list. Check the decisive fact ids and the state immediately '
                 'before or after the named event, then return only the typed answer. Do not truncate the history or '
                 'answer from the final state when the question names an earlier time.',
}


def babi(per_task=70, eval_per_task=8, seed=20261031, *, source='babi', task_filter=None, guided=False):
    out = []
    for split, cap in (('train', per_task), ('test', eval_per_task * 2)):
        by_task = defaultdict(list)
        for r in _rows(f'{BASE}/babi/{split}.jsonl'):
            if task_filter is not None and r['task'] not in task_filter:
                continue
            by_task[r['task']].append(r)
        for task, rows in sorted(by_task.items()):
            random.Random(seed).shuffle(rows)
            pool = [r['answer'] for r in rows[:500]]
            seen_stories = set()
            for r in rows:
                if r['story_id'] in seen_stories:
                    continue
                seen_stories.add(r['story_id'])
                facts = [f'{f["id"]}. {f["text"]}' for f in r['context']]
                unit = f'babi:{r["story_id"]}'
                out.append(make_case(
                    source=source, family=f'task{int(task):02d}', shape=f'babi-{r["task_name"]}', upstream='facebook/babi_qa',
                    upstream_id=f'{r["story_id"]}:{text_digest(r["question"])[:8]}', upstream_split='validation' if split == 'test' else 'train',
                    unit=unit, groups=[unit], root_name='answer_from_facts', args={'facts': 'string[]', 'question': 'string'},
                    returns='string', instruction=pick(BABI_INSTRUCTIONS, unit),
                    inputs={'facts': facts, 'question': r['question']}, expected=r['answer'], oracle='normalized',
                    level=min(3, len(r.get('supporting_fact_ids') or [1])), license_='BSD-3-Clause', domain='relational',
                    slice_='observation_followup', wrong=distinct_wrong(r['answer'], pool, 2),
                    aliases=[r['answer'].upper()],
                    revision=('tasks_1-20_v1-2+ordered-history-helper-v1' if guided else ''),
                    neuralese=BABI_HISTORY_NEURALESE if guided else None,
                    framing='neuralese-helper' if guided else None))
                if len(seen_stories) >= cap:
                    break
    return out


def babi_temporal3(per_task=70, eval_per_task=8, seed=20261031):
    """Guided bAbI three-supporting-facts successor; retain every ordered fact."""
    return babi(per_task, eval_per_task, seed, source='babi-temporal3', task_filter={3}, guided=True)


# ---------------------------------------------------------------- CLUTRR
CLUTRR_INSTRUCTIONS = [
    'story describes family relations between people named in brackets. Work out how target is related to anchor: '
    'return the single relation word that completes "target is anchor\'s ___" (for example son, aunt, '
    'grandfather, sister-in-law).',
    'From the family relations in story, find target\'s relation to anchor. Return one kinship word such that "target '
    'is the ___ of anchor" holds.',
]
CLUTRR_PATH_NEURALESE = {
    'helper': 'extract_relation_edge', 'args': {'sentence': 'string'}, 'returns': 'Neuralese<string>',
    'instruction': 'Extract only the direct family relation stated in sentence. Return a concise statement in the '
                   'form "PERSON_A is PERSON_B\'s RELATION". Preserve direction: "Beverly\'s daughter Paula" means '
                   'Paula is Beverly\'s daughter; "Paula\'s brother Samuel" means Samuel is Paula\'s brother. Do not '
                   'infer a relationship that the sentence does not state.',
    'root_note': 'Split story into its stated sentences and extract each direct edge with `const edge = await '
                 'extract_relation_edge(sentence);`. Await one call per sentence and retain every edge. Verify that each name and direct '
                 'relation appears in that sentence; then compose the connecting family path from target to anchor. '
                 'The requested string means "target is anchor\'s ___". Do not assume marriage or invent a parent edge. '
                 'Return only one relation word.',
}


def clutrr(cap_train=1000, cap_eval=150, seed=20261031, *, source='clutrr', min_hops=0, guided=False):
    out = []
    for split, cap in (('train', cap_train), ('test', cap_eval * 2)):
        rows = _rows(f'{BASE}/clutrr/gen_train234_test2to10__{split}.jsonl')
        random.Random(seed).shuffle(rows)
        labels = sorted({r['target_text'] for r in rows})
        rows = [r for r in rows if len(eval(r['edge_types'], {'__builtins__': {}})) >= min_hops]
        for r in rows[:cap]:
            anchor, target = eval(r['query'], {'__builtins__': {}})  # a literal ('A', 'B') tuple
            hops = len(eval(r['edge_types'], {'__builtins__': {}}))
            gold = r['target_text']
            unit = f'clutrr:{r["id"]}'
            out.append(make_case(
                source=source, family='kinship', shape=f'clutrr-{hops}hops', upstream='CLUTRR/v1', upstream_id=r['id'],
                upstream_split='validation' if split == 'test' else 'train', unit=unit, groups=[unit],
                root_name='find_relation', args={'story': 'string', 'anchor': 'string', 'target': 'string'},
                returns='string', instruction=pick(CLUTRR_INSTRUCTIONS, unit),
                inputs={'story': r['story'], 'anchor': anchor, 'target': target}, expected=gold, oracle='normalized',
                level=min(3, max(0, hops - 1)), license_='CC-BY-NC-4.0', domain='relational',
                wrong=[l for l in labels if l != gold][:2], aliases=[gold.capitalize()],
                revision=('gen_train234_test2to10+path-edge-helper-v1' if guided else ''),
                difficulty=f'{hops}hops', neuralese=CLUTRR_PATH_NEURALESE if guided else None,
                framing='neuralese-helper' if guided else None))
    return out


def clutrr_guided(cap_train=1000, cap_eval=150, seed=20261031, min_hops=4):
    """Guided multi-hop CLUTRR successor; retain original stories and related-world IDs."""
    return clutrr(cap_train, cap_eval, seed, source='clutrr-guided', min_hops=min_hops, guided=True)


# ---------------------------------------------------------------- LogiQA 2.0
LOGIQA_INSTRUCTIONS = [
    'Read passage and answer question by choosing among options. Return the letter of the option that follows '
    'logically.',
    'Decide which of options correctly answers question about passage, using strict logical reasoning. Return its '
    'letter.',
]


def logiqa(cap_train=1200, cap_eval=150, seed=20261031):
    out = []
    for split, cap in (('train', cap_train), ('dev', cap_eval), ('test', cap_eval)):
        rows = _rows(f'{BASE}/logiqa2/{split}.jsonl')
        random.Random(seed).shuffle(rows)
        for r in rows[:cap]:
            if len(r['options']) != 4:
                continue
            gold = LETTERS[r['answer']]
            unit = f'logiqa2:{split}:{r["id"]}'
            kinds = sorted(k for k, v in (r.get('type') or {}).items() if v)
            out.append(make_case(
                source='logiqa2', family='logic_mcq', shape=f'logiqa-{len(kinds)}kinds', upstream='LogiQA2.0',
                upstream_id=f'{split}:{r["id"]}', upstream_split='validation' if split != 'train' else 'train', unit=unit,
                groups=[unit], root_name='choose_answer', args={'passage': 'string', 'question': 'string',
                                                                 'options': 'Record<string, string>'},
                returns='"A" | "B" | "C" | "D"', instruction=pick(LOGIQA_INSTRUCTIONS, unit),
                inputs={'passage': r['text'], 'question': r['question'], 'options': dict(zip(LETTERS, r['options']))},
                expected=gold, oracle='exact', level=min(3, max(1, len(kinds))), license_='CC-BY-NC-SA-4.0',
                domain='logic', wrong=[l for l in LETTERS if l != gold][:2], aliases=[], revision=''))
    return out
