"""Composed Neuralese call-graph families (plans/neuralese/BROAD_RL_TASKS_20261031.md, "Composed Neuralese
structures"). The broad external source wraps almost every item in one root function with at most one
`read_X -> Neuralese<string>` helper. These families turn the same downloaded multi-document, long-article, table and
story sources into multi-function programs whose intended call graph moves Neuralese state in other ways:

- `neuralese_relay`: a producer's `Neuralese<string>` note is passed as an argument into another child (gather ->
  refine -> conclude, a hop-by-hop chain, evidence -> sharpen -> weigh), so state relays across two or more calls;
- `neuralese_typed`: helpers return structured `Neuralese<{...}>` results (per-document profiles, located answers,
  per-sentence relation lists), consumed by the parent or passed on as a typed array;
- `neuralese_iterate`: a reducer run by `iterateOn` carries `Neuralese` notes or a typed Neuralese state over the
  parts of a long input (article parts, story chunks, page sections) with a decreasing measure, then decides;
- `neuralese_fanin`: several distinct producer roles (table reader, text reader, row finder, page reader, planner)
  whose Neuralese outputs fan in to one decision;
- `multihop_deep`: call graphs of depth 3 with distinct subproblems (decompose -> per-step retrieve-and-read ->
  bridge -> confirm; lookup -> read entry -> compare; path tracing and relation composition).

Each case keeps the existing outcome oracle of its upstream item (span QA F1, exact letter, alias map, normalized,
numeric tolerance, answer list); rewards stay outcome-only and every fact the answer needs is in the inputs. The
reference replays a direct root answer (the collector replay has no Neuralese port); `structure.smoke` holds the
intended orchestration, executed per shape by `scripts/rl_tasks/structure_smoke.mjs` with the teacher-text transport.
Split units and groups are the ones the broad source uses for the same upstream items, so related worlds stay on
one split and custody relates them.
"""
from __future__ import annotations

import ast
import glob
import json
import math
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field

from ..common import SOURCES_DIR, distinct_wrong, make_case, nl_file, pick, resolve_str, text_digest
from .numeric import _plain_number, _sentence_keys, _tatqa_number, _tatqa_rights, html_table_rows

SOURCE = 'composed'
DOC = '{ title: string; text: string }'
LETTER = '"A" | "B" | "C" | "D"'
LETTERS = 'ABCD'
SPAN = {'level': 'span', 'normalization': 'qa', 'threshold': 0.8}
MULTIHIERTT_TOLERANCE = {'level': 'normalized', 'normalization': 'numeric-tolerance', 'tolerance': 1e-6,
                         'abs_tolerance': 0.006, 'percent_equivalent': True}
SHORT = 'a short phrase (a name, date, number, place, or yes or no), not a sentence'


@dataclass
class Program:
    """One program shape: root signature, root instruction variants and its callable folder."""
    name: str
    root: str
    args: dict
    returns: str
    instructions: list
    helpers: dict  # folder-relative path (no extension) -> (args, returns, instruction)
    depth: int
    producers: list
    relay: bool = False
    typed: bool = False
    iterate: bool = False
    smoke_code: str = ''
    smoke_children: dict = field(default_factory=dict)


def _files(program: Program) -> dict:
    return {f'{program.root}/{path}.nl': nl_file(*spec) for path, spec in program.helpers.items()}


def _case(program: Program, family: str, *, item: dict, inputs: dict, expected, oracle, level: int, returns=None,
          instruction_suffix: str = '', wrong=None, aliases=None, upstream: str, license_: str, license_note: str = '',
          revision: str = '', difficulty=None, smoke_children=None, slice_='nested_scoped', reference_value=None):
    instruction = pick(program.instructions, item['unit'] + '|' + program.name)
    if instruction_suffix:
        instruction = instruction.rstrip() + '\n' + instruction_suffix
    origin = item['source']
    case = make_case(
        source=SOURCE, family=family, shape=f'{program.name}-{item["shape"]}', upstream=upstream,
        upstream_id=item['id'], upstream_split=item['split'], unit=item['unit'],
        groups=sorted(set(item['groups'] + [f'ext-unit:{origin}:{item["unit"]}'])), root_name=program.root,
        args=program.args, returns=returns or program.returns, instruction=instruction, inputs=inputs,
        expected=expected, oracle=oracle, level=level, license_=license_, license_note=license_note,
        extra_files=_files(program), slice_=slice_, domain='relational', wrong=wrong or [], aliases=aliases or [],
        neuralese_room=f'{family}: {", ".join(program.producers)}', revision=revision, difficulty=difficulty,
        framing=family.replace('_', '-'),
        reference_root=([['return_result', {'status': 'success', 'value': reference_value}]]
                        if reference_value is not None else None))
    c = case['curriculum']
    c['named'] = 'required'
    if program.iterate:
        c['iterate'] = 'required'
    children = dict(program.smoke_children)
    children.update(smoke_children or {})
    final = reference_value if reference_value is not None else expected
    c['external']['origin_source'] = origin
    c['external']['structure'] = {
        'name': family, 'program': program.name, 'depth': program.depth, 'producers': program.producers,
        'relay': program.relay, 'typed_neuralese': program.typed, 'iterate': program.iterate,
        'functions': 1 + len(program.helpers),
        'smoke': {'root': [['eval', {'code': program.smoke_code}], ['return_result', {'status': 'success', 'value': final}]],
                  'children': children}}
    return case


def _take(items, n):
    return items[:n], items[n:]


def _spread(pool, programs, counts):
    """Disjoint slices of one source's items for several programs (different items, not the same item reframed)."""
    out, rest = [], list(pool)
    for program, n in zip(programs, counts):
        chosen, rest = _take(rest, n)
        out.append((program, chosen))
    return out


# ================================================================= source loaders (normalized items)

def _jsonl(path):
    with open(path) as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def musique_items(seed=20261103, limit=(1600, 160),
                  protected_groups=resolve_str('data_nvme', 'broad-rl-tasks-20261031', 'protected-musique-groups.txt')):
    """Answerable MuSiQue items, minus the protected seed groups (snapshot; assembly custody rechecks). Rows are
    kept as text until chosen, so the 470 MB train file is streamed."""
    blocked = set(open(protected_groups).read().split()) if os.path.exists(protected_groups) else set()
    out = {'train': [], 'validation': []}
    for (split, path), cap in zip((('train', 'train'), ('validation', 'validation')), limit):
        lines = []
        with open(f'{SOURCES_DIR}/musique/{path}.jsonl') as handle:
            for line in handle:
                head = json.loads(line[:200].split(', "paragraphs"')[0] + '}') if line.startswith('{"id"') else None
                if head is None:
                    lines.append(line)
                    continue
                rid = head['id']
                seeds = [p for p in rid.split('__', 1)[1].split('_') if p.isdigit()] if '__' in rid else []
                if blocked.intersection([f'musique:{rid}'] + [f'musique:seed:{x}' for x in seeds]):
                    continue
                lines.append(line)
        random.Random(seed).shuffle(lines)
        for line in lines:
            if len(out[split]) >= cap:
                break
            r = json.loads(line)
            if not r.get('answerable', True):
                continue
            hops = int(r['id'].split('hop')[0])
            seeds = [p for p in r['id'].split('__', 1)[1].split('_') if p.isdigit()] if '__' in r['id'] else []
            groups = [f'musique:{r["id"]}'] + [f'musique:seed:{x}' for x in seeds]
            if blocked.intersection(groups):
                continue
            steps = r.get('question_decomposition') or []
            out[split].append({
                'source': 'musique', 'id': r['id'], 'split': split, 'unit': r['id'], 'groups': groups,
                'question': r['question'], 'answer': r['answer'], 'aliases': list(r.get('answer_aliases') or []),
                'documents': [{'title': p['title'], 'text': p['paragraph_text']} for p in r['paragraphs']],
                'hops': hops, 'bridges': [x['answer'] for x in steps[:-1] if x.get('answer')],
                'shape': f'musique-{hops}hop'})
    return out


def twowiki_items(seed=20261103, scan=60000, limit=(2600, 200)):
    out = {'train': [], 'validation': []}
    for (split, path), cap in zip((('train', 'train'), ('validation', 'validation')), limit):
        rows = []
        with open(f'{SOURCES_DIR}/2wikimultihopqa/{path}.jsonl') as handle:
            for index, line in enumerate(handle):
                if index >= scan:
                    break
                rows.append(line)
        random.Random(seed).shuffle(rows)
        for line in rows:
            if len(out[split]) >= cap:
                break
            r = json.loads(line)
            if r.get('type') not in ('comparison', 'compositional', 'inference', 'bridge_comparison'):
                continue
            context = r['context']
            if isinstance(context, str):
                context = json.loads(context)
            if isinstance(context, dict):
                context = list(zip(context['title'], context.get('sentences') or context.get('content')))
            docs = [{'title': t, 'text': ' '.join(s) if isinstance(s, list) else s} for t, s in context]
            qid = r.get('_id') or r.get('id')
            evidences = r.get('evidences') or []
            if isinstance(evidences, str):
                evidences = json.loads(evidences)
            subjects = list(dict.fromkeys(e[0] for e in evidences if isinstance(e, (list, tuple)) and e))
            out[split].append({
                'source': '2wikimultihopqa', 'id': qid, 'split': split, 'unit': qid, 'groups': [f'2wiki-q:{qid}'],
                'question': r['question'], 'answer': str(r['answer']), 'aliases': [], 'documents': docs,
                'kind': r.get('type', 'qa'), 'subjects': subjects, 'shape': f'2wiki-{r.get("type", "qa")}'})
    return out


def quality_items(seed=20261103):
    out = {'train': [], 'validation': []}
    for split in ('train', 'validation'):
        rows = list({(text_digest(r['article']), r['question'].strip()): r
                     for r in _jsonl(f'{SOURCES_DIR}/quality/{split}.jsonl')}.values())
        random.Random(seed).shuffle(rows)
        for r in rows:
            unit = f'quality-article:{text_digest(r["article"])[:16]}'
            qid = f'{unit}:{text_digest(r["question"])[:10]}'
            out[split].append({
                'source': 'quality', 'id': qid, 'split': split, 'unit': unit, 'groups': [unit],
                'article': r['article'], 'question': r['question'].strip(),
                'options': {LETTERS[i]: o for i, o in enumerate(r['options'])}, 'answer': LETTERS[r['answer']],
                'hard': bool(r['hard']), 'shape': f'quality-{"hard" if r["hard"] else "easy"}'})
    return out


def squad_items(seed=20261103, per_case=(2, 6)):
    out = {'train': [], 'validation': []}
    for split in ('train', 'validation'):
        by_context, title_of = defaultdict(list), {}
        for r in _jsonl(f'{SOURCES_DIR}/squad_v1.1/{split}.jsonl'):
            by_context[r['context']].append(r)
            title_of[r['context']] = r['title']
        contexts = sorted(by_context, key=text_digest)
        random.Random(seed).shuffle(contexts)
        for context in contexts:
            n = per_case[0] + int(text_digest(context + '|n')[:4], 16) % (per_case[1] - per_case[0] + 1)
            qs = by_context[context][:n]
            if len(qs) < 2:
                continue
            unit = f'squad-ctx:{text_digest(context)[:16]}'
            out[split].append({
                'source': 'squad', 'id': ','.join(q['id'] for q in qs), 'split': split, 'unit': unit,
                'groups': [f'wiki:{title_of[context].lower()}', unit] + [f'squad-q:{q["id"]}' for q in qs],
                'article': context, 'questions': qs, 'shape': f'squad-{len(qs)}q'})
    return out


def clutrr_items(seed=20261103):
    out = {'train': [], 'validation': []}
    for split, path in (('train', 'train'), ('validation', 'test')):
        rows = list(_jsonl(f'{SOURCES_DIR}/clutrr/gen_train234_test2to10__{path}.jsonl'))
        random.Random(seed).shuffle(rows)
        for r in rows:
            anchor, target = ast.literal_eval(r['query'])
            hops = len(ast.literal_eval(r['edge_types']))
            unit = f'clutrr:{r["id"]}'
            out[split].append({
                'source': 'clutrr', 'id': r['id'], 'split': split, 'unit': unit, 'groups': [unit], 'story': r['story'],
                'anchor': anchor, 'target': target, 'answer': r['target_text'], 'hops': hops,
                'path': ast.literal_eval(r['edge_types']), 'shape': f'clutrr-{hops}hops'})
    return out


def babi_items(seed=20261103, tasks=(1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 14), min_facts=8):
    """One question per story: the one with the longest history (bAbI repeats a story with growing context)."""
    out = {'train': [], 'validation': []}
    for split, path in (('train', 'train'), ('validation', 'test')):
        best = {}
        for r in _jsonl(f'{SOURCES_DIR}/babi/{path}.jsonl'):
            if r['task'] not in tasks or len(r['context']) < min_facts:
                continue
            if r['story_id'] not in best or len(r['context']) > len(best[r['story_id']]['context']):
                best[r['story_id']] = r
        rows = sorted(best.values(), key=lambda r: r['story_id'])
        random.Random(seed).shuffle(rows)
        for r in rows:
            unit = f'babi:{r["story_id"]}'
            out[split].append({
                'source': 'babi', 'id': f'{r["story_id"]}:{text_digest(r["question"])[:8]}', 'split': split,
                'unit': unit, 'groups': [unit], 'facts': [f'{f["id"]}. {f["text"]}' for f in r['context']],
                'question': r['question'], 'answer': r['answer'], 'task': r['task'],
                'shape': f'babi-{r["task_name"]}'})
    return out


def trivia_items(seed=20261103, scan=12000, max_chars=60000, limit=(1200, 120)):
    """Streams the 3.2 GB train file: parses the first `scan` rows, keeps compact qualifying items up to the limit,
    then shuffles them."""
    out = {'train': [], 'validation': []}
    for (split, path), cap in zip((('train', 'rc.wikipedia__train.jsonl'), ('validation', 'rc.wikipedia__validation.jsonl')), limit):
        items = []
        with open(f'{SOURCES_DIR}/triviaqa_rc_wikipedia/{path}') as handle:
            for index, line in enumerate(handle):
                if index >= scan or len(items) >= cap:
                    break
                r = json.loads(line)
                pages = list(zip(r['entity_pages']['title'], r['entity_pages']['wiki_context']))
                text = ' '.join(c for _, c in pages)
                answer = r['answer']['value']
                aliases = list(dict.fromkeys(r['answer'].get('aliases') or []))[:30]
                if not pages or len(text) > max_chars or len(answer.split()) > 6:
                    continue
                if not any(a.lower() in text.lower() for a in [answer] + aliases):
                    continue
                items.append({
                    'source': 'triviaqa', 'id': r['question_id'], 'split': split,
                    'unit': f'triviaqa-q:{r["question_id"]}', 'groups': [f'triviaqa-q:{r["question_id"]}'],
                    'question': r['question'], 'pages': pages, 'answer': answer, 'aliases': aliases,
                    'shape': 'triviaqa'})
        random.Random(seed).shuffle(items)
        out[split] = items
    return out


def tatqa_items(seed=20261103):
    out = {'train': [], 'validation': []}
    for split, path in (('train', 'train'), ('validation', 'dev'), ('validation', 'test_gold')):
        contexts = list(_jsonl(f'{SOURCES_DIR}/tatqa/{path}.jsonl'))
        random.Random(seed).shuffle(contexts)
        for ctx in contexts:
            unit = f'tatqa:context:{ctx["table"]["uid"]}'
            paragraphs = [p['text'] for p in sorted(ctx['paragraphs'], key=lambda p: int(p['order']))]
            for q in ctx['questions'][:3]:
                out[split].append({
                    'source': 'tatqa', 'id': q['uid'], 'split': split, 'unit': unit, 'groups': [unit],
                    'question': q['question'], 'table': ctx['table']['table'], 'paragraphs': paragraphs,
                    'answer': q['answer'], 'kind': q['answer_type'], 'answer_from': q['answer_from'],
                    'derivation': q.get('derivation') or '', 'shape': f'tatqa-{q["answer_type"]}-{q["answer_from"]}'})
    return out


def multihiertt_items(seed=20261103, scan_factor=3, cap=1000):
    seen = set()
    for split in ('train', 'dev', 'test'):
        for r in _jsonl(f'{SOURCES_DIR}/finqa/{split}.jsonl'):
            seen |= _sentence_keys(r['pre_text'] + r['post_text'])
    for path in ('train', 'dev', 'test_gold'):
        for r in _jsonl(f'{SOURCES_DIR}/tatqa/{path}.jsonl'):
            seen |= _sentence_keys(p['text'] for p in r['paragraphs'])
    parse = lambda v: ast.literal_eval(v) if isinstance(v, str) else v
    out = {'train': [], 'validation': []}
    for split, path in (('train', 'train'), ('validation', 'dev')):
        rows, rng = [], random.Random(seed)
        with open(f'{SOURCES_DIR}/multihiertt/{path}.jsonl') as handle:
            for count, line in enumerate(handle):
                if len(rows) < cap * scan_factor:
                    rows.append(line)
                elif (j := rng.randrange(count + 1)) < cap * scan_factor:
                    rows[j] = line
        rng.shuffle(rows)
        for line in rows:
            r = json.loads(line)
            qa, paragraphs = parse(r['qa']), [p for p in parse(r['paragraphs']) if p.strip()]
            if _sentence_keys(paragraphs) & seen:
                continue
            tables = [html_table_rows(t) for t in parse(r['tables'])]
            if len(json.dumps(tables)) + len(json.dumps(paragraphs)) > 60000:
                continue
            unit = f'multihiertt:{r["uid"]}'
            ops = len([p for p in str(qa.get('program') or '').split('),') if p.strip()])
            out[split].append({
                'source': 'multihiertt', 'id': r['uid'], 'split': split, 'unit': unit, 'groups': [unit],
                'question': qa['question'], 'tables': tables, 'paragraphs': paragraphs, 'answer': qa['answer'],
                'kind': qa['question_type'], 'ops': ops, 'shape': f'multihiertt-{qa["question_type"]}'})
    return out


def hybrid_items(seed=20261103, max_page_chars=120000, limit=(700, 60)):
    root = glob.glob(f'{SOURCES_DIR}/hybridqa/WikiTables-WithLinks-*')[0]
    out = {'train': [], 'validation': []}
    title = lambda link: link.split('/wiki/', 1)[-1].replace('_', ' ')
    for (split, path), cap in zip((('train', 'train'), ('validation', 'dev')), limit):
        rows = list(_jsonl(f'{SOURCES_DIR}/hybridqa/{path}.jsonl'))
        random.Random(seed).shuffle(rows)
        for r in rows[:6000]:
            if len(out[split]) >= cap:
                break
            answer = r.get('answer-text') or ''
            if not answer or len(answer.split()) > 8:
                continue
            try:
                table = json.load(open(f'{root}/tables_tok/{r["table_id"]}.json'))
                passages = json.load(open(f'{root}/request_tok/{r["table_id"]}.json'))
            except OSError:
                continue
            cells = [[{'text': cell[0], 'pages': [title(link) for link in cell[1]]} for cell in row] for row in table['data']]
            pages = {title(link): text for link, text in passages.items()}
            if len(json.dumps(pages)) > max_page_chars or any('$' in t for t in pages):
                continue
            unit = f'hybridqa-table:{r["table_id"]}'
            out[split].append({
                'source': 'hybridqa', 'id': r['question_id'], 'split': split, 'unit': unit, 'groups': [unit],
                'question': r['question'], 'answer': answer,
                'table': {'title': table.get('title', ''), 'columns': [h[0] for h in table['header']], 'rows': cells},
                'pages': pages, 'shape': f'hybridqa-{min(len(cells) // 10, 5)}0rows'})
    return out


# ================================================================= shared answer helpers

def _span_checks(answer, pool, offset, extra_wrong=()):
    if answer.lower() in ('yes', 'no'):
        wrong = ['no' if answer.lower() == 'yes' else 'yes']
    else:
        wrong = [w for w in extra_wrong if w and w.lower() != answer.lower()][:2]
        wrong += distinct_wrong(answer, pool[offset % max(1, len(pool)):] + pool, 2)
    return wrong[:3]


def _span_oracle(answer, aliases):
    oracle = dict(SPAN)
    alts = [a for a in aliases if a and a != answer]
    if alts:
        oracle['alternates'] = alts
    return oracle, alts


def _parts(text, k):
    """Split a long text into k consecutive parts at paragraph boundaries, of roughly equal length."""
    paragraphs = [p for p in re.split(r'\n\s*\n', text) if p.strip()]
    if len(paragraphs) < k:
        paragraphs = [p for p in text.split('\n') if p.strip()]
    if len(paragraphs) < k:
        return None
    total, target, parts, current, size = sum(map(len, paragraphs)), None, [], [], 0
    target = total / k
    for p in paragraphs:
        current.append(p)
        size += len(p)
        if size >= target * (len(parts) + 1) and len(parts) < k - 1:
            parts.append('\n\n'.join(current))
            current = []
    if current:
        parts.append('\n\n'.join(current))
    return parts if len(parts) == k else None


# ================================================================= family 1: neuralese_relay

GATHER = ({'question': 'string', 'documents': DOC + '[]'}, 'Neuralese<string>',
          'Read every document in documents and write a private note of each fact that could bear on question: who '
          'or what each relevant document is about, the names, dates, places and numbers it gives, and any entity it '
          'mentions that another document describes. Leave out documents that have nothing to do with question.')
REFINE = ({'note': 'Neuralese<string>', 'question': 'string', 'documents': DOC + '[]'}, 'Neuralese<string>',
          'note is a first private note on documents for question. Check each fact in note against documents and '
          'correct it where it is wrong. Then follow every link note leaves open: when a fact names an entity that '
          'another document describes, read that document and add what it says. Write the refined note: the chain of '
          'facts from question to its answer, each with the title of the document it comes from.')
CONCLUDE = ({'note': 'Neuralese<string>', 'question': 'string'}, 'string',
            'note is a private note of the facts that answer question. Return the answer it supports as ' + SHORT + '.')

RELAY_GATHER = Program(
    name='gather-refine-conclude', root='answer_through_notes', args={'question': 'string', 'documents': DOC + '[]'},
    returns='string', depth=2, producers=['gather', 'refine'], relay=True,
    helpers={'gather': GATHER, 'refine': REFINE, 'conclude': CONCLUDE},
    instructions=[
        'Answer question from documents by passing a private note through three helpers:\n'
        '1. `const note = await gather(question, documents);` collects the facts that may matter.\n'
        '2. `const refined = await refine(note, question, documents);` checks those facts and follows the links '
        'between documents.\n'
        '3. `return await conclude(refined, question);` reads the refined note and gives the answer.\n'
        'Pass each note on as it is. The answer is ' + SHORT + '.',
        'documents are encyclopedia paragraphs; only a few of them matter for question, and the answer usually needs '
        'a fact in one document that leads to another document. Work through private notes: gather writes a first '
        'note from documents, refine takes that note together with documents and returns a corrected note in which '
        'each link is followed, and conclude turns a refined note into the answer. Call them in that order, each on '
        'the note the previous one returned, and return the answer conclude gives: ' + SHORT + '.',
        'Answer question from documents. First call gather(question, documents) for a private note of the relevant '
        'facts. Then call refine with that note, question and documents; refine checks the facts and follows links '
        'to other documents. When question chains through three or more entities (for example the spouse of the '
        'director of a film), call refine once more on its own result. Finally call conclude with the last note and '
        'question, and return its answer, ' + SHORT + '.',
    ],
    smoke_code='const note = await gather(question, documents);\nconst refined = await refine(note, question, documents);\n'
               'const answer = await conclude(refined, question);',
    smoke_children={'gather': 'first note', 'refine': 'refined note'})

RELAY_SELF = Program(
    name='gather-refine-answer', root='answer_from_refined_note', args={'question': 'string', 'documents': DOC + '[]'},
    returns='string', depth=2, producers=['gather', 'refine'], relay=True,
    helpers={'gather': GATHER, 'refine': REFINE},
    instructions=[
        '1. Call gather(question, documents); it returns a private note of the facts in documents that bear on '
        'question.\n'
        '2. Call refine(note, question, documents) with that note; it returns a checked note that follows each link '
        'between documents.\n'
        '3. Read the refined note and return the answer to question as ' + SHORT + '. When the note leaves the '
        'answer open, read the documents it points to yourself before answering.',
        'Use the two note helpers one after the other, then answer. gather makes a private note of everything in '
        'documents relevant to question; refine improves such a note against documents, fixing wrong facts and '
        'adding what linked documents say. Give refine the note from gather, then answer question from the refined '
        'note: return ' + SHORT + '.',
    ],
    smoke_code='const note = await gather(question, documents);\nconst refined = await refine(note, question, documents);',
    smoke_children={'gather': 'first note', 'refine': 'refined note'})

RELAY_HOPS = Program(
    name='hop-relay', root='follow_the_chain', args={'question': 'string', 'documents': DOC + '[]'}, returns='string',
    depth=2, producers=['first_link', 'next_link'], relay=True,
    helpers={
        'first_link': ({'question': 'string', 'documents': DOC + '[]'}, 'Neuralese<string>',
                       'question asks about something reached through a chain of facts (for example the birthplace of '
                       'the author of a book). Find in documents the first link of that chain: the entity question '
                       'starts from and the fact about it that leads onward. Write a private note of that fact, the '
                       'title of its document, and the entity it leads to.'),
        'next_link': ({'prior': 'Neuralese<string>', 'question': 'string', 'documents': DOC + '[]'},
                      'Neuralese<string>',
                      'prior is a private note of the chain of facts found so far for question. Starting from the '
                      'last entity in prior, find in documents the next fact question needs. Write a note that keeps '
                      'everything in prior, adds the new fact with its document title, and says whether the chain now '
                      'reaches the answer to question and what that answer is.'),
    },
    instructions=[
        'Answer question by following its chain of facts through documents with private notes:\n'
        '1. `let note = await first_link(question, documents);`\n'
        '2. Call `note = await next_link(note, question, documents);` once for each further link: a question with '
        'two linked facts needs one call, with three facts two calls, with four facts three calls.\n'
        '3. Read the final note and return the answer it reaches as ' + SHORT + '.',
        'question is a multi-hop question: each fact leads to an entity that the next fact is about. Start the chain '
        'with first_link(question, documents), which gives a private note of the first fact. Pass that note as prior '
        'to next_link(prior, question, documents), and keep passing each new note to the next call until the note '
        'says the chain reaches the answer. Return that answer as ' + SHORT + '.',
    ],
    smoke_code='let note = await first_link(question, documents);\nnote = await next_link(note, question, documents);',
    smoke_children={'first_link': 'first link', 'next_link': 'chain so far'})

EVIDENCE = ({'article': 'string', 'question': 'string'}, 'Neuralese<string>',
            'article is a long story or essay. Write a private note of every passage in article that bears on question: '
            'what happens, who says or does what, and the reasons the text gives, in the order they appear.')
WEIGH = ({'evidence': 'Neuralese<string>', 'question': 'string', 'options': 'Record<string, string>'}, LETTER,
         'evidence is a private note on how an article bears on question and its options. Return the letter of the '
         'option that evidence supports best.')
RELAY_QUALITY = Program(
    name='evidence-weigh', root='choose_supported_option',
    args={'article': 'string', 'question': 'string', 'options': 'Record<string, string>'}, returns=LETTER, depth=2,
    producers=['collect_evidence'], relay=True, helpers={'collect_evidence': EVIDENCE, 'weigh': WEIGH},
    instructions=[
        'Choose the option of options that article best supports as the answer to question:\n'
        '1. `const evidence = await collect_evidence(article, question);`\n'
        '2. `return await weigh(evidence, question, options);`\n'
        'weigh returns the letter of the chosen option; return that letter.',
        'article is long, so gather the evidence first and decide from it: collect_evidence writes a private note of '
        'the passages relevant to question, and weigh reads such a note and picks the option it supports. Give weigh '
        'the note from collect_evidence together with question and options, and return the letter it chooses.',
    ],
    smoke_code='const evidence = await collect_evidence(article, question);\nconst letter = await weigh(evidence, question, options);',
    smoke_children={'collect_evidence': 'evidence note'})
RELAY_QUALITY_SHARPEN = Program(
    name='evidence-sharpen-weigh', root='choose_after_comparing',
    args={'article': 'string', 'question': 'string', 'options': 'Record<string, string>'}, returns=LETTER, depth=2,
    producers=['collect_evidence', 'sharpen'], relay=True,
    helpers={'collect_evidence': EVIDENCE, 'weigh': WEIGH, 'sharpen': (
        {'evidence': 'Neuralese<string>', 'options': 'Record<string, string>', 'article': 'string'}, 'Neuralese<string>',
        'evidence is a private note of the passages of article that bear on a question. For each option in options, '
        'note which parts of evidence support it and which contradict it; look back at article where evidence is '
        'unclear about an option. Write the result as a new private note, option by option.')},
    instructions=[
        '1. Call collect_evidence(article, question) for a private note of the passages relevant to question.\n'
        '2. Call sharpen(evidence, options, article) with that note; it returns a note that sets each option against '
        'the evidence.\n'
        '3. Call weigh(sharpened, question, options) and return the letter it gives.',
        'The options are close to each other, so test them against the evidence before choosing. collect_evidence '
        'gives a private note of what article says about question; sharpen turns that note into one that weighs every '
        'option against it; weigh chooses the best-supported option from such a note. Run them in that order, each '
        'on the note before it, and return the letter weigh chooses.',
    ],
    smoke_code='const evidence = await collect_evidence(article, question);\n'
               'const sharpened = await sharpen(evidence, options, article);\nconst letter = await weigh(sharpened, question, options);',
    smoke_children={'collect_evidence': 'evidence note', 'sharpen': 'option-by-option note'})


def _doc_case(program, family, item, level, pool, offset, extra_wrong=(), smoke=None):
    answer = item['answer']
    oracle, alts = _span_oracle(answer, item.get('aliases') or [])
    return _case(program, family, item=item, inputs={'question': item['question'], 'documents': item['documents']},
                 expected=answer, oracle=oracle, level=level,
                 wrong=_span_checks(answer, pool, offset, extra_wrong),
                 aliases=[answer.upper(), 'The ' + answer + '.'] + alts[:2],
                 upstream='MuSiQue' if item['source'] == 'musique' else '2WikiMultihopQA',
                 license_='CC-BY-4.0' if item['source'] == 'musique' else 'Apache-2.0',
                 difficulty=item['shape'], smoke_children={**(smoke or {}), **(
                     {'conclude': answer} if 'conclude' in program.helpers else {})})


WIKI_LEVEL = {'comparison': 0, 'compositional': 1, 'inference': 1, 'bridge_comparison': 2}
MUSIQUE_LEVEL = {2: 1, 3: 2, 4: 3}


def _quality_level(item, base):
    return min(3, base + (1 if item['hard'] else 0))


RELAY_TRAIN = (250, 350, 450, 220, 220)
RELAY_HELD = 24
RELAY_MUSIQUE_ITEMS = RELAY_TRAIN[0] + RELAY_TRAIN[1]
RELAY_QUALITY_ITEMS = RELAY_TRAIN[3] + RELAY_TRAIN[4]


def relay(train=RELAY_TRAIN, held=RELAY_HELD, seed=20261103):
    family = 'neuralese_relay'
    out = []
    mus, wiki, qual = musique_items(seed), twowiki_items(seed), quality_items(seed)
    for split in ('train', 'validation'):
        n = train if split == 'train' else (held,) * 5
        mpool = [i['answer'] for i in mus[split][:3000]]
        wpool = [i['answer'] for i in wiki[split][:3000] if i['answer'].lower() not in ('yes', 'no')]
        m_items = mus[split]
        w_items = [i for i in wiki[split] if i['kind'] in WIKI_LEVEL]
        for k, item in enumerate(m_items[:n[0]]):
            out.append(_doc_case(RELAY_GATHER if k % 2 else RELAY_SELF, family, item, MUSIQUE_LEVEL[item['hops']],
                                 mpool, k, item['bridges']))
        for k, item in enumerate(m_items[n[0]:n[0] + n[1]]):
            out.append(_doc_case(RELAY_HOPS, family, item, MUSIQUE_LEVEL[item['hops']], mpool, k, item['bridges']))
        for k, item in enumerate(w_items[:n[2]]):
            wrong = [s for s in item['subjects'] if s != item['answer']] if item['kind'] == 'comparison' else []
            program = (RELAY_HOPS if item['kind'] == 'compositional' and k % 3 == 0 else
                       RELAY_GATHER if k % 2 else RELAY_SELF)
            out.append(_doc_case(program, family, item, WIKI_LEVEL[item['kind']], wpool, k, wrong))
        for k, item in enumerate(qual[split][:n[3] + n[4]]):
            program = RELAY_QUALITY if k < n[3] else RELAY_QUALITY_SHARPEN
            gold = item['answer']
            out.append(_case(program, family, item=item, inputs={'article': item['article'], 'question': item['question'],
                                                                 'options': item['options']},
                             expected=gold, oracle='exact', level=_quality_level(item, 2 if program is RELAY_QUALITY_SHARPEN else 1),
                             wrong=[l for l in LETTERS if l != gold][:2], upstream='emozilla/quality', license_='CC-BY-4.0',
                             revision='7142df0', difficulty=item['shape'], smoke_children={'weigh': gold}))
    return out


# ================================================================= family 2: neuralese_typed

PROFILE_T = '{ subject: string; relevant: boolean; facts: { attribute: string; value: string }[] }'
PROFILE = ({'question': 'string', 'document': DOC}, f'Neuralese<{PROFILE_T}>',
           'Describe document as a typed private profile for question. subject is who or what document is about; '
           'relevant says whether anything in document bears on question; facts lists each fact that could matter '
           '(birth and death dates, places, nationalities, founding or release dates, people related to subject) as '
           'an attribute with its value, copied as document states it.')
SMOKE_PROFILE = {'subject': 'Example', 'relevant': True, 'facts': [{'attribute': 'born', 'value': '1950'}]}
TYPED_PROFILES = Program(
    name='document-profiles', root='answer_from_profiles', args={'question': 'string', 'documents': DOC + '[]'},
    returns='string', depth=2, producers=['profile'], typed=True, helpers={'profile': PROFILE},
    instructions=[
        '1. For every document in documents, `await profile(question, document)` and keep the profiles in a list, one '
        'call per document.\n'
        '2. Leave out the profiles that are not relevant to question.\n'
        '3. Answer question from the relevant profiles: compare the attribute values it asks about (for example the '
        'earlier date or a shared country), or follow a profile\'s related person to that person\'s own profile.\n'
        '4. Return the answer as ' + SHORT + '.',
        'documents are encyclopedia paragraphs about different people, films, places or organisations. Have profile '
        'write a structured private profile of each document for question, then work from the profiles: question '
        'may ask which entity comes first or is older, whether two share a property, or who is linked to whom. '
        'Return only the answer, ' + SHORT + '.',
    ],
    smoke_code='const profiles = [];\nfor (const document of documents.slice(0, 2)) profiles.push(await profile(question, document));',
    smoke_children={'profile': SMOKE_PROFILE})
TYPED_JUDGE = Program(
    name='document-profiles-judge', root='judge_from_profiles', args={'question': 'string', 'documents': DOC + '[]'},
    returns='string', depth=2, producers=['profile'], typed=True, relay=True,
    helpers={'profile': PROFILE, 'judge': (
        {'question': 'string', 'profiles': f'Neuralese<{PROFILE_T}>[]'}, 'string',
        'profiles are typed private profiles of encyclopedia documents for question, one per document. Answer '
        'question from them: compare the facts it asks about, or follow a related person from one profile to that '
        'person\'s profile. Return ' + SHORT + '.')},
    instructions=[
        '1. Call profile(question, document) for every document in documents and collect the results in an array, '
        'in the order of documents.\n'
        '2. Call judge(question, profiles) with the whole array and return its answer.',
        'Answer question in two stages. First turn each document into a typed private profile with profile: who it '
        'is about, whether it matters for question, and its facts. Then hand the array of all profiles to judge, '
        'which compares them and answers. Return judge\'s answer, ' + SHORT + '.',
    ],
    smoke_code='const profiles = [];\nfor (const document of documents.slice(0, 2)) profiles.push(await profile(question, document));\n'
               'const answer = await judge(question, profiles);',
    smoke_children={'profile': SMOKE_PROFILE})

LOCATE_T = '{ answer: string; sentence: string }'
TYPED_SQUAD = Program(
    name='located-answers', root='answer_each_question',
    args={'article': 'string', 'questions': '{ id: string; question: string }[]'}, returns='Record<string, string>',
    depth=2, producers=['locate'], typed=True,
    helpers={'locate': ({'article': 'string', 'question': 'string'}, f'Neuralese<{LOCATE_T}>',
                         'Find where article answers question. answer is the shortest span of article that answers it, '
                         'copied exactly; sentence is the full sentence of article it comes from.')},
    instructions=[
        'For each entry of questions, call `await locate(article, entry.question)` and keep the typed private note '
        'under entry.id. Then build the result record: for each id, the answer span from its note, exactly as it is '
        'written in article. Return the record.',
        'Return a record that maps every question id in questions to its answer, a short span copied from article. '
        'Find each answer with locate(article, question), which gives a typed private note of the answering span and '
        'its sentence; one call per question. Take the span from each note.',
    ],
    smoke_code='const notes = {};\nfor (const entry of questions) notes[entry.id] = await locate(article, entry.question);',
    smoke_children={'locate': {'answer': 'span', 'sentence': 'The sentence.'}})

EDGE_T = '{ person: string; relation: string; relative: string }[]'
TYPED_CLUTRR = Program(
    name='sentence-relations', root='relation_between',
    args={'story': 'string', 'anchor': 'string', 'target': 'string'}, returns='string', depth=2,
    producers=['relations_in'], typed=True,
    helpers={'relations_in': ({'sentence': 'string'}, f'Neuralese<{EDGE_T}>',
                              'List every family relation that sentence states directly, one entry each: relative is '
                              'person\'s relation. For "[Ann]\'s son [Bo] ..." that is person Ann, relation son, '
                              'relative Bo. Use only the bracketed names and the relations sentence itself states.')},
    instructions=[
        '1. Split story into its sentences (an exact split in code).\n'
        '2. For each sentence, `await relations_in(sentence)` and keep every typed list.\n'
        '3. From those relations, find a path of family links from anchor to target and combine them into one '
        'relation word, so that target is anchor\'s ___ (for example son, aunt, grandfather, sister-in-law).\n'
        '4. Return that one word.',
        'story describes a family in short sentences, with names in brackets. Collect the stated relations of each '
        'sentence with relations_in, which returns a typed private list per sentence, then chain them from anchor to '
        'target. Return the single kinship word that completes "target is anchor\'s ___".',
    ],
    smoke_code='const sentences = story.split(/(?<=[.!?])\\s+/).filter(s => s.trim());\nconst lists = [];\n'
               'for (const sentence of sentences.slice(0, 2)) lists.push(await relations_in(sentence));',
    smoke_children={'relations_in': [{'person': 'A', 'relation': 'son', 'relative': 'B'}]})


def _clutrr_case(program, family, item, labels, smoke=None):
    gold = item['answer']
    return _case(program, family, item=item, inputs={'story': item['story'], 'anchor': item['anchor'],
                                                     'target': item['target']},
                 expected=gold, oracle='normalized', level=min(3, max(0, item['hops'] - 2)),
                 wrong=[l for l in labels if l != gold][:2], aliases=[gold.capitalize()], upstream='CLUTRR/v1',
                 license_='CC-BY-NC-4.0', difficulty=item['shape'], smoke_children=smoke)


def typed(train=(350, 350, 550, 450), held=24, seed=20261103):
    family = 'neuralese_typed'
    out = []
    wiki, squad, clutrr = twowiki_items(seed + 1), squad_items(seed), clutrr_items(seed)
    labels = sorted({i['answer'] for i in clutrr['train']})
    for split in ('train', 'validation'):
        n = train if split == 'train' else (held,) * 4
        wpool = [i['answer'] for i in wiki[split][:3000] if i['answer'].lower() not in ('yes', 'no')]
        w_items = [i for i in wiki[split] if i['kind'] in WIKI_LEVEL]
        for k, item in enumerate(w_items[:n[0] + n[1]]):
            program = TYPED_PROFILES if k < n[0] else TYPED_JUDGE
            wrong = [s for s in item['subjects'] if s != item['answer']] if item['kind'] == 'comparison' else []
            smoke = {'judge': item['answer']} if program is TYPED_JUDGE else None
            out.append(_doc_case(program, family, item, WIKI_LEVEL[item['kind']] + (k >= n[0] and item['kind'] != 'comparison'),
                                 wpool, k, wrong, smoke))
        pool = [q['answers']['text'][0] for i in squad[split][:3000] for q in i['questions']]
        for k, item in enumerate(squad[split][:n[2]]):
            qs = item['questions']
            gold = {f'q{j + 1}': sorted(set(q['answers']['text']), key=q['answers']['text'].index) for j, q in enumerate(qs)}
            first = {key: v[0] for key, v in gold.items()}
            keys = list(first)
            swapped = dict(first)
            swapped[keys[0]], swapped[keys[1]] = first[keys[1]], first[keys[0]]
            other = dict(first)
            other[keys[-1]] = distinct_wrong(first[keys[-1]], pool[(k * 7) % max(1, len(pool)):] + pool, 1)[0]
            alt = {key: (v[1] if len(v) > 1 else v[0]) for key, v in gold.items()}
            case = _case(TYPED_SQUAD, family, item=item,
                         inputs={'article': item['article'],
                                 'questions': [{'id': f'q{j + 1}', 'question': q['question']} for j, q in enumerate(qs)]},
                         expected=gold, oracle={'level': 'normalized', 'normalization': 'qa-alias-map', 'threshold': 0.8},
                         level={2: 0, 3: 1, 4: 2, 5: 2}.get(len(qs), 3), wrong=[w for w in (swapped, other) if w != first],
                         aliases=[first] + ([alt] if alt != first else []), upstream='rajpurkar/squad',
                         license_='CC-BY-SA-4.0', difficulty=item['shape'], reference_value=first)
            case['semantics']['expected'] = gold
            out.append(case)
        for item in clutrr[split][:n[3]]:
            out.append(_clutrr_case(TYPED_CLUTRR, family, item, labels))
    return out


# ================================================================= family 3: neuralese_iterate

def _iterate_steps(opener, stepper, coll, extra, state='notes'):
    """Shared numbered iterateOn steps (state field `state`, cursor `next`) for reducer instructions."""
    return (f'1. Start from `{{ {state}: await {opener}({coll}[0]{extra}), next: 1 }}`.\n'
            f'2. One step reads `{coll}[s.next]` with `await {stepper}(s.{state}, {coll}[s.next]{extra})` and returns '
            f'`{{ {state}: <that result>, next: s.next + 1 }}`.\n'
            f'3. Run all the steps in one eval: `const final = await iterateOn(step, start).withMeasure(s => '
            f'{coll}.length - s.next).until(s => s.next >= {coll}.length);`\n')


QOPTS = {'question': 'string', 'options': 'Record<string, string>'}
ITER_QUALITY = Program(
    name='article-parts-reducer', root='choose_after_reading',
    args={'parts': 'string[]', 'question': 'string', 'options': 'Record<string, string>'}, returns=LETTER, depth=2,
    producers=['open_notes', 'absorb'], relay=True, iterate=True,
    helpers={
        'open_notes': ({'part': 'string', **QOPTS}, 'Neuralese<string>',
                       'part is the first part of a long article. Write a private note of everything in it that bears '
                       'on question or on any of options.'),
        'absorb': ({'priorNotes': 'Neuralese<string>', 'part': 'string', **QOPTS}, 'Neuralese<string>',
                   'priorNotes are private notes on the earlier parts of an article for question and options. Read '
                   'part, the next part, and write updated notes: keep what still holds from priorNotes, add what part '
                   'says about question and each option, and correct anything part shows to be wrong.'),
        'settle': ({'notes': 'Neuralese<string>', **QOPTS}, LETTER,
                   'notes are private notes on a whole article for question. Return the letter of the option they '
                   'support best.'),
    },
    instructions=[
        'parts are the consecutive parts of one long article. Read them in order with iterateOn, carrying private '
        'notes from part to part:\n' + _iterate_steps('open_notes', 'absorb', 'parts', ', question, options') +
        '4. Return `await settle(final.notes, question, options)`, the letter of the best-supported option.',
        'parts split one long article into consecutive pieces. Build up private notes for question by iterating '
        'over the pieces with iterateOn: open_notes reads the first piece, and each step passes the notes so far and '
        'the next piece to absorb, which returns updated notes. The measure is the number of pieces not yet read; '
        'stop when none are left. Then let settle choose an option from the final notes and return its letter.',
    ],
    smoke_code='const start = { notes: await open_notes(parts[0], question, options), next: 1 };\n'
               'const final = await iterateOn(async s => ({ notes: await absorb(s.notes, parts[s.next], question, options), '
               'next: s.next + 1 }), start).withMeasure(s => parts.length - s.next).until(s => s.next >= parts.length);\n'
               'const letter = await settle(final.notes, question, options);',
    smoke_children={'open_notes': 'notes on part 1', 'absorb': 'notes so far'})
ITER_QUALITY_SELF = Program(
    name='article-parts-reducer-self', root='decide_after_reading',
    args={'parts': 'string[]', 'question': 'string', 'options': 'Record<string, string>'}, returns=LETTER, depth=2,
    producers=['open_notes', 'absorb'], relay=True, iterate=True,
    helpers={k: v for k, v in ITER_QUALITY.helpers.items() if k != 'settle'},
    instructions=[
        'Answer question about the article whose consecutive parts are parts. Carry private notes through the parts '
        'with iterateOn:\n' + _iterate_steps('open_notes', 'absorb', 'parts', ', question, options') +
        '4. Read the final notes and return the letter of the option in options they support best.',
        'The article is too long to judge at once, so reduce it part by part. open_notes gives private notes on '
        'parts[0]; absorb takes the notes so far and the next part and returns the updated notes. Let iterateOn run '
        'absorb over the remaining parts, with the count of unread parts as its measure. From the notes after the '
        'last part, choose the option of options that answers question and return its letter.',
    ],
    smoke_code=ITER_QUALITY.smoke_code.rsplit('\n', 1)[0],
    smoke_children={'open_notes': 'notes on part 1', 'absorb': 'notes so far'})

BABI_ANSWER = ('Return only the answer word, or words separated by commas when the answer has several (for '
               'example "apple,football"); yes or no for a yes/no question.')
WORLD = {'facts': 'string[]', 'question': 'string'}
ITER_BABI = Program(
    name='story-state-reducer', root='answer_after_story', args={'chunks': 'string[][]', 'question': 'string'},
    returns='string', depth=2, producers=['begin_state', 'advance'], relay=True, iterate=True,
    helpers={
        'begin_state': (WORLD, 'Neuralese<string>',
                        'facts are the first numbered facts of a story. Write a private state of the story\'s world '
                        'after them, for question: where each person and object is, who holds what, and anything else '
                        'question may need, with the fact numbers.'),
        'advance': ({'state': 'Neuralese<string>', **WORLD}, 'Neuralese<string>',
                    'state is a private record of a story\'s world after its earlier facts. Apply facts, the next '
                    'numbered facts in time order, and write the updated state. A later fact replaces an earlier one: '
                    'a person who moves is no longer in the old place, and an object dropped or given away is no longer '
                    'held. Keep the history question may ask about (earlier places, counts, the order of events).'),
        'answer_from_state': ({'state': 'Neuralese<string>', 'question': 'string'}, 'string',
                              'state is a private record of a story\'s world after all of its facts. Answer question '
                              'from it. ' + BABI_ANSWER),
    },
    instructions=[
        'chunks are the numbered facts of one story, in time order, in consecutive groups. Track the story\'s world '
        'with iterateOn, one chunk per step:\n' + _iterate_steps('begin_state', 'advance', 'chunks', ', question',
                                                                  'state') +
        '4. Return `await answer_from_state(final.state, question)`.',
        'Answer question about the story told by chunks (groups of numbered facts, in order). Keep a private world '
        'state: begin_state builds it from the first chunk, and advance updates a state with the next chunk. Drive '
        'advance over the remaining chunks with iterateOn, measuring the chunks still to apply, and pass the last '
        'state to answer_from_state. Return its answer.',
    ],
    smoke_code='const start = { state: await begin_state(chunks[0], question), next: 1 };\n'
               'const final = await iterateOn(async s => ({ state: await advance(s.state, chunks[s.next], question), '
               'next: s.next + 1 }), start).withMeasure(s => chunks.length - s.next).until(s => s.next >= chunks.length);\n'
               'const answer = await answer_from_state(final.state, question);',
    smoke_children={'begin_state': 'state 1', 'advance': 'state'})
ITER_BABI_SELF = Program(
    name='story-state-reducer-self', root='track_story', args={'chunks': 'string[][]', 'question': 'string'},
    returns='string', depth=2, producers=['begin_state', 'advance'], relay=True, iterate=True,
    helpers={k: v for k, v in ITER_BABI.helpers.items() if k != 'answer_from_state'},
    instructions=[
        'Each element of chunks is a group of numbered story facts; the groups are in time order. Follow the story '
        'with a private world state carried by iterateOn:\n' + _iterate_steps('begin_state', 'advance', 'chunks',
                                                                             ', question', 'state') +
        '4. Read the final state and answer question. ' + BABI_ANSWER,
        'Facts later in the story can undo earlier ones, so read it in order. begin_state turns chunks[0] into a '
        'private world state and advance applies the next chunk to a state. Run advance over the remaining chunks '
        'with iterateOn (measure: chunks left to apply) and answer question from the final state. ' + BABI_ANSWER,
    ],
    smoke_code=ITER_BABI.smoke_code.rsplit('\n', 1)[0],
    smoke_children={'begin_state': 'state 1', 'advance': 'state'})

CANDIDATES_T = '{ candidates: { answer: string; evidence: string }[] }'
SECTION = {'section': DOC, 'question': 'string'}
ITER_TRIVIA = Program(
    name='section-candidates-reducer', root='answer_from_sections',
    args={'sections': DOC + '[]', 'question': 'string'}, returns='string', depth=2,
    producers=['open_candidates', 'scan'], relay=True, typed=True, iterate=True,
    helpers={
        'open_candidates': (SECTION, f'Neuralese<{CANDIDATES_T}>',
                            'section is one section of an encyclopedia page (title names the page). List as typed '
                            'private candidates every phrase in it that could answer question, each with the words of '
                            'the section that support it. An empty list when nothing in it bears on question.'),
        'scan': ({'prior': f'Neuralese<{CANDIDATES_T}>', **SECTION}, f'Neuralese<{CANDIDATES_T}>',
                 'prior holds the typed private candidate answers to question found in earlier sections. Read section '
                 'and return the updated candidates: keep each prior candidate with its evidence, add new ones the '
                 'section supports, and drop a candidate the section shows to be wrong.'),
    },
    instructions=[
        'sections are the sections of encyclopedia pages about question\'s subject. Collect candidate answers '
        'section by section with iterateOn:\n' + _iterate_steps('open_candidates', 'scan', 'sections', ', question',
                                                                'candidates') +
        '4. Choose the candidate in the final state that the evidence supports best, and return it as the short '
        'answer (a name, place, date or number).',
        'Find the answer to question in sections by keeping typed private candidates: open_candidates lists the '
        'candidates of sections[0], and scan updates the candidates with one more section. Run scan over the other '
        'sections with iterateOn, measuring the sections still unread. Return the best-supported final candidate as '
        'a short answer (a name, place, date or number), not a sentence.',
    ],
    smoke_code='const start = { candidates: await open_candidates(sections[0], question), next: 1 };\n'
               'const final = await iterateOn(async s => ({ candidates: await scan(s.candidates, sections[s.next], '
               'question), next: s.next + 1 }), start).withMeasure(s => sections.length - s.next)'
               '.until(s => s.next >= sections.length);',
    smoke_children={'open_candidates': {'candidates': [{'answer': 'x', 'evidence': 'y'}]},
                    'scan': {'candidates': [{'answer': 'x', 'evidence': 'y'}]}})


def _sections(pages, size=5000):
    out = []
    for title, text in pages:
        paragraphs = [p for p in text.split('\n') if p.strip()]
        current = ''
        for p in paragraphs:
            if current and len(current) + len(p) > size:
                out.append({'title': title, 'text': current})
                current = ''
            current = (current + '\n' + p) if current else p
        if current:
            out.append({'title': title, 'text': current})
    return out


def iterate(train=(330, 330, 300, 300, 420), held=24, seed=20261103):
    family = 'neuralese_iterate'
    out = []
    qual, babi, trivia = quality_items(seed), babi_items(seed), trivia_items(seed)
    for split in ('train', 'validation'):
        n = train if split == 'train' else (held,) * 5
        # QuALITY: parts per article 2/3/5/8 by unit (related questions share one split of their article); items
        # after the ones the relay family uses.
        made = 0
        for item in qual[split][RELAY_QUALITY_ITEMS if split == 'train' else 2 * RELAY_HELD:]:
            if made >= n[0] + n[1]:
                break
            k = (2, 3, 5, 8)[int(text_digest(item['unit'] + '|parts')[:4], 16) % 4]
            parts = _parts(item['article'], k)
            if not parts:
                continue
            program = ITER_QUALITY if made < n[0] else ITER_QUALITY_SELF
            gold = item['answer']
            out.append(_case(program, family, item={**item, 'shape': f'{item["shape"]}-{k}parts'},
                             inputs={'parts': parts, 'question': item['question'], 'options': item['options']},
                             expected=gold, oracle='exact', level=min(3, {2: 0, 3: 1, 5: 2, 8: 3}[k] + item['hard']),
                             wrong=[l for l in LETTERS if l != gold][:2], upstream='emozilla/quality',
                             license_='CC-BY-4.0', revision='7142df0', difficulty=f'{k}parts', slice_='iterate',
                             smoke_children={'settle': gold}))
            made += 1
        # bAbI: four facts per chunk; level by chunk count.
        pools = defaultdict(list)
        for item in babi[split][:4000]:
            pools[item['task']].append(item['answer'])
        for k, item in enumerate(babi[split][:n[2] + n[3]]):
            chunks = [item['facts'][i:i + 4] for i in range(0, len(item['facts']), 4)]
            if len(chunks) < 2:
                continue
            program = ITER_BABI if k < n[2] else ITER_BABI_SELF
            level = 0 if len(chunks) <= 2 else 1 if len(chunks) == 3 else 2 if len(chunks) <= 5 else 3
            out.append(_case(program, family, item={**item, 'shape': f'{item["shape"]}-{min(len(chunks), 8)}chunks'},
                             inputs={'chunks': chunks, 'question': item['question']}, expected=item['answer'],
                             oracle={'level': 'normalized', 'normalization': 'qa'}, level=level,
                             wrong=distinct_wrong(item['answer'], pools[item['task']][k % 50:] + pools[item['task']], 2),
                             aliases=[item['answer'].upper()], upstream='facebook/babi_qa', license_='BSD-3-Clause',
                             difficulty=f'{len(chunks)}chunks', slice_='iterate',
                             smoke_children={'answer_from_state': item['answer']}))
        # TriviaQA: page sections, typed candidate state.
        tpool = [i['answer'] for i in trivia[split][:3000]]
        made = 0
        for k, item in enumerate(trivia[split]):
            if made >= n[4]:
                break
            sections = _sections(item['pages'])
            if not 2 <= len(sections) <= 14:
                continue
            answer = item['answer']
            oracle, alts = _span_oracle(answer, item['aliases'])
            level = 0 if len(sections) <= 3 else 1 if len(sections) <= 5 else 2 if len(sections) <= 8 else 3
            out.append(_case(ITER_TRIVIA, family, item={**item, 'shape': f'triviaqa-{min(len(sections), 10)}sections'},
                             inputs={'sections': sections, 'question': item['question']}, expected=answer,
                             oracle=oracle, level=level, wrong=distinct_wrong(answer, tpool[k % 500:] + tpool, 2),
                             aliases=alts[:2], upstream='mandarjoshi/trivia_qa', license_='Apache-2.0',
                             license_note='evidence pages are Wikipedia text (CC-BY-SA)',
                             difficulty=f'{len(sections)}sections', slice_='iterate'))
            made += 1
    return out


# ================================================================= family 4: neuralese_fanin

READ_TABLE = ({'table': 'string[][]', 'question': 'string'}, 'Neuralese<string>',
              'Find in table every row, column and value that question needs, and write a private note of them with '
              'their row and column labels and any unit or scale the table shows (for example "in millions" or %).')
READ_TEXT = ({'paragraphs': 'string[]', 'question': 'string'}, 'Neuralese<string>',
             'Find in paragraphs every statement that question needs: values, names, definitions, and any statement of '
             'units or scale. Write a private note of them, quoting the numbers exactly as written.')


def _tat_contract(returns):
    return {'number': 'Return a number in the report\'s own units (12.5 for 12.5%, 840.4 for $840.4 million when the '
                      'amounts are in millions).',
            'string': 'Return the exact words from the report that answer it.',
            'string[]': 'Return the list of the exact spans from the report, one per item asked for.',
            'number | string': 'Return a number, or for a question that asks which item or year, its text as written '
                               'in the report.'}[returns]


FANIN_TATQA = Program(
    name='table-and-text-readers', root='answer_from_report',
    args={'question': 'string', 'table': 'string[][]', 'paragraphs': 'string[]'}, returns='number', depth=2,
    producers=['read_table', 'read_text'], helpers={'read_table': READ_TABLE, 'read_text': READ_TEXT},
    instructions=[
        '1. Call read_table(table, question) and read_text(paragraphs, question); each returns a private note on one '
        'part of the annual-report excerpt.\n'
        '2. From both notes, find the values question needs. When it asks for a calculation (a change, a ratio, an '
        'average, a percentage), compute it exactly in eval from those values.\n'
        '3. Answer question.',
        'The report excerpt has two kinds of evidence, table rows and text paragraphs, and each has its own reader: '
        'read_table for table and read_text for paragraphs. Read both, combine the two private notes, and answer '
        'question; do any arithmetic exactly in eval.',
    ],
    smoke_code='const tableNote = await read_table(table, question);\nconst textNote = await read_text(paragraphs, question);',
    smoke_children={'read_table': 'table note', 'read_text': 'text note'})
FANIN_FILING_PLAN = Program(
    name='tables-text-planner', root='answer_from_filing',
    args={'question': 'string', 'tables': 'string[][][]', 'paragraphs': 'string[]'}, returns='number', depth=2,
    producers=['read_table', 'read_text', 'plan_calculation'], relay=True,
    helpers={'read_table': READ_TABLE, 'read_text': READ_TEXT, 'plan_calculation': (
        {'question': 'string', 'tableNotes': 'Neuralese<string>[]', 'textNote': 'Neuralese<string>'}, 'Neuralese<string>',
        'tableNotes are private notes on the tables of a financial report and textNote a note on its paragraphs, all '
        'for question. Write a private plan of the answer: which values to use (with the note and row each comes '
        'from) and the arithmetic steps in order.')},
    instructions=[
        '1. For each table in tables, call read_table(table, question); keep the notes in the order of tables.\n'
        '2. Call read_text(paragraphs, question).\n'
        '3. Call plan_calculation(question, tableNotes, textNote) for a private plan of the values and steps.\n'
        '4. Carry out the plan exactly in eval and answer question.',
        'The figures question needs can sit in several tables and in the text. Give each table to read_table and the '
        'paragraphs to read_text, then hand all of those notes to plan_calculation, which returns a private plan of '
        'the values and the arithmetic. Compute the plan exactly in eval and answer question.',
    ],
    smoke_code='const tableNotes = [];\nfor (const table of tables.slice(0, 2)) tableNotes.push(await read_table(table, question));\n'
               'const textNote = await read_text(paragraphs, question);\nconst plan = await plan_calculation(question, tableNotes, textNote);',
    smoke_children={'read_table': 'table note', 'read_text': 'text note', 'plan_calculation': 'plan'})
FANIN_FILING = Program(
    name='tables-and-text-readers', root='combine_report_notes',
    args={'question': 'string', 'tables': 'string[][][]', 'paragraphs': 'string[]'}, returns='number', depth=2,
    producers=['read_table', 'read_text'], helpers={'read_table': READ_TABLE, 'read_text': READ_TEXT},
    instructions=[
        'Read every table in tables with read_table(table, question) and the paragraphs with read_text(paragraphs, '
        'question). Combine the private notes to find the values question needs, compute exactly in eval, and '
        'answer question.',
        '1. One read_table call per table in tables, in order.\n2. One read_text call on paragraphs.\n3. From the '
        'notes, take the values question needs and compute the answer exactly in eval.\n4. Answer question.',
    ],
    smoke_code='const tableNotes = [];\nfor (const table of tables.slice(0, 2)) tableNotes.push(await read_table(table, question));\n'
               'const textNote = await read_text(paragraphs, question);',
    smoke_children={'read_table': 'table note', 'read_text': 'text note'})
HYBRID_TABLE = '{ title: string; columns: string[]; rows: { text: string; pages: string[] }[][] }'
FANIN_HYBRID = Program(
    name='rows-and-pages-readers', root='answer_from_table_and_links',
    args={'question': 'string', 'table': HYBRID_TABLE, 'pages': 'Record<string, string>'}, returns='string', depth=2,
    producers=['find_rows', 'read_page'],
    helpers={
        'find_rows': ({'table': HYBRID_TABLE, 'question': 'string'}, 'Neuralese<string>',
                      'Find the rows of table that question is about and write a private note of them: their cell '
                      'texts under each column, and the page titles each of those cells links to.'),
        'read_page': ({'question': 'string', 'title': 'string', 'page': 'string'}, 'Neuralese<string>',
                      'page is the encyclopedia page title. Write a private note of every fact in it that bears on '
                      'question, or that it has none.'),
    },
    instructions=[
        '1. `const rowNote = await find_rows(table, question);`\n'
        '2. From that note, pick the linked page titles that may hold the rest of the answer; for each one, '
        '`await read_page(question, title, pages[title])`.\n'
        '3. Combine the row note with the page notes and return the answer as ' + SHORT + '.',
        'question needs both the table and the pages its cells link to. find_rows writes a private note of the rows '
        'question is about and their links; read_page writes a note on one linked page. Use find_rows first, then '
        'read_page on the linked pages that matter, and answer from all the notes: ' + SHORT + '.',
    ],
    smoke_code='const rowNote = await find_rows(table, question);\nconst title = Object.keys(pages)[0];\n'
               'const pageNote = await read_page(question, title, pages[title]);',
    smoke_children={'find_rows': 'row note', 'read_page': 'page note'})


def _with_returns(program, returns):
    return program if program.returns == returns else Program(**{**program.__dict__, 'returns': returns})


def fanin(train=(500, 220, 220, 450), held=24, seed=20261103):
    family = 'neuralese_fanin'
    out = []
    dataset_license, dataset_revision, license_note = _tatqa_rights()
    tat, mh, hy = tatqa_items(seed), multihiertt_items(seed), hybrid_items(seed)
    for split in ('train', 'validation'):
        n = train if split == 'train' else (held * 2, held, held, held)
        made = 0
        for item in tat[split]:
            if made >= n[0]:
                break
            answer, kind = item['answer'], item['kind']
            if isinstance(answer, str) and answer.startswith('['):
                answer = ast.literal_eval(answer)
            if kind in ('arithmetic', 'count') or (kind == 'span' and isinstance(answer, list) and len(answer) == 1
                                                   and _tatqa_number(answer[0]) is not None):
                value = _tatqa_number(answer[0] if isinstance(answer, list) else answer)
                if value is None:
                    continue
                value = round(value, 4) if isinstance(value, float) else value
                returns, expected = 'number', value
                oracle = {'level': 'normalized', 'normalization': 'numeric-tolerance', 'tolerance': 1e-6,
                          'abs_tolerance': 0.006}
                wrong, aliases = [value * 1.1 + 1, -value - 1], [float(value)]
            elif kind == 'span' and isinstance(answer, list) and len(answer) == 1 and len(answer[0].split()) <= 8:
                returns, expected = 'string', answer[0]
                oracle = dict(SPAN)
                others = [p for p in item['paragraphs'] if p and answer[0] not in p][:1]
                wrong, aliases = ([others[0][:80]] if others else []) + ['none'], [answer[0].lower()]
            elif kind == 'multi-span' and isinstance(answer, list) and len(answer) > 1:
                returns, expected = 'string[]', answer
                oracle = {'level': 'normalized', 'normalization': 'answer-list'}
                wrong, aliases = [answer[:-1], answer + ['Total']], [list(reversed(answer))]
            else:
                continue
            ops = len(re.findall(r'[-+*/]', item['derivation']))
            level = (0 if kind in ('span', 'multi-span') and item['answer_from'] != 'table-text' else
                     1 if kind in ('span', 'multi-span', 'count') else 2 if ops <= 2 else 3)
            out.append(_case(_with_returns(FANIN_TATQA, returns), family, item=item,
                             inputs={'question': item['question'], 'table': item['table'], 'paragraphs': item['paragraphs']},
                             expected=expected, oracle=oracle, level=level, instruction_suffix=_tat_contract(returns),
                             wrong=wrong, aliases=aliases, upstream='NExTplusplus/TAT-QA', license_=dataset_license,
                             license_note=license_note, revision=dataset_revision, difficulty=item['shape']))
            made += 1
        made = 0
        for item in mh[split]:
            if made >= n[1] + n[2]:
                break
            answer = item['answer']
            if item['kind'] == 'arithmetic':
                value = round(float(answer), 6)
                value = int(value) if value.is_integer() else value
                # The gold is the upstream program's exact execution (5 decimals), so the FinQA 5e-3 relative band is
                # unnecessary and accepted the +7 negative control on large values (180 of 440 cases; 69 still at 1e-4
                # relative); a 0.006 absolute band (1e-6 relative) keeps 2-decimal rounding and percent forms.
                returns, expected, oracle = 'number', value, MULTIHIERTT_TOLERANCE
                wrong, aliases = [value * 1.5 + 1, -value - 1], [float(value)]
            else:
                text = str(answer).strip()
                number = _plain_number(text)
                if number is not None:
                    returns, expected = 'number | string', number
                    oracle = {'level': 'normalized', 'normalization': 'numeric-tolerance', 'tolerance': 1e-6}
                    wrong, aliases = [number + 1], [text]
                elif 0 < len(text.split()) <= 8:
                    returns, expected, oracle = 'string', text, dict(SPAN)
                    wrong, aliases = ['none', item['paragraphs'][0][:60]], [text.lower()]
                else:
                    continue
            program = FANIN_FILING_PLAN if made < n[1] else FANIN_FILING
            level = 1 if item['kind'] != 'arithmetic' else min(3, max(1, item['ops']))
            out.append(_case(_with_returns(program, returns), family, item={**item, 'shape': f'{item["shape"]}-{min(len(item["tables"]), 4)}tables'},
                             inputs={'question': item['question'], 'tables': item['tables'], 'paragraphs': item['paragraphs']},
                             expected=expected, oracle=oracle, level=level, instruction_suffix=_tat_contract(returns),
                             wrong=wrong, aliases=aliases, upstream='psunlpgroup/MultiHiertt', license_='MIT',
                             difficulty=f'{item["ops"]}ops'))
            made += 1
        hpool = [i['answer'] for i in hy[split][:3000]]
        for k, item in enumerate(hy[split][:n[3]]):
            rows = len(item['table']['rows'])
            level = 0 if rows <= 8 else 1 if rows <= 15 else 2 if rows <= 30 else 3
            out.append(_case(FANIN_HYBRID, family, item=item,
                             inputs={'question': item['question'], 'table': item['table'], 'pages': item['pages']},
                             expected=item['answer'], oracle=dict(SPAN), level=level,
                             wrong=distinct_wrong(item['answer'], hpool[k:] + hpool[:k], 2),
                             aliases=[item['answer'].upper()], upstream='wenhuchen/HybridQA', license_='CC-BY-4.0',
                             revision='f4ed68e', difficulty=f'{rows}rows'))
    return out


# ================================================================= family 5: multihop_deep

STEP_T = '{ step: string; answer: string }[]'
SPLIT_Q = ({'question': 'string'}, 'string[]',
           'Break question into the single-step questions that answer it, in order. A later step that needs an earlier '
           'answer refers to it as #1, #2 and so on: for "Where was the author of Dune born?" the steps are "Who is '
           'the author of Dune?" and "Where was #1 born?".')
SOLVE_STEP = ({'step': 'string', 'documents': DOC + '[]'}, 'string',
              'step is one single-step question. Pick the documents that may answer it (by title or because they '
              'mention its subject), read each of them with read_passage, and return the answer as a short phrase '
              'taken from the documents.')
READ_PASSAGE = ({'step': 'string', 'document': DOC}, 'Neuralese<string>',
                'Write a private note of what document says that answers step, or that it says nothing about it.')
DEEP_STEPS = Program(
    name='decompose-solve-read', root='answer_multihop', args={'question': 'string', 'documents': DOC + '[]'},
    returns='string', depth=3, producers=['read_passage'],
    helpers={'split_question': SPLIT_Q, 'solve_step': SOLVE_STEP, 'solve_step/read_passage': READ_PASSAGE},
    instructions=[
        '1. `const steps = await split_question(question);`\n'
        '2. Go through steps in order. In each step, replace every #k with the answer of step k (an exact text '
        'replacement in code), then `await solve_step(step, documents)` and keep its answer.\n'
        '3. Return the answer of the last step as ' + SHORT + '.',
        'question chains several facts. Let split_question break it into single-step questions, where #1, #2 stand '
        'for earlier answers. Solve the steps one at a time with solve_step, which reads the relevant documents, and '
        'put each answer into the later steps that refer to it before solving them. The answer to the last step '
        'answers question; return it as ' + SHORT + '.',
    ],
    smoke_code='const steps = await split_question(question);\nconst answers = [];\n'
               'for (const raw of steps) { const step = raw.replace(/#(\\d+)/g, (_, k) => answers[Number(k) - 1]); '
               'answers.push(await solve_step(step, documents)); }',
    smoke_children={'split_question': ['Who wrote it?', 'Where was #1 born?']})
DEEP_STEPS_CONFIRM = Program(
    name='decompose-solve-read-confirm', root='answer_and_confirm', args={'question': 'string', 'documents': DOC + '[]'},
    returns='string', depth=3, producers=['read_passage'],
    helpers={**DEEP_STEPS.helpers, 'confirm': (
        {'question': 'string', 'chain': STEP_T, 'documents': DOC + '[]'}, 'string',
        'chain lists single-step questions with the answers found for them, in order; together they should answer '
        'question. Check that the chain really answers question and that each answer is what documents say. Return '
        'the answer to question as ' + SHORT + ', correcting the chain\'s last answer if documents show it is wrong.')},
    instructions=[
        '1. `const steps = await split_question(question);`\n'
        '2. Solve the steps in order with `await solve_step(step, documents)`, first replacing each #k in a step with '
        'the answer of step k (exact text replacement in code). Keep every step with its answer as `{ step, answer }`.\n'
        '3. `return await confirm(question, chain, documents);` with that list as chain.',
        'Solve question in three parts: split_question gives the single-step questions (#k stands for the answer of '
        'step k), solve_step answers one step from documents, and confirm checks the whole chain of steps and '
        'answers against question and documents. Substitute earlier answers into later steps before solving them, '
        'then return the answer confirm gives: ' + SHORT + '.',
    ],
    smoke_code=DEEP_STEPS.smoke_code + '\nconst chain = steps.map((step, i) => ({ step, answer: answers[i] }));\n'
               'const final = await confirm(question, chain, documents);',
    smoke_children={'split_question': ['Who wrote it?', 'Where was #1 born?']})
LOOKUP = ({'entity': 'string', 'attribute': 'string', 'documents': DOC + '[]'}, 'string',
          'Find in documents the value of attribute for entity (for example the director of a film, or the date of '
          'birth of a person). Pick the documents about entity or that mention it, read each with read_entry, and '
          'return the value as written in the documents, or "unknown" when none of them gives it.')
READ_ENTRY = ({'entity': 'string', 'attribute': 'string', 'document': DOC}, 'Neuralese<string>',
              'Write a private note of what document says about attribute of entity, or that it says nothing about it.')
COMPARE = ({'question': 'string', 'findings': '{ entity: string; attribute: string; value: string }[]'}, 'string',
           'findings are facts looked up for question. Answer question from them: compare dates, numbers or places, '
           'decide whether two values are the same, or name the entity a chain of findings ends at. Return ' + SHORT + '.')
DEEP_LOOKUP = Program(
    name='lookup-read-compare', root='answer_by_lookups', args={'question': 'string', 'documents': DOC + '[]'},
    returns='string', depth=3, producers=['read_entry'],
    helpers={'lookup': LOOKUP, 'lookup/read_entry': READ_ENTRY, 'compare': COMPARE},
    instructions=[
        '1. Work out which facts question depends on, as pairs of an entity and an attribute: for "Which film came '
        'out first, X or Y?" they are (X, release date) and (Y, release date); for "Who is the mother of the '
        'director of X?" first (X, director), then (that director, mother).\n'
        '2. Get each fact with `await lookup(entity, attribute, documents)`; when a pair\'s entity is the value of an '
        'earlier pair, look that one up first.\n'
        '3. `return await compare(question, findings);` with every pair and its value as `{ entity, attribute, value }`.',
        'Answer question through lookups. Name the facts it needs as entity and attribute pairs (a chain like "the '
        'birthplace of the director of X" needs the director first). lookup finds one such fact in documents, '
        'reading the documents about the entity. Then give compare every looked-up fact; it compares them or '
        'follows the chain and answers. Return compare\'s answer: ' + SHORT + '.',
    ],
    smoke_code='const findings = [];\nconst director = await lookup(documents[0].title, "director", documents);\n'
               'findings.push({ entity: documents[0].title, attribute: "director", value: director });\n'
               'const answer = await compare(question, findings);',
    smoke_children={'lookup': {'$turns': [['eval', {'code': 'const note = await read_entry(entity, attribute, documents[0]);'}],
                                          ['return_result', {'status': 'success', 'value': 'someone'}]]},
                    'read_entry': 'entry note'})
DEEP_KIN = Program(
    name='trace-and-compose-kinship', root='kinship_by_path',
    args={'story': 'string', 'anchor': 'string', 'target': 'string'}, returns='string', depth=3,
    producers=['links_of'],
    helpers={
        'trace_path': ({'story': 'string', 'anchor': 'string', 'target': 'string'}, 'string[]',
                       'Find the chain of people that links anchor to target through the family relations story '
                       'states, using links_of to read the relations of each person on the way. Return the relation '
                       'words along the chain in order: the first is the relation of the second person to anchor, the '
                       'next the relation of the third person to the second, and so on up to target.'),
        'trace_path/links_of': ({'person': 'string', 'story': 'string'}, 'Neuralese<string>',
                                'Write a private note of every family relation story states directly between person '
                                'and someone else, with its direction (who is whose son, sister, husband and so on).'),
        'combine': ({'relations': 'string[]'}, 'string',
                    'relations is a chain of kinship words: each is the relation of the next person in a chain to the '
                    'person before. Fold the chain from the left with combine_pair into one kinship word, the '
                    'relation of the last person to the first, and return it.'),
        'combine/combine_pair': ({'first': 'string', 'second': 'string'}, 'string',
                                 'B is A\'s first and C is B\'s second. Return C\'s relation to A as one kinship word '
                                 '(for example son and then brother give son; mother and then brother give uncle).'),
    },
    instructions=[
        '1. `const relations = await trace_path(story, anchor, target);` gives the relation words along the chain '
        'from anchor to target.\n'
        '2. `return await combine(relations);` gives the one kinship word for target\'s relation to anchor.\n'
        'The answer completes "target is anchor\'s ___" (for example son, aunt, grandfather, sister-in-law).',
        'Two subproblems: find the family path, then compose it. trace_path follows the stated relations of story '
        'from anchor to target and returns the relation words along that path; combine folds such a list into a '
        'single kinship word. Return that word, which completes "target is anchor\'s ___".',
    ],
    smoke_code='const relations = await trace_path(story, anchor, target);\nconst word = await combine(relations);',
    smoke_children={'trace_path': {'$turns': [['eval', {'code': 'const note = await links_of(anchor, story);'}],
                                              ['return_result', {'status': 'success', 'value': ['son', 'brother']}]]},
                    'links_of': 'links note',
                    'combine': {'$turns': [['eval', {'code': 'const w = await combine_pair(relations[0], relations[1]);'}],
                                           ['return_result', {'status': 'success', 'value': 'son'}]]},
                    'combine_pair': 'son'})


def deep(train=(800, 650, 450), held=24, seed=20261103):
    family = 'multihop_deep'
    out = []
    mus, wiki, clutrr = musique_items(seed), twowiki_items(seed + 4), clutrr_items(seed + 5)
    labels = sorted({i['answer'] for i in clutrr['train']})
    for split in ('train', 'validation'):
        n = train if split == 'train' else (held * 2, held, held)
        mpool = [i['answer'] for i in mus[split][:3000]]
        skip = RELAY_MUSIQUE_ITEMS if split == 'train' else 2 * RELAY_HELD
        for k, item in enumerate(mus[split][skip:skip + n[0]]):
            program = DEEP_STEPS_CONFIRM if k % 2 else DEEP_STEPS
            out.append(_doc_case(program, family, item, MUSIQUE_LEVEL[item['hops']], mpool, k, item['bridges'],
                                 {'solve_step': {'$turns': [['eval', {'code': 'const note = await read_passage(step, documents[0]);'}],
                                                            ['return_result', {'status': 'success', 'value': item['answer']}]]},
                                  'read_passage': 'passage note', 'confirm': item['answer']}))
        wpool = [i['answer'] for i in wiki[split][:3000] if i['answer'].lower() not in ('yes', 'no')]
        w_items = [i for i in wiki[split] if i['kind'] in WIKI_LEVEL]
        for k, item in enumerate(w_items[:n[1]]):
            wrong = [s for s in item['subjects'] if s != item['answer']] if item['kind'] == 'comparison' else []
            level = {'comparison': 0, 'compositional': 1, 'inference': 2, 'bridge_comparison': 3}[item['kind']]
            out.append(_doc_case(DEEP_LOOKUP, family, item, level, wpool, k, wrong, {'compare': item['answer']}))
        for item in clutrr[split][:n[2]]:
            out.append(_clutrr_case(DEEP_KIN, family, item, labels))
    return out


# ================================================================= further root instruction variants
# Each program keeps one callable folder; these variants change how the root task is put (goal-first, reader request,
# numbered plan, check-before-answer, explicit stopping rule), not names or numbers, so a family has 20+ distinct
# normalized instruction texts (scripts/rl_tasks/measure_repetition.py). Picked per split unit, so related items share
# one wording.
EXTRA_INSTRUCTIONS = {
    'gather-refine-conclude': [
        'A reader wants the answer to question, and documents hold it only in pieces. Do not read all of documents in '
        'this call. Hand them to gather, which returns a private note; give that note, unchanged, to refine together '
        'with question and documents so it can verify each fact and chase links; give refine\'s note to conclude. The '
        'answer conclude returns is your result: ' + SHORT + '.',
        'Plan: collect, then correct, then conclude.\n- collect: gather(question, documents) -> a private note.\n'
        '- correct: refine(that note, question, documents) -> a note whose facts are checked against documents.\n'
        '- conclude: conclude(the corrected note, question) -> the answer.\nReturn the answer exactly as conclude gives '
        'it; it should be ' + SHORT + '.',
    ],
    'gather-refine-answer': [
        'Your job is the final judgment; the reading is delegated. gather turns documents into a private note for '
        'question, and refine checks a note against documents and follows its links to other documents. Get the '
        'refined note this way, then decide the answer yourself and return it as ' + SHORT + '. If the refined note '
        'names two candidates, prefer the one the documents tie to every part of question.',
        'question needs facts from more than one document. First obtain a note with gather(question, documents), then '
        'improve it with refine(note, question, documents); keep the improved note, not the first one. Answer from the '
        'improved note and check the answer against the document it cites. Return ' + SHORT + '.',
    ],
    'hop-relay': [
        'Count the links question chains through (the founder of the company that made X is two links). Call '
        'first_link(question, documents) for the first link, then next_link(prior, question, documents) with the newest '
        'note as prior, once per remaining link. Every note carries the whole chain so far. Return the answer the last '
        'note reaches, as ' + SHORT + '.',
        'Walk the chain of question one entity at a time. first_link gives a private note naming the starting entity '
        'and where it leads; next_link extends a note by one more fact found in documents. Stop extending as soon as a '
        'note states the answer to question, and return that answer: ' + SHORT + '.',
        'Answer question, which asks about an entity reached only through other entities. Do the first step with '
        'first_link and each later step with next_link, always passing the latest note as prior, never an earlier '
        'one. When a note says the chain is complete, read the answer from it and return ' + SHORT + '.',
    ],
    'evidence-weigh': [
        'The options of question differ in details buried in article. Have collect_evidence write a private note of '
        'the relevant passages, then pass that note to weigh with question and options and return the letter weigh '
        'chooses. Do not choose an option from the question wording alone.',
        '1. Gather evidence: collect_evidence(article, question) returns a private note.\n2. Decide: weigh(evidence, '
        'question, options) returns one letter.\nReturn that letter.',
    ],
    'evidence-sharpen-weigh': [
        'Three stages, each building on the last private note: collect_evidence finds what article says about '
        'question; sharpen sets every option against that evidence; weigh picks the letter of the best-supported '
        'option. Return the letter from weigh.',
        'Choose an answer for question by elimination. Get the evidence note from collect_evidence, let sharpen say '
        'for each option what supports and what contradicts it, and give the sharpened note to weigh with question and '
        'options. Return the letter weigh returns.',
    ],
    'document-profiles': [
        'Profile each document with profile(question, document); a profile is a typed private record of the document\'s '
        'subject, whether it matters for question, and its facts. Use the relevant profiles to answer: find the '
        'entities question names, then the facts it asks about (dates, places, nationalities, related people). Return '
        + SHORT + '.',
        'question compares or links entities described across documents. Build a typed profile of every document '
        'first, one profile call each, then answer from the profiles alone, reading a document again only when a '
        'profile leaves a needed fact out. Return ' + SHORT + '.',
        'Plan: 1. profiles = one profile(question, document) per document. 2. Keep the relevant ones. 3. Answer '
        'question by comparing or chaining their facts. 4. Return ' + SHORT + '.',
        'Work from structured notes, not raw text: profile writes a private profile (subject, relevant, facts) per '
        'document. Once every document has one, decide what question asks: an earlier or later date, a shared or '
        'different property, or the entity at the end of a relation. Return just that answer, ' + SHORT + '.',
    ],
    'document-profiles-judge': [
        'Separate reading from judging: profile reads one document into a typed private profile, judge reads an array '
        'of profiles and answers question. Profile every document, pass the array to judge, and return what judge '
        'returns (' + SHORT + ').',
        '1. For each document, `await profile(question, document)` and push the result.\n2. Give question and the '
        'array of profiles to judge.\n3. Return judge\'s answer unchanged.',
        'Build the evidence set as typed profiles (one profile call per document, in order) and let judge decide '
        'question from that set. Do not drop documents before judge sees them; judge uses each profile\'s relevant '
        'field itself. Return judge\'s answer, ' + SHORT + '.',
    ],
    'located-answers': [
        'questions are several reading questions about article. Answer each one separately: locate(article, question) '
        'gives a typed private note with the answering span and its sentence. Return a record from each question id to '
        'its span, copied exactly from article.',
        '1. Start with an empty record.\n2. For each entry of questions, `const found = await locate(article, '
        'entry.question)`.\n3. Set the record\'s entry.id to the answer span of found.\n4. Return the record.',
        'Return one short answer per question id. Each answer must be the exact words of article, so find it with '
        'locate rather than paraphrasing; check that the sentence locate reports really answers that question before '
        'you use its span.',
        'A reader asks the questions listed in questions about article. Locate the answer to each with locate (a typed '
        'private note: the span and its sentence) and answer with the spans, as a record keyed by question id.',
    ],
    'sentence-relations': [
        'Read story one sentence at a time with relations_in, which returns the family relations a sentence states as '
        'a typed private list. Join all the lists into a family graph, find how target is connected to anchor, and '
        'return the one kinship word for target\'s relation to anchor.',
        'Plan: (a) split story into sentences in code; (b) relations_in on each sentence; (c) chain the stated '
        'relations from anchor to target; (d) return the kinship word completing "target is anchor\'s ___".',
        'Every sentence of story states one or two family relations between bracketed names. Extract them with '
        'relations_in (one call per sentence), then work out target\'s relation to anchor by composing the relations '
        'along the path between them. Return a single word such as son, niece or father-in-law.',
        'question: how is target related to anchor? The answer is not stated directly; it follows from several '
        'relations in story. Collect each sentence\'s relations with relations_in, combine the ones on the path from '
        'anchor to target, and return the resulting kinship word.',
    ],
    'article-parts-reducer': [
        'Reduce the article in parts to private notes and then decide. open_notes reads parts[0]; one iterateOn step '
        'gives absorb the notes so far and the next part; the measure is parts.length minus the index of the next part '
        'and iteration stops when every part is absorbed. Return settle(final notes, question, options).',
        'Do not try to read parts at once. Keep a state { notes, next }: start it with open_notes on the first part and '
        'next = 1, step it with absorb on parts[next], and let iterateOn run the steps until next reaches parts.length, '
        'with the remaining count as its measure. Return the letter settle gives for the final notes.',
        'question asks about a long article split into parts. Iterate over the parts with iterateOn, carrying private '
        'notes built by open_notes and updated by absorb; each step moves to the next part. When all parts are read, '
        'call settle with the notes, question and options and return its letter.',
    ],
    'article-parts-reducer-self': [
        'Iterate over parts with iterateOn, keeping private notes: open_notes on parts[0], absorb for each later part, '
        'measure = parts not yet read. After the last part, compare options against the notes yourself and return the '
        'letter of the option they support.',
        'Read the article part by part, never all at once. The state is the private notes plus the index of the next '
        'part; absorb advances it by one part. Run the steps with iterateOn until no part is left, then answer question '
        'from the final notes with one letter.',
        'Build private notes over parts with open_notes and absorb, driven by iterateOn with a decreasing measure '
        '(unread parts). Then decide: the right option is the one the notes support and no later part contradicts. '
        'Return its letter.',
    ],
    'story-state-reducer': [
        'Simulate the story. Its facts arrive in chunks; begin_state builds a private world state from the first '
        'chunk, and each iterateOn step applies the next chunk with advance. The measure is the number of chunks not '
        'yet applied; stop when it is zero. Ask answer_from_state the question with the final state and return its '
        'answer.',
        'State tracking over chunks: state = begin_state(chunks[0], question); then iterateOn with a step that sets '
        'state = advance(state, chunks[next], question) and next = next + 1, until next equals chunks.length. Finally '
        'return answer_from_state(state, question).',
        'question depends on what happened in the story up to its end, and facts can change earlier facts. Carry a '
        'private world state through the chunks in order with iterateOn (begin_state, then advance per chunk), then '
        'let answer_from_state answer question from the last state. Return that answer.',
    ],
    'story-state-reducer-self': [
        'Carry a private world state through chunks with iterateOn: begin_state for chunks[0], advance for every later '
        'chunk, measure = chunks left. Answer question from the final state yourself. ' + BABI_ANSWER,
        'Track the story\'s world chunk by chunk. Start the state with begin_state, update it with advance one chunk '
        'per iterateOn step, and stop after the last chunk. Then read the state and answer question. ' + BABI_ANSWER,
        'The order of facts matters. Apply chunks to a private world state in order, using begin_state and advance '
        'inside iterateOn with the unapplied chunks as the measure. From the final state, answer question. ' +
        BABI_ANSWER,
    ],
    'section-candidates-reducer': [
        'Search sections for the answer to question with a typed private candidate list. open_candidates starts it '
        'from the first section; each iterateOn step passes the list and the next section to scan; the measure is the '
        'number of sections left. Return the final candidate with the strongest evidence, as a short answer.',
        'Read the sections in order, keeping candidates: state = { candidates, next }, started with open_candidates on '
        'sections[0], stepped with scan on sections[next], run by iterateOn until next reaches sections.length. Answer '
        'with the best-supported candidate (a name, place, date or number).',
        'question is a trivia question whose answer is somewhere in sections. Keep typed private candidates with their '
        'evidence, updating them section by section with scan inside iterateOn (open_candidates for the first '
        'section). After the last section, return the candidate the evidence supports best, as a short phrase.',
    ],
    'table-and-text-readers': [
        'Two readers, one answer. read_table gives a private note on the table, read_text a private note on the '
        'paragraphs; call both for question. Take the figures or words question needs from the two notes, compute '
        'anything arithmetic exactly in eval, and answer question.',
        'Plan:\n- tableNote = read_table(table, question)\n- textNote = read_text(paragraphs, question)\n- combine the '
        'notes; do arithmetic in eval\n- answer question.',
        'The answer may sit in the table, in the text, or need both (a figure from the table and a definition or '
        'scale from the text). Get both private notes, with read_table and read_text, before answering; use eval for '
        'any calculation.',
        'An analyst asks question about this report excerpt. Read the table with read_table and the paragraphs with '
        'read_text, check that the values you use carry the right year and unit, calculate exactly in eval where '
        'needed, and answer question.',
    ],
    'tables-text-planner': [
        'Fan the report in, then plan, then compute: read_table on every table, read_text on the paragraphs, '
        'plan_calculation on all of those notes. Execute the plan exactly in eval and answer question.',
        'Several hierarchical tables and the text may each hold part of what question needs. Collect one private note '
        'per table with read_table and one for the text with read_text, ask plan_calculation for a plan of values and '
        'steps from those notes, and carry it out in eval.',
        '1. tableNotes = read_table(table, question) for each table, in order.\n2. textNote = read_text(paragraphs, '
        'question).\n3. plan = plan_calculation(question, tableNotes, textNote).\n4. Compute the plan in eval; check '
        'each value against its table before answering question.',
    ],
    'tables-and-text-readers': [
        'Every table gets its own read_table call and the paragraphs one read_text call; each returns a private note. '
        'Pick the values question needs from those notes, compute exactly in eval, and answer question.',
        'question needs figures that may be spread over tables and text. Read them with the two readers (read_table '
        'per table, read_text for the paragraphs), combine the notes, and compute the answer in eval.',
        'Collect notes first, compute second: one read_table per table, one read_text, then an exact eval '
        'computation from the values in the notes. Answer question.',
    ],
    'rows-and-pages-readers': [
        'Start from the table: find_rows writes a private note of the rows question concerns and the pages their '
        'cells link to. Read only those linked pages with read_page, then answer from the notes: ' + SHORT + '.',
        'Plan:\n- rowNote = find_rows(table, question)\n- for each relevant linked title: read_page(question, title, '
        'pages[title])\n- answer question from the notes, as ' + SHORT + '.',
        'question joins a table lookup with facts from linked encyclopedia pages. Use find_rows for the lookup and '
        'read_page for each page that the found rows link to and that may hold the rest. Return ' + SHORT + '.',
        'Locate, follow, answer. find_rows locates the rows; their cells name linked pages; read_page reads each such '
        'page for question. Combine the notes and return the answer, ' + SHORT + '.',
    ],
    'decompose-solve-read': [
        'Plan: split_question breaks question into single steps; solve_step answers one step from documents by '
        'reading candidate documents with its own reader. Solve the steps in order, filling each #k with the answer '
        'of step k before solving it, and return the last answer as ' + SHORT + '.',
        'question is a chain. Ask split_question for its steps, then answer them one by one with solve_step(step, '
        'documents), replacing the #k placeholders in code with earlier answers. The answer of the final step is the '
        'answer to question: return ' + SHORT + '.',
        'Do not answer question in one go. Decompose it with split_question, resolve each step with solve_step '
        '(substituting answers for #1, #2, ... first), and keep a list of the answers. Return the final one, ' + SHORT + '.',
        'Three subproblems: decomposition (split_question), retrieval and reading for one step (solve_step), and '
        'bridging (putting step k\'s answer where a later step says #k, exactly, in code). Run them over all steps '
        'and return the last step\'s answer as ' + SHORT + '.',
    ],
    'decompose-solve-read-confirm': [
        'Decompose question with split_question, answer each step with solve_step after substituting earlier answers '
        'for #k, then have confirm check the whole chain against question and documents. Return confirm\'s answer as '
        + SHORT + '.',
        'Plan:\n- steps = split_question(question)\n- for each step: replace #k, then solve_step(step, documents)\n- '
        'chain = the steps with their answers\n- return confirm(question, chain, documents).',
        'Answer question by solving its single steps in order (split_question, then solve_step per step with earlier '
        'answers filled in) and finish with confirm, which may correct the last answer if documents disagree. Return '
        'what confirm returns: ' + SHORT + '.',
        'A chain of facts answers question. Find the chain with split_question and solve_step, record each step with '
        'its answer as { step, answer }, and have confirm verify it. Return confirm\'s answer, ' + SHORT + '.',
    ],
    'lookup-read-compare': [
        'Identify the facts question needs as (entity, attribute) pairs, look each up with lookup(entity, attribute, '
        'documents), and give compare the list of { entity, attribute, value }. When one fact depends on another (the '
        'birthplace of the director of X), look up the inner fact first. Return compare\'s answer: ' + SHORT + '.',
        'Plan:\n1. list the (entity, attribute) facts question depends on;\n2. value = lookup(entity, attribute, '
        'documents) for each, in dependency order;\n3. return compare(question, findings).',
        'Answer question by targeted lookups instead of reading everything: lookup reads the documents about one '
        'entity and returns one attribute value; compare answers question from all the looked-up values. Return '
        + SHORT + '.',
        'question compares entities or follows a relation between them. Turn it into lookups (an entity and the '
        'attribute wanted), perform each with lookup, and let compare draw the conclusion from the findings. Return '
        'its answer, ' + SHORT + '.',
    ],
    'trace-and-compose-kinship': [
        'First the path, then the word. trace_path returns the kinship words along the chain of people from anchor to '
        'target; combine folds them into one word. Return that word, the relation that completes "target is '
        'anchor\'s ___".',
        'Plan:\n- relations = trace_path(story, anchor, target)\n- word = combine(relations)\n- return word '
        '(for example uncle, granddaughter, mother-in-law).',
        'The relation of target to anchor is not stated; it follows from a path of stated relations. Let trace_path '
        'find that path in story as relation words, let combine reduce the words to one, and return it.',
        'Solve in two parts: path finding (trace_path, which reads the people on the way with its own helper) and '
        'relation composition (combine, which folds two steps at a time). Return the single kinship word for target '
        'relative to anchor.',
    ],
}

for _program in (RELAY_GATHER, RELAY_SELF, RELAY_HOPS, RELAY_QUALITY, RELAY_QUALITY_SHARPEN, TYPED_PROFILES,
                 TYPED_JUDGE, TYPED_SQUAD, TYPED_CLUTRR, ITER_QUALITY, ITER_QUALITY_SELF, ITER_BABI, ITER_BABI_SELF,
                 ITER_TRIVIA, FANIN_TATQA, FANIN_FILING_PLAN, FANIN_FILING, FANIN_HYBRID, DEEP_STEPS, DEEP_STEPS_CONFIRM,
                 DEEP_LOOKUP, DEEP_KIN):
    _program.instructions.extend(EXTRA_INSTRUCTIONS[_program.name])
