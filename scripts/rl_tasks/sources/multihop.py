"""Multi-document QA (HotpotQA distractor, MuSiQue, 2WikiMultihopQA): a question over a set of paragraphs, most
of them distractors. Natlang framing: the paragraphs are the function's `documents` input; per-document reading can
be handed to child calls that return evidence to the parent (Neuralese room). Oracle: token F1 >= 0.8 against the
gold answer or any upstream alias (SQuAD normalization; numbers must match exactly; yes/no exact)."""
from __future__ import annotations

import glob
import json
import random

from ..common import ARCHIVE_RAW, SOURCES_DIR, distinct_wrong, make_case, pick, resolve_str

ARGS = {'question': 'string', 'documents': '{ title: string; text: string }[]'}
INSTRUCTIONS = [
    'Answer question using documents. Only some documents are relevant, and the answer may need facts from two or '
    'more of them combined. Return the answer as a short phrase (a name, date, number, or yes/no), not a sentence.',
    'documents is a set of encyclopedia paragraphs, most of them distractors. Find the facts that answer question; '
    'it may take linking one document to another. Return just the short answer: an entity, number, date, or yes/no.',
    'Read documents to answer question. Return only the answer span as it would appear in the documents, or yes or '
    'no for a yes/no question.',
]
NEURALESE = {
    'helper': 'read_document', 'args': {'question': 'string', 'document': '{ title: string; text: string }'},
    'returns': 'Neuralese<string>',
    'instruction': 'Read document and write a private note of every fact in it that bears on question: names, dates, '
                   'numbers and relations, including facts that only help link to another document. Note that the '
                   'document is irrelevant when nothing in it bears on question.',
    'root_note': 'read_document gives a private note of what one document says about question. Collect the notes for '
                 'the documents, then decide the answer from them; read a document again yourself when a note leaves '
                 'the answer open.',
}
ORACLE = {'level': 'span', 'normalization': 'qa', 'threshold': 0.8}


def _case(source, upstream, revision, license_, item_id, split, question, documents, answer, aliases, level, shape,
          groups, pool, difficulty=None):
    instruction = pick(INSTRUCTIONS, item_id)
    oracle = dict(ORACLE)
    alts = [a for a in aliases if a and a != answer]
    if alts:
        oracle['alternates'] = alts
    if answer.lower() in ('yes', 'no'):
        wrong = ['no' if answer.lower() == 'yes' else 'yes']
    else:
        wrong = distinct_wrong(answer, pool, 2)
    return make_case(source=source, family='multidoc_qa', shape=shape, upstream=upstream, upstream_id=item_id,
                     upstream_split=split, unit=item_id, groups=groups, root_name='answer_from_documents', args=ARGS,
                     returns='string', instruction=instruction, inputs={'question': question, 'documents': documents},
                     expected=answer, oracle=oracle, level=level, license_=license_, domain='relational',
                     wrong=wrong, aliases=[answer.upper(), 'The ' + answer + '.'] + alts[:2],
                     neuralese_room='per-document reading delegated to children that return evidence to the parent',
                     revision=revision, difficulty=difficulty, neuralese=NEURALESE)


def hotpot(cap_train=1500, cap_eval=150, seed=20261031):
    import pyarrow.parquet as pq
    root = glob.glob(f'{ARCHIVE_RAW}/hotpotqa-distractor-*')[0]
    revision = root.rsplit('-', 1)[1]
    out = []
    for split, pattern, cap in (('train', 'train-*.parquet', cap_train), ('validation', 'validation-*.parquet', cap_eval * 2)):
        rows = []
        for path in sorted(glob.glob(f'{root}/{pattern}')):
            rows += pq.read_table(path).to_pylist()
        random.Random(seed).shuffle(rows)
        pool = [r['answer'] for r in rows[:2000] if r['answer'].lower() not in ('yes', 'no')]
        random.Random(seed + 1).shuffle(pool)
        for r in rows[:cap]:
            docs = [{'title': t, 'text': ''.join(s)} for t, s in zip(r['context']['title'], r['context']['sentences'])]
            level = {'easy': 1, 'medium': 1, 'hard': 2}.get(r['level'], 2)
            out.append(_case('hotpotqa', 'hotpotqa/hotpot_qa', revision, 'CC-BY-SA-4.0', r['id'], split, r['question'],
                             docs, r['answer'], [], level, f'hotpot-{r["type"]}-{len(docs)}docs',
                             [f'hotpotqa-q:{r["id"]}'], pool, difficulty=r['level']))
    return out


def musique(base=f'{SOURCES_DIR}/musique', cap_train=1500, cap_eval=150, seed=20261031,
            protected_groups=resolve_str('data_nvme', 'broad-rl-tasks-20261031', 'protected-musique-groups.txt')):
    """Most MuSiQue train questions share single-hop seed questions with protected held MuSiQue worlds; a snapshot
    of those protected `musique:` groups (written from ProtectedWorlds.load) is skipped here so the cap fills with
    usable items. Assembly still runs the full custody check."""
    import os
    blocked = set(open(protected_groups).read().split()) if os.path.exists(protected_groups) else set()
    out = []
    files = {'train': glob.glob(f'{base}/*train*.jsonl'), 'validation': glob.glob(f'{base}/*dev*.jsonl') +
             glob.glob(f'{base}/*validation*.jsonl')}
    for split, paths in files.items():
        rows = [json.loads(line) for path in paths for line in open(path) if line.strip()]
        rows = [r for r in rows if r.get('answerable', True)]
        random.Random(seed).shuffle(rows)
        pool = [r['answer'] for r in rows[:2000]]
        random.Random(seed + 1).shuffle(pool)
        taken = 0
        for r in rows:
            if taken >= (cap_train if split == 'train' else cap_eval * 2):
                break
            docs = [{'title': p['title'], 'text': p['paragraph_text']} for p in r['paragraphs']]
            hops = r['id'].split('hop')[0]
            seeds = [part for part in r['id'].split('__', 1)[1].split('_') if part.isdigit()] if '__' in r['id'] else []
            groups = [f'musique:{r["id"]}'] + [f'musique:seed:{s}' for s in seeds]
            if blocked.intersection(groups):
                continue
            level = {'2': 1, '3': 2, '4': 3}.get(hops, 2)
            out.append(_case('musique', 'MuSiQue', '', 'CC-BY-4.0', r['id'], split, r['question'], docs, r['answer'],
                             r.get('answer_aliases') or [], level, f'musique-{hops}hop-{len(docs)}docs', groups, pool,
                             difficulty=f'{hops}hop'))
            taken += 1
    return out


def twowiki(base=f'{SOURCES_DIR}/2wikimultihopqa', cap_train=1500, cap_eval=150, seed=20261031):
    out = []
    for split, pattern in (('train', 'train'), ('validation', 'dev')):
        paths = glob.glob(f'{base}/*{pattern}*.jsonl') + (glob.glob(f'{base}/*validation*.jsonl') if pattern == 'dev' else [])
        rows = [json.loads(line) for path in paths for line in open(path) if line.strip()]
        random.Random(seed).shuffle(rows)
        pool = [r['answer'] for r in rows[:2000] if str(r['answer']).lower() not in ('yes', 'no')]
        random.Random(seed + 1).shuffle(pool)
        for r in rows[:cap_train if split == 'train' else cap_eval * 2]:
            context = r['context']
            if isinstance(context, str):
                context = json.loads(context)
            if isinstance(context, dict):
                context = list(zip(context['title'], context.get('sentences') or context.get('content')))
            docs = [{'title': t, 'text': ' '.join(s) if isinstance(s, list) else s} for t, s in context]
            qid = r.get('_id') or r.get('id')
            kind = r.get('type', 'qa')
            level = {'comparison': 1, 'inference': 2, 'compositional': 2, 'bridge_comparison': 3}.get(kind, 2)
            out.append(_case('2wikimultihopqa', '2WikiMultihopQA', '', 'Apache-2.0', qid, split, r['question'], docs,
                             str(r['answer']), [], level, f'2wiki-{kind}-{len(docs)}docs', [f'2wiki-q:{qid}'], pool,
                             difficulty=kind))
    return out


HYBRID_INSTRUCTIONS = [
    'Answer question using table and the encyclopedia pages its cells link to (pages, keyed by title; each cell lists '
    'the titles it links to). The answer usually needs a table lookup combined with a fact from a linked page. Return '
    'just the short answer.',
    'question needs both table and pages: find the row(s) the question is about, then read the pages linked from '
    'those cells. Return only the answer phrase (a name, number, date or short span).',
]
HYBRID_NEURALESE = {
    'helper': 'read_page', 'args': {'question': 'string', 'title': 'string', 'page': 'string'},
    'returns': 'Neuralese<string>',
    'instruction': 'Read page (the encyclopedia page title) and write a private note of every fact in it that bears on '
                   'question, or note that it is irrelevant.',
    'root_note': 'read_page gives a private note of what one linked page says about question. Narrow down the rows '
                 'in eval first, then read the pages linked from those rows and decide from the notes.',
}


def hybridqa(cap_train=1500, cap_eval=150, seed=20261031):
    root = glob.glob(f'{SOURCES_DIR}/hybridqa/WikiTables-WithLinks-*')[0]
    out = []
    for split, path in (('train', 'train'), ('validation', 'dev')):
        rows = [json.loads(line) for line in open(f'{SOURCES_DIR}/hybridqa/{path}.jsonl')]
        random.Random(seed).shuffle(rows)
        pool = [r['answer-text'] for r in rows[:2000]]
        n = 0
        for r in rows:
            answer = r.get('answer-text') or ''
            if not answer or len(answer.split()) > 8:
                continue
            try:
                table = json.load(open(f'{root}/tables_tok/{r["table_id"]}.json'))
                passages = json.load(open(f'{root}/request_tok/{r["table_id"]}.json'))
            except OSError:
                continue
            title = lambda link: link.split('/wiki/', 1)[-1].replace('_', ' ')
            cells = [[{'text': cell[0], 'pages': [title(link) for link in cell[1]]} for cell in row] for row in table['data']]
            pages = {title(link): text for link, text in passages.items()}
            if len(json.dumps(pages)) > 120000:
                continue
            inputs = {'question': r['question'], 'table': {'title': table.get('title', ''),
                                                            'columns': [h[0] for h in table['header']], 'rows': cells},
                      'pages': pages}
            out.append(make_case(
                source='hybridqa', family='table_and_pages_qa', shape=f'hybridqa-{min(len(cells) // 10, 5)}0rows',
                upstream='wenhuchen/HybridQA', upstream_id=r['question_id'], upstream_split=split,
                unit=f'hybridqa-table:{r["table_id"]}', groups=[f'hybridqa-table:{r["table_id"]}'],
                root_name='answer_from_table_and_pages',
                args={'question': 'string', 'table': '{ title: string; columns: string[]; rows: { text: string; pages: string[] }[][] }',
                      'pages': 'Record<string, string>'},
                returns='string', instruction=pick(HYBRID_INSTRUCTIONS, r['question_id']), inputs=inputs, expected=answer,
                oracle=dict(ORACLE), level=2, license_='CC-BY-4.0', domain='relational',
                wrong=distinct_wrong(answer, pool[n:] + pool[:n], 2), aliases=[answer.upper()],
                neuralese_room='linked pages read by children returning notes; parent joins with the table lookup',
                neuralese=HYBRID_NEURALESE, revision='f4ed68e'))
            n += 1
            if n >= (cap_train if split == 'train' else cap_eval * 2):
                break
    return out


LONG_INSTRUCTIONS = [
    'Answer question using pages, encyclopedia pages about entities the question mentions. The pages are long; the '
    'answer may sit anywhere in them. Return only the short answer (a name, place, date or number).',
    'pages are full encyclopedia articles related to question. Find the passage that answers it and return just the '
    'answer phrase.',
]
LONG_NEURALESE = {
    'helper': 'read_section', 'args': {'question': 'string', 'title': 'string', 'section': 'string'},
    'returns': 'Neuralese<string>',
    'instruction': 'Read section (a part of the encyclopedia page title) and write a private note of anything in it '
                   'that answers or narrows down question, or note that it has nothing relevant.',
    'root_note': 'read_section gives a private note for one part of a page. Split long pages into sections (for '
                 'example by paragraph groups), collect the notes, and answer from them.',
}


def triviaqa(cap_train=1200, cap_eval=150, scan=20000, max_chars=150000, seed=20261031):
    base = f'{SOURCES_DIR}/triviaqa_rc_wikipedia'
    out = []
    for split, path in (('train', 'rc.wikipedia__train.jsonl'), ('validation', 'rc.wikipedia__validation.jsonl')):
        rows = []
        with open(f'{base}/{path}') as handle:
            for index, line in enumerate(handle):
                if index >= scan:
                    break
                rows.append(line)
        random.Random(seed).shuffle(rows)
        n, pool = 0, []
        for line in rows:
            r = json.loads(line)
            pages = [{'title': t, 'text': c} for t, c in zip(r['entity_pages']['title'], r['entity_pages']['wiki_context'])]
            text = ' '.join(p['text'] for p in pages)
            if not pages or len(text) > max_chars:
                continue
            answer = r['answer']['value']
            aliases = list(dict.fromkeys(r['answer'].get('aliases') or []))[:30]
            lowered = text.lower()
            if not any(a.lower() in lowered for a in [answer] + aliases) or len(answer.split()) > 6:
                continue
            pool.append(answer)
            if len(pool) < 3:
                continue
            out.append(make_case(
                source='triviaqa', family='long_page_qa', shape=f'triviaqa-{len(pages)}pages-{min(len(text) // 30000, 4)}',
                upstream='mandarjoshi/trivia_qa', upstream_id=r['question_id'], upstream_split=split,
                unit=f'triviaqa-q:{r["question_id"]}', groups=[f'triviaqa-q:{r["question_id"]}'],
                root_name='answer_from_pages', args={'question': 'string', 'pages': '{ title: string; text: string }[]'},
                returns='string', instruction=pick(LONG_INSTRUCTIONS, r['question_id']),
                inputs={'question': r['question'], 'pages': pages}, expected=answer,
                oracle={**ORACLE, 'alternates': [a for a in aliases if a != answer]}, level=2 if len(text) > 30000 else 1,
                license_='Apache-2.0', license_note='evidence pages are Wikipedia text (CC-BY-SA)', domain='relational',
                wrong=distinct_wrong(answer, pool[-3::-1], 2), aliases=[a for a in aliases if a != answer][:2],
                neuralese_room='long pages split into sections read by children returning notes',
                neuralese=LONG_NEURALESE, revision=''))
            n += 1
            if n >= (cap_train if split == 'train' else cap_eval * 2):
                break
    return out
