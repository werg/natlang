"""Multi-document QA (HotpotQA distractor, MuSiQue, 2WikiMultihopQA): a question over a set of paragraphs, most
of them distractors. Natlang framing: the paragraphs are the function's `documents` input; per-document reading can
be handed to child calls that return evidence to the parent (Neuralese room). Oracle: token F1 >= 0.8 against the
gold answer or any upstream alias (SQuAD normalization; numbers must match exactly; yes/no exact)."""
from __future__ import annotations

import glob
import json
import random

from ..common import distinct_wrong, make_case, pick

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
    root = glob.glob('/mnt/external/sdkb-archive/raw/hotpotqa-distractor-*')[0]
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


def musique(base='/home/werg/data/external-rl-sources/musique', cap_train=1500, cap_eval=150, seed=20261031):
    out = []
    files = {'train': glob.glob(f'{base}/*train*.jsonl'), 'validation': glob.glob(f'{base}/*dev*.jsonl') +
             glob.glob(f'{base}/*validation*.jsonl')}
    for split, paths in files.items():
        rows = [json.loads(line) for path in paths for line in open(path) if line.strip()]
        rows = [r for r in rows if r.get('answerable', True)]
        random.Random(seed).shuffle(rows)
        pool = [r['answer'] for r in rows[:2000]]
        random.Random(seed + 1).shuffle(pool)
        for r in rows[:cap_train if split == 'train' else cap_eval * 2]:
            docs = [{'title': p['title'], 'text': p['paragraph_text']} for p in r['paragraphs']]
            hops = r['id'].split('hop')[0]
            seeds = [part for part in r['id'].split('__', 1)[1].split('_') if part.isdigit()] if '__' in r['id'] else []
            groups = [f'musique:{r["id"]}'] + [f'musique:seed:{s}' for s in seeds]
            level = {'2': 1, '3': 2, '4': 3}.get(hops, 2)
            out.append(_case('musique', 'MuSiQue', '', 'CC-BY-4.0', r['id'], split, r['question'], docs, r['answer'],
                             r.get('answer_aliases') or [], level, f'musique-{hops}hop-{len(docs)}docs', groups, pool,
                             difficulty=f'{hops}hop'))
    return out


def twowiki(base='/home/werg/data/external-rl-sources/2wikimultihopqa', cap_train=1500, cap_eval=150, seed=20261031):
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
