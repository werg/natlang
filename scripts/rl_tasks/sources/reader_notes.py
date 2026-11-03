"""Reference notes for the single Neuralese helper of the reader-framed external cases (framing ``neuralese-helper``;
``scripts/rl_tasks/child_notes.py`` documents the entry shape and provenance vocabulary).

Each function takes the builder's own upstream row and returns entries for that source's helper, keyed by the
helper's call site (which document, page, question, sentence or file it is called on). Helpers whose call sites the
model itself chooses at run time (a QuALITY part, a TriviaQA section, a TabFact row slice, a knights-and-knaves
assumption, a natlang function name) get no note here: a note keyed by something the case does not fix could be wrong
for the call it reaches.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re

from ..common import SOURCES_DIR, text_digest
from ..child_notes import answer_sentences, child, clip, document_key, fragment, sentence_at, string_key
from . import composed_notes as composed

IRRELEVANT = 'The document is irrelevant to the question.'


def read_document_notes(question, documents, support, provenance):
    """read_document(question, document): the gold supporting note of a supporting document, else the gold irrelevance
    of a distractor. ``support`` is {title: note}."""
    titles = [d['title'] for d in documents]
    key = fragment(question)
    out = []
    for title in dict.fromkeys(titles):
        if titles.count(title) != 1:
            continue
        note = support.get(title)
        out.append(child('read_document', [key, document_key(title)], note or IRRELEVANT, provenance if note else 'gold'))
    return out


def derived_document_notes(row, context=None):
    """read_document notes of an existing case whose upstream annotation is not available: the sentences of each document
    that contain the gold answer (mechanical; a document without it gets no note, since distractors are not annotated)."""
    inputs, semantics = row['semantics']['inputs'], row['semantics']
    answer = semantics['expected']
    oracle = semantics.get('oracle') or {}
    names = [answer, *(oracle.get('alternates') or [])] if isinstance(answer, str) and answer.lower() not in ('yes', 'no') else []
    titles = [d['title'] for d in inputs['documents']]
    out = []
    for document in inputs['documents']:
        sentence = answer_sentences(document['text'], names, 500) if names and titles.count(document['title']) == 1 else None
        if sentence:
            out.append(child('read_document', [fragment(inputs['question']), document_key(document['title'])], sentence,
                             'mechanical:answer-lines'))
    return out


def hotpot_notes(row, documents):
    facts = row['supporting_facts']
    sentences = dict(zip(row['context']['title'], row['context']['sentences']))
    support = {}
    for title, index in zip(facts['title'], facts['sent_id']):
        parts = sentences.get(title) or []
        if 0 <= index < len(parts) and parts[index].strip():
            support[title] = clip((support[title] + '\n' if title in support else '') + parts[index].strip())
    return read_document_notes(row['question'], documents, support, 'gold')


def musique_notes(row, documents):
    item = {'documents': documents, 'answer': row['answer'],
            'steps': [{'question': s['question'], 'answer': s['answer'], 'support': s.get('paragraph_support_idx')}
                      for s in row['question_decomposition']],
            'support_paragraph': {p['idx']: p['title'] for p in row['paragraphs']}}
    return read_document_notes(row['question'], documents, composed.musique_support(item), 'checked')


def twowiki_notes(row, documents, context):
    support = row.get('supporting_facts') or {}
    if isinstance(support, str):
        support = ast.literal_eval(support)
    evidences = row.get('evidences') or []
    if isinstance(evidences, str):
        evidences = ast.literal_eval(evidences)
    item = {'documents': documents, 'support': list(zip(support.get('title') or [], support.get('sent_id') or [])),
            'evidences': [list(e) for e in evidences if isinstance(e, (list, tuple)) and len(e) == 3],
            'sentences': {t: list(s) for t, s in context if isinstance(s, (list, tuple))}}
    return read_document_notes(row['question'], documents, composed.wiki_support(item), 'gold')


def hybrid_page_notes(question, answer, table, pages):
    """read_page(question, title, page): pages that hold the gold answer text, with its sentence (found by code)."""
    entries = composed.hybrid_notes({'question': question, 'answer': answer, 'table': table, 'pages': pages})
    return [e for e in entries if e['match'][0].endswith('read_page(')]


def squad_notes(context, questions):
    """find_answer(article, question): the gold answer span and the sentence of its gold offset."""
    out = []
    for q in questions:
        text, start = q['answers']['text'][0], q['answers']['answer_start'][0]
        if context[start:start + len(text)] != text:
            continue
        sentence = sentence_at(context, start)
        if text in sentence:
            out.append(child('find_answer', [fragment(q['question'])], f'Answer: "{text}"\nSentence: {sentence}', 'checked'))
    return out


def _jsonl(path):
    with open(path) as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def derived_clutrr_notes(row, context):
    """extract_relation_edge notes of an existing guided CLUTRR case: the gold proof of its upstream row (looked up by id in
    the local CLUTRR files; a story that differs from the case's is not used)."""
    index = context.get('clutrr')
    if index is None:
        index = context['clutrr'] = {}
        for split in ('train', 'test'):
            for r in _jsonl(f'{SOURCES_DIR}/clutrr/gen_train234_test2to10__{split}.jsonl'):
                index[r['id']] = (r['story'], r.get('proof_state') or '')
    found = index.get(row['curriculum']['external']['upstream_id'])
    if found is None or found[0] != row['semantics']['inputs']['story']:
        return []
    return clutrr_edge_notes(*found)


def derived_babi_notes(row, context):
    """track_query_history notes of an existing guided bAbI case: the gold supporting facts of its upstream row."""
    index = context.get('babi')
    if index is None:
        index = context['babi'] = {}
        for split in ('train', 'test'):
            for r in _jsonl(f'{SOURCES_DIR}/babi/{split}.jsonl'):
                index[f'{r["story_id"]}:{text_digest(r["question"])[:8]}'] = r.get('supporting_fact_ids') or []
    inputs = row['semantics']['inputs']
    return babi_history_notes(inputs['facts'], index.get(row['curriculum']['external']['upstream_id'], []), inputs['question'])


def clutrr_edge_notes(story, proof):
    """extract_relation_edge(sentence): "B is A's relation" for the gold proof's base relations whose two people the
    sentence names (a sentence without one gets no note: it may state relations outside the proof)."""
    leaves = composed._clutrr_leaves(proof)
    sentences = composed.clutrr_sentences(story)
    out = []
    for sentence in sentences:
        names = set(re.findall(r'\[([^\]]+)\]', sentence))
        edges = [f"{b} is {a}'s {relation}" for a, relation, b in leaves if a in names and b in names]
        if edges and sentences.count(sentence) == 1:
            out.append(child('extract_relation_edge', [fragment(sentence)], '; '.join(edges), 'checked'))
    return out


def babi_history_notes(facts, support, question):
    """track_query_history(facts, question): the gold supporting facts, with their ids."""
    wanted = set(support)
    lines = [f for f in facts if int(f.split('.', 1)[0]) in wanted]
    if not lines:
        return []
    return [child('track_query_history', [fragment(question)], 'Decisive facts:\n' + '\n'.join(lines), 'gold')]


def multihiertt_table_notes(question, qa, tables, paragraphs, descriptions):
    """read_table(question, table): the dataset's own sentences for the gold evidence cells of that table."""
    parse = lambda v: ast.literal_eval(v) if isinstance(v, str) else (v or [])
    item = {'question': question, 'tables': tables, 'paragraphs': paragraphs, 'descriptions': dict(parse(descriptions)),
            'table_evidence': parse(qa.get('table_evidence')), 'text_evidence': [],
            'program': str(qa.get('program') or '')}

    class Plain:
        helpers = {}
    return [e for e in composed.multihiertt_notes(item, Plain) if e['match'][0].endswith('read_table(')]


def derived_sql_notes(row, context=None):
    """plan_query notes of an existing SQL case: the reference query the case recorded as ``external.gold_sql``."""
    sql = (row['curriculum']['external'] or {}).get('gold_sql')
    return sql_plan_notes(row['semantics']['inputs']['question'], sql) if sql else []


def sql_plan_notes(question, sql):
    """plan_query(question, schema): the dataset's reference query is the plan."""
    return [child('plan_query', [fragment(question)], clip('The reference query for this question:\n' + sql.strip()), 'gold')]


def _frontmatter(text):
    match = re.match(r'---\n([\s\S]*?)\n---', text)
    out, key = {}, ''
    for line in (match.group(1).split('\n') if match else []):
        head = re.match(r'^([A-Za-z_]+):\s?(.*)$', line)
        if head:
            key = head.group(1)
            out[key] = head.group(2)
        elif key:
            out[key] += '\n' + line
    return out


def derived_function_notes(row, context):
    """read_function notes of an existing natlang-codebase case: each function the exact gold answer (or the declared type
    the question is about) points at, described from its own file in the case: its declared parameters/result, the lines
    that mention the asked-about identifier, or the children it calls. The call is keyed by the function's own body, which
    the child's ``source`` argument shows, whatever name the model gives it."""
    ext = row['curriculum']['external']
    kind = ext['codebase']['kind'].removeprefix('explore-')
    x = ext['upstream_id'].split(':', 2)[2]
    answer = row['semantics']['expected']
    files = {path.removeprefix('explore_program/'): text for path, text in row['semantics']['files'].items()
             if path.startswith('explore_program/')}
    nl = {path: text for path, text in files.items() if path.endswith('.nl')}
    path_of = lambda dotted: dotted.replace('.', '/') + '.nl'
    body = lambda text: text.split('---', 2)[-1].strip()
    out = []

    def add(path, value):
        if path in nl and value.strip():
            out.append(child('read_function', [fragment(body(nl[path]))], clip(value), 'checked'))

    declared = lambda path: _frontmatter(nl[path])
    if kind == 'children' and isinstance(answer, list):
        add(path_of(x), f'{x} calls these named items directly: {", ".join(answer)}.')
    elif kind in ('param-type', 'return-type') and isinstance(answer, list):
        field = 'args' if kind == 'param-type' else 'returns'
        for dotted in answer:
            path = path_of(dotted)
            if path in nl:
                what = 'a parameter' if field == 'args' else 'a result'
                add(path, f'{dotted} declares {what} whose type mentions {x}:\n{declared(path).get(field, "").strip()}')
    elif kind == 'mentions' and isinstance(answer, list):
        for dotted in answer:
            path = path_of(dotted)
            if path in nl:
                lines = [line.strip()[:160] for line in body(nl[path]).splitlines() if x in line][:4]
                add(path, f'{dotted} mentions {x} in its instructions:\n' + '\n'.join(lines))
    elif kind == 'leaves' and isinstance(answer, list):
        for dotted in answer:
            add(path_of(dotted), f'{dotted} is a leaf of {x}: it calls no other named function.')
    elif kind in ('fields', 'optional-fields', 'field-type'):
        alias = x.split(' of ')[-1] if kind == 'field-type' else x
        fact = (f'{alias} has these fields: {", ".join(answer)}.' if kind == 'fields' and isinstance(answer, list) else
                f'{alias} has these optional fields: {", ".join(answer)}.' if kind == 'optional-fields' and isinstance(answer, list) else
                f'The type of {x} is declared as {answer}.' if kind == 'field-type' and isinstance(answer, str) else None)
        if fact:
            users = [path for path in sorted(nl) if re.search(rf'\b{re.escape(alias)}\b', declared(path).get('args', '') + declared(path).get('returns', ''))]
            for path in users[:6]:
                add(path, fact)
    return out


def derived_file_notes(row, context):
    """read_file notes of an existing codebase case, from the exact repository snapshot its service serves (the asset the
    case binds by SHA-256) and its recorded gold answer. ``context``: {'assets_v3': Path, 'cache': dict}."""
    external = row['curriculum']['external']
    codebase = external['codebase']
    files = context['cache'].get(codebase['asset_sha256'])
    if files is None:
        path = context['assets_v3'] / codebase['asset']
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != codebase['asset_sha256']:
            return []
        files = context['cache'][codebase['asset_sha256']] = json.loads(data)
    kind = codebase['kind']
    x = '' if kind == 'mutation' else external['upstream_id'].split(':', 2)[2]
    return file_notes(kind, x, row['semantics']['expected'], files)


def file_notes(kind, x, answer, files):
    """read_file(question, path, text): for each file the exact answer names, the matches in that file (the oracle's own
    list) and the lines that mention the asked-about symbol (found by code). A file the answer does not name gets no note."""
    by_file = {}
    for item in answer if isinstance(answer, list) else [answer]:
        path, _, name = str(item).partition(':')
        by_file.setdefault(path, []).append(name)
    symbol = x.rsplit('.', 1)[-1] if re.fullmatch(r'[A-Za-z_$][\w$.]*', x or '') else None
    out = []
    for path, names in by_file.items():
        text = files.get(path)
        if text is None:
            continue
        names = [n for n in names if n]
        if kind == 'definition':
            lines = [f'{path} defines {x}.']
        elif kind == 'importers':
            lines = [f'{path} imports {x}.']
        elif names:
            lines = [f'In {path}, matching the question: {", ".join(names)}.']
        else:
            lines = [f'{path} is one of the files the question asks for.']
        if symbol and len(symbol) > 2:
            hits = [f'L{number}: {line.strip()[:140]}' for number, line in enumerate(text.splitlines(), 1) if symbol in line][:6]
            if hits:
                lines += [f'Lines mentioning {symbol}:', *hits]
        out.append(child('read_file', [string_key('path', path)], clip('\n'.join(lines)), 'checked'))
    return out
