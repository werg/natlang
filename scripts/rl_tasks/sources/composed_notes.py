"""Gold, checked and mechanical reference notes for the composed families' Neuralese producers
(``scripts/rl_tasks/child_notes.py`` documents the shape and the provenance vocabulary).

One function per upstream dataset and helper role, each taking the normalized item the loaders in ``composed.py``
keep (the upstream annotations that the case's oracle does not use are exactly what these notes are made of) and
returning ``reference.children`` entries. Every note states only what the annotation or the inputs support.
"""
from __future__ import annotations

import ast
import re

from ..child_notes import (answer_sentences, child, clip, document_key, fragment, overlap, rendered, sentence_at,
                           string_key,
                           weakest)

PARENTHETICAL = re.compile(r'\s*\([^)]*\)\s*$')
NUMBER = re.compile(r'\d[\d,]*\.?\d*')


def _key(text):
    return fragment(text)


# ---------------------------------------------------------------- MuSiQue

def _musique_steps(item):
    """Hop lines of an answerable MuSiQue item: (line, source title, sentence, provenance)."""
    answers, out = [], []
    documents = {d['title']: d['text'] for d in item['documents']}
    for step in item['steps']:
        question = re.sub(r'#(\d+)', lambda m: answers[int(m.group(1)) - 1] if 0 < int(m.group(1)) <= len(answers) else m.group(0),
                          step['question'])
        answers.append(step['answer'])
        title = item['support_paragraph'].get(step['support']) if step['support'] is not None else None
        sentence = answer_sentences(documents.get(title, ''), [step['answer']], 400) if title else None
        out.append((f'{question} -> {step["answer"]}', title, sentence, 'checked' if sentence else 'gold'))
    return out


def musique_note(item, upto=None, answer=False):
    steps = _musique_steps(item)[:upto]
    lines = []
    for number, (line, title, sentence, _) in enumerate(steps, 1):
        lines.append(f'{number}. {line}' + (f' (document "{title}")' if title else ''))
        if sentence:
            lines.append(f'   {sentence}')
    if answer:
        lines.append(f'The chain reaches the answer: {item["answer"]}.')
    return clip('\n'.join(lines)), weakest(*(s[3] for s in steps))


def musique_relay(program, item):
    """gather / refine, or first_link / next_link, for a MuSiQue item."""
    question = _key(item['question'])
    out = []
    if program.name == 'hop-relay':
        first, provenance = musique_note(item, upto=1)
        full, full_provenance = musique_note(item, answer=True)
        out.append(child('first_link', [question], first, provenance))
        out.append(child('next_link', [question], full, full_provenance))
        return out
    gathered, provenance = musique_note(item)
    refined, refined_provenance = musique_note(item, answer=True)
    out.append(child('gather', [question], gathered, provenance))
    out.append(child('refine', [question], refined, refined_provenance))
    return out


def musique_support(item):
    """{title: note} of every gold supporting paragraph: the hop questions it answers and the sentence that holds each
    answer (found by code in the gold paragraph)."""
    documents = {d['title']: d for d in item['documents']}
    out = {}
    for line, title, sentence, _ in _musique_steps(item):
        if title in documents and sentence:
            out[title] = clip((out[title] + '\n' if title in out else '') + f'{line}\n{sentence}')
    return out


def musique_passage_notes(item):
    """read_passage(step, document) of the depth-3 program: one note per gold supporting paragraph."""
    return [child('read_passage', [document_key(title)], note, 'checked') for title, note in musique_support(item).items()]


# ---------------------------------------------------------------- 2WikiMultihopQA

def _wiki_sentences(item):
    out = []
    for title, sent_id in item['support']:
        sentences = item['sentences'].get(title) or []
        if 0 <= sent_id < len(sentences) and sentences[sent_id].strip():
            out.append((title, sentences[sent_id].strip()))
    return out


def wiki_note(item, upto=None, answer=False):
    triples = item['evidences'][:upto]
    lines = [f'{subject} | {relation} | {obj}' for subject, relation, obj in triples]
    sentences = _wiki_sentences(item)
    if upto is not None and triples:
        sentences = [(t, s) for t, s in sentences if triples[0][2].lower() in s.lower() or PARENTHETICAL.sub('', t) == triples[0][0]]
    lines += [f'[{title}] {sentence}' for title, sentence in sentences]
    if answer:
        lines.append(f'The chain reaches the answer: {item["answer"]}.')
    return (clip('\n'.join(lines)) if lines else None), 'gold'


def wiki_relay(program, item):
    question = _key(item['question'])
    out = []
    if program.name == 'hop-relay':
        first, provenance = wiki_note(item, upto=1)
        full, full_provenance = wiki_note(item, answer=True)
        out += [child('first_link', [question], first, provenance), child('next_link', [question], full, full_provenance)]
        return out
    gathered, provenance = wiki_note(item)
    refined, refined_provenance = wiki_note(item, answer=True)
    out += [child('gather', [question], gathered, provenance), child('refine', [question], refined, refined_provenance)]
    return out


def wiki_profiles(item):
    """profile(question, document) of every document: relevance is the gold supporting-document flag and the facts are
    the gold evidence triples whose subject is the document's subject."""
    question = _key(item['question'])
    supporting = {title for title, _ in item['support']}
    titles = [d['title'] for d in item['documents']]
    out = []
    for document in item['documents']:
        title = document['title']
        if titles.count(title) != 1:
            continue
        subject = PARENTHETICAL.sub('', title)
        facts = [{'attribute': relation, 'value': obj} for s, relation, obj in item['evidences'] if s in (title, subject)]
        out.append(child('profile', [question, document_key(title)], {'subject': title, 'relevant': title in supporting, 'facts': facts}, 'gold'))
    return out


def wiki_support(item):
    """{title: note} of every gold supporting document: its gold supporting sentences with the evidence triples about it."""
    documents = {d['title'] for d in item['documents']}
    out = {}
    for title in dict.fromkeys(t for t, _ in item['support']):
        if title not in documents:
            continue
        subject = PARENTHETICAL.sub('', title)
        lines = [s for t, s in _wiki_sentences(item) if t == title]
        lines += [f'{s} | {r} | {o}' for s, r, o in item['evidences'] if s in (title, subject)]
        if lines:
            out[title] = clip('\n'.join(lines))
    return out


def wiki_entry_notes(item):
    """read_entry(entity, attribute, document) of the lookup program. A non-supporting document gets no note (its
    content is not annotated)."""
    return [child('read_entry', [document_key(title)], note, 'gold') for title, note in wiki_support(item).items()]


# ---------------------------------------------------------------- QuALITY (no annotation of evidence: mechanical)

def quality_target(item):
    return item['question'] + ' ' + item['options'][item['answer']]


def quality_evidence(item):
    text = overlap(item['article'], quality_target(item))
    return text


def quality_relay(program, item):
    question = _key(item['question'])
    evidence = quality_evidence(item)
    out = [child('collect_evidence', [question], evidence, 'mechanical:overlap-sentences') if evidence else None]
    if 'sharpen' in program.helpers:
        lines = []
        for letter, option in item['options'].items():
            support = overlap(item['article'], item['question'] + ' ' + option, 360)
            if support:
                lines.append(f'{letter}) {option}\n   Closest passage: {support.splitlines()[0]}')
        if lines:
            out.append(child('sharpen', [_key(item['article'])], clip('\n'.join(lines)), 'mechanical:overlap-sentences'))
    return out


def quality_parts(item, parts):
    """open_notes / absorb over the parts of an article: the overlap sentences of the parts read so far."""
    target = quality_target(item)
    out, so_far = [], []
    for index, part in enumerate(parts):
        notes = overlap(part, target, 700)
        so_far.append(notes or '')
        cumulative = clip('\n'.join(n for n in so_far if n)) or 'Nothing in the parts read so far bears on the question.'
        out.append(child('open_notes' if index == 0 else 'absorb', [_key(part)], cumulative, 'mechanical:overlap-sentences'))
    return out


# ---------------------------------------------------------------- SQuAD

def squad_locate(item):
    """locate(article, question): the gold answer span and the sentence of its gold offset."""
    out = []
    context = item['article']
    for q in item['questions']:
        text, start = q['answers']['text'][0], q['answers']['answer_start'][0]
        if context[start:start + len(text)] != text:
            continue
        sentence = sentence_at(context, start)
        if text in sentence:
            out.append(child('locate', [_key(q['question'])], {'answer': text, 'sentence': sentence}, 'checked'))
    return out


# ---------------------------------------------------------------- CLUTRR

def _clutrr_leaves(proof):
    """Base facts of the gold proof: premises that no step derives."""
    try:
        steps = ast.literal_eval(proof) if isinstance(proof, str) else proof
    except (ValueError, SyntaxError):
        return []
    derived = {key for step in steps for key in step}
    leaves = []
    for step in steps:
        for premises in step.values():
            for fact in premises:
                if fact not in derived and fact not in leaves:
                    leaves.append(fact)
    return leaves


def clutrr_sentences(story):
    return [s for s in re.split(r'(?<=[.!?])\s+', story) if s.strip()]


def clutrr_relations(item):
    """relations_in(sentence): the gold proof's base relations whose two people the sentence names. A sentence with no
    base relation of the proof gets no note (it may state relations outside the proof)."""
    leaves = _clutrr_leaves(item['proof'])
    out = []
    seen = [s for s in clutrr_sentences(item['story'])]
    for sentence in seen:
        names = set(re.findall(r'\[([^\]]+)\]', sentence))
        facts = [{'person': a, 'relation': r, 'relative': b} for a, r, b in leaves if a in names and b in names]
        if facts and sum(s == sentence for s in seen) == 1:
            out.append(child('relations_in', [_key(sentence)], facts, 'checked'))
    return out


def clutrr_links(item):
    """links_of(person, story): the proof's base relations that involve the person, in words."""
    leaves = _clutrr_leaves(item['proof'])
    out = []
    for person in dict.fromkeys(name for a, _, b in leaves for name in (a, b)):
        lines = [f'{b} is {a}\'s {r}.' for a, r, b in leaves if person in (a, b)]
        out.append(child('links_of', [string_key('person', person)], '\n'.join(lines), 'checked'))
    return out


# ---------------------------------------------------------------- bAbI

def babi_states(item, chunks):
    """begin_state / advance: the gold supporting facts among the facts applied so far."""
    support = set(item['support'])
    facts = {int(f.split('.', 1)[0]): f for f in item['facts']}
    out = []
    for index, chunk in enumerate(chunks):
        last = int(chunk[-1].split('.', 1)[0])
        lines = [facts[i] for i in sorted(support) if i <= last and i in facts]
        text = ('Facts so far that bear on the question:\n' + '\n'.join(lines) if lines
                else 'None of the facts so far bears on the question.')
        out.append(child('begin_state' if index == 0 else 'advance', [_key(chunk[0])], text, 'gold'))
    return out


# ---------------------------------------------------------------- TriviaQA

def trivia_candidates(item, sections):
    """open_candidates / scan: the gold answer with the sentence that holds it, from the sections read so far."""
    names = [item['answer'], *item['aliases']]
    found = None
    out = []
    for index, section in enumerate(sections):
        if found is None:
            sentence = answer_sentences(section['text'], names, 300)
            if sentence:
                found = {'answer': item['answer'], 'evidence': sentence}
        value = {'candidates': [found] if found else []}
        key = [_key(section['text'])]
        out.append(child('open_candidates' if index == 0 else 'scan', key, value, 'checked'))
    return out


# ---------------------------------------------------------------- TAT-QA, MultiHiertt, HybridQA

def _numbers(text):
    return {n.replace(',', '').rstrip('.') for n in NUMBER.findall(text)}


def table_rows_text(rows, index):
    return ' | '.join(cell for cell in rows[index] if str(cell).strip())


def tatqa_notes(item, answer):
    """read_table / read_text of a TAT-QA question. ``answer_from`` and ``rel_paragraphs`` are annotations; table rows are
    the rows holding a number the gold derivation (or answer) uses."""
    question = _key(item['question'])
    out = []
    table = item['table']
    if item['answer_from'] == 'text':
        out.append(child('read_table', [question], 'Nothing in the table bears on the question.', 'gold'))
    else:
        wanted = _numbers(item['derivation']) | _numbers(' '.join(map(str, answer)) if isinstance(answer, list) else str(answer))
        rows = [i for i, row in enumerate(table) if i > 1 and wanted & set().union(*(_numbers(str(c)) for c in row))]
        if rows:
            header = [table_rows_text(table, i) for i in range(min(2, len(table))) if table_rows_text(table, i)]
            out.append(child('read_table', [question], clip('\n'.join(header + [table_rows_text(table, i) for i in rows])), 'checked'))
    paragraphs = item['paragraphs']
    if item['answer_from'] == 'table':
        out.append(child('read_text', [question], 'Nothing in the paragraphs bears on the question.', 'gold'))
    elif item['rel']:
        out.append(child('read_text', [question], clip('\n'.join(paragraphs[i] for i in item['rel'] if i < len(paragraphs))), 'gold'))
    return out


def evidence_positions(original, indices):
    """Positions in the non-empty paragraphs of the upstream paragraph indices an annotation names."""
    kept = {index: position for position, index in enumerate(i for i, p in enumerate(original) if str(p).strip())}
    return [kept[i] for i in indices if isinstance(i, int) and i in kept]


def multihiertt_notes(item, program):
    """read_table (per table), read_text and plan_calculation of a MultiHiertt question: the gold evidence cells'
    descriptions (the dataset's own sentences for them) and the gold evidence paragraphs."""
    question = _key(item['question'])
    by_table = {}
    for cell in item['table_evidence']:
        table_index = int(cell.split('-')[0])
        description = item['descriptions'].get(cell)
        if description:
            by_table.setdefault(table_index, []).append(description)
    tables = item['tables']
    cells = [{c for row in table for c in map(lambda x: str(x).strip(), row) if len(c) >= 6 and fragment(c)} for table in tables]
    out = []
    for index in range(len(tables)):
        # A cell names its table when no other table's cell shows its rendered fragment.
        others = [rendered(c) for j, group in enumerate(cells) if j != index for c in group]
        unique = next((c for c in sorted(cells[index], key=lambda c: (-len(c), c))
                       if not any(fragment(c) in other for other in others)), None)
        if unique is None:
            continue
        value = '\n'.join(by_table[index]) if by_table.get(index) else 'Nothing in this table bears on the question.'
        out.append(child('read_table', [question, fragment(unique)], clip(value), 'gold'))
    paragraphs = item['paragraphs']
    texts = [paragraphs[i] for i in item['text_evidence'] if isinstance(i, int) and i < len(paragraphs)]
    out.append(child('read_text', [question], clip('\n'.join(texts)) if texts else 'Nothing in the paragraphs bears on the question.', 'gold'))
    if 'plan_calculation' in program.helpers:
        values = [d for table in sorted(by_table) for d in by_table[table]]
        steps = item['program']
        plan = ('Values:\n' + '\n'.join(values) if values else '') + (f'\nSteps (in order): {steps}' if steps else '')
        out.append(child('plan_calculation', [question], clip(plan) if plan.strip() else None, 'gold'))
    return out


def hybrid_notes(item):
    """find_rows / read_page of a HybridQA question: the rows (and linked pages) that the gold answer text sits in,
    located by code in the cells and the linked pages."""
    question = _key(item['question'])
    answer = item['answer']
    table, pages = item['table'], item['pages']
    rows = []
    for number, row in enumerate(table['rows'], 1):
        linked = [title for cell in row for title in cell['pages'] if answer.lower() in pages.get(title, '').lower()]
        in_cell = any(answer.lower() in cell['text'].lower() for cell in row)
        if linked or in_cell:
            cells = '; '.join(f'{column}: {cell["text"]}' for column, cell in zip(table['columns'], row) if cell['text'].strip())
            titles = sorted({title for cell in row for title in cell['pages']})
            rows.append(f'Row {number}: {cells}. Linked pages: {", ".join(titles)}' if titles else f'Row {number}: {cells}')
    out = []
    if rows:
        out.append(child('find_rows', [question], clip('\n'.join(rows[:6])), 'checked'))
    for title, text in pages.items():
        sentence = answer_sentences(text, [answer], 500)
        if sentence:
            out.append(child('read_page', [question, string_key('title', title)], sentence, 'checked'))
    return out
