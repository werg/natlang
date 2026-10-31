#!/usr/bin/env python3
"""Source adapters for broad Neuralese producer/consumer graphs (plans/neuralese/NEURALESE_BROAD_GRAPHS_20261031.md).

Each adapter reads one pinned public dataset and emits source-agnostic cases (``natlang.broad-graph-case/1``): a
request, the documents only producers see, one gold-derived view per document (or the kind's exact "none" line), and
the dataset's gold answer. ``ts-host/scripts/inline-curriculum/build-broad-graph-calls.mjs`` turns cases into natlang
programs; nothing here renders, tokenizes or admits training rows.

Views are verbatim selections (dataset supporting facts, answer-span sentences, relevant turns, highlighted cells,
dataset event templates) or code-computed references to the gold target (SQL identifiers, gold-call tools, a greedy
ROUGE extractive oracle); every document records its ``view_origin``. No teacher text.

Splits: upstream test/validation -> ``test``; train-only sources are carved by world group. Groups follow the S1/view
corpus namespaces, and the S1 cross-corpus group index moves a case to ``test`` when S1 holds one of its groups there
(an upstream test case that S1 has in train is omitted). Cases sharing any group are closed into one component on one
split (``close``). Source-review holds are skipped by dataset id. Protected-world custody runs on the built IR
(``custody``), where ``ProtectedWorlds.check`` sees the exact program world.

    broad_graph_cases.py build --out DIR [--sources a,b] [--train-cap N] [--test-cap M]
    broad_graph_cases.py custody --ir DIR --out DIR
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import hashlib
import json
import os
import random
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'training' / 'neuralese'))
from natlang_neuralese.common.paths import resolve as data_path  # noqa: E402
from natlang_neuralese.common.hashing import sha256_hex  # noqa: E402

SCHEMA = 'natlang.broad-graph-case/1'
HF = Path.home() / '.cache/huggingface/hub'
VIEW3 = data_path('data_hdd', 'natlang-development-data/data/neuralese/raw/view-sources-20261010-v3')
SDKB = data_path('archive', 'raw/agentic-20260927')
S1_GROUPS = data_path('data_nvme', 'natlang-corpora/cross-corpus-index/s1-full-final-20261003/groups.json')
MAX_DOCS = 4


def text_sha(text):
    return sha256_hex(text.encode('utf-8'))


def file_binding(path):
    path = Path(path)
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 22), b''):
            digest.update(block)
    return {'path': str(path), 'sha256': digest.hexdigest(), 'bytes': path.stat().st_size}


def rng_for(*parts):
    return random.Random(int(text_sha('\0'.join(map(str, parts)))[:16], 16))


def carve(group, percent=5):
    """Deterministic test carve of a train-only source by its world group."""
    return 'test' if int(text_sha('broad-graph-carve:' + group)[:8], 16) % 100 < percent else 'train'


def wiki_group(title):
    return 'wiki:' + title.strip().replace(' ', '_')


ABBREVIATION = re.compile(r'(?:\b(?:Mr|Mrs|Ms|Dr|St|Jr|Sr|Inc|Co|Ltd|Corp|No|vs|Mt|Ft|Gen|Col|Lt|Sgt|Capt|Rev|Prof|Gov|'
                          r'Sen|Rep|Pres|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|etc|approx|ca|al|U\.S|U\.K|D\.C)'
                          r'|\b[A-Z])\.$')


def sentence_spans(text):
    """Verbatim sentence spans (start, end): split after . ! ? before a capital/digit/quote, never after an
    abbreviation or initial (the musique_hops splitter)."""
    spans, start = [], 0
    for match in re.finditer(r'[.!?]["\')\]]*\s+(?=["\'(\[]?[A-Z0-9])', text):
        end = match.start() + len(match.group(0).rstrip())
        if ABBREVIATION.search(re.sub(r'["\')\]]+$', '', text[start:end])):
            continue
        spans.append((start, end))
        start = match.end()
    if text[start:].strip():
        spans.append((start, len(text.rstrip())))
    out = []
    for a, b in spans:
        while a < b and text[a].isspace():
            a += 1
        if b > a:
            out.append((a, b))
    return out


# ---------------------------------------------------------------- model-facing wording per task kind

KINDS = {
    'evidence': {
        'request': 'question', 'none': 'No relevant facts.',
        'views': [
            ('find_evidence', 'The sentences of a document that help answer a question, word for word.',
             'Return, word for word, the sentences of document that state facts needed to answer question, in the order '
             'document gives them. If document states none of those facts, return exactly "No relevant facts."'),
            ('relevant_sentences', 'Only the sentences of a document that matter for a question, verbatim.',
             'Read document and keep only the sentences that matter for question. Return them exactly as document writes '
             'them, in document order. Return exactly "No relevant facts." when no sentence of document matters for question.')],
        'consumer': ('answer_from_notes', 'The short answer to a question from notes taken on several documents.',
                     'notes holds what was found in each document, one entry per document; "No relevant facts." means that '
                     'document had nothing for question. Answer question using only notes. Return only the answer, as a short phrase.'),
        'root': ('answer_question', 'Return only the answer, as a short phrase.')},
    'passage': {
        'request': 'question', 'none': 'No relevant facts.',
        'views': [
            ('relevant_passage', 'A passage word for word when it helps answer a question.',
             'document is one passage, headed by its title. If the passage states a fact needed to answer question, return '
             'the passage text word for word, without its title line. Otherwise return exactly "No relevant facts."'),
            ('passage_if_relevant', 'The text of a titled passage if a question needs it.',
             'document starts with a title line and then a passage. Return the passage exactly as written (not the title) '
             'when it contains a fact that question needs; return exactly "No relevant facts." when it does not.')],
        'consumer': ('answer_from_notes', 'The short answer to a question from the passages that bear on it.',
                     'notes holds one entry per passage: its text when it bears on question, or "No relevant facts.". '
                     'Answer question using only notes. Return only the answer, as a short phrase.'),
        'root': ('answer_question', 'Return only the answer, as a short phrase.')},
    'meeting': {
        'request': 'query', 'none': 'No relevant turns.',
        'views': [
            ('relevant_turns', 'The turns of a transcript part that a query needs, word for word.',
             'document is part of a meeting transcript, one turn per line as "Speaker: words". Return, word for word, the '
             'turns that are needed to answer query, one per line in transcript order. If no turn of document is needed, '
             'return exactly "No relevant turns."'),
            ('turns_for_query', 'Verbatim transcript turns that bear on a query.',
             'document holds consecutive turns of a meeting, each line "Speaker: words". Copy exactly the turns that '
             'query is about, one per line, keeping their order. Return exactly "No relevant turns." when none are.')],
        'consumer': ('answer_query', 'The answer to a query about a meeting from the turns found for it.',
                     'notes holds the turns found in each part of the transcript; "No relevant turns." means that part had '
                     'none. Answer query using only notes, in a few sentences. Return only the answer.'),
        'root': ('answer_meeting_query', 'Answer in a few sentences and return only the answer.')},
    'news': {
        'request': 'request', 'none': 'No relevant facts.',
        'views': [
            ('key_sentences', 'The sentences of a news article that carry its main facts, word for word.',
             'document is one news article. Return, word for word, at most three sentences of document that carry the '
             'main facts of the story request asks about, in document order. If document has none, return exactly '
             '"No relevant facts."'),
            ('main_facts', 'Up to three verbatim sentences with the main facts of an article.',
             'Read the article in document and copy exactly the (at most three) sentences that hold the facts a report on '
             'the story in request needs, in their order. Return exactly "No relevant facts." if none do.')],
        'consumer': ('write_report', 'A short news report written from the key sentences of several articles.',
                     'notes holds the key sentences of each article about one story. Write the report request asks for, '
                     'using only facts in notes. Return only the report.'),
        'root': ('report_story', 'Write the report request asks for and return only the report.')},
    'table': {
        'request': 'question', 'none': 'No relevant cells.',
        'views': [
            ('find_cells', 'The cells of a table part that a question needs.',
             'document is part of a table: its title, its column headers and some rows. Return the cells of document that '
             'are needed to answer question, one line per row, as "header: value" pairs separated by "; ", each line '
             'starting with the row\'s first filled column. If no cell of document is needed, return exactly "No relevant cells."'),
            ('needed_cells', 'Exactly the table cells a question depends on, row by row.',
             'document holds a few rows of a table under its headers. For every row with a cell that question needs, '
             'write one line "header: value; header: value" that begins with the row\'s first filled column and lists the needed '
             'cells. Return exactly "No relevant cells." if no row has one.')],
        'consumer': ('answer_from_cells', 'A one-sentence answer to a question from cells found in a table.',
                     'notes holds the cells found in each part of the table; "No relevant cells." means that part had none. '
                     'Answer question in one sentence using only notes. Return only that sentence.'),
        'root': ('answer_table_question', 'Answer in one sentence and return only that sentence.')},
    'schema': {
        'request': 'question', 'none': 'Not needed for this question.',
        'views': [
            ('relevant_columns', 'The columns of one database table that a SQL query for a question needs.',
             'document describes one table of a SQLite database: its name, its columns with types and keys, and a few '
             'sample rows. If a SQL query answering question needs this table, return "Table <name>: " followed by the '
             'needed columns as "column (type)", comma-separated, in the order document lists them. Otherwise return '
             'exactly "Not needed for this question."'),
            ('table_needs', 'Whether a SQL query for a question uses this table, and which of its columns.',
             'document is the description of one database table. Decide whether answering question with SQL uses this '
             'table. If it does, return "Table <name>: " and the columns the query uses, each as "column (type)", '
             'separated by commas, in document order. If it does not, return exactly "Not needed for this question."')],
        'consumer': ('write_sql', 'One SQLite query answering a question from notes on the tables it needs.',
                     'notes say, for each table of the database, which columns question needs ("Not needed for this '
                     'question." for a table it does not use). Write one SQLite query that answers question, using only '
                     'those tables and columns. Return only the query.'),
        'root': ('answer_with_sql', 'Then write one SQLite query that answers question, using only the tables and '
                 'columns the results name, and return only the query.')},
    'tools': {
        'request': 'request', 'none': 'Not needed for this request.',
        'views': [
            ('tool_use', 'Whether a request needs one tool, and which of its parameters.',
             'document is the JSON description of one tool. If fulfilling request needs this tool, return "Needed: " '
             'followed by the tool\'s name and the parameters request gives values for, as "name(parameter: type, ...)", '
             'in the order the tool declares them. Otherwise return exactly "Not needed for this request."'),
            ('check_tool', 'The call signature of a tool when a request needs it.',
             'Read the tool description in document. When request needs this tool, return "Needed: name(parameter: type, '
             '...)" with only the parameters request supplies, in declaration order. When it does not, return exactly '
             '"Not needed for this request."')],
        'consumer': ('write_calls', 'The JSON list of tool calls that fulfil a request.',
                     'notes say, for each available tool, whether request needs it and with which parameters. Return only '
                     'the JSON list of calls that fulfil request, as [{"name": ..., "arguments": {...}}], using only the '
                     'tools and parameters in notes.'),
        'root': ('plan_calls', 'Then return only the JSON list of calls that fulfil request, as [{"name": ..., '
                 '"arguments": {...}}], using only the tools and parameters the results name.')},
    'logs': {
        'request': 'request', 'none': 'No log lines in this window.',
        'views': [
            ('event_templates', 'The distinct event templates of a window of log lines.',
             'document is a window of consecutive log lines. Return the distinct event templates of its lines, one per '
             'line, in order of first appearance. An event template is a line\'s message with every variable part '
             '(numbers, identifiers, paths, addresses) replaced by <*>.'),
            ('log_templates', 'The event templates that occur in some log lines, in first-seen order.',
             'For the log lines in document, write each line\'s message as a template, replacing every part that varies '
             'between occurrences (numbers, ids, paths, addresses) with <*>. Return the distinct templates, one per '
             'line, in the order they first occur.')],
        'consumer': ('merge_templates', 'The distinct event templates over several log windows.',
                     'notes holds the event templates found in each log window, in window order. Return the distinct '
                     'templates over all windows, one per line, in order of first appearance.'),
        'root': ('collect_templates', 'Then return the distinct templates over all windows, one per line, in order of '
                 'first appearance.')},
}


def make_case(*, source, dataset, upstream_id, upstream_split, split, groups, kind, request_text, answer, documents,
              license, revision, files, answer_kind='text', oracle='normalized', answer_origin='dataset-answer',
              adaptation, domain, alternates=()):
    spec = KINDS[kind]
    variant = rng_for(source, upstream_id, 'variant').randrange(len(spec['views']))
    view_name, view_description, view_instructions = spec['views'][variant]
    consumer_name, consumer_description, consumer_instructions = spec['consumer']
    root_name, root_answer = spec['root']
    request = spec['request']
    read = (f'The documents for {request} are the files in docs/. Call {view_name} on each document\'s text with '
            f'{request}')
    if kind in ('schema', 'tools', 'logs'):
        root_instructions = f'{read} and print each result. {root_answer}'
    else:
        root_instructions = f'{read} and print each result. Then answer {request} from those results alone. {root_answer}'
    child_instructions = (f'{read}, collect the results in document order, and return what {consumer_name} gives for '
                          f'{request} and those results.')
    case = {
        'schema': SCHEMA, 'id': f'{source}:{upstream_id}', 'source': source, 'dataset': dataset,
        'upstream_id': str(upstream_id), 'upstream_split': upstream_split, 'split': split,
        'groups': sorted(set(groups)), 'task_kind': kind, 'domain': domain, 'variant': f'w{variant}',
        'request': {'name': request, 'text': request_text}, 'answer': answer, 'answer_kind': answer_kind,
        'answer_origin': answer_origin, 'oracle': oracle if not alternates else {'level': oracle, 'alternates': list(alternates)},
        'view': {'name': view_name, 'description': view_description, 'instructions': view_instructions,
                 'none': spec['none']},
        'consumer': {'name': consumer_name, 'description': consumer_description, 'instructions': consumer_instructions},
        'root': {'name': root_name, 'instructions_root': root_instructions, 'instructions_child': child_instructions},
        'documents': documents, 'license': license, 'revision': revision, 'files': files, 'adaptation': adaptation,
    }
    for doc in documents:
        doc.setdefault('view_origin', 'none-line' if doc['view'] == spec['none'] else 'dataset')
        doc['group'] = 'document:' + text_sha(doc['text'])
    case['groups'] = sorted(set(case['groups']) | {doc['group'] for doc in documents})
    return case


def answer_free(others, answer):
    """Distractors that do not state the answer string: supporting-fact annotations are not exhaustive (source
    review, hotpotqa knowledge_evidence holds), so a distractor naming the answer may hold a needed fact."""
    folded = ' '.join(answer.casefold().split())
    if folded in ('yes', 'no') or len(folded) < 2:
        return others
    return [doc for doc in others if folded not in ' '.join(doc['text'].casefold().split())]


def cap_documents(gold, others, rng, limit=MAX_DOCS):
    """Every gold document plus distractors up to ``limit`` documents (all gold kept even above it)."""
    room = max(0, limit - len(gold))
    return gold + rng.sample(others, min(room, len(others)))


# ---------------------------------------------------------------- multi-hop QA with supporting facts

def _parquet_rows(paths):
    import pyarrow.parquet as pq
    for path in paths:
        table = pq.read_table(path)
        for batch in table.to_batches(4096):
            yield from batch.to_pylist()


def _maybe_json(value):
    if isinstance(value, str):
        try:
            import ast
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return json.loads(value)
    return value


def supporting_fact_cases(source, dataset, files_by_split, license, revision, holds, domain='wiki-multihop'):
    for upstream_split, paths in files_by_split.items():
        split = 'train' if upstream_split == 'train' else 'test'
        files = [file_binding(path) for path in paths]
        for row in _parquet_rows(paths):
            if row['id'] in holds:
                yield None, 'source-review-hold'; continue
            context = _maybe_json(row['context']); facts = _maybe_json(row['supporting_facts'])
            titles, sentences = context['title'], context['sentences']
            wanted = collections.defaultdict(set)
            for title, sent_id in zip(facts['title'], facts['sent_id']):
                wanted[title].add(int(sent_id))
            gold, others, problem = [], [], None
            for title, sents in zip(titles, sentences):
                sents = [s.strip() for s in sents if s.strip()]
                text = f'# {title}\n' + ' '.join(sents)
                if title in wanted:
                    if any(i >= len(sents) for i in wanted[title]):
                        problem = 'supporting-sentence-out-of-range'; break
                    view = ' '.join(sents[i] for i in sorted(wanted[title]))
                    gold.append({'title': title, 'text': text, 'view': view, 'view_origin': 'dataset-supporting-facts'})
                else:
                    others.append({'title': title, 'text': text, 'view': 'No relevant facts.'})
            if problem:
                yield None, problem; continue
            if len(gold) != len(wanted) or not gold:
                yield None, 'supporting-title-missing'; continue
            if len(gold) > MAX_DOCS:
                yield None, 'too-many-supporting-documents'; continue
            if sum(len(d['text']) for d in gold) > 12000:
                yield None, 'documents-over-cap'; continue
            docs = cap_documents(gold, answer_free(others, row['answer']), rng_for(source, row['id'], 'docs'))
            groups = [f'{source}-q:{row["id"]}'] + [wiki_group(d['title']) for d in gold]
            if source == 'hotpotqa':
                groups.append(f'hotpotqa-q:{row["id"]}')
            yield make_case(source=source, dataset=dataset, upstream_id=row['id'], upstream_split=upstream_split,
                            split=split, groups=groups, kind='evidence', request_text=row['question'],
                            answer=row['answer'], documents=docs, license=license, revision=revision, files=files,
                            adaptation='supporting_facts sentences per paragraph as producer views; distractor '
                                       'paragraphs answer the none line; dataset answer as consumer target',
                            domain=domain), None


def hotpotqa(holds):
    base = glob.glob(str(HF / 'datasets--hotpotqa--hotpot_qa/snapshots/1908d6afbbead072334abe2965f91bd2709910ab/distractor'))[0]
    files = {'train': sorted(glob.glob(base + '/train-*.parquet')), 'validation': sorted(glob.glob(base + '/validation-*.parquet'))}
    yield from supporting_fact_cases('hotpotqa', 'hotpotqa/hotpot_qa distractor', files, {'spdx': 'CC-BY-SA-4.0'},
                                     'hotpotqa@1908d6afbbead072334abe2965f91bd2709910ab', holds.get('hotpotqa', set()))


def twowiki(holds):
    base = glob.glob(str(HF / 'datasets--framolfese--2WikiMultihopQA/snapshots/*/data'))[0]
    files = {'train': sorted(glob.glob(base + '/train-*.parquet')), 'validation': sorted(glob.glob(base + '/validation-*.parquet'))}
    revision = '2wikimultihopqa@' + base.split('/snapshots/')[1].split('/')[0]
    yield from supporting_fact_cases('2wiki', 'framolfese/2WikiMultihopQA', files, {'spdx': 'Apache-2.0'}, revision,
                                     holds.get('2wikimultihopqa', set()))


def musique(holds):
    base = glob.glob(str(HF / 'datasets--dgslibisey--MuSiQue/snapshots/*'))[0]
    revision = 'musique@' + base.rsplit('/', 1)[1]
    train_path, dev_path = base + '/musique_ans_v1.0_train.jsonl', base + '/musique_ans_v1.0_dev.jsonl'
    dev = [json.loads(line) for line in open(dev_path)]
    held_seeds = {str(step['id']) for row in dev for step in row.get('question_decomposition') or []}
    hold_ids = holds.get('musique', set())
    for upstream_split, path, rows in (('train', train_path, None), ('dev', dev_path, dev)):
        split = 'train' if upstream_split == 'train' else 'test'
        files = [file_binding(path)]
        for row in rows if rows is not None else (json.loads(line) for line in open(path)):
            if not row.get('answerable'):
                yield None, 'unanswerable'; continue
            if row['id'] in hold_ids:
                yield None, 'source-review-hold'; continue
            seeds = [str(step['id']) for step in row.get('question_decomposition') or []]
            if split == 'train' and set(seeds) & held_seeds:
                yield None, 'held-out-musique-seed'; continue
            gold = [{'title': p['title'], 'text': f'# {p["title"]}\n{p["paragraph_text"].strip()}',
                     'view': p['paragraph_text'].strip(), 'view_origin': 'dataset-supporting-paragraph'}
                    for p in row['paragraphs'] if p['is_supporting']]
            others = [{'title': p['title'], 'text': f'# {p["title"]}\n{p["paragraph_text"].strip()}',
                       'view': 'No relevant facts.'} for p in row['paragraphs'] if not p['is_supporting']]
            if not gold or len(gold) > MAX_DOCS:
                yield None, 'supporting-paragraph-count'; continue
            if any(len(d['view']) > 1500 for d in gold):
                yield None, 'supporting-paragraph-too-long'; continue
            docs = cap_documents(gold, answer_free(others, row['answer']), rng_for('musique', row['id'], 'docs'))
            groups = [f'musique:{row["id"]}'] + [f'musique:seed:{s}' for s in seeds]
            yield make_case(source='musique', dataset='dgslibisey/MuSiQue musique_ans_v1.0', upstream_id=row['id'],
                            upstream_split=upstream_split, split=split, groups=groups, kind='passage',
                            request_text=row['question'], answer=row['answer'], documents=docs,
                            license={'spdx': 'CC-BY-4.0'}, revision=revision, files=files,
                            alternates=row.get('answer_aliases') or (),
                            adaptation='supporting paragraphs verbatim as producer views (paragraph-level dataset flags; '
                                       'the held musique_hops decomposition/sentence selection is not used)',
                            domain='wiki-multihop'), None


# ---------------------------------------------------------------- SQuAD: same-article paragraphs

def squad(holds):
    path = glob.glob(str(HF / 'datasets--rajpurkar--squad/snapshots/*/plain_text/train-00000-of-00001.parquet'))[0]
    revision = 'squad@' + path.split('/snapshots/')[1].split('/')[0]
    files = [file_binding(path)]
    by_title = collections.defaultdict(list)
    for row in _parquet_rows([path]):
        by_title[row['title']].append(row)
    for title, rows in by_title.items():
        paragraphs = list(dict.fromkeys(r['context'] for r in rows))
        per_paragraph = collections.defaultdict(list)
        for r in rows:
            per_paragraph[r['context']].append(r)
        for context, questions in per_paragraph.items():
            row = rng_for('squad', title, text_sha(context)).choice(questions)
            answer, start = row['answers']['text'][0], row['answers']['answer_start'][0]
            end = start + len(answer)
            if context[start:end] != answer:
                yield None, 'answer-span-mismatch'; continue
            spans = [(a, b) for a, b in sentence_spans(context) if a < end and b > start]
            if not spans:
                yield None, 'answer-not-in-a-sentence'; continue
            view = context[spans[0][0]:spans[-1][1]]
            if len(view) < 16 or len(view) > 800:
                yield None, 'answer-sentence-length'; continue
            folded = answer.casefold()
            others = [p for p in paragraphs if p != context and folded not in p.casefold()]
            name = title.replace('_', ' ')
            gold = [{'title': name, 'text': f'# {name}\n{context}', 'view': view,
                     'view_origin': 'dataset-answer-span-sentence'}]
            distract = [{'title': name, 'text': f'# {name}\n{p}', 'view': 'No relevant facts.'} for p in others]
            docs = cap_documents(gold, distract, rng_for('squad', row['id'], 'docs'), limit=3)
            group = wiki_group(title)
            yield make_case(source='squad', dataset='rajpurkar/squad v1.1 train', upstream_id=row['id'],
                            upstream_split='train', split=carve(group), groups=[group, f'squad-q:{row["id"]}'],
                            kind='evidence', request_text=row['question'], answer=answer, documents=docs,
                            license={'spdx': 'CC-BY-SA-4.0'}, revision=revision, files=files,
                            adaptation='the sentence(s) holding the answer span as the gold paragraph view; other '
                                       'paragraphs of the same article without the answer string as distractors',
                            domain='wiki-qa'), None


# ---------------------------------------------------------------- QMSum: relevant turns of transcript parts

def qmsum(holds):
    base = VIEW3 / 'qmsum/data/ALL/jsonl'
    for upstream_split in ('train', 'val', 'test'):
        path = base / f'{upstream_split}.jsonl'
        files = [file_binding(path)]
        split = 'train' if upstream_split == 'train' else 'test'
        for meeting_index, line in enumerate(open(path)):
            meeting = json.loads(line)
            turns = [f'{t["speaker"]}: {" ".join(t["content"].split())}' for t in meeting['meeting_transcripts']]
            meeting_id = text_sha('\n'.join(turns[:5]))[:20]
            segments, current, size = [], [], 0
            for index, turn in enumerate(turns):
                if current and size + len(turn) + 1 > 4000:
                    segments.append(current); current, size = [], 0
                current.append(index); size += len(turn) + 1
            if current:
                segments.append(current)
            for q_index, query in enumerate(meeting['specific_query_list']):
                relevant = set()
                for a, b in query['relevant_text_span']:
                    relevant.update(range(int(a), int(b) + 1))
                hit = [i for i, seg in enumerate(segments) if relevant & set(seg)]
                if not hit or len(hit) > 3:
                    yield None, 'relevant-span-segments'; continue
                if any(len(turns[i]) > 4000 for i in relevant if i < len(turns)):
                    yield None, 'turn-over-cap'; continue
                rng = rng_for('qmsum', meeting_id, q_index)
                pool = [i for i in range(len(segments)) if i not in hit]
                chosen = sorted(hit + rng.sample(pool, min(len(pool), MAX_DOCS - len(hit))))
                docs, too_long = [], False
                for i in chosen:
                    seg = segments[i]
                    view_turns = [turns[j] for j in seg if j in relevant]
                    view = '\n'.join(view_turns) if view_turns else 'No relevant turns.'
                    if len(view) > 3000:
                        too_long = True
                    docs.append({'title': f'part {i + 1}', 'text': '\n'.join(turns[j] for j in seg), 'view': view,
                                 'view_origin': 'dataset-relevant-text-span' if view_turns else 'none-line'})
                if too_long:
                    yield None, 'relevant-turns-over-view-cap'; continue
                yield make_case(source='qmsum', dataset='Yale-LILY/QMSum specific queries',
                                upstream_id=f'{meeting_id}:{q_index}', upstream_split=upstream_split, split=split,
                                groups=[f'qmsum-meeting:{meeting_id}'], kind='meeting', request_text=query['query'],
                                answer=' '.join(query['answer'].split()), documents=docs,
                                license={'spdx': 'MIT', 'notes': 'AMI/ICSI transcripts CC-BY-4.0; parliamentary records open government licences'},
                                revision='qmsum@view-sources-20261010-v3', files=files, answer_kind='summary',
                                answer_origin='dataset-reference-answer',
                                adaptation='transcript cut into consecutive-turn parts of at most 4,000 characters; '
                                           'relevant_text_span turns as each part\'s view', domain='meeting'), None


# ---------------------------------------------------------------- FeTaQA: highlighted cells per row block

def fetaqa(holds):
    for upstream_split in ('train', 'dev', 'test'):
        path = VIEW3 / f'fetaqa/data/fetaQA-v1_{upstream_split}.jsonl'
        files = [file_binding(path)]
        split = 'train' if upstream_split == 'train' else 'test'
        for line in open(path):
            row = json.loads(line)
            table = row['table_array']
            header, body = table[0], table[1:]
            cells = collections.defaultdict(set)
            for r, c in row['highlighted_cell_ids']:
                cells[int(r)].add(int(c))
            if 0 in cells or not cells or any(r > len(body) for r in cells):
                yield None, 'highlight-outside-body'; continue
            blocks = [list(range(i + 1, min(i + 6, len(body) + 1))) for i in range(0, len(body), 5)]
            hit = [b for b, rows in enumerate(blocks) if cells.keys() & set(rows)]
            if len(hit) > MAX_DOCS:
                yield None, 'too-many-highlighted-blocks'; continue
            rng = rng_for('fetaqa', row['feta_id'])
            pool = [b for b in range(len(blocks)) if b not in hit]
            chosen = sorted(hit + rng.sample(pool, min(len(pool), MAX_DOCS - len(hit))))
            title = f'{row["table_page_title"]} — {row["table_section_title"]}'
            docs = []
            for b in chosen:
                rows = blocks[b]
                text = '\n'.join([f'Table: {title}', ' | '.join(header)] + [' | '.join(table[r]) for r in rows])
                lines = []
                for r in rows:
                    if r in cells:
                        key = next((c for c, value in enumerate(table[r]) if value.strip() not in ('', '-', '—', 'n/a')), 0)
                        cols = [key] + sorted(cells[r] - {key})
                        lines.append('; '.join(f'{header[c]}: {table[r][c]}' for c in cols))
                view = '\n'.join(lines) if lines else 'No relevant cells.'
                docs.append({'title': f'rows {rows[0]}-{rows[-1]}', 'text': text, 'view': view,
                             'view_origin': 'dataset-highlighted-cells' if lines else 'none-line'})
            if any(len(d['view']) < 16 for d in docs):
                yield None, 'view-under-edge-minimum'; continue
            page = row['page_wikipedia_url'].rsplit('/', 1)[-1]
            yield make_case(source='fetaqa', dataset='Yale-LILY/FeTaQA', upstream_id=row['feta_id'],
                            upstream_split=upstream_split, split=split,
                            groups=[f'fetaqa-table:{row["table_source_json"]}', wiki_group(page)], kind='table',
                            request_text=row['question'], answer=row['answer'], documents=docs,
                            license={'spdx': 'CC-BY-SA-4.0'}, revision='fetaqa@view-sources-20261010-v3', files=files,
                            answer_kind='sentence', answer_origin='dataset-free-form-answer',
                            adaptation='table cut into blocks of 5 rows under the header; highlighted_cell_ids as '
                                       'each block\'s view (row key = first filled column)', domain='table'), None


# ---------------------------------------------------------------- Multi-News: greedy ROUGE extractive oracle

def _ngrams(words, n):
    return collections.Counter(tuple(words[i:i + n]) for i in range(len(words) - n + 1))


def _rouge_f(candidate, reference, n):
    c, r = _ngrams(candidate, n), _ngrams(reference, n)
    overlap = sum((c & r).values())
    if not overlap:
        return 0.0
    p, q = overlap / max(1, sum(c.values())), overlap / max(1, sum(r.values()))
    return 2 * p * q / (p + q)


def greedy_oracle(sentences, summary, limit=3):
    words = lambda text: re.findall(r'\w+', text.lower())
    reference = words(summary)
    chosen, best = [], 0.0
    while len(chosen) < limit:
        step = None
        for i, s in enumerate(sentences):
            if i in chosen or len(words(s)) < 5:
                continue
            cand = words(' '.join(sentences[j] for j in sorted(chosen + [i])))
            score = _rouge_f(cand, reference, 1) + _rouge_f(cand, reference, 2)
            if score > best + 1e-9 and (step is None or score > step[0]):
                step = (score, i)
        if step is None:
            break
        best = step[0]; chosen.append(step[1])
    return sorted(chosen)


def multi_news(holds):
    base = data_path('data_hdd', 'bgkit-data/multi_news_v1/raw/data')
    for upstream_split in ('train', 'val', 'test'):
        src, tgt = base / f'{upstream_split}.src.cleaned', base / f'{upstream_split}.tgt'
        files = [file_binding(src), file_binding(tgt)]
        split = 'train' if upstream_split == 'train' else 'test'
        for index, (source_line, summary) in enumerate(zip(open(src), open(tgt))):
            if upstream_split == 'train' and int(text_sha(f'multi_news-select:{index}')[:8], 16) % 5:
                continue  # deterministic 1/5 pre-selection: the train cap (4,000) is far below 45k stories
            summary = ' '.join(summary.replace('NEWLINE_CHAR', ' ').split()).lstrip('– ').strip()
            articles = [' '.join(a.replace('NEWLINE_CHAR', ' ').split())
                        for a in re.split(r'\|\|\|\|\||story_separator_special_tag', source_line)]
            articles = [a for a in dict.fromkeys(articles) if len(a) >= 400]
            articles = [a for a in articles if len(a) <= 6000]
            if len(articles) < 2 or not summary or len(summary) > 2500:
                yield None, 'articles-or-summary-out-of-range'; continue
            articles = articles[:MAX_DOCS]
            docs = []
            for k, article in enumerate(articles):
                spans = sentence_spans(article)
                sentences = [article[a:b] for a, b in spans]
                picked = greedy_oracle(sentences, summary)
                view = ' '.join(sentences[i] for i in picked) if picked else 'No relevant facts.'
                docs.append({'title': f'article {k + 1}', 'text': article, 'view': view,
                             'view_origin': 'code-computed:greedy-rouge-oracle' if picked else 'none-line'})
            if not any(d['view_origin'] != 'none-line' for d in docs):
                yield None, 'no-oracle-sentences'; continue
            upstream_id = f'{upstream_split}:{index}'
            yield make_case(source='multi_news', dataset='alexfabbri/multi_news', upstream_id=upstream_id,
                            upstream_split=upstream_split, split=split, groups=[f'multi_news:{text_sha(source_line)[:24]}'],
                            kind='news', request_text='Write a short news report of the story these articles cover.',
                            answer=summary, documents=docs, license={'spdx': 'LicenseRef-multi-news-research',
                            'notes': 'card: other, non-commercial research terms; articles from newser.com sources'},
                            revision='multi_news@38cb2069', files=files, answer_kind='summary',
                            answer_origin='dataset-reference-summary',
                            adaptation='articles of 400-6,000 characters (at most 4); per article the greedy ROUGE-1+2 '
                                       'extractive oracle (at most 3 sentences) against the reference summary as view',
                            domain='news'), None


# ---------------------------------------------------------------- Spider / BIRD: schema views

def _sql_identifiers(sql):
    sql = re.sub(r"'(?:[^']|'')*'", "''", sql)
    found = set()
    for m in re.finditer(r'`([^`]+)`|"([^"]+)"|\[([^\]]+)\]|([A-Za-z_][A-Za-z0-9_]*)', sql):
        found.add(next(g for g in m.groups() if g is not None).lower())
    return found


def _schema_docs(schema, database_path, sample_rows=3):
    tables = schema['table_names_original']
    columns = collections.defaultdict(list)
    for index, ((table, name), kind) in enumerate(zip(schema['column_names_original'], schema['column_types'])):
        if table >= 0:
            columns[table].append((index, name, kind))
    primary = set()
    for key in schema.get('primary_keys') or []:
        primary.update(key if isinstance(key, list) else [key])
    foreign = collections.defaultdict(list)
    names = schema['column_names_original']
    for a, b in schema.get('foreign_keys') or []:
        foreign[a].append(f'{tables[names[b][0]]}.{names[b][1]}')
    connection = None
    if database_path and Path(database_path).is_file():
        connection = sqlite3.connect(f'file:{database_path}?mode=ro', uri=True)
        connection.text_factory = lambda b: b.decode('utf-8', 'replace')
    docs = {}
    for t, table in enumerate(tables):
        lines = [f'Table: {table}', 'Columns:']
        for index, name, kind in columns[t]:
            notes = [kind] + (['primary key'] if index in primary else []) + [f'references {x}' for x in foreign[index]]
            lines.append(f'- {name} ({", ".join(notes)})')
        if connection is not None:
            try:
                rows = connection.execute(f'SELECT * FROM "{table}" LIMIT {sample_rows}').fetchall()
            except sqlite3.Error:
                rows = []
            if rows:
                lines.append('Sample rows:')
                lines.append(' | '.join(name for _, name, _ in columns[t]))
                for row in rows:
                    lines.append(' | '.join(str(v)[:40] for v in row))
        docs[t] = {'table': table, 'columns': [(name, kind) for _, name, kind in columns[t]], 'text': '\n'.join(lines)}
    if connection is not None:
        connection.close()
    return docs


def sql_cases(source, rows, schemas, database_root, files, upstream_split, split_of, revision, license):
    cache = {}
    for index, row in enumerate(rows):
        db = row['db_id']
        if db not in schemas:
            yield None, 'schema-missing'; continue
        if db not in cache:
            cache.clear()
            cache[db] = _schema_docs(schemas[db], database_root / db / f'{db}.sqlite' if database_root else None)
        tables = cache[db]
        sql = row.get('SQL') or row.get('query')
        identifiers = _sql_identifiers(sql)
        gold, others = [], []
        for t, info in tables.items():
            if info['table'].lower() in identifiers:
                used = [f'{name} ({kind})' for name, kind in info['columns'] if name.lower() in identifiers]
                view = f'Table {info["table"]}: ' + (', '.join(used) if used else 'no specific columns')
                gold.append({'title': info['table'], 'text': info['text'], 'view': view,
                             'view_origin': 'code-computed:sql-identifier-reference'})
            else:
                others.append({'title': info['table'], 'text': info['text'], 'view': 'Not needed for this question.'})
        if not gold:
            yield None, 'no-referenced-table'; continue
        if len(gold) > MAX_DOCS + 1:
            yield None, 'too-many-referenced-tables'; continue
        if sum(len(d['text']) for d in gold) > 16000:
            yield None, 'schema-over-cap'; continue
        docs = cap_documents(gold, others, rng_for(source, db, index, 'docs'), limit=MAX_DOCS + 1)
        question = row['question'] + (f' (Hint: {row["evidence"]})' if row.get('evidence') else '')
        group = f'sql-db:{source}:{db}'
        yield make_case(source=source, dataset=revision.split('@')[0], upstream_id=f'{upstream_split}:{index}',
                        upstream_split=upstream_split, split=split_of(group), groups=[group], kind='schema',
                        request_text=question, answer=' '.join(sql.split()), documents=docs, license=license,
                        revision=revision, files=files, answer_kind='sql', oracle='exact',
                        answer_origin='dataset-gold-sql',
                        adaptation='one document per table (columns, types, keys, 3 sample rows read-only); view = '
                                   'the table and its columns referenced as identifiers by the gold SQL', domain='sql'), None


def spider(holds):
    base = SDKB / 'spider/official/spider_data'
    schemas = {s['db_id']: s for s in json.load(open(base / 'tables.json'))}
    for name in ('train_spider.json', 'train_others.json'):
        path = base / name
        rows = json.load(open(path))
        yield from sql_cases('spider', rows, schemas, base / 'database', [file_binding(path), file_binding(base / 'tables.json')],
                             name.removesuffix('.json'), lambda group: 'train', 'spider@official', {'spdx': 'CC-BY-SA-4.0'})
    path = base / 'dev.json'
    yield from sql_cases('spider', json.load(open(path)), schemas, base / 'database',
                         [file_binding(path), file_binding(base / 'tables.json')], 'dev', lambda group: 'test',
                         'spider@official', {'spdx': 'CC-BY-SA-4.0'})


def bird(holds):
    base = SDKB / 'bird/train'
    schemas = {s['db_id']: s for s in json.load(open(base / 'train_tables.json'))}
    path = base / 'train.json'
    database_root = base / 'train_databases'
    yield from sql_cases('bird', json.load(open(path)), schemas, database_root if database_root.is_dir() else None,
                         [file_binding(path), file_binding(base / 'train_tables.json')], 'train', lambda group: 'train',
                         'bird@train', {'spdx': 'CC-BY-SA-4.0'})


# ---------------------------------------------------------------- xLAM: tool-schema views

def xlam(holds):
    path = SDKB / 'xlam-function-calling-60k/xlam_function_calling_60k.json'
    files = [file_binding(path)]
    for row in json.load(open(path)):
        tools, answers = json.loads(row['tools']), json.loads(row['answers'])
        names = [t['name'] for t in tools]
        if len(tools) < 2 or len(set(names)) != len(names):
            yield None, 'fewer-than-two-distinct-tools'; continue
        if len(tools) > MAX_DOCS + 1:
            yield None, 'too-many-tools'; continue
        used = collections.defaultdict(set)
        for call in answers:
            used[call['name']].update((call.get('arguments') or {}).keys())
        if not set(used) <= set(names):
            yield None, 'call-to-undeclared-tool'; continue
        docs = []
        for tool in tools:
            params = tool.get('parameters') or {}
            if tool['name'] in used:
                listed = [f'{p}: {params[p].get("type", "any") if isinstance(params[p], dict) else "any"}'
                          for p in params if p in used[tool['name']]]
                if set(used[tool['name']]) - set(params):
                    docs = None; break
                view, origin = f'Needed: {tool["name"]}({", ".join(listed)})', 'code-computed:gold-call-reference'
            else:
                view, origin = 'Not needed for this request.', 'none-line'
            docs.append({'title': tool['name'], 'text': json.dumps(tool, ensure_ascii=False, indent=1), 'view': view,
                         'view_origin': origin})
        if docs is None:
            yield None, 'argument-not-declared'; continue
        group = 'tool-schema:' + '+'.join(sorted(names))
        yield make_case(source='xlam', dataset='Salesforce/xlam-function-calling-60k', upstream_id=row['id'],
                        upstream_split='train', split=carve(group), groups=[group, f'xlam-tools:{group[12:]}'],
                        kind='tools', request_text=row['query'], answer=json.dumps(answers, ensure_ascii=False),
                        documents=docs, license={'spdx': 'CC-BY-4.0'}, revision='xlam-function-calling-60k@sdkb-20260927',
                        files=files, answer_kind='calls', oracle='exact', answer_origin='dataset-gold-calls',
                        adaptation='one document per tool schema; view = the tool and the parameters the gold calls '
                                   'pass, or the none line', domain='tool-use'), None


# ---------------------------------------------------------------- LogHub: event templates per window

def loghub(holds, cases_per_system=120):
    base = VIEW3 / 'loghub'
    systems = sorted(p.name for p in base.iterdir() if p.is_dir())
    held = sorted(systems, key=lambda s: text_sha('loghub-split:' + s))[:3]
    for system in systems:
        log, structured = base / system / f'{system}_2k.log', base / system / f'{system}_2k.log_structured.csv'
        if not log.is_file() or not structured.is_file():
            yield None, 'system-files-missing'; continue
        raw = log.read_text(encoding='utf-8', errors='replace').splitlines()
        rows = list(csv.DictReader(open(structured, encoding='utf-8', errors='replace')))
        if len(rows) != len(raw):
            yield None, 'structured-line-count-mismatch'; continue
        files = [file_binding(log), file_binding(structured)]
        split = 'test' if system in held else 'train'
        rng = rng_for('loghub', system)
        for case in range(cases_per_system):
            windows = rng.randint(2, MAX_DOCS)
            sizes = [rng.choice((1200, 1800, 2400)) for _ in range(windows)]
            start = rng.randrange(0, max(1, len(raw) - 200))
            docs, cursor = [], start
            for size in sizes:
                lines, used = [], 0
                while cursor < len(raw) and used + len(raw[cursor]) + 1 <= size:
                    lines.append(cursor); used += len(raw[cursor]) + 1; cursor += 1
                if not lines:
                    break
                templates = list(dict.fromkeys(rows[i]['EventTemplate'] for i in lines))
                docs.append({'title': f'lines {lines[0] + 1}-{lines[-1] + 1}', 'text': '\n'.join(raw[i] for i in lines),
                             'view': '\n'.join(templates), 'view_origin': 'dataset-event-templates'})
            if len(docs) < 2:
                yield None, 'window-too-short'; continue
            answer = '\n'.join(dict.fromkeys(t for d in docs for t in d['view'].split('\n')))
            yield make_case(source='loghub', dataset='logpai/loghub 2k', upstream_id=f'{system}:{start}:{case}',
                            upstream_split='all', split=split, groups=[f'loghub-system:{system}'], kind='logs',
                            request_text=f'List the distinct event templates in these {system} log windows, in order '
                                         f'of first appearance.', answer=answer, documents=docs,
                            license={'spdx': 'NOASSERTION', 'notes': 'no licence file; for research'},
                            revision='loghub@dd61d095', files=files, answer_kind='list', oracle='exact',
                            answer_origin='dataset-event-templates',
                            adaptation='consecutive windows of 1,200-2,400 characters from a seeded start; the '
                                       'human-labelled EventTemplate of each line; whole systems held to test',
                            domain='logs'), None


ADAPTERS = {'hotpotqa': hotpotqa, '2wiki': twowiki, 'musique': musique, 'squad': squad, 'qmsum': qmsum,
            'fetaqa': fetaqa, 'multi_news': multi_news, 'spider': spider, 'bird': bird, 'xlam': xlam, 'loghub': loghub}


def source_review_holds():
    holds = collections.defaultdict(set)
    for line in open(ROOT / 'training/source-reviews/holds.jsonl'):
        if line.strip():
            row = json.loads(line)
            if row.get('status') == 'pending':
                holds[row['dataset']].update([row['id'], *(row.get('aliases') or [])])
    return holds


def close(cases, test_cap):
    """Hold at most ``test_cap`` test cases per source; drop every train case sharing a group with a kept test case.

    Transitive closure over shared Wikipedia paragraphs, tool schemas and databases merges most of a source into one
    component, so it is not used: the held graphs are what must stay unrelated to training. Upstream test cases beyond
    the cap are not written (they never train). Returns (kept cases, dropped-train count, dropped-test count)."""
    tests = sorted((c for c in cases if c['split'] == 'test'), key=lambda c: text_sha('broad-graph-select:' + c['id']))
    per, kept_tests = collections.Counter(), []
    for case in tests:
        per[case['source']] += 1
        if per[case['source']] <= test_cap:
            kept_tests.append(case)
    held = {group for case in kept_tests for group in case['groups']}
    train = [c for c in cases if c['split'] == 'train']
    kept_train = [c for c in train if not held & set(c['groups'])]
    return kept_train + kept_tests, len(train) - len(kept_train), len(tests) - len(kept_tests)


def cmd_build(args):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    holds = source_review_holds()
    s1 = json.load(open(S1_GROUPS))
    sources = args.sources.split(',') if args.sources else list(ADAPTERS)
    selected, report, train_caps = [], {}, {}
    for source in sources:
        omitted, candidates = collections.Counter(), {'train': [], 'test': []}
        for case, reason in ADAPTERS[source](holds):
            if case is None:
                omitted[reason] += 1; continue
            s1_splits = {s1[g] for g in case['groups'] if g in s1}
            if s1_splits - {'train'}:
                if case['split'] == 'train':
                    case['split'] = 'test'; case['split_reason'] = 's1-group-held'
            elif s1_splits and case['split'] == 'test':
                omitted['s1-has-upstream-test-world-in-train'] += 1; continue
            candidates[case['split']].append(case)
        caps = {'train': args.train_cap.get(source, args.default_train_cap), 'test': args.test_cap}
        train_caps[source] = caps['train']
        for split, items in candidates.items():
            items.sort(key=lambda c: text_sha('broad-graph-select:' + c['id']))
            # train: 30% headroom over the cap for cases dropped as related to a held case; trimmed after closure
            cap = int(caps[split] * 1.3) if split == 'train' else len(items)
            if len(items) > cap:
                omitted[f'{split}-cap'] += len(items) - cap
            selected.extend(items[:cap])
        report[source] = {'candidates': {k: len(v) for k, v in candidates.items()}, 'omitted': dict(omitted)}
        print(json.dumps({'source': source, **report[source]}), flush=True)
    kept, dropped_train, dropped_test = close(selected, args.test_cap)
    dropped_by = collections.Counter(c['source'] for c in selected if c['split'] == 'train') - \
        collections.Counter(c['source'] for c in kept if c['split'] == 'train')
    for source, n in dropped_by.items():
        report[source]['omitted']['train-related-to-held-case'] = n
    final = collections.Counter((c['source'], c['split']) for c in selected)
    trimmed, per = [], collections.Counter()
    for case in sorted(kept, key=lambda c: text_sha('broad-graph-select:' + c['id'])):
        if case['split'] == 'train':
            per[case['source']] += 1
            if per[case['source']] > train_caps[case['source']]:
                report[case['source']]['omitted']['train-cap-after-held-relation'] = \
                    report[case['source']]['omitted'].get('train-cap-after-held-relation', 0) + 1
                continue
        trimmed.append(case)
    kept = trimmed
    kept.sort(key=lambda c: (c['source'], text_sha('broad-graph-order:' + c['id'])))
    with open(out / 'cases.jsonl', 'w') as stream:
        for case in kept:
            stream.write(json.dumps(case, ensure_ascii=False, separators=(',', ':')) + '\n')
    summary = {'schema': 'natlang.broad-graph-cases-build/1', 'sources': report,
               'held_relation': {'dropped_train': dropped_train, 'unwritten_test_over_cap': dropped_test},
               'after_closure': {f'{s}:{k}': v for (s, k), v in sorted(final.items())},
               'written': dict(collections.Counter(f'{c["source"]}:{c["split"]}' for c in kept)),
               'cases': len(kept), 'builder': file_binding(__file__),
               's1_group_index': {'path': str(S1_GROUPS), 'entries': len(s1)}}
    (out / 'cases-build.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps({k: summary[k] for k in ('held_relation', 'cases', 'written')}))


_PROTECTED = None


def _custody_shard(task):
    """Check one IR shard; writes the kept shard, its omissions and a done marker bound to the protection digest."""
    global _PROTECTED
    from natlang_neuralese.common.custody import CustodyError, ProtectedWorlds
    if _PROTECTED is None:
        _PROTECTED = ProtectedWorlds.load()
    shard, target = map(Path, task)
    marker = target.with_suffix('.custody.json')
    if marker.exists() and json.loads(marker.read_text())['protected_sha256'] == _PROTECTED.digest:
        return json.loads(marker.read_text())['counts']
    counts, kept, omissions = collections.Counter(), [], []
    for line in open(shard):
        record = json.loads(line)
        if record['split'] == 'train':
            try:
                _PROTECTED.check(record)
            except CustodyError as error:
                omissions.append({'id': record['id'], 'source_ids': record['source_ids'], 'reason': str(error)})
                counts[shard.parent.name + ':custody-omitted'] += 1
                continue
        kept.append(line if line.endswith('\n') else line + '\n')
        counts[shard.parent.name + ':' + record['split']] += 1
    target.parent.mkdir(parents=True, exist_ok=True)
    target.with_suffix('.tmp').write_text(''.join(kept))
    os.replace(target.with_suffix('.tmp'), target)
    target.with_suffix('.omissions.tmp').write_text(''.join(json.dumps(row) + '\n' for row in omissions))
    os.replace(target.with_suffix('.omissions.tmp'), target.with_suffix('.custody-omissions.jsonl'))
    marker.write_text(json.dumps({'protected_sha256': _PROTECTED.digest, 'counts': dict(counts)}))
    return dict(counts)


def cmd_custody(args):
    """ProtectedWorlds.check on every train IR world; related worlds are omitted (never moved), test kept.
    Resumable per shard (done markers bound to the protection digest) and parallel over shards."""
    from concurrent.futures import ProcessPoolExecutor
    sys.path.insert(0, str(ROOT / 'training' / 'neuralese'))
    from natlang_neuralese.common.custody import ProtectedWorlds
    protected = ProtectedWorlds.load()
    ir, out = Path(args.ir), Path(args.out)
    tasks = [(str(shard), str(out / shard.parent.name / shard.name)) for shard in sorted(ir.glob('*/*.ir.jsonl'))]
    counts = collections.Counter()
    with ProcessPoolExecutor(args.workers) as pool:
        for done, shard_counts in enumerate(pool.map(_custody_shard, tasks), 1):
            counts.update(shard_counts)
            if done % 50 == 0:
                print(f'custody {done}/{len(tasks)} shards', flush=True)
    for source in sorted({Path(t).parent for _, t in tasks}):
        with open(source / 'custody-omissions.jsonl', 'w') as stream:
            for part in sorted(source.glob('*.custody-omissions.jsonl')):
                stream.write(part.read_text())
    receipt = {'schema': 'natlang.broad-graph-custody/1', 'protected_sha256': protected.digest,
               'protected_sources': [json.loads(s) for s in protected.sources], 'counts': dict(counts)}
    (out / 'custody.json').write_text(json.dumps(receipt, indent=1) + '\n')
    print(json.dumps(receipt['counts']))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build')
    b.add_argument('--out', required=True)
    b.add_argument('--sources')
    b.add_argument('--default-train-cap', type=int, default=6000)
    b.add_argument('--train-cap', type=json.loads, default={}, help='JSON {source: cap}')
    b.add_argument('--test-cap', type=int, default=300)
    c = sub.add_parser('custody')
    c.add_argument('--ir', required=True)
    c.add_argument('--out', required=True)
    c.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    {'build': cmd_build, 'custody': cmd_custody}[args.cmd](args)


if __name__ == '__main__':
    main()
