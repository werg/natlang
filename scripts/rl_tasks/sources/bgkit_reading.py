"""bgkit knowledge-reasoning stores as multi-call RL reader programs (plans/neuralese/BGKIT_KR_CENSUS_20261103.md).

The stores that the corrected SFT mixture, the RL pool and the operator cases did not use, and whose answers are exact or
checkable against their own inputs, become typed natlang programs that need constructs beyond one wrapper call:

- `bgkit_searchqa` (SearchQA, 117k Jeopardy clues with ~12 web-search snippets each; the bgkit provenance store
  `provenance/searchqa.jsonl` carries only the clue, answer and md5 key, the snippets come from the same upstream):
  per-snippet `Neuralese` readers whose notes are tallied in code (`vote`), typed per-snippet candidates handed to a judge
  (`judge`), and an `iterateOn` reducer over snippet groups that carries typed vote counts with a decreasing measure
  (`tally`);
- `bgkit_haystack` (the bgkit `babi_filler_v3` store: bAbI stories whose facts are scattered through wikitext, PG-19,
  fineweb-edu text): an `iterateOn` world-state reducer over raw text chunks (`track`) and typed per-chunk fact finders fanned
  in to a settler (`spot`);
- `bgkit_pubmed` (the 1,000 expert-labelled PubMedQA abstracts of the bgkit/decision source; the 211k artificial labels have no
  abstract text locally): typed per-section stance readers fanned in to a verdict (`stance`).

Every case keeps an exact outcome oracle taken from the upstream annotation (SearchQA's answer, the bAbI answer the generator
placed, the PubMedQA expert label). The reference replays a direct root answer; `structure.smoke` holds the intended
orchestration. Reference notes for the Neuralese producers are `checked` (the sentence that contains the gold answer, the
bAbI facts a pattern found in the chunk) and never invented. Rows are chosen by hashed split unit, so related items stay on
one side of the split; the assembler's custody then removes anything related to a protected world.
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

from natlang_neuralese.common.paths import resolve

from ..child_notes import answer_sentences, child, document_key, fragment, overlap
from ..common import distinct_wrong
from .composed import DOC, Program, _case, _iterate_steps, _span_oracle

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from scripts.neuralese_data.records import group_key, text_hash  # noqa: E402

SHORT = 'a short phrase (a name, place, date, number or title), not a sentence'
HF_HUB = Path(os.environ.get('HF_HUB_CACHE') or Path.home() / '.cache' / 'huggingface' / 'hub')
BGKIT_TASKS = Path(os.environ.get('NATLANG_BGKIT_TASKS') or Path.home() / 'bgkit-data-nvme' / 'bgkit2' / 'tasks')
PROTECTED_INDEX = resolve('data_hdd', 'natlang-development-data', 'data', 'neuralese', 'protected',
                          'bgkit-benchmarks.protected.json')
DECISION_SOURCES = resolve('data_hdd', 'natlang-development-data', 'data', 'decision-sources')
SOURCE = 'bgkit'


def _key(text):
    return fragment(text)


def _protected_question_hashes():
    try:
        return set(json.load(open(PROTECTED_INDEX))['question_hashes'])
    except OSError:
        return set()


# ================================================================= SearchQA

def _snippets(context):
    out = []
    for raw in context.split('[DOC]'):
        raw = raw.strip()
        if not raw:
            continue
        title, _, text = raw.partition('[PAR]')
        title = re.sub(r'^\[TLE\]\s*', '', title).strip()
        text = text.strip()
        if text and len(text) >= 40:
            out.append({'title': title[:160] or 'untitled', 'text': text})
    return out


def searchqa_items(seed=20261103, scan=60000, limit=(1500, 150)):
    """SearchQA clues whose answer is found in the snippets. Train rows are the upstream train split (the provenance store
    is the same 117,384 questions); the validation split gives the held dev and confirmation rows."""
    import glob

    import pyarrow.parquet as pq
    blocked = _protected_question_hashes()
    root = glob.glob(str(HF_HUB / 'datasets--lucadiliello--searchqa' / 'snapshots' / '*' / 'data'))[0]
    out = {}
    for split, pattern, cap in (('train', 'train-*.parquet', limit[0]), ('validation', 'validation-*.parquet', limit[1] * 2)):
        table = pq.read_table(glob.glob(f'{root}/{pattern}')[0], columns=['context', 'question', 'answers', 'key'])
        keys = list(range(table.num_rows))
        random.Random(seed).shuffle(keys)
        items, seen, answers = [], set(), []
        for index in keys[:scan]:
            r = table.slice(index, 1).to_pylist()[0]
            if len(r['answers']) != 1:
                continue
            answer = r['answers'][0].strip().replace("\\'", "'")
            question = r['question'].strip()
            snippets = _snippets(r['context'])
            if not 4 <= len(snippets) <= 24 or not answer or len(answer.split()) > 6:
                continue
            support = [i for i, s in enumerate(snippets) if answer.lower() in s['text'].lower()]
            if not support or text_hash(question) in blocked or text_hash(question) in seen:
                continue
            seen.add(text_hash(question))
            answers.append(answer)
            items.append({'id': r['key'], 'unit': f'searchqa-q:{r["key"]}', 'groups': [f'searchqa-q:{r["key"]}'],
                          'source': 'searchqa', 'split': split, 'question': question, 'answer': answer,
                          'snippets': snippets, 'support': support,
                          'shape': f'{min(len(snippets), 20) // 4 * 4}snip-{min(len(support), 4)}sup'})
            if len(items) >= cap:
                break
        out[split] = items
        out[split + '-pool'] = answers
    return out


def _alias_forms(answer):
    forms = {re.sub(r'\s*\([^)]*\)\s*', ' ', answer).strip(), re.sub(r'^(the|a|an)\s+', '', answer, flags=re.I).strip(),
             answer.replace('&', 'and')}
    return [f for f in forms if f and f.lower() != answer.lower()]


SNIP = {'question': 'string', 'snippet': DOC}
CAND_T = '{ answer: string; evidence: string }[]'
VOTES_T = '{ votes: { answer: string; snippets: number }[] }'
CLUE = ('question is a quiz clue (Jeopardy style); snippet is one web search result for it (title and text), which may '
        'or may not contain the answer')

SQ_VOTE = Program(
    name='snippet-notes-vote', root='answer_by_agreement', args={'question': 'string', 'snippets': DOC + '[]'},
    returns='string', depth=2, producers=['read_snippet'],
    helpers={'read_snippet': (SNIP, 'Neuralese<string>',
                              CLUE + '. Write a private note of the answer the snippet supports, quoting the words '
                              'that show it, or say that the snippet does not answer the clue. A snippet that merely '
                              'repeats the clue answers nothing.')},
    instructions=[
        'snippets are web search results for the quiz clue question. Different snippets often support different '
        'answers.\n1. Read every snippet with `await read_snippet(question, snippet)`, one call per snippet, and '
        'keep the notes in a list.\n2. Count how many notes support each distinct answer (in eval, normalising case).\n'
        '3. Return the answer supported by the most snippets as ' + SHORT + '; when they tie, the answer whose '
        'supporting words match the clue best.',
        'Find the answer to the quiz clue question in snippets, the search results for it. Have read_snippet write a '
        'private note for each snippet, then tally the notes in code: the answer with the most supporting snippets '
        'wins, and a tie goes to the answer whose evidence fits the clue more closely. Return it as ' + SHORT + '.',
        'Each element of snippets is a web search result for the quiz clue question. Do not trust one snippet: collect a '
        'private note per snippet with read_snippet, tally the supported answers in eval, and return the one that the '
        'largest number of snippets agree on, as ' + SHORT + '.',
    ],
    smoke_code='const notes = [];\nfor (const snippet of snippets.slice(0, 3)) notes.push(await read_snippet(question, snippet));',
    smoke_children={'read_snippet': 'The snippet supports the answer.'})
SQ_JUDGE = Program(
    name='snippet-candidates-judge', root='judge_search_results', args={'question': 'string', 'snippets': DOC + '[]'},
    returns='string', depth=2, producers=['candidates'], typed=True, relay=True,
    helpers={'candidates': (SNIP, f'Neuralese<{CAND_T}>',
                            CLUE + '. List as typed private candidates each phrase of the snippet that could answer the '
                            'clue, with the words that support it. An empty list when the snippet has nothing for the '
                            'clue.'),
             'judge': ({'question': 'string', 'candidates': f'Neuralese<{CAND_T}>[]'}, 'string',
                       'candidates are the typed private candidate answers of every web search result for the quiz '
                       'clue question, one list per result. Choose the answer that the most results agree on and the '
                       'clue fits best. Return it as ' + SHORT + '.')},
    instructions=[
        '1. For every snippet in snippets call `candidates(question, snippet)` and collect the results in an array, in '
        'the order of snippets.\n2. Call `judge(question, results)` with the whole array and return its answer.',
        'question is a quiz clue and snippets are web search results for it. Turn each result into typed private '
        'candidates with candidates, in order, then give all of the lists together to judge, which picks the answer. '
        'Return what judge returns, ' + SHORT + '.',
    ],
    smoke_code='const results = [];\nfor (const snippet of snippets.slice(0, 3)) results.push(await candidates(question, snippet));\n'
               'const answer = await judge(question, results);',
    smoke_children={'candidates': [{'answer': 'x', 'evidence': 'y'}]})
SQ_TALLY = Program(
    name='snippet-groups-tally', root='tally_results', args={'groups': DOC + '[][]', 'question': 'string'},
    returns='string', depth=2, producers=['open_votes', 'add_votes'], typed=True, relay=True, iterate=True,
    helpers={
        'open_votes': ({'group': DOC + '[]', 'question': 'string'}, f'Neuralese<{VOTES_T}>',
                       'group holds a few web search results for the quiz clue question. Return typed private vote '
                       'counts: for every distinct answer the results support, how many of them support it. Empty '
                       'votes when none answers the clue.'),
        'add_votes': ({'prior': f'Neuralese<{VOTES_T}>', 'group': DOC + '[]', 'question': 'string'}, f'Neuralese<{VOTES_T}>',
                      'prior holds the vote counts of the earlier groups of search results for the quiz clue question. '
                      'Read group, the next results, and return the combined counts: add this group\'s supporting '
                      'results to the matching answers (the same answer written differently counts as one) and add '
                      'new answers.'),
    },
    instructions=[
        'groups are consecutive groups of web search results for the quiz clue question. Keep typed private vote counts '
        'and update them group by group with iterateOn:\n' + _iterate_steps('open_votes', 'add_votes', 'groups',
                                                                              ', question', 'votes') +
        '4. Return the answer with the most votes in the final state as ' + SHORT + '.',
        'Answer the quiz clue question from groups, groups of web search results. open_votes counts which answers '
        'the first group supports and add_votes folds the next group into the counts. Run add_votes over the other '
        'groups with iterateOn, measuring the groups not yet read, and return the answer with the highest count as '
        + SHORT + '.',
    ],
    smoke_code='const start = { votes: await open_votes(groups[0], question), next: 1 };\n'
               'const final = await iterateOn(async s => ({ votes: await add_votes(s.votes, groups[s.next], question), '
               'next: s.next + 1 }), start).withMeasure(s => groups.length - s.next).until(s => s.next >= groups.length);',
    smoke_children={'open_votes': {'votes': [{'answer': 'x', 'snippets': 1}]},
                    'add_votes': {'votes': [{'answer': 'x', 'snippets': 2}]}})


SQ_CLUE_FIRST = Program(
    name='clue-note-then-snippets', root='answer_by_clue_type', args={'question': 'string', 'snippets': DOC + '[]'},
    returns='string', depth=2, producers=['understand_clue', 'read_snippet'], relay=True,
    helpers={'understand_clue': ({'question': 'string'}, 'Neuralese<string>',
                                 'question is a quiz clue (Jeopardy style). Write a private note of what it asks for: the '
                                 'kind of answer (a person, place, title, year, thing), what each phrase of the clue '
                                 'requires of the answer, and which words are only wordplay or a category hint.'),
             'read_snippet': ({'need': 'Neuralese<string>', 'snippet': DOC}, 'Neuralese<string>',
                              'need is a private note of what a quiz clue asks for and snippet is one web search result '
                              'for it. Write a private note of the answer snippet supports that fits need, quoting the '
                              'words that show it, or say that the snippet does not answer the clue.')},
    instructions=[
        'Understand the clue before reading the search results.\n1. `const need = await understand_clue(question);`\n'
        '2. For every snippet in snippets, `await read_snippet(need, snippet)` and keep the notes in a list.\n'
        '3. Tally in code how many notes support each distinct answer and return the best supported one as ' + SHORT + '.',
        'question is a quiz clue and snippets are web search results for it. First have understand_clue write a private '
        'note of what the clue asks, then pass that note to read_snippet for each snippet. Count the supporting notes per '
        'answer in eval and return the answer with the most, as ' + SHORT + '.',
        'Answer the quiz clue question from snippets. Start with understand_clue(question), whose private note says what '
        'kind of answer the clue wants; read every snippet with read_snippet(need, snippet) using that note, then choose '
        'the answer most snippets agree on. Return it as ' + SHORT + '.',
    ],
    smoke_code='const need = await understand_clue(question);\nconst notes = [];\n'
               'for (const snippet of snippets.slice(0, 3)) notes.push(await read_snippet(need, snippet));',
    smoke_children={'understand_clue': 'The clue asks for a name.', 'read_snippet': 'The snippet supports the answer.'})
SQ_PROPOSE = Program(
    name='propose-then-verify', root='confirm_answer', args={'question': 'string', 'snippets': DOC + '[]'},
    returns='string', depth=2, producers=['propose'], typed=True, relay=True,
    helpers={'propose': ({'question': 'string', 'snippets': DOC + '[]'}, 'Neuralese<{ answer: string; titles: string[] }[]>',
                         'question is a quiz clue and snippets are web search results for it. Return typed private '
                         'proposals: up to three distinct answers the snippets suggest, each with the titles of the '
                         'snippets that support it. An answer no snippet supports must not be proposed.'),
             'verify': ({'question': 'string', 'proposals': 'Neuralese<{ answer: string; titles: string[] }[]>',
                         'snippets': DOC + '[]'}, 'string',
                        'proposals are typed private candidate answers to the quiz clue question with the titles of their '
                        'supporting snippets. Check each proposal against the text of its snippets and against what the clue '
                        'says, and return the one that is correct as ' + SHORT + '.')},
    instructions=[
        '1. `const proposals = await propose(question, snippets);`\n2. Return `await verify(question, proposals, snippets)`: '
        'the checked answer as ' + SHORT + '.',
        'Answer the quiz clue question in two steps. propose reads all the snippets and writes typed private proposals '
        'with their supporting titles; verify takes the proposals together with the snippets, checks each against its '
        'evidence and the clue, and answers. Return that answer, ' + SHORT + '.',
        'question is a quiz clue, snippets are search results for it. Get proposals from propose(question, snippets), '
        'then let verify(question, proposals, snippets) pick the right one. Return what verify returns, ' + SHORT + '.',
    ],
    smoke_code='const proposals = await propose(question, snippets);\nconst answer = await verify(question, proposals, snippets);',
    smoke_children={'propose': [{'answer': 'x', 'titles': ['t']}]})


SQ_SELECT = Program(
    name='score-then-read-top', root='answer_from_best_snippets', args={'question': 'string', 'snippets': DOC + '[]'},
    returns='string', depth=2, producers=['read_snippet'],
    helpers={'relevance': (SNIP, 'number',
                           CLUE + '. Return a score from 0 to 10 for how likely the snippet is to contain the answer to '
                           'the clue: 10 when it plainly states it, 0 when it is unrelated.'),
             'read_snippet': SQ_VOTE.helpers['read_snippet']},
    instructions=[
        '1. Score every snippet in snippets with `await relevance(question, snippet)`, one call each.\n2. In eval, sort the '
        'snippets by score from high to low (keep the original order among equal scores) and keep the best four.\n'
        '3. Read only those four with `await read_snippet(question, snippet)`.\n4. Return the answer most of the notes '
        'support, as ' + SHORT + '.',
        'Most search results do not help, so triage them: relevance scores each snippet of snippets for the quiz clue question. '
        'Sort by score in code, keep the top four, have read_snippet write a private note for each of them, and return the '
        'answer the notes agree on as ' + SHORT + '.',
        'Pick the four most promising snippets by relevance score (sort in eval, highest first), read them with read_snippet, '
        'and answer the quiz clue question from those notes. Return ' + SHORT + '.',
    ],
    smoke_code='const scored = [];\nfor (const snippet of snippets) scored.push({ snippet, score: await relevance(question, snippet) });\n'
               'scored.sort((a, b) => b.score - a.score);\nconst notes = [];\n'
               'for (const item of scored.slice(0, 3)) notes.push(await read_snippet(question, item.snippet));',
    smoke_children={'relevance': 5, 'read_snippet': 'The snippet supports the answer.'})


def _sq_level(item):
    n, s = len(item['snippets']), len(item['support'])
    return 0 if s >= 3 and n <= 10 else 1 if s >= 2 else 2 if n <= 14 else 3


def _snippet_note(item, snippet, supports):
    sentence = answer_sentences(snippet['text'], [item['answer']], 320)
    if supports and sentence:
        return f'The snippet supports the answer "{item["answer"]}": {sentence}'
    return 'The snippet does not answer the clue.'


def _sq_notes(program, item, groups):
    answer, out = item['answer'], []
    if program in (SQ_VOTE, SQ_SELECT):
        for snippet in item['snippets']:
            hit = answer.lower() in snippet['text'].lower()
            out.append(child('read_snippet', [document_key(snippet['title']), _key(snippet['text'])],
                             _snippet_note(item, snippet, hit), 'checked'))
    elif program is SQ_JUDGE:
        for snippet in item['snippets']:
            sentence = answer_sentences(snippet['text'], [answer], 320)
            value = [{'answer': answer, 'evidence': sentence}] if sentence else []
            out.append(child('candidates', [document_key(snippet['title']), _key(snippet['text'])], value, 'checked'))
    elif program is SQ_CLUE_FIRST:
        out = [child('read_snippet', [document_key(sn['title']), _key(sn['text'])],
                     _snippet_note(item, sn, answer.lower() in sn['text'].lower()), 'checked') for sn in item['snippets']]
    elif program is SQ_PROPOSE:
        titles = [sn['title'] for sn in item['snippets'] if answer.lower() in sn['text'].lower()]
        out = [child('propose', [document_key(item['snippets'][0]['title'])], [{'answer': answer, 'titles': titles[:6]}], 'checked')]
    else:
        count = 0
        for index, group in enumerate(groups):
            count += sum(answer.lower() in s['text'].lower() for s in group)
            value = {'votes': [{'answer': answer, 'snippets': count}] if count else []}
            out.append(child('open_votes' if index == 0 else 'add_votes', [_key(group[0]['text'])], value, 'checked'))
    return out


def searchqa(train=(250, 250, 250, 250, 250, 250), held=40, seed=20261103):
    family = 'searchqa_reading'
    items = searchqa_items(seed, limit=(sum(train), held))
    out = []
    for split in ('train', 'validation'):
        pool = items[split + '-pool']
        counts = train if split == 'train' else (held // 2 + 1,) * 6
        cursor = 0
        for program, n in zip((SQ_VOTE, SQ_JUDGE, SQ_TALLY, SQ_CLUE_FIRST, SQ_PROPOSE, SQ_SELECT), counts):
            for k, item in enumerate(items[split][cursor:cursor + n]):
                answer = item['answer']
                groups = [item['snippets'][i:i + 3] for i in range(0, len(item['snippets']), 3)]
                inputs = ({'groups': groups, 'question': item['question']} if program is SQ_TALLY
                          else {'question': item['question'], 'snippets': item['snippets']})
                oracle, alts = _span_oracle(answer, _alias_forms(answer))
                out.append(_case(program, family, item=item, inputs=inputs, expected=answer, oracle=oracle,
                                 level=_sq_level(item) if program is not SQ_TALLY else min(3, _sq_level(item) + 1),
                                 wrong=distinct_wrong(answer, pool[(cursor + k) % 400:] + pool, 3),
                                 aliases=[answer.upper()] + alts[:2], upstream='lucadiliello/searchqa',
                                 license_='LicenseRef-searchqa-web-snippets',
                                 license_note='clues and answers are J! Archive items; snippets are public web text',
                                 difficulty=item['shape'], slice_='iterate' if program is SQ_TALLY else 'nested_scoped',
                                 smoke_children={**({'judge': answer} if program is SQ_JUDGE else {'verify': answer} if program is SQ_PROPOSE else {})},
                                 children=_sq_notes(program, item, groups), source=SOURCE))
            cursor += n
    return out


# ================================================================= bAbI facts in filler text (babi_filler_v3)

FACT = re.compile(r'(?<![A-Za-z])(?:Mary|John|Sandra|Daniel) (?:went|moved|travelled|journeyed|got|took|picked|grabbed|'
                  r'dropped|left|put|discarded|handed|gave|passed|is)\b[^.]*\.')
FILLER = {'wikitext': 'encyclopedia', 'pg19': 'book', 'fineweb_edu': 'web page', 'qa': 'passage'}


def _chunks(text, size=1800):
    pieces, chunks, current = re.split(r'(?<=[.!?])\s+', text), [], ''
    for piece in pieces:
        if current and len(current) + len(piece) > size:
            chunks.append(current)
            current = ''
        current = (current + ' ' + piece) if current else piece
    if current:
        chunks.append(current)
    return chunks


def haystack_items(seed=20261103, limit=(1400, 120), max_chars=20000):
    import pyarrow.parquet as pq
    table = pq.read_table(BGKIT_TASKS / 'babi_filler_v3.parquet', columns=['context', 'instruction', 'target', 'split', 'meta'])
    order = list(range(table.num_rows))
    random.Random(seed).shuffle(order)
    out = {'train': [], 'validation': []}
    pools = {'train': defaultdict(list), 'validation': defaultdict(list)}
    for index in order:
        r = table.slice(index, 1).to_pylist()[0]
        meta = json.loads(r['meta'])
        split = 'train' if r['split'] == 'train' else 'validation'
        cap = limit[0] if split == 'train' else limit[1] * 2
        if len(out[split]) >= cap or len(r['context']) > max_chars or meta['task'] == 'qa1' and meta['n_facts'] < 8:
            continue
        facts = FACT.findall(r['context'])
        chunks = _chunks(r['context'])
        if len(facts) != meta['n_facts'] or not 3 <= len(chunks) <= 16:
            continue  # the pattern must count exactly the facts the generator placed
        story = group_key('babi-story', meta['doc'].rsplit(':', 1)[0])
        unit = f'bgkit-babi-filler:{story}'  # every filler variant of one story stays on one side of the split
        item = {'id': f'{meta["doc"]}:{index}', 'unit': unit, 'groups': [story], 'source': 'babi_filler', 'split': split,
                'question': r['instruction'], 'answer': r['target'], 'chunks': chunks, 'facts': facts, 'task': meta['task'],
                'filler': meta['filler_source'], 'shape': f'{meta["task"]}-{len(chunks)}chunks-{meta["filler_source"]}'}
        pools[split][meta['task']].append(r['target'])
        out[split].append(item)
    for split in out:
        for item in out[split]:
            item['pool'] = pools[split][item['task']]
    return out


HAY_Q = {'chunks': 'string[]', 'question': 'string'}
HAY_ANSWER = 'Return only the answer word (a place or an object).'
HAY_TRACK = Program(
    name='haystack-state-reducer', root='answer_from_text', args=HAY_Q, returns='string', depth=2,
    producers=['open_world', 'read_on'], relay=True, iterate=True,
    helpers={
        'open_world': ({'chunk': 'string', 'question': 'string'}, 'Neuralese<string>',
                       'chunk is the first part of a long text. Most of the text is unrelated filler, but a few sentences '
                       'in it are short facts about where Mary, John, Sandra or Daniel went or what they took, dropped or '
                       'gave. Write a private world state: every such fact in chunk in order, and where each person and '
                       'object is after them, for question.'),
        'read_on': ({'state': 'Neuralese<string>', 'chunk': 'string', 'question': 'string'}, 'Neuralese<string>',
                    'state is a private world state built from the earlier parts of a long text. Read chunk, the next '
                    'part (mostly unrelated filler), find any short facts about people moving or handling objects and '
                    'apply them in order: a later fact replaces an earlier one. Write the updated state and keep the '
                    'history question may ask about.'),
        'answer_from_world': ({'state': 'Neuralese<string>', 'question': 'string'}, 'string',
                              'state is a private world state built from a whole long text. Answer question from it. '
                              + HAY_ANSWER),
    },
    instructions=[
        'chunks are the consecutive parts of one long text in which a few facts about people and objects are hidden among '
        'unrelated sentences. Track the world while you read, one chunk per step of iterateOn:\n'
        + _iterate_steps('open_world', 'read_on', 'chunks', ', question', 'state') +
        '4. Return `await answer_from_world(final.state, question)`.',
        'A story is scattered through the unrelated text of chunks. Read the chunks in order, keeping a private world '
        'state: open_world starts it from chunks[0] and read_on updates it with the next chunk. Drive read_on with iterateOn '
        '(measure: chunks not yet read) and have answer_from_world answer question from the last state. '
        + HAY_ANSWER,
    ],
    smoke_code='const start = { state: await open_world(chunks[0], question), next: 1 };\n'
               'const final = await iterateOn(async s => ({ state: await read_on(s.state, chunks[s.next], question), '
               'next: s.next + 1 }), start).withMeasure(s => chunks.length - s.next).until(s => s.next >= chunks.length);\n'
               'const answer = await answer_from_world(final.state, question);',
    smoke_children={'open_world': 'state 1', 'read_on': 'state'})
FACTS_T = '{ facts: string[] }'
HAY_SPOT = Program(
    name='haystack-fact-finders', root='answer_from_stream', args=HAY_Q, returns='string', depth=2,
    producers=['spot_facts'], typed=True, relay=True,
    helpers={
        'spot_facts': ({'chunk': 'string', 'question': 'string'}, f'Neuralese<{FACTS_T}>',
                       'chunk is one part of a long text that is mostly unrelated filler. List as typed private facts '
                       'every sentence of chunk that says where Mary, John, Sandra or Daniel went, or what one of them '
                       'took, picked up, dropped, left or gave, copied word for word and in order. Empty when there '
                       'is none.'),
        'settle': ({'finds': f'Neuralese<{FACTS_T}>[]', 'question': 'string'}, 'string',
                   'finds are the typed private facts of consecutive parts of a long text, one entry per part and in '
                   'text order. Put them together in order, follow each person and object through them (a later fact '
                   'replaces an earlier one) and answer question. ' + HAY_ANSWER),
    },
    instructions=[
        '1. Call `spot_facts(chunk, question)` for every chunk in chunks, one call per chunk, and collect the results in '
        'order.\n2. Call `settle(finds, question)` with the whole array and return its answer.',
        'chunks are the consecutive parts of one long text with a short story hidden in it. Have spot_facts pick the '
        'story sentences out of each chunk, and give all the lists, in order, to settle, which answers question. '
        + HAY_ANSWER,
        'Hidden in the unrelated sentences of chunks is a short story about people and objects. For each chunk, spot_facts '
        'returns the typed private list of story sentences it holds; call it for all chunks in order and pass the lists to '
        'settle with question. ' + HAY_ANSWER,
    ],
    smoke_code='const finds = [];\nfor (const chunk of chunks.slice(0, 3)) finds.push(await spot_facts(chunk, question));\n'
               'const answer = await settle(finds, question);',
    smoke_children={'spot_facts': {'facts': ['Mary moved to the hallway.']}})


HAY_FOCUS = Program(
    name='haystack-plan-then-spot', root='answer_with_plan', args=HAY_Q, returns='string', depth=2,
    producers=['plan_search', 'spot_facts'], typed=True, relay=True,
    helpers={
        'plan_search': ({'question': 'string'}, 'Neuralese<string>',
                        'question is about a short story of people moving between rooms and handling objects, hidden in '
                        'a long text. Write a private search plan: which people and objects the question follows and which '
                        'kinds of sentences (moves, pick-ups, drops, hand-overs) decide it, in the order they must be applied.'),
        'spot_facts': ({'plan': 'Neuralese<string>', 'chunk': 'string'}, f'Neuralese<{FACTS_T}>',
                       'plan is a private search plan for a question about a short story hidden in a long text. List as '
                       'typed private facts the sentences of chunk (one part of the text) that the plan needs, copied word '
                       'for word and in order. Empty when there is none.'),
        'settle': HAY_SPOT.helpers['settle'],
    },
    instructions=[
        '1. `const plan = await plan_search(question);`\n2. For every chunk in chunks, `await spot_facts(plan, chunk)` and '
        'collect the results in order.\n3. Return `await settle(finds, question)`.',
        'Plan first: plan_search(question) writes a private search plan. Then have spot_facts(plan, chunk) pick the story '
        'sentences out of each of the consecutive chunks, in order, and give all of the lists to settle, which answers '
        'question. ' + HAY_ANSWER,
        'A short story is hidden among the unrelated sentences of chunks, and question asks about it. Make a search plan with '
        'plan_search, run spot_facts with that plan over every chunk, and let settle answer from the collected lists. '
        + HAY_ANSWER,
    ],
    smoke_code='const plan = await plan_search(question);\nconst finds = [];\n'
               'for (const chunk of chunks.slice(0, 3)) finds.push(await spot_facts(plan, chunk));\n'
               'const answer = await settle(finds, question);',
    smoke_children={'plan_search': 'Follow the person asked about.', 'spot_facts': {'facts': ['Mary moved to the hallway.']}})


HAY_SELECT = Program(
    name='haystack-triage-then-spot', root='answer_from_flagged_chunks', args=HAY_Q, returns='string', depth=2,
    producers=['spot_facts'], typed=True, relay=True,
    helpers={
        'holds_story': ({'chunk': 'string'}, 'boolean',
                        'chunk is one part of a long text that is mostly unrelated filler. Return true only when it '
                        'contains a short sentence saying where Mary, John, Sandra or Daniel went, or what one of them '
                        'took, dropped or gave.'),
        'spot_facts': HAY_SPOT.helpers['spot_facts'],
        'settle': HAY_SPOT.helpers['settle'],
    },
    instructions=[
        '1. Call `holds_story(chunk)` for every chunk in chunks.\n2. In eval, keep only the chunks for which it returned true, '
        'in their original order.\n3. Call `spot_facts(chunk, question)` for each kept chunk and collect the results in order.\n'
        '4. Return `await settle(finds, question)`.',
        'Triage before reading closely: holds_story tells which chunks of chunks contain story sentences. Filter the chunks '
        'in code, run spot_facts on the flagged ones in order, and give the typed lists to settle with question. '
        + HAY_ANSWER,
        'Only a few chunks hold the hidden story. Flag them with holds_story, then spot_facts on just those (in text order) '
        'and let settle answer question from the lists. ' + HAY_ANSWER,
    ],
    smoke_code='const flagged = [];\nfor (const chunk of chunks) if (await holds_story(chunk)) flagged.push(chunk);\n'
               'const finds = [];\nfor (const chunk of flagged.slice(0, 3)) finds.push(await spot_facts(chunk, question));\n'
               'const answer = await settle(finds, question);',
    smoke_children={'holds_story': True, 'spot_facts': {'facts': ['Mary moved to the hallway.']}})


def _hay_notes(program, item):
    out, seen = [], []
    for index, chunk in enumerate(item['chunks']):
        found = FACT.findall(chunk)
        seen.extend(found)
        if program in (HAY_SPOT, HAY_FOCUS, HAY_SELECT):
            out.append(child('spot_facts', [_key(chunk)], {'facts': found}, 'checked'))
        else:
            text = ('Facts so far, in order:\n' + '\n'.join(seen)) if seen else 'No story fact has appeared so far.'
            out.append(child('open_world' if index == 0 else 'read_on', [_key(chunk)], text, 'checked'))
    return out


def haystack(train=(250, 250, 250, 250), held=36, seed=20261103):
    family = 'haystack_reading'
    items = haystack_items(seed, limit=(sum(train), held))
    out = []
    for split in ('train', 'validation'):
        counts = train if split == 'train' else (held // 2,) * 4
        cursor = 0
        for program, n in zip((HAY_TRACK, HAY_SPOT, HAY_FOCUS, HAY_SELECT), counts):
            for k, item in enumerate(items[split][cursor:cursor + n]):
                chunks = len(item['chunks'])
                level = 0 if chunks <= 4 else 1 if chunks <= 7 else 2 if chunks <= 11 else 3
                if item['task'] == 'qa3':
                    level = min(3, level + 1)
                answer = item['answer']
                out.append(_case(program, family, item=item, inputs={'chunks': item['chunks'], 'question': item['question']},
                                 expected=answer, oracle={'level': 'normalized', 'normalization': 'qa'}, level=level,
                                 wrong=distinct_wrong(answer, item['pool'][k % 50:] + item['pool'], 2),
                                 aliases=[answer.upper()], upstream='bgkit/babi_filler_v3',
                                 license_='BSD-3-Clause', license_note='bAbI tasks inside wikitext, PG-19 and fineweb-edu filler',
                                 difficulty=item['shape'], slice_='iterate' if program is HAY_TRACK else 'nested_scoped',
                                 smoke_children={**({'settle': answer} if program is not HAY_TRACK else {'answer_from_world': answer})},
                                 children=_hay_notes(program, item), source=SOURCE))
            cursor += n
    return out


# ================================================================= PubMedQA (expert-labelled)

STANCE_T = '{ stance: "yes" | "no" | "unclear"; evidence: string }'
VERDICT = '"yes" | "no" | "maybe"'
PUB_SECTION = {'question': 'string', 'label': 'string', 'text': 'string'}
PUB_STANCE = Program(
    name='abstract-stance-readers', root='answer_from_abstract', args={'question': 'string', 'sections': '{ label: string; text: string }[]'},
    returns=VERDICT, depth=2, producers=['read_section'], typed=True, relay=True,
    helpers={
        'read_section': (PUB_SECTION, f'Neuralese<{STANCE_T}>',
                         'text is one labelled section of a research abstract and question asks about the study. Return '
                         'the typed private stance of this section: "yes" when its findings answer question affirmatively, '
                         '"no" when they answer it negatively, "unclear" when it says nothing decisive; evidence quotes '
                         'the words that decide it, or is empty.'),
        'verdict': ({'question': 'string', 'readings': f'Neuralese<{STANCE_T}>[]'}, VERDICT,
                    'readings are the typed private stances of every section of one research abstract for question, in '
                    'order. Return "yes" or "no" when the decisive sections agree, and "maybe" when the results are '
                    'mixed, conditional or inconclusive.'),
    },
    instructions=[
        '1. Call `read_section(question, section.label, section.text)` for every section in sections and collect the '
        'readings in order.\n2. Return `await verdict(question, readings)`: "yes", "no" or "maybe".',
        'question asks about the study described by the abstract in sections. Read each labelled section with read_section '
        'to get its typed private stance, then let verdict combine all readings into the final answer, "yes", "no" or '
        '"maybe".',
        'Decide whether the study in sections supports question. read_section gives the typed private stance of one section; '
        'call it for each section in order, then ask verdict(question, readings) and return its "yes", "no" or "maybe".',
    ],
    smoke_code='const readings = [];\nfor (const section of sections) readings.push(await read_section(question, section.label, section.text));\n'
               'const answer = await verdict(question, readings);',
    smoke_children={'read_section': {'stance': 'yes', 'evidence': 'x'}})


EVIDENCE_T = '{ for_yes: string[]; for_no: string[] }'
PUB_ITER = Program(
    name='abstract-evidence-reducer', root='decide_from_abstract',
    args={'question': 'string', 'sections': '{ label: string; text: string }[]'}, returns=VERDICT, depth=2,
    producers=['open_evidence', 'add_evidence'], typed=True, relay=True, iterate=True,
    helpers={
        'open_evidence': ({'section': '{ label: string; text: string }', 'question': 'string'}, f'Neuralese<{EVIDENCE_T}>',
                          'section is the first labelled section of a research abstract. Return typed private evidence for '
                          'question: for_yes lists the statements that support an affirmative answer and for_no those that '
                          'support a negative one, copied from the section; both empty when it is only background.'),
        'add_evidence': ({'prior': f'Neuralese<{EVIDENCE_T}>', 'section': '{ label: string; text: string }', 'question': 'string'},
                         f'Neuralese<{EVIDENCE_T}>',
                         'prior is the typed private evidence for question from the earlier sections of an abstract. Read '
                         'section, the next one, and return the combined evidence: keep prior items that still hold, add '
                         'the section\'s statements and drop an item the section overturns.'),
        'verdict': ({'question': 'string', 'evidence': f'Neuralese<{EVIDENCE_T}>'}, VERDICT,
                    'evidence is the typed private evidence for and against an affirmative answer to question gathered from '
                    'a whole abstract. Return "yes" when it clearly favours yes, "no" when it clearly favours no, and '
                    '"maybe" when it is mixed, conditional or inconclusive.'),
    },
    instructions=[
        'sections are the labelled sections of one abstract, in order. Collect typed private evidence section by section with '
        'iterateOn:\n' + _iterate_steps('open_evidence', 'add_evidence', 'sections', ', question', 'evidence') +
        '4. Return `await verdict(question, final.evidence)`: "yes", "no" or "maybe".',
        'question asks about the study in sections. open_evidence turns the first section into typed private evidence for and '
        'against, add_evidence folds in each next section; drive it with iterateOn (measure: sections not yet read) and let '
        'verdict decide from the last state: "yes", "no" or "maybe".',
        'Weigh the abstract section by section: keep typed evidence for and against with open_evidence and add_evidence under '
        'iterateOn, then return verdict(question, evidence) as "yes", "no" or "maybe".',
    ],
    smoke_code='const start = { evidence: await open_evidence(sections[0], question), next: 1 };\n'
               'const final = await iterateOn(async s => ({ evidence: await add_evidence(s.evidence, sections[s.next], question), '
               'next: s.next + 1 }), start).withMeasure(s => sections.length - s.next).until(s => s.next >= sections.length);\n'
               'const answer = await verdict(question, final.evidence);',
    smoke_children={'open_evidence': {'for_yes': ['x'], 'for_no': []}, 'add_evidence': {'for_yes': ['x'], 'for_no': []}})


def pubmed_items(seed=20261103):
    import pyarrow.parquet as pq
    rows = pq.read_table(DECISION_SOURCES / 'qiaojin__PubMedQA' / 'pqa_labeled' / 'train' / '0000.parquet').to_pylist()
    items = []
    for r in rows:
        labels, texts = r['context']['labels'], r['context']['contexts']
        if len(labels) != len(texts) or len(texts) < 2 or r['final_decision'] not in ('yes', 'no', 'maybe'):
            continue
        sections = [{'label': l.title(), 'text': t.strip()} for l, t in zip(labels, texts) if t.strip()]
        unit = f'pubmedqa:{r["pubid"]}'
        items.append({'id': str(r['pubid']), 'unit': unit, 'groups': [f'pubmedqa:{r["pubid"]}'], 'source': 'pubmedqa',
                      'split': 'train-carve', 'question': r['question'].strip(), 'answer': r['final_decision'],
                      'sections': sections, 'long_answer': r['long_answer'],
                      'shape': f'{len(sections)}sections-{r["final_decision"]}'})
    random.Random(seed).shuffle(items)
    return items


DECISIVE_SECTIONS = ('RESULT', 'CONCLUSION', 'FINDING', 'OUTCOME', 'INTERPRETATION')


def _pubmed_notes(item):
    """Mechanical stance notes (provenance `mechanical:decisive-section-overlap`): the expert label is a verdict on the whole
    abstract, so only a results or conclusion section carries it (with the sentences that overlap the abstract's own long
    answer); every other section is `unclear`. A `maybe` abstract is `unclear` throughout."""
    out = []
    for section in item['sections']:
        decisive = section['label'].upper().startswith(DECISIVE_SECTIONS) and item['answer'] != 'maybe'
        evidence = (overlap(section['text'], item['long_answer'], 400) or '') if decisive else ''
        value = {'stance': item['answer'] if decisive else 'unclear', 'evidence': evidence}
        out.append(child('read_section', [_key(section['text'])], value, 'mechanical:decisive-section-overlap'))
    return out


def pubmed(train=900, seed=20261103):
    family = 'pubmed_reading'
    out = []
    for k, item in enumerate(pubmed_items(seed)[:train + 120]):
        answer = item['answer']
        wrong = [a for a in ('yes', 'no', 'maybe') if a != answer]
        program = PUB_ITER if k % 2 else PUB_STANCE
        out.append(_case(program, family, item={**item}, inputs={'question': item['question'], 'sections': item['sections']},
                         expected=answer, oracle='exact', level=2 if answer == 'maybe' else 1 if len(item['sections']) > 3 else 0,
                         wrong=wrong, aliases=[], upstream='qiaojin/PubMedQA', license_='MIT',
                         difficulty=item['shape'], smoke_children={'verdict': answer},
                         slice_='iterate' if program is PUB_ITER else 'nested_scoped',
                         children=_pubmed_notes(item) if program is PUB_STANCE else [], source=SOURCE))
    return out


# ================================================================= NarrativeQA summaries (qa_narrativeqa)

def _passages(text, size=700):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    out, current = [], ''
    for sentence in sentences:
        if current and len(current) + len(sentence) > size:
            out.append(current)
            current = ''
        current = (current + ' ' + sentence) if current else sentence
    if current:
        out.append(current)
    return out


def narrative_items(seed=20261103, limit=(900, 80)):
    """Questions over NarrativeQA plot summaries whose reference answer is a short span found in the summary."""
    import pyarrow.parquet as pq
    table = pq.read_table(BGKIT_TASKS / 'qa_narrativeqa.parquet', columns=['context', 'instruction', 'target', 'split', 'meta'])
    order = list(range(table.num_rows))
    random.Random(seed).shuffle(order)
    out = {'train': [], 'validation': []}
    answers = {'train': [], 'validation': []}
    seen = set()
    for index in order:
        r = table.slice(index, 1).to_pylist()[0]
        meta = json.loads(r['meta'])
        split = 'train' if r['split'] == 'train' else 'validation'
        if len(out[split]) >= (limit[0] if split == 'train' else limit[1] * 2):
            continue
        lowered = r['context'].lower()
        spans = [a.strip().rstrip('.') for a in meta.get('answers', []) if a.strip()]
        found = [a for a in spans if 1 <= len(a.split()) <= 6 and a.lower() in lowered]
        passages = _passages(r['context'])
        question = r['instruction'].strip()
        if not found or not 3 <= len(passages) <= 14 or text_hash(question + meta['doc']) in seen:
            continue
        seen.add(text_hash(question + meta['doc']))
        doc = group_key('narrativeqa-doc', meta['doc'])
        answers[split].append(found[0])
        out[split].append({'id': f'{meta["doc"]}:{index}', 'unit': f'bgkit-narrative:{doc}', 'groups': [doc],
                           'source': 'narrativeqa', 'split': split, 'question': question, 'answer': found[0],
                           'aliases': [a for a in spans if a != found[0] and len(a.split()) <= 10][:3],
                           'passages': passages, 'kind': meta.get('kind', ''),
                           'shape': f'{min(len(passages), 12)}passages-{meta.get("kind", "story")}'})
    for split in out:
        for item in out[split]:
            item['pool'] = answers[split]
    return out


NQ_NOTE_T = '{ found: boolean; quote: string }'
NQ_QUESTION = {'question': 'string', 'passage': 'string'}
NQ_JUDGE = Program(
    name='story-passage-finders', root='answer_about_story', args={'question': 'string', 'passages': 'string[]'},
    returns='string', depth=2, producers=['find_in_passage'], typed=True, relay=True,
    helpers={'find_in_passage': (NQ_QUESTION, f'Neuralese<{NQ_NOTE_T}>',
                                 'passage is one part of the plot summary of a novel or film. Say as a typed private '
                                 'note whether passage answers question (found) and quote the words that answer it '
                                 '(quote), or leave quote empty.'),
             'conclude': ({'question': 'string', 'finds': f'Neuralese<{NQ_NOTE_T}>[]'}, 'string',
                          'finds are the typed private notes of every part of a story summary for question, in order. '
                          'Answer question with the words the summary uses, as ' + SHORT + '.')},
    instructions=[
        '1. Call `find_in_passage(question, passage)` for every passage in passages, in order, and keep the results.\n'
        '2. Return `await conclude(question, finds)`, the answer as ' + SHORT + '.',
        'passages are consecutive parts of the plot summary of one story. Let find_in_passage check each part for the '
        'answer to question, then hand all the typed notes to conclude and return its answer, ' + SHORT + '.',
    ],
    smoke_code='const finds = [];\nfor (const passage of passages.slice(0, 3)) finds.push(await find_in_passage(question, passage));\n'
               'const answer = await conclude(question, finds);',
    smoke_children={'find_in_passage': {'found': True, 'quote': 'x'}})
NQ_ITER = Program(
    name='story-notes-reducer', root='answer_after_reading_story', args={'passages': 'string[]', 'question': 'string'},
    returns='string', depth=2, producers=['open_story_notes', 'extend_story_notes'], relay=True, iterate=True,
    helpers={
        'open_story_notes': ({'passage': 'string', 'question': 'string'}, 'Neuralese<string>',
                             'passage is the first part of the plot summary of a story. Write a private note of what it '
                             'says that bears on question: the names, places and events involved, in the summary\'s own words.'),
        'extend_story_notes': ({'notes': 'Neuralese<string>', 'passage': 'string', 'question': 'string'}, 'Neuralese<string>',
                               'notes are private notes on the earlier parts of a story summary for question. Read '
                               'passage, the next part, and write updated notes: keep what still matters, add what '
                               'passage says about question and correct anything it contradicts.'),
    },
    instructions=[
        'passages are the consecutive parts of one story summary. Read them in order with iterateOn, carrying private '
        'notes:\n' + _iterate_steps('open_story_notes', 'extend_story_notes', 'passages', ', question', 'notes') +
        '4. Answer question from the final notes with the summary\'s own words, as ' + SHORT + '.',
        'Answer question about the story told by passages. open_story_notes writes private notes on passages[0], and '
        'extend_story_notes folds in the next passage. Run it over the other passages with iterateOn (measure: passages '
        'not yet read) and answer from the final notes, as ' + SHORT + '.',
    ],
    smoke_code='const start = { notes: await open_story_notes(passages[0], question), next: 1 };\n'
               'const final = await iterateOn(async s => ({ notes: await extend_story_notes(s.notes, passages[s.next], question), '
               'next: s.next + 1 }), start).withMeasure(s => passages.length - s.next).until(s => s.next >= passages.length);',
    smoke_children={'open_story_notes': 'notes 1', 'extend_story_notes': 'notes'})


NQ_FOCUS = Program(
    name='story-outline-then-passages', root='answer_with_outline', args={'question': 'string', 'passages': 'string[]'},
    returns='string', depth=2, producers=['outline_story', 'find_in_passage'], typed=True, relay=True,
    helpers={'outline_story': ({'passages': 'string[]'}, 'Neuralese<string>',
                               'passages are consecutive parts of the plot summary of a novel or film. Write a private '
                               'outline of the whole story: the main characters and how they are related, the places, and '
                               'the order of the events.'),
             'find_in_passage': ({'outline': 'Neuralese<string>', **NQ_QUESTION}, f'Neuralese<{NQ_NOTE_T}>',
                                 'outline is a private outline of a whole story and passage is one part of its summary. Say '
                                 'as a typed private note whether passage answers question (found), using the outline to '
                                 'resolve who people are, and quote the words that answer it (quote), or leave quote empty.'),
             'conclude': NQ_JUDGE.helpers['conclude']},
    instructions=[
        '1. `const outline = await outline_story(passages);`\n2. For every passage in passages, in order, '
        '`await find_in_passage(outline, question, passage)` and keep the results.\n3. Return '
        '`await conclude(question, finds)`, the answer as ' + SHORT + '.',
        'Read the story as a whole first: outline_story(passages) gives a private outline. Then check each passage for the '
        'answer to question with find_in_passage(outline, question, passage), and let conclude combine the typed notes into '
        'the answer, ' + SHORT + '.',
        'question is about the story summarised in passages. Build an outline with outline_story, run find_in_passage with '
        'the outline over every passage, and return conclude(question, finds), as ' + SHORT + '.',
    ],
    smoke_code='const outline = await outline_story(passages);\nconst finds = [];\n'
               'for (const passage of passages.slice(0, 3)) finds.push(await find_in_passage(outline, question, passage));\n'
               'const answer = await conclude(question, finds);',
    smoke_children={'outline_story': 'outline', 'find_in_passage': {'found': True, 'quote': 'x'}})


NQ_SELECT = Program(
    name='story-triage-then-passages', root='answer_from_flagged_passages', args={'question': 'string', 'passages': 'string[]'},
    returns='string', depth=2, producers=['find_in_passage'], typed=True, relay=True,
    helpers={'bears_on': (NQ_QUESTION, 'boolean',
                          'passage is one part of the plot summary of a novel or film. Return true when it mentions a '
                          'character, place or event that question asks about.'),
             'find_in_passage': NQ_JUDGE.helpers['find_in_passage'],
             'conclude': NQ_JUDGE.helpers['conclude']},
    instructions=[
        '1. Call `bears_on(question, passage)` for every passage in passages.\n2. In eval keep the passages for which it '
        'returned true (all of them when none did), in order.\n3. Call `find_in_passage(question, passage)` on each kept '
        'passage and keep the results.\n4. Return `await conclude(question, finds)`, the answer as ' + SHORT + '.',
        'Narrow the story first: bears_on flags the passages of passages that touch question. Filter in code (fall back to '
        'every passage when none is flagged), run find_in_passage on the kept ones, then let conclude answer, as ' + SHORT + '.',
        'Skip the parts of the summary that have nothing to do with question: flag the relevant passages with bears_on, '
        'read only those with find_in_passage, and return conclude(question, finds), ' + SHORT + '.',
    ],
    smoke_code='const kept = [];\nfor (const passage of passages) if (await bears_on(question, passage)) kept.push(passage);\n'
               'const finds = [];\nfor (const passage of (kept.length ? kept : passages).slice(0, 3)) finds.push(await find_in_passage(question, passage));\n'
               'const answer = await conclude(question, finds);',
    smoke_children={'bears_on': True, 'find_in_passage': {'found': True, 'quote': 'x'}})


def _nq_notes(program, item):
    answer, out, seen = item['answer'], [], ''
    for index, passage in enumerate(item['passages']):
        sentence = answer_sentences(passage, [answer], 320)
        if program in (NQ_JUDGE, NQ_FOCUS, NQ_SELECT):
            value = {'found': bool(sentence), 'quote': sentence or ''}
            out.append(child('find_in_passage', [_key(passage)], value, 'checked'))
        else:
            if sentence and answer.lower() not in seen.lower():
                seen = (seen + '\n' + sentence).strip()
            text = f'Relevant so far: {seen}' if seen else 'Nothing in the story so far answers the question.'
            out.append(child('open_story_notes' if index == 0 else 'extend_story_notes', [_key(passage)], text, 'checked'))
    return out


def narrative(train=(225, 225, 225, 225), held=40, seed=20261103):
    family = 'narrative_reading'
    items = narrative_items(seed, limit=(sum(train), held))
    out = []
    for split in ('train', 'validation'):
        counts = train if split == 'train' else (held // 2,) * 4
        cursor = 0
        for program, n in zip((NQ_JUDGE, NQ_ITER, NQ_FOCUS, NQ_SELECT), counts):
            for k, item in enumerate(items[split][cursor:cursor + n]):
                answer = item['answer']
                oracle, alts = _span_oracle(answer, item['aliases'])
                parts = len(item['passages'])
                out.append(_case(program, family, item=item, inputs={'passages': item['passages'], 'question': item['question']},
                                 expected=answer, oracle=oracle, level=0 if parts <= 4 else 1 if parts <= 7 else 2 if parts <= 10 else 3,
                                 wrong=distinct_wrong(answer, item['pool'][k % 200:] + item['pool'], 3),
                                 aliases=[answer.upper()] + alts[:2], upstream='deepmind/narrativeqa', license_='Apache-2.0',
                                 license_note='summaries are Wikipedia-derived plot text', difficulty=item['shape'],
                                 slice_='iterate' if program is NQ_ITER else 'nested_scoped',
                                 smoke_children={'conclude': answer} if program is not NQ_ITER else {},
                                 children=_nq_notes(program, item), source=SOURCE))
            cursor += n
    return out


# ================================================================= memory of dated conversation sessions (memory_qa_v2_pairs)

def memory_items(seed=20261103, limit=(500, 40), max_chars=45000):
    """Questions over dated conversation sessions whose short answer occurs verbatim in a session (answers were written by
    a bgkit teacher, so an answer that no session contains is dropped, as are abstentions)."""
    import pyarrow.parquet as pq
    table = pq.read_table(BGKIT_TASKS / 'memory_qa_v2_pairs.parquet', columns=['context', 'prompt', 'target', 'split', 'meta'])
    order = list(range(table.num_rows))
    random.Random(seed).shuffle(order)
    out = {'train': [], 'validation': []}
    pool = {'train': [], 'validation': []}
    for index in order:
        r = table.slice(index, 1).to_pylist()[0]
        meta = json.loads(r['meta'])
        split = 'train' if r['split'] == 'train' else 'validation'
        if len(out[split]) >= (limit[0] if split == 'train' else limit[1] * 2):
            continue
        if len(r['context']) > max_chars or meta['qtype'] == 'abstention' or not 1 <= len(r['target'].split()) <= 6:
            continue
        sessions = json.loads(r['context'])
        answer = r['target'].strip()
        if not 3 <= len(sessions) <= 40 or not any(answer.lower() in s.lower() for s in sessions):
            continue
        question = re.sub(r'^\(asked on [^)]*\)\s*', '', json.loads(r['prompt'])[0]).strip()
        episode = group_key('memory-episode', f'{meta["source"]}:{meta["episode"]}')
        pool[split].append(answer)
        out[split].append({'id': f'{meta["episode"]}:{index}', 'unit': f'bgkit-memory:{episode}', 'groups': [episode],
                           'source': 'memory_qa', 'split': 'train-carve' if split == 'train' else split, 'question': question, 'asked_on': meta.get('asked_on', ''),
                           'answer': answer, 'sessions': sessions, 'qtype': meta['qtype'],
                           'shape': f'{min(len(sessions) // 8 * 8, 32)}sessions-{meta["qtype"]}'})
    for split in out:
        for item in out[split]:
            item['pool'] = pool[split]
    return out


MEM_T = '{ relevant: boolean; facts: string[] }'
MEM_FANIN = Program(
    name='session-recall-judge', root='answer_from_memory',
    args={'question': 'string', 'sessions': 'string[]'}, returns='string', depth=2, producers=['recall'], typed=True, relay=True,
    helpers={'recall': ({'question': 'string', 'session': 'string'}, f'Neuralese<{MEM_T}>',
                        'session is one dated conversation from a long chat history. Return typed private recall for '
                        'question: relevant says whether session says anything that bears on it, and facts lists each such '
                        'statement with its date, copied from the session.'),
             'settle': ({'question': 'string', 'recalls': f'Neuralese<{MEM_T}>[]'}, 'string',
                        'recalls are the typed private recalls of every dated session of a chat history for question, '
                        'oldest first. When the sessions disagree the newest statement holds. Answer question as ' + SHORT + '.')},
    instructions=[
        'sessions are the dated conversations of a long chat history, oldest first. Answer question from them.\n'
        '1. Call `recall(question, session)` for every session, in order, and collect the results.\n'
        '2. Return `await settle(question, recalls)`, the answer as ' + SHORT + '.',
        'question is about a long chat history given as the dated sessions in sessions, oldest first. Have recall pull '
        'what each session says about it, then give all the typed recalls to settle (a later session overrides an earlier '
        'one) and return its answer, ' + SHORT + '.',
        'Each element of sessions is one dated conversation, oldest first. Run recall on every session for question, keep '
        'the typed private recalls in order, and let settle answer from all of them. Return ' + SHORT + '.',
    ],
    smoke_code='const recalls = [];\nfor (const session of sessions.slice(0, 3)) recalls.push(await recall(question, session));\n'
               'const answer = await settle(question, recalls);',
    smoke_children={'recall': {'relevant': True, 'facts': ['x']}})


MEM_ITER = Program(
    name='session-notes-reducer', root='track_chat_history', args={'question': 'string', 'sessions': 'string[]'},
    returns='string', depth=2, producers=['open_memory', 'fold_memory'], relay=True, iterate=True,
    helpers={
        'open_memory': ({'session': 'string', 'question': 'string'}, 'Neuralese<string>',
                        'session is the oldest dated conversation of a chat history. Write a private note of what it says '
                        'that bears on question, with its date.'),
        'fold_memory': ({'notes': 'Neuralese<string>', 'session': 'string', 'question': 'string'}, 'Neuralese<string>',
                        'notes are private notes on the older sessions of a chat history for question. Read session, the '
                        'next (newer) one, and write updated notes: keep what still holds, add what it says about question '
                        'with its date, and replace anything it updates or contradicts.'),
        'answer_from_memory': ({'notes': 'Neuralese<string>', 'question': 'string'}, 'string',
                               'notes are private notes on a whole chat history for question; later facts replace earlier '
                               'ones. Answer question as ' + SHORT + '.'),
    },
    instructions=[
        'sessions are the dated conversations of a chat history, oldest first. Keep private notes while reading them in '
        'order with iterateOn:\n' + _iterate_steps('open_memory', 'fold_memory', 'sessions', ', question', 'notes') +
        '4. Return `await answer_from_memory(final.notes, question)`.',
        'Walk through the chat history in sessions from the oldest to the newest. open_memory notes the first session and '
        'fold_memory folds each newer session into the notes; drive it with iterateOn (measure: sessions not yet read). '
        'Then answer question from the final notes with answer_from_memory, as ' + SHORT + '.',
        'question depends on what was said in a long chat history (sessions, oldest first), where newer sessions can '
        'update older ones. Reduce the history session by session with iterateOn and the two note helpers, then let '
        'answer_from_memory answer. Return ' + SHORT + '.',
    ],
    smoke_code='const start = { notes: await open_memory(sessions[0], question), next: 1 };\n'
               'const final = await iterateOn(async s => ({ notes: await fold_memory(s.notes, sessions[s.next], question), '
               'next: s.next + 1 }), start).withMeasure(s => sessions.length - s.next).until(s => s.next >= sessions.length);\n'
               'const answer = await answer_from_memory(final.notes, question);',
    smoke_children={'open_memory': 'note 1', 'fold_memory': 'notes'})


MEM_SELECT = Program(
    name='session-triage-then-recall', root='answer_from_relevant_sessions', args={'question': 'string', 'sessions': 'string[]'},
    returns='string', depth=2, producers=['recall'], typed=True, relay=True,
    helpers={'mentions': ({'question': 'string', 'session': 'string'}, 'boolean',
                          'session is one dated conversation from a long chat history. Return true when it talks about '
                          'something question asks about.'),
             'recall': MEM_FANIN.helpers['recall'], 'settle': MEM_FANIN.helpers['settle']},
    instructions=[
        '1. Call `mentions(question, session)` for every session in sessions.\n2. In eval keep only the sessions for which it '
        'returned true (every session when none did), oldest first.\n3. Call `recall(question, session)` on each kept '
        'session and collect the results in order.\n4. Return `await settle(question, recalls)`, the answer as ' + SHORT + '.',
        'A long chat history rarely bears on a question in most sessions. Flag the sessions that do with mentions, filter '
        'them in code (all of them if none is flagged), recall from the flagged ones in order and let settle answer, '
        'as ' + SHORT + '.',
        'Narrow sessions down with mentions(question, session), run recall on what is left and give the typed recalls to '
        'settle; newer sessions override older ones. Return ' + SHORT + '.',
    ],
    smoke_code='const kept = [];\nfor (const session of sessions) if (await mentions(question, session)) kept.push(session);\n'
               'const recalls = [];\nfor (const session of (kept.length ? kept : sessions).slice(0, 3)) recalls.push(await recall(question, session));\n'
               'const answer = await settle(question, recalls);',
    smoke_children={'mentions': True, 'recall': {'relevant': True, 'facts': ['x']}})


def _memory_iter_notes(item):
    out, seen = [], []
    for index, session in enumerate(item['sessions']):
        sentence = answer_sentences(session, [item['answer']], 320)
        if sentence:
            seen.append(sentence)
        text = ('Statements so far that answer the question:\n' + '\n'.join(seen[-3:])) if seen else 'Nothing so far bears on the question.'
        out.append(child('open_memory' if index == 0 else 'fold_memory', [_key(session)], text, 'checked'))
    return out


def _memory_notes(item):
    out = []
    for session in item['sessions']:
        sentence = answer_sentences(session, [item['answer']], 320)
        out.append(child('recall', [_key(session)], {'relevant': bool(sentence), 'facts': [sentence] if sentence else []}, 'checked'))
    return out


def memory(train=420, held=30, seed=20261103):
    family = 'memory_reading'
    items = memory_items(seed, limit=(train, held))
    out = []
    for split in ('train', 'validation'):
        for k, item in enumerate(items[split]):
            answer = item['answer']
            n = len(item['sessions'])
            program = (MEM_FANIN, MEM_ITER, MEM_SELECT)[k % 3]
            out.append(_case(program, family, item=item, inputs={'question': item['question'], 'sessions': item['sessions']},
                             expected=answer, oracle=dict({'level': 'span', 'normalization': 'qa', 'threshold': 0.8}),
                             level=0 if n <= 8 else 1 if n <= 16 else 2 if n <= 28 else 3,
                             wrong=distinct_wrong(answer, item['pool'][k % 100:] + item['pool'], 3), aliases=[answer.upper()],
                             upstream='bgkit/memory_qa_v2_pairs', license_='LicenseRef-unverified',
                             license_note='sessions from Conversation Chronicles; questions and answers by a bgkit teacher, kept only when the answer occurs verbatim in a session',
                             difficulty=item['shape'], slice_='iterate' if program is MEM_ITER else 'nested_scoped',
                             smoke_children={'answer_from_memory': answer} if program is MEM_ITER else {'settle': answer},
                             children=_memory_iter_notes(item) if program is MEM_ITER else _memory_notes(item), source=SOURCE))
    return out
