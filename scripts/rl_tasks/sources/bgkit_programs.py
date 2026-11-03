"""More reader programs for the bgkit RL sources (scripts/rl_tasks/sources/bgkit_reading.py; plans/neuralese/BGKIT_KR_CENSUS_20261103.md).

The first version of the bgkit readers was world-diverse but program-poor: 3 to 6 call graphs per family, so adding the
whole supply to the pool cut its effective program count. These programs differ from those and from each other in what
they compute, who calls whom, which data is typed and which exact steps the root's eval code performs, not in wording:

- decision readouts (`readout: decision`, leaves that call nothing) whose value routes the rest of the program;
- plain typed helper results (`boolean`, `number`, `number[]`, records) that the root filters, sorts, tallies, replays or
  compares in eval (a Neuralese value is opaque to code, so exact arithmetic needs plain data);
- typed Neuralese producers fanned in to a consumer (`Neuralese<{...}>`), with reference notes only where the data
  derives them (`checked`, or `mechanical:`); a producer whose content the data cannot derive has no note, never an invented one;
- `iterateOn` with an explicit stop and a decreasing measure, forwards (typed state) and backwards (first decisive element);
- propose-and-verify with a retry branch, question decomposition, entity-keyword retrieval, recency-weighted votes,
  conflict detection, a person-by-person ledger, an exact event replay of a bAbI story.

Every case keeps the upstream gold as its exact oracle. `bgkit_reading` imports the program lists and note builders.
"""
from __future__ import annotations

import re

from ..child_notes import answer_sentences, child, document_key, fragment, overlap, string_key
from .composed import DOC, Program, _iterate_steps

SHORT = 'a short phrase (a name, place, date, number or title), not a sentence'
READOUTS = {}  # program name -> helper names written with `readout: decision`


def _key(text):
    return fragment(text)


def _decision(program_name, *helpers):
    READOUTS[program_name] = helpers


# ================================================================= SearchQA

SNIP = {'question': 'string', 'snippet': DOC}
CLUE = ('question is a quiz clue (Jeopardy style); snippet is one web search result for it (title and text), which may '
        'or may not contain the answer')
READ_SNIPPET = (SNIP, 'Neuralese<string>',
                CLUE + '. Write a private note of the answer the snippet supports, quoting the words that show it, or '
                'say that the snippet does not answer the clue. A snippet that merely repeats the clue answers nothing.')
RELEVANCE = (SNIP, 'number',
             CLUE + '. Return a score from 0 to 10 for how likely the snippet is to contain the answer to the clue: 10 '
             'when it plainly states it, 0 when it is unrelated.')
KIND_T = '"person" | "place" | "title" | "date_or_number" | "thing"'
READING_T = '{ answer: string; evidence: string }'

SQ_KIND = Program(
    name='answer-type-then-candidates', root='answer_of_the_asked_type', args={'question': 'string', 'snippets': DOC + '[]'},
    returns='string', depth=2, producers=[],
    helpers={'answer_kind': ({'question': 'string'}, KIND_T,
                             'question is a quiz clue (Jeopardy style). Decide what kind of answer it asks for: a person, '
                             'a place, a title (of a book, film, song or work), a date or number, or another thing.'),
             'candidates': (SNIP, '{ answer: string; kind: ' + KIND_T + ' }[]',
                            CLUE + '. List every phrase of the snippet that could answer the clue, each with its own kind '
                            '(person, place, title, date_or_number or thing). Empty when the snippet has nothing for it.')},
    instructions=[
        '1. `const kind = (await decide(answer_kind, question)).value;`\n2. For every snippet in snippets call '
        '`await candidates(question, snippet)` and put all the returned candidates into one array.\n3. In eval keep only the '
        'candidates whose kind equals `kind` (all of them when none do), count each distinct answer ignoring case, and '
        'return the most frequent one (the first seen on a tie) as ' + SHORT + '.',
        'Decide what type of answer the quiz clue question wants with the answer_kind readout, then read the web results in '
        'snippets: candidates lists typed candidate answers for one snippet. Filter the candidates by the wanted type in code, count the distinct '
        'answers and return the most frequent as ' + SHORT + '.',
    ],
    smoke_code='const kind = (await decide(answer_kind, question)).value;\nconst found = [];\n'
               'for (const snippet of snippets.slice(0, 3)) found.push(...(await candidates(question, snippet)));\n'
               'const typed = found.filter(c => c.kind === kind);',
    smoke_children={'answer_kind': 'person', 'candidates': [{'answer': 'x', 'kind': 'person'}]})
_decision('answer-type-then-candidates', 'answer_kind')

SQ_SKIM = Program(
    name='skim-then-read-chosen', root='read_the_promising_snippets', args={'question': 'string', 'snippets': DOC + '[]'},
    returns='string', depth=2, producers=['read_snippet'],
    helpers={'skim': ({'question': 'string', 'snippets': DOC + '[]'}, 'number[]',
                      'question is a quiz clue and snippets are web search results for it. Return the positions (counting '
                      'from 0) of the at most four snippets most likely to hold the answer, the most promising first.'),
             'read_snippet': READ_SNIPPET},
    instructions=[
        '1. `const picks = await skim(question, snippets);`\n2. In eval keep the integers of picks that are valid positions of '
        'snippets, each once, and at most four (positions 0, 1 and 2 when none is valid).\n3. Read each chosen snippet with '
        '`await read_snippet(question, snippet)`.\n4. Return the answer the most notes support, as ' + SHORT + '.',
        'Skim all the search results at once: skim(question, snippets) returns the positions worth reading. Check them in '
        'code (valid, distinct, at most four, with a fallback of the first three), have read_snippet write a private note '
        'for each chosen snippet, and return the answer most notes agree on, ' + SHORT + '.',
    ],
    smoke_code='const picks = await skim(question, snippets);\nconst chosen = [...new Set(picks.filter(i => Number.isInteger(i) && '
               'i >= 0 && i < snippets.length))].slice(0, 3);\nconst notes = [];\n'
               'for (const i of chosen) notes.push(await read_snippet(question, snippets[i]));',
    smoke_children={'skim': [0, 1], 'read_snippet': 'The snippet supports the answer.'})

SQ_STANCE = Program(
    name='propose-then-stance-tally', root='check_claims_against_snippets', args={'question': 'string', 'snippets': DOC + '[]'},
    returns='string', depth=2, producers=[],
    helpers={'propose_claims': ({'question': 'string', 'snippets': DOC + '[]'}, 'string[]',
                                'question is a quiz clue and snippets are web search results for it. Return up to three '
                                'different candidate answers the snippets suggest, each as a short phrase, the likeliest first.'),
             'stance': ({'question': 'string', 'claim': 'string', 'snippet': DOC}, '"supports" | "contradicts" | "silent"',
                        'claim is a proposed answer to the quiz clue question and snippet is one web search result. Say '
                        'whether the snippet supports claim as the answer, contradicts it (it gives a different answer or rules '
                        'it out) or is silent about it.')},
    instructions=[
        '1. `const claims = (await propose_claims(question, snippets)).slice(0, 2);`\n2. For every claim and every snippet get '
        '`(await decide(stance, question, claim, snippet)).value`.\n3. In eval score a claim: +1 for each "supports", -1 for '
        'each "contradicts".\n4. Return the claim with the highest score (the first on a tie) as ' + SHORT + '.',
        'Propose the two likeliest answers with propose_claims, then test each against every web result with the stance '
        'readout. Score supports minus contradicts for each claim in code and return the best-scoring claim, as ' + SHORT + '.',
    ],
    smoke_code='const claims = (await propose_claims(question, snippets)).slice(0, 2);\nconst scored = [];\n'
               'for (const claim of claims) {\n  let score = 0;\n  for (const snippet of snippets.slice(0, 3)) {\n'
               '    const d = await decide(stance, question, claim, snippet);\n'
               '    score += d.value === "supports" ? 1 : d.value === "contradicts" ? -1 : 0;\n  }\n  scored.push({ claim, score });\n}',
    smoke_children={'propose_claims': ['x', 'y'], 'stance': 'supports'})
_decision('propose-then-stance-tally', 'stance')

SQ_TWO_READERS = Program(
    name='strict-and-broad-readers-arbiter', root='adjudicate_two_readings', args={'question': 'string', 'snippets': DOC + '[]'},
    returns='string', depth=2, producers=[],
    helpers={'strict_reader': ({'question': 'string', 'snippets': DOC + '[]'}, READING_T,
                               'question is a quiz clue and snippets are web search results for it. Give an answer only when a '
                               'snippet states it outright, with the quoted words as evidence; otherwise an empty answer.'),
             'broad_reader': ({'question': 'string', 'snippets': DOC + '[]'}, READING_T,
                              'question is a quiz clue and snippets are web search results for it. Give the answer the results '
                              'most plausibly point to, also when it must be inferred from several of them, with the words '
                              'it rests on as evidence.'),
             'arbiter': ({'question': 'string', 'strict': READING_T, 'broad': READING_T, 'snippets': DOC + '[]'}, 'string',
                         'strict and broad are two readings (answer and evidence) of the web results snippets for the quiz clue '
                         'question that disagree. Check both against the snippets and the clue and return the correct '
                         'answer as ' + SHORT + '.')},
    instructions=[
        '1. `const strict = await strict_reader(question, snippets);` and `const broad = await broad_reader(question, '
        'snippets);`\n2. In eval, when both answers are non-empty and equal ignoring case and surrounding spaces, return that '
        'answer.\n3. Otherwise return `await arbiter(question, strict, broad, snippets)`, ' + SHORT + '.',
        'Two readers look at the same search results for the quiz clue question: strict_reader only accepts a stated answer and '
        'broad_reader accepts an inferred one. Compare their answers in code; if they agree return it, and if not let arbiter '
        'decide with both readings and the snippets. Return ' + SHORT + '.',
    ],
    smoke_code='const strict = await strict_reader(question, snippets);\nconst broad = await broad_reader(question, snippets);\n'
               'const same = strict.answer.trim().toLowerCase() === broad.answer.trim().toLowerCase();\n'
               'const answer = same ? strict.answer : await arbiter(question, strict, broad, snippets);',
    smoke_children={'strict_reader': {'answer': 'x', 'evidence': 'e'}, 'broad_reader': {'answer': 'y', 'evidence': 'e'}})

SQ_CONSTRAINTS = Program(
    name='clue-constraints-score', root='score_candidates_against_the_clue', args={'question': 'string', 'snippets': DOC + '[]'},
    returns='string', depth=2, producers=[],
    helpers={'clue_constraints': ({'question': 'string'}, 'string[]',
                                  'question is a quiz clue (Jeopardy style). Split it into two to four separate conditions '
                                  'the answer must meet, each a short phrase in the clue\'s own words. Wordplay and category '
                                  'hints are not conditions.'),
             'propose_candidates': ({'question': 'string', 'snippets': DOC + '[]'}, 'string[]',
                                    'question is a quiz clue and snippets are web search results for it. Return up to three '
                                    'different candidate answers the results suggest, as short phrases.'),
             'meets': ({'candidate': 'string', 'constraint': 'string', 'snippets': DOC + '[]'}, 'boolean',
                       'Return true only when the web search results snippets show that candidate meets constraint, a '
                       'condition taken from a quiz clue.')},
    instructions=[
        '1. `const constraints = (await clue_constraints(question)).slice(0, 3);` and '
        '`const candidates = (await propose_candidates(question, snippets)).slice(0, 3);`\n2. For every candidate and every '
        'constraint call `await meets(candidate, constraint, snippets)`.\n3. In eval a candidate\'s score is the number of '
        'constraints it meets.\n4. Return the candidate with the highest score (the first on a tie) as ' + SHORT + '.',
        'Break the quiz clue question into conditions with clue_constraints and get up to three candidate answers with '
        'propose_candidates. Ask meets for each pair against the snippets, count the met conditions per candidate in code '
        'and return the highest-scoring candidate, ' + SHORT + '.',
    ],
    smoke_code='const constraints = (await clue_constraints(question)).slice(0, 3);\n'
               'const candidates = (await propose_candidates(question, snippets)).slice(0, 3);\nconst scored = [];\n'
               'for (const candidate of candidates) {\n  let score = 0;\n'
               '  for (const constraint of constraints) if (await meets(candidate, constraint, snippets)) score += 1;\n'
               '  scored.push({ candidate, score });\n}',
    smoke_children={'clue_constraints': ['a', 'b'], 'propose_candidates': ['x', 'y'], 'meets': True})

SQ_TITLES = Program(
    name='title-triage-then-read', root='read_snippets_after_title_check', args={'question': 'string', 'snippets': DOC + '[]'},
    returns='string', depth=2, producers=['read_snippet'],
    helpers={'title_fits': ({'question': 'string', 'title': 'string'}, 'boolean',
                            'title is the title of one web search result for the quiz clue question. Return true when a page '
                            'with that title could plausibly contain the answer.'),
             'read_snippet': READ_SNIPPET},
    instructions=[
        '1. For every snippet in snippets call `await title_fits(question, snippet.title)` (the title only).\n2. In eval keep '
        'the snippets flagged true, in order and at most six (the first three when none is flagged).\n3. Read each kept snippet '
        'with `await read_snippet(question, snippet)`.\n4. Return the answer most notes support, as ' + SHORT + '.',
        'Look at the titles of the search results before their text: title_fits flags the promising ones. Keep the flagged '
        'snippets in code (up to six, else the first three), have read_snippet write a private note for each, and return the '
        'answer the notes agree on, ' + SHORT + '.',
    ],
    smoke_code='const flags = [];\nfor (const snippet of snippets) flags.push(await title_fits(question, snippet.title));\n'
               'const kept = snippets.filter((s, i) => flags[i]).slice(0, 6);\nconst notes = [];\n'
               'for (const snippet of (kept.length ? kept : snippets.slice(0, 3)).slice(0, 3)) notes.push(await read_snippet(question, snippet));',
    smoke_children={'title_fits': True, 'read_snippet': 'The snippet supports the answer.'})

SQ_UNTIL_AGREE = Program(
    name='ranked-read-until-two-agree', root='read_ranked_until_two_agree', args={'question': 'string', 'snippets': DOC + '[]'},
    returns='string', depth=2, producers=[], iterate=True,
    helpers={'relevance': RELEVANCE,
             'read_answer': (SNIP, 'string',
                             CLUE + '. Return the answer this one snippet gives to the clue as ' + SHORT +
                             ', or an empty string when the snippet does not answer it.')},
    instructions=[
        '1. Score every snippet with `await relevance(question, snippet)` and sort the pairs by score from high to low (keep '
        'the original order among equal scores) into `ranked`.\n2. Read the ranked snippets one per iterateOn step, keeping '
        'the non-empty answers: start from `{ answers: [], next: 0 }`; a step returns `{ answers: <answers plus the new '
        'one when non-empty>, next: s.next + 1 }` after `await read_answer(question, ranked[s.next].snippet)`.\n3. Run it with '
        '`iterateOn(step, start).withMeasure(s => ranked.length - s.next).until(...)`, stopping when `next` reaches '
        '`ranked.length` or the last two answers are equal ignoring case.\n4. Return the repeated last answer; when the loop '
        'ended without agreement, the most frequent answer (the first seen on a tie), as ' + SHORT + '.',
        'Rank the web results by relevance score in code, then read them best first with iterateOn, one snippet per step, until '
        'two consecutive non-empty answers agree (or the snippets run out); the measure is the ranked snippets not yet read. '
        'Return the agreed answer, or the most frequent one, as ' + SHORT + '.',
    ],
    smoke_code='const scored = [];\nfor (const snippet of snippets) scored.push({ snippet, score: await relevance(question, snippet) });\n'
               'const ranked = scored.map((p, i) => ({ ...p, i })).sort((a, b) => b.score - a.score || a.i - b.i);\n'
               'const start = { answers: [], next: 0 };\n'
               'const step = async s => { const a = await read_answer(question, ranked[s.next].snippet); '
               'return { answers: a.trim() ? [...s.answers, a] : s.answers, next: s.next + 1 }; };\n'
               'const final = await iterateOn(step, start).withMeasure(s => ranked.length - s.next).until(s => s.next >= ranked.length '
               '|| (s.answers.length >= 2 && s.answers[s.answers.length - 1].toLowerCase() === s.answers[s.answers.length - 2].toLowerCase()));',
    smoke_children={'relevance': 5, 'read_answer': 'x'})

SQ_DEDUPE = Program(
    name='dedupe-then-extract-vote', root='vote_over_distinct_snippets', args={'question': 'string', 'snippets': DOC + '[]'},
    returns='string', depth=2, producers=[],
    helpers={'extract': (SNIP, '{ answer: string; quote: string }',
                         CLUE + '. Return the answer this one snippet gives to the clue as ' + SHORT + ' and the words that '
                         'show it (quote); both empty when the snippet does not answer the clue.')},
    instructions=[
        '1. In eval drop every snippet whose text, lower-cased with spaces squeezed, starts with the same 120 characters as an '
        'earlier kept snippet (search engines repeat pages).\n2. Call `await extract(question, snippet)` for each kept snippet.\n'
        '3. In eval ignore empty answers, count each distinct answer ignoring case, and return the most frequent one (the first '
        'seen on a tie) as ' + SHORT + '.',
        'Web search results often repeat the same page. Remove the duplicates in code (same first 120 lower-cased characters), '
        'extract one typed answer with its quote from each remaining snippet, and return the answer with the most distinct '
        'snippets behind it, ' + SHORT + '.',
    ],
    smoke_code='const kept = [];\nconst seen = new Set();\nfor (const snippet of snippets) {\n'
               '  const head = snippet.text.toLowerCase().replace(/\\s+/g, " ").slice(0, 120);\n'
               '  if (!seen.has(head)) { seen.add(head); kept.push(snippet); }\n}\nconst found = [];\n'
               'for (const snippet of kept.slice(0, 3)) found.push(await extract(question, snippet));',
    smoke_children={'extract': {'answer': 'x', 'quote': 'q'}})

SQ_NEW = (SQ_KIND, SQ_SKIM, SQ_STANCE, SQ_TWO_READERS, SQ_CONSTRAINTS, SQ_TITLES, SQ_UNTIL_AGREE, SQ_DEDUPE)
SQ_HARDER = {SQ_STANCE.name, SQ_CONSTRAINTS.name, SQ_UNTIL_AGREE.name}


def _snippet_note(item, snippet):
    sentence = answer_sentences(snippet['text'], [item['answer']], 320)
    if answer_in(item, snippet) and sentence:
        return f'The snippet supports the answer "{item["answer"]}": {sentence}'
    return 'The snippet does not answer the clue.'


def answer_in(item, snippet):
    return item['answer'].lower() in snippet['text'].lower()


def sq_notes(program, item):
    """Checked notes for the Neuralese producers of the new SearchQA programs (the data's own answer sentence)."""
    if program in (SQ_SKIM, SQ_TITLES):
        return [child('read_snippet', [document_key(sn['title']), _key(sn['text'])], _snippet_note(item, sn), 'checked')
                for sn in item['snippets']]
    return []


# ================================================================= bAbI facts in filler text

NAMES = ('Mary', 'John', 'Sandra', 'Daniel')
MOVE = re.compile(r'^(\w+) (?:moved|travelled|journeyed|went(?: back)?) to the (\w+)\.$')
TAKE = re.compile(r'^(\w+) (?:took|got|grabbed|picked up) the (\w+)(?: there)?\.$')
DROP = re.compile(r'^(\w+) (?:dropped|left|put down|discarded) the (\w+)(?: there)?\.$')


def parse_event(sentence):
    """A story sentence as (actor, kind, thing): move (thing is the room), take or drop (thing is the object)."""
    for kind, pattern in (('move', MOVE), ('take', TAKE), ('drop', DROP)):
        found = pattern.match(sentence.strip())
        if found:
            return found.group(1), kind, found.group(2)
    return None


def replay(facts, question):
    """The answer an exact replay of the story sentences gives to the question (None when it cannot be answered)."""
    room, holder, dropped, trail = {}, {}, {}, {}
    for sentence in facts:
        event = parse_event(sentence)
        if event is None:
            return None
        actor, kind, thing = event
        if kind == 'move':
            room[actor] = thing
            for obj, who in holder.items():
                if who == actor:
                    _push(trail, obj, thing)
        elif kind == 'take':
            if actor not in room:
                return None
            holder[thing] = actor
            dropped.pop(thing, None)
            _push(trail, thing, room[actor])
        else:
            if actor not in room:
                return None
            if holder.get(thing) == actor:
                del holder[thing]
            dropped[thing] = room[actor]
            _push(trail, thing, room[actor])
    found = re.match(r'^Where is (\w+)\?$', question.strip())
    if found:
        name = found.group(1)
        if name in NAMES:
            return room.get(name)
        thing = name.lower()
        return room.get(holder[thing]) if thing in holder else dropped.get(thing)
    found = re.match(r'^Where is the (\w+)\?$', question.strip())
    if found:
        thing = found.group(1)
        return room.get(holder[thing]) if thing in holder else dropped.get(thing)
    found = re.match(r'^Where was the (\w+) before the (\w+)\?$', question.strip())
    if found:
        places = trail.get(found.group(1), [])
        target = found.group(2)
        indices = [i for i, p in enumerate(places) if p == target]
        return places[indices[-1] - 1] if indices and indices[-1] > 0 else None
    return None


def _push(trail, thing, place):
    places = trail.setdefault(thing, [])
    if not places or places[-1] != place:
        places.append(place)


def person_of(question):
    found = re.match(r'^Where is (\w+)\?$', question.strip())
    return found.group(1) if found and found.group(1) in NAMES else None


HAY_Q = {'chunks': 'string[]', 'question': 'string'}
HAY_ANSWER = 'Return only the answer word (a place or an object).'
STORY = ('a short story hidden among unrelated sentences: facts about where Mary, John, Sandra or Daniel went, or what one of '
         'them took, picked up, dropped, left or gave')
EVENT_T = '{ actor: string; kind: "move" | "take" | "drop"; thing: string }[]'
FACTS_T = '{ facts: string[] }'
SETTLE_LIST = ({'facts': 'string[]', 'question': 'string'}, 'string',
               'facts are the sentences of a short story about people and objects, in the order they occur in a long text. '
               'Follow each person and object through them (a later fact replaces an earlier one) and answer question. '
               + HAY_ANSWER)

HAY_REPLAY = Program(
    name='haystack-events-replay', root='replay_the_hidden_story', args=HAY_Q, returns='string', depth=2, producers=[],
    helpers={'read_events': ({'chunk': 'string'}, EVENT_T,
                             'chunk is one part of a long text that is mostly unrelated filler. List, in order, every sentence '
                             'of chunk that says where Mary, John, Sandra or Daniel went or what one of them took, picked up, '
                             'dropped, left or put down, as an event: actor is the person; for a move kind is "move" and thing '
                             'the room; for taking an object kind is "take" and thing the object; for letting go of it kind is '
                             '"drop" and thing the object. Empty when chunk has no such sentence.')},
    instructions=[
        'chunks are the consecutive parts of one long text. A short story is hidden in it.\n1. For every chunk, in order, call '
        '`await read_events(chunk)` and join the lists into one array `events`.\n2. Replay `events` in eval, in order, with '
        'three records: `room` (person to room), `holder` (object to the person carrying it) and `dropped` (object to the room '
        'where it lies), and a list `trail[object]` of the rooms the object has been in (add a room only when it differs from '
        'the last). "move": `room[actor] = thing` and append the new room to the trail of every object that actor carries. '
        '"take": `holder[thing] = actor`, delete `dropped[thing]`, append `room[actor]` to the trail. "drop": delete '
        '`holder[thing]`, set `dropped[thing] = room[actor]`.\n3. Answer question: "Where is NAME?" is `room[NAME]`; "Where '
        'is the OBJ?" is `room[holder[OBJ]]` when the object is carried, else `dropped[OBJ]`; "Where was the OBJ before the '
        'ROOM?" is the room just before the last ROOM in `trail[OBJ]`. ' + HAY_ANSWER,
        'Read the hidden story as typed events and compute the answer yourself: read_events turns each chunk into '
        '{ actor, kind, thing } events. Concatenate the events in text order and replay them in eval (rooms of people, who '
        'carries which object, where an object was dropped, and each object\'s trail of rooms), then read off the answer to '
        'question: a person\'s room, an object\'s room, or the room before a named room in the object\'s trail. ' + HAY_ANSWER,
    ],
    smoke_code='const events = [];\nfor (const chunk of chunks.slice(0, 3)) events.push(...(await read_events(chunk)));\n'
               'const room = {};\nfor (const e of events) if (e.kind === "move") room[e.actor] = e.thing;',
    smoke_children={'read_events': [{'actor': 'Mary', 'kind': 'move', 'thing': 'hallway'}]})

TRAIL_T = '{ places: string[] }'
HAY_TRAIL = Program(
    name='haystack-person-trail', root='follow_the_person', args=HAY_Q, returns='string', depth=2, producers=[], iterate=True,
    helpers={
        'start_trail': ({'chunk': 'string', 'question': 'string'}, TRAIL_T,
                        'chunk is the first part of a long text that is mostly unrelated filler. question asks where one of '
                        'Mary, John, Sandra or Daniel is. Return places: the rooms that person went to in chunk, in order, '
                        'each as the one word the sentence names; empty when chunk says nothing about that person.'),
        'follow_on': ({'trail': TRAIL_T, 'chunk': 'string', 'question': 'string'}, TRAIL_T,
                      'trail holds the rooms the person in question went to in the earlier parts of a long text. Read chunk, '
                      'the next part (mostly unrelated filler), and return the updated places: the same rooms followed by the '
                      'rooms that person goes to in chunk, in order.')},
    instructions=[
        'question asks where one person is, and chunks are the parts of a long text with a short story hidden in it. Carry '
        'only that person\'s typed trail of rooms with iterateOn:\n' + _iterate_steps('start_trail', 'follow_on', 'chunks', ', question', 'trail') +
        '4. Return the last element of `final.trail.places`. ' + HAY_ANSWER,
        'Follow one person through the long text in chunks. start_trail gives the rooms that person visits in chunks[0], and '
        'follow_on appends the rooms from each next chunk; run it with iterateOn (measure: chunks not yet read) and return '
        'the last room of the final trail. ' + HAY_ANSWER,
    ],
    smoke_code='const start = { trail: await start_trail(chunks[0], question), next: 1 };\n'
               'const final = await iterateOn(async s => ({ trail: await follow_on(s.trail, chunks[s.next], question), '
               'next: s.next + 1 }), start).withMeasure(s => chunks.length - s.next).until(s => s.next >= chunks.length);\n'
               'const answer = final.trail.places[final.trail.places.length - 1];',
    smoke_children={'start_trail': {'places': ['hallway']}, 'follow_on': {'places': ['hallway']}})

HAY_GREP = Program(
    name='haystack-grep-then-verify', root='answer_from_grepped_sentences', args=HAY_Q, returns='string', depth=2, producers=[],
    helpers={'is_story_fact': ({'sentence': 'string'}, 'boolean',
                               'Return true only when sentence is a short fact about one of Mary, John, Sandra or Daniel '
                               'moving to a room or taking, picking up, dropping or leaving an object. A sentence about a '
                               'real person, a place or an event of the world is not one.'),
             'settle_facts': SETTLE_LIST},
    instructions=[
        '1. In eval split every chunk into sentences with `chunk.split(/(?<=[.!?])\\s+/)`, in text order, and keep the '
        'sentences that start with Mary, John, Sandra or Daniel followed by a space.\n2. Call `await is_story_fact(sentence)` for '
        'each kept sentence and keep those that return true, in order.\n3. Return `await settle_facts(facts, question)`. '
        + HAY_ANSWER,
        'Search the text with code before reading: split the chunks into sentences, keep those starting with one of the four '
        'names, and let is_story_fact confirm which really belong to the hidden story. Hand the confirmed sentences, in order, '
        'to settle_facts together with question. ' + HAY_ANSWER,
    ],
    smoke_code='const candidates = [];\nfor (const chunk of chunks) for (const sentence of chunk.split(/(?<=[.!?])\\s+/)) '
               'if (/^(Mary|John|Sandra|Daniel) /.test(sentence)) candidates.push(sentence);\nconst facts = [];\n'
               'for (const sentence of candidates.slice(0, 4)) if (await is_story_fact(sentence)) facts.push(sentence);\n'
               'const answer = await settle_facts(facts, question);',
    smoke_children={'is_story_fact': True})

LEDGER_T = '{ person: string; moves: string[] }'
HAY_LEDGERS = Program(
    name='haystack-person-ledgers', root='answer_from_person_ledgers', args=HAY_Q, returns='string', depth=2,
    producers=['ledger'], typed=True, relay=True,
    helpers={'story_people': ({'chunk': 'string'}, 'string[]',
                              'chunk is one part of a long text that is mostly unrelated filler. Return the names (Mary, John, '
                              'Sandra, Daniel) of the people that story sentences in chunk say moved somewhere; empty when none.'),
             'ledger': ({'person': 'string', 'chunks': 'string[]'}, f'Neuralese<{LEDGER_T}>',
                        'chunks are the parts of a long text mostly made of unrelated filler. Return the typed private ledger of '
                        'person: moves lists, word for word and in text order, every sentence saying where person went.'),
             'settle_ledgers': ({'ledgers': f'Neuralese<{LEDGER_T}>[]', 'question': 'string'}, 'string',
                                'ledgers hold the typed private moves of each person in a long text, one ledger per person. '
                                'Answer question, which asks where one person is now: the room of the last move in that '
                                'person\'s ledger. ' + HAY_ANSWER)},
    instructions=[
        'question asks where one person is. chunks are the parts of a long text with a short story hidden in it.\n1. Call '
        '`await story_people(chunk)` for every chunk and collect the distinct names in eval, in first-seen order.\n2. For each '
        'name call `await ledger(name, chunks)` and keep the ledgers in that order.\n3. Return `await settle_ledgers(ledgers, '
        'question)`. ' + HAY_ANSWER,
        'Work person by person instead of chunk by chunk. First find which people act in the text (story_people on each '
        'chunk, distinct names in code), then have ledger write the typed private move list of each of them from all the chunks, '
        'and let settle_ledgers answer where the person in question is. ' + HAY_ANSWER,
    ],
    smoke_code='const names = [];\nfor (const chunk of chunks) for (const name of await story_people(chunk)) if (!names.includes(name)) names.push(name);\n'
               'const ledgers = [];\nfor (const name of names.slice(0, 3)) ledgers.push(await ledger(name, chunks));\n'
               'const answer = await settle_ledgers(ledgers, question);',
    smoke_children={'story_people': ['Mary'], 'ledger': {'person': 'Mary', 'moves': ['Mary moved to the hallway.']}})

HAY_BACKWARD = Program(
    name='haystack-backward-scan', root='scan_back_for_the_person', args=HAY_Q, returns='string', depth=2, producers=[],
    iterate=True,
    helpers={'last_move': ({'person': 'string', 'chunk': 'string'}, 'string',
                           'chunk is one part of a long text that is mostly unrelated filler. Return the room that person '
                           'last went to according to a sentence of chunk (the one word the sentence names), or an empty '
                           'string when chunk says nothing about where person went.')},
    instructions=[
        'question asks where one person is: `person` is the name after "Where is" (without the question mark). The newest '
        'sentence about that person decides, so read the chunks from the last to the first and stop at the first answer.\n'
        '1. Start from `{ place: "", next: chunks.length - 1 }`.\n2. A step returns `{ place: await last_move(person, chunks[s.next]), '
        'next: s.next - 1 }`.\n3. Run `iterateOn(step, start).withMeasure(s => s.next + 1).until(s => s.place !== "" || s.next < 0)` '
        'in one eval.\n4. Return `final.place`. ' + HAY_ANSWER,
        'Scan the long text backwards for the person asked about in question (the name after "Where is"): last_move(person, '
        'chunk) gives the room of the last sentence about them in a chunk. Step with iterateOn from the final chunk towards '
        'the first, measuring the chunks left, and stop as soon as a room is found. Return it. ' + HAY_ANSWER,
    ],
    smoke_code='const person = question.replace("Where is ", "").replace("?", "");\n'
               'const start = { place: "", next: chunks.length - 1 };\n'
               'const final = await iterateOn(async s => ({ place: await last_move(person, chunks[s.next]), next: s.next - 1 }), start)'
               '.withMeasure(s => s.next + 1).until(s => s.place !== "" || s.next < 0);',
    smoke_children={'last_move': 'hallway'})

HAY_VERIFY = Program(
    name='haystack-answer-verify-retry', root='answer_then_check', args=HAY_Q, returns='string', depth=2, producers=[],
    helpers={'answer_from_text': (HAY_Q, 'string',
                                  'chunks are the parts of a long text with ' + STORY + '. Answer question from that story. '
                                  + HAY_ANSWER),
             'story_supports': ({'chunks': 'string[]', 'question': 'string', 'answer': 'string'}, 'boolean',
                                'chunks are the parts of a long text with ' + STORY + '. Return true only when following that '
                                'story in order really makes answer the correct answer to question.'),
             'answer_again': ({'chunks': 'string[]', 'question': 'string', 'rejected': 'string'}, 'string',
                              'chunks are the parts of a long text with ' + STORY + '. The answer rejected to question was '
                              'checked and is wrong. Follow the story sentence by sentence and give the correct answer. '
                              + HAY_ANSWER)},
    instructions=[
        '1. `const first = await answer_from_text(chunks, question);`\n2. `const ok = await story_supports(chunks, question, '
        'first);`\n3. In eval return `first` when `ok` is true; otherwise return `await answer_again(chunks, question, first)`. '
        + HAY_ANSWER,
        'Answer, check, and only then retry: answer_from_text gives a first answer from the whole text, story_supports checks '
        'it against the story, and when the check fails answer_again redoes the work knowing the first answer was wrong. '
        + HAY_ANSWER,
    ],
    smoke_code='const first = await answer_from_text(chunks, question);\nconst ok = await story_supports(chunks, question, first);\n'
               'const answer = ok ? first : await answer_again(chunks, question, first);',
    smoke_children={'answer_from_text': 'x', 'story_supports': False})

HAY_COUNT = Program(
    name='haystack-count-guided-spotting', root='count_then_spot', args=HAY_Q, returns='string', depth=2, producers=[],
    helpers={'story_count': ({'chunk': 'string'}, 'number',
                             'chunk is one part of a long text that is mostly unrelated filler. Return how many of its '
                             'sentences say where Mary, John, Sandra or Daniel went or what one of them took, picked up, dropped '
                             'or left (0 when none).'),
             'spot_exact': ({'chunk': 'string', 'count': 'number'}, 'string[]',
                            'chunk is one part of a long text that is mostly unrelated filler and holds exactly count sentences '
                            'about where Mary, John, Sandra or Daniel went or what one of them took, picked up, dropped or left. '
                            'Return those sentences word for word, in order.'),
             'settle_facts': SETTLE_LIST},
    instructions=[
        '1. Call `await story_count(chunk)` for every chunk.\n2. For each chunk with a count above 0 call `await spot_exact(chunk, '
        'count)`; when the returned list does not have exactly `count` sentences call it once more and keep that second result.\n'
        '3. Join the lists in text order and return `await settle_facts(facts, question)`. ' + HAY_ANSWER,
        'First count the story sentences of each chunk with story_count, then extract exactly that many with spot_exact for the '
        'chunks that have any, checking the list length in code and retrying a wrong length once. Give the ordered sentences '
        'and question to settle_facts. ' + HAY_ANSWER,
    ],
    smoke_code='const counts = [];\nfor (const chunk of chunks) counts.push(await story_count(chunk));\nconst facts = [];\n'
               'for (const [i, chunk] of chunks.entries()) {\n  if (counts[i] <= 0) continue;\n'
               '  let found = await spot_exact(chunk, counts[i]);\n  if (found.length !== counts[i]) found = await spot_exact(chunk, counts[i]);\n'
               '  facts.push(...found);\n}\nconst answer = await settle_facts(facts, question);',
    smoke_children={'story_count': 1, 'spot_exact': ['Mary moved to the hallway.']})

HAY_WINDOWS = Program(
    name='haystack-window-readers', root='answer_from_overlapping_windows', args=HAY_Q, returns='string', depth=2,
    producers=['read_window'], typed=True, relay=True,
    helpers={'read_window': ({'window': 'string', 'question': 'string'}, f'Neuralese<{FACTS_T}>',
                             'window is two neighbouring parts of a long text that is mostly unrelated filler. List as typed '
                             'private facts every sentence of window that says where Mary, John, Sandra or Daniel went or what '
                             'one of them took, picked up, dropped or left, word for word and in order.'),
             'settle_windows': ({'finds': f'Neuralese<{FACTS_T}>[]', 'question': 'string'}, 'string',
                                'finds are typed private facts of overlapping windows of a long text, in text order; '
                                'neighbouring windows share one part, so the same sentence can occur in two lists. Count each '
                                'sentence once, follow each person and object through the story (a later fact replaces an '
                                'earlier one) and answer question. ' + HAY_ANSWER)},
    instructions=[
        '1. In eval build the windows: for each i from 0 to `chunks.length - 2`, `chunks[i] + "\\n\\n" + chunks[i + 1]` (each chunk '
        'is read twice, with its predecessor and its successor).\n2. Call `await read_window(window, question)` for every window, '
        'in order, and collect the results.\n3. Return `await settle_windows(finds, question)`. ' + HAY_ANSWER,
        'Read the long text through overlapping windows, each made of two neighbouring chunks (built in code), so that no story '
        'sentence depends on one chunk boundary. read_window returns the typed private facts of a window; give all of them, '
        'in order, to settle_windows with question. ' + HAY_ANSWER,
    ],
    smoke_code='const windows = [];\nfor (let i = 0; i < chunks.length - 1; i++) windows.push(chunks[i] + "\\n\\n" + chunks[i + 1]);\n'
               'const finds = [];\nfor (const window of windows.slice(0, 3)) finds.push(await read_window(window, question));\n'
               'const answer = await settle_windows(finds, question);',
    smoke_children={'read_window': {'facts': ['Mary moved to the hallway.']}})

HAY_NEW = (HAY_REPLAY, HAY_TRAIL, HAY_GREP, HAY_LEDGERS, HAY_BACKWARD, HAY_VERIFY, HAY_COUNT, HAY_WINDOWS)
HAY_PERSON_ONLY = {HAY_TRAIL.name, HAY_LEDGERS.name, HAY_BACKWARD.name}  # the question names one person
HAY_HARDER = {HAY_REPLAY.name, HAY_LEDGERS.name, HAY_COUNT.name}


def hay_notes(program, item, fact_pattern):
    """Checked notes: the story sentences the pattern finds in the chunk or window, or in the whole text for one person."""
    chunks = item['chunks']
    if program is HAY_WINDOWS:
        return [child('read_window', [_key(chunks[i] + '\n\n' + chunks[i + 1])], {'facts': fact_pattern.findall(chunks[i] + ' ' + chunks[i + 1])},
                      'checked') for i in range(len(chunks) - 1)]
    if program is HAY_LEDGERS:
        people = []
        for sentence in item['facts']:
            event = parse_event(sentence)
            if event and event[1] == 'move' and event[0] not in people:
                people.append(event[0])
        out = []
        for person in people:
            moves = [s for s in item['facts'] if (parse_event(s) or ('',))[0] == person and parse_event(s)[1] == 'move']
            out.append(child('ledger', [string_key('person', person)], {'person': person, 'moves': moves}, 'checked'))
        return out
    return []


def grep_eligible(item):
    """Every story sentence must survive the code pre-filter of the grep program (a sentence of its own that starts with a name)."""
    sentences = []
    for chunk in item['chunks']:
        sentences.extend(s for s in re.split(r'(?<=[.!?])\s+', chunk) if re.match(r'^(Mary|John|Sandra|Daniel) ', s))
    return all(f in sentences for f in item['facts'])


def count_eligible(item):
    """The per-chunk counts of story sentences must be exact in the chunks the generator wrote."""
    return all(len(parse_chunk(chunk)) == len(chunk_facts(chunk)) for chunk in item['chunks'])


def parse_chunk(chunk):
    return [s for s in re.split(r'(?<=[.!?])\s+', chunk) if parse_event(s) is not None]


def chunk_facts(chunk):
    return re.findall(r'(?<![A-Za-z])(?:Mary|John|Sandra|Daniel) (?:went|moved|travelled|journeyed|got|took|picked|grabbed|dropped|left|put|discarded)\b[^.]*\.', chunk)


# ================================================================= NarrativeQA

NQ_NOTE_T = '{ found: boolean; quote: string }'
NQ_QUESTION = {'question': 'string', 'passage': 'string'}
NQ_FIND = ({'question': 'string', 'passage': 'string'}, f'Neuralese<{NQ_NOTE_T}>',
           'passage is one part of the plot summary of a novel or film. Say as a typed private note whether passage answers '
           'question (found) and quote the words that answer it (quote), or leave quote empty.')
NQ_CONCLUDE = ({'question': 'string', 'finds': f'Neuralese<{NQ_NOTE_T}>[]'}, 'string',
               'finds are the typed private notes of every part of a story summary for question, in order. Answer question '
               'with the words the summary uses, as ' + SHORT + '.')
CAND_T = '{ answer: string; quote: string }[]'

NQ_ROUTED = Program(
    name='story-kind-routed-candidates', root='answer_by_question_kind', args={'question': 'string', 'passages': 'string[]'},
    returns='string', depth=2, producers=[],
    helpers={'question_kind': ({'question': 'string'}, '"person" | "place" | "other"',
                               'question asks about a story. Decide whether the answer is the name of a character (person), '
                               'a place, or something else (an object, an event, a reason, a time, a number or a description).'),
             'name_candidates': (NQ_QUESTION, CAND_T,
                                 'passage is one part of the plot summary of a story. List the names of characters in passage '
                                 'that could answer question, each with the quoted words that show it; empty when none.'),
             'place_candidates': (NQ_QUESTION, CAND_T,
                                  'passage is one part of the plot summary of a story. List the places in passage that could '
                                  'answer question, each with the quoted words that show it; empty when none.'),
             'phrase_candidates': (NQ_QUESTION, CAND_T,
                                   'passage is one part of the plot summary of a story. List the short phrases in passage '
                                   '(an object, an event, a reason, a time, a number) that could answer question, each with the '
                                   'quoted words that show it; empty when none.')},
    instructions=[
        '1. `const kind = (await decide(question_kind, question)).value;`\n2. Pick the finder for the kind: `name_candidates` for '
        '"person", `place_candidates` for "place", `phrase_candidates` otherwise, and call it as `finder(question, passage)` for '
        'every passage, collecting all candidates in one array.\n3. In eval count each distinct answer ignoring case and return the '
        'most frequent one (the first seen on a tie) as ' + SHORT + '.',
        'Route by the kind of question: the question_kind readout says person, place or other, and each kind has its own '
        'candidate finder (name_candidates, place_candidates, phrase_candidates). Run the matching finder over every passage, count '
        'the distinct candidate answers in code, and return the most frequent as ' + SHORT + '.',
    ],
    smoke_code='const kind = (await decide(question_kind, question)).value;\nconst found = [];\n'
               'for (const passage of passages.slice(0, 3)) {\n'
               '  found.push(...(kind === "person" ? await name_candidates(question, passage) : kind === "place" ? '
               'await place_candidates(question, passage) : await phrase_candidates(question, passage)));\n}',
    smoke_children={'question_kind': 'person', 'name_candidates': [{'answer': 'x', 'quote': 'q'}]})
_decision('story-kind-routed-candidates', 'question_kind')

EVENTS_T = '{ actor: string; did: string }[]'
NQ_EVENTS = Program(
    name='story-who-did-what', root='answer_from_story_events', args={'question': 'string', 'passages': 'string[]'},
    returns='string', depth=2, producers=['events_in'], typed=True, relay=True,
    helpers={'events_in': ({'passage': 'string'}, f'Neuralese<{EVENTS_T}>',
                           'passage is one part of the plot summary of a story. Return the typed private events of passage in '
                           'order: for each, actor is the character who does it and did says what happens, in the summary\'s '
                           'own words.'),
             'answer_from_events': ({'question': 'string', 'events': f'Neuralese<{EVENTS_T}>[]'}, 'string',
                                    'events are the typed private events of every part of a story summary, in story order. '
                                    'Answer question about the story from them, using the summary\'s own words, as ' + SHORT + '.')},
    instructions=[
        '1. Call `await events_in(passage)` for every passage in passages, in order, and keep the results (they do not depend '
        'on the question).\n2. Return `await answer_from_events(question, events)`, the answer as ' + SHORT + '.',
        'Turn the story summary into who-did-what events first: events_in(passage) returns the typed private events of one passage. '
        'Collect them for all passages in order and let answer_from_events answer question from them, as ' + SHORT + '.',
    ],
    smoke_code='const events = [];\nfor (const passage of passages.slice(0, 3)) events.push(await events_in(passage));\n'
               'const answer = await answer_from_events(question, events);',
    smoke_children={'events_in': [{'actor': 'x', 'did': 'y'}]})

TIMELINE_T = '{ events: string[] }'
NQ_TIMELINE = Program(
    name='story-event-timeline', root='answer_from_event_timeline', args={'passages': 'string[]', 'question': 'string'},
    returns='string', depth=2, producers=['open_timeline', 'extend_timeline'], typed=True, relay=True, iterate=True,
    helpers={'open_timeline': ({'passage': 'string', 'question': 'string'}, f'Neuralese<{TIMELINE_T}>',
                               'passage is the first part of the plot summary of a story. Return the typed private timeline for '
                               'question: events lists, in order and in the summary\'s own words, the sentences of passage that '
                               'bear on question.'),
             'extend_timeline': ({'timeline': f'Neuralese<{TIMELINE_T}>', 'passage': 'string', 'question': 'string'},
                                 f'Neuralese<{TIMELINE_T}>',
                                 'timeline holds the events bearing on question from the earlier parts of a story summary. Read '
                                 'passage, the next part, and return the updated timeline: keep the earlier events and append the '
                                 'sentences of passage that bear on question; drop an event the passage shows to be wrong.'),
             'answer_from_timeline': ({'timeline': f'Neuralese<{TIMELINE_T}>', 'question': 'string'}, 'string',
                                      'timeline holds the events of a whole story summary that bear on question, in order. '
                                      'Answer question from it with the summary\'s own words, as ' + SHORT + '.')},
    instructions=[
        'passages are the consecutive parts of one story summary. Build a typed timeline of the events that matter, one passage '
        'per iterateOn step:\n' + _iterate_steps('open_timeline', 'extend_timeline', 'passages', ', question', 'timeline') +
        '4. Return `await answer_from_timeline(final.timeline, question)`, the answer as ' + SHORT + '.',
        'open_timeline starts a typed private timeline of the events bearing on question from passages[0]; extend_timeline adds each '
        'next passage (drive it with iterateOn, measure: passages not yet read); answer_from_timeline then answers from the last '
        'timeline, as ' + SHORT + '.',
    ],
    smoke_code='const start = { timeline: await open_timeline(passages[0], question), next: 1 };\n'
               'const final = await iterateOn(async s => ({ timeline: await extend_timeline(s.timeline, passages[s.next], question), '
               'next: s.next + 1 }), start).withMeasure(s => passages.length - s.next).until(s => s.next >= passages.length);\n'
               'const answer = await answer_from_timeline(final.timeline, question);',
    smoke_children={'open_timeline': {'events': ['e1']}, 'extend_timeline': {'events': ['e1']}})

NQ_SORT = Program(
    name='story-score-sort-reread', root='answer_from_top_passages', args={'question': 'string', 'passages': 'string[]'},
    returns='string', depth=2, producers=[],
    helpers={'relevance': (NQ_QUESTION, 'number',
                           'passage is one part of the plot summary of a novel or film. Return a score from 0 to 10 for how much '
                           'passage tells about what question asks: 10 when it contains the answer, 0 when it is unrelated.'),
             'answer_from': ({'question': 'string', 'passages': 'string[]'}, 'string',
                             'passages are the most relevant parts of a story summary, in story order. Answer question from '
                             'them with the summary\'s own words, as ' + SHORT + '.')},
    instructions=[
        '1. Score every passage with `await relevance(question, passage)`.\n2. In eval sort the passage positions by score from '
        'high to low (the original order among equal scores), keep the best three, and put them back in story order.\n3. Return '
        '`await answer_from(question, kept)` with the kept passages as an array of strings, the answer as ' + SHORT + '.',
        'Rate each passage with relevance, keep the three best-scoring ones in code, restore their story order, and let '
        'answer_from answer question from just those. Return it as ' + SHORT + '.',
    ],
    smoke_code='const scores = [];\nfor (const passage of passages) scores.push(await relevance(question, passage));\n'
               'const top = scores.map((score, i) => ({ score, i })).sort((a, b) => b.score - a.score || a.i - b.i).slice(0, 3)'
               '.sort((a, b) => a.i - b.i);\nconst kept = top.map(t => passages[t.i]);\nconst answer = await answer_from(question, kept);',
    smoke_children={'relevance': 5, 'answer_from': 'x'})

NQ_VERIFY = Program(
    name='story-propose-count-support', root='answer_then_count_support', args={'question': 'string', 'passages': 'string[]'},
    returns='string', depth=2, producers=[],
    helpers={'propose_answer': ({'question': 'string', 'passages': 'string[]'}, 'string',
                                'passages are the consecutive parts of a story summary. Answer question as ' + SHORT + ', in the '
                                'summary\'s own words.'),
             'states_it': ({'question': 'string', 'answer': 'string', 'passage': 'string'}, 'boolean',
                           'passage is one part of a story summary. Return true only when passage itself states answer as the '
                           'answer to question.'),
             'propose_again': ({'question': 'string', 'passages': 'string[]', 'rejected': 'string'}, 'string',
                               'passages are the consecutive parts of a story summary. No passage states rejected as the answer '
                               'to question. Find the passage that does answer it and give its answer as ' + SHORT + '.')},
    instructions=[
        '1. `const first = await propose_answer(question, passages);`\n2. Count in eval how many passages return true from '
        '`await states_it(question, first, passage)`.\n3. When the count is at least 1 return `first`; otherwise return '
        '`await propose_again(question, passages, first)`, the answer as ' + SHORT + '.',
        'Answer first, then test the answer passage by passage: states_it says whether one passage states it. If at least one '
        'passage does, keep the answer; if none does, ask propose_again for a new answer knowing the first was not stated. '
        'Return it as ' + SHORT + '.',
    ],
    smoke_code='const first = await propose_answer(question, passages);\nlet support = 0;\n'
               'for (const passage of passages) if (await states_it(question, first, passage)) support += 1;\n'
               'const answer = support >= 1 ? first : await propose_again(question, passages, first);',
    smoke_children={'propose_answer': 'x', 'states_it': False, 'propose_again': 'y'})

NQ_ENTITIES = Program(
    name='story-entity-filter-read', root='answer_from_entity_passages', args={'question': 'string', 'passages': 'string[]'},
    returns='string', depth=2, producers=['find_in_passage'], typed=True, relay=True,
    helpers={'question_entities': ({'question': 'string'}, 'string[]',
                                   'question asks about a story. Return the names of the characters, places and other named '
                                   'things it mentions, exactly as written in question; empty when it names none.'),
             'find_in_passage': NQ_FIND, 'conclude': NQ_CONCLUDE},
    instructions=[
        '1. `const entities = await question_entities(question);`\n2. In eval keep the passages whose lower-cased text contains the '
        'lower-cased form of some entity of at least 3 characters (every passage when none does), in order.\n3. Call '
        '`await find_in_passage(question, passage)` on each kept passage and keep the results.\n4. Return `await '
        'conclude(question, finds)`, the answer as ' + SHORT + '.',
        'Retrieve by name: question_entities lists what the question names, and code keeps the passages that mention one of '
        'them (all passages when none does). Read those with find_in_passage, then let conclude answer from the typed notes, '
        'as ' + SHORT + '.',
    ],
    smoke_code='const entities = await question_entities(question);\nconst kept = passages.filter(p => entities.some(e => e.length >= 3 '
               '&& p.toLowerCase().includes(e.toLowerCase())));\nconst finds = [];\n'
               'for (const passage of (kept.length ? kept : passages).slice(0, 3)) finds.push(await find_in_passage(question, passage));\n'
               'const answer = await conclude(question, finds);',
    smoke_children={'question_entities': ['x'], 'find_in_passage': {'found': True, 'quote': 'q'}})

STEP_T = '{ step: string; answer: string }'
NQ_DECOMPOSE = Program(
    name='story-question-decomposed', root='answer_step_by_step', args={'question': 'string', 'passages': 'string[]'},
    returns='string', depth=2, producers=[],
    helpers={'split_question': ({'question': 'string'}, 'string[]',
                                'question asks about a story. Return one to three sub-questions, in the order they must be '
                                'answered, whose last answer is the answer to question. A simple question is its own only step.'),
             'answer_step': ({'step': 'string', 'passages': 'string[]', 'earlier': f'{STEP_T}[]'}, 'string',
                             'passages are the parts of a story summary. earlier lists the steps already answered, with their '
                             'answers. Answer step from the passages (using earlier where step refers to it) as ' + SHORT + '.'),
             'combine_steps': ({'question': 'string', 'steps': f'{STEP_T}[]'}, 'string',
                               'steps are the answered sub-questions of the story question question, in order. Return the answer '
                               'to question as ' + SHORT + '.')},
    instructions=[
        '1. `const parts = (await split_question(question)).slice(0, 3);`\n2. In eval go through `parts` in order and for each call '
        '`await answer_step(part, passages, steps)`, where `steps` is the array of `{ step, answer }` records answered so far; '
        'push the new record.\n3. Return `await combine_steps(question, steps)`, the answer as ' + SHORT + '.',
        'Decompose the story question with split_question (at most three sub-questions), answer them one after another with '
        'answer_step, passing the records of the earlier steps each time, and have combine_steps return the final answer to '
        'question, as ' + SHORT + '.',
    ],
    smoke_code='const parts = (await split_question(question)).slice(0, 3);\nconst steps = [];\n'
               'for (const part of parts) steps.push({ step: part, answer: await answer_step(part, passages, steps) });\n'
               'const answer = await combine_steps(question, steps);',
    smoke_children={'split_question': ['a', 'b'], 'answer_step': 'x'})

NQ_VOTE = Program(
    name='story-answers-vote', root='vote_over_passages', args={'question': 'string', 'passages': 'string[]'},
    returns='string', depth=2, producers=[],
    helpers={'answer_in': (NQ_QUESTION, 'string',
                           'passage is one part of the plot summary of a novel or film. Return the answer to question that passage '
                           'itself gives as ' + SHORT + ', in the summary\'s own words, or an empty string when it gives none.')},
    instructions=[
        '1. Call `await answer_in(question, passage)` for every passage, in order.\n2. In eval ignore the empty answers, count each '
        'distinct answer ignoring case and surrounding spaces, and return the most frequent one (the earliest in the story on a '
        'tie) as ' + SHORT + '.',
        'Ask every part of the summary for its own answer with answer_in, tally the non-empty answers in code and return the most '
        'frequent one (the earliest on a tie), as ' + SHORT + '.',
    ],
    smoke_code='const answers = [];\nfor (const passage of passages) answers.push(await answer_in(question, passage));\n'
               'const counts = {};\nfor (const a of answers.filter(a => a.trim())) counts[a.trim().toLowerCase()] = (counts[a.trim().toLowerCase()] || 0) + 1;',
    smoke_children={'answer_in': 'x'})

NQ_NEW = (NQ_ROUTED, NQ_EVENTS, NQ_TIMELINE, NQ_SORT, NQ_VERIFY, NQ_ENTITIES, NQ_DECOMPOSE, NQ_VOTE)
NQ_HARDER = {NQ_DECOMPOSE.name, NQ_VERIFY.name, NQ_TIMELINE.name}


def nq_notes(program, item):
    """Checked notes for the Neuralese producers of the new narrative programs."""
    answer, out, seen = item['answer'], [], []
    if program is NQ_ENTITIES:
        for passage in item['passages']:
            sentence = answer_sentences(passage, [answer], 320)
            out.append(child('find_in_passage', [_key(passage)], {'found': bool(sentence), 'quote': sentence or ''}, 'checked'))
    elif program is NQ_TIMELINE:
        for index, passage in enumerate(item['passages']):
            sentence = answer_sentences(passage, [answer], 320)
            if sentence and sentence not in seen:
                seen.append(sentence)
            out.append(child('open_timeline' if index == 0 else 'extend_timeline', [_key(passage)], {'events': list(seen[-6:])}, 'checked'))
    return out


# ================================================================= memory of dated conversation sessions

MEM_T = '{ relevant: boolean; facts: string[] }'
MEM_RECALL = ({'question': 'string', 'session': 'string'}, f'Neuralese<{MEM_T}>',
              'session is one dated conversation from a long chat history. Return typed private recall for question: relevant says '
              'whether session says anything that bears on it, and facts lists each such statement with its date, copied from '
              'session.')
MEM_SETTLE = ({'question': 'string', 'recalls': f'Neuralese<{MEM_T}>[]'}, 'string',
              'recalls are the typed private recalls of dated sessions of a chat history for question, oldest first. When they '
              'disagree the newest statement holds. Answer question as ' + SHORT + '.')

MEM_WEIGHTED = Program(
    name='session-recency-weighted-vote', root='weigh_sessions_by_recency', args={'question': 'string', 'sessions': 'string[]'},
    returns='string', depth=2, producers=[],
    helpers={'session_answer': ({'question': 'string', 'session': 'string'}, 'string',
                                'session is one dated conversation from a long chat history. Return what it says in answer to '
                                'question as ' + SHORT + ', copied from session, or an empty string when it says nothing about it.')},
    instructions=[
        'sessions are the dated conversations of a chat history, oldest first.\n1. Call `await session_answer(question, session)` for '
        'every session, in order.\n2. In eval give each non-empty answer the weight `index + 1` (index is the session\'s position, so '
        'newer sessions weigh more), add the weights of equal answers (ignoring case and surrounding spaces) and return the answer '
        'with the largest total (the newer one on a tie), as ' + SHORT + '.',
        'Ask every session what it says about question (session_answer, empty when nothing), then vote in code with recency: the '
        'answer of the session at position i (oldest is 0) weighs i + 1. Return the answer with the largest total weight, as '
        + SHORT + '.',
    ],
    smoke_code='const answers = [];\nfor (const session of sessions) answers.push(await session_answer(question, session));\n'
               'const weight = {};\nanswers.forEach((a, i) => { if (a.trim()) weight[a.trim().toLowerCase()] = (weight[a.trim().toLowerCase()] || 0) + i + 1; });',
    smoke_children={'session_answer': 'x'})

CLAIM_T = '{ stated: boolean; value: string }'
MEM_CONFLICT = Program(
    name='session-conflict-check', root='answer_unless_sessions_conflict', args={'question': 'string', 'sessions': 'string[]'},
    returns='string', depth=2, producers=[],
    helpers={'claim_in': ({'question': 'string', 'session': 'string'}, CLAIM_T,
                          'session is one dated conversation from a long chat history. Return stated true and the value that '
                          'session gives in answer to question (copied from session, a short phrase) when it says something '
                          'about it; stated false and an empty value otherwise.'),
             'resolve_conflict': ({'question': 'string', 'claims': '{ session: number; value: string }[]'}, 'string',
                                  'claims are the values that different sessions of a chat history give for question, each with '
                                  'the position of its session (0 is the oldest). They disagree. Decide which value is current '
                                  'or correct (a newer session usually updates an older one) and return it as ' + SHORT + '.')},
    instructions=[
        '1. Call `await claim_in(question, session)` for every session, in order.\n2. In eval build `claims`: `{ session: <position>, '
        'value }` for each result that is stated with a non-empty value.\n3. When all the values are equal ignoring case and spaces '
        '(and there is at least one), return that value. Otherwise return `await resolve_conflict(question, claims)`, ' + SHORT + '.',
        'Collect what each session claims about question with claim_in, keep the stated ones with their positions, and check in code '
        'whether they all agree. If they do, return the value; if the sessions conflict (or none claims anything) call '
        'resolve_conflict with the claims. Return ' + SHORT + '.',
    ],
    smoke_code='const claims = [];\nfor (const [i, session] of sessions.entries()) {\n  const c = await claim_in(question, session);\n'
               '  if (c.stated && c.value.trim()) claims.push({ session: i, value: c.value });\n}\n'
               'const same = claims.length > 0 && claims.every(c => c.value.trim().toLowerCase() === claims[0].value.trim().toLowerCase());\n'
               'const answer = same ? claims[0].value : await resolve_conflict(question, claims);',
    smoke_children={'claim_in': {'stated': True, 'value': 'x'}, 'resolve_conflict': 'x'})

MEM_TWO_HOP = Program(
    name='session-two-hop-locate-read', root='locate_then_read_sessions', args={'question': 'string', 'sessions': 'string[]'},
    returns='string', depth=2, producers=[],
    helpers={'locate_sessions': ({'question': 'string', 'sessions': 'string[]'}, 'number[]',
                                 'sessions are the dated conversations of a long chat history, oldest first. Return the positions '
                                 '(counting from 0) of the at most two sessions that say something about question.'),
             'read_session': ({'question': 'string', 'session': 'string'}, 'string',
                              'session is one dated conversation. Return what it says in answer to question as ' + SHORT +
                              ', copied from session, or an empty string when it says nothing about it.')},
    instructions=[
        '1. `const picks = await locate_sessions(question, sessions);`\n2. In eval keep the integers of picks that are valid positions '
        'of sessions, each once, at most two, sorted from the oldest to the newest (the last two sessions when none is valid).\n'
        '3. Call `await read_session(question, sessions[i])` for each pick.\n4. Return the last non-empty answer (the newest '
        'session wins), as ' + SHORT + '.',
        'First find where the answer lives: locate_sessions returns the positions of the sessions that matter. Validate and sort '
        'them in code (at most two, oldest first), read each with read_session, and return the newest non-empty answer, as '
        + SHORT + '.',
    ],
    smoke_code='const picks = await locate_sessions(question, sessions);\nconst chosen = [...new Set(picks.filter(i => Number.isInteger(i) && '
               'i >= 0 && i < sessions.length))].sort((a, b) => a - b).slice(0, 2);\nconst answers = [];\n'
               'for (const i of chosen) answers.push(await read_session(question, sessions[i]));',
    smoke_children={'locate_sessions': [0, 1], 'read_session': 'x'})

MEM_SUMMARIES = Program(
    name='session-summaries-bottleneck', root='answer_from_session_summaries', args={'question': 'string', 'sessions': 'string[]'},
    returns='string', depth=2, producers=['summarize_session'], relay=True,
    helpers={'summarize_session': ({'session': 'string'}, 'Neuralese<string>',
                                   'session is one dated conversation from a long chat history. Write a private summary of it: its '
                                   'date and every concrete statement people make about themselves, their plans, preferences, '
                                   'numbers and events, in their own words.'),
             'answer_from_summaries': ({'question': 'string', 'summaries': 'Neuralese<string>[]'}, 'string',
                                       'summaries are the private summaries of the dated sessions of a chat history, oldest first. '
                                       'Answer question from them (a newer statement replaces an older one) as ' + SHORT + '.')},
    instructions=[
        '1. Call `await summarize_session(session)` for every session, in order (the summaries do not depend on the question).\n'
        '2. Return `await answer_from_summaries(question, summaries)`, the answer as ' + SHORT + '.',
        'Compress the chat history before asking: summarize_session writes a private summary of each dated session, and '
        'answer_from_summaries answers question from all the summaries together. Return the answer as ' + SHORT + '.',
    ],
    smoke_code='const summaries = [];\nfor (const session of sessions.slice(0, 3)) summaries.push(await summarize_session(session));\n'
               'const answer = await answer_from_summaries(question, summaries);',
    smoke_children={'summarize_session': 'summary'})

MEM_LEDGER = Program(
    name='session-topic-ledger', root='look_up_in_topic_ledger', args={'question': 'string', 'sessions': 'string[]'},
    returns='string', depth=2, producers=[],
    helpers={'list_facts': ({'session': 'string'}, '{ topic: string; value: string }[]',
                            'session is one dated conversation. List every concrete fact people state in it as a topic (a short '
                            'noun phrase such as "favourite food" or "daughter\'s school") and its value, a short phrase copied '
                            'from the session.'),
             'topic_asked': ({'question': 'string', 'topics': 'string[]'}, 'string',
                             'topics are the topics of facts stated in a chat history. Return the one topic that question asks '
                             'about, written exactly as in topics; an empty string when none fits.')},
    instructions=[
        '1. Call `await list_facts(session)` for every session from the oldest to the newest.\n2. In eval build `ledger` (lower-cased '
        'topic to value; a newer session overwrites an older value) and `topics` (the distinct topics as first written).\n'
        '3. `const topic = await topic_asked(question, topics);`\n4. Return `ledger[topic.toLowerCase()]` (an empty string when '
        'missing), the value as ' + SHORT + '.',
        'Keep a ledger of the chat history in code: list_facts gives the typed topic and value facts of each session, and each newer '
        'value overwrites the older one for its topic. Then ask topic_asked which topic question is about and look the value up in the '
        'ledger. Return it as ' + SHORT + '.',
    ],
    smoke_code='const ledger = {};\nconst topics = [];\nfor (const session of sessions) for (const f of await list_facts(session)) {\n'
               '  const key = f.topic.toLowerCase();\n  if (!(key in ledger)) topics.push(f.topic);\n  ledger[key] = f.value;\n}\n'
               'const topic = await topic_asked(question, topics);\nconst answer = ledger[topic.toLowerCase()] ?? "";',
    smoke_children={'list_facts': [{'topic': 'colour', 'value': 'red'}], 'topic_asked': 'colour'})

MEM_NEWEST_FIRST = Program(
    name='session-newest-first-scan', root='scan_back_through_sessions', args={'question': 'string', 'sessions': 'string[]'},
    returns='string', depth=2, producers=[], iterate=True,
    helpers={'check_session': ({'question': 'string', 'session': 'string'}, '{ relevant: boolean; answer: string }',
                               'session is one dated conversation. Return relevant true and the answer it gives to question as '
                               + SHORT + ' (copied from session) when it says something about question; relevant false and an '
                               'empty answer otherwise.')},
    instructions=[
        'The newest statement wins, so read the sessions from the newest to the oldest and stop at the first that answers.\n'
        '1. Start from `{ found: "", next: sessions.length - 1 }`.\n2. A step reads `sessions[s.next]` with `await check_session(question, '
        'sessions[s.next])` and returns `{ found: <its answer when relevant, else "">, next: s.next - 1 }`.\n3. Run it with '
        '`iterateOn(step, start).withMeasure(s => s.next + 1).until(s => s.found !== "" || s.next < 0)` in one eval.\n'
        '4. Return `final.found`, as ' + SHORT + '.',
        'Scan the chat history backwards with iterateOn: check_session says whether one session answers question and what the '
        'answer is. Start at the newest session, measure the sessions left, and stop at the first answer (or when none is left). '
        'Return it as ' + SHORT + '.',
    ],
    smoke_code='const start = { found: "", next: sessions.length - 1 };\n'
               'const step = async s => { const r = await check_session(question, sessions[s.next]); '
               'return { found: r.relevant ? r.answer : "", next: s.next - 1 }; };\n'
               'const final = await iterateOn(step, start).withMeasure(s => s.next + 1).until(s => s.found !== "" || s.next < 0);',
    smoke_children={'check_session': {'relevant': True, 'answer': 'x'}})

MEM_BATCHES = Program(
    name='session-batches-recall', root='answer_from_session_batches', args={'question': 'string', 'sessions': 'string[]'},
    returns='string', depth=2, producers=['recall_batch'], typed=True, relay=True,
    helpers={'recall_batch': ({'question': 'string', 'batch': 'string[]'}, f'Neuralese<{MEM_T}>',
                              'batch is a few consecutive dated conversations from a long chat history. Return typed private recall '
                              'for question: relevant says whether any of them bears on it, and facts lists each such statement with '
                              'its date, copied, oldest first.'),
             'settle': MEM_SETTLE},
    instructions=[
        '1. In eval cut `sessions` into consecutive batches of four (the last batch may be shorter), oldest first.\n2. Call `await '
        'recall_batch(question, batch)` for every batch, in order, and collect the results.\n3. Return `await settle(question, '
        'recalls)`, the answer as ' + SHORT + '.',
        'Group the dated sessions into consecutive batches of four in code, have recall_batch write the typed private recall of each '
        'batch for question, and give all of them, oldest first, to settle. Return the answer as ' + SHORT + '.',
    ],
    smoke_code='const batches = [];\nfor (let i = 0; i < sessions.length; i += 4) batches.push(sessions.slice(i, i + 4));\n'
               'const recalls = [];\nfor (const batch of batches.slice(0, 3)) recalls.push(await recall_batch(question, batch));\n'
               'const answer = await settle(question, recalls);',
    smoke_children={'recall_batch': {'relevant': True, 'facts': ['x']}})

MEM_NEW = (MEM_WEIGHTED, MEM_CONFLICT, MEM_TWO_HOP, MEM_SUMMARIES, MEM_LEDGER, MEM_NEWEST_FIRST, MEM_BATCHES)
MEM_HARDER = {MEM_CONFLICT.name, MEM_LEDGER.name, MEM_TWO_HOP.name}


def mem_notes(program, item):
    answer, out = item['answer'], []
    if program is MEM_SUMMARIES:
        for session in item['sessions']:
            sentence = answer_sentences(session, [answer], 320)
            if sentence:
                header = session.split('\n', 1)[0].strip()
                out.append(child('summarize_session', [_key(session)], f'{header}\n{sentence}', 'checked'))
    elif program is MEM_BATCHES:
        sessions = item['sessions']
        for start in range(0, len(sessions), 4):
            batch = sessions[start:start + 4]
            facts = [s for s in (answer_sentences(x, [answer], 320) for x in batch) if s]
            out.append(child('recall_batch', [_key(batch[0])], {'relevant': bool(facts), 'facts': facts}, 'checked'))
    return out


# ================================================================= PubMedQA abstracts

SECTION_T = '{ label: string; text: string }'
VERDICT = '"yes" | "no" | "maybe"'
STANCE3 = '"yes" | "no" | "unclear"'
PUB_ARGS = {'question': 'string', 'sections': f'{SECTION_T}[]'}

PUB_VOTE = Program(
    name='abstract-decisive-vote', root='vote_on_decisive_sections', args=PUB_ARGS, returns=VERDICT, depth=2, producers=[],
    helpers={'section_stance': ({'question': 'string', 'section': SECTION_T}, STANCE3,
                                'section is one labelled section of a research abstract and question asks about the study. Say '
                                'whether the section\'s findings answer question affirmatively (yes), negatively (no) or say '
                                'nothing decisive (unclear).')},
    instructions=[
        '1. In eval keep the sections whose label, in upper case, starts with RESULT, CONCLUSION or FINDING (all sections when none '
        'does).\n2. For each kept section get `(await decide(section_stance, question, section)).value`.\n3. Count the "yes" and the '
        '"no" values: return "yes" when yes is larger, "no" when no is larger, and "maybe" when they are equal (also when all are '
        '"unclear").',
        'Only the results and conclusions of an abstract answer the question. Keep those sections in code (every section if none is '
        'labelled so), read each with the section_stance readout, and vote: "yes" if more yes than no, "no" if more no than yes, '
        'otherwise "maybe".',
    ],
    smoke_code='const kept = sections.filter(s => /^(RESULT|CONCLUSION|FINDING)/.test(s.label.toUpperCase()));\nlet yes = 0, no = 0;\n'
               'for (const section of (kept.length ? kept : sections)) {\n  const d = await decide(section_stance, question, section);\n'
               '  if (d.value === "yes") yes += 1; else if (d.value === "no") no += 1;\n}',
    smoke_children={'section_stance': 'yes'})
_decision('abstract-decisive-vote', 'section_stance')

PUB_BACKWARD = Program(
    name='abstract-conclusion-first-scan', root='scan_back_through_abstract', args=PUB_ARGS, returns=VERDICT, depth=2, producers=[],
    iterate=True,
    helpers={'stance_of': ({'question': 'string', 'section': SECTION_T}, STANCE3,
                           'section is one labelled section of a research abstract and question asks about the study. Return '
                           '"yes" when the section\'s findings answer question affirmatively, "no" when negatively, and '
                           '"unclear" when it says nothing decisive (background and methods usually do not).')},
    instructions=[
        'The conclusion decides, so read the sections from the last to the first and stop at the first decisive stance.\n'
        '1. Start from `{ stance: "unclear", next: sections.length - 1 }`.\n2. A step returns `{ stance: await stance_of(question, '
        'sections[s.next]), next: s.next - 1 }`.\n3. Run `iterateOn(step, start).withMeasure(s => s.next + 1).until(s => s.stance !== '
        '"unclear" || s.next < 0)` in one eval.\n4. Return `final.stance` when it is "yes" or "no", otherwise "maybe".',
        'Read the abstract backwards with iterateOn: stance_of gives the stance of one section. Start at the last section, measure the '
        'sections left, and stop at the first section that is not "unclear". Return its "yes" or "no"; when every section is unclear '
        'return "maybe".',
    ],
    smoke_code='const start = { stance: "unclear", next: sections.length - 1 };\n'
               'const final = await iterateOn(async s => ({ stance: await stance_of(question, sections[s.next]), next: s.next - 1 }), start)'
               '.withMeasure(s => s.next + 1).until(s => s.stance !== "unclear" || s.next < 0);',
    smoke_children={'stance_of': 'yes'})

PUB_HEDGE = Program(
    name='abstract-hedging-rule-table', root='decide_with_hedging_rules', args=PUB_ARGS, returns=VERDICT, depth=2, producers=[],
    helpers={'conclusion_stance': ({'question': 'string', 'text': 'string'}, STANCE3,
                                   'text is the conclusion of a research abstract and question asks about the study. Return "yes" '
                                   'when text answers question affirmatively, "no" when negatively, "unclear" when it does not '
                                   'answer it.'),
             'hedging': ({'text': 'string'}, '"firm" | "hedged"',
                         'Decide whether text states its finding firmly or hedges it (may, might, suggests, appears, '
                         'possibly, further research is needed, mixed or limited evidence).')},
    instructions=[
        '1. In eval let `conclusion` be the last section whose label, in upper case, starts with CONCLUSION (the last section when '
        'there is none).\n2. `const stance = await conclusion_stance(question, conclusion.text);` and `const tone = (await '
        'decide(hedging, conclusion.text)).value;`\n3. Apply this rule table in eval: tone "hedged" gives "maybe"; tone "firm" with '
        'stance "yes" gives "yes"; with "no" gives "no"; with "unclear" gives "maybe". Return the result.',
        'Judge the conclusion of the abstract in two independent ways: conclusion_stance says what it answers and the hedging '
        'readout says whether it is firm or hedged. Pick the conclusion section in code and combine the two with a fixed rule '
        'table: any hedged conclusion is "maybe"; a firm one gives its "yes" or "no", and "maybe" when unclear.',
    ],
    smoke_code='const conclusion = [...sections].reverse().find(s => s.label.toUpperCase().startsWith("CONCLUSION")) ?? sections[sections.length - 1];\n'
               'const stance = await conclusion_stance(question, conclusion.text);\nconst tone = (await decide(hedging, conclusion.text)).value;\n'
               'const answer = tone === "hedged" ? "maybe" : stance === "unclear" ? "maybe" : stance;',
    smoke_children={'conclusion_stance': 'yes', 'hedging': 'firm'})
_decision('abstract-hedging-rule-table', 'hedging')

PUB_GIST = Program(
    name='abstract-gist-then-verdict', root='decide_from_abstract_gist', args=PUB_ARGS, returns=VERDICT, depth=2,
    producers=['abstract_gist'], relay=True,
    helpers={'abstract_gist': ({'sections': f'{SECTION_T}[]'}, 'Neuralese<string>',
                      'sections are the labelled sections of one research abstract. Write a private gist of what the study '
                      'found and concluded, in the abstract\'s own words, leaving out background and methods.'),
             'verdict_from_gist': ({'question': 'string', 'gist': 'Neuralese<string>'}, VERDICT,
                                   'gist is a private gist of what a research study found and concluded. Return "yes" when it '
                                   'answers question affirmatively, "no" when negatively, and "maybe" when the findings are mixed, '
                                   'conditional or inconclusive.')},
    instructions=[
        '1. `const note = await abstract_gist(sections);`\n2. Return `await verdict_from_gist(question, note)`: "yes", "no" or "maybe".',
        'Compress the whole abstract first: abstract_gist(sections) writes a private gist of its findings and conclusions. Then let '
        'verdict_from_gist decide from the gist alone whether the study answers question with "yes", "no" or "maybe".',
    ],
    smoke_code='const note = await abstract_gist(sections);\nconst answer = await verdict_from_gist(question, note);',
    smoke_children={'abstract_gist': 'The study found an effect.'})

PUB_DEBATE = Program(
    name='abstract-yes-no-debate', root='weigh_the_two_cases', args=PUB_ARGS, returns=VERDICT, depth=2,
    producers=['case_for_yes', 'case_for_no'], relay=True,
    helpers={'case_for_yes': (PUB_ARGS, 'Neuralese<string>',
                              'sections are the labelled sections of a research abstract. Write the strongest private case, from the '
                              'abstract\'s own statements only, that the answer to question is yes.'),
             'case_for_no': (PUB_ARGS, 'Neuralese<string>',
                             'sections are the labelled sections of a research abstract. Write the strongest private case, from the '
                             'abstract\'s own statements only, that the answer to question is no.'),
             'judge_cases': ({'question': 'string', 'yes_case': 'Neuralese<string>', 'no_case': 'Neuralese<string>'}, VERDICT,
                             'yes_case and no_case are private cases for and against answering question yes, drawn from one '
                             'research abstract. Return "yes" or "no" when one case clearly outweighs the other, and "maybe" when '
                             'they balance or the evidence is inconclusive.')},
    instructions=[
        '1. `const yes_case = await case_for_yes(question, sections);` and `const no_case = await case_for_no(question, sections);`\n'
        '2. Return `await judge_cases(question, yes_case, no_case)`: "yes", "no" or "maybe".',
        'Let two private cases compete: case_for_yes and case_for_no each argue from the abstract in sections for their answer to '
        'question, and judge_cases weighs them and returns "yes", "no" or "maybe".',
    ],
    smoke_code='const yes_case = await case_for_yes(question, sections);\nconst no_case = await case_for_no(question, sections);\n'
               'const answer = await judge_cases(question, yes_case, no_case);',
    smoke_children={'case_for_yes': 'case for yes', 'case_for_no': 'case for no'})

PUB_SENTENCES = Program(
    name='abstract-sentence-decisions', root='count_deciding_sentences', args=PUB_ARGS, returns=VERDICT, depth=2, producers=[],
    helpers={'sentence_verdict': ({'question': 'string', 'sentence': 'string'}, '"yes" | "no" | "none"',
                                  'sentence is one sentence of the results or conclusion of a research abstract and question asks '
                                  'about the study. Say whether the sentence on its own answers question affirmatively (yes), '
                                  'negatively (no) or does not bear on it (none).')},
    instructions=[
        '1. In eval collect the sentences (`text.split(/(?<=[.!?])\\s+/)`) of every section whose label, in upper case, starts with '
        'RESULT or CONCLUSION (of all sections when none does), in order.\n2. For each sentence get `(await decide(sentence_verdict, '
        'question, sentence)).value`.\n3. Count "yes" and "no". Return "yes" when yes is at least 2 and larger than no, "no" when no '
        'is at least 2 and larger than yes, and "maybe" otherwise.',
        'Decide sentence by sentence: code splits the results and conclusion sections into sentences, the sentence_verdict readout '
        'classifies each as yes, no or none, and code counts them. Answer "yes" or "no" only when that side has at least two '
        'sentences and more than the other; otherwise "maybe".',
    ],
    smoke_code='const kept = sections.filter(s => /^(RESULT|CONCLUSION)/.test(s.label.toUpperCase()));\nconst sentences = [];\n'
               'for (const s of (kept.length ? kept : sections)) sentences.push(...s.text.split(/(?<=[.!?])\\s+/));\nlet yes = 0, no = 0;\n'
               'for (const sentence of sentences.slice(0, 4)) {\n  const d = await decide(sentence_verdict, question, sentence);\n'
               '  if (d.value === "yes") yes += 1; else if (d.value === "no") no += 1;\n}',
    smoke_children={'sentence_verdict': 'yes'})
_decision('abstract-sentence-decisions', 'sentence_verdict')

PUB_NEW = (PUB_VOTE, PUB_BACKWARD, PUB_HEDGE, PUB_GIST, PUB_DEBATE, PUB_SENTENCES)
PUB_HARDER = {PUB_HEDGE.name, PUB_SENTENCES.name}


def pub_notes(program, item):
    """Mechanical gist note (provenance `mechanical:decisive-section-overlap`): the sentences of the results and conclusion
    sections that overlap the abstract's own long answer. A `maybe` abstract has no decisive gist, so it gets no note."""
    if program is not PUB_GIST or item['answer'] == 'maybe':
        return []
    parts = []
    for section in item['sections']:
        if section['label'].upper().startswith(('RESULT', 'CONCLUSION', 'FINDING')):
            text = overlap(section['text'], item['long_answer'], 400)
            if text:
                parts.append(text)
    if not parts:
        return []
    return [child('abstract_gist', [_key(item['sections'][0]['text'])], ' '.join(parts), 'mechanical:decisive-section-overlap')]
