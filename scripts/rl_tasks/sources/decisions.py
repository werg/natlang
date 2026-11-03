"""Decision-readout tasks from labelled choice datasets (CLINC150, BANKING77, MASSIVE, TREC, SMS spam, VitaminC,
BoardgameQA). See plans/neuralese/BROAD_RL_TASKS_20261031.md, "Decision-readout families".

Every program has at least one `readout: decision` function whose declared result is a finite literal union, and a
root that acts on what it decides (natlang-native: the judgment is a typed decision; the routing, counting, effects
and combination around it are crisp eval code over named data). Shapes:

  route      one decision gates a lookup: the request's intent picks a team from `routes`, an out-of-scope decision
             goes to "human_review" (abstention is data-labelled: CLINC's own out-of-scope utterances, or a request
             from an intent outside the candidates).
  desk       decisions gate service effects: each ticket is assigned to its intent's team or escalated, each spam
             message is junked, through a `desk`/`inbox` service; the answer is the service's confirmation code of its
             final state (state oracle, as the tau-bench store: verify.mjs materializes it from the gold actions).
  tally      several decisions aggregated in code: per-label counts, the ids of one label, or the most frequent label.
  tree       two chained decisions: scenario then intent (MASSIVE), coarse then fine question type (TREC), each level a
             decision function of its own and the second chosen by the first's answer.
  verdict    a decision over a Neuralese view: a helper writes a private `Neuralese<string>` note on the evidence (claim
             evidence for VitaminC, the game's facts/rules/preferences for BoardgameQA) and the decision reads the note.

Abstention or escalation on a probability floor (`decide(...).confidence`) has no exact gold, so no case grades a
floor; the programs state finite decisions and act on their values. Gold labels come from the datasets. Custody:
CLINC/BANKING77/SMS items carry the inline curriculum's own content ids (`<dataset>:sha256(dataset\\0text\\0)`) and its
content split (ts-host/scripts/inline-curriculum/source-split.mjs, including the test reservations), so the protected
held-set closure relates them exactly; other datasets keep their upstream evaluation splits as held dev/confirmation.
Every item is used in at most one case.
"""
from __future__ import annotations

import collections
import hashlib
import json
import random
import re
from pathlib import Path

import pyarrow.parquet as pq

from ..common import make_case, nl_file, pick, resolve_str

REPO = Path(resolve_str('repo'))
SOURCES = Path(resolve_str('data_hdd', 'natlang-development-data', 'data', 'decision-sources'))
VENDOR = REPO / 'vendor' / 'datasets'
RESERVED = {r['id'] for r in json.loads((REPO / 'training' / 'source-split-reservations.json').read_text())['records']}
TEAMS = ['alpha', 'bravo', 'charlie', 'delta', 'echo', 'foxtrot', 'golf', 'hotel']
LICENSES = {'clinc_oos': 'CC-BY-3.0', 'banking77': 'CC-BY-4.0', 'massive_intent': 'CC-BY-4.0',
            'trec': 'unknown (TREC QC, Li & Roth 2002)', 'sms_spam': 'CC-BY-4.0', 'vitaminc': 'CC-BY-SA-3.0',
            'boardgameqa': 'CC-BY-4.0'}
UPSTREAM = {'clinc_oos': 'clinc/clinc_oos@155b9c71 plus', 'banking77': 'PolyAI banking77@57ec275d',
            'massive_intent': 'mteb/amazon_massive_intent en', 'trec': 'SetFit/TREC-QC', 'sms_spam': 'ucirvine/sms_spam',
            'vitaminc': 'tals/vitaminc', 'boardgameqa': 'tasksource/Boardgame-QA'}
OOS = 'out_of_scope'


def _norm(text):
    return re.sub(r'\s+', ' ', str(text).replace('\\', ' ')).strip()


def _rid(dataset, text):
    return hashlib.sha256(f'{dataset}\0{text}\0'.encode()).hexdigest()


def _content_split(rid):
    """The inline curriculum's content partition (source-split.mjs): reservations and a tenth of ids are test."""
    return 'held' if rid in RESERVED or int(rid[:8], 16) % 10 == 0 else 'train'


def _held_half(key):
    return 'dev' if int(hashlib.sha256(f'decisions-held|{key}'.encode()).hexdigest()[:8], 16) % 2 == 0 else 'confirmation'


def _jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _parquet(path):
    return pq.read_table(path).to_pylist()


def _labelled(dataset, rows):
    """{text, label} rows -> unique items; texts with conflicting labels are dropped (as the inline curriculum)."""
    seen, conflicts = {}, set()
    for text, label in rows:
        text = _norm(text)
        if not text or len(text) > 2000:
            continue
        rid = _rid(dataset, text)
        if rid in seen and seen[rid]['label'] != label:
            conflicts.add(rid)
        seen.setdefault(rid, {'text': text, 'label': label, 'rid': rid, 'group': f'{dataset}:{rid}', 'dataset': dataset})
    return [item for rid, item in sorted(seen.items()) if rid not in conflicts]


def _load():
    """dataset -> {'train': items, 'dev': items, 'confirmation': items, 'held': items (content-split, halves per case)}."""
    out = {}
    clinc = _jsonl(VENDOR / 'clinc_oos/155b9c710419136e17307b80d0a13e68cd46b4ec/plus/train-00000-of-00001.parquet.jsonl')
    banking = _jsonl(VENDOR / 'banking77/57ec275d8078af65b7731c2a98be812d844a6d6b/banking_data/train.csv.jsonl')
    sms = _jsonl(VENDOR / 'sms_spam/cae486f927c250fe1d4a5b55f11357964ed1646c/plain_text/train-00000-of-00001.parquet.jsonl')
    for dataset, rows in (('clinc_oos', [(r['text'], r['intent']) for r in clinc]),
                          ('banking77', [(r['text'], r['category']) for r in banking]),
                          ('sms_spam', [(r['sms'], 'spam' if str(r['label']) in ('1', 'spam') else 'ham') for r in sms])):
        items = [i for i in _labelled(dataset, rows) if re.fullmatch(r'[a-z0-9_]+', i['label'])]
        out[dataset] = {'train': [i for i in items if _content_split(i['rid']) == 'train'],
                        'held': [i for i in items if _content_split(i['rid']) == 'held']}
    for dataset, folder, fields in (('massive_intent', 'mteb__amazon_massive_intent/en', ('text', 'label_text')),
                                    ('trec', 'SetFit__TREC-QC/default', ('text', None))):
        out[dataset] = {}
        for upstream, split in (('train', 'train'), ('validation', 'dev'), ('test', 'held')):
            path = SOURCES / folder / upstream / '0000.parquet'
            if not path.exists():
                continue
            rows = _parquet(path)
            if dataset == 'trec':
                pairs = [(r['text'], f'{r["label_coarse_original"]}|{r["label_text"]}') for r in rows]
            else:
                pairs = [(r[fields[0]], r[fields[1]]) for r in rows]
            items = _labelled(dataset, pairs)
            if dataset == 'massive_intent' and split == 'held':
                split = 'confirmation'
            out[dataset][split] = items
    return out


class Pool:
    """Items drawn without replacement, per label."""

    def __init__(self, items, rng):
        self.by = collections.defaultdict(list)
        for item in items:
            self.by[item['label']].append(item)
        for v in self.by.values():
            rng.shuffle(v)

    def labels(self, minimum=1):
        return sorted(k for k, v in self.by.items() if len(v) >= minimum)

    def take(self, label):
        return self.by[label].pop() if self.by.get(label) else None


# ------------------------------------------------------------------ model-facing text (several genuinely different)

ROUTE = [
    'Classify request with intent. If its answer is one of the intents in routes, return the team routes names for it; '
    'if intent says out_of_scope, return "human_review" so a person handles it.',
    'Decide which of the desk\'s intents request expresses with intent, then return the team that routes assigns to '
    'that intent. When intent finds the request outside every listed intent, return "human_review".',
    'A request came in at the help desk. Classify it with intent; routes says which team handles each intent. Return '
    'the name of that team, or "human_review" if the request matches none of the intents.',
    'Route request: ask intent which intent it is and look the answer up in routes. A request that fits none of the '
    'intents is not routed to a team: return "human_review" for it. Otherwise return the team.',
]
INTENT_CHILD = [
    'Which of these intents does request express? Choose out_of_scope when it expresses none of them.',
    'Classify the customer request into one of the listed intents, or out_of_scope if none applies.',
]
DESK = [
    'For every ticket: ask intent what the ticket wants; a listed intent means desk.assign to the team in routes, '
    'out_of_scope means desk.escalate. Each ticket gets exactly one action. Return desk.finish() at the end.',
    'Route the whole queue through desk. A ticket is assigned (desk.assign) to routes[its intent] when intent finds a '
    'listed intent, and escalated (desk.escalate) otherwise. After the last ticket return desk.finish().',
    'Every ticket in tickets needs routing. Use intent on each ticket\'s text: assign the ticket with desk.assign to '
    'the team routes gives for its intent, and escalate it with desk.escalate when it is out of scope. When every '
    'ticket is handled, return desk.finish().',
    'Work through the queue of tickets. For each one, decide its intent with intent; a ticket whose intent is listed '
    'in routes goes to that team (desk.assign), any other ticket goes to a human (desk.escalate). Handle each ticket '
    'once, then return the confirmation code from desk.finish().',
]
DESK_SERVICE = r'''import { createHash } from 'node:crypto';
const TICKETS: string[] = __TICKETS__;
const TEAMS: string[] = __TEAMS__;
const SALT = __SALT__;
const routed: Record<string, string> = {};
/** Assign a ticket to a team. A later assign or escalate of the same ticket replaces this one. */
export function assign(args: { ticket: string; team: string }): string {
  if (!TICKETS.includes(args.ticket)) return `Error: no ticket ${args.ticket}`;
  if (!TEAMS.includes(args.team)) return `Error: no team ${args.team}; teams: ${TEAMS.join(', ')}`;
  routed[args.ticket] = 'team:' + args.team;
  return `ticket ${args.ticket} assigned to ${args.team}`;
}
/** Send a ticket to human review instead of a team. A later assign of the same ticket replaces this. */
export function escalate(args: { ticket: string }): string {
  if (!TICKETS.includes(args.ticket)) return `Error: no ticket ${args.ticket}`;
  routed[args.ticket] = 'human';
  return `ticket ${args.ticket} escalated to human review`;
}
/** Close the session: the confirmation code of how every ticket was routed. */
export function finish(): string {
  const state = TICKETS.map(id => [id, routed[id] ?? 'open']);
  return 'desk-' + createHash('sha256').update(SALT + JSON.stringify(state)).digest('hex').slice(0, 16);
}
'''
INBOX = [
    'Junk every spam text with inbox.junk and keep every ham text; is_spam decides which is which. Return the '
    'confirmation code of inbox.finish() once all messages are handled.',
    'Go through messages one by one. When is_spam answers spam, call inbox.junk with that message\'s id; otherwise do '
    'nothing to it. Finish with inbox.finish() and return what it gives.',
    'messages is today\'s inbox. Use is_spam on each message and move every spam message to junk with inbox.junk; '
    'leave the rest. Then return inbox.finish().',
    'Clean up the inbox: a message is junked (inbox.junk) exactly when is_spam says it is spam. Leave ham messages '
    'where they are and return the code inbox.finish() gives at the end.',
]
SPAM_CHILD = ['Is this text message spam (unsolicited advertising, prizes, scams) or ham (an ordinary personal or '
              'service message)?', 'Decide whether message is spam or ham.']
INBOX_SERVICE = r'''import { createHash } from 'node:crypto';
const MESSAGES: string[] = __TICKETS__;
const SALT = __SALT__;
const junked = new Set<string>();
/** Move a message to the junk folder. */
export function junk(args: { id: string }): string {
  if (!MESSAGES.includes(args.id)) return `Error: no message ${args.id}`;
  junked.add(args.id);
  return `message ${args.id} moved to junk`;
}
/** Move a junked message back to the inbox. */
export function restore(args: { id: string }): string {
  if (!MESSAGES.includes(args.id)) return `Error: no message ${args.id}`;
  junked.delete(args.id);
  return `message ${args.id} restored`;
}
/** Close the session: the confirmation code of the inbox's final state. */
export function finish(): string {
  return 'inbox-' + createHash('sha256').update(SALT + JSON.stringify(MESSAGES.map(id => [id, junked.has(id)]))).digest('hex').slice(0, 16);
}
'''
TALLY_CHILD = ['Which of these intents does message express?',
               'Classify the message: which listed intent is the sender\'s request?']
TALLY = {
    'counts': [
        'Tally the messages: classify each with intent, then return, for every intent intent can give, how many '
        'messages have it (zero included).',
        'Classify every message in messages with intent and return how many messages have each intent: a record with '
        'one entry per intent intent can answer, including those with zero messages.',
        'Count the messages per intent. Decide each message\'s intent with intent, then return a record from every '
        'intent intent offers to its number of messages (0 when none).',
    ],
    'select': [
        'Find the messages whose intent is {target}: run intent on each and return the matching ids (order does not '
        'matter).',
        'Which messages are about {target}? Decide each message\'s intent with intent and return the ids of the '
        'messages whose intent is {target}, in any order.',
        'Return the ids of all messages in messages that intent classifies as {target}.',
    ],
    'majority': [
        'Which intent is most common in messages? Decide every message\'s intent with intent, count them and return '
        'the intent with the highest count.',
        'Decide the intent of every message with intent and return the intent that the most messages have.',
        'What do most of these messages want? Classify each one with intent and return the most frequent intent.',
    ],
}
TREE = {
    'massive_intent': [
        'Find what utterance asks the assistant to do. First decide its scenario with scenario; then call the intent '
        'function of that scenario (named <scenario>_intent) on utterance and return its answer.',
        'utterance was spoken to a voice assistant. Use scenario to pick the scenario, then the matching '
        '<scenario>_intent function to pick the intent within it. Return that intent.',
    ],
    'trec': [
        'What kind of answer does question ask for? Decide the broad answer type with answer_type, then call the '
        'function for that type (named <type>_detail, for example NUM_detail) on question and return its answer.',
        'Classify question in two steps: answer_type gives the coarse answer type; the <type>_detail function of that '
        'type gives the fine one. Return the fine answer type.',
    ],
}
TREC_COARSE = {'ABBR': 'an abbreviation or its expansion', 'DESC': 'a description, definition, manner or reason',
               'ENTY': 'an entity (animal, food, product, colour, event, ...)', 'HUM': 'a person, group or title',
               'LOC': 'a location', 'NUM': 'a number, date, count, amount or measure'}
VERDICT = {
    'vitaminc': [
        'Check claim against evidence. read_evidence writes a private note on what evidence says about the claim; '
        'verdict decides from that note whether the evidence supports or refutes the claim or is not enough. Return '
        '"publish" when it supports the claim, "correct" when it refutes it and "needs_source" otherwise.',
        'Use read_evidence on claim and evidence, give its note to verdict, and return verdict\'s answer.',
        'A fact-check desk. Have read_evidence study evidence for claim, then let verdict decide from the note. If the '
        'evidence refutes the claim return "correct"; if it supports it return "publish"; if it is not enough return '
        '"needs_source".',
    ],
    'boardgameqa': [
        'game describes a board game: its current state, rules and which rules are preferred over others. Have '
        'analyze trace the rules that bear on question into a private note, then let outcome decide from that note '
        'whether question is proved, disproved or unknown. Return its answer.',
        'Answer a player\'s question about the game in two steps: analyze reads game (state, rules, preferences) and '
        'writes a private analysis of the rules relevant to question; outcome turns that analysis into proved, '
        'disproved or unknown. Return outcome\'s answer unchanged.',
        'Work out whether the statement in question follows from game. analyze writes notes on the facts, rules and '
        'preferences that matter; outcome reads them and decides. Return "proved", "disproved" or "unknown".',
    ],
}
VITAMINC_ACTIONS = {'SUPPORTS': ('supports', 'publish'), 'REFUTES': ('refutes', 'correct'),
                    'NOT ENOUGH INFO': ('not_enough_info', 'needs_source')}


# Deployment settings, one sentence each, so programs of one family read as different applications rather than one
# template; combined with the family's procedure wordings (several per family).
FRAMES = {
    'clinc_oos': [
        'You run the request router of a general-purpose personal assistant.',
        'This program sits behind a smart-speaker assistant and turns what users say into work for its back-end teams.',
        'A travel, banking and home assistant receives free-form requests; most fit a known intent, some fit none.',
        'Users of a chat assistant type short requests. The assistant only acts on the intents it was built for.',
        'The assistant\'s operators split incoming requests between specialised teams and keep everything else for a human.',
        'Requests arrive one by one from an in-car voice assistant whose skills cover a fixed set of intents.',
    ],
    'banking77': [
        'You triage messages sent to an online bank\'s customer support.',
        'A neobank\'s help chat receives questions about cards, transfers, top-ups and identity checks.',
        'Customers of a card issuer write in about payments, deliveries and account problems.',
        'This program is the first line of a bank\'s support desk: it sorts customer messages before an agent sees them.',
        'Support tickets from a mobile banking app arrive with only the customer\'s own words.',
        'A payments company routes customer enquiries to the teams that can resolve them.',
    ],
    'massive_intent': [
        'You interpret commands spoken to a home voice assistant.',
        'A smart-home hub hears utterances about alarms, music, lights, calendars, weather and more.',
        'Utterances come from a phone assistant used in many everyday situations.',
        'This program is the understanding step of a voice assistant: it decides what an utterance asks for.',
        'A kitchen speaker receives spoken requests; each must reach the part of the assistant that can act on it.',
        'Users talk to an assistant on their watch; utterances are short and informal.',
    ],
    'trec': [
        'A question-answering service decides what kind of answer a question needs before searching.',
        'You classify questions typed into a reference library\'s help box.',
        'Quiz questions are sorted by the type of answer they expect, so the right retriever is used.',
        'Before answering, an assistant decides whether a question wants a person, a place, a number, a definition or a thing.',
        'A search engine routes factual questions to specialised answer extractors.',
        'This program labels trivia questions by expected answer type.',
    ],
    'sms_spam': [
        'You keep a phone\'s SMS inbox clean.',
        'A family shares one phone number; its text messages mix personal chat with advertising and scams.',
        'Text messages arrive at a small shop\'s phone; prize offers and premium-rate come-ons are spam.',
        'This program filters an SMS inbox before its owner reads it.',
        'A messaging app moves unwanted texts out of the way and leaves real conversations alone.',
        'A user asked for help cleaning up a cluttered text-message inbox.',
    ],
    'vitaminc': [
        'You check claims about Wikipedia facts against the current article text.',
        'A newsroom verifies statements against a quoted source sentence before they are printed.',
        'Editors of a reference site compare user-submitted claims with the revised article text.',
        'This program decides whether a revised article sentence backs a claim, contradicts it or says too little.',
    ],
    'boardgameqa': [
        'You referee a board game whose rules can conflict and are ranked by preference.',
        'A rules engine for a tabletop game answers players\' questions about what follows from the current state.',
        'Players ask whether a statement follows from the game state, its rules and which rules take precedence.',
        'This program settles disputes in a game where facts, rules and rule preferences are written in prose.',
    ],
}


def _framed(dataset, key, text):
    return pick(FRAMES[dataset], key + '|frame') + '\n\n' + text

def _decision_file(args, options, instruction):
    returns = ' | '.join(json.dumps(o) for o in options)
    return nl_file(args, returns, instruction).replace('---\n', '---\nreadout: decision\n', 1)


def _id(i):
    return f'm{i + 1}'


# ------------------------------------------------------------------ families

def _route(dataset, item, candidates, rng, split, key):
    teams = TEAMS[:max(2, len(candidates) - 1)]
    routes = {intent: teams[i % len(teams)] for i, intent in enumerate(candidates)}
    in_scope = item['label'] in candidates
    gold = routes[item['label']] if in_scope else 'human_review'
    options = candidates + [OOS]
    child = _decision_file({'request': 'string'}, options, pick(INTENT_CHILD, key + '|child'))
    code = ('const d = await decide(intent, request);\n'
            f'return d.value === {json.dumps(OOS)} ? "human_review" : routes[d.value];')
    wrong = sorted({t for t in routes.values() if t != gold} | ({'human_review'} if in_scope else set()))[:3]
    level = 0 if in_scope and len(candidates) <= 4 else 1
    case = _make(dataset, 'route', f'route-{"in" if in_scope else "oos"}-k{len(candidates)}', [item], split, key,
                 root='route_request', args={'request': 'string', 'routes': 'Record<string, string>'}, returns='string',
                 instruction=_framed(dataset, key, pick(ROUTE, key)), inputs={'request': item['text'], 'routes': routes}, expected=gold,
                 oracle='normalized', level=level, files={'route_request/intent.nl': child},
                 code=code, children=[(item, OOS if not in_scope else item['label'])], wrong=wrong,
                 aliases=[gold.upper()])
    return case


def _desk(dataset, items, candidates, rng, split, key):
    teams = TEAMS[:max(2, len(candidates) - 1)]
    routes = {intent: teams[i % len(teams)] for i, intent in enumerate(candidates)}
    tickets = [{'id': f't{100 + i}', 'text': it['text']} for i, it in enumerate(items)]
    actions = []
    for t, it in zip(tickets, items):
        if it['label'] in candidates:
            actions.append({'name': 'assign', 'kwargs': {'ticket': t['id'], 'team': routes[it['label']]}})
        else:
            actions.append({'name': 'escalate', 'kwargs': {'ticket': t['id']}})
    wrong = []
    for i, a in enumerate(actions):
        if a['name'] == 'assign':
            other = next((x for x in teams if x != a['kwargs']['team']), None)
            if other:
                wrong.append(actions[:i] + [{'name': 'assign', 'kwargs': {'ticket': a['kwargs']['ticket'], 'team': other}}] + actions[i + 1:])
            wrong.append(actions[:i] + [{'name': 'escalate', 'kwargs': {'ticket': a['kwargs']['ticket']}}] + actions[i + 1:])
        else:
            wrong.append(actions[:i] + [{'name': 'assign', 'kwargs': {'ticket': a['kwargs']['ticket'], 'team': teams[0]}}] + actions[i + 1:])
        if len(wrong) >= 3:
            break
    wrong.append(actions[:-1])  # the last ticket left unhandled
    salt = hashlib.sha256(f'desk|{key}'.encode()).hexdigest()[:24]
    service = (DESK_SERVICE.replace('__TICKETS__', json.dumps([t['id'] for t in tickets]))
               .replace('__TEAMS__', json.dumps(teams)).replace('__SALT__', json.dumps(salt)))
    options = candidates + [OOS]
    child = _decision_file({'request': 'string'}, options, pick(INTENT_CHILD, key + '|child'))
    code = ('for (const ticket of tickets) {\n'
            '  const d = await decide(intent, ticket.text);\n'
            f'  if (d.value === {json.dumps(OOS)}) await desk.escalate({{ ticket: ticket.id }});\n'
            '  else await desk.assign({ ticket: ticket.id, team: routes[d.value] });\n'
            '}\nreturn desk.finish();')
    labels = [it['label'] if it['label'] in candidates else OOS for it in items]
    case = _make(dataset, 'desk', f'desk-n{len(items)}', items, split, key, root='route_tickets',
                 args={'tickets': '{ id: string; text: string }[]', 'routes': 'Record<string, string>'},
                 returns='string', instruction=_framed(dataset, key, pick(DESK, key)), inputs={'tickets': tickets, 'routes': routes},
                 expected=None, oracle='exact', level=2 if len(items) <= 4 else 3,
                 files={'route_tickets/intent.nl': child}, code=code, children=list(zip(items, labels)),
                 services={'desk': service}, slice_='inline_placement', domain='actor')
    case['curriculum']['external']['call_sets'] = {'service': 'desk', 'mode': 'state', 'finish': 'finish',
                                                   'gold': actions, 'wrong': wrong, 'alias': []}
    case['curriculum']['external']['oracle_depth'] = 'effect-state'
    return case


def _inbox(items, rng, split, key):
    msgs = [{'id': _id(i), 'text': it['text']} for i, it in enumerate(items)]
    actions = [{'name': 'junk', 'kwargs': {'id': m['id']}} for m, it in zip(msgs, items) if it['label'] == 'spam']
    wrong = []
    if actions:
        wrong.append(actions[1:])
    hams = [m for m, it in zip(msgs, items) if it['label'] == 'ham']
    if hams:
        wrong.append(actions + [{'name': 'junk', 'kwargs': {'id': hams[0]['id']}}])
    wrong.append([{'name': 'junk', 'kwargs': {'id': m['id']}} for m in msgs])
    salt = hashlib.sha256(f'inbox|{key}'.encode()).hexdigest()[:24]
    service = INBOX_SERVICE.replace('__TICKETS__', json.dumps([m['id'] for m in msgs])).replace('__SALT__', json.dumps(salt))
    child = _decision_file({'message': 'string'}, ['spam', 'ham'], pick(SPAM_CHILD, key + '|child'))
    code = ('const verdicts = await Promise.all(messages.map(m => decide(is_spam, m.text)));\n'
            'for (const [i, m] of messages.entries()) if (verdicts[i].value === "spam") await inbox.junk({ id: m.id });\n'
            'return inbox.finish();')
    case = _make('sms_spam', 'inbox', f'inbox-n{len(items)}', items, split, key, root='clean_inbox',
                 args={'messages': '{ id: string; text: string }[]'}, returns='string', instruction=_framed('sms_spam', key, pick(INBOX, key)),
                 inputs={'messages': msgs}, expected=None, oracle='exact', level=2 if len(items) <= 6 else 3,
                 files={'clean_inbox/is_spam.nl': child}, code=code,
                 children=[(it, it['label']) for it in items], services={'inbox': service},
                 slice_='inline_placement', domain='actor')
    case['curriculum']['external']['call_sets'] = {'service': 'inbox', 'mode': 'state', 'finish': 'finish',
                                                   'gold': actions, 'wrong': wrong, 'alias': []}
    case['curriculum']['external']['oracle_depth'] = 'effect-state'
    return case


def _tally(dataset, items, candidates, kind, rng, split, key):
    msgs = [{'id': _id(i), 'text': it['text']} for i, it in enumerate(items)]
    counts = collections.Counter(it['label'] for it in items)
    child = _decision_file({'message': 'string'}, candidates, pick(TALLY_CHILD, key + '|child'))
    head = ('const labels = {};\nfor (const m of messages) labels[m.id] = (await decide(intent, m.text)).value;\n')
    if kind == 'counts':
        gold = {c: counts.get(c, 0) for c in candidates}
        returns, oracle = 'Record<string, number>', 'exact'
        code = head + ('const counts = Object.fromEntries(' + json.dumps(candidates) +
                       '.map(c => [c, 0]));\nfor (const v of Object.values(labels)) counts[v]++;\nreturn counts;')
        moved = dict(gold)
        a, b = candidates[0], candidates[1]
        moved[a], moved[b] = moved[a] + 1, moved[b] - 1 if moved[b] else moved[b] + 1
        wrong = [moved, {c: v for c, v in gold.items() if v}] if 0 in gold.values() else [moved]
        aliases = [dict(reversed(list(gold.items())))]
        instruction = _framed(dataset, key, pick(TALLY['counts'], key))
    elif kind == 'select':
        target = max(candidates, key=lambda c: (counts.get(c, 0), c))
        gold = [m['id'] for m, it in zip(msgs, items) if it['label'] == target]
        returns, oracle = 'string[]', {'level': 'normalized', 'normalization': 'answer-list'}
        code = head + f'return messages.filter(m => labels[m.id] === {json.dumps(target)}).map(m => m.id);'
        other = [m['id'] for m in msgs if m['id'] not in gold]
        wrong = [gold[:-1] or other[:1], gold + other[:1]]
        aliases = [list(reversed(gold))]
        instruction = _framed(dataset, key, pick(TALLY['select'], key).format(target=target))
    else:
        gold = counts.most_common(1)[0][0]
        returns, oracle = ' | '.join(json.dumps(c) for c in candidates), 'exact'
        code = head + ('const counts = {};\nfor (const v of Object.values(labels)) counts[v] = (counts[v] ?? 0) + 1;\n'
                       'return Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0];')
        wrong = [c for c in candidates if c != gold][:2]
        aliases = []
        instruction = _framed(dataset, key, pick(TALLY['majority'], key))
    level = 2 if len(items) <= 8 else 3
    return _make(dataset, 'tally', f'tally-{kind}-n{len(items)}', items, split, key, root='review_messages',
                 args={'messages': '{ id: string; text: string }[]'}, returns=returns, instruction=instruction,
                 inputs={'messages': msgs}, expected=gold, oracle=oracle, level=level,
                 files={'review_messages/intent.nl': child}, code=code,
                 children=[(it, it['label']) for it in items], wrong=wrong, aliases=aliases)


FOLDER = [
    'The folder holds incoming requests as inbox/<id>.md files. Classify each file\'s text with intent and move the '
    'file into the folder of its team, <team>/ with the team from routes, keeping its filename; a request intent finds '
    'out of scope goes to human_review/. Return the number of files moved.',
    'Empty inbox/: every request file there is moved, under the same name, to the directory named after the team '
    'that routes gives for its intent (decide the intent with intent), or to human_review/ when it is out of scope. '
    'Return how many files you moved.',
    'Sort the request files in inbox/ by team. For each file, read it, let intent decide its intent, and move it to '
    '<team>/<same name> using routes; out-of-scope requests go to human_review/<same name>. Leave nothing in inbox/ '
    'and return the count of moved files.',
    'Each file in inbox/ is one customer request. Route the files on disk: intent classifies the text, routes maps '
    'the intent to a team directory, and anything out of scope is filed under human_review/. Keep filenames and '
    'contents unchanged and return the number of files you moved.',
]
SKILL_ROUTE = [
    'Decide request\'s intent with intent. The desks that handle requests are described by the skills bound to this '
    'call: each skill\'s description says which intents its desk takes, and its instructions give the queue code for '
    'each of them. Read the skill that covers the intent and return the queue code it lists for that intent. Return '
    '"human_review" when intent finds the request out of scope.',
    'Find the queue code for request. First classify it with intent; then choose the desk skill whose description '
    'covers that intent, read its instructions with read_code and return the code they give for the intent. An '
    'out-of-scope request has no desk: return "human_review".',
    'Use intent, then the skills. intent tells you what request is about; exactly one bound skill describes a desk for '
    'that intent, and its instructions list a queue code per intent. Return that code, or "human_review" if the '
    'request is out of scope.',
    'Every desk publishes its own routing table as a skill. Classify request with intent, read the routing table of '
    'the desk whose skill description mentions that intent, and return the queue code listed there for it. Requests '
    'outside every listed intent go to "human_review".',
]
DESK_NAMES = ['north', 'harbor', 'summit', 'river', 'cedar', 'meadow']


def _folder_route(dataset, items, candidates, rng, split, key):
    teams = TEAMS[:max(2, len(candidates) - 1)]
    routes = {intent: teams[i % len(teams)] for i, intent in enumerate(candidates)}
    ids = [hashlib.sha256((key + it['rid']).encode()).hexdigest()[:10] for it in items]
    folder = {f'inbox/{fid}.md': it['text'] + '\n' for fid, it in zip(ids, items)}
    expected_files = {}
    labels = []
    for fid, it in zip(ids, items):
        team = routes[it['label']] if it['label'] in candidates else 'human_review'
        labels.append(it['label'] if it['label'] in candidates else OOS)
        expected_files[f'{team}/{fid}.md'] = it['text'] + '\n'
    child = _decision_file({'request': 'string'}, candidates + [OOS], pick(INTENT_CHILD, key + '|child'))
    code = ('const files = await folder.files("inbox/*.md");\n'
            'for (const file of files) {\n'
            '  const d = await decide(intent, (await file.readText()).trim());\n'
            f'  await file.moveTo((d.value === {json.dumps(OOS)} ? "human_review" : routes[d.value]) + "/");\n'
            '}\nreturn files.length;')
    case = _make(dataset, 'folder_route', f'folder-route-n{len(items)}', items, split, key, root='route_folder',
                 args={'routes': 'Record<string, string>'}, returns='number',
                 instruction=_framed(dataset, key, pick(FOLDER, key)), inputs={'routes': routes}, expected=len(items),
                 oracle='exact', level=2 if len(items) <= 5 else 3, files={'route_folder/intent.nl': child}, code=code,
                 children=list(zip(items, labels)), slice_='folder_failure', domain='actor')
    root = case['semantics']['root']
    case['semantics']['files'][root] = case['semantics']['files'][root].replace('---\nargs:', '---\nkind: directory-reducer\nargs:', 1)
    case['semantics']['folder_files'] = folder
    case['semantics']['expected_files'] = expected_files
    case['semantics']['files_oracle'] = {'compare': 'content', 'threshold': 1}
    # Wrong effects: nothing moved; one file in another team's folder; files copied instead of moved.
    moved = next(iter(expected_files))
    wrong_team = next(t for t in teams + ['human_review'] if t != moved.split('/')[0])
    case['curriculum']['external']['checks'] = {'wrong_code': [
        'return ' + str(len(items)) + ';',
        code.replace('return files.length;', f'await folder.file({json.dumps(moved)}).moveTo({json.dumps(wrong_team + "/")});\nreturn files.length;'),
        code.replace('await file.moveTo(', 'await folder.file((d.value === ' + json.dumps(OOS) + ' ? "human_review" : routes[d.value]) + "/" + file.name).writeText(await file.readText()); void (')],
        'wrong_answers': [], 'alias_answers': []}
    case['curriculum']['external']['oracle_depth'] = 'effect-state'
    return case


def _skill_route(dataset, item, candidates, rng, split, key):
    k = len(candidates)
    desks = rng.sample(DESK_NAMES, 2 if k <= 4 else 3)
    owner = {c: desks[i % len(desks)] for i, c in enumerate(candidates)}
    codes = {c: f'Q-{100 + int(hashlib.sha256((key + c).encode()).hexdigest()[:6], 16) % 900}' for c in candidates}
    files = {f'handle_request/intent.nl': _decision_file({'request': 'string'}, candidates + [OOS],
                                                        pick(INTENT_CHILD, key + '|child'))}
    for desk in desks:
        mine = [c for c in candidates if owner[c] == desk]
        words = ', '.join(c.replace('_', ' ') for c in mine)
        table = '\n'.join(f'- {c}: queue {codes[c]}' for c in mine)
        files[f'handle_request/skills/{desk}-desk/SKILL.md'] = (
            f'---\nname: {desk}-desk\ndescription: {json.dumps(f"The {desk} desk handles requests whose intent is one of: {words}.")}\n---\n\n'
            f'Queue codes of the {desk} desk, by intent:\n\n{table}\n\nReturn the code exactly as written.\n')
    in_scope = item['label'] in candidates
    gold = codes[item['label']] if in_scope else 'human_review'
    reference = [['eval', {'code': 'const d = await decide(intent, request);\nd.value;'}]]
    if in_scope:
        reference.append(['read_code', {'name': f'skills.{owner[item["label"]]}-desk'}])
    reference.append(['return_result', {'status': 'success', 'value': gold}])
    case = _make(dataset, 'skill_route', f'skill-route-{"in" if in_scope else "oos"}-d{len(desks)}', [item], split, key,
                 root='handle_request', args={'request': 'string'}, returns='string',
                 instruction=_framed(dataset, key, pick(SKILL_ROUTE, key)), inputs={'request': item['text']},
                 expected=gold, oracle='normalized', level=1 if len(desks) == 2 else 2, files=files, code='',
                 children=[(item, item['label'] if in_scope else OOS)],
                 wrong=sorted({v for v in codes.values() if v != gold} | ({'human_review'} if in_scope else set()))[:3],
                 aliases=[gold.lower()])
    case['curriculum']['reference']['root'] = reference
    return case


def _identifier(text):
    return re.sub(r'[^A-Za-z0-9]+', '_', text).strip('_')


def _tree(dataset, item, groups, split, key):
    """groups: {top: [fine labels]}; item['top'], item['fine']."""
    tops = sorted(groups)
    if dataset == 'massive_intent':
        top_fn, suffix, arg = 'scenario', '_intent', 'utterance'
        top_text = 'Which scenario of the voice assistant is utterance about?'
        fine_text = 'Which {top} intent does utterance express?'
        root = 'understand_utterance'
    else:
        top_fn, suffix, arg = 'answer_type', '_detail', 'question'
        top_text = 'What broad type of answer does question ask for? ' + '; '.join(
            f'{t}: {TREC_COARSE[t]}' for t in tops) + '.'
        fine_text = 'question asks for {desc}. Which kind exactly?'
        root = 'classify_question'
    files = {f'{root}/{top_fn}.nl': _decision_file({arg: 'string'}, tops, top_text)}
    for top in tops:
        desc = TREC_COARSE.get(top, top)
        files[f'{root}/{_identifier(top)}{suffix}.nl'] = _decision_file(
            {arg: 'string'}, groups[top], fine_text.format(top=top, desc=desc))
    table = '{ ' + ', '.join(f'{json.dumps(t)}: {_identifier(t)}{suffix}' for t in tops) + ' }'
    code = (f'const top = (await decide({top_fn}, {arg})).value;\n'
            f'const next = {table}[top];\n'
            f'return (await decide(next, {arg})).value;')
    gold = item['fine']
    siblings = [f for f in groups[item['top']] if f != gold]
    others = [f for t in tops if t != item['top'] for f in groups[t]]
    wrong = siblings[:2] + others[:1]
    children = [{'match': [f'You are inside this call: {top_fn}(', item['text'][:80]], 'value': item['top']},
                {'match': [f'You are inside this call: {_identifier(item["top"])}{suffix}(', item['text'][:80]],
                 'value': gold}]
    returns = ' | '.join(json.dumps(f) for f in sorted({f for fs in groups.values() for f in fs}))
    case = _make(dataset, 'tree', f'tree-{len(tops)}x', [item], split, key, root=root, args={arg: 'string'},
                 returns=returns, instruction=_framed(dataset, key, pick(TREE[dataset], key)), inputs={arg: item['text']}, expected=gold,
                 oracle='exact', level=2, files=files, code=code, children=None, wrong=wrong, aliases=[])
    case['curriculum']['reference']['children'] = children
    return case


def _verdict(dataset, row, split, key):
    if dataset == 'vitaminc':
        options = ['supports', 'refutes', 'not_enough_info']
        label, action = VITAMINC_ACTIONS[row['label']]
        variant = pick([0, 1, 2], key)
        instruction = _framed('vitaminc', key, VERDICT['vitaminc'][variant])
        args = {'claim': 'string', 'evidence': 'string'}
        inputs = {'claim': _norm(row['claim']), 'evidence': _norm(row['evidence'])}
        helper = ('read_evidence', {'claim': 'string', 'evidence': 'string'},
                  'Read evidence for what it says about claim and write a private note: the exact figures, names or '
                  'dates in evidence that bear on the claim, and whether they agree with it, contradict it or do not '
                  'settle it.')
        decision = ('verdict', {'claim': 'string', 'note': 'Neuralese<string>'},
                    'From the note on the evidence, does the evidence support claim, refute it, or is it not enough '
                    'to tell?')
        note = (f'Evidence: {inputs["evidence"][:300]} -> {label.replace("_", " ")} the claim.')
        if variant == 1:
            gold, returns = label, ' | '.join(json.dumps(o) for o in options)
            wrong = [o for o in options if o != label]
        else:
            gold, returns = action, '"publish" | "correct" | "needs_source"'
            wrong = [a for _, a in VITAMINC_ACTIONS.values() if a != action]
        level = 2
        root = 'check_claim'
        match = inputs['claim'][:80]
    else:
        options = ['proved', 'disproved', 'unknown']
        label = row['label']
        instruction = _framed('boardgameqa', key, pick(VERDICT['boardgameqa'], key))
        args = {'game': 'string', 'question': 'string'}
        inputs = {'game': _norm(row['example'].split('Based on the game state and the rules and preferences,')[0]),
                  'question': _norm('Based on the game state and the rules and preferences,' +
                                    row['example'].split('Based on the game state and the rules and preferences,')[-1])}
        helper = ('analyze', {'game': 'string', 'question': 'string'},
                  'Trace how the facts, rules and rule preferences in game bear on question and write a private note: '
                  'each rule that fires or is blocked, which preferred rule wins a conflict, and the conclusion they '
                  'lead to.')
        decision = ('outcome', {'question': 'string', 'analysis': 'Neuralese<string>'},
                    'Given the analysis of the game, is the statement in question proved, disproved, or unknown '
                    '(neither follows)?')
        note = _norm(row['proof'])[:1200] or 'No rule decides the question.'
        gold, returns = label, ' | '.join(json.dumps(o) for o in options)
        wrong = [o for o in options if o != label]
        depth = int(re.search(r'depth(\d)', row['config']).group(1)) if re.search(r'depth(\d)', row['config']) else 2
        level = min(3, depth)
        root = 'judge_game'
        match = inputs['question'][-80:]
    h_name, h_args, h_text = helper
    d_name, d_args, d_text = decision
    files = {f'{root}/{h_name}.nl': nl_file(h_args, 'Neuralese<string>', h_text),
             f'{root}/{d_name}.nl': _decision_file(d_args, options, d_text)}
    item = {'text': match, 'group': row['_group'], 'rid': row['_rid'], 'dataset': dataset, 'label': label}
    case = _make(dataset, 'verdict', f'verdict-{dataset}-{"action" if gold != label else "label"}', [item], split, key,
                 root=root, args=args, returns=returns, instruction=instruction, inputs=inputs, expected=gold,
                 oracle='exact', level=level, files=files, code='', children=None, wrong=wrong, aliases=[])
    # The reference returns the gold directly, as every Neuralese-helper framing does: a scripted child cannot write
    # Neuralese<string> without the actor's write port. The program text is unchanged; rollouts run the helper.
    case['curriculum']['reference'] = {'root': [['return_result', {'status': 'success', 'value': gold}]]}
    case['curriculum']['external']['reference_note'] = note[:400]
    case['curriculum']['external']['framing'] = 'neuralese-helper'
    case['curriculum']['external']['neuralese_room'] = 'the decision reads a private Neuralese note on the evidence'
    return case


def _make(dataset, family, shape, items, split, key, *, root, args, returns, instruction, inputs, expected, oracle,
          level, files, code, children, wrong=None, aliases=None, services=None, slice_='nested_scoped',
          domain='relational'):
    unit = f'decisions:{dataset}:{key}'
    upstream_split = {'train': 'train', 'dev': 'validation', 'confirmation': 'test'}[split]
    case = make_case(
        source='decisions', family=f'{family}', shape=shape, upstream=UPSTREAM[dataset], upstream_id=key,
        upstream_split=upstream_split, unit=unit, groups=sorted({it['group'] for it in items}), root_name=root,
        args=args, returns=returns, instruction=instruction, inputs=inputs, expected=expected, oracle=oracle,
        level=level, license_=LICENSES[dataset], extra_files=files, slice_=slice_, domain=domain,
        reference_root=[['eval', {'code': code, 'finish': True}]], wrong=wrong or [], aliases=aliases or [],
        split=split, services=services, framing='plain')
    case['source_ids'] = sorted({f'{dataset}:{it["rid"]}' for it in items})
    if children is not None:
        # Longest fragment first, so an item whose text contains another item's text gets its own answer.
        case['curriculum']['reference']['children'] = sorted(
            ({'match': [it['text'][:80]], 'value': value} for it, value in children),
            key=lambda child: -len(child['match'][0]))
    case['curriculum']['named'] = 'required'
    case['curriculum']['external']['decision'] = {'dataset': dataset, 'labels': [it['label'] for it in items]}
    case['generation']['method'] = ('labelled dataset items arranged into a typed natlang program whose decision '
                                    'readout function(s) the root acts on; gold from the upstream labels; no '
                                    'sampled trajectory')
    return case


# ------------------------------------------------------------------ build

def cases(seed=20261103, route=560, desk=220, inbox=120, tally=480, tree=300, vitaminc=300, boardgame=220,
          folder=260, skill=240, held_fraction=0.12):
    data = _load()
    out = []
    rng = random.Random(seed)

    def plan(n):
        held = max(2, int(round(n * held_fraction)))
        return [('train', i) for i in range(n)] + [('held', i) for i in range(held)]

    pools = {}

    def pool(dataset, split):
        key = (dataset, split)
        if key not in pools:
            pools[key] = Pool(data[dataset].get(split, []), random.Random(f'{seed}|{dataset}|{split}'))
        return pools[key]

    intent_sets = ('clinc_oos', 'banking77', 'massive_intent')

    def draw(dataset, part, k, minimum):
        labels = [l for l in pool(dataset, part).labels(minimum) if l != 'oos']
        return rng.sample(labels, k) if len(labels) >= k else None

    def parts(dataset, s, key):
        """(pool part, case split) for a planned train/held slot."""
        if s == 'train':
            return 'train', 'train'
        if dataset == 'massive_intent':
            return ('dev', 'dev') if _held_half(key) == 'dev' else ('confirmation', 'confirmation')
        return 'held', _held_half(key)

    # route
    for s, i in plan(route):
        dataset = intent_sets[i % 3]
        key = f'route|{dataset}|{s}|{i}'
        part, split = parts(dataset, s, key)
        k = rng.randint(3, 7)
        cands = draw(dataset, part, k, 1)
        if not cands:
            continue
        p = pool(dataset, part)
        mode = rng.random()
        if mode < 0.72:
            item = p.take(rng.choice(cands))
        elif dataset == 'clinc_oos' and p.by.get('oos'):
            item = p.take('oos')
        else:
            others = [l for l in p.labels(1) if l not in cands and l != 'oos']
            item = p.take(rng.choice(others)) if others else None
        if item:
            out.append(_route(dataset, item, cands, rng, split, key))
    # desk (service effects)
    for s, i in plan(desk):
        dataset = intent_sets[i % 3]
        key = f'desk|{dataset}|{s}|{i}'
        part, split = parts(dataset, s, key)
        cands = draw(dataset, part, rng.randint(3, 5), 2)
        if not cands:
            continue
        p = pool(dataset, part)
        n = rng.randint(3, 7)
        items = []
        for j in range(n):
            if j == 0 or rng.random() < 0.8:
                it = p.take(rng.choice(cands))
            else:
                others = [l for l in p.labels(1) if l not in cands]
                it = p.take(rng.choice(others)) if others else None
            if it:
                items.append(it)
        if len(items) >= 3:
            out.append(_desk(dataset, items, cands, rng, split, key))
    # inbox (spam, service effects)
    for s, i in plan(inbox):
        key = f'inbox|sms_spam|{s}|{i}'
        part, split = parts('sms_spam', s, key)
        p = pool('sms_spam', part)
        n = rng.randint(4, 9)
        items = [p.take('spam' if rng.random() < 0.4 else 'ham') for _ in range(n)]
        items = [it for it in items if it]
        if len(items) >= 4 and any(it['label'] == 'spam' for it in items):
            out.append(_inbox(items, rng, split, key))
    # tally
    for s, i in plan(tally):
        dataset = (intent_sets + ('trec',))[i % 4]
        kind = ('counts', 'select', 'majority')[(i // 4) % 3]
        key = f'tally|{dataset}|{kind}|{s}|{i}'
        part, split = parts(dataset, s, key)
        p = pool(dataset, part)
        cands = draw(dataset, part, rng.randint(3, 5), 3)
        if not cands:
            continue
        n = rng.randint(5, 11)
        items = [it for it in (p.take(rng.choice(cands)) for _ in range(n)) if it]
        counts = collections.Counter(it['label'] for it in items)
        top = counts.most_common(2)
        if len(items) < 5 or (kind == 'majority' and len(top) > 1 and top[0][1] == top[1][1]):
            continue
        if dataset == 'trec':
            items = [dict(it, label=it['label'].split('|')[1]) for it in items]
            cands = [c.split('|')[1] for c in cands]
            if len(set(cands)) < len(cands):
                continue
        out.append(_tally(dataset, items, cands, kind, rng, split, key))
    # tree (two chained decisions)
    for s, i in plan(tree):
        dataset = ('massive_intent', 'trec')[i % 2]
        key = f'tree|{dataset}|{s}|{i}'
        part, split = parts(dataset, s, key)
        p = pool(dataset, part)
        if dataset == 'massive_intent':
            tops = collections.defaultdict(list)
            for label in p.labels(1):
                tops[label.split('_', 1)[0]].append(label)
            tops = {t: v for t, v in tops.items() if len(v) >= 2}
            chosen = rng.sample(sorted(tops), min(len(tops), rng.randint(3, 5)))
            groups = {t: sorted(tops[t])[:6] for t in chosen}
            top = rng.choice(chosen)
            label = rng.choice(groups[top])
            item = p.take(label)
            if not item:
                continue
            item = dict(item, top=top, fine=label)
        else:
            tops = collections.defaultdict(list)
            for label in p.labels(1):
                coarse, fine = label.split('|')
                tops[coarse].append(fine)
            groups = {t: sorted(set(v)) for t, v in tops.items() if len(set(v)) >= 2}
            top = rng.choice(sorted(groups))
            fine = rng.choice(groups[top])
            item = p.take(f'{top}|{fine}')
            if not item:
                continue
            item = dict(item, top=top, fine=fine, label=fine)
        out.append(_tree(dataset, item, groups, split, key))
    # folder reducer: decisions move request files on disk
    for s, i in plan(folder):
        dataset = intent_sets[i % 3]
        key = f'folder|{dataset}|{s}|{i}'
        part, split = parts(dataset, s, key)
        cands = draw(dataset, part, rng.randint(3, 5), 2)
        if not cands:
            continue
        p = pool(dataset, part)
        items = []
        for j in range(rng.randint(3, 7)):
            if j == 0 or rng.random() < 0.8:
                it = p.take(rng.choice(cands))
            else:
                others = [l for l in p.labels(1) if l not in cands]
                it = p.take(rng.choice(others)) if others else None
            if it:
                items.append(it)
        if len(items) >= 3:
            out.append(_folder_route(dataset, items, cands, rng, split, key))
    # skills: desk routing tables published as skills
    for s, i in plan(skill):
        dataset = intent_sets[i % 3]
        key = f'skill|{dataset}|{s}|{i}'
        part, split = parts(dataset, s, key)
        cands = draw(dataset, part, rng.randint(3, 7), 1)
        if not cands:
            continue
        p = pool(dataset, part)
        if rng.random() < 0.8:
            item = p.take(rng.choice(cands))
        else:
            others = [l for l in p.labels(1) if l not in cands and l != 'oos']
            item = p.take('oos') if dataset == 'clinc_oos' and p.by.get('oos') else (p.take(rng.choice(others)) if others else None)
        if item:
            out.append(_skill_route(dataset, item, cands, rng, split, key))
    # verdicts over a Neuralese note
    out += _vitaminc(vitaminc, held_fraction, seed)
    out += _boardgame(boardgame, held_fraction, seed)
    ids = [c['id'] for c in out]
    assert len(ids) == len(set(ids))
    return out


def _vitaminc(n, held_fraction, seed):
    rng = random.Random(f'{seed}|vitaminc')
    out = []
    for upstream, split, count in (('train', 'train', n), ('validation', 'dev', int(n * held_fraction / 2) + 1),
                                   ('test', 'confirmation', int(n * held_fraction / 2) + 1)):
        rows = _parquet(SOURCES / 'tals__vitaminc/default' / upstream / '0000.parquet')
        pages = collections.defaultdict(list)
        for r in rows:
            if r['revision_type'] == 'real' and 30 <= len(r['evidence']) <= 1200 and r['label'] in VITAMINC_ACTIONS:
                pages[r['page']].append(r)
        names = sorted(pages)
        rng.shuffle(names)
        for page in names[:count]:
            row = rng.choice(pages[page])
            row['_rid'] = _rid('vitaminc', _norm(row['claim']) + '\0' + _norm(row['evidence']))
            row['_group'] = 'vitaminc:page:' + hashlib.sha256(page.encode()).hexdigest()[:24]
            out.append(_verdict('vitaminc', row, split, f'vitaminc|{split}|{row["unique_id"]}'))
    return out


def _boardgame(n, held_fraction, seed):
    rng = random.Random(f'{seed}|boardgame')
    out = []
    for upstream, split, count in (('train', 'train', n), ('valid', 'dev', int(n * held_fraction / 2) + 1),
                                   ('test', 'confirmation', int(n * held_fraction / 2) + 1)):
        rows = [r for r in _parquet(SOURCES / 'tasksource__Boardgame-QA/default' / upstream / '0000.parquet')
                if r['label'] in ('proved', 'disproved', 'unknown') and len(r['example']) <= 6000
                and 'Based on the game state and the rules and preferences,' in r['example']]
        rng.shuffle(rows)
        by_label = collections.defaultdict(list)
        for r in rows:
            by_label[r['label']].append(r)
        picked = []
        while len(picked) < count and any(by_label.values()):
            for label in ('proved', 'disproved', 'unknown'):
                if by_label[label] and len(picked) < count:
                    picked.append(by_label[label].pop())
        for row in picked:
            ident = hashlib.sha256(row['example'].encode()).hexdigest()
            row['_rid'] = ident
            row['_group'] = 'boardgameqa:' + ident[:32]
            out.append(_verdict('boardgameqa', row, split, f'boardgame|{split}|{ident[:16]}'))
    return out
