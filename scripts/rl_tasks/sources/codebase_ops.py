"""Several different operations against each of the repository's natlang codebases (codebases/<name>: named
natural-language functions with callable folders, exact TypeScript children and `types.ts` aliases), each judged by an
exact effect or state check. The codebase plus its operations is the unit: every operation on one codebase stays on
that codebase's split, the same split the natlang-codebase explore/repair families give it (`natcode._split`).
See plans/neuralese/BROAD_RL_TASKS_20261031.md, "Multi-operation codebase families".

Operations (each over the codebase's own pure TypeScript functions; values are computed at build time by running the
original code through the collector's own authoring harness, scripts/rl_tasks/run_exports.mjs):

  trace    investigate: run the program's function on given inputs (one call, or folded over a sequence of items)
           and return its exact result. Read-only; oracle normalized exact.
  select   plan under a constraint: which candidate arguments make the function's result satisfy a stated property
           (a field equal to a value). Read-only; oracle answer-list.
  repair   behavioural repair (directory reducer): one operator of one TypeScript child was flipped; a visible table of
           calls with their correct results fails. The edited codebase must load, keep every other file byte-identical
           and keep the export header, and the repaired function must return the correct results on the visible
           table and on further inputs (semantics.authoring + semantics.codebase_check).
  feature  implement a small, fully specified feature (directory reducer): a new module beside a function that folds
           it over a sequence, applies it to many argument records, or applies it while tolerating rejected calls.
           Judged by running the new export on several inputs, plus loading and preservation of every existing file.
  drive    changing-state control through a service: the service holds the application's state and applies moves
           through the application's own transition function; the model reaches a stated target with as few moves
           as possible and returns the service's confirmation code (final projected state plus the applied move set;
           verify.mjs materializes the gold code from the gold moves, as for the tau-bench store).
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

from ..common import make_case, nl_file, pick, resolve_str
from .natcode import _split as natcode_split

REPO = Path(resolve_str('repo'))
RUNTIME = os.environ.get('NATLANG_RL_EXPORT_RUNTIME', str(REPO / 'ts-host'))

# ------------------------------------------------------------------ per-codebase world generators
# Each generator returns one argument list for the function, drawn from small domains so calls interact (ids repeat,
# states reference the same orders/tasks/goods).


def _ids(prefix, n):
    return [f'{prefix}{i}' for i in range(1, n + 1)]


# order_saga
SAGA_KINDS = ['start', 'paid', 'failed', 'sent', 'cancel']
SAGA_ORDER = ['reserved', 'paid', 'done', 'cancelled']
SAGA_TEXT = {'start': 'Customer placed a new order.', 'paid': 'Payment confirmed.', 'failed': 'Shipment failed.',
             'sent': 'Shipping completed.', 'cancel': 'Customer asks to cancel.'}


def saga_state(r):
    orders = {o: r.choice(SAGA_ORDER) for o in _ids('o', 3) if r.random() < 0.6}
    return {'seen': r.sample(_ids('e', 6), r.randint(0, 3)), 'orders': orders, 'outbox': []}


def saga_event(r, kind=None):
    kind = kind or r.choice(SAGA_KINDS)
    return {'id': r.choice(_ids('e', 9)), 'order': r.choice(_ids('o', 3)), 'text': SAGA_TEXT[kind]}


# dependency_plan
PLAN_DESCRIPTIONS = ['Patch the active security vulnerability.', 'Improve the customer-facing checkout page.',
                     'Clean up internal build logs.', 'Rotate the leaked API key.', 'Fix the broken signup form.',
                     'Archive old sprint boards.']


def plan_tasks(r, n=None):
    n = n or r.randint(3, 6)
    ids = _ids('t', n)
    tasks = []
    for i, tid in enumerate(ids):
        needs = sorted(r.sample(ids[:i], min(i, r.randint(0, 2)))) if i else []
        if r.random() < 0.08:
            needs = needs + ['t9']
        tasks.append({'id': tid, 'needs': needs, 'description': r.choice(PLAN_DESCRIPTIONS)})
    return tasks


def plan_state(r, tasks=None):
    tasks = tasks or plan_tasks(r)
    order, done = [], set()
    for _ in range(r.randint(0, len(tasks) - 1)):
        ready = [t['id'] for t in tasks if t['id'] not in done and all(n in done for n in t['needs'])]
        if not ready:
            break
        pick_ = r.choice(ready)
        order.append(pick_)
        done.add(pick_)
    return {'tasks': tasks, 'order': order, 'blocked': [], 'finished': False}


# legal_move
def tictac_cells(r):
    cells = ['empty'] * 9
    moves = r.randint(0, 7)
    slots = r.sample(range(9), moves)
    for i, s in enumerate(slots):
        cells[s] = 'X' if i % 2 == 0 else 'O'
    if r.random() < 0.1:
        cells[r.randrange(9)] = r.choice(['X', 'O'])
    return cells


# shopkeeper
GOODS = ['apple', 'bread', 'candle', 'rope', 'lamp']


def shop(r):
    goods = r.sample(GOODS, r.randint(2, 4))
    return {'stock': {g: r.randint(0, 5) for g in goods}, 'prices': {g: r.choice([2, 3, 5, 8, 10, 12]) for g in goods},
            'coins': r.randint(0, 40), 'persona': 'a gruff but fair village shopkeeper', 'ledger': []}


def shop_intent(r, s):
    good = r.choice(list(s['prices']) + ['silk'])
    kind = r.choice(['buy', 'ask_price', 'haggle', 'chat'])
    price = s['prices'].get(good, 5)
    return {'kind': kind, 'good': good if kind != 'chat' else '', 'qty': r.randint(1, 4) if kind in ('buy', 'haggle') else 0,
            'offer': r.randint(max(1, price - 4), price + 1) if kind == 'haggle' else 0}


def shop_action(r, s):
    good = r.choice(list(s['prices']))
    code = r.choice(['sell_list_price', 'sell_at_offer', 'quote', 'decline', 'counter_offer'])
    qty = r.randint(1, 2) if code.startswith('sell') else 0
    price = s['prices'][good] - (1 if code == 'sell_at_offer' else 0)
    return {'code': code, 'good': good if code != 'decline' else '', 'qty': qty, 'price': price if code != 'decline' else 0}


# webserver
SITE_ROUTES = [{'path': '/', 'methods': ['GET'], 'kind': 'page', 'purpose': 'home page'},
               {'path': '/guestbook', 'methods': ['GET', 'POST'], 'kind': 'form', 'purpose': 'sign the guestbook'},
               {'path': '/style.css', 'methods': ['GET'], 'kind': 'static', 'purpose': 'stylesheet', 'asset': 'style.css'},
               {'path': '/robots.txt', 'methods': ['GET'], 'kind': 'static', 'purpose': 'robots', 'asset': 'robots.txt'},
               {'path': '/about', 'methods': ['GET'], 'kind': 'page', 'purpose': 'about the site'}]


def site(r):
    sessions = {f's{i}x{i}': {'id': f's{i}x{i}', 'visits': r.randint(1, 4), 'name': ''} for i in range(1, r.randint(1, 3) + 1)}
    return {'name': 'Village Notes', 'about': 'A small community site.', 'routes': SITE_ROUTES,
            'assets': {'style.css': 'body{margin:0}', 'robots.txt': 'User-agent: *'},
            'entries': [{'author': 'ann', 'message': 'Lovely site!'}][:r.randint(0, 1)], 'sessions': sessions, 'log': []}


def web_request(r, s=None):
    path = r.choice(['/', '/guestbook', '/style.css', '/about/', '/missing', '/robots.txt', '/guestbook?x=1'])
    method = r.choice(['GET', 'GET', 'POST', 'put'])
    sid = r.choice(list((s or {}).get('sessions', {})) + ['', 'zz9'])
    headers = {'Cookie': f'sid={sid}'} if sid else {}
    body = 'author=Bo&message=Hello+there' if method == 'POST' else ''
    return {'id': f'r{r.randint(1, 99)}', 'method': method, 'path': path, 'headers': headers, 'body': body}


def parsed(r, s=None):
    req = web_request(r, s)
    path = req['path'].split('?')[0].rstrip('/') or '/'
    return {'id': req['id'], 'method': req['method'].upper(), 'path': path, 'query': {}, 'form': {},
            'session_id': (req['headers'].get('Cookie') or 'sid=').split('=')[1]}


def review(r):
    return {'accept': r.random() < 0.6, 'reason': 'ok', 'author': r.choice(['  Bo ', '', 'Cy']),
            'message': r.choice(['Hello there', '  ', 'Nice photos of the fair!'])}


# reconciliation
def recon_customers(r):
    return [{'id': c, 'tier': r.choice(['gold', 'basic'])} for c in _ids('c', r.randint(2, 4))]


def recon_events(r):
    out = []
    for _ in range(r.randint(3, 7)):
        if out and r.random() < 0.2:
            out.append(dict(r.choice(out), message='duplicate delivery'))
            continue
        out.append({'id': r.choice(_ids('ev', 8)), 'customer': r.choice(_ids('c', 5)),
                    'message': r.choice(['Checkout is down', 'Question about invoice', 'Card declined']),
                    'cents': r.choice([0, 250, 1999, 4500, 12000])})
    return out


def recon_join(r):
    customers, events = recon_customers(r), recon_events(r)
    ids = {c['id']: c for c in customers}
    seen, rows, dup = set(), [], 0
    for e in events:
        if e['id'] in seen:
            dup += 1
            continue
        seen.add(e['id'])
        rows.append({**e, 'matched': e['customer'] in ids, 'tier': ids[e['customer']]['tier'] if e['customer'] in ids else 'unknown'})
    return {'rows': rows, 'duplicates': dup}


# experiment_lab
def lab_candidates(r):
    return [{'id': c, 'source_revision': 'rev' + c, 'model_id': 'm', 'model_seed': i} for i, c in enumerate(_ids('cand', r.randint(1, 3)))]


def lab_trials(r, cands):
    out = []
    for _ in range(r.randint(2, 8)):
        out.append({'candidate': r.choice([c['id'] for c in cands] + ['cand9']), 'case_id': r.choice(['k1', 'k2', 'k3']),
                    'attempt': r.randint(1, 3), 'status': r.choice(['done', 'done', 'quiesced', 'exception', 'missing']),
                    'label': 'x', 'provenance': r.random() < 0.7, 'quality': r.choice(['pass', 'fail', 'pending']),
                    'value_digest': r.choice(['d1', 'd1', 'd2', '']), 'trace_id': 't'})
    return out


# spell_arena
def arena_world(r):
    actors = [{'id': a, 'hp': r.randint(0, 10), 'shield': r.randint(0, 3), 'x': r.randint(0, 4), 'y': r.randint(0, 4)}
              for a in ['hero', 'goblin', 'troll'][:r.randint(2, 3)]]
    return {'revision': r.randint(0, 5), 'energy': r.randint(5, 30), 'actors': actors}


ARENA_RULES = {'max_damage': 5, 'max_shield': 4, 'max_move': 3, 'damage_cost': 2, 'shield_cost': 1, 'move_cost': 1}


def arena_meaning(r, world):
    effects = []
    for _ in range(r.randint(0, 3)):
        kind = r.choice(['damage', 'shield', 'move'])
        target = r.choice([a['id'] for a in world['actors']] + ['dragon'])
        if kind == 'move':
            effects.append({'kind': 'move', 'target': target, 'amount': 0, 'x': r.randint(0, 5), 'y': r.randint(0, 5)})
        else:
            effects.append({'kind': kind, 'target': target, 'amount': r.randint(0, 6), 'x': 0, 'y': 0})
    status = r.choice(['plan', 'plan', 'plan', 'clarify'])
    rev = world['revision'] if r.random() < 0.85 else world['revision'] + 1
    return {'status': status, 'plan': {'revision': rev, 'effects': effects if status == 'plan' else []},
            'explanation': 'as asked'}


# highlighter
NL_SNIPPETS = ['---\nargs:\n  x: string\nreturns: string\n---\nfunction f(x) -> string\n  y = clean(x)   # exact\n  return y\n',
               'import helper from "./helper.nl"\n---\nargs: {}\nreturns: number\n---\nrepeat 3 times:\n  total = add(total, 1)\nreturn total\n',
               '/*---\ndescription: tool\n---*/\nexport default function g(a: number) { return a + 1 }\n',
               'Plain text with no frontmatter.\nfor each item in items: check(item)\n']


def hl_file(r):
    text = r.choice(NL_SNIPPETS)
    return {'path': r.choice(['f.nl', 'g.ts', 'notes.nl']) if not text.startswith('/*') else 'g.ts', 'text': text}


# nlprolog
def prolog_state(r):
    facts = ['Tweety is a bird.', 'Opus is a penguin.', 'Rex is a dog.', 'tweety is a bird', 'Rex barks.']
    known = r.sample(facts, r.randint(1, 3))
    return {'known': known, 'derived': [], 'grew': r.random() < 0.5}


def prolog_found(r):
    pool = ['Tweety can fly.', 'TWEETY CAN FLY', 'Rex is loyal.', 'Opus cannot fly.', 'tweety is a bird.']
    return [r.sample(pool, r.randint(0, 2)) for _ in range(r.randint(1, 3))]


# type_studio
def ts_candidate(r):
    return {'args': {'x': 'string'}, 'returns': 'number', 'effects': [], 'reason': 'r',
            'alternatives': [] if r.random() < 0.5 else ['string']}


def ts_fit(r):
    return {'parseable': r.random() < 0.8, 'obligations_ok': r.random() < 0.8, 'checked': r.randint(0, 3), 'diagnostics': []}


# mail_rules
def mail_decisions(r):
    return [{'action': r.choice(['calendar', 'reply_later', 'archive']), 'date': '2026-11-0' + str(r.randint(1, 9))}
            for _ in range(r.randint(0, 7))]


# ------------------------------------------------------------------ codebases and their functions
# fold: (state parameter index, item parameter names) for functions f(state, ...item) -> state.
CODEBASES = {
    'order_saga': {
        'step/transition.ts': {'gen': lambda r: [saga_state(r), saga_event(r), r.choice(SAGA_KINDS)],
                               'fold': ('acc', ['item', 'kind'])},
        'step/seen.ts': {'gen': lambda r: [saga_state(r), saga_event(r)]},
        'step/clear_outbox.ts': {'gen': lambda r: [dict(saga_state(r), outbox=[{'key': 'e1:0', 'order': 'o1', 'operation': 'reserve'}])]},
    },
    'dependency_plan': {
        'plan/step/ready_tasks.ts': {'gen': lambda r: [plan_state(r)]},
        'plan/step/advance.ts': {'gen': lambda r: (lambda s: [s, r.choice([t['id'] for t in s['tasks']])])(plan_state(r)),
                                 'fold': ('state', ['chosen'])},
        'plan/step/stall.ts': {'gen': lambda r: [plan_state(r)]},
        'plan/prepare.ts': {'gen': lambda r: [plan_tasks(r)]},
    },
    'legal_move': {
        'check_move/board_problem.ts': {'gen': lambda r: [tictac_cells(r), r.choice(['X', 'O'])]},
        'check_move/judge_move.ts': {'gen': lambda r: [tictac_cells(r), r.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), r.choice(['X', 'O'])]},
        'check_move/draw.ts': {'gen': lambda r: [tictac_cells(r)]},
    },
    'shopkeeper': {
        'serve/legal_actions.ts': {'gen': lambda r: (lambda s: [s, shop_intent(r, s)])(shop(r))},
        'serve/apply_action.ts': {'gen': lambda r: (lambda s: [s, shop_action(r, s), r.choice(['ann', 'bo'])])(shop(r)),
                                  'fold': ('acc', ['action', 'customer'])},
        'serve/checked_action.ts': {'gen': lambda r: [r.choice(['quote', 'sell_list_price', ' decline ', 'steal']),
                                                      [{'code': c, 'good': 'apple', 'qty': 1, 'price': 3} for c in r.sample(['decline', 'quote', 'sell_list_price'], 2)]]},
        'serve/goods_of.ts': {'gen': lambda r: [shop(r)]},
    },
    'webserver': {
        'handle/parse_request.ts': {'gen': lambda r: [web_request(r)]},
        'handle/match_route.ts': {'gen': lambda r: [SITE_ROUTES, parsed(r)]},
        'handle/session_for.ts': {'gen': lambda r: (lambda s: [s, parsed(r, s)])(site(r))},
        'handle/apply_submission.ts': {'gen': lambda r: [site(r), review(r)], 'fold': ('acc', ['review'])},
        'handle/error_response.ts': {'gen': lambda r: [{'kind': r.choice(['not_found', 'method_not_allowed']), 'name': '/x', 'purpose': 'GET, POST', 'asset': ''}]},
        'handle/static_response.ts': {'gen': lambda r: [site(r), {'kind': 'static', 'name': '/a', 'purpose': '', 'asset': r.choice(['style.css', 'robots.txt', 'logo.png'])}]},
    },
    'reconciliation': {
        'reconcile/join_events.ts': {'gen': lambda r: [recon_customers(r), recon_events(r)]},
        'reconcile/summarize.ts': {'gen': lambda r: (lambda j: [j, [r.choice(['urgent', 'normal']) for _ in j['rows']]])(recon_join(r))},
    },
    'experiment_lab': {
        'report/summarize.ts': {'gen': lambda r: (lambda c: [c, lab_trials(r, c)])(lab_candidates(r))},
    },
    'spell_arena': {
        'cast/resolve.ts': {'gen': lambda r: (lambda w: [w, dict(ARENA_RULES), arena_meaning(r, w)])(arena_world(r))},
    },
    'highlighter': {
        'highlight/highlight_file/split_source.ts': {'gen': lambda r: [hl_file(r)]},
    },
    'nlprolog': {
        'solve/derive/merge.ts': {'gen': lambda r: [prolog_state(r), prolog_found(r)], 'fold': ('state', ['found'])},
        'solve/settled.ts': {'gen': lambda r: [prolog_state(r)]},
    },
    'type_studio': {
        'infer/finalize.ts': {'gen': lambda r: [ts_candidate(r), ts_fit(r)]},
    },
    'mail_rules': {
        'process_mail/tally_actions.ts': {'gen': lambda r: [mail_decisions(r)]},
    },
}

# Drive: the move set, the state projection the goal names, and how the goal reads.
DRIVE = {
    'order_saga': {'module': 'step/transition.ts', 'service': 'saga', 'state': saga_state,
                   'moves': lambda r: [{'item': {'id': f'e{20 + i}', 'order': r.choice(_ids('o', 3)), 'text': SAGA_TEXT[k]}, 'kind': k}
                                       for i, k in enumerate(r.choices(SAGA_KINDS, k=6))],
                   'args': ['item', 'kind'], 'project': lambda s: s['orders'], 'field': 'orders',
                   'goal': 'the saga\'s orders (saga.state().orders) equal target exactly'},
    'dependency_plan': {'module': 'plan/step/advance.ts', 'service': 'planner', 'state': lambda r: plan_state(r, plan_tasks(r, r.randint(4, 6))) | {'order': []},
                        'moves': None, 'args': ['chosen'], 'project': lambda s: sorted(s['order']), 'field': 'order',
                        'goal': 'every task in target is in the plan\'s order (planner.state().order holds exactly '
                                'those tasks, in any order)'},
    'shopkeeper': {'module': 'serve/apply_action.ts', 'service': 'till', 'state': shop,
                   'moves': lambda r: None, 'args': ['action', 'customer'],
                   'project': lambda s: {'stock': s['stock'], 'coins': s['coins']}, 'field': None,
                   'goal': 'the shop\'s stock and coins (till.state().stock and .coins) equal target exactly'},
    'webserver': {'module': 'handle/apply_submission.ts', 'service': 'site', 'state': site,
                  'moves': lambda r: [{'review': review(r)} for _ in range(6)], 'args': ['review'],
                  'project': lambda s: s['entries'], 'field': 'entries',
                  'goal': 'the guestbook entries (site.state().entries) equal target exactly, in that order'},
}

TRACE = {
    'call': ['The program in your callable folder is a natlang application. Call its function {dotted} with this '
             'call\'s arguments ({names}) and return exactly what it returns.',
             'What does {dotted} of the program in your callable folder return for {names}? Use the program\'s own '
             'code and return its result unchanged.'],
    'fold': ['Start from {state} and apply {dotted} of the program in your callable folder to each entry of steps in '
             'order: the result of each call is the {state} of the next one, and the entry\'s fields are the other '
             'arguments ({rest}). Return the final {state}.',
             'Replay steps through the program\'s {dotted}: begin with {state}, call it once per entry (in order) with '
             'the entry\'s {rest}, and keep each result as the new {state}. What is the last {state}?'],
}
SELECT = ['candidates maps ids to values of the argument {param}. For each candidate, call {dotted} of the program in '
          'your callable folder with that value as {param} and the other arguments from fixed. Return the ids of the '
          'candidates for which the result{accessor} is {value}.',
          'Which candidates make {dotted} return a result whose{accessor_words} is {value}? Call it with each '
          'candidate as {param} (the remaining arguments come from fixed) and return the ids of those candidates.']
FOLDER_TOOLS = ('The supplied folder is the application to work on. The `{root}` operation is the task entrypoint '
                'outside that application; it is not a file to search for. Inspect the application with the folder '
                'tools (`list_files`, `read_file`, `search_files`), edit with `write_file` or `editor`, and review '
                'your change with `diff_files`.\n\n')
REPAIR = ['{tools}A behaviour of the application broke: one operator in one of its TypeScript modules was replaced by '
          'another (for example < by <=, === by !==, && by ||, Math.max by Math.min). These calls of {dotted} must '
          'return the listed results and some do not:\n\n{table}\n\nRestore the changed operator so that all of them '
          'do. Change nothing else, and keep every function\'s name and parameters. Return exactly `repair complete`.',
          '{tools}Tests of {dotted} fail since one operator in a TypeScript module of the application was changed. '
          'Each line is a call and the result it must give:\n\n{table}\n\nFind the changed operator and put the '
          'original back, leaving every other file and line as it is. Return exactly `repair complete`.']
FEATURE = {
    'fold': ['{tools}Add the module {new} to the application. It exports `export function {name}({state}: {S}, steps: '
             '{T}[]): {S}`, which starts from {state} and applies {fn} to each entry of steps in order (the entry\'s '
             'fields are {fn}\'s other arguments: {rest}); each result is the {state} of the next call, and it returns '
             'the last one. Call the existing {fn} module for each step: `import {fn} from "./{fn}.ts";`. Change no '
             'existing file. Return exactly `feature added`.'],
    'each': ['{tools}Add the module {new} to the application, exporting `export function {name}(calls: {T}[]): {R}[]`. '
             'Each entry of calls holds the arguments of {fn} by name ({params}); {name} returns {fn}\'s result for '
             'every entry, in the same order. Use the existing module: `import {fn} from "./{fn}.ts";`. Change no '
             'existing file. Return exactly `feature added`.'],
    'tolerant': ['{tools}Add the module {new} to the application, exporting `export function {name}({state}: {S}, '
                 'steps: {T}[]): {{ state: {S}; rejected: number[] }}`. It applies {fn} (`import {fn} from '
                 '"./{fn}.ts";`) to each entry of steps in order, starting from {state}; the entry\'s fields are '
                 '{fn}\'s other arguments ({rest}). When {fn} throws for a step, that step is skipped: the state stays '
                 'as it was and the step\'s index (from 0) is added to rejected. It returns the final state and the '
                 'rejected indexes in order. Change no existing file. Return exactly `feature added`.'],
}
DRIVE_TEXT = ['{service} holds the state of the application in your callable folder. moves lists the moves you may '
              'make, by id; {service}.apply({{ id }}) makes one through the application\'s own {fn} (a move it rejects '
              'changes nothing), and each move can be made at most once. Make {goal}, using as few moves as possible, '
              'then return {service}.finish().',
              'Bring the application to a target using {service}. Each id in moves names one call of the program\'s '
              '{fn} that {service}.apply({{ id }}) performs on the current state (once per id; {service}.state() shows '
              'the state). Use the fewest moves with which {goal}, and finish by returning the code {service}.finish() '
              'gives.',
              'Plan before you act: the program\'s {fn} (in your callable folder) is exactly what {service}.apply runs, '
              'so you can try moves on copies of {service}.state() in eval first. Then make the smallest set of moves '
              'from moves after which {goal}, apply them with {service}.apply({{ id }}), and return {service}.finish().',
              'Moves cannot be undone and each one may be used only once. Find the fewest moves (ids of moves) after '
              'which {goal}; a move that {fn} rejects leaves the state unchanged and does not count. Apply them through '
              '{service}.apply({{ id }}) and return the confirmation code from {service}.finish().',
              'Your job is a state change, not an answer: {goal}. Read the current state with {service}.state(), work '
              'out with {fn} which of moves get there in as few steps as possible, make exactly those moves with '
              '{service}.apply({{ id }}), and return {service}.finish(). Extra moves change the confirmation code.']
DRIVE_SERVICE = r'''import { createHash } from 'node:crypto';
__FUNCTION__
const MOVES: Record<string, unknown[]> = __MOVES__;
const PROJECT = (s: any): unknown => __PROJECT__;
const SALT = __SALT__;
let current: any = __INITIAL__;
const applied: string[] = [];
function canon(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canon);
  if (value && typeof value === 'object') return Object.fromEntries(Object.keys(value).sort().map(k => [k, canon((value as any)[k])]));
  return value;
}
/** The application's current state. */
export function state(): any { return structuredClone(current); }
/** Make one move by its id: the application's own __NAME__ applied to the current state. Each move can be made once; a move the application rejects changes nothing. */
export function apply(args: { id: string }): string {
  const move = MOVES[args.id];
  if (!move) return `Error: no move ${args.id}`;
  if (applied.includes(args.id)) return `Error: move ${args.id} was already made`;
  try { current = (__NAME__ as any)(structuredClone(current), ...structuredClone(move)); }
  catch (error) { return `Error: rejected: ${(error as Error).message}`; }
  applied.push(args.id);
  return `made ${args.id}`;
}
/** Finish: the confirmation code of the final state and the moves made. */
export function finish(): string {
  return '__SERVICE__-' + createHash('sha256').update(SALT + JSON.stringify([canon(PROJECT(current)), [...applied].sort()])).digest('hex').slice(0, 16);
}
'''
DRIVE_PROJECT_TS = {'order_saga': 's.orders', 'dependency_plan': '[...s.order].sort()',
                    'shopkeeper': '({ stock: s.stock, coins: s.coins })', 'webserver': 's.entries'}

MUTATIONS = [(r'===', '!=='), (r'!==', '==='), (r'<=', '<'), (r'>=', '>'), (r'(?<![<=!>])<(?![=<])', '<='),
             (r'(?<![=>\-])>(?![=>])', '>='), (r'&&', '||'), (r'\|\|', '&&'), (r'Math\.max', 'Math.min'),
             (r'Math\.min', 'Math.max'), (r'\+ 1\b', '- 1'), (r'- 1\b', '+ 1')]


# ------------------------------------------------------------------ helpers

def _files(name):
    root = REPO / 'codebases' / name
    out = {}
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.suffix in ('.nl', '.ts') and '.test.' not in path.name and not path.name.endswith('.d.ts'):
            out[str(path.relative_to(root))] = path.read_text()
    return out


def _signature(text):
    m = re.search(r'export default function (\w+)\(([^)]*)\)\s*:\s*([^{]+?)\s*\{', text)
    params, depth, cur = [], 0, ''
    for ch in m.group(2) + ',':
        depth += {'{': 1, '<': 1, '(': 1, '[': 1, '}': -1, '>': -1, ')': -1, ']': -1}.get(ch, 0)
        if ch == ',' and depth == 0:
            if cur.strip():
                pname, ptype = cur.split(':', 1)
                params.append((pname.strip(), ptype.strip()))
            cur = ''
        else:
            cur += ch
    return m.group(1), params, m.group(3).strip()


def _dotted(path):
    return path[:-3].replace('/', '.')


def _job_key(job):
    return hashlib.sha256(json.dumps([RUNTIME, job['files'], job['module'], job['export'], job['args']],
                                     sort_keys=True).encode()).hexdigest()


_CACHE = None


def _cache():
    """Results of earlier runs under the same runtime path, so a preempted build resumes (NATLANG_RL_EXPORT_CACHE)."""
    global _CACHE
    if _CACHE is None:
        _CACHE = {}
        path = os.environ.get('NATLANG_RL_EXPORT_CACHE')
        if path and os.path.exists(path):
            for line in open(path):
                try:
                    row = json.loads(line)
                    _CACHE[row['key']] = (row['ok'], row['value'])
                except (ValueError, KeyError):
                    continue
    return _CACHE


def _run(jobs):
    """Run export jobs through run_exports.mjs; returns {id: (ok, value or error)}."""
    cache, out, todo = _cache(), {}, []
    for job in jobs:
        key = _job_key(job)
        if key in cache:
            out[job['id']] = cache[key]
        else:
            todo.append((key, job))
    if not todo:
        return out
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / 'jobs.jsonl', Path(tmp) / 'out.jsonl'
        src.write_text(''.join(json.dumps(dict(j, id=k)) + '\n' for k, j in todo))
        subprocess.run(['node', str(REPO / 'scripts/rl_tasks/run_exports.mjs'), RUNTIME, str(src), str(dst)],
                       check=True, cwd=str(REPO), stdout=subprocess.DEVNULL)
        results = {r['id']: (r['ok'], r.get('value') if r['ok'] else r.get('error'))
                   for r in map(json.loads, dst.read_text().splitlines())}
    path = os.environ.get('NATLANG_RL_EXPORT_CACHE')
    with open(path, 'a') if path else open(os.devnull, 'w') as handle:
        for key, job in todo:
            cache[key] = results[key]
            out[job['id']] = results[key]
            handle.write(json.dumps({'key': key, 'ok': results[key][0], 'value': results[key][1]}) + '\n')
    return out


def _key(value):
    return json.dumps(value, sort_keys=True)


def _accessors(value, prefix='', depth=0):
    """(accessor, words, value) for scalar leaves of a result, a few levels deep."""
    out = []
    if isinstance(value, dict) and depth < 3:
        for k, v in value.items():
            if re.fullmatch(r'[A-Za-z_]\w*', k):
                out += _accessors(v, f'{prefix}.{k}', depth + 1)
            else:
                out += _accessors(v, f'{prefix}[{json.dumps(k)}]', depth + 1)
    elif isinstance(value, list) and depth < 3:
        out.append((f'{prefix}.length', f'{prefix or " result"}.length' if prefix else ' length', len(value)))
    elif isinstance(value, (str, int, float, bool)) and not (isinstance(value, str) and len(value) > 40):
        out.append((prefix, prefix if prefix else '', value))
    return out


# ------------------------------------------------------------------ operations

def _layout(files, root):
    """A read-only/service task: the codebase is the root function's callable folder; types reach the root too."""
    program = {f'{root}/{p}': t for p, t in files.items()}
    if 'types.ts' in files:
        program['types.ts'] = files['types.ts']  # the root's own signature names the codebase's types
    program['natlang.json'] = '{"schema":"natlang.package/v2"}\n'
    return program


def _case(name, kind, shape, item, *, split, root, args, returns, instruction, inputs, expected, oracle, level,
          files, reference_root, wrong=None, aliases=None, services=None, extra=None, slice_='nested_scoped',
          domain='relational', revision=''):
    unit = f'natlang-codebase:codebases/{name}'
    upstream_split = {'train': 'train', 'dev': 'validation', 'confirmation': 'test'}[split]
    case = make_case(
        source='codebase-ops', family=f'{kind}', shape=shape, upstream='natlang:codebases', upstream_id=item,
        upstream_split=upstream_split, unit=unit, groups=[unit, f'codebase-ops:{name}'], root_name=root, args=args,
        returns=returns, instruction=instruction, inputs=inputs, expected=expected, oracle=oracle, level=level,
        license_='natlang repository (owner)', extra_files=files, slice_=slice_, domain=domain,
        reference_root=reference_root, wrong=wrong or [], aliases=aliases or [], split=split, services=services,
        extra_semantics=extra, revision=revision, framing='plain')
    case['curriculum']['external']['codebase'] = {'package': f'codebases/{name}', 'operation': kind, 'version': revision}
    case['generation']['method'] = ('an operation against a repository natlang codebase; expected values computed by '
                                    'running the codebase\'s own TypeScript through the collector authoring harness; '
                                    'no sampled trajectory')
    return case


def _calls(name, files, rng, path, spec, n):
    """n distinct argument lists for one function with their results (only calls that return)."""
    seen, args = set(), []
    for _ in range(n * 4):
        a = spec['gen'](rng)
        if _key(a) not in seen:
            seen.add(_key(a))
            args.append(a)
        if len(args) >= n * 2:
            break
    results = _run([{'id': str(i), 'files': files, 'module': path, 'export': 'default', 'args': a} for i, a in enumerate(args)])
    return [(a, results[str(i)][1]) for i, a in enumerate(args) if results[str(i)][0]][:n]


def build_codebase(name, rng, revision):
    files = _files(name)
    split = natcode_split(f'codebases/{name}', {})
    out = []
    for path, spec in CODEBASES[name].items():
        fn, params, ret = _signature(files[path])
        dotted = _dotted(path)
        calls = _calls(name, files, rng, path, spec, 14)
        if len(calls) < 4:
            continue
        pool = [res for _, res in calls]
        # trace: one call
        for index, (a, res) in enumerate(calls[:8]):
            key = f'{name}|trace|{path}|{index}'
            wrong = [w for w in pool if _key(w) != _key(res)][:2]
            names = ', '.join(pn for pn, _ in params)
            out.append(_case(name, 'trace', 'trace-call', key, split=split, root='inspect_program',
                             args=dict(params), returns=ret,
                             instruction=pick(TRACE['call'], key).format(dotted=dotted, names=names),
                             inputs={pn: a[i] for i, (pn, _) in enumerate(params)},
                             expected=res, oracle='exact', level=1, files=_layout(files, 'inspect_program'),
                             reference_root=[['eval', {'code': f'return await {dotted}({names});', 'finish': True}]],
                             wrong=wrong, revision=revision))
        # select: one argument varies over candidates; property from a leaf of the results
        vary = max(range(len(params)), key=lambda i: len({_key(a[i]) for a, _ in calls}))
        base = calls[0][0]
        cands = {f'c{i + 1}': a[vary] for i, (a, _) in enumerate(calls[:8])}
        jobs = [{'id': cid, 'files': files, 'module': path, 'export': 'default',
                 'args': base[:vary] + [v] + base[vary + 1:]} for cid, v in cands.items()]
        results = {cid: v for cid, (ok, v) in _run(jobs).items() if ok}
        props = defaultdict(set)
        for cid, res in results.items():
            for acc, words, val in _accessors(res):
                props[(acc, words, _key(val))].add(cid)
        good = [(k, ids) for k, ids in props.items() if 1 <= len(ids) < len(results) - 0 and len(results) >= 4]
        rng.shuffle(good)
        for index, ((acc, words, val), ids) in enumerate(good[:3]):
            key = f'{name}|select|{path}|{acc}|{val}'
            value = json.loads(val)
            fixed = {pn: base[i] for i, (pn, _) in enumerate(params) if i != vary}
            gold = sorted(ids)
            others = sorted(set(results) - ids)
            ptype = params[vary][1]
            args_ = {'candidates': f'Record<string, {ptype}>', 'fixed': '{ ' + '; '.join(
                f'{pn}: {pt}' for i, (pn, pt) in enumerate(params) if i != vary) + ' }'} if fixed else {
                'candidates': f'Record<string, {ptype}>'}
            inputs = {'candidates': {cid: cands[cid] for cid in results}, **({'fixed': fixed} if fixed else {})}
            call_args = ', '.join(('v' if i == vary else f'fixed.{pn}') for i, (pn, _) in enumerate(params))
            code = (f'const ids = [];\nfor (const [id, v] of Object.entries(candidates)) {{\n'
                    f'  const result = await {dotted}({call_args});\n'
                    f'  if (JSON.stringify(result{acc}) === {json.dumps(json.dumps(value))}) ids.push(id);\n}}\nreturn ids;')
            out.append(_case(name, 'select', f'select-{"leaf" if "." in acc or "[" in acc else "value"}', key,
                             split=split, root='inspect_program', args=args_, returns='string[]',
                             instruction=pick(SELECT, key).format(param=params[vary][0], dotted=dotted,
                                                                  accessor=acc or '', accessor_words=(words or ' value'),
                                                                  value=json.dumps(value)),
                             inputs=inputs, expected=gold, oracle={'level': 'normalized', 'normalization': 'answer-list'},
                             level=2, files=_layout(files, 'inspect_program'),
                             reference_root=[['eval', {'code': code, 'finish': True}]],
                             wrong=[gold[:-1] + others[:1], gold + others[:1]] if others else [gold[:-1] or ['c9']],
                             aliases=[list(reversed(gold))], revision=revision))
        # trace-fold, feature fold/tolerant
        if spec.get('fold'):
            state_name, rest = spec['fold']
            sidx = [pn for pn, _ in params].index(state_name)
            rest_types = {pn: pt for pn, pt in params if pn in rest}
            item_type = '{ ' + '; '.join(f'{pn}: {pt}' for pn, pt in rest_types.items()) + ' }'
            out += _folds(name, files, rng, path, spec, fn, dotted, params, ret, state_name, rest, sidx, item_type,
                          calls, split, revision)
        # feature: each
        out += _feature_each(name, files, rng, path, fn, params, ret, calls, split, revision)
        # repair: mutants
        out += _repairs(name, files, rng, path, fn, dotted, params, calls, split, revision)
    if name in DRIVE:
        out += _drives(name, files, rng, split, revision)
    return out


def _folds(name, files, rng, path, spec, fn, dotted, params, ret, state_name, rest, sidx, item_type, calls, split, revision):
    out = []
    names = [pn for pn, _ in params]
    S = dict(params)[state_name]
    folder = 'feature_' + fn
    new = f'{path.rsplit("/", 1)[0]}/{fn}_steps.ts' if '/' in path else f'{fn}_steps.ts'
    worlds = []
    for w in range(6):
        start = spec['gen'](rng)[sidx]
        steps = [{pn: a[names.index(pn)] for pn in rest} for a in (spec['gen'](rng) for _ in range(rng.randint(2, 5)))]
        worlds.append((start, steps))
    # reference fold module, run once per world through the harness
    fold_src = (f'import {fn} from "./{fn}.ts";\n'
                f'export function {fn}_all({state_name}: {S}, steps: {item_type}[]): {S} {{\n'
                f'  let current = {state_name};\n'
                f'  for (const step of steps) current = {fn}(' + ', '.join('current' if pn == state_name else f'step.{pn}' for pn in names) + ');\n'
                f'  return current;\n}}\n')
    tolerant_src = (f'import {fn} from "./{fn}.ts";\n'
                    f'export function {fn}_tolerant({state_name}: {S}, steps: {item_type}[]): {{ state: {S}; rejected: number[] }} {{\n'
                    f'  let current = {state_name};\n  const rejected: number[] = [];\n'
                    f'  steps.forEach((step, index) => {{\n'
                    f'    try {{ current = {fn}(' + ', '.join('current' if pn == state_name else f'step.{pn}' for pn in names) + '); }\n'
                    f'    catch {{ rejected.push(index); }}\n  }});\n'
                    f'  return {{ state: current, rejected }};\n}}\n')
    with_fold = dict(files, **{new: fold_src})
    tol_path = new.replace('_steps.ts', '_tolerant.ts')
    with_tol = dict(files, **{tol_path: tolerant_src})
    jobs = [{'id': f'f{i}', 'files': with_fold, 'module': new, 'export': f'{fn}_all', 'args': [s, st]} for i, (s, st) in enumerate(worlds)]
    jobs += [{'id': f't{i}', 'files': with_tol, 'module': tol_path, 'export': f'{fn}_tolerant', 'args': [s, st]} for i, (s, st) in enumerate(worlds)]
    res = _run(jobs)
    folded = [(worlds[i], res[f'f{i}'][1]) for i in range(len(worlds)) if res[f'f{i}'][0]]
    tolerant = [(worlds[i], res[f't{i}'][1]) for i in range(len(worlds)) if res[f't{i}'][0]]
    rest_text = ', '.join(rest)
    # trace-fold
    pool = [r for _, r in folded]
    for index, ((start, steps), result) in enumerate(folded[:3]):
        key = f'{name}|trace-fold|{path}|{index}'
        # The loop variable must not shadow a callable of the folder (order_saga's operation is named step).
        code = (f'let current = {state_name};\nfor (const entry of steps) current = await {dotted}('
                + ', '.join('current' if pn == state_name else f'entry.{pn}' for pn in names) + ');\nreturn current;')
        out.append(_case(name, 'trace', 'trace-fold', key, split=split, root='inspect_program',
                         args={state_name: S, 'steps': f'{item_type}[]'}, returns=ret,
                         instruction=pick(TRACE['fold'], key).format(dotted=dotted, state=state_name, rest=rest_text),
                         inputs={state_name: start, 'steps': steps}, expected=result, oracle='exact', level=2,
                         files=_layout(files, 'inspect_program'), reference_root=[['eval', {'code': code, 'finish': True}]],
                         wrong=[w for w in pool if _key(w) != _key(result)][:1] + [start] if _key(start) != _key(result) else
                         [w for w in pool if _key(w) != _key(result)][:2], revision=revision))
    # feature: fold module and tolerant module (directory reducers judged by running the new export)
    for kind, src, module, export, runs_from in (('fold', fold_src, new, f'{fn}_all', folded),
                                                 ('tolerant', tolerant_src, tol_path, f'{fn}_tolerant', tolerant)):
        if len(runs_from) < 3:
            continue
        runs = [{'args': [s, st], 'expected': r} for (s, st), r in runs_from]
        if kind == 'tolerant' and not any(r['rejected'] for _, r in runs_from) and name != 'dependency_plan':
            continue
        key = f'{name}|feature-{kind}|{path}'
        text = pick(FEATURE[kind], key).format(tools=FOLDER_TOOLS.format(root='add_feature'), new=module, name=export,
                                               state=state_name, S=S, T=item_type, fn=fn, rest=rest_text)
        wrong_src = src.replace('for (const step of steps)', 'for (const step of steps.slice(1))').replace(
            'steps.forEach(', 'steps.slice(0, -1).forEach(')
        out.append(_feature_case(name, key, kind, module, export, src, wrong_src, files, runs, text, split, revision, path))
    return out


def _feature_case(name, key, kind, module, export, src, wrong_src, files, runs, text, split, revision, base_path):
    preserve = {p: hashlib.sha256(t.encode()).hexdigest() for p, t in files.items()}
    write = lambda body: (f'await folder.file({json.dumps(module)}).writeText({json.dumps(body)}); '
                          'return "feature added";')
    case = _case(name, 'feature', f'feature-{kind}', key, split=split, root='add_feature', args={}, returns='string',
                 instruction=text, inputs={}, expected='feature added', oracle='exact', level=3,
                 files={}, reference_root=[['eval', {'code': write(src), 'finish': True}]], revision=revision,
                 slice_='folder_failure',
                 extra={'folder_files': dict(files),
                        'authoring': {'module': module, 'export': export, 'runs': runs},
                        'codebase_check': {'load': True, 'preserve': preserve, 'require': [module],
                                           'allow_new_prefixes': [module]}})
    root = case['semantics']['root']
    case['semantics']['files'][root] = case['semantics']['files'][root].replace('---\nargs:', '---\nkind: directory-reducer\nargs:', 1)
    edit_other = (f'await folder.file({json.dumps(base_path)}).writeText({json.dumps(files[base_path] + chr(10) + "// touched" + chr(10))}); '
                  + write(src).split('; ', 1)[0] + '; return "feature added";')
    case['curriculum']['external']['checks'] = {
        'wrong_code': ['return "feature added";', write(wrong_src), edit_other], 'wrong_answers': [], 'alias_answers': []}
    return case


def _feature_each(name, files, rng, path, fn, params, ret, calls, split, revision):
    if len(calls) < 4:
        return []
    module = (path.rsplit('/', 1)[0] + '/' if '/' in path else '') + f'{fn}_each.ts'
    export = f'{fn}_each'
    T = '{ ' + '; '.join(f'{pn}: {pt}' for pn, pt in params) + ' }'
    names = [pn for pn, _ in params]
    src = (f'import {fn} from "./{fn}.ts";\nexport function {export}(calls: {T}[]): {ret}[] {{\n'
           f'  return calls.map(call => {fn}(' + ', '.join(f'call.{pn}' for pn in names) + '));\n}\n')
    batches = []
    for b in range(4):
        chosen = rng.sample(calls, min(len(calls), rng.randint(2, 4)))
        batches.append(([{pn: a[i] for i, pn in enumerate(names)} for a, _ in chosen], [r for _, r in chosen]))
    runs = [{'args': [batch], 'expected': results} for batch, results in batches]
    # Check the reference module through the harness (also proves the sibling import form).
    check = _run([{'id': str(i), 'files': dict(files, **{module: src}), 'module': module, 'export': export,
                   'args': r['args']} for i, r in enumerate(runs)])
    if not all(check[str(i)][0] and _key(check[str(i)][1]) == _key(r['expected']) for i, r in enumerate(runs)):
        return []
    key = f'{name}|feature-each|{path}'
    text = pick(FEATURE['each'], key).format(tools=FOLDER_TOOLS.format(root='add_feature'), new=module, name=export,
                                             T=T, R=ret, fn=fn, params=', '.join(names))
    wrong_src = src.replace('calls.map(', 'calls.slice().reverse().map(')
    if all(len(r['expected']) < 2 or _key(r['expected']) == _key(list(reversed(r['expected']))) for r in runs):
        wrong_src = src.replace('calls.map(', 'calls.slice(1).map(')
    return [_feature_case(name, key, 'each', module, export, src, wrong_src, files, runs, text, split, revision, path)]


def _repairs(name, files, rng, path, fn, dotted, params, calls, split, revision):
    text = files[path]
    head_end = text.index('{', text.index('export default function')) + 1
    sites = []
    for pattern, repl in MUTATIONS:
        for m in re.finditer(pattern, text[head_end:]):
            start = head_end + m.start()
            mutant = text[:start] + re.sub(pattern, repl, text[start:start + len(m.group(0))], count=1) + text[start + len(m.group(0)):]
            if mutant != text:
                sites.append((pattern, start, mutant))
    rng.shuffle(sites)
    out = []
    args_pool = [a for a, _ in calls]
    for pattern, start, mutant in sites:
        if len(out) >= 6:
            break
        mfiles = dict(files, **{path: mutant})
        res = _run([{'id': str(i), 'files': mfiles, 'module': path, 'export': 'default', 'args': a} for i, a in enumerate(args_pool)])
        diff, same = [], []
        for i, (a, good) in enumerate(calls):
            ok, got = res[str(i)]
            if not ok or _key(got) != _key(good):
                diff.append(i)
            else:
                same.append(i)
        if len(diff) < 3:
            continue
        shown = diff[:2] + same[:1]
        hidden = diff[2:5] + same[1:4]
        table = '\n'.join(f'- {fn}({", ".join(json.dumps(x) for x in calls[i][0])}) must return {json.dumps(calls[i][1])}'
                          for i in shown)
        runs = [{'args': calls[i][0], 'expected': calls[i][1]} for i in shown + hidden]
        key = f'{name}|repair|{path}|{start}'
        write = lambda body: (f'await folder.file({json.dumps(path)}).writeText({json.dumps(body)}); '
                              'return "repair complete";')
        preserve = {p: hashlib.sha256(t.encode()).hexdigest() for p, t in files.items() if p != path}
        header = re.search(r'export default function [^{]+', text).group(0)
        instruction = pick(REPAIR, key).format(tools=FOLDER_TOOLS.format(root='repair_program'), dotted=dotted, table=table)
        case = _case(name, 'repair', 'repair-operator', key, split=split, root='repair_program', args={},
                     returns='string', instruction=instruction, inputs={}, expected='repair complete', oracle='exact',
                     level=3, files={}, reference_root=[['eval', {'code': write(text), 'finish': True}]],
                     revision=revision, slice_='folder_failure',
                     extra={'folder_files': mfiles,
                            'authoring': {'module': path, 'export': 'default', 'runs': runs},
                            'codebase_check': {'load': True, 'preserve': preserve, 'require': [path],
                                               'contains': [{'path': path, 'text': header}]}})
        root = case['semantics']['root']
        case['semantics']['files'][root] = case['semantics']['files'][root].replace('---\nargs:', '---\nkind: directory-reducer\nargs:', 1)
        # Wrong repairs: none; special-casing only the visible table (hidden failing runs reject it); deleting.
        visible = [calls[i] for i in shown]
        lookup = (text[:head_end] + '\n  const __table: [string, unknown][] = ' +
                  json.dumps([[json.dumps(a, separators=(',', ':'), ensure_ascii=False), r] for a, r in visible]) + ';\n' +
                  '  const __hit = __table.find(([k]) => k === JSON.stringify([...arguments]));\n'
                  '  if (__hit) return __hit[1] as any;\n' + mutant[head_end:])
        wrong_code = ['return "repair complete";', write(lookup),
                     f'await folder.file({json.dumps(path)}).remove(); return "repair complete";']
        case['curriculum']['external']['checks'] = {'wrong_code': wrong_code, 'wrong_answers': [], 'alias_answers': []}
        case['curriculum']['external']['mutation'] = {'path': path, 'offset': start, 'pattern': pattern}
        out.append(case)
    return out


def _drives(name, files, rng, split, revision):
    d = DRIVE[name]
    path = d['module']
    fn, params, ret = _signature(files[path])
    names = [pn for pn, _ in params]
    out = []
    for w in range(36):
        start = d['state'](rng)
        if name == 'dependency_plan':
            moves = [{'chosen': t['id']} for t in start['tasks']]
        elif name == 'shopkeeper':
            moves = [{'action': shop_action(rng, start), 'customer': rng.choice(['ann', 'bo'])} for _ in range(6)]
        else:
            moves = d['moves'](rng)
        ids = [f'm{i + 1}' for i in range(len(moves))]
        move_args = {mid: [m[pn] for pn in names[1:]] for mid, m in zip(ids, moves)}
        # breadth-first over move sets (order matters for the state; at most 3 moves)
        frontier = [((), start)]
        reach = {}
        for depth in range(1, 4):
            jobs = []
            for seq, st in frontier:
                for mid in ids:
                    if mid not in seq:
                        jobs.append({'id': json.dumps(list(seq) + [mid]), 'files': files, 'module': path,
                                     'export': 'default', 'args': [st] + move_args[mid]})
            res = _run(jobs)
            nxt = []
            for job in jobs:
                ok, val = res[job['id']]
                seq = tuple(json.loads(job['id']))
                if not ok:
                    continue
                proj = _key(d['project'](val))
                reach.setdefault(proj, []).append(seq)
                nxt.append((seq, val))
            rng.shuffle(nxt)
            frontier = nxt[:40]
        start_proj = _key(d['project'](start))
        options = []
        for proj, seqs in reach.items():
            if proj == start_proj:
                continue
            best = min(len(s) for s in seqs)
            sets = {frozenset(s) for s in seqs if len(s) == best}
            if best >= 2 and len(sets) == 1:
                options.append((proj, sorted(sets.pop()), best, next(s for s in seqs if len(s) == best)))
        if not options:
            continue
        proj, gold_set, best, seq = rng.choice(sorted(options))
        key = f'{name}|drive|{w}'
        salt = hashlib.sha256(f'drive|{key}'.encode()).hexdigest()[:24]
        func = re.sub(r'export default function', 'function', files[path])
        service = (DRIVE_SERVICE.replace('__FUNCTION__', func).replace('__MOVES__', json.dumps(move_args))
                   .replace('__PROJECT__', DRIVE_PROJECT_TS[name]).replace('__SALT__', json.dumps(salt))
                   .replace('__INITIAL__', json.dumps(start)).replace('__NAME__', fn).replace('__SERVICE__', d['service']))
        target = json.loads(proj)
        actions = [{'name': 'apply', 'kwargs': {'id': mid}} for mid in seq]
        others = [mid for mid in ids if mid not in gold_set]
        wrong = [actions[:-1], actions + [{'name': 'apply', 'kwargs': {'id': others[0]}}] if others else actions[1:]]
        sv = d['service']
        code = '\n'.join(f'await {sv}.apply({{ id: {json.dumps(mid)} }});' for mid in seq) + f'\nreturn {sv}.finish();'
        moves_input = {mid: m for mid, m in zip(ids, moves)}
        S = dict(params)[names[0]]
        case = _case(name, 'drive', f'drive-{best}moves', key, split=split, root='operate',
                     args={'moves': 'Record<string, { ' + '; '.join(f'{pn}: {pt}' for pn, pt in params[1:]) + ' }>',
                           'target': 'unknown'}, returns='string',
                     instruction=pick(DRIVE_TEXT, key).format(service=sv, fn=fn, goal=d['goal']),
                     inputs={'moves': moves_input, 'target': target}, expected=None, oracle='exact',
                     level=2 if best == 2 else 3, files=_layout(files, 'operate'),
                     reference_root=[['eval', {'code': code, 'finish': True}]], services={sv: service},
                     revision=revision, slice_='inline_placement', domain='actor')
        case['curriculum']['external']['call_sets'] = {'service': sv, 'mode': 'state', 'finish': 'finish',
                                                       'gold': actions, 'wrong': wrong, 'alias': []}
        case['curriculum']['external']['oracle_depth'] = 'effect-state'
        out.append(case)
    return out


def cases(seed=20261103, only=''):
    revision = subprocess.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    names = [n for n in CODEBASES if not only or n in only.split(',')]
    out = []
    for name in names:
        rng = random.Random(f'{seed}|{name}')
        out += build_codebase(name, rng, revision)
    ids = [c['id'] for c in out]
    dup = {i for i in ids if ids.count(i) > 1}
    assert not dup, dup
    return out
