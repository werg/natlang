"""Algorithmic and architectural examples with independent state/effect oracles."""
from pathlib import Path

from ..host import load, load_fold
from ..surface import is_previewed
from ..types import format_type
from .programs import Plan, Program

CB = Path(__file__).resolve().parents[2] / 'codebases'
HANDCRAFTED_RATE = .25


def call(fn, to, **inputs):
    return [('call', {'function': fn, 'to': to, 'inputs': inputs})]


def mark(start, end=None, skipped=False):
    return [('mark_done', {'start': start, 'end': end or start, **({'skipped': True} if skipped else {})})]


def leaf(answer):
    def script(lam):
        hidden = [f'args/{name}' for name, value in lam.in_.items() if is_previewed(value)]
        if hidden:
            yield [('read', {'path': path}) for path in hidden]
        yield [('write', {'path': 'return', 'type': format_type(lam.type.returns), 'value': answer(lam.in_)})]
    return Plan('script', script=script)


MESSAGES = [
    ('All customers are unable to pay right now.', 'urgent'),
    ('We have an active security breach.', 'urgent'),
    ('The service is down for everyone.', 'urgent'),
    ('Yesterday\'s outage was resolved; everything works now.', 'normal'),
    ('Could a future outage affect scheduled payments?', 'normal'),
    ('Please send a copy of last month\'s invoice.', 'normal'),
]


def reconciliation(rng):
    if rng.random() < HANDCRAFTED_RATE:
        # Keep a few exact, small traces in the corpus.  The values fit in the
        # opening workspace listing, so the lesson is to pass paths directly to
        # the helpers instead of reading every event field first.
        cases = [
            ([{'id': 'constructor', 'tier': 'priority'}],
             [{'id': 'e0', 'customer': 'constructor',
               'message': 'We have an active security breach.', 'cents': 125}]),
            ([{'id': 'constructor', 'tier': 'standard'}, {'id': '__proto__', 'tier': 'priority'}],
             [{'id': 'e0', 'customer': 'constructor',
               'message': 'Please send a copy of last month\'s invoice.', 'cents': 80},
              {'id': 'e1', 'customer': '__proto__',
               'message': 'The service is down for everyone.', 'cents': -20},
              {'id': 'e0', 'customer': 'constructor',
               'message': 'We have an active security breach.', 'cents': 9999}]),
            ([], [{'id': 'e0', 'customer': 'unknown',
                   'message': 'Could a future outage affect scheduled payments?', 'cents': 7}]),
        ]
        customers, events = cases[rng.randrange(len(cases))]
        truth = dict(MESSAGES)
    else:
        customers = [{'id': key, 'tier': rng.choice(['priority', 'standard'])}
                     for key in ['constructor', '__proto__', 'acme'][:rng.randint(0, 3)]]
        truth, events = {}, []
        for i in range(rng.randint(0, 14)):
            text, label = rng.choice(MESSAGES)
            truth[text] = label
            events.append({'id': f'e{i}', 'customer': rng.choice(['constructor', '__proto__', 'acme', 'unknown']),
                           'message': text, 'cents': rng.randint(-200, 400)})
            if rng.random() < .3:
                # Conflicting redeliveries must not change the first event's amounts or classification.
                events.append({**events[-1], 'cents': 9999, 'message': 'We have an active security breach.'})
    totals, urgent, unmatched, seen = {}, [], [], set()
    ids = {c['id'] for c in customers}
    for e in events:
        if e['id'] in seen:
            continue
        seen.add(e['id'])
        if e['customer'] not in ids:
            unmatched.append(e['id'])
        else:
            totals[e['customer']] = totals.get(e['customer'], 0) + e['cents']
            if truth[e['message']] == 'urgent':
                urgent.append(e['id'])
    expected = dict(totals=totals, urgent=urgent, unmatched=unmatched, duplicates=len(events)-len(seen))

    def root(lam):
        yield call('join_events', 'let/joined', customers='args/customers', events='args/events')
        yield mark(2)
        yield [('call', {'function': 'assess', 'to': 'let/labels', 'over': 'let/joined/rows'})]
        yield mark(3)
        yield call('summarize', 'return', joined='let/joined', labels='let/labels')
        yield mark(4)
    plans = {'reconcile': Plan('script', script=root), 'assess': leaf(lambda a: truth[a['row']['message']])}
    source = CB/'reconciliation/reconcile.nl'
    return Program('cb_reconciliation', {}, {}, expected, plans,
                   loader=lambda: load(source, {'customers': customers, 'events': events}),
                   source_semantics={'codebase_file': str(source),
                       'inputs': {'customers': customers, 'events': events},
                       'operations': [
                           {'op': 'invoke', 'function': 'join_events', 'target': 'let/joined',
                            'arguments': {'customers': 'args/customers', 'events': 'args/events'}},
                           {'op': 'invoke', 'function': 'assess', 'target': 'let/labels',
                            'foreach': 'let/joined/rows'},
                           {'op': 'invoke', 'function': 'summarize', 'target': 'return',
                            'arguments': {'joined': 'let/joined', 'labels': 'let/labels'}}],
                       'leaf_oracles': {'assess': {'parameter': 'row', 'lookup_path': ['message'],
                           'cases': [{'input': message, 'output': label} for message, label in truth.items()]}}})


DESCRIPTIONS = [('Patch the active security vulnerability.', 0),
                ('Improve the customer-facing checkout page.', 1),
                ('Clean up internal build logs.', 2)]


def dependency_plan(rng):
    if rng.random() < HANDCRAFTED_RATE:
        # Include both sides of the repeat guard in a short, readable state:
        # an independent task exercises init/until/max, while a missing
        # dependency exercises the stall path and its skipped alternatives.
        cases = [
            ([{'id': 'constructor', 'needs': [],
               'description': 'Patch the active security vulnerability.'}],
             {'constructor': 0}),
            ([{'id': '__proto__', 'needs': ['missing-dependency'],
               'description': 'Improve the customer-facing checkout page.'}],
             {'__proto__': 1}),
            ([], {}),
        ]
        tasks, priorities = cases[rng.randrange(len(cases))]
    else:
        tasks, priorities = [], {}
        for i in range(rng.randint(0, 8)):
            text, priority = rng.choice(DESCRIPTIONS)
            key = ['constructor', '__proto__'][i] if i < 2 else f't{i}'
            priorities[key] = priority
            tasks.append({'id': key, 'needs': [t['id'] for t in tasks if rng.random() < .3], 'description': text})
        if tasks and rng.random() < .35:
            tasks[0]['needs'] = [tasks[-1]['id']]  # cycle or self-cycle
        if tasks and rng.random() < .2:
            tasks[-1]['needs'].append('missing-dependency')
    order = []
    while len(order) < len(tasks):
        ready = [t for t in tasks if t['id'] not in order and all(n in order for n in t['needs'])]
        if not ready:
            break
        order.append(min(ready, key=lambda t: (priorities[t['id']], t['id']))['id'])
    expected = {'tasks': tasks, 'order': order, 'blocked': [t['id'] for t in tasks if t['id'] not in order], 'finished': True}

    def root(lam):
        yield call('prepare', 'let/initial', tasks='args/tasks')
        r = yield [('read', {'path': 'let/initial/finished'})]
        yield mark(2, 3)
        if r.value:
            yield [('write', {'path': 'return', 'type': 'State', 'source': 'let/initial'})]
            yield mark(4)
            yield mark(5, 6, skipped=True)
        else:
            yield mark(4, skipped=True)
            yield [('call', {'function': 'step', 'to': 'return', 'init': 'let/initial', 'until': 'finished', 'max': 16})]
            yield mark(5, 6)

    def step(lam):
        r = yield call('ready_tasks', 'let/ready', state='args/state')
        yield mark(2, 3)
        if not r.value:
            yield call('stall', 'return', state='args/state')
            yield mark(4)
            yield mark(5, 7, skipped=True)
        else:
            yield mark(4, skipped=True)
            yield call('choose', 'let/chosen', ready='let/ready')
            yield mark(5, 6)
            yield call('advance', 'return', state='args/state', chosen='let/chosen')
            yield mark(7)
    plans = {'plan': Plan('script', script=root), 'step': Plan('script', script=step),
             'choose': leaf(lambda a: min(a['ready'], key=lambda t: (priorities[t['id']], t['id']))['id'])}
    source = CB/'dependency_plan/plan.nl'
    root_ops = [
        {'op': 'invoke', 'function': 'prepare', 'target': 'let/initial',
         'arguments': {'tasks': 'args/tasks'}},
        {'op': 'branch', 'test_path': 'let/initial/finished', 'test': 'truthy',
         'then': [{'op': 'assign', 'target': 'return', 'value_type': 'State', 'from': 'let/initial'}],
         'else': [{'op': 'invoke', 'function': 'step', 'target': 'return',
                   'initial': 'let/initial', 'until': 'finished', 'max_steps': 16}]}]
    step_ops = [
        {'op': 'invoke', 'function': 'ready_tasks', 'target': 'let/ready',
         'arguments': {'state': 'args/state'}},
        {'op': 'branch', 'test_path': 'let/ready', 'test': 'empty',
         'then': [{'op': 'invoke', 'function': 'stall', 'target': 'return',
                   'arguments': {'state': 'args/state'}}],
         'else': [{'op': 'invoke', 'function': 'choose', 'target': 'let/chosen',
                   'arguments': {'ready': 'let/ready'}},
                  {'op': 'invoke', 'function': 'advance', 'target': 'return',
                   'arguments': {'state': 'args/state', 'chosen': 'let/chosen'}}]}]
    return Program('cb_dependency_plan', {}, {}, expected, plans,
                   loader=lambda: load(source, {'tasks': tasks}),
                   source_semantics={'codebase_file': str(source), 'inputs': {'tasks': tasks},
                       'operations': root_ops, 'functions': {'step': step_ops},
                       'leaf_rules': {'choose': {'kind': 'min_priority_then_id', 'priorities': priorities}}})


EVENT_TEXT = {'start': 'I would like to place this new order.', 'paid': 'Payment has succeeded.',
              'failed': 'The shipment failed; it could not be delivered.', 'sent': 'Shipping is complete.',
              'cancel': 'Please cancel my order.'}


def order_saga(rng):
    handcrafted = rng.random() < HANDCRAFTED_RATE
    if handcrafted:
        # Duplicate suppression and compensation in one tiny fold.  The first
        # delivery is deliberately made to lose its acknowledgement so the
        # target includes a same-call resume with only function + destination.
        events = [
            {'id': 'e0', 'order': 'constructor', 'text': EVENT_TEXT['start']},
            {'id': 'e0', 'order': 'constructor', 'text': EVENT_TEXT['start']},
            {'id': 'e1', 'order': 'constructor', 'text': EVENT_TEXT['cancel']},
        ]
    else:
        events = []
        for order in ['constructor', '__proto__'][:rng.randint(1, 2)]:
            sequence = ['start', 'paid', rng.choice(['failed', 'cancel', 'sent'])]
            if rng.random() < .3:
                sequence.insert(0, 'paid')  # out-of-order confirmation is ignored
            for kind in sequence:
                event = {'id': f'e{len(events)}', 'order': order, 'text': EVENT_TEXT[kind]}
                events.append(event)
                if rng.random() < .5:
                    events.append(dict(event))
    seen, orders, commands = [], {}, []
    kinds = {v: k for k, v in EVENT_TEXT.items()}
    for e in events:
        if e['id'] in seen:
            continue
        seen.append(e['id'])
        old, kind, operations = orders.get(e['order']), kinds[e['text']], []
        if old is None and kind == 'start':
            orders[e['order']], operations = 'reserved', ['reserve']
        elif old == 'reserved' and kind == 'paid':
            orders[e['order']], operations = 'paid', ['ship']
        elif old == 'paid' and kind == 'sent':
            orders[e['order']] = 'done'
        elif old in ('reserved', 'paid') and kind in ('cancel', 'failed'):
            orders[e['order']], operations = 'cancelled', ['refund', 'release'] if old == 'paid' else ['release']
        commands += [dict(key=e['id']+':'+str(i), order=e['order'], operation=op) for i, op in enumerate(operations)]
    expected = {'seen': seen, 'orders': orders, 'outbox': []}
    delivered, delivered_keys, failed = [], set(), set()
    fail_key = (commands[0]['key'] if handcrafted and commands else
                rng.choice(commands)['key'] if commands and rng.random() < .7 else None)

    def send(args):
        command = args[0]
        if command['key'] not in delivered_keys:
            delivered_keys.add(command['key'])
            delivered.append(command)
        if command['key'] == fail_key and fail_key not in failed:
            failed.add(fail_key)
            raise RuntimeError('delivery succeeded but its acknowledgement was lost')
        return None

    def step(lam):
        r = yield call('seen', 'let/duplicate', acc='args/acc', item='args/item')
        yield mark(2, 3)
        if r.value:
            yield [('write', {'path': 'return', 'type': 'State', 'source': 'args/acc'})]
            yield mark(4)
            yield mark(5, 11, skipped=True)
            return
        yield mark(4, skipped=True)
        yield call('read_event', 'let/kind', text='args/item/text')
        yield mark(5, 6)
        yield call('transition', 'let/next', acc='args/acc', item='args/item', kind='let/kind')
        yield mark(7)
        r = yield call('dispatch', 'let/sent', commands='let/next/outbox')
        if r.kind == 'quiesced':
            r = yield [('call', {'function': 'dispatch', 'to': 'let/sent'})]
            assert r.kind == 'done'
            yield mark(9)
        else:
            yield mark(9, skipped=True)
        yield mark(8)
        yield mark(10, skipped=True)
        yield call('clear_outbox', 'return', state='let/next')
        yield mark(11)
    plans = {'step': Plan('script', script=step), 'read_event': leaf(lambda a: kinds[a['text']])}
    source = CB/'order_saga/step.nl'
    initial = {'seen': [], 'orders': {}, 'outbox': []}
    step_ops = [
        {'op': 'invoke', 'function': 'seen', 'target': 'let/duplicate',
         'arguments': {'acc': 'args/acc', 'item': 'args/item'}},
        {'op': 'branch', 'test_path': 'let/duplicate', 'test': 'truthy',
         'then': [{'op': 'assign', 'target': 'return', 'value_type': 'State', 'from': 'args/acc'}],
         'else': [
             {'op': 'invoke', 'function': 'read_event', 'target': 'let/kind',
              'arguments': {'text': 'args/item/text'}},
             {'op': 'invoke', 'function': 'transition', 'target': 'let/next',
              'arguments': {'acc': 'args/acc', 'item': 'args/item', 'kind': 'let/kind'}},
             {'op': 'invoke', 'function': 'dispatch', 'target': 'let/sent',
              'arguments': {'commands': 'let/next/outbox'}, 'retry_on_quiesced': 1},
             {'op': 'invoke', 'function': 'clear_outbox', 'target': 'return',
              'arguments': {'state': 'let/next'}}]}]
    return Program('cb_order_saga', {}, {}, lambda v: v == expected and delivered == commands, plans,
                   loader=lambda: load_fold(source, initial, list(events)),
                   capabilities={'queue.send': send},
                   source_semantics={'fold': True, 'inputs': {}, 'events': events,
                       'expected': expected, 'operations': step_ops,
                       'leaf_oracles': {'read_event': {'parameter': 'text', 'cases': [
                           {'input': text, 'output': kind} for text, kind in kinds.items()]}},
                       'effects': {'queue.send': {'kind': 'deliver_once_ack_loss',
                           'fail_key': fail_key, 'expected_delivered': commands}}})


ARCHITECTURES = {'cb_reconciliation': reconciliation, 'cb_dependency_plan': dependency_plan, 'cb_order_saga': order_saga}
