#!/usr/bin/env python3
"""Neuralese read-back acceptance probe: does a reader get back the text a producer wrote?

usage: neuralese_readback_probe.py --endpoint URL [--out OUT.jsonl] [--temperature 0] [--top-p 1] [--n 24]

Through the serving front's public API only (`/v1/neuralese/native-states/write-text`, `/v1/chat/completions`), so it
measures any engine that serves the protocol. For each text it writes the state, then asks the `natlang.read` call
(the runtime's own typed readout prompt, `ts-host/src/neuralese/combinators.ts` `natlang:model-read/1`) three ways:

- state: the written state imported where the reader declares its value;
- inline: the same text in its place (the upper bound: what the reader does with the text itself);
- shuffled: another text's state (the floor: a reader that ignores the state matches nothing).

It reports the exact read-back rate of each arm. A text-primed state must read back about as well as its inline text
and far above the shuffled arm (AGENTS: "A span that ablation cannot tell from shuffled state is a defect"); the
2026-11-23 failure (1 of 87 held read-backs) had state ~ shuffled ~ 0. The texts are synthetic (never held inputs).
"""
import argparse
import json
import random
import re
import sys
import urllib.request

TEXTS = [
    'Gate C opens at 06:40.', 'Ticket T-118 is waiting for the supplier.', 'The blue folder holds the March invoices.',
    'Mira booked the north room for Thursday.', 'Batch 7 failed the humidity check.',
    'Register after 2 notices.\nCase QX/31/0772.', 'Owner: Dana Kell\nStatus: approved',
    'Shelf 4 has 12 jars of plum jam.', 'The courier left the parcel with the porter.',
    'Order 5531 ships on 2026-12-03.', 'Pump B was serviced in May; pump A is overdue.',
    'Lena asks Omar to swap shifts.', 'The kettle was in the attic before it was in the garage.',
    'Route 9 is closed between Ash Lane and the mill.', 'Invoice total: 418.20 EUR',
    'Step S-04 pending\nStep S-05 applied', 'The vote passed 7 to 2.', 'Room 12 needs two more chairs.',
    'Tomas lent the drill to his neighbour.', 'Sample R-2 tested negative.', 'Train 41 departs from platform 3.',
    'The draft mentions the harbour twice.', 'Unit E-11 is due for inspection on 2027-01-15.',
    'Kai finished the bridge section first.',
]

READ_BODY = ('Read the supplied Neuralese value. Return exactly the value it contains in its declared element type. '
             'Preserve its content and structure; do not summarize, add fields, or add commentary.')
TOOLS = [{'type': 'function', 'function': {'name': 'return_result', 'description': "Return this call's result.",
          'parameters': {'type': 'object', 'properties': {'status': {'type': 'string', 'enum': ['success', 'failure']},
                                                          'value': {}}, 'required': ['status', 'value']}}}]


def post(endpoint, path, body):
    request = urllib.request.Request(endpoint.rstrip('/') + path, data=json.dumps(body).encode(),
                                     headers={'content-type': 'application/json'})
    with urllib.request.urlopen(request, timeout=900) as response:
        return json.load(response)


def reader_messages(part):
    """The natlang.read call as the runtime opens it: the instructions, then the declaring eval result."""
    return [
        {'role': 'system', 'content': 'Answer with the return_result tool: call it with status "success" and a value '
                                      'of the declared type (for a string result, the text itself).'},
        {'role': 'user', 'content': 'You are inside this call: natlang.read(value: Neuralese<string>): string\n\n'
                                    f'Instructions:\n{READ_BODY}'},
        {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'c0', 'type': 'function', 'function': {
            'name': 'eval', 'arguments': json.dumps({'code': 'const value: Neuralese<string> = read_inputs().value;'})}}]},
        {'role': 'tool', 'tool_call_id': 'c0', 'content': [
            {'type': 'text', 'text': 'value: Neuralese<string> = '}, part,
            {'type': 'text', 'text': '\nDeclared value for the rest of this call.'}]},
    ]


def returned(reply):
    message = reply['choices'][0]['message']
    for call in message.get('tool_calls') or ():
        if call['function']['name'] == 'return_result':
            try:
                value = json.loads(call['function']['arguments']).get('value')
            except (ValueError, AttributeError):
                return call['function']['arguments']
            return value if isinstance(value, str) else json.dumps(value)
    return message.get('content') or ''


def same(a, b):
    squash = lambda text: re.sub(r'\s+', ' ', text).strip()
    return squash(a) == squash(b)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--endpoint', required=True)
    parser.add_argument('--out')
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top-p', type=float, default=1.0)
    parser.add_argument('--max-tokens', type=int, default=700)
    parser.add_argument('--n', type=int, default=len(TEXTS))
    parser.add_argument('--seed', type=int, default=20261123)
    a = parser.parse_args()
    texts = TEXTS[:a.n]
    states = [post(a.endpoint, '/v1/neuralese/native-states/write-text', {'text': text, 'type': 'Neuralese<string>'})['id']
              for text in texts]
    shuffled = states[1:] + states[:1]
    rng = random.Random(a.seed)
    rows, hits = [], {'state': 0, 'inline': 0, 'shuffled': 0}
    for index, text in enumerate(texts):
        for arm, part in (('state', {'type': 'native_state', 'id': states[index]}), ('inline', {'type': 'text', 'text': text}),
                          ('shuffled', {'type': 'native_state', 'id': shuffled[index]})):
            reply = post(a.endpoint, '/v1/chat/completions', {
                'messages': reader_messages(part), 'tools': TOOLS, 'max_tokens': a.max_tokens,
                'temperature': a.temperature, 'top_p': a.top_p, 'seed': rng.randrange(1 << 30)})
            got = returned(reply)
            hit = same(got, text)
            hits[arm] += hit
            imports = (reply.get('native_state') or {}).get('imports')
            rows.append({'arm': arm, 'text': text, 'returned': got, 'exact': hit, 'imports': imports})
            print(json.dumps({'arm': arm, 'exact': hit, 'text': text[:60], 'returned': got[:80],
                              'modes': [i.get('mode') for i in imports or ()]}), flush=True)
    if a.out:
        with open(a.out, 'w') as stream:
            stream.writelines(json.dumps(row) + '\n' for row in rows)
    summary = {arm: f'{count}/{len(texts)}' for arm, count in hits.items()}
    print(json.dumps({'summary': summary, 'endpoint': a.endpoint, 'temperature': a.temperature, 'top_p': a.top_p}))
    # Acceptance: the state reads back nearly as well as the inline text and clearly above the shuffled state.
    ok = hits['state'] >= max(1, hits['inline'] - max(1, len(texts) // 8)) and hits['state'] > hits['shuffled'] + len(texts) // 4
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
