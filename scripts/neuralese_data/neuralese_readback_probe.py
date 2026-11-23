#!/usr/bin/env python3
"""Neuralese read-back acceptance probe: does a reader get back the text a producer wrote?

usage: neuralese_readback_probe.py --endpoint URL [--out OUT.jsonl] [--temperature 0] [--top-p 1] [--n 24]
                                   [--tokenizer PATH] [--arms state,inline,shuffled,captured]

Through the serving front's public API only (`/v1/neuralese/native-states/write-text`, `/v1/chat/completions`), so it
measures any engine that serves the protocol. For each text it writes the state, then asks the `natlang.read` call
(the runtime's own typed readout prompt, `ts-host/src/neuralese/combinators.ts` `natlang:model-read/1`) three ways:

- state: the written state (`native.text-write`) imported where the reader declares its value;
- captured: the same text as a produce span captured in a producer's own call (`native.capture` of an assistant reply
  that writes it, `select` = the text), imported the same way (needs `--tokenizer` to render the producer call);
- inline: the same text in its place (the upper bound: what the reader does with the text itself);
- shuffled: another text's state (the floor: a reader that ignores the state matches nothing).

It reports the exact read-back rate of each arm, the import receipts' modes, and how many replies carry a Neuralese
role marker (`<|neuralese_…|>`: a reader echoing an untrained marker row, or a sampled stray marker). A text-primed state must read back about as well as its inline text
and far above the shuffled arm (AGENTS: "A span that ablation cannot tell from shuffled state is a defect"); the
2026-11-23 failure (1 of 87 held read-backs) had state ~ shuffled ~ 0. The texts are synthetic (never held inputs).
"""
import argparse
import json
import random
import re
import sys
import urllib.error
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


def post(endpoint, path, body, attempts=3):
    """POST JSON; a server error or a dropped connection is retried (a busy shared front restarts or sheds load)."""
    import time
    for attempt in range(attempts):
        request = urllib.request.Request(endpoint.rstrip('/') + path, data=json.dumps(body).encode(),
                                         headers={'content-type': 'application/json'})
        try:
            with urllib.request.urlopen(request, timeout=900) as response:
                return json.load(response)
        except (urllib.error.URLError, ConnectionError, TimeoutError) as error:
            if attempt + 1 == attempts or (isinstance(error, urllib.error.HTTPError) and error.code < 500):
                raise
            time.sleep(15 * (attempt + 1))


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


MARKER = re.compile(r'<\|/?neuralese_(?:produce|consume)\|>')


def producer_ids(tokenizer, text):
    """(prompt ids, reply ids) of a producer call whose assistant reply writes ``text`` (the captured arm)."""
    messages = [{'role': 'system', 'content': 'You keep short notes for the team.'},
                {'role': 'user', 'content': 'Write down the note for the log.'}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    opened = prompt.rstrip('\n').endswith('<think>')
    reply = ('</think>\n\n' if opened else '') + 'Note:\n' + text + (tokenizer.eos_token or '')
    return (tokenizer(prompt, add_special_tokens=False)['input_ids'], tokenizer(reply, add_special_tokens=False)['input_ids'])


def raw_reply(reply):
    message = reply['choices'][0]['message']
    return (message.get('content') or '') + ''.join(call['function']['arguments'] for call in message.get('tool_calls') or ())


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
    parser.add_argument('--tokenizer', help='tokenizer of the served model (renders the captured arm\'s producer call)')
    parser.add_argument('--arms', default='state,inline,shuffled,captured')
    a = parser.parse_args()
    arms = a.arms.split(',')
    if 'captured' in arms and not a.tokenizer:
        arms.remove('captured')
    texts = TEXTS[:a.n]
    states = [post(a.endpoint, '/v1/neuralese/native-states/write-text', {'text': text, 'type': 'Neuralese<string>'})['id']
              for text in texts]
    shuffled = states[1:] + states[:1]
    captured = []
    if 'captured' in arms:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=False)
        for text in texts:
            prompt, reply = producer_ids(tokenizer, text)
            captured.append(post(a.endpoint, '/v1/neuralese/native-states/capture', {
                'prompt_token_ids': prompt, 'reply_token_ids': reply, 'select': text, 'type': 'Neuralese<string>'})['id'])
    rng = random.Random(a.seed)
    rows, hits, marked, modes = [], {arm: 0 for arm in arms}, {arm: 0 for arm in arms}, {arm: {} for arm in arms}
    refused = {arm: 0 for arm in arms}
    for index, text in enumerate(texts):
        parts = {'state': {'type': 'native_state', 'id': states[index]}, 'inline': {'type': 'text', 'text': text},
                 'shuffled': {'type': 'native_state', 'id': shuffled[index]},
                 **({'captured': {'type': 'native_state', 'id': captured[index]}} if captured else {})}
        for arm in arms:
            try:
                reply = post(a.endpoint, '/v1/chat/completions', {
                    'messages': reader_messages(parts[arm]), 'tools': TOOLS, 'max_tokens': a.max_tokens,
                    'temperature': a.temperature, 'top_p': a.top_p, 'seed': rng.randrange(1 << 30)})
            except urllib.error.HTTPError as error:
                # a refused reply (e.g. a protocol error for a sampled role marker) is a miss, recorded with its cause
                body = error.read().decode('utf-8', 'replace')[:600]
                refused[arm] += 1
                rows.append({'arm': arm, 'text': text, 'exact': False, 'error': {'status': error.code, 'body': body}})
                print(json.dumps({'arm': arm, 'exact': False, 'text': text[:60], 'error': error.code, 'body': body[:300]}),
                      flush=True)
                continue
            got = returned(reply)
            hit = same(got, text)
            hits[arm] += hit
            marker = bool(MARKER.search(raw_reply(reply)))
            marked[arm] += marker
            imports = (reply.get('native_state') or {}).get('imports')
            for receipt in imports or ():
                modes[arm][receipt.get('mode')] = modes[arm].get(receipt.get('mode'), 0) + 1
            rows.append({'arm': arm, 'text': text, 'returned': got, 'exact': hit, 'marker': marker, 'imports': imports})
            print(json.dumps({'arm': arm, 'exact': hit, 'text': text[:60], 'returned': got[:80], 'marker': marker,
                              'modes': [i.get('mode') for i in imports or ()]}), flush=True)
    if a.out:
        with open(a.out, 'w') as stream:
            stream.writelines(json.dumps(row) + '\n' for row in rows)
    summary = {arm: f'{count}/{len(texts)}' for arm, count in hits.items()}
    print(json.dumps({'summary': summary, 'marker_replies': marked, 'refused': refused, 'modes': modes, 'endpoint': a.endpoint,
                      'temperature': a.temperature, 'top_p': a.top_p}))
    # Acceptance: the state reads back nearly as well as the inline text and clearly above the shuffled state.
    state, inline, floor = hits.get('state', 0), hits.get('inline', 0), hits.get('shuffled', 0)
    ok = state >= max(1, inline - max(1, len(texts) // 8)) and state > floor + len(texts) // 4
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
