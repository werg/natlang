#!/usr/bin/env python3
"""Python native engine vs SGLang actor argmax on record decisions (semantic-FP B10 "done": 20 record decisions).

Twenty finite-record decisions in the runtime's readout layout (``DECISION_SYSTEM_PROMPT``, the call's opening, the
readout request): joint mode (the product of the options <= 32, every record one closed option, the existing request
text) and field mode (the FORM request, one ``open_reply`` step per field). Both engines score the same checkpoint.

    build OUT                       write OUT/decisions.json
    native OUT --checkpoint C       the native engine in this process (``decide_many_native``, the served path): joint
                                    decisions, and field decisions step by step along the native argmax; OUT/native.json
    sglang OUT --endpoint URL       the SGLang actor front: joint items through POST /v1/natlang/score (closed, the
                                    served path) and every item's options through POST /v1/neuralese/sentinel (the
                                    SGLang input log-probabilities of exactly the native items' prompt + continuation
                                    ids; field steps use this, so an actor whose code predates ``open_reply`` scores
                                    them too); sequential requests; OUT/sglang.json
    llama-items OUT                 the native run's score items (messages, continuations, open_reply) as
                                    OUT/llama-items.json for scripts/wllama_smoke.mjs --items (the managed llama-server
                                    and the in-page engine score them in one run, same GGUF, release template)
    llama OUT --smoke RESULT        the smoke's --out file as OUT/llama.json and OUT/page.json
    compare OUT                     per decision (field: per step, on the native prefix) argmax agreement and the
                                    largest |log-prob difference| per option against the native engine, for SGLang and,
                                    when present, the llama-server and the page; OUT/result.json

Rendering is ``decision_items`` on the LFM3 tokenizer (``tokenizer_renderer``), the rendering both engines serve with
(the options' shared tokens, ``</think>`` and the record's opening, are prompt). sglang.json keeps each option's
per-token sentinel log-probabilities. The 2026-11-23 hunt (DECISIONS "decision parity: FP8 KV, BF16 head logits"):
the native-vs-SGLang gap is the actor's FP8 KV (BF16 KV: TV mean 0.012 vs 0.038), the score-vs-sentinel gap batch
composition and radix splits, amplified by FP8 KV.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / 'training/neuralese'), str(REPO)]

from natlang_neuralese.data.decision_graph import (DECISION_SYSTEM_PROMPT, decision_prompt,  # noqa: E402
                                                   field_candidates)

JOINT_LIMIT = 32
LFM3 = Path.home() / '.cache/huggingface/hub/models--LiquidAI--LFM3-1.7B/snapshots/40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd'
FIELDS_ONE = 'Answer this call now, without tools. Reply with exactly one JSON value of this form and nothing else: '
FIELDS_MANY = 'Answer this call now, without tools. Reply with exactly one JSON value of one of these forms and nothing else: '

# (function, instructions, arguments, arms); an arm is a list of (key, values) or a scalar JSON value (('value', v))
CASES = [
    ('triageEmail', 'Classify the email as spam or not, and say whether it needs an answer today.',
     {'email': 'Subject: Your invoice #4471 is overdue\n\nHi, the invoice from March is still open. Please pay by Friday '
               'or call me. -- Dana, Accounts'},
     [[('label', ['spam', 'ham']), ('urgent', [True, False])]]),
    ('readSentiment', 'Judge the review\'s overall polarity and whether it is sarcastic.',
     {'review': 'Oh great, another phone that lasts a whole four hours on a charge. Truly revolutionary.'},
     [[('polarity', ['positive', 'negative', 'neutral']), ('sarcastic', [True, False])]]),
    ('rateBug', 'Rate the bug report\'s severity and the component it belongs to.',
     {'report': 'Checkout button returns HTTP 500 for every user since the 14:00 deploy; orders cannot be placed.'},
     [[('severity', ['low', 'medium', 'high', 'critical']), ('component', ['ui', 'api', 'db'])]]),
    ('decideRefund', 'Decide whether to approve the refund request and the reason that applies under the policy.',
     {'request': 'Bought headphones 10 days ago, left earcup stopped working. Have the receipt.',
      'policy': 'Defective items within 30 days are refunded.'},
     [[('approve', [True, False]), ('reason', ['policy', 'fraud', 'goodwill'])]]),
    ('reviewChange', 'Give the code review verdict and whether the change needs tests.',
     {'diff': '- return a + b\n+ return a - b  # fix sign', 'description': 'Fix subtraction helper'},
     [[('verdict', ['approve', 'request_changes', 'comment']), ('needs_tests', [True, False])]]),
    ('detectLanguage', 'Name the language of the text and whether its register is formal.',
     {'text': 'Sehr geehrte Damen und Herren, hiermit kündige ich meinen Vertrag fristgerecht.'},
     [[('lang', ['en', 'de', 'fr', 'es']), ('formal', [True, False])]]),
    ('answerInvite', 'Decide how to answer the meeting invitation given the calendar, and whether to notify the team.',
     {'invite': 'Quarterly planning, Thu 10:00-12:00', 'calendar': 'Thu 09:00-13:00 dentist (blocked)'},
     [[('action', ['accept', 'decline', 'tentative']), ('notify', [True, False])]]),
    ('resolveClaim', 'Resolve the warranty claim: no action, or a refund of the right amount.',
     {'claim': 'The $50 kettle arrived with a cracked lid.', 'policy': 'Refund the item price for damage in transit.'},
     [('value', 'none'), [('kind', ['refund']), ('amount', [10, 50, 100])]]),
    ('routeEmail', 'Route the message to a team and set its priority.',
     {'message': 'Our enterprise contract renewal is next week and legal has not received the redlines yet.'},
     [[('team', ['sales', 'support', 'billing', 'legal']), ('priority', ['p1', 'p2'])]]),
    ('moderatePost', 'Decide whether the post is allowed and its category.',
     {'post': 'BUY CHEAP WATCHES!!! visit cheap-watchez dot biz, 90% off today only'},
     [[('allowed', [True, False]), ('category', ['hate', 'spam', 'nsfw', 'ok'])]]),
    ('triageTicket', 'Label the ticket, rate its urgency and route it.',
     {'ticket': 'I was charged twice for my subscription this month. Please refund the duplicate charge.'},
     [[('label', ['spam', 'ham', 'other']), ('urgency', ['low', 'mid', 'high']),
       ('route', ['billing', 'tech', 'sales', 'abuse'])]]),
    ('scoreReview', 'Read the product review: its star rating, main topic and whether the writer recommends it.',
     {'review': 'Took three weeks to arrive and the box was crushed, but the blender itself is excellent. 4/5.'},
     [[('stars', [1, 2, 3, 4, 5]), ('topic', ['price', 'quality', 'shipping', 'service']),
       ('recommend', [True, False])]]),
    ('triagePatient', 'Assign the acuity level (1 most urgent), the department and whether a follow-up is needed.',
     {'note': 'Adult, sudden crushing chest pain radiating to left arm for 20 minutes, sweating.'},
     [[('acuity', [1, 2, 3, 4, 5]), ('department', ['er', 'gp', 'derm', 'ortho', 'cardio']),
       ('followup', [True, False])]]),
    ('classifyPullRequest', 'Classify the pull request by type and scope, and say whether it breaks the API.',
     {'title': 'cli: rename --out to --output and drop the old flag', 'files': 'src/cli/args.ts, docs/cli.md'},
     [[('type', ['feat', 'fix', 'docs', 'refactor', 'test', 'chore']), ('scope', ['core', 'cli', 'ui', 'ci']),
       ('breaking', [True, False])]]),
    ('handleRequest', 'Handle the customer request: a refund (amount, approval, channel), an escalation, or nothing.',
     {'request': 'Your courier left my $100 order in the rain and it is ruined. I paid by card.'},
     [[('kind', ['refund']), ('amount', [10, 25, 50, 100, 250]), ('approved', [True, False]),
       ('channel', ['card', 'wallet', 'check'])],
      [('kind', ['escalate']), ('team', ['billing', 'legal', 'security', 'exec']), ('urgent', [True, False])],
      ('value', 'none')]),
    ('screenApplicant', 'Screen the application: the decision, the seniority, whether remote, and the best team.',
     {'cv': '9 years building data pipelines (Spark, Airflow), led a team of 4, based in Lisbon, wants remote.',
      'opening': 'Senior data engineer, remote OK'},
     [[('decision', ['interview', 'reject', 'hold']), ('seniority', ['junior', 'mid', 'senior', 'staff']),
       ('remote', [True, False]), ('team', ['platform', 'data', 'mobile'])]]),
    ('pickOutfit', 'Pick an outfit for the forecast.',
     {'forecast': '4 °C, steady rain, wind 30 km/h'},
     [[('top', ['tshirt', 'sweater', 'jacket', 'coat']), ('bottom', ['shorts', 'jeans', 'trousers']),
       ('umbrella', [True, False]), ('shoes', ['sneakers', 'boots', 'sandals'])]]),
    ('readLogLine', 'Read the log line: its level, source and whether someone must act on it.',
     {'line': '2026-11-23T03:12:09Z ERROR db: connection pool exhausted (max=50), requests queued'},
     [[('level', ['debug', 'info', 'warn', 'error', 'fatal']), ('source', ['auth', 'db', 'net', 'disk']),
       ('actionable', [True, False])]]),
    ('rateTranslation', 'Rate the translation\'s adequacy and fluency (1 worst, 4 best) and its main error.',
     {'source': 'Der Zug fällt heute wegen eines Streiks aus.', 'translation': 'The train is late today.'},
     [[('adequacy', [1, 2, 3, 4]), ('fluency', [1, 2, 3, 4]),
       ('error', ['none', 'mistranslation', 'omission', 'addition'])]]),
    ('tagRecipe', 'Tag the recipe by course, diet, spiciness and preparation time.',
     {'recipe': 'Chickpea curry: onions, garlic, chili, coconut milk, chickpeas, spinach; simmer 25 minutes.'},
     [[('course', ['starter', 'main', 'dessert']), ('diet', ['vegan', 'vegetarian', 'omnivore']),
       ('spicy', [True, False]), ('time', ['quick', 'medium', 'long'])]]),
]

dumps = lambda value: json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def arm_values(arm):
    if isinstance(arm, tuple):
        return [dumps(arm[1])]
    return ['{' + ','.join(f'{dumps(k)}:{dumps(v)}' for (k, _), v in zip(arm, combo)) + '}'
            for combo in itertools.product(*[values for _, values in arm])]


def arm_form(arm):
    if isinstance(arm, tuple):
        return dumps(arm[1])
    return '{' + ','.join(f'{dumps(k)}:' + (dumps(vs[0]) if len(vs) == 1 else '<' + ' | '.join(map(dumps, vs)) + '>')
                          for k, vs in arm) + '}'


def build(out):
    decisions = []
    for function, instructions, arguments, arms in CASES:
        opening = (f'Function: {function}\n\nInstructions: {instructions}\n\nArguments:\n'
                   + '\n'.join(f'{name}: {dumps(value)}' for name, value in arguments.items()))
        values = list(dict.fromkeys(v for arm in arms for v in arm_values(arm)))
        joint = len(values) <= JOINT_LIMIT
        forms = [arm_form(arm) for arm in arms]
        request = (decision_prompt(values) if joint else
                   (FIELDS_ONE if len(forms) == 1 else FIELDS_MANY) + ', '.join(forms) + '.')
        decisions.append({'id': function, 'mode': 'joint' if joint else 'field', 'size': len(values),
                          'messages': [{'role': 'system', 'content': DECISION_SYSTEM_PROMPT},
                                       {'role': 'user', 'content': opening}, {'role': 'user', 'content': request}],
                          **({'options': values} if joint else {'forms': forms})})
    out.mkdir(parents=True, exist_ok=True)
    (out / 'decisions.json').write_text(json.dumps(decisions, indent=1, ensure_ascii=False) + '\n')
    print(json.dumps({'decisions': len(decisions), 'joint': sum(d['mode'] == 'joint' for d in decisions),
                      'field': sum(d['mode'] == 'field' for d in decisions)}))


def native(out, checkpoint, base):
    import torch
    from natlang_neuralese.common.code_provenance import code_provenance
    from natlang_neuralese.data.decision_graph import decision_forms
    from natlang_neuralese.serve import load_native_engine
    from natlang_neuralese.serve.native_learning import decide_many_native
    decisions = json.loads((out / 'decisions.json').read_text())
    started = time.time()
    engine = load_native_engine(str(checkpoint), base=str(base), device='cuda', dtype=torch.bfloat16,
                                training_state=False)
    items = []

    def score(decision, step, options, open_reply):
        item = {'messages': decision['messages'], 'continuations': options, **({'open_reply': True} if open_reply else {})}
        result = decide_many_native(engine, {'items': [item]})['results'][0]
        if 'error' in result:
            raise RuntimeError(f"{decision['id']} step {step}: {result['error']}")
        if open_reply and result.get('open_reply') is not True:
            raise RuntimeError('an open item came back without open_reply')
        items.append({'decision': decision['id'], 'mode': decision['mode'], 'step': step, 'options': options,
                      'open_reply': open_reply, 'log_probs': result['log_probs'], 'tokens': result['tokens']})
        return result['log_probs']
    for decision in decisions:
        if decision['mode'] == 'joint':
            score(decision, 0, decision['options'], False)
            continue
        forms, chosen = decision_forms(decision['messages']), []
        while True:
            candidates = field_candidates(forms, chosen)
            if not candidates:
                break
            texts = [text for text, _ in candidates]
            if len(texts) == 1:
                chosen.append(texts[0])
                continue
            scores = score(decision, len(chosen), texts, True)
            chosen.append(texts[max(range(len(texts)), key=scores.__getitem__)])
    info = engine.model_config
    (out / 'native.json').write_text(json.dumps({
        'engine': 'python-native', 'checkpoint': str(checkpoint), 'code': code_provenance(REPO),
        'checkpoint_sha256': info.get('checkpoint_sha256') or getattr(engine, 'native_identity', {}).get('checkpoint_sha256'),
        'dtype': 'bfloat16', 'seconds': round(time.time() - started, 1), 'items': items}, indent=1) + '\n')
    print(json.dumps({'items': len(items), 'seconds': round(time.time() - started, 1)}))


def _post(url, body, timeout=300):
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def sglang(out, endpoint):
    from transformers import AutoTokenizer
    from natlang_neuralese.data.decision_graph import tokenizer_renderer
    from natlang_neuralese.serve.native_learning import decision_items
    renderer = tokenizer_renderer(AutoTokenizer.from_pretrained(str(LFM3), local_files_only=True,
                                                                trust_remote_code=False), model_type='lfm3')
    native_items = json.loads((out / 'native.json').read_text())['items']
    decisions = {d['id']: d for d in json.loads((out / 'decisions.json').read_text())}
    info = json.loads(urllib.request.urlopen(endpoint + '/v1/neuralese/info', timeout=30).read())
    joint = [item for item in native_items if not item['open_reply']]
    started = time.time()
    served = _post(endpoint + '/v1/natlang/score', {'items': [
        {'messages': decisions[item['decision']]['messages'], 'continuations': item['options']} for item in joint]})
    served = dict(zip((item['decision'] for item in joint), served['results']))
    items = []
    for item in native_items:
        prompt, continuations = decision_items(renderer, decisions[item['decision']]['messages'], None, item['options'],
                                               item['open_reply'])
        ids = [int(v) for _, v in prompt]
        sums, per_token = [], []
        for continuation in continuations:
            logps = _post(endpoint + '/v1/neuralese/sentinel', {'token_ids': ids + continuation})['sglang']
            tail = logps[len(ids) - 1:]
            assert len(tail) == len(continuation)
            sums.append(float(sum(tail)))
            per_token.append([float(x) for x in tail])
        score = served.get(item['decision']) if not item['open_reply'] else None
        items.append({**{k: item[k] for k in ('decision', 'mode', 'step', 'options', 'open_reply')},
                      'sentinel_log_probs': sums, 'sentinel_token_log_probs': per_token,
                      **({'score_log_probs': score['log_probs']} if score and 'log_probs' in score else {}),
                      **({'score_error': score['error']} if score and 'error' in score else {})})
    binding = info.get('rl_policy_binding') or {}
    (out / 'sglang.json').write_text(json.dumps({
        'engine': 'sglang-actor', 'endpoint': endpoint, 'checkpoint_sha256': (info.get('native_state') or {}).get(
            'checkpoint_sha256'), 'actor_code': binding.get('code'),
        'actor_export': binding.get('actor_export'), 'kv_cache_dtype': (binding.get('serving_engine') or {}).get(
            'kv_cache_dtype'), 'seconds': round(time.time() - started, 1), 'items': items}, indent=1) + '\n')
    print(json.dumps({'items': len(items), 'seconds': round(time.time() - started, 1)}))


def llama_items(out):
    native_items = json.loads((out / 'native.json').read_text())['items']
    decisions = {d['id']: d for d in json.loads((out / 'decisions.json').read_text())}
    items = [{'messages': decisions[item['decision']]['messages'], 'continuations': item['options'],
              **({'open_reply': True} if item['open_reply'] else {})} for item in native_items]
    (out / 'llama-items.json').write_text(json.dumps(items) + '\n')
    print(json.dumps({'items': len(items), 'path': str(out / 'llama-items.json')}))


def llama(out, smoke):
    run = json.loads(Path(smoke).read_text())
    native_items = json.loads((out / 'native.json').read_text())['items']
    assert len(run['items']) == len(native_items), 'the smoke scored other items'
    for engine, results in (('llama-server', run['native']), ('wllama-page', run['page'])):
        items = []
        for item, result in zip(native_items, results):
            if item['open_reply'] and result.get('open_reply') is not True:
                raise SystemExit(f'{engine}: an open item was not scored open: {result}')
            items.append({**{k: item[k] for k in ('decision', 'mode', 'step', 'options', 'open_reply')},
                          **({'log_probs': result['log_probs'], 'tokens': result.get('tokens')} if 'log_probs' in result
                             else {'error': result.get('error')})})
        (out / ('llama.json' if engine == 'llama-server' else 'page.json')).write_text(json.dumps({
            'engine': engine, 'model': run['model'], 'server': run['server'], 'template': run['template'],
            'template_kwargs': run['template_kwargs'], 'items': items}, indent=1) + '\n')
    print(json.dumps({'items': len(native_items)}))


def compare(out):
    native_run = json.loads((out / 'native.json').read_text())
    sglang_run = json.loads((out / 'sglang.json').read_text())
    argmax = lambda xs: max(range(len(xs)), key=xs.__getitem__)
    rows, by_decision = [], {}
    for a, b in zip(native_run['items'], sglang_run['items']):
        assert (a['decision'], a['step'], a['options']) == (b['decision'], b['step'], b['options'])
        row = {'decision': a['decision'], 'mode': a['mode'], 'step': a['step'], 'options': len(a['options']),
               'native_argmax': a['options'][argmax(a['log_probs'])],
               'sglang_argmax': b['options'][argmax(b['sentinel_log_probs'])],
               'max_abs_diff': max(abs(x - y) for x, y in zip(a['log_probs'], b['sentinel_log_probs'])),
               'native_margin': sorted(a['log_probs'])[-1] - sorted(a['log_probs'])[-2]}
        row['agree'] = row['native_argmax'] == row['sglang_argmax']
        if 'score_log_probs' in b:
            row['score_route_argmax'] = a['options'][argmax(b['score_log_probs'])]
            row['score_route_max_abs_diff'] = max(abs(x - y) for x, y in zip(a['log_probs'], b['score_log_probs']))
            row['score_vs_sentinel_max_abs_diff'] = max(abs(x - y) for x, y in zip(b['score_log_probs'],
                                                                                  b['sentinel_log_probs']))
        rows.append(row)
        by_decision.setdefault(a['decision'], []).append(row)
    summary = {
        'decisions': len(by_decision), 'items': len(rows),
        'decisions_all_steps_agree': sum(all(r['agree'] for r in rs) for rs in by_decision.values()),
        'items_agree': sum(r['agree'] for r in rows),
        'joint_score_route_agree': sum(r.get('score_route_argmax') == r['native_argmax'] for r in rows
                                       if 'score_route_argmax' in r),
        'max_abs_diff': max(r['max_abs_diff'] for r in rows),
        'max_abs_diff_joint': max((r['max_abs_diff'] for r in rows if r['mode'] == 'joint'), default=None),
        'max_abs_diff_field': max((r['max_abs_diff'] for r in rows if r['mode'] == 'field'), default=None),
        'max_score_route_abs_diff': max((r['score_route_max_abs_diff'] for r in rows if 'score_route_max_abs_diff' in r),
                                        default=None),
        'native': {k: native_run.get(k) for k in ('checkpoint', 'checkpoint_sha256', 'code', 'dtype')},
        'sglang': {k: sglang_run.get(k) for k in ('endpoint', 'checkpoint_sha256', 'actor_code', 'actor_export',
                                                  'kv_cache_dtype')}}
    # the GGUF engines (Q8_0: int8 activation rounding, docs/ENGINE_PARITY.md numerics) against the native engine
    for name in ('llama', 'page'):
        path = out / f'{name}.json'
        if not path.exists():
            continue
        run = json.loads(path.read_text())
        agree = gaps = 0
        worst, per_decision = 0.0, {}
        for a, b, row in zip(native_run['items'], run['items'], rows):
            assert (a['decision'], a['step'], a['options']) == (b['decision'], b['step'], b['options'])
            if 'log_probs' not in b:
                row[f'{name}_error'] = b.get('error'); per_decision.setdefault(a['decision'], []).append(False); continue
            same = a['options'][argmax(b['log_probs'])] == row['native_argmax']
            row[f'{name}_argmax'] = a['options'][argmax(b['log_probs'])]
            row[f'{name}_max_abs_diff'] = max(abs(x - y) for x, y in zip(a['log_probs'], b['log_probs']))
            row[f'{name}_tokens_equal'] = b.get('tokens') == a.get('tokens')
            agree += same; gaps += not row[f'{name}_tokens_equal']; worst = max(worst, row[f'{name}_max_abs_diff'])
            per_decision.setdefault(a['decision'], []).append(same)
        summary[name] = {'engine': run['engine'], 'model': run['model'], 'template_kwargs': run['template_kwargs'],
                         'items_agree_with_native': agree, 'decisions_all_steps_agree': sum(all(v) for v in per_decision.values()),
                         'items_with_other_token_counts': gaps, 'max_abs_diff_vs_native': worst}
    (out / 'result.json').write_text(json.dumps({'summary': summary, 'items': rows}, indent=1) + '\n')
    print(json.dumps(summary, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('build', 'native', 'sglang', 'llama-items', 'llama', 'compare'):
        p = sub.add_parser(name)
        p.add_argument('out', type=Path)
        if name == 'native':
            p.add_argument('--checkpoint', type=Path, required=True)
            p.add_argument('--base', type=Path, default=LFM3)
        if name == 'sglang':
            p.add_argument('--endpoint', required=True)
        if name == 'llama':
            p.add_argument('--smoke', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'build':
        build(args.out)
    elif args.command == 'native':
        native(args.out, args.checkpoint, args.base)
    elif args.command == 'sglang':
        sglang(args.out, args.endpoint.rstrip('/'))
    elif args.command == 'llama-items':
        llama_items(args.out)
    elif args.command == 'llama':
        llama(args.out, args.smoke)
    else:
        compare(args.out)


if __name__ == '__main__':
    main()
