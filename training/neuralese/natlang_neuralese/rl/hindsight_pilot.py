"""Hindsight-hint pilot (plans/neuralese/NEURALESE_HINDSIGHT_HINTS.md, "Pilot").

Stages, each resumable into ``--out``:

* ``select``: successful (quality >= 1) RL episodes of custody-checked train worlds, shortest per task; writes each
  tau*'s ``writeHint`` inputs (root consumer, the call value, the crisp overview, the root's visible facts other than
  the trajectory itself) and the node classes (``hindsight.node_classes``).
* ``write``: ``writeHint`` through the natlang runtime against a native server, K times per tau* (``natlang call``).
* ``score``: log p(tau*) of the root's actions under the frozen policy with no hint, the facts-only text hint, each
  written hint imported as Neuralese (channel off/on), the hint as plain text and tau* verbatim (a leakage reference).
* ``train``: phase 1, pathwise only on the written hints: the producer adapter and a writer LoRA bound to the hint
  invocations; the backbone and every other invocation stay frozen. Reports reconstruction on held-out tau*.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[4]
WRITER = REPO / 'applications/guidance-writer/writeHint.nl'


def _jsonl(path):
    with open(path) as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=1))
    os.replace(tmp, path)


# ---------------------------------------------------------------- tau* as a value

def _content(message):
    value = message.get('content')
    if isinstance(value, list):
        return '\n'.join(part.get('text', '') for part in value if isinstance(part, dict))
    return value or ''


def call_value(rollout, tokenizer):
    """The root call of a single-call episode as a ``TrajectoryCall`` value (steps in the TranscriptEntry shape)."""
    turns = rollout['turns']
    last = turns[-1]
    messages = last['messages']
    prompt = '\n'.join(_content(m) for m in messages if m['role'] == 'user')[:20000]
    steps, turn = [], 0
    for index, message in enumerate(messages):
        if message['role'] != 'assistant':
            continue
        outputs = [m for m in messages[index + 1:] if m['role'] == 'tool']
        for number, call in enumerate(message.get('tool_calls') or []):
            function = call.get('function') or {}
            try:
                arguments = json.loads(function.get('arguments') or '{}')
            except json.JSONDecodeError:
                arguments = {'raw': function.get('arguments')}
            output = _content(outputs[number]) if number < len(outputs) else ''
            step = {'turn': turn, 'tool': function.get('name') or 'reply', 'arguments': arguments,
                    'status': 'error' if output.lstrip().lower().startswith(('error', 'typeerror', 'referenceerror'))
                    else 'ok', 'output': output}
            if isinstance(arguments, dict) and isinstance(arguments.get('code'), str):
                step['code'] = arguments['code']
                step['arguments'] = {key: value for key, value in arguments.items() if key != 'code'}
            reasoning = message.get('reasoning_content')
            if reasoning and number == 0:
                step['reasoning'] = reasoning
            steps.append(step)
        turn += 1
    final = tokenizer.decode([sample['token_id'] for sample in last['sampling']['samples']], skip_special_tokens=True)
    steps.append({'turn': turn, 'tool': 'final', 'arguments': {}, 'status': 'completed', 'output': final})
    return {'call': last['invocation_id'], 'function': rollout.get('family') or rollout['task_id'], 'role': 'root',
            'prompt': prompt, 'steps': steps, 'result': {'status': 'success'}, 'children': []}


def overview(call):
    """Mirror of applications/guidance-writer/trajectoryOverview.ts, checked against it by the select stage."""
    size = lambda value: len(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False))
    return {'call': call['call'], 'function': call['function'], 'role': call['role'], 'promptChars': size(call['prompt']),
            'turns': len({step['turn'] for step in call['steps']}),
            'steps': [{'index': i, 'turn': s['turn'], 'tool': s['tool'], 'status': s['status'],
                       'chars': size(s.get('code', s['arguments'])) + size(s['output'])} for i, s in enumerate(call['steps'])],
            'errors': sum(s['status'] not in ('ok', 'completed') for s in call['steps']), 'result': call['result']['status'],
            'children': []}


def select(args):
    from ..common.custody import CustodyError, ProtectedWorlds
    from ..rl.privileged_context import applicable, case_facts, fact_request
    from .hindsight import hindsight_declaration, node_classes
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    custody = ProtectedWorlds.load()
    cases = {case['id']: case for path in args.cases for case in _jsonl(path)}
    best = {}
    for path in args.rollouts:
        for rollout in _jsonl(path):
            if ((rollout.get('reward') or {}).get('quality') or 0) < 1 or rollout.get('split') != 'train':
                continue
            calls = {turn['invocation_id'] for turn in rollout['turns']}
            base = rollout['task_id'].removesuffix(':hinted')
            case = cases.get(base)
            if case is None:
                continue
            tokens = sum(len(turn['sampling']['samples']) for turn in rollout['turns'])
            last_prompt = sum(len(s.get('token_ids') or []) for s in rollout['turns'][-1]['sampling']['prompt'])
            key = base
            if key not in best or last_prompt + tokens < best[key][0]:
                best[key] = (last_prompt + tokens, path, rollout, case, len(calls))
    declaration = hindsight_declaration({'child_hint_min_tokens': args.child_hint_min_tokens})
    selected, refused = [], []
    for key, (size, path, rollout, case, calls) in sorted(best.items(), key=lambda item: item[1][0])[:args.limit]:
        try:
            _, facts = case_facts(case, custody=custody)
        except CustodyError as error:
            refused.append({'task_id': key, 'reason': str(error)})
            continue
        if calls != 1:
            refused.append({'task_id': key, 'reason': f'{calls} calls: multi-call replay not wired in this pilot'})
            continue
        call = call_value(rollout, tokenizer)
        consumer = {'name': call['call'], 'role': 'root', 'text': call['prompt'][:4000]}
        visible = [fact for fact in facts if fact.kind != 'teacher_trajectory'
                   and applicable(fact, {'name': call['call'], 'role': 'root', 'text': call['prompt']})]
        action_tokens = sum(len(turn['sampling']['samples']) for turn in rollout['turns'])
        selected.append({'task_id': key, 'source': str(path), 'rollout_sha': rollout.get('seed'), 'size': size,
                         'action_tokens': action_tokens,
                         'classes': node_classes({call['call']: {'role': 'root', 'action_tokens': action_tokens}},
                                                 declaration),
                         'inputs': {'consumer': consumer, 'execution': call, 'overview': overview(call),
                                    'facts': [fact_request(fact) for fact in visible]}})
        _write(Path(args.out) / 'tau' / f'{len(selected) - 1:03d}.json', {**selected[-1], 'rollout': rollout})
    _write(Path(args.out) / 'selection.json', {'selected': [{k: v for k, v in item.items() if k != 'inputs'}
                                                            for item in selected], 'refused': refused,
                                               'protected_sha256': custody.digest,
                                               'child_hint_min_tokens': args.child_hint_min_tokens})
    print(json.dumps({'selected': len(selected), 'refused': len(refused)}))


# ---------------------------------------------------------------- writer episodes

def write(args):
    """``writeHint`` K times per tau* against a native server profile (``neuraleseServer``: the configured Neuralese
    writer materializes the returned text as a native state), ``--parallel`` calls at a time."""
    from concurrent.futures import ThreadPoolExecutor
    out = Path(args.out)
    node = os.environ.get('NATLANG_NODE', 'node')
    config = out / 'natlang-config'
    _write(config / 'config.json', {'defaultProfile': 'pilot', 'profiles': {'pilot': {
        'endpoint': args.server, 'neuraleseServer': True, 'model': 'native'}}})
    environment = {key: value for key, value in os.environ.items() if key != 'NATLANG_SERVER'}
    environment.update(NATLANG_CONFIG_HOME=str(config), NATLANG_PROFILE='pilot', NATLANG_PROGRESS='off')
    jobs = []
    for path in sorted((out / 'tau').glob('*.json')):
        for sample in range(args.k):
            if not (out / 'hints' / f'{path.stem}-{sample}.json').exists():
                jobs.append((path, sample))

    def run(job):
        path, sample = job
        tau = json.loads(path.read_text())
        inputs = out / 'hints' / f'{path.stem}-{sample}.inputs.json'
        _write(inputs, tau['inputs'])
        live = out / 'hints' / f'{path.stem}-{sample}.live.jsonl'
        started = time.time()
        try:
            process = subprocess.run([node, str(REPO / 'ts-host/bin/natlang.mjs'), 'call', str(WRITER), '--inputs',
                                      str(inputs)], capture_output=True, text=True, timeout=args.timeout,
                                     env={**environment, 'NATLANG_LIVE_FILE': str(live)})
            record = {'returncode': process.returncode, 'stdout': process.stdout[-20000:],
                      'stderr': process.stderr[-4000:]}
        except subprocess.TimeoutExpired:
            record = {'returncode': None, 'stdout': '', 'stderr': f'timeout after {args.timeout}s'}
        _write(out / 'hints' / f'{path.stem}-{sample}.json', {**record, 'seconds': time.time() - started})
        print(path.stem, sample, record['returncode'], round(time.time() - started), flush=True)
    with ThreadPoolExecutor(args.parallel) as pool:
        list(pool.map(run, jobs))


def _state_text(server, state_id):
    import urllib.request
    from ..serve.store import decode_native_state
    with urllib.request.urlopen(f"{server.rstrip('/')}/v1/neuralese/native-states/{state_id}", timeout=120) as response:
        return decode_native_state(response.read())[0].text


def hint_text(record, server):
    """The hint's text: the written native state's own text (fetched from the writer's server) when the result names
    one, else the plain stdout."""
    import re
    text = record['stdout'].strip()
    found = re.search(r'nzs1_[a-z2-7]{52}', text)
    if found:
        return _state_text(server, found.group(0))
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return text or None
    return value if isinstance(value, str) else None


# ---------------------------------------------------------------- scoring

def _engine(args):
    from ..serve import load_native_engine
    return load_native_engine(args.checkpoint, device=args.device, dtype=torch.bfloat16)


def _tau_replay(engine, rollout):
    """tau*'s compiled replay (root call): invocations and actions with own-token ``local_query``."""
    from .episode_graph import whole_episode_turns
    from .native_replay import compile_replay
    turns = whole_episode_turns(rollout)
    identity = dict(rollout['turns'][0]['sampling']['policy'])
    replay = compile_replay(turns, identity={key: identity[key] for key in
                                             ('checkpoint_sha256', 'tokenizer_fingerprint', 'embedding_weight_sha256',
                                              'architecture')},
                            roles=engine.role_tokens, fetch_block=lambda key: None)
    offsets = dict(zip((node.name for node in replay.plan.invocations), replay.plan.offsets))
    actions = [{'turn': a['turn'], 'local_query': a['query'] - offsets[a['turn']], 'token_id': a['token_id']}
               for a in replay.actions]
    return list(replay.plan.invocations), actions


def _text_hinted(invocations, actions, texts, rendering):
    """Plan with a text guidance turn (``rendering.turn``) inserted in every invocation, and requeried actions."""
    from ..model.native_graph_plan import compile_graph
    from .privileged_scoring import _Edit, _rebase_plan
    plan = compile_graph(invocations)
    offsets = dict(zip((node.name for node in plan.invocations), plan.offsets))
    edits = {}
    if texts:
        tokens = tuple(rendering.turn(texts))
        for node in invocations:
            if not node.cache_prefix:
                edits[node.name] = _Edit(rendering.insertion_point(node.token_ids)[0], 0, tokens)
    rebased, requery = _rebase_plan(plan, edits)
    return rebased, [requery({**a, 'query': offsets[a['turn']] + a['local_query']}) for a in actions]


def _nll(model, plan, actions, channel=None):
    from ..model.import_channel import import_channel
    from .hindsight import reconstruction_nll
    from contextlib import nullcontext
    with torch.no_grad(), (import_channel(channel) if channel is not None else nullcontext()):
        result = model(plan, retain_states=False)
        return float(reconstruction_nll(model, result['h_final'][0], actions)), result.get('import_channel')


def score(args):
    from ..model.import_channel import ImportChannel, measure_scales
    from .hindsight import Hint, hinted_graph
    out = Path(args.out)
    engine = _engine(args)
    model, roles, rendering = engine.shared_model, engine.role_tokens, engine._privileged_rendering()
    tokenize = lambda text: engine.tokenizer(text, add_special_tokens=False)['input_ids']
    scales = None
    rows = []
    for path in sorted((out / 'tau').glob('*.json')):
        done = out / 'scores' / f'{path.stem}.json'
        hints_now = sorted((out / 'hints').glob(f'{path.stem}-[0-9].json'))
        if done.exists() and json.loads(done.read_text()).get('hint_files') == [h.name for h in hints_now]:
            rows.append(json.loads(done.read_text()))
            continue
        tau = json.loads(path.read_text())
        invocations, actions = _tau_replay(engine, tau['rollout'])
        # tau* is the root call; each of its model turns is a graph invocation and gets the call's hint.
        turns = [n.name for n in invocations if not n.cache_prefix]
        T = len(actions)
        if scales is None:
            from ..model.native_graph_plan import compile_graph
            scales = measure_scales(model, compile_graph(invocations))
            _write(out / 'channel-scales.json', [[layer, name, value] for (layer, name), value in scales.items()])
        row = {'tau': path.stem, 'task_id': tau['task_id'], 'T': T, 'turns': len(turns)}
        plan, acts = _text_hinted(invocations, actions, [], rendering)
        row['nll_none'] = _nll(model, plan, acts)[0] / T
        facts = [fact['text'] for fact in tau['inputs']['facts']]
        if facts:
            plan, acts = _text_hinted(invocations, actions, facts, rendering)
            row['nll_facts_text'] = _nll(model, plan, acts)[0] / T
        verbatim = json.dumps(tau['inputs']['execution']['steps'], ensure_ascii=False)[:6000]
        plan, acts = _text_hinted(invocations, actions, ['A verified successful trajectory:\n' + verbatim], rendering)
        row['nll_verbatim_text'] = _nll(model, plan, acts)[0] / T
        row['hints'] = []
        for hint_path in sorted((out / 'hints').glob(f'{path.stem}-[0-9].json')):
            text = hint_text(json.loads(hint_path.read_text()), args.server)
            if not text:
                row['hints'].append({'hint': hint_path.stem, 'failed': True})
                continue
            ids = tuple(tokenize(text))
            item = {'hint': hint_path.stem, 'tokens': len(ids), 'text': text[:2000]}
            plan, member_actions, names = hinted_graph(invocations, actions, [{t: Hint(t, ids) for t in turns}],
                                                       rendering, roles)
            item['nll_import'] = _nll(model, plan, member_actions[0])[0] / T
            channel = ImportChannel(tuple(names[0]), args.sigma, 1., scales, seed=1)
            value, report = _nll(model, plan, member_actions[0], channel)
            item['nll_import_channel'] = value / T
            item['kl'] = float(report['kl'].sum())
            plan, acts = _text_hinted(invocations, actions, [text], rendering)
            item['nll_text'] = _nll(model, plan, acts)[0] / T
            row['hints'].append(item)
        row['hint_files'] = [h.name for h in hints_now]
        _write(done, row)
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != 'hints'}), flush=True)
    _write(out / 'scores.json', rows)


# ---------------------------------------------------------------- phase 1 (pathwise, fixed hints)

def train(args):
    from .hindsight import Hint, hinted_graph, reconstruction_nll
    out = Path(args.out)
    engine = _engine(args)
    model, roles, rendering = engine.shared_model, engine.role_tokens, engine._privileged_rendering()
    tokenize = lambda text: engine.tokenizer(text, add_special_tokens=False)['input_ids']
    items = []
    for path in sorted((out / 'tau').glob('*.json')):
        tau = json.loads(path.read_text())
        invocations, actions = _tau_replay(engine, tau['rollout'])
        for hint_path in sorted((out / 'hints').glob(f'{path.stem}-[0-9].json')):
            text = hint_text(json.loads(hint_path.read_text()), args.server)
            if text:
                items.append((path.stem, invocations, actions, tuple(tokenize(text))))
    stems = sorted({item[0] for item in items})
    dev = set(stems[::4])
    train_items = [item for item in items if item[0] not in dev]
    dev_items = [item for item in items if item[0] in dev]
    parameters = list(model.adapter.parameters())
    for parameter in parameters:
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.)

    def loss_of(item, grad):
        stem, invocations, actions, ids = item
        hints = {n.name: Hint(n.name, ids) for n in invocations if not n.cache_prefix}
        plan, member_actions, _ = hinted_graph(invocations, actions, [hints], rendering, roles)
        with torch.set_grad_enabled(grad):
            result = model(plan, retain_states=False)
            return reconstruction_nll(model, result['h_final'][0], member_actions[0]) / len(actions)

    history = []
    def evaluate(step):
        with torch.no_grad():
            value = sum(float(loss_of(item, False)) for item in dev_items) / max(1, len(dev_items))
        history.append({'step': step, 'dev_nll_per_token': value})
        print(json.dumps(history[-1]), flush=True)
    evaluate(0)
    generator = torch.Generator().manual_seed(0)
    for step in range(1, args.steps + 1):
        item = train_items[int(torch.randint(len(train_items), (1,), generator=generator))]
        loss = loss_of(item, True)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        history.append({'step': step, 'train_nll_per_token': float(loss)})
        if step % args.eval_every == 0:
            evaluate(step)
    _write(out / 'train.json', {'trainable': 'producer adapter (phase-1 writer parameters); backbone frozen',
                                'lr': args.lr, 'steps': args.steps, 'train': len(train_items), 'dev': len(dev_items),
                                'history': history})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('stage', choices=('select', 'write', 'score', 'train'))
    parser.add_argument('--out', required=True)
    parser.add_argument('--rollouts', nargs='*', default=())
    parser.add_argument('--cases', nargs='*', default=())
    parser.add_argument('--tokenizer')
    parser.add_argument('--limit', type=int, default=24)
    parser.add_argument('--child-hint-min-tokens', type=int, default=256)
    parser.add_argument('--server', default='http://127.0.0.1:18431')
    parser.add_argument('--k', type=int, default=2)
    parser.add_argument('--timeout', type=int, default=1800)
    parser.add_argument('--parallel', type=int, default=4)
    parser.add_argument('--checkpoint')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--sigma', type=float, default=.5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--eval-every', type=int, default=50)
    args = parser.parse_args(argv)
    {'select': select, 'write': write, 'score': score, 'train': train}[args.stage](args)


if __name__ == '__main__':
    main()
