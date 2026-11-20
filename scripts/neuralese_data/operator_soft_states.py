#!/usr/bin/env python3
"""The learned operators' own soft prompt states, trained with the CSPD recipe on verified operator rows.

Neuralese spans, soft instruction and skill states, memetic guidance states and operator states are one family: native
states primed from text in their real rendered call, then shaped by training (DECISIONS 2026-11-20). This script makes
the operators members of it through the one mechanism every member uses, `natlang_neuralese.tune` soft states:

* `turns`: each compacted source of an operator case (`operator_rows`: the runtime's own site rendering, the held
  closure, the case seed) becomes one `natlang.tune-turn/v1` row of the binding `prompt:view@<operator>`: the operator
  call's messages, its prompt text (view's body, the system message) as the binding's instruction text, the seed as
  the reply. Only verified seeds are positive rows (`gold`: dataset spans, `checked`: verified against the case's
  answer; evidence `rootOracle`); `mechanical` seeds bear the answer by construction and are admitted only with
  `--admit-mechanical` at the weak weight (`enclosingSuccess`).
* `plan`: a `tune.build` plan: soft states on, every adapter rank 0 (CSPD's frozen backbone), prefix-robust rows at
  the recipe default (`soft_prefix_share` 0.5; `--prefix-share 0` opts out).
* `python -m natlang_neuralese.tune.build --plan PLAN --out OUT` then primes each operator's state from view's body at
  its position in a real operator call (`tune.soft.initialize_soft_states`, `native_init_body`) and trains it. The
  runtime serves `OUT` with `NATLANG_NEURALESE_PROMPT_STATES=OUT` (ts-host model/prompt-states.ts): an operator call
  imports its own state where view's body stood.

    operator_soft_states.py turns --out TURNS.jsonl [--cases ...] [--per-source N] [--operators a,b] [--admit-mechanical]
    operator_soft_states.py plan --turns TURNS.jsonl --checkpoint CKPT.pt --out PLAN.json [--steps N] [--batch-rows B]
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import operator_rows  # noqa: E402
from operator_rows import CASES_ROOT, HELD_SETS, LFM3_BASE, HeldClosure, RuntimeSites, case_files, site_request  # noqa: E402
from natlang_neuralese.data.operator_graph import OperatorCaseError, TokenRenderer  # noqa: E402

TURN_SCHEMA = 'natlang.tune-turn/v1'
ACTION_POLICY = 'natlang.tune-action-outcome/2'
VERIFIED = ('gold', 'checked')
# The runtime's operator sites: compact-value and compact-params are view at the runtime's sites (one summarizer
# family); each has its own state of view's body (system-prompts.ts scoped entries `view@<operator>`).
SITE_OPERATORS = ('compact-value', 'compact-params')
CASE_OPERATORS = ('compact-value', 'compact-params', 'extract')


def binding(operator):
    return f'prompt:view@{operator}'


def evidence(provenance):
    verified = provenance in VERIFIED
    return {'rootOracle': verified, 'enclosingSuccess': True, 'callOutcome': 'done', 'actionPolicy': ACTION_POLICY,
            'actionDisposition': 'admissible', 'failedHistory': False, 'actionEvents': [],
            'operatorSeed': provenance}


def operator_turn(site, seed_text, provenance, *, case, index):
    """One operator call of the runtime (its site) with the seed as its reply, bound to the operator's own state."""
    messages = site['messages']
    if not messages or messages[0].get('role') != 'system' or not isinstance(messages[0].get('content'), str):
        raise OperatorCaseError('site-without-system-prompt')
    operator = site['operator']
    return {'schema': TURN_SCHEMA, 'root': case, 'component': f'operator:{operator}', 'partition': 'train',
            'binding': binding(operator), 'call': f'source-{index}', 'turn': 1, 'tools': site['tools'] or [],
            'teacher': f'operator-seed:{provenance}', 'source': 'operator-case', 'messages': messages,
            'texts': {'instructions': messages[0]['content']}, 'reply': {'text': seed_text},
            'evidence': evidence(provenance)}


def turn_chars(messages, reply):
    return sum(len(m['content']) for m in messages if isinstance(m.get('content'), str)) + len(reply)


def turns(args):
    tokenizer, roles, _ = operator_rows.load_tokenizer(args.base)
    renderer = TokenRenderer(tokenizer, role_ids=roles)
    sites = RuntimeSites()
    held = HeldClosure.from_cases(args.held)
    # Case directories whose cases render at a runtime site (extract cases are string arguments: compact-params);
    # compact-history is the model's own note, captured in place, with no operator call to prime.
    operators = set(args.operators.split(',')) if args.operators else set(CASE_OPERATORS)
    counts = collections.Counter()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as stream:
        for path in case_files(args.cases, operators):
            name = path.parent.name
            kept = read = 0
            with open(path, 'rb') as handle:
                for line in handle:
                    # A source whose cases are rarely compacted or verified is not read to its end.
                    if args.per_source and (kept >= args.per_source or read >= 20 * args.per_source):
                        break
                    read += 1
                    record = json.loads(line)
                    if (record.get('split') in ('test', 'validation')) != (args.split == 'test'):
                        continue   # train turns from train cases; test turns (evaluate) from test cases only
                    if held.hit(record):
                        counts['held_dropped'] += 1
                        continue
                    case = hashlib.sha256(line.rstrip(b'\n')).hexdigest()[:20]
                    wrote = False
                    for index, source in enumerate(record['sources']):
                        try:
                            site = sites.site({**site_request(name, record, source, index, len(record['sources'])),
                                               'context_tokens': args.context_tokens})
                            if not site['compacted'] or site['operator'] not in SITE_OPERATORS:
                                counts['not_compacted'] += 1
                                continue
                            seed_ids, seed = renderer.seed(name, record, source)
                        except OperatorCaseError as error:
                            counts[f'reject:{str(error).split(":")[0]}'] += 1
                            continue
                        if seed['provenance'] not in VERIFIED and not args.admit_mechanical:
                            counts[f'unverified:{seed["provenance"]}'] += 1
                            continue
                        text = tokenizer.decode(seed_ids, skip_special_tokens=False)
                        if args.max_chars and turn_chars(site['messages'], text) > args.max_chars:
                            counts['over_max_chars'] += 1   # a row's activations scale with its length (fit memory)
                            continue
                        stream.write(json.dumps(operator_turn(site, text, seed['provenance'], case=case, index=index)) + '\n')
                        counts[f'{site["operator"]}:{seed["provenance"]}'] += 1
                        wrote = True
                    kept += wrote
    sites.close()
    print(json.dumps({'out': str(out), 'counts': dict(counts)}, sort_keys=True))
    return counts


def evaluate(args):
    """Held-out check of a fit on test-split operator turns, forward only on the frozen backbone: the token-weighted NLL
    of each verified reply under view's text, under the state primed from it (the fit's initial state) and under the
    trained state, paired per row."""
    import random
    import torch
    from natlang_neuralese.common.cuda_context import prepare_gpu
    from natlang_neuralese.data.native_graph import graph_loss
    from natlang_neuralese.data.native_graph_schema import parse_graph
    from natlang_neuralese.serve import load_native_engine
    from natlang_neuralese.tune.build import fit_store_context, import_archive
    from natlang_neuralese.tune.rows import turn_graph
    build = Path(args.build)
    result = json.loads((build / 'result.json').read_text()) if (build / 'result.json').exists() else None
    state = json.loads((build / 'state.json').read_text())
    init = json.loads((build / 'soft-init.json').read_text())
    prepare_gpu(args.device)
    config = {'device': args.device}
    with fit_store_context(config) as store:
        engine = load_native_engine(args.checkpoint, device=args.device, dtype=getattr(torch, args.dtype), store=store)
        import_archive(engine, [build / 'artifacts'], [build / 'states'])
        turns = [json.loads(line) for line in Path(args.turns).read_text().splitlines() if line.strip()]
        random.Random(0).shuffle(turns)
        turns = turns[:args.rows] if args.rows else turns
        lengths = [turn_chars(turn['messages'], turn['reply']['text']) for turn in turns]
        arms = {'text': None, 'primed': {key: record['state'] for key, record in init.items()},
                'trained': dict(state['soft_states'])}
        losses = {arm: [] for arm in arms}
        engine.shared_model.eval()
        for turn in turns:
            turn = {**turn, 'partition': 'train'}   # rendering only; never trained
            for arm, soft in arms.items():
                row = turn_graph(engine, turn, prompt_weight=0., soft=soft)
                example = parse_graph(row['graph'], tokenizer_fingerprint=engine.model_config['tokenizer_fingerprint'])
                blocks = {name: engine.store.get_native_state(spec['id']).block
                          for name, spec in (row['graph'].get('external_states') or {}).items()}
                identity = {name: {key: getattr(block, key) for key in
                                   ('checkpoint_sha256', 'tokenizer_fingerprint', 'embedding_weight_sha256', 'architecture')}
                            for name, block in blocks.items()}
                with torch.no_grad():
                    loss, _ = graph_loss(engine.shared_model, example, checkpoint_layers=False,
                                         **({'external_blocks': blocks, 'external_identity': identity} if blocks else {}))
                losses[arm].append(float(loss))
    report = {'build': str(build), 'turns': args.turns, 'rows': len(turns), 'step': state.get('step'),
              'complete': result is not None, 'long_chars': args.long_chars,
              **paired_report(losses, range(len(turns)))}
    # The fit's row cap (turns --max-chars) bounds what it trained on; report rows beyond it on their own so a state
    # fit on short inputs is not credited or blamed for long ones on an average.
    if args.long_chars:
        for name, keep in (('short', lambda n: n <= args.long_chars), ('long', lambda n: n > args.long_chars)):
            rows = [i for i, n in enumerate(lengths) if keep(n)]
            if rows:
                report[name] = {'rows': len(rows), **paired_report(losses, rows)}
    print(json.dumps(report, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps({**report, 'losses': losses, 'chars': lengths}, indent=1) + '\n')


def paired_report(losses, rows):
    """Mean NLL per arm and paired changes (text -> primed -> trained) over the given row indices."""
    import math
    rows = list(rows)

    def paired(a, b):
        diffs = [losses[b][i] - losses[a][i] for i in rows]
        mean = sum(diffs) / len(diffs)
        sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / max(len(diffs) - 1, 1))
        return {'mean_change': mean, 'se': sd / math.sqrt(len(diffs)), 'lower_rows': sum(d < 0 for d in diffs),
                'rows': len(diffs)}
    return {'mean_nll': {arm: sum(values[i] for i in rows) / len(rows) for arm, values in losses.items()},
            'primed_vs_text': paired('text', 'primed'), 'trained_vs_primed': paired('primed', 'trained'),
            'trained_vs_text': paired('text', 'trained')}


def manifest_for(bindings, program_hash):
    """A `natlang.adapters/v1` manifest whose bindings are the operators' prompt states (no scope nodes train: every
    rank is 0)."""
    return {'schema': 'natlang.adapters/v1', 'program': {'id': 'natlang-runtime-prompts', 'buildHash': program_hash},
            'base': {}, 'tree': [{'id': 'tree:', 'level': 'tree', 'path': '', 'parent': None}],
            'bindings': [{'id': item, 'component': 'operator:' + item.rsplit('@', 1)[-1], 'definitionId': item,
                          'contextId': None, 'origin': 'named', 'ancestry': ['tree:'],
                          'semanticHash': hashlib.sha256(item.encode()).hexdigest()} for item in sorted(bindings)],
            'artifacts': [], 'families': [], 'memberships': [], 'scales': {}, 'validatedReductions': [],
            'fallback': 'base', 'receipts': {}}


def plan(args):
    rows = Path(args.turns)
    data = rows.read_bytes()
    bindings = {json.loads(line)['binding'] for line in data.decode().splitlines() if line.strip()}
    if not bindings:
        raise SystemExit(f'{rows} holds no operator turns')
    recipe = {'ranks': {'tree': 0, 'folder': 0, 'lambda': 0}, 'steps': args.steps, 'batch_rows': args.batch_rows,
              'prompt_weight': 0., 'state_optimizer': {'name': 'adam', 'learning_rate': args.lr}}
    if args.prefix_share is not None:
        recipe['soft_prefix_share'] = args.prefix_share
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plan_json = {'manifest': manifest_for(bindings, hashlib.sha256(data).hexdigest()), 'rows': str(rows.resolve()),
                 'softStates': True, 'recipe': recipe, 'label': 'operator-soft-states',
                 'student': {'name': 'lfm3', 'checkpoint': args.checkpoint,
                             'engine': {'device': args.device, 'dtype': args.dtype,
                                        **({'store_scratch': args.store_scratch} if args.store_scratch else {})}}}
    out.write_text(json.dumps(plan_json, indent=1, sort_keys=True) + '\n')
    print(json.dumps({'plan': str(out), 'bindings': sorted(bindings)}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    t = sub.add_parser('turns')
    t.add_argument('--cases', nargs='+', default=[str(CASES_ROOT)])
    t.add_argument('--out', required=True)
    t.add_argument('--operators', help='comma list of case operator directories (default: every one)')
    t.add_argument('--held', nargs='+', default=[str(p) for p in HELD_SETS])
    t.add_argument('--base', default=LFM3_BASE)
    t.add_argument('--per-source', type=int, default=200, help='cases kept per source file (0: all)')
    t.add_argument('--context-tokens', type=int, default=131072)
    t.add_argument('--admit-mechanical', action='store_true',
                   help='admit mechanical seeds as weak (enclosing-success) rows; off by default: they are not verified')
    t.add_argument('--split', choices=('train', 'test'), default='train')
    t.add_argument('--max-chars', type=int, default=40000,
                   help='longest row (site and reply characters; ~10k tokens): bounds the fit\'s activation memory (0: no cap)')
    e = sub.add_parser('evaluate')
    e.add_argument('--build', required=True)
    e.add_argument('--turns', required=True, help='test-split turns (turns --split test)')
    e.add_argument('--checkpoint', required=True)
    e.add_argument('--rows', type=int, default=0, help='test turns evaluated (0: all)')
    e.add_argument('--long-chars', type=int, default=40000,
                   help='report rows above this many characters separately (the fit\'s row cap; 0: one pool)')
    e.add_argument('--device', default='cuda')
    e.add_argument('--dtype', default='bfloat16')
    e.add_argument('--out')
    p = sub.add_parser('plan')
    p.add_argument('--turns', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--batch-rows', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--prefix-share', type=float, default=None, help='CSPD prefix-robust share (default: the recipe 0.5; 0 opts out)')
    p.add_argument('--device', default='cuda')
    p.add_argument('--dtype', default='bfloat16')
    p.add_argument('--store-scratch', default=None)
    args = parser.parse_args(argv)
    {'turns': turns, 'plan': plan, 'evaluate': evaluate}[args.command](args)


if __name__ == '__main__':
    main()
