"""`python -m natlang_neuralese.tune.learned.lane`: glue of the LFM3 tuning and learned-optimizer lane
(`training/neuralese/learned-optimizer/run-lfm3-lane.sh`).

- ``tune-result --program DIR --out FILE``: one program's tune outcome after ``natlang tune DIR --rounds`` and
  ``natlang tune DIR --evaluate development``: the selected round, its development success, the measured untuned and
  active arms, and the plan of the round-0 supervised fit (its rows are the program's admitted training rows:
  reused verified data and oracle-verified privileged self-teaching) that bank-init and the corpus refit.
- ``specs --op DIR --checkpoint CK --templates DIR``: the operation's bank-init and corpus specs and its train/meta
  templates from the ``lfm3-*-spec.json`` templates and the recorded tune results. Programs are split by FAMILY into the optimizer's
  corpus side and its held evaluation side (``optimizer-split.json``): held families never supply bank sources,
  corpus targets or meta-examples, so their later ``natlang tune --learned`` lift measures programs that did not
  train the optimizer.

- ``outer-next --op DIR``: the next outer-training action (train to the next bank publication, adopt a publication
  and refit labels against it, or done) with its spec; the lane runs each as its own ledger unit (``outer_next``).

Nothing here trains or evaluates; every number it records was measured by the tuner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

RESULT_SCHEMA = 'natlang.lopt.lane-tune-result/1'


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=1, sort_keys=True) + '\n')
    temporary.replace(path)


def description(root):
    """The program's purpose: the first sentence of its root function's instruction text."""
    text = (Path(root) / 'main.nl').read_text() if (Path(root) / 'main.nl').exists() else ''
    body = text.split('---', 2)[-1].strip() if text.startswith('---') else text.strip()
    sentence = re.split(r'(?<=[.!?])\s', body, maxsplit=1)[0] if body else ''
    return sentence or Path(root).name


def _arm_success(arm):
    """Successes as `ts-host/src/tune/evaluate.ts` rootSuccess counts them (oracle roots: correct; others: completed)."""
    roots = arm.get('roots') or []
    success = sum(1 for root in roots if (root.get('correct') is True if root.get('oracle') else bool(root.get('ok'))))
    return {'success': success, 'roots': len(roots)}


def tune_result(program, out, family=None):
    root = Path(program).resolve()
    tune = root / '.natlang' / 'tune'
    journals = sorted(tune.glob('rounds/*/journal.json'), key=lambda path: path.stat().st_mtime)
    if not journals:
        raise SystemExit(f'{root}: no rounds journal; run natlang tune {root} --rounds first')
    journal = json.loads(journals[-1].read_text())
    rounds = journal['rounds']
    if not journal.get('best') or not rounds or rounds[0].get('status') != 'evaluated':
        raise SystemExit(f'{root}: the rounds journal has no evaluated supervised round ({journal.get("stopped")})')
    plan = Path(rounds[0]['build']).parent / 'plan.json'
    if not plan.exists():
        raise SystemExit(f'{root}: round 0 build {rounds[0]["build"]} has no plan.json')
    arms = {}
    for path in sorted((tune / 'evaluations' / 'development').glob('*.json')):
        arm = json.loads(path.read_text())
        arms[arm.get('arm', path.stem)] = {**_arm_success(arm), 'release': arm.get('release')}
    rows = sum(1 for line in (plan.parent / json.loads(plan.read_text())['rows']).read_text().splitlines() if line.strip())
    result = {'schema': RESULT_SCHEMA, 'program': root.name, 'root': str(root), 'family': family or re.sub(r'-v\d+$', '', root.name),
              'description': description(root), 'sft_plan': str(plan), 'sft_rows': rows, 'journal': str(journals[-1]),
              'best': journal['best'], 'stopped': journal.get('stopped'),
              'rounds': [{key: item.get(key) for key in ('index', 'kind', 'status', 'development', 'improved', 'reason')}
                         for item in rounds],
              'development_arms': arms,
              'note': 'measured by natlang tune on development roots; confirmation untouched; no production exposure'}
    _write(out, result)
    print(json.dumps({'command': 'tune-result', 'program': root.name, 'best': journal['best']['success'],
                      'roots': next((item['development']['roots'] for item in rounds if item.get('development')), None)}))
    return result


def _held(family, fraction):
    return int(hashlib.sha256(f'natlang.lopt.family-split/1:{family}'.encode()).hexdigest()[:8], 16) / 2 ** 32 < fraction


def _fill(value, mapping):
    if isinstance(value, str):
        if value in mapping and not isinstance(mapping[value], str):
            return mapping[value]
        for key, item in mapping.items():
            if isinstance(item, str):
                value = value.replace(key, item)
        return value
    if isinstance(value, list):
        return [_fill(item, mapping) for item in value]
    if isinstance(value, dict):
        return {key: _fill(item, mapping) for key, item in value.items()}
    return value


def _merge(base, extra):
    out = dict(base)
    for key, value in extra.items():
        out[key] = _merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


def specs(op, checkpoint, templates, *, held_fraction=.25, min_programs=2, extra=None):
    op, templates = Path(op).resolve(), Path(templates)
    results = [json.loads(path.read_text()) for path in sorted((op / 'tunes').glob('*.json'))]
    results = [item for item in results if item.get('schema') == RESULT_SCHEMA and item.get('sft_rows', 0) > 0]
    families = sorted({item['family'] for item in results})
    held_families = {family for family in families if _held(family, held_fraction)}
    if held_families == set(families):  # the corpus side never empties: keep the family with the lowest hash
        held_families.discard(min(families, key=lambda family: hashlib.sha256(family.encode()).hexdigest()))
    corpus = [item for item in results if item['family'] not in held_families]
    held = [item for item in results if item['family'] in held_families]
    if len(corpus) < min_programs:
        raise SystemExit(f'{len(corpus)} tuned corpus-side program(s) with admitted rows; bank-init needs at least {min_programs}')
    _write(op / 'optimizer-split.json', {'schema': 'natlang.lopt.program-split/1', 'held_fraction': held_fraction,
                                         'corpus': [item['program'] for item in corpus], 'held': [item['program'] for item in held],
                                         'held_families': sorted(held_families)})
    mapping = {'{op}': str(op), '{checkpoint}': str(Path(checkpoint).resolve()),
               '{sources}': [{'name': item['program'], 'kind': 'program', 'plan': item['sft_plan'], 'description': item['description']}
                             for item in corpus],
               '{programs}': [{'name': item['program'], 'plan': item['sft_plan'], 'root': item['root']} for item in corpus],
               '{meta_programs}': '{meta_programs}'}
    written = {}
    # bank-init and the first corpus run as written; train and meta are templates the outer cycles bind (outer-next).
    for name, target in (('bank-init', 'bank-init-spec.json'), ('corpus', 'corpus-spec.json'),
                         ('train', 'train-template.json'), ('meta', 'meta-template.json')):
        spec = _fill(json.loads((templates / f'lfm3-{name}-spec.json').read_text()), mapping)
        for key in ('*', name):
            spec = _merge(spec, (extra or {}).get(key, {}))
        path = op / target
        if path.exists() and json.loads(path.read_text()) != spec:
            raise SystemExit(f'{path} binds other programs or settings; use a new operation id for a changed lane')
        _write(path, spec)
        written[name] = str(path)
    print(json.dumps({'command': 'specs', 'corpus': len(corpus), 'held': len(held), 'specs': written}))
    return written


def _cycle(op, index):
    """Cycle 0 trains on the bank-init bank and the first corpus; cycle k>0 on the k-th adopted publication and its refit."""
    if index == 0:
        corpus = json.loads((op / 'corpus-spec.json').read_text())
        return {'bank': corpus['bank'], 'corpus': corpus['work'], 'targets': corpus['out']}
    return {'bank': str(op / 'cycles' / str(index) / 'bank.json'), 'corpus': str(op / 'cycles' / str(index) / 'corpus'),
            'targets': str(op / 'cycles' / str(index) / 'corpus' / 'targets.jsonl')}


def outer_next(op):
    """The next outer-training action of the operation, with its spec written:

    - ``train``: ``cycles/<k>/train-<step>.json``, the updates up to the next publication boundary (or the total);
    - ``adopt``: after a publication, ``cycles/<k+1>/adopt.json`` (the trainer transition, `adopt-bank`) and
      ``cycles/<k+1>/corpus-spec.json`` (the label refit against the published bank in a fresh work directory);
      the lane marks ``cycles/<k+1>/complete`` after both succeed;
    - ``done``: ``train-spec.json`` (bound to the final bank and targets, for the outer meta-RL round) and
      ``meta-spec.json`` (meta-examples over the final corpus plans) are written."""
    op = Path(op).resolve()
    template = json.loads((op / 'train-template.json').read_text())
    total, every = int(template['steps']), int(template.get('recipe', {}).get('publish_every', 50))
    index = 0
    while (op / 'cycles' / str(index + 1) / 'complete').exists():
        index += 1
    current = _cycle(op, index)
    state_path = Path(template['state'])
    state = json.loads(state_path.read_text()) if state_path.exists() else {'step': 0, 'labels_stale': False}
    step = int(state['step'])
    common = {key: template[key] for key in template if key not in ('steps', 'bank', 'targets')}
    if state.get('labels_stale'):
        directory = op / 'cycles' / str(index + 1)
        nxt = _cycle(op, index + 1)
        adopt = {key: template[key] for key in ('checkpoint', 'store', 'device', 'dtype', 'state', 'recipe') if key in template}
        _write(directory / 'adopt.json', {**adopt, 'out': nxt['bank']})
        corpus = json.loads((op / 'corpus-spec.json').read_text())
        _write(directory / 'corpus-spec.json', {**corpus, 'bank': nxt['bank'], 'work': nxt['corpus'], 'out': nxt['targets']})
        action = {'action': 'adopt', 'cycle': index + 1, 'adopt': str(directory / 'adopt.json'),
                  'corpus': str(directory / 'corpus-spec.json'), 'complete': str(directory / 'complete')}
    elif step < total:
        steps = min(total - step, every - step % every)
        path = op / 'cycles' / str(index) / f'train-{step}.json'
        _write(path, {**common, 'bank': current['bank'], 'targets': current['targets'], 'steps': steps})
        action = {'action': 'train', 'cycle': index, 'step': step, 'steps': steps, 'spec': str(path)}
    else:
        _write(op / 'train-spec.json', {**common, 'bank': current['bank'], 'targets': current['targets'], 'steps': 0})
        corpus = Path(current['corpus'])
        programs = json.loads((op / 'corpus-spec.json').read_text())['programs']
        meta = _fill(json.loads((op / 'meta-template.json').read_text()), {
            '{meta_programs}': [{'plan': str(corpus / program['name'] / 'plan.json')} for program in programs]})
        meta['learned'] = {**meta['learned'], 'bank': current['bank'], 'state': template['state']}
        _write(op / 'meta-spec.json', meta)
        action = {'action': 'done', 'cycle': index, 'step': step, 'train': str(op / 'train-spec.json'), 'meta': str(op / 'meta-spec.json')}
    print(json.dumps(action))
    return action


def summary(op):
    """What the operation actually produced: stage statuses, tune results, the optimizer split, outer training
    (steps, publications, adoptions), the meta-RL round and meta-example counts. No capability claim is derived."""
    op = Path(op).resolve()
    status = {path.name[:-len('.status')]: int(path.read_text().strip() or -1) for path in sorted(op.glob('*.status'))}
    tunes = {}
    for path in sorted((op / 'tunes').glob('*.json')):
        item = json.loads(path.read_text())
        tunes[item['program']] = {'sft_rows': item['sft_rows'], 'best': item['best'], 'development_arms': item['development_arms'],
                                  'rounds': [(r['kind'], r['status'], r.get('development')) for r in item['rounds']]}
    failed = [line for line in (op / 'tunes' / 'failed.txt').read_text().splitlines() if line] if (op / 'tunes' / 'failed.txt').exists() else []
    template = json.loads((op / 'train-template.json').read_text()) if (op / 'train-template.json').exists() else {}
    state_path = Path(template.get('state', op / 'optimizer' / 'state.json'))
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    records = op / 'outer-rl' / 'records.jsonl'
    meta = op / 'meta' / 'tune.examples.jsonl'
    count = lambda path: sum(1 for line in path.read_text().splitlines() if line.strip()) if path.exists() else 0
    out = {'schema': 'natlang.lopt.lane-summary/1', 'operation': json.loads((op / 'operation.json').read_text()),
           'stages': status, 'tunes': tunes, 'failed_tunes': failed,
           'optimizer_split': json.loads((op / 'optimizer-split.json').read_text()) if (op / 'optimizer-split.json').exists() else None,
           'outer': {'step': state.get('step'), 'bank': state.get('bank'), 'published': len(state.get('published', [])),
                     'adoptions': [{key: item[key] for key in ('step', 'from', 'to', 'bank_optimizer')} for item in state.get('adoptions', [])],
                     'labels_stale': state.get('labels_stale'), 'targets_seen': sum(state.get('exposure', {}).values())},
           'outer_rl_records': count(records), 'meta_examples': count(meta),
           'note': 'saved artifacts and passing stages are not capability; lift needs development and confirmation outcomes'}
    print(json.dumps(out, indent=1, sort_keys=True))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest='command', required=True)
    one = commands.add_parser('tune-result')
    one.add_argument('--program', required=True)
    one.add_argument('--out', required=True)
    one.add_argument('--family')
    many = commands.add_parser('specs')
    many.add_argument('--op', required=True)
    many.add_argument('--checkpoint', required=True)
    many.add_argument('--templates', required=True)
    many.add_argument('--held-fraction', type=float, default=.25)
    many.add_argument('--min-programs', type=int, default=2)
    many.add_argument('--extra', help='JSON {"*"|spec name: settings merged into the specs} (dry runs)')
    step = commands.add_parser('outer-next')
    step.add_argument('--op', required=True)
    report = commands.add_parser('summary')
    report.add_argument('--op', required=True)
    args = parser.parse_args(argv)
    if args.command == 'tune-result':
        tune_result(args.program, args.out, args.family)
    elif args.command == 'outer-next':
        outer_next(args.op)
    elif args.command == 'summary':
        summary(args.op)
    else:
        specs(args.op, args.checkpoint, args.templates, held_fraction=args.held_fraction, min_programs=args.min_programs,
              extra=json.loads(args.extra) if args.extra else None)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
