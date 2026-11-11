#!/usr/bin/env python3
"""Audit the scripted-template families for useless rows and queue the withdrawal (owner order 2026-11-10/11).

Appends cannot remove rows (``native_resident_ce.extension_policy`` rejects a smaller family map), so a withdrawal is a static
source correction. That is no longer a manual cut-over: the controller applies a queued plan itself at its next SFT window
boundary (``rl/source_withdrawal.py``, one change with the queued packages; see that module for the plan format, the guards and
the failure handling). This script is the producer of the template plan and a rehearsal tool:

- ``audit`` decides what is useless (owner clarification 2026-11-11: withdraw only what is truly useless, never a family
  blindly): per-row criteria over the raw rows of ``port-record-wrapper`` and ``native-communication-template``, written to
  ``withdrawal-plan.json`` (``natlang.source-withdrawal-plan/1``, checkpoint independent) with decoded samples of both sides
  and a visit cap for useful but over-represented rows; ``--queue CONTROLLER_OUT`` appends it to that controller's
  ``memetic/distill/withdrawal-queue.jsonl`` (any plan: ``queue_source_withdrawal.py``);
- ``rehearse`` runs the boundary's preparation on a scratch copy of a controller's config, state and queues (the live
  controller is not touched): the filtered leaves, correction proof, CE policy, packed geometry, successor focus and recipes
  it would commit;
- ``preflight`` is the trainer's CPU continuation preflight (``--continuation-preflight-only --device cpu``, about 48 GB)
  of the rehearsed window.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

from natlang_neuralese.common.hashing import sha256_file_hex
from natlang_neuralese.common.jsonio import write_json_durable
from natlang_neuralese.data.native_sources import NativeGraphSource
from natlang_neuralese.rl import source_withdrawal
from natlang_neuralese.rl.source_withdrawal import PLAN_SCHEMA, leaves

FAMILIES = ('port-record-wrapper', 'native-communication-template')
REASON = 'owner order 2026-11-10: excise scripted template data'


def ref(path):
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'sha256': sha256_file_hex(path)}


def stage_view(recipe, stage_id):
    from natlang_neuralese.train.recipe import effective_stage_parameters, load_recipe, resolve_stage_inputs
    recipe = load_recipe(recipe)
    stage = next(item for item in recipe['stages'] if item['id'] == stage_id)
    return stage, resolve_stage_inputs(recipe, stage, {}), effective_stage_parameters(stage, [], recipe)


def effective_recipe(controller_out, config, override=None):
    """The recipe the lineage's next window starts from: the admitted successor's settled form, else the controller's
    (``override``: an explicit recipe, for a package against an older boundary)."""
    state = json.loads((Path(controller_out) / 'state.json').read_text())
    if override:
        return str(Path(override).resolve(strict=True)), state
    recipe = ((state.get('interleave') or {}).get('memetic_distillation') or {}).get('recipe')
    if recipe:
        if sha256_file_hex(recipe['path']) != recipe['sha256']:
            raise ValueError('admitted memetic successor recipe changed')
        return recipe['path'], state
    return config['production_recipe'], state


# ---- audit: which rows of the two families are useless ---------------------------------------------------------------
WRAPPER, TEMPLATE = FAMILIES
NEAR_DUPLICATE_JACCARD = 0.9   # wrapper: exact Jaccard of the 4-gram sets of the masked target (first 6000 tokens)
CONSTANT_PRODUCER_MIN_ROWS = 8  # template: a producer reply repeated verbatim by this many rows is not input-specific
TARGET_WEIGHT = 0.5             # reply/reasoning weight is 1.0; context 0.05 and feedback 0.25 are not targets
SHINGLE, SHINGLE_TOKENS, PERMUTATIONS = 4, 6000, 64


def _shingles(tokens):
    import numpy as np
    t = np.asarray(tokens[:SHINGLE_TOKENS], dtype=np.uint64)
    if len(t) < SHINGLE:
        t = np.concatenate([t, np.zeros(SHINGLE - len(t), dtype=np.uint64)])
    sh = t[:len(t) - SHINGLE + 1] * np.uint64(1000003)
    for j in range(1, SHINGLE):
        sh = sh * np.uint64(1000003) + t[j:len(t) - SHINGLE + 1 + j]
    return np.unique(sh)


def _minhash(shingles):
    import numpy as np
    rng = np.random.default_rng(1)
    a = rng.integers(1, (1 << 61) - 1, PERMUTATIONS, dtype=np.uint64)
    b = rng.integers(0, (1 << 61) - 1, PERMUTATIONS, dtype=np.uint64)
    return np.array([(shingles * a[i] + b[i]).min() for i in range(PERMUTATIONS)], dtype=np.uint64)


def _jaccard(x, y):
    import numpy as np
    inter = len(np.intersect1d(x, y, assume_unique=True))
    return inter / (len(x) + len(y) - inter)


def _target_runs(row):
    """Per node the tokens that carry reply/reasoning loss (weight >= TARGET_WEIGHT)."""
    return [[t for t, w in zip(n['token_ids'], n['target_weights']) if w >= TARGET_WEIGHT] for n in row['nodes']]


def audit(args):
    """Write ``withdrawal-plan.json`` (``natlang.source-withdrawal-plan/1``) for the named families from the raw rows of the
    live source; ``--queue CONTROLLER_OUT`` also queues it for that controller's next window boundary."""
    import hashlib
    import random
    import numpy as np
    from natlang_neuralese.common.jsonio import open_jsonl_text
    from natlang_neuralese.train.checkpoint_policy import CheckpointPolicy
    from natlang_neuralese.train.native_continuation import collect_sft_progress
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads((Path(args.controller_out) / 'config.json').read_text())
    stage_id = config['production_stage']
    base_recipe, state = effective_recipe(args.controller_out, config, args.recipe)
    checkpoint = Path(args.checkpoint or state['checkpoint']).resolve(strict=True)
    _, inputs, parameters = stage_view(base_recipe, stage_id)
    declared = parameters.get('sft_packed_geometry')
    identity0 = json.loads(Path((json.loads(declared) if isinstance(declared, str) else declared)['path']).read_text())['identity']
    source = NativeGraphSource(inputs['native_graphs']['path'], identity0['tokenizer'], required_split='train',
                               role_ids=identity0['roles'])
    policy = json.loads(Path(inputs['ce_supervision_policy']['path']).read_text())
    family_map = json.loads(Path(policy['objective']['family_balance']['map']['path']).read_text())
    family_of = {row['id']: row['family'] for row in family_map['rows']}
    frame = collect_sft_progress(CheckpointPolicy.load(checkpoint)['training'], borrow_frames=True)[stage_id]
    credits, quotas = frame.get('exposure_counts') or {}, frame.get('required_visits') or {}
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['base'])

    # pass 1: stream the rows of both families
    info = {}
    for leaf in leaves(source):
        wanted = {e['id'] for e in leaf.entries if family_of[e['id']] in FAMILIES}
        if not wanted:
            continue
        with open_jsonl_text(leaf.path) as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row['id'] not in wanted:
                    continue
                runs = _target_runs(row)
                flat = [t for run in runs for t in run]
                family = family_of[row['id']]
                item = {'family': family, 'n': len(flat), 'text': [tokenizer.decode(run[:300]) for run in runs],
                        'groups': (row.get('source_groups') or [])[:2]}
                if family == WRAPPER:
                    item['tokens'] = np.asarray(flat[:SHINGLE_TOKENS], dtype=np.uint32)
                    item['sig'] = _minhash(_shingles(flat))
                else:
                    if [n.get('name') for n in row['nodes']] != ['producer', 'consumer']:
                        raise ValueError('template edge with unexpected nodes: ' + row['id'])
                    item['producer'] = hashlib.sha256(json.dumps(runs[0]).encode()).hexdigest()
                    item['consumer'] = hashlib.sha256(json.dumps(runs[1]).encode()).hexdigest()
                info[row['id']] = item
        print(json.dumps({'event': 'audit_leaf', 'leaf': Path(leaf.path).name, 'rows': len(info)}), flush=True)

    earned = lambda i: int(credits.get(i, 0)) >= int(quotas.get(i, 1))  # noqa: E731
    decisions = {}  # id -> (criterion, kept representative)
    # wrapper: one row per task already (distinct programs); withdraw only repeats of an already kept row
    wrapper = [i for i, x in info.items() if x['family'] == WRAPPER]
    buckets = collections.defaultdict(list)
    for i in wrapper:
        sig = info[i]['sig']
        for band in range(PERMUTATIONS // 4):
            buckets[(band, bytes(sig[band * 4:(band + 1) * 4]))].append(i)
    parent = {i: i for i in wrapper}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for members in buckets.values():
        if 1 < len(members) <= 60:
            for a in range(len(members)):
                for b in range(a + 1, len(members)):
                    x, y = members[a], members[b]
                    if find(x) != find(y) and _jaccard(_shingles(info[x]['tokens']), _shingles(info[y]['tokens'])) >= NEAR_DUPLICATE_JACCARD:
                        parent[find(x)] = find(y)
    clusters = collections.defaultdict(list)
    for i in wrapper:
        clusters[find(i)].append(i)
    for members in clusters.values():
        if len(members) > 1:
            # keep the most complete row (an earned row first: it stays anyway), withdraw the rest
            members.sort(key=lambda i: (not earned(i), -info[i]['n'], i))
            for i in members[1:]:
                if not earned(i):
                    decisions[i] = ('wrapper near-duplicate target (exact Jaccard >= %.2f) of a kept row'
                                    % NEAR_DUPLICATE_JACCARD, members[0])
    # template: input-specific producers stay; constant and repeated producer replies go
    template = [i for i, x in info.items() if x['family'] == TEMPLATE]
    producer_rows = collections.Counter(info[i]['producer'] for i in template)
    seen = {}
    for i in sorted(template):
        x = info[i]
        if earned(i):
            continue
        if producer_rows[x['producer']] >= CONSTANT_PRODUCER_MIN_ROWS:
            decisions[i] = ('template producer reply repeated verbatim by >= %d edges (constant, not input-specific)'
                            % CONSTANT_PRODUCER_MIN_ROWS, None)
        elif (x['consumer'], x['producer']) in seen:
            decisions[i] = ('template edge repeating the producer reply of a kept edge into the same consumer answer',
                            seen[(x['consumer'], x['producer'])])
        else:
            seen[(x['consumer'], x['producer'])] = i
    # visit cap: a kept template edge re-trains its (shared) consumer answer on every visit
    cap = {i: {'required_visits': 1} for i in template
           if i not in decisions and int(quotas.get(i, 1)) > 1}
    kept = {f: [i for i, x in info.items() if x['family'] == f and i not in decisions] for f in FAMILIES}
    rs = random.Random(20261111)

    def example(i):
        x = info[i]
        return {'id': i, 'groups': x['groups'], 'target_tokens': x['n'],
                'decoded_targets_by_node': [t[:360] for t in x['text']],
                **({'criterion': decisions[i][0], 'kept_representative': decisions[i][1]} if i in decisions else {})}

    def withdrawn_of(f):
        return sorted(i for i in decisions if info[i]['family'] == f)
    plan = {'schema': PLAN_SCHEMA, 'reason': REASON, 'families': list(FAMILIES),
            'criteria': {WRAPPER: 'the 2026-11-03 review kept one row per wrapper task; every row here is its own task '
                                  '(distinct teacher program). Withdraw only a row whose masked target is a near-duplicate '
                                  '(exact Jaccard >= %.2f of 4-gram sets) of a kept row of another task; the most complete '
                                  'row of a cluster stays.' % NEAR_DUPLICATE_JACCARD,
                         TEMPLATE: 'keep edges whose producer reply is input-specific and distinct (they teach genuine '
                                   'Neuralese communication). Withdraw an edge whose producer reply is repeated verbatim '
                                   'by >= %d edges, or repeats the producer reply of a kept edge into the same consumer '
                                   'answer. Kept edges are capped at one visit (was two): the consumer answer shared by the '
                                   'edges of one fan-in is otherwise re-trained twice per edge.' % CONSTANT_PRODUCER_MIN_ROWS},
            # checkpoint independent: the boundary applies the plan against its own parent frame (an earned row is
            # kept, a cap only lowers a higher saved quota)
            'withdraw': {i: {'family': info[i]['family'], 'criterion': d[0], 'kept_representative': d[1],
                             'binding': source.bindings[i]} for i, d in sorted(decisions.items())},
            'visit_cap': {i: {'required_visits': item['required_visits']} for i, item in cap.items()},
            'evidence': [],
            'audit': {'parent_checkpoint': ref(checkpoint),
                      'source': {'path': inputs['native_graphs']['path'], 'sha256': inputs['native_graphs']['sha256']}},
            'counts': {f: {'rows': sum(1 for x in info.values() if x['family'] == f), 'withdrawn': len(withdrawn_of(f)),
                           'kept': len(kept[f]), 'earned_kept': sum(1 for i in kept[f] if earned(i)),
                           'visit_capped': sum(1 for i in cap if info[i]['family'] == f)} for f in FAMILIES},
            'withdrawn_by_criterion': dict(collections.Counter(d[0] for d in decisions.values())),
            'examples': {f: {'withdrawn': [example(i) for i in rs.sample(withdrawn_of(f), min(5, len(withdrawn_of(f))))],
                             'kept': [example(i) for i in rs.sample(kept[f], 5)]} for f in FAMILIES},
            'audit_sample': {f: [example(i) for i in rs.sample([i for i, x in info.items() if x['family'] == f], 60)]
                             for f in FAMILIES},
            'structure': {'wrapper_distinct_programs': len({i.split(':decision')[0] for i in wrapper}),
                          'wrapper_near_duplicate_clusters': sum(1 for m in clusters.values() if len(m) > 1),
                          'template_distinct_producer_replies': len(producer_rows),
                          'template_distinct_consumer_answers': len({info[i]['consumer'] for i in template}),
                          'template_edges_per_consumer_answer': dict(collections.Counter(collections.Counter(
                              info[i]['consumer'] for i in template).values()).most_common(8))}}
    write_json_durable(out / 'withdrawal-plan.json', plan)
    print(json.dumps({'event': 'audit', 'counts': plan['counts'], 'by_criterion': plan['withdrawn_by_criterion'],
                      'structure': plan['structure']}), flush=True)
    if args.queue:
        source_withdrawal.queue_plan(args.queue, out / 'withdrawal-plan.json')
    return plan


def rehearse(args):
    """The window boundary's withdrawal (and package append) on a SCRATCH copy of the controller's config, state and
    queues: nothing of the live controller is touched. ``--plan`` queues one more plan in the scratch copy first. The
    result is exactly what the controller commits at its next boundary: ``<out>/state.json`` (ledger, failures) and
    ``<out>/sft-window-NNNNNN/memetic-admission`` (record, proof, recipes). Run ``preflight`` on it next."""
    import shutil
    from natlang_neuralese.rl import interleave, memetic_admission
    out = Path(args.out).resolve()
    source = Path(args.controller_out).resolve(strict=True)
    config = json.loads((source / 'config.json').read_text())
    state = json.loads((source / 'state.json').read_text())
    config['out'] = str(out)
    queues = out / 'memetic' / 'distill'
    queues.mkdir(parents=True, exist_ok=True)
    if (source / 'memetic' / 'distill' / 'withdrawal-queue.jsonl').is_file():
        shutil.copy(source / 'memetic' / 'distill' / 'withdrawal-queue.jsonl', queues / 'withdrawal-queue.jsonl')
    packages = source / 'memetic' / 'distill' / 'admission-queue.jsonl'
    if packages.is_file():  # --families: rehearse with the packages of these families only (a boundary's host memory grows with them)
        lines = [line for line in packages.read_text().splitlines() if line.strip()]
        keep = [line for line in lines if not args.families or json.loads(line)['family'] in args.families]
        (queues / 'admission-queue.jsonl').write_text(''.join(line + '\n' for line in keep))
    if args.plan:
        source_withdrawal.queue_plan(out, args.plan)
    state_path = out / 'state.json'
    write_json_durable(state_path, state)
    config, declared = interleave.bind_stage(config, state, state_path)
    window = state['interleave']['sft_windows']
    recipe, _, record = memetic_admission.prepare_window(
        config, declared, state, state_path, out / f'sft-window-{window:06d}')
    ledger = state['interleave'].get('memetic_distillation') or {}
    print(json.dumps({'event': 'rehearsal', 'window': window, 'recipe': recipe, 'record': record,
                      'withdrawals': ledger.get('withdrawals'),
                      'failures': state['interleave'].get('memetic_distillation_failures')}, indent=1), flush=True)


def preflight(args):
    """The trainer's CPU continuation preflight of a rehearsed window: the command ``interleave.run_sft_window`` would
    build from the rehearsal's append recipe (``continuation_config``), run with ``--continuation-preflight-only --device
    cpu`` into ``<out>/preflight`` (never the controller or window directories). Needs about 48 GB: run it through the
    ledger."""
    from natlang_neuralese.rl.native_supervised import _resume_source_arguments, continuation_config
    from natlang_neuralese.rl.production import expand_command
    from natlang_neuralese.train.native_state_sft import main as trainer_main
    from natlang_neuralese.train.recipe import stage_parameter_args
    out = Path(args.out).resolve()
    state = json.loads((out / 'state.json').read_text())
    config = json.loads((Path(args.controller_out) / 'config.json').read_text())
    append = state['interleave']['memetic_distillation']['admissions'][-1]['recipe']['path']
    stage, _, _ = stage_view(append, config['production_stage'])
    directory = out / 'preflight'
    (directory / 'resident-data').mkdir(parents=True, exist_ok=True)
    source = {k: v for k, v in config.items() if k != 'interleaved_rl_stage'}
    source['resident_data_directory'] = str(directory / 'resident-data')
    source['production_recipe'] = append
    checkpoint = Path(args.checkpoint or state['checkpoint']).resolve(strict=True)
    window, _ = continuation_config(source, stage, checkpoint=str(checkpoint), out=str(directory))
    window.pop('rolling_sft_resume', None)
    write_json_durable(directory / 'config.json', window)
    parameters = window['trainer_parameters']
    learner = directory / 'learner'
    values = {**{k: v for k, v in window.items() if isinstance(v, (str, int, float, list))}, 'checkpoint': str(checkpoint),
              'learner_out': str(learner), 'trainer_args': stage_parameter_args(parameters),
              'seed_trainer_args': stage_parameter_args(parameters)}
    command, reused = _resume_source_arguments(window, learner, values)
    argv = expand_command(command, values)
    start = argv.index('natlang_neuralese.train.native_state_sft') + 1
    trainer = argv[start:] + ['--device', 'cpu', '--continuation-preflight-only']
    trainer[trainer.index('--out') + 1] = str(learner)
    learner.mkdir(parents=True, exist_ok=True)
    write_json_durable(directory / 'trainer-argv.json', {'reused': reused, 'argv': trainer})
    return trainer_main(trainer)




def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)
    a = sub.add_parser('audit', help='decide the useless rows: writes withdrawal-plan.json')
    a.add_argument('--controller-out', required=True)
    a.add_argument('--checkpoint')
    a.add_argument('--recipe', help='explicit base recipe (default: the controller\'s effective recipe)')
    a.add_argument('--out', required=True)
    a.add_argument('--queue', metavar='CONTROLLER_OUT', help='also queue the plan for that controller\'s next window boundary')
    r = sub.add_parser('rehearse', help='the boundary\'s withdrawal on a scratch copy of the controller state')
    r.add_argument('--controller-out', required=True)
    r.add_argument('--out', required=True, help='a new scratch directory')
    r.add_argument('--plan', help='withdrawal plan to queue in the scratch copy first')
    r.add_argument('--families', nargs='*', help='queue only the packages of these families (default: all)')
    p = sub.add_parser('preflight', help='CPU continuation preflight of a rehearsed window (about 48 GB)')
    p.add_argument('--controller-out', required=True)
    p.add_argument('--checkpoint', help='default: the checkpoint the rehearsal continued')
    p.add_argument('--out', required=True, help='the rehearsal directory')
    args = ap.parse_args(argv)
    if args.command == 'audit':
        audit(args)
    elif args.command == 'rehearse':
        rehearse(args)
    else:
        return preflight(args)


if __name__ == '__main__':
    sys.exit(main())
