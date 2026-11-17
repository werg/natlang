#!/usr/bin/env python3
"""Per-program LoRA tuning on the LFM3 stack (plans/neuralese/TUNING_RESTORE_20261117.md).

A program is one task family of the GRPO pools: one typed function, many inputs. Its pool rows (held-v3/v2 closure and
pool errata removed) are split by split group into support / dev / confirmation; the adapter is fitted only from
verified successes of support tasks and judged by task outcomes on dev (held_eval, paired with the unadapted policy).

Stages (each reads the previous stage's output in DIR):

  split  pool rows of --family -> DIR/split.json (ids per split, pool digests) and one held_eval manifest per split,
         DIR/{support,dev,confirm}/manifest.json, with the held set's engine pin (engine.json): dev and confirmation
         are measured, support is sampled on-policy (held_eval measure) for fit data.
  rows   verified successes of support tasks -> one
         natlang.tune-turn/v1 row per model call in DIR/rows.jsonl. Sources: --rollouts, on-policy samples of the
         support split (`held_eval measure --manifest DIR/support/manifest.json`; the start policy's own verified
         successes), and --sources, an `sft_sources.py` work directory (recorded GRPO/teacher successes replayed
         under the current runtime and oracle). A tool call whose result is a runtime error is not a
         target (`rejected`): its turn is dropped and later turns keep it as failed history (prompt weight 0 anyway).
  plan   DIR/fit/plan.json for `python -m natlang_neuralese.tune.build --plan DIR/fit/plan.json --out DIR/fit/build`:
         one scope node (the whole program), free LoRA of --rank over the LFM3 target table, frozen base in BF16 with
         FP32 adapter parameters, optional GGUF export (--gguf-base-sha256, --hf-config).

  bank   bank meta-learning (tune/learned/meta_bank.py) over several programs' own fits: the basis and components
         come from the fits (PCA), Meta-SGD outer steps on their support worlds; each --held program is adapted by
         the bank's inner fit on its support rows into OUT/adapt/<family>/, a release served and measured like a fit.

Measure: serve the start policy twice with the held engine pin (`serve_native_checkpoint.py`, the second with
`--adapter-release DIR/fit/build`) and run `held_eval measure --manifest DIR/dev/manifest.json` against each, then
`held_eval compare` (paired per task; the adapted actor's policy id is CHECKPOINT+stack digest). Confirmation once,
for the selected release.

The fitter, export and serving are the tuner's (`tune.build`, `tune.export`, the engines' adapter paths): this driver
only selects data and writes the plan.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / 'training/neuralese'), str(REPO), str(HERE)]

from natlang_neuralese.common.hashing import canonical_json_sha256_hex  # noqa: E402
from natlang_neuralese.common.hashing import sha256_file_hex as sha256_file  # noqa: E402

SPLITS = ('support', 'dev', 'confirm')
FRACTIONS = (0.6, 0.25, 0.15)
ACTION_POLICY = 'natlang.tune-action-outcome/2'
TURN_SCHEMA = 'natlang.tune-turn/v1'
BINDING = 'program'
# Runtime error results of a tool call: a thrown JS error, a checker diagnostic (`LINE:COL kind: ...`), a refused
# return (`rejected`), a failed native call (`name: failed: ...`) or an `Error:` service result.
ERROR_RESULT = re.compile(r'^\s*(?:[A-Z][A-Za-z]*Error\b|Error\b|\d+:\d+ [a-z][a-z-]*:|rejected\b|[A-Za-z_][\w.]*: failed:)')


def read_jsonl(path):
    with open(path, encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def split_of(family, group, fractions=FRACTIONS):
    """The split of a split group: a stable hash rank of (family, group) against the cumulative fractions."""
    value = int(hashlib.sha256(f'program-split:{family}:{group}'.encode()).hexdigest()[:15], 16) / 16 ** 15
    edge = 0.
    for name, share in zip(SPLITS, fractions):
        edge += share
        if value < edge:
            return name
    return SPLITS[-1]


def split_group(row):
    return (row.get('curriculum') or {}).get('split_group') or row['id']


def split_manifest(directory, family, name, items, held):
    """A held_eval manifest of one split's tasks (``items``: (pool, row index, row)) with the held set's engine pin:
    dev and confirmation are measured with it, support is sampled with it (on-policy fit data)."""
    from natlang_neuralese.rl.held_eval import write_manifest, MANIFEST_SCHEMA
    from natlang_neuralese.rl.curriculum import metadata
    if not items:
        raise SystemExit(f'the {name} split is empty')
    pool = items[0][0]
    manifest = {'schema': MANIFEST_SCHEMA, 'seed': f'program-split:{family}', 'included': 0,
                'program': {'family': family, 'split': name},
                'tasks': [{'id': row['id'], 'pool_row': index, 'family': metadata(row)['family'],
                           'level': metadata(row)['level'], 'split_group': split_group(row),
                           'row_sha256': hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()}
                          for _, index, row in items],
                'excluded_split_groups': sorted({split_group(row) for _, _, row in items}),
                # held_eval.exclusion_ids reads one pool: a family's rows come from one pool file
                'pool': {'path': str(pool), 'sha256': sha256_file(pool), 'rows': sum(1 for _ in open(pool))}}
    write_manifest(directory, manifest, [row for _, _, row in items])
    engine = Path(held).parent / 'engine.json'
    if engine.exists():
        shutil.copyfile(engine, directory / 'engine.json')
    return {'path': str(directory / 'manifest.json'), 'tasks': len(items)}


# ---- split -----------------------------------------------------------------------------------------------------------
def split(args):
    from natlang_neuralese.rl.held_eval import exclusion_ids
    import grpo
    out = Path(args.out)
    if (out / 'split.json').exists():
        raise SystemExit(f'{out}/split.json exists: a program split is pinned once')
    pools = args.pool or grpo.POOLS
    held = args.held_manifest or grpo.HELD
    rows, seen = [], set()
    for pool in pools:
        for index, row in enumerate(read_jsonl(pool)):
            if row.get('family') == args.family and row['id'] not in seen and row.get('split', 'train') == 'train':
                seen.add(row['id'])
                rows.append((pool, index, row))
    if not rows:
        raise SystemExit(f'no pool rows of family {args.family}')
    excluded = exclusion_ids(held, [row for _, _, row in rows])
    errata = set()
    for path in (args.errata or sorted(Path(grpo.ERRATA).glob('*.jsonl'))):
        errata |= {row['id'] for row in read_jsonl(path)}
    exposed = set()
    for path in args.exposed or ():
        exposed |= set(json.loads(Path(path).read_text()))
    # A group touched by an excluded row stays out whole (siblings share their source).
    blocked = {split_group(row) for _, _, row in rows if row['id'] in excluded or row['id'] in exposed}
    kept = [(pool, index, row) for pool, index, row in rows
            if row['id'] not in errata and split_group(row) not in blocked]
    parts = {name: [] for name in SPLITS}
    for item in kept:
        parts[split_of(args.family, split_group(item[2]), args.fractions)].append(item)
    record = {'schema': 'natlang.program-split/1', 'family': args.family, 'fractions': list(args.fractions),
              'pools': [{'path': str(p), 'sha256': sha256_file(p)} for p in pools],
              'held_manifest': {'path': str(held), 'sha256': sha256_file(held)},
              'excluded': {'held_closure': sum(1 for _, _, r in rows if r['id'] in excluded),
                           'errata': sum(1 for _, _, r in rows if r['id'] in errata),
                           'exposed': sum(1 for _, _, r in rows if r['id'] in exposed),
                           'blocked_groups': len(blocked)},
              'counts': {name: len(items) for name, items in parts.items()},
              'groups': {name: len({split_group(r) for _, _, r in items}) for name, items in parts.items()},
              'ids': {name: sorted(r['id'] for _, _, r in items) for name, items in parts.items()}}
    for name in SPLITS:
        items = parts[name][:args.max_tasks] if args.max_tasks and name != 'support' else parts[name]
        record[f'{name}_manifest'] = split_manifest(out / name, args.family, name, items, held)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'split.json').write_text(json.dumps(record, indent=1) + '\n')
    print(json.dumps({k: record[k] for k in ('family', 'counts', 'groups', 'excluded')}))


# ---- rows ------------------------------------------------------------------------------------------------------------
def message_text(message):
    content = message.get('content')
    if isinstance(content, str):
        return content
    return ''.join(part.get('text', '') for part in content or () if isinstance(part, dict))


def tool_results(messages):
    """tool_call_id -> result text of the tool messages in a request."""
    return {m.get('tool_call_id'): message_text(m) for m in messages if m.get('role') == 'tool'}


def call_outcome(name, result, *, final):
    if result is None:
        # The episode's last reply ends it: its oracle verdict is the result of its actions.
        return 'ok' if final else 'unverified'
    return 'rejected' if ERROR_RESULT.match(result) else 'ok'


def reply_of(target):
    calls = []
    for call in target.get('tool_calls') or ():
        function = call['function']
        arguments = function.get('arguments')
        calls.append([function['name'], json.loads(arguments) if isinstance(arguments, str) and arguments else (arguments or {})])
    reply = {'text': target.get('content') or '', 'calls': calls}
    if target.get('reasoning_content'):
        reply['reasoning'] = target['reasoning_content']
    return reply


def episode_turns(task_id, key, turns):
    """natlang.tune-turn/v1 rows of one verified successful episode, and per-turn dispositions."""
    rows, dispositions = [], []
    failed = False
    for index, turn in enumerate(turns):
        target = turn['target']
        later = [t['messages'] for t in turns[index + 1:]]
        events = []
        for call in target.get('tool_calls') or ():
            result = next((tool_results(messages).get(call.get('id')) for messages in later
                           if call.get('id') in tool_results(messages)), None)
            function = call['function']
            arguments = function.get('arguments')
            arguments = json.loads(arguments) if isinstance(arguments, str) and arguments else (arguments or {})
            events.append({'toolCallId': call.get('id') or f'call_{len(events)}', 'name': function['name'],
                           'argumentsSha256': canonical_json_sha256_hex(arguments),
                           'outcome': call_outcome(function['name'], result, final=index == len(turns) - 1)})
        outcomes = {e['outcome'] for e in events}
        disposition = 'rejected' if 'rejected' in outcomes else 'unverified' if 'unverified' in outcomes else 'admissible'
        dispositions.append(disposition)
        if disposition == 'admissible':
            # The tuner's outcome vocabulary: a verified action is "ok"; ids as reply_message renders them.
            events = [{**e, 'toolCallId': f'call_{i}'} for i, e in enumerate(events)]
            rows.append({'schema': TURN_SCHEMA, 'root': f'{task_id}#{key}', 'component': task_id, 'partition': 'train',
                         'binding': BINDING, 'call': turn.get('invocation_id') or 'root', 'turn': index,
                         'messages': turn['messages'], 'tools': turn.get('tools') or [], 'reply': reply_of(target),
                         'teacher': key.split(':', 1)[0], 'source': key,
                         'evidence': {'rootOracle': True, 'enclosingSuccess': True, 'callOutcome': 'done',
                                      'actionPolicy': ACTION_POLICY, 'actionDisposition': 'admissible',
                                      'failedHistory': failed, 'actionEvents': events}})
        else:
            failed = True
    return rows, dispositions


def sampled_episodes(directory):
    """(key, task id, source, turns) of the successful episodes of an on-policy sample of the support split (a
    `held_eval measure` directory of DIR/support/manifest.json): its rollouts as natlang.rollout/1 records
    (`task_rollouts.score` emits them once), scored by the current runtime and oracle that ran them."""
    from natlang_neuralese.rl import task_rollouts
    directory = Path(directory)
    emitted = directory / 'rollouts.jsonl'
    if not emitted.exists():
        task_rollouts.score(directory, emitted, neuralese_bonus=0., efficiency_weight=0.)
    label = 'sample:' + '/'.join(directory.parts[-2:])
    for row in read_jsonl(emitted):
        if row.get('outcome') == 'task_success' and float((row.get('reward') or {}).get('quality') or 0) >= 1:
            yield f"{label}:{row['group']}:{row['sample']}", row['task_id'], 'sampled', row['turns']


def replayed_episodes(work, family, stats):
    """(key, task id, source, turns) of replay-verified successes in an `sft_sources.py` work directory: the current
    runtime's replay of each recorded success (replay/<name>/rollouts.jsonl)."""
    from sft_sources import zstd_lines
    work = Path(work)
    replays = {}
    for line in (work / 'replay.jsonl').open():
        if line.strip():
            item = json.loads(line)
            if 'key' in item:
                replays[item['key']] = item
    for path in sorted(work.glob('episodes*.jsonl.zst')):
        stats['sources'].append({'path': str(path), 'sha256': sha256_file(path)})
        for line in zstd_lines(path):
            episode = json.loads(line)
            if episode.get('family') != family:
                continue
            replay = replays.get(episode['key'])
            if replay is None:
                stats['not_replayed'] += 1
                continue
            if replay.get('outcome') != 'task_success' or not replay.get('exact'):
                stats['replay_failed'] += 1
                continue
            replayed = read_jsonl(work / 'replay' / replay['name'] / 'rollouts.jsonl')[0]
            yield episode['key'], episode['task_id'], episode['source'], replayed['turns']


def rows(args):
    directory = Path(args.dir)
    record = json.loads((directory / 'split.json').read_text())
    support = set(record['ids']['support'])
    stats = {'episodes': 0, 'not_support': 0, 'not_replayed': 0, 'replay_failed': 0, 'turns': 0, 'rows': 0,
             'rejected_turns': 0, 'unverified_turns': 0, 'tasks': set(), 'by_source': {}, 'sources': []}
    episodes = []
    for work in args.sources or ():
        episodes.append(replayed_episodes(work, record['family'], stats))
    for sample in args.rollouts or ():
        stats['sources'].append({'path': str(Path(sample).resolve()), 'kind': 'support sample'})
        episodes.append(sampled_episodes(sample))
    if not episodes:
        raise SystemExit('give --sources and/or --rollouts')
    out, keys = [], set()
    for stream in episodes:
        for key, task_id, source, turns in stream:
            stats['episodes'] += 1
            if task_id not in support:
                stats['not_support'] += 1
                continue
            if key in keys:
                continue
            keys.add(key)
            made, dispositions = episode_turns(task_id, key, turns)
            stats['turns'] += len(dispositions)
            stats['rejected_turns'] += dispositions.count('rejected')
            stats['unverified_turns'] += dispositions.count('unverified')
            stats['rows'] += len(made)
            stats['tasks'].add(task_id)
            stats['by_source'][source] = stats['by_source'].get(source, 0) + 1
            out.extend(made)
    stats['tasks'] = len(stats['tasks'])
    stats['support_tasks'] = len(support)
    if not out:
        raise SystemExit(f'no verified support successes of {record["family"]}: {stats}')
    with open(directory / 'rows.jsonl', 'w', encoding='utf-8') as handle:
        for row in out:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    (directory / 'rows.json').write_text(json.dumps(stats, indent=1) + '\n')
    print(json.dumps({k: v for k, v in stats.items() if k != 'sources'}))


# ---- plan ------------------------------------------------------------------------------------------------------------
def manifest(family, base):
    """A one-scope adapter manifest: the program is one tree node every call binds to."""
    return {'schema': 'natlang.adapters/v1', 'program': {'id': f'program:{family}', 'buildHash': family},
            'base': base, 'tree': [{'id': 'tree:', 'level': 'tree', 'path': '', 'parent': None}],
            'bindings': [{'id': BINDING, 'component': f'{family}::program', 'definitionId': 'tree:', 'contextId': None,
                          'origin': 'named', 'ancestry': ['tree:'], 'semanticHash': family}],
            'artifacts': [], 'families': [], 'memberships': [], 'scales': {}, 'validatedReductions': [],
            'fallback': 'base', 'receipts': {}}


def plan(args):
    directory = Path(args.dir)
    record = json.loads((directory / 'split.json').read_text())
    if not (directory / 'rows.jsonl').exists():
        raise SystemExit('run `rows` first')
    sha = args.checkpoint_sha256 or sha256_file(args.checkpoint)
    fit = directory / 'fit'
    fit.mkdir(parents=True, exist_ok=True)
    recipe = {'parameterization': 'free-lora', 'ranks': {'tree': args.rank, 'folder': 0, 'lambda': 0},
              'level_lr_scale': {'tree': 1., 'folder': 1., 'lambda': 1.},
              'optimizer': {'name': 'adam', 'learning_rate': args.lr}, 'batch_rows': args.batch_rows,
              'prompt_weight': 0., 'seed': args.seed, **({'target_classes': args.targets} if args.targets else {}),
              **({'epochs': args.epochs} if args.epochs else {'steps': args.steps})}
    student = {'checkpoint': str(Path(args.checkpoint).resolve())}
    exports = []
    if args.gguf_base_sha256:
        if not args.hf_config:
            raise SystemExit('--gguf-base-sha256 needs --hf-config (the base model config for the GGUF converter)')
        student.update(baseGgufSha256=args.gguf_base_sha256, hfConfigDir=str(Path(args.hf_config).resolve()))
        exports = ['gguf']
    body = {'manifest': manifest(record['family'], {'checkpoint_sha256': sha}),
            'engine': {'checkpoint': student['checkpoint'], 'base': args.base, 'device': args.device, 'dtype': args.dtype,
                       **({'store_scratch': args.store_scratch} if args.store_scratch else {})},
            'student': student, 'rows': str((directory / 'rows.jsonl').resolve()), 'recipe': recipe,
            'exports': exports, 'label': f'program:{record["family"]}'}
    (fit / 'plan.json').write_text(json.dumps(body, indent=1) + '\n')
    print(json.dumps({'plan': str(fit / 'plan.json'), 'recipe': recipe, 'exports': exports}))


# ---- bank ------------------------------------------------------------------------------------------------------------
def program_rows(engine, directory, *, sha):
    """Pool rows of one program (meta_bank.build_pool input): its support rows lowered with the engine's template,
    one world per support task."""
    from natlang_neuralese.tune.rows import lower_rows
    directory = Path(directory)
    family = json.loads((directory / 'split.json').read_text())['family']
    turns = read_jsonl(directory / 'rows.jsonl')
    lowered = lower_rows(engine, turns, prompt_weight=0.)
    return [{'family': family, 'world': turn['component'], 'actor': {'checkpoint_sha256': sha}, 'graph': row['graph']}
            for turn, row in zip(turns, lowered)], family


def program_fit(engine, directory):
    """``{name: (B, A)}`` (``Delta W = B A``) of a program's own fit (DIR/fit/build): the actual per-program LoRA fit
    the bank is built from and its mixture labels are projections of."""
    from natlang_neuralese.tune.build import import_archive
    from natlang_neuralese.tune.trainer import adapter_factors
    build = Path(directory) / 'fit' / 'build'
    result = json.loads((build / 'result.json').read_text())
    import_archive(engine, [build / 'artifacts'], [])
    factors = {}
    for item in result['artifacts']:
        for name, (factor_b, factor_a) in adapter_factors(engine, item['artifact']).items():
            factors[name] = (factor_b.detach().float().cpu(), factor_a.detach().float().cpu())
    return factors


def write_release(engine, record_id, directory, provenance):
    """A servable adapter release (`serve_native_checkpoint.py --adapter-release`): the composed adapter and its
    dependency closure (the bank basis) as store files, and the result fragment naming it."""
    from natlang_neuralese.tune.build import persist_artifacts, _write
    directory = Path(directory)
    persist_artifacts(engine, [record_id], directory / 'artifacts', directory / 'states')
    _write(directory / 'result.json', {'schema': 'natlang.tune.build-result/1', 'memberships': [],
                                        'artifacts': [{'node': 'tree:', 'artifact': record_id, 'requires': []}],
                                        'receipt': provenance})


def bank(args):
    """Bank meta-learning over program fits: the shared basis is the SVD of the meta-training programs' own LoRA fits,
    the components their principal directions (meta_bank PCA initialization), then Meta-SGD outer steps on the
    programs' support worlds; held programs are adapted by the bank's inner fit on their support rows and written as
    servable releases (OUT/adapt/<family>/) for the task-outcome gate (held_eval on each held program's dev split)."""
    import torch
    from natlang_neuralese.serve import load_native_engine
    from natlang_neuralese.tune.build import fit_store_context, _write
    from natlang_neuralese.tune.learned import meta_bank as MB
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sha = args.checkpoint_sha256 or sha256_file(args.checkpoint)
    recipe = MB.MetaRecipe.from_json({'target_pattern': args.target_pattern, 'seed': args.seed,
                                      **json.loads(args.recipe or '{}')})
    operation = {'checkpoint_sha256': sha, 'recipe': recipe.as_json(),
                 'programs': [str(Path(p).resolve()) for p in args.programs],
                 'held': [str(Path(p).resolve()) for p in args.held]}
    if (out / 'operation.json').exists() and json.loads((out / 'operation.json').read_text()) != operation:
        raise SystemExit(f'{out} is bound to another checkpoint, recipe or program set')
    _write(out / 'operation.json', operation)
    config = {'store_scratch': args.store_scratch} if args.store_scratch else {}
    with fit_store_context(config) as store:
        engine = load_native_engine(str(args.checkpoint), base=args.base, device=args.device,
                                    dtype=getattr(torch, args.dtype) if args.dtype else None, store=store)
        rows, fits, held_dirs = [], {}, {}
        for directory in args.programs:
            made, family = program_rows(engine, directory, sha=sha)
            rows.append((made, family, 'meta-train'))
            fits[family] = program_fit(engine, directory)
        for directory in args.held:
            made, family = program_rows(engine, directory, sha=sha)
            rows.append((made, family, 'held'))
            held_dirs[family] = Path(directory)
        pool = {'families': {}}
        for made, family, role in rows:
            # build_pool's role draw is replaced by the declared role (held programs are the gate's families)
            part = MB.build_pool(made, held_fraction=1. if role == 'held' else 0., rows_per_world=args.rows_per_world,
                                 support_worlds=recipe.support_worlds, query_worlds=recipe.query_worlds, seed=recipe.seed)
            if family not in part['families']:
                print(json.dumps({'event': 'program skipped', 'family': family, 'role': role,
                                  'worlds': len({r['world'] for r in made})}), flush=True)
                continue
            pool = {**part, 'families': {**pool['families'], family: part['families'][family]}}
        if not any(f['role'] == 'meta-train' for f in pool['families'].values()):
            raise SystemExit('no meta-training program has enough support worlds')
        receipt = MB.write_pool(pool, out / 'pool')
        meta_fits = {family: fits[family] for family, f in pool['families'].items() if f['role'] == 'meta-train'}
        learner = MB.BankMetaLearner(engine, pool, recipe, out / 'learner', parent_sha256=sha, fits=meta_fits)
        print(json.dumps({'event': 'start', 'pool': receipt, 'step': learner.state['step'],
                          'targets': learner.state['targets'], 'components': learner.state['components'],
                          'spectrum': learner.state.get('spectrum')}), flush=True)
        while learner.state['step'] < args.steps:
            entry = learner.step()
            learner.save()
            print(json.dumps({'event': 'outer', 'step': entry['step'], 'seconds': entry['seconds'],
                              'grad': entry['grad_norm'], 'query': [round(r['query_fitted'], 4) for r in entry['families']]}),
                  flush=True)
        if any(f['role'] == 'held' for f in pool['families'].values()):
            evaluation = learner.evaluate('dev')
            learner.save()
            print(json.dumps({'event': 'teacher-forced dev', 'mean': evaluation['mean'],
                              'meta_wins': evaluation['meta_wins']}), flush=True)
        manifest = learner.publish()
        learner.save()
        for family, directory in held_dirs.items():
            made = [r for made, f, _ in rows if f == family for r in made]
            bank_ = learner.bank_of('meta')
            fitted = learner.fit(made, 'meta')
            cores = bank_.composed_cores(list(range(len(bank_.components))), fitted['coefficients'])
            record = bank_.compose_adapter(engine, cores, provenance={'bank_adaptation': 'meta', 'family': family,
                                                                      'support': [r['graph']['id'] for r in made]})
            write_release(engine, record.id, out / 'adapt' / family,
                          {'bank': manifest['snapshot'], 'family': family, 'support_rows': len(made),
                           'support_loss': fitted['history'][-1], 'program': str(directory.resolve())})
            print(json.dumps({'event': 'adapted', 'family': family, 'release': str(out / 'adapt' / family),
                              'support_loss': [round(fitted['history'][0], 4), round(fitted['history'][-1], 4)]}),
                  flush=True)
        _write(out / 'bank-manifest.json', manifest)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    s = sub.add_parser('split')
    s.add_argument('--family', required=True)
    s.add_argument('--out', required=True)
    s.add_argument('--pool', action='append', help='task pools (default grpo.POOLS)')
    s.add_argument('--held-manifest', help='default grpo.HELD (held-v3; its exclusion covers held-v2)')
    s.add_argument('--errata', action='append', help='errata JSONL (default every grpo.ERRATA/*.jsonl)')
    s.add_argument('--exposed', action='append',
                   help='JSON list of task ids the start policy trained on (a later parent than the base): kept out')
    s.add_argument('--fractions', type=float, nargs=3, default=FRACTIONS, metavar=('SUPPORT', 'DEV', 'CONFIRM'))
    s.add_argument('--max-tasks', type=int, help='cap dev/confirmation manifests at this many tasks (in pool order)')
    r = sub.add_parser('rows')
    r.add_argument('--dir', required=True)
    r.add_argument('--sources', action='append', help='an sft_sources.py work directory (extracted and replayed)')
    r.add_argument('--rollouts', action='append',
                   help='a held_eval measure directory of DIR/support/manifest.json (on-policy support samples)')
    p = sub.add_parser('plan')
    p.add_argument('--dir', required=True)
    p.add_argument('--checkpoint', required=True, help='the start policy (native checkpoint)')
    p.add_argument('--checkpoint-sha256')
    p.add_argument('--base', required=True, help='the HF base directory (config and tokenizer)')
    p.add_argument('--device', default='cuda')
    p.add_argument('--dtype', default='bfloat16', help='frozen base precision (adapter parameters are FP32)')
    p.add_argument('--store-scratch')
    p.add_argument('--rank', type=int, default=16)
    p.add_argument('--targets', nargs='+', choices=('attention', 'conv-projection', 'mlp'),
                   help='target classes (default: all). The native engine applies an adapter by substituting each '
                        'target\'s full effective weight (FP32 delta + its gradient), so memory grows with the targets')
    p.add_argument('--lr', type=float, default=2e-3)
    p.add_argument('--batch-rows', type=int, default=8)
    p.add_argument('--epochs', type=float)
    p.add_argument('--steps', type=int, default=1)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--gguf-base-sha256', help='sha256 of the base GGUF the LoRA exports bind to')
    p.add_argument('--hf-config', help='HF config directory of the base for the GGUF LoRA converter')
    b = sub.add_parser('bank', help='bank meta-learning over program fits; held programs adapted for the outcome gate')
    b.add_argument('--programs', nargs='+', required=True, help='meta-training program directories (fit/build done)')
    b.add_argument('--held', nargs='*', default=[], help='held program directories (their own fits are references)')
    b.add_argument('--out', required=True)
    b.add_argument('--checkpoint', required=True)
    b.add_argument('--checkpoint-sha256')
    b.add_argument('--base')
    b.add_argument('--device', default='cuda')
    b.add_argument('--dtype', help='frozen base precision (default: the checkpoint\'s)')
    b.add_argument('--store-scratch')
    b.add_argument('--steps', type=int, default=0, help='outer meta-training steps (0: the PCA bank alone)')
    b.add_argument('--rows-per-world', type=int, default=8)
    b.add_argument('--target-pattern', default=r'self_attn\.(q_proj|k_proj|v_proj|o_proj)|conv\.(in_proj|out_proj)',
                   help='bank target matrices (default: the program fits\' attention and conv projections)')
    b.add_argument('--recipe', help='JSON overrides of meta_bank.MetaRecipe')
    b.add_argument('--seed', type=int, default=0)
    args = parser.parse_args(argv)
    {'split': split, 'rows': rows, 'plan': plan, 'bank': bank}[args.command](args)


if __name__ == '__main__':
    main()
