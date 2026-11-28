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
  teacher-tasks  a teacher_gen.py WORK over the support split (tasks.jsonl, pool.json): API teachers' verified
         successes, replayed by sft_sources.py, are fit rows (--sources) when no actor is free for on-policy samples.
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
import math
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
SEGMENT_EXIT = 75     # tune.build: a segment ended before the last step (rerun to continue)
TURN_SCHEMA = 'natlang.tune-turn/v1'
BINDING = 'program'
# Runtime error results of a tool call: a thrown JS error, a checker diagnostic (`LINE:COL kind: ...`), a refused
# return (`rejected`), a failed native call (`name: failed: ...`) or an `Error:` service result.
ERROR_RESULT = re.compile(r'^\s*(?:[A-Z][A-Za-z]*Error\b|Error\b|\d+:\d+ [a-z][a-z-]*:|rejected\b|[A-Za-z_][\w.]*: failed:)')


def logical(path):
    """The absolute path as written, symlinks kept: a plan or operation record must not bind to where a symlink
    pointed when it was written (2026-11-20: plans resolved through the archive symlink to the HDD archive, the
    directory moved back to the NVMe, and every fit of the chain failed on a missing rows.jsonl)."""
    import os
    return os.path.abspath(path)


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


def subset(args):
    """DIR/<NAME>/: the first --tasks of a split's manifest in stable hash order, as its own pinned held_eval manifest
    (the split's engine.json beside it; ``subset_of`` names the parent manifest's sha256). A measurement arm of every
    program on the whole dev split costs ~4,200 episodes at 2 samples; a fixed subset is the same population, cheaper."""
    from natlang_neuralese.rl.held_eval import write_manifest
    directory = Path(args.dir)
    parent = directory / args.split
    manifest = json.loads((parent / 'manifest.json').read_text())
    family = (manifest.get('program') or {}).get('family') or directory.name
    order = sorted(manifest['tasks'], key=lambda t: hashlib.sha256(f'subset:{family}:{t["id"]}'.encode()).hexdigest())
    keep = order[:args.tasks]
    ids = {t['id'] for t in keep}
    rows = [row for row in read_jsonl(parent / 'cases.jsonl') if row['id'] in ids]
    out = directory / (args.name or f'{args.split}{args.tasks}')
    body = {k: v for k, v in manifest.items() if k != 'cases'}
    body.update(tasks=[t for t in manifest['tasks'] if t['id'] in ids],
                excluded_split_groups=manifest['excluded_split_groups'],
                subset_of={'manifest_sha256': sha256_file(parent / 'manifest.json'), 'split': args.split,
                           'tasks': len(ids), 'order': f'sha256(subset:{family}:<task id>)'})
    sha = write_manifest(out, body, rows)
    if (parent / 'engine.json').exists():
        shutil.copyfile(parent / 'engine.json', out / 'engine.json')
    print(json.dumps({'manifest': str(out / 'manifest.json'), 'sha256': sha, 'tasks': len(rows)}))


# ---- split -----------------------------------------------------------------------------------------------------------
def split(args):
    from natlang_neuralese.rl.held_eval import training_exclusion_ids
    import grpo
    out = Path(args.out)
    if (out / 'split.json').exists():
        raise SystemExit(f'{out}/split.json exists: a program split is pinned once')
    pools = args.pool or grpo.default_pools()
    held = args.held_manifest or grpo.default_held()
    rows, seen = [], set()
    for pool in pools:
        for index, row in enumerate(read_jsonl(pool)):
            if row.get('family') == args.family and row['id'] not in seen and row.get('split', 'train') == 'train':
                seen.add(row['id'])
                rows.append((pool, index, row))
    if not rows:
        raise SystemExit(f'no pool rows of family {args.family}')
    excluded = training_exclusion_ids([row for _, _, row in rows], held)
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
    # A call's last turn ends that invocation (a nested call's result goes to its caller, not to a later turn of its
    # own): in a verified exact episode its actions are judged by the episode's outcome, as the root's last turn is.
    last_of = {turn.get('invocation_id'): index for index, turn in enumerate(turns)}
    # a reasoned readout's samples train only when their answer is the one the program used
    # (data.decision_graph.reasoned_decision_plan, as sft_sources build)
    from natlang_neuralese.data.decision_graph import reasoned_decision_plan
    from natlang_neuralese.tune.rows import empty_reply_message, train_empty_replies
    reasoned = reasoned_decision_plan(turns)
    for index, turn in enumerate(turns):
        if reasoned[index] is not None and not reasoned[index]['train']:
            dispositions.append('reasoned-sample-not-used')
            continue
        target = turn['target']
        if empty_reply_message(target) and not train_empty_replies():
            # no action (no text, no tool call): its only target would be an immediate end of turn
            # (tune.rows.empty_reply; NATLANG_TUNE_EMPTY_REPLIES=train opts out)
            dispositions.append('empty-reply')
            continue
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
                           'outcome': call_outcome(function['name'], result, final=index == len(turns) - 1 or last_of[turn.get('invocation_id')] == index)})
        outcomes = {e['outcome'] for e in events}
        disposition = 'rejected' if 'rejected' in outcomes else 'unverified' if 'unverified' in outcomes else 'admissible'
        dispositions.append(disposition)
        if disposition == 'admissible':
            # The tuner's outcome vocabulary: a verified action is "ok"; ids as reply_message renders them.
            events = [{**e, 'toolCallId': f'call_{i}'} for i, e in enumerate(events)]
            rows.append({'schema': TURN_SCHEMA, 'root': f'{task_id}#{key}', 'component': task_id, 'partition': 'train',
                         'binding': BINDING, 'call': turn.get('invocation_id') or 'root', 'turn': index,
                         'messages': turn['messages'], 'tools': turn.get('tools') or [], 'reply': reply_of(target),
                         **({'decision_reasoning': {k: turn['decision_reasoning'].get(k)
                                                    for k in ('index', 'samples', 'action')}}
                            if reasoned[index] is not None else {}),
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


def replayed_episodes(work, family, stats, drifted=True):
    """(key, task id, source, turns) of replay-verified successes in an `sft_sources.py` work directory: the current
    runtime's replay of each recorded success (replay/<name>/rollouts.jsonl). ``drifted``: also teacher successes whose
    replay display drifted (``sft_sources.drifted_success``, the rule ``verified_episodes`` applies)."""
    from sft_sources import drifted_success, zstd_lines
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
            if replay.get('outcome') != 'task_success' or not (
                    replay.get('exact') or (drifted and episode.get('source') == 'teacher'
                                            and drifted_success(replay, episode['turns']))):
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
        episodes.append(replayed_episodes(work, record['family'], stats,
                                          drifted=getattr(args, 'admit_drifted', 'on') == 'on'))
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
            stats['empty_reply_turns'] = stats.get('empty_reply_turns', 0) + dispositions.count('empty-reply')
            stats['reasoned_samples_not_used'] = (stats.get('reasoned_samples_not_used', 0)
                                                  + dispositions.count('reasoned-sample-not-used'))
            stats['rows'] += len(made)
            stats['tasks'].add(task_id)
            stats['by_source'][source] = stats['by_source'].get(source, 0) + 1
            out.extend(made)
    stats['tasks'] = len(stats['tasks'])
    stats['support_tasks'] = len(support)
    stats['system_prompts'] = system_prompt_counts(row['messages'] for row in out)
    if not out:
        raise SystemExit(f'no verified support successes of {record["family"]}: {stats}')
    with open(directory / 'rows.jsonl', 'w', encoding='utf-8') as handle:
        for row in out:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    (directory / 'rows.json').write_text(json.dumps(stats, indent=1) + '\n')
    print(json.dumps({k: v for k, v in stats.items() if k != 'sources'}))


def system_prompt_counts(requests):
    """{sha256 of a request's system prompt (first 16 hex): count} over requests (message lists)."""
    counts = {}
    for messages in requests:
        first = (messages or [{}])[0]
        if first.get('role') == 'system':
            key = hashlib.sha256(message_text(first).encode()).hexdigest()[:16]
            counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: -item[1]))


def prompt_parity(directory, arm_dir):
    """How many of DIR's fit rows were rendered with a system prompt the arm's episodes saw (first request of each):
    the fit trains on the runtime its rows were replayed with, the arm serves the episodes' pin. 2026-11-25: every
    program's rows (replayed 11-17..11-20) carried an 18.2k-character system prompt, the dev arms (pin de2b36add) a
    14.8k one: 0% parity. Rows rebuilt from a replay at the arm's pin restore it (``rows``)."""
    rows = [json.loads(line) for line in (directory / 'rows.jsonl').open() if line.strip()]
    trained = system_prompt_counts(row['messages'] for row in rows)
    served = system_prompt_counts(step['context'] for path in sorted(Path(arm_dir).glob('row-*/sample-*/results.jsonl'))
                                  for line in path.open() if line.strip()
                                  for step in json.loads(line).get('trajectory', [])[:1] if step.get('context'))
    same = sum(count for key, count in trained.items() if key in served)
    return {'schema': 'natlang.program-prompt-parity/1', 'rows': len(rows), 'rows_with_served_prompt': same,
            'share': same / len(rows) if rows else None, 'trained': trained, 'served': served}


# ---- plan ------------------------------------------------------------------------------------------------------------
def manifest(family, base):
    """A one-scope adapter manifest: the program is one tree node every call binds to."""
    return {'schema': 'natlang.adapters/v1', 'program': {'id': f'program:{family}', 'buildHash': family},
            'base': base, 'tree': [{'id': 'tree:', 'level': 'tree', 'path': '', 'parent': None}],
            'bindings': [{'id': BINDING, 'component': f'{family}::program', 'definitionId': 'tree:', 'contextId': None,
                          'origin': 'named', 'ancestry': ['tree:'], 'semanticHash': family}],
            'artifacts': [], 'families': [], 'memberships': [], 'scales': {}, 'validatedReductions': [],
            'fallback': 'base', 'receipts': {}}


DEFAULT_SFT_LR = 5e-5
MAX_SAFE_SFT_LR = 5e-4
DEFAULT_DEPTH_PENALTY = 1e-3


def plan(args):
    directory = Path(args.dir)
    record = json.loads((directory / 'split.json').read_text())
    rows_path = directory / 'rows.jsonl'
    if not rows_path.exists():
        raise SystemExit('run `rows` first')
    if args.lr > MAX_SAFE_SFT_LR and not getattr(args, 'allow_high_lr', False):
        raise SystemExit(
            f'--lr {args.lr} exceeds safe LFM3 LoRA limit {MAX_SAFE_SFT_LR} '
            '(2e-3 collapsed ext_decisions_tally/tree by -30 to -40 pts); pass --allow-high-lr to override'
        )
    n_rows = sum(1 for line in rows_path.open(encoding='utf-8') if line.strip())
    rows_meta = json.loads((directory / 'rows.json').read_text()) if (directory / 'rows.json').exists() else {}
    n_tasks = int(rows_meta.get('tasks') or 0)
    min_rows = int(getattr(args, 'min_rows', 0) or 0)
    min_tasks = int(getattr(args, 'min_tasks', 0) or 0)
    if getattr(args, 'strict_support', False):
        min_rows = max(min_rows, 32)
        min_tasks = max(min_tasks, 8)
    if min_rows and n_rows < min_rows:
        raise SystemExit(f'{directory}: only {n_rows} support rows (< --min-rows {min_rows}); collect more support data or use `grpo`')
    if min_tasks and rows_meta and n_tasks < min_tasks:
        raise SystemExit(f'{directory}: only {n_tasks} distinct support tasks (< --min-tasks {min_tasks}); collect more support data or use `grpo`')
    steps = args.steps
    max_epochs = getattr(args, 'max_epochs', None)
    capped_steps = None
    if not args.epochs and max_epochs is not None and max_epochs > 0 and n_rows > 0:
        cap = max(1, math.ceil(float(max_epochs) * n_rows / max(1, args.batch_rows)))
        if steps > cap:
            capped_steps = cap
            steps = cap
    effective_epochs = (float(args.epochs) if args.epochs else (steps * args.batch_rows) / max(1, n_rows))
    sha = args.checkpoint_sha256 or sha256_file(args.checkpoint)
    fit = directory / 'fit'
    fit.mkdir(parents=True, exist_ok=True)
    depth_penalty = float(getattr(args, 'depth_penalty', DEFAULT_DEPTH_PENALTY))
    recipe = {'parameterization': 'free-lora', 'ranks': {'tree': args.rank, 'folder': 0, 'lambda': 0},
              'level_lr_scale': {'tree': 1., 'folder': 1., 'lambda': 1.},
              'depth_penalty': {'tree': depth_penalty, 'folder': depth_penalty, 'lambda': depth_penalty},
              'optimizer': {'name': 'adam', 'learning_rate': args.lr}, 'batch_rows': args.batch_rows,
              'prompt_weight': 0., 'seed': args.seed, **({'target_classes': args.targets} if args.targets else {}),
              **({'epochs': args.epochs} if args.epochs else {'steps': steps})}
    student = {'checkpoint': logical(args.checkpoint)}
    exports = []
    if args.gguf_base_sha256:
        if not args.hf_config:
            raise SystemExit('--gguf-base-sha256 needs --hf-config (the base model config for the GGUF converter)')
        student.update(baseGgufSha256=args.gguf_base_sha256, hfConfigDir=logical(args.hf_config))
        exports = ['gguf']
    body = {'manifest': manifest(record['family'], {'checkpoint_sha256': sha}),
            'engine': {'checkpoint': student['checkpoint'], 'base': args.base, 'device': args.device, 'dtype': args.dtype,
                       **({'store_scratch': args.store_scratch} if args.store_scratch else {})},
            'student': student, 'rows': logical(rows_path), 'recipe': recipe,
            'exports': exports, 'label': f'program:{record["family"]}'}
    (fit / 'plan.json').write_text(json.dumps(body, indent=1) + '\n')
    summary = {'plan': str(fit / 'plan.json'), 'recipe': recipe, 'exports': exports,
               'support_rows': n_rows, 'support_tasks': n_tasks or None,
               'effective_epochs': round(effective_epochs, 2)}
    if capped_steps is not None:
        summary['capped_steps'] = capped_steps
    if effective_epochs > 3.0 and n_rows > 1:
        summary['warning'] = f'effective_epochs={effective_epochs:.1f} > 3.0 risks memorizing support rows'
    print(json.dumps(summary))


# ---- bank ------------------------------------------------------------------------------------------------------------
def world_order(family, task):
    return hashlib.sha256(f'bank-world:{family}:{task}'.encode()).hexdigest()


def program_rows(engine, directory, *, sha, max_worlds=None):
    """Pool rows of one program (meta_bank.build_pool input): its support rows lowered with the engine's template,
    one world per support task; with ``max_worlds``, only that many support tasks in stable hash order (an episode
    draws a few worlds: lowering every row of a 12k-row program costs time and memory and adds nothing)."""
    from natlang_neuralese.tune.rows import lower_rows
    directory = Path(directory)
    family = json.loads((directory / 'split.json').read_text())['family']
    turns = read_jsonl(directory / 'rows.jsonl')
    if max_worlds:
        keep = set(sorted({t['component'] for t in turns}, key=lambda task: world_order(family, task))[:max_worlds])
        turns = [t for t in turns if t['component'] in keep]
    lowered = lower_rows(engine, turns, prompt_weight=0.)
    return [{'family': family, 'world': turn['component'], 'actor': {'checkpoint_sha256': sha}, 'graph': row['graph']}
            for turn, row in zip(turns, lowered)], family


def program_fit(engine, directory):
    """``{name: (B, A)}`` (``Delta W = B A``) of a program's own fit (DIR/fit/build): the actual per-program LoRA fit
    the bank is built from and its mixture labels are projections of."""
    from natlang_neuralese.tune.build import import_archive
    from natlang_neuralese.tune.trainer import adapter_factors
    if not (Path(directory) / 'fit' / 'release' / 'result.json').exists():
        release(argparse.Namespace(dir=directory))   # read one artifact, not every step's (12 GB on LFM3)
    build = Path(directory) / 'fit' / 'release'
    result = json.loads((build / 'result.json').read_text())
    import_archive(engine, [build / 'artifacts'], [])
    factors = {}
    for item in result['artifacts']:
        for name, (factor_b, factor_a) in adapter_factors(engine, item['artifact']).items():
            factors[name] = (factor_b.detach().float().cpu(), factor_a.detach().float().cpu())
    return factors


def artifact_closure(artifacts, ids):
    """The `.nnt` files of ``ids`` and their dependency closure in an artifacts directory."""
    from natlang_neuralese.model.native_artifact import decode_native_tensor_artifact, native_tensor_dependencies
    artifacts, seen, todo = Path(artifacts), {}, list(ids)
    while todo:
        ident = todo.pop()
        if ident in seen:
            continue
        path = artifacts / f'{ident}.nnt'
        if not path.is_file():
            raise SystemExit(f'{path}: an artifact of the release closure is missing')
        seen[ident] = path
        artifact, _ = decode_native_tensor_artifact(path.read_bytes())
        todo += [d for d in native_tensor_dependencies(artifact) if isinstance(d, str) and d.startswith('nnt1_')]
    return sorted(seen.values())


def release(args):
    """DIR/fit/release: the fit's result.json (byte-identical, so the served policy id is unchanged) and only the
    artifacts it names with their dependency closure. A tune.build output keeps every step's adapter and optimizer
    state for resumption (12 GB for a 60-step all-target rank-16 fit); a served release needs one 66 MB artifact."""
    directory = Path(args.dir)
    build, out = directory / 'fit' / 'build', directory / 'fit' / 'release'
    result = json.loads((build / 'result.json').read_text())
    if (build / 'states').is_dir() and any((build / 'states').iterdir()):
        raise SystemExit(f'{build}/states: soft-state releases are copied whole (not handled here)')
    files = artifact_closure(build / 'artifacts', [item['artifact'] for item in result['artifacts']])
    temp = out.with_name('release.tmp')
    shutil.rmtree(temp, ignore_errors=True)
    (temp / 'artifacts').mkdir(parents=True)
    for path in files:
        shutil.copyfile(path, temp / 'artifacts' / path.name)
    shutil.copyfile(build / 'result.json', temp / 'result.json')
    shutil.rmtree(out, ignore_errors=True)
    temp.rename(out)
    print(json.dumps({'release': str(out), 'artifacts': [p.name for p in files],
                      'bytes': sum(p.stat().st_size for p in files)}))
    return out


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
                 'programs': [logical(p) for p in args.programs],
                 'held': [logical(p) for p in args.held]}
    if (out / 'operation.json').exists() and json.loads((out / 'operation.json').read_text()) != operation:
        raise SystemExit(f'{out} is bound to another checkpoint, recipe or program set')
    _write(out / 'operation.json', operation)
    config = {'store_scratch': args.store_scratch} if args.store_scratch else {}
    with fit_store_context(config) as store:
        engine = load_native_engine(str(args.checkpoint), base=args.base, device=args.device,
                                    dtype=getattr(torch, args.dtype) if args.dtype else None, store=store, training_state=False)
        rows, fits, held_dirs = [], {}, {}
        for directory in args.programs:
            made, family = program_rows(engine, directory, sha=sha, max_worlds=args.max_worlds)
            rows.append((made, family, 'meta-train'))
            fits[family] = program_fit(engine, directory)
        for directory in args.held:
            made, family = program_rows(engine, directory, sha=sha, max_worlds=args.max_worlds)
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
            # the bank adapts from a few support rows (its point is few-shot adaptation): --held-support-rows of the
            # held program's support rows, whole worlds in stable hash order
            made = sorted((r for made, f, _ in rows if f == family for r in made),
                          key=lambda r: (world_order(family, r['world']), r['graph']['id']))[:args.held_support_rows]
            bank_ = learner.bank_of('meta')
            fitted = learner.fit(made, 'meta')
            cores = bank_.composed_cores(list(range(len(bank_.components))), fitted['coefficients'])
            record = bank_.compose_adapter(engine, cores, provenance={'bank_adaptation': 'meta', 'family': family,
                                                                      'support': [r['graph']['id'] for r in made]})
            write_release(engine, record.id, out / 'adapt' / family,
                          {'bank': manifest['snapshot'], 'family': family, 'support_rows': len(made),
                           'support_loss': fitted['history'][-1], 'program': logical(directory)})
            print(json.dumps({'event': 'adapted', 'family': family, 'release': str(out / 'adapt' / family),
                              'support_loss': [round(fitted['history'][0], 4), round(fitted['history'][-1], 4)]}),
                  flush=True)
        _write(out / 'bank-manifest.json', manifest)


# ---- Colab tenants (DECISIONS 2026-11-24 "program tuning on the A100") -----------------------------------------------
def remote_plan(plan, *, checkpoint, base, rows, scratch):
    """The plan with its paths on the VM (the recipe, manifest and labels unchanged)."""
    return {**plan, 'engine': {**plan['engine'], 'checkpoint': checkpoint, 'base': base, 'store_scratch': scratch},
            'student': {**plan['student'], 'checkpoint': checkpoint}, 'rows': rows}


def colab_fit(args):
    """DIR's fit as a tenant of the Colab session (beside its actor; yields to a round's learner and restarts from its
    own state.json): the start checkpoint and rows shipped (Drive), tune.build at the tenant commit, the release made on
    the VM and pulled back verified into DIR/fit/release (and DIR/fit/build/{result,state}.json). Exit 0 when the release
    is home; another status when the tenant was evicted or failed (rerun: it resumes)."""
    import shlex
    import time
    import pop_rl_actor as A
    directory = Path(args.dir)
    fit = directory / 'fit'
    if (fit / 'release' / 'result.json').exists():
        print(json.dumps({'event': 'release present', 'dir': str(directory)}))
        return 0
    state = fit / 'build' / 'state.json'
    if state.exists() and not args.take_over:
        raise SystemExit(f'{state}: a local fit is in progress (--take-over to fit it on Colab from scratch)')
    plan = json.loads((fit / 'plan.json').read_text())
    family = json.loads((directory / 'split.json').read_text())['family']
    cfg = A.host_config('colab', host=args.host)
    host = cfg['host']
    # one host per program: another session's claim stands unless --take-over (two Colab sessions, 2026-11-24)
    claim_file = fit / 'colab-claim.json'
    if claim_file.exists() and not args.take_over:
        other = json.loads(claim_file.read_text()).get('host')
        if other and other != host:
            raise SystemExit(f'{directory.name} is claimed by {other} (--take-over to fit it here)')
    # the DGX fit chain (fit-seq.sh) skips a program with this marker: one program is never fitted twice at once
    (fit / 'colab-claim.json').write_text(json.dumps({'host': host, 'commit': args.commit, 'at': time.time()}) + '\n')
    tcfg = A.colab_tenant_checkout(host, cfg, args.commit)
    remote = f'{cfg["work_root"]}/tenant-fit-{family}'
    sha = plan['manifest']['base']['checkpoint_sha256']
    started = time.time()
    A.ensure_remote_checkpoint(host, cfg, plan['engine']['checkpoint'], sha)
    rows_sha, remote_rows = sha256_file(plan['rows']), f'{remote}/rows.jsonl'
    have = A.ssh_host(host, f'sha256sum {shlex.quote(remote_rows)} 2>/dev/null || true', check=False).split()
    if not have or have[0] != rows_sha:
        A.ship_file(plan['rows'], remote_rows, host, cfg, rows_sha)
    body = remote_plan(plan, checkpoint=f'{cfg["checkpoints"]}/{sha}.pt', base=cfg['base'], rows=remote_rows,
                       scratch=f'{remote}/fit-scratch')
    # the fit store's scratch directory must exist (tempfile.mkdtemp under it; the DGX plans' existed already)
    A.ssh_host(host, f'mkdir -p {shlex.quote(remote)}/fit {shlex.quote(remote)}/fit-scratch && '
                     f'cat > {shlex.quote(remote)}/fit/plan.json',
               input=json.dumps(body, indent=1) + '\n', timeout=60)
    claim = dict(A.colab_layout(A.colab_hardware(host))['tenants']['fit'])
    if args.vram_gib:     # two fits share the L4 (each ~8 GB, one CPU core bound: 9% GPU at 2.5 min/step, 11-24)
        claim['vram_gib'] = float(args.vram_gib)
    segment = ['--segment-steps', str(args.segment_steps)] if args.segment_steps else []
    command = A.colab_tenant_command(cfg, tcfg, claim, f'fit-{family[:40]}',
                                     [cfg['python'], '-m', 'natlang_neuralese.tune.build', '--plan',
                                      f'{remote}/fit/plan.json', '--out', f'{remote}/fit/build', *segment],
                                     mps_percent=args.mps_percent, beside=args.beside or None)
    shipped = time.time()
    rc = A.run_detached(host, cfg, remote, command)
    while args.segment_steps and rc == SEGMENT_EXIT:    # a segment ended (state saved; its memory growth released): the next one resumes
        rc = A.run_detached(host, cfg, remote, command)
    trained = time.time()
    A.log(event='Colab tenant fit ended', host=host, family=family, returncode=rc, ship_seconds=round(shipped - started, 1),
          fit_seconds=round(trained - shipped, 1))
    if rc != 0:
        return rc if rc else 1
    A.ssh_host(host, shlex.join(['/usr/bin/env', f'PYTHONPATH={tcfg["repo"]}/training/neuralese:{tcfg["repo"]}',
                                 cfg['python'], f'{tcfg["repo"]}/scripts/neuralese_data/program_tune.py', 'release',
                                 '--dir', remote]), timeout=600)
    names = A.ssh_host(host, f'ls {shlex.quote(remote)}/fit/release/artifacts', timeout=60).split()
    staged = fit / 'release.pull'
    shutil.rmtree(staged, ignore_errors=True)
    (staged / 'artifacts').mkdir(parents=True)
    for name in names:
        A.pull_verified(f'{remote}/fit/release/artifacts/{name}', staged / 'artifacts' / name, host)
    A.pull_verified(f'{remote}/fit/release/result.json', staged / 'result.json', host)
    (fit / 'build').mkdir(parents=True, exist_ok=True)
    A.pull_verified(f'{remote}/fit/build/state.json', fit / 'build' / 'state.json', host)
    shutil.copyfile(staged / 'result.json', fit / 'build' / 'result.json')
    shutil.rmtree(fit / 'release', ignore_errors=True)
    staged.rename(fit / 'release')
    record = {'schema': 'natlang.program-fit-colab/1', 'host': host, 'tenant_commit': tcfg['commit'], 'remote': remote,
              'gpu': A.colab_hardware(host).get('gpu_name'), 'ship_seconds': round(shipped - started, 1),
              'fit_seconds': round(trained - shipped, 1), 'pull_seconds': round(time.time() - trained, 1),
              'mps': bool(A.COLAB_TENANT_MPS or args.mps_percent is not None),
              'mps_percent': args.mps_percent if args.mps_percent is not None else
              (A.COLAB_TENANT_MPS_PERCENT if A.COLAB_TENANT_MPS else None)}
    (fit / 'colab.json').write_text(json.dumps(record, indent=1) + '\n')
    print(json.dumps({'event': 'release home', 'dir': str(directory), **record}))
    return 0


def colab_bank(args):
    """The bank (``bank``) as a tenant of the Colab session: each program's split.json, rows.jsonl and fit release (the
    meta-training fits; held programs' rows) shipped to the VM, ``bank`` run there at the tenant commit, and each held
    program's adapted release (OUT/adapt/<family>/), the bank manifest and the learner state pulled back verified into
    --out. Resumable (the learner saves every outer step); exit 0 when the releases are home."""
    import shlex
    import time
    import pop_rl_actor as A
    out = Path(args.out)
    cfg = A.host_config('colab', host=args.host)
    host = cfg['host']
    tcfg = A.colab_tenant_checkout(host, cfg, args.commit)
    remote = f'{cfg["work_root"]}/tenant-bank-{out.name}'
    first = json.loads((Path(args.programs[0]) / 'fit' / 'plan.json').read_text())
    sha = first['manifest']['base']['checkpoint_sha256']
    A.ensure_remote_checkpoint(host, cfg, first['engine']['checkpoint'], sha)
    remote_dirs = {}
    for item in [*args.programs, *args.held]:
        directory = Path(item)
        family = json.loads((directory / 'split.json').read_text())['family']
        target = f'{remote}/programs/{family}'
        A.ssh_host(host, shlex.join(['mkdir', '-p', f'{target}/fit']))
        A.copy_resumable(str(directory / 'split.json'), f'{host}:{target}/split.json')
        rows_sha = sha256_file(directory / 'rows.jsonl')
        have = A.ssh_host(host, f'sha256sum {shlex.quote(target)}/rows.jsonl 2>/dev/null || true', check=False).split()
        if not have or have[0] != rows_sha:
            A.ship_file(str(directory / 'rows.jsonl'), f'{target}/rows.jsonl', host, cfg, rows_sha)
        if item in args.programs:
            A.ship_release(host, cfg, str(directory / 'fit' / 'release'), f'{target}/fit/release')
        remote_dirs[item] = target
    command = [cfg['python'], f'{tcfg["repo"]}/scripts/neuralese_data/program_tune.py', 'bank',
               '--programs', *[remote_dirs[p] for p in args.programs], '--held', *[remote_dirs[p] for p in args.held],
               '--out', f'{remote}/out', '--checkpoint', f'{cfg["checkpoints"]}/{sha}.pt', '--checkpoint-sha256', sha,
               '--base', cfg['base'], '--dtype', 'bfloat16', '--store-scratch', f'{remote}/scratch',
               '--steps', str(args.steps), '--max-worlds', str(args.max_worlds),
               '--held-support-rows', str(args.held_support_rows), *(['--recipe', args.recipe] if args.recipe else [])]
    A.ssh_host(host, shlex.join(['mkdir', '-p', f'{remote}/scratch']))
    claim = {'vram_gib': args.vram_gib, 'host_ram_gib': args.host_ram_gib}
    started = time.time()
    rc = A.run_detached(host, cfg, remote, A.colab_tenant_command(cfg, tcfg, claim, f'bank-{out.name[:30]}', command,
                                                                  beside=args.beside or None))
    A.log(event='Colab tenant bank ended', host=host, returncode=rc, seconds=round(time.time() - started, 1))
    if rc != 0:
        return rc if rc else 1
    out.mkdir(parents=True, exist_ok=True)
    for name in ('bank-manifest.json', 'operation.json', 'learner/state.json'):
        A.pull_verified(f'{remote}/out/{name}', out / name, host)
    for item in args.held:
        family = json.loads((Path(item) / 'split.json').read_text())['family']
        src, staged = f'{remote}/out/adapt/{family}', out / 'adapt' / f'{family}.pull'
        shutil.rmtree(staged, ignore_errors=True)
        (staged / 'artifacts').mkdir(parents=True)
        for name in A.ssh_host(host, f'ls {shlex.quote(src)}/artifacts', timeout=60).split():
            A.pull_verified(f'{src}/artifacts/{name}', staged / 'artifacts' / name, host)
        A.pull_verified(f'{src}/result.json', staged / 'result.json', host)
        shutil.rmtree(out / 'adapt' / family, ignore_errors=True)
        staged.rename(out / 'adapt' / family)
    (out / 'colab.json').write_text(json.dumps({'host': host, 'tenant_commit': tcfg['commit'], 'remote': remote,
                                                'seconds': round(time.time() - started, 1)}, indent=1) + '\n')
    print(json.dumps({'event': 'bank home', 'out': str(out)}))
    return 0


def merge_release(args):
    """DIR's fit release (or --release) merged into its start checkpoint: OUT/checkpoint.pt (default DIR/fit/merged),
    a plain native checkpoint any engine serves without adapter slots, and OUT/merged.json (identity and how much of
    the delta survives the checkpoint's precision). The merge is the learner's (``lfm3_learner.merge_policy_state``:
    ``W = bf16(fp32(W0) + s * B @ A)``); a stack of several artifacts on one target is summed (B concatenated, scaled).
    Why: SGLang's adapter slots crash (CUDA illegal memory access) above 2 concurrent adapted requests and their parity is
    unverified (DECISIONS 2026-11-24 05:10 T3), so an adapter arm measured on slots is not trustworthy yet."""
    import torch
    from natlang_neuralese.train.checkpoint_policy import CheckpointPolicy
    from natlang_neuralese.train.learner_state import _write_merged
    from natlang_neuralese.train.lfm3_learner import merge_policy_state
    from natlang_neuralese.tune.stack_export import load_artifacts
    directory = Path(args.dir)
    release = Path(args.release) if args.release else directory / 'fit' / 'release'
    out = Path(args.out) if args.out else directory / 'fit' / 'merged'
    plan = json.loads((directory / 'fit' / 'plan.json').read_text())
    base_path = Path(plan['engine']['checkpoint'])
    base_sha = plan['manifest']['base']['checkpoint_sha256']
    result = json.loads((release / 'result.json').read_text())
    if result.get('softStates'):
        raise SystemExit(f'{release}: soft states do not merge into weights')
    artifacts = load_artifacts(release / 'artifacts')
    tensors, scales, specs = merged_factors(artifacts, [(item['artifact'], args.scale) for item in result['artifacts']])
    if sha256_file(base_path) != base_sha:
        raise SystemExit(f'{base_path} is not the plan\'s start checkpoint {base_sha[:12]}')
    state = CheckpointPolicy.load(base_path)
    backbone, interface, changed = merge_policy_state(state, tensors, scales)
    retained = {}
    original = state['backbone_state']
    for name in scales:
        delta = float(scales[name]) * (tensors[f'lora_B.{name}'].float() @ tensors[f'lora_A.{name}'].float())
        served = backbone[name].float() - original[name].to(torch.bfloat16).float()
        retained[name] = (float(delta.norm()), float(served.norm()),
                          float(torch.nn.functional.cosine_similarity(delta.flatten(), served.flatten(), dim=0)))
    norm = sum(d * d for d, _, _ in retained.values()) ** .5
    stats = {'targets': len(retained), 'delta_norm': norm,
             'served_over_delta_norm': (sum(s * s for _, s, _ in retained.values()) ** .5) / norm if norm else None,
             'min_cosine': min((c for _, _, c in retained.values()), default=None),
             'mean_cosine': sum(c for _, _, c in retained.values()) / max(1, len(retained)), 'changed_elements': changed}
    effective = canonical_json_sha256_hex({'base': base_sha, 'release': sha256_file(release / 'result.json'),
                                           'scale': args.scale})
    out.mkdir(parents=True, exist_ok=True)
    record = _write_merged(out / 'checkpoint.pt', state, backbone, interface, learner_step=int(state['step']),
                           adapter_spec={'kind': 'program-release-merge', 'specs': specs}, effective_sha=effective,
                           source={'training_checkpoint_sha256': base_sha,
                                   'training_checkpoint_bytes': base_path.stat().st_size,
                                   'release_result_sha256': sha256_file(release / 'result.json')})
    merged = {'schema': 'natlang.program-merged/1', 'release': str(release), 'base_checkpoint_sha256': base_sha,
              'scale': args.scale, **record, 'retention': stats}
    (out / 'merged.json').write_text(json.dumps(merged, indent=1) + '\n')
    print(json.dumps({'event': 'merged', 'dir': str(directory), 'checkpoint_sha256': record['checkpoint_sha256'],
                      **stats}))
    return 0


def merged_factors(artifacts, stack):
    """``merge_policy_state`` inputs for a runtime stack [(artifact id, scale)] of free-LoRA coefficient artifacts:
    ``lora_A.<backbone name>`` / ``lora_B.<backbone name>`` (several artifacts on one target concatenated along the
    rank, each B scaled by its stack scale x alpha / rank) and a scale of 1 per target; the native target names
    (``port.hf.<backbone name>``) map to the checkpoint's backbone names."""
    import torch
    parts, specs = {}, []
    for ident, scale in stack:
        record = artifacts.get(ident)
        if record is None or record.kind != 'adapter-coefficients':
            raise SystemExit(f'stack entry {ident} is not a coefficient artifact of the release')
        spec = record.provenance['adapter_spec']
        if spec['kind'] != 'free-lora':
            raise SystemExit(f'{ident}: only free-lora artifacts merge here, not {spec["kind"]}')
        specs.append({'artifact': ident, 'scale': scale, 'alpha': spec['alpha'], 'targets': len(spec['parameter_names'])})
        for name in spec['parameter_names']:
            B, A = record.tensors['lora_B.' + name].float(), record.tensors['lora_A.' + name].float()
            if B.dim() != 2:
                raise SystemExit(f'{ident}: {name} has batched factors {tuple(B.shape)}')
            key = name[len('port.hf.'):] if name.startswith('port.hf.') else name
            parts.setdefault(key, []).append((B * (float(scale) * float(spec['alpha']) / A.shape[0]), A))
    tensors, scales = {}, {}
    for key, items in parts.items():
        tensors[f'lora_B.{key}'] = torch.cat([b for b, _ in items], dim=1)
        tensors[f'lora_A.{key}'] = torch.cat([a for _, a in items], dim=0)
        scales[key] = 1.0
    return tensors, scales, specs


def colab_measure(args):
    """One held_eval arm on dev manifests against a Colab tenant actor: the start checkpoint (``--arm base``, every
    --dir on one actor) or it with DIR's fit release on an SGLang adapter slot (``--arm adapter``, one --dir). The
    episodes run on the DGX (a ledger unit at the experiment pin); the policy id comes from the actor's attested
    binding (``CHECKPOINT+stack digest`` for the adapter). Resumable; a lost (evicted) actor stops the measure, whose
    unscored episodes are left out, and the next call resumes."""
    import subprocess
    import time
    import urllib.request
    import pop_rl_actor as A
    dirs = [Path(d) for d in args.dir]
    if args.arm in ('adapter', 'bank') and len(dirs) != 1:
        raise SystemExit(f'--arm {args.arm} measures one program')
    if (args.arm == 'bank') != bool(args.release):
        raise SystemExit('--arm bank measures a --release (a bank-adapted program, colab-bank OUT/adapt/<family>)')
    if args.arm == 'merged' and len(dirs) != 1:
        raise SystemExit('--arm merged measures one program')
    plan = json.loads((dirs[0] / 'fit' / 'plan.json').read_text())
    sha, checkpoint = plan['manifest']['base']['checkpoint_sha256'], plan['engine']['checkpoint']
    if args.arm == 'merged':     # the release merged into the weights (`merge`): a plain checkpoint, no adapter slot
        merged = json.loads((dirs[0] / 'fit' / 'merged' / 'merged.json').read_text())
        sha, checkpoint = merged['checkpoint_sha256'], merged['path']
    release = (Path(args.release) if args.arm == 'bank' else
               dirs[0] / 'fit' / 'release' if args.arm == 'adapter' else None)
    record = A.serve_colab_tenant_actor(host=args.host, checkpoint=checkpoint, sha=sha,
                                        commit=args.commit, release=str(release) if release else None,
                                        beside=args.beside or None)
    pin = Path(args.pin).resolve()
    sys.path.insert(0, str(pin / 'training/neuralese'))
    from natlang_neuralese.rl.held_eval import policy_id_of
    info = json.load(urllib.request.urlopen(f'{record["endpoint"]}/v1/neuralese/info', timeout=30))
    policy = policy_id_of(info.get('rl_policy_binding') or {})
    results = {}
    for directory in dirs:
        out = directory / 'eval' / f'{args.arm}-{args.split}-{args.series}'
        out.mkdir(parents=True, exist_ok=True)
        unit = f'progdev-{directory.name[:40]}-{args.arm}-{int(time.time())}'
        env = (f'NATLANG_REPO={pin} PYTHONPATH={pin}/training/neuralese:{pin} HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 '
               f'MKL_NUM_THREADS=1')
        measure = (f'exec /usr/bin/env {env} {sys.executable} -m natlang_neuralese.rl.held_eval measure '
                   f'--manifest {directory}/{args.split}/manifest.json --endpoint {record["endpoint"]}={args.slots} '
                   f'--policy-id {policy} --top-p 0.95 --samples {args.samples} --out {out} '
                   f'--label {directory.name}-{args.arm}-{args.series} --allow-other-engine --max-model-requests 40 '
                   f'>> {out}/measure.log 2>&1')
        subprocess.run(['/usr/bin/python3', str(REPO / 'scripts/memory_ledger.py'), 'run', '--unit', unit, '--budget-gb',
                        str(args.budget_gb), '--priority', args.priority, '--wait', '3600', '--workdir', str(pin), '--',
                        '/bin/sh', '-c', measure], check=True)
        while subprocess.run(['systemctl', '--user', 'is-active', '-q', f'{unit}.service']).returncode == 0:
            time.sleep(30)
        result = json.loads((out / 'result.json').read_text()) if (out / 'result.json').exists() else {}
        if (directory / 'rows.jsonl').exists():
            parity = prompt_parity(directory, out)
            (out / 'prompt-parity.json').write_text(json.dumps(parity, indent=1) + '\n')
            if parity['share'] is not None and parity['share'] < .5:
                print(json.dumps({'event': 'prompt parity', 'program': directory.name, 'arm': args.arm,
                                  'warning': 'most fit rows were rendered with a system prompt this arm never saw; '
                                             'rebuild the rows from a replay at the arm pin',
                                  **{k: parity[k] for k in ('rows', 'rows_with_served_prompt', 'share')}}), flush=True)
        results[directory.name] = {k: result.get(k) for k in ('policy_id', 'complete', 'scored', 'successes',
                                                              'success_rate', 'samples_per_task')}
        print(json.dumps({'event': 'arm measured', 'program': directory.name, 'arm': args.arm, **results[directory.name]}),
              flush=True)
    return 0 if all(r.get('complete') for r in results.values()) else 3


# ---- Per-workflow on-policy binary GRPO + local eval + Layer-1 decision calibration ----------------------------------
def _resolve_dev_manifest(directory: Path, dev_split: str | None) -> Path:
    if dev_split:
        path = directory / dev_split / 'manifest.json'
        if not path.exists():
            raise SystemExit(f'{path}: dev manifest does not exist')
        return path
    for candidate in ('dev64', 'dev'):
        path = directory / candidate / 'manifest.json'
        if path.exists():
            return path
    raise SystemExit(f'{directory}: no dev64/manifest.json or dev/manifest.json found')


def grpo_program(args):
    """Per-workflow on-policy binary GRPO on one program's leak-free support split.

    Trains strictly on DIR/split.json['ids']['support'] (excluding DIR/dev, DIR/confirm, and all held-v3/v4 sets),
    evaluates periodically on DIR/<dev-split>/manifest.json, and exports compact Mode-P LoRA factors into
    WORK/factors. Zero paid teacher calls required.
    """
    import shlex
    import subprocess
    directory = Path(args.dir).resolve()
    record = json.loads((directory / 'split.json').read_text())
    support_ids = list(record['ids']['support'])
    if not support_ids:
        raise SystemExit(f'{directory}: empty support split')
    dev_manifest = _resolve_dev_manifest(directory, args.dev_split)
    confirm_manifest = directory / 'confirm' / 'manifest.json'
    work = Path(args.work).resolve() if args.work else (directory / 'grpo')
    work.mkdir(parents=True, exist_ok=True)
    task_ids_path = work / 'support-task-ids.json'
    task_ids_path.write_text(json.dumps(support_ids, indent=1) + '\n')
    pin = Path(args.pin).resolve() if args.pin else REPO
    sha = args.checkpoint_sha256 or sha256_file(args.checkpoint)
    py = str(pin / '.venv-neuralese' / 'bin' / 'python') if (pin / '.venv-neuralese' / 'bin' / 'python').exists() else sys.executable
    unit_prefix = args.unit_prefix or f'proggrpo-{record["family"][:28]}'
    cmd = [
        py, str(pin / 'scripts' / 'neuralese_data' / 'grpo.py'), 'train',
        '--work', str(work / 'run'),
        '--state-dir', str(work / 'state'),
        '--checkpoint', logical(args.checkpoint),
        '--checkpoint-sha256', sha,
        '--task-ids', str(task_ids_path),
        '--held-manifest', str(dev_manifest),
        '--reward', args.reward,
        '--pass-band', str(args.pass_band[0]), str(args.pass_band[1]),
        '--explore', str(args.explore),
        '--lr', str(args.lr),
        '--warmup', str(args.warmup),
        '--kl-beta', str(args.kl_beta),
        '--lora-rank', str(args.lora_rank),
        '--lr-reference-spread', str(args.lr_reference_spread),
        '--sd-weight', str(args.sd_weight),
        '--concise-pass-bonus', str(args.concise_pass_bonus),
        '--min-groups', str(args.min_groups),
        '--max-groups', str(args.max_groups),
        '--group', str(args.group),
        '--slots', str(args.slots),
        '--max-steps', str(args.max_steps),
        '--save-every', str(args.save_every),
        '--state-every', str(args.state_every),
        '--branch-states', str(args.branch_states),
        '--export-factors-dir', str(work / 'factors'),
        '--held-samples', str(args.held_samples),
        '--held-samples-start', str(args.held_samples),
        '--held-every-hours', str(args.held_every_hours),
        '--unit-prefix', unit_prefix,
    ]
    if confirm_manifest.exists():
        cmd.extend(['--exclude-manifest', str(confirm_manifest)])
    if args.pass_rates:
        cmd.extend(['--pass-rates', logical(args.pass_rates)])
    if args.held_remote and args.held_remote != 'none':
        cmd.extend(['--held-remote', args.held_remote])
    if args.held_strict_series:
        cmd.append('--held-strict-series')
    if getattr(args, 'neuralese_variants', None):
        cmd.extend([
            '--neuralese-variants', logical(args.neuralese_variants),
            '--neuralese-variant-share', str(args.neuralese_variant_share),
        ])
    if args.extra:
        cmd.extend(shlex.split(args.extra))
    env = {
        'NATLANG_REPO': str(pin),
        'PYTHONPATH': f'{pin / "training" / "neuralese"}:{pin}',
        'HF_HUB_OFFLINE': '1',
        'NATLANG_ALLOW_UNREGISTERED_DATA': '1',
    }
    plan_record = {
        'schema': 'natlang.program-grpo/1',
        'family': record['family'],
        'support_tasks': len(support_ids),
        'dev_manifest': str(dev_manifest),
        'confirm_excluded': str(confirm_manifest) if confirm_manifest.exists() else None,
        'checkpoint': logical(args.checkpoint),
        'checkpoint_sha256': sha,
        'pin': str(pin),
        'work': str(work),
        'env': env,
        'command': cmd,
    }
    (work / 'launch.json').write_text(json.dumps(plan_record, indent=1) + '\n')
    if args.dry_run:
        print(json.dumps(plan_record))
        return 0
    if args.ledger_unit:
        ledger_cmd = [
            '/usr/bin/python3', str(REPO / 'scripts' / 'memory_ledger.py'), 'run',
            '--unit', args.ledger_unit,
            '--budget-gb', str(args.budget_gb),
            '--priority', args.priority,
            '--class', 'experiment',
            '--resumable',
            '--workdir', str(pin),
        ]
        for k, v in env.items():
            ledger_cmd.extend(['--env', f'{k}={v}'])
        ledger_cmd.append('--')
        ledger_cmd.extend(cmd)
        subprocess.run(ledger_cmd, check=True)
        print(json.dumps({'event': 'launched', 'unit': args.ledger_unit, 'work': str(work)}))
        return 0
    full_env = {**dict(__import__('os').environ), **env}
    proc = subprocess.run(cmd, cwd=str(pin), env=full_env)
    return int(proc.returncode)


def _lower_subspace_rollout_turns(row: dict, stats, states=None) -> list[dict]:
    """Extract tokenized (prompt, sample, imports) turns from either real grpo.py rollouts (via lower_rollout_turns)
    or pre-tokenized {'prompt_ids', 'sample_ids'} test rows."""
    raw_turns = row.get('turns') or ()
    pre = [
        {'prompt': [int(x) for x in t['prompt_ids']], 'sample': [int(x) for x in t['sample_ids']], 'imports': []}
        for t in raw_turns
        if t.get('prompt_ids') and t.get('sample_ids')
    ]
    if pre:
        return pre
    from grpo import trainable_rows
    from lower_rollout_turns import episode_turns
    calls = episode_turns(row, stats, states) if raw_turns else []
    rows = trainable_rows(calls, stats)
    out = []
    for r in rows:
        imps = list(r.get('imports') or ())
        if imps and states is None:
            continue
        p = [int(x) for x in r['prompt'].tolist()]
        s = [int(x) for x in r['sample'].tolist()]
        if p and s:
            out.append({'prompt': p, 'sample': s, 'imports': imps})
    return out


def adapt_subspace_cmd(args):
    """Execution-grounded low-DOF subspace adaptation over a shared Mode-P LoRA basis (tune.low_dof).

    Loads a shared ``FactorSubspaceBasis`` extracted from verified workflow LoRAs, lowers real on-policy support
    rollouts (``rollouts.jsonl[.zst]`` via ``lower_rollout_turns.episode_turns``), runs O(1)-turn-VRAM
    KL-anchored advantage-weighted optimization gated by a mandatory held-out query split, and exports standard
    Mode-P LoRA factors (``adapter_factors_fp32.safetensors`` + ``policy_manifest.json``) when accepted.
    """
    import collections
    import torch
    from grpo import episode_training_reward
    from lower_rollout_turns import NativeStates
    from natlang_neuralese.rl.episode_files import read_rows
    from natlang_neuralese.train.lfm3_fast import hidden as lfm3_hidden
    from natlang_neuralese.train.lfm3_fast import import_tensors as lfm3_import_tensors
    from natlang_neuralese.train.lfm3_fast import linear_nll as lfm3_linear_nll
    from natlang_neuralese.train.lfm3_learner import AdapterSpec, LayerAdapterBundle, Lfm3Learner, LinearFactorLeaf
    from natlang_neuralese.tune.low_dof import (
        FactorSubspaceBasis,
        LowDofExecConfig,
        compose_subspace_factors,
        export_subspace_factors,
        fit_subspace_execution_accum,
    )

    basis = FactorSubspaceBasis.load(args.basis)
    states = NativeStates(args.store_dir) if getattr(args, 'store_dir', None) else None
    stats = collections.Counter()
    raw_rows = list(read_rows(Path(args.rollouts)))
    examples = []
    for idx, row in enumerate(raw_rows):
        rew = row.get('reward')
        if rew is None or row.get('unscored'):
            continue
        turns = _lower_subspace_rollout_turns(row, stats, states)
        if not turns:
            continue
        r_val = episode_training_reward(rew.get('quality', 0.0), rew.get('graded'), mode=args.reward)
        examples.append({
            'group': str(row.get('task_id') or f'task-{idx}'),
            'reward': float(r_val),
            'turns': turns,
        })
    if len(examples) < 2:
        raise SystemExit(f'{args.rollouts}: need >= 2 lowerable scored rollouts, got {len(examples)} (stats={dict(stats)})')

    dtype = getattr(torch, args.dtype)
    learner = Lfm3Learner.from_checkpoint(
        args.checkpoint, args.base, AdapterSpec.for_rl(rank=1, trainable_rows=False),
        device=args.device, dtype=dtype, compile_layers=False,
    )
    device = learner.device

    # Keep basis tensors resident on device once (eliminates per-turn CPU->GPU copies).
    basis = FactorSubspaceBasis(
        target_names=basis.target_names,
        left_u={k: v.to(device=device, dtype=torch.float32) for k, v in basis.left_u.items()},
        right_v={k: v.to(device=device, dtype=torch.float32) for k, v in basis.right_v.items()},
        components={k: v.to(device=device, dtype=torch.float32) for k, v in basis.components.items()},
        singular_values=basis.singular_values.to(device=device, dtype=torch.float32),
        donors=basis.donors,
    )

    def _turn_logp(turn: dict, adapters):
        p_ids = turn['prompt']
        s_ids = turn['sample']
        seq = torch.tensor(p_ids + s_ids, dtype=torch.long, device=device)
        positions = torch.arange(len(p_ids) - 1, len(p_ids) + len(s_ids) - 1, dtype=torch.long, device=device)
        targets = seq[len(p_ids):]
        imports = None
        if turn.get('imports') and states is not None:
            imports = lfm3_import_tensors(
                learner.model,
                [(pos, states(bid)) for pos, bid in turn['imports']],
                device=device,
            )
        h = lfm3_hidden(learner.model, seq, imports=imports, adapters=adapters, trainable_rows=False)
        return (-lfm3_linear_nll(h[positions], learner.model.lm_head.weight, targets, trainable_rows=False)).mean()

    with torch.no_grad():
        for ep in examples:
            ep['ref_lp'] = sum(float(_turn_logp(t, ()).item()) for t in ep['turns']) / len(ep['turns'])

    def _build_bundles(coeffs: torch.Tensor) -> tuple[LayerAdapterBundle, ...]:
        factors = compose_subspace_factors(basis, coeffs.to(device))
        per_layer: dict[int, dict[str, LinearFactorLeaf]] = {i: {} for i in range(len(learner.model.model.layers))}
        for param_name, (layer_idx, _kind, short_name) in learner.target_map.items():
            if param_name in factors:
                b, a = factors[param_name]
                per_layer[layer_idx][short_name] = LinearFactorLeaf(a=a, b=b, scale=1.0)
        return tuple(
            LayerAdapterBundle(layer_idx=idx, layer_kind=('attn' if layer.is_attention_layer else 'conv'), **per_layer[idx])
            for idx, layer in enumerate(learner.model.model.layers)
        )

    _eval_cache: dict[str, object] = {'key': None, 'bundles': None}

    def turn_logp_fn(coeffs: torch.Tensor, turn: dict):
        if not torch.is_grad_enabled():
            key = coeffs.detach().cpu().numpy().tobytes()
            if _eval_cache['key'] != key:
                _eval_cache['key'] = key
                _eval_cache['bundles'] = _build_bundles(coeffs)
            return _turn_logp(turn, _eval_cache['bundles'])
        return _turn_logp(turn, _build_bundles(coeffs))

    cfg = LowDofExecConfig(
        steps=int(args.steps),
        lr=float(args.lr),
        kl_beta=float(args.kl_beta),
        prox_lambda=float(args.prox_lambda),
        eval_every=int(args.eval_every),
        min_query_gain=float(args.min_query_gain),
        query_fraction=float(args.query_fraction),
    )
    res = fit_subspace_execution_accum(turn_logp_fn, examples, num_components=basis.num_components, cfg=cfg)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    exported = None
    if res.accepted:
        exported = str(export_subspace_factors(basis, res.coeffs_best, out))
    report = {
        'schema': 'natlang.program-adapt-subspace/1',
        'basis': str(Path(args.basis).resolve()),
        'rollouts': str(Path(args.rollouts).resolve()),
        'examples': len(examples),
        'num_components': basis.num_components,
        'accepted': bool(res.accepted),
        'best_step': int(res.best_step),
        'base_query_score': round(float(res.base_query_score), 6),
        'best_query_score': round(float(res.best_query_score), 6),
        'coeffs': [round(float(x), 8) for x in res.coeffs_best.detach().cpu().tolist()],
        'exported_factors': exported,
        'history': res.history,
    }
    (out / 'adapt-report.json').write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(report))
    return 0


def extract_basis_cmd(args):
    """Extract a shared low-DOF Mode-P LoRA subspace basis from verified workflow factor exports."""
    from natlang_neuralese.tune.low_dof import extract_factor_basis

    basis = extract_factor_basis(
        args.factors,
        max_rank=int(args.max_rank),
        max_components=int(args.max_components),
        include_family_slices=bool(getattr(args, 'family_slices', True)),
    )
    out = basis.save(args.out)
    report = {
        'schema': 'natlang.program-extract-basis/1',
        'out': str(out),
        'num_components': basis.num_components,
        'targets': len(basis.target_names),
        'donors': list(basis.donors),
        'singular_values': [round(float(v), 6) for v in basis.singular_values.tolist()],
    }
    print(json.dumps(report))
    return 0


def extract_grpo_decision_rows(work_dir: str | Path) -> list[dict]:
    """Extract calibration rows ``{"option_logits", "target", "labels"}`` from GRPO's ``WORK/decisions/step-*.jsonl``.

    Keeps positive-advantage (passing-episode) decision records where ``scores`` and ``chosen`` (or ``target``) are
    present so ``calibrate-decisions`` can run with zero extra GPU compute directly on logged GRPO traces.
    """
    dec_dir = Path(work_dir) / 'decisions'
    if not dec_dir.is_dir():
        return []
    rows: list[dict] = []
    for path in sorted(dec_dir.glob('step-*.jsonl')):
        for rec in read_jsonl(path):
            scores = rec.get('option_scores') or rec.get('scores') or rec.get('option_logits')
            if not isinstance(scores, list) or len(scores) < 2:
                continue
            # Prefer explicit target; otherwise use chosen option on positive-advantage/passing episodes.
            target = rec.get('target')
            if target is None:
                adv = rec.get('advantage')
                quality = rec.get('quality')
                if (quality is not None and float(quality) < 1.0) or (quality is None and adv is not None and float(adv) <= 0.0):
                    continue
                target = rec.get('chosen')
                if target is None:
                    target = rec.get('choice')
            if target is None:
                continue
            idx = int(target)
            if not (0 <= idx < len(scores)):
                continue
            labels = list(rec.get('labels') or rec.get('options') or [f'opt_{i}' for i in range(len(scores))])
            rows.append({'option_logits': [float(x) for x in scores], 'target': idx, 'labels': labels})
    return rows


def calibrate_decisions(args):
    """Layer-1 single-pass decision calibration (tune.decision_compile) for a workflow's readout: decision calls.

    Reads JSONL rows of ``{"option_logits": [float, ...], "target": int, "labels": [str, ...]}`` (or extracts them
    directly from ``--from-grpo-work WORK/decisions/step-*.jsonl``), fits scalar temperature T_f minimizing NLL,
    measures ECE & margin-gated fast-path settlement rate (0 decode tokens when confident), writes
    DIR/fit/decision-calibration.json, and optionally attaches ``decisionCalibration`` into a WeightManifest JSON.
    """
    import torch
    import torch.nn.functional as F
    from natlang_neuralese.tune.decision_compile import compute_ece, fit_temperature, should_settle_fast

    directory = Path(args.dir).resolve()
    family = json.loads((directory / 'split.json').read_text())['family'] if (directory / 'split.json').exists() else directory.name
    if getattr(args, 'from_grpo_work', None):
        fit_rows = extract_grpo_decision_rows(args.from_grpo_work)
    elif args.rows:
        fit_rows = read_jsonl(args.rows)
    else:
        raise SystemExit('calibrate-decisions requires either --rows or --from-grpo-work')
    if not fit_rows:
        raise SystemExit('no decision rows to calibrate')
    eval_rows = read_jsonl(args.eval_rows) if args.eval_rows else fit_rows

    def _to_tensors(items):
        logits = torch.tensor([row['option_logits'] for row in items], dtype=torch.float32)
        targets = torch.tensor([int(row['target']) for row in items], dtype=torch.long)
        labels = [list(row.get('labels') or [f'opt_{i}' for i in range(len(row['option_logits']))]) for row in items]
        return logits, targets, labels

    fit_logits, fit_targets, _ = _to_tensors(fit_rows)
    eval_logits, eval_targets, eval_labels = _to_tensors(eval_rows)
    cal = fit_temperature(fit_logits, fit_targets, min_temp=args.min_temp, max_temp=args.max_temp, steps=args.steps)
    risk = json.loads(args.risk_thresholds) if args.risk_thresholds else None

    with torch.no_grad():
        scaled = eval_logits / cal.temperature
        probs = F.softmax(scaled, dim=-1)
        eval_nll_before = float(F.cross_entropy(eval_logits, eval_targets).item())
        eval_nll_after = float(F.cross_entropy(scaled, eval_targets).item())
        eval_ece_before = compute_ece(F.softmax(eval_logits, dim=-1), eval_targets)
        eval_ece_after = compute_ece(probs, eval_targets)
        eval_acc = float((probs.argmax(dim=-1) == eval_targets).float().mean().item())
        settled_total, settled_correct = 0, 0
        for i in range(len(eval_rows)):
            settle, pred_label, _ = should_settle_fast(
                probs[i], eval_labels[i], fast_threshold=args.fast_threshold, risk_thresholds=risk,
            )
            if settle:
                settled_total += 1
                if pred_label == eval_labels[i][int(eval_targets[i].item())]:
                    settled_correct += 1

    out_path = Path(args.out) if args.out else (directory / 'fit' / 'decision-calibration.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        'schema': 'natlang.program-decision-calibration/1',
        'family': family,
        'fit_rows': len(fit_rows),
        'eval_rows': len(eval_rows),
        'temperature': round(cal.temperature, 6),
        'fast_threshold': float(args.fast_threshold),
        'risk_thresholds': risk,
        'fit': {
            'nll_before': round(cal.nll_before, 6),
            'nll_after': round(cal.nll_after, 6),
            'ece_before': round(cal.ece_before, 6),
            'ece_after': round(cal.ece_after, 6),
        },
        'eval': {
            'nll_before': round(eval_nll_before, 6),
            'nll_after': round(eval_nll_after, 6),
            'ece_before': round(eval_ece_before, 6),
            'ece_after': round(eval_ece_after, 6),
            'top1_accuracy': round(eval_acc, 6),
            'fast_settled_count': settled_total,
            'fast_settled_rate': round(settled_total / max(1, len(eval_rows)), 6),
            'fast_settled_accuracy': round(settled_correct / settled_total, 6) if settled_total else None,
        },
    }
    out_path.write_text(json.dumps(report, indent=1) + '\n')
    attach = getattr(args, 'attach_manifest', None)
    if attach:
        m_path = Path(attach)
        manifest = json.loads(m_path.read_text(encoding='utf-8'))
        cal_map = dict(manifest.get('decisionCalibration') or {})
        binding_key = getattr(args, 'binding_key', None) or family
        entry = {'temperature': round(cal.temperature, 6), 'fastThreshold': float(args.fast_threshold)}
        if risk:
            entry['riskThresholds'] = risk
        cal_map[binding_key] = entry
        manifest['decisionCalibration'] = cal_map
        # Reseal digest matching ts-host manifestDigest(rest, 'natlang.adapters/v1')
        rest = {k: v for k, v in manifest.items() if k != 'digest'}
        manifest['digest'] = hashlib.sha256(
            f'natlang.adapters/v1\n{json.dumps(rest, sort_keys=True, separators=(",", ":"))}'.encode('utf-8')
        ).hexdigest()
        m_path.write_text(json.dumps(manifest, indent=1) + '\n', encoding='utf-8')
        report['attached_manifest'] = str(m_path)
    print(json.dumps(report))
    return 0


# ---- teacher tasks ---------------------------------------------------------------------------------------------------
def teacher_tasks(args):
    """A `teacher_gen.py` WORK directory over programs' support splits only: WORK/tasks.jsonl (support rows of every
    --dir, round-robin over the programs, each in split-group hash order, at most --max-tasks per program) and
    WORK/pool.json (the splits' receipts, pools and held manifest). Teacher successes, replayed through
    `sft_sources.py` (extract --teacher WORK, replay), become fit rows (`rows --sources`). Dev and confirmation tasks
    never reach the teachers."""
    work = Path(args.work)
    if (work / 'tasks.jsonl').exists():
        raise SystemExit(f'{work}/tasks.jsonl exists: teacher_gen resumes it as it is')
    programs, queues = [], []
    for item in args.dir:
        directory = Path(item)
        record = json.loads((directory / 'split.json').read_text())
        support = read_jsonl(directory / 'support' / 'cases.jsonl')
        if {row['id'] for row in support} != set(record['ids']['support']):
            raise SystemExit(f'{directory}: the support manifest differs from split.json')
        support.sort(key=lambda row: hashlib.sha256(
            f"teacher:{record['family']}:{split_group(row)}:{row['id']}".encode()).hexdigest())
        support = support[:args.max_tasks] if args.max_tasks else support
        queues.append(support)
        programs.append({'split': str((directory / 'split.json').resolve()), 'sha256': sha256_file(directory / 'split.json'),
                         'family': record['family'], 'tasks': len(support), 'pools': record['pools'],
                         'held_manifest': record['held_manifest']})
    ordered = [queue[i] for i in range(max(map(len, queues))) for queue in queues if i < len(queue)]
    work.mkdir(parents=True, exist_ok=True)
    with open(work / 'tasks.jsonl', 'w', encoding='utf-8') as handle:
        for row in ordered:
            handle.write(json.dumps({'row': row, 'base_rank': 'support'}) + '\n')
    (work / 'pool.json').write_text(json.dumps({'schema': 'natlang.program-support-teacher-pool/1', 'split': 'support',
                                                'programs': programs, 'tasks': len(ordered)}, indent=1) + '\n')
    print(json.dumps({'work': str(work), 'tasks': len(ordered), 'programs': {p['family']: p['tasks'] for p in programs}}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    s = sub.add_parser('split')
    s.add_argument('--family', required=True)
    s.add_argument('--out', required=True)
    s.add_argument('--pool', action='append', help='task pools (default: the registry role grpo-pool)')
    s.add_argument('--held-manifest', help='default: the registry role held-eval (held-v3; its exclusion covers held-v2)')
    s.add_argument('--errata', action='append', help='errata JSONL (default every grpo.ERRATA/*.jsonl)')
    s.add_argument('--exposed', action='append',
                   help='JSON list of task ids the start policy trained on (a later parent than the base): kept out')
    s.add_argument('--fractions', type=float, nargs=3, default=FRACTIONS, metavar=('SUPPORT', 'DEV', 'CONFIRM'))
    s.add_argument('--max-tasks', type=int, help='cap dev/confirmation manifests at this many tasks (in pool order)')
    r = sub.add_parser('rows')
    r.add_argument('--dir', required=True)
    r.add_argument('--sources', action='append', help='an sft_sources.py work directory (extracted and replayed)')
    r.add_argument('--admit-drifted', choices=('on', 'off'), default='on',
                   help='teacher successes whose replay display drifted (sft_sources.drifted_success) join the exact ones')
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
                   help='target classes (default: all; training applies the adapter factored, NATLANG_FACTORED_LORA)')
    p.add_argument('--lr', type=float, default=DEFAULT_SFT_LR,
                   help=f'LoRA Adam learning rate (default {DEFAULT_SFT_LR}; values > {MAX_SAFE_SFT_LR} require --allow-high-lr)')
    p.add_argument('--allow-high-lr', action='store_true',
                   help=f'permit --lr > {MAX_SAFE_SFT_LR} (unsafe on LFM3-1.7B free-LoRA)')
    p.add_argument('--depth-penalty', type=float, default=DEFAULT_DEPTH_PENALTY,
                   help=f'L2 anchor penalty on LoRA weights (default {DEFAULT_DEPTH_PENALTY})')
    p.add_argument('--min-rows', type=int, default=0, help='refuse to write a plan below this many support rows')
    p.add_argument('--min-tasks', type=int, default=0, help='refuse to write a plan below this many distinct support tasks')
    p.add_argument('--strict-support', action='store_true',
                   help='enforce safe production floors (--min-rows >= 32, --min-tasks >= 8)')
    p.add_argument('--max-epochs', type=float, default=None,
                   help='cap --steps to at most this many epochs over rows.jsonl (prevents tiny-set memorization)')
    p.add_argument('--batch-rows', type=int, default=8)
    p.add_argument('--epochs', type=float)
    p.add_argument('--steps', type=int, default=1)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--gguf-base-sha256', help='sha256 of the base GGUF the LoRA exports bind to')
    p.add_argument('--hf-config', help='HF config directory of the base for the GGUF LoRA converter')
    gp = sub.add_parser('grpo', help='per-workflow on-policy binary GRPO on DIR/support with dev evaluation')
    gp.add_argument('--dir', required=True)
    gp.add_argument('--checkpoint', required=True, help='start policy (native checkpoint)')
    gp.add_argument('--checkpoint-sha256')
    gp.add_argument('--pin', help='pinned repo checkout (default: current repo)')
    gp.add_argument('--work', help='GRPO work directory (default: DIR/grpo)')
    gp.add_argument('--dev-split', help='dev manifest directory name (default: dev64 if present else dev)')
    gp.add_argument('--pass-rates', help='optional measured pass-rates JSON to seed task selection')
    gp.add_argument('--reward', choices=('binary', 'graded', 'bounded-graded'), default='bounded-graded')
    gp.add_argument('--pass-band', type=float, nargs=2, default=[0.0, 0.99])
    gp.add_argument('--explore', type=float, default=0.5)
    gp.add_argument('--lr', type=float, default=1e-4)
    gp.add_argument('--warmup', type=int, default=2)
    gp.add_argument('--kl-beta', type=float, default=0.01)
    gp.add_argument('--lora-rank', type=int, default=16)
    gp.add_argument('--lr-reference-spread', type=int, default=6)
    gp.add_argument('--sd-weight', type=float, default=0.1)
    gp.add_argument('--concise-pass-bonus', type=float, default=0.05)
    gp.add_argument('--min-groups', type=int, default=12)
    gp.add_argument('--max-groups', type=int, default=24)
    gp.add_argument('--group', type=int, default=8)
    gp.add_argument('--slots', type=int, default=48)
    gp.add_argument('--max-steps', type=int, default=20)
    gp.add_argument('--save-every', type=int, default=5)
    gp.add_argument('--state-every', type=int, default=5)
    gp.add_argument('--branch-states', type=int, default=2)
    gp.add_argument('--held-remote', choices=('none', 'pop', 'colab_g4'), default='none')
    gp.add_argument('--held-samples', type=int, default=2)
    gp.add_argument('--held-every-hours', type=float, default=0.5)
    gp.add_argument('--held-strict-series', action='store_true')
    gp.add_argument('--neuralese-variants', help='optional neuralese_variants.py JSONL so GRPO exercises producer->consumer Neuralese turns')
    gp.add_argument('--neuralese-variant-share', type=float, default=0.25,
                    help='fraction of groups using the Neuralese variant when --neuralese-variants is set (default 0.25)')
    gp.add_argument('--unit-prefix')
    gp.add_argument('--ledger-unit', help='if set, launch through scripts/memory_ledger.py under this systemd unit name')
    gp.add_argument('--budget-gb', type=float, default=85.0)
    gp.add_argument('--priority', choices=('low', 'medium', 'high'), default='high')
    gp.add_argument('--extra', default='', help='extra CLI flags forwarded verbatim to grpo.py train')
    gp.add_argument('--dry-run', action='store_true', help='write support-task-ids.json and launch.json without running')
    eb = sub.add_parser('extract-basis', help='extract a low-DOF shared Mode-P LoRA subspace basis from workflow factor exports')
    eb.add_argument('--factors', nargs='+', required=True, help='one or more verified factor directories (adapter_factors_fp32.safetensors)')
    eb.add_argument('--out', required=True, help='output directory for subspace_basis_fp32.safetensors + subspace_basis.json')
    eb.add_argument('--max-rank', type=int, default=16)
    eb.add_argument('--max-components', type=int, default=32)
    eb.add_argument('--no-family-slices', dest='family_slices', action='store_false', default=True,
                    help='disable per-family x layer-band basis expansion (use donor SVD directions only)')
    asub = sub.add_parser('adapt-subspace', help='execution-grounded low-DOF adaptation over a shared Mode-P LoRA subspace')
    asub.add_argument('--basis', required=True, help='FactorSubspaceBasis directory from extract-basis')
    asub.add_argument('--rollouts', required=True, help='scored on-policy rollouts JSONL/zst')
    asub.add_argument('--store-dir', help='optional actor native block store directory for lowering native_state turns')
    asub.add_argument('--checkpoint', required=True, help='start policy native checkpoint')
    asub.add_argument('--base', required=True, help='HF base model directory')
    asub.add_argument('--out', required=True, help='output directory for adapt-report.json and collapsed Mode-P LoRA factors')
    asub.add_argument('--reward', choices=('binary', 'graded', 'bounded-graded'), default='bounded-graded')
    asub.add_argument('--steps', type=int, default=24)
    asub.add_argument('--lr', type=float, default=0.05)
    asub.add_argument('--kl-beta', type=float, default=0.01)
    asub.add_argument('--prox-lambda', type=float, default=0.05)
    asub.add_argument('--eval-every', type=int, default=4)
    asub.add_argument('--min-query-gain', type=float, default=1e-4)
    asub.add_argument('--query-fraction', type=float, default=0.25)
    asub.add_argument('--device', default='cuda')
    asub.add_argument('--dtype', default='bfloat16')
    cd = sub.add_parser('calibrate-decisions', help='Layer-1 single-pass decision temperature calibration & fast-path gate')
    cd.add_argument('--dir', required=True)
    cd.add_argument('--rows', help='JSONL of {"option_logits", "target", "labels"} for fitting T_f')
    cd.add_argument('--from-grpo-work', help='extract decision rows directly from GRPO WORK/decisions/step-*.jsonl')
    cd.add_argument('--eval-rows', help='optional held-out JSONL to measure calibrated ECE & fast-path settlement')
    cd.add_argument('--out', help='output JSON report path (default: DIR/fit/decision-calibration.json)')
    cd.add_argument('--attach-manifest', help='optional natlang.adapters.json path to attach decisionCalibration into')
    cd.add_argument('--binding-key', help='definition id/name key for --attach-manifest (default: program family)')
    cd.add_argument('--fast-threshold', type=float, default=0.85)
    cd.add_argument('--risk-thresholds', help='optional JSON map of {label: floor} forcing fallback')
    cd.add_argument('--min-temp', type=float, default=0.1)
    cd.add_argument('--max-temp', type=float, default=10.0)
    cd.add_argument('--steps', type=int, default=80)
    v = sub.add_parser('release', help='DIR/fit/release: the fit\'s result and only its artifact closure (to serve)')
    v.add_argument('--dir', required=True)
    t = sub.add_parser('teacher-tasks', help='a teacher_gen.py WORK over the support split only')
    t.add_argument('--dir', action='append', required=True, help='program directory (repeatable)')
    t.add_argument('--work', required=True)
    t.add_argument('--max-tasks', type=int, help='at most this many support tasks per program (stable hash order)')
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
    b.add_argument('--max-worlds', type=int, default=64, help='support tasks per program lowered into the pool')
    b.add_argument('--held-support-rows', type=int, default=64,
                   help='support rows a held program is adapted from (the bank inner fit; recorded in the release)')
    b.add_argument('--seed', type=int, default=0)
    cf = sub.add_parser('colab-fit', help='a fit as a tenant of the Colab session (yields to rounds; resumable)')
    cf.add_argument('--dir', required=True)
    cf.add_argument('--host', default=None)
    cf.add_argument('--beside', default='actor-',
                        help='tenant stands beside a live claim with this label prefix; \'\' on a session of its own (the L4)')
    cf.add_argument('--commit', required=True, help='the tenant checkout commit (pushed to origin)')
    cf.add_argument('--mps-percent', type=int, help='join MPS with this SM share (default: outside MPS, its own context)')
    cf.add_argument('--take-over', action='store_true', help='fit on Colab although a local fit state exists')
    cf.add_argument('--vram-gib', type=float, help='the fit\'s admission claim (default: colab_layout tenants fit)')
    cf.add_argument('--segment-steps', type=int, default=0,
                    help='tune.build --segment-steps: end each process after N steps and rerun it (bounds its growth)')
    cm = sub.add_parser('colab-measure', help='one dev arm against a Colab tenant actor (held_eval on the DGX)')
    cm.add_argument('--dir', action='append', required=True)
    cm.add_argument('--arm', choices=('base', 'adapter', 'bank', 'merged'), required=True,
                    help='merged: DIR/fit/merged (`merge`) served as a plain checkpoint')
    cm.add_argument('--release', help='--arm bank: the bank-adapted release directory')
    cm.add_argument('--host', default=None)
    cm.add_argument('--beside', default='actor-',
                        help='tenant stands beside a live claim with this label prefix; \'\' on a session of its own (the L4)')
    cm.add_argument('--commit', required=True)
    cm.add_argument('--pin', required=True, help='experiment pin (scripts/pin_checkout.py) the episodes run from')
    cm.add_argument('--split', default='dev', help='the manifest directory measured (dev, or a subset such as dev64)')
    cm.add_argument('--series', default='a100', help='result directory suffix: DIR/eval/<arm>-<split>-<series>')
    cb = sub.add_parser('colab-bank', help='bank meta-learning as a tenant of the Colab session; held releases home')
    cb.add_argument('--programs', nargs='+', required=True)
    cb.add_argument('--held', nargs='+', required=True)
    cb.add_argument('--out', required=True)
    cb.add_argument('--host', default=None)
    cb.add_argument('--beside', default='actor-',
                        help='tenant stands beside a live claim with this label prefix; \'\' on a session of its own (the L4)')
    cb.add_argument('--commit', required=True)
    cb.add_argument('--steps', type=int, default=0)
    cb.add_argument('--max-worlds', type=int, default=64)
    cb.add_argument('--held-support-rows', type=int, default=64)
    cb.add_argument('--recipe')
    cb.add_argument('--vram-gib', type=float, default=14.)
    cb.add_argument('--host-ram-gib', type=float, default=24.)
    ss = sub.add_parser('subset', help='a stable-hash subset of a split as its own pinned manifest (DIR/<split><tasks>)')
    ss.add_argument('--dir', required=True)
    ss.add_argument('--split', default='dev')
    ss.add_argument('--tasks', type=int, required=True)
    ss.add_argument('--name')
    mg = sub.add_parser('merge', help='a fit release merged into its start checkpoint (DIR/fit/merged), served plain')
    mg.add_argument('--dir', required=True)
    mg.add_argument('--release', help='default DIR/fit/release')
    mg.add_argument('--out', help='default DIR/fit/merged')
    mg.add_argument('--scale', type=float, default=1.0, help='stack scale of every artifact (serving default 1.0)')
    cm.add_argument('--samples', type=int, default=2)
    cm.add_argument('--slots', type=int, default=32)
    cm.add_argument('--budget-gb', type=float, default=12.)
    cm.add_argument('--priority', default='low')
    args = parser.parse_args(argv)
    status = {'split': split, 'rows': rows, 'plan': plan, 'release': release, 'teacher-tasks': teacher_tasks,
              'bank': bank, 'colab-fit': colab_fit, 'colab-measure': colab_measure, 'colab-bank': colab_bank,
              'subset': subset, 'merge': merge_release, 'grpo': grpo_program,
              'extract-basis': extract_basis_cmd,
              'adapt-subspace': adapt_subspace_cmd,
              'calibrate-decisions': calibrate_decisions}[args.command](args)
    return status if isinstance(status, int) else 0


if __name__ == '__main__':
    sys.exit(main() or 0)
