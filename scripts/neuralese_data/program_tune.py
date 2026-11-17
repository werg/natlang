#!/usr/bin/env python3
"""Per-program LoRA tuning on the LFM3 stack (plans/neuralese/TUNING_RESTORE_20261117.md).

A program is one task family of the GRPO pools: one typed function, many inputs. Its pool rows (held-v3/v2 closure and
pool errata removed) are split by split group into support / dev / confirmation; the adapter is fitted only from
verified successes of support tasks and judged by task outcomes on dev (held_eval, paired with the unadapted policy).

Stages (each reads the previous stage's output in DIR):

  split  pool rows of --family -> DIR/split.json (ids per split, pool digests) and two held_eval manifests,
         DIR/dev/manifest.json and DIR/confirm/manifest.json, with the held set's engine pin (engine.json).
  rows   replay-verified successes of support tasks (an `sft_sources.py` work directory: episodes*.jsonl.zst,
         replay.jsonl and replay/<name>/rollouts.jsonl, the current runtime's record of each episode) -> one
         natlang.tune-turn/v1 row per model call in DIR/rows.jsonl. A tool call whose result is a runtime error is not a
         target (`rejected`): its turn is dropped and later turns keep it as failed history (prompt weight 0 anyway).
  plan   DIR/fit/plan.json for `python -m natlang_neuralese.tune.build --plan DIR/fit/plan.json --out DIR/fit/build`:
         one scope node (the whole program), free LoRA of --rank over the LFM3 target table, frozen base in BF16 with
         FP32 adapter parameters, optional GGUF export (--gguf-base-sha256, --hf-config).

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


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


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


# ---- split -----------------------------------------------------------------------------------------------------------
def split(args):
    from natlang_neuralese.rl.held_eval import exclusion_ids, write_manifest, MANIFEST_SCHEMA
    from natlang_neuralese.rl.curriculum import metadata
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
    engine = Path(held).parent / 'engine.json'
    for name in ('dev', 'confirm'):
        items = parts[name][:args.max_tasks] if args.max_tasks else parts[name]
        if not items:
            raise SystemExit(f'the {name} split is empty')
        manifest = {'schema': MANIFEST_SCHEMA, 'seed': f'program-split:{args.family}', 'included': 0,
                    'program': {'family': args.family, 'split': name},
                    'tasks': [{'id': row['id'], 'pool_row': index, 'family': metadata(row)['family'],
                               'level': metadata(row)['level'], 'split_group': split_group(row),
                               'row_sha256': hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()}
                              for _, index, row in items],
                    'excluded_split_groups': sorted({split_group(row) for _, _, row in items}),
                    # held_eval.exclusion_ids reads one pool: the family's rows all come from these pools
                    'pool': {'path': str(items[0][0]), 'sha256': sha256_file(items[0][0]),
                             'rows': sum(1 for _ in open(items[0][0]))}}
        write_manifest(out / name, manifest, [row for _, _, row in items])
        if engine.exists():
            shutil.copyfile(engine, out / name / 'engine.json')
        record[f'{name}_manifest'] = {'path': str(out / name / 'manifest.json'), 'tasks': len(items)}
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


def rows(args):
    from sft_sources import zstd_lines
    directory = Path(args.dir)
    record = json.loads((directory / 'split.json').read_text())
    support = set(record['ids']['support'])
    work = Path(args.sources)
    replays = {}
    for line in (work / 'replay.jsonl').open():
        if line.strip():
            item = json.loads(line)
            if 'key' in item:
                replays[item['key']] = item
    stats = {'episodes': 0, 'not_support': 0, 'not_replayed': 0, 'replay_failed': 0, 'turns': 0, 'rows': 0,
             'rejected_turns': 0, 'unverified_turns': 0, 'tasks': set(), 'by_source': {}}
    out = []
    paths = sorted(work.glob('episodes*.jsonl.zst'))
    for path in paths:
        for line in zstd_lines(path):
            episode = json.loads(line)
            if episode.get('family') != record['family']:
                continue
            stats['episodes'] += 1
            if episode['task_id'] not in support:
                stats['not_support'] += 1
                continue
            replay = replays.get(episode['key'])
            if replay is None:
                stats['not_replayed'] += 1
                continue
            if replay.get('outcome') != 'task_success' or not replay.get('exact'):
                stats['replay_failed'] += 1
                continue
            replayed = read_jsonl(work / 'replay' / replay['name'] / 'rollouts.jsonl')[0]
            made, dispositions = episode_turns(episode['task_id'], episode['key'], replayed['turns'])
            stats['turns'] += len(dispositions)
            stats['rejected_turns'] += dispositions.count('rejected')
            stats['unverified_turns'] += dispositions.count('unverified')
            stats['rows'] += len(made)
            stats['tasks'].add(episode['task_id'])
            stats['by_source'][episode['source']] = stats['by_source'].get(episode['source'], 0) + 1
            out.extend(made)
    stats['tasks'] = len(stats['tasks'])
    stats['support_tasks'] = len(support)
    stats['sources'] = [{'path': str(p), 'sha256': sha256_file(p)} for p in paths]
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
    r.add_argument('--sources', required=True, help='an sft_sources.py work directory (extracted and replayed)')
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
    args = parser.parse_args(argv)
    {'split': split, 'rows': rows, 'plan': plan}[args.command](args)


if __name__ == '__main__':
    main()
