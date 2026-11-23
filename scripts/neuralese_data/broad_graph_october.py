#!/usr/bin/env python3
"""FALLOW-22 broad-graph trajectories -> October per-call teacher rows for ``sft_sources.py extract --october-teacher``.

The fallow-22 whole-chain Neuralese graphs (``/mnt/external/natlang-fallow22-20261110/<source>-s<N>``: fetaqa, 2wiki,
bird) are token graphs on an old surface (a 33.4k-character system prompt) with native-write spans inside producer
calls, a form today's runtime does not produce (NEURALESE_PASSTHROUGH_20261117.md). Their source records
(``shard-*.records.jsonl`` under each shard's receipt ``inputs.work``) keep every call's original plain messages and
reply (``native_source_conversion.source_messages`` / ``source_target``: the trajectories ran without a Neuralese
server; the spans were a post-hoc conversion) and the task program (``task.program_ir``). This writes them in the
October row form the extract stage turns into ``teacher`` episodes, so ``sft_sources.py replay`` runs each task again
on today's runtime against the recorded replies and the build renders today's prompts. A producer's result reaches its
consumer as text in the consumer's own context, which is how today's runtime reads a text-written state too
(in-context-text, d16437be8/f109bedfd/d6b3218e7).

Selection: the train graphs each shard kept (its ``ids-*heldfiltered.txt``: train split, the old held filter), less
trajectories whose task id, source ids or source groups are in ``--exclude-ids`` / ``--exclude-groups`` (held-v3/v4
verify hits and their siblings). The synthetic ``action-notes`` reasoning ("I run the next step in eval.") is dropped:
no reasoning is better than a constant note.

    broad_graph_october.py --fallow /mnt/external/natlang-fallow22-20261110 --out DIR
      -> DIR/october/fallow22-broad-graphs.jsonl.zst (per-call rows), DIR/sources/tasks.jsonl.zst (one task row per
         trajectory, the rows' ``source_file``), DIR/conversion.json, DIR/avoid-ids.json, DIR/avoid-groups.json (for ``held_eval verify``)
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]

GROUP = 'fallow22-broad-graphs'
MODEL = 'port-record-reference'


def shard_dirs(fallow):
    return sorted(d for d in Path(fallow).iterdir() if d.is_dir() and (d / 'receipt.json').exists()
                  and (d / 'graphs-train.jsonl.zst').exists())


def kept_trajectories(shard):
    """Trajectory ids of the shard's held-filtered train graphs (``chain:<source>:<trajectory>``)."""
    lists = sorted(shard.glob('ids-graphs-heldfiltered.txt')) or sorted(shard.glob('ids-heldfiltered.txt'))
    out = set()
    for path in lists:
        for line in path.read_text().splitlines():
            line = line.strip()
            if line:
                out.add(line.split(':', 2)[2] if line.startswith('chain:') else line)
    return out


def record_files(shard):
    work = Path(json.loads((shard / 'receipt.json').read_text())['source_binding']['inputs']['work'])
    return sorted(work.glob('shard-*.records.jsonl'))


def plain_call(record):
    """(messages, tools, target) of a record as the call originally ran: the pre-conversion messages and reply, the
    tools offered; the synthetic reasoning note left out."""
    conversion = record.get('native_source_conversion') or {}
    messages = conversion.get('source_messages') or record['messages']
    target = dict(conversion.get('source_target') or record['target'])
    target.pop('reasoning_content', None)
    target.pop('reasoning', None)
    return messages, record['tools'], target


def october_row(record, source_file):
    messages, tools, target = plain_call(record)
    program = record['task']['program_ir']
    ref = record['source_ref']
    return {'source': GROUP, 'id': record['id'], 'task_id': program['id'],
            'family': program.get('family') or record.get('task_family'), 'domain': 'other',
            'source_ids': list(program.get('source_ids') or record.get('source_ids') or ()),
            'source_groups': list(program.get('source_groups') or record.get('source_groups') or ()),
            'messages': messages, 'tools': tools, 'target': target, 'rerendered': False,
            'provenance': {'source_file': str(source_file), 'model': (record.get('provenance') or {}).get('model', MODEL),
                           'collection_role': 'reference', 'generator': 'natlang.broad_graph_calls/1',
                           'trajectory': ref['trajectory_id'], 'invocation': ref.get('invocation_id'),
                           'target_tokens': None}}


def excluded(program, ids, groups):
    return (program['id'] in ids or any(s in ids for s in program.get('source_ids') or ())
            or any(g in groups for g in program.get('source_groups') or ())
            or (program.get('curriculum') or {}).get('split_group') in groups)


def read_list(paths):
    """Ids or groups from files: a JSON list, else one per line."""
    out = set()
    for path in paths or ():
        text = Path(path).read_text()
        out |= set(json.loads(text)) if text.lstrip().startswith('[') else \
            {line.strip() for line in text.splitlines() if line.strip()}
    return out


def convert(fallow, out, exclude_ids=(), exclude_groups=()):
    from sft_sources import ZstdWriter
    out = Path(out)
    (out / 'october').mkdir(parents=True, exist_ok=True)
    (out / 'sources').mkdir(parents=True, exist_ok=True)
    ids, groups = read_list(exclude_ids), read_list(exclude_groups)
    source_file = out / 'sources' / 'tasks.jsonl.zst'
    rows_out = ZstdWriter(out / 'october' / f'{GROUP}.jsonl.zst')
    tasks_out = ZstdWriter(source_file)
    c = collections.Counter()
    per_shard, seen = {}, set()
    avoid_ids, avoid_groups = set(), set()
    for shard in shard_dirs(fallow):
        wanted = kept_trajectories(shard) - seen
        by_trajectory = collections.defaultdict(list)
        for path in record_files(shard):
            for line in path.open():
                if not line.strip():
                    continue
                record = json.loads(line)
                trajectory = record['source_ref']['trajectory_id']
                if trajectory in wanted:
                    by_trajectory[trajectory].append(record)
        s = collections.Counter(wanted=len(wanted), found=len(by_trajectory))
        for trajectory, records in sorted(by_trajectory.items()):
            records.sort(key=lambda r: int(r['id'].rsplit(':', 1)[1]))
            program = records[0]['task']['program_ir']
            if program.get('split') != 'train' or any(r.get('split') != 'train' for r in records):
                s['not_train'] += 1
                continue
            if excluded(program, ids, groups):
                s['held_excluded'] += 1
                continue
            if [int(r['id'].rsplit(':', 1)[1]) for r in records] != list(range(len(records))):
                s['decision_gaps'] += 1
                continue
            tasks_out.write({'source_ref': {'trajectory_id': trajectory}, 'task': {'program_ir': program}})
            for record in records:
                rows_out.write(october_row(record, source_file))
                s['calls'] += 1
            s['trajectories'] += 1
            seen.add(trajectory)
            avoid_ids.add(program['id'])
            avoid_ids.update(program.get('source_ids') or ())
            avoid_groups.update(program.get('source_groups') or ())
            if (program.get('curriculum') or {}).get('split_group'):
                avoid_groups.add(program['curriculum']['split_group'])
        per_shard[shard.name] = dict(s)
        c.update(s)
    rows_out.close()
    tasks_out.close()
    # JSON lists: ``held_eval verify --avoid-ids/--avoid-groups`` read them as they are
    (out / 'avoid-ids.json').write_text(json.dumps(sorted(avoid_ids)) + '\n')
    (out / 'avoid-groups.json').write_text(json.dumps(sorted(avoid_groups)) + '\n')
    report = {'schema': 'natlang.fallow22-october/1', 'fallow': str(fallow), 'group': GROUP, 'totals': dict(c),
              'shards': per_shard, 'excluded_ids': len(ids), 'excluded_groups': len(groups)}
    (out / 'conversion.json').write_text(json.dumps(report, indent=1) + '\n')
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--fallow', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--exclude-ids', action='append', default=[], help='file of task/source ids to leave out')
    p.add_argument('--exclude-groups', action='append', default=[], help='file of source/split groups to leave out')
    a = p.parse_args(argv)
    print(json.dumps(convert(a.fallow, a.out, a.exclude_ids, a.exclude_groups)['totals'], indent=1))


if __name__ == '__main__':
    main()
