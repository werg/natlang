"""compact-history cases from SWE trajectories (COMPACTION_OPERATOR_CONVERTERS.md section 5).

An earlier stretch of a coding session is the operator's source; the task statement and the most recent assistant
turns with their observations are the consumer's context; the next assistant message is the gold target. Replaces the
S1-era `swe_compaction.py` builder: its natural-boundary windows already live in the S1 closed records, and the
upstream message helpers are kept here for the bgkit store.

Sources (one CaseSink file each, spec section 5):
  s1-swe                S1 `trajectory_continuation_swe` records labelled `checked` by upstream SWE-rebench tests.
  bgkit-compaction-v2   bgkit `compaction_v2` windows re-materialised from the upstream trajectories they index.

Selection is two pass and streaming: pass 1 keeps the best `stable_rank` candidates per split as file offsets,
pass 2 seeks to them. At most MAX_WINDOWS windows per trajectory overall (across both sources).
"""
from __future__ import annotations

import argparse
import heapq
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neuralese_data.operator_cases_common import (  # noqa: E402
    OUT_ROOT, RULES, CaseSink, data_path, source_text, stable_rank)
from neuralese_data.records import VERSION, exact_refs_from, group_key, license_, seal, strip_markup  # noqa: E402

OPERATOR = 'compact-history'
S1_FILE = data_path('data_hdd', 'natlang-development-data', 'data', 'neuralese', 'port-records', 'full-20261003',
                    'resume-20261003T1948-review1', 'final', 'trajectory_continuation_swe.port-records.jsonl')
TRAJECTORIES_EXT = data_path('data_hdd', 'bgkit-data', 'trajectories_ext')
BENCH_KEYS = TRAJECTORIES_EXT / 'bench_dedupe' / 'bench_keys.json'
S1_NAME = 's1-swe'
TRAIN_CAP, TEST_CAP = 15_000, 1_000
MAX_WINDOWS = 2        # windows per trajectory overall
OVERSAMPLE = 4         # candidates kept beyond the cap: trajectory cap, bench drops and dedupe rejects consume some
VERIFIED = 'swe-rebench-tests'
CACHE_WINDOW = 1 << 28  # bytes read between page-cache releases


def load_bench_instances(path: Path = BENCH_KEYS) -> set:
    """Benchmark instance ids (SWE-bench and relatives) that must stay out of training."""
    if not path.exists():
        return set()
    keys = json.loads(path.read_text()).get('keys', [])
    return {k[1] for k in keys if isinstance(k, list) and len(k) >= 2 and k[0] == 'id'}


def split_of(record: dict) -> str:
    return 'test' if record['split'] in ('test', 'validation') else 'train'


class Pool:
    """Bounded best-ranked candidate set per split: (rank, offset, trajectory) with no record bodies."""

    def __init__(self, source: str, caps: dict):
        self.source, self.caps = source, caps
        self.heaps = {'train': [], 'test': []}
        self.eligible = Counter()

    def add(self, split: str, record_id: str, offset, trajectory: str):
        self.eligible[split] += 1
        heap, keep = self.heaps[split], self.caps[split] * OVERSAMPLE
        item = (-stable_rank(record_id, self.source), offset, trajectory)
        if len(heap) < keep:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)

    def ranked(self, split: str) -> list:
        return sorted((-negative, offset, trajectory) for negative, offset, trajectory in self.heaps[split])


def release_cache(*streams):
    """Flush and drop the page cache of bulk streams so a multi-GB pass stays inside the unit's memory budget."""
    for stream in streams:
        if hasattr(stream, 'flush') and not stream.closed and 'w' in getattr(stream, 'mode', ''):
            stream.flush()
            os.fsync(stream.fileno())
        os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)


def take_ranked(pool: Pool, windows: Counter, sink: CaseSink, build, label: str, taken: set,
                streams=()) -> Counter:
    """Offer ranked candidates until each split cap is met; the per-trajectory window cap spans both sources."""
    notes, offered = Counter(), 0
    for split in ('train', 'test'):
        for _, offset, trajectory in pool.ranked(split):
            if sink.counts[split] >= pool.caps[split]:
                notes[f'over-cap-{split}'] += 1
                continue
            if windows[trajectory] >= MAX_WINDOWS:
                sink.reject(f'{label}:{trajectory}', f'window-cap: more than {MAX_WINDOWS} windows per trajectory')
                continue
            record = build(offset, split)
            offered += 1
            if offered % 500 == 0:
                release_cache(sink.out, *streams)
            if sink.offer(record):
                windows[trajectory] += 1
                taken.add((trajectory, tuple((record['sources'][0].get('meta') or {}).get('span') or ())))
    return notes


# ---------------------------------------------------------------------------------------------------------
# (a) S1 records
# ---------------------------------------------------------------------------------------------------------

def s1_reason(record: dict, bench: set) -> str | None:
    """None when an S1 record is a verified compact-history candidate, else the drop reason."""
    outcome = record.get('outcome') or {}
    if outcome.get('label') != 'checked' or outcome.get('checked') != VERIFIED:
        return 'not-swe-rebench-verified'
    sources = record.get('sources') or []
    if len(sources) != 1 or sources[0].get('role') != 'trajectory':
        return 'source-shape'
    length = len(source_text(sources[0]))
    rules = RULES[OPERATOR]
    if length < rules['min_chars']:
        return 'below-8000-chars'
    if length > rules['max_chars']:
        return 'above-200000-chars'
    notes = (record.get('lineage') or {}).get('notes') or {}
    instance = notes.get('instance_id')
    if (notes.get('benchmark_protected') or instance in bench) and split_of(record) == 'train':
        return 'benchmark-instance-in-train'
    return None


def select_s1(path: Path, sink: CaseSink, bench: set, limit: int | None, caps: dict):
    pool, dropped, offset, released = Pool(S1_NAME, caps), Counter(), 0, 0
    with open(path, 'rb') as stream:
        for count, line in enumerate(stream):
            if limit is not None and count >= limit:
                break
            start, offset = offset, offset + len(line)
            if offset - released >= CACHE_WINDOW:     # a 22 GB scan must not fill the unit's page-cache charge
                os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                released = offset
            if VERIFIED.encode() not in line:        # cheap pre-filter: only SWE-rebench-verified rows are parsed
                dropped['not-swe-rebench-verified'] += 1
                continue
            record = json.loads(line)
            reason = s1_reason(record, bench)
            if reason:
                dropped[reason] += 1
                continue
            pool.add(split_of(record), record['id'], start, record['lineage']['upstream_id'])
    for reason, number in dropped.items():
        sink.rejects[reason] += number
    return pool, dropped


def convert_s1(out_root: Path, limit: int | None, windows: Counter, bench: set, path: Path = S1_FILE,
               train_cap: int | None = None) -> dict:
    caps = {'train': TRAIN_CAP if train_cap is None else train_cap, 'test': TEST_CAP,
            'max_windows_per_trajectory': MAX_WINDOWS}
    taken: set = set()
    with CaseSink(OPERATOR, S1_NAME, converter=__file__, inputs=[path], out_root=out_root, caps=caps) as sink:
        pool, dropped = select_s1(path, sink, bench, limit, caps)
        stream = open(path, 'rb')

        def build(offset, split):
            stream.seek(offset)
            return json.loads(stream.readline())

        notes = take_ranked(pool, windows, sink, build, S1_NAME, taken, (stream,))
        stream.close()
        sink.notes = ('S1 trajectory_continuation_swe rows with outcome.checked == swe-rebench-tests, records kept as '
                      f'sealed in S1 (split closure and protected moves already applied); eligible {dict(pool.eligible)}; '
                      f'dropped before selection {dict(sorted(dropped.items()))}; selection {dict(notes)}')
        return {'source': S1_NAME, 'counts': dict(sink.counts), 'eligible': dict(pool.eligible),
                'dropped': dict(dropped), 'notes': dict(notes), 'taken': taken}



# ---------------------------------------------------------------------------------------------------------
# (b) bgkit compaction_v2 windows, re-materialised from the upstream trajectories they index
# ---------------------------------------------------------------------------------------------------------

BGKIT_NAME = 'bgkit-compaction-v2'
BGKIT_TASKS = Path.home() / 'bgkit-data-nvme' / 'bgkit2' / 'tasks' / 'compaction_v2.parquet'
UPSTREAM_TRAJECTORIES = TRAJECTORIES_EXT / 'swe_rebench_openhands' / 'trajectories.parquet'
UPSTREAM = 'nebius/SWE-rebench-openhands-trajectories'
CONVERTER = 'scripts/neuralese_data/operator_cases_compact_history.py@1'
COMPACTED_MARKER = '[Earlier conversation compacted]'
INSTRUCTIONS = ('Read this earlier part of the coding session so that the task can be continued from the most '
                'recent steps.')


def message_of(m: dict) -> dict:
    """An upstream OpenHands message without its scaffolding markup (the system prompt is dropped by the caller)."""
    out = {'role': m.get('role')}
    content = m.get('content')
    out['content'] = strip_markup(content) if isinstance(content, str) else None
    for key in ('name', 'tool_call_id'):
        if m.get(key):
            out[key] = m[key]
    if m.get('tool_calls'):
        out['tool_calls'] = [{'id': c.get('id'), 'type': c.get('type') or 'function',
                              'function': {'name': (c.get('function') or {}).get('name'),
                                           'arguments': (c.get('function') or {}).get('arguments')}}
                             for c in m['tool_calls']]
    return out


def prune(value):
    """Drop nulls that Arrow's struct unification adds to upstream tool schemas."""
    if isinstance(value, dict):
        return {k: prune(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [prune(v) for v in value]
    return value


def upstream_status(path: Path = UPSTREAM_TRAJECTORIES) -> dict:
    """{trajectory_id: (instance_id, repo, resolved, row_group)} from the three small columns only."""
    import pyarrow.parquet as pq
    status, parquet = {}, pq.ParquetFile(path)
    for group in range(parquet.num_row_groups):
        table = parquet.read_row_group(group, columns=['trajectory_id', 'instance_id', 'repo', 'resolved'])
        for row in zip(*(table.column(name).to_pylist() for name in ('trajectory_id', 'instance_id', 'repo',
                                                                      'resolved'))):
            status[row[0]] = (row[1], row[2], bool(row[3]), group)
    return status


def bgkit_candidates(tasks_path: Path, status: dict, bench: set, limit: int | None, sink: CaseSink):
    """Task rows joined to the upstream resolved status; the drops are counted as sink rejects by reason."""
    import pyarrow.parquet as pq
    names = ('family', 'instruction', 'target', 'split', 'meta')
    out, index = [], 0
    for batch in pq.ParquetFile(tasks_path).iter_batches(batch_size=1000, columns=list(names)):
        for family, instruction, target, split, meta in zip(*(batch.column(name).to_pylist() for name in names)):
            index += 1
            if limit is not None and index > limit:
                return out
            meta = json.loads(meta)
            tid, span = meta['trajectory_id'], tuple(meta['span'])
            record_id = f'{BGKIT_NAME}:{tid}:{span[0]}-{span[1]}'
            if family != 'compaction':
                sink.reject(record_id, f'not-a-continuation: family {family} targets a probe, not the next '
                                       'assistant message')
            elif tid not in status:
                sink.reject(record_id, 'no-upstream: trajectory not in the upstream resolved-status table')
            elif not status[tid][2]:
                sink.reject(record_id, 'unresolved: upstream tests did not resolve the instance')
            else:
                # The stored session is collapsed: [system, task, compaction marker, *recent steps]; the recent
                # steps are upstream messages[stop : stop + n], the target is the next assistant message.
                session = json.loads(instruction)
                if len(session) < 4 or not (session[2].get('content') or '').startswith(COMPACTED_MARKER):
                    sink.reject(record_id, 'misaligned: stored session has no compaction marker')
                    continue
                instance = status[tid][0]
                out.append({'id': record_id, 'tid': tid, 'span': span, 'recent': len(session) - 3,
                            'target': target,
                            'split': 'test' if split in ('eval', 'test') or instance in bench else 'train',
                            'group': status[tid][3]})
    return out


def bgkit_record(candidate: dict, status: dict, trajectory: list, tools) -> dict | str:
    """The sealed record for one candidate, or the reject reason when upstream does not line up."""
    start, stop = candidate['span']
    target_index = stop + candidate['recent']
    if (len(trajectory) <= target_index or len(trajectory) < 2 or trajectory[0].get('role') != 'system'
            or trajectory[1].get('role') != 'user' or not 2 <= start < stop):
        return 'misaligned: span outside the upstream trajectory'
    upstream_target = trajectory[target_index]
    if upstream_target.get('role') != 'assistant' or trajectory[stop].get('role') != 'assistant':
        return 'misaligned: recent steps do not start and end at assistant turns'
    target = strip_markup(upstream_target.get('content') or '')
    if not target.strip() or target.strip() != candidate['target'].strip():
        return 'target-mismatch: stored target is not the upstream next assistant text'
    task = message_of(trajectory[1])
    earlier = [message_of(m) for m in trajectory[start:stop]]
    instance, repo, _, _ = status[candidate['tid']]
    text = '\n'.join(str(m.get('content') or '') for m in earlier)
    record = {
        'version': VERSION, 'id': f'upstream:swe-rebench-openhands:{candidate["tid"]}:{target_index}',
        'family': 'trajectory_continuation_swe', 'task': 'continue',
        'sources': [{'role': 'trajectory', 'messages': earlier, 'exact_refs': exact_refs_from(text),
                     'meta': {'span': [start, stop]}}],
        'writer': {'instructions': INSTRUCTIONS, 'result_type': 'Neuralese<SessionHistory>', 'context': [task]},
        'consumer': {'context': [task, *[message_of(m) for m in trajectory[stop:target_index]]],
                     'tools': prune(list(tools or [])), 'withheld': ['sources']},
        'target': {'kind': 'message', 'value': {'role': 'assistant', 'content': target}, 'alternatives': []},
        'outcome': {'label': 'checked', 'checked': VERIFIED,
                    'details': {'resolved': True, 'checker_evidence': {
                        'origin': 'upstream_resolved_status', 'raw_status_field': 'resolved', 'raw_status': True,
                        'local_execution': False}}},
        'lineage': {'project': 'bgkit', 'store': 'compaction_v2', 'store_version': None, 'row': candidate['id'],
                    'upstream': UPSTREAM, 'upstream_id': candidate['tid'], 'upstream_revision': None,
                    'teacher': {'model': 'Qwen/Qwen3-Coder-480B-A35B-Instruct', 'provider': 'nebius (dataset)'},
                    'converter': CONVERTER, 'sha256': '',
                    'notes': {'instance_id': instance, 'upstream_system_prompt': 'dropped (OpenHands scaffolding)',
                              'benchmark_protected': candidate['split'] == 'test'}},
        'license': license_('CC-BY-4.0', False, 'Trajectories CC-BY-4.0; embedded repository code under each '
                                                "repository's licence."),
        'split': candidate['split'],
        'split_groups': [group_key('repo', repo), group_key('swe-instance', instance)],
    }
    return seal(record)


def open_bgkit_sink(out_root: Path, tasks_path: Path, upstream_path: Path) -> CaseSink:
    return CaseSink(OPERATOR, BGKIT_NAME, converter=__file__, inputs=[tasks_path, upstream_path], out_root=out_root,
                    caps={'train': TRAIN_CAP, 'test': TEST_CAP, 'max_windows_per_trajectory': MAX_WINDOWS})


def convert_bgkit(sink: CaseSink, status: dict, candidates: list, windows: Counter, taken: set,
                  upstream_path: Path = UPSTREAM_TRAJECTORIES) -> dict:
    """Rank, apply the window caps, then re-materialise the chosen windows one upstream row group at a time."""
    import pyarrow.parquet as pq
    chosen, notes, counted, per_split = [], Counter(), Counter(), Counter()
    for split in ('train', 'test'):
        pool = sorted((c for c in candidates if c['split'] == split), key=lambda c: stable_rank(c['id'], BGKIT_NAME))
        for candidate in pool:
            if per_split[split] >= sink.caps[split]:
                notes[f'over-cap-{split}'] += 1
            elif (candidate['tid'], candidate['span']) in taken:
                sink.reject(candidate['id'], 'duplicate: window already taken from the S1 records')
            elif windows[candidate['tid']] + counted[candidate['tid']] >= MAX_WINDOWS:
                sink.reject(candidate['id'], f'window-cap: more than {MAX_WINDOWS} windows per trajectory')
            else:
                counted[candidate['tid']] += 1
                per_split[split] += 1
                chosen.append(candidate)
    parquet = pq.ParquetFile(upstream_path)
    for group in sorted({c['group'] for c in chosen}):
        wanted = {c['tid'] for c in chosen if c['group'] == group}
        table = parquet.read_row_group(group, columns=['trajectory_id', 'trajectory', 'tools'])
        by_id = {tid: (trajectory, tools) for tid, trajectory, tools in zip(
            *(table.column(name).to_pylist() for name in ('trajectory_id', 'trajectory', 'tools'))) if tid in wanted}
        del table
        for candidate in (c for c in chosen if c['group'] == group):
            trajectory, tools = by_id[candidate['tid']]
            built = bgkit_record(candidate, status, trajectory, tools)
            if isinstance(built, str):
                sink.reject(candidate['id'], built)
            elif sink.offer(built):
                windows[candidate['tid']] += 1
    return dict(notes)

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--limit', type=int, default=None, help='smoke: read only the first N input rows')
    parser.add_argument('--out-root', default=str(OUT_ROOT))
    args = parser.parse_args(argv)
    out_root, windows, bench = Path(args.out_root), Counter(), load_bench_instances()
    # The bgkit join runs first only to learn how many windows it can supply: whatever part of its 15,000-window
    # budget it cannot fill is given to the S1 records, so the train total approaches the 25-30k target.
    status = upstream_status()
    with open_bgkit_sink(out_root, BGKIT_TASKS, UPSTREAM_TRAJECTORIES) as sink:
        candidates = bgkit_candidates(BGKIT_TASKS, status, bench, args.limit, sink)
        available = sum(1 for c in candidates if c['split'] == 'train')
        s1_cap = 2 * TRAIN_CAP - min(TRAIN_CAP, available)
        summary = convert_s1(out_root, args.limit, windows, bench, train_cap=s1_cap)
        taken = summary.pop('taken')
        print(json.dumps(summary), flush=True)
        notes = convert_bgkit(sink, status, candidates, windows, taken)
        sink.notes = (f'bgkit compaction_v2 joined to {UPSTREAM} resolved status; {len(candidates)} joined candidates; '
                      f'selection {notes}')
    print(json.dumps({'source': BGKIT_NAME, 'counts': dict(sink.counts), 'rejects': dict(sink.rejects),
                      'joined_candidates': len(candidates)}), flush=True)


if __name__ == '__main__':
    main()
