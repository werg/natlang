"""compact-history converter over inline S1-shaped fixture rows (scripts/neuralese_data/operator_cases_compact_history.py)."""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))

from neuralese_data import operator_cases_compact_history as ch  # noqa: E402
from neuralese_data.records import VERSION, group_key, license_, seal, source, validate  # noqa: E402

def valid(case):
    """Schema-valid; the sink adds `meta.operator_seed` after sealing, so only the content hash is exempt."""
    return [e for e in validate(case) if 'sha256' not in e] == []


TASK = {'role': 'user', 'content': 'Fix the bug in pkg/a.py'}


def messages(count, tag):
    out = []
    for index in range(count):
        out.append({'role': 'assistant', 'content': f'{tag} step {index} ' + 'x' * 120,
                    'tool_calls': [{'id': f'c{index}', 'type': 'function',
                                    'function': {'name': 'bash', 'arguments': '{"cmd": "cat pkg/a.py"}'}}]})
        out.append({'role': 'tool', 'content': f'{tag} output {index} ' + 'y' * 120, 'tool_call_id': f'c{index}'})
    return out


def row(index, *, trajectory=None, split='train', label='checked', checked='swe-rebench-tests', turns=40,
        instance='o__r-1', protected=False):
    trajectory = trajectory or f'traj-{index}'
    return seal({
        'version': VERSION, 'id': f'upstream:fixture:{trajectory}:{index}', 'family': 'trajectory_continuation_swe',
        'task': 'continue', 'sources': [source('trajectory', messages=messages(turns, f't{index}'))],
        'writer': {'instructions': 'Read this earlier part of the coding session.',
                   'result_type': 'Neuralese<SessionHistory>', 'context': [TASK]},
        'consumer': {'context': [TASK, *messages(2, 'recent')], 'withheld': ['sources']},
        'target': {'kind': 'message', 'value': {'role': 'assistant', 'content': f'next {index}'}, 'alternatives': []},
        'outcome': {'label': label, 'checked': checked},
        'lineage': {'project': 'upstream', 'store': 'fixture', 'store_version': None, 'row': index, 'upstream': 'x',
                    'upstream_id': trajectory, 'upstream_revision': None, 'teacher': None, 'converter': 'test',
                    'sha256': '', 'notes': {'instance_id': instance, 'benchmark_protected': protected}},
        'license': license_('MIT'), 'split': split,
        'split_groups': [group_key('repo', 'o/r'), group_key('swe-instance', instance)]})


def test_s1_selection_keeps_only_verified_rows_and_caps_windows_per_trajectory(tmp_path):
    rows = [row(0), row(1, trajectory='traj-0'), row(2, trajectory='traj-0'),          # three windows of one trajectory
            row(3, split='test'), row(4, label='teacher', checked=None), row(5, checked='other-tests'),
            row(6, turns=2), row(7, turns=4000), row(8, protected=True, instance='bench__x-1')]
    s1 = tmp_path / 's1.jsonl'
    s1.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    windows = Counter()
    summary = ch.convert_s1(tmp_path / 'out', None, windows, set(), path=s1)
    assert summary['counts'] == {'train': 2, 'test': 1}
    assert windows['traj-0'] == ch.MAX_WINDOWS
    out = tmp_path / 'out' / 'compact-history'
    cases = [json.loads(line) for line in (out / 's1-swe.port-records.jsonl').read_text().splitlines()]
    assert len(cases) == 3 and all(valid(case) for case in cases)
    assert {case['outcome']['checked'] for case in cases} == {'swe-rebench-tests'}
    assert all(case['sources'][0]['meta']['operator_seed']['provenance'] == 'mechanical:history-outline' for case in cases)
    assert all(case['sources'][0]['role'] == 'trajectory' and 'sources' in case['consumer']['withheld']
               for case in cases)
    rejects = json.loads((out / 's1-swe.manifest.json').read_text())['rejects']
    assert rejects['not-swe-rebench-verified'] == 2 and rejects['window-cap'] == 1
    assert rejects['below-8000-chars'] == 1 and rejects['above-200000-chars'] == 1
    assert rejects['benchmark-instance-in-train'] == 1


def test_s1_selection_is_deterministic_under_a_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(ch, 'TRAIN_CAP', 2)
    rows = [row(index) for index in range(8)]
    s1 = tmp_path / 's1.jsonl'
    s1.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    picked = []
    for name in ('a', 'b'):
        ch.convert_s1(tmp_path / name, None, Counter(), set(), path=s1)
        picked.append([json.loads(line)['id'] for line in
                       (tmp_path / name / 'compact-history' / 's1-swe.port-records.jsonl').read_text().splitlines()])
    assert picked[0] == picked[1] and len(picked[0]) == 2


def upstream_trajectory(tag, turns=20):
    messages = [{'role': 'system', 'content': 'You are OpenHands.'}, TASK]
    for index in range(turns):
        messages.append({'role': 'assistant', 'content': f'{tag} step {index} ' + 'x' * 500})
        messages.append({'role': 'tool', 'content': f'{tag} output {index} ' + 'y' * 500})
    return messages


def write_parquets(tmp_path, upstream, tasks):
    import pyarrow as pa
    import pyarrow.parquet as pq
    rows = [{'trajectory_id': tid, 'instance_id': instance, 'repo': 'o/r', 'resolved': resolved,
             'trajectory': upstream_trajectory(tid), 'tools': [{'function': {'name': 'bash'}}]}
            for tid, instance, resolved in upstream]
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / 'upstream.parquet')
    task_rows = []
    for tid, span, family, split, recent in tasks:
        trajectory = upstream_trajectory(tid)
        session = [trajectory[0], trajectory[1], {'role': 'user', 'content': ch.COMPACTED_MARKER + '\n<|reserved_6|>'},
                   *trajectory[span[1]:span[1] + recent]]
        target = trajectory[span[1] + recent]['content'] if family == 'compaction' else '/workspace/file.py'
        task_rows.append({'family': family, 'instruction': json.dumps(session), 'target': target, 'split': split,
                          'meta': json.dumps({'trajectory_id': tid, 'span': list(span), 'mode': 'segments'})})
    pq.write_table(pa.Table.from_pylist(task_rows), tmp_path / 'tasks.parquet')
    return tmp_path / 'tasks.parquet', tmp_path / 'upstream.parquet'


def test_bgkit_join_keeps_resolved_windows_and_rejects_the_rest_by_reason(tmp_path):
    tasks, upstream = write_parquets(
        tmp_path, [('a', 'o__r-1', True), ('b', 'o__r-2', False), ('d', 'o__r-4', True), ('e', 'bench__x-1', True)],
        [('a', (2, 26), 'compaction', 'train', 4), ('a', (2, 22), 'compaction', 'train', 4),
         ('a', (2, 30), 'compaction', 'train', 4),                                   # third window of one trajectory
         ('a', (2, 18), 'compaction_probe', 'train', 4), ('b', (2, 26), 'compaction', 'train', 4),
         ('c', (2, 26), 'compaction', 'train', 4), ('d', (2, 26), 'compaction', 'eval', 4),
         ('e', (2, 26), 'compaction', 'train', 4)])
    status = ch.upstream_status(upstream)
    out = tmp_path / 'out'
    windows = Counter()
    with ch.open_bgkit_sink(out, tasks, upstream) as sink:
        candidates = ch.bgkit_candidates(tasks, status, {'bench__x-1'}, None, sink)
        ch.convert_bgkit(sink, status, candidates, windows, set(), upstream_path=upstream)
    assert dict(sink.counts) == {'train': 2, 'test': 2}
    cases = [json.loads(line) for line in (out / 'compact-history' / 'bgkit-compaction-v2.port-records.jsonl')
             .read_text().splitlines()]
    assert all(valid(case) for case in cases)
    train = next(case for case in cases if case['split'] == 'train')
    assert train['outcome']['checked'] == 'swe-rebench-tests' and train['consumer']['context'][0] == TASK
    assert train['sources'][0]['messages'][0]['role'] == 'assistant'             # the system prompt is dropped
    assert len(train['consumer']['context']) == 1 + 4 and train['target']['value']['role'] == 'assistant'
    assert train['split_groups'] == [group_key('repo', 'o/r'), group_key('swe-instance', 'o__r-1')]
    assert {case['split'] for case in cases if 'o__r-4' in json.dumps(case['split_groups'])} == {'test'}
    assert {case['split'] for case in cases if 'bench__x-1' in json.dumps(case['split_groups'])} == {'test'}
    rejects = json.loads((out / 'compact-history' / 'bgkit-compaction-v2.manifest.json').read_text())['rejects']
    assert rejects == {'not-a-continuation': 1, 'unresolved': 1, 'no-upstream': 1, 'window-cap': 1}


def test_bgkit_window_already_taken_from_s1_is_not_repeated(tmp_path):
    tasks, upstream = write_parquets(tmp_path, [('a', 'o__r-1', True)], [('a', (2, 26), 'compaction', 'train', 4)])
    status = ch.upstream_status(upstream)
    with ch.open_bgkit_sink(tmp_path / 'out', tasks, upstream) as sink:
        candidates = ch.bgkit_candidates(tasks, status, set(), None, sink)
        ch.convert_bgkit(sink, status, candidates, Counter(), {('a', (2, 26))}, upstream_path=upstream)
    assert dict(sink.counts) == {} and dict(sink.rejects) == {'duplicate': 1}


def test_swe_zero_labels_admit_only_locally_resolved_trajectories(tmp_path, monkeypatch):
    """SWE-Zero trajectories join the locally executed test labels; unlabelled instances are rejected by status."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    data = tmp_path / 'swe_zero' / 'data'
    data.mkdir(parents=True)
    ids = ['z1', 'z2', 'z3', 'z4']
    pq.write_table(pa.Table.from_pylist([
        {'trajectory_id': tid, 'instance_id': f'o__r-{n}', 'repo': 'o/r', 'trajectory': upstream_trajectory(tid)}
        for n, tid in enumerate(ids)]), data / 'train-00000-of-00001.parquet')
    (tmp_path / 'swe_zero' / 'tools.json').write_text(json.dumps([{'type': 'function', 'function': {'name': 'bash'}}]))
    monkeypatch.setattr(ch, 'SWE_ZERO', tmp_path / 'swe_zero')
    harness = {'name': 'swe-rebench-fork', 'git_commit': 'abc'}
    labels = tmp_path / 'labels.jsonl'
    labels.write_text(''.join(json.dumps(row) + '\n' for row in [
        {'trajectory_id': 'z1', 'instance_id': 'o__r-0', 'resolved': True, 'status': 'resolved',
         'test_log_sha256': 'f' * 64, 'harness': harness, 'image': 'img', 'tests': {}},
        {'trajectory_id': 'z2', 'instance_id': 'o__r-1', 'resolved': False, 'status': 'unresolved'},
        {'trajectory_id': 'z3', 'instance_id': 'o__r-2', 'resolved': None, 'status': 'no-environment'},
    ]))
    status = ch.swe_zero_status(labels, data)
    assert {tid: value[2] for tid, value in status.items()} == {'z1': True, 'z2': False, 'z3': None}
    tasks_rows = [{'family': 'compaction', 'target': upstream_trajectory(tid)[26]['content'], 'split': 'train',
                   'instruction': json.dumps([upstream_trajectory(tid)[0], TASK,
                                              {'role': 'user', 'content': ch.COMPACTED_MARKER + '\n<|reserved_6|>'},
                                              *upstream_trajectory(tid)[22:26]]),
                   'meta': json.dumps({'trajectory_id': tid, 'span': [2, 22], 'mode': 'segments'})} for tid in ids]
    pq.write_table(pa.Table.from_pylist(tasks_rows), tmp_path / 'tasks.parquet')
    out = tmp_path / 'out'
    # The OpenHands upstream file is not consulted for SWE-Zero ids; an empty one stands in for it.
    pq.write_table(pa.Table.from_pylist([{'trajectory_id': 'x', 'instance_id': 'i', 'repo': 'r', 'resolved': True,
                                          'trajectory': [], 'tools': []}]), tmp_path / 'upstream.parquet')
    with ch.open_bgkit_sink(out, tmp_path / 'tasks.parquet', tmp_path / 'upstream.parquet') as sink:
        candidates = ch.bgkit_candidates(tmp_path / 'tasks.parquet', status, set(), None, sink)
        ch.convert_bgkit(sink, status, candidates, Counter(), set(), upstream_path=tmp_path / 'upstream.parquet')
    cases = [json.loads(line) for line in (out / 'compact-history' / 'bgkit-compaction-v2.port-records.jsonl')
             .read_text().splitlines()]
    assert len(cases) == 1 and valid(cases[0])
    assert cases[0]['outcome']['checked'] == ch.SWE_ZERO_VERIFIED
    evidence = cases[0]['outcome']['details']['checker_evidence']
    assert evidence['local_execution'] and evidence['gold_validated'] and evidence['test_log_sha256'] == 'f' * 64
    assert cases[0]['lineage']['upstream'] == ch.SWE_ZERO_UPSTREAM and cases[0]['consumer']['tools']
    rejects = json.loads((out / 'compact-history' / 'bgkit-compaction-v2.manifest.json').read_text())['rejects']
    assert rejects == {'unresolved': 1, 'no-label-no-environment': 1, 'no-upstream': 1}
