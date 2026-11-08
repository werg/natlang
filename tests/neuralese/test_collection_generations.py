"""A collection resumes across restarts that change endpoints, worker counts and code (rl.collection_generations)."""
import json
from pathlib import Path

import pytest

from natlang_neuralese.rl import production
from natlang_neuralese.rl.collection_generations import Generations, comparable_binding, code_changes
from natlang_neuralese.common.jsonio import read_jsonl

REPO = Path(__file__).resolve().parents[2]
POLICY = 'a' * 64


class Harness:
    """Fake collector: completes entries until ``budget`` attempts are used, then 'dies' with a start and no finish."""

    def __init__(self, tmp_path, monkeypatch, entries=6, teachers=0):
        from natlang_neuralese.rl import actor_client
        self.tmp = tmp_path
        self.runtime = tmp_path / 'ts-host'
        (self.runtime / 'dist/teacher').mkdir(parents=True)
        (self.runtime / 'dist/teacher/cli.js').write_text('1')
        self.cases = tmp_path / 'cases.jsonl'
        self.cases.write_text('{}\n')
        self.out = tmp_path / 'round'
        self.entries, self.budget, self.ran = entries, 0, []
        monkeypatch.setattr(production, 'root', lambda name: REPO)

        class Server:
            def __init__(self, url, **kw): self.url = url
            def get_bytes(self, path): return json.dumps({'rl_policy_binding': {'checkpoint_sha256': POLICY}}).encode()
        monkeypatch.setattr(actor_client, 'NativeActorClient', Server)

        def prepare(args):
            queue = [{'key': f's7-task-{i:03d}', 'index': 0, 'group': i, 'sample': 0, 'seed': i,
                      'output': str(self.out / f'r{i}.jsonl'), 'jobs': str(self.out), 'log': str(self.out / 'l')}
                     for i in range(entries)]
            (self.out / 'queue.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in queue))
            return {'ids': ['t'] * entries}
        monkeypatch.setattr(production, 'prepare', prepare)
        monkeypatch.setattr(production, 'score', lambda out, emit, **kw: (Path(emit).write_text(''), {'scored': 1})[1])
        monkeypatch.setattr(production, '_run_collector', self.collector)

    def collector(self, command, log, control, custody_receipt=None):
        queue, journal = Path(command[2]), Path(command[3])
        endpoint = command[command.index('--server') + 1]
        finished = {row['key'] for row in read_jsonl(journal) if row['event'] == 'finish'} if journal.exists() else set()
        for entry in read_jsonl(queue):
            if entry['key'] in finished:   # run_bonsai_queue: restart skips attempted entries
                continue
            self.ran.append((entry['key'], endpoint, journal.name))
            with journal.open('a') as stream:
                stream.write(json.dumps({'event': 'start', 'key': entry['key']}) + '\n')
                if self.budget <= 0:
                    return 143   # killed mid-entry: a start, never a finish
                self.budget -= 1
                stream.write(json.dumps({'event': 'finish', 'key': entry['key'], 'status': 'complete'}) + '\n')
        return 0

    def collect(self, **overrides):
        kwargs = dict(cases=self.cases, out=self.out, endpoints=['http://a'], policy_id=POLICY, rows=self.entries,
                      workers=2, runtime=self.runtime, neuralese_server=False)
        return production.collect(**{**kwargs, **overrides})


def finishes(out):
    return [row['key'] for path in Generations(out).journals() for row in read_jsonl(path) if row.get('event') == 'finish']


def test_restart_with_other_endpoints_and_workers_runs_only_pending_entries_once(tmp_path, monkeypatch):
    h = Harness(tmp_path, monkeypatch)
    h.budget = 3
    first = h.collect()
    assert first['collection_complete'] is True   # (no stop control: the fake collector just ran out of budget)
    assert len(finishes(h.out)) == 3
    binding = json.loads((h.out / 'collection-binding.json').read_text())
    assert 'endpoints' not in binding and 'remote_workers' not in binding   # transport is not provenance
    # restart: a remote joins, workers change; the 3 finished entries stay finished, the 3 others run
    (h.out / 'score.json').unlink()
    h.ran.clear()
    h.budget = 99
    h.collect(endpoints=['http://a', 'http://b'], workers=1, remote_workers={'http://b': {'workers': 2, 'weight': 1}})
    keys = finishes(h.out)
    assert sorted(keys) == [f's7-task-{i:03d}' for i in range(6)] and len(keys) == len(set(keys))
    assert sorted(key for key, *_ in h.ran) == sorted({f's7-task-{i:03d}' for i in range(6)} - set(keys[:3]))
    gen2 = [row for row in h.ran]
    assert {journal for _, _, journal in gen2} <= {f'journal-g2-{i}.jsonl' for i in range(3)}
    assert {endpoint for _, endpoint, _ in gen2} == {'http://a', 'http://b'}
    history = [row for row in read_jsonl(h.out / 'generations.jsonl') if row['event'] == 'start']
    assert [row['generation'] for row in history] == [1, 2]
    assert history[1]['endpoints'] == ['http://a', 'http://b'] and history[0]['endpoints'] == ['http://a']
    # merged journal reads every generation; rollouts keep the endpoint that actually sampled them
    merged = [json.loads(line) for line in (h.out / 'journal.jsonl').read_text().splitlines()]
    assert sum(row['event'] == 'finish' for row in merged) == 6
    sampled = Generations(h.out).finished_endpoints()
    assert len(sampled) == 6 and {sampled[k] for k in keys[:3]} == {'http://a'}


def test_the_interrupted_entry_is_retried_not_counted_and_a_completed_round_dispatches_nothing(tmp_path, monkeypatch):
    h = Harness(tmp_path, monkeypatch, entries=4)
    h.budget = 1
    h.collect()
    started = Generations(h.out).started()
    assert len(started) > 1 and len(Generations(h.out).attempted()) == 1   # the killed entry has a start only
    (h.out / 'score.json').unlink()
    h.budget, h.ran = 99, []
    h.collect()
    assert len(h.ran) == 3 and len(set(finishes(h.out))) == 4 == len(finishes(h.out))
    h.ran.clear()
    assert h.collect(endpoints=['http://elsewhere'], workers=7)['collection_complete'] is True
    assert h.ran == []   # completed: nothing dispatched, transport irrelevant


def test_code_identity_may_change_on_resume_but_sampling_may_not(tmp_path, monkeypatch):
    h = Harness(tmp_path, monkeypatch, entries=2)
    h.budget = 1
    h.collect()
    (h.out / 'score.json').unlink()
    (h.runtime / 'dist/new.js').write_text('changed collector code')   # a successor runtime
    h.budget = 99
    h.collect()
    start = [row for row in read_jsonl(h.out / 'generations.jsonl') if row['event'] == 'start'][-1]
    assert list(start['code_changes']) == ['runtime']
    (h.out / 'score.json').unlink()
    with pytest.raises(ValueError, match='resume policy/cases/sampling/runtime changed: runtime'):
        h.collect(allow_code_change=False)
    with pytest.raises(ValueError, match='rows'):
        h.collect(rows=1)


def test_teacher_plan_is_persisted_and_reused_by_a_restart(tmp_path, monkeypatch):
    from natlang_neuralese.rl import jolt
    h = Harness(tmp_path, monkeypatch, entries=2)
    monkeypatch.setattr(jolt, 'student_guidance', lambda *a, **k: None)
    plans = []

    def plan(student_entries, rows, **kw):
        plans.append(1)
        teacher = {**student_entries[0], 'key': f's7-task-000-t0{len(plans)}', 'jolt_role': 'teacher'}
        return {'entries': [teacher], 'refusal': None, 'plan': {'implementation': 'crisp'}}
    monkeypatch.setattr(jolt, 'teacher_entries', plan)
    kwargs = dict(teacher_attempts=1, teacher_plan='crisp', prompt_rendering=None)
    h.budget = 2                      # both students finish; the teacher is killed mid-run
    h.collect(**kwargs)
    saved = json.loads((h.out / 'teacher-plan.json').read_text())
    assert saved['teachers']['entries'][0]['key'] == 's7-task-000-t01' and len(plans) == 1
    (h.out / 'score.json').unlink()
    h.budget, h.ran = 99, []
    h.collect(**kwargs)
    assert len(plans) == 1            # not re-planned: the started teacher key is the one that now finishes
    assert [key for key, *_ in h.ran] == ['s7-task-000-t01']
    assert sorted(finishes(h.out)) == ['s7-task-000', 's7-task-000-t01', 's7-task-001']
    queue = [row['key'] for row in read_jsonl(h.out / 'queue.jsonl')]
    assert queue == ['s7-task-000', 's7-task-001', 's7-task-000-t01']


def test_comparable_binding_and_code_changes():
    a = {'rows': 1, 'endpoints': ['x'], 'remote_workers': {}, 'runtime': {'a.js': '1'}, 'collector_sha256': 'c',
         'expected_native_actor': {'k': 1, 'python_runtime': {'sha256': 'p'}}}
    b = {**a, 'endpoints': ['y'], 'runtime': {'a.js': '2'}, 'expected_native_actor': {'k': 1, 'python_runtime': {'sha256': 'q'}}}
    assert comparable_binding(a, code=True) == comparable_binding(b, code=True) == {'rows': 1, 'expected_native_actor': {'k': 1}}
    assert comparable_binding(a) != comparable_binding(b)
    assert sorted(code_changes(a, b)) == ['actor_python_runtime', 'runtime']


def test_a_round_bound_by_an_older_controller_resumes_with_new_transport_and_code():
    recorded = {'schema': 'natlang.rl-collection-binding/1', 'cases_sha256': 'c', 'policy_id': 'p', 'rows': 16, 'group': 8,
                'endpoints': ['http://127.0.0.1:8200'], 'neuralese_server': True, 'runtime': {'dist/a.js': '1'},
                'collector_sha256': 'old', 'jolt_teacher_attempts': 1,
                'jolt_guidance': {'schema': 'natlang.guidance-run/1', 'teacher_plan': 'crisp'},
                'expected_native_actor': {'native_identity': {'x': 1}, 'python_runtime': {'sha256': 'v57'}}}
    current = {k: v for k, v in recorded.items() if k not in ('endpoints', 'jolt_teacher_attempts', 'jolt_guidance')}
    current.update(runtime={'dist/a.js': '2'}, collector_sha256='new',
                   expected_native_actor={'native_identity': {'x': 1}, 'python_runtime': {'sha256': 'v58'}})
    strict, changed, bonus, teachers = production.compare_recorded_binding(recorded, current, 1, 'crisp')
    assert strict == [] and teachers == 1 and sorted(changed) == ['actor_python_runtime', 'collector_sha256', 'runtime']
    strict, changed, _, _ = production.compare_recorded_binding(recorded, current, 1, 'crisp', allow_code_change=False)
    assert 'collector_sha256' in strict and changed == {}
    # sampling changes still refuse
    strict, *_ = production.compare_recorded_binding(recorded, {**current, 'rows': 32, 'seed_base': 5}, 1, 'crisp')
    assert strict == ['rows', 'seed_base']
    # a round bound before teacher attempts existed keeps its behaviour even when endpoints were recorded
    old = {k: v for k, v in recorded.items() if not k.startswith('jolt')}
    strict, _, _, teachers = production.compare_recorded_binding(old, current, 1, 'crisp')
    assert strict == [] and teachers == 0
