"""A collection resumes across restarts that change endpoints, worker counts and code (rl.collection_generations)."""
import json
from pathlib import Path

import pytest

from natlang_neuralese.rl import production
from natlang_neuralese.common import code_provenance as provenance
from natlang_neuralese.rl.collection_generations import Generations, IncompleteCollection, comparable_binding, dispatch
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
            (self.out / 'packet.jsonl').write_text('{}\n')
            (self.out / 'preparation.json').write_text('{}')
            return {'ids': ['t'] * entries}
        monkeypatch.setattr(production, 'prepare', prepare)
        monkeypatch.setattr(production, 'score', lambda out, emit, **kw: (Path(emit).write_text(''), {'scored': 1})[1])
        monkeypatch.setattr(production, '_run_collector', self.collector)

    def collector(self, command, log, control, custody_receipt=None, detach_path=None):
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
    # The fake collector dies after 3 finishes: the round must not seal, it stays resumable.
    with pytest.raises(IncompleteCollection, match='3 of 6 expected keys have no finish row'):
        h.collect()
    assert not (h.out / 'score.json').exists()
    assert len(finishes(h.out)) == 3
    events = read_jsonl(h.out / 'generations.jsonl')
    dispatched = [row for row in events if row['event'] == 'dispatch'][0]
    assert (dispatched['pending'], dispatched['finished'], dispatched['remaining']) == (6, 3, 3)
    incomplete = [row for row in events if row['event'] == 'incomplete'][0]
    assert incomplete['finished'] == 3 and len(incomplete['missing']) == 3
    binding = json.loads((h.out / 'collection-binding.json').read_text())
    assert 'endpoints' not in binding and 'remote_workers' not in binding   # transport is not provenance
    # restart: a remote joins, workers change; the 3 finished entries stay finished, the 3 others run
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
    with pytest.raises(IncompleteCollection):
        h.collect()
    started = Generations(h.out).started()
    assert len(started) > 1 and len(Generations(h.out).attempted()) == 1   # the killed entry has a start only
    h.budget, h.ran = 99, []
    complete = h.collect()
    assert complete['collection_keys'] == {'expected': 4, 'finished': 4, 'unfinished': []}
    assert len(h.ran) == 3 and len(set(finishes(h.out))) == 4 == len(finishes(h.out))
    h.ran.clear()
    assert h.collect(endpoints=['http://elsewhere'], workers=7)['collection_complete'] is True
    assert h.ran == []   # completed: nothing dispatched, transport irrelevant


def test_a_code_change_never_blocks_a_resume_and_each_start_records_its_commit(tmp_path, monkeypatch):
    h = Harness(tmp_path, monkeypatch, entries=2)
    commits = iter(['c1' * 20, 'c2' * 20])
    monkeypatch.setattr(provenance, 'code_provenance',
                        lambda *a, **k: {'schema': provenance.SCHEMA, 'git_commit': next(commits), 'dirty': False})
    h.budget = 1
    with pytest.raises(IncompleteCollection):
        h.collect()
    (h.runtime / 'dist/new.js').write_text('changed collector code')   # main moved: the next start runs newer code
    h.budget = 99
    h.collect()
    starts = [row for row in read_jsonl(h.out / 'generations.jsonl') if row['event'] == 'start']
    assert [row['code']['git_commit'] for row in starts] == ['c1' * 20, 'c2' * 20]
    assert starts[0]['code_changes'] == {'git_commit': [None, 'c1' * 20]}
    assert starts[1]['code_changes'] == {'git_commit': ['c1' * 20, 'c2' * 20]}
    assert json.loads((h.out / 'score.json').read_text())['code']['git_commit'] == 'c2' * 20
    assert 'runtime' not in json.loads((h.out / 'collection-binding.json').read_text())
    (h.out / 'score.json').unlink()
    with pytest.raises(ValueError, match='resume policy/cases/sampling changed'):   # sampling still may not change
        h.collect(rows=1)


class Gated(Harness):
    """The last student waits until ``gate`` plans exist: proof that teachers are planned while students still run."""

    def __init__(self, *a, gate=1, **k):
        super().__init__(*a, **k)
        self.gate, self.plans, self.finished_at_plan = gate, [], []

    def collector(self, command, log, control, custody_receipt=None, detach_path=None):
        import time
        queue = [e['key'] for e in read_jsonl(command[2])]
        if f's7-task-{self.entries - 1:03d}' in queue:
            end = time.monotonic() + 20
            while len(self.plans) < self.gate and time.monotonic() < end:
                time.sleep(0.02)
        return super().collector(command, log, control, custody_receipt, detach_path)


def test_teachers_are_planned_while_students_still_run_and_the_budget_carries(tmp_path, monkeypatch):
    from natlang_neuralese.rl import jolt
    h = Gated(tmp_path, monkeypatch, entries=4, gate=1)
    h.budget = 99
    monkeypatch.setattr(jolt, 'student_guidance', lambda *a, **k: None)

    def plan(student_entries, rows, *, carry=0, attempts, **kw):
        groups = sorted({e['group'] for e in student_entries})
        h.finished_at_plan.append(len(Generations(h.out).attempted()))
        h.plans.append({'groups': groups, 'carry': carry})
        # only group 0 asks for a teacher: the rest of the budget stays for later batches
        entries = [{**e, 'key': f's7-task-{e["group"]:03d}-t00', 'jolt_role': 'teacher'}
                   for e in student_entries if e['group'] == 0]
        return {'entries': entries, 'refusal': None, 'plan': {'implementation': 'crisp'},
                'accounting': {'budget': attempts * len(groups) + carry, 'planned': len(entries)}}
    monkeypatch.setattr(jolt, 'teacher_entries', plan)
    report = h.collect(teacher_attempts=1, teacher_plan='crisp', prompt_rendering=None, slot_poll_seconds=0.02)
    assert h.plans[0]['carry'] == 0 and h.finished_at_plan[0] < 4   # planned before the last student finished
    covered = [g for p in h.plans for g in p['groups']]
    assert sorted(covered) == [0, 1, 2, 3]                          # every group planned exactly once
    assert h.plans[1]['carry'] == len(h.plans[0]['groups']) - 1 if len(h.plans) > 1 else True
    assert sorted(finishes(h.out)) == sorted([f's7-task-{i:03d}' for i in range(4)] + ['s7-task-000-t00'])
    saved = json.loads((h.out / 'teacher-plan.json').read_text())
    assert saved['schema'] == 'natlang.rl-teacher-plan/2' and len(saved['student_keys']) == 4
    assert report['jolt_teacher_attempts']['entries'] == 1
    events = [row for row in read_jsonl(h.out / 'generations.jsonl') if row['event'] == 'teacher_plan']
    assert sorted(g for row in events for g in row['groups']) == [0, 1, 2, 3]
    assert [row['key'] for row in read_jsonl(h.out / 'queue.jsonl')][-1] == 's7-task-000-t00'


def test_teacher_plan_is_persisted_per_batch_and_reused_by_a_restart(tmp_path, monkeypatch):
    from natlang_neuralese.rl import jolt
    h = Harness(tmp_path, monkeypatch, entries=2)
    monkeypatch.setattr(jolt, 'student_guidance', lambda *a, **k: None)
    plans = []

    def plan(student_entries, rows, *, carry=0, attempts, **kw):
        plans.append(sorted({e['group'] for e in student_entries}))
        teachers = [{**e, 'key': f's7-task-{e["group"]:03d}-t{len(plans):02d}', 'jolt_role': 'teacher'}
                    for e in student_entries]
        return {'entries': teachers, 'refusal': None, 'plan': {'implementation': 'crisp'},
                'accounting': {'budget': attempts * len(plans[-1]) + carry, 'planned': len(teachers)}}
    monkeypatch.setattr(jolt, 'teacher_entries', plan)
    kwargs = dict(teacher_attempts=1, teacher_plan='crisp', prompt_rendering=None, workers=1, slot_poll_seconds=0.02,
                  teacher_plan_batch=2)
    h.budget = 2                      # both students finish; the teachers are killed mid-run
    with pytest.raises(IncompleteCollection, match='teacher collection incomplete: 2 of 2'):
        h.collect(**kwargs)
    saved = json.loads((h.out / 'teacher-plan.json').read_text())
    assert saved['schema'] == 'natlang.rl-teacher-plan/2'
    keys = sorted(e['key'] for b in saved['batches'] for e in b['entries'])
    assert keys == ['s7-task-000-t01', 's7-task-001-t01'] and plans == [[0, 1]]
    h.budget, h.ran = 99, []
    h.collect(**kwargs)
    assert plans == [[0, 1]]          # not re-planned: the started teacher keys are the ones that now finish
    assert sorted(key for key, *_ in h.ran) == keys
    assert sorted(finishes(h.out)) == ['s7-task-000', 's7-task-000-t01', 's7-task-001', 's7-task-001-t01']


def test_a_round_planned_by_the_old_whole_round_planner_resumes_unchanged(tmp_path, monkeypatch):
    from natlang_neuralese.rl import jolt
    from natlang_neuralese.rl.collection_generations import persist_teacher_plan
    h = Harness(tmp_path, monkeypatch, entries=2)
    monkeypatch.setattr(jolt, 'student_guidance', lambda *a, **k: None)
    monkeypatch.setattr(jolt, 'teacher_entries', lambda *a, **k: pytest.fail('a recorded plan is never re-planned'))
    h.out.mkdir(parents=True, exist_ok=True)
    teacher = {'key': 's7-task-000-t00', 'index': 0, 'group': 0, 'sample': 2, 'seed': 9, 'jolt_role': 'teacher',
               'output': str(h.out / 't.jsonl'), 'jobs': str(h.out), 'log': str(h.out / 'l')}
    persist_teacher_plan(h.out / 'teacher-plan.json', ['s7-task-000', 's7-task-001'],
                         {'entries': [teacher], 'refusal': None, 'plan': {'implementation': 'crisp'}})
    h.budget = 99
    report = h.collect(teacher_attempts=1, teacher_plan='crisp', prompt_rendering=None)
    assert json.loads((h.out / 'teacher-plan.json').read_text())['schema'] == 'natlang.rl-teacher-plan/1'
    assert sorted(finishes(h.out)) == ['s7-task-000', 's7-task-000-t00', 's7-task-001']
    assert report['jolt_teacher_attempts']['entries'] == 1


def test_dispatch_feed_entries_are_taken_by_idle_slots_and_a_stalled_student_does_not_hang(tmp_path):
    class Feed:
        def __init__(self): self.sent, self.polls = False, 0
        def poll(self, *, flush=False):
            self.polls += 1
            if self.sent or not (tmp_path / 'k0-done').exists():
                return []
            self.sent = True
            return [{'key': 'teacher0'}]
        def complete(self): return False   # k1's students never finish: only the quiet rule can close the feed

    ran = []

    def run_worker(index, name, queue, journal):
        key = json.loads(queue.read_text().splitlines()[-1])['key']
        ran.append(key)
        if key == 'k0':
            (tmp_path / 'k0-done').write_text('1')
        if key != 'k1':   # k1's collector dies without a finish row
            with journal.open('a') as stream:
                stream.write(json.dumps({'event': 'finish', 'key': key}) + '\n')
        return 0

    entries = [{'key': 'k0'}, {'key': 'k1'}]
    results, counts = dispatch(entries, directory=tmp_path, generation=1, phase='', endpoints=['a', 'b'],
                               run_worker=run_worker, attempted=set(), poll_seconds=0.02, feed=Feed())
    assert sorted(ran) == ['k0', 'k1', 'teacher0'] and [e['key'] for e in entries] == ['k0', 'k1', 'teacher0']
    assert counts == {'initial': 3, 'attempted': 3, 'finished': 2, 'remaining': 1}


def test_opt_out_plans_the_whole_round_after_the_students_and_reuses_it_on_restart(tmp_path, monkeypatch):
    from natlang_neuralese.rl import jolt
    h = Harness(tmp_path, monkeypatch, entries=2)
    monkeypatch.setattr(jolt, 'student_guidance', lambda *a, **k: None)
    plans = []

    def plan(student_entries, rows, **kw):
        plans.append(1)
        teacher = {**student_entries[0], 'key': f's7-task-000-t0{len(plans)}', 'jolt_role': 'teacher'}
        return {'entries': [teacher], 'refusal': None, 'plan': {'implementation': 'crisp'}}
    monkeypatch.setattr(jolt, 'teacher_entries', plan)
    kwargs = dict(teacher_attempts=1, teacher_plan='crisp', prompt_rendering=None, teacher_overlap=False)
    h.budget = 2                      # both students finish; the teacher is killed mid-run
    with pytest.raises(IncompleteCollection, match='teacher collection incomplete: 1 of 1'):
        h.collect(**kwargs)
    saved = json.loads((h.out / 'teacher-plan.json').read_text())
    assert saved['teachers']['entries'][0]['key'] == 's7-task-000-t01' and len(plans) == 1
    h.budget, h.ran = 99, []
    h.collect(**kwargs)
    assert len(plans) == 1            # not re-planned: the started teacher key is the one that now finishes
    assert [key for key, *_ in h.ran] == ['s7-task-000-t01']
    assert sorted(finishes(h.out)) == ['s7-task-000', 's7-task-000-t01', 's7-task-001']
    queue = [row['key'] for row in read_jsonl(h.out / 'queue.jsonl')]
    assert queue == ['s7-task-000', 's7-task-001', 's7-task-000-t01']


def test_comparable_binding_ignores_transport_and_every_code_identity_field():
    a = {'rows': 1, 'endpoints': ['x'], 'remote_workers': {}, 'runtime': {'a.js': '1'}, 'collector_sha256': 'c',
         'expected_native_actor': {'k': 1, 'python_runtime': {'sha256': 'p'}}}
    b = {**a, 'endpoints': ['y'], 'runtime': {'a.js': '2'}, 'expected_native_actor': {'k': 1, 'python_runtime': {'sha256': 'q'}}}
    assert comparable_binding(a) == comparable_binding(b) == {'rows': 1, 'expected_native_actor': {'k': 1}}


def test_a_round_bound_by_an_older_controller_resumes_with_new_transport_and_code():
    recorded = {'schema': 'natlang.rl-collection-binding/1', 'cases_sha256': 'c', 'policy_id': 'p', 'rows': 16, 'group': 8,
                'endpoints': ['http://127.0.0.1:8200'], 'neuralese_server': True, 'runtime': {'dist/a.js': '1'},
                'collector_sha256': 'old', 'jolt_teacher_attempts': 1,
                'jolt_guidance': {'schema': 'natlang.guidance-run/1', 'teacher_plan': 'crisp'},
                'expected_native_actor': {'native_identity': {'x': 1}, 'python_runtime': {'sha256': 'v57'}}}
    current = {k: v for k, v in recorded.items() if k not in ('endpoints', 'jolt_teacher_attempts', 'jolt_guidance')}
    current.update(runtime={'dist/a.js': '2'}, collector_sha256='new',
                   expected_native_actor={'native_identity': {'x': 1}, 'python_runtime': {'sha256': 'v58'}})
    strict, bonus, teachers = production.compare_recorded_binding(recorded, current, 1, 'crisp')
    assert strict == [] and teachers == 1
    # a binding written by the new code carries no code identity at all and still resumes the old round
    new_binding = {k: v for k, v in current.items() if k not in ('runtime', 'collector_sha256')}
    new_binding['expected_native_actor'] = {'native_identity': {'x': 1}}
    strict, *_ = production.compare_recorded_binding(recorded, new_binding, 1, 'crisp')
    assert strict == []
    # sampling changes still refuse
    strict, *_ = production.compare_recorded_binding(recorded, {**current, 'rows': 32, 'seed_base': 5}, 1, 'crisp')
    assert strict == ['rows', 'seed_base']
    # a round bound before teacher attempts existed keeps its behaviour even when endpoints were recorded
    old = {k: v for k, v in recorded.items() if not k.startswith('jolt')}
    strict, _, teachers = production.compare_recorded_binding(old, current, 1, 'crisp')
    assert strict == [] and teachers == 0


def test_lost_block_exclusions_drop_only_named_scored_rollouts_and_record_them(tmp_path, monkeypatch):
    h = Harness(tmp_path, monkeypatch, entries=2)
    h.budget = 99
    rows = [{'group': 0, 'sample': 0, 'reward': 1.}, {'group': 0, 'sample': 1, 'reward': 0.}]
    monkeypatch.setattr(production, 'score', lambda out, emit, **kw: (
        Path(emit).write_text(''.join(json.dumps(r) + '\n' for r in rows)), {'scored': 2})[1])
    lost = {'reason': 'actor store deleted before bundling', 'samples': [{'group': 0, 'sample': 1}]}
    report = h.collect(lost_block_exclusions=lost)
    assert [r['sample'] for r in read_jsonl(h.out / 'rollouts.jsonl')] == [0]
    assert len(list(read_jsonl(h.out / 'rollouts.lost-block-unfiltered.jsonl'))) == 2
    assert report['lost_block_exclusions']['excluded'] == [[0, 1]]
    (h.out / 'score.json').unlink()
    with pytest.raises(ValueError, match='lost-block exclusions'):
        h.collect(lost_block_exclusions={'reason': 'x', 'samples': [{'group': 3, 'sample': 0}]})


def test_dispatch_reports_initial_attempted_finished_and_remaining(tmp_path):
    entries = [{'key': f'k{i}'} for i in range(4)]

    def run_worker(index, name, queue, journal):
        taken = [json.loads(line)['key'] for line in queue.read_text().splitlines()]
        with journal.open('a') as stream:   # finishes the first entry, then the collector dies on the second
            stream.write(json.dumps({'event': 'finish', 'key': taken[-1]}) + '\n') if len(taken) == 1 else None
        return 1

    results, counts = dispatch(entries, directory=tmp_path, generation=1, phase='', endpoints=['a'],
                               run_worker=run_worker, attempted={'k0'})
    # k0 was finished earlier; one slot takes k1 (finished) and k2 (left without a finish row) and stops there
    assert counts == {'initial': 3, 'attempted': 2, 'finished': 1, 'remaining': 2}
    assert results[0]['entries'] == 2


def test_teacher_plan_is_not_fixed_on_an_incomplete_student_sample(tmp_path, monkeypatch):
    h = Harness(tmp_path, monkeypatch, entries=4)
    h.budget = 2
    planned = []
    from natlang_neuralese.rl import jolt
    monkeypatch.setattr(jolt, 'teacher_entries', lambda *a, **k: planned.append(1) or {'entries': [], 'refusal': None})
    monkeypatch.setattr(jolt, 'student_guidance', lambda *a, **k: None)
    with pytest.raises(IncompleteCollection, match='2 of 4'):
        h.collect(teacher_attempts=1, teacher_overlap=False)   # the opt-out plans only a complete sample
    assert planned == [] and not (h.out / 'teacher-plan.json').exists()


def test_overlap_plans_only_the_groups_whose_students_all_finished(tmp_path, monkeypatch):
    from natlang_neuralese.rl import jolt
    h = Harness(tmp_path, monkeypatch, entries=4)
    h.budget = 2
    seen = []
    monkeypatch.setattr(jolt, 'teacher_entries', lambda students, rows, **k: seen.append(sorted({e['group'] for e in students}))
                        or {'entries': [], 'refusal': None, 'plan': {'implementation': 'crisp'}})
    monkeypatch.setattr(jolt, 'student_guidance', lambda *a, **k: None)
    with pytest.raises(IncompleteCollection, match='2 of 4'):
        h.collect(teacher_attempts=1, teacher_plan='crisp', slot_poll_seconds=0.02)
    assert sorted(g for batch in seen for g in batch) == sorted(
        int(key.rsplit('-', 1)[1]) for key in finishes(h.out))   # exactly the finished students' groups
