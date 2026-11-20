import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'neuralese_data'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training' / 'neuralese'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import teacher_gen  # noqa: E402


def record(index, family, model, outcome):
    return {'index': index, 'task_id': f't{index}', 'family': family, 'teacher': {'model_id': model},
            'episodes': [{'outcome': outcome, 'reward': 1.0 if outcome == 'task_success' else 0.0}]}


def campaign(tmp_path, tasks, earlier=(), own=(), teachers=(), **flags):
    old = tmp_path / 'old'
    old.mkdir(parents=True, exist_ok=True)
    (old / 'manifest.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in earlier))
    work = tmp_path / 'work'
    work.mkdir(parents=True, exist_ok=True)
    (work / 'tasks.jsonl').write_text(''.join(json.dumps(t) + '\n' for t in tasks))
    (work / 'manifest.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in own))
    args = SimpleNamespace(work=str(work), covered_by=[str(old)], attempts=2, route_ratio=2.0, route_floor=0.03,
                           route_min_scored=30, limit=0, shard=(0, 1), pause_below_available_gb=0, provider_cooldown=0,
                           **flags)
    return teacher_gen.Generator(args, list(teachers))


def teacher(model, **spec):
    return teacher_gen.make_teacher({'model_id': model, 'provider': 'p', **spec})


def task(family, failed_by=()):
    return {'row': {'id': 'x', 'family': family}, 'base_rank': 'teacher_failed',
            **({'failed_by': list(failed_by)} if failed_by else {})}


def test_family_routing_sends_a_family_to_the_teacher_that_solves_it(tmp_path):
    earlier = ([record(100 + i, 'read', 'strong', 'task_success') for i in range(20)] +
               [record(200 + i, 'read', 'weak', 'semantic_mismatch') for i in range(20)] +
               [record(300 + i, 'math', 'weak', 'task_success') for i in range(20)] +
               [record(400 + i, 'math', 'strong', 'task_success') for i in range(20)])
    weak, strong = teacher('weak'), teacher('strong')
    g = campaign(tmp_path, [task('read'), task('math'), task('read')], earlier, teachers=[weak, strong])
    assert g._take(weak) == 1                  # the read tasks are left for strong
    assert g.passed == [0]
    assert g._take(strong) == 0
    assert g._take(strong) == 2
    assert g._take(weak) is None


def test_a_failed_task_goes_on_to_another_teacher_and_keeps_both_episodes(tmp_path):
    a, b = teacher('a'), teacher('b')
    g = campaign(tmp_path, [task('f')], own=[record(0, 'f', 'a', 'semantic_mismatch')], teachers=[a, b])
    assert 0 not in g.done and g.tasks[0]['failed_by'] == ['a']
    claims = tmp_path / 'work' / 'claims'
    (claims / 't000000').write_text(json.dumps({'pid': 1, 'teacher': 'a', 'state': 'done'}))
    assert g._take(a) is None                  # a tried it here already
    assert g._take(b) == 0                     # its done claim was one verdict, not the end
    g.tried[0].add('b')
    assert g._finished(0)                      # --attempts 2 teachers
    solved = campaign(tmp_path / 's', [task('f')], own=[record(0, 'f', 'a', 'task_success')], teachers=[a, b])
    assert 0 in solved.done


def test_min_failed_reserves_a_teacher_for_tasks_others_failed(tmp_path):
    cheap, sol = teacher('cheap'), teacher('sol', min_failed=1)
    g = campaign(tmp_path, [task('f'), task('f', ['cheap'])], teachers=[cheap, sol])
    assert g._take(sol) == 1
    assert g._take(cheap) == 0
    assert g._take(sol) is None


def test_hopeless_family_is_left_alone(tmp_path):
    earlier = [record(i, 'f', 'bad', 'semantic_mismatch') for i in range(60)]
    bad = teacher('bad')
    g = campaign(tmp_path, [task('f')], earlier, teachers=[bad])
    assert g._take(bad) is None and g._finished(0)


def test_refused_account_retires_the_teacher_not_the_task(tmp_path):
    assert any('credit balance is too low' in m for m in teacher_gen.PERMANENT_PROVIDER_FAILURES)
    a, b = teacher('a'), teacher('b')
    g = campaign(tmp_path, [task('f')], teachers=[a, b])
    a.enabled = False
    assert g._take(a) is None and g._eligible(0, b)
