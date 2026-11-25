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
    args = SimpleNamespace(**{'work': str(work), 'covered_by': [str(old)], 'attempts': 2, 'route_ratio': 2.0,
                              'route_floor': 0.03, 'route_min_scored': 30, 'limit': 0, 'shard': (0, 1),
                              'pause_below_available_gb': 0, 'provider_cooldown': 0, **flags})
    return teacher_gen.Generator(args, list(teachers))


def teacher(model, **spec):
    return teacher_gen.make_teacher({'model_id': model, 'provider': 'p', **spec})


def task(family, failed_by=(), tier='teacher_failed'):
    return {'row': {'id': 'x', 'family': family}, 'base_rank': tier,
            **({'failed_by': list(failed_by)} if failed_by else {})}


def test_family_routing_sends_a_family_to_the_teacher_that_solves_it(tmp_path):
    earlier = ([record(100 + i, 'read', 'strong', 'task_success') for i in range(20)] +
               [record(200 + i, 'read', 'weak', 'semantic_mismatch') for i in range(20)] +
               [record(300 + i, 'math', 'weak', 'task_success') for i in range(20)] +
               [record(400 + i, 'math', 'strong', 'task_success') for i in range(20)])
    weak, strong = teacher('weak'), teacher('strong')
    g = campaign(tmp_path, [task('read'), task('math'), task('read')], earlier, teachers=[weak, strong])
    assert g._take(weak) == 1                  # the read tasks are left for strong
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
    earlier = [{**record(9, 'f', 'cheap', 'semantic_mismatch'), 'task_id': 'x'}]
    g = campaign(tmp_path, [task('f'), task('f', ['cheap'])], earlier, teachers=[cheap, sol])
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


def test_a_harness_infrastructure_error_is_no_teacher_verdict(tmp_path):
    g = campaign(tmp_path, [task('f')], teachers=[teacher('a')])
    crash = {**record(0, 'f', 'a', 'infrastructure_error'), 'provider_failure': False}
    assert g._no_verdict(crash)
    assert not g._no_verdict(record(0, 'f', 'a', 'budget_exhaustion'))


def test_a_usage_limit_hit_mid_episode_is_no_teacher_verdict(tmp_path):
    # a nested child call refused by the subscription's usage window makes the root fail: the provider's verdict
    g = campaign(tmp_path, [])
    limited = {**record(0, 'f', 'a', 'semantic_mismatch'), 'provider_limited': True}
    assert g._no_verdict(limited)
    assert not g._no_verdict({**record(0, 'f', 'a', 'task_success'), 'provider_limited': True})
    assert any(limit in 'Error: You have hit your ChatGPT usage limit.' for limit in teacher_gen.TEMPORARY_PROVIDER_LIMITS)


def test_voided_verdicts_are_no_try_and_keep_their_episode_directory(tmp_path):
    failed = {**record(0, 'f', 'a', 'semantic_mismatch'), 'rollouts': '/x/t000000/rollouts.jsonl'}
    earlier = [{**record(5, 'f', 'b', 'semantic_mismatch'), 'task_id': 'x', 'rollouts': '/old/t5/rollouts.jsonl'}]
    a, b = teacher('a'), teacher('b')
    g0 = campaign(tmp_path, [task('f', ['b'])], earlier, own=[failed], teachers=[a, b])
    assert g0.tasks[0]['failed_by'] == ['a', 'b'] and 'a' in g0.tried[0]
    (tmp_path / 'work' / 'voided.jsonl').write_text(json.dumps({'rollouts': failed['rollouts']}) + '\n')
    (tmp_path / 'old' / 'voided.jsonl').write_text(json.dumps({'rollouts': '/old/t5/rollouts.jsonl'}) + '\n')
    g = campaign(tmp_path, [task('f', ['b'])], earlier, own=[failed], teachers=[a, b])
    assert g.tasks[0]['failed_by'] == [] and 0 not in g.tried and g.recorded[0] == 1
    assert g._take(a) == 0                     # a may try again; its next episodes go to t000000-a1


def fresh(index, model, outcome):
    return {**record(index, 'old', model, outcome), 'base_rank': 'teacher_untried'}


def test_a_weak_teacher_leaves_the_student_rare_tiers_to_the_strong_ones(tmp_path):
    """2026-11-24: on the aei1 student-rare tasks Nemotron scored 0/99 while Luna and Sol solved 42% of those no teacher
    had failed; a weak teacher's failure there spends one of the task's attempts and demotes it for later campaigns."""
    earlier = ([fresh(i, 'nemotron', 'semantic_mismatch') for i in range(40)] + [fresh(50, 'nemotron', 'task_success')] +
               [fresh(100 + i, 'luna', 'task_success' if i % 2 else 'semantic_mismatch') for i in range(40)])
    nemotron, luna = teacher('nemotron'), teacher('luna')
    tasks = [task('f', tier='student_rare'), task('f', tier='student_band'), task('f', tier='teacher_untried'),
             task('f', tier='student_rare_failed')]
    g = campaign(tmp_path, tasks, earlier, teachers=[nemotron, luna], weak_yield=0.25, route_ratio=0)
    assert g.weak(nemotron) and not g.weak(luna)
    assert g.strength(nemotron) == ((1 + 0.3 * 5) / 46, 41)
    assert g._take(nemotron) == 1              # the band task: the student-rare one is luna's
    assert g._take(luna) == 0                  # the strong teacher takes the hard task first
    assert g._take(nemotron) == 2 and g._take(nemotron) is None   # never the student_rare_failed one
    assert g._take(luna) == 3
    off = campaign(tmp_path / 'off', tasks, earlier, teachers=[nemotron, luna], weak_yield=0, route_ratio=0)
    assert off._take(nemotron) == 0


def test_the_standing_queue_has_no_end_but_its_tasks_and_drains_on_request(tmp_path):
    """No --limit: every ordered task; a deferred task waits for its retry time at its place in the queue; a task a
    teacher failed here stays at its place for the next teacher; WORK/DRAIN stops the slots."""
    a, b = teacher('a'), teacher('b')
    g = campaign(tmp_path, [task('f') for _ in range(5)], teachers=[a, b], weak_yield=0.25)
    assert [g._take(a) for _ in range(5)] == [0, 1, 2, 3, 4]
    g.active.clear()
    g.retry[0] = teacher_gen.time.time() + 600
    g.tried[1] = {'a'}
    g.tasks[1]['failed_by'] = ['a']
    g.done.update({2, 3, 4})
    for index in range(5):
        (tmp_path / 'work' / 'claims' / f't{index:06d}').unlink(missing_ok=True)
    assert g._take(a) is None and g._take(b) == 1
    g.retry[0] = 0
    assert g._take(a) == 0 and 0 not in g.retry
    (tmp_path / 'work' / 'DRAIN').touch()
    g.slot(a)                                   # returns at once: takes nothing
    assert g.running == 0


GEMINI_FREE_429 = ('You exceeded your current quota, please check your plan and billing details. \\n* Quota exceeded for '
                   'metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: {n}, model: '
                   'gemini-3.5-flash-lite, description: Generate content requests per minute limit exceeded on Free Tier.'
                   '\\nPlease retry in 41.2s.')


def test_a_free_tier_quota_window_is_a_temporary_limit_not_a_refused_account():
    # AI Studio's per-model free-tier 429 shares OpenAI's spent-account text; with a nonzero limit it passes
    window = GEMINI_FREE_429.format(n=15)
    assert teacher_gen.provider_refusal(window) is None and teacher_gen.provider_limit(window)
    # no free quota at all for the model (limit 0) and OpenAI's spent account stay refusals
    assert teacher_gen.provider_refusal(GEMINI_FREE_429.format(n=0)) == 'exceeded your current quota'
    assert not teacher_gen.provider_limit(GEMINI_FREE_429.format(n=0))
    openai = 'You exceeded your current quota, please check your plan and billing details. insufficient_quota'
    assert teacher_gen.provider_refusal(openai) and not teacher_gen.provider_limit(openai)
    assert teacher_gen.provider_limit('Error: You have hit your ChatGPT usage limit.')


def test_a_teacher_key_comes_from_the_provider_key_files_when_not_sourced(tmp_path, monkeypatch):
    (tmp_path / 'groq.env').write_text('export GROQ_API_KEY="k-groq"\n')
    (tmp_path / 'other.env').write_text('OTHER=1\n')
    monkeypatch.setattr(teacher_gen, 'CONFIG_ENV_DIR', tmp_path)
    assert teacher_gen.config_env_key('GROQ_API_KEY', tmp_path) == 'k-groq'
    assert teacher_gen.config_env_key('MISSING', tmp_path) is None
    t = teacher_gen.make_teacher({'model_id': 'm', 'endpoint': 'https://api.groq.com/openai/v1', 'api_key_env': 'GROQ_API_KEY'})
    assert teacher_gen.episode_env(t, {})['GROQ_API_KEY'] == 'k-groq'
    assert teacher_gen.episode_env(t, {'GROQ_API_KEY': 'set'})['GROQ_API_KEY'] == 'set'
