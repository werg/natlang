import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'neuralese_data'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training' / 'neuralese'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import teacher_gen  # noqa: E402


def test_gemini_default_omits_the_seed_field():
    gemini = 'https://generativelanguage.googleapis.com/v1beta/openai/v1/chat/completions'
    assert teacher_gen.default_chat_config(gemini) == {
        'top_p': 0.95, 'x_natlang_omit_seed': True, 'x_natlang_host_call_fields': teacher_gen.GEMINI_HOST_CALL_FIELDS}
    assert teacher_gen.default_chat_config('https://openrouter.ai/api/v1/chat/completions') == {'top_p': 0.95}


def test_provider_teacher_runs_the_pi_transport_without_http_controls(tmp_path):
    import pytest
    luna = teacher_gen.make_teacher({'model_id': 'gpt-6-luna', 'provider': 'openai-codex', 'slots': 3})
    assert (luna.slots, luna.reasoning_effort, luna.endpoint) == (3, 'low', None)
    command = teacher_gen.queue_command(luna, tmp_path, tmp_path / 'q.jsonl', tmp_path / 'j.jsonl')
    assert command[command.index('--provider') + 1] == 'openai-codex'
    assert command[command.index('--model-id') + 1] == 'gpt-6-luna'
    assert command[command.index('--reasoning-effort') + 1] == 'low'
    assert not {'--server', '--chat-completions-url', '--api-key-env', '--chat-request-config'} & set(command)
    assert teacher_gen.teacher_identity(luna) == {'model_id': 'gpt-6-luna', 'provider': 'openai-codex', 'reasoning_effort': 'low'}
    http = teacher_gen.make_teacher({'model_id': 'm', 'endpoint': 'https://openrouter.ai/api/v1/', 'api_key_env': 'K'})
    command = teacher_gen.queue_command(http, tmp_path, tmp_path / 'q.jsonl', tmp_path / 'j.jsonl')
    assert command[command.index('--chat-completions-url') + 1] == 'https://openrouter.ai/api/v1/chat/completions'
    assert command[command.index('--api-key-env') + 1] == 'K' and '--provider' not in command
    assert teacher_gen.teacher_identity(http) == {'model_id': 'm', 'provider': 'openrouter',
                                                  'endpoint': 'https://openrouter.ai/api/v1'}
    for bad in ({'model_id': 'm'}, {'model_id': 'm', 'endpoint': 'http://x', 'provider': 'openai-codex'},
                {'model_id': 'm', 'provider': 'openai-codex', 'api_key_env': 'K'}):
        with pytest.raises(ValueError):
            teacher_gen.make_teacher(bad)


def _manifest(work, records):
    import json
    work.mkdir(parents=True, exist_ok=True)
    (work / 'manifest.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))


def _record(task_id, teacher, outcome, reward):
    return {'task_id': task_id, 'teacher': {'model_id': teacher},
            'episodes': [{'outcome': outcome, 'reward': reward}]}


def test_gap_selection_tiers_and_failed_by(tmp_path):
    import json
    _manifest(tmp_path / 'old', [_record('solved', 'haiku', 'task_success', 1.0),
                                 _record('failed', 'haiku', 'semantic_mismatch', 0.0),
                                 _record('rare', 'luna', 'budget_exhaustion', 0.0),
                                 _record('often', 'luna', 'semantic_mismatch', 0.0)])
    coverage = teacher_gen.teacher_coverage([tmp_path / 'old'])
    assert coverage['solved']['success'] and coverage['failed']['failed_by'] == {'haiku'}
    group = tmp_path / 'run' / 'groups' / 'g000001'
    group.mkdir(parents=True)
    (group / 'rewards.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in [
        {'task_id': 'often', 'reward': 1.0}, {'task_id': 'often', 'reward': 1.0},
        {'task_id': 'often', 'reward': 0.0, 'jolt_role': 'teacher'},
        {'task_id': 'supply-often', 'reward': 1.0}, {'task_id': 'rare', 'reward': 0.0}]))
    rates = teacher_gen.student_rates({'rare': {'successes': 1, 'scored': 4}}, [str(tmp_path / 'run/groups/g*')])
    assert rates['often'] == (2, 2) and rates['rare'] == (1, 5)
    rank = teacher_gen.gap_rank(coverage, rates, {'supply', 'supply-often'}, 0.25)
    tiers = {i: rank({'id': i}) for i in ('solved', 'failed', 'rare', 'often', 'supply', 'supply-often', 'untried')}
    assert tiers == {'solved': None, 'failed': 3, 'rare': 0, 'often': None, 'supply': 1, 'supply-often': 1,
                     'untried': 2}
    rows = [{'id': i, 'family': f} for i, f in (('failed', 'a'), ('rare', 'b'), ('supply', 'a'), ('untried', 'c'),
                                                ('solved', 'a'))]
    assert [r['id'] for r in teacher_gen.order_tasks(rows, rank, 1)] == ['rare', 'supply', 'untried', 'failed']


def test_task_a_teacher_failed_goes_to_another_teacher_first(tmp_path):
    import json
    from types import SimpleNamespace
    tasks = [{'row': {'id': 'x'}, 'base_rank': 'teacher_failed', 'failed_by': ['haiku']},
             {'row': {'id': 'y'}, 'base_rank': 'teacher_failed'},
             {'row': {'id': 'z'}, 'base_rank': 'teacher_failed', 'failed_by': ['haiku', 'luna']}]
    (tmp_path / 'tasks.jsonl').write_text(''.join(json.dumps(t) + '\n' for t in tasks))
    haiku, luna = (teacher_gen.make_teacher({'model_id': m, 'provider': 'p'}) for m in ('haiku', 'luna'))
    generator = teacher_gen.Generator(SimpleNamespace(work=str(tmp_path), limit=0, shard=(0, 1), covered_by=[], attempts=2,
                                                      route_ratio=2.0, route_floor=0.03, route_min_scored=30), [haiku, luna])
    assert generator._take(haiku) == 1          # x passed on, y taken
    assert generator._take(luna) == 0           # luna takes the passed x first
    assert generator._take(haiku) == 2          # every teacher failed z: no one to pass it to
    assert generator._take(luna) is None


def test_every_teacher_identity_names_its_provider():
    for endpoint, provider in (('https://integrate.api.nvidia.com/v1', 'nvidia-nim'),
                               ('https://generativelanguage.googleapis.com/v1beta/openai', 'google-gemini'),
                               ('https://api.z.ai/api/paas/v4', 'zai'), ('http://127.0.0.1:8082/v1', 'local:8082'),
                               ('https://example.org/v1', 'example.org')):
        teacher = teacher_gen.make_teacher({'model_id': 'm', 'endpoint': endpoint})
        identity = teacher_gen.teacher_identity(teacher)
        assert identity['provider'] == provider and identity['model_id'] == 'm' and identity['endpoint'] == endpoint
