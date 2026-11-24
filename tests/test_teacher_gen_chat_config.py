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
    assert teacher_gen.teacher_identity(luna) == {'model_id': 'gpt-6-luna', 'provider': 'openai-codex', 'reasoning_effort': 'low',
                                                  'reasoning': 'summary'}
    http = teacher_gen.make_teacher({'model_id': 'm', 'endpoint': 'https://openrouter.ai/api/v1/', 'api_key_env': 'K'})
    command = teacher_gen.queue_command(http, tmp_path, tmp_path / 'q.jsonl', tmp_path / 'j.jsonl')
    assert command[command.index('--chat-completions-url') + 1] == 'https://openrouter.ai/api/v1/chat/completions'
    assert command[command.index('--api-key-env') + 1] == 'K' and '--provider' not in command
    assert teacher_gen.teacher_identity(http) == {'model_id': 'm', 'provider': 'openrouter',
                                                  'endpoint': 'https://openrouter.ai/api/v1', 'reasoning': 'raw'}
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
    rates = teacher_gen.student_rates({'rare': {'successes': 1, 'scored': 4}, 'band': {'successes': 0, 'scored': 9}},
                                      [str(tmp_path / 'run/groups/g*')])
    # the newest measurement wins: the lineage's groups replace the base rate of 'rare'
    assert rates['often'] == (2, 2) and rates['rare'] == (0, 1) and rates['band'] == (0, 9)
    select = tmp_path / 'tasks.rates.json'
    select.write_text(json.dumps({'band': {'successes': 1, 'scored': 2}, 'rare': {'successes': 2, 'scored': 2},
                                  'untried-measured': {'successes': 3, 'scored': 4}}))
    frontier = tmp_path / 'frontier.json'
    frontier.write_text(json.dumps({'tasks': {'often': {'successes': 3, 'scored': 4, 'tier': 'sometimes'}}}))
    rates = teacher_gen.student_rates({}, [str(tmp_path / 'run/groups/g*')], [select], [frontier])
    assert rates['band'] == (1, 2) and rates['rare'] == (0, 1) and rates['often'] == (3, 4)
    rank = teacher_gen.gap_rank(coverage, rates, {'supply', 'supply-often'}, 0.25)
    tiers = {i: rank({'id': i}) for i in ('solved', 'failed', 'rare', 'often', 'supply', 'supply-often', 'untried',
                                          'band', 'untried-measured')}
    # rare and luna failed it: last; 'often' at 3/4 is in the band, luna failed it: teacher_failed
    assert tiers == {'solved': None, 'failed': 4, 'rare': 5, 'often': 4, 'supply': 1, 'supply-often': 1,
                     'untried': 3, 'band': 2, 'untried-measured': 2}
    assert teacher_gen.GAP_TIERS[0] == 'student_rare' and teacher_gen.GAP_TIERS[-1] == 'student_rare_failed'
    solved = teacher_gen.gap_rank(coverage, {'x': (9, 10)}, set(), 0.25)
    assert solved({'id': 'x'}) is None and teacher_gen.gap_rank({}, {'x': (0, 3)}, set(), 0.25)({'id': 'x'}) == 0
    rows = [{'id': i, 'family': f} for i, f in (('failed', 'a'), ('rare', 'b'), ('supply', 'a'), ('untried', 'c'),
                                                ('solved', 'a'), ('band', 'b'))]
    assert [r['id'] for r in teacher_gen.order_tasks(rows, rank, 1)] == ['supply', 'band', 'untried', 'failed', 'rare']
    # within a tier the tasks fewer teachers failed come first
    many = {'id': 'many', 'family': 'a'}
    order = teacher_gen.order_tasks([many, {'id': 'one', 'family': 'a'}], lambda row: 4,
                                    1, sub=lambda row: 3 if row['id'] == 'many' else 1)
    assert [r['id'] for r in order] == ['one', 'many']


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


def test_vertex_gemini_teacher_mints_its_token_per_episode_and_records_hidden_reasoning():
    import sys
    import pytest
    vertex = 'https://aiplatform.googleapis.com/v1/projects/p/locations/global/endpoints/openapi'
    teacher = teacher_gen.make_teacher({'model_id': 'google/gemini-3.1-pro-preview', 'endpoint': vertex,
                                        'api_key_env': 'VERTEX_ACCESS_TOKEN',
                                        'api_key_command': [sys.executable, '-c', 'print("tok-1")']})
    assert teacher.chat_config == teacher_gen.default_chat_config(vertex)
    assert teacher.chat_config['x_natlang_omit_seed'] is True
    identity = teacher_gen.teacher_identity(teacher)
    assert (identity['provider'], identity['reasoning']) == ('google-vertex', 'hidden')
    env = teacher_gen.episode_env(teacher, {'A': '1'})
    assert env == {'A': '1', 'VERTEX_ACCESS_TOKEN': 'tok-1'}
    plain = teacher_gen.make_teacher({'model_id': 'm', 'endpoint': 'https://integrate.api.nvidia.com/v1'})
    assert teacher_gen.episode_env(plain, {'A': '1'}) == {'A': '1'}
    assert teacher_gen.teacher_identity(plain)['reasoning'] == 'raw'
    override = teacher_gen.make_teacher({'model_id': 'm', 'endpoint': vertex, 'reasoning': 'summary'})
    assert teacher_gen.teacher_identity(override)['reasoning'] == 'summary'
    for bad in ({'model_id': 'm', 'endpoint': vertex, 'reasoning': 'thoughts'},
                {'model_id': 'm', 'endpoint': vertex, 'api_key_command': ['gcloud']},       # no api_key_env
                {'model_id': 'm', 'provider': 'openai-codex', 'api_key_command': ['x'], 'api_key_env': 'K'}):
        with pytest.raises(ValueError):
            teacher_gen.make_teacher(bad)
    empty = teacher_gen.make_teacher({'model_id': 'm', 'endpoint': vertex, 'api_key_env': 'K',
                                      'api_key_command': [sys.executable, '-c', 'pass']})
    with pytest.raises(RuntimeError):
        teacher_gen.episode_env(empty, {})


def test_an_account_without_balance_retires_its_teacher():
    assert any(m in '{"error":{"code":"1113","message":"Insufficient balance or no resource package."}}'
               for m in teacher_gen.PERMANENT_PROVIDER_FAILURES)


def test_completion_urls_keep_the_api_base_path_and_nim_sends_one_reasoning_alias():
    url = teacher_gen.completions_url
    assert url('http://127.0.0.1:8082') == 'http://127.0.0.1:8082/v1/chat/completions'
    assert url('https://integrate.api.nvidia.com/v1/') == 'https://integrate.api.nvidia.com/v1/chat/completions'
    assert url('https://openrouter.ai/api/v1') == 'https://openrouter.ai/api/v1/chat/completions'
    assert url('https://api.z.ai/api/paas/v4') == 'https://api.z.ai/api/paas/v4/chat/completions'
    assert url('https://generativelanguage.googleapis.com/v1beta/openai') == \
        'https://generativelanguage.googleapis.com/v1beta/openai/chat/completions'
    vertex = 'https://aiplatform.googleapis.com/v1/projects/p/locations/global/endpoints/openapi'
    assert url(vertex) == vertex + '/chat/completions'
    assert teacher_gen.default_chat_config(url('https://integrate.api.nvidia.com/v1')) == {
        'top_p': 0.95, 'x_natlang_history_reasoning': 'reasoning_content'}
