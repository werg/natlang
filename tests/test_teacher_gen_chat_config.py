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
    assert teacher_gen.teacher_identity(http) == {'model_id': 'm', 'endpoint': 'https://openrouter.ai/api/v1'}
    for bad in ({'model_id': 'm'}, {'model_id': 'm', 'endpoint': 'http://x', 'provider': 'openai-codex'},
                {'model_id': 'm', 'provider': 'openai-codex', 'api_key_env': 'K'}):
        with pytest.raises(ValueError):
            teacher_gen.make_teacher(bad)
