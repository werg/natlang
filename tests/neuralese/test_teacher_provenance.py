import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / 'scripts' / 'neuralese_data'), str(REPO / 'training' / 'neuralese'), str(REPO)]

import teacher_provenance as tp  # noqa: E402


def _zst(path, rows):
    import zstandard
    path.write_bytes(zstandard.ZstdCompressor().compress(''.join(json.dumps(r) + '\n' for r in rows).encode()))


def test_old_endpoint_records_get_their_provider(tmp_path):
    pool = tmp_path / 'campaign' / 'pool-v1'
    (pool / 'episodes' / 't000001').mkdir(parents=True)
    records = [
        {'task_id': 'a', 'teacher': {'model_id': 'nvidia/nemotron-3-ultra-550b-a55b', 'endpoint': 'https://integrate.api.nvidia.com/v1'},
         'episodes': [{'sample': 0, 'outcome': 'task_success', 'reward': 1.0, 'turns': 3}],
         'rollouts': str(pool / 'episodes/t000001/rollouts.jsonl')},
        {'task_id': 'b', 'teacher': {'model_id': 'claude-haiku-5-5', 'provider': 'anthropic', 'reasoning_effort': 'medium'},
         'episodes': [{'sample': 0, 'outcome': 'semantic_mismatch', 'reward': 0.0, 'turns': 2}],
         'rollouts': str(pool / 'episodes/t000002/rollouts.jsonl')}]
    (pool / 'manifest.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
    rows = list(tp.scan_pool(pool, tmp_path))
    assert [(r['model_id'], r['provider'], r['success']) for r in rows] == [
        ('nvidia/nemotron-3-ultra-550b-a55b', 'nvidia-nim', True), ('claude-haiku-5-5', 'anthropic', False)]
    assert rows[0]['key'] == 'teacher:campaign/pool-v1:t000001:0'


def test_episodes_and_october_rows_carry_model_provider_and_reasoning(tmp_path):
    turn = lambda reasoning, tool: {'target': {'reasoning_content': reasoning,
                                               'tool_calls': [{'function': {'name': tool}}]}}
    _zst(tmp_path / 'episodes.jsonl.zst', [
        {'source': 'rl-own', 'run': 'rl:x', 'key': 'k0', 'turns': []},
        {'source': 'teacher', 'run': 'teacher:c/p', 'key': 'k1', 'task_id': 't',
         'teacher': {'model_id': 'gemini-3.5-flash-lite', 'endpoint': 'https://generativelanguage.googleapis.com/v1beta/openai'},
         'turns': [turn('abc', 'eval'), turn('', 'return_result')]}])
    rows = list(tp.scan_episodes(tmp_path / 'episodes.jsonl.zst', tmp_path))
    assert len(rows) == 1 and rows[0]['provider'] == 'google-gemini'
    assert rows[0]['reasoning_chars'] == [3, 0] and rows[0]['tools'] == ['eval', 'return_result']
    october = tmp_path / 'october'
    october.mkdir()
    _zst(october / 'qwen-s72.jsonl.zst', [
        {'task_id': 't', 'target': {'reasoning_content': 'x' * 10, 'tool_calls': [{'function': {'name': 'eval'}}]},
         'provenance': {'model': 'nvidia/Qwen3.6-35B-A3B-NVFP4', 'trajectory': 'tr1', 'source_file': 'f',
                        'outcome': {'accepted': True}}},
        {'task_id': 't', 'target': {'content': 'done'},
         'provenance': {'model': 'nvidia/Qwen3.6-35B-A3B-NVFP4', 'trajectory': 'tr1', 'source_file': 'f'}}])
    [traj] = list(tp.scan_october(october, tmp_path))
    assert (traj['model_id'], traj['provider'], traj['calls'], traj['reasoning_chars'], traj['success']) == (
        'nvidia/Qwen3.6-35B-A3B-NVFP4', 'self-hosted-vllm-dgx', 2, [10, 0], True)
    [summary] = [s for s in tp.summarize([traj]) if s['model_id'] == 'nvidia/Qwen3.6-35B-A3B-NVFP4']
    assert summary['calls_with_reasoning'] == 1 and summary['reasoning_chars_p50'] == 10
    assert summary['missing_provider'] == 0
