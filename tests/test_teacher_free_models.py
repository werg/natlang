import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training' / 'neuralese'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import teacher_free_models as t  # noqa: E402


def test_new_models_are_open_candidates_and_a_missing_key_records_the_signup(tmp_path):
    catalogue = {'a': {'tools': True}}

    def failing():
        raise RuntimeError('down')

    def keyless():
        raise t.MissingKey('cerebras')
    providers = {'or': lambda: dict(catalogue), 'down': failing, 'cerebras': keyless}
    first = t.check(tmp_path, providers, now='T1')
    assert first['new'] == ['or:a'] and 'error' in first['providers']['down']
    assert first['providers']['cerebras']['needs_key'] and 'cloud.cerebras.ai' in first['providers']['cerebras']['signup']
    catalogue['b:free'] = {'tools': True, 'reasoning': True}
    second = t.check(tmp_path, providers, now='T2')
    assert second['new'] == ['or:b:free']
    assert [r['model'] for r in t.open_candidates(tmp_path)] == ['a', 'b:free']
    del catalogue['a']
    third = t.check(tmp_path, providers, now='T3')
    assert third['new'] == [] and third['providers']['or']['gone'] == ['a']
    t.mark('b:free', 'qualified', note='3/20', out=tmp_path)
    assert [r['model'] for r in t.open_candidates(tmp_path)] == ['a']
    rows = [json.loads(line) for line in (tmp_path / 'candidates.jsonl').open()]
    assert rows[1]['status'] == 'qualified' and rows[1]['note'] == '3/20'


def test_openrouter_free_means_the_free_suffix_or_zero_prices():
    assert t.openrouter_free({'id': 'x/y:free', 'pricing': {'prompt': '0.1'}})
    assert t.openrouter_free({'id': 'x/y', 'pricing': {'prompt': '0', 'completion': '0'}})
    assert not t.openrouter_free({'id': 'x/y', 'pricing': {'prompt': '0', 'completion': '0.2'}})


def test_free_filters_of_public_listings():
    assert t.zero_price({'pricing': {'prompt': 0, 'completion': '0'}})
    assert not t.zero_price({'pricing': {'prompt': 0.039, 'completion': 1.25}}) and not t.zero_price({})
    assert t.FREE_FILTERS['hf-is-free']({'providers': [{'is_free': True, 'supports_tools': True}]})
    assert not t.FREE_FILTERS['hf-is-free']({'providers': [{'is_free': False, 'supports_tools': True}]})
    assert t.FREE_FILTERS['llm7-tier']({'tier': 'turbo'}) and not t.FREE_FILTERS['llm7-tier']({'tier': 'pro'})


def test_the_screen_keeps_strong_chat_models_and_drops_small_or_unusable_ones():
    def row(provider, model, **kw):
        return {'provider': provider, 'model': model, **kw}
    assert t.screen(row('ai-studio', 'gemini-3.8-flash')) is None
    assert t.screen(row('nvidia-nim', 'nvidia/nemotron-3.5-super-vl-120b-a12b')) is None
    assert t.screen(row('nvidia-nim', 'nvidia/nemotron-3.5-lightning-30b-a3b')) is None
    assert 'small' in t.screen(row('nvidia-nim', 'google/gemma-3-12b-it'))
    assert t.screen(row('openrouter', 'x/embed-large:free'))
    assert 'tool' in t.screen(row('openrouter', 'x/y:free', tools=False))
    assert 'tokens/min' in t.screen(row('groq', 'openai/gpt-oss-120b'))


def test_probe_outcomes(monkeypatch):
    monkeypatch.setattr(t, 'provider_key', lambda provider: 'k')
    row = {'provider': 'opencode-zen', 'model': 'big-pickle'}
    replies = {'free': (429, '{"error":{"type":"FreeUsageLimitError"}}', {}),
               'busy': (503, 'high demand', {}),
               'tool': (200, {'choices': [{'message': {'tool_calls': [{'function': {'name': 'calc'}}]}}]}, {}),
               'text': (200, {'choices': [{'message': {'content': '391'}}]}, {}),
               'kilo': (429, 'PROMOTION_MODEL_LIMIT_REACHED', {})}
    for name, expected in (('free', False), ('busy', None), ('tool', True), ('text', False), ('kilo', t.NEEDS_KEY)):
        monkeypatch.setattr(t, 'post_json', lambda *a, reply=replies[name], **k: reply)
        assert t.probe(row)[0] == expected, name
    monkeypatch.setattr(t, 'provider_key', lambda provider: None)
    assert t.probe({'provider': 'cerebras', 'model': 'gpt-oss-120b'})[0] == t.NEEDS_KEY


def qual_manifest(work, outcomes, seconds=120):
    work.mkdir(parents=True, exist_ok=True)
    with (work / 'manifest.jsonl').open('w') as handle:
        for i, outcome in enumerate(outcomes):
            handle.write(json.dumps({'index': i, 'seconds': seconds, 'episodes': [
                {'outcome': outcome, 'reward': 1.0 if outcome == 'task_success' else 0.0}]}) + '\n')


def test_qualify_starts_one_per_quota_group_and_adopt_writes_teachers_with_sized_slots(tmp_path, monkeypatch):
    monkeypatch.setattr(t, 'QUAL_ROOT', tmp_path / 'qual')
    monkeypatch.setattr(t, 'seed_work', lambda work: work.mkdir(parents=True, exist_ok=True))
    monkeypatch.setattr(t, 'env_key', lambda name, file: 'k')
    out = tmp_path / 'free'
    for provider, model in (('ai-studio', 'gemini-3.8-flash'), ('ai-studio', 'gemini-3.6-flash'),
                            ('nvidia-nim', 'nvidia/big-120b'), ('nvidia-nim', 'nvidia/other-120b'),
                            ('zai', 'glm-9-flash')):
        t.add(provider, model, out=out, now='T')
    t.update_candidates(out, lambda row: row.update(status='probed'))
    launched = []
    started = t.qualify_all(out, pin=tmp_path, runner=lambda argv, **k: launched.append(argv))
    # per-model AI Studio quotas run side by side; NIM's one account limit runs one qualification at a time
    assert started == ['ai-studio:gemini-3.8-flash', 'ai-studio:gemini-3.6-flash', 'nvidia-nim:nvidia/big-120b', 'zai:glm-9-flash']
    assert all('--priority' in argv and 'low' in argv for argv in launched)
    script = (tmp_path / 'qual' / 'zai--glm-9-flash' / 'run.sh').read_text()
    assert '--attempts 1 --limit 20' in script and 'zai.env' in script and 'printf' in script

    qual_manifest(tmp_path / 'qual' / 'ai-studio--gemini-3-8-flash', ['task_success'] * 9 + ['budget_exhaustion'] * 11, 150)
    qual_manifest(tmp_path / 'qual' / 'ai-studio--gemini-3-6-flash', ['task_success'] * 3 + ['budget_exhaustion'] * 17)
    qual_manifest(tmp_path / 'qual' / 'nvidia-nim--nvidia-big-120b', ['task_success'] * 10 + ['budget_exhaustion'] * 10)
    qual_manifest(tmp_path / 'qual' / 'zai--glm-9-flash', ['task_success'] * 2)          # still running
    teachers = tmp_path / 'teachers.json'
    teachers.write_text(json.dumps([{'model_id': 'gpt-6-luna', 'provider': 'openai-codex', 'slots': 6},
                                    {'model_id': 'nvidia/super', 'endpoint': 'https://integrate.api.nvidia.com/v1',
                                     'api_key_env': 'NVIDIA_API_KEY', 'slots': 3}]))
    decided = t.adopt_all(out, teachers, active=lambda unit: 'glm' in unit, now='2026-11-25')
    status = {d['model']: (d['status'], d['note']) for d in decided}
    assert status['gemini-3.8-flash'][0] == 'admitted' and status['gemini-3.6-flash'][0] == 'rejected'
    assert status['nvidia/big-120b'][0] == 'qualified' and 'nim full' in status['nvidia/big-120b'][1]
    assert 'glm-9-flash' not in status
    specs = json.loads(teachers.read_text())
    added = specs[-1]
    assert added['model_id'] == 'gemini-3.8-flash' and added['slots'] == 1 and added['api_key_env'] == 'GEMINI_API_KEY'
    assert 'api_key_command' not in added and added['free_check']['yield'] == '9/20'
    assert len(specs) == 3
    row = next(r for r in t.read_jsonl(out / 'candidates.jsonl') if r['model'] == 'gemini-3.8-flash')
    assert row['expected_successes_per_hour'] == round(3600 / 150 * 0.45, 1)


def test_slots_follow_provider_limits():
    assert t.slots_for('ai-studio', 150) == 1
    assert t.slots_for('openrouter', 120) == 1 and t.slots_for('openrouter', 120, group_slots_used=1) == 0
    assert t.slots_for('nvidia-nim', 600) == 3 and t.slots_for('nvidia-nim', 600, group_slots_used=2) == 1


def test_a_free_tier_of_a_few_requests_a_day_is_rejected_and_its_qualification_stopped(tmp_path, monkeypatch):
    monkeypatch.setattr(t, 'env_key', lambda name, file: 'k')
    out = tmp_path / 'free'
    t.add('ai-studio', 'gemini-3.6-flash', out=out, now='T')
    work = tmp_path / 'qual'
    qual_manifest(work, ['task_success'])
    (work / 'episodes' / 't000001').mkdir(parents=True)
    (work / 'episodes' / 't000001' / 's7.log').write_text(
        'generate_content_free_tier_requests, limit: 20, model: gemini-3.6-flash, description: Generate content requests '
        'per day limit exceeded on Free Tier.\n generate_content_free_tier_requests, limit: 5, model: gemini-3.6-flash, '
        'description: Generate content requests per minute limit exceeded on Free Tier.')
    assert t.daily_request_limit(work) == 20
    t.update_candidates(out, lambda row: row.update(status='qualifying', work=str(work), unit='tq-x'))
    stopped = []
    decided = t.adopt_all(out, tmp_path / 'teachers.json', active=lambda unit: True, stop=stopped.append)
    assert stopped == ['tq-x'] and decided[0]['status'] == 'rejected' and '20 requests/day' in decided[0]['note']
    assert not (tmp_path / 'teachers.json').exists()
