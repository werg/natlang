"""Actual live identity and immutable teacher configuration; no generation."""
import json
import os

import pytest

from scripts.coreweave import teacher_generation_guardian as guardian


def selected_teacher(tmp_path):
    actor = tmp_path / 'serving-intent.json'
    guardian.write(actor, {'model': 'RecursiveSI/flint', 'revision': 'pinned-quant'})
    current = guardian.process(os.getpid())
    owner = {**guardian.public_identity(current),
             'argv_sha256': guardian.hashlib.sha256(b'\0'.join(current['argv'])).hexdigest()}
    identity = tmp_path / 'identity.json'
    guardian.write(identity, {'schema': 'natlang.self-hosted-teacher-execution-identity/1',
        'server': 'http://localhost:8002', 'model_id': 'flint',
        'actor': guardian.binding(actor), 'processes': [owner]})
    sampling = tmp_path / 'sampling.json'
    guardian.write(sampling, {'top_p': .95, 'top_k': 20, 'min_p': 0})
    source = tmp_path / 'teacher-source.json'
    guardian.write(source, {'schema': 'natlang.self-hosted-teacher-source/1',
        'server': 'http://localhost:8002', 'model_id': 'flint',
        'execution_identity': guardian.binding(identity),
        'teacher': {'model': 'RecursiveSI/flint', 'revision': 'pinned-quant'},
        'sampling': {'temperature': 1.0, 'request_config': guardian.binding(sampling)}})
    return {'server': 'http://localhost:8002', 'model_id': 'flint',
            'teacher_identity': guardian.binding(identity), 'teacher_source': guardian.binding(source)}


def test_selected_teacher_environment_binds_actual_live_actor_and_sampling(tmp_path):
    config = selected_teacher(tmp_path)
    env = guardian.teacher_environment(config)
    assert env['NATLANG_TEACHER_SERVER'] == config['server']
    assert env['NATLANG_TEACHER_MODEL_ID'] == 'flint'
    assert json.loads(env['NATLANG_TEACHER_SOURCE_BINDING']) == config['teacher_source']


@pytest.mark.parametrize('changed', ['source', 'identity', 'sampling', 'actor', 'endpoint', 'model'])
def test_selected_teacher_rejects_changed_runtime_or_bytes(tmp_path, changed):
    config = selected_teacher(tmp_path)
    source = json.loads(open(config['teacher_source']['path']).read())
    if changed == 'endpoint':
        config['server'] = 'http://other'
    elif changed == 'model':
        config['model_id'] = 'old-teacher'
    else:
        paths = {'source': config['teacher_source']['path'],
                 'identity': config['teacher_identity']['path'],
                 'sampling': source['sampling']['request_config']['path'],
                 'actor': json.loads(open(config['teacher_identity']['path']).read())['actor']['path']}
        with open(paths[changed], 'a') as stream:
            stream.write('\n')
    with pytest.raises(ValueError):
        guardian.teacher_environment(config)


def test_legacy_teacher_cannot_inherit_an_undeclared_new_descriptor(monkeypatch):
    monkeypatch.setenv('NATLANG_TEACHER_SOURCE_BINDING', '{"path":"wrong"}')
    with pytest.raises(ValueError, match='ambient'):
        guardian.teacher_environment({'server': 'http://localhost:8001', 'model_id': 'pennyroyal'})


def test_repair_uses_new_teacher_sampling_and_preserves_original_queue(tmp_path, monkeypatch):
    from test_teacher_generation_guardian import repair_fixture
    config, receipt_path = repair_fixture(tmp_path)
    original = tmp_path / 'repairs/batch-0013/queue.jsonl'
    original_bytes = original.read_bytes()
    config.update(selected_teacher(tmp_path))
    calls = []
    monkeypatch.setattr(guardian, 'recover_token', lambda *a, **k: None)
    def capture(command, **kwargs):
        calls.append(command)
        raise RuntimeError('intent captured before spawn')
    monkeypatch.setattr(guardian.subprocess, 'Popen', capture)
    with pytest.raises(RuntimeError, match='before spawn'):
        guardian.launch_repair(tmp_path, {}, config)
    attempt = json.loads(receipt_path.read_text())['current']
    assert attempt['teacher_source'] == config['teacher_source']
    assert attempt['sampling']['temperature'] == 1.0
    assert '--chat-request-config' in calls[0]
    assert calls[0][calls[0].index('--model-id') + 1] == 'flint'
    assert all(json.loads(line)['temperature'] == 1.0
               for line in open(attempt['queue']) if line.strip())
    assert original.read_bytes() == original_bytes
