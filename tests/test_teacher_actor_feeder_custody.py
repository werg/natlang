"""Actual live actor/process custody; no model calls or generated training data."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from scripts.coreweave import teacher_generation_guardian as guardian


def wait_process(child, argument):
    for _ in range(200):
        value = guardian.process(child.pid)
        if value and argument in value['argv']:
            return value
        time.sleep(.005)
    raise AssertionError('fixture did not publish its actual executable arguments')


def teacher(tmp_path, name, owner, *, server=None):
    folder = tmp_path / name
    folder.mkdir()
    actor = folder / 'actor.json'
    guardian.write(actor, {'fixture': name, 'actual_actor_pid': owner['pid']})
    identity = folder / 'identity.json'
    endpoint = server or 'http://127.0.0.1:8000/' + name
    guardian.write(identity, {'schema': 'natlang.self-hosted-teacher-execution-identity/1',
        'server': endpoint, 'model_id': name, 'actor': guardian.binding(actor),
        'processes': [{**guardian.public_identity(owner),
            'argv_sha256': hashlib.sha256(b'\0'.join(owner['argv'])).hexdigest()}]})
    source = folder / 'source.json'
    guardian.write(source, {'schema': 'natlang.self-hosted-teacher-source/1',
        'server': endpoint, 'model_id': name, 'execution_identity': guardian.binding(identity),
        'teacher': {'model': 'fixture/' + name, 'revision': 'exact-fixture-revision'},
        'sampling': {'temperature': .6}})
    config = {'teacher_identity': guardian.binding(identity), 'server': endpoint, 'model_id': name}
    env = {**os.environ, 'NATLANG_TEACHER_SOURCE_BINDING': json.dumps(guardian.binding(source)),
        'NATLANG_TEACHER_IDENTITY': str(identity), 'NATLANG_TEACHER_SERVER': endpoint,
        'NATLANG_TEACHER_MODEL_ID': name}
    return config, env, source


@pytest.fixture
def actors(tmp_path):
    command = 'import time;time.sleep(60)'
    actor = subprocess.Popen([sys.executable, '-c', command])
    children = [actor]
    try:
        independent = wait_process(actor, command.encode())
        selected = guardian.process(os.getpid())
        config, _, _ = teacher(tmp_path, 'selected', selected)
        script = tmp_path / 'feed-evidence-fixture.py'
        script.write_text('import time\ntime.sleep(60)\n')

        def feeder(name, owner):
            _, env, source = teacher(tmp_path, name, owner)
            child = subprocess.Popen([sys.executable, str(script)], env=env)
            children.append(child)
            return wait_process(child, str(script).encode()), source

        yield config, selected, independent, feeder, children
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
            child.wait(timeout=10)


def test_distinct_verified_live_actor_can_finish_alongside_selected_teacher(actors):
    config, _, independent, feeder, _ = actors
    other, _ = feeder('other', independent)
    assert guardian.feeder_conflicts(other, config) is False


def test_endpoint_or_model_alias_cannot_claim_independent_custody_of_same_actor(actors):
    config, selected, _, feeder, _ = actors
    other, _ = feeder('different-alias-and-endpoint', selected)
    assert guardian.feeder_conflicts(other, config) is True


def test_mutated_source_binding_does_not_authorize_concurrent_generation(actors):
    config, _, independent, feeder, _ = actors
    other, source = feeder('mutated', independent)
    value = json.loads(source.read_bytes())
    value['teacher']['revision'] = 'changed-after-launch'
    guardian.write(source, value)
    assert guardian.feeder_conflicts(other, config) is True


def test_unknown_feeder_environment_keeps_exclusive_custody(actors, tmp_path):
    config, _, _, _, children = actors
    script = tmp_path / 'feed-evidence-unknown.py'
    script.write_text('import time\ntime.sleep(60)\n')
    env = {k: v for k, v in os.environ.items() if not k.startswith('NATLANG_TEACHER_')}
    child = subprocess.Popen([sys.executable, str(script)], env=env)
    children.append(child)
    other = wait_process(child, str(script).encode())
    assert guardian.feeder_conflicts(other, config) is True


def test_changed_selected_execution_identity_is_rejected_before_launch(actors):
    config, _, independent, feeder, _ = actors
    other, _ = feeder('independent', independent)
    path = Path(config['teacher_identity']['path'])
    value = json.loads(path.read_bytes())
    value['model_id'] = 'tampered'
    guardian.write(path, value)
    with pytest.raises(ValueError, match='identity binding changed'):
        guardian.feeder_conflicts(other, config)


def test_exited_feeder_is_not_an_active_reservation(actors):
    config, _, independent, feeder, children = actors
    other, _ = feeder('exited', independent)
    child = children[-1]
    child.terminate()
    child.wait(timeout=10)
    assert guardian.feeder_conflicts(other, config) is False


@pytest.mark.parametrize('same_actor', [False, True])
def test_prefetch_obeys_the_same_actor_custody_as_initial_launch(actors, tmp_path, monkeypatch, same_actor):
    config, selected, independent, feeder, children = actors
    feeder('previous-teacher', selected if same_actor else independent)
    monkeypatch.setenv('PATH', str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH', ''))
    next_feeder = tmp_path / 'feed-evidence-prefetch.py'
    next_feeder.write_text('import time\ntime.sleep(60)\n')
    source = tmp_path / 'source.json'
    guardian.write(source, {'fixture': 'immutable source'})
    worker = tmp_path / 'run_bonsai_queue.py'
    worker.write_text('# fixture worker bytes; not executed by the feeder stub\n')
    config = {**config, 'root': str(tmp_path / 'generation')}
    slot = {'token': os.urandom(12).hex(), 'batch': 1,
        'feeder': guardian.binding(next_feeder), 'source_configuration': guardian.binding(source),
        'task_runtime': '/qualified/task-runtime',
        'collection_context': {'repo': str(tmp_path), 'node_bin': '/qualified/node/bin',
                               'queue_worker': guardian.binding(worker)}}
    popen = subprocess.Popen

    def spawn(*args, **kwargs):
        child = popen(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(guardian.subprocess, 'Popen', spawn)
    if same_actor:
        with pytest.raises(ValueError, match='unowned feeder'):
            guardian._spawn_primary_prefetch(tmp_path, {}, config, slot, 1)
        assert 'owner' not in slot
    else:
        guardian._spawn_primary_prefetch(tmp_path, {}, config, slot, 1)
        assert guardian.alive(slot['owner'])
        assert children[-1].pid == slot['owner']['pid']


def test_repair_recovery_uses_recorded_versioned_worker_after_config_change(tmp_path, monkeypatch):
    queue = tmp_path / 'repairs/batch-0000/queue.jsonl'
    queue.parent.mkdir(parents=True)
    queue.write_text(json.dumps({'key': 'case-0', 'source': '/source', 'index': 0, 'seed': 7}) + '\n')
    config = {'task_runtime': '/qualified/runtime', 'server': 'http://localhost:8001',
        'model_id': 'fixture', 'node_bin': '/node/bin',
        'queue_worker': '/controller/run_bonsai_queue-versioned-original.py'}
    monkeypatch.setattr(guardian, 'recover_token', lambda *a, **k: None)
    monkeypatch.setattr(guardian.subprocess, 'Popen',
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError('before spawn')))
    with pytest.raises(RuntimeError, match='before spawn'):
        guardian._launch_one_repair(tmp_path, {}, config, source_queue=queue)
    seen = []
    owner = guardian.process(os.getpid())

    def recover(token, *, script_prefix):
        seen.append(script_prefix)
        return owner

    monkeypatch.setattr(guardian, 'recover_token', recover)
    state = {}
    guardian._launch_one_repair(tmp_path, state,
        {**config, 'queue_worker': '/controller/run_bonsai_queue-future.py'}, source_queue=queue)
    assert seen == ['run_bonsai_queue-versioned-original.py']
    assert state['repair_owner']['pid'] == os.getpid()
