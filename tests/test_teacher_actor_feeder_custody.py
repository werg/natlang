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
