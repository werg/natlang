"""Actual exited process custody and full metadata; CPU protocol fixtures only."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
import torch

from test_native_resident import setup
from natlang_neuralese.train import native_resident as resident
from natlang_neuralese.train import native_resident_geometry_refresh as refresh
from natlang_neuralese.train.native_continuation import TRAINING_SCHEMA
from natlang_neuralese.model.native_model import CHECKPOINT_SCHEMA
from natlang_neuralese.rl import native_bridge as bridge, native_raw_handoff as hand


@pytest.fixture
def recovery(tmp_path, monkeypatch, request):
    options = getattr(request, 'param', {})
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE', str(tmp_path / 'cache'))
    model_config = {'architecture': 'fixture', 'dtype': 'torch.float32'}
    previous = {'model_config': model_config, 'ffn_chunk_tokens': 0}
    paths, directory, manager, recipe = setup(tmp_path, geometry_identity=previous)
    resident.prepare(directory, recipe, 'sft', tokenizer_fingerprint='tokenizer-sha', role_ids=())
    assert manager.poll(7) is not None
    current = resident.current(directory)
    row_ids = manager.train.ordered_ids()
    training = {'schema': TRAINING_SCHEMA, 'step': 9, 'global_step': 9,
        'optimizer': {'ordinary': {'moments': torch.tensor([1., 2.])}},
        'rng': {'torch': torch.get_rng_state()},
        'source_row_bindings': dict(manager.train.bindings),
        'held_row_bindings': dict(manager.held.bindings), 'source_row_order': row_ids,
        'exposure_counts': {key: int(key != 'c') for key in row_ids},
        'source_receipts': [current['payload'][role] for role in ('train', 'held')],
        'options': {'resident_adoption': {'generation': current['generation'], 'history': current['history']}}}
    if options.get('bad_exposure'):
        training['exposure_counts'].pop(row_ids[-1])
    if options.get('bad_adoption'):
        training['options']['resident_adoption']['history'] = []
    if options.get('missing_optimizer'):
        training['optimizer'] = {}
    checkpoint = tmp_path / 'full-state.pt'
    torch.save({'schema': CHECKPOINT_SCHEMA, 'step': 9, 'model_config': model_config,
                'backbone_state': {'weight': torch.tensor([3., 4.])}, 'training': training}, checkpoint)
    checkpoint_bytes = checkpoint.read_bytes()
    operation = tmp_path / 'operation'
    learner = operation / 'learner'
    learner.mkdir(parents=True)
    proposed = {**previous, 'ffn_chunk_tokens': 4096}
    cases = [{'id': 'c', **current['payload']['geometry']}]
    memory_request = learner / 'capacity-request.json'
    resident.write_json(memory_request, {'cases': cases})
    proof = learner / 'capacity-proof.json'
    resident.write_json(proof, {'schema': 'natlang.native-memory-qualification/1',
        'passed': True, 'optimizer_updates': 0, 'parent_checkpoint_sha256': resident.sha(checkpoint),
        'identity': proposed, 'runtime_code_sha256': resident.digest(current['payload']['plan']['code']),
        'request': resident.binding(memory_request), 'numerical_reference': {'passed': True},
        'capacity': {'passed': True, 'cases': [{**cases[0], 'status': 'passed'}]},
        'scope': 'CPU protocol fixture; no production numerical or GPU capacity credit'})
    qualification = learner / 'capacity.json'
    resident.write_json(qualification, {'schema': 'natlang.native-resident-geometry-qualification/1',
        'passed': True, 'identity': proposed, 'limits': current['payload']['geometry'],
        'runtime_code_sha256': resident.digest(current['payload']['plan']['code']),
        'evidence': resident.binding(proof)})
    config = operation / 'config.json'
    resident.write_json(config, {'native_state': True, 'checkpoint': str(checkpoint),
        'out': str(operation), 'resident_data_directory': str(directory), 'production_stage': 'sft',
        'trainer_parameters': {'ffn_chunk_tokens': 4096, 'memory_qualification': [resident.binding(memory_request)]}})
    bridge_request = bridge.write_request(operation / 'bridge', kind='round', out=operation, config=config, device='cpu')
    # Run an actual short-lived process with the real trainer module/--out argv
    # shape; its isolated module is explicitly only a process-protocol fixture.
    runtime = tmp_path / 'runtime'
    package = runtime / 'natlang_neuralese/train'
    package.mkdir(parents=True)
    (runtime / 'natlang_neuralese/__init__.py').write_text('')
    (package / '__init__.py').write_text('')
    (package / 'native_state_sft.py').write_text('import sys\nprint("ready",flush=True)\nsys.stdin.buffer.read()\nsys.exit(1)\n')
    argv = [sys.executable, '-m', 'natlang_neuralese.train.native_state_sft', '--out', str(learner)]
    child = subprocess.Popen(argv, env={**os.environ, 'PYTHONPATH': str(runtime)}, stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == 'ready'
        observed = hand.observe(child.pid)
        launch = tmp_path / 'actual-launch.json'
        resident.write_json(launch, {'at': time.time(), 'pid': child.pid, 'argv': argv, 'identity': observed})
        captured = hand.capture_drain_intent(resident.binding(launch), tmp_path / 'captured-owner.json')
        initial = json.loads((directory / 'initial.json').read_bytes())
        initial['owner'] = observed
        resident.write_json(directory / 'initial.json', initial)
        if not options.get('live_owner'):
            child.communicate('')
            assert child.returncode == 1
        value = bridge.read_request(bridge_request)
        result = operation / 'failed-result.json'
        resident.write_json(result, {'schema': bridge.RESULT_SCHEMA, 'kind': 'round', 'status': 'failed',
            'request_id': value['request_id'], 'request_sha256': resident.sha(bridge_request),
            'error': 'actual exited fixture owner; no production credit', 'artifacts': bridge.inventory(operation)})
        kwargs = {'checkpoint': resident.binding(checkpoint), 'qualification': resident.binding(qualification),
            'failed_request': resident.binding(bridge_request), 'failed_result': resident.binding(result),
            'captured_owner': captured, 'out': tmp_path / 'workspace-recovery.json',
            'reason': 'Use a newly qualified bounded FFN workspace without resetting earned state.'}
        yield {'directory': directory, 'manager': manager, 'kwargs': kwargs,
               'checkpoint': checkpoint, 'checkpoint_bytes': checkpoint_bytes,
               'initial': (directory / 'initial.json').read_bytes(),
               'current': (directory / 'current.json').read_bytes(),
               'pending': (directory / 'pending.json').read_bytes(), 'result': result,
               'qualification': qualification, 'proof': proof, 'learner': learner}
    finally:
        if child.poll() is None:
            child.communicate('')


def test_recovery_preserves_all_earned_state_and_data_and_normal_constructor_accepts(recovery):
    x = recovery
    ref = refresh.recover_saved_workspace(x['directory'], **x['kwargs'])
    receipt = json.loads(resident.bound(ref).read_bytes())
    assert receipt['training_credit'] == 0 and receipt['readiness_granted'] is False
    assert receipt['original_operation_still_failed'] is True
    after = json.loads((x['directory'] / 'initial.json').read_bytes())
    before = json.loads(x['initial'])
    assert {k: v for k, v in after.items() if k not in receipt['changed_fields']} == {
        k: v for k, v in before.items() if k not in receipt['changed_fields']}
    assert after['geometry_identity']['ffn_chunk_tokens'] == 4096
    assert x['checkpoint'].read_bytes() == x['checkpoint_bytes']
    assert (x['directory'] / 'current.json').read_bytes() == x['current']
    assert (x['directory'] / 'pending.json').read_bytes() == x['pending']
    m = x['manager']
    resident.ResidentAdoption(x['directory'], stage_id='sft', recipe_binding=m.recipe_binding,
        train=m.train, held=m.held, geometry_qualification=x['qualification'],
        geometry_identity=after['geometry_identity'])
    assert refresh.recover_saved_workspace(x['directory'], **x['kwargs']) == ref


@pytest.mark.parametrize('recovery', [{'live_owner': True}, {'bad_exposure': True},
                                     {'bad_adoption': True}, {'missing_optimizer': True}], indirect=True)
def test_live_owner_or_incomplete_saved_authority_is_rejected_without_metadata_mutation(recovery):
    x = recovery
    with pytest.raises(ValueError):
        refresh.recover_saved_workspace(x['directory'], **x['kwargs'])
    assert (x['directory'] / 'initial.json').read_bytes() == x['initial']
    assert (x['directory'] / 'current.json').read_bytes() == x['current']
    assert (x['directory'] / 'pending.json').read_bytes() == x['pending']
    assert x['checkpoint'].read_bytes() == x['checkpoint_bytes']


def test_new_optimizer_progress_cannot_be_rewound_to_qualified_parent(recovery):
    x = recovery
    resident.write_json(x['learner'] / 'progress.json', {'global_step': 10})
    with pytest.raises(ValueError, match='newer optimizer progress'):
        refresh.recover_saved_workspace(x['directory'], **x['kwargs'])
    assert (x['directory'] / 'initial.json').read_bytes() == x['initial']


def test_actual_numerical_proof_must_have_been_inventoried_by_failed_operation(recovery):
    x = recovery
    result = json.loads(x['result'].read_bytes())
    result['artifacts'] = [a for a in result['artifacts'] if a['path'] != str(x['proof'])]
    resident.write_json(x['result'], result)
    x['kwargs']['failed_result'] = resident.binding(x['result'])
    with pytest.raises(ValueError, match='absent from actual failed operation'):
        refresh.recover_saved_workspace(x['directory'], **x['kwargs'])
    assert (x['directory'] / 'initial.json').read_bytes() == x['initial']


def test_crash_after_atomic_initial_commit_finishes_same_transaction_without_rewriting(recovery, monkeypatch):
    x = recovery
    original = resident.write_json
    with monkeypatch.context() as patch:
        def fail_commit(path, value):
            if Path(path) == x['kwargs']['out']:
                raise OSError('crash after initial commit')
            return original(path, value)
        patch.setattr(resident, 'write_json', fail_commit)
        with pytest.raises(OSError, match='crash after initial commit'):
            refresh.recover_saved_workspace(x['directory'], **x['kwargs'])
    initial = (x['directory'] / 'initial.json').read_bytes()
    reference = refresh.recover_saved_workspace(x['directory'], **x['kwargs'])
    assert (x['directory'] / 'initial.json').read_bytes() == initial
    assert refresh.recover_saved_workspace(x['directory'], **x['kwargs']) == reference


@pytest.mark.parametrize('change', ['model', 'parent', 'cases'])
def test_consistently_rebound_but_invalid_capacity_evidence_is_rejected(recovery, change):
    x = recovery
    proof = json.loads(x['proof'].read_bytes())
    certificate = json.loads(x['qualification'].read_bytes())
    if change == 'model':
        proof['identity']['model_config']['dtype'] = 'torch.bfloat16'
        certificate['identity'] = proof['identity']
    elif change == 'parent':
        proof['parent_checkpoint_sha256'] = '0' * 64
    else:
        proof['capacity']['cases'] = []
    resident.write_json(x['proof'], proof)
    certificate['evidence'] = resident.binding(x['proof'])
    resident.write_json(x['qualification'], certificate)
    result = json.loads(x['result'].read_bytes())
    result['artifacts'] = [resident.binding(item['path']) for item in result['artifacts']]
    resident.write_json(x['result'], result)
    x['kwargs'].update(qualification=resident.binding(x['qualification']), failed_result=resident.binding(x['result']))
    with pytest.raises(ValueError):
        refresh.recover_saved_workspace(x['directory'], **x['kwargs'])
    assert (x['directory'] / 'initial.json').read_bytes() == x['initial']
    assert x['checkpoint'].read_bytes() == x['checkpoint_bytes']


def test_declared_byte_counts_are_verified_in_addition_to_sha(recovery):
    x = recovery
    kwargs = {**x['kwargs'], 'checkpoint': {**x['kwargs']['checkpoint'], 'bytes': 1}}
    with pytest.raises(ValueError, match='byte count changed'):
        refresh.recover_saved_workspace(x['directory'], **kwargs)
    assert (x['directory'] / 'initial.json').read_bytes() == x['initial']


def test_new_pending_append_during_verification_is_preserved_and_cas_rejects(recovery, monkeypatch):
    x = recovery
    from natlang_neuralese.train.checkpoint_policy import CheckpointPolicy
    original = CheckpointPolicy.load
    changed = b'{"new_pending_obligation":"preserve"}\n'
    def advance_pending(*args, **kwargs):
        value = original(*args, **kwargs)
        (x['directory'] / 'pending.json').write_bytes(changed)
        return value
    monkeypatch.setattr(CheckpointPolicy, 'load', advance_pending)
    with pytest.raises(ValueError, match='authority changed during'):
        refresh.recover_saved_workspace(x['directory'], **x['kwargs'])
    assert (x['directory'] / 'initial.json').read_bytes() == x['initial']
    assert (x['directory'] / 'pending.json').read_bytes() == changed
