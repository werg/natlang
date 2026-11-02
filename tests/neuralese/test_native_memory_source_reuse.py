"""Prepared capacity checks reuse actual loaded data without weakening its identity."""
import copy
import hashlib
import json
import os
import random
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from natlang_neuralese.data.native_sources import NativeGraphSource
from natlang_neuralese.eval import native_memory as memory
from natlang_neuralese.train import native_resident as resident
from natlang_neuralese.train.native_state_sft import native_training_update
from test_native_memory_capacity import prepared_fixture
from test_native_optimizer_checkpointing import assert_exact


def active_request(tmp_path, monkeypatch, family='lfm'):
    model, optimizer, leaf, roles, config, args = prepared_fixture(
        tmp_path, monkeypatch, family=family)
    root = tmp_path/'loaded-train.json'
    root.write_text(json.dumps({'schema': 'natlang.native-graph-sources/1',
        'sources': [{'kind': 'native-graphs', **resident.binding(leaf.path)}]})+'\n')
    train = NativeGraphSource(root, config['tokenizer_fingerprint'],
                              required_split='train', role_ids=roles.ids)
    request_path = resident.bound(json.loads(args.memory_qualification))
    request = json.loads(request_path.read_text())
    request.update(source=resident.binding(leaf.path), indexes=resident._source_witness(train.children[0]),
                   cases=[dict(request['reference'])])
    request_path.write_text(json.dumps(request))
    args.memory_qualification = json.dumps(resident.binding(request_path))
    return model, optimizer, train, roles, config, args, request_path


@pytest.mark.parametrize('family', ['lfm'])
def test_prepared_real_backward_reuses_loaded_subtree_and_preserves_full_state(tmp_path, monkeypatch, family):
    model, optimizer, train, roles, config, args, _ = active_request(tmp_path, monkeypatch, family)
    native_training_update(model, optimizer, list(train.iter_indices([0])),
                           checkpoint_layers=True, chunk_tokens=3)
    optimizer.zero_grad(set_to_none=True)
    before = copy.deepcopy(model.state_dict()), copy.deepcopy(model.port.hf.state_dict()), copy.deepcopy(optimizer.state_dict())
    rng, python_rng = torch.get_rng_state().clone(), random.getstate()
    caches = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (tmp_path/'indexes').glob('*.pkl')}
    original = NativeGraphSource.__init__

    def reject_reopening(self, path, *a, **kw):
        assert Path(path).resolve() not in {train.path.resolve(), train.children[0].path.resolve()}, \
            'already validated source must not be reconstructed'
        return original(self, path, *a, **kw)

    monkeypatch.setattr(NativeGraphSource, '__init__', reject_reopening)
    result = memory.run_prepared(model, optimizer, train, roles, config, args,
                                 backend=None, checkpoint_sha256='a'*64)
    assert result['passed'] and result['numerical_reference']['passed']
    case, = result['capacity']['cases']
    assert case['id'] == 'active' and case['gradients'] and case['status'] == 'passed'
    assert result['optimizer_updates'] == 0
    for expected, actual in zip(before, (model.state_dict(), model.port.hf.state_dict(), optimizer.state_dict())):
        assert_exact(expected, actual)
    assert torch.equal(rng, torch.get_rng_state()) and python_rng == random.getstate()
    assert all(parameter.grad is None for parameter in model.parameters())
    assert caches == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in caches}


def no_model_work(monkeypatch):
    for name in ('offload_numerical_parity', 'qualify_capacity'):
        monkeypatch.setattr(memory, name, lambda *a, **kw: pytest.fail('invalid source reached model work'))


def test_retained_leaf_change_with_same_size_and_mtime_is_rejected_before_backward(tmp_path, monkeypatch):
    model, optimizer, train, roles, config, args, request_path = active_request(tmp_path, monkeypatch)
    leaf = train.children[0].path
    before = leaf.stat()
    raw = leaf.read_text()
    changed = raw.replace('"id": "active"', '"id": "edited"')
    assert raw != changed and len(raw) == len(changed)
    leaf.write_text(changed)
    os.utime(leaf, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert leaf.stat().st_size == before.st_size and leaf.stat().st_mtime_ns == before.st_mtime_ns
    # Point at the unchanged manifest so the retained dependency verifier must
    # detect the mutation; a fresh top-level binding check alone cannot.
    request = json.loads(request_path.read_text())
    request['source'] = resident.binding(train.path)
    request_path.write_text(json.dumps(request))
    args.memory_qualification = json.dumps(resident.binding(request_path))
    no_model_work(monkeypatch)
    with pytest.raises(ValueError, match='reused preparation source changed'):
        memory.run_prepared(model, optimizer, train, roles, config, args,
                            backend=None, checkpoint_sha256='a'*64)
    assert not list(args.out.rglob('proof.json'))


@pytest.mark.parametrize('contract', ['roles', 'tokenizer', 'split'])
def test_loaded_source_cannot_be_reinterpreted_for_another_contract(tmp_path, monkeypatch, contract):
    model, optimizer, train, roles, config, args, request_path = active_request(tmp_path, monkeypatch)
    request = json.loads(request_path.read_text())
    if contract == 'roles':
        roles = SimpleNamespace(ids=(54,55,56,57))
    elif contract == 'tokenizer':
        config = {**config, 'tokenizer_fingerprint': 'different-tokenizer'}
        request['identity'] = memory.capacity_identity(config, args, None)
    else:
        request['split'] = 'test'
    request_path.write_text(json.dumps(request))
    args.memory_qualification = json.dumps(resident.binding(request_path))
    no_model_work(monkeypatch)
    with pytest.raises(ValueError, match='tokenizer or role contract|expected test'):
        memory.run_prepared(model, optimizer, train, roles, config, args,
                            backend=None, checkpoint_sha256='a'*64)
    assert not list(args.out.rglob('proof.json'))


def test_dense_continuation_reuses_authenticated_measurement_after_fresh_parity(tmp_path, monkeypatch):
    from natlang_neuralese.train import native_packed_capacity as packed
    model, optimizer, train, roles, config, args, request_path = active_request(tmp_path, monkeypatch)
    config = {**config, 'architecture': 'lfm3'}
    request = json.loads(request_path.read_text())
    request['identity'] = memory.capacity_identity(config, args, None)
    request_path.write_text(json.dumps(request))
    args.memory_qualification = json.dumps(resident.binding(request_path))
    monkeypatch.setattr(memory, '_dense_capacity_reuse_supported', lambda *a: True)
    identity = {'execution': 'dense', 'optimizer': packed.optimizer_shape_identity(model, optimizer)}
    monkeypatch.setattr(packed, 'oversize_execution_identity', lambda *a, **kw: identity)
    monkeypatch.setattr(packed, '_capacity_memory_context', lambda device: {'device_type':'cpu', 'host_available_bytes': 1 << 40})
    monkeypatch.setattr(packed, '_incremental_headroom_covers', lambda row, context: True)
    first = memory.run_prepared(model, optimizer, train, roles, config, args,
                                backend=None, checkpoint_sha256='a'*64)
    proof = resident.binding(args.out/'memory-qualifications'/first['request']['sha256']/('a'*64)/'proof.json')
    calls = []
    monkeypatch.setattr(memory, 'qualify_capacity', lambda *a, **kw: pytest.fail('full graph was repeated'))
    monkeypatch.setattr(memory, 'offload_numerical_parity', lambda *a, **kw: (calls.append('parity'), {'passed':True})[1])
    second = memory.run_prepared(model, optimizer, train, roles, config, args,
                                 backend=None, checkpoint_sha256='b'*64, prior_proofs=[proof])
    assert calls == ['parity']
    assert second['passed'] and second['capacity_reuse']['measured_proof'] == proof
    assert second['capacity']['parent_checkpoint_sha256'] == 'a'*64
    assert second['parent_checkpoint_sha256'] == 'b'*64

    # A changed proof cannot authorize another continuation.
    Path(proof['path']).write_text('{}')
    measured = []
    def remeasure(*a, **kw):
        measured.append(True)
        return first['capacity']
    monkeypatch.setattr(memory, 'qualify_capacity', remeasure)
    third = memory.run_prepared(model, optimizer, train, roles, config, args,
                                backend=None, checkpoint_sha256='c'*64, prior_proofs=[proof])
    assert third['passed'] and measured == [True] and not third.get('capacity_reuse')


def test_optimizer_shape_identity_tracks_state_and_parameter_partition():
    from natlang_neuralese.train.native_packed_capacity import optimizer_shape_identity
    from natlang_neuralese.model.shared_state import SharedStateModel
    from test_shared_state import tiny
    model = SharedStateModel(tiny('lfm'), (58,59,60,61), rank=4)
    optimizer = torch.optim.Adam(model.parameters())
    before = optimizer_shape_identity(model, optimizer)
    parameter = next(model.parameters())
    optimizer.state[parameter]['moment'] = torch.zeros_like(parameter)
    after = optimizer_shape_identity(model, optimizer)
    assert before != after
    optimizer.state[parameter]['moment'].fill_(3)
    assert optimizer_shape_identity(model, optimizer) == after


def test_file_scratch_reuse_requires_measured_peak_and_storage_headroom(tmp_path, monkeypatch):
    from natlang_neuralese.train import checkpoint_safety
    import shutil
    monkeypatch.setattr(checkpoint_safety, 'checkpoint_minimum_free_bytes', lambda path: 100)
    monkeypatch.setattr(shutil, 'disk_usage', lambda path: type('Space', (), {'free': 2 << 30})())
    assert not memory._scratch_headroom_covers([{}], str(tmp_path))
    cases = [{'checkpoint_input_offload': {'peak_live_bytes': 1 << 20}}]
    assert memory._scratch_headroom_covers(cases, str(tmp_path))
    monkeypatch.setattr(shutil, 'disk_usage', lambda path: type('Space', (), {'free': 1000})())
    assert not memory._scratch_headroom_covers(cases, str(tmp_path))
