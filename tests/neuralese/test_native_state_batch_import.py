"""Saved-fit import verifies complete numeric histories without repeating every prefix."""
import pytest
import torch

from test_native_serving_adapters import engine_fixture
from natlang_neuralese.model.native_artifact import NativeTensorArtifact
from natlang_neuralese.model.native_state_derivation import optimize_state, state_values
from natlang_neuralese.serve.store import TensorStore, NativeTensorRecord, encode_native_state


def _history(tmp_path, family, algorithm):
    engine = engine_fixture(tmp_path, family, service_template=True)
    root = engine.store.get_native_state(engine.native_write_text('t5 t6')['id'])
    records = [root]
    moment = None
    for _ in range(4):
        parent = records[-1]
        name = next(iter(state_values(parent.block)))
        gradient = engine.store.put_native_tensor_artifact(NativeTensorArtifact(
            'state-gradient', engine.native_identity,
            {name: torch.full_like(state_values(parent.block)[name], .03)},
            parent.id, {'scope': 'test-real-state-update'}))
        result, _, _ = optimize_state(engine, gradient,
            {'name': algorithm, 'learning_rate': .01, 'momentum': .4, 'eps': 1e-6}, moment)
        moment = result['optimizer_state_id']
        records.append(engine.store.get_native_state(result['native_state_id']))
    return engine, records


def _artifacts(source, target):
    for record in list(source._resident.values()):
        if isinstance(record, NativeTensorRecord):
            target.put_native_tensor_artifact(record.artifact)


@pytest.mark.parametrize('family', ['lfm', 'mellum'])
@pytest.mark.parametrize('algorithm', ['adam', 'sgd'])
def test_batch_keeps_exact_history_and_checks_each_edge_once(tmp_path, monkeypatch, family, algorithm):
    engine, records = _history(tmp_path, family, algorithm)
    target = TensorStore()
    try:
        _artifacts(engine.store, target)
        import natlang_neuralese.model.native_state_derivation as derivation
        original = derivation._verify_optimizer_derivation
        calls = []

        def checked(store, block):
            calls.append(block.derivation['parent_state_id'])
            return original(store, block)

        monkeypatch.setattr(derivation, '_verify_optimizer_derivation', checked)
        install = target._put_verified_native_state

        def parent_first(record, owner):
            if record.block.derivation is not None:
                assert target.get_native_state(record.block.derivation['parent_state_id']) is not None
            return install(record, owner)

        monkeypatch.setattr(target, '_put_verified_native_state', parent_first)
        result = target.put_native_states([record.block for record in reversed(records)], 'fit')
        assert [record.id for record in result] == [record.id for record in reversed(records)]
        assert len(calls) == 4  # Eager per-state verification checks 1+2+3+4 edges.
        for record in records:
            actual = target.get_native_state(record.id)
            assert encode_native_state(actual.block) == encode_native_state(record.block)
            assert target.holders(record.id) == {'fit'}
        # Publication owns the tensors rather than borrowing caller memory.
        records[-1].block.embeddings.add_(1)
        assert not torch.equal(records[-1].block.embeddings, result[0].block.embeddings)
    finally:
        target.close()
        engine.store.close()


def test_missing_ancestor_admits_no_partial_state(tmp_path):
    engine, records = _history(tmp_path, 'lfm', 'adam')
    target = TensorStore()
    try:
        _artifacts(engine.store, target)
        with pytest.raises(ValueError, match='archived parent'):
            target.put_native_states([record.block for record in records[1:]])
        assert all(target.get_native_state(record.id) is None for record in records)
    finally:
        target.close()
        engine.store.close()


def test_invalid_disconnected_state_is_checked_before_any_publication(tmp_path):
    engine, records = _history(tmp_path, 'lfm', 'adam')
    target = TensorStore()
    try:
        _artifacts(engine.store, target)
        from natlang_neuralese.model.native_state_derivation import replace_state_values
        bad = records[2].block
        values = {key: value + 1 for key, value in state_values(bad).items()}
        forged = replace_state_values(bad, values)
        with pytest.raises(ValueError, match='numeric update'):
            target.put_native_states([*[record.block for record in records], forged])
        assert all(target.get_native_state(record.id) is None for record in records)
    finally:
        target.close()
        engine.store.close()


def test_batch_can_continue_exact_external_parent(tmp_path):
    engine, records = _history(tmp_path, 'mellum', 'adam')
    target = TensorStore()
    try:
        _artifacts(engine.store, target)
        target.put_native_state(records[0].block)
        target.put_native_states([record.block for record in records[1:]], 'fit')
        assert encode_native_state(target.get_native_state(records[-1].id).block) == encode_native_state(records[-1].block)
    finally:
        target.close()
        engine.store.close()


def test_saved_fit_import_uses_verified_batch_and_checks_file_names(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from natlang_neuralese.tune.build import import_archive
    engine, records = _history(tmp_path, 'lfm', 'adam')
    target = TensorStore()
    directory = tmp_path / 'archive'
    directory.mkdir()
    try:
        _artifacts(engine.store, target)
        for record in records:
            (directory / f'{record.id}.nzs').write_bytes(encode_native_state(record.block))
        batch = target.put_native_archive_payloads
        sizes = []

        def checked(payloads):
            values = list(payloads)
            sizes.append(len(values))
            return batch(values)

        monkeypatch.setattr(target, 'put_native_archive_payloads', checked)
        import_archive(SimpleNamespace(store=target), state_directories=[directory])
        assert sizes == [5]
        assert encode_native_state(target.get_native_state(records[-1].id).block) == encode_native_state(records[-1].block)
        (directory / 'wrong.nzs').write_bytes(encode_native_state(records[0].block))
        with pytest.raises(ValueError, match='named content'):
            import_archive(SimpleNamespace(store=target), state_directories=[directory])
        assert sizes == [5]
    finally:
        target.close()
        engine.store.close()
