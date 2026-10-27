import copy
from types import SimpleNamespace

import pytest

from natlang_neuralese.serve.store import TensorStore
from natlang_neuralese.tune.build import import_archive, persist_artifacts
from natlang_neuralese.serve.native_derivative import execute_derivative_program
from test_native_derivative_program import program
from test_native_serving_adapters import nonzero_adapter
from test_native_state_derivation import engine_fixture


@pytest.mark.parametrize('family', ['lfm', 'mellum'])
@pytest.mark.parametrize('objective_kind', ['ce', 'query'])
def test_program_state_archive_cold_restore(tmp_path, family, objective_kind):
    engine = engine_fixture(tmp_path, family, service_template=True)
    adapter, _ = nonzero_adapter(engine)
    source = program(engine, adapter, 'sgd')
    ident = engine.native_write_text('t5 t6')['id']
    source['inputs'] = {'a': {'kind': 'native-state', 'id': ident}}
    if objective_kind == 'query':
        from natlang_neuralese.model.native_readout import snapshot_readout
        readout = snapshot_readout(engine).id
    for objective in [source['outer'], *(node for node in source['nodes'] if node['kind'] == 'gradient')]:
        objective['request'] = copy.deepcopy(objective['request'])
        objective['request'].pop('adapters')
        graph = objective['request']['graph']
        graph['external_states'] = {'prior': {'id': ident, 'token_count': 2}}
        graph['nodes'] = graph['nodes'][1:]
        graph['nodes'][0]['imports'] = [{'source': 'prior', 'start': 0, 'stop': 2, 'at': 2}]
        if objective_kind == 'query':
            from test_native_query_objective import query_spec
            node = graph['nodes'][0]
            objective['request']['objective'] = query_spec(graph, [{'kind': 'nll', 'reduction': 'mean', 'readout': readout, 'actions': [
                {'query': index - 1, 'token_id': token} for index, (token, weight) in
                enumerate(zip(node['token_ids'], node['target_weights'])) if weight]}])
    target = SimpleNamespace(store=TensorStore())
    try:
        result = execute_derivative_program(engine, source)
        root = result['outputs']['a2']
        persist_artifacts(engine, [root, result['optimizer_states']['a2']], tmp_path / 'artifacts', tmp_path / 'states')
        import_archive(target, [tmp_path / 'artifacts'], [tmp_path / 'states'])
        assert target.store.get_native_state(root).block.derivation is not None
        from natlang_neuralese.model.native_state_derivation import verify_state_derivation
        verify_state_derivation(target.store, target.store.get_native_state(root).block)
        resumed = copy.deepcopy(source)
        resumed['inputs']['a']['id'] = root
        for objective in [resumed['outer'], *(node for node in resumed['nodes'] if node['kind'] == 'gradient')]:
            objective['request']['graph']['external_states']['prior']['id'] = root
            if objective_kind == 'query':
                from natlang_neuralese.common.hashing import canonical_json_sha256_hex
                objective['request']['objective']['graph_sha256'] = canonical_json_sha256_hex(objective['request']['graph'])
        original_store, engine.store = engine.store, target.store
        try:
            continued = execute_derivative_program(engine, resumed)
            assert target.store.get_native_state(continued['outputs']['a2']) is not None
            from natlang_neuralese.model.native_state_derivation import optimize_state
            gradient = target.store.get_native_tensor_artifact(continued['outputs']['g0'])
            update, _, _ = optimize_state(engine, gradient, source['nodes'][1]['optimizer'], result['optimizer_states']['a2'],
                program_id=gradient.artifact.provenance['derivative_program_id'])
            assert target.store.get_native_state(update['native_state_id']).block.derivation['optimizer_state_parent_id'] == result['optimizer_states']['a2']
        finally:
            engine.store = original_store
        missing = tmp_path / 'states' / f'{ident}.nzs'
        missing.unlink()
        empty = SimpleNamespace(store=TensorStore())
        try:
            with pytest.raises(ValueError):
                import_archive(empty, [tmp_path / 'artifacts'], [tmp_path / 'states'])
            assert len(empty.store) == 0
        finally:
            empty.store.close()
    finally:
        target.store.close()
        engine.store.close()


def test_cycle_rejected_before_admission(tmp_path, monkeypatch):
    from test_native_state_batch_import import _history
    import natlang_neuralese.model.state_block as state_block
    from natlang_neuralese.serve.store import native_state_id
    engine, records = _history(tmp_path, 'lfm', 'adam')
    target = TensorStore()
    try:
        blocks = [records[0].block, records[-1].block]
        first, last = [native_state_id(block) for block in blocks]
        monkeypatch.setattr(state_block, 'native_state_dependencies',
                            lambda block: [last if native_state_id(block) == first else first])
        with pytest.raises(ValueError, match='cyclic'):
            target.put_native_archive(blocks, [])
        assert len(target) == 0
    finally:
        target.close()
        engine.store.close()


def test_file_import_decodes_once_and_owns_values(tmp_path, monkeypatch):
    from test_native_state_batch_import import _history
    import natlang_neuralese.serve.store as store_module
    import natlang_neuralese.model.native_artifact as tensor_module
    engine, records = _history(tmp_path, 'lfm', 'adam')
    target = SimpleNamespace(store=TensorStore())
    try:
        persist_artifacts(engine, [records[-1].id], tmp_path / 'artifacts', tmp_path / 'states')
        decoded = {}
        state_decode, tensor_decode = store_module.decode_native_state, tensor_module.decode_native_tensor_artifact
        def state(body):
            value, ident = state_decode(body)
            assert ident not in decoded
            decoded[ident] = value
            return value, ident
        def tensor(body):
            value, ident = tensor_decode(body)
            assert ident not in decoded
            decoded[ident] = value
            return value, ident
        monkeypatch.setattr(store_module, 'decode_native_state', state)
        monkeypatch.setattr(tensor_module, 'decode_native_tensor_artifact', tensor)
        import_archive(target, [tmp_path / 'artifacts'], [tmp_path / 'states'])
        assert len(decoded) == len(list((tmp_path / 'artifacts').glob('*.nnt'))) + len(list((tmp_path / 'states').glob('*.nzs')))
        for ident, value in decoded.items():
            record = target.store.get_native_state(ident) if ident.startswith('nzs1_') else target.store.get_native_tensor_artifact(ident)
            assert (record.block if ident.startswith('nzs1_') else record.artifact) is value
        before = target.store.get_native_state(records[-1].id).block.embeddings.clone()
        (tmp_path / 'states' / f'{records[-1].id}.nzs').write_bytes(b'changed')
        assert target.store.get_native_state(records[-1].id).block.embeddings.equal(before)
    finally:
        target.store.close()
        engine.store.close()


def test_object_admission_copies_inputs_and_rejects_mutable_payload(tmp_path):
    from test_native_state_batch_import import _history
    from natlang_neuralese.serve.store import encode_native_state
    engine, records = _history(tmp_path, 'lfm', 'adam')
    target = TensorStore()
    try:
        root = records[0]
        target.put_native_archive([root.block], [])
        before = target.get_native_state(root.id).block.embeddings.clone()
        root.block.embeddings.add_(1)
        assert target.get_native_state(root.id).block.embeddings.equal(before)
        with pytest.raises(TypeError, match='immutable bytes'):
            target.put_native_archive_payloads([(root.id, bytearray(encode_native_state(root.block)))])
        assert len(target) == 1
    finally:
        target.close()
        engine.store.close()


def test_bounded_archive_preserves_history_and_rejects_missing_before_publication(tmp_path):
    from test_native_state_batch_import import _history
    from natlang_neuralese.model.native_state_derivation import verify_state_derivation
    engine, records = _history(tmp_path, 'lfm', 'adam')
    target = SimpleNamespace(store=TensorStore(tmp_path / 'bounded', resident_bytes=1))
    try:
        persist_artifacts(engine, [records[-1].id], tmp_path / 'artifacts', tmp_path / 'states')
        import_archive(target, [tmp_path / 'artifacts'], [tmp_path / 'states'])
        verify_state_derivation(target.store, target.store.get_native_state(records[-1].id).block)
        assert target.store.resident <= 1
        count = len(target.store)
        import_archive(target, [tmp_path / 'artifacts'], [tmp_path / 'states'])
        assert len(target.store) == count
        assert not list((tmp_path / 'bounded').glob('archive-*'))
        target.store.close()
        empty = SimpleNamespace(store=TensorStore(tmp_path / 'empty', resident_bytes=1))
        try:
            (tmp_path / 'states' / f'{records[0].id}.nzs').unlink()
            with pytest.raises(ValueError):
                import_archive(empty, [tmp_path / 'artifacts'], [tmp_path / 'states'])
            assert len(empty.store) == 0
        finally:
            empty.store.close()
    finally:
        engine.store.close()


def test_bounded_cycle_rejected_before_admission(tmp_path, monkeypatch):
    from test_native_state_batch_import import _history
    import natlang_neuralese.model.state_block as state_block
    from natlang_neuralese.serve.store import native_state_id, encode_native_state
    engine, records = _history(tmp_path, 'lfm', 'adam')
    target = TensorStore(tmp_path / 'cycle', resident_bytes=1)
    try:
        blocks = [records[0].block, records[-1].block]
        first, last = [native_state_id(block) for block in blocks]
        monkeypatch.setattr(state_block, 'native_state_dependencies',
                            lambda block: [last if native_state_id(block) == first else first])
        with pytest.raises(ValueError, match='cyclic'):
            target.put_native_archive_payloads(((native_state_id(b), encode_native_state(b)) for b in blocks), bounded=True)
        assert len(target) == 0
    finally:
        target.close()
        engine.store.close()


def test_fit_store_owns_disposable_scratch(tmp_path):
    import gc
    from natlang_neuralese.tune.build import fit_store
    store = fit_store({'store_scratch': str(tmp_path), 'store_resident_bytes': 1})
    directory = store.directory
    assert directory.exists() and store.resident_bytes == 1
    store.close()
    del store
    gc.collect()
    assert not directory.exists()
    with pytest.raises(ValueError):
        fit_store({'store_resident_bytes': -1})
