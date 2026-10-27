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
@pytest.mark.parametrize('bounded', [False, True])
def test_program_state_archive_cold_restore(tmp_path, family, objective_kind, bounded):
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
    target = SimpleNamespace(store=TensorStore(tmp_path / 'target', 1) if bounded else TensorStore())
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
        empty = SimpleNamespace(store=TensorStore(tmp_path / 'empty-target', 1) if bounded else TensorStore())
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


@pytest.mark.parametrize('family', ['lfm', 'mellum'])
@pytest.mark.parametrize('algorithm', ['adam', 'sgd'])
def test_bounded_archive_preserves_history_and_rejects_missing_before_publication(tmp_path, family, algorithm):
    from test_native_state_batch_import import _history
    from natlang_neuralese.model.native_state_derivation import verify_state_derivation
    engine, records = _history(tmp_path, family, algorithm)
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


def test_bounded_malformed_and_collision_leave_existing_store_unchanged(tmp_path):
    from test_native_state_batch_import import _history
    from natlang_neuralese.serve.store import encode_native_state
    engine, records = _history(tmp_path, 'lfm', 'adam')
    target = TensorStore(tmp_path / 'target-errors', resident_bytes=1)
    try:
        root = records[0]
        body = encode_native_state(root.block)
        with pytest.raises(ValueError):
            target.put_native_archive_payloads([(root.id, body), (records[-1].id, b'bad')], bounded=True)
        assert len(target) == 0
        assert not list(target.directory.glob('archive-*'))
        target.put_native_archive_payloads([(root.id, body)], bounded=True)
        # Existing bytes are compared before any new archive item is admitted.
        target._path(root.id).write_bytes(b'collision')
        with pytest.raises(ValueError, match='collision'):
            target.put_native_archive_payloads([(root.id, body)], bounded=True)
        assert len(target) == 1
        assert not list(target.directory.glob('archive-*'))
    finally:
        target.close()
        engine.store.close()


def test_fit_store_context_cleans_on_exception(tmp_path):
    from natlang_neuralese.tune.build import fit_store_context
    with pytest.raises(RuntimeError, match='failure'):
        with fit_store_context({'store_scratch': str(tmp_path)}) as store:
            directory = store.directory
            assert directory.exists()
            raise RuntimeError('failure')
    assert not directory.exists()


def test_bounded_rejects_forged_gradient_origin_without_publication(tmp_path):
    from dataclasses import replace
    from natlang_neuralese.model.native_artifact import encode_native_tensor_artifact, native_tensor_artifact_id
    engine = engine_fixture(tmp_path, 'lfm', service_template=True)
    target = SimpleNamespace(store=TensorStore(tmp_path / 'target-gradient', resident_bytes=1))
    try:
        adapter, _ = nonzero_adapter(engine)
        result = execute_derivative_program(engine, program(engine, adapter, 'sgd'))
        gradient_id = result['gradients']['a']['id']
        gradient = engine.store.get_native_tensor_artifact(gradient_id).artifact
        persist_artifacts(engine, [gradient_id, result['outputs']['a2']], tmp_path / 'artifacts', tmp_path / 'states')
        import_archive(target, [tmp_path / 'artifacts'], [tmp_path / 'states'])
        count = len(target.store)
        provenance = copy.deepcopy(gradient.provenance)
        provenance['evaluation_inputs']['a']['value_id'] = adapter.id
        forged = replace(gradient, provenance=provenance)
        with pytest.raises(ValueError):
            target.store.put_native_archive_payloads([(native_tensor_artifact_id(forged), encode_native_tensor_artifact(forged))], bounded=True)
        assert len(target.store) == count
        assert not list(target.store.directory.glob('archive-*'))
    finally:
        target.store.close()
        engine.store.close()


def test_repeat_persist_uses_dependency_metadata_and_repairs_missing_descendant(tmp_path, monkeypatch):
    from test_native_state_batch_import import _history
    engine, records = _history(tmp_path, 'mellum', 'adam')
    target = SimpleNamespace(store=TensorStore(tmp_path / 'persist-store', resident_bytes=1))
    try:
        persist_artifacts(engine, [records[-1].id], tmp_path / 'source-artifacts', tmp_path / 'source-states')
        import_archive(target, [tmp_path / 'source-artifacts'], [tmp_path / 'source-states'])
        output, states = tmp_path / 'out-artifacts', tmp_path / 'out-states'
        persist_artifacts(target, [records[-1].id], output, states)
        target.store._resident.clear()
        target.store._resident_size = 0
        loads = []
        original = target.store._load
        def count_load(ident):
            loads.append(ident)
            return original(ident)
        monkeypatch.setattr(target.store, '_load', count_load)
        persist_artifacts(target, [records[-1].id], output, states)
        assert loads == []
        descendant = states / f'{records[0].id}.nzs'
        descendant.unlink()
        persist_artifacts(target, [records[-1].id], output, states)
        assert descendant.exists()
        assert set(loads) <= {records[0].id}
        assert target.store.get_native_state(records[-1].id) is not None
    finally:
        target.store.close()
        engine.store.close()
