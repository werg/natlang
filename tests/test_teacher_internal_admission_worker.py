import json
import hashlib
from pathlib import Path

import pytest

from scripts.coreweave import teacher_internal_admission_worker as worker
from scripts.coreweave import teacher_internal_conversion as conversion_module
from scripts.coreweave.teacher_tranche_admission import file_binding, preceding_tranches
from scripts.neuralese_data.build_native_semantic_pairs_e2e import CompressedRows
from scripts.neuralese_data.build_verified_teacher_trajectory_graphs import (
    graph_content_sha256, teacher_action_graph_id,
)


def write_json(path: Path, value: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))
    return file_binding(path)


def register_proof(root: Path, corpus_id: str, proof_ref: dict) -> None:
    entry = {'id': corpus_id, 'path': f'runs/{corpus_id}', 'training_admission': True,
             'source_binding': {'registration_proof': {
                 'path': f'runs/{corpus_id}/evidence/registration-proof.json',
                 'sha256': proof_ref['sha256'], 'bytes': proof_ref['bytes']}}}
    write_json(root / 'training/neuralese_corpora.json', {'corpora': [entry]})
    write_json(root / 'training/corpus-manifests' / f'{corpus_id}.json', {
        'id': corpus_id, 'path': entry['path'], 'files': [{
            'path': 'evidence/registration-proof.json', 'sha256': proof_ref['sha256'],
            'bytes': proof_ref['bytes']}],
    })


def zero_conversion(tmp_path: Path) -> Path:
    source = tmp_path / 'source.jsonl'
    source_row = {'id': 'source-row-1', 'trajectory': [{
        'phase': 'action', 'invocation_id': 'invocation-1', 'request_sha256': 'a' * 64,
        'raw_response_sha256': 'b' * 64, 'assistant': {'content': 'captured answer'},
        'model_response': {'raw_calls': []},
    }]}
    source.write_text(json.dumps(source_row, sort_keys=True) + '\n')
    source_row_sha = hashlib.sha256(source.read_bytes().rstrip(b'\r\n')).hexdigest()
    source_ref = {**file_binding(source), 'rows': 1,
                  'row_bindings': [{'id': 'source-row-1', 'sha256': source_row_sha}]}
    source_proof_ref = write_json(tmp_path / 'source-proof.json', {
        'status': 'passed', 'training_admission': False, 'source': source_ref,
        'actor': {'sha256': 'a' * 64}, 'execution_identity': {'sha256': 'b' * 64},
        'current_protection': {'sha256': 'd' * 64}, 'rows': 1, 'teacher_action_turns': 1,
    })
    settings_ref = write_json(tmp_path / 'settings.json', {
        'teacher_namespace': 'qwen36-verified-positive-native-graphs-20261022-recovery7853',
        'internal_continuation': {'schema': 'natlang.teacher-internal-source-continuation/1',
                                  'training_admission': False},
    })
    predecessor_id = 'registered-native-predecessor'
    package = tmp_path / 'runs' / predecessor_id
    manifest_files, graph_refs, receipts, models = [], {}, {}, []
    decisions_row = {'id': 'source-row-1:decision:0000',
        'source_ref': {'trajectory_id': 'source-row-1', 'invocation_id': 'invocation-1'},
        'decision': {'index': 0, 'training_approved': True, 'failed_action': False,
                     'source_raw_response_sha256': 'b' * 64},
        'training_admission': {'approved': True, 'reason': 'qualified'}}
    for model in ('lfm25', 'mellum21'):
        fingerprint = ('be949ec48f44e17b2b17e1131aeb1dab4262175ba20f9783ecd8e59c27a58105'
                       if model == 'lfm25' else
                       'fbf22d8773d09be802cb9977895067db36cdfba747fcb24b02ddbf1312d3778f')
        graph_id = 'text:' + teacher_action_graph_id('source-row-1', source_row_sha, 0,
            'keep' if model == 'lfm25' else None, 'mask-runtime-failures')
        graph = {'schema': 'natlang.neuralese-native-graph/1', 'id': graph_id,
                 'split': 'train', 'source_groups': ['fixture:train'],
                 'tokenizer_fingerprint': fingerprint,
                 'nodes': [{'name': 'action', 'token_ids': [1, 2], 'target_weights': [0, 1]}]}
        graph_path = package / model / 'train/native-graphs.jsonl.zst'
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        graph_writer = CompressedRows(graph_path); graph_writer.write(graph); graph_writer.finish()
        graph_ref = file_binding(graph_path); graph_refs[model] = graph_ref
        manifest_files.append({'path': f'{model}/train/native-graphs.jsonl.zst',
                               'sha256': graph_ref['sha256'], 'bytes': graph_ref['bytes']})
    write_json(tmp_path / 'training/neuralese_corpora.json', {'corpora': [
        {'id': predecessor_id, 'path': f'runs/{predecessor_id}', 'training_admission': True}]})
    write_json(tmp_path / 'training/corpus-manifests' / f'{predecessor_id}.json', {
        'id': predecessor_id, 'path': f'runs/{predecessor_id}', 'files': manifest_files})
    converter = Path(conversion_module.__file__).resolve().parents[2] / 'scripts/neuralese_data/build_verified_teacher_trajectory_graphs.py'
    for model in ('lfm25', 'mellum21'):
        fingerprint = ('be949ec48f44e17b2b17e1131aeb1dab4262175ba20f9783ecd8e59c27a58105'
                       if model == 'lfm25' else
                       'fbf22d8773d09be802cb9977895067db36cdfba747fcb24b02ddbf1312d3778f')
        decision_dir = tmp_path / model / 'decision-admission-cache'
        decisions_path = decision_dir / 'decisions.jsonl'
        decisions_path.parent.mkdir(parents=True, exist_ok=True)
        decisions_path.write_text(json.dumps(decisions_row) + '\n')
        decision_receipt_ref = write_json(decision_dir / 'receipt.json', {
            'schema': 'natlang.verified-teacher-decision-materialization/1',
            'identity': {'source_sha256': source_ref['sha256'], 'direct_answers': True,
                         'runtime': {'reader.js': 'c' * 64}},
            'output_sha256': file_binding(decisions_path)['sha256'],
            'summary': {'training_decisions': 1, 'output': 'decisions.jsonl'},
        })
        outputs = {}
        for split in ('train', 'test'):
            output_path = tmp_path / model / split / 'native-graphs.jsonl.zst'
            output_path.parent.mkdir(parents=True, exist_ok=True)
            writer = CompressedRows(output_path); outputs[split] = {
                **writer.finish(), 'tokens': 0, 'max_row_tokens': 0}
        graph = {'schema': 'natlang.neuralese-native-graph/1', 'id':
                 'text:' + teacher_action_graph_id('source-row-1', source_row_sha, 0,
                   'keep' if model == 'lfm25' else None, 'mask-runtime-failures'),
                 'split': 'train', 'source_groups': ['fixture:train'],
                 'tokenizer_fingerprint': fingerprint,
                 'nodes': [{'name': 'action', 'token_ids': [1, 2], 'target_weights': [0, 1]}]}
        content_sha = graph_content_sha256(graph)
        receipt = {'schema': 'natlang.verified-teacher-native-graph-build/1', 'status': 'passed',
            'model_key': model, 'source': {'path': str(source), 'sha256': source_ref['sha256'], 'rows': 1,
                'row_bindings': [{'id': 'source-row-1', 'sha256': source_row_sha}]},
            'tokenizer': {'expanded_fingerprint': fingerprint,
                **({'history_reasoning': {'policy': 'keep'}} if model == 'lfm25' else {})},
            **({'explicit_history_reasoning_policy': 'keep'} if model == 'lfm25' else {}),
            'failed_assistant_history_policy': 'mask-runtime-failures', 'turns': 1,
            'action_omissions': [], 'outputs': outputs,
            'decision_admission': {'path': decision_receipt_ref['path'],
                'sha256': decision_receipt_ref['sha256'],
                'output_sha256': file_binding(decisions_path)['sha256'],
                'summary': {'training_decisions': 1, 'output': 'decisions.jsonl'}},
            'deduplication': {'inputs': [{'path': graph_refs[model]['path'],
                'sha256': graph_refs[model]['sha256'], 'bytes': graph_refs[model]['bytes'], 'rows': 1}],
                'emitted_rows': 0, 'omitted_rows': 1,
                'omissions': [{'id': graph['id'], 'content_sha256': content_sha,
                    'reason': 'exact token IDs and target weights already represented',
                    'represented_by': {'path': graph_refs[model]['path'], 'id': graph['id']}}]},
        }
        receipts[model] = write_json(tmp_path / model / 'receipt.json', receipt)
        models.append({'model': model, 'receipt': receipts[model]})
    conversion = tmp_path / 'conversion'
    write_json(conversion / 'both-port-conversion.json', {
        'schema': 'natlang.teacher-internal-both-port-conversion/1', 'batch': 21, 'tranche': 1,
        'source': source_ref, 'source_proof': source_proof_ref, 'settings': settings_ref,
        'converter': file_binding(converter), 'protection_manifest': {'sha256': 'd' * 64},
        'training_registration': False, 'training_exposure': False,
        'admitted_predecessors': [predecessor_id], 'same_actor_source_boundary_inputs': [],
        'models': models,
    })
    return conversion


def test_zero_rows_require_both_exact_dedup_receipts_and_bound_inputs(tmp_path, monkeypatch):
    conversion = zero_conversion(tmp_path)
    monkeypatch.setattr(worker, 'ROOT', tmp_path)
    monkeypatch.setattr(worker, '_registered_ids', lambda: set())
    monkeypatch.setattr(conversion_module, '_legacy_inputs', lambda root, model: [])
    ctx = worker.Tranche(21, 1, teacher_namespace='qwen36-verified-positive-native-graphs-20261022-recovery7853')
    result = worker._publish_if_new_delta(ctx, conversion)
    assert result['status'] == 'verified-zero-delta'
    assert result['training_admission'] is False
    assert result['training_exposure'] is False
    conversion_proof = json.loads((conversion / 'both-port-conversion.json').read_text())
    source_proof = json.loads(Path(conversion_proof['source_proof']['path']).read_text())
    checked = conversion_module._verified_zero_conversion(tmp_path, conversion / 'both-port-conversion.json',
        worker.Tranche(21, 2, teacher_namespace='qwen36-verified-positive-native-graphs-20261022-recovery7853'), source_proof, {
            'lfm25': {'tokenizer_fingerprint': 'be949ec48f44e17b2b17e1131aeb1dab4262175ba20f9783ecd8e59c27a58105'},
            'mellum21': {'tokenizer_fingerprint': 'fbf22d8773d09be802cb9977895067db36cdfba747fcb24b02ddbf1312d3778f'},
        })
    assert (checked['batch'], checked['tranche']) == (21, 1)
    assert checked['training_admission'] is False

    receipt_path = Path(json.loads((conversion / 'both-port-conversion.json').read_text())['models'][0]['receipt']['path'])
    receipt = json.loads(receipt_path.read_text())
    receipt['deduplication']['omissions'][0]['represented_by']['path'] = str(tmp_path / 'unbound')
    changed_ref = write_json(receipt_path, receipt)
    conversion_path = conversion / 'both-port-conversion.json'
    conversion_data = json.loads(conversion_path.read_text())
    conversion_data['models'][0]['receipt'] = changed_ref
    write_json(conversion_path, conversion_data)
    with pytest.raises(ValueError, match='zero-boundary omission'):
        worker._publish_if_new_delta(ctx, conversion)


@pytest.mark.parametrize('tamper', [None, 'source', 'cache', 'policy', 'manifest'])
def test_historical_cache_receipt_rechecks_restored_source(tmp_path, monkeypatch, tamper):
    from natlang_neuralese.common import row_index
    from natlang_neuralese.common.hashing import canonical_json_sha256_hex
    conversion = zero_conversion(tmp_path)
    monkeypatch.setattr(worker, 'ROOT', tmp_path)
    monkeypatch.setattr(conversion_module, '_legacy_inputs', lambda root, model: [])
    proof_path = conversion / 'both-port-conversion.json'
    proof = json.loads(proof_path.read_text())
    sources, indexes = [], []
    for model in proof['models']:
        receipt_path = Path(model['receipt']['path'])
        receipt = json.loads(receipt_path.read_text())
        original = receipt['deduplication']['inputs'][0]
        source = Path(original['path']); sources.append(source)
        contents, rows = conversion_module._validated_content_fingerprints(
            source, receipt['tokenizer']['expanded_fingerprint'])
        key = {'source_sha256': original['sha256'],
               'tokenizer_fingerprint': receipt['tokenizer']['expanded_fingerprint'],
               'policy_sha256': conversion_module._content_index_policy_sha256(),
               'kind': 'teacher-content-fingerprints-v1'}
        index = tmp_path / (model['model'] + '-historical-index.pkl'); indexes.append(index)
        value = {'key': key, 'fingerprints': contents, 'rows': rows,
                 'metadata_sha256': canonical_json_sha256_hex({'rows': rows, 'fingerprints': contents})}
        if tamper == 'policy':
            value = {**value, 'key': {**key, 'policy_sha256': '0' * 64}}
        row_index.save(index, value)
        historical_path = ('registered-cache:registered-native-predecessor/'
                           + model['model'] + '/train/native-graphs.jsonl.zst')
        receipt['deduplication']['inputs'][0] = {
            **original, 'path': historical_path, 'content_index_cache': file_binding(index),
            'content_index_key': key, 'cache_only_source': True}
        receipt['deduplication']['omissions'][0]['represented_by']['path'] = historical_path
        model['receipt'] = write_json(receipt_path, receipt)
    write_json(proof_path, proof)
    if tamper == 'source':
        sources[0].write_bytes(b'changed restored source')
    elif tamper == 'cache':
        indexes[0].write_bytes(b'changed index')
    elif tamper == 'manifest':
        path = tmp_path / 'training/corpus-manifests/registered-native-predecessor.json'
        value = json.loads(path.read_text()); value['files'][0]['sha256'] = '0' * 64
        write_json(path, value)
    ctx = worker.Tranche(21, 1,
        teacher_namespace='qwen36-verified-positive-native-graphs-20261022-recovery7853')
    if tamper:
        with pytest.raises(ValueError, match='(restored dedup source|historical dedup content index)'):
            worker._publish_if_new_delta(ctx, conversion)
    else:
        result = worker._publish_if_new_delta(ctx, conversion)
        assert result['status'] == 'verified-zero-delta'
        assert result['training_admission'] is False
        assert result['training_exposure'] is False


@pytest.mark.parametrize('field,value', [('content_sha256', '0' * 64), ('represented_id', 'forged-row-id')])
def test_zero_row_rejects_forged_content_or_represented_id_with_same_graph_bytes(tmp_path, monkeypatch, field, value):
    conversion = zero_conversion(tmp_path)
    monkeypatch.setattr(worker, 'ROOT', tmp_path)
    monkeypatch.setattr(worker, '_registered_ids', lambda: set())
    monkeypatch.setattr(conversion_module, '_legacy_inputs', lambda root, model: [])
    proof_path = conversion / 'both-port-conversion.json'
    proof = json.loads(proof_path.read_text())
    receipt_path = Path(proof['models'][0]['receipt']['path'])
    receipt = json.loads(receipt_path.read_text())
    omission = receipt['deduplication']['omissions'][0]
    if field == 'content_sha256':
        omission['content_sha256'] = value
    else:
        omission['represented_by']['id'] = value
    receipt_ref = write_json(receipt_path, receipt)
    proof['models'][0]['receipt'] = receipt_ref
    write_json(proof_path, proof)
    with pytest.raises(ValueError, match='(omission|deduplication identity)'):
        worker._publish_if_new_delta(worker.Tranche(21, 1,
            teacher_namespace='qwen36-verified-positive-native-graphs-20261022-recovery7853'), conversion)


def test_zero_row_requires_readable_empty_native_output_streams(tmp_path, monkeypatch):
    conversion = zero_conversion(tmp_path)
    monkeypatch.setattr(worker, 'ROOT', tmp_path)
    monkeypatch.setattr(worker, '_registered_ids', lambda: set())
    monkeypatch.setattr(conversion_module, '_legacy_inputs', lambda root, model: [])
    proof_path = conversion / 'both-port-conversion.json'
    proof = json.loads(proof_path.read_text())
    receipt_path = Path(proof['models'][0]['receipt']['path'])
    receipt = json.loads(receipt_path.read_text())
    output_path = receipt_path.parent / 'train/native-graphs.jsonl.zst'
    output_path.write_bytes(b'not-a-zstd-stream')
    output_ref = file_binding(output_path)
    receipt['outputs']['train'].update({key: output_ref[key] for key in ('path', 'sha256', 'bytes')})
    receipt_ref = write_json(receipt_path, receipt)
    proof['models'][0]['receipt'] = receipt_ref
    write_json(proof_path, proof)
    with pytest.raises((ValueError, OSError, RuntimeError)):
        worker._publish_if_new_delta(worker.Tranche(21, 1,
            teacher_namespace='qwen36-verified-positive-native-graphs-20261022-recovery7853'), conversion)


def test_registered_exact_conversion_is_idempotent_but_immutable_conflicts_fail(tmp_path, monkeypatch):
    conversion = tmp_path / 'conversion.json'
    conversion_ref = write_json(conversion, {'conversion': 'exact'})
    corpus_id = 'qwen36-verified-positive-native-graphs-20261023-internal-batch0021-tranche0001'
    proof_path = tmp_path / 'runs' / corpus_id / 'evidence/registration-proof.json'
    proof = {'schema': 'natlang.teacher-internal-tranche-registration-proof/1',
             'corpus_id': corpus_id, 'training_admission': True,
             'conversion': conversion_ref}
    proof_ref = write_json(proof_path, proof)
    register_proof(tmp_path, corpus_id, proof_ref)
    monkeypatch.setattr(worker, 'ROOT', tmp_path)
    assert worker._existing_conversion(corpus_id, conversion_ref)['idempotent'] is True
    with pytest.raises(ValueError, match='different conversion'):
        worker._existing_conversion(corpus_id, {'sha256': '0' * 64,
                                                'bytes': conversion_ref['bytes']})


@pytest.mark.parametrize('missing', ['manifest', 'proof'])
def test_registered_conversion_with_missing_manifest_or_proof_fails_closed(tmp_path, monkeypatch, missing):
    conversion_ref = write_json(tmp_path / 'conversion.json', {'conversion': 'exact'})
    corpus_id = 'qwen36-verified-positive-native-graphs-20261023-internal-batch0021-tranche0001'
    proof_path = tmp_path / 'runs' / corpus_id / 'evidence/registration-proof.json'
    proof_ref = write_json(proof_path, {
        'schema': 'natlang.teacher-internal-tranche-registration-proof/1',
        'corpus_id': corpus_id, 'training_admission': True, 'conversion': conversion_ref})
    register_proof(tmp_path, corpus_id, proof_ref)
    monkeypatch.setattr(worker, 'ROOT', tmp_path)
    missing_path = (tmp_path / 'training/corpus-manifests' / f'{corpus_id}.json'
                    if missing == 'manifest' else proof_path)
    missing_path.unlink()
    with pytest.raises(FileNotFoundError):
        worker._existing_conversion(corpus_id, conversion_ref)


def test_repeated_poll_recognizes_exact_registered_conversion_without_republishing(tmp_path, monkeypatch):
    conversion = tmp_path / 'conversion'
    models = []
    for model in ('lfm25', 'mellum21'):
        models.append({'model': model, 'receipt': write_json(conversion / model / 'receipt.json', {
            'outputs': {'train': {'rows': 1}},
        })})
    conversion_ref = write_json(conversion / 'both-port-conversion.json', {'models': models})
    corpus_id = ('qwen36-verified-positive-native-graphs-20261023-internal-'
                 'batch0021-tranche0001')
    proof_ref=write_json(tmp_path / 'runs' / corpus_id / 'evidence/registration-proof.json', {
        'schema': 'natlang.teacher-internal-tranche-registration-proof/1',
        'corpus_id': corpus_id, 'training_admission': True,
        'conversion': conversion_ref,
    })
    register_proof(tmp_path, corpus_id, proof_ref)
    monkeypatch.setattr(worker, 'ROOT', tmp_path)
    monkeypatch.setattr(worker, '_registered_ids', lambda: {corpus_id})
    monkeypatch.setattr(worker.subprocess, 'run', lambda *args, **kwargs: pytest.fail('republished exact conversion'))
    first = worker._publish_if_new_delta(worker.Tranche(21, 1), conversion)
    second = worker._publish_if_new_delta(worker.Tranche(21, 1), conversion)
    assert first['idempotent'] is True
    assert second == first


def internal_predecessor_fixture(root: Path, *, source=b'raw source', actor='a' * 64,
                                 runtime='b' * 64, lfm_history='keep'):
    corpus_id = ('qwen36-verified-positive-native-graphs-20261023-internal-'
                 'batch0021-tranche0000')
    package = root / 'runs' / corpus_id
    package.mkdir(parents=True)
    source_path = package / 'evidence/native-trajectories.jsonl'
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(source)
    source_ref = file_binding(source_path)
    source_proof_path = package / 'evidence/canonical-source-verification.json'
    source_proof_ref = write_json(source_proof_path, {
        'status': 'passed', 'training_admission': False, 'source': source_ref,
        'actor': {'sha256': actor}, 'execution_identity': {'sha256': runtime},
        'current_protection': {'sha256': 'd' * 64},
    })
    source_result_ref = write_json(package / 'evidence/source-result.json', {
        'training_admission': False, 'training_exposure': False,
        'source_proof': source_proof_ref,
    })
    model_proofs = {}
    files = []
    for rel in ('evidence/native-trajectories.jsonl',
                'evidence/canonical-source-verification.json', 'evidence/source-result.json'):
        ref = file_binding(package / rel)
        files.append({'path': rel, 'sha256': ref['sha256'], 'bytes': ref['bytes']})
    for model in ('lfm25', 'mellum21'):
        explicit = lfm_history if model == 'lfm25' else None
        receipt_ref = write_json(package / f'{model}/receipt.json', {
            'status': 'passed', 'model_key': model,
            'source': {'sha256': source_ref['sha256']},
            'tokenizer': {'expanded_fingerprint': 'c' * 64},
            'explicit_history_reasoning_policy': explicit,
            'failed_assistant_history_policy': 'mask-runtime-failures',
        })
        graph_path = package / f'{model}/train/native-graphs.jsonl.zst'
        graph_path.parent.mkdir(parents=True)
        graph_path.write_bytes((model + ' graph').encode())
        graph_ref = file_binding(graph_path)
        files.extend([
            {'path': f'{model}/receipt.json', 'sha256': receipt_ref['sha256'], 'bytes': receipt_ref['bytes']},
            {'path': f'{model}/train/native-graphs.jsonl.zst', 'sha256': graph_ref['sha256'], 'bytes': graph_ref['bytes']},
        ])
        model_proofs[model] = {'receipt': receipt_ref, 'train': graph_ref}
    conversion_ref = write_json(package / 'evidence/both-port-conversion.json', {
        'schema': 'natlang.teacher-internal-both-port-conversion/1',
        'batch': 21, 'tranche': 0,
        'source': source_ref, 'training_registration': False, 'training_exposure': False,
        'models': [{'model': model, 'receipt': model_proofs[model]['receipt']}
                   for model in ('lfm25', 'mellum21')],
    })
    files.append({'path': 'evidence/both-port-conversion.json',
                  'sha256': conversion_ref['sha256'], 'bytes': conversion_ref['bytes']})
    proof_path = package / 'evidence/registration-proof.json'
    proof_ref = write_json(proof_path, {
        'schema': 'natlang.teacher-internal-tranche-registration-proof/1',
        'status': 'passed', 'training_admission': True, 'training_exposure': False,
        'corpus_id': corpus_id,
        'source_namespace': 'qwen36-verified-positive-native-graphs-20261022-recovery7853',
        'batch': 21, 'tranche': 0,
        'source': source_ref, 'source_verification': source_proof_ref,
        'source_result': source_result_ref, 'conversion': conversion_ref,
        'current_protection_manifest_sha256': 'd' * 64,
        'models': model_proofs,
    })
    files.append({'path': 'evidence/registration-proof.json',
                  'sha256': proof_ref['sha256'], 'bytes': proof_ref['bytes']})
    entry = {'id': corpus_id, 'path': f'runs/{corpus_id}', 'training_admission': True,
             'source_binding': {'registration_proof': {
                 'path': f'runs/{corpus_id}/evidence/registration-proof.json',
                 'sha256': proof_ref['sha256'], 'bytes': proof_ref['bytes']}}}
    write_json(root / 'training/neuralese_corpora.json', {'corpora': [entry]})
    write_json(root / 'training/corpus-manifests' / f'{corpus_id}.json', {
        'id': corpus_id, 'path': entry['path'], 'files': files,
    })
    rows = [{'id': corpus_id, 'outputs': {
        model: file_binding(package / f'{model}/train/native-graphs.jsonl.zst')
        for model in ('lfm25', 'mellum21')}, 'registry_entry': entry}]
    return rows, source_ref


def test_internal_predecessor_graphs_keep_exact_historical_identity_without_relabeling(tmp_path):
    rows, source_ref = internal_predecessor_fixture(tmp_path)
    ctx = worker.Tranche(21, 1, teacher_namespace='qwen36-verified-positive-native-graphs-20261022-recovery7853')
    source_proof = {'source': {'sha256': source_ref['sha256']},
                    'actor': {'sha256': 'a' * 64},
                    'execution_identity': {'sha256': 'b' * 64},
                    'current_protection': {'sha256': 'd' * 64}}
    contracts = {model: {'tokenizer_fingerprint': 'c' * 64}
                 for model in ('lfm25', 'mellum21')}
    checked = conversion_module._verified_internal_predecessors(
        tmp_path, ctx, source_proof, contracts, rows)
    assert [(row['batch'], row['tranche']) for row in checked] == [(21, 0)]
    admitted, rejected = preceding_tranches(
        tmp_path, tmp_path / 'spool', ctx,
        accounted_internal_boundaries={(row['batch'], row['tranche']) for row in checked})
    assert admitted == [] and rejected == []
    for changed in (
        {**source_proof, 'actor': {'sha256': 'e' * 64}},
        {**source_proof, 'execution_identity': {'sha256': 'f' * 64}},
        {**source_proof, 'current_protection': {'sha256': '0' * 64}},
    ):
        # These values describe the current boundary. Historical graphs retain
        # their exact actor/runtime/protection and remain dedup inputs.
        assert conversion_module._verified_internal_predecessors(
            tmp_path, ctx, changed, contracts, rows)
    source_path = Path(source_ref['path'])
    source_path.write_bytes(b'changed source')
    with pytest.raises(ValueError, match='source custody'):
        conversion_module._verified_internal_predecessors(
            tmp_path, ctx, source_proof, contracts, rows)


def test_internal_predecessor_rejects_registered_history_policy_mismatch(tmp_path):
    rows, source_ref = internal_predecessor_fixture(tmp_path, lfm_history=None)
    source_proof = {'source': {'sha256': source_ref['sha256']},
                    'actor': {'sha256': 'a' * 64},
                    'execution_identity': {'sha256': 'b' * 64},
                    'current_protection': {'sha256': 'd' * 64}}
    contracts = {model: {'tokenizer_fingerprint': 'c' * 64}
                 for model in ('lfm25', 'mellum21')}
    with pytest.raises(ValueError, match='history policy'):
        conversion_module._verified_internal_predecessors(
            tmp_path, worker.Tranche(21, 1, teacher_namespace='qwen36-verified-positive-native-graphs-20261022-recovery7853'), source_proof, contracts, rows)


class StopPolling(Exception):
    pass


@pytest.mark.parametrize('tamper', [None,'completed-result','source-cases','relocation-bytes'])
def test_source_retry_uses_exact_recovered_closure_without_cloud_fetch(tmp_path,monkeypatch,tamper):
    spool=tmp_path/'spool';monkeypatch.setattr(worker,'SPOOL',spool)
    ctx=worker.Tranche(63,7,teacher_namespace='fixture-teacher')
    target='a'*64
    relocation=tmp_path/'recovered-relocation.json'
    reference=write_json(relocation,{'source_cases_sha256':'b'*64 if tamper=='source-cases'else target})
    if tamper=='completed-result':
        reference=write_json(tmp_path/'relocation-result.json',{
            'schema':'natlang.teacher-ready-packet-relocation-result/1',
            'relocation_declaration':reference})
    write_json(spool/'source-executions'/ctx.corpus_id/'execution.json',{
        'ready_packet_relocation':reference})
    if tamper=='relocation-bytes':relocation.write_text('{}')
    monkeypatch.setattr(worker,'_exists_valid_source',lambda ctx:False)
    calls=[];monkeypatch.setattr(worker,'_run',lambda command,*args:calls.append(command))
    if tamper in ('source-cases','relocation-bytes'):
        with pytest.raises(ValueError):worker._ensure_source(ctx,target)
        assert not calls
    else:
        worker._ensure_source(ctx,target)
        assert len(calls)==1
        assert calls[0][1].endswith('/teacher_internal_source.py')
        assert calls[0][calls[0].index('--ready-packet-relocation')+1]==str(relocation)


def polling_fixture(tmp_path,monkeypatch):
    spool=tmp_path/'spool';monkeypatch.setattr(worker,'SPOOL',spool)
    settings=tmp_path/'settings.json';write_json(settings,{'teacher_namespace':'fixture-teacher'})
    publisher=tmp_path/'publisher.json';write_json(publisher,{
        'schema':'natlang.teacher-incremental-export-publisher/1','directory_root':str(spool),'first_export_batch':41})
    for name,path in [('ORIGINAL_SETTINGS',settings),('PUBLISHER_CONFIG',publisher),
                      ('DEPLOYMENT',tmp_path/'deployment.json'),('TOKENIZERS',tmp_path/'tokenizers.json'),
                      ('PROTECTION',tmp_path/'protection.json')]:
        if not path.exists():write_json(path,{'revision':0})
        monkeypatch.setattr(worker,name,path)
    monkeypatch.setattr(worker,'ORIGINAL_SETTINGS_SHA',file_binding(settings)['sha256'])
    ctx=worker.Tranche(41,0,teacher_namespace='fixture-teacher')
    private=spool/'batch-0041/private-receipts/tranche-0000.json';write_json(private,{'revision':0})
    write_json(ctx.admission(spool)/'canonical-source-verification.json',{'fixture':'already-normal-verified'})
    conversion=spool/'conversions/batch-0041/tranche-0000-attempt-0000';models=[]
    for model in ('lfm25','mellum21'):
        ref=write_json(conversion/(model+'.json'),{'outputs':{'train':{'rows':0}}})
        models.append({'model':model,'receipt':ref})
    write_json(conversion/'both-port-conversion.json',{'models':models})
    tranche={'context':{'source-cases.jsonl':{'sha256':'a'*64}}}
    monkeypatch.setattr(worker,'_boundaries',lambda *args:[(ctx,tranche)])
    monkeypatch.setattr(worker,'_next_conversion',lambda ctx:conversion)
    monkeypatch.setattr(worker,'_publish_if_new_delta',lambda *args:{'status':'verified-zero-delta'})
    return private


def test_poll_skips_only_completed_unchanged_work_and_rechecks_changed_custody_and_policy(tmp_path,monkeypatch):
    private=polling_fixture(tmp_path,monkeypatch);verified=[];polls=[]
    monkeypatch.setattr(worker,'_ensure_source',lambda ctx,target:verified.append(ctx.corpus_id))
    def sleep(seconds):
        polls.append(seconds)
        if len(polls)==2:write_json(private,{'revision':1})
        if len(polls)==3:write_json(worker.PROTECTION,{'revision':1})
        if len(polls)==4:raise StopPolling
    monkeypatch.setattr(worker.time,'sleep',sleep)
    with pytest.raises(StopPolling):worker.run(once=False,poll_seconds=60)
    assert len(verified)==3
    records=[json.loads(line)for path in (worker.SPOOL/'worker-logs').glob('worker-*.jsonl')for line in path.read_text().splitlines()]
    assert len(records)==3 and all(row['status']=='verified-zero-delta'for row in records)
    assert all(row['training_admission']is False and row['training_exposure']is False for row in records)
    assert records[0]['private_receipt']!=records[1]['private_receipt']


def test_failed_poll_retries_before_any_completed_scheduling_reuse(tmp_path,monkeypatch):
    polling_fixture(tmp_path,monkeypatch);attempts=[];polls=[]
    def ensure(ctx,target):
        attempts.append(ctx.corpus_id)
        if len(attempts)==1:raise ValueError('missing exact source; no admission')
    monkeypatch.setattr(worker,'_ensure_source',ensure)
    def sleep(seconds):
        polls.append(seconds)
        if len(polls)==3:raise StopPolling
    monkeypatch.setattr(worker.time,'sleep',sleep)
    with pytest.raises(StopPolling):worker.run(once=False,poll_seconds=60)
    assert len(attempts)==2
    records=[json.loads(line)for path in (worker.SPOOL/'worker-logs').glob('worker-*.jsonl')for line in path.read_text().splitlines()]
    assert [row['status']for row in records]==['error','verified-zero-delta']


@pytest.mark.parametrize('tamper', [None, 'source', 'outcomes', 'source-result', 'row-type'])
def test_zero_positive_source_is_terminal_without_numeric_conversion(tmp_path, monkeypatch, tamper):
    polling_fixture(tmp_path, monkeypatch)
    ctx=worker.Tranche(41,0,teacher_namespace='fixture-teacher')
    source=ctx.admission(worker.SPOOL)/'native-trajectories.jsonl'
    source.write_bytes(b'')
    proof={'schema':'natlang.teacher-live-tranche-canonical-source-verification/1',
           'status':'passed','training_admission':False,'rows':0,
           'source':{**file_binding(source),'rows':0},'teacher_action_turns':0,
           'selected_outputs':1,'rejected_outputs':[{'source_output':'outputs/failed.jsonl',
               'sha256':'a'*64,'reason':'oracle rejected incorrect execution'}]}
    if tamper=='outcomes':proof['rejected_outputs']=[]
    if tamper=='row-type':proof['rows']=False
    ref=write_json(ctx.admission(worker.SPOOL)/'canonical-source-verification.json',proof)
    write_json(worker.SPOOL/'source-executions'/ctx.corpus_id/'source-result.json',{
        'source_proof':ref,'source':proof['source'],'training_admission':False,
        'training_exposure':False,'source_disposition':(
            'positive-source'if tamper=='source-result'else'verified-zero-positive-source')})
    if tamper=='source':source.write_bytes(b'not empty')
    monkeypatch.setattr(worker,'_ensure_source',lambda *args:None)
    monkeypatch.setattr(worker,'_next_conversion',lambda *args:pytest.fail('converted an empty source'))
    worker.run(once=True,poll_seconds=60)
    records=[json.loads(line)for p in(worker.SPOOL/'worker-logs').glob('worker-*.jsonl')
             for line in p.read_text().splitlines()]
    assert len(records)==1
    if tamper:
        assert records[0]['status']=='error'
    else:
        assert records[0]['status']=='verified-zero-positive-source'
        assert records[0]['row_counts']=={'lfm25':0,'mellum21':0}
    assert records[0]['training_admission']is False
    assert records[0]['training_exposure']is False


@pytest.mark.parametrize('changed',['source-proof','source-result','conversion','model-receipt','new-conversion'])
def test_poll_rechecks_changed_source_and_conversion_stages(tmp_path,monkeypatch,changed):
    polling_fixture(tmp_path,monkeypatch);verified=[];polls=[]
    ctx=worker.Tranche(41,0,teacher_namespace='fixture-teacher')
    conversion=worker.SPOOL/'conversions/batch-0041/tranche-0000-attempt-0000'
    monkeypatch.setattr(worker,'_ensure_source',lambda ctx,target:verified.append(ctx.corpus_id))
    def sleep(seconds):
        polls.append(seconds)
        if len(polls)==2:
            if changed=='source-proof':path=ctx.admission(worker.SPOOL)/'canonical-source-verification.json'
            elif changed=='source-result':path=worker.SPOOL/'source-executions'/ctx.corpus_id/'source-result.json'
            elif changed=='conversion':path=conversion/'both-port-conversion.json'
            elif changed=='model-receipt':path=conversion/'lfm25.json'
            else:path=conversion.parent/'tranche-0000-attempt-0001/both-port-conversion.json'
            data=json.loads(path.read_text()) if path.exists() else {}
            write_json(path,{**data,'revision':1})
        if len(polls)==4:raise StopPolling
    monkeypatch.setattr(worker.time,'sleep',sleep)
    with pytest.raises(StopPolling):worker.run(once=False,poll_seconds=60)
    # A changed receipt with its old parent binding is invalid and must keep
    # retrying, rather than becoming a cached success after the first failure.
    assert len(verified)==(3 if changed=='model-receipt' else 2)
