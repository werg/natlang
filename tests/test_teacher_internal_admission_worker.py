import json
from pathlib import Path

import pytest

from scripts.coreweave import teacher_internal_admission_worker as worker
from scripts.coreweave import teacher_internal_conversion as conversion_module
from scripts.coreweave.teacher_tranche_admission import file_binding, preceding_tranches


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
    source.write_text('{"source":"verified"}\n')
    source_ref = file_binding(source)
    source_proof_path = tmp_path / 'source-proof.json'
    source_proof = {'status': 'passed', 'training_admission': False,
                    'source': source_ref, 'actor': {'sha256': 'a' * 64},
                    'execution_identity': {'sha256': 'b' * 64},
                    'current_protection': {'sha256': 'd' * 64}}
    source_proof_ref = write_json(source_proof_path, source_proof)
    settings_ref = write_json(tmp_path / 'settings.json', {
        'teacher_namespace': 'qwen36-verified-positive-native-graphs-20261022-recovery7853',
        'internal_continuation': {'schema': 'natlang.teacher-internal-source-continuation/1',
                                  'training_admission': False},
    })
    predecessor_id = 'registered-native-predecessor'
    package = tmp_path / 'runs' / predecessor_id
    manifest_files = []
    model_graphs = {}
    for model in ('lfm25', 'mellum21'):
        represented = package / model / 'train/native-graphs.jsonl.zst'
        represented.parent.mkdir(parents=True, exist_ok=True)
        represented.write_bytes((model + ' graph').encode())
        represented_ref = file_binding(represented)
        model_graphs[model] = represented_ref
        manifest_files.append({'path': f'{model}/train/native-graphs.jsonl.zst',
                               'sha256': represented_ref['sha256'], 'bytes': represented_ref['bytes']})
    write_json(tmp_path / 'training/neuralese_corpora.json', {
        'corpora': [{'id': predecessor_id, 'path': f'runs/{predecessor_id}',
                     'training_admission': True}]})
    write_json(tmp_path / 'training/corpus-manifests' / f'{predecessor_id}.json', {
        'id': predecessor_id, 'path': f'runs/{predecessor_id}', 'files': manifest_files})
    models = []
    for model in ('lfm25', 'mellum21'):
        represented_ref = model_graphs[model]
        outputs = {}
        for split in ('train', 'test'):
            output_path = tmp_path / model / split / 'native-graphs.jsonl.zst'
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b'empty native graph stream')
            outputs[split] = {**file_binding(output_path), 'rows': 0}
        receipt_path = tmp_path / model / 'receipt.json'
        receipt = {'status': 'passed', 'model_key': model,
                   'source': {'sha256': source_ref['sha256']},
                   'tokenizer': {'expanded_fingerprint': (
                       'be949ec48f44e17b2b17e1131aeb1dab4262175ba20f9783ecd8e59c27a58105'
                       if model == 'lfm25' else
                       'fbf22d8773d09be802cb9977895067db36cdfba747fcb24b02ddbf1312d3778f')},
                   'explicit_history_reasoning_policy': 'keep' if model == 'lfm25' else None,
                   'failed_assistant_history_policy': 'mask-runtime-failures',
                   'outputs': outputs,
                   'deduplication': {'inputs': [{'path': represented_ref['path'],
                                                 'sha256': represented_ref['sha256']}],
                                     'emitted_rows': 0, 'omitted_rows': 1,
                                     'omissions': [{'content_sha256': 'a' * 64,
                                         'represented_by': {'path': represented_ref['path'],
                                                            'id': 'sample-native-row'}}]}}
        models.append({'model': model, 'receipt': write_json(receipt_path, receipt)})
    conversion = tmp_path / 'conversion'
    write_json(conversion / 'both-port-conversion.json', {
        'schema': 'natlang.teacher-internal-both-port-conversion/1',
        'batch': 21, 'tranche': 1,
        'source': source_ref, 'source_proof': source_proof_ref,
        'settings': settings_ref,
        'protection_manifest': {'sha256': 'd' * 64},
        'training_registration': False, 'training_exposure': False,
        'admitted_predecessors': [predecessor_id],
        'same_actor_source_boundary_inputs': [],
        'models': models,
    })
    return conversion


def test_zero_rows_require_both_exact_dedup_receipts_and_bound_inputs(tmp_path, monkeypatch):
    conversion = zero_conversion(tmp_path)
    monkeypatch.setattr(worker, 'ROOT', tmp_path)
    monkeypatch.setattr(worker, '_registered_ids', lambda: set())
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


def test_internal_predecessor_graphs_require_same_actor_source_and_history_contract(tmp_path):
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
        with pytest.raises(ValueError):
            conversion_module._verified_internal_predecessors(
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
