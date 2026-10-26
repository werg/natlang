import hashlib
import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/neuralese_data/prepare_source_correction_continuation.py'
SPEC = importlib.util.spec_from_file_location('source_correction_preparer', SCRIPT)
PREPARER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARER)


def _entry(path):
    payload = Path(path).read_bytes()
    return {'source_binding': {'models': {'lfm25': {
        'model_id': 'lfm2:model', 'tokenizer_fingerprint': 'a' * 64,
        'train': {'path': str(path), 'bytes': len(payload),
                  'sha256': hashlib.sha256(payload).hexdigest()}}}}}


def test_registered_package_role_binds_exact_model_split_bytes(tmp_path):
    graph = tmp_path / 'native-graphs.jsonl.zst'
    graph.write_bytes(b'authenticated package leaf')
    PREPARER._validate_registered_role(_entry(graph), 'lfm25', 'train', graph,
        model_id='lfm2:model', fingerprint='a' * 64)
    with pytest.raises(ValueError, match='registered model binding'):
        PREPARER._validate_registered_role(_entry(graph), 'lfm25', 'train', graph,
            model_id='mellum:model', fingerprint='a' * 64)
    with pytest.raises(ValueError, match='registered model binding'):
        PREPARER._validate_registered_role(_entry(graph), 'lfm25', 'train', graph,
            model_id='lfm2:model', fingerprint='b' * 64)
    damaged = _entry(graph)
    damaged['source_binding']['models']['lfm25']['train']['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='registered model binding'):
        PREPARER._validate_registered_role(damaged, 'lfm25', 'train', graph,
            model_id='lfm2:model', fingerprint='a' * 64)
    with pytest.raises(ValueError, match='no exact mellum21 source binding'):
        PREPARER._validate_registered_role(_entry(graph), 'mellum21', 'train', graph,
            model_id='lfm2:model', fingerprint='a' * 64)


def test_registered_package_may_use_an_authenticated_external_storage_symlink(tmp_path):
    external = tmp_path / 'external' / 'native-graphs.jsonl.zst'
    external.parent.mkdir()
    external.write_bytes(b'authenticated package leaf')
    package = tmp_path / 'package'
    package.mkdir()
    graph = package / 'native-graphs.jsonl.zst'
    graph.symlink_to(external)
    PREPARER._validate_registered_role(_entry(graph), 'lfm25', 'train', graph,
        model_id='lfm2:model', fingerprint='a' * 64)
