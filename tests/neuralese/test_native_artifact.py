import pytest
import torch

from natlang_neuralese.model.native_artifact import (NativeTensorArtifact, decode_native_tensor_artifact,
    encode_native_tensor_artifact, native_tensor_artifact_id)
from natlang_neuralese.serve.http import block_ids
from natlang_neuralese.serve.store import TensorStore


def artifact():
    return NativeTensorArtifact('parameter-gradient', {
        'checkpoint_sha256': '1' * 64, 'tokenizer_fingerprint': '2' * 64,
        'embedding_weight_sha256': '3' * 64, 'architecture': 'lfm2'},
        {'adapter.down.weight': torch.arange(6, dtype=torch.float32).reshape(2, 3),
         'input_rows': torch.ones((4, 3), dtype=torch.bfloat16)},
        source_artifact_id='nzs1_example', provenance={'objective': 'fixture', 'order': 1})


def test_native_tensor_artifact_round_trip_binds_names_identity_and_provenance():
    value = artifact()
    body = encode_native_tensor_artifact(value)
    restored, artifact_id = decode_native_tensor_artifact(body)
    assert artifact_id == native_tensor_artifact_id(value)
    assert artifact_id.startswith('nnt1_')
    assert encode_native_tensor_artifact(restored) == body
    assert restored.kind == 'parameter-gradient'
    assert restored.source_artifact_id == value.source_artifact_id
    assert list(sorted(restored.tensors)) == ['adapter.down.weight', 'input_rows']
    torch.testing.assert_close(restored.tensors['input_rows'], value.tensors['input_rows'])
    assert artifact_id in block_ids({'gradient': artifact_id})


def test_native_tensor_artifact_rejects_content_corruption_and_identity_mismatch():
    body = bytearray(encode_native_tensor_artifact(artifact()))
    body[-1] ^= 1
    with pytest.raises(ValueError, match='ID does not match'):
        decode_native_tensor_artifact(bytes(body))
    value = artifact()
    value.native_identity['architecture'] = ''
    with pytest.raises(ValueError, match='architecture'):
        encode_native_tensor_artifact(value)


def test_native_tensor_artifact_uses_shared_store_holds_and_survives_restart(tmp_path):
    store = TensorStore(tmp_path, resident_bytes=1)
    record = store.put_native_tensor_artifact(artifact(), 'owner-a')
    store.pin(record.id, 'owner-b')
    store.close()
    store = TensorStore(tmp_path, resident_bytes=1)
    restored = store.get_native_tensor_artifact(record.id)
    assert restored is not None and restored.id == record.id
    torch.testing.assert_close(restored.artifact.tensors['adapter.down.weight'],
                               artifact().tensors['adapter.down.weight'])
    assert store.holders(record.id) == {'owner-a', 'owner-b'}
    store.collect(set(), 'owner-a')
    assert store.get_native_tensor_artifact(record.id) is not None
    store.unpin(record.id, 'owner-b')
    assert store.collect(set(), 'owner-b') == [record.id]
    store.close()
