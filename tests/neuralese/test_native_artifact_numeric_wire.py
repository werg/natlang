"""Existing native IDs survive Python/JS numeric spelling differences."""
import json
from pathlib import Path
import subprocess

import pytest
import torch
from natlang_neuralese.model.native_artifact import NativeTensorArtifact, native_tensor_artifact_id, encode_native_tensor_artifact, decode_native_tensor_artifact
from natlang_neuralese.serve.store import TensorStore


def test_actual_python_js_numeric_metadata_identity_and_immutable_store(tmp_path):
    repo=Path(__file__).resolve().parents[2]
    artifact=NativeTensorArtifact('parameter-gradient',{'checkpoint_sha256':'1'*64,
        'tokenizer_fingerprint':'2'*64,'embedding_weight_sha256':'3'*64,'architecture':'numeric-wire'},
        {'weight':torch.tensor([[1.,2.]])},provenance={'loss':1.0,'small':1e-9,'negative_zero':-0.0,
            'nested':{'stats':[1.0,1e-9,1e20,1e21,0.000001,0.0000001]}})
    original_id=native_tensor_artifact_id(artifact)
    original=tmp_path/'python.safetensors';original.write_bytes(encode_native_tensor_artifact(artifact))
    returned=tmp_path/'js.safetensors'
    result=subprocess.run(['node',str(repo/'ts-host/test/fixtures/native-tensor-metadata-roundtrip.mjs'),
        str(original),str(returned)],check=True,capture_output=True,text=True)
    assert json.loads(result.stdout)['id']==original_id
    restored,ident=decode_native_tensor_artifact(returned.read_bytes())
    assert ident==original_id and restored.provenance==artifact.provenance
    store=TensorStore(tmp_path/'store')
    assert store.put_native_tensor_artifact(restored).id==original_id
    store.close()
    reopened=TensorStore(tmp_path/'store')
    assert native_tensor_artifact_id(reopened.get_native_tensor_artifact(original_id).artifact)==original_id
    reopened.close()
    restored.provenance={**restored.provenance,'loss':2.}
    with pytest.raises(ValueError,match='preserved metadata bytes'):
        native_tensor_artifact_id(restored)
