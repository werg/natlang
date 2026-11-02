import hashlib
import json
import pytest
from natlang_neuralese.export.native_hf import export_loaded, SCHEMA


def test_cached_export_must_bind_nonlegacy_execution_profile(tmp_path):
    weights=tmp_path/'model.safetensors';weights.write_bytes(b'weights')
    manifest={'schema':SCHEMA,'checkpoint_sha256':'a'*64,'dtype':'bfloat16',
        'weights_sha256':hashlib.sha256(b'weights').hexdigest(), 'native_shared_execution':'ffn-stable-v1'}
    (tmp_path/'natlang-export.json').write_text(json.dumps(manifest))
    def check():
        return export_loaded(None,None,None,{'native_shared_execution':'ffn-stable-v1'},
                             checkpoint='unused',checkpoint_sha256='a'*64,out_dir=tmp_path)
    (tmp_path/'config.json').write_text(json.dumps({'natlang_shared_execution':'source-v1'}))
    with pytest.raises(ValueError,match='config does not bind'):check()
    (tmp_path/'config.json').write_text(json.dumps({'natlang_shared_execution':'ffn-stable-v1'}))
    assert check()==manifest
    manifest['native_shared_execution']='source-v1'
    (tmp_path/'natlang-export.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='different native shared'):check()
