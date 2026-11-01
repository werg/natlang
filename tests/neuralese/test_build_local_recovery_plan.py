import os,sys
from scripts.neuralese_data.build_local_recovery_plan import ldd_files,mapped_executables


def test_inventories_real_process_and_executable_dependencies():
    assert os.path.realpath(sys.executable) in mapped_executables(os.getpid())
    libraries=ldd_files(sys.executable)
    assert libraries and all(os.path.isabs(path) and os.path.isfile(path) for path in libraries)


def test_full_checkpoint_requires_only_pinned_config_and_tokenizer(tmp_path):
    import hashlib
    import pytest
    from scripts.neuralese_data.build_local_recovery_plan import checkpoint_base_files
    (tmp_path/'config.json').write_bytes(b'config')
    (tmp_path/'tokenizer.json').write_bytes(b'tokenizer')
    digest=lambda data:hashlib.sha256(data).hexdigest()
    inventory={'config.json':digest(b'config'),'tokenizer.json':digest(b'tokenizer'),
               'model.safetensors':digest(b'pinned weights')}
    state={'model_config':{'base_inventory':inventory},
           'backbone_state':{'model.embed_tokens.weight':object()}}
    assert checkpoint_base_files(tmp_path,state)=={k:v for k,v in inventory.items() if not k.endswith('.safetensors')}
    with pytest.raises(ValueError,match='backbone'):
        checkpoint_base_files(tmp_path,dict(state,backbone_state={}))
    (tmp_path/'tokenizer.json').write_bytes(b'changed')
    with pytest.raises(ValueError,match='differs'):
        checkpoint_base_files(tmp_path,state)
