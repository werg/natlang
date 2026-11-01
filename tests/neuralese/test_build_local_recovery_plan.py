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


def test_control_snapshot_survives_source_edit_and_rejects_dirty_capture(tmp_path):
    import subprocess
    from pathlib import Path
    import pytest
    from scripts.neuralese_data.build_local_recovery_plan import snapshot_control
    repo=tmp_path/'repo';repo.mkdir()
    subprocess.run(['git','init','-q',str(repo)],check=True)
    source=repo/'qualify_local_checkpoint_recovery.py';source.write_text('committed\n')
    subprocess.run(['git','-C',str(repo),'add',source.name],check=True)
    subprocess.run(['git','-C',str(repo),'-c','user.name=Test','-c','user.email=test@example.invalid','commit','-qm','initial'],check=True)
    snapshot=snapshot_control(source,tmp_path/'snapshots',repo)
    assert Path(snapshot['path']).read_text()=='committed\n'
    source.write_text('new uncommitted work\n')
    assert Path(snapshot['path']).read_text()=='committed\n'
    with pytest.raises(ValueError,match='differs from its committed source'):
        snapshot_control(source,tmp_path/'snapshots',repo)


def test_historical_program_mapping_binds_archived_bytes_not_mutable_original(tmp_path):
    import hashlib,json
    from pathlib import Path
    import pytest
    from scripts.neuralese_data.build_local_recovery_plan import historical_program_bindings
    original=tmp_path/'original.py';original.write_bytes(b'new mutable code')
    archive=tmp_path/'archived.py';archive.write_bytes(b'old exact code')
    digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    receipt=tmp_path/'receipt.json';receipt.write_text(json.dumps({'schema':'natlang.historical-source-program-preservation/1','programs':[{'original_path':str(original),'path':str(archive),'sha256':digest(archive),'bytes':archive.stat().st_size}]}))
    def add(path,role,expected=None):
        actual=digest(path)
        if expected is not None and actual!=expected:raise ValueError('archive digest differs')
        return {'path':str(Path(path).resolve()),'sha256':actual,'role':role}
    mapping=historical_program_bindings([receipt],add)
    assert mapping[0]['original_path']==str(original) and mapping[0]['archive']['sha256']==digest(archive)
    archive.write_bytes(b'bad alteration')
    with pytest.raises(ValueError,match='digest differs'):historical_program_bindings([receipt],add)
