import io,json,tarfile
import pytest
from scripts.neuralese_data.archive_local_recovery_runtime import seal
from scripts.neuralese_data.qualify_local_checkpoint_recovery import extract_archive,_mapped_path


def test_extracts_and_validates_runtime_archive(tmp_path):
    root=tmp_path/'source';root.mkdir();(root/'module.py').write_text('value=4\n')
    archive=tmp_path/'runtime.tar';seal(root,archive,tmp_path/'receipt.json')
    manifest=extract_archive(archive,tmp_path/'restore');assert manifest['schema']=='natlang.local-recovery-runtime-archive/1'
    assert (tmp_path/'restore/runtime/module.py').read_text()=='value=4\n'


def test_restore_rejects_escaping_archive_and_nonabsolute_source(tmp_path):
    archive=tmp_path/'bad.tar'
    with tarfile.open(archive,'w') as tar:
        info=tarfile.TarInfo('runtime/../../escape');data=b'bad';info.size=len(data);tar.addfile(info,io.BytesIO(data))
    with pytest.raises((ValueError,tarfile.FilterError)):extract_archive(archive,tmp_path/'restore')
    assert not (tmp_path/'escape').exists()
    with pytest.raises(ValueError,match='absolute'):_mapped_path(tmp_path,'relative/source')
