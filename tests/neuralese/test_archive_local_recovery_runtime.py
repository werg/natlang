import json,tarfile
from pathlib import Path
import pytest
from scripts.neuralese_data.archive_local_recovery_runtime import seal


def test_runtime_archive_readback_and_internal_symlink(tmp_path):
    root=tmp_path/'runtime';(root/'package').mkdir(parents=True);(root/'package'/'module.py').write_text('answer=42\n');(root/'alias').symlink_to('package',target_is_directory=True)
    archive=tmp_path/'archive.tar';receipt=tmp_path/'receipt.json';result=seal(root,archive,receipt)
    assert result['all_members_independently_reread'] and json.loads(receipt.read_text())==result
    with tarfile.open(archive) as tar:
        assert tar.extractfile('runtime/package/module.py').read()==b'answer=42\n'
        assert tar.getmember('runtime/alias').issym()


def test_runtime_archive_refuses_external_symlink(tmp_path):
    root=tmp_path/'runtime';root.mkdir();outside=tmp_path/'outside';outside.write_text('outside');(root/'link').symlink_to(outside)
    with pytest.raises(ValueError,match='escapes'):seal(root,tmp_path/'archive.tar',tmp_path/'receipt.json')
    assert not (tmp_path/'archive.tar').exists()


def test_explicit_system_file_link_is_copied_and_bound(tmp_path):
    root=tmp_path/'system';root.mkdir();outside=tmp_path/'library';outside.write_text('library');(root/'link').symlink_to(outside)
    archive=tmp_path/'archive.tar';seal(root,archive,tmp_path/'receipt.json',allow_external_file_links=True)
    with tarfile.open(archive) as tar:
        assert tar.extractfile('runtime/link').read()==b'library'
        manifest=json.load(tar.extractfile('runtime-manifest.json'));assert manifest['entries'][0]['original_link']==str(outside)
