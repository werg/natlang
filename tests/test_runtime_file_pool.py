import concurrent.futures
import os
import shutil
import stat
from pathlib import Path

import pytest

from scripts import runtime_file_pool as pool


def snapshot(root,content=b'unchanged input'):
    root.mkdir()
    (root/'script.js').write_bytes(content)
    (root/'script.js').chmod(0o755)
    (root/'link.js').symlink_to('script.js')
    return root


def test_snapshots_share_immutable_bytes_and_keep_paths_and_symlinks(tmp_path):
    a=snapshot(tmp_path/'a');b=snapshot(tmp_path/'b')
    first=pool.seal_runtime_tree(a);second=pool.seal_runtime_tree(b)
    assert first['new_blob_bytes']==second['reused_bytes']==len(b'unchanged input')
    assert os.path.samefile(a/'script.js',b/'script.js')
    assert (a/'link.js').readlink()==Path('script.js')
    assert stat.S_IMODE((a/'script.js').stat().st_mode)==0o555
    # Replacing a path for a new version never changes an already sealed peer.
    (a/'script.js').unlink();(a/'script.js').write_text('new version')
    pool.seal_runtime_tree(a)
    assert (b/'script.js').read_bytes()==b'unchanged input'
    assert not os.path.samefile(a/'script.js',b/'script.js')


def test_concurrent_publication_and_reference_based_pruning(tmp_path):
    a=snapshot(tmp_path/'a');b=snapshot(tmp_path/'b')
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(pool.seal_runtime_tree,[a,b]))
    store=tmp_path/'.native-runtime-file-pool'
    assert os.path.samefile(a/'script.js',b/'script.js')
    shutil.rmtree(a)
    assert pool.prune_unreferenced(store)==0
    assert (b/'script.js').read_bytes()==b'unchanged input'
    shutil.rmtree(b)
    assert pool.prune_unreferenced(store)==len(b'unchanged input')


def test_corrupt_pool_fails_before_replacing_new_snapshot(tmp_path):
    a=snapshot(tmp_path/'a');pool.seal_runtime_tree(a)
    blob=next((tmp_path/'.native-runtime-file-pool').glob('[0-9a-f][0-9a-f]/*'))
    blob.chmod(0o755);blob.write_bytes(b'corrupt input');blob.chmod(0o555)
    b=snapshot(tmp_path/'b')
    with pytest.raises(ValueError,match='corrupt'):
        pool.seal_runtime_tree(b)
    assert (b/'script.js').read_bytes()==b'unchanged input'


def test_source_mutation_is_detected_without_overwriting_it(tmp_path,monkeypatch):
    a=snapshot(tmp_path/'a');real=pool._hash
    def changed(path):
        value=real(path)
        if Path(path)==a/'script.js':Path(path).write_bytes(b'concurrent edit')
        return value
    monkeypatch.setattr(pool,'_hash',changed)
    with pytest.raises(ValueError,match='changed during hashing'):
        pool.seal_runtime_tree(a)
    assert (a/'script.js').read_bytes()==b'concurrent edit'


def test_temporary_cleanup_checks_actual_writer_identity(tmp_path):
    store=tmp_path/'pool';store.mkdir()
    boot,pid,start=pool._writer_identity()
    live=store/f'.copy.{boot}.{pid}.{start}.live';live.write_bytes(b'live')
    dead=store/f'.copy.{boot}.999999999.0.dead';dead.write_bytes(b'dead')
    assert pool.prune_unreferenced(store)==4
    assert live.read_bytes()==b'live' and not dead.exists()


def test_pool_inside_snapshot_is_rejected(tmp_path):
    a=snapshot(tmp_path/'a')
    with pytest.raises(ValueError,match='outside snapshot'):
        pool.seal_runtime_tree(a,a/'pool')
