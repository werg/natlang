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


def test_closure_copy_is_bounded_ordered_and_propagates_failure(tmp_path, monkeypatch):
    import threading, time
    from scripts.neuralese_data import qualify_local_checkpoint_recovery as q
    lock=threading.Lock(); state={'active':0,'peak':0}
    def copy(source,target,digest):
        with lock:
            state['active']+=1;state['peak']=max(state['peak'],state['active'])
        try:
            time.sleep(.01)
            if source=='bad':raise ValueError('invalid content')
            return [digest]
        finally:
            with lock:state['active']-=1
    monkeypatch.setattr(q,'verified_recovery_blob',copy)
    rows=[{'path':str(i),'sha256':str(i)} for i in range(12)]
    result=q.copy_closure_blobs(rows,tmp_path)
    assert list(result)==[(x['path'],x['sha256']) for x in rows]
    assert 1 < state['peak'] <= 4
    with pytest.raises(ValueError,match='invalid content'):
        q.copy_closure_blobs(rows+[{'path':'bad','sha256':'bad'}],tmp_path)


def test_closure_copy_refills_while_first_copy_is_slow(tmp_path, monkeypatch):
    import threading
    from scripts.neuralese_data import qualify_local_checkpoint_recovery as q
    progressed=threading.Event()
    def copy(source,target,digest):
        if source=='0':
            assert progressed.wait(2), 'later work did not advance while first copy waited'
        if source=='9':progressed.set()
        return [digest]
    monkeypatch.setattr(q,'verified_recovery_blob',copy)
    rows=[{'path':str(i),'sha256':str(i)} for i in range(12)]
    assert len(q.copy_closure_blobs(rows,tmp_path))==12


def test_historical_program_routes_use_archived_bytes_and_fail_on_conflict():
    from scripts.neuralese_data.qualify_local_checkpoint_recovery import add_historical_program_routes
    import pytest
    row={'original_path':'/old/program.py','archive':{'path':'/archive/exact.py','sha256':'a'*64},'receipt':{'path':'/receipt','sha256':'b'*64}}
    paths={'/archive/exact.py':'/restored/exact.py'}
    add_historical_program_routes(paths,[row]);assert paths['/old/program.py']=='/restored/exact.py'
    with pytest.raises(ValueError,match='conflicts'):
        add_historical_program_routes({'/old/program.py':'/other','/archive/exact.py':'/restored/exact.py'},[row])
    with pytest.raises(ValueError,match='absent'):
        add_historical_program_routes({},[row])
