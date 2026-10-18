"""Independent exporters share durability without uploading the same bytes twice."""
import json
import multiprocessing
from pathlib import Path
import sys
import time

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/coreweave'))
from private_export_guard import export_lease

IDENTITY={'sha256':'a'*64,'bytes':48_000_000_000}


def receipt(identity=IDENTITY):
    return {'private':True,'identity':dict(identity),'independent_verification':dict(identity),
            'run':'werg/natlang/run','name':'mellum/'+identity['sha256']+'/checkpoint.pt','source':'/immutable/first.pt'}


def concurrent_worker(directory, barrier):
    barrier.wait()
    with export_lease(directory,scope='mellum:sandbox',identity=IDENTITY,object_exists=lambda _:True) as lease:
        if lease.receipt is None:
            with (Path(directory)/'uploads').open('a') as stream:stream.write('actual PUT and GET\n')
            time.sleep(.1)
            lease.record(receipt())


def test_independent_processes_export_once(tmp_path):
    ctx=multiprocessing.get_context('spawn');barrier=ctx.Barrier(2)
    workers=[ctx.Process(target=concurrent_worker,args=(str(tmp_path),barrier)) for _ in range(2)]
    for worker in workers:worker.start()
    for worker in workers:
        worker.join(15)
        if worker.is_alive():worker.terminate();worker.join();pytest.fail('export lock did not drain')
        assert worker.exitcode==0
    assert (tmp_path/'uploads').read_text().splitlines()==['actual PUT and GET']


def test_deleted_private_object_requires_new_export_and_keeps_original_provenance(tmp_path):
    with export_lease(tmp_path,scope='owner',identity=IDENTITY,object_exists=lambda _:True) as lease:lease.record(receipt())
    with export_lease(tmp_path,scope='owner',identity=IDENTITY,object_exists=lambda _:False) as lease:assert lease.receipt is None
    with export_lease(tmp_path,scope='owner',identity=IDENTITY,object_exists=lambda _:True) as lease:
        assert lease.receipt['source']=='/immutable/first.pt'
    other={**IDENTITY,'sha256':'b'*64}
    with export_lease(tmp_path,scope='owner',identity=other,object_exists=lambda _:pytest.fail('different content reused cache')) as lease:assert lease.receipt is None


def test_failed_upload_never_publishes_receipt_and_provider_failure_is_not_absence(tmp_path):
    with pytest.raises(RuntimeError):
        with export_lease(tmp_path,scope='owner',identity=IDENTITY,object_exists=lambda _:True):raise RuntimeError('interrupted PUT')
    assert not list(tmp_path.glob('*.json'))
    with export_lease(tmp_path,scope='owner',identity=IDENTITY,object_exists=lambda _:True) as lease:
        bad=receipt();bad['independent_verification']={'sha256':'a'*64}
        with pytest.raises(ValueError):lease.record(bad)
        assert not list(tmp_path.glob('*.json'))
        lease.record(receipt())
    def unavailable(_):raise ConnectionError('provider unavailable')
    with pytest.raises(ConnectionError):
        with export_lease(tmp_path,scope='owner',identity=IDENTITY,object_exists=unavailable):pytest.fail('must not upload again')


def test_equivalent_path_preserves_actual_upload_and_full_get_history():
    from private_export_guard import receipt_for_source
    original=receipt();original['verified_at']=123
    alias=receipt_for_source(original,source='/different/pin.pt',identity=IDENTITY)
    assert original['source']=='/immutable/first.pt'
    assert alias['source']=='/different/pin.pt' and alias['verified_at']==123
    assert alias['reused_verified_export']['upload_source']==original['source']
    assert not alias['reused_verified_export']['new_upload']
    with pytest.raises(ValueError):receipt_for_source(original,source='/different/pin.pt',identity={**IDENTITY,'bytes':1})


def test_cache_bounded_reader_preserves_read_seek_and_readinto(tmp_path, monkeypatch):
    import os
    from private_export_guard import CacheBoundedReader
    path=tmp_path/'source';path.write_bytes(b'0123456789')
    advised=[]
    monkeypatch.setattr(os,'posix_fadvise',lambda fd,start,count,mode:advised.append((start,count,mode)))
    with CacheBoundedReader(path,release_bytes=3)as stream:
        assert stream.read(2)==b'01' and stream.read(2)==b'23'
        assert advised==[(0,4,os.POSIX_FADV_DONTNEED)]
        stream.seek(8)
        assert stream.read(2)==b'89' and stream.read(2)==b''
        assert advised[-1]==(8,2,os.POSIX_FADV_DONTNEED)
        stream.seek(0);buffer=bytearray(10)
        assert stream.readinto(buffer)==10 and bytes(buffer)==path.read_bytes()
        assert stream.tell()==10
    with pytest.raises(ValueError):CacheBoundedReader(path,release_bytes=0)


def test_cache_bounded_reader_is_length_preserving_http_put_body(tmp_path):
    import hashlib
    import http.server
    import threading
    import requests
    from private_export_guard import CacheBoundedReader
    body=bytes(range(256))*8192
    path=tmp_path/'source';path.write_bytes(body)
    received={}
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_PUT(self):
            received['length']=int(self.headers['Content-Length'])
            received['encoding']=self.headers.get('Transfer-Encoding')
            received['sha256']=hashlib.sha256(self.rfile.read(received['length'])).hexdigest()
            self.send_response(200);self.end_headers()
        def log_message(self,*args):pass
    with http.server.HTTPServer(('127.0.0.1',0),Handler)as server:
        server.timeout=5
        worker=threading.Thread(target=server.handle_request);worker.start()
        try:
            with CacheBoundedReader(path,release_bytes=16384)as stream:
                result=requests.put(f'http://127.0.0.1:{server.server_port}/object',data=stream,timeout=5)
            assert result.status_code==200
        finally:worker.join(6)
    assert received=={'length':len(body),'encoding':None,'sha256':hashlib.sha256(body).hexdigest()}
