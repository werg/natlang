import hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
import pytest
from scripts.coreweave import send_actor_transport as sender


def binding(data):return hashlib.sha256(data).hexdigest()


def transport(root, data):
    root.mkdir();view_sha=binding(data);view=root/(view_sha+'.pt');view.write_bytes(data)
    ref=root/'view.json';ref.write_text(json.dumps({'synthetic':True,'view':view_sha}));ref_sha=binding(ref.read_bytes())
    renamed=root/(ref_sha+'.json');ref.rename(renamed)
    value={'view':{'path':view.name,'bytes':len(data),'sha256':view_sha},
        'source':{'sha256':binding(b'synthetic training owner'+data[:64])},'view_receipt':{'sha256':ref_sha}}
    p=root/'transport.json';p.write_text(json.dumps(value));return p,binding(p.read_bytes()),view


@pytest.fixture
def local_peer(monkeypatch):
    # Exercise the real receiver programs, codec, disk publication and readback;
    # replace only SSH/SCP, with a separate local peer filesystem.
    def remote(host, code, *args):
        result=subprocess.run([sys.executable,'-I','-',*map(str,args)],input=code,text=True,capture_output=True)
        if result.returncode:raise RuntimeError(result.stderr)
        return json.loads(result.stdout)
    def copy(host, source, target, **kwargs):shutil.copyfile(source,target)
    monkeypatch.setattr(sender,'remote',remote);monkeypatch.setattr(sender,'copy',copy)


def test_full_then_delta_then_verified_reuse_preserve_exact_bytes_and_metadata(tmp_path,local_peer):
    old=os.urandom(2*1024*1024);new=old[:8192]+b'changed!'+old[8200:]
    p1,sha1,view1=transport(tmp_path/'sender one',old);peer=tmp_path/'peer with spaces'
    first=sender.send(p1,sha1,'test-peer',str(peer));assert first['mode']=='full'
    p2,sha2,view2=transport(tmp_path/'sender two',new)
    second=sender.send(p2,sha2,'test-peer',str(peer));assert second['mode']=='xor-zlib'
    assert second['payload_bytes']<len(new)*.05
    reconstructed=Path(second['verified']['receiver_view']);assert reconstructed.read_bytes()==new
    metadata=reconstructed.parent/p2.name;before=metadata.stat()
    third=sender.send(p2,sha2,'test-peer',str(peer));assert third['mode']=='already-verified' and third['payload_bytes']==0
    assert metadata.stat().st_ctime_ns==before.st_ctime_ns


def test_corrupt_peer_base_is_rejected_without_fabricating_target(tmp_path,local_peer):
    old=os.urandom(1024*1024);new=old[:-1]+b'x'
    p,sha,_=transport(tmp_path/'first',old);peer=tmp_path/'peer';first=sender.send(p,sha,'peer',str(peer))
    Path(first['verified']['receiver_view']).write_bytes(b'corrupt')
    p,sha,_=transport(tmp_path/'next',new)
    with pytest.raises(RuntimeError,match='base differs'):sender.send(p,sha,'peer',str(peer))
    target=peer/json.loads(p.read_text())['source']['sha256']/json.loads(p.read_text())['view']['path']
    assert not target.exists()


def test_receiver_uses_full_copy_when_previous_sender_cache_is_missing(tmp_path,local_peer):
    p,sha,view=transport(tmp_path/'first',b'old data'*100);peer=tmp_path/'peer';sender.send(p,sha,'peer',str(peer));view.unlink()
    p,sha,_=transport(tmp_path/'next',b'new data'*100)
    assert sender.send(p,sha,'peer',str(peer))['mode']=='full'


def zip_bytes(metadata,tensor):
    import io,zipfile
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_STORED) as archive:
        archive.writestr('archive/data.pkl',metadata)
        archive.writestr('archive/data/0',tensor)
    return stream.getvalue()


def test_shifted_zip_payload_is_preferred_and_retry_reuses_cache(tmp_path,local_peer,monkeypatch):
    data=os.urandom(2*1024*1024);old=zip_bytes(b'old',data);new=zip_bytes(b'new metadata with shifted offset',data)
    p,sha,_=transport(tmp_path/'first-zip',old);peer=tmp_path/'peer';sender.send(p,sha,'peer',str(peer))
    p,sha,_=transport(tmp_path/'next-zip',new)
    def fail_xor(*args,**kwargs):raise AssertionError('should not build XOR for small ZIP delta')
    monkeypatch.setattr(sender,'build',fail_xor)
    result=sender.send(p,sha,'peer',str(peer));assert result['mode']=='zip-payload-zlib'
    assert result['payload_bytes']<len(new)*.05
    assert Path(result['verified']['receiver_view']).read_bytes()==new
    assert sender.send(p,sha,'peer',str(peer))['mode']=='already-verified'


def test_corrupt_cached_zip_is_rejected_before_copy(tmp_path,local_peer):
    data=os.urandom(1024*1024);old=zip_bytes(b'old',data);new=zip_bytes(b'new longer',data)
    p,sha,_=transport(tmp_path/'first-zip',old);peer=tmp_path/'peer';sender.send(p,sha,'peer',str(peer))
    p,sha,view=transport(tmp_path/'next-zip',new)
    old_sha=binding(old);cache=p.parent/'.transfer-deltas'/(old_sha+'-'+binding(new))/'zip'
    sender.artifact_zip_delta.build(json.loads((peer/'.natlang-actor-transfer.json').read_text())['sender_view'],view,cache)
    payload=cache/'payload.zlib';payload.write_bytes(payload.read_bytes()+b'bad')
    with pytest.raises(ValueError,match='cached ZIP delta'):sender.send(p,sha,'peer',str(peer))


def test_zip_without_shared_payload_falls_back_to_full(tmp_path,local_peer):
    old=zip_bytes(b'a',os.urandom(1024*1024));new=zip_bytes(b'longer',os.urandom(1024*1024))
    p,sha,_=transport(tmp_path/'first-zip',old);peer=tmp_path/'peer';sender.send(p,sha,'peer',str(peer))
    p,sha,_=transport(tmp_path/'next-zip',new)
    result=sender.send(p,sha,'peer',str(peer));assert result['mode']=='full'
    assert Path(result['verified']['receiver_view']).read_bytes()==new


def test_interrupted_zip_send_retries_from_verified_cache(tmp_path,local_peer,monkeypatch):
    data=os.urandom(1024*1024);old=zip_bytes(b'old',data);new=zip_bytes(b'changed-header',data)
    p,sha,_=transport(tmp_path/'old',old);peer=tmp_path/'peer';sender.send(p,sha,'peer',str(peer))
    p,sha,_=transport(tmp_path/'new',new)
    real_copy=sender.copy;failed=[]
    def interrupted(host,source,target,**kwargs):
        if Path(source).name=='payload.zlib' and not failed:
            failed.append(True);raise TimeoutError('synthetic interrupted transfer')
        return real_copy(host,source,target,**kwargs)
    monkeypatch.setattr(sender,'copy',interrupted)
    with pytest.raises(TimeoutError):sender.send(p,sha,'peer',str(peer))
    def no_rebuild(*args,**kwargs):raise AssertionError('verified ZIP cache should be reused')
    monkeypatch.setattr(sender.artifact_zip_delta,'build',no_rebuild)
    result=sender.send(p,sha,'peer',str(peer))
    assert result['mode']=='zip-payload-zlib' and Path(result['verified']['receiver_view']).read_bytes()==new


def test_zip_peer_base_tampering_rejects_reconstruction(tmp_path,local_peer):
    data=os.urandom(1024*1024);old=zip_bytes(b'old',data);new=zip_bytes(b'changed-header',data)
    p,sha,_=transport(tmp_path/'old',old);peer=tmp_path/'peer';first=sender.send(p,sha,'peer',str(peer))
    peer_base=Path(first['verified']['receiver_view']);raw=bytearray(peer_base.read_bytes());raw[-1]^=1;peer_base.write_bytes(raw)
    p,sha,_=transport(tmp_path/'new',new)
    with pytest.raises(RuntimeError,match='base binding differs'):sender.send(p,sha,'peer',str(peer))
    target=peer/json.loads(p.read_text())['source']['sha256']/json.loads(p.read_text())['view']['path']
    assert not target.exists()
