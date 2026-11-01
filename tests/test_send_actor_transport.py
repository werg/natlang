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
