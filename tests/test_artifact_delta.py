import hashlib,json,os,zlib
from pathlib import Path
import pytest
from scripts.coreweave.artifact_delta import build,apply


@pytest.mark.parametrize('length',[113,1024*1024+19,2*1024*1024+33])
def test_exact_reconstruction_across_block_boundaries_and_different_lengths(tmp_path,length):
    base=tmp_path/'base';base.write_bytes(os.urandom(1024*1024+19))
    raw=(base.read_bytes()+b'extra'*300000)[:length];raw=raw[:17]+b'changed'+raw[24:]
    target=tmp_path/'target';target.write_bytes(raw)
    receipt=build(base,target,tmp_path/'delta');out=tmp_path/'restored'
    result=apply(base,receipt['path'],out,expected_sha256=hashlib.sha256(raw).hexdigest(),expected_bytes=len(raw))
    assert out.read_bytes()==raw and result['sha256']==receipt['target']['sha256']


@pytest.mark.parametrize('fault',['base','payload','truncated','binding','trailing','oversize'])
def test_rejects_corruption_without_publishing_or_leaving_pending_files(tmp_path,fault):
    base=tmp_path/'base';base.write_bytes(b'a'*10000);target=tmp_path/'target';target.write_bytes(b'a'*9999+b'b')
    receipt=build(base,target,tmp_path/'delta');meta=Path(receipt['path']);payload=meta.parent/'delta.zlib'
    expected=receipt['target']['sha256'];length=10000
    if fault=='base':base.write_bytes(b'x'*10000)
    elif fault=='payload':payload.write_bytes(bytes([payload.read_bytes()[0]^1])+payload.read_bytes()[1:])
    elif fault=='truncated':payload.write_bytes(payload.read_bytes()[:-2])
    elif fault=='binding':expected='0'*64
    elif fault=='trailing':payload.write_bytes(payload.read_bytes()+b'extra')
    else:payload.write_bytes(zlib.compress(b'\0'*10001))
    if fault in ('truncated','trailing','oversize'):
        value=json.loads(meta.read_text());value['delta']['bytes']=payload.stat().st_size;value['delta']['sha256']=hashlib.sha256(payload.read_bytes()).hexdigest();meta.write_text(json.dumps(value))
    out=tmp_path/'restored'
    with pytest.raises((ValueError,zlib.error)):
        apply(base,meta,out,expected_sha256=expected,expected_bytes=length)
    assert not out.exists() and not list(tmp_path.glob('.reconstruct-*'))


def test_receiver_never_replaces_an_existing_artifact(tmp_path):
    base=tmp_path/'base';base.write_bytes(b'base');target=tmp_path/'target';target.write_bytes(b'next')
    receipt=build(base,target,tmp_path/'delta')
    with pytest.raises(ValueError,match='replace'):
        apply(base,receipt['path'],target,expected_sha256=receipt['target']['sha256'],expected_bytes=4)
    assert target.read_bytes()==b'next'
