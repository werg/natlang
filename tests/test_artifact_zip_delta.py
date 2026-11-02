import hashlib, importlib.util, json, zipfile
from pathlib import Path
import pytest

path=Path(__file__).resolve().parents[1]/'scripts/coreweave/artifact_zip_delta.py'
spec=importlib.util.spec_from_file_location('artifact_zip_delta',path)
import sys
sys.path.insert(0,str(path.parent))
delta=importlib.util.module_from_spec(spec);spec.loader.exec_module(delta)

def make(path,metadata,tensor):
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_STORED) as z:
        z.writestr('archive/data.pkl',metadata)
        z.writestr('archive/data/0',tensor)
        z.writestr('archive/data/1',b'changed-'+metadata)

def test_shifted_entries_exact_reconstruction_and_tampering(tmp_path):
    base=tmp_path/'base';target=tmp_path/'target';tensor=bytes(range(256))*300
    make(base,b'old',tensor);make(target,b'new metadata of different length',tensor)
    result=delta.build(base,target,tmp_path/'patch')
    assert result['copied_bytes']>=len(tensor)
    args=dict(expected_sha256=delta.digest_file(target),expected_bytes=target.stat().st_size,expected_manifest_sha256=result['sha256'])
    assert delta.verify(base,result['path'],**args)['sha256']==args['expected_sha256']
    out=tmp_path/'out';delta.verify(base,result['path'],out=out,**args)
    assert out.read_bytes()==target.read_bytes()
    with pytest.raises(ValueError):delta.verify(base,result['path'],out=out,**args)
    base.write_bytes(base.read_bytes()+b'!')
    with pytest.raises(ValueError,match='base binding'):delta.verify(base,result['path'],**args)


def test_payload_receipt_and_copy_range_corruption_rejected(tmp_path):
    base=tmp_path/'base';target=tmp_path/'target';make(base,b'a',b'x'*5000);make(target,b'bb',b'x'*5000)
    result=delta.build(base,target,tmp_path/'patch');receipt=Path(result['path']);payload=receipt.parent/'payload.zlib'
    args=dict(expected_sha256=delta.digest_file(target),expected_bytes=target.stat().st_size,expected_manifest_sha256=result['sha256'])
    original=payload.read_bytes();payload.write_bytes(original+b'!')
    with pytest.raises(ValueError,match='payload binding'):delta.verify(base,receipt,**args)
    payload.write_bytes(original)
    value=json.loads(receipt.read_text());next(op for op in value['operations'] if op['kind']=='copy')['offset']=10**9
    receipt.write_text(json.dumps(value));args['expected_manifest_sha256']=delta.digest_file(receipt)
    with pytest.raises(ValueError,match='base range'):delta.verify(base,receipt,**args)
    with pytest.raises(ValueError,match='receipt hash'):delta.verify(base,receipt,**{**args,'expected_manifest_sha256':'0'*64})


def _fixture_bytes(seed, size):
    import random
    return random.Random(seed).randbytes(size)


def test_changed_payloads_choose_smallest_and_preserve_unequal_lengths(tmp_path, monkeypatch):
    monkeypatch.setattr(delta, 'CHUNK', 1024)
    old = _fixture_bytes(11, 18000)
    changed = bytearray(old); changed[103:109] = b'ABCDEF'
    unrelated = _fixture_bytes(12, 18000)
    base, target = tmp_path/'base', tmp_path/'target'
    def archive(path, metadata, entries):
        with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_STORED) as z:
            z.writestr('archive/data.pkl', metadata)
            for name, payload in entries.items(): z.writestr(name, payload)
    archive(base, b'old header', {'same': old, 'changed': old, 'longer': old,
            'shorter': old, 'unrelated': old})
    archive(target, b'much longer pickle shifts all offsets', {'same': old,
            'changed': bytes(changed), 'longer': old+b'extended', 'shorter': old[:-17],
            'unrelated': unrelated})
    result=delta.build(base,target,tmp_path/'patch',min_free_bytes=0)
    receipt=Path(result['path']); manifest=json.loads(receipt.read_text())
    assert manifest['schema']==delta.SCHEMA and result['copied_bytes'] >= len(old)
    assert result['xored_bytes'] >= len(changed)+len(old+b'extended')+len(old[:-17])
    ops=manifest['operations']; assert sum(op['kind']=='xor' for op in ops)>=3
    for op in ops:
        if op['kind']=='xor':
            with open(base,'rb') as src,open(target,'rb') as dst:
                target_range=next((off,size) for off,size,_ in delta.ranges(target).values()
                    if size==op['output_bytes'] and delta.range_hash(dst,off,size)==op['sha256'])
                raw=b''.join(delta.stream_range(dst,*target_range))
                assert op['bytes'] < len(__import__('zlib').compress(raw,1))
    args=dict(expected_sha256=delta.digest_file(target),expected_bytes=target.stat().st_size,
              expected_manifest_sha256=result['sha256'],min_free_bytes=0)
    # Stream verification without allocating any output or reconstruction temp file.
    monkeypatch.setattr(delta.tempfile,'mkstemp',lambda *a,**kw: pytest.fail('unexpected output materialization'))
    assert delta.verify(base,receipt,**args)['sha256']==args['expected_sha256']
    monkeypatch.undo()
    reconstructed=tmp_path/'restored'
    delta.verify(base,receipt,out=reconstructed,**args)
    assert reconstructed.read_bytes()==target.read_bytes()


def test_schema1_receipt_remains_recoverable(tmp_path):
    base,target=tmp_path/'base',tmp_path/'target'
    for path,metadata in ((base,_fixture_bytes(5,100)),(target,_fixture_bytes(6,117))):
        with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_STORED) as z:
            z.writestr('archive/data.pkl',metadata)
            z.writestr('archive/data/0',b'unchanged'*1000)
    result=delta.build(base,target,tmp_path/'patch',min_free_bytes=0)
    receipt=Path(result['path']); value=json.loads(receipt.read_text())
    # The v1 dialect cannot contain XOR operations.
    assert not any(op['kind']=='xor' for op in value['operations'])
    value['schema']=delta.SCHEMA_V1;value.pop('xored_bytes')
    receipt.write_text(json.dumps(value))
    result=delta.verify(base,receipt,expected_sha256=delta.digest_file(target),
        expected_bytes=target.stat().st_size,expected_manifest_sha256=delta.digest_file(receipt),min_free_bytes=0)
    assert result['sha256']==delta.digest_file(target)


@pytest.mark.parametrize('mutation', ['base_offset','base_size','base_hash','output_hash',
                                      'payload_offset','payload_size','output_size','v1_xor'])
def test_xor_receipt_rejects_corruption_and_out_of_bounds(tmp_path, mutation):
    old=_fixture_bytes(91,20000);new=bytearray(old);new[200:204]=b'edit'
    base,target=tmp_path/'base',tmp_path/'target'
    make(base,b'a',old);make(target,b'longer pickle',bytes(new))
    result=delta.build(base,target,tmp_path/'patch',min_free_bytes=0)
    receipt=Path(result['path']);value=json.loads(receipt.read_text())
    op=next(op for op in value['operations'] if op['kind']=='xor')
    if mutation=='base_offset':op['base_offset']=base.stat().st_size
    elif mutation=='base_size':op['base_bytes']=-1
    elif mutation=='base_hash':op['base_sha256']='0'*64
    elif mutation=='output_hash':op['sha256']='0'*64
    elif mutation=='payload_offset':op['offset']+=1
    elif mutation=='payload_size':op['bytes']+=10**9
    elif mutation=='output_size':op['output_bytes']+=1
    else:value['schema']=delta.SCHEMA_V1
    receipt.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        delta.verify(base,receipt,expected_sha256=delta.digest_file(target),
            expected_bytes=target.stat().st_size,expected_manifest_sha256=delta.digest_file(receipt),min_free_bytes=0)


def test_xor_payload_corruption_rejected_even_with_rebound_payload_hash(tmp_path):
    old=_fixture_bytes(92,24000);new=bytearray(old);new[2500]=new[2500]^1
    base,target=tmp_path/'base',tmp_path/'target'
    make(base,b'a',old);make(target,b'longer metadata',bytes(new))
    result=delta.build(base,target,tmp_path/'patch',min_free_bytes=0)
    receipt=Path(result['path']);value=json.loads(receipt.read_text());payload=receipt.parent/value['payload']['path']
    op=next(op for op in value['operations'] if op['kind']=='xor')
    data=bytearray(payload.read_bytes());data[op['offset']]^=1;payload.write_bytes(data)
    value['payload']['sha256']=delta.digest_file(payload);receipt.write_text(json.dumps(value))
    import zlib
    with pytest.raises((ValueError,zlib.error)):
        delta.verify(base,receipt,expected_sha256=delta.digest_file(target),
            expected_bytes=target.stat().st_size,expected_manifest_sha256=delta.digest_file(receipt),min_free_bytes=0)


def test_floor_is_enforced_before_payload_writes_and_no_artifact_is_published(tmp_path, monkeypatch):
    old=_fixture_bytes(14,8000);new=_fixture_bytes(15,8000)
    base,target=tmp_path/'base',tmp_path/'target'
    make(base,b'old',old);make(target,b'new metadata',new)
    seen=[]
    def floor(path, minimum):
        seen.append(minimum)
        if minimum>12345:raise ValueError('synthetic floor reached')
    monkeypatch.setattr(delta,'require_free',floor)
    with pytest.raises(ValueError,match='floor'):
        delta.build(base,target,tmp_path/'patch',min_free_bytes=12345)
    assert seen[0]==12345 and any(value>12345 for value in seen)
    assert list((tmp_path/'patch').iterdir())==[]


def test_deflated_entries_and_changed_zip_methods_reconstruct_exactly(tmp_path):
    old=_fixture_bytes(301,14000);new=bytearray(old);new[400]^=1
    base,target=tmp_path/'base',tmp_path/'target'
    with zipfile.ZipFile(base,'w',compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('metadata',b'old');z.writestr('tensor',old)
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_STORED) as z:
        z.writestr('metadata',b'new and shifted');z.writestr('tensor',new)
    result=delta.build(base,target,tmp_path/'patch',min_free_bytes=0)
    output=tmp_path/'out'
    delta.verify(base,result['path'],expected_sha256=delta.digest_file(target),
        expected_bytes=target.stat().st_size,expected_manifest_sha256=result['sha256'],out=output,min_free_bytes=0)
    assert output.read_bytes()==target.read_bytes()


def test_xor_stream_trailing_bytes_rejected_after_payload_rebinding(tmp_path):
    old=_fixture_bytes(432,20000);new=bytearray(old);new[40]^=1
    base,target=tmp_path/'base',tmp_path/'target'
    make(base,b'old',old);make(target,b'new shifted',new)
    result=delta.build(base,target,tmp_path/'patch',min_free_bytes=0)
    receipt=Path(result['path']);value=json.loads(receipt.read_text());payload=receipt.parent/value['payload']['path']
    op=next(op for op in value['operations'] if op['kind']=='xor')
    insertion=op['offset']+op['bytes'];data=payload.read_bytes();payload.write_bytes(data[:insertion]+b'!'+data[insertion:])
    op['bytes']+=1
    for other in value['operations']:
        if other is not op and other['kind'] in ('xor','inline') and other['offset']>=insertion:
            other['offset']+=1
    value['payload']['bytes']+=1;value['payload']['sha256']=delta.digest_file(payload)
    receipt.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='compressed data'):
        delta.verify(base,receipt,expected_sha256=delta.digest_file(target),expected_bytes=target.stat().st_size,
                     expected_manifest_sha256=delta.digest_file(receipt),min_free_bytes=0)
