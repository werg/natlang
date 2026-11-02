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
