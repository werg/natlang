"""Only complete byte-identical artifacts may cross lossless-transfer publication."""
import hashlib
import json
from pathlib import Path
import struct

import pytest
import zstandard as zstd

from scripts.coreweave import lossless_file_transfer as codec

assert Path(codec.__file__).resolve()==Path(__file__).resolve().parents[1]/'scripts/coreweave/lossless_file_transfer.py'


def identity(raw):return {'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}


def prepare(tmp_path,target=b'new-data'*4096,basis=b'old-data'*4096):
    a,b,c=tmp_path/'target.pt',tmp_path/'basis.pt',tmp_path/'wire.nldelta'
    a.write_bytes(target);b.write_bytes(basis)
    record=codec.encode(a,b,c,target_identity=identity(target),basis_identity=identity(basis),block_bytes=257)
    return a,b,c,record,dict(container_identity=record['identity'],target_identity=identity(target),basis_identity=identity(basis))


@pytest.mark.parametrize('target,basis',[(b'',b''),(b'',b'old'),(b'new',b''),
    (bytes(range(256))*101,bytes(range(255,-1,-1))*93),(b'x'*9,b'x'*90),(b'x'*90,b'x'*9)],
    ids=['empty','basis-only','target-only','long-target','short-target','long-basis-tail'])
def test_full_stream_roundtrip_across_block_and_file_boundaries(tmp_path,target,basis):
    a,b,c,record,args=prepare(tmp_path,target,basis)
    assert record['identity']==identity(c.read_bytes())
    assert codec.verify(c,b,**args)['exact_reconstruction']
    out=tmp_path/'restored.pt';result=codec.decode(c,b,out,block_bytes=113,**args)
    assert out.read_bytes()==a.read_bytes()==target and result['identity']==identity(target)
    assert b.read_bytes()==basis and out.stat().st_mode&0o222==0
    assert not list(tmp_path.glob('.*pending-*'))


@pytest.mark.parametrize('damage',['basis','basis_size','container','container_size','target_sha',
    'target_size','basis_identity','boolean_bytes','wrong_codec'])
def test_damaged_identity_or_input_never_publishes_target(tmp_path,damage):
    a,b,c,record,args=prepare(tmp_path)
    if damage=='basis':b.write_bytes(b'!'+b.read_bytes()[1:])
    if damage=='basis_size':b.write_bytes(b.read_bytes()[:-1])
    if damage=='container':
        c.chmod(0o600);raw=c.read_bytes();c.write_bytes(raw[:-1]+bytes([raw[-1]^1]))
    if damage=='container_size':c.chmod(0o600);c.write_bytes(c.read_bytes()[:-1])
    if damage=='target_sha':args['target_identity']={**args['target_identity'],'sha256':'0'*64}
    if damage=='target_size':args['target_identity']={**args['target_identity'],'bytes':1}
    if damage=='basis_identity':args['basis_identity']={**args['basis_identity'],'sha256':'0'*64}
    if damage=='boolean_bytes':args['target_identity']={**args['target_identity'],'bytes':True}
    if damage=='wrong_codec':
        raw=c.read_bytes();n=len(codec.MAGIC);length=struct.unpack('>I',raw[n:n+4])[0];header=json.loads(raw[n+4:n+4+length]);header['codec']='other';encoded=json.dumps(header).encode();raw=codec.MAGIC+struct.pack('>I',len(encoded))+encoded+raw[n+4+length:];c.chmod(0o600);c.write_bytes(raw);args['container_identity']=identity(raw)
    out=tmp_path/'restored.pt'
    with pytest.raises((ValueError,zstd.ZstdError)):codec.decode(c,b,out,**args)
    assert not out.exists()and not list(tmp_path.glob('.*pending-*'))
    assert a.exists()and b.exists()and c.exists()


@pytest.mark.parametrize('raw',[b'',codec.MAGIC,codec.MAGIC+b'\x00\x00\x00\x00',
    codec.MAGIC+struct.pack('>I',codec.MAX_HEADER_BYTES+1),codec.MAGIC+struct.pack('>I',8)+b'{}'])
def test_malformed_header_is_rejected_before_model_publication(tmp_path,raw):
    b,c=tmp_path/'basis',tmp_path/'container';b.write_bytes(b'');c.write_bytes(raw);out=tmp_path/'restored.pt'
    with pytest.raises(ValueError):codec.decode(c,b,out,container_identity=identity(raw),
        target_identity=identity(b''),basis_identity=identity(b''))
    assert not out.exists()and not list(tmp_path.glob('.*pending-*'))


def test_decompressed_tail_is_rejected_even_with_a_matching_container_digest(tmp_path):
    _,b,c,_,args=prepare(tmp_path);raw=c.read_bytes()+zstd.ZstdCompressor().compress(b'extra')
    c.chmod(0o600);c.write_bytes(raw);args['container_identity']=identity(raw)
    with pytest.raises(ValueError,match='exceeds'):codec.decode(c,b,tmp_path/'restored.pt',**args)
    assert not(tmp_path/'restored.pt').exists()


def test_full_input_authentication_precedes_encoding_publication(tmp_path):
    a,b=tmp_path/'target',tmp_path/'basis';a.write_bytes(b'actual');b.write_bytes(b'basis');out=tmp_path/'wire'
    with pytest.raises(ValueError,match='content differs'):codec.encode(a,b,out,
        target_identity=identity(b'wrong!'),basis_identity=identity(b'basis'))
    assert not out.exists()and not list(tmp_path.glob('.*pending-*'))


@pytest.mark.parametrize('operation',['encode','decode'])
def test_interrupted_operation_preserves_inputs_and_removes_its_pending_file(tmp_path,operation):
    a,b,c,record,args=prepare(tmp_path);out=tmp_path/'new-output'
    def interrupt(_):raise InterruptedError('controlled interruption')
    with pytest.raises(InterruptedError):
        if operation=='encode':codec.encode(a,b,out,target_identity=args['target_identity'],
            basis_identity=args['basis_identity'],block_bytes=13,progress=interrupt)
        else:codec.decode(c,b,out,block_bytes=13,progress=interrupt,**args)
    assert not out.exists()and not list(tmp_path.glob('.*pending-*'))
    assert identity(a.read_bytes())==args['target_identity']and identity(b.read_bytes())==args['basis_identity']


def test_existing_output_and_concurrent_publication_are_never_overwritten(tmp_path):
    a,b,c,record,args=prepare(tmp_path);out=tmp_path/'existing';out.write_bytes(b'keep')
    with pytest.raises(FileExistsError):codec.decode(c,b,out,**args)
    assert out.read_bytes()==b'keep'
    out.unlink()
    def publish_other(_):
        if not out.exists():out.write_bytes(b'other-writer')
    with pytest.raises(FileExistsError):codec.decode(c,b,out,progress=publish_other,**args)
    assert out.read_bytes()==b'other-writer'and not list(tmp_path.glob('.*pending-*'))


def test_source_changed_during_encoding_cannot_be_published(tmp_path):
    a,b=tmp_path/'target',tmp_path/'basis';a.write_bytes(b'x'*1024);b.write_bytes(b'y'*1024);out=tmp_path/'wire'
    def change(_):
        if a.read_bytes().endswith(b'x'):
            with a.open('r+b')as f:f.seek(-1,2);f.write(b'z')
    with pytest.raises(ValueError):codec.encode(a,b,out,target_identity=identity(b'x'*1024),
        basis_identity=identity(b'y'*1024),block_bytes=7,progress=change)
    assert not out.exists()and not list(tmp_path.glob('.*pending-*'))
