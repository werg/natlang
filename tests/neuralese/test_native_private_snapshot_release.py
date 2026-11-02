"""Private reconstruction lineage is checked, never inferred from success flags."""
import hashlib,json
from pathlib import Path
import pytest
from natlang_neuralese.model.native_bundle import _fresh_file_binding
from natlang_neuralese.model.native_snapshot_publication import _verify_private_reconstruction


def fixture(tmp_path,target=None):
    def write(name,value):
        p=tmp_path/name;p.write_text(json.dumps(value,sort_keys=True));return p
    def ref(path,role):
        v=_fresh_file_binding(path);return {'original_path':str(path),'sha256':v['sha256'],'bytes':v['bytes'],'role':role}
    base={'sha256':'a'*64,'bytes':12};base0={'sha256':'b'*64,'bytes':13}
    bpayload={'sha256':'c'*64,'bytes':14};bdelta={'sha256':'d'*64,'bytes':15}
    target=target or {'sha256':'e'*64,'bytes':99};payload={'sha256':'f'*64,'bytes':10}
    parent=ref(write('parent.json',{'schema':'natlang.exact-native-execution-recovery-capsule/1','run':'private/run',
        'base':base0,'targets':{'parent':{'target':base,'payload':bpayload,'receipt':bdelta}}}),
        'retained-private-BF95-base-capsule')
    decoders=[ref(write(f'decoder-{i}.json',{'version':i}),'exact-reconstruction-decoder') for i in range(2)]
    delta=ref(write('delta.json',{'schema':'natlang.artifact-zip-delta/2','base':base,'target':target,
                               'payload':payload}),'exact-full-state-delta-receipt')
    payloadref={'original_path':str(tmp_path/'payload.zlib'),'role':'exact-full-state-delta-payload',**payload}
    presence=[{'name':f"blobs/{v['sha256']}.p0000",'bytes':v['bytes'],'http_status':206}
              for v in (base0,bpayload,bdelta)]
    stream=ref(write('stream.json',{'sha256':target['sha256'],'bytes':target['bytes'],'path':None,
        'receipt_sha256':delta['sha256'],'payload_get_sha256':payload['sha256'],'payload_get_bytes':payload['bytes'],
        'verified_unix':123.,'base_private_presence':presence}),'independent-private-full-target-stream-readback')
    local=ref(write('local.json',{'schema':'natlang.local-exact-checkpoint-reconstruction/1',
        'all_full_state_bytes_verified':True,'target':target,'base':base,'receipt':delta,'payload':payload,
        'decoders':decoders,'retained_independent_private_base_capsule':parent}),'local-full-state-reconstruction-capsule')
    files=[local,delta,payloadref,stream,parent,*decoders]
    capsule={'schema':'natlang.private-exact-earned-SFT-recovery/1','run':'private/run','target':target,'base':base,
             'complete_target_stream_verified':True,'training_restore_qualified':False,'files':files}
    cp=write('capsule.json',capsule);cref=_fresh_file_binding(cp)
    def object_entry(v):return {'phase':'independently-verified','bytes':v['bytes'],'role':v.get('role'),
        'source':v.get('original_path'),'owner_get':{'sha256':v['sha256'],'bytes':v['bytes'],
            'mechanism':'complete private GET on DGX; no sandbox','unix':123.},
        'parts':[{'index':0,'offset':0,'name':f"blobs/{v['sha256']}.p0000",'bytes':v['bytes'],'sha256':v['sha256']}]}
    objects={v['sha256']:object_entry(v) for v in files};objects[cref['sha256']]=object_entry(cref)
    state={'schema':'natlang.private-native-admission-evidence/1','run':'private/run','complete':True,'target':target,
        'capsule':{'sha256':cref['sha256'],'bytes':cref['bytes'],'role':'earned-SFT-exact-full-state-private-capsule'},'objects':objects}
    ep=write('export.json',state)
    return capsule,cref,ep,target,stream,payload


def test_private_reconstruction_binds_stream_base_decoder_and_publisher(tmp_path):
    capsule,cref,export,target,stream,payload=fixture(tmp_path)
    _verify_private_reconstruction(capsule,cref,_fresh_file_binding(export),target)
    state=json.loads(export.read_text());state['objects'][payload['sha256']]['owner_get']['sha256']='0'*64
    export.write_text(json.dumps(state))
    with pytest.raises(ValueError,match='independent GET'):
        _verify_private_reconstruction(capsule,cref,_fresh_file_binding(export),target)
    state['objects'][payload['sha256']]['owner_get']['sha256']=payload['sha256'];export.write_text(json.dumps(state))
    Path(stream['original_path']).write_text('{}')
    with pytest.raises(ValueError,match='local evidence changed'):
        _verify_private_reconstruction(capsule,cref,_fresh_file_binding(export),target)


def test_private_reconstruction_rejects_foreign_full_source(tmp_path):
    capsule,cref,export,target,*_=fixture(tmp_path)
    with pytest.raises(ValueError,match='capsule or publisher'):
        _verify_private_reconstruction(capsule,cref,_fresh_file_binding(export),{**target,'sha256':'0'*64})


def test_private_reconstruction_releases_pin_only_after_delivery_and_all_leases(tmp_path):
    from natlang_neuralese.model.native_snapshot_publication import release_pending
    root=tmp_path/'owner';root.mkdir();pin=root/'source.pt';pin.write_bytes(b'full-state-fixture')
    binding=_fresh_file_binding(pin);target={k:binding[k] for k in ('sha256','bytes')}
    capsule,cref,export,_,_,_=fixture(tmp_path,target)
    lease={'role':'actor-view-source:fixture','consumer_id':'owner','identity':target}
    st=pin.stat()
    record={'status':'ready','full_source_pinned':True,'source':binding,'pin':str(pin),'lease':lease,
            'source_identity':{'device':st.st_dev,'inode':st.st_ino,'size':st.st_size,'mtime_ns':st.st_mtime_ns},
            'view':{'view_receipt':{'sha256':'9'*64}}}
    (root/'actor-view-pending.json').write_text(json.dumps(record))
    leases=root/'standalone-retention-consumer-leases.json'
    leases.write_text(json.dumps({'schema':'natlang.retention-consumer-leases/1','leases':[lease,{**lease,'consumer_id':'other'}]}))
    transport=tmp_path/'transport.json';transport.write_text(json.dumps({'schema':'natlang.native-actor-transport/1',
        'source':target,'view_receipt':{'sha256':'9'*64},'view':{'sha256':'8'*64}}))
    tr=_fresh_file_binding(transport)
    delivery=tmp_path/'delivery.json';delivery.write_text(json.dumps({'schema':'natlang.actor-transfer-readback/1',
        'transport_sha256':tr['sha256'],'verified':{'source_checkpoint_sha256':target['sha256'],'sha256':'8'*64}}))
    args={'recovery_ref':cref,'recovery_export_ref':_fresh_file_binding(export),'transport_ref':tr,'delivery_ref':_fresh_file_binding(delivery)}
    with pytest.raises(ValueError,match='additional consumer'):
        release_pending(root,**args)
    assert pin.exists()
    leases.write_text(json.dumps({'schema':'natlang.retention-consumer-leases/1','leases':[lease]}))
    result=release_pending(root,**args)
    assert result['source_sha256']==target['sha256'] and not pin.exists()
    assert result['recovery_export']==_fresh_file_binding(export)


def test_private_reconstruction_rejects_mutated_decoder_and_base_presence(tmp_path):
    capsule,cref,export,target,stream,_=fixture(tmp_path)
    decoder=next(item for item in capsule['files'] if item['role']=='exact-reconstruction-decoder')
    path=Path(decoder['original_path']);original=path.read_bytes();path.write_bytes(original+b' ')
    with pytest.raises(ValueError,match='decoder changed'):
        _verify_private_reconstruction(capsule,cref,_fresh_file_binding(export),target)
    path.write_bytes(original)
    data=json.loads(Path(stream['original_path']).read_text());data['base_private_presence'][0]['http_status']=404
    Path(stream['original_path']).write_text(json.dumps(data))
    with pytest.raises(ValueError,match='local evidence changed'):
        _verify_private_reconstruction(capsule,cref,_fresh_file_binding(export),target)
