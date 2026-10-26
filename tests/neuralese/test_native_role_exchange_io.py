import hashlib
import os
from pathlib import Path
import re
import sys
import pytest
from natlang_neuralese.common.hashing import canonical_json_sha256_hex

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/coreweave'))
import native_role_exchange_io as transport


class Operation:
    def __init__(self,callback):self.callback=callback
    def result(self,timeout=None):return self.callback()


class FakeSandbox:
    def __init__(self):self.files={}
    def write_file_streaming(self,path,source,**kwargs):
        return Operation(lambda:self.files.__setitem__(path,b''.join(source)))
    def read_file_streaming(self,path,**kwargs):
        class Reader:
            def __init__(self,data):self.data=data
            def __iter__(self):return iter([self.data[i:i+17]for i in range(0,len(self.data),17)])
            def close(self):pass
        return Reader(self.files[path])


class FakeSDK:
    def __init__(self,sandbox):self.instance=sandbox
    def remote(self,sid,code,timeout=120):
        pair=re.search(r"os\.replace\((['\"])(.*?)\1,\s*(['\"])(.*?)\3\)",code)
        if pair:
            source,target=pair.group(2),pair.group(4)
            self.instance.files[target]=self.instance.files.pop(source)
        return 'true'
    def sandbox(self,sid):return self.instance
    def mirror_many(self,*args,**kwargs):return True
    def read(self,*args,**kwargs):return b'cloud metadata'
    def read_optional(self,*args,**kwargs):return None
    def write_atomic(self,*args,**kwargs):return None
    def dependency_closure(self,*args,**kwargs):return {'assets':[],'files':[]}
    def verify_assets(self,*args,**kwargs):return None


def make_io(tmp_path,monkeypatch):
    runtime=tmp_path/'runtime';(runtime/'training/neuralese').mkdir(parents=True)
    root=tmp_path/'local';root.mkdir()
    provider={'sandboxes':{'mellum':{'runtime':str(runtime)}},'receipts':str(tmp_path/'receipts'),
        'ferry_command':['unused'],'local_source_host':{'source_host_id':'dgx-local',
            'runtime':str(runtime),'python':sys.executable,'read_roots':[str(root)],'write_roots':[str(root)]}}
    io=transport.NativeRoleExchangeIO(provider);fake=FakeSandbox();io.sdk=FakeSDK(fake)
    monkeypatch.setattr(transport,'checked_binding',lambda value:{'path':value['path'],
        'sha256':value['sha256'],'bytes':value['bytes']})
    return io,root,fake


def test_local_source_host_exec_and_metadata_roots_are_exact(tmp_path,monkeypatch):
    io,root,_=make_io(tmp_path,monkeypatch)
    assert io.remote('dgx-local','import json;print(json.dumps("local-ok"))')=='local-ok'
    path=root/'proof.json';io.write_atomic('dgx-local',str(path),b'{"proof":true}')
    assert io.read('dgx-local',str(path))==b'{"proof":true}'
    with pytest.raises(ValueError,match='escaped'):
        io.read('dgx-local',str(tmp_path/'outside.json'))
    linked=root/'escape';linked.symlink_to(tmp_path/'outside',target_is_directory=True)
    with pytest.raises(ValueError,match='symlinks'):
        io.read('dgx-local',str(linked/'proof.json'))


def test_local_to_cloud_stream_is_hash_checked_and_atomic(tmp_path,monkeypatch):
    io,root,fake=make_io(tmp_path,monkeypatch)
    monkeypatch.setattr(io,'_cloud_target_state',lambda sid,item:'missing')
    source=root/'bundle.pt';data=bytes(range(251))*101;source.write_bytes(data)
    item={'path':str(source),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
    target='/work/native-production/bundle.pt'
    io._stream_local_to_cloud('dgx-local','mellum',item,target_path=target)
    assert fake.files[target]==data
    source.write_bytes(data+b'changed')
    with pytest.raises(ValueError,match='differ'):
        io._stream_local_to_cloud('dgx-local','mellum',item)


def test_large_local_stage_pauses_for_dynamic_independent_receipt_then_resumes(tmp_path,monkeypatch):
    io,root,fake=make_io(tmp_path,monkeypatch)
    inventory=root/'control'/'private-export-inventory.json'
    inventory.parent.mkdir()
    io.local['private_export_role']='lfm'
    io.local['private_export_inventory']=str(inventory)
    monkeypatch.setattr(io,'_cloud_target_state',lambda sid,item:'missing')
    source=root/'large-source.bin';data=bytes(range(251))*(33*1024*1024//251+1);source.write_bytes(data)
    item={'path':str(source),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
    entry={'source_host_id':'dgx-local','target_host_id':'mellum',
        'target':'/work/native-production/large-source.bin','source':item}
    assert io.stage_prepared_files([entry],source_host_id='dgx-local') is False
    assert not fake.files
    receipt={'private':True,'role':'lfm','source':str(source),'identity':{'sha256':item['sha256'],'bytes':item['bytes']},
        'independent_verification':{'sha256':item['sha256'],'bytes':item['bytes']},
        'run':'fixture/natlang','name':'large-source'}
    inventory.write_text(__import__('json').dumps({'schema':'natlang.native-private-export-inventory/1',
        'role':'lfm','entries':[{'source_path':str(source),'identity':receipt['identity'],'receipt':receipt}]}))
    copied=io.stage_prepared_files([entry],source_host_id='dgx-local')
    assert copied and fake.files[entry['target']]==data


def test_large_local_stage_rejects_stale_dynamic_receipt(tmp_path,monkeypatch):
    io,root,fake=make_io(tmp_path,monkeypatch)
    inventory=root/'inventory.json';io.local['private_export_role']='lfm'
    io.local['private_export_inventory']=str(inventory)
    source=root/'large.bin';data=bytes(range(251))*(33*1024*1024//251+1);source.write_bytes(data)
    item={'path':str(source),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
    wrong={'private':True,'role':'lfm','source':str(source),'identity':{'sha256':item['sha256'],'bytes':item['bytes']},
        'independent_verification':{'sha256':'0'*64,'bytes':item['bytes']},
        'run':'fixture/natlang','name':'large'}
    inventory.write_text(__import__('json').dumps({'schema':'natlang.native-private-export-inventory/1',
        'role':'lfm','entries':[{'source_path':str(source),'identity':wrong['identity'],'receipt':wrong}]}))
    with pytest.raises(ValueError,match='independent'):
        io._require_private_export(item)
    assert not fake.files


def test_prepared_files_stage_to_explicit_cloud_target_and_verify_existing(tmp_path,monkeypatch):
    io,root,fake=make_io(tmp_path,monkeypatch)
    monkeypatch.setattr(io,'_cloud_target_state',lambda sid,item:'missing')
    source=root/'local-plan.json';data=b'{"cycle":1}';source.write_bytes(data)
    target='/work/native-production/semantic/local-plan.json'
    entry={'source_host_id':'dgx-local','target_host_id':'mellum','target':target,
        'source':{'path':str(source),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}}
    staged=io.stage_prepared_files([entry],source_host_id='dgx-local')
    assert fake.files[target]==data
    assert staged==[{'source_host_id':'dgx-local','target_host_id':'mellum','target':target,
        'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}]
    with pytest.raises(ValueError,match='source host'):
        io.stage_prepared_files([dict(entry,source_host_id='other')],source_host_id='dgx-local')


def test_prepared_files_copy_local_targets_atomically(tmp_path,monkeypatch):
    io,root,_=make_io(tmp_path,monkeypatch)
    source=root/'plan-source.json';target=root/'prepared'/'plan.json';data=b'local exact plan'
    source.write_bytes(data)
    entry={'source_host_id':'dgx-local','target_host_id':'dgx-local','target':str(target),
        'source':{'path':str(source),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}}
    io.stage_prepared_files([entry],source_host_id='dgx-local')
    assert target.read_bytes()==data


def test_cloud_preparation_never_overwrites_a_different_existing_target(tmp_path,monkeypatch):
    io,root,fake=make_io(tmp_path,monkeypatch)
    monkeypatch.setattr(io,'_cloud_target_state',lambda sid,item:'different')
    source=root/'source.json';data=b'new';source.write_bytes(data)
    entry={'source_host_id':'dgx-local','target_host_id':'mellum',
        'target':'/work/native-production/existing.json',
        'source':{'path':str(source),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}}
    with pytest.raises(ValueError,match='different bytes'):
        io.stage_prepared_files([entry],source_host_id='dgx-local')
    assert not fake.files


def test_exchange_stages_prepared_inventory_once_under_exact_receipt(tmp_path):
    from native_role_exchange import stage_preparation_files
    class StageIO:
        local={'source_host_id':'dgx-local'}
        def __init__(self):self.receipts={};self.stages=0
        def read_optional(self,sid,path):return self.receipts.get((sid,path))
        def stage_prepared_files(self,files,*,source_host_id):
            self.stages+=1
            return [{'sha256':item['source']['sha256'],'target':item['target']}for item in files]
        def write_atomic(self,sid,path,data):self.receipts[(sid,path)]=data
    io=StageIO()
    files=[{'source_host_id':'dgx-local','target_host_id':'cloud','target':'/work/native-production/a.json',
        'source':{'path':'/inputs/a.json','sha256':'a'*64,'bytes':1}}]
    staged={'schema':'natlang.native-role-preparation-files/1','source_host_id':'dgx-local',
        'files':files,'receipt_path':'/work/native-production/semantic/staged.json'}
    staged['inventory_sha256']=canonical_json_sha256_hex({'schema':staged['schema'],
        'source_host_id':staged['source_host_id'],'files':files})
    config={'hosts':{'lfm':{'source_host_id':'dgx-local'},'mellum':{'source_host_id':'cloud'}},
        'local_cycle_returns':{'consumer':'mellum'},'preparation_stage_files':staged}
    first=stage_preparation_files(config,io)
    second=stage_preparation_files(config,io)
    assert first==second and io.stages==1
    io.receipts[('cloud',staged['receipt_path'])]=b'{"schema":"wrong"}'
    with pytest.raises(ValueError,match='another source inventory'):
        stage_preparation_files(config,io)


def test_exchange_keeps_large_preparation_pending_until_dynamic_private_receipt(tmp_path):
    from native_role_exchange import stage_preparation_files
    class PendingIO:
        local={'source_host_id':'dgx-local'}
        def __init__(self):self.receipts={};self.stages=0
        def read_optional(self,sid,path):return self.receipts.get((sid,path))
        def stage_prepared_files(self,files,*,source_host_id):self.stages+=1;return False
        def write_atomic(self,*args):raise AssertionError('pending private export must not publish readiness')
    io=PendingIO();files=[{'source_host_id':'dgx-local','target_host_id':'cloud',
        'target':'/work/native-production/big-shared-object','source':{
            'path':'/data/shared-object','sha256':'a'*64,'bytes':40*1024**2}}]
    staged={'schema':'natlang.native-role-preparation-files/1','source_host_id':'dgx-local',
        'files':files,'receipt_path':'/work/native-production/semantic/staged.json'}
    staged['inventory_sha256']=canonical_json_sha256_hex({'schema':staged['schema'],
        'source_host_id':staged['source_host_id'],'files':files})
    config={'hosts':{'lfm':{'source_host_id':'dgx-local'},'mellum':{'source_host_id':'cloud'}},
        'local_cycle_returns':{'consumer':'mellum'},'preparation_stage_files':staged}
    assert stage_preparation_files(config,io)=={
        'status':'prepared-files-private-export-pending','receipt_path':staged['receipt_path']}
    assert io.stages==1 and io.receipts=={}


def test_cloud_to_local_stream_uses_verified_durable_download(tmp_path,monkeypatch):
    io,root,fake=make_io(tmp_path,monkeypatch)
    remote='/work/native-production/shared/bundle.pt';data=os.urandom(131_071);fake.files[remote]=data
    item={'path':remote,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
    # The remote path maps into a temporary local test root only for this call.
    target=root/'shared/bundle.pt'
    monkeypatch.setattr(io,'_local_path',lambda path,write=False:target)
    io._stream_cloud_to_local('mellum','dgx-local',item)
    assert target.read_bytes()==data
    fake.files[remote]=data[:-1]
    target.unlink()
    with pytest.raises(ValueError):io._stream_cloud_to_local('mellum','dgx-local',item)
    assert not target.exists()


def test_local_dependency_closure_and_asset_verification_use_declared_runtime(tmp_path,monkeypatch):
    io,root,_=make_io(tmp_path,monkeypatch)
    calls=[]
    monkeypatch.setattr(io,'remote',lambda sid,code,**kwargs:(calls.append((sid,code,kwargs)) or
        ('{"assets":[]}\n' if 'dependency_closure(' in code else '')))
    assert io.dependency_closure('dgx-local',{'semantic_config':{'path':'/inputs/curriculum.json'}})=={'assets':[]}
    io.verify_assets('dgx-local',[])
    assert len(calls)==2 and all(sid=='dgx-local' for sid,_,_ in calls)
    assert all(str(root) in code for _,code,_ in calls)
    assert 'dependency_closure(' in calls[0][1] and 'verify_assets(' in calls[1][1]
