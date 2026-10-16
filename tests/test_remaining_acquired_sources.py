"""Archive durability and derived two-actor provenance, without invented state."""
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from neuralese_data import acquired_agents as acquired,agents,acquired_archives as archives
from neuralese_data.acquired_transformed_coop import qwen_comm4_acquired
from neuralese_data.common import Reject


def pinned(tmp_path,monkeypatch,ref,files):
    root=tmp_path/ref;root.mkdir();entries=[]
    for name,data in files.items():
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        entries.append({'path':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
    (root/'acquisition.json').write_text(json.dumps({'status':'acquired','repo':'source/'+ref,'revision':'pinned',
        'card':{},'files':entries}))
    monkeypatch.setattr(agents,'EXPANSION_ROOT',tmp_path);monkeypatch.setattr(agents,'_bench_ids',lambda:set())
    monkeypatch.setenv('NATLANG_HASH_CACHE',str(tmp_path/'hash-cache.jsonl'))
    return root


def tar_bytes(files):
    buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w:gz') as archive:
        for name,value in files:
            raw=value if isinstance(value,bytes) else json.dumps(value).encode()
            member=tarfile.TarInfo(name);member.size=len(raw);archive.addfile(member,io.BytesIO(raw))
    return buffer.getvalue()


def test_archive_complete_stream_target_and_neighbors_are_exact(tmp_path,monkeypatch):
    parent='run/team/repo_task/17/f1_f2/';target='Original final response'
    stream='Original task and full observed tool output <|im_end|>\ntokens used\n99\n'+target+'\n'
    data=tar_bytes([(parent+'agent2_traj.json',{'repo':'repo_task','task_id':17,'agent_id':'agent1',
            'status':'Submitted','messages':[{'role':'assistant','content':target}]}),
        (parent+'agent1_stream.log',stream.encode()),(parent+'eval.json',{'repo':'repo_task','task_id':17,'both_passed':True})])
    root=pinned(tmp_path,monkeypatch,'D12',{'run.tar.gz':data});monkeypatch.setattr(acquired,'ACQUIRED_FILES',frozenset({'run.tar.gz'}))
    values=list(archives.cooper_archive_acquired());good=[(r,t) for r,t in values if not isinstance(t,Reject)]
    assert len(good)==1 and len(values)==3
    rid,t=good[0];assert rid.startswith('run.tar.gz:0:') and t.messages[-1]['content']==target
    assert t.messages[1]['content']==stream[:-len(target+'\n')]
    note=t.lineage['notes']['original_cooperation_context'];assert note['neighbor_bindings']['eval.json']['sha256']
    assert not note['archive_context']['original_context_roles_reconstructed']
    assert acquired.supported_acquired_file('run.tar.gz')
    rows=list(acquired.iter_acquired_file_rows(root/'run.tar.gz'));assert [n for n,_ in rows]==[0,1,2]


def test_archive_traversal_and_duplicate_names_fail_closed(tmp_path):
    for names in ([('../escape.json',{})],[('same.json',{}),('same.json',{})]):
        path=tmp_path/'archive.tar.gz';path.write_bytes(tar_bytes(names))
        with pytest.raises(Reject):list(archives.iter_archive_rows(path))


def transformed_fixture(tmp_path,monkeypatch):
    parent='repo_task/17/f1_f2/';path=parent+'readable_merged.json'
    entries=[{'index':0,'agent':'agent1','actor':'CODER-agent1','type':'thinking','content':'Original reasoning'},
        {'index':1,'agent':'agent1','actor':'CODER-agent1','type':'action','content':'cat source.py'},
        {'index':2,'agent':'agent1','actor':'CODER-agent1','type':'action','content':'(result: exit 0)'},
        {'index':3,'agent':'agent1','actor':'CODER-agent1','type':'observation','content':'ENTIRE FILE'},
        {'index':4,'agent':'agent2','actor':'CODER-agent2','type':'action','content':'Original peer reply'}]
    buffer=io.StringIO();writer=csv.DictWriter(buffer,fieldnames=['repo','task_id','features','log_dir','model','passed','metadata']);writer.writeheader()
    writer.writerow({'repo':'repo_task','task_id':17,'features':'1_2','log_dir':path,'model':'source-teacher','passed':'',
        'metadata':json.dumps({'agent_statuses':{'agent1':'Submitted','agent2':'LimitsExceeded'}})})
    root=pinned(tmp_path,monkeypatch,'D15',{'index.csv':buffer.getvalue().encode(),path:json.dumps({'entries':entries,'run_info':{'model':'actual-model'}}).encode(),
        parent+'transform_metadata.json':json.dumps({'model':'source-transformer','rounds':[{'original':'all audit facts'}]}).encode()})
    monkeypatch.setattr(acquired,'ACQUIRED_FILES',frozenset({path}))
    return root,path,entries


def test_transformed_source_keeps_entire_order_and_both_actor_views(tmp_path,monkeypatch):
    root,path,entries=transformed_fixture(tmp_path,monkeypatch);values=list(qwen_comm4_acquired())
    assert len(values)==2 and all(not isinstance(t,Reject) for _,t in values)
    first,second=[t for _,t in values];assert first.outcome['label']=='teacher' and second.outcome['label']=='failed'
    assert first.messages[1]['reasoning']=='Original reasoning' and first.messages[2]['content']=='cat source.py'
    assert first.messages[3]['role']=='tool' and '(result: exit 0)' in first.messages[3]['content']
    assert all(t.messages[-1]['content'] for t in (first,second))
    for t in (first,second):
        note=t.lineage['notes']['original_transformed_cooperation'];assert [v['original'] for v in note['entries']]==entries
        assert not note['original_four_agent_contexts_available'] and not note['source_test_result_available']
    audit=root/Path(path).parent/'transform_metadata.json';audit.write_text('tamper')
    assert isinstance(next(qwen_comm4_acquired())[1],Reject)


def test_transformed_unknown_actor_or_entry_cannot_drop_context(tmp_path,monkeypatch):
    root,path,entries=transformed_fixture(tmp_path,monkeypatch)
    # Exercise unsupported input through the adapter while keeping its bound
    # neighboring index/audit documents authentic.
    original=acquired.acquired_rows
    def rows(ref,limit=None):
        for receipt,rid,row in original(ref,limit):
            row['entries'][0]['actor']='Unrecorded-third-party';yield receipt,rid,row
    monkeypatch.setattr(acquired,'acquired_rows',rows)
    assert isinstance(next(qwen_comm4_acquired())[1],Reject)


def test_malformed_archive_member_is_accounted_without_losing_good_members(tmp_path):
    path=tmp_path/'archive.tar.gz'
    path.write_bytes(tar_bytes([('broken.json',b''),('good.json',{'messages':['original']}),('raw.bin',b'\xff')]))
    rows=list(archives.iter_archive_rows(path))
    assert len(rows)==3 and rows[0][1]['kind']=='invalid-json-source-artifact'
    assert rows[1][1]['value']=={'messages':['original']} and rows[2][1]['kind']=='binary-source-artifact'
    assert rows[0][1]['member']['sha256']==hashlib.sha256(b'').hexdigest()


def test_bound_transformed_index_cannot_relabel_another_task(tmp_path,monkeypatch):
    root,path,_=transformed_fixture(tmp_path,monkeypatch)
    index=root/'index.csv';index.write_text(index.read_text().replace('repo_task,17,','other_repo,17,'))
    receipt=json.loads((root/'acquisition.json').read_text());entry=next(e for e in receipt['files'] if e['path']=='index.csv')
    data=index.read_bytes();entry.update(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    (root/'acquisition.json').write_text(json.dumps(receipt))
    value=next(qwen_comm4_acquired())[1]
    assert isinstance(value,Reject) and 'another original benchmark task' in str(value)


def test_archive_incomplete_stream_cannot_be_labeled_complete_context(tmp_path,monkeypatch):
    parent='run/team/repo/17/f1_f2/'
    data=tar_bytes([(parent+'agent1_traj.json',{'repo':'repo','task_id':17,'agent_id':'agent1',
        'messages':[{'role':'assistant','content':'Independently stored final response'}]}),
        (parent+'agent1_stream.log',b'Original transcript cut off before final response\n')])
    pinned(tmp_path,monkeypatch,'D12',{'run.tar.gz':data});monkeypatch.setattr(acquired,'ACQUIRED_FILES',frozenset({'run.tar.gz'}))
    values=list(archives.cooper_archive_acquired())
    assert all(isinstance(value,Reject) for _,value in values)
    assert any('does not bind exact original stream suffix' in str(value) for _,value in values)
