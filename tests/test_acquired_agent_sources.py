"""New snapshots preserve complete sessions, exact source facts and omissions."""
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from neuralese_data import acquired_agents as acquired, agents
from neuralese_data.common import Reject
from neuralese_data.records import group_key


def fixture(tmp_path, monkeypatch, ref, rows):
    root = tmp_path/ref
    root.mkdir()
    data = b'fixture parquet source'
    (root/'sessions.parquet').write_bytes(data)
    receipt = {'status':'acquired','repo':'fixture/'+ref,'revision':'pinned',
               'card':{'license':'source-license'},
               'files':[{'path':'sessions.parquet','bytes':len(data),
                         'sha256':hashlib.sha256(data).hexdigest()}]}
    (root/'acquisition.json').write_text(json.dumps(receipt))
    monkeypatch.setenv('NATLANG_HASH_CACHE',str(tmp_path/'hashes.jsonl'))
    monkeypatch.setattr(agents,'EXPANSION_ROOT',tmp_path)
    monkeypatch.setattr(agents,'_parquet_groups',lambda path:iter(enumerate(rows)))
    monkeypatch.setattr(agents,'_bench_ids',lambda:{'protected'})
    return root


def messages():
    return [{'role':'user','content':'Repair this task.'},
            {'role':'assistant','content':None,'reasoning_content':'Inspect first.',
             'tool_calls':[{'id':'call','function':{'name':'bash','arguments':'{"command":"ls"}'}}]},
            {'role':'tool','content':'Observed file listing.','tool_call_id':'call'},
            {'role':'assistant','content':'Completed patch.'}]


@pytest.mark.parametrize('ref,adapter,key',[
    ('D22',acquired.open_swe_acquired,'messages'),
    ('D25',acquired.swe_rebench_acquired,'trajectory')])
def test_coding_preserves_calls_reasoning_protection_and_failure_without_oracle(tmp_path,monkeypatch,ref,adapter,key):
    row={key:messages(),'trajectory_id':'actor-1','instance_id':'protected','repo':'org/repo',
         'resolved':0,'metadata':{'teacher_model':{'name':'actual teacher'}},
         'tools':[json.dumps({'type':'function','function':{'name':'bash'}})]}
    fixture(tmp_path,monkeypatch,ref,[row])
    rid,t=next(adapter())
    assert t.split=='test' and t.outcome['label']=='failed' and t.outcome['checked'] is None
    assert t.outcome['details']['executed_locally'] is False
    assert t.lineage['upstream_revision']=='pinned'
    assert t.lineage['teacher']['model']=='actual teacher'
    assert t.messages[1]['reasoning']=='Inspect first.'
    assert t.messages[1]['tool_calls'][0]['id']=='call'
    assert t.messages[2]['tool_call_id']=='call'
    assert t.messages[-1]['content']=='Completed patch.'
    assert group_key('swe-instance','protected') in t.split_groups
    assert group_key('repo','org/repo') in t.split_groups


def test_terminal_siblings_share_task_and_keep_distinct_runs_with_all_turns(tmp_path,monkeypatch):
    rows=[{'task':'same-task','run_id':run,'model':'teacher','model_provider':'self-hosted',
           'conversations':messages()} for run in ('a','b')]
    fixture(tmp_path,monkeypatch,'D35',rows)
    first,second=[t for _,t in acquired.nemotron_terminal_acquired()]
    assert first.id!=second.id and first.split_groups==second.split_groups
    assert len(first.messages)==4 and first.outcome['checked'] is None
    assert first.lineage['notes']['original_terminal']['run_id']=='a'
    assert first.lineage['teacher']['model']=='teacher'


def test_cooper_keeps_all_original_agents_and_shared_task_and_communication_facts(tmp_path,monkeypatch):
    row={'pair_id':'pair','repo':'org/repo','task_id':'task','both_passed':False,
         'model':'teacher','split':'sft','conversation':[{'text':'Original exchange.'}],
         'trajectories':[{'agent':name,'messages_json':json.dumps(messages())} for name in ('a','b')]}
    fixture(tmp_path,monkeypatch,'D17',[row])
    values=[t for _,t in acquired.cooperdata_acquired()]
    assert len(values)==2 and values[0].id!=values[1].id
    assert set(values[0].split_groups)==set(values[1].split_groups)
    assert all(t.outcome['label']=='failed' and t.outcome['checked'] is None for t in values)
    assert values[0].lineage['notes']['original_cooperation']['conversation']==row['conversation']


def test_malformed_session_is_recorded_and_following_valid_session_survives(tmp_path,monkeypatch):
    fixture(tmp_path,monkeypatch,'D35',[{'task':'bad'},
        {'task':'good','conversations':messages(),'model':'teacher'}])
    values=list(acquired.nemotron_terminal_acquired())
    assert isinstance(values[0][1],Reject) and not isinstance(values[1][1],Reject)


def test_missing_cooper_session_is_explicit_omission_and_does_not_stop_source(tmp_path,monkeypatch):
    fixture(tmp_path,monkeypatch,'D17',[{'pair_id':'empty','trajectories':[]},
        {'pair_id':'good','task_id':'t','repo':'r','trajectories':[{'agent':'a','messages_json':json.dumps(messages())}]}])
    values=list(acquired.cooperdata_acquired())
    assert isinstance(values[0][1],Reject) and not isinstance(values[1][1],Reject)


def test_acquired_reader_rejects_mutated_source_and_unfinished_receipt(tmp_path,monkeypatch):
    root=fixture(tmp_path,monkeypatch,'D35',[])
    (root/'sessions.parquet').write_bytes(b'changed')
    with pytest.raises(Reject,match='source file differs'):
        list(acquired.acquired_rows('D35'))
    receipt=json.loads((root/'acquisition.json').read_text());receipt['status']='partial'
    (root/'acquisition.json').write_text(json.dumps(receipt))
    with pytest.raises(Reject,match='not completely acquired'):
        list(acquired.acquired_rows('D35'))


def test_receipt_file_shards_retain_exact_original_row_identity(tmp_path,monkeypatch):
    root=fixture(tmp_path,monkeypatch,'D35',[{'task':'t','conversations':messages()}])
    monkeypatch.setattr(acquired,'ACQUIRED_FILES',frozenset({'sessions.parquet'}))
    selected=list(acquired.acquired_rows('D35'))
    assert selected[0][1]=='sessions.parquet:0'
    monkeypatch.setattr(acquired,'ACQUIRED_FILES',frozenset({'other.parquet'}))
    with pytest.raises(Reject,match='not a bound source'):
        list(acquired.acquired_rows('D35'))


def test_sharded_source_receipt_and_file_index_are_cached_but_stat_changes_revalidate(tmp_path,monkeypatch):
    root=fixture(tmp_path,monkeypatch,'D35',[{'task':'t','conversations':messages()}])
    monkeypatch.setattr(acquired,'ACQUIRED_FILES',frozenset({'sessions.parquet'}))
    original=acquired.validate_acquisition;calls=[]
    def counted(*args,**kwargs):
        calls.append(args[0]);return original(*args,**kwargs)
    monkeypatch.setattr(acquired,'validate_acquisition',counted)
    assert len(list(acquired.acquired_rows('D35')))==1
    assert len(list(acquired.acquired_rows('D35')))==1
    assert calls==['D35']
    receipt_path=root/'acquisition.json';value=json.loads(receipt_path.read_text());value['status']='partial'
    receipt_path.write_text(json.dumps(value))
    stat=receipt_path.stat()
    import os
    os.utime(receipt_path,ns=(stat.st_atime_ns,stat.st_mtime_ns+1_000_000))
    with pytest.raises(Reject,match='not completely acquired'):
        list(acquired.acquired_rows('D35'))
    assert calls==['D35','D35']


def test_cached_receipt_still_rehashes_each_selected_source_file(tmp_path,monkeypatch):
    root=fixture(tmp_path,monkeypatch,'D35',[{'task':'t','conversations':messages()}])
    monkeypatch.setattr(acquired,'ACQUIRED_FILES',frozenset({'sessions.parquet'}))
    assert len(list(acquired.acquired_rows('D35')))==1
    (root/'sessions.parquet').write_bytes(b'mutated after receipt cache population')
    with pytest.raises(Reject,match='source file differs'):
        list(acquired.acquired_rows('D35'))


def test_conversion_records_row_rejection_then_keeps_following_trajectory(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import neuralese_port_records as cli
    row=SimpleNamespace(lineage={},outcome={'label':'teacher'})
    monkeypatch.setitem(agents.CORPORA,'fixture-acquired',
        (lambda limit:iter([('bad',Reject('missing original session')),('good',row)]),4))
    seen=[]
    monkeypatch.setattr(cli.trajectory,'records_for',lambda t,*args:seen.append(t))
    cli.cmd_convert_agents(SimpleNamespace(corpora=['fixture-acquired'],limit=None,out=tmp_path,windows=None))
    assert seen==[row]
    rejection=json.loads((tmp_path/'upstream.fixture-acquired.rejects.jsonl').read_text())
    assert rejection['row']=='bad' and rejection['reason']=='missing original session'


@pytest.mark.parametrize('ref',['D26','D27','D28','D38','D23','D36','D37'])
def test_general_sources_keep_structured_history_and_metadata(ref,tmp_path,monkeypatch):
    rows=[{'messages':json.dumps(messages()),'uuid':'session','tools':'[]',
           'metadata':json.dumps({'instance_id':'protected','repo':'org/repo','model':'source-teacher'}),
           'reward':0 if ref=='D36' else 1}]
    root=fixture(tmp_path,monkeypatch,ref,rows)
    if ref=='D36':
        receipt=json.loads((root/'acquisition.json').read_text());entry=receipt['files'][0]
        entry['path']='trajectories/sessions.parquet';(root/'trajectories').mkdir()
        (root/'sessions.parquet').rename(root/entry['path']);(root/'acquisition.json').write_text(json.dumps(receipt))
    _,trajectory=next(acquired.general_agent_acquired(ref))
    assert not isinstance(trajectory,Reject)
    assert trajectory.messages[1]['reasoning']=='Inspect first.'
    assert trajectory.messages[2]['content']=='Observed file listing.'
    assert trajectory.split=='test' and group_key('swe-instance','protected') in trajectory.split_groups
    assert trajectory.outcome['label']==('failed' if ref=='D36' else 'teacher')


def test_bound_jsonl_is_streamed_and_images_remain_explicit_omissions(tmp_path,monkeypatch):
    root=fixture(tmp_path,monkeypatch,'D28',[])
    path=root/'agent.jsonl.gz'
    import gzip
    rows=[{'messages':messages(),'uuid':'one'}, {'messages':messages(),'uuid':'two','images':[{'path':'unavailable.png'}]}]
    with gzip.open(path,'wt') as stream:
        for row in rows:stream.write(json.dumps(row)+'\n')
    receipt=json.loads((root/'acquisition.json').read_text());receipt['files']=[{'path':path.name,
        'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}]
    (root/'acquisition.json').write_text(json.dumps(receipt))
    result=list(acquired.nemotron_agentic_acquired())
    assert len(result)==2 and result[0][0]=='agent.jsonl.gz:0'
    assert not isinstance(result[0][1],Reject)
    assert isinstance(result[1][1],Reject) and 'multimodal' in str(result[1][1])


@pytest.mark.parametrize('suffix',['.json','.json.gz'])
@pytest.mark.parametrize('array',[False,True])
def test_whole_json_reader_preserves_original_integer_rows(tmp_path,suffix,array):
    import gzip
    value=[{'id':'first','messages':messages()},{'id':'second','content':'entire original text'}]
    source=value if array else value[0]
    path=tmp_path/('sessions'+suffix);raw=json.dumps(source,ensure_ascii=False).encode()
    path.write_bytes(gzip.compress(raw) if suffix.endswith('.gz') else raw)
    assert acquired.supported_acquired_file(path.name)
    assert list(acquired.iter_acquired_file_rows(path))==list(enumerate(value if array else [value[0]]))


@pytest.mark.parametrize('suffix',['.jsonl','.jsonl.gz','.jsonl.zst'])
def test_shared_line_reader_retains_blank_line_indices_and_all_content(tmp_path,suffix):
    import gzip
    source='\n'+json.dumps({'id':'first','text':'α\nfull'})+'\n\n'+json.dumps({'id':'last'})+'\n'
    path=tmp_path/('sessions'+suffix)
    if suffix.endswith('.gz'):path.write_bytes(gzip.compress(source.encode()))
    elif suffix.endswith('.zst'):
        import zstandard
        path.write_bytes(zstandard.ZstdCompressor().compress(source.encode()))
    else:path.write_text(source)
    assert list(acquired.iter_acquired_file_rows(path))==[(1,{'id':'first','text':'α\nfull'}),(3,{'id':'last'})]


def test_shared_reader_rejects_unsupported_files(tmp_path):
    path=tmp_path/'README.md';path.write_text('source card')
    assert not acquired.supported_acquired_file(path)
    with pytest.raises(Reject,match='unsupported acquired'):list(acquired.iter_acquired_file_rows(path))


def test_recovered_acquisition_requires_exact_archive_and_complete_inventory(tmp_path):
    import tarfile
    from io import BytesIO
    from neuralese_data.acquired_agents import validate_recovery_manifest
    root=tmp_path/'D13';root.mkdir()
    payload=root/'sessions.jsonl';payload.write_text('{"messages":[]}\n')
    files=[{'path':payload.name,'bytes':payload.stat().st_size,
            'sha256':hashlib.sha256(payload.read_bytes()).hexdigest()}]
    inventory_sha=hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    embedded={'schema':'natlang.pinned-public-source-archive/1','acquisition':{
        'schema':'natlang.external-source-acquisition/1','ref':'D13','repo':'CooperBench/team-coop',
        'revision':'7348f4baa12d53c2c2d537943e7846ff8587e411','files':files}}
    embedded_bytes=json.dumps(embedded,separators=(',',':')).encode()
    archive=tmp_path/'D13-pinned-source.tar'
    with tarfile.open(archive,'w') as stream:
        info=tarfile.TarInfo('D13/sessions.jsonl');info.size=payload.stat().st_size
        stream.addfile(info,BytesIO(payload.read_bytes()))
        info=tarfile.TarInfo('D13/durability-manifest.json');info.size=len(embedded_bytes)
        stream.addfile(info,BytesIO(embedded_bytes))
    receipt={
        'schema':'natlang.external-source-recovery/1','ref':'D13','repo':'CooperBench/team-coop',
        'revision':'7348f4baa12d53c2c2d537943e7846ff8587e411','status':'recovered',
        'training_admission':False,'files':files,'file_inventory_sha256':inventory_sha,
        'expected_source_inventory':{'files':1,'bytes':payload.stat().st_size,'sha256':inventory_sha},
        'historical_acquisition_receipt':{'available':False,'expected_sha256':'1'*64},
        'source_archive':{'verified':True,'private':True,'path':str(archive),'bytes':archive.stat().st_size,
            'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'run':'werg/natlang/ywcj2mvb',
            'name':'mellum/example/D13-pinned-source.tar'},
        'embedded_archive_manifest':{'path':'D13/durability-manifest.json','bytes':len(embedded_bytes),
            'sha256':hashlib.sha256(embedded_bytes).hexdigest(),'schema':'natlang.pinned-public-source-archive/1'},
    }
    manifest=root/'recovery-manifest.json';manifest.write_text(json.dumps(receipt))
    assert validate_recovery_manifest('D13',receipt,manifest,root=root)==receipt
    receipt['expected_source_inventory']['bytes']+=1
    with pytest.raises(Reject,match='inventory differs'):
        validate_recovery_manifest('D13',receipt,manifest,root=root)
    receipt['expected_source_inventory']['bytes']-=1
    payload.write_text('{"messages":[1]}\n')
    with pytest.raises(Reject,match='inventory file differs'):
        validate_recovery_manifest('D13',receipt,manifest,root=root)
    payload.write_text('{"messages":[]}\n')
    receipt['source_archive']['sha256']='0'*64
    with pytest.raises(Reject,match='archive differs'):
        validate_recovery_manifest('D13',receipt,manifest,root=root)
    receipt['source_archive']['sha256']=hashlib.sha256(archive.read_bytes()).hexdigest()
    receipt['files'][0]['bytes']+=1
    with pytest.raises(Reject,match='not bound to its retained archive proof'):
        validate_recovery_manifest('D13',receipt,manifest,root=root)
