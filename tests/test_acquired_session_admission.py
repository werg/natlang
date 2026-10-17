"""Original source authenticity and whole-session coverage at admission."""
import copy
import pytest
from transformers import AutoTokenizer

from test_acquired_full_session_graphs import value, BASES
from scripts.neuralese_data.build_acquired_full_session_graphs import session_graphs
from scripts.neuralese_data.build_native_semantic_pairs_e2e import provision
from scripts.neuralese_data.prepare_acquired_session_admission import verify_graph_source
from natlang_neuralese.data.text_corpus import tokenizer_fingerprint


@pytest.mark.parametrize('fail,quality,pending,decision,expected',[
    (True,True,True,True,'source-reported-non-positive'),
    (False,True,True,True,'current-source-quality-or-pending-protection'),
    (False,False,True,True,'current-source-quality-or-pending-protection'),
    (False,False,False,True,'current-decision-hold'),
    (False,False,False,False,None)])
def test_shared_exclusion_retains_original_conversion_precedence(fail,quality,pending,decision,expected):
    from scripts.neuralese_data.build_acquired_full_session_graphs import session_exclusion
    row=value();row['positive_candidate']=not fail
    protection={'quality':{'source-task'} if quality else set(),'pending':{'source-task'} if pending else set()}
    assert session_exclusion(row,protection,{'source-session:1'} if decision else set())==expected


def test_decision_hold_contract_uses_exact_immutable_file_identity_across_checkouts():
    from scripts.neuralese_data.prepare_acquired_session_admission import decision_contract
    first=[{'path':'/frozen/training/decision-holds/held.jsonl','sha256':'a'*64,'bytes':10,'kind':'decision-hold'}]
    second=[{**first[0],'path':'/active/training/decision-holds/held.jsonl'}]
    assert decision_contract(first)==decision_contract(second)
    for key,value in [('sha256','b'*64),('bytes',11),('path','/active/other.jsonl')]:
        assert decision_contract(first)!=decision_contract([{**second[0],key:value}])


def test_incremental_member_requires_one_exact_projected_export_and_lists_pending_siblings():
    from scripts.neuralese_data.prepare_acquired_session_admission import source_member_scope
    declared=[{'path':f'/source/{i}.json','sha256':str(i)*64,'source_sha256':str(i+1)*64} for i in range(15)]
    selected=[{'path':'/alias/0.json','sha256':declared[0]['sha256'],
               'source_sha256':declared[0]['source_sha256']}]
    scope=source_member_scope({'normalized_export_bindings':declared},selected)
    assert scope['kind']=='complete-normalized-source-member'
    assert scope['selected_exports']==[declared[0]]
    assert scope['pending_export_count']==14 and len(scope['pending_exports'])==14
    with pytest.raises(ValueError):source_member_scope({'normalized_export_bindings':declared},selected*2)
    with pytest.raises(ValueError):source_member_scope({'normalized_export_bindings':declared},[
        {**selected[0],'source_sha256':'f'*64}])


def test_incremental_member_accepts_declared_complete_trajectory_without_broadening_other_sources():
    from scripts.neuralese_data.prepare_acquired_session_admission import incremental_member_source_kind
    d13={'contract':{'source':{'path':'qwen35-cooperdata/agent_traj.json'},
        'source_rows':1,'trajectory_representation':'existing complete Trajectory dataclass'},
        'complete_sessions':1}
    assert incremental_member_source_kind(d13)=='complete-trajectory-single-source-row'
    assert incremental_member_source_kind({**d13,'complete_sessions':4})=='complete-trajectory-single-source-row'
    assert incremental_member_source_kind({'contract':{'source':{'path':'jsonl_sft/train.jsonl'}}})=='jsonl_sft'
    assert incremental_member_source_kind({**d13,'contract':{**d13['contract'],'source_rows':2}})=='complete-trajectory-source-export'
    assert incremental_member_source_kind({**d13,'contract':{**d13['contract'],'source_rows':True}}) is None
    d32={'contract':{'source':{'path':'seed_42/train-00002-of-00003.parquet'},
        'source_rows':2034,'trajectory_representation':'existing complete Trajectory dataclass'},
        'complete_sessions':1803}
    assert incremental_member_source_kind(d32)=='complete-trajectory-source-export'
    assert incremental_member_source_kind({**d32,'complete_sessions':True}) is None
    assert incremental_member_source_kind({**d32,'complete_sessions':0}) is None
    assert incremental_member_source_kind({**d32,'contract':{**d32['contract'],'source_rows':True}}) is None
    assert incremental_member_source_kind({**d32,'contract':{**d32['contract'],'source_rows':0}}) is None
    assert incremental_member_source_kind({'contract':{'source':{'path':'dpo/train.jsonl'},
        'source_rows':1,'trajectory_representation':'existing complete Trajectory dataclass'},
        'complete_sessions':1}) is None


def test_native_conversion_directory_resolves_source_sha_and_legacy_export_identity(tmp_path):
    from scripts.neuralese_data.prepare_acquired_session_admission import native_conversion_directory
    root=tmp_path/'native';export_path=tmp_path/'exports'/'export-id'/'export.json'
    source_sha='a'*64;model='lfm25';by_source=root/source_sha/model;by_source.mkdir(parents=True)
    (by_source/'receipt.json').write_text('{}')
    export={'contract':{'source':{'sha256':source_sha}}}
    assert native_conversion_directory(root,export_path,export,model)==by_source.resolve()
    by_export=root/'export-id'/model;by_export.mkdir(parents=True)
    (by_export/'receipt.json').write_text('{}')
    with pytest.raises(ValueError,match='ambiguous'):
        native_conversion_directory(root,export_path,export,model)


def test_native_conversion_directory_keeps_legacy_fallback_for_unbound_source_hash(tmp_path):
    from scripts.neuralese_data.prepare_acquired_session_admission import native_conversion_directory
    root=tmp_path/'native';export_path=tmp_path/'exports'/'export-id'/'export.json'
    legacy=root/'export-id'/'mellum21';legacy.mkdir(parents=True)
    (legacy/'receipt.json').write_text('{}')
    assert native_conversion_directory(root,export_path,{'contract':{'source':{}}},'mellum21')==legacy.resolve()


@pytest.fixture(scope='module',params=BASES,ids=['lfm','mellum'])
def admitted(request):
    base=request.param
    if not (base/'tokenizer.json').is_file():pytest.skip('production tokenizer unavailable')
    tokenizer=AutoTokenizer.from_pretrained(base,local_files_only=True,trust_remote_code=False)
    source_fp=tokenizer_fingerprint(tokenizer);provision(tokenizer);fp=tokenizer_fingerprint(tokenizer)
    row=value();row['trajectory']['split']='train';row['trajectory']['messages'][0]['content']='complete original task'
    graphs=list(session_graphs(row,tokenizer,source_fingerprint=source_fp,expanded_fingerprint=fp,split='train'))
    return row,graphs,fp


def verify(graph,row,fp,**kwargs):
    return verify_graph_source(graph,row,protection={'held':set(),'quality':set(),'pending':set()},
        held_sources=set(),fingerprint=fp,**kwargs)


def test_actual_both_tokenizers_authenticate_complete_source_and_exact_views(admitted):
    row,graphs,fp=admitted;seen=set();expected=set()
    for graph in graphs:
        ids,width,split=verify(graph,row,fp);seen.add(graph['id']);expected.update(ids)
        assert width>0 and split=='train'
    assert seen==expected


@pytest.mark.parametrize('change',['target','source-row','groups','view','oracle','missing-reasoning'])
def test_forged_provenance_and_source_mutations_fail_closed(admitted,change):
    row,graphs,fp=admitted;graph=copy.deepcopy(graphs[0]);row=copy.deepcopy(row)
    if change=='target':row['trajectory']['messages'][-1]['content']='different source answer'
    elif change=='source-row':row['source_row']='different-occurrence'
    elif change=='groups':graph['source_groups']=['different-task']
    elif change=='view':graph['id']+=':reasoning-turn:100'
    elif change=='oracle':graph['conversion']['acquired_session']['oracle_verified']=True
    else:graph['conversion']['acquired_session']['reasoning_coverage']={}
    with pytest.raises(ValueError):verify(graph,row,fp)


@pytest.mark.parametrize('kind',['failed','nonpositive','quality','pending','decision','held'])
def test_source_failure_and_current_protections_cannot_become_positive_train(admitted,kind):
    row,graphs,fp=admitted;row=copy.deepcopy(row);graph=graphs[0]
    protection={'held':set(),'quality':set(),'pending':set()};held_sources=set()
    if kind=='failed':row['trajectory']['outcome']['label']='failed'
    elif kind=='nonpositive':row['positive_candidate']=False
    elif kind=='decision':held_sources.add(row['trajectory']['id']+':1')
    else:protection[kind].add('source-task')
    with pytest.raises(ValueError):verify_graph_source(graph,row,protection=protection,
        held_sources=held_sources,fingerprint=fp)


@pytest.mark.parametrize('base',BASES,ids=['lfm','mellum'])
def test_actual_completed_durable_conversion_rejoins_all_positive_and_failed_sources(base,tmp_path,monkeypatch):
    import json
    from pathlib import Path
    from test_prepared_acquired_tokenizers import export
    from scripts.neuralese_data import build_acquired_full_session_graphs as builder
    from scripts.neuralese_data.prepare_acquired_session_admission import verify_conversion
    from scripts.neuralese_data.decision_holds import load_decision_holds
    from natlang_neuralese.common.hashing import sha256_file_hex
    from natlang_neuralese.common.jsonio import write_jsonl_text
    if not (base/'tokenizer.json').is_file():pytest.skip('production tokenizer unavailable')
    receipt=export(tmp_path/'source','source-one');source=receipt.parent/'full-trajectories.jsonl.zst'
    rows=list(builder.sessions(source));failed=copy.deepcopy(rows[0]);failed['trajectory']['id']='failed'
    failed['source_row']='failed:1';failed['trajectory']['outcome']['label']='failed';rows.append(failed)
    with write_jsonl_text(source) as stream:
        for row in rows:stream.write(json.dumps(row)+'\n')
    metadata=json.loads(receipt.read_text());metadata['complete_sessions']=2
    metadata['outputs'][source.name]=sha256_file_hex(source);receipt.write_text(json.dumps(metadata))
    protection={'held':set(),'quality':set(),'pending':set(),'proof':{}}
    monkeypatch.setattr(builder,'current_protection',lambda path:protection)
    holds=sorted((Path(builder.__file__).resolve().parents[2]/'training/decision-holds').glob('*.jsonl'))
    ids,bindings=load_decision_holds(holds);output=tmp_path/'graphs'
    argv=['--export-receipt',str(receipt),'--base-tokenizer',str(base),'--protection','fixture','--output',str(output)]
    for hold in holds:argv+=['--decision-hold',str(hold)]
    builder.main(argv)
    checked,outputs,proofs,groups=verify_conversion(output,receipt,protection=protection,
        held_sources=ids|{v.split(':decision:',1)[0] for v in ids},decision_bindings=bindings,source_rows=rows)
    assert checked['counts']['positive_sessions']==1 and checked['counts']['source-reported-non-positive']==1
    assert all(p['source_outcome']['label']!='failed' for p in proofs)
    assert len(proofs)==sum(item['rows'] for item in outputs) and not groups['test']
    alias=tmp_path/'logical-graphs';alias.symlink_to(output,target_is_directory=True)
    assert verify_conversion(alias,receipt,protection=protection,
        held_sources=ids|{v.split(':decision:',1)[0] for v in ids},decision_bindings=bindings,
        source_rows=rows)==(checked,outputs,proofs,groups)


def test_invalid_original_tool_argument_is_explicit_pending_and_admission_rechecks_it(tmp_path,monkeypatch):
    import json,sys
    from pathlib import Path
    from types import SimpleNamespace
    from scripts.neuralese_data import build_acquired_full_session_graphs as builder
    from scripts.neuralese_data.prepare_acquired_session_admission import verify_conversion
    from scripts.neuralese_data.decision_holds import load_decision_holds
    from natlang_neuralese.common.hashing import sha256_file_hex
    from natlang_neuralese.common.jsonio import write_jsonl_text
    row=value(); row['trajectory']['split']='train'
    row['trajectory']['messages'][1]['tool_calls'][0]['function']['arguments']=(
        '{"query": \\"City Reapers win OTE championship\\" 2023", "topn": 10}')
    original=copy.deepcopy(row)
    source=tmp_path/'full-trajectories.jsonl.zst'
    with write_jsonl_text(source) as stream:stream.write(json.dumps(row)+'\n')
    export=tmp_path/'export.json'
    export.write_text(json.dumps({'schema':'natlang.normalized-agent-source-file/1',
        'contract':{'source':row['source_file'],'revision':row['source_revision']},
        'outputs':{source.name:sha256_file_hex(source)},'complete_sessions':1}))
    prepared=SimpleNamespace(tokenizer=object(),source_fingerprint='a'*64,expanded_fingerprint='a'*64,
        rendering_configuration={'history_reasoning':{'backbone':'lfm2'}},
        binding={'schema':'test-prepared-tokenizer','rendering_configuration':
            {'history_reasoning':{'backbone':'lfm2'}}},validate=lambda base:None)
    monkeypatch.setattr(builder,'prepare_tokenizer',lambda base:prepared)
    monkeypatch.setattr(builder.AutoTokenizer,'from_pretrained',lambda *a,**k:object())
    monkeypatch.setattr(builder,'tokenizer_fingerprint',lambda t:'a'*64)
    monkeypatch.setattr(builder,'provision',lambda t:None)
    protection={'held':set(),'quality':set(),'pending':set(),'proof':{}}
    monkeypatch.setattr(builder,'current_protection',lambda p:protection)
    holds=sorted((Path(builder.__file__).resolve().parents[2]/'training/decision-holds').glob('*.jsonl'))
    monkeypatch.setattr(builder,'load_decision_holds',lambda paths:(set(),[]))
    output=tmp_path/'graphs';argv=['builder','--export-receipt',str(export),'--base-tokenizer','fixture',
        '--protection','fixture','--output',str(output)]
    for hold in holds:argv+=['--decision-hold',str(hold)]
    monkeypatch.setattr(sys,'argv',argv)
    builder.main()
    assert row==original
    receipt=json.loads((output/'receipt.json').read_text())
    assert receipt['counts']['source-render-pending:tool-arguments']==1
    assert receipt['counts'].get('positive_sessions',0)==0
    omissions=[json.loads(line) for line in (output/'omissions.jsonl').read_text().splitlines()]
    assert len(omissions)==1 and omissions[0]['pending_status']=='source-faithful-review-pending'
    checked,_,proofs,_=verify_conversion(output,export,protection=protection,held_sources=set(),
        decision_bindings=[],source_rows=[row])
    assert checked['counts']['source-render-pending:tool-arguments']==1 and not proofs


@pytest.mark.parametrize(('backbone','pending'), [('lfm2', True), ('mellum', False)])
def test_admission_prevalidation_is_bound_to_actual_template_and_conserves_scalar_source(backbone,pending):
    from natlang_neuralese.common.hashing import canonical_json_sha256_hex
    from scripts.neuralese_data.prepare_acquired_session_admission import _prevalidate_source_for_conversion
    from natlang_neuralese.serve.chat import RequestError
    row=value()
    row['trajectory']['messages'][1]['tool_calls'][0]['function']['arguments']='"plain scalar"'
    source_sha=canonical_json_sha256_hex(row)
    target_ids=['source-session:1','source-session:3']
    receipt={'prepared_tokenizer':{'rendering_configuration':{
        'history_reasoning':{'backbone':backbone}}}}
    if pending:
        with pytest.raises(RequestError) as error:
            _prevalidate_source_for_conversion(row,receipt)
        assert error.value.code=='tool-arguments'
    else:
        normalized,conversions=_prevalidate_source_for_conversion(row,receipt)
        assert normalized[1]['tool_calls'][0]['function']['arguments']=='plain scalar'
        assert conversions
    assert canonical_json_sha256_hex(row)==source_sha
    assert [row['trajectory']['id']+':'+str(i) for i,m in enumerate(row['trajectory']['messages'])
        if m.get('role')=='assistant']==target_ids


def test_paired_admission_keeps_mellum_scalar_while_lfm2_records_exact_pending_occurrence():
    from scripts.neuralese_data.prepare_acquired_session_admission import (
        verify_model_source_parity, whole_trajectory_source_view)
    key=('source-sha','source.parquet:17')
    accepted={('source-session:source-occurrence:x','source-sha','train','whole-trajectory')}
    proofs=[
        {'id':'source-session:source-occurrence:x','normalized_session_sha256':'source-sha',
            'source_row':'source.parquet:17','split':'train','view':'whole-trajectory'},
        {'id':'per-model-auxiliary-lfm','normalized_session_sha256':'source-sha',
            'source_row':'source.parquet:17','split':'train','view':'full-prefix-reasoning-target'},
    ]
    assert whole_trajectory_source_view(proofs)=={key:accepted}
    verify_model_source_parity({
        'lfm25':{'backbone':'lfm2','accepted':{},'pending':{key},
            'auxiliary':{'view':'per-model-diagnostic','proofs':{'lfm-only'}}},
        'mellum21':{'backbone':'mellum','accepted':{key:accepted},'pending':set(),
            'auxiliary':{'view':'per-model-diagnostic','proofs':{'mellum-only'}}}})
    with pytest.raises(ValueError,match='both accepted and render-pending'):
        verify_model_source_parity({
            'lfm25':{'backbone':'lfm2','accepted':{key:accepted},'pending':{key}},
            'mellum21':{'backbone':'mellum','accepted':{key:accepted},'pending':set()}})
    with pytest.raises(ValueError,match='cross-model source outcome'):
        verify_model_source_parity({
            'lfm25':{'backbone':'lfm2','accepted':{key:accepted},'pending':set()},
            'mellum21':{'backbone':'mellum','accepted':{},'pending':{key}}})
    with pytest.raises(ValueError,match='missing from a complete'):
        verify_model_source_parity({
            'lfm25':{'backbone':'lfm2','accepted':{},'pending':{key}},
            'mellum21':{'backbone':'mellum','accepted':{},'pending':set()}})
