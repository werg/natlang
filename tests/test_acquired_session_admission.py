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
