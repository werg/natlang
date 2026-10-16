"""Immutable offline pair joins, whole-pair holds and durable resumable rendering."""
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from natlang_neuralese.common.hashing import sha256_file_hex
from natlang_neuralese.common.jsonio import write_jsonl_text,write_json_durable
from scripts.neuralese_data import acquired_granted as granted
from scripts.neuralese_data.acquired_preferences import normalized_preference_records
from scripts.neuralese_data import build_acquired_preference_graphs as builder
from scripts.neuralese_data.build_acquired_full_session_graphs import sessions


def normalized(tmp_path,count=2):
    source=tmp_path/'full-trajectories.jsonl.zst'
    contract={'source':{'path':'dpo/pairs.json','sha256':'a'*64,'bytes':123},'revision':'b'*40}
    receipt={'repo':'miromind-ai/MiroVerse-v0.1','revision':contract['revision'],
             'files':[contract['source']],'card':{'license':'source-pinned'}}
    values=[]
    for index in range(count):
        raw={'chosen':[{'role':'user','content':'Question '+str(index)},
                       {'role':'assistant','content':'A correct useful response '+str(index)}],
             'rejected':[{'role':'user','content':'Question '+str(index)},
                         {'role':'assistant','content':'An inferior response '+str(index)}]}
        for rid,t in granted.miro_pair_trajectories(receipt,'dpo/pairs.json:'+str(index),raw):
            values.append({'trajectory':asdict(t),'source_row':rid,'source_file':contract['source'],
                           'source_revision':contract['revision'],'positive_candidate':rid.endswith(':chosen')})
    with write_jsonl_text(source) as stream:
        for value in values:stream.write(json.dumps(value)+'\n')
    export=tmp_path/'export.json'
    write_json_durable(export,{'schema':'natlang.normalized-agent-source-file/1','contract':contract,
        'complete_sessions':len(values),'outputs':{source.name:sha256_file_hex(source)}})
    return source,export,values


def test_exact_adjacent_normalized_source_join_and_false_pair_rejected(tmp_path):
    source,_,values=normalized(tmp_path)
    records=list(normalized_preference_records(source))
    assert [row[0] for row in records]==[(0,1),(2,3)]
    assert all(row[2] is None for row in records)
    assert records[0][1]['alternatives']['chosen']['messages']==values[0]['trajectory']['messages']
    assert records[0][1]['alternatives']['rejected']['messages']==values[1]['trajectory']['messages']
    values[1]['trajectory']['lineage']['notes']['source_preference']['pair_sha256']='f'*64
    with write_jsonl_text(source) as stream:
        for value in values:stream.write(json.dumps(value)+'\n')
    with pytest.raises(ValueError,match='exact adjacent original pair'):
        list(normalized_preference_records(source))


def test_source_failure_never_becomes_positive_pair_and_both_original_rows_accounted(tmp_path):
    source,_,values=normalized(tmp_path,count=1)
    values[0]['trajectory']['outcome']['label']='failed'
    with write_jsonl_text(source) as stream:
        for value in values:stream.write(json.dumps(value)+'\n')
    indices,record,error=next(normalized_preference_records(source))
    assert indices==(0,1) and record is None and 'source-reported failed' in str(error)



def test_failed_rejected_alternative_remains_usable_only_as_explicit_negative_preference(tmp_path):
    source,_,values=normalized(tmp_path,count=1)
    values[1]['trajectory']['outcome']['label']='failed'
    with write_jsonl_text(source) as stream:
        for value in values:stream.write(json.dumps(value)+'\n')
    indices,record,error=next(normalized_preference_records(source))
    assert indices==(0,1) and error is None
    assert record['alternatives']['rejected']['outcome']['label']=='failed'
    assert record['source_preference']=='chosen-over-rejected' and not record['training_admission']

def setup_conversion(tmp_path,monkeypatch):
    from tests.neuralese.test_native_serve import _tiny_source as tiny_source
    source,export,values=normalized(tmp_path)
    base=tmp_path/'base';tiny_source(base)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(base)
    tokenizer.chat_template=("{% for message in messages %}{{ message['role'] }}: {% if message['role'] == 'assistant' and message.get('reasoning_content') and loop.last %}{{ message['reasoning_content'] }}{% endif %}{{ message['content'] }}\n{% endfor %}"
                             "{% if add_generation_prompt %}assistant: {% endif %}")
    tokenizer.save_pretrained(base)
    proof=tmp_path/'protection.json'
    write_json_durable(proof,{'normalized_export_bindings':[
        {'sha256':sha256_file_hex(export),'source_sha256':sha256_file_hex(source)}]})
    protected={'held':set(),'quality':set(),'pending':set(),'proof':{'sha256':'c'*64}}
    monkeypatch.setattr(builder,'current_protection',lambda path:protected)
    monkeypatch.setattr(builder,'verify_global_source_closure',lambda *args:None)
    root=tmp_path/'runtime';(root/'training/decision-holds').mkdir(parents=True)
    hold=root/'training/decision-holds/holds.jsonl';hold.write_text('')
    monkeypatch.setattr(builder,'ROOT',root)
    monkeypatch.setattr(builder,'_runtime',lambda:{'fingerprint':'d'*64})
    monkeypatch.setattr(builder,'load_decision_holds',lambda paths:(set(),[]))
    args=['--export-receipt',str(export),'--base-tokenizer',str(base),'--protection',str(proof),
          '--decision-hold',str(hold),'--output',str(tmp_path/'output'),'--chunk-pairs','1']
    return args,protected,values


def test_durable_shards_resume_without_replaying_and_bind_objective_sidecar(tmp_path,monkeypatch):
    args,_,_=setup_conversion(tmp_path,monkeypatch)
    original=builder.pair_graph;calls=[]
    def interrupted(*positional,**kwargs):
        calls.append(1)
        if len(calls)==2:raise RuntimeError('interrupted after first durable pair')
        return original(*positional,**kwargs)
    monkeypatch.setattr(builder,'pair_graph',interrupted)
    with pytest.raises(RuntimeError,match='interrupted'):builder.main(args)
    assert (tmp_path/'output/pair-shards/shard-00000000/receipt.json').is_file()
    calls.clear();monkeypatch.setattr(builder,'pair_graph',lambda *a,**kw:(calls.append(1),original(*a,**kw))[1])
    result=builder.main(args)
    assert len(calls)==1 and result['counts']['graphs:train']==2 and result['counts']['normalized_sessions']==4
    objective=json.loads(Path(result['objective']['path']).read_text())
    graphs=list(sessions(tmp_path/'output/train/native-graphs.jsonl.zst'))
    assert set(objective['rows'])=={row['id'] for row in graphs}
    assert all(not any(node['target_weights']) for row in graphs for node in row['nodes'])
    assert builder.main(args)==result and len(calls)==1
    shard=tmp_path/'output/pair-shards/shard-00000000/objective.json'
    shard.write_text(shard.read_text()+' ')
    with pytest.raises(ValueError,match='differs from its receipt'):builder.main(args)


def test_connected_hold_moves_both_alternatives_and_never_partially_trains_pair(tmp_path,monkeypatch):
    args,protected,values=setup_conversion(tmp_path,monkeypatch)
    protected['held'].update(values[0]['trajectory']['split_groups'])
    result=builder.main(args)
    assert result['counts']['graphs:test']>=1
    train=list(sessions(tmp_path/'output/train/native-graphs.jsonl.zst'))
    test=list(sessions(tmp_path/'output/test/native-graphs.jsonl.zst'))
    assert len(train)+len(test)==2
    assert all(row['split']=='test' and all(node['name'].split(':')[0] in ('chosen','rejected') for node in row['nodes']) for row in test)



def test_independent_admission_checks_every_pair_source_mask_and_protection(tmp_path,monkeypatch):
    from scripts.neuralese_data import prepare_acquired_preference_admission as admission
    args,protected,_=setup_conversion(tmp_path,monkeypatch)
    result=builder.main(args)
    monkeypatch.setattr(admission,'ROOT',builder.ROOT)
    monkeypatch.setattr(admission,'current_protection',builder.current_protection)
    monkeypatch.setattr(admission,'verify_global_source_closure',builder.verify_global_source_closure)
    monkeypatch.setattr(admission,'load_decision_holds',builder.load_decision_holds)
    monkeypatch.setattr(admission,'_runtime',builder._runtime)
    report=admission.verify_conversion(tmp_path/'output',protection_path=tmp_path/'protection.json',base=tmp_path/'base')
    assert report['counts']['pairs:train']==2 and report['normalized_sessions']==4
    assert len(report['proofs'])==2
    protected['proof']={'sha256':'new stricter source closure'}
    with pytest.raises(ValueError,match='different current source protection'):
        admission.verify_conversion(tmp_path/'output',protection_path=tmp_path/'protection.json',base=tmp_path/'base')
    protected['proof']={'sha256':'c'*64}
    path=Path(result['objective']['path']);objective=json.loads(path.read_text())
    first=next(iter(objective['rows'].values()))
    first['suffixes'][next(iter(first['suffixes']))]['suffix_start']=1
    write_json_durable(path,objective)
    receipt=tmp_path/'output/receipt.json';r=json.loads(receipt.read_text())
    r['objective']=builder._bound(path);write_json_durable(receipt,r)
    with pytest.raises(ValueError,match='immutable graph identity'):
        admission.verify_conversion(tmp_path/'output',protection_path=tmp_path/'protection.json',base=tmp_path/'base')


def test_verified_model_view_cache_skips_rerender_only_for_same_exact_inputs(tmp_path,monkeypatch):
    from scripts.neuralese_data import prepare_acquired_preference_admission as admission
    args,_,_=setup_conversion(tmp_path,monkeypatch)
    builder.main(args)
    for name in ('current_protection','verify_global_source_closure','load_decision_holds','_runtime'):
        monkeypatch.setattr(admission,name,getattr(builder,name))
    monkeypatch.setattr(admission,'ROOT',builder.ROOT)
    cache=tmp_path/'view-verification.json'
    verified=set()
    original_pair=admission.pair_graph
    calls=[]
    def interrupt_after_one(record,tokenizer,**kwargs):
        calls.append(1)
        if len(calls)==2:raise RuntimeError('simulated stop between verified source pairs')
        return original_pair(record,tokenizer,**kwargs)
    monkeypatch.setattr(admission,'pair_graph',interrupt_after_one)
    with pytest.raises(RuntimeError,match='simulated stop'):
        admission.verify_conversion(tmp_path/'output',protection_path=tmp_path/'protection.json',
            base=tmp_path/'base',verified_closures=verified,verification_receipt_path=cache)
    checkpoint=next(tmp_path.glob('view-verification-*.renders.jsonl'))
    assert len(checkpoint.read_text().splitlines())==1 and not cache.exists()
    calls.clear();monkeypatch.setattr(admission,'pair_graph',lambda *a,**kw:(calls.append(1),original_pair(*a,**kw))[1])
    first=admission.verify_conversion(tmp_path/'output',protection_path=tmp_path/'protection.json',
        base=tmp_path/'base',verified_closures=verified,verification_receipt_path=cache)
    assert first['exact_original_alternative_renders']==2 and cache.is_file() and len(calls)==1
    # The source, model tokenizer, protection, holds and code bindings are
    # unchanged. Both completed record checks and the complete view are reused.
    calls.clear()
    monkeypatch.setattr(admission,'pair_graph',lambda *a,**kw: (_ for _ in ()).throw(AssertionError('rerendered cached view')))
    second=admission.verify_conversion(tmp_path/'output',protection_path=tmp_path/'protection.json',
        base=tmp_path/'base',verified_closures=verified,verification_receipt_path=cache)
    assert second==first
    prepared=admission.prepare_tokenizer(tmp_path/'base')
    prepared.tokenizer.chat_template+='\n'
    with pytest.raises(ValueError,match='tokenizer vocabulary/backend/template was mutated'):
        admission.verify_conversion(tmp_path/'output',protection_path=tmp_path/'protection.json',
            base=tmp_path/'base',prepared=prepared,verified_closures=verified,
            verification_receipt_path=cache)
    # A damaged durable attestation cannot authorize reuse.
    record=json.loads(cache.read_text());record['payload']['result']['source_units']+=1
    write_json_durable(cache,record)
    # The full-view attestation is now invalid, but the per-record checkpoints
    # still let us reconstruct it without re-running either tokenizer render.
    repaired=admission.verify_conversion(tmp_path/'output',protection_path=tmp_path/'protection.json',
        base=tmp_path/'base',verified_closures=verified,verification_receipt_path=cache)
    assert not calls and repaired['source_units']==first['source_units']


def test_source_eligibility_index_recovers_durable_prefix_and_reuses_across_tokenizer_views(tmp_path,monkeypatch):
    from scripts.neuralese_data import prepare_acquired_preference_admission as admission
    _,protected,values=setup_conversion(tmp_path,monkeypatch)
    source=tmp_path/'full-trajectories.jsonl.zst';export_path=tmp_path/'export.json'
    export=json.loads(export_path.read_text())
    source_binding={'path':str(source.resolve()),'sha256':sha256_file_hex(source),'bytes':source.stat().st_size}
    # One complete original pair has a source-quality hold and must remain an explicit omission.
    protected['quality'].update(values[0]['trajectory']['split_groups'])
    root=tmp_path/'shared-source-index';original=admission.normalized_preference_records
    original_sessions=admission.sessions
    def interrupt_after_one(source_path):
        iterator=original(source_path)
        yield next(iterator)
        raise RuntimeError('simulated stop after first indexed pair')
    monkeypatch.setattr(admission,'normalized_preference_records',interrupt_after_one)
    with pytest.raises(RuntimeError,match='simulated stop'):
        admission._source_eligibility_index(source,export_path,export,source_binding,None,
            protected,set(),[],root,expected_units=2,batch_units=1)
    partial=next(root.glob('*.state.json'))
    state=json.loads(partial.read_text())
    assert state['units']==1 and state['source_frontier']==2
    # Resume from the durable pair boundary; all normalized rows are accounted
    # for and the protected occurrence remains a source-level omission.
    monkeypatch.setattr(admission,'normalized_preference_records',original)
    expected,omitted,frontier,units,receipt=admission._source_eligibility_index(
        source,export_path,export,source_binding,None,protected,set(),[],root,
        expected_units=2,batch_units=1)
    assert frontier==4 and units==2 and len(expected)==1 and len(omitted)==1
    assert omitted[0]['normalized_source_rows']==[0,1]
    # The completed index is source-only and can be reused by the sibling
    # tokenizer view without another source parser pass.
    def forbidden(_):raise AssertionError('re-read completed shared source index')
    monkeypatch.setattr(admission,'sessions',forbidden)
    monkeypatch.setattr(admission,'normalized_preference_records',forbidden)
    again=admission._source_eligibility_index(source,export_path,export,source_binding,None,
        protected,set(),[],root,expected_units=2,batch_units=1)
    assert again==(expected,omitted,frontier,units,receipt)
    assert len(list(root.glob('*.receipt.json')))==1
    # A changed current protection gets a new identity and cannot reuse the
    # earlier eligibility/omission decision.
    monkeypatch.setattr(admission,'sessions',original_sessions)
    monkeypatch.setattr(admission,'normalized_preference_records',original)
    changed={**protected,'quality':set()}
    new_expected,new_omitted,_,_,new_receipt=admission._source_eligibility_index(
        source,export_path,export,source_binding,None,changed,set(),[],root,
        expected_units=2,batch_units=1)
    assert len(new_expected)==2 and not new_omitted
    assert new_receipt['identity_sha256']!=receipt['identity_sha256']
    assert len(list(root.glob('*.receipt.json')))==2


def test_conversion_runtime_drift_requires_the_exact_named_full_rerender_transition():
    from copy import deepcopy
    from scripts.neuralese_data import prepare_acquired_preference_admission as admission
    from scripts.neuralese_data import build_acquired_preference_graphs as source_builder
    current=source_builder._runtime()
    transition=admission._D24_RENDER_REQUALIFICATION
    old=deepcopy(current)
    for key,(before,after) in transition['changed_files'].items():
        assert old['files'][key]['sha256']==after
        old['files'][key]['sha256']=before
    old['files']['tokenizer_inventory']['sha256']='99eee6d5b2bbe87b8957b3df486951fdf37e8a102d73eaf0cbd91f7274924e9d'
    old['fingerprint']=transition['from_fingerprint']
    original_current=deepcopy(current)
    original_current['files']['tokenizer_inventory']['sha256']='99eee6d5b2bbe87b8957b3df486951fdf37e8a102d73eaf0cbd91f7274924e9d'
    original_current['fingerprint']=admission.canonical_json_sha256_hex(original_current['files'])
    original=admission._conversion_runtime_transition(old,original_current)
    assert original['kind']=='exact-all-pair-rerender-continuation'
    assert 'function_scoped_tokenizer_inventory_transition' not in original
    proof=admission._conversion_runtime_transition(old,current)
    assert proof['kind']=='exact-all-pair-rerender-continuation'
    assert set(proof['changed_files'])==set(transition['changed_files'])
    scoped=proof['function_scoped_tokenizer_inventory_transition']
    assert scoped['conversion_module_sha256']==old['files']['tokenizer_inventory']['sha256']
    assert scoped['verification_module_sha256']==current['files']['tokenizer_inventory']['sha256']
    assert scoped['qualified_function_ast_sha256']==admission._TOKENIZER_INVENTORY_WITNESS['functions']
    mutated=deepcopy(old);mutated['files']['graph_writer']['sha256']='0'*64
    mutated['fingerprint']=transition['from_fingerprint']
    with pytest.raises(ValueError,match='exact all-pair rerender continuation'):
        admission._conversion_runtime_transition(mutated,current)


def test_tokenizer_inventory_transition_rejects_function_ast_drift(tmp_path,monkeypatch):
    from scripts.neuralese_data import prepare_acquired_preference_admission as admission
    from scripts.neuralese_data import build_acquired_preference_graphs as source_builder
    import training.neuralese.natlang_neuralese.model.native_model as native_model
    import inspect
    from copy import deepcopy
    current=source_builder._runtime()
    old=deepcopy(current)
    for key,(before,_) in admission._D24_RENDER_REQUALIFICATION['changed_files'].items():
        old['files'][key]['sha256']=before
    old['files']['tokenizer_inventory']['sha256']='99eee6d5b2bbe87b8957b3df486951fdf37e8a102d73eaf0cbd91f7274924e9d'
    old['fingerprint']=admission._D24_RENDER_REQUALIFICATION['from_fingerprint']
    module=tmp_path/'training/neuralese/natlang_neuralese/model/native_model.py'
    module.parent.mkdir(parents=True)
    source=inspect.getsource(native_model).replace(
        'return _input_inventory(base, include_weights=False)',
        'return _input_inventory(base, include_weights=True)')
    module.write_text(source)
    monkeypatch.setattr(admission,'CODE_ROOT',tmp_path)
    with pytest.raises(ValueError,match='function witness'):
        admission._conversion_runtime_transition(old,current)
    unchanged=inspect.getsource(native_model)+'\n\ndef untrusted_rebind():\n    global Path\n    Path = object\n'
    module.write_text(unchanged)
    with pytest.raises(ValueError,match='global is rebound'):
        admission._tokenizer_inventory_ast_witness(module)
    module.write_text(inspect.getsource(native_model)+'\n\ntokenizer_inventory = lambda base: {}\n')
    with pytest.raises(ValueError,match='global is rebound'):
        admission._tokenizer_inventory_ast_witness(module)
    module.write_text(inspect.getsource(native_model)+'\n\ndef tokenizer_inventory(base):\n    return {}\n')
    with pytest.raises(ValueError,match='function witness is incomplete'):
        admission._tokenizer_inventory_ast_witness(module)


def test_admission_cli_preserves_existing_held_prefix_and_emits_executable_shared_stage_inputs(tmp_path,monkeypatch):
    from scripts.neuralese_data import prepare_acquired_preference_admission as admission
    args,_,values=setup_conversion(tmp_path,monkeypatch)
    result=builder.main(args)
    for name in ('current_protection','verify_global_source_closure','load_decision_holds','_runtime'):
        monkeypatch.setattr(admission,name,getattr(builder,name))
    monkeypatch.setattr(admission,'ROOT',builder.ROOT)
    receipt=json.loads((tmp_path/'output/receipt.json').read_text())
    fingerprint=receipt['binding']['prepared_tokenizer']['expanded_fingerprint']
    held=tmp_path/'ordinary-held.jsonl'
    row={'schema':'natlang.neuralese-native-graph/1','id':'ordinary-held','split':'test',
        'source_groups':['protected-existing-held'],'tokenizer_fingerprint':fingerprint,
        'nodes':[{'name':'answer','token_ids':[1,2,3],'target_weights':[0,1,1]}]}
    held.write_text(json.dumps(row)+'\n')
    declaration={'base':str(tmp_path/'base'),'conversions':[str(tmp_path/'output')],
                 'ordinary_held':builder._bound(held)}
    models={name:declaration for name in ('fixture-view-a','fixture-view-b')}
    report=admission.main(['--models',json.dumps(models),'--protection',str(tmp_path/'protection.json'),
                           '--output',str(tmp_path/'admission')])
    assert report['status']=='passed' and not report['capacity_qualified'] and not report['training_admission']
    assert report['training_entrypoints']['fixture-view-a']['kind']=='native_offline_preference'
    held_manifest=json.loads((tmp_path/'admission/fixture-view-a-test-sources.json').read_text())
    assert held_manifest['sources'][0]['sha256']==sha256_file_hex(held)
    from natlang_neuralese.data.native_sources import NativeGraphSource
    source=NativeGraphSource(tmp_path/'admission/fixture-view-a-train-sources.json',fingerprint,required_split='train')
    assert len(source)==2
    # A protected prior held group cannot be hidden by a wrong source projection.
    row['source_groups']=values[0]['trajectory']['split_groups'];held.write_text(json.dumps(row)+'\n')
    for value in models.values():value['ordinary_held']=builder._bound(held)
    with pytest.raises(ValueError,match='global train/held overlap'):
        admission.main(['--models',json.dumps(models),'--protection',str(tmp_path/'protection.json'),
                        '--output',str(tmp_path/'bad-admission')])


def test_admission_resumes_exact_partial_entrypoint_publication_and_rejects_changed_request(tmp_path,monkeypatch):
    from scripts.neuralese_data import prepare_acquired_preference_admission as admission
    args,_,_=setup_conversion(tmp_path,monkeypatch)
    builder.main(args)
    for name in ('ROOT','current_protection','verify_global_source_closure','load_decision_holds','_runtime'):
        monkeypatch.setattr(admission,name,getattr(builder,name))
    held=tmp_path/'ordinary-held.jsonl'
    receipt=json.loads((tmp_path/'output/receipt.json').read_text())
    fingerprint=receipt['binding']['prepared_tokenizer']['expanded_fingerprint']
    held.write_text(json.dumps({'schema':'natlang.neuralese-native-graph/1','id':'held',
        'split':'test','source_groups':['prior-held'],'tokenizer_fingerprint':fingerprint,
        'nodes':[{'name':'answer','token_ids':[1,2,3],'target_weights':[0,1,1]}]})+'\n')
    declaration={'base':str(tmp_path/'base'),'conversions':[str(tmp_path/'output')],
        'ordinary_held':builder._bound(held)}
    models={name:declaration for name in ('fixture-view-a','fixture-view-b')}
    output=tmp_path/'admission-resume'
    original=admission._write_owned_json;calls=[]
    def interrupt_after_one(path,value):
        calls.append(Path(path).name)
        if len(calls)==2:raise RuntimeError('stop during final proposal publication')
        return original(path,value)
    monkeypatch.setattr(admission,'_write_owned_json',interrupt_after_one)
    with pytest.raises(RuntimeError,match='final proposal publication'):
        admission.main(['--models',json.dumps(models),'--protection',str(tmp_path/'protection.json'),
            '--output',str(output)])
    assert (output/'.proposal-request.json').is_file()
    assert len(list(output.glob('*-sources.json')))==1
    monkeypatch.setattr(admission,'_write_owned_json',original)
    resumed=admission.main(['--models',json.dumps(models),'--protection',str(tmp_path/'protection.json'),
        '--output',str(output)])
    assert resumed['status']=='passed' and (output/'audit.json').is_file()
    changed={name:{**declaration,'ordinary_held':{**builder._bound(held),'sha256':'0'*64}}
        for name in models}
    with pytest.raises(ValueError,match='different proposal request'):
        admission.main(['--models',json.dumps(changed),'--protection',str(tmp_path/'protection.json'),
            '--output',str(output)])


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_actual_miro_full_pair_durable_conversion_both_production_tokenizers(tmp_path,monkeypatch,family):
    import hashlib
    from natlang_neuralese.common.paths import resolve
    from natlang_neuralese.data.native_preference import pair_graph
    from scripts.neuralese_data.build_acquired_full_session_graphs import prepare_tokenizer
    inspection=resolve('data_hdd','natlang-development-data/granted-format-inspection-20261016/inspection.json')
    base=resolve('data_nvme','dgxops/qad-drift-probe/base') if family=='lfm' else resolve('models','mellum21-12b-a2.5b-thinking')
    if not inspection.exists() or not (base/'tokenizer.json').exists():pytest.skip('pinned actual source/tokenizer absent')
    transfer=json.loads((inspection.parent/'transfer-receipt.json').read_text())
    assert hashlib.sha256(inspection.read_bytes()).hexdigest()==transfer['source']['sha256']
    entry=next(item for item in json.loads(inspection.read_text())['records'] if item['ref']=='D24' and item['catalog_file']['path'].startswith('dpo/'))
    original=entry['records'][0]
    source_binding={'path':entry['catalog_file']['path'],'bytes':entry['source_bytes'],'sha256':entry['source_sha256']}
    receipt={'repo':entry['repo'],'revision':entry['revision'],'files':[source_binding],'card':{'license':'source-pinned'}}
    source=tmp_path/'full-trajectories.jsonl.zst';values=[]
    for rid,trajectory in granted.miro_pair_trajectories(receipt,entry['catalog_file']['path']+':0',original):
        values.append({'source_row':rid,'source_file':source_binding,'source_revision':entry['revision'],
                       'trajectory':asdict(trajectory),'positive_candidate':rid.endswith(':chosen')})
    with write_jsonl_text(source) as stream:
        for value in values:stream.write(json.dumps(value)+'\n')
    export=tmp_path/'export.json';write_json_durable(export,{'schema':'natlang.normalized-agent-source-file/1',
        'contract':{'source':source_binding,'revision':entry['revision']},'complete_sessions':2,
        'outputs':{source.name:sha256_file_hex(source)}})
    protection=tmp_path/'protection.json';write_json_durable(protection,{'normalized_export_bindings':[
        {'sha256':sha256_file_hex(export),'source_sha256':sha256_file_hex(source)}]})
    root=tmp_path/'runtime';(root/'training/decision-holds').mkdir(parents=True)
    hold=root/'training/decision-holds/held.jsonl';hold.write_text('')
    runtime=builder._runtime()
    monkeypatch.setattr(builder,'ROOT',root)
    monkeypatch.setattr(builder,'_runtime',lambda:runtime)
    monkeypatch.setattr(builder,'current_protection',lambda path:{'held':set(),'quality':set(),'pending':set(),'proof':{'sha256':'c'*64}})
    monkeypatch.setattr(builder,'verify_global_source_closure',lambda *a:None)
    monkeypatch.setattr(builder,'load_decision_holds',lambda paths:(set(),[]))
    prepared=prepare_tokenizer(base)
    report=builder.main(['--export-receipt',str(export),'--base-tokenizer',str(base),'--protection',str(protection),
        '--decision-hold',str(hold),'--output',str(tmp_path/'native')],prepared=prepared)
    graphs=[row for split in ('train','test') for row in sessions(tmp_path/'native'/split/'native-graphs.jsonl.zst')]
    assert len(graphs)==1
    _,record,error=next(normalized_preference_records(source));assert error is None
    expected,spec=pair_graph(record,prepared.tokenizer,tokenizer_fingerprint=prepared.expanded_fingerprint)
    assert graphs[0]==expected
    objective=json.loads(Path(report['objective']['path']).read_text())
    assert objective['rows'][expected['id']]==spec
    assert spec['token_counts']['chosen']>0 and spec['token_counts']['rejected']>0
    from scripts.neuralese_data import prepare_acquired_preference_admission as admission
    for name in ('ROOT','current_protection','verify_global_source_closure','load_decision_holds','_runtime'):
        monkeypatch.setattr(admission,name,getattr(builder,name))
    verified=admission.verify_conversion(tmp_path/'native',protection_path=protection,base=base)
    assert verified['source_render_verified'] and verified['exact_original_alternative_renders']==1


def test_admission_rejects_coordinated_token_objective_and_row_proof_forgery(tmp_path,monkeypatch):
    from natlang_neuralese.common.hashing import canonical_json_sha256_hex as digest
    from natlang_neuralese.data.native_graph_schema import parse_graph
    from natlang_neuralese.rl.native_replay import graph_identity
    from natlang_neuralese.train.native_preference import NativePreferenceObjective
    from scripts.neuralese_data import prepare_acquired_preference_admission as admission
    args,_,_=setup_conversion(tmp_path,monkeypatch)
    result=builder.main(args)
    for name in ('ROOT','current_protection','verify_global_source_closure','load_decision_holds','_runtime'):
        monkeypatch.setattr(admission,name,getattr(builder,name))
    directory=tmp_path/'output'
    source_sha=sha256_file_hex(tmp_path/'full-trajectories.jsonl.zst')
    graph=next(sessions(directory/'train/native-graphs.jsonl.zst'))
    old_id=graph['id']
    objective=json.loads((directory/'objective.json').read_text())
    spec=objective['rows'][old_id]
    node=graph['nodes'][0]
    node['token_ids'][-1]=1 if node['token_ids'][-1]!=1 else 2
    spec['suffixes'][node['name']]['token_ids_sha256']=digest(node['token_ids'])
    graph['id']='offline-preference:'+digest({'source_record_sha256':spec['source_record_sha256'],
        'suffixes':spec['suffixes'],'token_counts':spec['token_counts']})
    spec['graph_sha256']=digest(graph_identity(parse_graph(graph,tokenizer_fingerprint=graph['tokenizer_fingerprint'])))
    def rewrite(directory):
        graphs=directory/'train/native-graphs.jsonl.zst'
        rows=list(sessions(graphs))
        if not any(row['id']==old_id for row in rows):return
        with write_jsonl_text(graphs) as stream:
            for row in rows:stream.write(json.dumps(graph if row['id']==old_id else row)+'\n')
        path=directory/'objective.json';value=json.loads(path.read_text())
        del value['rows'][old_id];value['rows'][graph['id']]=spec;write_json_durable(path,value)
        path=directory/'row-proof.jsonl';proofs=list(sessions(path))
        for proof in proofs:
            if proof['id']==old_id:proof.update(id=graph['id'],graph_sha256=digest(graph))
        path.write_text(''.join(json.dumps(proof)+'\n' for proof in proofs))
    rewrite(directory)
    for shard in result['shards']:
        path=Path(shard['path']);rewrite(path.parent)
        value=json.loads(path.read_text())
        for output in value['outputs']:
            bound=builder._bound(path.parent/output['path'])
            output.update({key:bound[key] for key in ('sha256','bytes')})
        write_json_durable(path,value)
        shard['sha256']=sha256_file_hex(path)
    for output in result['outputs']:
        bound=builder._bound(output['path']);output.update({key:bound[key] for key in ('sha256','bytes')})
    for key in ('objective','row_proof'):result[key]=builder._bound(result[key]['path'])
    write_json_durable(directory/'receipt.json',result)
    # All altered files, immutable IDs, suffix token hashes and internal masks
    # agree; only the original source still establishes that these are false.
    import torch
    with torch.no_grad():
        NativePreferenceObjective(directory/'objective.json').forward_inputs({'examples':[
            parse_graph(graph,tokenizer_fingerprint=graph['tokenizer_fingerprint'])]})
    assert sha256_file_hex(tmp_path/'full-trajectories.jsonl.zst')==source_sha
    with pytest.raises(ValueError,match='exact original-alternative tokenizer rendering'):
        admission.verify_conversion(directory,protection_path=tmp_path/'protection.json',base=tmp_path/'base')
