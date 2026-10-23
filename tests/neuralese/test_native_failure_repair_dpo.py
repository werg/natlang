"""Verified whole-graph failure repair admission and genuine reference DPO."""
import copy
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from natlang_neuralese.common.hashing import canonical_json_sha256_hex
from natlang_neuralese.data import native_failure_repair as repair
from natlang_neuralese.data.native_failure_repair import (_merge_graphs, _write_admitted_rows,
    _collector_actor_binding, PAIR_SCHEMA)
from natlang_neuralese.data.native_graph_schema import SCHEMA, parse_graph
from natlang_neuralese.rl.native_replay import graph_identity
from natlang_neuralese.train.native_preference import NativePreferenceObjective
from natlang_neuralese.train.native_state_sft import packed_graph_loss


def record(task_sha='a'*64):
    original={'task_sha256':task_sha,'world_sha256':'b'*64,'oracle_sha256':'c'*64,
              'source_groups':['task-1'], 'prompt_sha256':'d'*64}
    alternatives={}
    for label, token in [('failed',10),('repaired',11)]:
        graph={'schema':SCHEMA,'id':'capture-'+label,'split':'train','source_groups':['task-1'],
            'tokenizer_fingerprint':'tok','nodes':[
              {'name':'writer','token_ids':[1,2,3,4],'target_weights':[0.,0.,0.,0.], 'producing':[[2,4]]},
              {'name':'reader','token_ids':[5,6,token,7],'target_weights':[0.,0.,0.,0.],
               'imports':[{'at':1,'source':'writer','start':2,'stop':4}]}]}
        actions=[{'node':'writer','position':2,'token_id':3,'kind':'sampled','reference_logp':-.4},
                 {'node':'writer','position':3,'token_id':4,'kind':'forced','reference_logp':-.3},
                 {'node':'reader','position':2,'token_id':token,'kind':'sampled','reference_logp':-.5}]
        ref={'checkpoint_sha256':'1'*64,'tokenizer_fingerprint':'tok',
             'runtime_sha256':'2'*64,'graph_action_sha256':canonical_json_sha256_hex(actions),
             'normalization':'token-mean-v1'}
        receipt={'schema':'natlang.native-fixed-reference-scores/1',**ref,'actions':actions}
        ref.update(receipt=receipt,receipt_sha256=canonical_json_sha256_hex(receipt))
        actor={'checkpoint_sha256':('e' if label=='failed' else 'a')*64,
               'runtime_sha256':('f' if label=='failed' else 'b')*64,'actor_id':f'student-{label}'}
        capture={'schema':'natlang.native-failure-execution-capture/1',
            'task_sha256':original['task_sha256'],'world_sha256':original['world_sha256'],
            'prompt_sha256':original['prompt_sha256'],'actions':[
                {key:value[key] for key in ('node','position','token_id','kind')} for value in actions],
            'actor':actor}
        capture_sha=canonical_json_sha256_hex(capture)
        replay={'schema':'natlang.native-graph-replay-receipt/1',
            'graph_sha256':canonical_json_sha256_hex(graph),'execution_capture_sha256':capture_sha,
            'occurrence_bindings_verified':True}
        alternatives[label]={'graph':graph,'task_sha256':original['task_sha256'],
            'world_sha256':original['world_sha256'],'oracle_sha256':original['oracle_sha256'],
            'source_groups':original['source_groups'],'prompt_sha256':original['prompt_sha256'],
            'prompt_view':'original-unhinted','hint_in_student_prompt':False,
            'execution_capture_sha256':capture_sha,'execution_receipt':capture,
            'graph_replay_sha256':canonical_json_sha256_hex(replay),'graph_replay_receipt':replay,
            'complete_graph':True,'occurrence_bindings_verified':True,
            'actor':actor,
            'reference_score':ref,'actions':actions}
    oracle_evidence={}
    for label in ('failed','repaired'):
        evidence={'outcome':'semantic_mismatch' if label=='failed' else 'task_success',
            'task_sha256':original['task_sha256'],'world_sha256':original['world_sha256'],
            'oracle_sha256':original['oracle_sha256'],'source_groups':original['source_groups'],
            'packet_sha256':'4'*64,'result_sha256':('5' if label=='failed' else '6')*64,
            'finish_sha256':'7'*64,'collection_binding_sha256':'9'*64}
        receipt={'schema':'natlang.task-oracle-evaluation/1',**evidence,
            'task_id':'task-1','evaluator_sha256':'8'*64}
        evidence['evaluation_receipt']=receipt
        evidence['evaluation_receipt_sha256']=canonical_json_sha256_hex(receipt)
        oracle_evidence[label]=evidence
    return {'schema':PAIR_SCHEMA,'split':'train','protected':False,'original_task':original,
            'oracle_evidence':oracle_evidence,
            'alternatives':alternatives}


class Tiny(nn.Module):
    def __init__(self):
        super().__init__();self.embedding=nn.Embedding(20,6);self.readout=nn.Linear(6,20,bias=False)
        self.port=SimpleNamespace(checkpoint_layers=False);self.calls=0
    def forward(self,plan,retain_states=False,**kwargs):
        self.calls+=1
        values=[];by_name={}
        for node,offset in zip(plan.invocations,plan.offsets):
            own=self.embedding(torch.tensor(node.token_ids));by_name[node.name]=own
            values.append(own)
        # Carry each imported producer span into consumer hidden state so the
        # test detects a detached/consumer-only implementation.
        for index,node in enumerate(plan.invocations):
            own=values[index]
            for edge in node.imports:
                source=by_name[edge.source][edge.start:edge.stop]
                updated=own.clone();updated[edge.at:edge.at+len(source)]=updated[edge.at:edge.at+len(source)]+source
                own=updated
            values[index]=own
        return {'h_final':torch.cat(values)[None]}
    def logits(self,states):return self.readout(states)


def test_admission_retains_complete_internal_graph_and_reference_dpo_updates_producer(tmp_path):
    pair=record();graph,specrow=_merge_graphs(pair,'tok')
    assert pair['alternatives']['failed']['actor']['checkpoint_sha256']!=\
        pair['alternatives']['repaired']['actor']['checkpoint_sha256']
    example=parse_graph(graph,tokenizer_fingerprint='tok')
    assert len(example.plan.invocations)==4
    assert any(edge.source=='failed/writer' for node in example.plan.invocations for edge in node.imports)
    assert any(edge.source=='repaired/writer' for node in example.plan.invocations for edge in node.imports)
    spec={'schema':'natlang.native-failure-repair-dpo-objective/1','kind':'native-failure-repair-dpo',
          'normalization':'token-mean-v1','beta':.2,'rows':{graph['id']:specrow}}
    path=tmp_path/'objective.json';path.write_text(json.dumps(spec));objective=NativePreferenceObjective(path)
    example.id=graph['id']
    model=Tiny()
    result=model(example.plan);reader=next(node for node in example.plan.invocations if node.name=='repaired/reader')
    reader_offset=example.plan.offsets[example.plan.invocations.index(reader)]
    from natlang_neuralese.rl.native_replay import action_logps
    consumer_logp,_=action_logps(model,result['h_final'][0],[{'query':reader_offset+1,'token_id':11,
        'temperature':1.,'allowed_token_ids':None,'banned_token_ids':None,'behavior_logp':0.}])
    producer_gradient=torch.autograd.grad(consumer_logp.sum(),model.embedding.weight,retain_graph=True)[0]
    assert producer_gradient[3:5].norm()>0
    loss,detail=packed_graph_loss(model,[example],objective=objective,checkpoint_layers=False)
    loss.backward()
    assert torch.isfinite(loss) and model.calls==2
    assert detail['preference_pairs']==1 and detail['scored_action_tokens']==6
    assert model.embedding.weight.grad is not None and model.embedding.weight.grad.norm()>0


@pytest.mark.parametrize('mutation', ['hint','held','oracle','world','prompt','missing_graph','source_protected'])
def test_admission_rejects_hint_leakage_or_protected_or_mismatched_capture(mutation):
    pair=record()
    if mutation=='hint':pair['alternatives']['repaired']['hint_in_student_prompt']=True
    elif mutation=='held':pair['split']='test'
    elif mutation=='oracle':pair['oracle_evidence']['failed']['outcome']='task_success'
    elif mutation=='world':pair['alternatives']['repaired']['world_sha256']='8'*64
    elif mutation=='prompt':pair['alternatives']['failed']['prompt_view']='hinted'
    else:pair['alternatives']['failed'].pop('graph')
    with pytest.raises(ValueError):_merge_graphs(pair,'tok',protected_groups={'task-1'} if mutation=='source_protected' else set())


def test_reference_custody_and_source_dynamic_append(tmp_path):
    first=record();second=record('7'*64)
    one=_write_admitted_rows([first],tmp_path/'first',tokenizer_fingerprint='tok',protected_groups=set())
    two=_write_admitted_rows([first,second],tmp_path/'second',tokenizer_fingerprint='tok',protected_groups=set())
    spec=json.loads((tmp_path/'first'/'repair-objective.json').read_text())
    appended=json.loads((tmp_path/'second'/'repair-objective.json').read_text())
    from natlang_neuralese.train.native_state_sft import append_only_rows
    old={key:canonical_json_sha256_hex(row) for key,row in spec['rows'].items()}
    new={key:canonical_json_sha256_hex(row) for key,row in appended['rows'].items()}
    assert len(old)==one['pairs']==1 and len(new)==two['pairs']==2
    assert len(append_only_rows(old,new,label='test objective'))==1
    changed=copy.deepcopy(new);changed[next(iter(old))]='0'*64
    with pytest.raises(ValueError,match='modified/removed'):
        append_only_rows(old,changed,label='test objective')
    forged=record();forged['alternatives']['failed']['reference_score']['checkpoint_sha256']='z'*64
    with pytest.raises(ValueError,match='fixed reference score receipt'):
        _merge_graphs(forged,'tok')


def test_batch_builder_loads_fixed_reference_once_and_preserves_prior_rows(tmp_path,monkeypatch):
    held=tmp_path/'held.jsonl';held.write_text('{}\n')
    reference=tmp_path/'reference.pt';reference.write_bytes(b'reference')
    parent=tmp_path/'parent.pt';parent.write_bytes(b'parent')
    prior=tmp_path/'prior'
    _write_admitted_rows([record()],prior,tokenizer_fingerprint='tok',protected_groups=set())
    output=tmp_path/'batch';loads=[]
    model=SimpleNamespace(eval=lambda:None);roles=SimpleNamespace(ids=[1,2,3,4])
    monkeypatch.setattr('natlang_neuralese.model.native_model.load_native_checkpoint',
        lambda *a,**k:(loads.append(1) or (model,object(),roles,{'tokenizer_fingerprint':'tok'},None)))
    monkeypatch.setattr('natlang_neuralese.data.text_corpus.tokenizer_fingerprint',lambda _t:'tok')
    monkeypatch.setattr('natlang_neuralese.data.native_sources.NativeGraphSource',
        lambda *a,**k:SimpleNamespace(groups=[]))
    monkeypatch.setattr('natlang_neuralese.model.native_model.native_identity',
        lambda *a,**k:{'checkpoint_sha256':'f'*64})
    monkeypatch.setattr('scripts.source_review.pending_unpinned_source_groups',lambda:set())
    def fake_pair(**kwargs):
        tasksha=kwargs['task_id'].ljust(64,'0')[:64]
        _write_admitted_rows([record(tasksha)],kwargs['output_dir'],tokenizer_fingerprint='tok',protected_groups=set())
        return {'graphs':str(kwargs['output_dir']/'repair-graphs.jsonl'),
            'objective':str(kwargs['output_dir']/'repair-objective.json')}
    monkeypatch.setattr(repair,'admit_run_pair',fake_pair)
    result=repair.admit_run_pairs(pairs=[{'failed_run_dir':'f','repaired_run_dir':'r','task_id':'e'*64,
        'failed_sample':0,'repaired_sample':1}],reference_checkpoint=reference,base=tmp_path,
        held_native_graphs=held,output_dir=output,continue_from=parent,stage_id='repair-dpo',
        prior_admission_dir=prior)
    graph_rows=[json.loads(line) for line in (output/'repair-graphs.jsonl').read_text().splitlines()]
    prior_rows=[json.loads(line) for line in (prior/'repair-graphs.jsonl').read_text().splitlines()]
    objective=json.loads((output/'repair-objective.json').read_text())
    assert len(loads)==1 and result['new_pairs']==1 and result['pairs']==2
    assert graph_rows[0]==prior_rows[0]
    assert graph_rows[0]['id'] in objective['rows'] and graph_rows[1]['id'] in objective['rows']


def test_recipe_stage_addition_and_sft_to_dpo_continuation(tmp_path):
    held=tmp_path/'held.jsonl';held.write_text('{}\n')
    parent=tmp_path/'parent.pt';parent.write_bytes(b'parent checkpoint')
    out=tmp_path/'admission'
    result=_write_admitted_rows([record()],out,tokenizer_fingerprint='tok',protected_groups=set(),
        stage_inputs={'held_native_graphs':held,'continue_from':parent,'stage_id':'failure-repair-dpo',
            'requires':['native-offline-preference'],'beta':.1})
    addition=json.loads((out/'recipe-stage-addition.json').read_text())
    assert result['recipe_stage_addition'].endswith('recipe-stage-addition.json')
    assert addition['stage']['kind']=='native_offline_preference'
    assert addition['stage']['inputs'].keys()=={'native_graphs','objective_spec','held_native_graphs','continue_from'}
    assert addition['continuation']['weights_optimizer_rng_exposure']=='preserved'
    from natlang_neuralese.train.native_state_sft import verify_failure_repair_objective_continuation
    graph,row=_merge_graphs(record(),'tok')
    spec={'schema':'natlang.native-failure-repair-dpo-objective/1','kind':'native-failure-repair-dpo',
        'normalization':'token-mean-v1','beta':.1,'rows':{graph['id']:row}}
    objective=NativePreferenceObjective.__new__(NativePreferenceObjective)
    objective.mode='failure_repair_dpo';objective.kind='native-failure-repair-dpo';objective._spec=spec
    # A normal SFT parent has no DPO row bindings; the first DPO stage starts cleanly.
    assert verify_failure_repair_objective_continuation({'objective_kind':'native-sft'},
        {'stage_id':'sft'},objective,'failure-repair-dpo') is False
    objective.row_bindings=lambda:{graph['id']:canonical_json_sha256_hex(row)}
    current_binding=canonical_json_sha256_hex(row)
    assert verify_failure_repair_objective_continuation({'objective_kind':'native-failure-repair-dpo'},
        {'stage_id':'failure-repair-dpo','failure_repair_objective_row_bindings':{graph['id']:current_binding}},
        objective,'failure-repair-dpo') is True
    with pytest.raises(ValueError,match='lacks exact prior'):
        verify_failure_repair_objective_continuation({'objective_kind':'native-failure-repair-dpo'},
            {'stage_id':'failure-repair-dpo'},objective,'failure-repair-dpo')


def test_objective_loader_rechecks_reference_receipt_custody(tmp_path):
    graph,row=_merge_graphs(record(),'tok')
    spec={'schema':'natlang.native-failure-repair-dpo-objective/1','kind':'native-failure-repair-dpo',
        'normalization':'token-mean-v1','beta':.1,'rows':{graph['id']:row}}
    spec['rows'][graph['id']]['reference']['failed']['receipt']['checkpoint_sha256']='9'*64
    path=tmp_path/'forged-objective.json';path.write_text(json.dumps(spec))
    with pytest.raises(ValueError,match='reference score receipt'):
        NativePreferenceObjective(path)


def test_actual_collector_binding_keeps_python_and_typescript_runtime_inventories_distinct(tmp_path):
    path=tmp_path/'collection-binding.json'
    checkpoint='1'*64;identity={'architecture':'lfm2','checkpoint_sha256':checkpoint,
        'embedding_weight_sha256':'2'*64,'tokenizer_fingerprint':'3'*64}
    runtime={'schema':'natlang.python-runtime/1','files':337,'sha256':'4'*64}
    path.write_text(json.dumps({'schema':'natlang.rl-collection-binding/1','policy_id':checkpoint,
        'expected_native_actor':{'native_identity':identity,'python_runtime':runtime},
        'collector_sha256':'6'*64,'runtime':{'dist/runtime.js':'5'*64}}))
    got_path,got_identity,got_runtime,collector_runtime=_collector_actor_binding(tmp_path,{'policy_checkpoint_sha256':checkpoint})
    assert got_path==path and got_identity==identity and got_runtime==runtime['sha256']
    assert collector_runtime==canonical_json_sha256_hex({'dist/runtime.js':'5'*64})
    value=json.loads(path.read_text());value['expected_native_actor']['native_identity']['checkpoint_sha256']='6'*64
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='collector binding'):
        _collector_actor_binding(tmp_path,{'policy_checkpoint_sha256':checkpoint})


def test_actual_collector_binding_rejects_malformed_runtime_inventories(tmp_path):
    path=tmp_path/'collection-binding.json';checkpoint='1'*64
    value={'schema':'natlang.rl-collection-binding/1','policy_id':checkpoint,
        'expected_native_actor':{'native_identity':{'architecture':'lfm2','checkpoint_sha256':checkpoint,
            'embedding_weight_sha256':'2'*64,'tokenizer_fingerprint':'3'*64},
            'python_runtime':{'schema':'natlang.python-runtime/1','files':337,'sha256':'4'*64}},
        'collector_sha256':'5'*64,'runtime':{'../escape.js':'6'*64}}
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='collector binding'):
        _collector_actor_binding(tmp_path,{'policy_checkpoint_sha256':checkpoint})


def test_materialized_qwen_capture_uses_exact_live_actor_and_normal_collector_layout(tmp_path,monkeypatch):
    from natlang_neuralese.data.native_failure_repair import materialize_qwen_repair_run
    failed=tmp_path/'failed';failed.mkdir();checkpoint='1'*64
    identity={'architecture':'lfm2','checkpoint_sha256':checkpoint,'embedding_weight_sha256':'2'*64,
        'tokenizer_fingerprint':'3'*64}
    runtime={'schema':'natlang.python-runtime/1','files':3,'sha256':'4'*64}
    task={'id':'task-1','split':'train','source_groups':['family-1'],'program_ir':{'id':'program-1'}}
    packet=(json.dumps(task)+'\n').encode();(failed/'packet.jsonl').write_bytes(packet)
    from natlang_neuralese.common.hashing import sha256_file_hex
    (failed/'preparation.json').write_text(json.dumps({'schema':'natlang.task-rollout-preparation/1',
        'packet_sha256':sha256_file_hex(failed/'packet.jsonl'),'policy_checkpoint_sha256':checkpoint}))
    (failed/'collection-binding.json').write_text(json.dumps({'schema':'natlang.rl-collection-binding/1',
        'policy_id':checkpoint,'endpoints':['http://student.test'],'expected_native_actor':{
            'native_identity':identity,'python_runtime':runtime},'collector_sha256':'5'*64,
        'runtime':{'dist/collector.js':'6'*64}}))
    old_results=failed/'old.jsonl';old_results.write_text('{}\n')
    (failed/'queue.jsonl').write_text(json.dumps({'index':0,'sample':0,'group':'family-1',
        'key':'failed-key','output':str(old_results)})+'\n')
    failed_rollout={'schema':'natlang.rollout/1','task_id':'task-1','split':'train','group':'family-1','sample':0,
        'policy':{'checkpoint_sha256':checkpoint},'turns':[],'execution_graph':[],'outcome':'semantic_mismatch',
        'reward':{'quality':0.}}
    monkeypatch.setattr(repair,'_run_member',lambda *_args:(task,failed_rollout,{'outcome':'semantic_mismatch'}))
    live_binding={'native_identity':identity,'python_runtime':runtime}
    class Response:
        def __enter__(self):return self
        def __exit__(self,*_args):return None
        def read(self):return json.dumps({'rl_policy_binding':live_binding}).encode()
    monkeypatch.setattr('urllib.request.urlopen',lambda *_args,**_kwargs:Response())
    def save(_rollouts,out,**kwargs):
        path=out/'manifest.json';path.write_text(json.dumps({'schema':'natlang.rl-block-bundle/1',
            'policy_id':kwargs['policy_id'],'blocks':{},**kwargs['manifest_fields']}));return json.loads(path.read_text())
    monkeypatch.setattr('natlang_neuralese.rl.graph_replay.save_bundle',save)
    messages=[{'role':'system','content':'original prompt'}];tools=[]
    prompt_sha=canonical_json_sha256_hex({'messages':messages,'tools':tools})
    capture={'schema':'natlang.qwen-teacher-repair-execution/1','run_id':'fresh-replay',
        'target_actor':{'endpoint':'http://student.test','native_identity':identity,'python_runtime':runtime,
            'binding_sha256':canonical_json_sha256_hex(live_binding)},
        'row':{'task':{'program_ir':task['program_ir']}},'run':{'outcome':{'status':'done','accepted':True,
            'execution_graph':[{'kind':'invocation','phase':'start'}]}},'trajectory':[{'phase':'action',
                'context':messages,'tools_offered':tools,'model_response':{'calls':[],'text':'done','raw_calls':[],
                    'rl_sampling':{'schema':'natlang.native-teacher-forced-capture/1',
                        'replay_prompt_sha256':prompt_sha,'policy':identity}},'assistant':{'content':'done','calls':[]}}]}
    capture_path=tmp_path/'capture.json';capture_path.write_text(json.dumps(capture))
    result=materialize_qwen_repair_run(failed_run_dir=failed,task_id='task-1',failed_sample=0,
        repaired_sample=1,capture_path=capture_path,output_dir=tmp_path/'repaired',endpoint='http://student.test')
    assert result['outcome']=='task_success' and result['block_bundle'].endswith('blocks/manifest.json')
    materialized=tmp_path/'repaired';rows=[json.loads(line) for line in (materialized/'results/repaired.jsonl').read_text().splitlines()]
    assert rows[0]['task']=={'program_ir':task['program_ir']} and rows[0]['trajectory'][0]['model_response']['rl_sampling']['schema']==\
        'natlang.native-teacher-forced-capture/1'
    assert json.loads((materialized/'queue.jsonl').read_text().splitlines()[0])['sample']==1


def test_zero_result_model_budget_failure_uses_existing_complete_prefix_verifier(tmp_path,monkeypatch):
    from natlang_neuralese.common.hashing import sha256_file_hex
    run=tmp_path/'partial';run.mkdir();checkpoint='1'*64
    identity={'architecture':'lfm2','checkpoint_sha256':checkpoint,'embedding_weight_sha256':'2'*64,
        'tokenizer_fingerprint':'3'*64}
    runtime={'schema':'natlang.python-runtime/1','files':1,'sha256':'4'*64}
    task={'id':'partial-task','split':'train','source_groups':['source-1'],'program_ir':{'id':'program-1'}}
    packet=run/'packet.jsonl';packet.write_text(json.dumps(task)+'\n')
    packet_sha=sha256_file_hex(packet)
    (run/'preparation.json').write_text(json.dumps({'schema':'natlang.task-rollout-preparation/1',
        'packet_sha256':packet_sha,'policy_checkpoint_sha256':checkpoint}))
    (run/'collection-binding.json').write_text(json.dumps({'schema':'natlang.rl-collection-binding/1',
        'policy_id':checkpoint,'expected_native_actor':{'native_identity':identity,'python_runtime':runtime},
        'collector_sha256':'5'*64,'runtime':{'dist/collector.js':'6'*64}}))
    missing=run/'result.jsonl';entry={'index':0,'sample':0,'group':'source-1','key':'job-key','output':str(missing)}
    (run/'queue.jsonl').write_text(json.dumps(entry)+'\n')
    finish={'event':'finish','key':'job-key','status':'complete','resource_limit_reason':'model_request_budget',
        'output_accounting':{'job_states':[{'index':0,'state':'failed','program_id':'partial-task',
            'resource_limit_reason':'model_request_budget'}]}}
    (run/'journal.jsonl').write_text(json.dumps(finish)+'\n')
    context=[{'role':'system','content':'exact prompt'}]
    frame={'phase':'action','invocation_id':'run/execution-1','context':context,'tools_offered':[],
        'model_response':{'calls':[],'raw_calls':[],'text':'','rl_sampling':{'schema':'natlang.native-rl-sampling/2',
            'policy':identity,'samples':[{'token_id':9,'forced':False,'logp':-.1,'input_role':'ordinary',
                'terminal':False,'query_own_position':1}]}},'assistant':{'content':'','calls':[]}}
    prefix={'turns':[{'messages':context,'tools':None,'target':{'role':'assistant','content':''},
        'invocation_id':'run/execution-1','sampling':frame['model_response']['rl_sampling']}],
        'trajectory':[frame],'execution_graph':[{'kind':'invocation','phase':'start','call_id':'run/execution-1'}],
        'evidence':{'path':'partial.jsonl','sha256':'7'*64,'attempt_id':'attempt-1',
            'failure_reason':'whole-case model request budget exceeded (2)'}}
    calls=[]
    def recover(entry_arg,task_arg,finish_arg,*,packet_source_path,packet_source_sha256):
        calls.append((entry_arg,task_arg,finish_arg,packet_source_path,packet_source_sha256));return prefix
    monkeypatch.setattr('natlang_neuralese.rl.task_rollouts._budget_terminated_prefix',recover)
    _,rollout,evidence=repair._run_member(run,'partial-task',0)
    assert len(calls)==1 and calls[0][-1]==packet_sha
    assert rollout['outcome']=='budget_exhaustion' and rollout['reward']['quality']==0.
    assert rollout['execution_graph']==prefix['execution_graph'] and evidence['partial_evidence']==prefix['evidence']
    assert not missing.exists()


def test_teacher_forced_capture_replays_actual_tokens_without_behavior_logps():
    from natlang_neuralese.rl.native_replay import _input_stream
    capture={'schema':'natlang.native-teacher-forced-capture/1','samples':[
        {'index':0,'token_id':4,'logp':None,'forced':True,'input_role':'ordinary','terminal':False,'query_own_position':3},
        {'index':1,'token_id':64,'logp':None,'forced':True,'input_role':'ordinary','terminal':False,'query_own_position':4},
        {'index':2,'token_id':8,'logp':None,'forced':True,'input_role':'produce','terminal':False,'query_own_position':5},
        {'index':3,'token_id':65,'logp':None,'forced':True,'input_role':'ordinary','terminal':False,'query_own_position':6}],
        'generated_inputs':[{'token_id':value['token_id'],'input_role':value['input_role'],'sample_index':index}
            for index,value in enumerate([
                {'token_id':4,'input_role':'ordinary'},{'token_id':64,'input_role':'ordinary'},
                {'token_id':8,'input_role':'produce'},{'token_id':65,'input_role':'ordinary'}])],
        'writes':[{'begin_sample':1,'end_sample':4,'payload_begin':2,'payload_end':3,
            'own_begin':5,'own_end':8,'payload_own_begin':6,'payload_own_end':7}]}
    stream,writes=_input_stream(capture,4)
    assert len(stream)==4 and all(item['forced'] and item['source_sample'] is None for item in stream)
    assert all(item['logp'] is None for item in stream) and writes==capture['writes']
