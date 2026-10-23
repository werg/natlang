"""Verified whole-graph failure repair admission and genuine reference DPO."""
import copy
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from natlang_neuralese.common.hashing import canonical_json_sha256_hex
from natlang_neuralese.data.native_failure_repair import _merge_graphs, _write_admitted_rows, PAIR_SCHEMA
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
        actor={'checkpoint_sha256':'e'*64,'runtime_sha256':'f'*64,'actor_id':'student'}
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
            'finish_sha256':'7'*64}
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
