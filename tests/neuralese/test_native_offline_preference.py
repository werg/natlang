"""Offline pair masks, analytical gradients and one actual native graph forward."""
import copy
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from natlang_neuralese.common.hashing import canonical_json_sha256_hex
from natlang_neuralese.data.native_graph_schema import SCHEMA,parse_graph
from natlang_neuralese.rl.native_replay import graph_identity
from natlang_neuralese.train.native_preference import NativePreferenceObjective,SCHEMA as OBJECTIVE_SCHEMA,KIND
from natlang_neuralese.train.native_state_sft import packed_graph_loss


def example_spec(tmp_path,*,split='train',fingerprint='tokenizer',source_sha='a'*64):
    row={'schema':SCHEMA,'id':'offline-pair','split':split,'source_groups':['exact-task'],
         'tokenizer_fingerprint':fingerprint,'nodes':[
             {'name':'chosen:2','token_ids':[1,2,3,4],'target_weights':[0.]*4},
             {'name':'rejected:2','token_ids':[1,2,5,6,7],'target_weights':[0.]*5}]}
    example=parse_graph(row,tokenizer_fingerprint=fingerprint)
    objective={'graph_sha256':canonical_json_sha256_hex(graph_identity(example)),
        'split':split,'suffixes':{},'token_counts':{'chosen':2,'rejected':3}}
    for node in example.plan.invocations:
        objective['suffixes'][node.name]={'alternative':node.name.split(':')[0],'source_message_index':2,
            'suffix_start':2,'suffix_stop':len(node.token_ids),
            'token_ids_sha256':canonical_json_sha256_hex(list(node.token_ids))}
    objective['source_record_sha256']=source_sha
    row['id']='offline-preference:'+canonical_json_sha256_hex({key:objective[key] for key in ['source_record_sha256','suffixes','token_counts']})
    example=parse_graph(row,tokenizer_fingerprint=fingerprint)
    objective['graph_sha256']=canonical_json_sha256_hex(graph_identity(example))
    spec={'schema':OBJECTIVE_SCHEMA,'kind':KIND,'beta':2.3,'margin':.4,'rows':{example.id:objective},'chunk_tokens':2}
    path=tmp_path/'objective.json';path.write_text(json.dumps(spec))
    return example,NativePreferenceObjective(path)


class TinyGraphModel(nn.Module):
    def __init__(self):
        super().__init__();self.embedding=nn.Embedding(16,7);self.readout=nn.Linear(7,16,bias=False)
        self.port=SimpleNamespace(checkpoint_layers=False);self.calls=0
    def forward(self,plan,retain_states=False,**kwargs):
        self.calls+=1
        return {'h_final':self.embedding(torch.tensor([token for node in plan.invocations for token in node.token_ids]))[None]}
    def logits(self,states):return self.readout(states)


def manual(model,example,objective):
    values={'chosen':[],'rejected':[]}
    for node in example.plan.invocations:
        ids=torch.tensor(node.token_ids);logps=model.logits(model.embedding(ids)).float().log_softmax(-1)
        suffix=objective.rows[example.id]['suffixes'][node.name]
        values[suffix['alternative']].extend(logps[position-1,ids[position]] for position in range(2,len(ids)))
    difference=torch.stack(values['chosen']).mean()-torch.stack(values['rejected']).mean()
    return -F.logsigmoid(objective.beta*difference-objective.margin)


def test_exact_pairwise_loss_and_all_parameter_gradients_match_analytical_oracle(tmp_path):
    torch.manual_seed(42);model=TinyGraphModel();reference=copy.deepcopy(model)
    example,objective=example_spec(tmp_path)
    actual,detail=packed_graph_loss(model,[example],objective=objective,checkpoint_layers=False)
    expected=manual(reference,example,objective)
    torch.testing.assert_close(actual,expected,rtol=0,atol=1e-7)
    actual.backward();expected.backward()
    for left,right in zip(model.parameters(),reference.parameters()):
        torch.testing.assert_close(left.grad,right.grad,rtol=2e-6,atol=1e-7)
    assert model.calls==1 and detail['preference_pairs']==1 and detail['scored_assistant_tokens']==5
    assert model.embedding.weight.grad[0].count_nonzero()==0


def test_ordinary_sft_cannot_train_rejected_alternative_and_forged_masks_fail_before_forward(tmp_path):
    example,objective=example_spec(tmp_path);model=TinyGraphModel()
    with pytest.raises(ValueError,match='declared offline/policy objective'):
        packed_graph_loss(model,[example],checkpoint_layers=False)
    assert model.calls==0
    objective.rows[example.id]['suffixes']['chosen:2']['suffix_start']=1
    with pytest.raises(ValueError,match='immutable graph identity'):
        packed_graph_loss(model,[example],objective=objective,checkpoint_layers=False)
    assert model.calls==0


def test_held_pair_cannot_receive_training_gradient(tmp_path):
    example,objective=example_spec(tmp_path,split='test');model=TinyGraphModel()
    with pytest.raises(ValueError,match='protected offline preference'):
        packed_graph_loss(model,[example],objective=objective,checkpoint_layers=False)
    assert model.calls==0
    with torch.no_grad():loss,_=packed_graph_loss(model,[example],objective=objective,checkpoint_layers=False)
    assert torch.isfinite(loss) and model.calls==1


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_both_actual_ports_one_forward_matches_cached_independent_history_loss_and_gradients(tmp_path,family):
    from test_shared_state import tiny
    from natlang_neuralese.model.shared_state import SharedStateModel
    example,objective=example_spec(tmp_path)
    model=SharedStateModel(tiny(family),(58,59,60,61),rank=4)
    model.port.checkpoint_layers=False
    if family=='mellum':model.port.hf.natlang_precision='bf16'
    from natlang_neuralese.train.native_state_sft import make_native_optimizer
    _,named=make_native_optimizer(model,lr=1e-3,in_backward=False)
    forwards=[];hook=model.register_forward_hook(lambda *args:forwards.append(1))
    actual,_=packed_graph_loss(model,[example],objective=objective,checkpoint_layers=False)
    parameters=tuple(parameter for _,parameter in named if parameter.requires_grad)
    gradients=torch.autograd.grad(actual,parameters,allow_unused=True)
    assert forwards==[1];hook.remove()
    reference=model(example.plan,reference=True)
    context={'examples':[example],'row_offsets':{example.id:{'physical_start':0}}}
    expected,_=objective(model,reference,context)
    expected_gradients=torch.autograd.grad(expected,parameters,allow_unused=True)
    torch.testing.assert_close(actual,expected,rtol=2e-5,atol=3e-6)
    for left,right in zip(gradients,expected_gradients):
        assert (left is None)==(right is None)
        if left is not None:torch.testing.assert_close(left,right,rtol=3e-5,atol=3e-6)
    assert any(value is not None and value.norm()>0 for value in gradients)


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_pair_capacity_executes_actual_joint_objective_without_updates_or_rng_changes(tmp_path,family):
    from test_shared_state import tiny
    from test_native_optimizer_checkpointing import assert_exact
    from natlang_neuralese.model.shared_state import SharedStateModel
    from natlang_neuralese.train.native_state_sft import make_native_optimizer
    from natlang_neuralese.eval.native_memory import qualify_capacity,offload_numerical_parity
    example,objective=example_spec(tmp_path)
    model=SharedStateModel(tiny(family),(58,59,60,61),rank=4)
    if family=='mellum':model.port.hf.natlang_precision='bf16'
    optimizer,_=make_native_optimizer(model,lr=3e-6,in_backward=True)
    before=copy.deepcopy(model.state_dict());backbone=copy.deepcopy(model.port.hf.state_dict())
    state=copy.deepcopy(optimizer.state_dict());rng=torch.get_rng_state().clone()
    report=qualify_capacity(model,optimizer,[example],objective=objective,
        checkpoint_sha256='b'*64,source_binding={'path':'paired-fixture','sha256':'c'*64})
    assert report['passed'] and report['optimizer_updates']==0
    assert report['objective']==objective.binding()
    assert report['cases'][0]['forward_tokens']==sum(len(n.token_ids) for n in example.plan.invocations)
    assert_exact(before,model.state_dict());assert_exact(backbone,model.port.hf.state_dict())
    assert_exact(state,optimizer.state_dict());assert torch.equal(rng,torch.get_rng_state())
    parity=offload_numerical_parity(model,optimizer,example,objective=objective,
        checkpoint_sha256='b'*64,source_binding={})
    assert parity['passed'] and parity['gradient_parameters']>0


def test_actual_shared_trainer_stop_resume_tweak_and_pair_exposure(tmp_path,monkeypatch):
    from test_native_model import tiny_source
    from test_native_state_sft import graph
    from natlang_neuralese.model.native_model import load_native_model
    from natlang_neuralese.train import native_state_sft as trainer
    from natlang_neuralese.train.native_preference import main
    base=tmp_path/'base';tiny_source(base)
    _,_,_,config=load_native_model(base,dtype=torch.float32,rank=4)
    fingerprint=config['tokenizer_fingerprint']
    example,objective=example_spec(tmp_path,fingerprint=fingerprint)
    row={'schema':SCHEMA,'id':example.id,'split':'train','source_groups':['exact-task'],
         'tokenizer_fingerprint':fingerprint,'nodes':[
             {'name':node.name,'token_ids':list(node.token_ids),'target_weights':[0.]*len(node.token_ids)}
             for node in example.plan.invocations]}
    train=tmp_path/'train.jsonl';train.write_text(json.dumps(row)+'\n')
    held=tmp_path/'held.jsonl';held.write_text(json.dumps(dict(graph('held'),split='test',
        source_groups=['held-task'],tokenizer_fingerprint=fingerprint))+'\n')
    out=tmp_path/'run'
    args=['--base',str(base),'--dtype','float32','--rank','4','--device','cpu',
        '--native-graphs',str(train),'--held-native-graphs',str(held),'--objective-spec',str(objective.path),
        '--epochs','3','--eval-every','100','--out',str(out)]
    def stop(policy,step,**kwargs):policy.stop.requested=True;return False
    with monkeypatch.context() as patch:
        patch.setattr(trainer.CheckpointPolicy,'due',stop)
        assert main(args)==0
    first=torch.load(out/'checkpoint.pt',weights_only=False)
    assert first['training']['step']==1 and first['training']['exposure_counts']=={example.id:1}
    assert first['training']['identity']['objective_kind']==KIND
    assert first['training']['optimizer'] and first['training']['rng']
    assert main(args+['--preference-beta','1.7'])==0
    last=torch.load(out/'checkpoint.pt',weights_only=False)
    assert last['training']['step']==3 and last['training']['exposure_counts']=={example.id:3}
    assert last['training']['identity']['objective']['beta']==1.7
    assert last['training']['optimizer']
    assert json.loads((out/'report.json').read_text())['completed'] is True


def test_recipe_preference_stage_reuses_native_defaults_and_explicit_objective_options():
    from natlang_neuralese.train import recipe
    stage={'kind':'native_offline_preference','parameters':{'preference_beta':.5,'preference_margin':.2},'requires':['foundation']}
    config={'stage_parameter_defaults':{'native_state_sft':{'learning_rate':3e-6,'fp8_recipe':'current-bf16-dx',
        'communication_foundation':True,'share_causal_prefixes':True}}}
    values=recipe.effective_stage_parameters(stage,[],config)
    assert values=={'learning_rate':3e-6,'fp8_recipe':'current-bf16-dx','preference_beta':.5,'preference_margin':.2}
    assert recipe.HANDLERS[stage['kind']]['module']=='natlang_neuralese.train.native_preference'
    assert 'objective_spec' in recipe.HANDLERS[stage['kind']]['required_inputs']
    assert stage['kind'] in recipe.NATIVE_TRAINING_KINDS
    args=recipe.stage_parameter_args(values)
    assert '--preference-beta' in args and '--preference-margin' in args


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_actual_miro_pair_both_tokenizers_preserves_full_histories_and_only_divergent_suffixes(monkeypatch,family):
    from natlang_neuralese.common.paths import resolve
    from natlang_neuralese.data.native_preference import pair_graph
    from natlang_neuralese.data.text_corpus import native_gold_packet
    from scripts.neuralese_data import acquired_agents,acquired_preferences
    from scripts.neuralese_data.build_acquired_full_session_graphs import prepare_tokenizer
    from scripts.neuralese_data.common import Reject
    path=resolve('data_hdd','natlang-development-data/granted-format-inspection-20261016/inspection.json')
    base=resolve('data_nvme','dgxops/qad-drift-probe/base') if family=='lfm' else resolve('models','mellum21-12b-a2.5b-thinking')
    if not path.exists() or not (base/'tokenizer.json').exists():pytest.skip('pinned source inspection/tokenizer absent')
    inspection=json.loads(path.read_text())
    source=next(row for row in inspection['records'] if row['ref']=='D24' and row['catalog_file']['path'].startswith('dpo/'))
    raw=source['records'][0];rid=source['catalog_file']['path']+':0'
    receipt={'repo':source['repo'],'revision':source['revision'],'card':{'license':'source-pinned'},
             'files':[{'path':source['catalog_file']['path'],'sha256':source['source_sha256'],'bytes':source['source_bytes']}]}
    monkeypatch.setattr(acquired_agents,'acquired_rows',lambda ref,limit=None:iter([(receipt,rid,raw)]))
    _,record=next(acquired_preferences.miroverse_preference_records())
    assert not isinstance(record,Reject)
    prepared=prepare_tokenizer(base)
    tokenizer=prepared.tokenizer
    row,spec=pair_graph(record,tokenizer,tokenizer_fingerprint='diagnostic-exact-tokenizer')
    assert all(not any(node['target_weights']) for node in row['nodes'])
    assert len(row['nodes'])==sum(sum(m['role']=='assistant' for m in value['messages'][record['common_message_count']:])
                                  for value in record['alternatives'].values())
    for node in row['nodes']:
        suffix=spec['suffixes'][node['name']];alternative=node['name'].split(':')[0]
        messages=copy.deepcopy(record['alternatives'][alternative]['messages'])
        for message in messages:
            if 'reasoning' in message:message['reasoning_content']=message.pop('reasoning')
        index=suffix['source_message_index']
        _,ids,boundary=native_gold_packet(tokenizer,messages[:index],messages[index],record['alternatives'][alternative].get('tools'))
        assert node['token_ids']==ids and suffix['suffix_start']==boundary and suffix['suffix_stop']==len(ids)
        assert index>=record['common_message_count']
