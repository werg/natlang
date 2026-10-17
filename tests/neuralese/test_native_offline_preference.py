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


def test_recipe_input_reaches_actual_preference_entrypoint(tmp_path,monkeypatch):
    from natlang_neuralese.train import native_preference,native_state_sft
    from natlang_neuralese.train.recipe import stage_input_args
    _,objective=example_spec(tmp_path)
    arguments=stage_input_args({'objective_spec':{'path':str(objective.path)}},'native_offline_preference')
    received=[]
    def trainer(argv):
        path=argv[argv.index('--objective-spec')+1]
        received.append(NativePreferenceObjective(path).binding())
        assert '--objective-continuation' in argv
        return 0
    monkeypatch.setattr(native_state_sft,'main',trainer)
    assert native_preference.main(arguments)==0
    assert received==[objective.binding()]


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


def test_offline_admission_prepares_held_manifest_with_actual_role_ids(monkeypatch, tmp_path):
    from scripts.neuralese_data import prepare_acquired_preference_admission as admission

    calls=[]
    class Prepared:
        expanded_fingerprint='prepared-fingerprint'
        tokenizer=object()
        def validate(self, base):
            calls.append(('validate', base))

    expected=(101,102,103,104)
    monkeypatch.setattr('natlang_neuralese.model.shared_state.provision_role_tokenizer',
        lambda tokenizer: (calls.append(('provision', tokenizer)) or expected))

    class CapturedSource:
        def __init__(self, path, fingerprint, *, required_split, role_ids):
            calls.append(('source', path, fingerprint, required_split, role_ids))
            self.groups={'held:fixture'}

    monkeypatch.setattr('natlang_neuralese.data.native_sources.NativeGraphSource', CapturedSource)
    source=admission._load_held_graph_source(tmp_path/'held-sources.json',base=tmp_path/'base',prepared=Prepared())
    assert source.groups=={'held:fixture'}
    assert calls==[
        ('validate',tmp_path/'base'),
        ('provision',Prepared.tokenizer),
        ('validate',tmp_path/'base'),
        ('source',tmp_path/'held-sources.json','prepared-fingerprint','test',expected),
    ]


def test_held_role_plumbing_continuation_reuses_only_exact_prior_verification(tmp_path):
    from scripts.neuralese_data import prepare_acquired_preference_admission as admission

    identity={'verifier_dependencies':{'admission':{'path':'scripts/neuralese_data/prepare_acquired_preference_admission.py',
        'bytes':1,'sha256':'current'}},'other':'unchanged'}
    prior=admission._prior_role_plumbing_identity(identity)
    assert prior['verifier_dependencies']['admission']['sha256']==admission._HELD_ROLE_PLUMBING_TRANSITION['old_module_sha256']
    payload={'identity':prior,'identity_sha256':admission.canonical_json_sha256_hex(prior),'result':{'verified':1}}
    path=tmp_path/'cache.json'
    path.write_text(json.dumps({'schema':'natlang.offline-preference-view-verification/1','payload':payload,
        'payload_sha256':admission.canonical_json_sha256_hex(payload)}))
    assert admission._cached_verification(path,identity)=={'verified':1}
    assert admission._cached_verification(path,{**identity,'other':'changed'}) is None


def test_held_role_plumbing_transition_rejects_other_admission_edits(monkeypatch, tmp_path):
    from scripts.neuralese_data import prepare_acquired_preference_admission as admission

    original=admission.Path(admission.__file__).read_text()
    altered=original.replace('role_ids=provision_role_tokenizer(prepared.tokenizer)',
        'role_ids={"assistant":0}',1)
    assert altered!=original
    path=tmp_path/'prepare_acquired_preference_admission.py';path.write_text(altered)
    monkeypatch.setattr(admission,'__file__',str(path))
    with pytest.raises(ValueError,match='outside the exact held-role plumbing'):
        admission._held_role_plumbing_transition()


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
    logged=json.loads((out/'train.jsonl').read_text().splitlines()[0])
    assert logged['offline_preference']['scored_assistant_tokens']==5
    assert logged['supervised_tokens']==0  # no rejected-positive CE credit
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
    assert 'pending_native_sources' in recipe.HANDLERS[stage['kind']]['optional_inputs']
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


def test_completed_objective_tweak_adds_honest_full_pair_pass_without_reset(tmp_path,monkeypatch):
    # Reuse the actual shared trainer setup, then inspect its completed state.
    from test_native_model import tiny_source
    from test_native_state_sft import graph
    from natlang_neuralese.model.native_model import load_native_model
    from natlang_neuralese.train.native_preference import main
    base=tmp_path/'base';tiny_source(base)
    _,_,_,config=load_native_model(base,dtype=torch.float32,rank=4)
    example,objective=example_spec(tmp_path,fingerprint=config['tokenizer_fingerprint'])
    row={'schema':SCHEMA,'id':example.id,'split':'train','source_groups':['exact-task'],
        'tokenizer_fingerprint':config['tokenizer_fingerprint'],'nodes':[
            {'name':n.name,'token_ids':list(n.token_ids),'target_weights':[0.]*len(n.token_ids)} for n in example.plan.invocations]}
    train=tmp_path/'train.jsonl';train.write_text(json.dumps(row)+'\n')
    held=tmp_path/'held.jsonl';held.write_text(json.dumps(dict(graph('held'),split='test',
        source_groups=['held'],tokenizer_fingerprint=config['tokenizer_fingerprint']))+'\n')
    out=tmp_path/'run';args=['--base',str(base),'--dtype','float32','--rank','4','--device','cpu',
        '--native-graphs',str(train),'--held-native-graphs',str(held),'--objective-spec',str(objective.path),
        '--epochs','1','--eval-every','100','--out',str(out)]
    assert main(args)==0
    before=torch.load(out/'checkpoint.pt',weights_only=False)['training']
    assert before['exposure_counts']=={example.id:1}
    assert main(args+['--preference-margin','.8'])==0
    after=torch.load(out/'checkpoint.pt',weights_only=False)['training']
    assert after['step']==2 and after['exposure_counts']=={example.id:2}
    transition=after['options']['offline_objective_transition']
    assert transition['additional_full_pair_visits']==1 and transition['previous_stage_complete']
    frame=after['sft_progress'][KIND]
    assert frame['required_visits']=={example.id:2}
    assert frame['options']['epochs']==1  # original declared quota remains recorded
    assert main(args+['--preference-margin','.8'])==0
    assert torch.load(out/'checkpoint.pt',weights_only=False)['training']['step']==2
    # A newly admitted pair adds its own quota; unchanged completed rows are not replayed.
    extra_dir=tmp_path/'appended';extra_dir.mkdir()
    extra,extra_objective=example_spec(extra_dir,fingerprint=config['tokenizer_fingerprint'],source_sha='b'*64)
    new_row={**row,'id':extra.id}
    combined=tmp_path/'combined.jsonl';combined.write_text(train.read_text()+json.dumps(new_row)+'\n')
    specification=copy.deepcopy(objective.spec)
    specification['rows'].update(extra_objective.spec['rows'])
    combined_spec=tmp_path/'appended-objective.json';combined_spec.write_text(json.dumps(specification))
    continued=list(args)
    continued[continued.index('--native-graphs')+1]=str(combined)
    continued[continued.index('--objective-spec')+1]=str(combined_spec)
    assert main(continued+['--preference-margin','.8','--continued-native-graphs',str(train)])==0
    appended=torch.load(out/'checkpoint.pt',weights_only=False)['training']
    assert appended['step']==3 and appended['exposure_counts']=={example.id:2,extra.id:1}


def test_actual_preference_deficits_survive_three_phase_semantic_training_and_next_real_update(tmp_path,monkeypatch):
    from test_native_semantic_stage import _fixture
    from test_native_semantic import setup_pair
    from natlang_neuralese.train import native_semantic_stage as semantic
    from natlang_neuralese.train.native_continuation import collect_sft_progress,restore_sft_progress,progress_obligations
    from natlang_neuralese.train.native_state_sft import native_training_update,make_native_optimizer,pending_exposure_batches
    from natlang_neuralese.model.native_model import restore_model_state
    path,manifest,_,_=_fixture(tmp_path,monkeypatch)
    original_load=semantic.load_participant;frames={}
    def load(name,*args,**kwargs):
        participant=original_load(name,*args,**kwargs)
        directory=tmp_path/name;directory.mkdir(exist_ok=True)
        example,objective=example_spec(directory,fingerprint=participant.config['tokenizer_fingerprint'])
        participant.optimizer.zero_grad(set_to_none=True)
        loss,_=packed_graph_loss(participant.model,[example],objective=objective,checkpoint_layers=False)
        loss.backward();participant.optimizer.step();participant.optimizer.zero_grad(set_to_none=True)
        assert torch.isfinite(loss)
        frame={'schema':'natlang.neuralese-native-sft-training/1','identity':{'objective_kind':KIND,'objective':objective.binding()},
            'step':1,'global_step':1,'epoch':1,'cursor':1,'source_row_bindings':{example.id:'exact-binding'},
            'source_row_order':[example.id],'exposure_counts':{example.id:1},'source_receipts':[{'sha256':'train'},{'sha256':'held'}],
            'options':{'stage_id':'offline-pairs','epochs':2},'optimizer':participant.optimizer.state_dict()}
        participant.sft_progress=collect_sft_progress(frame);participant.global_step=1
        frames[name]=(example,objective,copy.deepcopy(participant.sft_progress))
        return participant
    monkeypatch.setattr(semantic,'load_participant',load)
    report=semantic.run(path,manifest,tmp_path/'semantic',device='cpu')
    assert report['completed']
    from natlang_neuralese.train.checkpoint_policy import CheckpointPolicy
    saved=CheckpointPolicy.load(tmp_path/'semantic/checkpoint.pt')
    for name,entry in saved['participants'].items():
        example,objective,original=frames[name]
        assert entry['sft_progress']==original and entry['global_step']==3
        assert progress_obligations(entry['sft_progress'])['remaining_graph_visits']==1
        participant=setup_pair()[0][name]
        if name=='mellum':participant.model.port.hf.natlang_precision='bf16'
        restore_model_state(participant.model,entry['model_state'])
        optimizer,_=make_native_optimizer(participant.model,lr=1e-4,in_backward=False)
        optimizer.load_state_dict(entry['optimizer'])
        restored=restore_sft_progress({**entry,'rng':saved['training']['rng']},'offline-pairs')
        quotas,batches=pending_exposure_batches([{'id':example.id,'forward_tokens':9}],restored['exposure_counts'],epochs=2)
        assert quotas=={example.id:2} and len(batches)==1
        optimizer.zero_grad(set_to_none=True)
        loss,_=packed_graph_loss(participant.model,[example],objective=objective,checkpoint_layers=False)
        loss.backward();optimizer.step()
        restored['exposure_counts'][example.id]+=1
        restored['step']+=1;restored['global_step']+=1
        assert progress_obligations(collect_sft_progress(restored))['complete']
