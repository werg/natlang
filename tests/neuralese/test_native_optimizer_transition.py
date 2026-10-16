"""No native ordinary table may be silently reclassified as a Muon matrix."""
import copy
import pytest
import torch

from test_shared_state import tiny
from test_native_optimizer_checkpointing import examples,assert_exact
from natlang_neuralese.model.shared_state import SharedStateModel
from natlang_neuralese.train.native_state_sft import make_native_optimizer,native_training_update
from natlang_neuralese.train import optim
from natlang_neuralese.train.optim_restore import OptimizerRestoreError,port_saved_slot_names
from natlang_neuralese.train.native_optimizer_migration import POLICY,validate_native_optimizer_migration


def build(family):
    model=SharedStateModel(tiny(family),(58,59,60,61),rank=4,token_override_ids=(54,55))
    if family=='mellum':model.port.hf.natlang_precision='bf16'
    return model


def histories(optimizer):
    result={}
    for child in (optimizer.muon,optimizer.auxiliary,optimizer.latent):
        if child is None:continue
        for group in child.param_groups:
            for p in group['params']:result[optimizer._parameter_names[id(p)]]=(group,child.state.get(p,{}))
    return result


def legacy_optimizer(model,monkeypatch):
    constructor=optim.PortMuonAdamW
    def historical(*args,**kwargs):
        kwargs.pop('force_adamw_ids',None)
        return constructor(*args,**kwargs)
    with monkeypatch.context() as old:
        old.setattr(optim,'PortMuonAdamW',historical)
        return make_native_optimizer(model,lr=3e-6,in_backward=True,interface_row_lr=.003)[0]


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_native_ordinary_contract_routes_rows_and_narrow_adapter_to_adamw(family):
    model=build(family);optimizer,named=make_native_optimizer(model,lr=3e-6,in_backward=True)
    latent,ordinary=optim.latent_split(named,'muonsr');actual={v['name']:v['optimizer'] for v in optimizer.schema}
    assert ordinary and all(actual[name]=='adamw' for name,_ in ordinary)
    assert all(actual[name]=='muonsr' for name,_ in latent)
    assert all(actual[name]=='adamw' for name in actual if name.startswith('interface.'))
    p=torch.nn.Parameter(torch.ones(4,32))
    legacy=optim.PortMuonAdamW([('legacy.hidden',p)],lr=.001,vocab_size=999)
    assert legacy.schema[0]['optimizer']=='muon','legacy generic callers keep their declared classifier'


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_explicit_transition_preserves_numeric_history_rng_weights_and_exact_future_resume(family,monkeypatch):
    model=build(family);old=legacy_optimizer(model,monkeypatch)
    native_training_update(model,old,examples(),checkpoint_layers=True,chunk_tokens=3)
    saved=copy.deepcopy(old.state_dict());untouched=copy.deepcopy(saved);before=copy.deepcopy(model.state_dict())
    old_named=histories(old);new,_=make_native_optimizer(model,lr=3e-6,in_backward=True)
    changed={v['name'] for v in saved['schema'] if v['optimizer']=='muon'}
    assert changed and any('.adapter.' in name for name in changed)
    rng=torch.get_rng_state().clone()
    with pytest.raises(OptimizerRestoreError,match='explicitly declare'):new.load_state_dict(saved)
    new.load_state_dict(saved,native_optimizer_migration=POLICY)
    assert torch.equal(rng,torch.get_rng_state());assert_exact(model.state_dict(),before)
    assert_exact(saved,untouched,'source checkpoint remains unchanged')
    assert len(new._native_optimizer_transitions)==1
    provenance=new._native_optimizer_transitions[0]
    assert {p['name'] for p in provenance['parameters']}==changed
    now=histories(new)
    for name,(group,state) in now.items():
        if name in changed:
            assert int(state['step'])==0 and not torch.count_nonzero(state['exp_avg']) and not torch.count_nonzero(state['exp_avg_sq'])
            record=next(p for p in provenance['parameters'] if p['name']==name)
            assert_exact(record['source_state'],old_named[name][1],name+'/original_muon')
            assert_exact(record['source_group'],{**old_named[name][0],'params':record['source_group']['params']})
            assert group['lr']==old_named[name][0]['lr']
        else:
            assert_exact(state,old_named[name][1],name+'/unaffected')
            assert {k:v for k,v in group.items() if k!='params'}=={k:v for k,v in old_named[name][0].items() if k!='params'}
    # A new checkpoint carries exact original Muon tensors and actual new Adam
    # updates through ordinary named serialization, without repeating transition.
    resumed=build(family);resumed.load_state_dict(model.state_dict());resumed.port.hf.load_state_dict(model.port.hf.state_dict())
    loaded,_=make_native_optimizer(resumed,lr=3e-6,in_backward=True)
    loaded.load_state_dict(copy.deepcopy(new.state_dict()))
    assert_exact(loaded._native_optimizer_transitions,provenance and new._native_optimizer_transitions)
    for update in range(2):
        rng=torch.get_rng_state()
        expected,_=native_training_update(model,new,examples(),checkpoint_layers=True,chunk_tokens=3)
        torch.set_rng_state(rng)
        observed,_=native_training_update(resumed,loaded,examples(),checkpoint_layers=True,chunk_tokens=3)
        assert torch.equal(expected,observed);assert_exact(model.state_dict(),resumed.state_dict())
        assert_exact(new.state_dict(),loaded.state_dict())
    again=copy.deepcopy(new.state_dict());loaded.load_state_dict(again,native_optimizer_migration=POLICY)
    assert len(loaded._native_optimizer_transitions)==1


def test_transition_refuses_undeclared_nonordinary_change_and_bad_momentum(monkeypatch):
    model=build('lfm');old=legacy_optimizer(model,monkeypatch)
    native_training_update(model,old,examples(),checkpoint_layers=False,chunk_tokens=3)
    saved=copy.deepcopy(old.state_dict());target,_=make_native_optimizer(model,lr=3e-6,in_backward=True)
    bad=copy.deepcopy(saved);names=port_saved_slot_names(bad,'muon',bad['muon'])
    slot=next(iter(bad['muon']['state']));bad['muon']['state'][slot]['momentum_buffer'].fill_(float('nan'))
    with pytest.raises(OptimizerRestoreError,match='numeric'):target.load_state_dict(bad,native_optimizer_migration=POLICY)
    target._force_adamw_names.clear()
    with pytest.raises(OptimizerRestoreError,match='only proven ordinary'):target.load_state_dict(saved,native_optimizer_migration=POLICY)
    with pytest.raises(ValueError):validate_native_optimizer_migration('guess-second-moments')


def test_real_sft_explicit_migration_only_checkpoint_preserves_weights_rng_and_completed_row_credits(tmp_path,monkeypatch):
    import json
    from test_native_model import tiny_source
    from natlang_neuralese.model.native_model import load_native_model
    from natlang_neuralese.train import native_state_sft as trainer
    base=tmp_path/'base';tiny_source(base)
    _,_,roles,config=load_native_model(base,dtype=torch.float32,rank=4)
    row={'schema':'natlang.neuralese-native-graph/1','id':'train-one','split':'train','source_groups':['train-one'],
         'tokenizer_fingerprint':config['tokenizer_fingerprint'],'nodes':[{'name':'own','token_ids':[4,roles.ids[0],5,roles.ids[1]],'target_weights':[0,1,.25,1]}]}
    held={**row,'id':'held','split':'test','source_groups':['held-one']}
    train=tmp_path/'train.jsonl';train.write_text(json.dumps(row)+'\n');hold=tmp_path/'held.jsonl';hold.write_text(json.dumps(held)+'\n')
    def args(out):
        return ['--native-graphs',str(train),'--held-native-graphs',str(hold),'--out',str(out),'--base',str(base),
            '--device','cpu','--dtype','float32','--rank','4','--epochs','1','--eval-every','1','--batch-graphs','1',
            '--learning-rate','3e-6','--fp8-training','off','--forward-atol','1e-4','--gradient-atol','1e-5']
    original=optim.PortMuonAdamW
    def historical(*a,**kw):kw.pop('force_adamw_ids',None);return original(*a,**kw)
    old_out=tmp_path/'old'
    with monkeypatch.context() as legacy:
        legacy.setattr(optim,'PortMuonAdamW',historical)
        assert trainer.main(args(old_out))==0
    checkpoint=old_out/'checkpoint.pt';before=torch.load(checkpoint,map_location='cpu',weights_only=False)
    new_out=tmp_path/'new';command=args(new_out)+['--continue-from',str(checkpoint)]
    with pytest.raises(OptimizerRestoreError,match='explicitly declare'):trainer.main(command)
    assert trainer.main(command+['--native-optimizer-migration',POLICY])==0
    after=torch.load(new_out/'checkpoint.pt',map_location='cpu',weights_only=False)
    for key in ('backbone_state','interface_state'):assert_exact(before[key],after[key])
    for key in ('exposure_counts','source_row_bindings','source_row_order','rng','global_step','epoch','cursor'):
        assert_exact(before['training'][key],after['training'][key],key)
    assert len(after['training']['optimizer']['native_optimizer_transitions'])==1


def test_recipe_and_semantic_declarations_use_one_explicit_migration_policy():
    from natlang_neuralese.train.recipe import HANDLERS,validate_stage_parameters
    from natlang_neuralese.train.native_optimizer_migration import native_optimizer_migration_option
    for kind in ('native_state_sft','native_state_rl'):
        assert 'native_optimizer_migration' in HANDLERS[kind]['parameters']
        validate_stage_parameters(kind,{'native_optimizer_migration':POLICY})
        with pytest.raises(ValueError):validate_stage_parameters(kind,{'native_optimizer_migration':'automatic'})
    assert native_optimizer_migration_option({'options':{'native_optimizer_migration':POLICY}},'student')==POLICY
    assert native_optimizer_migration_option({'native_optimizer_migration':POLICY,'models':{'student':{'native_optimizer_migration':None}}},'student') is None


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_corrected_adam_interface_learns_real_native_role_targets(family):
    model=build(family)
    optimizer,_=make_native_optimizer(model,lr=3e-6,interface_row_lr=3e-4,in_backward=True)
    from natlang_neuralese.train.native_state_sft import packed_graph_loss
    before=float(packed_graph_loss(model,examples(),checkpoint_layers=True,chunk_tokens=3)[0].detach())
    for _ in range(8):native_training_update(model,optimizer,examples(),checkpoint_layers=True,chunk_tokens=3)
    after=float(packed_graph_loss(model,examples(),checkpoint_layers=True,chunk_tokens=3)[0].detach())
    assert after<before,'actual native producer/consumer objectives must learn under corrected ordinary AdamW'
    assert model.native_sparse_updates['input_rows']['rows'][0]['update_l2']>0


def test_transition_composes_with_declared_master_promotion_and_added_rows():
    p=torch.nn.Parameter(torch.ones(4,32,dtype=torch.bfloat16));latent=torch.nn.Parameter(torch.ones(16,16))
    old=optim.PortMuonAdamW([('rows',p)],lr=3e-6,vocab_size=999,
        latent=[('hidden',latent,3e-6,1)],latent_optimizer='muonsr')
    old.muon.state[p]['momentum_buffer']=torch.arange(p.numel(),dtype=torch.bfloat16).reshape_as(p)
    saved=copy.deepcopy(old.state_dict())
    promoted=torch.nn.Parameter(p.float());added=torch.nn.Parameter(torch.ones(2,32))
    new=optim.PortMuonAdamW([('rows',promoted),('added',added)],lr=3e-6,vocab_size=999,
        force_adamw_ids={id(promoted),id(added)},latent=[('hidden',latent,3e-6,1)],latent_optimizer='muonsr')
    new.load_state_dict(saved,allow_added_names={'added'},allow_promoted_names={'rows'},native_optimizer_migration=POLICY)
    retained=new._native_optimizer_transitions[0]['parameters'][0]['source_state']['momentum_buffer']
    assert retained.dtype==torch.bfloat16 and torch.equal(retained,old.muon.state[p]['momentum_buffer'])
    assert new.auxiliary.state[promoted]['exp_avg'].dtype==torch.float32
    assert added not in new.auxiliary.state
    bad=copy.deepcopy(new.state_dict());bad['native_optimizer_transitions'][0]['parameters'][0]['source_state']['momentum_buffer']=retained.float()
    with pytest.raises(OptimizerRestoreError,match='original contract'):new.load_state_dict(bad)


def test_production_width_weighted_role_readout_matches_independent_float64_ce_adjoint():
    from torch import nn
    from natlang_neuralese.data.native_graph import parse_graph,SCHEMA,graph_loss_from_result
    class ReadoutPort(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding_weight=nn.Parameter(torch.zeros(64,2304),requires_grad=False)
            self.output_weight=nn.Parameter(torch.zeros(64,2304),requires_grad=False)
            self.final_norm=nn.Identity()
    torch.manual_seed(123)
    model=SharedStateModel(ReadoutPort(),(60,61,62,63),rank=4,token_override_ids=(58,59))
    hidden=torch.randn(5,2304);ids=[4,60,5,61,58];weights=[0,1,.05,.25,1]
    graph=parse_graph({'schema':SCHEMA,'id':'math','split':'train','source_groups':['diagnostic'],
        'tokenizer_fingerprint':'x','nodes':[{'name':'n','token_ids':ids,'target_weights':weights}]},tokenizer_fingerprint='x')
    with torch.no_grad():model.output_rows.normal_(0,.01);model.extra_output_rows.normal_(0,.01)
    loss,_=graph_loss_from_result(model,graph,{'plan':graph.plan,'h_final':hidden.unsqueeze(0)},chunk_tokens=2)
    loss.backward()
    matrix=model.port.output_weight.detach().clone()
    matrix[list(model.marker_ids)]=model.output_rows.detach()
    matrix[list(model.token_override_ids)]=model.extra_output_rows.detach()
    logits=hidden[:4].double()@matrix.double().T;labels=torch.tensor(ids[1:]);weight=torch.tensor(weights[1:],dtype=torch.float64)
    oracle=((logits.softmax(-1)-torch.nn.functional.one_hot(labels,64))*weight[:,None]/weight.sum()).T@hidden[:4].double()
    reference=(torch.nn.functional.cross_entropy(logits,labels,reduction='none')*weight).sum()/weight.sum()
    torch.testing.assert_close(loss.double(),reference,atol=2e-6,rtol=0)
    torch.testing.assert_close(model.output_rows.grad.double(),oracle[list(model.marker_ids)],atol=2e-7,rtol=2e-6)
