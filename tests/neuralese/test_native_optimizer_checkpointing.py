"""One packed backward must not mutate weights needed by later recomputation."""
import copy
import pytest
import torch
from natlang_neuralese.data.native_graph import parse_graph, SCHEMA
from natlang_neuralese.model.shared_state import SharedStateModel
from natlang_neuralese.train.native_state_sft import make_native_optimizer, packed_graph_loss, native_training_update
from test_shared_state import tiny


def examples():
    result=[]
    for index in range(3):
        row={'schema':SCHEMA,'id':str(index),'split':'train','source_groups':['train'],
             'tokenizer_fingerprint':'fixture','nodes':[
                {'name':'writer','token_ids':[1,58,5+index,7,8,59],
                 'producing':[[2,5]],'target_weights':[0,1,.25,1,.5,1]},
                {'name':'reader','token_ids':[3,60,61,11+index,12,13,2],
                 'imports':[{'source':'writer','start':2,'stop':5,'at':2}],
                 'target_weights':[0,.25,0,1,1,1,1]}]}
        result.append(parse_graph(row,tokenizer_fingerprint='fixture'))
    return result


def assert_exact(left,right,path='root'):
    if isinstance(left,torch.Tensor):
        assert left.dtype==right.dtype,path
        torch.testing.assert_close(left,right,rtol=0,atol=0,msg=path)
    elif isinstance(left,dict):
        assert left.keys()==right.keys(),path
        for key in left:assert_exact(left[key],right[key],path+'/'+str(key))
    elif isinstance(left,(list,tuple)):
        assert len(left)==len(right),path
        for index,(a,b) in enumerate(zip(left,right)):assert_exact(a,b,path+'/'+str(index))
    else:assert left==right,path


def qualify_updates(hook_model,batch,*,checkpoint_layers,chunk_tokens,objective=None,controlled_rounding=False):
    reference=copy.deepcopy(hook_model)
    hook_optimizer,hook_named=make_native_optimizer(hook_model,lr=.003,in_backward=True)
    reference_optimizer,reference_named=make_native_optimizer(reference,lr=.003,in_backward=False)
    names={id(value):name for name,value in hook_named}
    observed={}
    original=hook_optimizer.latent._muon_update
    deferred=reference_optimizer.latent._muon_update
    def apply(fn,parameter,group):
        if not controlled_rounding:return fn(parameter,group)
        # Stochastic rounding draws occur in different parameter orders. Give
        # each parameter the SAME draw to isolate recomputation correctness.
        import hashlib
        seed=int.from_bytes(hashlib.sha256((str(update)+group['name']).encode()).digest()[:8],'little')
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            return fn(parameter,group)
    reference_optimizer.latent._muon_update=lambda parameter,group:apply(deferred,parameter,group)
    def record(parameter,group):
        name=names[id(parameter)]
        assert name not in observed,'multiple updates in one backward: '+name
        observed[name]=parameter.grad.detach().clone()
        return apply(original,parameter,group)
    hook_optimizer.latent._muon_update=record
    for update in range(2):
        observed.clear()
        rng=torch.get_rng_state()
        reference_optimizer.zero_grad(set_to_none=True)
        expected,_=packed_graph_loss(reference,batch,checkpoint_layers=checkpoint_layers,chunk_tokens=chunk_tokens,objective=objective)
        expected.backward()
        gradients={name:value.grad.detach().clone() for name,value in reference_named if value.grad is not None}
        reference_optimizer.step()
        torch.set_rng_state(rng)
        actual,_=native_training_update(hook_model,hook_optimizer,batch,
            checkpoint_layers=checkpoint_layers,chunk_tokens=chunk_tokens,objective=objective)
        assert_exact(actual.detach(),expected.detach(),'loss')
        for name,gradient in observed.items():assert_exact(gradient,gradients[name],'gradient/'+name)
        latent_names={group['name'] for group in reference_optimizer.latent.param_groups}
        assert set(observed)==latent_names & gradients.keys()
        assert observed,'test must execute actual in-backward updates'
        assert_exact(hook_model.state_dict(),reference.state_dict(),'weights')
        assert_exact(hook_optimizer.state_dict(),reference_optimizer.state_dict(),'optimizer')


@pytest.mark.parametrize('family',['lfm','mellum'])
@pytest.mark.parametrize('checkpoint_layers',[False,True])
@pytest.mark.parametrize('chunk_tokens',[1,3])
def test_in_backward_matches_deferred_gradients_weights_and_optimizer(family,checkpoint_layers,chunk_tokens):
    torch.manual_seed(14)
    model=SharedStateModel(tiny(family),(58,59,60,61),rank=4)
    if family=='mellum':model.port.hf.natlang_precision='bf16'
    qualify_updates(model,examples(),checkpoint_layers=checkpoint_layers,chunk_tokens=chunk_tokens)


@pytest.mark.parametrize('family',['lfm','mellum'])
@pytest.mark.parametrize('mode',['rft','grpo'])
@pytest.mark.parametrize('chunk_tokens',[1,3])
def test_native_policy_in_backward_matches_deferred_exactly(tmp_path,family,mode,chunk_tokens):
    import json
    from test_native_rl_replay import actor_rollout
    from natlang_neuralese.rl.native_replay import prepare_objective,NativePolicyObjective
    model,roles,identity,_,block,turns=actor_rollout(family)
    if family=='mellum':model.port.hf.natlang_precision='bf16'
    raw=[{'messages':[],'target':{},'invocation_id':turn['graph_turn'],
          'sampling':turn['sampling'],'replay':{'raw_calls_retained':True}} for turn in turns]
    episodes=[{'task_id':'task-'+str(i),'sample':0,'split':'train','group':'g-'+str(i),
        'reward':{'quality':1},'policy':{'checkpoint_sha256':identity['checkpoint_sha256']},'turns':copy.deepcopy(raw)}
        for i in range(3)]
    if mode=='grpo':
        for positive in episodes[:]:
            negative=copy.deepcopy(positive);negative.update(sample=1,reward={'quality':0})
            episodes.append(negative)
    graphs,spec=prepare_objective(episodes,identity=identity,roles=roles,fetch_block=lambda _:block,mode=mode)
    spec['chunk_tokens']=chunk_tokens
    path=tmp_path/'objective.json';path.write_text(json.dumps(spec))
    batch=[parse_graph(row,tokenizer_fingerprint=identity['tokenizer_fingerprint']) for row in graphs]
    qualify_updates(model,batch,checkpoint_layers=True,chunk_tokens=chunk_tokens,objective=NativePolicyObjective(path))


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_bfloat16_in_backward_matches_deferred_with_identical_rounding_draws(family):
    model=SharedStateModel(tiny(family),(58,59,60,61),rank=4).to(dtype=torch.bfloat16)
    if family=='mellum':model.port.hf.natlang_precision='bf16'
    qualify_updates(model,examples(),checkpoint_layers=True,chunk_tokens=1,controlled_rounding=True)
