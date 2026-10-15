"""Exact cached-prefix layout and complete ordinary SFT gradient preservation."""
from types import SimpleNamespace
import copy
import pytest
import torch
from natlang_neuralese.model.shared_state import Invocation, ImportSpan, compile_graph, SharedStateModel
from natlang_neuralese.train.native_prefix_packing import pack_causal_prefixes
from natlang_neuralese.train.native_state_sft import packed_graph_loss, pack_native_graphs
from test_shared_state import tiny


def examples():
    sequences = [(2,3,4,5,6,7),(2,3,4,5,8),(2,3,4,5),(2,3,4,5,6,7),(9,10,11)]
    return [SimpleNamespace(id=f'e{index}',plan=compile_graph([Invocation('answer',tokens)]),
        target_weights=tuple([0]+[(.05 if position%2 else 1)*(index+1)
                                  for position in range(1,len(tokens))]))
            for index,tokens in enumerate(sequences)]


def test_prefix_layout_retains_duplicates_and_internal_terminal_queries_without_adaptation():
    rows=examples();plan,maps=pack_causal_prefixes(rows)
    assert plan.token_count == 10
    assert not any(plan.producing)
    assert maps[0] == maps[3] and maps[2] == maps[0][:4]
    tokens=[token for node in plan.invocations for token in node.token_ids]
    for example,mapping in zip(rows,maps):
        assert tuple(tokens[index] for index in mapping) == example.plan.invocations[0].token_ids
        for logical,physical in enumerate(mapping):
            assert plan.own_positions[physical] == logical
            assert plan.history(physical) == mapping[:logical+1]


@pytest.mark.parametrize('family',['lfm','mellum'])
@pytest.mark.parametrize('checkpoint_layers',[False,True])
def test_prefix_logits_loss_and_all_gradients_match_disjoint_packing(family,checkpoint_layers):
    model=SharedStateModel(tiny(family),(58,59,60,61),rank=4)
    with torch.no_grad():model.adapter.up.weight.normal_(std=.1)
    reference=copy.deepcopy(model);rows=examples()
    shared,maps=pack_causal_prefixes(rows)
    disjoint,_,ownership,_=pack_native_graphs(rows)
    with torch.no_grad():
        actual=model(shared,retain_states=False)['h_final'][0]
        expected=reference(disjoint,retain_states=False)['h_final'][0]
        for mapping,(first,last) in zip(maps,ownership):
            start=disjoint.offsets[first];stop=disjoint.offsets[last-1]+len(disjoint.invocations[last-1].token_ids)
            torch.testing.assert_close(actual[list(mapping)],expected[start:stop],atol=2e-6,rtol=2e-5)
    loss,details=packed_graph_loss(model,rows,checkpoint_layers=checkpoint_layers,chunk_tokens=2,
                                  share_causal_prefixes=True)
    reference_loss,reference_details=packed_graph_loss(reference,rows,checkpoint_layers=checkpoint_layers,chunk_tokens=2)
    torch.testing.assert_close(loss,reference_loss,atol=1e-6,rtol=1e-6)
    assert [row['weight'] for row in details] == [row['weight'] for row in reference_details]
    assert [row['supervised_tokens'] for row in details] == [row['supervised_tokens'] for row in reference_details]
    loss.backward();reference_loss.backward()
    for (name,parameter),(other_name,other) in zip(model.named_parameters(),reference.named_parameters()):
        assert name==other_name
        assert (parameter.grad is None)==(other.grad is None),name
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad,other.grad,atol=3e-6,rtol=2e-4,msg=name)


def test_authenticated_producing_or_imported_graphs_keep_existing_layout():
    rows=examples()
    rows[0].plan=compile_graph([Invocation('writer',(2,3,4),producing=((1,3),)),
        Invocation('reader',(5,6),imports=(ImportSpan(0,'writer',1,3),))])
    assert pack_causal_prefixes(rows) is None
    rows[0].plan=compile_graph([Invocation('writer',(2,3,4),producing=((1,3),))])
    assert pack_causal_prefixes(rows) is None


@pytest.mark.parametrize('edge',[ImportSpan(0,'prefix',1,3,True),ImportSpan(1,'prefix',0,3,True),
                                 ImportSpan(0,'prefix',0,3,False)])
def test_raw_cache_requires_complete_positionally_identical_prefix(edge):
    with pytest.raises(ValueError,match='complete causal prefix'):
        compile_graph([Invocation('prefix',(2,3,4),cache_prefix=True),Invocation('tail',(5,),imports=(edge,))])


def test_raw_cache_export_never_activates_adapter_or_accepts_external_or_neuralese_parent():
    with pytest.raises(ValueError,match='unadapted'):
        compile_graph([Invocation('prefix',(2,3,4),producing=((0,3),),cache_prefix=True)])
    with pytest.raises(ValueError,match='raw complete prefix'):
        compile_graph([Invocation('prefix',(2,3,4),producing=((0,3),)),
            Invocation('tail',(5,),imports=(ImportSpan(0,'prefix',0,3),),cache_prefix=True)])
    with pytest.raises(ValueError,match='raw complete prefix'):
        compile_graph([Invocation('tail',(5,),imports=(ImportSpan(0,'external',0,3),),cache_prefix=True)],
                      external_sizes={'external':3})


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_two_actual_muon_updates_and_optimizer_histories_match_disjoint_reference(family):
    from natlang_neuralese.train.native_state_sft import make_native_optimizer,native_training_update
    model=SharedStateModel(tiny(family),(58,59,60,61),rank=4)
    if family=='mellum':model.port.hf.natlang_precision='bf16'
    reference=copy.deepcopy(model)
    model.share_causal_prefixes=True
    optimizer,_=make_native_optimizer(model,lr=1e-5,in_backward=True)
    reference_optimizer,_=make_native_optimizer(reference,lr=1e-5,in_backward=True)
    gradients = {'shared':{},'reference':{}}
    # Prefix sharing changes the autograd traversal, not the stochastic rounding
    # distribution. Match each named matrix's draw to isolate its actual update.
    def controlled(function,parameter,group,owner,label):
        import hashlib
        seed=int.from_bytes(hashlib.sha256((str(step)+group['name']).encode()).digest()[:8],'little')
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            old=owner.state[parameter].get('momentum',torch.zeros_like(parameter,dtype=torch.bfloat16))
            exact_m=old.float()*group['momentum']+parameter.grad.float()
            gradients[label][group['name']]=parameter.grad.detach().float().clone()
            result=function(parameter,group)
            # Explicitly explain every stored BF16 momentum bit with its exact
            # FP32 input. Near a cast midpoint, tiny valid accumulation changes
            # can change a BF16 bit; do not hide that behind wider tolerances.
            torch.testing.assert_close(owner.state[parameter]['momentum'],exact_m.to(torch.bfloat16),atol=0,rtol=0)
            return result
    actual_update=optimizer.latent._muon_update
    expected_update=reference_optimizer.latent._muon_update
    optimizer.latent._muon_update=lambda parameter,group:controlled(actual_update,parameter,group,optimizer.latent,'shared')
    reference_optimizer.latent._muon_update=lambda parameter,group:controlled(expected_update,parameter,group,reference_optimizer.latent,'reference')
    def compare(left,right,path='state'):
        if isinstance(left,torch.Tensor):
            if path.endswith('/momentum') and left.dtype==torch.bfloat16:
                # The exact per-matrix recurrence/cast oracle above and strict
                # raw-gradient comparison below qualify these states. A prior
                # rounding-bit difference propagates through the next exact
                # recurrence; unrelated cross-state tolerances cannot test it.
                return
            torch.testing.assert_close(left,right,atol=3e-6,rtol=3e-4,msg=path)
        elif isinstance(left,dict):
            assert left.keys()==right.keys(),path
            for key in left:compare(left[key],right[key],path+'/'+str(key))
        elif isinstance(left,(tuple,list)):
            assert len(left)==len(right),path
            for index,(a,b) in enumerate(zip(left,right)):compare(a,b,path+'/'+str(index))
        else:assert left==right,path
    for step in range(2):
        rng=torch.get_rng_state()
        expected,_=native_training_update(reference,reference_optimizer,examples(),chunk_tokens=2)
        torch.set_rng_state(rng)
        actual,_=native_training_update(model,optimizer,examples(),chunk_tokens=2)
        torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-6)
        compare(model.state_dict(),reference.state_dict(),'weights')
        compare(gradients['shared'],gradients['reference'],'raw_gradients')
        compare(optimizer.state_dict(),reference_optimizer.state_dict(),'optimizer')


def test_original_rl_graph_identity_does_not_gain_default_cache_field():
    from dataclasses import asdict
    from natlang_neuralese.model.native_graph_plan import invocation_dict
    node=Invocation('original',(2,3,4))
    old=asdict(node);old.pop('cache_prefix')
    assert invocation_dict(node)==old
    assert invocation_dict(Invocation('cached',(2,3),cache_prefix=True))['cache_prefix'] is True


def test_actual_trainer_cache_switch_retains_saved_exposures_and_optimizer_on_resume(tmp_path):
    import json
    from test_native_model import tiny_source
    from natlang_neuralese.model.native_model import load_native_model
    from natlang_neuralese.data.native_graph import SCHEMA
    from natlang_neuralese.train.native_state_sft import main
    base=tmp_path/'base';tiny_source(base)
    _,_,_,config=load_native_model(base,dtype=torch.float32,rank=4)
    def row(name,tokens,split):
        return {'schema':SCHEMA,'id':name,'split':split,'source_groups':[name],
            'tokenizer_fingerprint':config['tokenizer_fingerprint'],'nodes':[
                {'name':'answer','token_ids':tokens,'target_weights':[0,1,1]}]}
    train=tmp_path/'train.jsonl';held=tmp_path/'held.jsonl';out=tmp_path/'run'
    train.write_text(''.join(json.dumps(row(name,tokens,'train'))+'\n'
                            for name,tokens in [('a',[1,2,3]),('b',[1,2,4])]))
    held.write_text(json.dumps(row('held',[5,6,7],'test'))+'\n')
    argv=['--base',str(base),'--native-graphs',str(train),'--held-native-graphs',str(held),
        '--out',str(out),'--dtype','float32','--device','cpu','--rank','4',
        '--batch-graphs','2','--epochs','1','--share-causal-prefixes','--checkpoint-every','1','--eval-every','1',
        '--forward-atol','1e-4','--gradient-atol','1e-5','--chunk-tokens','2']
    assert main(argv)==0
    saved=torch.load(out/'checkpoint.pt',weights_only=False)
    assert saved['training']['options']['share_causal_prefixes'] is True
    assert saved['training']['exposure_counts']=={'a':1,'b':1}
    progress=[json.loads(line) for line in (out/'train.jsonl').read_text().splitlines()]
    assert progress[-1]['physical_forward_tokens']==4
    assert progress[-1]['reused_prefix_tokens']==2
    argv[argv.index('--epochs')+1]='2'
    argv[argv.index('--share-causal-prefixes')]='--no-share-causal-prefixes'
    assert main(argv)==0
    continued=torch.load(out/'checkpoint.pt',weights_only=False)
    assert continued['training']['exposure_counts']=={'a':2,'b':2}
    assert continued['training']['step']==saved['training']['step']+1
    assert continued['training']['options']['share_causal_prefixes'] is False
