"""Actual native ports and the real inference codec law, with live source AD."""
from dataclasses import replace
from types import SimpleNamespace
import pytest
import torch
import torch.nn.functional as F
from test_native_semantic_runtime import fixture
from natlang_neuralese.model.shared_state import Invocation, ImportSpan, compile_graph, capture_block
from natlang_neuralese.model.native_codec_graph import CodecDependency, execute_codec_graph


def case(tmp_path,source,target,stopped=False,checkpointed=False,source_ids=(4,5)):
    import os
    participants,channel=fixture(tmp_path)
    device=os.environ.get('NATLANG_NATIVE_ADAPTER_TEST_DEVICE','cpu')
    channel.vae.to(device)
    for name,participant in participants.items():
        participant.model.to(device)
        participant.identity=channel.bindings[name]
        participant.model.port.checkpoint_layers=checkpointed
        with torch.no_grad():participant.model.adapter.up.weight.normal_(std=.01)
    sp,tp=participants[source],participants[target]
    text=sp.tokenizer.decode(list(source_ids),skip_special_tokens=False)
    target_ids=tuple(tp.tokenizer(text,add_special_tokens=False)['input_ids'])
    op,close=sp.model.marker_ids[:2];_,_,co,cc=tp.model.marker_ids
    writer=Invocation('writer',(op,*source_ids,close),producing=((1,1+len(source_ids)),))
    reader=Invocation('reader',(7,co,cc,8),imports=(ImportSpan(2,'wire',0,len(target_ids),stop_gradient=stopped),))
    plans={source:compile_graph([writer])}
    if source==target:plans[source]=compile_graph([writer,reader],external_sizes={'wire':len(target_ids)})
    else:plans[target]=compile_graph([reader],external_sizes={'wire':len(target_ids)})
    dependency=CodecDependency('wire',target,source,target,text,'source-message','2'*64,source_ids,target_ids,
        source_node='writer',source_start=1,source_stop=1+len(source_ids))
    return participants,channel,plans,dependency


def objective(participants,results,target):
    model=participants[target].model;result=results[target]
    index=next(index for index,node in enumerate(result['plan'].invocations) if node.name=='reader')
    node=result['plan'].invocations[index]
    query=result['plan'].offsets[index]+len(node.token_ids)-1
    logits=model.logits(result['h_final'][:,query]).float()
    return F.cross_entropy(logits,torch.tensor([9],device=logits.device))


def reference(participants,channel,dependency):
    sp=participants[dependency.source_model];tp=participants[dependency.target_participant]
    op,close=sp.model.marker_ids[:2];_,_,co,cc=tp.model.marker_ids
    end=1+len(dependency.source_token_ids)
    plan=compile_graph([Invocation('writer',(op,*dependency.source_token_ids,close),producing=((1,end),))])
    result=sp.model(plan)
    block=capture_block(result,'writer',1,end,text=dependency.text,**sp.identity)
    decoded,_=channel.adapt(block,target_name=dependency.target_model,target_model=tp.model,
        target_tokenizer=tp.tokenizer,target_identity=tp.identity,allow_nearest=True,source_state_id=dependency.source_state_id)
    consumer=compile_graph([Invocation('reader',(7,co,cc,8),imports=(ImportSpan(2,'wire',0,len(decoded.token_ids),
        stop_gradient=False),))],external_sizes={'wire':len(decoded.token_ids)})
    out=tp.model(consumer,external_blocks={'wire':decoded},external_identity={'wire':tp.identity})
    return {dependency.target_participant:out}


@pytest.mark.parametrize('source,target',[('lfm','lfm'),('mellum','mellum'),('lfm','mellum'),('mellum','lfm')])
@pytest.mark.parametrize('checkpointed',[False,True])
def test_real_codec_dependency_matches_explicit_semantic_composition_and_source_adjoints(tmp_path,source,target,checkpointed):
    torch.manual_seed(1107)
    participants,channel,plans,dependency=case(tmp_path,source,target,checkpointed=checkpointed)
    selected=[participants[source].model.adapter.up.weight,participants[target].model.adapter.up.weight,
              *channel.vae.posterior.parameters(),*channel.vae.shared_decoder.parameters()]
    selected=list({id(value):value for value in selected}.values())
    results,execution=execute_codec_graph(participants,plans,[dependency],channel=channel)
    got=objective(participants,results,target)
    gradients=torch.autograd.grad(got,selected,allow_unused=True)
    expected_results=reference(participants,channel,dependency)
    expected=objective(participants,expected_results,target)
    expected_grad=torch.autograd.grad(expected,selected,allow_unused=True)
    torch.testing.assert_close(got,expected,rtol=0,atol=0)
    assert gradients[0].norm()>0, 'consumer loss must reach the actual producer through encoder/code/decoder'
    assert sum(float(gradient.norm()) for gradient in gradients[1:] if gradient is not None)>0
    for actual,wanted in zip(gradients,expected_grad):
        assert (actual is None)==(wanted is None)
        if actual is not None:torch.testing.assert_close(actual,wanted,rtol=0,atol=0)
    assert [entry['posterior'] for entry in execution if entry['kind']=='semantic-codec']==['mean']
    assert sum(entry['kind']=='semantic-target-baseline' for entry in execution)==1
    seen=[name for entry in execution if entry['kind']=='native-graph-segment' for name in entry['invocations']]
    assert sorted(seen)==['reader','writer']
    assert sum(entry.get('tokens',0) for entry in execution)==len(dependency.target_token_ids)+2


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_semantic_constant_import_cuts_only_consumed_codec_path(tmp_path,family):
    participants,channel,plans,dependency=case(tmp_path,family,family,stopped=True)
    results,_=execute_codec_graph(participants,plans,[dependency],channel=channel)
    loss=objective(participants,results,family)
    values=[participants[family].model.adapter.up.weight,*channel.vae.parameters()]
    gradients=torch.autograd.grad(loss,values,allow_unused=True)
    assert all(value is None or torch.count_nonzero(value)==0 for value in gradients)


@pytest.mark.parametrize('source,target',[('lfm','lfm'),('mellum','mellum'),('lfm','mellum'),('mellum','lfm')])
def test_codec_graph_matches_actual_incremental_producer_and_consumer_adjoints(tmp_path,source,target):
    from natlang_neuralese.model.shared_state import native_step,concatenate_captures,splice_block
    from natlang_neuralese.model.state_block import NativeStateBlock
    torch.manual_seed(1107)
    participants,channel,plans,dependency=case(tmp_path,source,target)
    sp,tp=participants[source],participants[target]
    values=list({id(value):value for value in [sp.model.adapter.up.weight,tp.model.adapter.up.weight,
                 *channel.vae.posterior.parameters(),*channel.vae.shared_decoder.parameters()]}.values())
    results,_=execute_codec_graph(participants,plans,[dependency],channel=channel)
    actual_loss=objective(participants,results,target)
    actual_gradients=torch.autograd.grad(actual_loss,values,allow_unused=True)
    def step(participant,ids,cache=None,producing=False):
        return native_step(participant.model,torch.tensor([ids],device=participant.model.input_rows.device),
                           cache,producing=producing)
    opened=step(sp,[sp.model.marker_ids[0]])
    payload=step(sp,list(dependency.source_token_ids),opened['cache'],True)
    step(sp,[sp.model.marker_ids[1]],payload['cache'])
    block=NativeStateBlock(text=dependency.text,**sp.identity,**concatenate_captures([payload])).validate()
    decoded,_=channel.adapt(block,target_name=target,target_model=tp.model,target_tokenizer=tp.tokenizer,
                           target_identity=tp.identity,allow_nearest=True,source_state_id=dependency.source_state_id)
    prefix=step(tp,[7,tp.model.marker_ids[2]])
    cache=splice_block(tp.model,prefix['cache'],decoded,**{key:value for key,value in tp.identity.items() if key!='architecture'})
    suffix=step(tp,[tp.model.marker_ids[3],8],cache)
    logits=tp.model.logits(suffix['h_final'][:,-1]).float()
    wanted=F.cross_entropy(logits,torch.tensor([9],device=logits.device))
    wanted_gradients=torch.autograd.grad(wanted,values,allow_unused=True)
    torch.testing.assert_close(actual_loss,wanted,rtol=2e-5,atol=3e-6)
    for actual,expected in zip(actual_gradients,wanted_gradients):
        assert (actual is None)==(expected is None)
        if actual is None:continue
        torch.testing.assert_close(actual,expected,rtol=2e-5,atol=3e-6)
        if expected.norm()>0:
            assert (actual-expected).norm()/expected.norm()<2e-5


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_one_live_and_one_constant_use_share_codec_values_without_coalescing_gradients(tmp_path,family):
    participants,channel,plans,dependency=case(tmp_path,family,family)
    writer,reader=plans[family].invocations
    stopped=replace(reader,name='constant-reader',imports=tuple(replace(edge,stop_gradient=True) for edge in reader.imports))
    plans[family]=compile_graph([writer,reader,stopped],external_sizes={'wire':len(dependency.target_token_ids)})
    results,execution=execute_codec_graph(participants,plans,[dependency],channel=channel)
    result=results[family];model=participants[family].model
    positions=[offset+len(node.token_ids)-1 for node,offset in zip(result['plan'].invocations,result['plan'].offsets)
               if node.name!='writer']
    # Compare each physical row against the SAME layout with only the AD flag
    # changed. Distinct CPU GEMM rows can round differently despite equal input;
    # comparing sibling positions would test that unrelated arithmetic promise.
    all_live=replace(stopped,imports=reader.imports)
    live_plan=compile_graph([writer,reader,all_live],external_sizes={'wire':len(dependency.target_token_ids)})
    unchanged,_=execute_codec_graph(participants,{family:live_plan},[dependency],channel=channel)
    torch.testing.assert_close(result['h_final'],unchanged[family]['h_final'],rtol=0,atol=0)
    values=[model.adapter.up.weight,*channel.vae.posterior.parameters()]
    logits=model.logits(result['h_final'][:,positions]).float()
    losses=[F.cross_entropy(logits[:,index],torch.tensor([9],device=logits.device)) for index in range(2)]
    constant=torch.autograd.grad(losses[1],values,retain_graph=True,allow_unused=True)
    assert all(value is None or torch.count_nonzero(value)==0 for value in constant)
    live=torch.autograd.grad(losses[0],values,allow_unused=True)
    assert all(value is not None and value.norm()>0 for value in live)
    assert sum(entry['kind']=='semantic-codec' for entry in execution)==1


def test_semantic_cycle_rejects_before_any_model_execution(tmp_path,monkeypatch):
    participants,channel,plans,dependency=case(tmp_path,'lfm','lfm')
    writer,reader=plans['lfm'].invocations
    writer=replace(writer,imports=(ImportSpan(1,'wire',0,len(dependency.target_token_ids)),))
    # The public plan compiler sees an external codec input. The dependency
    # scheduler must still detect its actual source call cycle before forward.
    plans['lfm']=compile_graph([writer,reader],external_sizes={'wire':len(dependency.target_token_ids)})
    monkeypatch.setattr(participants['lfm'].model,'forward',lambda *a,**kw:pytest.fail('executed a cyclic codec graph'))
    with pytest.raises(ValueError,match='acyclic'):
        execute_codec_graph(participants,plans,[dependency],channel=channel)


@pytest.mark.parametrize('source,target',[('lfm','mellum'),('mellum','lfm')])
def test_empty_semantic_payload_retains_real_control_trajectory_without_invented_tokens(tmp_path,source,target):
    participants,channel,plans,dependency=case(tmp_path,source,target,source_ids=())
    results,execution=execute_codec_graph(participants,plans,[dependency],channel=channel)
    expected=reference(participants,channel,dependency)
    torch.testing.assert_close(results[target]['h_final'],expected[target]['h_final'],rtol=0,atol=0)
    assert [entry['tokens'] for entry in execution if entry['kind']=='semantic-target-baseline']==[2]
    assert dependency.source_token_ids==dependency.target_token_ids==()
