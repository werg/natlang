"""Real native adapter application, gradient and durability contracts."""
from dataclasses import replace
import pytest
import torch

from test_native_full_updates import source
from natlang_neuralese.model.native_model import load_native_model, model_state, CHECKPOINT_SCHEMA
from natlang_neuralese.model.native_artifact import NativeTensorArtifact, encode_native_tensor_artifact, decode_native_tensor_artifact
from natlang_neuralese.model.native_adapters import create_adapter, run_native, resolve_adapters, target_parameters, effective_identity
from natlang_neuralese.model.shared_state import compile_graph, Invocation
from natlang_neuralese.serve import load_native_engine
from natlang_neuralese.serve.native_grad import native_graph_gradient
from natlang_neuralese.serve.native_optim import native_optimizer_step


def engine_fixture(tmp_path, family, *, service_template=False, context=None):
    import os
    device=os.environ.get('NATLANG_NATIVE_ADAPTER_TEST_DEVICE','cpu')
    dtype = source(tmp_path/'base', family)
    if context is not None:
        from transformers import AutoConfig
        config=AutoConfig.from_pretrained(tmp_path/'base')
        config.max_position_embeddings=context
        config.save_pretrained(tmp_path/'base')
    if service_template:
        from transformers import AutoTokenizer
        from tokenizers.pre_tokenizers import WhitespaceSplit
        tokenizer=AutoTokenizer.from_pretrained(tmp_path/'base')
        tokenizer.backend_tokenizer.pre_tokenizer=WhitespaceSplit()
        last_only=' and loop.last' if family=='lfm' else ''
        tokenizer.chat_template=("{% for message in messages %}{{ message['role'] }}: "
            "{% if message['role'] == 'assistant' and message.get('reasoning_content')"+last_only+" %}"
            "{{ message['reasoning_content'] }} {% endif %}{{ message['content'] }} {% endfor %}"
            "{% if add_generation_prompt %}assistant: {% endif %}")
        tokenizer.save_pretrained(tmp_path/'base')
    model, _, _, config = load_native_model(tmp_path/'base', device=device, dtype=dtype, rank=4)
    checkpoint = tmp_path/'native.pt'
    torch.save({'schema':CHECKPOINT_SCHEMA,'model_config':config,**model_state(model)},checkpoint)
    return load_native_engine(str(checkpoint), device=device, dtype=dtype)


def nonzero_adapter(engine, kind='xs'):
    available = target_parameters(engine.shared_model)
    name = next(name for name in available if name.endswith('q_proj.weight'))
    record = create_adapter(engine, {'kind':kind,'rank':2,'parameter_names':[name]})
    artifact = record.artifact
    values = {name:torch.ones_like(value)*.04 for name,value in artifact.tensors.items()}
    return engine.store.put_native_tensor_artifact(replace(artifact,tensors=values)), name


@pytest.mark.parametrize('family',['lfm','mellum'])
@pytest.mark.parametrize('kind',['xs'])
def test_native_adapter_cached_graph_gradient_and_exact_restoration(tmp_path,family,kind):
    engine = engine_fixture(tmp_path,family)
    adapter,name = nonzero_adapter(engine,kind)
    entries = [{'id':adapter.id,'scale':.75}]
    model = engine.shared_model
    originals = dict(model.named_parameters())
    values = {name:value.detach().clone() for name,value in originals.items()}
    plan = compile_graph([Invocation('answer',(4,5,6,7))])
    baseline = model(plan)['h_final'].detach()
    coefficient = adapter.artifact.tensors[name].clone().requires_grad_(True)
    def execute():
        graph = model(plan)['h_final']
        cache = None
        for token in (4,5,6,7):
            output=engine._native_forward([token],cache)
            cache=output['cache']
        torch.testing.assert_close(output['h_final'][:,-1],graph[:,-1],atol=5e-3 if family=='mellum' else 3e-6,rtol=5e-3 if family=='mellum' else 2e-5)
        loss=model.logits(graph).float().square().mean()
        gradient=torch.autograd.grad(loss,coefficient)[0]
        assert gradient.norm()>0
        return graph.detach(), gradient
    adapted,gradient=run_native(engine,execute,entries,leaves={adapter.id:{name:coefficient}})
    assert not torch.equal(adapted,baseline)
    assert all(dict(model.named_parameters())[name] is value for name,value in originals.items())
    assert all(torch.equal(value,values[name]) for name,value in model.named_parameters())
    assert torch.equal(model(plan)['h_final'],baseline)
    def fail():
        assert dict(model.named_parameters())[name] is not originals[name]
        raise RuntimeError('fixture callback failed')
    with pytest.raises(RuntimeError,match='fixture callback failed'):
        run_native(engine,fail,entries)
    assert all(dict(model.named_parameters())[name] is value for name,value in originals.items())
    raw=encode_native_tensor_artifact(adapter.artifact)
    restored,ident=decode_native_tensor_artifact(raw)
    assert ident==adapter.id
    assert restored.source_artifact_id==adapter.artifact.source_artifact_id
    altered=replace(adapter.artifact,native_identity={**engine.native_identity,'checkpoint_sha256':'1'*64})
    wrong=engine.store.put_native_tensor_artifact(altered)
    with pytest.raises(ValueError,match='exact live checkpoint'):
        resolve_adapters(engine,[wrong.id])
    engine.store.close()


@pytest.mark.parametrize('family',['lfm','mellum'])
@pytest.mark.parametrize('algorithm',['sgd','adam'])
def test_adapter_graph_gradients_optimizer_continuation_and_native_state_identity(tmp_path,family,algorithm):
    engine=engine_fixture(tmp_path,family)
    adapter,name=nonzero_adapter(engine)
    original_id=adapter.id
    roles=engine.role_tokens
    row={'schema':'natlang.neuralese-native-graph/1','id':'adapter-learn','split':'train',
         'source_groups':['adapter-service-fixture'],'tokenizer_fingerprint':engine.native_identity['tokenizer_fingerprint'],
         'nodes':[{'name':'answer','token_ids':[4,roles.produce_open,5,6,roles.produce_close,7],
                   'target_weights':[0,1,1,1,1,1],'producing':[[2,4]]}]}
    reference=torch.nn.Parameter(adapter.artifact.tensors[name].clone())
    optimizer=(torch.optim.Adam([reference],lr=.01) if algorithm=='adam' else
               torch.optim.SGD([reference],lr=.01,momentum=.8))
    state=None
    for step in range(1,4):
        gradient=native_graph_gradient(engine,{'schema':'natlang.native-gradient-request/1','graph':row,
            'native_identity':engine.native_identity,'adapters':[adapter.id],
            'target':{'kind':'adapter','id':adapter.id}},None)
        artifact=engine.store.get_native_tensor_artifact(gradient['gradient']['id']).artifact
        assert artifact.kind=='adapter-gradient' and artifact.source_artifact_id==adapter.id
        assert artifact.tensors[name].norm()>0
        optimizer.zero_grad(set_to_none=True)
        reference.grad=artifact.tensors[name].clone()
        optimizer.step()
        updated=native_optimizer_step(engine,{'schema':'natlang.native-optimizer-request/1',
            'native_identity':engine.native_identity,'gradient_id':gradient['gradient']['id'],
            'optimizer_state_id':state,'optimizer':{'name':algorithm,'learning_rate':.01,'momentum':.8}},None)
        adapter=engine.store.get_native_tensor_artifact(updated['adapter']['id'])
        assert updated['step']==step
        assert torch.equal(adapter.artifact.tensors[name],reference)
        state=updated['optimizer_state_id']
    assert adapter.id!=original_id
    written=engine.native_write_text('t5 t6',adapters=[adapter.id])
    block=engine.store.get_native_state(written['id']).block
    assert block.checkpoint_sha256==effective_identity(engine,resolve_adapters(engine,[adapter.id]))['checkpoint_sha256']
    assert block.adapter_binding['base_native_identity']==engine.native_identity
    assert not engine._native_exact(block)
    assert run_native(engine,lambda:engine._native_exact(block),[adapter.id])
    from natlang_neuralese.serve.store import encode_native_state,decode_native_state
    restored,ident=decode_native_state(encode_native_state(block))
    assert restored.adapter_binding==block.adapter_binding and ident==written['id']
    engine.store.close()


def test_actual_mellum_expert_buffer_adapter_and_coefficient_gradient(tmp_path):
    engine=engine_fixture(tmp_path,'mellum')
    names=[name for name,value in target_parameters(engine.shared_model).items() if value.ndim==3]
    assert names, 'actual Mellum frozen expert banks must be exposed, not omitted from named_parameters'
    name=names[0]
    adapter=create_adapter(engine,{'rank':2,'parameter_names':[name]})
    adapter=engine.store.put_native_tensor_artifact(replace(adapter.artifact,
        tensors={name:torch.full_like(adapter.artifact.tensors[name],.2)}))
    leaf=adapter.artifact.tensors[name].clone().requires_grad_(True)
    ids=[4,5,6,7]
    plan=compile_graph([Invocation('answer',tuple(ids))])
    original=dict(engine.shared_model.named_buffers())[name]
    baseline=engine.shared_model(plan)['h_final'].detach()
    def execute():
        hidden=engine.shared_model(plan)['h_final']
        cache=None
        for token in ids:
            result=engine._native_forward([token],cache)
            cache=result['cache']
        torch.testing.assert_close(hidden[:,-1],result['h_final'][:,-1],atol=.006,rtol=.006)
        grad=torch.autograd.grad(hidden.float().square().sum(),leaf)[0]
        assert grad.norm()>0
        return hidden.detach()
    adapted=run_native(engine,execute,[adapter.id],leaves={adapter.id:{name:leaf}})
    assert not torch.equal(adapted,baseline)
    assert dict(engine.shared_model.named_buffers())[name] is original
    engine.store.close()


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_actual_http_typed_adapter_learning_scoring_generation_and_portable_restart(tmp_path,monkeypatch,family):
    import json,subprocess,threading
    from pathlib import Path
    from natlang_neuralese.serve.http import serve
    engine=engine_fixture(tmp_path,family,service_template=True)
    target=next(name for name in target_parameters(engine.shared_model) if name.endswith('q_proj.weight'))
    roles=engine.role_tokens
    fixture={'identity':engine.native_identity,'target':target,'graph':{
        'schema':'natlang.neuralese-native-graph/1','id':'real-http-adapter-learning','split':'train',
        'source_groups':['native-adapter-service-fixture'],'tokenizer_fingerprint':engine.native_identity['tokenizer_fingerprint'],
        'nodes':[{'name':'answer','token_ids':[4,roles.produce_open,5,6,roles.produce_close,7],
                  'target_weights':[0,1,1,1,1,1],'producing':[[2,4]]}]}}
    binding=tmp_path/'service.json';binding.write_text(json.dumps(fixture))
    monkeypatch.setattr(engine,'_native_token',lambda seq:(2,-.1))
    engine.start();server=serve(engine,'127.0.0.1',0)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    endpoint=f'http://127.0.0.1:{server.server_address[1]}'
    try:
        repo=Path(__file__).resolve().parents[2]
        result=subprocess.run(['node',str(repo/'ts-host/test/fixtures/native-adapter-service.mjs'),endpoint,
            str(tmp_path/'archive'),str(binding)],capture_output=True,text=True)
        assert result.returncode==0,result.stdout+'\n'+result.stderr
        assert json.loads(result.stdout)['restored'] is True
    finally:
        server.shutdown();server.server_close();engine.stop();engine.store.close()


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_adapted_real_dag_all_hidden_logits_and_coefficient_adjoints_match_reference(tmp_path,family):
    from test_shared_state import tiny,graph
    from natlang_neuralese.model.shared_state import SharedStateModel
    from types import SimpleNamespace
    import threading
    import os
    port=tiny(family).to(os.environ.get('NATLANG_NATIVE_ADAPTER_TEST_DEVICE','cpu'))
    model=SharedStateModel(port,(60,61,62,63),rank=4).eval()
    torch.nn.init.normal_(model.adapter.up.weight,std=.03)
    for parameter in model.parameters():parameter.requires_grad_(False)
    from natlang_neuralese.serve.store import TensorStore
    engine=SimpleNamespace(shared_model=model,native_execution_lock=threading.RLock(),store=TensorStore(),
        native_identity={'checkpoint_sha256':'1'*64,'tokenizer_fingerprint':'2'*64,
                         'embedding_weight_sha256':'3'*64,'architecture':family})
    adapter,name=nonzero_adapter(engine)
    coefficient=adapter.artifact.tensors[name].clone().requires_grad_(True)
    plan=graph()
    def execute(reference=False):
        result=model(plan,reference=reference)
        logits=model.logits(result['h_final'])
        last=len(plan.invocations[-1].token_ids)
        loss=logits[:,-last:].square().mean()
        gradient=torch.autograd.grad(loss,coefficient)[0]
        assert gradient.norm()>0
        return result['h_final'].detach(),result['embeddings'].detach(),logits.detach(),gradient.detach()
    actual=run_native(engine,execute,[adapter.id],leaves={adapter.id:{name:coefficient}})
    reference=run_native(engine,lambda:execute(True),[adapter.id],leaves={adapter.id:{name:coefficient}})
    for observed,expected in zip(actual,reference):
        torch.testing.assert_close(observed,expected,atol=3e-6,rtol=2e-5)
    engine.store.close()


def test_model_lock_blocks_concurrent_unadapted_reader_and_retention_follows_bases(tmp_path):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    engine=engine_fixture(tmp_path,'lfm')
    adapter,name=nonzero_adapter(engine)
    baseline=engine._native_forward([4,5],None)['h_final'].detach()
    entered,release,reader=threading.Event(),threading.Event(),threading.Event()
    def adapted():
        def operation():
            entered.set()
            assert release.wait(10)
            return engine._native_forward([4,5],None)['h_final'].detach()
        return run_native(engine,operation,[adapter.id])
    def unadapted():
        assert entered.wait(10)
        reader.set()
        return engine._native_forward([4,5],None)['h_final'].detach()
    with ThreadPoolExecutor(max_workers=2) as pool:
        first=pool.submit(adapted)
        second=pool.submit(unadapted)
        assert reader.wait(10)
        assert not second.done()
        release.set()
        assert not torch.equal(first.result(),baseline)
        assert torch.equal(second.result(),baseline)
    engine.store.collect({adapter.id})
    assert engine.store.get_native_tensor_artifact(adapter.artifact.source_artifact_id) is not None
    engine.store.collect(set())
    assert engine.store.get_native_tensor_artifact(adapter.id) is None
    engine.store.close()


def test_native_adapter_preserves_tied_parameters_and_rejects_incomplete_optimizer(tmp_path):
    import threading
    from types import SimpleNamespace
    from test_shared_state import tiny
    from natlang_neuralese.model.shared_state import SharedStateModel
    from natlang_neuralese.serve.store import TensorStore
    from natlang_neuralese.model.native_adapters import optimize_adapter
    port=tiny('lfm')
    original=port.layers[1].self_attn.q_proj.weight
    port.layers[3].self_attn.q_proj.weight=original
    model=SharedStateModel(port,(60,61,62,63),rank=4)
    engine=SimpleNamespace(shared_model=model,native_execution_lock=threading.RLock(),store=TensorStore(),
        native_identity={'checkpoint_sha256':'1'*64,'tokenizer_fingerprint':'2'*64,
                         'embedding_weight_sha256':'3'*64,'architecture':'lfm'})
    adapter,name=nonzero_adapter(engine)
    def inspect_ties():
        assert port.layers[1].self_attn.q_proj.weight is port.layers[3].self_attn.q_proj.weight
        assert port.layers[1].self_attn.q_proj.weight is not original
    run_native(engine,inspect_ties,[adapter.id])
    assert port.layers[1].self_attn.q_proj.weight is original
    assert port.layers[3].self_attn.q_proj.weight is original
    gradient=NativeTensorArtifact('adapter-gradient',engine.native_identity,
        {name:torch.ones_like(adapter.artifact.tensors[name])},adapter.id)
    first=optimize_adapter(engine,gradient,{'name':'adam'},None)
    state=engine.store.get_native_tensor_artifact(first['optimizer_state_id']).artifact
    broken=engine.store.put_native_tensor_artifact(replace(state,tensors={key:value for key,value in state.tensors.items() if key!='m.'+name}))
    updated=engine.store.get_native_tensor_artifact(first['adapter']['id'])
    next_gradient=replace(gradient,source_artifact_id=updated.id)
    with pytest.raises(ValueError,match='incomplete'):
        optimize_adapter(engine,next_gradient,{'name':'adam'},broken.id)
    with pytest.raises(ValueError,match='exact coefficient parent'):
        optimize_adapter(engine,gradient,{'name':'adam'},first['optimizer_state_id'])
    engine.store.close()


def test_mellum_warm_deployed_cache_never_ignores_or_leaks_expert_adapter(tmp_path):
    from natlang_neuralese.maple.generation_gate import deployed_cache
    from natlang_neuralese.maple.ternary import QUANT_MIX
    engine=engine_fixture(tmp_path,'mellum')
    available=target_parameters(engine.shared_model)
    name=next(name for name,value in available.items() if value.ndim==3)
    original=available[name]
    adapter=create_adapter(engine,{'rank':2,'parameter_names':[name]})
    adapter=engine.store.put_native_tensor_artifact(replace(adapter.artifact,
        tensors={name:torch.full_like(adapter.artifact.tensors[name],.2)}))
    with torch.no_grad(),deployed_cache():
        base=engine._native_forward([4,5,6],None)['h_final'].clone()
        cache=QUANT_MIX['cache']
        saved={key:tuple(value.data_ptr() for value in values) for key,values in cache.items()}
        adapted=run_native(engine,lambda:engine._native_forward([4,5,6],None)['h_final'].clone(),[adapter.id])
        assert not torch.equal(adapted,base)
        assert QUANT_MIX['cache'] is cache
        assert saved=={key:tuple(value.data_ptr() for value in values) for key,values in cache.items()}
        assert torch.equal(engine._native_forward([4,5,6],None)['h_final'],base)
        assert target_parameters(engine.shared_model)[name] is original
    assert QUANT_MIX['cache'] is None
    engine.store.close()


def test_native_model_lora_basis_is_sign_canonical_for_matrix_and_batched_weights():
    from natlang_neuralese.model.native_adapter_math import singular_bases
    torch.manual_seed(20261016)
    for shape in ((11,7),(3,11,7)):
        weight=torch.randn(shape)
        u,_,vh=torch.linalg.svd(weight,full_matrices=False)
        left,right=u[...,:3],vh[...,:3,:].transpose(-2,-1)
        signs=torch.sign(left.gather(-2,left.abs().argmax(-2,keepdim=True))).clamp_min(0)*2-1
        actual=singular_bases(weight,3)
        assert torch.equal(actual[0],left*signs)
        assert torch.equal(actual[1],right*signs)
@pytest.mark.parametrize('family',['lfm','mellum'])
def test_adapted_stored_state_import_gradients_require_the_exact_effective_model(tmp_path,family):
    from natlang_neuralese.serve.chat import RequestError
    engine=engine_fixture(tmp_path,family,service_template=True)
    adapter,name=nonzero_adapter(engine)
    produced=engine.native_write_text('t5 t6',adapters=[adapter.id])
    block=engine.store.get_native_state(produced['id']).block
    count=len(block.token_ids)
    assert count==2
    row={'schema':'natlang.neuralese-native-graph/1','id':'adapted-import-learning','split':'train',
         'source_groups':['native-adapter-service-fixture'],'tokenizer_fingerprint':engine.native_identity['tokenizer_fingerprint'],
         'external_states':{'prior':{'id':produced['id'],'token_count':count}},
         'nodes':[{'name':'consumer','token_ids':[4,engine.role_tokens.consume_open,engine.role_tokens.consume_close,5,6],
                   'target_weights':[0,0,0,1,1],
                   'imports':[{'at':2,'source':'prior','start':0,'stop':count}]}]}
    request={'schema':'natlang.native-gradient-request/1','graph':row,'native_identity':engine.native_identity,
             'adapters':[adapter.id],'target':{'kind':'adapter','id':adapter.id}}
    result=native_graph_gradient(engine,request,None)
    gradient=engine.store.get_native_tensor_artifact(result['gradient']['id']).artifact
    assert gradient.tensors[name].norm()>0
    request['target']={'kind':'native-state','id':produced['id']}
    result=native_graph_gradient(engine,request,None)
    gradient=engine.store.get_native_tensor_artifact(result['gradient']['id']).artifact
    assert gradient.kind=='state-gradient' and any(value.norm()>0 for value in gradient.tensors.values())
    with pytest.raises(RequestError,match='foreign native state checkpoint'):
        native_graph_gradient(engine,{**request,'adapters':None},None)
    engine.store.close()
