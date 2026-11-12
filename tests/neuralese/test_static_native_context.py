import copy
import pytest
from test_own_output_supervision import objective,turn,USER,ASSISTANT,FP
from test_shared_state import tiny
from natlang_neuralese.data.static_native_context import graphify,verify_graphification
from natlang_neuralese.data.native_graph import parse_graph,graph_loss
from natlang_neuralese.model.shared_state import SharedStateModel
from natlang_neuralese.train.own_output_supervision import graph_weights,apply_to_example

def row():
    ids=turn(USER,[20]*25+[21]*25+[22]*25)+turn(ASSISTANT,[23,24,25])
    return {'schema':'natlang.neuralese-native-graph/1','id':'original','split':'train','source_groups':['world'],
      'tokenizer_fingerprint':FP,'nodes':[{'name':'text','token_ids':ids,'target_weights':[0.]+[.05]*(len(ids)-1)}],
      'conversion':{'role_interpretation':'whole-trajectory-chat-roles','runtime_opening_assistant_turns':0}}

def test_static_graph_preserves_tokens_objective_and_imports_causally():
    original=row();obj=objective(mass='token');child,proof=graphify(original,obj,max_tokens=32,consumer_context=6,state_tokens=2)
    assert proof['producer_count']>=2 and child['id']==original['id']
    assert verify_graphification(original,child,obj)
    assert graph_weights(original,obj)==graph_weights(child,obj)
    assert [t for n in child['nodes'] for t in n['token_ids']]==original['nodes'][0]['token_ids']
    ex=apply_to_example(parse_graph(child,tokenizer_fingerprint=FP),child,obj)
    model=SharedStateModel(tiny('lfm'),marker_ids=(58,59,60,61),rank=4)
    model.port.hf.requires_grad_(True)
    loss,_=graph_loss(model,ex,checkpoint_layers=False,chunk_tokens=32)
    loss.backward()
    grad=model.port.hf.get_input_embeddings().weight.grad
    assert grad is not None
    assert all(grad[token].abs().sum()>0 for token in (20,21,22))

@pytest.mark.parametrize('mutation',['edge','producer','weights','provenance'])
def test_static_graph_rejects_forgery(mutation):
    original=row();obj=objective(mass='token');child,_=graphify(original,obj,max_tokens=32,consumer_context=6,state_tokens=2)
    bad=copy.deepcopy(child)
    if mutation=='edge':bad['nodes'][-1]['imports'][0]['start']+=1
    elif mutation=='producer':bad['nodes'][0]['token_ids'][0]+=1
    elif mutation=='weights':bad['conversion']['static_native_context']['effective_weights'][-2]+=1
    else:bad['source_groups']=['different']
    with pytest.raises(ValueError):verify_graphification(original,bad,obj)

class WordTokenizer:
    """Deterministic text/token map: every token id decodes to a word, and words encode into small in-vocab ids."""
    def decode(self,ids,skip_special_tokens=False):
        return ' '.join(f'w{i}' for i in ids if not (skip_special_tokens and i<20))+'.'
    def encode(self,text,add_special_tokens=False):
        import hashlib
        return [40+int(hashlib.md5(word.encode()).hexdigest(),16)%12 for word in text.split()]


def test_seeded_static_graph_exports_a_chunk_summary_with_provenance_under_bounded_producer_loss():
    original=row();obj=objective(mass='token')
    child,proof=graphify(original,obj,max_tokens=32,consumer_context=6,state_tokens=2,tokenizer=WordTokenizer(),seed_tokens=4)
    meta=child['conversion']['static_native_context']
    assert meta['schema']=='natlang.static-native-context-graph/2' and proof['schema']=='natlang.static-native-context-proof/2'
    assert verify_graphification(original,child,obj)
    producers=child['nodes'][:-1];spans=meta['producer_spans']
    assert len(producers)>=2 and all(len(n['token_ids'])<=32 for n in producers)
    for node,(a,b),seed in zip(producers,spans,meta['seeds']):
        n=b-a;k=seed['tokens']
        # Original chunk tokens come first and stay unsupervised; the summary is the exported, supervised span.
        assert node['token_ids'][:n]==original['nodes'][0]['token_ids'][a:b]
        assert node['target_weights']==[0.]*n+[1.]*k and node['producing']==[[n,n+k]]
        assert 1<=k<=4 and seed['provenance']=='mechanical:chunk-summary' and seed['parts']
    assert [t for n,(a,b) in zip(producers,spans) for t in n['token_ids'][:b-a]]+child['nodes'][-1]['token_ids']==original['nodes'][0]['token_ids']
    assert [(e['start'],e['stop']) for e in child['nodes'][-1]['imports']]==[tuple(n['producing'][0]) for n in producers]
    weights,categories=graph_weights(child,obj)
    ex=apply_to_example(parse_graph(child,tokenizer_fingerprint=FP),child,obj)
    assert len(weights)==len(ex.target_weights)==sum(len(n['token_ids']) for n in child['nodes'])
    total=sum(s['tokens'] for s in meta['seeds']);seed_mass=meta['seed_weight']*total
    parent_mass=sum(graph_weights(original,obj)[0])
    assert 0<seed_mass<=meta['seed_share']*parent_mass*(1+1e-9)
    assert proof['imported_state_tokens']==total and proof['seed_provenance']=={'mechanical:chunk-summary':len(producers)}
    model=SharedStateModel(tiny('lfm'),marker_ids=(58,59,60,61),rank=4)
    model.port.hf.requires_grad_(True)
    loss,_=graph_loss(model,ex,checkpoint_layers=False,chunk_tokens=32)
    loss.backward()
    grad=model.port.hf.get_input_embeddings().weight.grad
    assert grad is not None and all(grad[token].abs().sum()>0 for token in (20,21,22))


def test_seed_is_question_selected_and_names_exact_content():
    from natlang_neuralese.data.operator_seeds import chunk_summary_seed
    text='The vault opens at nine. Gears turn slowly. The pump P-17 failed on 2026-03-04. Nothing else happened.'
    focused=chunk_summary_seed(text,'when did pump P-17 fail')
    assert focused['parts'][0]=='question-overlap' and 'P-17 failed' in focused['text']
    assert focused['provenance']=='mechanical:chunk-summary'
    blind=chunk_summary_seed(text)
    assert 'question-overlap' not in blind['parts'] and 'P-17' in blind['text']
    assert chunk_summary_seed(text,'when did pump P-17 fail')==focused


@pytest.mark.parametrize('mutation',['seed-token','seed-weight','seed-provenance','import','producing','tail-fragment'])
def test_seeded_static_graph_rejects_forgery(mutation):
    original=row();obj=objective(mass='token')
    child,_=graphify(original,obj,max_tokens=32,consumer_context=6,state_tokens=2,tokenizer=WordTokenizer(),seed_tokens=4)
    bad=copy.deepcopy(child);meta=bad['conversion']['static_native_context']
    if mutation=='seed-token':bad['nodes'][0]['token_ids'][-1]+=1
    elif mutation=='seed-weight':bad['nodes'][0]['target_weights'][-1]=0.
    elif mutation=='seed-provenance':meta['seeds'][0]['provenance']=''
    elif mutation=='import':bad['nodes'][-1]['imports'][0]['stop']-=1
    elif mutation=='producing':bad['nodes'][0]['producing']=[[len(bad['nodes'][0]['token_ids'])-2,len(bad['nodes'][0]['token_ids'])]]
    else:bad['nodes'][0]['token_ids'][0]+=1
    with pytest.raises(ValueError):verify_graphification(original,bad,obj)


def test_seeded_geometry_falls_back_through_the_same_position_error():
    original=row()
    with pytest.raises(ValueError,match='exceed the position limit'):
        graphify(original,objective(mass='token'),max_tokens=32,consumer_context=6,state_tokens=8,tokenizer=WordTokenizer(),seed_tokens=8)
    with pytest.raises(ValueError,match='invalid static native context geometry'):
        graphify(original,objective(mass='token'),max_tokens=32,consumer_context=6,state_tokens=2,tokenizer=WordTokenizer(),seed_tokens=20)


def test_tokenizerless_graphify_keeps_the_legacy_tail_export_for_replay():
    original=row();child,_=graphify(original,objective(mass='token'),max_tokens=32,consumer_context=6,state_tokens=2)
    assert child['conversion']['static_native_context']['schema']=='natlang.static-native-context-graph/1'
    assert 'seeds' not in child['conversion']['static_native_context']
