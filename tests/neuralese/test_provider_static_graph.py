import hashlib
from types import SimpleNamespace
import pytest
from natlang_neuralese.data.provider_static_graph import _compile_loaded_episode

class Tokenizer:
    def __call__(self,text,**kw):return {'input_ids':[ord(c)+100 for c in text]}
class Renderer:
    tokenizer=Tokenizer();role_tokens=SimpleNamespace(produce_open=1,produce_close=2,consume_open=3,consume_close=4);stop_ids={9};specials=()
    def _template_tokens(self,text,nonce=''):return [ord(c)+100 for c in text]
    def render_request(self,messages,tools):
        parts=messages[0]['content']; blocks=[p['id'] for p in parts if p['type']=='native_state']; return SimpleNamespace(segments=['p',*sum(([i,'q'] for i in range(len(blocks))),[])],blocks=blocks,escape_nonce='')

def fixture():
    body='body'; digest=hashlib.sha256(body.encode()).hexdigest(); block='nzt1_'+('a'*52)
    base={'source_row_id':'teacher-program:fixture','source_row_sha256':'a'*64,'source_file_sha256':'b'*64,'trace_sha256':'c'*64,'plan_sha256':'d'*64,'runtime_receipt_sha256':'e'*64,'source_groups':['train:fixture'],'split':'train','actor':{'model':'gpt-6-luna'},'target_supervision':True}
    producer={**base,'source_turn_index':0,'target_text_spans':[{'source_turn_id':'producer/1','marker':'M'}],'typed_text_writes':[]}
    consumer={**base,'source_turn_index':1,'typed_text_writes':[{'block_id':block,'writer_call_id':'producer/1','writer_node':'producer/1#1','body_sha256':digest,'context_occurrences':2}]}
    loaded0={'source_receipt':producer,'record':{'target':{'role':'assistant'}},'native_text_spans':[{'marker':'M','path':['content'],'text':body}],'typed_text_bodies':[],'audit_sha256':'f'*64}
    typed={'block_id':block,'body':body,'body_sha256':digest,'type':'Neuralese<string>','context_occurrences':2}
    loaded1={'source_receipt':consumer,'record':{'target':{'role':'assistant'}},'native_text_spans':[],'typed_text_bodies':[typed],'audit_sha256':'f'*64}
    return [(loaded0,[{'role':'user','content':[{'type':'text','text':'p'}]}],[]),(loaded1,[{'role':'user','content':[{'type':'teacher_text','id':block},{'type':'teacher_text','id':block}]}],[])]

def test_inline_writer_and_multiple_imports_preserve_target(monkeypatch):
    import natlang_neuralese.data.text_corpus as corpus
    monkeypatch.setattr(corpus,'tokenizer_fingerprint',lambda tokenizer:'1'*64)
    import natlang_neuralese.data.provider_static_renderer as render
    monkeypatch.setattr(render,'render_forced_message_plan',lambda renderer,message,tools,opening,spans: [7,1,*[ord(c)+100 for c in 'body'],2,8,9,10] if spans else [7,8,9,10])
    row, example=_compile_loaded_episode(fixture(),Renderer(),tokenizer_fingerprint='1'*64)
    assert len(row['nodes'])==2
    producer,consumer=row['nodes'];assert producer['name']=='producer/1#1'
    assert len(producer['producing'])==1 and len(consumer['imports'])==2
    assert all(edge['source']==producer['name'] for edge in consumer['imports'])
    assert sum(producer['target_weights'])==8 and sum(consumer['target_weights'])==2
    assert row['static_training_provenance']['scope'].startswith('Authenticated')

def test_body_tamper_fails_before_graph(monkeypatch):
    import natlang_neuralese.data.text_corpus as corpus
    monkeypatch.setattr(corpus,'tokenizer_fingerprint',lambda tokenizer:'1'*64)
    rows=fixture();rows[1][0]['typed_text_bodies'][0]['body']='tampered'
    with pytest.raises(ValueError,match='body identity'):_compile_loaded_episode(rows,Renderer(),tokenizer_fingerprint='1'*64)


def test_frozen_provider_runtime_rejects_external_code(tmp_path,monkeypatch):
    import sys,types
    from scripts.neuralese_data.provider_execution_runtime import require_frozen_provider_runtime
    import natlang_neuralese.common.runtime_inventory as inventory
    monkeypatch.setattr(inventory,'immutable_native_runtime_tree',lambda root:{'kind':'test'})
    fake=types.ModuleType('natlang_neuralese.bad_external');fake.__file__=__file__
    monkeypatch.setitem(sys.modules,'natlang_neuralese.bad_external',fake)
    with pytest.raises(ValueError,match='outside its frozen runtime'):require_frozen_provider_runtime(tmp_path)


def test_static_and_reference_graph_loss_and_gradients_match(monkeypatch):
    import copy,torch
    import natlang_neuralese.data.text_corpus as corpus
    import natlang_neuralese.data.provider_static_renderer as render
    from natlang_neuralese.data.native_graph import graph_loss,parse_graph
    from natlang_neuralese.model.shared_state import SharedStateModel
    from test_shared_state import tiny
    monkeypatch.setattr(corpus,'tokenizer_fingerprint',lambda tokenizer:'1'*64)
    monkeypatch.setattr(render,'render_forced_message_plan',lambda renderer,message,tools,opening,spans: [7,1,*[ord(c)+100 for c in 'body'],2,8,9,10] if spans else [7,8,9,10])
    row,_=_compile_loaded_episode(fixture(),Renderer(),tokenizer_fingerprint='1'*64)
    for node in row['nodes']:node['token_ids']=[x%50 for x in node['token_ids']]
    source,consumer=row['nodes']
    expected={'schema':'natlang.neuralese-native-graph/1','id':'reference','split':'train','source_groups':['train:fixture'],'tokenizer_fingerprint':'1'*64,'nodes':[{'name':source['name'],'token_ids':[12,7,1,48,11,0,21,2,8],'target_weights':[0,1,1,1,1,1,1,1,1],'producing':[[3,7]]},{'name':consumer['name'],'token_ids':[12,3,4,13,3,4,13,7,8],'target_weights':[0,0,0,0,0,0,0,1,1],'imports':[{'at':2,'source':source['name'],'start':3,'stop':7},{'at':5,'source':source['name'],'start':3,'stop':7}]}]}
    a=parse_graph(row,tokenizer_fingerprint='1'*64);b=parse_graph(expected,tokenizer_fingerprint='1'*64)
    model=SharedStateModel(tiny('lfm'),marker_ids=(58,59,60,61),rank=4);reference=copy.deepcopy(model)
    loss,_=graph_loss(model,a,checkpoint_layers=False,chunk_tokens=16);other,_=graph_loss(reference,b,checkpoint_layers=False,chunk_tokens=16)
    loss.backward();other.backward();torch.testing.assert_close(loss,other,rtol=0,atol=0)
    for (name,p),(name2,q) in zip(model.named_parameters(),reference.named_parameters(),strict=True):
        assert name==name2 and ((p.grad is None)==(q.grad is None))
        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)
    disconnected=copy.deepcopy(expected);disconnected['nodes'][1]['imports']=[]
    isolated=copy.deepcopy(reference);isolated.zero_grad(set_to_none=True)
    cut,_=graph_loss(isolated,parse_graph(disconnected,tokenizer_fingerprint='1'*64),checkpoint_layers=False,chunk_tokens=16)
    cut.backward()
    assert any(p.grad is not None and q.grad is not None and not torch.equal(p.grad,q.grad)
               for p,q in zip(model.parameters(),isolated.parameters(),strict=True))


def test_static_target_renderer_tracks_serving_implementation():
    import ast,inspect,textwrap
    from natlang_neuralese.serve.engine import Engine
    from natlang_neuralese.data.provider_static_renderer import render_forced_message_plan
    serving=ast.parse(textwrap.dedent(inspect.getsource(Engine._forced_message_plan))).body[0].body
    start=next(i for i,node in enumerate(serving) if isinstance(node,ast.If) and 'not isinstance(message, dict)' in ast.unparse(node.test))
    body=serving[start:]
    body=[node for node in body if not (isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='spans')]
    static=ast.parse(textwrap.dedent(inspect.getsource(render_forced_message_plan))).body[0].body
    if isinstance(static[0],ast.Expr) and isinstance(static[0].value,ast.Constant):static=static[1:]
    for node in ast.walk(ast.Module(body=static,type_ignores=[])):
        if isinstance(node,ast.ImportFrom) and node.module=='serve.chat' and node.level==2:
            node.module='chat';node.level=1
    assert ast.dump(ast.Module(body=body,type_ignores=[]))==ast.dump(ast.Module(body=static,type_ignores=[]))
