from types import SimpleNamespace
from scripts.teacher_behavior_probe import assess


def event(name, args, kind='ok'):
    import json
    return {'action': name+' '+json.dumps(args), 'kind':kind}


def test_correct_value_cannot_hide_skipped_or_redirected_required_call():
    call={'function':'size_of','to':'let/size','inputs':{'items':'args/items'},'done':1}
    case={'steps':[('call',call)],'expected':{'size':1},'failure':None,'effects':[]}
    out=SimpleNamespace(kind='done',detail='')
    assert not assess(case,out,{'size':1},[],[])['pass']
    redirected={**call,'to':'return/size'}
    assert not assess(case,out,{'size':1},[event('call',redirected,'done')],[])['pass']
    assert assess(case,out,{'size':1},[event('call',call,'done')],[])['pass']
    log=[event('mark_done',{'start':1}),event('call',call,'done')]
    assert assess(case,out,{'size':1},log,[])['premature_marks']==[1]


def test_validation_failure_is_separate_from_explicit_correct_failure():
    case={'steps':[],'expected':None,'failure':'error','effects':[]}
    score=assess(case,SimpleNamespace(kind='quiesced',detail='validation failed: wrong type'),None,[],[])
    assert score['safe_validation_failure'] and not score['pass']
    score=assess(case,SimpleNamespace(kind='quiesced',detail='error: incompatible required type'),None,[],[])
    assert score['pass']


def test_requested_direct_call_can_fail_at_runtime_without_becoming_a_fudge():
    direct={'function':'size_of','to':'return','inputs':{'items':'args/items'}}
    case={'steps':[],'expected':None,'failure':'error','effects':[], 'allowed_failure_call':direct}
    out=SimpleNamespace(kind='quiesced',detail='validation failed: type-does-not-fit-slot')
    assert assess(case,out,None,[event('call',direct,'rejected')],[])['pass']
    redirected={**direct,'to':'let/result'}
    assert not assess(case,out,None,[event('call',redirected,'rejected')],[])['pass']


def test_batched_flow_fixtures_execute_through_real_runtime():
    from scripts.teacher_behavior_probe import flow_cases
    from natlang.gen.policy import ReferenceAgent
    from natlang.gen.programs import Plan
    from natlang.runtime import Runtime
    from natlang.values import load_program, dump
    for case in flow_cases(883):
        rt=Runtime(lambda lam:ReferenceAgent(Plan('calls',steps=case['steps']),[]))
        out,val=rt.run_root(load_program(case['root']))
        assert out.kind=='done' and dump(val)==case['expected']


def test_missing_read_is_explicit_and_empty_text_is_preserved():
    from natlang.runtime import Runtime, Session
    from natlang.types import TypeEnv
    from natlang.values import load_program
    for args in ({}, {'document':''}):
        root=load_program({'$lambda':{'type':'Lambda<{ document?: Text }, Text>',
            'instructions':'Copy document.', 'args':args}})
        result=Session(Runtime(None),root,TypeEnv())._op_read({'path':'args/document'})
        assert result.kind=='ok'
        assert result.text=='' if args else 'not supplied' in result.text


def test_initializer_quote_normalization_preserves_text_and_rejects_wrappers():
    import pytest
    from natlang.diag import Reject
    from natlang.runtime import Runtime, Session
    from natlang.types import TypeEnv
    from natlang.values import load_program
    session=Session(Runtime(None),load_program({'$lambda':{'type':'Lambda<{}, Num>',
        'instructions':'Return 5.'}}),TypeEnv())
    assert session._init('5','Num')=={'init':5}
    assert session._init('5','Text')=={'init':'5'}
    assert session._init('"5"','Text')=={'init':'"5"'}
    assert session._init(5,'Text')=={'init':'5'}
    with pytest.raises(Reject):
        session._init('{"value":5}','Num')


def test_fold_accepts_equivalent_quoted_literal_initializer():
    from scripts.teacher_behavior_probe import flow_cases
    from natlang.gen.policy import ReferenceAgent
    from natlang.gen.programs import Plan
    from natlang.runtime import Runtime
    from natlang.values import load_program
    case=list(flow_cases(885))[1]
    call={**case['steps'][0][1], 'init':'5'}
    rt=Runtime(lambda lam:ReferenceAgent(Plan('calls',steps=[('call',call)]),[]))
    out,val=rt.run_root(load_program(case['root']))
    assert assess(case,out,val,[event('call',call,'done')],[])['pass']


def test_reply_validates_return_and_open_lines():
    from natlang.decoder import ChatTurn
    from natlang.runtime import Runtime, Session
    from natlang.surface import ToolSurface
    from natlang.types import TypeEnv
    from natlang.values import load_program
    from natlang.tool_agent import ToolAgent
    from scripts.generate_agent_support import support_cases
    case=next(c for c in support_cases(881,0) if c['name']=='binding_repair')
    session=Session(Runtime(None),load_program(case['root']),TypeEnv())
    surface=ToolSurface(marks=True)
    assert 'done' not in [tool['function']['name'] for tool in surface.tools(session)]
    for name,args in case['steps']:
        assert surface.apply(session,name,args).kind in ('done','ok')
    assert surface.pending(session) == []
    assert session.finish()
    class Scripted:
        def chat(self,*args,**kwargs):
            return ChatTurn(text='Finished.',completion_tokens=1)
    root=load_program({'$lambda':{'type':'Lambda<{}, Num>','instructions':'Return 7.','effects':['out.emit']}})
    rt=Runtime(lambda lam:ToolAgent(Scripted()))
    out,value=rt.run_root(root)
    assert out.kind=='quiesced' and 'return' in out.detail and rt.emitted==[]


def test_blank_lines_do_not_block_natural_completion():
    from natlang.decoder import ChatTurn
    from natlang.runtime import Runtime
    from natlang.tool_agent import ToolAgent
    from natlang.values import load_program

    class Scripted:
        def __init__(self, mark):
            self.turns = iter([ChatTurn([('write', {'path': 'return', 'type': 'Num', 'value': 7,
                                                       **({'done': 1} if mark else {})})], completion_tokens=1,
                                        raw_response={'choices': [{'message': {'reasoning_content': 'write seven'}}]}),
                               ChatTurn(text='Finished.', completion_tokens=1),
                               ChatTurn(text='Finished.', completion_tokens=1),
                               ChatTurn(text='Finished.', completion_tokens=1)])

        def chat(self, *_args, **_kwargs):
            return next(self.turns)

    def root(mark_second_line=False):
        return load_program({'$lambda': {'type': 'Lambda<{}, Num>',
            'instructions': 'Return 7.\n\n   \n# comment' + ('\nSay that you finished.' if mark_second_line else ''),
            'codebase': {'unused': {'args': {}, 'returns': 'Num', 'instructions': 'Return 1.'}}}})

    transcript, teacher_turns = [], []
    out, value = Runtime(lambda lam: ToolAgent(Scripted(True), transcript=transcript,
                                               teacher_turns=teacher_turns)).run_root(root())
    assert out.kind == 'done' and value == 7
    assert transcript[-1] == {'role': 'assistant', 'content': 'Finished.'}
    assert teacher_turns[0]['response']['choices'][0]['message']['reasoning_content'] == 'write seven'
    assert teacher_turns[0]['executions'][0]['kind'] == 'ok'
    assert teacher_turns[1]['text'] == 'Finished.'
    out, _ = Runtime(lambda lam: ToolAgent(Scripted(True))).run_root(root(True))
    assert out.kind == 'quiesced' and 'unfinished lines: 5' in out.detail


def test_faithful_wrong_type_write_can_fail_without_fudging_but_wrong_value_cannot():
    write={'path':'return','type':'Text','value':'blue'}
    case={'steps':[],'expected':None,'failure':'error','effects':[], 'allowed_failure_write':write}
    out=SimpleNamespace(kind='quiesced',detail='validation failed: type-mismatch')
    assert assess(case,out,None,[event('write',write,'rejected')],[])['pass']
    assert not assess(case,out,None,[event('write',{**write,'value':{'args':False}},'rejected')],[])['pass']


def test_json_text_transport_does_not_mutate_tools_or_unwrap_values(monkeypatch):
    import json
    import urllib.request
    from natlang.decoder import LlamaServerDecoder
    from natlang.runtime import Runtime, Session
    from natlang.surface import ToolSurface
    from natlang.types import TypeEnv
    from natlang.values import load_program
    root=load_program({'$lambda':{'type':'Lambda<{}, { size: Num }>','instructions':'Return size 7.'}})
    session=Session(Runtime(None),root,TypeEnv())
    tools=ToolSurface().tools(session)
    before=json.dumps(tools)
    captured=[]
    class Response:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def read(self):return json.dumps({'choices':[{'message':{'tool_calls':[{'type':'function','function':
            {'name':'write','arguments':json.dumps({'path':'return','type':'{ size: Num }','value':'{"size":7}'})}}]}}],
            'usage':{'completion_tokens':1}}).encode()
    def open_(req,**kwargs):
        captured.append(json.loads(req.data));return Response()
    monkeypatch.setattr(urllib.request,'urlopen',open_)
    turn=LlamaServerDecoder(json_text_values=True).chat([],tools,temperature=0)
    wire=next(t for t in captured[0]['tools'] if t['function']['name']=='write')
    assert wire['function']['parameters']['properties']['value']['type']=='string'
    assert json.dumps(tools)==before
    assert turn.calls[0][1]['value']=='{"size":7}'
    assert session.apply(*turn.calls[0]).kind=='ok' and root.ret=={'size':7}
