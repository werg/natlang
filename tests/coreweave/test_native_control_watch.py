import hashlib,json,types
import pytest
from scripts.coreweave.native_control_watch import attempt_control


def test_attempt_records_selection_and_retries_source_free(tmp_path,monkeypatch):
    helper=tmp_path/'helper.py';helper.write_text('print("hello")');sha=hashlib.sha256(helper.read_bytes()).hexdigest()
    selection={'helper':{'path':str(helper),'sha256':sha},'target_helper':{'path':str(helper)},'target_selection':{'path':'/work/selection.json','sha256':'a'*64},'control_directory':'/work/control','role_prefix':'peer2','phase_directory':str(tmp_path)}
    pin={'path':'/work/pin','sha256':'b'*64,'step':2,'sandbox_id':'box'};state={'exports':[{}]};entry=state['exports'][0];journal=tmp_path/'journal';calls=[]
    class Sandbox:
        def exec(self,argv,**kwargs):
            assert argv[2]==helper.read_text() and '--selection-sha256' in argv
            result=types.SimpleNamespace(returncode=0,stdout='NATLANG_CONTROL_COLLECTOR_INVOKED\n'+json.dumps({'checkpoint_sha256':'b'*64,'archive':{'path':'/work/archive','sha256':'c'*64,'bytes':23},'control':{'path':'/work/proof','sha256':'d'*64}}),stderr='')
            return types.SimpleNamespace(result=lambda **kwargs:result)
    monkeypatch.setattr('scripts.coreweave.native_control_watch.subprocess.run',lambda *a,**k:types.SimpleNamespace(returncode=1))
    def persist(path,value):path.write_text(json.dumps(value))
    attempt_control(pin,Sandbox(),entry,state,journal,selection,persist,presence=lambda *a:False,launch=lambda *a:calls.append(a))
    assert entry['control']['selection_sha256']=='a'*64 and entry['control']['collector_invoked'] and len(calls)==1
    attempt_control(pin,None,entry,state,journal,selection,persist,presence=lambda *a:True,launch=lambda *a:calls.append(a))
    assert calls[-1][-1] is True
    with pytest.raises(ValueError,match='selection epoch'):
        attempt_control(pin,None,entry,state,journal,{**selection,'target_selection':{'path':'/work/selection.json','sha256':'e'*64}},persist,presence=lambda *a:True,launch=lambda *a:None)


def test_historical_unbound_entry_is_not_silently_relabelled(tmp_path):
    helper=tmp_path/'helper';helper.write_text('pass');sha=hashlib.sha256(helper.read_bytes()).hexdigest();selection={'helper':{'path':str(helper),'sha256':sha},'target_selection':{'sha256':'a'*64}}
    with pytest.raises(ValueError,match='historical control'):
        attempt_control({},None,{'control':{'attempts':1}},{},tmp_path/'j',selection,lambda *a:None,presence=lambda *a:False,launch=lambda *a:None)


def test_missing_hook_reports_failure_without_executing_untrusted_code():
    import ast
    from pathlib import Path
    source=Path(__file__).resolve().parents[2]/'scripts/coreweave/native_standalone_control_watch.py'
    node=next(x for x in ast.parse(source.read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='control_attempt')
    env={'hook':None,'hook_error':'ValueError: changed helper'}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),env)
    with pytest.raises(ValueError,match='selected control hook unavailable'):
        env['control_attempt']({},None,{})
