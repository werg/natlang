import hashlib,json,types
import pytest
from scripts.coreweave.native_control_watch import attempt_control


def test_attempt_records_selection_and_retries_source_free(tmp_path,monkeypatch):
    helper=tmp_path/'helper.py';helper.write_text('print("hello")');sha=hashlib.sha256(helper.read_bytes()).hexdigest()
    selection={'helper':{'path':str(helper),'sha256':sha},'target_helper':{'path':str(helper)},'target_selection':{'path':'/work/selection.json','sha256':'a'*64},'control_directory':'/work/control','role_prefix':'peer2','phase_directory':str(tmp_path)}
    pin={'path':'/work/pin','sha256':'b'*64,'step':2,'sandbox_id':'box'};state={'exports':[{}]};entry=state['exports'][0];journal=tmp_path/'journal';calls=[]
    class Sandbox:
        def exec(self,argv,**kwargs):
            assert 'os.nice(15)' in argv[2] and 'OMP_NUM_THREADS' in argv[2] and helper.read_text() in argv[2] and '--selection-sha256' in argv
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


def test_durable_retirement_intent_fences_source_free_control_retry(tmp_path):
    helper=tmp_path/'helper';helper.write_text('pass');sha=hashlib.sha256(helper.read_bytes()).hexdigest()
    selection={'helper':{'path':str(helper),'sha256':sha},'target_selection':{'sha256':'a'*64},'role_prefix':'peer2','phase_directory':str(tmp_path)}
    pin={'step':2,'sandbox_id':'box'}; entry={'control':{'attempts':1,'selection_sha256':'a'*64,'archive':{'path':'/work/archive','sha256':'c'*64,'bytes':23}}};state={'exports':[entry]}; journal=tmp_path/'journal';calls=[]
    (tmp_path/'standalone-retired-cloud-objects.json').write_text(json.dumps({'operations':[{'control_role':'peer2-step2-publication-control','phase':'intent'}]}))
    attempt_control(pin,None,entry,state,journal,selection,lambda p,v:p.write_text(json.dumps(v)),presence=lambda *a:True,launch=lambda *a:calls.append(a))
    assert entry['control']['status']=='retired' and not calls


def test_missing_hook_reports_failure_without_executing_untrusted_code():
    import ast
    from pathlib import Path
    source=Path(__file__).resolve().parents[2]/'scripts/coreweave/native_standalone_control_watch.py'
    node=next(x for x in ast.parse(source.read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='control_attempt')
    env={'hook':None,'hook_error':'ValueError: changed helper'}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),env)
    with pytest.raises(ValueError,match='selected control hook unavailable'):
        env['control_attempt']({},None,{})


def test_complete_publication_requires_exact_private_selector_owner_get(tmp_path,monkeypatch):
    import ast,sys
    from pathlib import Path
    source=Path(__file__).resolve().parents[2]/'scripts/coreweave/native_standalone_control_watch.py'
    node=next(x for x in ast.parse(source.read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='refresh_publication')
    class Api:
        def run(self,*args):return self
        def file(self,*args):return types.SimpleNamespace(size=2)
    monkeypatch.setitem(sys.modules,'wandb',types.SimpleNamespace(Api=lambda:Api()))
    intent=tmp_path/'intent';intent.write_text(json.dumps({'sha256':'b'*64}))
    def receipt(role,sha):
        path=tmp_path/('coreweave-production-'+role+'-private-export-receipt-20261014.json')
        path.write_text(json.dumps({'role':role,'private':True,'run':'r','name':role,'identity':{'sha256':sha,'bytes':2},'independent_verification':{'readback_location':'owner','sha256':sha,'bytes':2}}));return path
    full=receipt('full','b'*64);control=receipt('control','c'*64);selector=receipt('selector','a'*64)
    def path_factory(x):return tmp_path if str(x)=='/home/werg/data/dgxops' else Path(x)
    env={'selection':{'publication_dependencies':[]},'selector_receipt':selector,'selector_receipt_sha':hashlib.sha256(selector.read_bytes()).hexdigest(),'selection_sha':'a'*64,'Path':path_factory,'json':json,'write_json_durable':lambda *args:None,'journal':tmp_path/'j','s':{}}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),env)
    entry={'full_backup_verified':True,'role':'full','intent':str(intent),'control':{'role':'control','status':'independently_verified','archive':{'sha256':'c'*64}}}
    env['refresh_publication'](entry);assert entry['publication_complete']
    env['selection_sha']='d'*64;env['refresh_publication'](entry);assert not entry['publication_complete']


def test_owner_binding_rejects_changed_sandbox_plan_or_process():
    import ast
    from pathlib import Path
    source=Path(__file__).resolve().parents[2]/'scripts/coreweave/native_standalone_control_watch.py'
    node=next(x for x in ast.parse(source.read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='validate_owner')
    env={};exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),env)
    accepted={'accepted':True,'owner_id':'o','generation':4,'boot_id':'b','pid':7,'process_start_ticks':'12'}
    value={'sandbox_id':'s','plan_sha256':'p','runtime_commit':'r','source_sha256':'t','accepted':accepted}
    env['validate_owner'](value,value)
    with pytest.raises(ValueError,match='plan_sha256'):env['validate_owner']({**value,'plan_sha256':'other'},value)
    with pytest.raises(ValueError,match='pid'):env['validate_owner']({**value,'accepted':{**accepted,'pid':8}},value)


def test_minimum_step_never_scans_older_or_equal_parent():
    import ast
    from pathlib import Path
    source=Path(__file__).resolve().parents[2]/'scripts/coreweave/native_standalone_control_watch.py'
    node=next(x for x in ast.parse(source.read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='minimum_step')
    env={};exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),env)
    assert env['minimum_step'](700,800,900)==900
    assert env['minimum_step'](1000,800,900)==1000
    with pytest.raises(ValueError):env['minimum_step'](1,2,-1)


def test_remote_scanner_owner_guard_rejects_dead_reused_and_reset_processes():
    import ast,subprocess,sys
    from pathlib import Path
    source=Path(__file__).resolve().parents[2]/'scripts/coreweave/native_standalone_control_watch.py'
    node=next(x for x in ast.parse(source.read_text()).body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='OWNER_GUARD' for t in x.targets));guard=ast.literal_eval(node.value)
    prefix="import json,os,pathlib,sys\npid=os.getpid();fields=pathlib.Path('/proc/%d/stat'%pid).read_text().rsplit(')',1)[1].split();boot=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip();expected={'pid':pid,'boot_id':boot,'process_start_ticks':fields[19]}\n"
    for alteration,accepted in [('',True),("expected['pid']=999999999\n",False),("expected['process_start_ticks']='0'\n",False),("expected['boot_id']='other'\n",False)]:
        code=prefix+alteration+"sys.argv=['x','2','0',json.dumps(expected)]\n"+guard+"\nprint('SCANNER_REACHED')\n"
        output=subprocess.check_output([sys.executable,'-c',code],text=True)
        assert ('SCANNER_REACHED' in output)==accepted
        if not accepted:assert json.loads(output)['ready'] is False
