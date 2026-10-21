import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import pytest

spec=importlib.util.spec_from_file_location('cursor_export',Path(__file__).parents[1]/'scripts/coreweave/cursor_checkpoint_export.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_actual_owner_output_is_bound_role_scoped_and_cannot_escape(tmp_path):
    control=tmp_path/'control';out=control/'lfm378-ordinary-current'/'learner';out.mkdir(parents=True)
    argv=tmp_path/'argv.json';argv.write_text(json.dumps(['python','--task-eval','','--out',str(out)]))
    ref={'path':str(argv),'sha256':hashlib.sha256(argv.read_bytes()).hexdigest(),'allowed_output_root':str(out.parent)}
    assert module.bound_output_checkpoint(ref,'lfm',production_root=tmp_path,control_root=control)==out/'checkpoint.pt'
    for changed in ({**ref,'sha256':'0'*64},{**ref,'allowed_output_root':str(control/'mellum-current')},
                    {**ref,'allowed_output_root':str(tmp_path)}):
        with pytest.raises(ValueError):module.bound_output_checkpoint(changed,'lfm',production_root=tmp_path,control_root=control)
    argv.write_text(json.dumps(['--out',str(out),'--out',str(out)]));ref['sha256']=hashlib.sha256(argv.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='exactly one'):module.bound_output_checkpoint(ref,'lfm',production_root=tmp_path,control_root=control)


@pytest.mark.parametrize('role', ['lfm', 'mellum'])
def test_recovered_owner_output_retains_role_and_exact_lineage_boundary(tmp_path, role):
    recovery=tmp_path/'recovered-training'
    lineage=recovery/(role+'-full-recipe-26320')
    output=lineage/'native_full_sft';output.mkdir(parents=True)
    argv=tmp_path/'argv.json'
    def reference(target, allowed):
        argv.write_text(json.dumps(['python','--out',str(target)]))
        return {'path':str(argv),'sha256':hashlib.sha256(argv.read_bytes()).hexdigest(),
                'allowed_output_root':str(allowed)}
    assert module.bound_output_checkpoint(reference(output,lineage),role,
        production_root=tmp_path)==output/'checkpoint.pt'
    foreign=recovery/('mellum-other' if role=='lfm' else 'lfm-other')
    for target,allowed in ((foreign/'native_full_sft',foreign),(output,recovery),
                           (recovery/(role+'-other')/'native_full_sft',lineage)):
        with pytest.raises(ValueError):
            module.bound_output_checkpoint(reference(target,allowed),role,production_root=tmp_path)
    escaped=lineage/'escaped';escaped.symlink_to(tmp_path/'unrelated',target_is_directory=True)
    with pytest.raises(ValueError,match='escapes'):
        module.bound_output_checkpoint(reference(escaped/'native_full_sft',lineage),role,
            production_root=tmp_path)


def test_pending_export_recovery_never_recopies_or_advances_before_full_readback(tmp_path):
    chosen={'source':'canonical/learner/checkpoint.pt','pin':'immutable/pin.pt','step':12,'bytes':123}
    calls=[]
    def observe(**kwargs):calls.append('observe');return {'chosen':chosen,'observations':{'source':{'step':12}}}
    def failed(*args):calls.append('failed export');raise OSError('temporary private transport failure')
    with pytest.raises(OSError):module.run_once(tmp_path,'mellum',observe=observe,export=failed)
    paths=module.paths_for(tmp_path,'mellum');assert paths['pending'].exists()and not paths['state'].exists()
    def forbidden(**kwargs):raise AssertionError('must resume pending pin, not recopy')
    def export(value,path):
        calls.append('resumed export');assert value==chosen
        return {'source':chosen['pin'],'identity':{'sha256':'f'*64,'bytes':123},
                'independent_verification':{'sha256':'f'*64,'bytes':123}}
    result=module.run_once(tmp_path,'mellum',observe=forbidden,export=export)
    assert result['verified_step']==12 and calls==['observe','failed export','resumed export']
    assert not paths['pending'].exists()and json.loads(paths['state'].read_text())['last_verified_step']==12


@pytest.mark.parametrize('field',['source','bytes','sha256','readback-bytes'])
def test_wrong_export_readback_preserves_pending_and_prior_state(tmp_path,field):
    chosen={'source':'slot','pin':'pin','step':13,'bytes':123};receipt={'source':'pin','identity':{'sha256':'f'*64,'bytes':123},'independent_verification':{'sha256':'f'*64,'bytes':123}}
    if field=='source':receipt['source']='foreign'
    elif field=='bytes':receipt['identity']['bytes']=124
    elif field=='sha256':receipt['independent_verification']['sha256']='a'*64
    else:receipt['independent_verification']['bytes']=124
    with pytest.raises(ValueError,match='full readback'):
        module.run_once(tmp_path,'lfm',observe=lambda **kw:{'chosen':chosen,'observations':{}},export=lambda *a:receipt)
    paths=module.paths_for(tmp_path,'lfm');assert paths['pending'].exists()and not paths['state'].exists()


@pytest.mark.parametrize('relative',['production/round-000001/learner','cycles/000003/supervised/001/learner','cycles/000003/supervised/001/retry-predecessor-v2/learner'])
@pytest.mark.parametrize('failure',[None,'incomplete','conflict'])
def test_remote_actual_full_checkpoint_global_step_independent_pin_and_failure_cleanup(tmp_path,failure,relative):
    import torch
    root=tmp_path/'production';slot=root/'rounds/lfm'/relative/'checkpoint.pt';slot.parent.mkdir(parents=True)
    torch.save({'step':42,'training':{'step':5,'optimizer':{}if failure=='incomplete'else{'moment':torch.tensor([1.25])},
        'rng':{'cpu':torch.get_rng_state()},'source_receipts':[{'path':'actual-source'}]}},slot)
    pins=tmp_path/'pins';pins.mkdir();target=pins/'lfm-native-checkpoint-step-42.pt'
    if failure=='conflict':target.write_bytes(b'preserved conflicting immutable target')
    code=module.remote_code(role='lfm',root=root,known={},last=0,extra=None,pin_directory=pins,minimum_available_bytes=0)
    result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,timeout=30)
    assert not list(pins.glob('.lfm-cursor-observation-*.pt'))
    if failure:
        assert result.returncode!=0
        if failure=='conflict':assert target.read_bytes()==b'preserved conflicting immutable target'
    else:
        assert result.returncode==0,result.stderr
        observed=json.loads(result.stdout);assert observed['chosen']['step']==42
        assert target.read_bytes()==slot.read_bytes()and target.stat().st_ino!=slot.stat().st_ino
        saved=torch.load(target,weights_only=True);assert saved['training']['step']==5
        assert saved['training']['optimizer']['moment'].item()==1.25



@pytest.mark.parametrize('relative',['production/round-1/learner','cycles/000003/supervised/001/learner','cycles/000003/supervised/001/retry-predecessor-v2/learner'])
def test_canonical_glob_rejects_foreign_role_checkpoint_symlink_before_copy(tmp_path,relative):
    root=tmp_path/'production';foreign=root/'rounds/mellum/production/round-1/learner/checkpoint.pt';foreign.parent.mkdir(parents=True);foreign.write_bytes(b'foreign checkpoint never decoded')
    alias=root/'rounds/lfm'/relative/'checkpoint.pt';alias.parent.mkdir(parents=True);alias.symlink_to(foreign)
    pins=tmp_path/'pins';pins.mkdir()
    code=module.remote_code(role='lfm',root=root,known={},last=0,extra=None,pin_directory=pins,minimum_available_bytes=0)
    result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,timeout=30)
    assert result.returncode!=0 and 'escaped its role namespace'in result.stderr
    assert not list(pins.iterdir())and foreign.read_bytes()==b'foreign checkpoint never decoded'


@pytest.mark.parametrize('sidecar,sha',[('writer.py',None),(None,'f'*64),('writer.py','not-a-sha')])
def test_sidecar_binding_requires_complete_explicit_options(sidecar,sha):
    with pytest.raises(ValueError,match='sidecar'):
        module.remote_code(role='lfm',root='/root',known={},last=0,extra=None,
            pin_directory='/pins',minimum_available_bytes=0,actor_view_directory='/views',
            actor_view_runtime='/runtime',actor_view_sidecar=sidecar,actor_view_sidecar_sha256=sha)


def test_default_remote_code_is_byte_identical_when_sidecar_unused(tmp_path):
    import subprocess
    raw=subprocess.check_output(['git','show','904834c47:scripts/coreweave/cursor_checkpoint_export.py'],cwd=Path(__file__).parents[1],text=True)
    original_path=tmp_path/'published_actor_exporter.py';original_path.write_text(raw)
    spec=importlib.util.spec_from_file_location('published_actor_exporter',original_path)
    original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
    options=dict(role='lfm',root='/root',known={},last=0,extra=None,pin_directory='/pins',minimum_available_bytes=0)
    import inspect
    old_bound=inspect.getsource(original.bound_output_checkpoint)
    new_bound=inspect.getsource(module.bound_output_checkpoint)
    expected=original.remote_code(**options).replace(old_bound,new_bound).replace(
        'cycles/*/supervised/*/learner/checkpoint.pt','cycles/*/supervised/*/**/learner/checkpoint.pt')
    assert module.remote_code(**options)==expected
    options.update(actor_view_directory='/views',actor_view_runtime='/runtime')
    expected=original.remote_code(**options).replace(old_bound,new_bound).replace(
        'cycles/*/supervised/*/learner/checkpoint.pt','cycles/*/supervised/*/**/learner/checkpoint.pt')
    assert module.remote_code(**options)==expected


def test_completed_remote_observation_recovers_after_lost_rpc_without_decode(tmp_path):
    root=tmp_path/'source';root.mkdir();pin=root/'pin.pt';pin.write_bytes(b'actual independently pinned full state')
    marker="print(json.dumps({'at':time.time(),'observations':updates,'chosen':chosen,'cursor_binding_only':c}))"
    counter=root/'calls';code="import time,pathlib,json\np=pathlib.Path("+repr(str(counter))+");p.write_text(str(int(p.read_text())+1)if p.exists()else'1')\nupdates={'source':{'step':42}};chosen={'pin':"+repr(str(pin))+",'step':42};c=None\n"+marker
    wrapped=module.recoverable_remote_code(code,role='mellum',root=root)
    first=subprocess.run([sys.executable,'-c',wrapped],text=True,capture_output=True,check=True)
    second=subprocess.run([sys.executable,'-c',wrapped],text=True,capture_output=True,check=True)
    assert json.loads(first.stdout)==json.loads(second.stdout)and counter.read_text()=='1'
    report=next(root.glob('mellum-cursor-observation-*.json'));record=json.loads(report.read_text());record['request_sha256']='0'*64;report.write_text(json.dumps(record))
    failed=subprocess.run([sys.executable,'-c',wrapped],text=True,capture_output=True)
    assert failed.returncode!=0 and 'request differs'in failed.stderr and counter.read_text()=='1'


def test_completed_remote_observation_rejects_pin_mutation_before_replay(tmp_path):
    root=tmp_path/'source';root.mkdir();pin=root/'pin.pt';pin.write_bytes(b'original')
    marker="print(json.dumps({'at':time.time(),'observations':updates,'chosen':chosen,'cursor_binding_only':c}))"
    code="import time,json,pathlib\nupdates={};chosen={'pin':"+repr(str(pin))+",'step':1};c=None\n"+marker
    wrapped=module.recoverable_remote_code(code,role='lfm',root=root)
    subprocess.run([sys.executable,'-c',wrapped],check=True,capture_output=True)
    import os
    previous=pin.stat();pin.write_bytes(b'changed!');os.utime(pin,ns=(previous.st_atime_ns,previous.st_mtime_ns))
    assert pin.stat().st_size==previous.st_size and pin.stat().st_mtime_ns==previous.st_mtime_ns
    failed=subprocess.run([sys.executable,'-c',wrapped],text=True,capture_output=True)
    assert failed.returncode!=0 and 'pin changed'in failed.stderr


def test_actual_full_remote_observation_replay_keeps_pin_without_source_slot(tmp_path):
    import torch
    root=tmp_path/'source';slot=root/'rounds/lfm/production/round-1/learner/checkpoint.pt';slot.parent.mkdir(parents=True)
    torch.save({'step':7,'training':{'optimizer':{'moment':torch.tensor([2.])},'rng':{'cpu':torch.get_rng_state()},'source_receipts':[{'actual':True}]}},slot)
    pins=tmp_path/'pins';pins.mkdir()
    raw=module.remote_code(role='lfm',root=root,known={},last=0,extra=None,pin_directory=pins,minimum_available_bytes=0)
    wrapped=module.recoverable_remote_code(raw,role='lfm',root=root)
    first=subprocess.run([sys.executable,'-c',wrapped],capture_output=True,text=True,check=True,timeout=30)
    slot.unlink()
    replay=subprocess.run([sys.executable,'-c',wrapped],capture_output=True,text=True,check=True,timeout=30)
    assert json.loads(replay.stdout)==json.loads(first.stdout)
    assert (pins/'lfm-native-checkpoint-step-7.pt').exists()


def test_empty_remote_observation_does_not_hide_a_later_actual_checkpoint(tmp_path):
    import torch
    root=tmp_path/'source';root.mkdir();pins=tmp_path/'pins';pins.mkdir()
    raw=module.remote_code(role='lfm',root=root,known={},last=0,extra=None,pin_directory=pins,minimum_available_bytes=0)
    wrapped=module.recoverable_remote_code(raw,role='lfm',root=root)
    empty=subprocess.run([sys.executable,'-c',wrapped],capture_output=True,text=True,check=True,timeout=30)
    assert json.loads(empty.stdout)['chosen']is None
    assert not list(root.glob('lfm-cursor-observation-*.json'))
    slot=root/'rounds/lfm/production/round-1/learner/checkpoint.pt';slot.parent.mkdir(parents=True)
    torch.save({'step':8,'training':{'optimizer':{'moment':torch.tensor([2.])},'rng':{'cpu':torch.get_rng_state()},'source_receipts':[{'actual':True}]}},slot)
    found=subprocess.run([sys.executable,'-c',wrapped],capture_output=True,text=True,check=True,timeout=30)
    assert json.loads(found.stdout)['chosen']['step']==8


def test_owner_following_uses_new_generation_journal_without_erasing_old(tmp_path):
    generations=['1'*64,'2'*64]
    calls=[];exports=[]
    def observe(**kwargs):
        calls.append(kwargs)
        index=len(calls)-1
        generation=generations[min(index,1)]
        chosen={'source':f'round-{index}/learner/checkpoint.pt','pin':f'pin-{index}.pt',
                'step':10+index,'bytes':123}
        return {'owner':{'generation_sha256':generation,
                         'manifest':{'path':f'/owner-{index}.json','sha256':generation,'bytes':12}},
                'chosen':chosen,'observations':{}}
    def export(chosen,path):
        exports.append(chosen)
        return {'source':chosen['pin'],'identity':{'sha256':'f'*64,'bytes':chosen['bytes']},
                'independent_verification':{'sha256':'f'*64,'bytes':chosen['bytes']}}
    first=module.run_owner_follow_once(tmp_path,'lfm',observe=observe,export=export)
    second=module.run_owner_follow_once(tmp_path,'lfm',observe=observe,export=export)
    assert first['verified_step']==10 and second['verified_step']==11
    assert calls[0]['known_owner_generation'] is None
    assert calls[1]['known_owner_generation']=='1'*64
    index=json.loads((tmp_path/'owned-publication-owner-index.json').read_text())
    assert index['current']['generation_sha256']=='2'*64
    assert {row['generation_sha256'] for row in index['generations']}==set(generations)
    assert module.paths_for(tmp_path/'owner-generations'/('1'*64),'lfm')['state'].exists()
    assert module.paths_for(tmp_path/'owner-generations'/('2'*64),'lfm')['state'].exists()
    assert len(exports)==2


def test_owned_output_path_may_follow_only_role_named_round_tree(tmp_path):
    production=tmp_path/'native-production';round_out=production/'shared/deployment/rounds/lfm/cycles/4/supervised/1/learner'
    round_out.mkdir(parents=True);argv=tmp_path/'argv.json';argv.write_text(json.dumps(['python','--out',str(round_out)]))
    ref={'path':str(argv),'sha256':hashlib.sha256(argv.read_bytes()).hexdigest(),
         'allowed_output_root':str(round_out.parent)}
    assert module.bound_output_checkpoint(ref,'lfm',production_root=production,
        control_root=production/'control')==round_out/'checkpoint.pt'
    foreign=production/'shared/deployment/rounds/mellum/cycles/4/supervised/1/learner';foreign.mkdir(parents=True)
    argv.write_text(json.dumps(['python','--out',str(foreign)]));ref.update(sha256=hashlib.sha256(argv.read_bytes()).hexdigest(),allowed_output_root=str(foreign.parent))
    with pytest.raises(ValueError,match='outside its role-owned'):
        module.bound_output_checkpoint(ref,'lfm',production_root=production,control_root=production/'control')


def test_owner_following_keeps_failed_old_upload_while_publishing_new_owner(tmp_path):
    calls=[]
    def observe(**kwargs):
        generation='1'*64 if not calls else '2'*64
        calls.append(kwargs)
        index=len(calls)-1
        return {'owner':{'generation_sha256':generation,
            'manifest':{'path':f'/manifest-{index}','sha256':generation,'bytes':10}},
            'chosen':{'source':f'source-{index}','pin':f'pin-{index}.pt','step':20+index,'bytes':10},
            'observations':{}}
    def export(chosen,path):
        if chosen['pin']=='pin-0.pt':raise OSError('old full GET is retrying')
        return {'source':chosen['pin'],'identity':{'sha256':'a'*64,'bytes':10},
            'independent_verification':{'sha256':'a'*64,'bytes':10}}
    with pytest.raises(OSError):
        module.run_owner_follow_once(tmp_path,'lfm',observe=observe,export=export)
    second=module.run_owner_follow_once(tmp_path,'lfm',observe=observe,export=export)
    assert second['verified_step']==21 and len(second['pending_failures'])==1
    assert second['pending_failures'][0]['generation_sha256']=='1'*64
    index=json.loads((tmp_path/'owned-publication-owner-index.json').read_text())
    assert index['current']['generation_sha256']=='2'*64
    old=module.paths_for(tmp_path/'owner-generations'/('1'*64),'lfm')
    new=module.paths_for(tmp_path/'owner-generations'/('2'*64),'lfm')
    assert old['pending'].exists() and new['state'].exists()


def test_owner_follower_current_private_export_runs_while_old_get_is_blocked(tmp_path):
    import threading
    generations=['1'*64,'2'*64];observations=[];export_calls=[]
    old_entered=threading.Event();release_old=threading.Event();new_finished=threading.Event()
    workers=module.OwnerFollowWorkers()
    def observe(**kwargs):
        index=len(observations);observations.append(kwargs);generation=generations[min(index,1)]
        return {'owner':{'generation_sha256':generation,
            'manifest':{'path':f'/manifest-{index}','sha256':generation,'bytes':10}},
            'chosen':{'source':f'source-{index}','pin':f'pin-{index}.pt','step':30+index,'bytes':10},
            'observations':{}}
    def export(chosen,path):
        export_calls.append((chosen,path))
        if len(export_calls)==1:raise OSError('first private GET fails after durable pending selection')
        if generations[0] in str(path):
            old_entered.set()
            if not release_old.wait(5):raise TimeoutError('old private GET fixture was not released')
        else:
            if not old_entered.wait(5):raise AssertionError('old export was not actually blocked before current export')
            new_finished.set()
        return {'source':chosen['pin'],'identity':{'sha256':'a'*64,'bytes':chosen['bytes']},
            'independent_verification':{'sha256':'a'*64,'bytes':chosen['bytes']}}
    with pytest.raises(RuntimeError,match='export worker failed'):
        module.run_owner_follow_once(tmp_path,'lfm',observe=observe,export=export,
            workers=workers,wait_current=True)
    assert module.paths_for(tmp_path/'owner-generations'/generations[0],'lfm')['pending'].exists()
    second=module.run_owner_follow_once(tmp_path,'lfm',observe=observe,export=export,
        workers=workers,wait_current=True)
    assert second['owner_generation']==generations[1]
    assert second['current_export']['verified_step']==31
    assert new_finished.is_set() and old_entered.is_set()
    assert not release_old.is_set()  # current full private export completed while old GET remained blocked
    release_old.set()
    old_result=workers.wait('generation:'+generations[0])
    assert old_result['verified_step']==30
    assert module.paths_for(tmp_path/'owner-generations'/generations[0],'lfm')['state'].exists()


def test_owner_follower_rotates_between_two_pending_old_generations_after_retry(tmp_path):
    import pytest
    old_generations=['1'*64,'2'*64]
    current_generation='3'*64
    owner_root=tmp_path/'owner-generations'
    index={'schema':'natlang.owned-publication-owner-index/1','role':'lfm',
        'generations':[{'generation_sha256':value} for value in old_generations],
        'current':None}
    (tmp_path/'owned-publication-owner-index.json').write_text(json.dumps(index))
    for generation in old_generations:
        paths=module.paths_for(owner_root/generation,'lfm')
        paths['pending'].parent.mkdir(parents=True,exist_ok=True)
        module.save(paths['pending'],{'source':'pin-'+generation,'pin':'pin-'+generation,
            'step':10,'bytes':4})
    workers=module.OwnerFollowWorkers()
    attempts=[]
    def observe(**kwargs):
        return {'owner':{'generation_sha256':current_generation,
            'manifest':{'path':'/manifest-current','sha256':current_generation,'bytes':10}},
            'chosen':{'source':'current-source','pin':'current-pin','step':11,'bytes':4},
            'observations':{}}
    def export(chosen,path):
        generation=Path(path).parent.name
        attempts.append(generation)
        if generation==old_generations[0]:
            raise OSError('temporary failure; retain this old pending export')
        return {'source':chosen['pin'],'identity':{'sha256':'a'*64,'bytes':chosen['bytes']},
            'independent_verification':{'sha256':'a'*64,'bytes':chosen['bytes']}}
    first=module.run_owner_follow_once(tmp_path,'lfm',observe=observe,export=export,
        workers=workers,wait_current=True)
    assert first['old_export_started']
    try:
        workers.wait('generation:'+old_generations[0])
    except RuntimeError as error:
        assert 'worker failed' in str(error)
    assert old_generations[0] in attempts
    assert module.paths_for(owner_root/old_generations[0],'lfm')['pending'].exists()
    second=module.run_owner_follow_once(tmp_path,'lfm',observe=observe,export=export,
        workers=workers,wait_current=True)
    assert second['old_export_started']
    workers.wait('generation:'+old_generations[1])
    assert old_generations[0] in attempts and old_generations[1] in attempts
    assert attempts.index(old_generations[0]) < attempts.index(old_generations[1])
    assert module.paths_for(owner_root/old_generations[0],'lfm')['pending'].exists()
    assert not module.paths_for(owner_root/old_generations[1],'lfm')['pending'].exists()


def test_current_owner_cli_builds_remote_control_root_without_path_alias(tmp_path,monkeypatch):
    import sys
    from types import SimpleNamespace
    from scripts.coreweave import cwctl
    monkeypatch.setitem(sys.modules,'cwctl',cwctl)
    captured={}
    class Rpc:
        def result(self,timeout):
            return SimpleNamespace(returncode=0,stdout=json.dumps({'owner':None,'chosen':None,'observations':{}}),stderr='')
    class Sandbox:
        def exec(self,argv,timeout_seconds):
            captured['code']=argv[2]
            return Rpc()
    monkeypatch.setattr(cwctl,'load_env_file',lambda:None)
    monkeypatch.setattr(cwctl,'require_key',lambda:None)
    monkeypatch.setattr(cwctl,'read_sandbox',lambda sandbox_id,patience:Sandbox())
    root=tmp_path/'native-production'
    state=tmp_path/'owner-journal'
    pointer=root/'control'/'lfm-current-owner.json'
    result=module.main(['--role','lfm','--sandbox-id','fixture','--root',str(root),
        '--state-dir',str(state),'--export-command','["true"]',
        '--publication-owner-pointer',str(pointer),'--once'])
    assert result==0
    assert str(root/'control') in captured['code']


def test_remote_owner_observer_code_compiles_and_waits_for_first_owner(tmp_path):
    from scripts.coreweave.owned_publication_export import owner_remote_code
    # The exporter passes pathlib paths from argparse. The observer source is
    # serialized for execution remotely, where PosixPath is not defined.
    code=owner_remote_code(tmp_path/'control'/'lfm-current-owner.json',role='lfm',
        production_root=tmp_path,control_root=tmp_path/'control')
    compile(code,'remote-owner-observer','exec')
    result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,check=True)
    assert json.loads(result.stdout)=={'owner':None,'chosen':None,'observations':{}}
    pointer=tmp_path/'control'/'lfm-current-owner.json';pointer.parent.mkdir(parents=True)
    pointer.symlink_to(tmp_path/'missing-target.json')
    code=owner_remote_code(str(pointer),role='lfm',production_root=str(tmp_path),
        control_root=str(tmp_path/'control'))
    result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
    assert result.returncode and 'pointer may not be a symlink' in result.stderr


def test_remote_owner_observer_accepts_real_native_preference_publication(tmp_path):
    import shutil
    import tempfile
    from natlang_neuralese.model.native_checkpoint_control import publication_selection
    from natlang_neuralese.rl import native_publication_owner as owner
    from scripts.coreweave.owned_publication_export import owner_remote_code

    runtime = Path(__file__).resolve().parents[1]
    production = runtime
    test_root = Path(tempfile.mkdtemp(prefix='.owned-publication-observer-', dir=runtime))
    parent = test_root / 'parent.pt'
    parent.write_bytes(b'full selected parent')
    output = test_root / 'rounds' / 'lfm' / 'cycles' / '000004' / 'multi-rft' / '000'
    learner = output / 'learner-kind'
    learner.mkdir(parents=True)
    (learner / 'checkpoint.pt').write_bytes(b'full saved child')
    selection = publication_selection(namespace=str(production),
        readonly_input_roots=[str(production), str(runtime)])
    control = production / 'control'
    control.mkdir(exist_ok=True)
    pointer = control / (test_root.name + '-lfm-current-owner.json')
    config = {'runtime': '/work/native-production/task-runtimes/ts-host',
        'model': 'lfm', 'native_actor_mode': 'raw-state',
        'native_checkpoint_publication_role': 'lfm',
        'native_checkpoint_publication_runtime_root': str(runtime),
        'native_checkpoint_publication': selection,
        'native_checkpoint_publication_owner_pointer': str(pointer),
        'request_id': 'preference-operation'}
    raw_command = ['env', f'PYTHONPATH={runtime}/training/neuralese:{runtime}',
        f'NATLANG_REPO={runtime}', '/usr/bin/python3.11', '-m', 'natlang_neuralese.train.native_preference',
        '--out', str(output / 'learner-kind'), '--continue-from', str(parent), '--checkpoint-publication',
        '--checkpoint-publication-namespace', selection['namespace'],
        '--checkpoint-publication-readonly-input-roots', json.dumps(selection['readonly_input_roots']),
        '--objective-spec', str(tmp_path / 'objective.json')]
    try:
        owner.emit_current(config, raw_command, out=output, selected_checkpoint=parent,
            runtime_root=config['native_checkpoint_publication_runtime_root'])
        code = owner_remote_code(str(pointer), role='lfm', production_root=str(production),
            control_root=str(control))
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)['chosen'] is None
    finally:
        pointer.unlink(missing_ok=True)
        shutil.rmtree(test_root, ignore_errors=True)
