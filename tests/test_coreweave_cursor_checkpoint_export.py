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


@pytest.mark.parametrize('failure',[None,'incomplete','conflict'])
def test_remote_actual_full_checkpoint_global_step_independent_pin_and_failure_cleanup(tmp_path,failure):
    import torch
    root=tmp_path/'production';slot=root/'rounds/lfm/production/round-000001/learner/checkpoint.pt';slot.parent.mkdir(parents=True)
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



def test_canonical_glob_rejects_foreign_role_checkpoint_symlink_before_copy(tmp_path):
    root=tmp_path/'production';foreign=root/'rounds/mellum/production/round-1/learner/checkpoint.pt';foreign.parent.mkdir(parents=True);foreign.write_bytes(b'foreign checkpoint never decoded')
    alias=root/'rounds/lfm/production/round-1/learner/checkpoint.pt';alias.parent.mkdir(parents=True);alias.symlink_to(foreign)
    pins=tmp_path/'pins';pins.mkdir()
    code=module.remote_code(role='lfm',root=root,known={},last=0,extra=None,pin_directory=pins,minimum_available_bytes=0)
    result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,timeout=30)
    assert result.returncode!=0 and 'escaped its role namespace'in result.stderr
    assert not list(pins.iterdir())and foreign.read_bytes()==b'foreign checkpoint never decoded'
