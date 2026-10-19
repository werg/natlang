"""Exact in-process checkpoint import preserves the existing RAW recipe debt."""
import json
import fcntl
import os
import sys
from pathlib import Path

import pytest

from natlang_neuralese.common.jsonio import write_json_durable as write
from natlang_neuralese.rl import native_raw_rounds as raw
from natlang_neuralese.train import native_resident as resident


@pytest.fixture
def case(tmp_path, monkeypatch):
    root=tmp_path/'raw-owner';production=root/'production';production.mkdir(parents=True)
    config_path=root/'config.json';config={'out':str(production),'raw_model_id':'mellum21:base-sha',
        'production_recipe':'recipe.json'};write(config_path,config)
    old=root/'old.pt';old.write_bytes(b'old-verified-complete-state')
    new=root/'new.pt';new.write_bytes(b'new-verified-complete-state')
    oldref=resident.binding(old);newref={**resident.binding(new),'bytes':new.stat().st_size}
    state_path=production/'state.json'
    state={'schema':'natlang.rl-production-state/1','round':9,'offset':31,
        'checkpoint':oldref['path'],'phase':'collect','trained_rounds':8,
        'collected_task_ids':['case-a','case-b'],'attempted_task_ids':['case-a','case-b','case-c'],
        'trained_task_ids':['case-a'],'trained_supervised_tokens':50,'trained_target_tokens':13,
        'production_complete':False,'collection_attempt':3,'group_stats':[{'groups':4}],
        'last_learner_progress':{'opaque':'preserve me'},'ready_marker':'stale'}
    write(state_path,state)
    cursor={'cycle':11,'phase':'rl','next_stage':2,'current_checkpoint':oldref,
        'task_cases':{'pending':['case-c','case-d']},'pending_collection':{'offset':31},
        'readiness':{'checkpoint_sha256':oldref['sha256']},'foundation_qualified':True,
        'opaque_cursor_debt':{'must':'survive'}}
    cursor_path=root/'cycle.json';write(cursor_path,cursor)
    wrapper=root/'owned-entry.py';wrapper.write_text('# in-process owner entry fixture\n')
    argv=['/usr/bin/python3','-u',str(wrapper),'fallback']
    trainer_args=['--out',str(root/'learner'),'--continue-from',str(new),'--stage-id','native_full_sft']
    args_file=root/'effective-trainer-argv-list.json';write(args_file,trainer_args)
    full_command=[sys.executable,'-u','-m','natlang_neuralese.train.native_state_sft',*trainer_args]
    command_file=root/'effective-trainer-command.json';write(command_file,full_command)
    gate_spec_path=root/'gate-spec.json';gate_spec={'trainer_arguments':trainer_args,
        'effective_trainer_argv':resident.binding(args_file),'parent':newref,
        'out':str(root/'learner'),'stage_id':'native_full_sft'};write(gate_spec_path,gate_spec)
    owner={'pid':12345,'boot_id':'boot-fixture','start_ticks':'987','argv_hex':'fixture'}
    from natlang_neuralese.rl import native_raw_handoff as handoff
    monkeypatch.setattr(handoff,'observe',lambda pid:dict(owner))
    monkeypatch.setattr(handoff,'process_argv',lambda identity:list(argv))
    package=Path(raw.__file__).resolve().parents[1]
    runtime_root=package.parents[2]
    runtime_code=resident.digest(resident.code_inventory(package))
    gate_spec.update(runtime=str(runtime_root),runtime_code_sha256=runtime_code)
    write(gate_spec_path,gate_spec)
    authority={'schema':'natlang.native-raw-current-owner/2','owner':owner,
        'wrapper':resident.binding(wrapper),'process_argv':list(argv),'config':resident.binding(config_path),
        'package':{'path':str(package),'code':resident.code_inventory(package)},
        'wrapper_configuration':resident.binding(gate_spec_path),
        'effective_argv_file':resident.binding(command_file),
        'effective_argv_binding':resident.binding(args_file),'selected_checkpoint':newref,
        'saved_state':{'checkpoint':newref,'global_step':24,'model_id':config['raw_model_id']}}
    loaded={'training':{'global_step':24,'optimizer':{'opaque':'already loaded'},'rng':{'opaque':'already loaded'}},
        'model_config':{'architecture':'mellum21','base_inventory_sha256':'base-sha'}}
    monkeypatch.setattr('natlang_neuralese.rl.native_raw_policy.require_raw_training',lambda value:None)
    return {'root':root,'config_path':config_path,'state_path':state_path,'cursor_path':cursor_path,
        'state':state,'cursor':cursor,'old':oldref,'new':newref,'authority':authority,'loaded':loaded,
        'owner':owner,'argv':argv,'receipt':root/'current-owner-import.json'}


def call(x,**kwargs):
    return raw.import_current_owner_checkpoint(x['config_path'],x['cursor_path'],x['new'],x['loaded'],
        x['authority'],x['receipt'],**kwargs)


def test_import_changes_only_current_checkpoint_and_invalidates_current_readiness(case):
    x=case;before_state=json.loads(x['state_path'].read_text());before_cursor=json.loads(x['cursor_path'].read_text())
    result=call(x,recovery_reason='supervisor selected the latest verified full checkpoint after owner recovery')
    state=json.loads(x['state_path'].read_text());cursor=json.loads(x['cursor_path'].read_text())
    assert result==resident.binding(x['receipt'])
    assert state['checkpoint']==x['new']['path'] and state['actor_policy_pending_load']==x['new']['sha256']
    assert state['phase']==before_state['phase']=='collect'
    for key,value in before_state.items():
        if key not in ('checkpoint','actor_policy_pending_load','readiness'):assert state[key]==value
    assert 'readiness' not in state
    assert cursor['current_checkpoint']==x['new']
    assert cursor['cycle']==before_cursor['cycle'] and cursor['phase']==before_cursor['phase']
    for key,value in before_cursor.items():
        if key not in ('current_checkpoint','readiness','foundation_qualified'):assert cursor[key]==value
    assert 'readiness' not in cursor and 'foundation_qualified' not in cursor
    proof=json.loads(x['receipt'].read_text())
    intent=json.loads(Path(proof['intent']['path']).read_text())
    assert proof['whole_recipe_completed'] is False and proof['readiness_granted'] is False
    assert intent['whole_recipe_completed'] is False
    # Historical receipt retry never rewinds future state.
    cursor['later_debt']={'retained':True};write(x['cursor_path'],cursor)
    assert call(x,recovery_reason='supervisor selected the latest verified full checkpoint after owner recovery')==result
    assert json.loads(x['cursor_path'].read_text())['later_debt']=={'retained':True}


def test_import_rejects_checkpoint_replacement_without_supervised_reason(case):
    x=case;before={p:p.read_bytes() for p in (x['state_path'],x['cursor_path'])}
    with pytest.raises(ValueError,match='recovery reason'):call(x)
    assert all(p.read_bytes()==value for p,value in before.items())


def test_import_reuses_existing_canonical_controller_fd_without_unlocking(case):
    x=case;lock_path=x['root'].parent/'controller.lock'
    with lock_path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        result=call(x,recovery_reason='explicit recovery',owned_controller_fd=lock.fileno())
        assert json.loads(Path(result['path']).read_text())['whole_recipe_completed'] is False
        assert any('FLOCK' in row and 'WRITE' in row for row in Path('/proc/self/fdinfo',str(lock.fileno())).read_text().splitlines())


def test_import_rejects_unlocked_or_substituted_controller_fd(case,tmp_path):
    x=case;other=tmp_path/'other.lock';other.touch()
    with other.open('a') as lock,pytest.raises(ValueError,match='descriptor'):
        call(x,recovery_reason='explicit recovery',owned_controller_fd=lock.fileno())


@pytest.mark.parametrize('change',['pending-round','learner-pending','pending-checkpoint','wrong-checkpoint'])
def test_import_cannot_skip_a_sealed_or_active_raw_operation(case,change):
    x=case
    if change=='pending-round':
        cursor=json.loads(x['cursor_path'].read_text());cursor['pending_rl_round']=10;write(x['cursor_path'],cursor)
    elif change=='wrong-checkpoint':
        cursor=json.loads(x['cursor_path'].read_text());cursor['current_checkpoint']=x['new'];write(x['cursor_path'],cursor)
    else:
        state=json.loads(x['state_path'].read_text())
        if change=='learner-pending':state['phase']='learner_pending'
        else:state['pending_native_checkpoint']=str(x['root']/'sealed.pt')
        write(x['state_path'],state)
    with pytest.raises(ValueError):call(x,recovery_reason='explicit recovery')


@pytest.mark.parametrize('change',['owner','argv','effective-command','argument-binding','wrapper','config','package','checkpoint','loaded-step','model'])
def test_import_rejects_changed_current_owner_or_loaded_checkpoint_authority(case,change,monkeypatch):
    x=case;a=x['authority']
    if change=='owner':
        from natlang_neuralese.rl import native_raw_handoff as handoff
        monkeypatch.setattr(handoff,'observe',lambda pid:{**a['owner'],'start_ticks':'changed'})
    elif change=='argv':a['process_argv'][-1]='different-role'
    elif change=='effective-command':
        command=Path(a['effective_argv_file']['path']);value=json.loads(command.read_text());value[-1]='different-role';write(command,value)
    elif change=='argument-binding':
        gate=Path(a['wrapper_configuration']['path']);value=json.loads(gate.read_text());value['trainer_arguments'][-1]='different-role';write(gate,value)
    elif change=='wrapper':Path(a['wrapper']['path']).write_text('# changed entry\n')
    elif change=='config':
        value=json.loads(x['config_path'].read_text());value['port']=32123;write(x['config_path'],value)
    elif change=='package':a['package']['code']['fake.py']='0'*64
    elif change=='checkpoint':Path(x['new']['path']).write_bytes(b'different full state bytes')
    elif change=='loaded-step':x['loaded']['training']['global_step']=25
    else:a['saved_state']['model_id']='another:base'
    with pytest.raises((ValueError,OSError)):
        call(x,recovery_reason='explicit recovery')


def test_partial_durable_commit_retries_exact_candidates_but_not_unknown_successor(case,monkeypatch):
    x=case;cursor=x['cursor_path'];real=raw.write_bytes_durable
    def crash_before_cursor(path,data):
        if Path(path)==cursor:raise OSError('simulated controller stop after state CAS')
        return real(path,data)
    with monkeypatch.context() as patch:
        patch.setattr(raw,'write_bytes_durable',crash_before_cursor)
        with pytest.raises(OSError):call(x,recovery_reason='explicit recovery')
    assert json.loads(x['state_path'].read_text())['checkpoint']==x['new']['path']
    receipt=call(x,recovery_reason='explicit recovery')
    assert json.loads(cursor.read_text())['current_checkpoint']==x['new']
    state=json.loads(x['state_path'].read_text());state['unknown_successor']=True;write(x['state_path'],state)
    assert call(x,recovery_reason='explicit recovery')==receipt
    assert json.loads(x['state_path'].read_text())['unknown_successor'] is True


def test_incomplete_import_never_overwrites_unknown_state_successor(case,monkeypatch):
    x=case;cursor=x['cursor_path'];real=raw.write_bytes_durable
    def crash_before_cursor(path,data):
        if Path(path)==cursor:raise OSError('simulated controller stop after state CAS')
        return real(path,data)
    with monkeypatch.context() as patch:
        patch.setattr(raw,'write_bytes_durable',crash_before_cursor)
        with pytest.raises(OSError):call(x,recovery_reason='explicit recovery')
    state=json.loads(x['state_path'].read_text());state['unknown_successor']=True;write(x['state_path'],state)
    with pytest.raises(ValueError,match='unknown controller successor'):
        call(x,recovery_reason='explicit recovery')
    assert json.loads(cursor.read_text())['current_checkpoint']==x['old']
    assert json.loads(x['state_path'].read_text())['unknown_successor'] is True


def test_new_owner_resumes_only_same_durable_import_after_original_owner_exit(case):
    x=case;cursor=x['cursor_path'];real=raw.write_bytes_durable
    def crash_before_cursor(path,data):
        if Path(path)==cursor:raise OSError('simulated owner exit after state CAS')
        return real(path,data)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(raw,'write_bytes_durable',crash_before_cursor)
        with pytest.raises(OSError):call(x,recovery_reason='explicit recovery')
    # A restarted in-process wrapper must attest its new owner, but may finish
    # only the exact durable decision after the original PID has exited.
    x['owner'].update(pid=12346,start_ticks='988')
    x['authority']['owner']=x['owner']
    result=call(x,recovery_reason='explicit recovery')
    receipt=json.loads(Path(result['path']).read_text())
    intent=json.loads(Path(receipt['intent']['path']).read_text())
    assert intent['authority']['owner']['pid']==12345
    from natlang_neuralese.common.hashing import canonical_json_sha256_hex
    assert receipt['request_sha256']==canonical_json_sha256_hex(intent['request'])
    assert receipt['resumed_by_authority']==x['authority']
    assert receipt['resumed_by_authority_sha256']==canonical_json_sha256_hex(x['authority'])
