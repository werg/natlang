"""A deferred loaded parent joins its real drained child without fake credit."""
import copy
import json
import shutil
import sys
import time
from pathlib import Path

import pytest
import torch

from test_native_raw_handoff import actual_drain
from natlang_neuralese.common.jsonio import write_json_durable as write
from natlang_neuralese.rl import native_raw_rounds as raw
from natlang_neuralese.rl import native_raw_handoff as handoff
from natlang_neuralese.train import native_resident as resident
from test_native_resident import setup


@pytest.fixture
def bridge_case(actual_drain, tmp_path, monkeypatch):
    process, old_intent_ref, terminal, slot, pin = actual_drain
    raw_intent = json.loads(Path(old_intent_ref['path']).read_text())
    owner = raw_intent['owner']
    source_launch = Path(raw_intent['source_launch']['path'])
    selected_path = tmp_path/'selected-startup-parent.pt'
    selected_path.write_bytes(b'bound selected full startup checkpoint')
    selected = {**resident.binding(selected_path), 'bytes': selected_path.stat().st_size}
    wrapper = resident.binding(Path(sys.executable).resolve())
    package = Path(raw.__file__).resolve().parents[1]
    runtime_root = package.parents[2]
    code = resident.code_inventory(package)
    trainer_args = ['--out', str(tmp_path/'entry'/'learner'), '--continue-from', str(selected_path),
                    '--stage-id', 'native_full_sft']
    args_file = tmp_path/'trainer-args.json'; write(args_file, trainer_args)
    full_command = [sys.executable, '-u', '-m', 'natlang_neuralese.train.native_state_sft', *trainer_args]
    command_file = tmp_path/'trainer-command.json'; write(command_file, full_command)
    continuation_file = tmp_path/'dispatch-continuation.json'
    write(continuation_file, {'parent': selected, 'runtime': str(runtime_root),
                              'runtime_code_sha256': resident.digest(code),
                              'whole_recipe_completed': False})
    dispatch_file = tmp_path/'dispatch.json'
    write(dispatch_file, {'schema': 'natlang.native-inprocess-trainer-dispatch/1',
        'owner': owner, 'wrapper': wrapper, 'effective_argv': full_command,
        'continuation': resident.binding(continuation_file), 'env': {}})
    argv = handoff.process_argv(owner)
    launch_value = {'at': raw_intent['at'], 'pid': owner['pid'], 'identity': owner,
                    'argv': argv, 'inprocess_dispatch': resident.binding(dispatch_file)}
    write(source_launch, launch_value)
    drain_ref = handoff.capture_drain_intent(resident.binding(source_launch), tmp_path/'actual-drain.json')
    drain = json.loads(Path(drain_ref['path']).read_text())

    saved = torch.load(slot, weights_only=False)
    model_id = 'fixture:' + 'a'*64
    saved['model_config'] = {'architecture': 'fixture', 'base_inventory_sha256': 'a'*64}
    saved['training'].update(global_step=25, options={'continuation': {'checkpoint_sha256': selected['sha256']}})
    torch.save(saved, slot); shutil.copyfile(slot, pin)
    process.communicate()
    write(terminal, {'at': time.time(), 'returncode': process.returncode})
    child_handoff = handoff.complete_drain(drain_ref, resident.binding(terminal),
        resident.binding(pin), tmp_path/'completed-handoff.json', original_slot=resident.binding(slot))

    root = tmp_path/'rounds'/'fixture'; production = root/'production'; production.mkdir(parents=True)
    config_path = root/'config.json'
    config = {'out': str(production), 'raw_model_id': model_id, 'production_recipe': str(tmp_path/'recipe.json')}
    write(config_path, config)
    old_path = tmp_path/'stale-raw-parent.pt'; old_path.write_bytes(b'old raw parent')
    old = resident.binding(old_path)
    state_path = production/'state.json'
    cursor_path = root/'cycle.json'
    retired_request = 'retirement-request-sha'
    retired_state = {'schema': 'natlang.rl-production-state/1', 'round': 3,
        'checkpoint': old['path'], 'phase': 'collect', 'collection_attempt': 4,
        'trained_rounds': 2, 'selection': {'sealed': ['a', 'b']},
        'stale_pending_raw_retirement': {'request_sha256': retired_request,
            'round': 3, 'pending_round': 4, 'previous_checkpoint': old,
            'whole_recipe_completed': False, 'training_credit': 0},
        'debt': {'rows': {'a': 2, 'b': 1}}}
    retired_cursor = {'cycle': 3, 'phase': 'sft', 'next_stage': 0,
        'current_checkpoint': old, 'raw_retry_generation': 4,
        'stale_pending_raw_retirement': {'request_sha256': retired_request,
            'round': 3, 'pending_round': 4, 'whole_recipe_completed': False},
        'debt': {'cases': ['a', 'b']}}
    write(state_path, retired_state); write(cursor_path, retired_cursor)

    gate_path = tmp_path/'entry'/'gate-spec.json'; gate_path.parent.mkdir()
    gate = {'trainer_arguments': trainer_args, 'effective_trainer_argv': resident.binding(args_file),
        'parent': selected, 'out': trainer_args[1], 'stage_id': 'native_full_sft',
        'runtime': str(runtime_root), 'runtime_code_sha256': resident.digest(code)}
    write(gate_path, gate)
    authority = {'schema': 'natlang.native-raw-current-owner/2', 'owner': owner,
        'process_argv': argv, 'wrapper': wrapper, 'config': resident.binding(config_path),
        'package': {'path': str(package), 'code': code},
        'wrapper_configuration': resident.binding(gate_path),
        'effective_argv_file': resident.binding(command_file),
        'effective_argv_binding': resident.binding(args_file), 'selected_checkpoint': selected,
        'saved_state': {'checkpoint': selected, 'global_step': 24, 'model_id': model_id}}
    authority_path = tmp_path/'entry'/'current-owner-authority.json'; write(authority_path, authority)
    deferred_path = tmp_path/'entry'/'current-owner-import-deferred.json'
    deferred = {'at': time.time(),
        'reason': 'current-owner import cannot bypass a pending sealed RAW round',
        'authority': resident.binding(authority_path), 'selected_checkpoint': selected,
        'saved_state': authority['saved_state'], 'config': resident.binding(config_path),
        'cursor': resident.binding(cursor_path), 'launch': resident.binding(source_launch),
        'drain_intent': drain_ref, 'readiness_granted': False, 'whole_recipe_completed': False}
    write(deferred_path, deferred)
    from natlang_neuralese.rl import native_owned_dispatch
    def verified_dispatch(captured, observed, snapshots=None):
        assert captured['owner'] == owner and observed['inprocess_dispatch'] == resident.binding(dispatch_file)
        return json.loads(continuation_file.read_text()), full_command, {
            'supervisor': wrapper, 'intent': resident.binding(dispatch_file),
            'continuation': resident.binding(continuation_file)}
    monkeypatch.setattr(native_owned_dispatch, 'verify', verified_dispatch)

    # A real retirement receipt binds the exact candidate files. The retire
    # verifier is independently covered by its own transaction suite; this
    # fixture isolates the subsequent handoff bridge.
    state_candidate = tmp_path/'retired-state.json'; cursor_candidate = tmp_path/'retired-cursor.json'
    write(state_candidate, retired_state); write(cursor_candidate, retired_cursor)
    retire_intent = {'schema': 'natlang.native-raw-stale-pending-retirement/1',
        'authority': {'fixture': True}, 'request': {'request_sha256': retired_request},
        'candidates': {'state': resident.binding(state_candidate), 'cursor': resident.binding(cursor_candidate)}}
    retire_intent_path = tmp_path/'retirement-intent.json'; write(retire_intent_path, retire_intent)
    retirement_path = tmp_path/'retirement.json'
    write(retirement_path, {'schema': 'natlang.native-raw-stale-pending-retirement/1',
        'request_sha256': retired_request, 'intent': resident.binding(retire_intent_path),
        'retired_round': 3, 'pending_round': 4, 'collection_attempt': 4,
        'whole_recipe_completed': False, 'training_credit': 0,
        'same_selection': {'path': str(tmp_path/'selection.json'), 'sha256': 'selection'}})
    monkeypatch.setattr(raw, 'retire_stale_pending_raw_round',
        lambda config, cursor, authority, out: resident.binding(out))
    return {'root': root, 'config': config_path, 'cursor': cursor_path, 'state': state_path,
        'selected': selected, 'child': resident.binding(pin), 'deferred': resident.binding(deferred_path),
        'handoff': child_handoff, 'retirement': resident.binding(retirement_path),
        'out': tmp_path/'bridge.json', 'authority_path': authority_path,
        'authority': authority, 'deferred_value': deferred, 'saved': saved}


def bridge(x, **kwargs):
    return raw.bridge_deferred_owner_handoff(x['config'], x['cursor'], x['deferred'],
        x['handoff'], x['retirement'], x['out'], recovery_reason='resume the exact selected startup optimizer after stale-round retirement',
        verified_handoff=handoff.verify_handoff(x['handoff']), **kwargs)


def test_deferred_startup_parent_bridges_to_completed_real_handoff_without_recipe_credit(bridge_case):
    x = bridge_case
    before_state = json.loads(x['state'].read_text()); before_cursor = json.loads(x['cursor'].read_text())
    result = bridge(x)
    state = json.loads(x['state'].read_text()); cursor = json.loads(x['cursor'].read_text())
    assert result == resident.binding(x['out'])
    assert state['checkpoint'] == x['child']['path']
    assert state['actor_policy_pending_load'] == x['child']['sha256']
    assert cursor['current_checkpoint'] == x['child']
    assert cursor['phase'] == 'sft' and cursor['next_stage'] == 0
    assert state['round'] == before_state['round'] and cursor['cycle'] == before_cursor['cycle']
    assert state['debt'] == before_state['debt'] and cursor['debt'] == before_cursor['debt']
    assert state['stale_pending_raw_retirement'] == before_state['stale_pending_raw_retirement']
    bridge_record = cursor['deferred_owner_handoff_bridge']
    assert bridge_record['whole_recipe_completed'] is False
    assert bridge_record['raw_completion_credit'] == 0 and bridge_record['readiness_granted'] is False
    assert bridge_record['runtime_handoff_required'] is True
    assert 'readiness' not in state and 'readiness' not in cursor and 'foundation_qualified' not in cursor
    proof = json.loads(x['out'].read_text())
    assert proof['selected_startup_parent'] == x['selected']
    assert proof['published_checkpoint'] == x['child']
    assert proof['training_credit'] == 0 and proof['whole_recipe_completed'] is False


def test_bridge_cursor_is_an_explicit_sft_first_runtime_boundary(bridge_case):
    from natlang_neuralese.rl import native_runtime_handoff as runtime
    x=bridge_case
    bridge_ref=bridge(x)
    cursor=json.loads((x['root']/'cycle.json').read_text())
    state=json.loads((Path(json.loads(x['config'].read_text())['out'])/'state.json').read_text())
    shutil.copyfile(x['config'],x['root']/'runtime-config-snapshot.json')
    proof={'deferred_owner_bridge':bridge_ref,'checkpoint':x['child'],
        'sources':{'config':resident.binding(x['config']),
            'cursor':resident.binding(x['root']/'cycle.json'),
            'state':resident.binding(Path(json.loads(x['config'].read_text())['out'])/'state.json')},
        'snapshots':{'config':resident.binding(x['root']/'runtime-config-snapshot.json')},
        'handoff':x['handoff']}
    recipe={'stages':[{'id':'sft','kind':'native_state_sft'}]}
    runtime._verified_deferred_owner_bridge(bridge_ref,x['child'],proof['sources'],
        config_snapshot=proof['snapshots']['config'],
        cursor=cursor,state=state,handoff=x['handoff'])
    assert runtime._owned_cycle_boundary(proof,cursor,state,recipe,'sft')
    bad=copy.deepcopy(cursor);bad['deferred_owner_handoff_bridge']['request_sha256']='0'*64
    assert not runtime._owned_cycle_boundary(proof,bad,state,recipe,'sft')
    bad=copy.deepcopy(state);bad['checkpoint']=str(x['selected']['path'])
    assert not runtime._owned_cycle_boundary(proof,cursor,bad,recipe,'sft')


def test_bridge_runtime_handoff_reaches_owed_sft_without_owned_operation_credit(
        bridge_case,tmp_path,monkeypatch):
    from types import SimpleNamespace
    from natlang_neuralese.rl import native_runtime_handoff as runtime
    from natlang_neuralese.rl import native_rounds
    import natlang_neuralese.rl.native_supervised as native_supervised

    x=bridge_case
    # Install a small ordinary SFT recipe in the existing controller namespace.
    (tmp_path/'recipe-fixture').mkdir()
    paths,recipe_directory,_,_=setup(tmp_path/'recipe-fixture')
    recipe_path=tmp_path/'recipe-fixture'/'old-recipe.json'
    recipe_plan=json.loads((recipe_directory.parent/'recipe-plan.json').read_text())
    cases=tmp_path/'cases.jsonl';cases.write_text('{"id":"ordinary-case","split":"train"}\n')
    collector=tmp_path/'collector';(collector/'dist/teacher').mkdir(parents=True)
    (collector/'dist/teacher/cli.js').write_text('/* fixture */')
    package=Path(raw.__file__).resolve().parents[1]
    config=json.loads(x['config'].read_text())
    config.update(bootstrap_binding={'plan_sha256':'fixture-plan','handoff':x['handoff']},
        production_recipe=str(recipe_path),recipe_lineage_sha256=recipe_plan['recipe_sha256'],
        production_stage='sft',trainer_parameters={},seed_trainer_parameters={},
        seed_command=['env','PYTHONPATH='+str(package.parent),'python','-m','natlang_neuralese.train.native_state_sft'],
        cases=str(cases),runtime='sealed-runtime',actor_command=['old-actor'],
        before_collect=['old-before'],task_source={'fixture':True},source_admission={'fixture':True},
        native_state=True,native_actor_mode='raw-state')
    write(x['config'],config)
    authority=json.loads(x['authority_path'].read_text());authority['config']=resident.binding(x['config'])
    write(x['authority_path'],authority)
    deferred_path=Path(x['deferred']['path']);deferred=json.loads(deferred_path.read_text())
    deferred.update(authority=resident.binding(x['authority_path']),config=resident.binding(x['config']))
    write(deferred_path,deferred);x['deferred']=resident.binding(deferred_path)
    bridge_ref=bridge(x)
    # Runtime handoff references can carry the standard path/SHA/size shape.
    # Validate the byte count at the actual producer-to-runtime boundary.
    bridge_ref={**bridge_ref,'bytes':Path(bridge_ref['path']).stat().st_size}

    qualification=tmp_path/'fixture-qualification.json'
    write(qualification,{'passed':True,'fixture_only':True})
    bare={'recipe':resident.binding(recipe_path),'package':str(package),
        'ordinary_capacity':resident.binding(qualification)}
    # The production handoff verifier is separately exercised by the runtime
    # suite. This fixture isolates the new bridge authority edge while keeping
    # the real publication, configuration and scheduler calls.
    recipe_binding=resident.binding(recipe_path)
    transition={'bare':True,'recipe':recipe_binding,'new_code':resident.code_inventory(package),
        'lineage':recipe_plan['recipe_sha256'],'inputs':{}}
    monkeypatch.setattr(runtime,'verify_handoff_metadata',lambda *args:None)
    monkeypatch.setattr(runtime,'_authority',lambda *args:transition)
    monkeypatch.setattr(runtime,'_verify',lambda value,*args:
        runtime._verified_deferred_owner_bridge(value['deferred_owner_bridge'],value['checkpoint'],
            value['sources'],config_snapshot=value['snapshots']['config'],
            handoff=value['handoff']))
    proof_path=tmp_path/'runtime-handoff.json'
    wrong_size={**bridge_ref,'bytes':bridge_ref['bytes']+1}
    with pytest.raises(ValueError,match='deferred-owner bridge receipt changed'):
        runtime.prepare_runtime_handoff({'path':'fixture-bootstrap'},x['handoff'],None,
            x['config'],tmp_path/'wrong-size-runtime-handoff.json',bare_direct_owner=bare,
            deferred_owner_bridge=wrong_size,verified_handoff=handoff.verify_handoff(x['handoff']))
    runtime_ref=runtime.prepare_runtime_handoff({'path':'fixture-bootstrap'},x['handoff'],None,
        x['config'],proof_path,bare_direct_owner=bare,deferred_owner_bridge=bridge_ref,
        verified_handoff=handoff.verify_handoff(x['handoff']))
    proof=json.loads(proof_path.read_text())
    assert proof['deferred_owner_bridge']==bridge_ref and proof['owned_operations'] is None

    declaration={'production_stage':'sft','handoff_requirement':{'model_id':config['raw_model_id']},
        'runtime_handoff':runtime_ref,'cases':resident.binding(cases),'runtime':str(collector),
        'commands':{'actor_command':['new-actor']}}
    plan_path=tmp_path/'plan.json';write(plan_path,{'models':[declaration]})
    monkeypatch.setattr(raw,'_scheduling_controls',lambda *args,**kwargs:None)
    def provider(config,declaration,*args,**kwargs):
        config['actor_command']=declaration['commands']['actor_command']
    monkeypatch.setattr(native_rounds,'configure_provider',provider)
    import natlang_neuralese.rl.native_bootstrap_metadata as bootstrap_metadata
    monkeypatch.setattr(bootstrap_metadata,'save',lambda *args,**kwargs:None)
    monkeypatch.setattr(native_supervised,'continuation_config',lambda config,stage,**kwargs:
        ({'trainer_parameters':{'stage_id':stage['id'],'epochs':1},'production_stage':stage['id']},{}))
    monkeypatch.setattr(native_supervised,'stage_pending',lambda *args,**kwargs:True)
    configured=raw.configure_model(plan_path,declaration,proof['bootstrap'],x['root'])
    installed=json.loads(configured.read_text())
    assert installed['runtime_handoff']==runtime_ref and installed['actor_command']==['new-actor']
    # The config is now mutable current state. Reconfiguration and verification
    # must use the runtime proof's immutable config snapshot, not reopen it as
    # if it were still the startup config.
    assert raw.configure_model(plan_path,declaration,proof['bootstrap'],x['root'])==configured
    runtime_after_config=runtime.verify_runtime_handoff(runtime_ref,bootstrap=proof['bootstrap'])
    assert runtime_after_config['deferred_owner_bridge']==bridge_ref

    calls=[];stop=SimpleNamespace(requested=False)
    def supervised(continuation,jobdir,config,cursor,owner_stop):
        calls.append((continuation['production_stage'],Path(jobdir)))
        owner_stop.requested=True
    original_configure=raw.configure_model
    def configure_again(*args):
        calls.append(('configure',Path(args[3])))
        return original_configure(*args)
    monkeypatch.setattr(raw,'configure_model',configure_again)
    monkeypatch.setattr(raw,'_supervised_job',supervised)
    result=raw.student_cycle(plan_path,declaration,proof['bootstrap'],x['root'],stop=stop)
    assert any(call[0]=='sft' for call in calls)
    assert result['phase']=='sft' and result['next_stage']==0
    assert result.get('whole_recipe_completed') is not True
    cursor_after=json.loads((x['root']/'cycle.json').read_text())
    assert 'deferred_owner_handoff_bridge' not in cursor_after
    consumed=cursor_after['deferred_owner_handoff_bridge_consumed']
    assert consumed['runtime_handoff']==runtime_ref and consumed['training_credit']==0
    assert not runtime._owned_cycle_boundary(proof,
        json.loads((x['root']/'cycle.json').read_text()),
        json.loads((Path(json.loads(x['config'].read_text())['out'])/'state.json').read_text()),
        {'stages':[{'id':'sft','kind':'native_state_sft'}]},'sft')
    # A retry after the one-time config application must keep the same
    # authority and must not classify the consumed marker as a fresh boundary.
    retry=raw.student_cycle(plan_path,declaration,proof['bootstrap'],x['root'],stop=stop)
    assert retry['deferred_owner_handoff_bridge_consumed']==consumed


def test_deferred_bridge_rejects_child_without_selected_parent_chain(bridge_case):
    x = bridge_case
    authority = json.loads(x['authority_path'].read_text())
    authority['saved_state']['global_step'] = 26
    write(x['authority_path'], authority)
    deferred = json.loads(Path(x['deferred']['path']).read_text())
    deferred['authority'] = resident.binding(x['authority_path'])
    deferred['saved_state'] = authority['saved_state']
    write(x['deferred']['path'], deferred)
    x['deferred'] = resident.binding(Path(x['deferred']['path']))
    with pytest.raises(ValueError, match='completed handoff'):
        bridge(x)
