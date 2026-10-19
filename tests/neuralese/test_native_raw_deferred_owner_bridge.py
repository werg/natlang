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
    def verified_dispatch(captured, observed):
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
