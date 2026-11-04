"""Interleaved whole-graph JOLT+ RL and joint SFT windows (recipe kind ``native_interleaved_rl``).

The recipe declares one interleaved stage that requires an earlier
``native_state_sft`` stage. An ordinary RL production controller binds it with
``interleaved_rl_stage``; the stage then owns the controller's task source,
selector, round budget and policy loss. The same controller alternates
``rl_rounds_per_window`` completed RL rounds with one SFT window: the existing
mandatory continuation (``native_supervised.continuation_config``/``execute``)
from the controller's current checkpoint, run to the SFT stage's declared
``refresh_after_updates`` semantic-refresh boundary in its own directory. The
window's earned checkpoint becomes the controller's head and the next RL actor's
policy. SFT row quotas, exposure, optimizer/RNG and source custody stay inside
the carried checkpoint frames exactly as in separate controllers; nothing here
counts an SFT window as an RL round or the reverse.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from ..common.hashing import canonical_json_sha256_hex, sha256_file_hex_cached
from ..common.jsonio import write_json_durable

SCHEMA = 'natlang.rl-sft-interleave/1'
CONTROLLER_KEYS = {'rows_per_round': 'rows_per_round', 'group': 'group', 'task_curriculum': 'task_curriculum',
                   'rl_policy_loss': 'policy_loss', 'update_objective': 'update_objective',
                   'max_model_requests': 'max_model_requests', 'max_turns': 'max_turns',
                   'case_seconds': 'case_seconds', 'neuralese_use_bonus': 'neuralese_use_bonus',
                   # JOLT+ guidance authoring: 'backbone' records text twins without a writer; 'writer' runs the
                   # (text) guidance writer, whose checked outputs seed the guidance learned operator (fallow C6).
                   'guidance_stage': 'jolt_guidance_stage'}
GUIDANCE_STAGES = ('backbone', 'writer')


def validate_stage_parameters(parameters):
    from .curriculum import policy_options
    from .policy_math import policy_loss_declaration
    if not isinstance(parameters.get('sft_stage'), str) or not parameters['sft_stage']:
        raise ValueError('interleaved RL declares the joint SFT stage it alternates with (sft_stage)')
    rounds = parameters.get('rl_rounds_per_window')
    if type(rounds) is not int or rounds < 1:
        raise ValueError('interleaved RL rl_rounds_per_window must be a positive integer')
    if parameters.get('start_with', 'rl') not in ('rl', 'sft'):
        raise ValueError('interleaved RL start_with must be rl or sft')
    for key, low in (('rows_per_round', 1), ('group', 2), ('max_model_requests', 1), ('max_turns', 1),
                     ('case_seconds', 1)):
        if key in parameters and (type(parameters[key]) is not int or parameters[key] < low):
            raise ValueError(f'interleaved RL {key} must be an integer >= {low}')
    if 'task_curriculum' in parameters:
        policy_options(parameters['task_curriculum'])
    if 'rl_policy_loss' in parameters:
        loss = policy_loss_declaration(parameters['rl_policy_loss'])
        if loss is None or loss != parameters['rl_policy_loss']:
            raise ValueError('interleaved RL rl_policy_loss must be a canonical policy loss declaration')
    if parameters.get('update_objective', 'policy') not in ('policy', 'self-distillation'):
        raise ValueError('interleaved RL update_objective must be policy or self-distillation')
    if parameters.get('guidance_stage', 'backbone') not in GUIDANCE_STAGES:
        raise ValueError('interleaved RL guidance_stage is backbone or writer')
    from .memetic_admission import validate_parameters as validate_distillation
    validate_distillation(parameters)
    bonus = parameters.get('neuralese_use_bonus')
    if bonus is not None and (type(bonus) not in (int, float) or not 0 <= bonus <= 1):
        raise ValueError('interleaved RL neuralese_use_bonus must be in [0, 1]')


def declared_stage(config):
    """The recipe's interleaved stage, its resolved task source and the SFT stage it alternates with."""
    from ..train.recipe import load_recipe, resolve_stage_inputs, effective_stage_parameters, recipe_identity_sha256
    stage_id = config.get('interleaved_rl_stage')
    if stage_id is None:
        return None
    recipe = load_recipe(config['production_recipe'])
    stages = {stage['id']: stage for stage in recipe['stages']}
    stage = stages.get(stage_id)
    if stage is None or stage['kind'] != 'native_interleaved_rl':
        raise ValueError('interleaved_rl_stage does not name a native_interleaved_rl recipe stage')
    parameters = effective_stage_parameters(stage, [], recipe)
    if parameters['sft_stage'] != config.get('production_stage'):
        raise ValueError('interleaved RL alternates with a different SFT stage than the controller obligation')
    inputs = resolve_stage_inputs(recipe, stage, {})
    sft = stages[parameters['sft_stage']]
    memetic = next((item for item in recipe['stages'] if item['kind'] == 'native_memetic_rl'
                    and (item.get('parameters') or {}).get('rl_stage') == stage_id), None)
    return {'id': stage_id, 'parameters': parameters, 'task_source': inputs['rl_task_source'],
            'sft_stage': sft, 'sft_window_updates': effective_stage_parameters(sft, [], recipe)['refresh_after_updates'],
            'recipe_sha256': recipe_identity_sha256(config['production_recipe']),
            'stage_sha256': canonical_json_sha256_hex(stage),
            'memetic': None if memetic is None else {
                'id': memetic['id'], 'parameters': effective_stage_parameters(memetic, [], recipe),
                'program_source': resolve_stage_inputs(recipe, memetic, {})['program_source'],
                # The registered evaluation-only memetic held packet (optional; rl.memetic_stage held-packet).
                'held_program_source': resolve_stage_inputs(recipe, memetic, {}).get('held_program_source'),
                'stage_sha256': canonical_json_sha256_hex(memetic)}}


def bind_stage(config, state, state_path):
    """Apply the declared stage to this round's controller config and record its binding once."""
    declared = declared_stage(config)
    if declared is None:
        return config, None
    declared['controller_out'] = config.get('out')
    if config.get('mandatory_sft_only'):
        raise ValueError('an interleaved RL controller is not a mandatory SFT-only continuation')
    result = dict(config)
    result['cases'] = declared['task_source']['path']
    for key, target in CONTROLLER_KEYS.items():
        if key in declared['parameters']:
            result[target] = copy.deepcopy(declared['parameters'][key])
    interleave = state.setdefault('interleave', {'schema': SCHEMA, 'stage': declared['id'], 'sft_windows': 0,
        'rl_rounds_at_last_window': state.get('round', 0), 'windows': [], 'stage_bindings': [],
        'pending_initial_window': declared['parameters'].get('start_with', 'rl') == 'sft'})
    if interleave.get('stage') != declared['id']:
        raise ValueError('controller interleave state belongs to another declared stage')
    binding = {'recipe_sha256': declared['recipe_sha256'], 'stage_sha256': declared['stage_sha256'],
               'task_source': {'path': declared['task_source']['path'], 'sha256': declared['task_source']['sha256']},
               'round': state.get('round', 0)}
    bindings = interleave['stage_bindings']
    if not bindings or {k: v for k, v in bindings[-1].items() if k != 'round'} != {
            k: v for k, v in binding.items() if k != 'round'}:
        bindings.append(binding)
        write_json_durable(state_path, state)
    if declared.get('memetic') is not None and state.get('checkpoint'):
        # A due memetic round ("improve program P" episodes, rl/memetic_stage.py) replaces this round's task source.
        from .memetic_stage import bind_round
        before = json.dumps(interleave.get('memetic'), sort_keys=True)
        memetic = bind_round(result, declared['memetic'], state, sha256_file_hex_cached(state['checkpoint']))
        if json.dumps(interleave.get('memetic'), sort_keys=True) != before:
            write_json_durable(state_path, state)
        if memetic is not None:
            result = memetic
    return result, declared


def window_due(declared, state):
    if declared is None:
        return False
    interleave = state['interleave']
    if interleave.get('sft_complete'):
        # Newly queued memetic distillation packages reopen a completed joint SFT stage (rl/memetic_admission.py).
        from .memetic_admission import unseen_packages
        return bool(unseen_packages(declared.get('controller_out'), state))
    if interleave.get('active_window') is not None or interleave.get('pending_initial_window'):
        return True
    return state.get('round', 0) - interleave['rl_rounds_at_last_window'] >= declared['parameters']['rl_rounds_per_window']


def _rolling_control(window_config, window_state):
    """Resume an interrupted window from its earned slot (natlang.rolling-sft-resume/1)."""
    from ..train.checkpoint_policy import CheckpointPolicy
    slot = Path(window_config['out']) / 'learner' / 'checkpoint.pt'
    if not slot.is_file():
        return None
    slot = slot.resolve(strict=True)
    parent = Path(window_state['checkpoint']).resolve(strict=True)
    if slot == parent or sha256_file_hex_cached(slot) == sha256_file_hex_cached(parent):
        return None
    training = CheckpointPolicy.load(slot)['training']
    if not training.get('optimizer') or not training.get('rng'):
        raise ValueError('interrupted SFT window slot lacks optimizer or RNG state')
    return {'schema': 'natlang.rolling-sft-resume/1', 'checkpoint_path': str(slot),
            'checkpoint_sha256': sha256_file_hex_cached(slot), 'previous_checkpoint_path': str(parent),
            'previous_checkpoint_sha256': sha256_file_hex_cached(parent)}


def run_sft_window(config, declared, state, state_path, control):
    """One SFT window to its semantic-refresh boundary; adopt its earned checkpoint as the controller head.

    Returns ``'adopted'``, ``'complete'`` (the SFT stage has no remaining work) or ``'stopped'``.
    """
    from .native_supervised import continuation_config, execute
    from .production import initial_controller_state, run_command, invalidate_checkpoint_readiness
    interleave = state['interleave']
    index = interleave['sft_windows']
    directory = Path(config['out']).resolve() / f'sft-window-{index:06d}'
    config_path = directory / 'config.json'
    if config_path.exists():
        window_config = json.loads(config_path.read_text())
        declared_window = window_config.get('interleaved_window') or {}
        parent = declared_window.get('parent') or {}
        if (declared_window.get('window') != index or declared_window.get('stage') != declared['id']
                or sha256_file_hex_cached(window_config['checkpoint']) != parent.get('sha256')
                or parent.get('sha256') != sha256_file_hex_cached(state['checkpoint'])):
            raise ValueError('interleaved SFT window changed its recorded parent')
        if interleave.get('active_window') is None:
            # The window config was written but the controller state was not:
            # adopt the exact durable window rather than preparing another.
            interleave['active_window'] = {'window': index, 'directory': str(directory),
                'config': {'path': str(config_path), 'sha256': sha256_file_hex_cached(config_path)},
                'parent': {**parent, 'path': window_config['checkpoint']}}
            state['phase'] = 'interleaved_sft_window'
            write_json_durable(state_path, state)
    else:
        directory.mkdir(parents=True, exist_ok=True)
        source = {key: value for key, value in config.items() if key not in ('interleaved_rl_stage',)}
        source['resident_data_directory'] = str(directory / 'resident-data')
        # Pending memetic distillation packages become a recorded source append of this window; afterwards the
        # lineage's windows continue on the appended source (rl/memetic_admission.py).
        from .memetic_admission import prepare_window
        recipe_path, sft_stage, admission = prepare_window(config, declared, state, state_path, directory)
        if recipe_path is not None:
            source['production_recipe'] = str(recipe_path)
        window_config, _ = continuation_config(source, sft_stage, checkpoint=state['checkpoint'],
                                               out=str(directory))
        window_config.pop('rolling_sft_resume', None)
        parent = {'path': str(Path(state['checkpoint']).resolve(strict=True)),
                  'sha256': sha256_file_hex_cached(state['checkpoint'])}
        window_config['interleaved_window'] = {'schema': SCHEMA, 'stage': declared['id'], 'window': index,
            'parent': parent, 'controller_round': state.get('round', 0),
            'sft_updates': declared['sft_window_updates'],
            **({'memetic_admission': admission} if admission is not None else {})}
        write_json_durable(config_path, window_config)
        interleave['active_window'] = {'window': index, 'directory': str(directory),
            'config': {'path': str(config_path), 'sha256': sha256_file_hex_cached(config_path)},
            'parent': {**parent, 'path': window_config['checkpoint']}}
        state['phase'] = 'interleaved_sft_window'
        write_json_durable(state_path, state)
    window_state_path = directory / 'state.json'
    window_state = (json.loads(window_state_path.read_text()) if window_state_path.exists()
                    else initial_controller_state(window_config))
    write_json_durable(window_state_path, window_state)
    if window_state.get('phase') != 'awaiting_semantic_refresh':
        rolling = _rolling_control(window_config, window_state)
        effective = dict(window_config)
        if rolling is not None:
            effective['rolling_sft_resume'] = rolling
            history = interleave['active_window'].setdefault('rolling_resumes', [])
            if not history or history[-1] != rolling:
                history.append(rolling)
                write_json_durable(state_path, state)
        window_state = execute(effective, window_state, window_state_path, control)
    if window_state.get('phase') != 'awaiting_semantic_refresh':
        if control is not None and control.stop.requested:
            state['phase'] = 'requested_stop'
            write_json_durable(state_path, state)
            return 'stopped'
        raise RuntimeError('interleaved SFT window did not reach its semantic-refresh boundary')
    checkpoint = Path(window_state['pending_native_checkpoint']).resolve(strict=True)
    if checkpoint != (directory / 'learner' / 'checkpoint.pt').resolve(strict=True):
        raise ValueError('interleaved SFT window boundary is not its own learner slot')
    report_path = directory / 'learner' / 'report.json'
    report = json.loads(report_path.read_text())
    checkpoint_sha = sha256_file_hex_cached(checkpoint)
    if report.get('checkpoint_sha256') not in (None, checkpoint_sha):
        raise ValueError('interleaved SFT window report describes another checkpoint')
    boundary = report.get('continuation_boundary') or {}
    updates = boundary.get('new_optimizer_updates', 0)
    parent = interleave['active_window']['parent']
    record = {'window': index, 'directory': str(directory), 'config': interleave['active_window']['config'],
              'parent': parent, 'checkpoint': {'path': str(checkpoint), 'sha256': checkpoint_sha,
                                               'bytes': checkpoint.stat().st_size},
              'report': {'path': str(report_path), 'sha256': sha256_file_hex_cached(report_path)},
              'new_optimizer_updates': updates, 'completed_stage': report.get('completed') is True,
              'controller_round': state.get('round', 0),
              'rolling_resumes': interleave['active_window'].get('rolling_resumes', []),
              'mandatory_sft': window_state.get('mandatory_sft')}
    interleave['windows'].append(record)
    interleave.pop('active_window', None)
    interleave.pop('pending_initial_window', None)
    interleave['sft_windows'] = index + 1
    interleave['rl_rounds_at_last_window'] = state.get('round', 0)
    if checkpoint_sha == parent['sha256'] or (report.get('completed') is True and not updates):
        # No remaining SFT work: later windows would repeat zero-update work.
        interleave['sft_complete'] = True
    if checkpoint_sha != parent['sha256']:
        values = {**{k: v for k, v in config.items() if isinstance(v, (str, int, float, list))},
                  'checkpoint': str(checkpoint), 'policy_id': checkpoint_sha}
        if config.get('after_update'):
            run_command(config['after_update'], values)
        else:
            state['actor_policy_pending_load'] = checkpoint_sha
        invalidate_checkpoint_readiness(state, checkpoint, reason='interleaved-sft-window')
        state['checkpoint'] = str(checkpoint)
    state['phase'] = 'collect'
    write_json_durable(state_path, state)
    return 'complete' if interleave.get('sft_complete') else 'adopted'
