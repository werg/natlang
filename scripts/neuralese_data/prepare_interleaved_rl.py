#!/usr/bin/env python3
"""Prepare the interleaved JOLT+ RL + joint-SFT control from an earned mandatory SFT window.

The previous control is a mandatory joint-SFT window (natlang.rl-production-config/1, mandatory_sft_only) at its
earned semantic-refresh boundary (or an explicit earned ``--checkpoint``). The prepared control:

- appends a ``native_interleaved_rl`` stage to that window's recipe (same lineage; the existing SFT stage, its
  inputs and parameters are unchanged), binding the RL task source, selector /2, round budget and JOLT+ loss;
- is an ordinary RL production controller with ``interleaved_rl_stage`` and ``production_stage`` = the SFT stage,
  rebound to a frozen runtime that contains ``rl/interleave.py`` (actor command, environment, Python inventory,
  rendering assets and the learner admission wrapper);
- carries the previous controller's task-selection observations, projected onto the new task packet with absent
  task worlds excluded explicitly (rl/selection_history.py).

Nothing is launched and no checkpoint is written. Launch (DGX):

  python3 scripts/memory_ledger.py run --unit <unit> --budget-gb 110 --priority high -- \\
    env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
    NATLANG_REPO=<runtime> PYTHONPATH=<runtime>/training/neuralese:<runtime> \\
    <repo>/.venv-neuralese/bin/python -m natlang_neuralese.rl.production run --config <config-out>

Run this script with the frozen runtime's package on PYTHONPATH.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path

# Task-source oracles that need runtime code newer than the interleave stage itself.
REQUIRED_RUNTIME_COMMITS = (
    ('577fc3271', 'codebase-ops repair/feature oracles call default exports; older runtimes grade every rollout wrong'),
)
DEFAULT_LOSS = {'schema': 'jolt-plus/1', 'beta_s': 1.0, 'beta_t': 0.5,
                'outcome': {'schema': 'score-centered/1', 'k': 128, 'eps': 1e-6, 'importance': 'tis'}}


def _require_new(path):
    path = Path(path)
    if path.exists():
        raise SystemExit(f'{path} exists: a prepared control is immutable')
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def prior_state_path(previous_config):
    """The controller state that ended at this control's boundary: the control's own once it has run, else (a never-started
    prepared control) the completed prior window's, as its branch origin records it."""
    own = Path(previous_config['out']) / 'state.json'
    if own.is_file():
        return own
    origin = previous_config.get('branch_origin') or {}
    prior = (origin.get('prepared_next_window') or {}).get('prior_out')
    if prior and (Path(prior) / 'state.json').is_file():
        return Path(prior) / 'state.json'
    completed = (origin.get('actual_completed_controller') or {}).get('path')
    if completed and Path(completed).is_file():
        return Path(completed)
    raise SystemExit(f'{own} is absent and the control records no completed prior controller state')


def earned_parent(previous_config, checkpoint=None):
    from natlang_neuralese.train.native_resident import binding
    from natlang_neuralese.train.checkpoint_policy import CheckpointPolicy
    from natlang_neuralese.train.native_continuation import collect_sft_progress
    own = Path(previous_config['out']) / 'state.json'
    state = json.loads(prior_state_path(previous_config).read_text())
    if not own.is_file():
        # A never-started prepared control (a relocated DiLoCo session worker): its parent is its own checkpoint.
        if checkpoint is None or Path(checkpoint).resolve() != Path(previous_config['checkpoint']).resolve():
            raise SystemExit('a never-started control converts only with --checkpoint equal to its own checkpoint')
    if checkpoint is None:
        if state.get('phase') != 'awaiting_semantic_refresh':
            raise SystemExit(f'previous window is {state.get("phase")!r}; pass an explicit earned --checkpoint')
        checkpoint = state['pending_native_checkpoint']
    path = Path(checkpoint).resolve(strict=True)
    training = CheckpointPolicy.load(path)['training']
    if not training.get('optimizer') or not training.get('rng'):
        raise SystemExit('parent lacks optimizer or RNG state')
    if previous_config['production_stage'] not in collect_sft_progress(training, borrow_frames=True):
        raise SystemExit('parent does not carry the joint SFT stage frame')
    return binding(path), training.get('global_step'), state


def successor_recipe(previous_recipe_path, out, *, stage_id, sft_stage, task_source, parameters, sft_window=None,
                     stage_parameters=None, focus=None):
    from natlang_neuralese.common.jsonio import write_json_durable
    from natlang_neuralese.common.hashing import sha256_file_hex_cached
    from natlang_neuralese.train.recipe import load_recipe, resolve_recipe_data
    recipe = copy.deepcopy(resolve_recipe_data(previous_recipe_path))
    if any(stage['id'] == stage_id for stage in recipe['stages']):
        raise SystemExit(f'recipe already declares {stage_id}')
    if sft_window is not None:
        # The SFT window length is the joint stage's semantic-refresh cadence (as sft_window.py next --window).
        next(stage for stage in recipe['stages'] if stage['id'] == sft_stage)['parameters']['refresh_after_updates'] = \
            sft_window
    if stage_parameters:
        apply_stage_parameters(recipe, sft_stage, stage_parameters)
    if focus is not None:
        rebind_pending_focus(recipe, sft_stage, focus)
    name = 'rl_task_source_' + re.sub(r'[^a-z0-9]+', '_', Path(task_source).parent.name.lower()).strip('_')[:48]
    recipe.setdefault('input_bindings', {})[name] = {'path': str(Path(task_source).resolve(strict=True)),
                                                     'sha256': sha256_file_hex_cached(task_source)}
    recipe['stages'].append({'id': stage_id, 'kind': 'native_interleaved_rl', 'requires': [sft_stage],
                             'inputs': {'rl_task_source': name}, 'parameters': parameters})
    path = Path(out) / 'recipe-runtime.json'
    write_json_durable(_require_new(path), recipe)
    load_recipe(path)
    return path


# Execution and optimizer controls of a window. The interleaved controller rebuilds every SFT window's trainer
# parameters from the recipe stage (native_supervised.continuation_config), not from the previous window's control,
# so an LR/memory decision made on the window config alone silently reverts to the recipe's values at the first
# interleaved window (2026-11-05: the bf95 low-LR restart would have returned to 2e-5).
WINDOW_PARAMETER_KEYS = ('learning_rate', 'interface_row_learning_rate', 'learned_row_learning_rate', 'batch_tokens',
                         'batch_graphs', 'accumulate_batches', 'chunk_tokens', 'ffn_chunk_tokens',
                         'checkpoint_input_offload', 'lr_warmup', 'lr_warmup_start', 'weight_decay')
WINDOW_PARAMETER_PREFIXES = ('checkpoint_input_offload_',)


def window_parameter_overrides(previous, recipe, sft_stage):
    """Stage-parameter changes that make the recipe's SFT stage produce the previous window's effective trainer
    parameters (None removes a key the previous window does not set)."""
    actual = previous.get('trainer_parameters') or {}
    declared = next(stage for stage in recipe['stages'] if stage['id'] == sft_stage)['parameters']
    changes = {}
    for key in sorted(set(actual) | set(declared)):
        if key not in WINDOW_PARAMETER_KEYS and not key.startswith(WINDOW_PARAMETER_PREFIXES):
            continue
        if actual.get(key) != declared.get(key):
            changes[key] = actual.get(key)
    return changes


def apply_stage_parameters(recipe, sft_stage, changes):
    parameters = next(stage for stage in recipe['stages'] if stage['id'] == sft_stage)['parameters']
    for key, value in changes.items():
        if value is None:
            parameters.pop(key, None)
        else:
            parameters[key] = value


def pending_focus_binding(focus_path, parent_training, sft_stage):
    """A new one-time focus request for the SFT stage: bound, authenticated, and admissible against the parent's
    actual focus history (the prior request completed, or the new one names it in ``supersedes``)."""
    from natlang_neuralese.common.hashing import sha256_file_hex_cached
    from natlang_neuralese.train.native_continuation import collect_sft_progress
    from natlang_neuralese.train.native_pending_focus import FINISHED, dependencies
    path = Path(focus_path).resolve(strict=True)
    reference = {'path': str(path), 'sha256': sha256_file_hex_cached(path)}
    document, _ = dependencies(reference)
    frame = collect_sft_progress(parent_training, borrow_frames=True)[sft_stage]
    history = (frame.get('options') or {}).get('pending_focus_history') or []
    if history and history[-1].get('request', {}).get('sha256') == reference['sha256']:
        return reference, 'already the parent\'s focus request'
    if history and history[-1].get('state') not in FINISHED:
        named = (document.get('supersedes') or {}).get('request_sha256')
        if named != history[-1]['request']['sha256']:
            raise SystemExit(f'parent focus {history[-1]["request"]["sha256"][:12]} is {history[-1].get("state")!r}: the '
                             'new focus request must name it in "supersedes" (request_sha256, reason)')
        return reference, 'supersedes the parent\'s active focus request'
    return reference, 'parent focus history finished' if history else 'parent has no focus history'


def rebind_pending_focus(recipe, sft_stage, reference):
    stage = next(item for item in recipe['stages'] if item['id'] == sft_stage)
    old = stage['inputs'].get('pending_focus')
    name = 'pending_focus_' + reference['sha256'][:12]
    recipe.setdefault('input_bindings', {})[name] = dict(reference)
    stage['inputs']['pending_focus'] = name
    if old and old != name and not any(old in (item.get('inputs') or {}).values() for item in recipe['stages']):
        recipe['input_bindings'].pop(old, None)


def recipe_with_stages(previous_recipe_path, stages_from, out, *, sft_stage, sft_window=None, runtime=None,
                       ce_policy=None, stage_parameters=None, focus=None):
    """The previous window's recipe plus the RL stages a declared successor recipe adds (interleaved, memetic), with
    their input bindings. The joint SFT stage keeps its inputs and parameters except the declared changes: window
    length, task-eval runtime and a CE objective transition (``ce_policy``). Returns the written recipe path."""
    from natlang_neuralese.common.jsonio import write_json_durable
    from natlang_neuralese.train.recipe import load_recipe, resolve_recipe_data
    recipe = copy.deepcopy(resolve_recipe_data(previous_recipe_path))
    donor = resolve_recipe_data(stages_from)
    present = {stage['id'] for stage in recipe['stages']}
    added = [copy.deepcopy(stage) for stage in donor['stages'] if stage['id'] not in present]
    if not any(stage['kind'] == 'native_interleaved_rl' for stage in added):
        raise SystemExit(f'{stages_from} adds no native_interleaved_rl stage to the previous recipe')
    for stage in added:
        if stage.get('parameters', {}).get('sft_stage', sft_stage) != sft_stage:
            raise SystemExit(f'{stage["id"]} alternates with another SFT stage')
        for name in (stage.get('inputs') or {}).values():
            if name not in donor.get('input_bindings', {}):
                raise SystemExit(f'{stage["id"]} input {name} is not bound in {stages_from}')
            existing = recipe.setdefault('input_bindings', {}).get(name)
            if existing is not None and existing != donor['input_bindings'][name]:
                raise SystemExit(f'input binding {name} differs between the recipes')
            recipe['input_bindings'][name] = donor['input_bindings'][name]
        recipe['stages'].append(stage)
    sft = next(stage for stage in recipe['stages'] if stage['id'] == sft_stage)
    if sft_window is not None:
        sft['parameters']['refresh_after_updates'] = sft_window
    if runtime is not None and isinstance(sft['parameters'].get('task_eval'), dict):
        sft['parameters']['task_eval']['runtime'] = str(Path(runtime) / 'ts-host')
    if stage_parameters:
        apply_stage_parameters(recipe, sft_stage, stage_parameters)
    if focus is not None:
        rebind_pending_focus(recipe, sft_stage, focus)
    if ce_policy is not None:
        name = 'ce_policy_' + ce_policy['sha256'][:12]
        replaced = sft['inputs'].get('ce_supervision_policy')
        recipe['input_bindings'][name] = {'path': ce_policy['path'], 'sha256': ce_policy['sha256']}
        sft['inputs']['ce_supervision_policy'] = name
        if replaced and not any(replaced in (stage.get('inputs') or {}).values() for stage in recipe['stages']):
            recipe['input_bindings'].pop(replaced)  # the transition's predecessor stays bound in the parent recipe
    path = Path(out) / 'recipe-runtime.json'
    write_json_durable(_require_new(path), recipe)
    load_recipe(path)
    return path


def boundary_weight_policy(previous, parent, weight, out, token_ids=(125033, 125034)):
    """CE objective transition at the earned boundary: the window's policy plus ``boundary_weights`` (producer
    open/close at least ``weight`` reply tokens), ``objective_transition.from`` = the current objective, and the
    quota-aware basis of the actual boundary frame (same source: remaining quotas and credits unchanged)."""
    from natlang_neuralese.common.jsonio import write_json_durable
    from natlang_neuralese.train.native_resident import binding
    from natlang_neuralese.train.checkpoint_policy import CheckpointPolicy
    from natlang_neuralese.train.native_continuation import collect_sft_progress
    from natlang_neuralese.train.native_quota_aware_ce import make_basis
    from natlang_neuralese.train.own_output_supervision import BOUNDARY_WEIGHT_SCHEMA, identity, policy_objective
    from natlang_neuralese.train.recipe import load_recipe, resolve_stage_inputs
    stage_id = previous['production_stage']
    recipe = load_recipe(previous['production_recipe'])
    stage = next(item for item in recipe['stages'] if item['id'] == stage_id)
    current = resolve_stage_inputs(recipe, stage, {})['ce_supervision_policy']
    policy = json.loads(Path(current['path']).read_text())
    before = identity(policy_objective(policy))
    policy['objective'] = {**policy['objective'], 'boundary_weights': {
        'schema': BOUNDARY_WEIGHT_SCHEMA, 'token_ids': sorted(token_ids), 'min_weight': float(weight)}}
    policy['objective_transition'] = {'from': before['sha256']}
    training = CheckpointPolicy.load(Path(parent['path']))['training']
    frame = collect_sft_progress(training, borrow_frames=True)[stage_id]
    source = (frame.get('source_receipts') or [{}])[0].get('sha256')
    parameters = stage['parameters']
    policy['quota_aware_basis'] = make_basis(frame, stage_id=stage_id, source_sha256=source, candidate_sha256=source,
                                             epochs=parameters.get('epochs'),
                                             communication_min_visits=parameters.get('communication_min_visits'))
    path = _require_new(Path(out) / 'ce-policy.json')
    write_json_durable(path, policy)
    after = identity(policy_objective(policy))
    return {**binding(path), 'objective_transition': {'from': before, 'to': after}}


def cloud_commands(previous):
    """Cloud window controls run the trainer modules directly (no DGX admission wrapper); a DiLoCo worker's seed
    command runs behind the session's join.py, which the RL learner command takes too (join.py wraps
    natlang_neuralese.rl.native_stage: relay, worker env, exec)."""
    seed, learner = list(previous['seed_command']), list(previous['learner_command'])
    if learner[1] != '-m':
        return None
    if len(seed) > 2 and Path(seed[1]).name == 'join.py':
        cut = seed.index('--')
        if learner[1] == '-m' and Path(learner[1]).name != 'join.py':
            learner = [seed[0], *seed[1:cut + 1], *learner]
    return {'seed_command': seed, 'learner_command': learner}


def learner_wrapper(previous_wrapper, out, runtime):
    text = Path(previous_wrapper).read_text()
    pattern = re.compile(r"^runtime=Path\('[^']*'\)$", re.M)
    if len(pattern.findall(text)) != 1:
        raise SystemExit('learner admission wrapper has no single runtime binding to rebind')
    path = _require_new(Path(out) / 'admit-learner.py')
    path.write_text(pattern.sub(f"runtime=Path('{runtime}')", text))
    return path


def prepare(args):
    from natlang_neuralese.common.jsonio import write_json_durable
    from natlang_neuralese.common.hashing import sha256_file_hex_cached
    from natlang_neuralese.rl import production, interleave, selection_history
    from natlang_neuralese.rl.curriculum import DEFAULT as CURRICULUM, task_descriptors
    from natlang_neuralese.rl.native_supervised import bind_recipe_lineage, recipe_lineage
    from natlang_neuralese.train.native_resident import binding
    runtime = Path(args.runtime).resolve(strict=True)
    if not (runtime / 'training/neuralese/natlang_neuralese/rl/interleave.py').is_file():
        raise SystemExit('runtime lacks rl/interleave.py: freeze a runtime from main first')
    closure = runtime / 'runtime-input-closure.json'
    if closure.is_file():
        import subprocess
        commit = json.loads(closure.read_text()).get('commit')
        checkout = Path(__file__).resolve().parents[2]
        bundle = checkout / '.coreweave-code-commit'
        # A cloud code bundle is a git archive without history: a runtime frozen at exactly the bundle's own commit
        # contains every required commit this list names (each entry was added after its commit landed on main).
        same_as_bundle = (not (checkout / '.git').exists() and bundle.is_file()
                          and bundle.read_text().strip() == (commit or ''))
        for required, reason in REQUIRED_RUNTIME_COMMITS:
            # The frozen tree is a git archive (and NATLANG_REPO names it); ancestry is checked in the checkout this
            # script runs from. Run it from the canonical checkout or a code bundle of the runtime's own commit.
            if same_as_bundle:
                continue
            if subprocess.run(['git', '-C', str(checkout), 'merge-base', '--is-ancestor', required,
                               commit or 'missing'], capture_output=True).returncode != 0:
                raise SystemExit(f'frozen runtime {commit} predates {required}: {reason}')
    previous_path = Path(args.previous_config).resolve(strict=True)
    previous = json.loads(previous_path.read_text())
    if not previous.get('mandatory_sft_only'):
        raise SystemExit('previous control is not a mandatory joint-SFT window')
    stages_from = None
    if args.stages_from:
        from natlang_neuralese.train.recipe import load_recipe, resolve_stage_inputs
        stages_from = load_recipe(args.stages_from)
        declared = next(stage for stage in stages_from['stages'] if stage['kind'] == 'native_interleaved_rl')
        args.stage_id = declared['id']
        task = resolve_stage_inputs(stages_from, declared, {})['rl_task_source']
        if args.task_source and Path(args.task_source).resolve() != Path(task['path']).resolve():
            raise SystemExit('--task-source differs from the declared stage task source')
        args.task_source = task['path']
    if not args.task_source:
        from natlang_neuralese.train.recipe import load_recipe, resolve_stage_inputs
        loaded = load_recipe(json.loads(Path(args.previous_config).read_text())['production_recipe'])
        declared = next((stage for stage in loaded['stages'] if stage['kind'] == 'native_interleaved_rl'), None)
        if declared is None:
            raise SystemExit('--task-source or --stages-from is required')
        args.task_source = resolve_stage_inputs(loaded, declared, {})['rl_task_source']['path']
    task_source = Path(args.task_source).resolve(strict=True)
    rows = 0
    with task_source.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get('split') != 'train':
                raise SystemExit('RL task source must contain only train-split tasks')
            task_descriptors([row])
            rows += 1
    parent, step, previous_state = earned_parent(previous, args.checkpoint)
    out = Path(args.out).resolve() if args.out else None
    control = Path(args.control_dir).resolve()
    if out is not None and out.exists() and any(out.iterdir()):
        raise SystemExit(f'{out} is not empty')
    control.mkdir(parents=True, exist_ok=True)
    loss = json.loads(args.policy_loss) if args.policy_loss else DEFAULT_LOSS
    parameters = {'sft_stage': previous['production_stage'], 'rl_rounds_per_window': args.rl_rounds_per_window,
                  'start_with': args.start_with, 'rows_per_round': args.rows_per_round, 'group': args.group,
                  'task_curriculum': CURRICULUM, 'rl_policy_loss': loss, 'update_objective': 'policy',
                  'max_model_requests': previous.get('max_model_requests', 64),
                  'max_turns': previous.get('max_turns', 128), 'case_seconds': previous.get('case_seconds', 3600)}
    ce_policy = (boundary_weight_policy(previous, parent, args.boundary_weight, control)
                 if args.boundary_weight is not None else None)
    own = json.loads(Path(previous['production_recipe']).read_text())
    stage_parameters = focus = None
    if not args.keep_recipe_window_parameters:
        from natlang_neuralese.train.recipe import resolve_recipe_data
        stage_parameters = window_parameter_overrides(previous, resolve_recipe_data(previous['production_recipe']),
                                                      previous['production_stage'])
    if args.pending_focus:
        from natlang_neuralese.train.checkpoint_policy import CheckpointPolicy
        focus, focus_note = pending_focus_binding(args.pending_focus, CheckpointPolicy.load(Path(parent['path']))['training'],
                                                  previous['production_stage'])
        print(json.dumps({'pending_focus': focus, 'admission': focus_note}), flush=True)
    present = next((stage for stage in own['stages'] if stage['kind'] == 'native_interleaved_rl'), None)
    if present is not None:
        # A session worker control already runs the successor recipe (session.py prepare --recipe): its recipe and
        # trainer recipe binding are the session template's; adopt them unchanged.
        if args.boundary_weight is not None or args.sft_window_updates is not None or args.pending_focus:
            raise SystemExit('the previous recipe already declares the interleaved stage: no further recipe changes')
        stage_parameters = None  # its recipe is the session template's, adopted unchanged
        if present['parameters'].get('start_with', 'rl') != args.start_with:
            raise SystemExit(f"declared start_with is {present['parameters'].get('start_with', 'rl')!r}")
        args.stage_id = present['id']
        recipe = Path(previous['production_recipe']).resolve(strict=True)
        parameters = present['parameters']
        loss = parameters['rl_policy_loss']
    elif stages_from is not None:
        recipe = recipe_with_stages(previous['production_recipe'], args.stages_from, control,
                                    sft_stage=previous['production_stage'], sft_window=args.sft_window_updates,
                                    runtime=runtime, ce_policy=ce_policy, stage_parameters=stage_parameters,
                                    focus=focus)
        declared_parameters = next(stage for stage in json.loads(recipe.read_text())['stages']
                                   if stage['id'] == args.stage_id)['parameters']
        parameters = {**parameters, **declared_parameters, 'start_with': args.start_with}
        if parameters != declared_parameters:
            # start_with is the only per-boundary choice; record it in the stage itself.
            body = json.loads(recipe.read_text())
            next(stage for stage in body['stages'] if stage['id'] == args.stage_id)['parameters'] = parameters
            write_json_durable(recipe, body)
        loss = parameters['rl_policy_loss']
    else:
        if ce_policy is not None:
            raise SystemExit('--boundary-weight needs --stages-from (the recipe successor binds the CE policy)')
        recipe = successor_recipe(previous['production_recipe'], control, stage_id=args.stage_id,
                                  sft_stage=previous['production_stage'], task_source=task_source,
                                  parameters=parameters, sft_window=args.sft_window_updates,
                                  stage_parameters=stage_parameters, focus=focus)
    if args.recipe_only:
        # The successor recipe alone (e.g. for session.py prepare --recipe); controls are prepared from it later.
        result = {'recipe': str(recipe), 'ce_objective_transition': ce_policy, 'parent': parent,
                  'global_step': step, 'stage': args.stage_id, 'sft_stage_parameter_changes': stage_parameters}
        print(json.dumps(result), flush=True)
        return result
    config = copy.deepcopy(previous)
    for key in ('mandatory_sft_only', 'rolling_sft_resume', 'execution_transition_source_controller',
                'task_selection_history', 'branch_origin'):
        config.pop(key, None)
    lineage = recipe_lineage(previous)
    commands = cloud_commands(previous)
    if commands is not None:
        config.update(commands)
    else:
        wrapper = learner_wrapper(Path(previous['learner_command'][1]), control, runtime)
        for key in ('learner_command', 'seed_command'):
            command = list(config[key])
            if command[1] != previous['learner_command'][1]:
                raise SystemExit(f'{key} does not use the shared learner admission wrapper')
            command[1] = str(wrapper)
            config[key] = command
    environment = production.actor_runtime_configuration(runtime)
    config.update(environment)
    config['actor_command'] = production.bind_actor_runtime_command(config['actor_command'], runtime)
    config['runtime'] = str(runtime / 'ts-host')
    config.update(id=args.id, out=str(out), checkpoint=parent['path'], cases=str(task_source),
                  production_recipe=str(recipe), interleaved_rl_stage=args.stage_id,
                  resident_data_directory=str(out / 'resident-data'), task_curriculum=CURRICULUM,
                  rows_per_round=args.rows_per_round, group=args.group, policy_loss=loss)
    bind_recipe_lineage(config, lineage)
    previous_state_path = prior_state_path(previous)
    if previous_state.get('task_curriculum_history'):
        selector = selection_history.snapshot({'path': str(previous_state_path),
                                               'sha256': sha256_file_hex_cached(previous_state_path)},
                                              control / 'control-inputs', str(task_source))
        if selector is not None:
            config['task_selection_history'] = selector
    config['branch_origin'] = {
        'parent': parent, 'parent_global_step': step, 'prior_control': binding(previous_path),
        'prior_controller_state': binding(previous_state_path), 'runtime': str(runtime),
        'interleaved_stage': args.stage_id, 'task_source': {'path': str(task_source), 'rows': rows,
                                                            'sha256': sha256_file_hex_cached(task_source)},
        **({'ce_objective_transition': ce_policy} if ce_policy is not None else {}),
        **({'stages_from': {'path': str(Path(args.stages_from).resolve()),
                            'sha256': sha256_file_hex_cached(args.stages_from)}} if args.stages_from else {}),
        'reason': 'Start declared whole-graph JOLT+ RL interleaved with the joint SFT stage '
                  '(2026-11-03 recipe review), from the actual earned joint-SFT boundary.',
        'new_optimizer_updates': 0, 'capability_lift_claimed': False}
    production.validate_actor_runtime_configuration(config)
    config_out = _require_new(args.config_out)
    write_json_durable(config_out, config)
    declared = interleave.declared_stage(config)
    if declared['task_source']['sha256'] != sha256_file_hex_cached(task_source):
        raise SystemExit('declared stage does not bind the prepared task source')
    result = {'prepared': str(config_out), 'recipe': str(recipe), 'out': str(out), 'parent': parent,
              'global_step': step, 'task_rows': rows, 'stage': declared['id'],
              'sft_window_updates': declared['sft_window_updates'], 'runtime': str(runtime),
              'sft_stage_parameter_changes': stage_parameters}
    print(json.dumps(result), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--previous-config', required=True, help='earned mandatory joint-SFT window control')
    parser.add_argument('--checkpoint', help='explicit earned parent (default: the window boundary checkpoint)')
    parser.add_argument('--task-source', help='train-only RL task cases JSONL (default: the --stages-from stage\'s)')
    parser.add_argument('--stages-from', help='declared successor recipe whose added stages (interleaved RL, memetic) '
                        'are adopted with their inputs and parameters')
    parser.add_argument('--boundary-weight', type=float,
                        help='CE objective transition: producer open/close boundary targets at least this many reply '
                             'tokens (own-output boundary_weights)')
    parser.add_argument('--pending-focus', help='new one-time SFT focus request (natlang.native-pending-focus/1) bound '
                        'as the joint stage\'s pending_focus input; it must name the parent\'s active request in '
                        '"supersedes" unless that request finished')
    parser.add_argument('--keep-recipe-window-parameters', action='store_true',
                        help='keep the recipe stage\'s own LR/batch/offload/chunk parameters. By default the SFT stage '
                             'adopts the previous window\'s effective trainer parameters, because every interleaved '
                             'window is rebuilt from the recipe stage')
    parser.add_argument('--runtime', required=True, help='frozen runtime root containing rl/interleave.py')
    parser.add_argument('--control-dir', required=True, help='new directory for the recipe, wrapper and inputs')
    parser.add_argument('--config-out')
    parser.add_argument('--out', help='new controller output directory')
    parser.add_argument('--id')
    parser.add_argument('--recipe-only', action='store_true',
                        help='write only the successor recipe (and its CE transition) into --control-dir')
    parser.add_argument('--stage-id', default='interleaved_jolt_rl_v1')
    parser.add_argument('--rl-rounds-per-window', type=int, default=2)
    parser.add_argument('--start-with', choices=('rl', 'sft'), default='rl')
    parser.add_argument('--sft-window-updates', type=int,
                        help="SFT updates per window (the joint stage's refresh_after_updates; default: unchanged)")
    parser.add_argument('--rows-per-round', type=int, default=16)
    parser.add_argument('--group', type=int, default=8)
    parser.add_argument('--policy-loss', help='canonical policy loss JSON (default JOLT+ tis)')
    args = parser.parse_args(argv)
    if not args.recipe_only and not (args.config_out and args.out and args.id):
        parser.error('--config-out, --out and --id are required unless --recipe-only')
    prepare(args)


if __name__ == '__main__':
    sys.exit(main())
