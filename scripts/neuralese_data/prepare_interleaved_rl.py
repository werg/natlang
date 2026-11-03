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


def earned_parent(previous_config, checkpoint=None):
    from natlang_neuralese.train.native_resident import binding
    from natlang_neuralese.train.checkpoint_policy import CheckpointPolicy
    from natlang_neuralese.train.native_continuation import collect_sft_progress
    state = json.loads((Path(previous_config['out']) / 'state.json').read_text())
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


def successor_recipe(previous_recipe_path, out, *, stage_id, sft_stage, task_source, parameters, sft_window=None):
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
    name = 'rl_task_source_' + re.sub(r'[^a-z0-9]+', '_', Path(task_source).parent.name.lower()).strip('_')[:48]
    recipe.setdefault('input_bindings', {})[name] = {'path': str(Path(task_source).resolve(strict=True)),
                                                     'sha256': sha256_file_hex_cached(task_source)}
    recipe['stages'].append({'id': stage_id, 'kind': 'native_interleaved_rl', 'requires': [sft_stage],
                             'inputs': {'rl_task_source': name}, 'parameters': parameters})
    path = Path(out) / 'recipe-runtime.json'
    write_json_durable(_require_new(path), recipe)
    load_recipe(path)
    return path


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
        for required, reason in REQUIRED_RUNTIME_COMMITS:
            # The frozen tree is a git archive (and NATLANG_REPO names it); ancestry is checked in the checkout this
            # script runs from. Run it from the canonical checkout.
            checkout = Path(__file__).resolve().parents[2]
            if subprocess.run(['git', '-C', str(checkout), 'merge-base', '--is-ancestor', required,
                               commit or 'missing'], capture_output=True).returncode != 0:
                raise SystemExit(f'frozen runtime {commit} predates {required}: {reason}')
    previous_path = Path(args.previous_config).resolve(strict=True)
    previous = json.loads(previous_path.read_text())
    if not previous.get('mandatory_sft_only'):
        raise SystemExit('previous control is not a mandatory joint-SFT window')
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
    out = Path(args.out).resolve()
    control = Path(args.control_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f'{out} is not empty')
    control.mkdir(parents=True, exist_ok=True)
    loss = json.loads(args.policy_loss) if args.policy_loss else DEFAULT_LOSS
    parameters = {'sft_stage': previous['production_stage'], 'rl_rounds_per_window': args.rl_rounds_per_window,
                  'start_with': args.start_with, 'rows_per_round': args.rows_per_round, 'group': args.group,
                  'task_curriculum': CURRICULUM, 'rl_policy_loss': loss, 'update_objective': 'policy',
                  'max_model_requests': previous.get('max_model_requests', 64),
                  'max_turns': previous.get('max_turns', 128), 'case_seconds': previous.get('case_seconds', 3600)}
    recipe = successor_recipe(previous['production_recipe'], control, stage_id=args.stage_id,
                              sft_stage=previous['production_stage'], task_source=task_source, parameters=parameters,
                              sft_window=args.sft_window_updates)
    config = copy.deepcopy(previous)
    for key in ('mandatory_sft_only', 'rolling_sft_resume', 'execution_transition_source_controller',
                'task_selection_history', 'branch_origin'):
        config.pop(key, None)
    lineage = recipe_lineage(previous)
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
    previous_state_path = Path(previous['out']) / 'state.json'
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
              'sft_window_updates': declared['sft_window_updates'], 'runtime': str(runtime)}
    print(json.dumps(result), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--previous-config', required=True, help='earned mandatory joint-SFT window control')
    parser.add_argument('--checkpoint', help='explicit earned parent (default: the window boundary checkpoint)')
    parser.add_argument('--task-source', required=True, help='train-only RL task cases JSONL')
    parser.add_argument('--runtime', required=True, help='frozen runtime root containing rl/interleave.py')
    parser.add_argument('--control-dir', required=True, help='new directory for the recipe, wrapper and inputs')
    parser.add_argument('--config-out', required=True)
    parser.add_argument('--out', required=True, help='new controller output directory')
    parser.add_argument('--id', required=True)
    parser.add_argument('--stage-id', default='interleaved_jolt_rl_v1')
    parser.add_argument('--rl-rounds-per-window', type=int, default=2)
    parser.add_argument('--start-with', choices=('rl', 'sft'), default='rl')
    parser.add_argument('--sft-window-updates', type=int,
                        help="SFT updates per window (the joint stage's refresh_after_updates; default: unchanged)")
    parser.add_argument('--rows-per-round', type=int, default=16)
    parser.add_argument('--group', type=int, default=8)
    parser.add_argument('--policy-loss', help='canonical policy loss JSON (default JOLT+ tis)')
    prepare(parser.parse_args(argv))


if __name__ == '__main__':
    sys.exit(main())
