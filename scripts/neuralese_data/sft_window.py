#!/usr/bin/env python3
"""Windows of one mandatory joint-SFT continuation (natlang.rl-production-config/1 with mandatory_sft_only).

A mandatory supervised controller trains until its qualified semantic-refresh boundary, then stops with
``phase: awaiting_semantic_refresh`` and ``pending_native_checkpoint``: one *window*. The next window is a new
controller (its own ``out``) continuing from that earned checkpoint with the same recipe, objective, focus,
optimizer/RNG/exposure and scored-graph anchor (native_supervised.continuation_config). An interrupted window resumes
in its own slot through ``rolling_sft_resume``. The same steps serve the DGX and the cloud lineages
(scripts/coreweave/rl_cloud.py, lfm3_rl_lineage.sh).

Run with the package of the runtime that trains the lineage on PYTHONPATH (a frozen runtime; this file only uses
functions that runtime already has):

  sft_window.py next --previous-config CFG --out DIR --config-out PATH [--checkpoint-path P] [--window N] [--id ID]
  sft_window.py rolling --config CFG --config-out PATH
  sft_window.py closure --config CFG -o OUT.json     exact input files a window's controller and learner read
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

GIB = 1024 ** 3


def _require_new(path):
    path = Path(path)
    if path.exists():
        raise SystemExit(f'{path} exists: a prepared control is immutable')
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _folded_admissions(training, options):
    history = (options.get('native_shared_execution_history')
               or (training.get('options') or {}).get('native_shared_execution_history') or [])
    return [e for e in history if isinstance(e, dict) and isinstance(e.get('qualification'), dict)]


def next_window(previous_config, out, config_out, *, checkpoint_path=None, window=None, ident=None):
    """The next window from the previous window's actual earned boundary (generalizes the DGX
    prepare-earned-window-continuation operator; no host-only checks or wrappers)."""
    from natlang_neuralese.common.jsonio import write_json_durable
    from natlang_neuralese.train.native_resident import binding, bound
    from natlang_neuralese.train.checkpoint_policy import CheckpointPolicy
    from natlang_neuralese.train.native_continuation import collect_sft_progress
    from natlang_neuralese.train.native_operation_report import supervised_progress_numerics_ok
    from natlang_neuralese.train.recipe import load_recipe
    from natlang_neuralese.train.native_execution_transition import require_qualified_for_training
    from natlang_neuralese.rl import native_supervised as ns

    previous_config = Path(previous_config)
    c = json.loads(previous_config.read_text())
    if not c.get('mandatory_sft_only'):
        raise SystemExit('not a mandatory supervised controller config')
    previous = Path(c['out'])
    state = json.loads((previous / 'state.json').read_text())
    if state.get('phase') != 'awaiting_semantic_refresh':
        raise SystemExit(f'previous window is {state.get("phase")!r}, not at its semantic-refresh boundary')
    parent = Path(state['pending_native_checkpoint']).resolve(strict=True)
    if parent != (previous / 'learner/checkpoint.pt').resolve(strict=True):
        raise SystemExit('pending checkpoint is not the previous window learner slot')
    ref = binding(parent)
    report_path = previous / 'learner/report.json'
    report = json.loads(report_path.read_text())
    if report.get('checkpoint_sha256') != ref['sha256'] or not supervised_progress_numerics_ok(report, ref):
        raise SystemExit('previous window report does not qualify its checkpoint')
    boundary = report.get('continuation_boundary') or {}
    if not (boundary.get('kind') == 'semantic-refresh' and boundary.get('reached') is True
            and boundary.get('new_optimizer_updates', 0) >= 1):
        raise SystemExit('previous window did not reach an earned semantic-refresh boundary')
    training = CheckpointPolicy.load(parent)['training']
    if not training.get('optimizer') or not training.get('rng'):
        raise SystemExit('earned boundary lacks optimizer or RNG state')
    stage_id = c['production_stage']
    frame = collect_sft_progress(training, borrow_frames=True)[stage_id]
    options = frame.get('options') or {}
    anchor = copy.deepcopy(options.get('sft_update_budget_reference'))
    folded = _folded_admissions(training, options)
    if folded:
        if len(folded) != 1:
            raise SystemExit('earned boundary carries more than one folded execution admission')
        require_qualified_for_training(training, admission_ref=folded[0]['qualification']['admission'],
                                       checkpoint_path=str(parent))
        # The trained successor carries its folded admission itself; the parent controller's authority is not
        # re-inherited (main 90bbf7fc7 _successor_folded_admission; frozen runtimes predate it).
        ns.inherited_execution_admission = lambda config: None
    receipt = json.loads((previous / 'learner/native-training-receipt.json').read_text())
    if receipt.get('recipe') != c['trainer_parameters']['recipe_binding']:
        raise SystemExit('previous window receipt recipe differs from its control')
    recipe = json.loads(bound({'path': receipt['recipe']['path'], 'sha256': receipt['recipe']['file_sha256']}).read_text())
    stage = next(s for s in recipe['stages'] if s['id'] == stage_id)
    prior_window = stage['parameters'].get('refresh_after_updates')
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f'{out} is not empty')
    out.mkdir(parents=True, exist_ok=True)
    config_out = _require_new(config_out)
    if window is not None and window != prior_window:
        stage['parameters']['refresh_after_updates'] = window
        recipe_path = out / 'recipe-runtime.json'
        write_json_durable(recipe_path, recipe)
        load_recipe(recipe_path)
        c['production_recipe'] = str(recipe_path)
    c['resident_data_directory'] = str(out / 'resident-data')
    c, sources = ns.continuation_config(c, stage, checkpoint=str(parent), out=str(out))
    c.pop('execution_transition_source_controller', None)
    c.pop('rolling_sft_resume', None)  # a same-slot retry control never carries into a new window
    if checkpoint_path is not None:  # the same bytes at another lineage's path (a relocated cloud parent)
        c['checkpoint'] = str(checkpoint_path)
    c['seed_trainer_parameters'] = copy.deepcopy(c['trainer_parameters'])
    if ident:
        c['id'] = ident
    c['branch_origin'] = {
        'parent': ref, 'parent_global_step': training['global_step'], 'actual_report': binding(report_path),
        'actual_completed_controller': binding(previous / 'state.json'), 'prior_control': binding(previous_config),
        'scored_graph_anchor': anchor, 'changed_controls': ({'refresh_after_updates': [prior_window, window]}
                                                            if window is not None and window != prior_window else {}),
        'reason': 'Next window of the same joint SFT continuation from the actual earned boundary.',
        'execution_admission': ({'folded_in_parent_history': folded[0]['qualification']['admission'],
                                 'controller_inheritance': 'omitted'} if folded else None),
        'new_optimizer_updates': 0, 'capability_lift_claimed': False}
    write_json_durable(config_out, c)
    result = {'prepared': str(config_out), 'out': str(out), 'parent': ref, 'global_step': training['global_step'],
              'source_roles': sorted(sources)}
    print(json.dumps(result), flush=True)
    return result


def rolling(config, config_out):
    """Bind an interrupted window's earned rolling slot (natlang.rolling-sft-resume/1); the control is otherwise the
    exact prior control."""
    from natlang_neuralese.common.jsonio import write_json_durable
    from natlang_neuralese.common.hashing import sha256_file_hex_cached
    from natlang_neuralese.train.checkpoint_policy import CheckpointPolicy
    c = json.loads(Path(config).read_text())
    out = Path(c['out'])
    state = json.loads((out / 'state.json').read_text())
    if state.get('pending_native_checkpoint') or state.get('phase') not in ('mandatory_sft_pending', 'updating_mandatory_sft'):
        raise SystemExit(f'window is {state.get("phase")!r}: no interrupted slot to resume')
    slot = (out / 'learner/checkpoint.pt').resolve(strict=True)
    prev = Path(state['checkpoint']).resolve(strict=True)
    if slot == prev:
        raise SystemExit('rolling slot is the window parent: nothing earned to resume')
    training = CheckpointPolicy.load(slot)['training']
    if not training.get('optimizer') or not training.get('rng'):
        raise SystemExit('rolling slot lacks optimizer or RNG state')
    control = {'schema': 'natlang.rolling-sft-resume/1', 'checkpoint_path': str(slot),
               'checkpoint_sha256': sha256_file_hex_cached(slot), 'previous_checkpoint_path': str(prev),
               'previous_checkpoint_sha256': sha256_file_hex_cached(prev)}
    c = copy.deepcopy(c)
    c['rolling_sft_resume'] = control
    write_json_durable(_require_new(config_out), c)
    print(json.dumps({'prepared': str(config_out), 'global_step': training['global_step'], 'rolling': control}),
          flush=True)


# -- input closure ---------------------------------------------------------------------------------------------------

def _paths(value, out):
    if isinstance(value, str):
        if value.startswith('/') and '\n' not in value and len(value) < 4096:
            out.add(value)
    elif isinstance(value, list):
        for item in value:
            _paths(item, out)
    elif isinstance(value, dict):
        for item in value.values():
            _paths(item, out)


def closure(config_path, *, max_file_bytes=4 * GIB):
    """The files a window's controller and learner read, by the runtime's own resolvers: every absolute path the
    control names, the recipe resolver's files and stage inputs (manifests, shards, CE/focus/correction dependencies),
    and the runtime's protected-custody closure. Paths recorded inside those files (receipts, reports, provenance) are
    not followed: the typed resolvers define what is read, and historical source paths are evidence, not inputs.
    Excluded and reported: the runtime itself (shipped as one archive), the base model (Hugging Face),
    the Python environment, checkpoints (shipped as exact blobs), the window's own output, scratch directories, and
    files above ``max_file_bytes`` (external provenance such as raw archives; never training payloads)."""
    from natlang_neuralese.common.hashing import sha256_file_hex_cached
    from natlang_neuralese.train.recipe import recipe_file_dependencies, _binding_files
    from natlang_neuralese.rl.native_raw_handoff import _recipe_plan
    config = json.loads(Path(config_path).read_text())
    runtime = Path((config.get('actor_environment') or {}).get('NATLANG_REPO') or Path(config['runtime']).parent)
    params = config.get('trainer_parameters') or {}
    roots = {
        'runtime': runtime, 'base': Path(config['base']), 'environment': Path(sys.prefix),
        'window-output': Path(config['out']),
    }
    for key in ('checkpoint_input_offload_directory',):
        if params.get(key):
            roots['scratch'] = Path(params[key])
    if config.get('llama_cpp_converter'):
        roots['llama-cpp'] = Path(config['llama_cpp_converter'])
    candidates, reasons = {}, {}

    def want(path, why):
        candidates.setdefault(path, set()).add(why)

    declared = set()
    _paths(config, declared)
    for path in declared:
        want(path, 'control')
    for item in recipe_file_dependencies(config['production_recipe']):
        want(item['path'] if isinstance(item, dict) else str(item), 'recipe')
    plan = _recipe_plan(Path(config['production_recipe']), 'cuda')
    for stage, roles in plan['stage_inputs'].items():
        for value in roles.values():
            if isinstance(value, dict) and 'path' in value:
                for item in _binding_files(value):
                    want(item['path'], f'stage-input:{stage}')
    custody = runtime / 'runtime-input-closure.json'
    if custody.is_file():
        for item in json.loads(custody.read_text()).get('bound_files', []):
            want(item['path'], 'runtime-protected-custody')
    files, excluded, missing = {}, [], []
    for raw, why in sorted(candidates.items()):
        path, why = Path(raw), sorted(why)
        root = next((name for name, r in roots.items() if path == r or r in path.parents), None)
        if root is not None:
            excluded.append({'path': raw, 'reason': root, 'from': why})
            continue
        if path.name.endswith('.pt'):
            excluded.append({'path': raw, 'reason': 'checkpoint', 'from': why})
            continue
        if not path.exists():
            missing.append({'path': raw, 'from': why})
            continue
        members = [path] if path.is_file() else sorted(p for p in path.rglob('*') if p.is_file()) if path.is_dir() else []
        for member in members:
            key = str(member)
            if key in files:
                files[key]['from'] = sorted(set(files[key]['from']) | set(why))
                continue
            size = member.stat().st_size
            if size > max_file_bytes:
                excluded.append({'path': key, 'reason': 'large', 'bytes': size, 'from': why})
                continue
            files[key] = {'path': key, 'bytes': size, 'sha256': sha256_file_hex_cached(member), 'from': why}
    unique = {v['sha256'] for v in files.values()}
    return {'schema': 'natlang.sft-window-input-closure/1', 'config': str(Path(config_path).resolve()),
            'runtime': str(runtime), 'files': sorted(files.values(), key=lambda v: v['path']),
            'bytes': sum(v['bytes'] for v in files.values()), 'unique_contents': len(unique),
            'excluded': sorted(excluded, key=lambda v: v['path']), 'missing': sorted(missing, key=lambda v: v['path'])}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('next')
    p.add_argument('--previous-config', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--config-out', required=True)
    p.add_argument('--checkpoint-path')
    p.add_argument('--window', type=int)
    p.add_argument('--id')
    p = sub.add_parser('rolling')
    p.add_argument('--config', required=True)
    p.add_argument('--config-out', required=True)
    p = sub.add_parser('closure')
    p.add_argument('--config', required=True)
    p.add_argument('-o', '--output', required=True)
    args = parser.parse_args(argv)
    if args.command == 'next':
        next_window(args.previous_config, args.out, args.config_out, checkpoint_path=args.checkpoint_path,
                    window=args.window, ident=args.id)
    elif args.command == 'rolling':
        rolling(args.config, args.config_out)
    else:
        from natlang_neuralese.common.jsonio import write_json_durable
        value = closure(args.config)
        write_json_durable(Path(args.output), value)
        print(json.dumps({'files': len(value['files']), 'bytes': value['bytes'], 'excluded': len(value['excluded']),
                          'missing': len(value['missing'])}), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
