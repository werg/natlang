"""Prepare and optionally run paired BF16/TE-FP8 direct-SFT comparisons.

This is an experiment coordinator, not a trainer. It clones reviewed declared
``neuralese_direct`` recipes and invokes the shared declared-stage launcher;
the trainer, data path, optimizer, evaluator, and checkpoint format remain the
same in both arms. Preparation is the default. ``--run`` is explicit because
the comparison owns the GPU for the duration of its sequential runs.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parents[2]
sys.path.insert(0, str(PACKAGE_ROOT.parent))

from natlang_neuralese.common.hashing import sha256_file_hex_cached
from natlang_neuralese.train.recipe import (DIRECT_STAGE_SCHEMA, load_recipe, validate_direct_stage_recipe,
                                            write_json)


DEFAULT_FP8_IMAGE = 'sha256:170d746f59d6e6114bb90b957b3acd3d7f1b8c042b308e47c7cbfec26834eca6'


def package_code_inventory() -> dict[str, str]:
    relative_package = PACKAGE_ROOT.relative_to(REPOSITORY_ROOT)
    status = subprocess.run(['git', 'status', '--porcelain', '--', str(relative_package)],
                            cwd=REPOSITORY_ROOT, text=True, capture_output=True, check=True).stdout.strip()
    if status:
        raise RuntimeError('shared training package has uncommitted changes; commit/review it before freezing a comparison')
    return {str(path.relative_to(PACKAGE_ROOT)): sha256_file_hex_cached(path)
            for path in sorted(PACKAGE_ROOT.rglob('*.py'))
            if '__pycache__' not in path.parts}


def _validate_inputs(recipe: dict[str, Any]) -> None:
    paths = {binding['path']: binding['sha256'] for binding in recipe['inputs'].values()}
    for evidence in recipe['lineage'].get('evidence', []):
        paths[evidence['path']] = evidence['sha256']
    for raw_path, expected in paths.items():
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f'declared comparison input is missing: {path}')
        actual = sha256_file_hex_cached(path)
        if actual != expected:
            raise ValueError(f'declared comparison input digest differs: {path}')


def _normalized_pair(recipe: dict[str, Any]) -> dict[str, Any]:
    value = json.loads(json.dumps(recipe))
    value.pop('id', None)
    value.pop('outputs', None)
    value.pop('runtime', None)
    for key in ('fp8_training', 'fp8_recipe', 'fp8_amax_history_len'):
        value['parameters'].pop(key, None)
    return value


def _profile_identity(recipe: dict[str, Any]) -> dict[str, Any]:
    """Ensure short/long profiles share data, optimizer, and objective identity."""
    value = _normalized_pair(recipe)
    for key in ('tokens', 'target_tokens', 'prefix_tokens', 'batch_tokens'):
        value['parameters'].pop(key, None)
    value['lineage'].pop('scope', None)
    return value


def build_pair(template_path: Path, output_root: Path, profile: str, image: str,
               *, saved_tensor_audit_steps: int = 1) -> list[dict[str, Any]]:
    template = load_recipe(template_path)
    if template.get('schema') != DIRECT_STAGE_SCHEMA or template.get('stage', {}).get('kind') != 'neuralese_direct':
        raise ValueError(f'{template_path} must be a declared neuralese_direct stage recipe')
    validate_direct_stage_recipe(template)
    _validate_inputs(template)
    if template['parameters'].get('fp8_training', 'off') not in ('off', 'te-fp8'):
        raise ValueError('template has an unsupported FP8 backend declaration')
    if int(template['parameters'].get('batch', 1)) < 2:
        raise ValueError('paired gradient audit requires at least two windows per update')
    if template['parameters'].get('window_execution', 'auto') == 'packed':
        raise ValueError('paired gradient audit requires sequential or auto window execution')
    if 'steps' in template['parameters'] and template['parameters']['steps'] < 5:
        raise ValueError('comparison recipe must contain at least four optimizer updates for warm timing')

    runtime = dict(template['runtime'])
    runtime['image'] = image
    runtime['package_code'] = package_code_inventory()
    variants = []
    for backend, recipe_name in (('bf16', 'delayed'), ('te_fp8', 'delayed')):
        recipe = json.loads(json.dumps(template))
        recipe['id'] = f"{template['id']}-{profile}-{backend.replace('_', '-')}"
        recipe['runtime'] = runtime
        recipe['outputs'] = {'directory': str((output_root / 'outputs' / profile / backend).resolve())}
        recipe['lineage']['scope'] = (
            str(recipe['lineage']['scope']).rstrip() +
            f' Paired diagnostic profile={profile}; no task/runtime qualification or admission.')
        recipe['parameters']['fp8_training'] = 'off' if backend == 'bf16' else 'te-fp8'
        recipe['parameters']['fp8_recipe'] = recipe_name
        recipe['parameters']['fp8_amax_history_len'] = 16
        recipe['parameters']['saved_tensor_audit_steps'] = saved_tensor_audit_steps
        recipe['parameters']['gradient_audit_steps'] = 1
        validate_direct_stage_recipe(recipe)
        path = output_root / 'recipes' / f'{profile}-{backend}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, recipe)
        variants.append({'backend': backend, 'recipe_path': str(path.resolve()),
                         'recipe_sha256': sha256_file_hex_cached(path), 'recipe': recipe})

    if _normalized_pair(variants[0]['recipe']) != _normalized_pair(variants[1]['recipe']):
        raise AssertionError('paired recipes differ outside precision backend, runtime, or output identity')
    return variants


def _gpu_sample(stop: threading.Event, destination: Path, interval: float) -> None:
    fields = ('timestamp,clocks.current.sm,clocks.current.memory,power.draw,temperature.gpu,'
              'memory.used,memory.total')
    with destination.open('w', encoding='utf-8') as handle:
        while not stop.is_set():
            result = subprocess.run(
                ['nvidia-smi', f'--query-gpu={fields}', '--format=csv,noheader,nounits'],
                text=True, capture_output=True, check=False)
            handle.write(json.dumps({'time': time.time(), 'returncode': result.returncode,
                                     'sample': result.stdout.strip(),
                                     'error': result.stderr.strip()[:300]}) + '\n')
            handle.flush()
            stop.wait(interval)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _execution_plans(path: Path) -> list[dict[str, Any]]:
    plans = []
    if not path.is_file():
        return plans
    with path.open(encoding='utf-8', errors='replace') as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get('event') == 'execution_plan':
                plans.append({key: row.get(key) for key in
                              ('step', 'source_documents', 'mode', 'checkpoint_layers',
                               'context_tokens', 'target_tokens', 'activation_geometry_bytes')})
    return plans


def _window_schedule(plans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: row.get(key) for key in
             ('step', 'source_documents', 'mode', 'context_tokens', 'target_tokens')}
            for row in plans]


def _telemetry_summary(path: Path) -> dict[str, Any]:
    samples = []
    if path.is_file():
        with path.open(encoding='utf-8') as handle:
            for line in handle:
                try:
                    outer = json.loads(line)
                    fields = [value.strip() for value in outer.get('sample', '').split(',')]
                    if len(fields) < 7:
                        continue
                    samples.append({'sm_clock_mhz': float(fields[1]), 'memory_clock_mhz': float(fields[2]),
                                    'power_watts': float(fields[3]), 'temperature_c': float(fields[4]),
                                    'gpu_memory_used_mib': float(fields[5]), 'gpu_memory_total_mib': float(fields[6])})
                except (ValueError, TypeError, json.JSONDecodeError):
                    continue
    result = {'sample_count': len(samples)}
    for key in ('sm_clock_mhz', 'memory_clock_mhz', 'power_watts', 'temperature_c',
                'gpu_memory_used_mib', 'gpu_memory_total_mib'):
        values = [row[key] for row in samples]
        result['median_' + key] = statistics.median(values) if values else None
        result['max_' + key] = max(values) if values else None
    return result


def _numeric_deltas(reference: Any, candidate: Any, prefix: str = '') -> dict[str, Any]:
    if isinstance(reference, dict) and isinstance(candidate, dict):
        output = {}
        for key in sorted(set(reference) & set(candidate)):
            if key in {'step', 'seconds', 'time'}:
                continue
            child = _numeric_deltas(reference[key], candidate[key], f'{prefix}.{key}' if prefix else key)
            output.update(child)
        return output
    if isinstance(reference, (int, float)) and not isinstance(reference, bool) and \
            isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
        delta = float(candidate) - float(reference)
        return {prefix: {'bf16': reference, 'te_fp8': candidate, 'delta': delta,
                         'relative_delta': delta / abs(float(reference)) if reference else None}}
    return {}


def _gradient_deltas(reference: list[dict[str, Any]] | None,
                     candidate: list[dict[str, Any]] | None) -> dict[str, Any] | None:
    if not isinstance(reference, list) or not isinstance(candidate, list):
        return None
    left = {row['name']: row for row in reference if row.get('present')}
    right = {row['name']: row for row in candidate if row.get('present')}
    shared = sorted(set(left) & set(right))
    return {
        'shared_parameter_count': len(shared),
        'missing_from_te_fp8': sorted(set(left) - set(right)),
        'missing_from_bf16': sorted(set(right) - set(left)),
        'per_parameter': {name: {
            'bf16_l2': left[name].get('l2'), 'te_fp8_l2': right[name].get('l2'),
            'l2_delta': (right[name]['l2'] - left[name]['l2'])
            if isinstance(left[name].get('l2'), (int, float)) and
               isinstance(right[name].get('l2'), (int, float)) else None,
            'bf16_max_abs': left[name].get('max_abs'), 'te_fp8_max_abs': right[name].get('max_abs'),
            'finite_bf16': left[name].get('finite'), 'finite_te_fp8': right[name].get('finite')}
            for name in shared},
    }


def summarize_run(output: Path) -> dict[str, Any]:
    train = _read_jsonl(output / 'direct_sft' / 'train.jsonl')
    updates = [row for row in train if isinstance(row.get('seconds'), (int, float))]
    warmed = updates[3:]
    def median(rows, key):
        vals = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
        return statistics.median(vals) if vals else None
    aggregate_seconds = sum(float(row['seconds']) for row in warmed
                            if isinstance(row.get('seconds'), (int, float)))
    aggregate_tokens = sum(int(row['supervised_tokens']) for row in warmed
                           if isinstance(row.get('supervised_tokens'), (int, float)))
    evaluations = _read_jsonl(output / 'direct_sft' / 'eval.jsonl')
    report = json.loads((output / 'direct_sft' / 'report.json').read_text(encoding='utf-8'))
    runtime_inventory = _read_jsonl(output / 'direct_sft' / 'fp8-runtime-inventory.jsonl')[-1:] or None
    initial_fp8 = report.get('fp8_training')
    def dtype_bytes(inventory):
        if not isinstance(inventory, dict):
            return {}
        totals = {}
        for module in (inventory.get('runtime_module_tensor_inventory') or {}).values():
            for dtype, count in (module.get('unique_storage_bytes_by_dtype') or {}).items():
                totals[dtype] = totals.get(dtype, 0) + int(count)
        return totals
    fp8_plan = updates[0].get('memory_plan', {}) if updates else {}
    return {
        'updates_logged': len(updates),
        'warm_updates_used': len(warmed),
        'warm_aggregate_wall_seconds': aggregate_seconds,
        'warm_aggregate_supervised_tokens': aggregate_tokens,
        'warm_aggregate_supervised_tokens_per_second':
            aggregate_tokens / aggregate_seconds if aggregate_seconds else None,
        'cold_first_update_seconds': updates[0].get('seconds') if updates else None,
        'first_update_loss': updates[0].get('loss') if updates else None,
        'warm_median_update_seconds': median(warmed, 'seconds'),
        'warm_median_supervised_tokens_per_second': median(warmed, 'supervised_tokens_per_second'),
        'peak_cuda_allocated_gb': max((float(row['cuda_peak_allocated_gb']) for row in updates
                                       if isinstance(row.get('cuda_peak_allocated_gb'), (int, float))), default=None),
        'peak_cuda_reserved_gb': max((float(row['cuda_peak_reserved_gb']) for row in updates
                                      if isinstance(row.get('cuda_peak_reserved_gb'), (int, float))), default=None),
        'peak_observed_incremental_bytes': max((int(row['observed_incremental_peak_bytes']) for row in updates
                                                if isinstance(row.get('observed_incremental_peak_bytes'), (int, float))),
                                               default=None),
        'first_evaluation': evaluations[:1] or None,
        'last_evaluation': evaluations[-1:] or None,
        'saved_tensor_audit_first_update': updates[0].get('saved_tensor_audit') if updates else None,
        'gradient_audit_first_update': updates[0].get('gradient_audit') if updates else None,
        'fp8_training_first_update': updates[0].get('fp8_training') if updates else None,
        'fp8_runtime_inventory': runtime_inventory,
        'fp8_memory_attribution': {
            'kind': 'post-update module-owned unique tensor storage inventory; excludes transient kernels/workspaces and is not total allocator memory',
            'eligible_bf16_parameter_bytes': initial_fp8.get('eligible_parameter_bytes')
            if isinstance(initial_fp8, dict) else None,
            'initial_module_owned_bytes_by_dtype': dtype_bytes(initial_fp8),
            'post_update_module_owned_bytes_by_dtype': dtype_bytes(runtime_inventory[0])
            if runtime_inventory else {},
            'plan_fp8_weight_cache_estimate_bytes': fp8_plan.get('fp8_weight_cache_estimate_bytes'),
            'plan_fp8_weight_cache_reservation_bytes': fp8_plan.get('fp8_weight_cache_reservation_bytes'),
            'peak_cuda_allocated_gb': max((float(row['cuda_peak_allocated_gb']) for row in updates
                                            if isinstance(row.get('cuda_peak_allocated_gb'), (int, float))),
                                           default=None),
            'peak_cuda_reserved_gb': max((float(row['cuda_peak_reserved_gb']) for row in updates
                                           if isinstance(row.get('cuda_peak_reserved_gb'), (int, float))),
                                          default=None),
            'bf16_master_parameters_and_optimizer_retained': True,
        },
    }


def run_variant(variant: dict[str, Any], output_root: Path, device: str, artifact_root: Path,
                hf_hub_cache: Path | None, sample_seconds: float):
    recipe_path = Path(variant['recipe_path'])
    recipe = variant['recipe']
    output = Path(recipe['outputs']['directory'])
    try:
        output.relative_to(artifact_root.resolve())
        output_root.resolve().relative_to(artifact_root.resolve())
    except ValueError as exc:
        raise ValueError('comparison recipes, outputs, and receipts must be under the declared artifact root') from exc
    metadata = output_root / 'launches' / f"{recipe['id']}.json"
    metadata.parent.mkdir(parents=True, exist_ok=True)
    # Keep the coordinator's stdout separate from the declared container log:
    # the launcher otherwise defaults to metadata.with_suffix('.log'), which
    # collides with a wrapper log named after the recipe ID.
    launch_log = output_root / 'launches' / f"{recipe['id']}.wrapper.log"
    container_log = output_root / 'launches' / f"{recipe['id']}.container.log"
    telemetry = output_root / 'telemetry' / f"{recipe['id']}.jsonl"
    telemetry.parent.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    thread = threading.Thread(target=_gpu_sample, args=(stop, telemetry, sample_seconds), daemon=True)
    thread.start()
    env = os.environ.copy()
    argv = [sys.executable, str(REPOSITORY_ROOT / 'scripts' / 'run_neuralese_recipe.py'),
            '--recipe', str(recipe_path), '--artifact-root', str(artifact_root),
            '--launch-metadata', str(metadata), '--log', str(container_log), '--device', device]
    if hf_hub_cache is not None:
        argv.extend(['--hf-hub-cache', str(hf_hub_cache)])
    try:
        with launch_log.open('w', encoding='utf-8') as log:
            completed = subprocess.run(argv, cwd=REPOSITORY_ROOT, env=env,
                                       stdout=log, stderr=subprocess.STDOUT, check=False)
    finally:
        stop.set()
        thread.join()
    summary = {'recipe_id': recipe['id'], 'recipe_path': str(recipe_path),
               'recipe_sha256': variant['recipe_sha256'], 'output': str(output),
               'launch_metadata': str(metadata), 'launch_log': str(launch_log),
               'container_log': str(container_log),
               'telemetry': str(telemetry), 'exit_code': completed.returncode,
               'measurements': summarize_run(output),
               'gpu_telemetry_summary': _telemetry_summary(telemetry),
               'execution_plans': _execution_plans(container_log)}
    summaries = output_root / 'summaries'
    summaries.mkdir(parents=True, exist_ok=True)
    write_json(summaries / f"{recipe['id']}.json", summary)
    if completed.returncode:
        raise RuntimeError(f"comparison variant failed ({completed.returncode}); wrapper={launch_log}; container={container_log}")
    if summary['measurements']['warm_updates_used'] < 2:
        raise RuntimeError(f"comparison variant has fewer than two warmed updates; wrapper={launch_log}; container={container_log}")
    return summary


def comparison_result(root: Path, plan_path: Path, summaries: list[dict[str, Any]]) -> dict[str, Any]:
    by_profile = {}
    for summary in summaries:
        recipe_id = summary['recipe_id']
        profile = 'short' if '-short-' in recipe_id else 'long'
        backend = 'te_fp8' if '-te-fp8' in recipe_id else 'bf16'
        by_profile.setdefault(profile, {})[backend] = summary
    paired = {}
    for profile, runs in by_profile.items():
        if set(runs) != {'bf16', 'te_fp8'}:
            raise RuntimeError(f'incomplete comparison pair for {profile}')
        bf16, fp8 = runs['bf16'], runs['te_fp8']
        if _window_schedule(bf16['execution_plans']) != _window_schedule(fp8['execution_plans']):
            raise RuntimeError(f'{profile} pair sampled different update windows; measurements are not comparable')
        b, f = bf16['measurements'], fp8['measurements']
        btime, ftime = b['warm_median_update_seconds'], f['warm_median_update_seconds']
        btok, ftok = b['warm_median_supervised_tokens_per_second'], f['warm_median_supervised_tokens_per_second']
        paired[profile] = {
            'warm_median_update_speedup': btime / ftime if btime and ftime else None,
            'warm_median_supervised_tokens_per_second_ratio': ftok / btok if btok and ftok else None,
            'same_schedule_warm_aggregate_supervised_tokens_per_second_ratio':
                (f['warm_aggregate_supervised_tokens_per_second'] /
                 b['warm_aggregate_supervised_tokens_per_second']
                 if b.get('warm_aggregate_supervised_tokens_per_second') and
                    f.get('warm_aggregate_supervised_tokens_per_second') else None),
            'incremental_peak_bytes_ratio': (f['peak_observed_incremental_bytes'] /
                                             b['peak_observed_incremental_bytes']
                                             if b['peak_observed_incremental_bytes'] and
                                             f['peak_observed_incremental_bytes'] else None),
            'total_allocated_peak_gb_ratio': (f['peak_cuda_allocated_gb'] / b['peak_cuda_allocated_gb']
                                              if b['peak_cuda_allocated_gb'] and
                                                 f['peak_cuda_allocated_gb'] else None),
            'first_update_loss_delta': (f['first_update_loss'] - b['first_update_loss']
                                        if isinstance(b['first_update_loss'], (int, float)) and
                                           isinstance(f['first_update_loss'], (int, float)) else None),
            'bf16_initial_and_final_eval': {'initial': b['first_evaluation'], 'final': b['last_evaluation']},
            'fp8_initial_and_final_eval': {'initial': f['first_evaluation'], 'final': f['last_evaluation']},
            'initial_eval_numeric_deltas': _numeric_deltas(
                b['first_evaluation'][0] if b['first_evaluation'] else {},
                f['first_evaluation'][0] if f['first_evaluation'] else {}),
            'final_eval_numeric_deltas': _numeric_deltas(
                b['last_evaluation'][0] if b['last_evaluation'] else {},
                f['last_evaluation'][0] if f['last_evaluation'] else {}),
            'first_update_gradient_deltas': _gradient_deltas(
                b['gradient_audit_first_update'], f['gradient_audit_first_update']),
            'same_execution_windows': True,
            'checkpointed_update_counts': {
                'bf16': sum(bool(row.get('checkpoint_layers')) for row in bf16['execution_plans']),
                'te_fp8': sum(bool(row.get('checkpoint_layers')) for row in fp8['execution_plans'])},
        }
    result = {'schema': 'natlang.neuralese-fp8-direct-comparison-result/1',
              'plan': str(plan_path), 'paired_profiles': paired, 'runs': summaries}
    write_json(root / 'comparison-result.json', result)
    return result


def execute_prepared_plan(plan_path: Path, *, artifact_root: Path, hf_hub_cache: Path | None,
                          device: str | None, sample_seconds: float) -> int:
    plan_path = plan_path.expanduser().resolve()
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    if plan.get('schema') != 'natlang.neuralese-fp8-direct-comparison/1':
        raise ValueError('prepared comparison plan has an unsupported schema')
    root = Path(plan.get('output_root', '')).expanduser().resolve()
    if root != plan_path.parent:
        raise ValueError('prepared plan must live at the declared comparison output root')
    artifact_root = artifact_root.expanduser().resolve()
    actual_device = device or plan.get('device', 'cuda')
    image = plan.get('runtime_image')
    if not isinstance(image, str) or not image.startswith('sha256:') or len(image) != 71:
        raise ValueError('prepared plan must declare its pinned runtime image')
    variants = {}
    for entries in plan.get('profiles', {}).values():
        for entry in entries:
            recipe_path = Path(entry['recipe_path']).expanduser().resolve()
            if sha256_file_hex_cached(recipe_path) != entry['recipe_sha256']:
                raise ValueError(f'prepared recipe SHA changed: {recipe_path}')
            recipe = load_recipe(recipe_path)
            validate_direct_stage_recipe(recipe)
            _validate_inputs(recipe)
            if recipe.get('runtime', {}).get('image') != image:
                raise ValueError(f'prepared recipe runtime differs from plan: {recipe_path}')
            variants[str(recipe_path)] = {**entry, 'recipe_path': str(recipe_path), 'recipe': recipe}
    order = plan.get('execution_order')
    if not isinstance(order, list) or len(order) != len(variants) or set(order) != set(variants):
        raise ValueError('prepared execution order must name every recipe exactly once')
    if (root / 'execution-intent.json').exists() or (root / 'comparison-result.json').exists():
        raise FileExistsError('prepared plan has already been executed or has an execution receipt')
    intent = {'schema': 'natlang.neuralese-fp8-direct-comparison-execution/1',
              'plan_path': str(plan_path), 'plan_sha256': sha256_file_hex_cached(plan_path),
              'runtime_image': image, 'device': actual_device, 'started_at_unix': time.time(),
              'execution_order': order, 'status': 'running'}
    write_json(root / 'execution-intent.json', intent)
    summaries = []
    for raw_path in order:
        summaries.append(run_variant(variants[raw_path], root, actual_device, artifact_root,
                                     hf_hub_cache, sample_seconds))
    result = comparison_result(root, plan_path, summaries)
    intent.update(status='complete', completed_at_unix=time.time(),
                  result_path=str(root / 'comparison-result.json'),
                  result_sha256=sha256_file_hex_cached(root / 'comparison-result.json'))
    write_json(root / 'execution-intent.json', intent)
    print(json.dumps({'result': str(root / 'comparison-result.json'), 'runs': len(summaries)}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--short-recipe', type=Path,
                        help='reviewed shared neuralese_direct recipe for the short-context profile')
    parser.add_argument('--long-recipe', type=Path,
                        help='reviewed shared neuralese_direct recipe for the long-context profile')
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--run-plan', type=Path,
                        help='execute an immutable prepared comparison plan without rebuilding its recipes')
    parser.add_argument('--runtime-image', default=DEFAULT_FP8_IMAGE)
    parser.add_argument('--artifact-root', type=Path, default=Path('/srv/storage'))
    parser.add_argument('--hf-hub-cache', type=Path,
                        help='optional host Hugging Face hub root passed to the container launcher')
    parser.add_argument('--device')
    parser.add_argument('--saved-tensor-audit-steps', type=int, default=1)
    parser.add_argument('--gpu-sample-seconds', type=float, default=1.0)
    parser.add_argument('--run', action='store_true', help='execute sequential paired recipes; otherwise prepare only')
    args = parser.parse_args(argv)
    if args.saved_tensor_audit_steps < 0 or args.gpu_sample_seconds <= 0:
        raise ValueError('audit steps must be nonnegative and GPU sampling interval positive')
    if args.run_plan is not None:
        if args.short_recipe or args.long_recipe or args.output_root or args.run:
            parser.error('--run-plan cannot be combined with recipe templates, --output-root, or --run')
        return execute_prepared_plan(args.run_plan, artifact_root=args.artifact_root,
                                     hf_hub_cache=args.hf_hub_cache, device=args.device,
                                     sample_seconds=args.gpu_sample_seconds)
    if args.short_recipe is None or args.long_recipe is None or args.output_root is None:
        parser.error('preparation requires --short-recipe, --long-recipe, and --output-root')
    if args.device is None:
        args.device = 'cuda'
    root = args.output_root.resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError(f'comparison output root must be empty: {root}')
    if not args.runtime_image.startswith('sha256:') or len(args.runtime_image) != 71:
        raise ValueError('comparison requires an image pinned by sha256 digest')
    root.mkdir(parents=True, exist_ok=True)
    order = []
    profiles = {}
    loaded_templates = [load_recipe(args.short_recipe), load_recipe(args.long_recipe)]
    for template in loaded_templates:
        if template.get('schema') != DIRECT_STAGE_SCHEMA or template.get('stage', {}).get('kind') != 'neuralese_direct':
            raise ValueError('both profiles must use declared neuralese_direct templates')
        validate_direct_stage_recipe(template)
        _validate_inputs(template)
    if _profile_identity(loaded_templates[0]) != _profile_identity(loaded_templates[1]):
        raise ValueError('short and long profiles may differ only in the declared context-length parameters')
    for profile, template_path in (('short', args.short_recipe), ('long', args.long_recipe)):
        variants = build_pair(template_path, root, profile, args.runtime_image,
                              saved_tensor_audit_steps=args.saved_tensor_audit_steps)
        profiles[profile] = [{'backend': value['backend'], 'recipe_path': value['recipe_path'],
                              'recipe_sha256': value['recipe_sha256']} for value in variants]
        # Alternate ordering across profiles to reduce a systematic clock/time-of-run bias.
        order.extend(variants if profile == 'short' else list(reversed(variants)))
    plan = {'schema': 'natlang.neuralese-fp8-direct-comparison/1', 'output_root': str(root),
            'runtime_image': args.runtime_image, 'device': args.device,
            'templates': {'short': {'path': str(args.short_recipe.resolve()),
                                    'sha256': sha256_file_hex_cached(args.short_recipe)},
                          'long': {'path': str(args.long_recipe.resolve()),
                                   'sha256': sha256_file_hex_cached(args.long_recipe)}},
            'profiles': profiles, 'execution_order': [v['recipe_path'] for v in order],
            'timing_policy': 'discard first three updates; report median of remaining updates',
            'gpu_telemetry': 'nvidia-smi SM/memory clocks, power, temperature and memory sampled each second',
            'task_quality_scope': 'only the existing declared recipe evaluations; no runtime qualification or admission',
            'run_requested': args.run}
    write_json(root / 'comparison-plan.json', plan)
    if not args.run:
        print(json.dumps({'prepared': str(root / 'comparison-plan.json'), 'run_requested': False}, indent=2))
        return 0
    summaries = []
    for variant in order:
        summaries.append(run_variant(variant, root, args.device, args.artifact_root,
                                     args.hf_hub_cache, args.gpu_sample_seconds))
    by_profile = {}
    for summary in summaries:
        recipe_id = summary['recipe_id']
        profile = 'short' if '-short-' in recipe_id else 'long'
        backend = 'te_fp8' if '-te-fp8' in recipe_id else 'bf16'
        by_profile.setdefault(profile, {})[backend] = summary
    result = comparison_result(root, root / 'comparison-plan.json', summaries)
    print(json.dumps({'result': str(root / 'comparison-result.json'), 'runs': len(summaries)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
