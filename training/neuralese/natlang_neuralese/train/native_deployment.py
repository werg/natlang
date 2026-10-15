"""Bind the existing foundation rendezvous and RL controller to real cloud assets.

Run on the coordinator sandbox after the owner mirrors both release recipes and
source closures. No provider credentials, resources, models or trainers here.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

from ..common.hashing import sha256_file_hex_cached, canonical_json_sha256_hex
from ..common.jsonio import write_json_durable, iter_jsonl_rows
from .native_readiness import _bound
from .recipe import resolve_recipe_data, resolve_stage_inputs, effective_stage_parameters, recipe_identity_sha256


def binding(path):
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'sha256': sha256_file_hex_cached(path)}


def prepare(release_path, curriculum_path, out, *, participants, cases, runtime, namespace='/work/native-production'):
    """Create runnable configuration without inventing future trained identities."""
    release_path, curriculum_path, out = map(Path, (release_path, curriculum_path, out))
    records = json.loads(release_path.read_text())
    if {row['role'] for row in records} != {'lfm', 'mellum'} or len(records) != 2:
        raise ValueError('native deployment requires exactly the two current training releases')
    if set(participants) != {'lfm', 'mellum'} or len(set(participants.values())) != 2:
        raise ValueError('native deployment must name both exact distinct participants')
    runtime = Path(runtime).resolve(strict=True)
    if not (runtime/'ts-host/dist/teacher/cli.js').is_file():
        raise ValueError('native deployment requires the actual built runtime')
    cases_binding = binding(cases)
    case_rows = [row for _, row in iter_jsonl_rows(cases)]
    if not case_rows or any(row.get('split') != 'train' for row in case_rows):
        raise ValueError('native deployment requires the actual admitted nonempty train-only case packet')
    semantic = json.loads(curriculum_path.read_text())
    if semantic.get('schema') != 'natlang.native-semantic-training/1':
        raise ValueError('native deployment requires the declared semantic curriculum')
    for key in ('pairs', 'held_pairs'):
        _bound(semantic[key])
    out = out.resolve(); out.mkdir(parents=True, exist_ok=True)
    queue = str(Path(namespace)/'shared/bridge-ipc')
    client = [sys.executable, '-m', 'natlang_neuralese.rl.native_bridge', '--queue', queue,
              '--request', '{request}', '--result', '{result}', '--stop-file', '{stop_file}']
    models, requirements = [], []
    for record in sorted(records, key=lambda row: row['role']):
        role = record['role']; line = 'lfm25' if role == 'lfm' else 'mellum21'
        recipe_path = _bound({'path': record['recipe_path'], 'sha256': record['recipe_sha256']})
        recipe = resolve_recipe_data(recipe_path)
        stages = [stage for stage in recipe['stages'] if stage['kind'] == 'native_state_sft']
        if len(stages) != 1:
            raise ValueError('foundation release must identify its single shared native SFT stage')
        stage = stages[0]
        inputs = resolve_stage_inputs(recipe, stage, {})
        parameters = effective_stage_parameters(stage, [], recipe)
        if not parameters.get('task_eval') or not parameters.get('task_eval_every'):
            raise ValueError('native deployment cannot omit actual task evaluation')
        for key in ('native_graphs', 'held_native_graphs', 'foundation_native_graphs'):
            _bound(inputs[key])
        if (Path(inputs['foundation_native_graphs']['path']).resolve() != Path(record['foundation_manifest']['path']).resolve()
                or inputs['foundation_native_graphs']['sha256'] != record['foundation_manifest']['sha256']):
            raise ValueError('release foundation differs from the actual training recipe')
        if role == 'mellum':
            semantic['task_eval'] = parameters['task_eval']
        declared_base = record.get('base') or parameters.get('base')
        if not declared_base:
            raise ValueError('native deployment requires an explicit actual base path from release or recipe')
        base = declared_base['path'] if isinstance(declared_base, dict) else declared_base
        from ..model.native_model import base_inventory
        inventory = base_inventory(base)
        inventory_sha = canonical_json_sha256_hex(inventory)
        if (inventory_sha != participants[role].split(':', 1)[-1]
                or isinstance(declared_base, dict) and declared_base.get('inventory_sha256') != inventory_sha):
            raise ValueError('native deployment resident base inventory differs from its exact participant')
        semantic.setdefault('models', {}).setdefault(participants[role], {})['base'] = str(base)
        models.append({'model': role, 'native_participant': participants[role],
            'production_recipe': binding(recipe_path), 'production_stage': stage['id'],
            'seed_native_graphs': inputs['native_graphs'], 'held_native_graphs': inputs['held_native_graphs'],
            'cases': cases_binding, 'runtime': str(runtime/'ts-host'), 'base': base,
            'base_inventory_sha256': inventory_sha,
            'port': 8097 if role == 'lfm' else 8098, 'provider_queue': queue,
            'provider': {'sandbox_id': record['sandbox_id']}})
        requirements.append({'model_id': participants[role],
            'readiness': str(Path(namespace)/line/'native_full_sft/native-training-receipt.json'),
            'trainer_exit': str(Path(record['control'])/'exit.json'),
            'trainer_launch': binding(Path(record['control'])/'launch.json'),
            'trainer_recipe': str(recipe_path),
            'recipe_sha256': recipe_identity_sha256(recipe_path), 'stage_ids': [stage['id']],
            'readiness_boundary': 'communication-foundation',
            'foundation_sources': inputs['foundation_native_graphs'],
            'foundation_epochs': parameters.get('epochs', 1.), 'forward_atol': 1e-4, 'gradient_atol': 1e-5})
    curriculum = out/'curriculum.json'; write_json_durable(curriculum, semantic)
    bootstrap = out/'rl-bootstrap.json'
    write_json_durable(bootstrap, {'schema': 'natlang.native-rl-bootstrap/1', 'models': models,
        'semantic_refresh_config': binding(curriculum), 'refresh_command': client,
        'refresh_provider': next(model['provider'] for model in models if model['model'] == 'mellum')})
    controller = out/'controller.json'
    write_json_durable(controller, {'schema': 'natlang.native-shared-stage-controller/1',
        'out': str(out/'shared-stage'), 'curriculum': str(curriculum),
        'participants': requirements, 'poll_seconds': 60,
        'command': [sys.executable, '-m', 'natlang_neuralese.train.native_semantic_stage',
            '--config', str(curriculum), '--participants', '{participants}', '--out', '{out}', '--device', 'cuda'],
        'rl_bootstrap': binding(bootstrap), 'rl_out': str(out/'rl'), 'rl_device': 'cuda',
        'deployment_inputs': {'release': binding(release_path), 'curriculum': binding(curriculum_path)}})
    return controller


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('release', 'curriculum', 'out', 'cases', 'runtime'):
        parser.add_argument('--'+flag, type=Path, required=True)
    parser.add_argument('--participant', action='append', required=True, help='role=exact architecture:base-inventory identity')
    parser.add_argument('--namespace', default='/work/native-production')
    args = parser.parse_args(argv)
    participants = dict(item.split('=', 1) for item in args.participant)
    print(prepare(args.release, args.curriculum, args.out, participants=participants, cases=args.cases,
        runtime=args.runtime, namespace=args.namespace))


if __name__ == '__main__':
    main()
