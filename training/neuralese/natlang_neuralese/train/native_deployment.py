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


def prepare(release_path, curriculum_path, out, *, participants, cases=None, task_source_manifest=None,
            runtime, namespace='/work/native-production',
            refresh_curriculum_path=None):
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
    task_source = None
    if task_source_manifest is None and cases is None:
        task_source_manifest = Path(__file__).resolve().parents[4] / 'runs/decision-rich-train-20261016-v1/task-source-manifest.json'
    if task_source_manifest is not None:
        from ..rl.task_source import resolve_task_source
        task_source = resolve_task_source(task_source_manifest)
        if cases is not None and Path(cases).resolve(strict=True) != task_source['cases']:
            raise ValueError('explicit RL cases differ from the selected admitted task-source manifest')
        cases = task_source['cases']
    if cases is None:
        raise ValueError('native deployment requires an admitted RL task source')
    cases_binding = binding(cases)
    case_rows = [row for _, row in iter_jsonl_rows(cases)]
    if not case_rows or any(row.get('split') != 'train' for row in case_rows):
        raise ValueError('native deployment requires the actual admitted nonempty train-only case packet')
    semantic = json.loads(curriculum_path.read_text())
    refresh = json.loads(Path(refresh_curriculum_path).read_text()) if refresh_curriculum_path else semantic
    for declaration in (semantic, refresh):
        if declaration.get('schema') != 'natlang.native-semantic-training/1':
            raise ValueError('native deployment requires the declared semantic curriculum')
        for key in ('pairs', 'held_pairs'):
            _bound(declaration[key])
        # A refresh quantum controls safe handoff, never production completion.
        # Collected whole episodes supply the actual policy_training sources;
        # an empty source declaration must not masquerade as trained exposure.
        declaration.setdefault('policy_schedule', {'id':'policy', 'epochs':1,
            'every_semantic_updates':1, 'updates_per_refresh':16})
    out = out.resolve(); out.mkdir(parents=True, exist_ok=True)
    queue = str(Path(namespace)/'shared/bridge-ipc')
    client = [sys.executable, '-m', 'natlang_neuralese.rl.native_bridge', '--queue', queue,
              '--request', '{request}', '--result', '{result}', '--stop-file', '{stop_file}']
    models, requirements = [], []
    for record in sorted(records, key=lambda row: row['role']):
        role = record['role']; line = 'lfm25' if role == 'lfm' else 'mellum21'
        recipe_path = _bound({'path': record['recipe_path'], 'sha256': record['recipe_sha256']})
        adoption_binding = record.get('resident_adoption')
        if adoption_binding:
            from .native_resident import current
            adoption_path = _bound(adoption_binding)
            adoption = current(adoption_path.parent)
            if (not adoption or adoption_path.name != 'current.json' or adoption['payload']['recipe']['path'] != str(recipe_path)
                    or adoption['payload']['recipe']['file_sha256'] != record['recipe_sha256']):
                raise ValueError('resident release does not bind its actual current recipe')
        recipe = resolve_recipe_data(recipe_path)
        stages = [stage for stage in recipe['stages'] if stage['kind'] == 'native_state_sft']
        if len(stages) != 1:
            raise ValueError('foundation release must identify its single shared native SFT stage')
        stage = stages[0]
        inputs = resolve_stage_inputs(recipe, stage, {})
        parameters = effective_stage_parameters(stage, [], recipe)
        cadence = parameters.get('task_eval_every', 0)
        if not parameters.get('task_eval') or type(cadence) is not int or cadence < 0:
            raise ValueError('native deployment requires actual boundary task evaluation and nonnegative routine cadence')
        # Zero disables routine synchronous pauses, while the native trainer
        # still evaluates exact-current foundation/refresh/final weights.
        for key in ('native_graphs', 'held_native_graphs', 'foundation_native_graphs'):
            _bound(inputs[key])
        if (Path(inputs['foundation_native_graphs']['path']).resolve() != Path(record['foundation_manifest']['path']).resolve()
                or inputs['foundation_native_graphs']['sha256'] != record['foundation_manifest']['sha256']):
            raise ValueError('release foundation differs from the actual training recipe')
        if role == 'mellum':
            for declaration in (semantic, refresh):
                declaration['task_eval'] = parameters['task_eval']
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
        for declaration in (semantic, refresh):
            declaration.setdefault('models', {}).setdefault(participants[role], {})['base'] = str(base)
        models.append({'model': role, 'native_participant': participants[role],
            'production_recipe': binding(recipe_path), 'production_stage': stage['id'],
            'seed_native_graphs': inputs['native_graphs'], 'held_native_graphs': inputs['held_native_graphs'],
            'cases': cases_binding, 'runtime': str(runtime/'ts-host'), 'base': base,
            **({'task_source': {'manifest': str(task_source['manifest']),
                'manifest_sha256': task_source['manifest_sha256'], 'cases_sha256': task_source['cases_sha256'],
                'rows': task_source['rows'], 'audit_sha256': task_source['audit_sha256'],
                'selection_proof_sha256': task_source['selection_proof_sha256']}} if task_source else {}),
            'base_inventory_sha256': inventory_sha,
            'port': 8097 if role == 'lfm' else 8098, 'provider_queue': queue,
            'provider': {'sandbox_id': record['sandbox_id']}})
        requirements.append({'model_id': participants[role],
            'readiness': str(Path(namespace)/line/'native_full_sft/native-training-receipt.json'),
            'trainer_exit': str(Path(record['control'])/'exit.json'),
            'trainer_launch': binding(Path(record['control'])/'launch.json'),
            'trainer_recipe': str(recipe_path),
            **({'resident_adoption': adoption_binding} if adoption_binding else {}),
            'recipe_sha256': recipe_identity_sha256(recipe_path), 'stage_ids': [stage['id']],
            'readiness_boundary': 'communication-foundation',
            'foundation_sources': inputs['foundation_native_graphs'],
            'foundation_epochs': parameters.get('epochs', 1.), 'forward_atol': 1e-4, 'gradient_atol': 1e-5})
    curriculum = out/'curriculum.json'; write_json_durable(curriculum, semantic)
    refresh_curriculum = curriculum
    if refresh_curriculum_path:
        refresh_curriculum = out/'refresh-curriculum.json'
        write_json_durable(refresh_curriculum, refresh)
    bootstrap = out/'rl-bootstrap.json'
    write_json_durable(bootstrap, {'schema': 'natlang.native-rl-bootstrap/1', 'models': models,
        'semantic_refresh_config': binding(refresh_curriculum), 'refresh_command': client,
        'refresh_provider': next(model['provider'] for model in models if model['model'] == 'mellum')})
    raw_bootstrap=out/'raw-rl-bootstrap.json'
    write_json_durable(raw_bootstrap, {'schema':'natlang.native-raw-rl-bootstrap/1',
        'models':[{**model,'handoff_requirement':requirement}
                  for model,requirement in zip(models,requirements)]})
    controller = out/'controller.json'
    shared = Path(namespace)/'shared'
    write_json_durable(controller, {'schema': 'natlang.native-shared-stage-controller/1',
        'out': str(shared/'native_shared_stage'), 'curriculum': str(curriculum),
        'participants': requirements, 'poll_seconds': 60,
        'command': [sys.executable, '-m', 'natlang_neuralese.train.native_semantic_stage',
            '--config', str(curriculum), '--participants', '{participants}', '--out', '{out}', '--device', 'cuda'],
        'rl_bootstrap': binding(bootstrap), 'rl_out': str(shared/'native_rl'), 'rl_device': 'cuda',
        'raw_rl_bootstrap':binding(raw_bootstrap),
        'deployment_inputs': {'release': binding(release_path), 'curriculum': binding(curriculum_path),
            **({'refresh_curriculum':binding(refresh_curriculum_path)} if refresh_curriculum_path else {})}})
    return controller


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('release', 'curriculum', 'out', 'runtime'):
        parser.add_argument('--'+flag, type=Path, required=True)
    parser.add_argument('--cases', type=Path, help='explicit admitted train task packet; defaults to the bound 128-row RL source')
    parser.add_argument('--task-source-manifest', type=Path,
                        help='integrity-bound admitted RL task source; defaults to decision-rich-train-20261016-v1')
    parser.add_argument('--participant', action='append', required=True, help='role=exact architecture:base-inventory identity')
    parser.add_argument('--namespace', default='/work/native-production')
    parser.add_argument('--refresh-curriculum', type=Path,
                        help='expanded refresh source; keeps the first shared foundation curriculum fixed')
    args = parser.parse_args(argv)
    participants = dict(item.split('=', 1) for item in args.participant)
    print(prepare(args.release, args.curriculum, args.out, participants=participants, cases=args.cases,
        task_source_manifest=args.task_source_manifest,
        runtime=args.runtime, namespace=args.namespace, refresh_curriculum_path=args.refresh_curriculum))


if __name__ == '__main__':
    main()
