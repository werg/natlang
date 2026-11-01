"""Bind resident source authority before a direct supervised trainer allocates a model."""
from pathlib import Path
import json


def ensure_plan(config, *, package=None):
    from ..train import native_resident as resident
    from ..train.recipe import load_recipe, resolve_stage_inputs, _binding_files
    from ..common.jsonio import write_json_durable
    from .native_supervised import recipe_lineage

    parameters = config.get('trainer_parameters') or {}
    directory = parameters.get('resident_data_directory', config.get('resident_data_directory'))
    if not directory:
        return None
    directory = Path(directory).resolve()
    package = Path(package or Path(__file__).resolve().parents[1])
    code = resident.code_inventory(package)
    path = directory.parent / 'recipe-plan.json'
    # A committed resident generation owns its plan, even when launch controls
    # still name the initial source. Never overwrite a published adoption.
    state = resident.current(directory)
    if state:
        plan = state['payload']['plan']
        if plan.get('code') != code:
            raise ValueError('supervised resident plan needs a drained runtime handoff')
        if plan.get('recipe_sha256') != recipe_lineage(config):
            raise ValueError('supervised resident plan changed recipe lineage')
        resident.recover(directory)
        return resident.binding(path)
    recipe = load_recipe(config['production_recipe'])
    inputs = {stage['id']: ({} if stage.get('admitted') is False else
              resolve_stage_inputs(recipe, stage, {})) for stage in recipe['stages']}
    stage = config['production_stage']
    actual = inputs[stage]
    for role, key in (('native_graphs', 'seed_native_graphs'), ('held_native_graphs', 'held_native_graphs')):
        if resident.binding(config[key]) != {k: actual[role][k] for k in ('path', 'sha256')}:
            raise ValueError('supervised resident plan differs from launch source roles')
    plan = {'schema': 'natlang.neuralese-recipe-plan/1', 'recipe': recipe,
            'recipe_sha256': recipe_lineage(config), 'code': code,
            'device': parameters.get('device', 'cuda'), 'stage_inputs': inputs,
            'inputs': {ref['path']: ref['sha256'] for resolved in inputs.values()
                       for value in resolved.values() for ref in _binding_files(value)}}
    with resident.locked(directory):
        # Serialize creation with resident publication and refuse to silently
        # replace an existing plan belonging to another runtime or recipe.
        if resident.current(directory):
            raise ValueError('resident generation appeared during supervised plan initialization')
        if path.exists():
            old = json.loads(path.read_text())
            for key in ('schema', 'recipe', 'recipe_sha256', 'code', 'stage_inputs'):
                if old.get(key) != plan[key]:
                    raise ValueError('supervised resident plan changed; use a distinct drained authority directory')
        else:
            write_json_durable(path, plan)
    return resident.binding(path)
