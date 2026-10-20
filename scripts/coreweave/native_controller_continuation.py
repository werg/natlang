"""Exact, metadata-only continuation of an authenticated supervised handoff."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path


def prepare_drained_supervised_continuation(record, saved, *, drain, resident,
                                            package, directory, quantum_updates,
                                            write):
    """Apply a recipe owner's declared quantum without granting stage credit.

    ``drain`` and ``resident`` are the caller's already authenticated runtime
    modules. This function never loads a checkpoint, model, or tokenizer.
    """
    if type(quantum_updates) is not int or quantum_updates <= 0:
        raise ValueError('supervised continuation needs a positive declared quantum')
    _, plan, stage_id = drain._drained_plan_authority(record, saved)
    if plan['code'] != resident.code_inventory(package):
        raise ValueError('drained continuation runtime changed')
    initial_ref = record['resident']['initial']
    initial = json.loads(resident.bound(initial_ref).read_bytes())
    capacity = initial.get('geometry_qualification')
    if not capacity:
        raise ValueError('drained continuation needs actual resident capacity')
    limits = resident.geometry_capacity(capacity, identity=initial['geometry_identity'],
                                        code=plan['code'])
    recipe = copy.deepcopy(plan['recipe'])
    stages = [stage for stage in recipe['stages'] if stage['id'] == stage_id]
    if len(stages) != 1 or stages[0]['kind'] != 'native_state_sft':
        raise ValueError('drained continuation needs its actual supervised stage kind')
    parameters = stages[0].setdefault('parameters', {})
    parameters['refresh_after_updates'] = quantum_updates
    parameters['resident_geometry_qualification'] = capacity['path']
    parameters.pop('memory_qualification', None)
    directory = Path(directory)

    def emit(path, value):
        if path.exists():
            if json.loads(path.read_bytes()) != value:
                raise ValueError('immutable drained continuation changed on retry')
        else:
            write(path, value)
        return resident.binding(path)

    reference = emit(directory / 'drained-continuing-recipe.json', recipe)
    proof = {'schema': 'natlang.owned-drained-continuation/1',
             'source_plan_sha256': resident.digest(plan),
             'source_resident': record['resident'], 'recipe': reference,
             'stage_id': stage_id, 'capacity': capacity, 'limits': limits,
             'runtime_code_sha256': resident.digest(plan['code']),
             'whole_recipe_completed': False, 'exposure_credit': 0}
    emit(directory / 'drained-continuation-proof.json', proof)
    return reference


def runtime_child_environment(runtime, environment=None):
    """Propagate the exact bound runtime to controller subprocess imports.

    The caller verifies the runtime inventory. Dynamic ``sys.path`` changes
    affect only the current interpreter and cannot substitute for this binding.
    """
    root = Path(runtime).resolve(strict=True)
    package = root / 'training/neuralese/natlang_neuralese'
    if not (package / '__init__.py').is_file():
        raise ValueError('controller child runtime lacks its Python package')
    result = dict(os.environ if environment is None else environment)
    result['NATLANG_REPO'] = str(root)
    result['PYTHONPATH'] = str(root / 'training/neuralese') + os.pathsep + str(root)
    return result
