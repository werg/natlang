"""Carry exact measured derivatives across an explicitly reviewed metadata transition."""
from __future__ import annotations
import json
from pathlib import Path
from . import native_resident as resident
from . import native_source_classification as classification

# The complete R7 -> R8 source diff was independently reviewed. This registry
# grants no permission for a different hash, even in a metadata-named module.
APPROVED = {('52bd76263cb757ffef1a197c755efde1323a6e272d73cab910495a5c6335e996', 'db405a13512666a9c9f38ea5d7130d77513e708b2a5b9cabf2d9e9898cae154e'): {'eval/task_cadence.py': {'old': '2af90e850ade60b5146b5c303201065235bf28032665a2af80becb2c306c0562', 'new': '2c3e547ac64ac9512264091812a1f86eab5a0d71680d79475dc5aef5bea5c4f7'}, 'rl/native_completed_supervised.py': {'old': '281b1f30ec1f403f10fe262cb0ff829d5bcdb295de2f92b8d74b13dd1a3f77ed', 'new': '0563edba9b62758f8055fa13f441c4caef8dbc44b1138b453c088f4b2b2044c4'}, 'rl/native_raw_rounds.py': {'old': '4d8f624224ae2e624f36d09eb7e9d4ef4b033d26541ea09ec9ffeaf8cf2cdcfe', 'new': 'e5efafbfbbe51e9ec032d430921b33f800ac3fc29e0c5ce5f1263d2cf14e8608'}, 'rl/native_runtime_handoff.py': {'old': 'd828e9f6d73f4ada469ca2f5695ff29acdb791a21ab4021dde19c8bff422c7a7', 'new': '269445c6b86c5aa836d5cbcd7956b74e0cfcc05217cff8ee2ed893fecb36b261'}, 'train/native_continuation.py': {'old': '72f50dd2c8d56cd0df422a37266397c22c1326d672d5d91ab8261701eda18d2d', 'new': 'adb3bc85ade188f2407d3525cadb86138ca1162b1bc3e96ef4dbcfcfbfb84dda'}, 'train/native_geometry_classification.py': {'old': None, 'new': '954f061bb68fd116e37b4d40bac24f1013c40b5345332c52e7b1b1571a8d1927'}, 'train/native_operation_report.py': {'old': '8993b26bd89941f07d598668bd02a8a1fb1066f7a5c68a82faa074e504339664', 'new': 'c0c0f99f54dd8e0ef59e766f1f4e6f239a03024ef800e2217056ceed1a25e7b2'}, 'train/native_state_sft.py': {'old': '3eb9a2c4304577778591b478d9287685e159875925a7eb029a8213b3d741b973', 'new': 'eff5414b6b7f4d6e08801ed0008c14b10f609f0c106290251eb3f0ea04d4c16d'}}}


def inventory_equivalence(old, new):
    pair = (resident.digest(old), resident.digest(new))
    expected = APPROVED.get(pair)
    changes = {name: {'old': old.get(name), 'new': value}
               for name, value in new.items() if old.get(name) != value}
    if expected is None or changes != expected or set(old) - set(new):
        raise ValueError('gradient reuse requires an exact reviewed metadata-only inventory transition')
    return {'qualified_runtime_code': pair[0], 'executing_runtime_code': pair[1],
            'changes': changes, 'model_optimizer_objective_gradient_modules_unchanged': True}


def _proof_contents(reference):
    value = json.loads(classification._checked(reference).read_text())
    # Only a path relocation of the same immutable producer is permitted.
    for role in ('indexes', 'held_indexes'):
        for part in value.get(role, []):
            classification._checked(part['producer'])
            part['producer'] = {k: v for k, v in part['producer'].items() if k != 'path'}
    return value


def normalized_arguments(arguments, parser, spec):
    value = vars(parser.parse_args(arguments)).copy()
    for key in ('out', 'resident_data_directory'):
        value.pop(key, None)
    recipe = value.get('recipe_binding')
    if recipe is not None:
        value['recipe_binding'] = {'lineage_sha256': json.loads(recipe)['lineage_sha256']}
    proof = value.pop('source_classification_migration', None)
    if proof is not None:
        value['classification_contents'] = _proof_contents(classification._binding(proof))
    for key in ('checkpoint_input_offload_directory', 'checkpoint_publication_readonly_input_roots'):
        if value.get(key) is not None:
            value[key] = str(value[key]).replace(str(Path(spec['out']).parent), '<owner>').replace(spec['runtime'], '<runtime>')
    return json.loads(json.dumps(value, default=str, sort_keys=True))


def carry_qualification(reference, spec, *, package, inventory, roles, actor_identity, parser):
    """Verify a real prior PASS; preserve its original report/runtime identity."""
    request = json.loads(classification._checked(reference).read_text())
    required = {'schema', 'gate_spec', 'performance', 'inventory', 'precision', 'ffn',
                'actor', 'pre_capacity', 'failure'}
    if set(request) != required or request.get('schema') != 'natlang.gradient-metadata-equivalence-request/1':
        raise ValueError('gradient reuse request is incomplete')
    docs = {name: json.loads(classification._checked(ref).read_text())
            for name, ref in request.items() if name != 'schema'}
    prior = docs['gate_spec']
    old_package = Path(prior['runtime']) / 'training/neuralese/natlang_neuralese'
    old_inventory, new_inventory = resident.code_inventory(old_package), resident.code_inventory(package)
    equivalent = inventory_equivalence(old_inventory, new_inventory)
    if (prior['runtime_code_sha256'] != equivalent['qualified_runtime_code']
            or spec['runtime_code_sha256'] != equivalent['executing_runtime_code']
            or prior['parent'] != spec['parent']
            or docs['performance'].get('passed') is not True
            or docs['performance'].get('parent') != spec['parent']
            or docs['performance'].get('runtime_code_sha256') != prior['runtime_code_sha256']
            or docs['performance'].get('optimizer_updates_from_measurement') != 0
            or docs['actor'].get('passed') is not True
            or docs['actor'].get('parent') != spec['parent']
            or docs['actor'].get('candidate_identity') != actor_identity
            or docs['actor'].get('optimizer_updates') != 0
            or docs['pre_capacity'].get('passed') is not True
            or docs['pre_capacity'].get('parent') != spec['parent']
            or docs['pre_capacity'].get('new_optimizer_updates') != 0
            or docs['ffn'].get('passed') is not True
            or docs['failure'].get('error') != 'geometry evidence source prefix/count/metadata differs'):
        raise ValueError('gradient reuse lost its exact measured parent/execution/pre-update PASS')
    observed = docs['inventory']
    if (observed.get('parent') != spec['parent']
            or observed.get('runtime_code_sha256') != prior['runtime_code_sha256']
            or observed.get('inventory') != inventory
            or observed.get('roles') != roles
            or observed.get('inventory_sha256') != resident.digest(inventory)
            or observed.get('actual_count') != len(inventory)
            or set(observed.get('roles', {})) != {'backbone', 'adapter', 'learned_input', 'learned_output'}
            or any(not values for values in observed['roles'].values())):
        raise ValueError('gradient reuse changed the actual complete trainable cohort')
    if normalized_arguments(prior['trainer_arguments'], parser, prior) != normalized_arguments(spec['trainer_arguments'], parser, spec):
        raise ValueError('gradient reuse changed the numerical trainer invocation')
    precision = docs['precision']; runtime = precision.get('runtime_reference') or {}
    complete = runtime.get('complete_gradient_qualification') or {}
    from ..eval.native_gradients import validate_complete_proof
    args = parser.parse_args(spec['trainer_arguments'])
    validate_complete_proof(complete, runtime.get('native_identity'), gradient_atol=args.gradient_atol)
    if (precision.get('parent_weights_sha256') != spec['parent']['sha256']
            or runtime.get('checkpoint_sha256') != spec['parent']['sha256']
            or runtime.get('passed') is not True
            or runtime.get('gradient_scope') != 'complete-production-trainables'
            or complete.get('inventory') != inventory
            or complete.get('derivative_comparison_count') != len(inventory) + 1
            or runtime.get('forward_atol', float('inf')) > args.forward_atol):
        raise ValueError('gradient reuse lacks the original complete numerical proof')
    return {'schema': 'natlang.carried-gradient-qualification/1', 'passed': True,
            'request': dict(reference), 'equivalence': equivalent, 'parent': spec['parent'],
            'precision_transition': precision, 'ffn_transition': docs['ffn'],
            'numerical_measurement_performed': False, 'optimizer_updates': 0,
            'scope': 'Original measured runtime retained; only reviewed metadata execution changed'}
