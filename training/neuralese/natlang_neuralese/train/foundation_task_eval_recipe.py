"""Prepare an immutable foundation recipe with its registered boundary task check.

The natural-language recipe selects the held packet and the boundary at which
it runs. This module verifies those exact declarations and writes a successor
recipe plus a byte-bound preparation receipt; it does not train, serve, or
change a live controller.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from ..common.hashing import sha256_file_hex
from ..common.jsonio import write_json_durable
from ..eval.task_cadence import parse_task_eval


def _read_bound(path: str | Path, expected_sha256: str) -> tuple[Path, bytes, dict]:
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f'recipe input changed: {path}')
    value = json.loads(raw)
    if not isinstance(value, dict) or value.get('schema') != 'natlang.neuralese-training-recipe/1':
        raise ValueError(f'unsupported recipe input: {path}')
    if 'extends' in value or 'overrides' in value:
        raise ValueError('foundation recipe preparation requires concrete, already-resolved recipe files')
    return path, raw, value


def prepare_foundation_boundary_recipe(*, parent_recipe: str | Path, parent_sha256: str,
                                       task_eval_recipe: str | Path, task_eval_recipe_sha256: str,
                                       stage_id: str, task_eval_stage_id: str,
                                       output_recipe: str | Path, proof_path: str | Path) -> dict:
    """Copy a concrete foundation recipe, adding only task_eval and zero cadence."""
    parent_path, parent_raw, parent = _read_bound(parent_recipe, parent_sha256)
    task_path, task_raw, source = _read_bound(task_eval_recipe, task_eval_recipe_sha256)
    stages = [stage for stage in parent.get('stages', []) if stage.get('id') == stage_id]
    source_stages = [stage for stage in source.get('stages', []) if stage.get('id') == task_eval_stage_id]
    if len(stages) != 1 or stages[0].get('kind') != 'native_state_sft':
        raise ValueError('target recipe must contain exactly one declared native_state_sft stage')
    if stages[0].get('parameters', {}).get('communication_foundation') is not True:
        raise ValueError('target stage is not the declared communication-foundation SFT stage')
    if len(source_stages) != 1:
        raise ValueError('task-evaluation source must contain exactly one named stage')
    if source_stages[0].get('kind') != 'native_state_sft':
        raise ValueError('registered task-evaluation source must be a native_state_sft recipe')
    spec = source_stages[0].get('parameters', {}).get('task_eval')
    packet = spec.get('packet') if isinstance(spec, dict) else None
    if (not isinstance(packet, dict) or not isinstance(packet.get('path'), str)
            or not isinstance(packet.get('sha256'), str) or len(packet['sha256']) != 64):
        raise ValueError('registered task-evaluation recipe lacks an exact held packet binding')
    packet_path = Path(packet['path'])
    if not packet_path.is_absolute():
        from ..common.paths import root as logical_root
        packet_path = logical_root('repo') / packet_path
    packet_path = packet_path.resolve(strict=True)
    if sha256_file_hex(packet_path) != packet['sha256']:
        raise ValueError('registered held task packet SHA-256 does not match its recipe')
    # Validate the complete task-evaluation shape using the trainer's parser.
    # A zero cadence means boundary-only; it is intentionally not passed here
    # as a task-evaluation option.
    parsed_spec = parse_task_eval(copy.deepcopy(spec))
    if parsed_spec is None or parsed_spec['packet'] != packet:
        raise ValueError('registered task-evaluation spec failed trainer validation')
    if parsed_spec.get('neuralese_server') is not True:
        raise ValueError('native foundation task evaluation must use native-state transport')
    from ..eval.task_packet import read_jsonl
    packet_rows = read_jsonl(packet_path)
    if max(parsed_spec['cases']) >= len(packet_rows) or any(
            packet_rows[index].get('split') != 'test' for index in parsed_spec['cases']):
        raise ValueError('registered task-evaluation cases must all be held test rows')

    target = stages[0]
    parameters = target.get('parameters')
    if not isinstance(parameters, dict):
        raise ValueError('target foundation stage has no parameter object')
    if parameters.get('task_eval') not in (None, '', 'null'):
        raise ValueError('target foundation stage already has a task-evaluation policy')
    if type(parameters.get('task_eval_every')) is not int or parameters['task_eval_every'] != 0:
        raise ValueError('target foundation stage must already declare boundary-only task cadence 0')
    successor = copy.deepcopy(parent)
    successor_stage = next(stage for stage in successor['stages'] if stage.get('id') == stage_id)
    successor_stage['parameters']['task_eval'] = copy.deepcopy(spec)
    successor_stage['parameters']['task_eval_every'] = 0

    output_recipe = Path(output_recipe).resolve()
    proof_path = Path(proof_path).resolve()
    if output_recipe == proof_path:
        raise ValueError('recipe and proof outputs must have distinct paths')
    if output_recipe.exists() or proof_path.exists():
        raise FileExistsError('recipe successor and proof paths must be new immutable outputs')
    output_recipe.parent.mkdir(parents=True, exist_ok=True)
    proof_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_durable(output_recipe, successor)
    written = json.loads(output_recipe.read_bytes())
    if written != successor:
        raise IOError('written foundation recipe failed JSON round-trip verification')
    output_sha = sha256_file_hex(output_recipe)
    proof = {
        'schema': 'natlang.foundation-boundary-task-eval-recipe-preparation/1',
        'parent_recipe': {'path': str(parent_path), 'sha256': hashlib.sha256(parent_raw).hexdigest(),
                          'bytes': len(parent_raw)},
        'task_eval_source_recipe': {'path': str(task_path), 'sha256': hashlib.sha256(task_raw).hexdigest(),
                                    'bytes': len(task_raw)},
        'task_eval_stage_id': task_eval_stage_id,
        'held_packet': {'path': str(packet_path), 'sha256': packet['sha256'],
                        'bytes': packet_path.stat().st_size},
        'target_stage_id': stage_id,
        'changed_parameters': ['task_eval', 'task_eval_every'],
        'task_eval_every': 0,
        'success_threshold_is_not_foundation_qualification': True,
        'output_recipe': {'path': str(output_recipe), 'sha256': output_sha,
                          'bytes': output_recipe.stat().st_size},
    }
    write_json_durable(proof_path, proof)
    if json.loads(proof_path.read_bytes()) != proof:
        raise IOError('written preparation proof failed JSON round-trip verification')
    return proof


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-recipe', type=Path, required=True)
    parser.add_argument('--parent-sha256', required=True)
    parser.add_argument('--task-eval-recipe', type=Path, required=True)
    parser.add_argument('--task-eval-recipe-sha256', required=True)
    parser.add_argument('--stage-id', required=True)
    parser.add_argument('--task-eval-stage-id', required=True)
    parser.add_argument('--output-recipe', type=Path, required=True)
    parser.add_argument('--proof', type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(prepare_foundation_boundary_recipe(
        parent_recipe=args.parent_recipe, parent_sha256=args.parent_sha256,
        task_eval_recipe=args.task_eval_recipe, task_eval_recipe_sha256=args.task_eval_recipe_sha256,
        stage_id=args.stage_id, task_eval_stage_id=args.task_eval_stage_id,
        output_recipe=args.output_recipe, proof_path=args.proof), sort_keys=True))


if __name__ == '__main__':
    main()
