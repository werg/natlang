#!/usr/bin/env python3
"""Create a resumable, single-phase SFT continuation from verified teacher corrections.

The initial production recipe is immutable. Each improvement round gets a new
recipe and run directory, initialized from the previous completed adapter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_training_pipeline import atomic_json


def validate_correction_turns(teacher_turns):
    approved = 0
    with Path(teacher_turns).open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get('version') != 'natlang.teacher_training_turn.native/1':
                raise ValueError(f'correction turn {line_number} has the wrong version')
            if (not isinstance(row.get('id'), str) or not row['id']
                    or not isinstance(row.get('messages'), list) or not row['messages']
                    or not isinstance(row.get('target'), dict)
                    or not isinstance(row.get('program_id'), str) or not row['program_id']
                    or not isinstance(row.get('source_groups'), list) or not row['source_groups']
                    or not isinstance(row.get('source_ref', {}).get('source_row_sha256'), str)):
                raise ValueError(f'correction turn {line_number} lacks native decision evidence')
            if row.get('outcome', {}).get('accepted') is not True:
                raise ValueError(f'correction turn {line_number} lacks an accepted final oracle')
            admission = row.get('training_admission', {})
            if admission.get('kind') != 'exact-native-runtime-oracle' or not isinstance(admission.get('approved'), bool):
                raise ValueError(f'correction turn {line_number} lacks native training admission')
            if row.get('trace_admission', {}).get('admitted') is not True:
                raise ValueError(f'correction turn {line_number} lacks admitted trace evidence')
            approved += admission['approved']
    if not approved:
        raise ValueError('correction turns contain no approved decisions')
    return approved


def improvement_recipe(base_recipe, base_run, teacher_turns, *, deferred_turns=False):
    base_recipe = Path(base_recipe).resolve()
    base_run = Path(base_run).resolve()
    teacher_turns = Path(teacher_turns).resolve()
    source = json.loads(base_recipe.read_text())
    if source.get('version') != 'natlang.training_pipeline/1':
        raise ValueError('base recipe must be a natlang training pipeline')
    stages = {stage['id']: stage for stage in source['stages']}
    prior_phase = 'correction' if 'train-correction' in stages else 'teacher'
    needed = tuple(f'{kind}-{prior_phase}' for kind in ('render', 'audit', 'train'))
    if any(name not in stages for name in needed):
        raise ValueError('base recipe lacks a completed distillation training phase')
    state_path = base_run / 'pipeline-state.json'
    state = json.loads(state_path.read_text())
    identity = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
    if state.get('config_sha256') != identity:
        raise ValueError('base run does not match the base recipe')
    if state.get('stages', {}).get(f'train-{prior_phase}', {}).get('status') != 'complete':
        raise ValueError('base distillation training stage must be complete')
    checkpoint = base_run / f'train-{prior_phase}/checkpoint'
    checkpoint_state = json.loads((checkpoint / 'state.json').read_text())
    if checkpoint_state.get('trained_examples', 0) < checkpoint_state.get('corpus', {}).get('target_examples', 1):
        raise ValueError('base teacher checkpoint is partial')
    adapter = checkpoint / 'weights/adapter_model.safetensors'
    if not adapter.is_file():
        raise ValueError('base teacher adapter is missing')
    registry = (Path(source['continuation']['split_registry']) if prior_phase == 'correction'
                else base_run / 'prepared/splits.json')
    if not registry.is_file():
        raise ValueError('base split registry is missing')
    if not deferred_turns and not teacher_turns.is_file():
        raise ValueError('verified correction turns file is missing')
    if not deferred_turns:
        validate_correction_turns(teacher_turns)
    repo = Path(source['repository']).resolve()
    # Docker recipes only mount the repository and the new run directory.
    if stages[f'train-{prior_phase}']['command'][0] == 'docker':
        for path in (base_run, teacher_turns, registry):
            if not path.is_relative_to(repo):
                raise ValueError('Docker continuation inputs must be inside the repository')
    def substitute(stage, replacements):
        return {key: ([replace(value, replacements) for value in stage[key]]
                      if key in ('command', 'inputs', 'outputs') else stage[key])
                for key in stage}
    def replace(value, replacements):
        for before, after in replacements:
            value = value.replace(before, after)
        return value
    prior = str(base_run)
    prepared = '${run}/prepared-correction'
    python_command = stages[f'render-{prior_phase}']['command']
    # Preserve the base recipe's Python/image invocation and model/tokenizer options.
    prepare_prefix = python_command[:python_command.index('${repo}/scripts/render_training_corpus.py')]
    prepare = {'id': 'prepare-correction',
               'command': [*prepare_prefix, '${repo}/scripts/prepare_training_stages.py',
                           '--output', prepared, '--teacher', str(teacher_turns),
                           '--registry', str(registry)],
               'inputs': ['${repo}/scripts/prepare_training_stages.py', str(teacher_turns), str(registry)],
               'outputs': [f'{prepared}/manifest.json', f'{prepared}/teacher.jsonl']}
    validate = {'id': 'validate-corrections',
                'command': [*prepare_prefix, '${repo}/scripts/validate_correction_turns.py',
                            '--input', str(teacher_turns), '--output', '${run}/corrections.validation.json'],
                'inputs': ['${repo}/scripts/validate_correction_turns.py',
                           '${repo}/scripts/create_improvement_round.py', str(teacher_turns)],
                'outputs': ['${run}/corrections.validation.json']}
    prepare['inputs'].append('${run}/corrections.validation.json')
    render = substitute(stages[f'render-{prior_phase}'], [
        ('${run}/prepared-teacher/teacher.jsonl', f'{prepared}/teacher.jsonl'),
        ('${run}/teacher.sft.jsonl', '${run}/correction.sft.jsonl')])
    render['id'] = 'render-correction'
    audit = substitute(stages[f'audit-{prior_phase}'], [
        ('${run}/teacher.sft.jsonl', '${run}/correction.sft.jsonl'),
        ('${run}/teacher.ready.jsonl', '${run}/correction.ready.jsonl')])
    audit['id'] = 'audit-correction'
    train = substitute(stages[f'train-{prior_phase}'], [
        ('${run}/train-coding/checkpoint/weights', str(checkpoint / 'weights')),
        ('${run}/train-coding/checkpoint/state.json', str(checkpoint / 'state.json')),
        ('${run}/teacher.ready.jsonl', '${run}/correction.ready.jsonl'),
        ('${run}/train-teacher/', '${run}/train-correction/'),
        ('${run}/teacher.tokens.sqlite', '${run}/correction.tokens.sqlite'),
        ('${run}/training-readiness.json', str(base_run / 'training-readiness.json'))])
    train['id'] = 'train-correction'
    train['training_state'] = '${run}/train-correction/checkpoint/state.json'
    prior_init = train['command'][train['command'].index('--init-adapter') + 1]
    train['command'][train['command'].index('--init-adapter') + 1] = str(checkpoint / 'weights')
    train['inputs'] = [item for item in train['inputs'] if item not in
                       (prior_init + '/adapter_model.safetensors',
                        str(source.get('continuation', {}).get('base_recipe', '')),
                        str(source.get('continuation', {}).get('base_run', '')) + '/pipeline-state.json')]
    train['inputs'].extend([str(checkpoint / 'state.json'), str(adapter), str(base_recipe), str(state_path)])
    train['inputs'] = list(dict.fromkeys(train['inputs']))
    return {'version': 'natlang.training_pipeline/1', 'repository': str(repo),
            'continuation': {'base_recipe': str(base_recipe), 'base_run': prior,
                             'teacher_turns': str(teacher_turns), 'split_registry': str(registry)},
            'stages': [validate, prepare, render, audit, train]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-recipe', required=True, type=Path)
    parser.add_argument('--base-run', required=True, type=Path)
    parser.add_argument('--teacher-turns', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    config = improvement_recipe(args.base_recipe, args.base_run, args.teacher_turns)
    if args.output.exists():
        if json.loads(args.output.read_text()) != config:
            raise ValueError('refusing to replace a different improvement recipe')
    else:
        atomic_json(args.output, config)
    print(args.output)


if __name__ == '__main__':
    main()
