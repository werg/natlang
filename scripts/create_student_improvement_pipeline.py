#!/usr/bin/env python3
"""Create one durable student-rollout -> teacher-correction -> training round."""
import argparse
import json
import sys
from pathlib import Path

from create_improvement_round import improvement_recipe
from run_training_pipeline import atomic_json
from self_improvement_data import current_improvement_turns

# Stages run with the interpreter that created the round: a bare `python` need not exist on the host (only python3).
PYTHON = sys.executable


def improvement_pipeline(base_recipe, base_run, programs, run, student_server, student_model,
                         teacher_server, teacher_model, *, limit=0, root_seed=42, workers=1, max_turns=None):
    run = Path(run).resolve()
    programs = Path(programs).resolve()
    if not programs.is_file():
        raise ValueError('program IR file is missing')
    if not student_server or not student_model or not teacher_server or not teacher_model:
        raise ValueError('both student and teacher model endpoints and IDs are required')
    if not isinstance(limit, int) or limit < 0 or not isinstance(workers, int) or workers < 1:
        raise ValueError('limit must be nonnegative and workers positive')
    base = improvement_recipe(base_recipe, base_run, run / 'verified-turns.jsonl', deferred_turns=True)
    p, r = '${repo}', '${run}'
    frozen = f'{r}/runtime-host'
    runtime_hash = f'{frozen}/frozen-runtime.json'
    def add(name, command, inputs, outputs, *, output_directories=(), indirect_inputs=()):
        stage = {'id': name, 'command': command, 'inputs': inputs, 'outputs': outputs}
        if output_directories:
            stage['output_directories'] = list(output_directories)
        if indirect_inputs:
            stage['indirect_inputs'] = list(indirect_inputs)
        return stage
    stages = [add('freeze-runtime', [PYTHON, f'{p}/scripts/freeze_training_runtime.py',
                                     f'{p}/ts-host', frozen],
                  [f'{p}/scripts/freeze_training_runtime.py'],
                  [runtime_hash, f'{frozen}/dist', f'{frozen}/scripts', f'{frozen}/src', f'{frozen}/prelude.js'],
                  output_directories=[f'{frozen}/dist', f'{frozen}/scripts', f'{frozen}/src'])]
    def collect(role, source, endpoint, model, output, extra=()):
        args = ['node', f'{frozen}/scripts/teacher-collector.mjs', source,
                f'{r}/{role}-jobs', output, '--model-id', model, '--server', endpoint,
                '--root-seed', str(root_seed), '--workers', str(workers),
                '--collection-role', role, '--execution-adapter', f'{frozen}/dist/improvement/teacher.js', *(['--max-turns', str(max_turns)] if max_turns else []), *extra]
        if role == 'student':
            args += ['--all'] if limit == 0 else ['--limit', str(limit)]
        else:
            args += ['--all']
        return args
    student_rows = f'{r}/student-trajectories.jsonl'
    handoffs = f'{r}/handoffs.ir.jsonl'
    teacher_rows = f'{r}/teacher-corrections.jsonl'
    student_turns = f'{r}/student-success-turns.jsonl'
    teacher_turns = f'{r}/teacher-correction-turns.jsonl'
    combined = f'{r}/verified-turns.jsonl'
    repo = Path(base['repository']).resolve()
    native_improvement_turns = current_improvement_turns(repo)
    original_recipe = json.loads(Path(base_recipe).read_text())
    derivations = original_recipe.get('reviewed_turn_identity_derivations', [])
    derived = {str((repo / item['source']).resolve()): item['output'] for item in derivations}
    needed = {path for path in native_improvement_turns if path in derived}
    required_stages = {f"namespace-reviewed-turn-identities-{item['lane']}"
                       for item in derivations if str((repo / item['source']).resolve()) in needed}
    namespace_stages = [stage for stage in original_recipe['stages'] if stage['id'] in required_stages]
    if {stage['id'] for stage in namespace_stages} != required_stages:
        raise ValueError('base recipe lacks reviewed legacy identity derivation stages')
    # Carry the same approval-bound identities into correction rounds. Raw legacy
    # row IDs can collide across separately reviewed runtime API lanes.
    stages.extend(namespace_stages)
    native_improvement_turns = [derived.get(path, path) for path in native_improvement_turns]
    stages.append(add('collect-student', collect('student', str(programs), student_server, student_model,
                                                  student_rows),
                      [str(programs), runtime_hash, f'{frozen}/dist/teacher/collector.js',
                       f'{frozen}/dist/native/runtime.js'],
                      [student_rows, f'{student_rows}.manifest.json'],
                      indirect_inputs=[runtime_hash, f'{frozen}/dist/teacher/collector.js',
                                       f'{frozen}/dist/native/runtime.js']))
    # Each failed student run becomes handoff tasks: the run replayed up to a failure, the rest the teacher's.
    stages.append(add('build-handoffs', ['node', f'{frozen}/scripts/build-handoffs.mjs', handoffs, student_rows],
                      [student_rows, runtime_hash, f'{frozen}/scripts/build-handoffs.mjs'], [handoffs],
                      indirect_inputs=[runtime_hash]))
    stages.append(add('collect-corrections', collect('teacher', handoffs, teacher_server, teacher_model, teacher_rows),
                      [handoffs, runtime_hash, f'{frozen}/dist/teacher/collector.js', f'{frozen}/dist/native/runtime.js'],
                      [teacher_rows, f'{teacher_rows}.manifest.json'],
                      indirect_inputs=[runtime_hash, f'{frozen}/dist/teacher/collector.js',
                                       f'{frozen}/dist/native/runtime.js']))
    for role, source, target in (('student', student_rows, student_turns),
                                 ('teacher', teacher_rows, teacher_turns)):
        stages.append(add(f'materialize-{role}', ['node', f'{frozen}/scripts/materialize-native-teacher.mjs',
                                                   source, target, '--replace'],
                          [source, runtime_hash, f'{frozen}/dist/teacher/native-materializer.js'], [target],
                          indirect_inputs=[runtime_hash, f'{frozen}/dist/teacher/native-materializer.js']))
    preferences = f'{r}/preference-pairs.jsonl'
    stages.append(add('build-preferences', ['node', f'{frozen}/scripts/build-preference-pairs.mjs',
                                            preferences, '--handoffs', teacher_rows],
                      [teacher_rows, runtime_hash, f'{frozen}/scripts/build-preference-pairs.mjs'], [preferences],
                      indirect_inputs=[runtime_hash]))
    stages.append(add('combine-verified', [PYTHON, f'{p}/scripts/combine_verified_turns.py',
                                           '--student', student_turns, '--teacher', teacher_turns,
                                           '--output', combined, *[arg for path in native_improvement_turns for arg in ('--additional-teacher', path)]],
                      [f'{p}/scripts/combine_verified_turns.py', f'{p}/scripts/create_improvement_round.py',
                       student_turns, teacher_turns, *native_improvement_turns],
                      [combined, f'{combined}.manifest.json']))
    base['stages'] = [*stages, *base['stages']]
    base['collection'] = {'programs': str(programs), 'student_server': student_server,
                          'student_model': student_model, 'teacher_server': teacher_server,
                          'teacher_model': teacher_model, 'root_seed': root_seed, 'workers': workers,
                          'limit': limit, 'runtime': runtime_hash}
    base['run_directory'] = str(run)
    return base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base-recipe', 'base-run', 'programs', 'run', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    for name in ('student-server', 'student-model', 'teacher-server', 'teacher-model'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--limit', type=int, default=0, help='student programs; zero means all')
    parser.add_argument('--root-seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--max-turns', type=int, help='model turns allowed per collected call (unbounded if omitted)')
    args = parser.parse_args()
    config = improvement_pipeline(args.base_recipe, args.base_run, args.programs, args.run,
                                  args.student_server, args.student_model, args.teacher_server,
                                  args.teacher_model, limit=args.limit, root_seed=args.root_seed,
                                  workers=args.workers, max_turns=args.max_turns)
    if args.output.exists():
        if json.loads(args.output.read_text()) != config:
            raise ValueError('refusing to replace a different improvement pipeline')
    else:
        atomic_json(args.output, config)
    print(args.output)


if __name__ == '__main__':
    main()
