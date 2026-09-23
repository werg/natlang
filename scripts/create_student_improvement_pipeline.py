#!/usr/bin/env python3
"""Create one durable student-rollout -> teacher-correction -> training round."""
import argparse
import json
from pathlib import Path

from create_improvement_round import improvement_recipe
from run_training_pipeline import atomic_json


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
    def add(name, command, inputs, outputs):
        return {'id': name, 'command': command, 'inputs': inputs, 'outputs': outputs}
    stages = [add('freeze-runtime', ['python', f'{p}/scripts/freeze_training_runtime.py',
                                     f'{p}/ts-host', frozen],
                  [f'{p}/scripts/freeze_training_runtime.py'],
                  [runtime_hash, f'{frozen}/dist', f'{frozen}/scripts', f'{frozen}/src', f'{frozen}/prelude.js'])]
    def collect(role, source, endpoint, model, output, extra=()):
        args = ['node', f'{frozen}/scripts/teacher-collector.mjs', source,
                f'{r}/{role}-jobs', output, '--model-id', model, '--server', endpoint,
                '--root-seed', str(root_seed), '--workers', str(workers),
                '--segment-turns', '24', '--segment-messages', '48',
                '--collection-role', role, *(['--max-turns', str(max_turns)] if max_turns else []), *extra]
        if role == 'student':
            args += ['--all'] if limit == 0 else ['--limit', str(limit)]
        else:
            args += ['--all']
        return args
    student_rows = f'{r}/student-trajectories.jsonl'
    queue = f'{r}/hard-state-queue.jsonl'
    teacher_rows = f'{r}/teacher-corrections.jsonl'
    student_turns = f'{r}/student-success-turns.jsonl'
    teacher_turns = f'{r}/teacher-correction-turns.jsonl'
    combined = f'{r}/verified-turns.jsonl'
    stages.append(add('collect-student', collect('student', str(programs), student_server, student_model,
                                                 student_rows),
                      [str(programs), runtime_hash, f'{frozen}/dist/teacher/collector.js',
                       f'{frozen}/dist/native/runtime.js'],
                      [student_rows, f'{student_rows}.manifest.json']))
    stages.append(add('build-hard-states', ['node', f'{frozen}/scripts/build-hard-state-queue.mjs',
                                            student_rows, queue],
                      [student_rows, runtime_hash, f'{frozen}/scripts/build-hard-state-queue.mjs'],
                      [queue, f'{queue}.programs.jsonl', f'{queue}.manifest.json']))
    stages.append(add('collect-corrections', collect('teacher', f'{queue}.programs.jsonl', teacher_server,
                                                    teacher_model, teacher_rows, ('--handoff-queue', queue)),
                      [queue, f'{queue}.programs.jsonl', runtime_hash,
                       f'{frozen}/dist/teacher/collector.js', f'{frozen}/dist/native/runtime.js'],
                      [teacher_rows, f'{teacher_rows}.manifest.json']))
    for role, source, target in (('student', student_rows, student_turns),
                                 ('teacher', teacher_rows, teacher_turns)):
        stages.append(add(f'materialize-{role}', ['node', f'{frozen}/scripts/materialize-native-teacher.mjs',
                                                   source, target, '--replace'],
                          [source, runtime_hash, f'{frozen}/dist/teacher/native-materializer.js'], [target]))
    preferences = f'{r}/preference-pairs.jsonl'
    stages.append(add('build-preferences', ['node', f'{frozen}/scripts/build-preference-pairs.mjs',
                                            student_rows, teacher_rows, teacher_turns, preferences],
                      [student_rows, teacher_rows, teacher_turns, runtime_hash,
                       f'{frozen}/scripts/build-preference-pairs.mjs'],
                      [preferences, f'{preferences}.manifest.json']))
    stages.append(add('combine-verified', ['python', f'{p}/scripts/combine_verified_turns.py',
                                           '--student', student_turns, '--teacher', teacher_turns,
                                           '--output', combined],
                      [f'{p}/scripts/combine_verified_turns.py', f'{p}/scripts/create_improvement_round.py',
                       student_turns, teacher_turns],
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
    parser.add_argument('--workers', type=int, default=1)
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
