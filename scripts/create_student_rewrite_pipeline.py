#!/usr/bin/env python3
"""Prepare a rewrite -> ordinary rendering -> token audit pipeline; never launch it."""
import argparse
import json
from pathlib import Path

try:
    from .run_training_pipeline import digest_file, validate_stage_paths
except ImportError:
    from run_training_pipeline import digest_file, validate_stage_paths


def build(plan_path, *, model, revision, max_len, python):
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text())
    if plan.get('schema') not in ('natlang.student_rewrite_plan/1', 'natlang.student_projection_plan/1', 'natlang.student_chunk_rewrite_plan/1'):
        raise ValueError('unsupported student collection plan')
    projection = plan['schema'] == 'natlang.student_projection_plan/1'
    chunk = plan['schema'] == 'natlang.student_chunk_rewrite_plan/1'
    if len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
        raise ValueError('tokenizer revision must be an immutable commit SHA')
    if max_len < 2:
        raise ValueError('training context must allow a prompt and completion')
    repo = Path(__file__).resolve().parents[1]
    output = str(Path(plan['output']).resolve())
    turns = output + '/turns.jsonl'
    collector = str(repo / ('ts-host/scripts/rewrite-student-chunks.mjs' if chunk else 'ts-host/scripts/project-student-trajectories.mjs' if projection else 'ts-host/scripts/rewrite-student-trajectories.mjs'))
    inputs = sorted(set([str(plan_path), collector, *plan['pins'],
                         *(item['path'] for item in plan['teacher_artifacts']),
                         *plan.get('student', {}).get('weight_pins', {})]))
    stages = [
        {'id': 'student-rewrite', 'command': ['node', collector, str(plan_path),
            *(['--execute', digest_file(plan_path)] if chunk else ['--execute', '--sha256', digest_file(plan_path)])], 'inputs': inputs,
         'indirect_inputs': inputs[0:],
         'outputs': [output, turns, output + '/summary.json', output + '/admitted.jsonl'],
         'output_directories': [output]},
        {'id': 'render-rewrites', 'command': [python, str(repo / 'scripts/render_training_corpus.py'),
            '--inputs', turns, '--output', '${run}/rendered.jsonl', '--model', model,
            '--revision', revision, '--streaming'], 'inputs': [turns],
         'outputs': ['${run}/rendered.jsonl', '${run}/rendered.jsonl.manifest.json']},
        {'id': 'audit-rewrites', 'command': [python, str(repo / 'scripts/audit_training_corpus.py'),
            '--input', '${run}/rendered.jsonl', '--output', '${run}/ready.jsonl',
            '--model', model, '--revision', revision, '--max-len', str(max_len), '--streaming'],
         'inputs': ['${run}/rendered.jsonl', '${run}/rendered.jsonl.manifest.json'],
         'outputs': ['${run}/ready.jsonl', '${run}/ready.jsonl.manifest.json', '${run}/ready.jsonl.audit.json']},
    ]
    if projection:
        for stage in stages:
            stage['id'] = stage['id'].replace('rewrite', 'projection')
    return {'version': 'natlang.training_pipeline/1', 'repo': str(repo), 'stages': stages,
            'method': ('verified-threshold-chunk-rewrite/1' if chunk else 'request-boundary-proposal-corrected-mh/1' if projection else 'verified-student-rewrite'), ('projection_plan_sha256' if projection else 'rewrite_plan_sha256'): digest_file(plan_path),
            'root_approved': plan.get('root_approved') is True,
            'requires_existing_pinned_student_server': True,
            'automatic_training_publication': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--model', required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--max-len', type=int, default=16384)
    parser.add_argument('--python', default='python3')
    args = parser.parse_args()
    config = build(args.plan, model=args.model, revision=args.revision, max_len=args.max_len, python=args.python)
    # The generic runner consumes this exact graph without changing the active SFT recipe.
    validate_stage_paths(config['stages'], args.output.resolve().parent / 'pipeline-run', Path(config['repo']))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(config, stream, indent=2, sort_keys=True)
        stream.write('\n')


if __name__ == '__main__':
    main()
