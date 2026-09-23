#!/usr/bin/env python3
"""Create an owned Docker service plan for sequential student/Bonsai serving."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_managed_improvement import recipe_digest
from run_training_pipeline import atomic_json, digest_file


def service_port(endpoint):
    parsed = urlparse(endpoint)
    if parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost') or not parsed.port or parsed.path not in ('', '/'):
        raise ValueError('managed Docker serving requires a local HTTP endpoint with no path')
    return parsed.port


def model_swap_config(recipe, run, student_gguf):
    recipe, run = Path(recipe).resolve(), Path(run).resolve()
    config = json.loads(recipe.read_text())
    if config.get('run_directory') != str(run) or 'collection' not in config:
        raise ValueError('expected a student-improvement recipe bound to this run')
    if config['collection'].get('teacher_model') != 'Ternary-Bonsai-2-27B':
        raise ValueError('the default managed teacher launcher only serves Ternary-Bonsai-2-27B; use a custom service plan')
    repo = Path(config['repository']).resolve()
    if Path(student_gguf).name != student_gguf or not student_gguf.endswith('.gguf'):
        raise ValueError('student GGUF must be a filename under repository models/')
    if not (repo / 'models' / student_gguf).is_file():
        raise ValueError('student GGUF is missing from repository models/')
    tag = hashlib.sha256(str(run).encode()).hexdigest()[:12]
    student_endpoint = config['collection']['student_server']
    teacher_endpoint = config['collection']['teacher_server']
    student_port, teacher_port = map(service_port, (student_endpoint, teacher_endpoint))
    student_container, teacher_container = f'natlang-student-{tag}', f'natlang-teacher-{tag}'
    base_run = Path(config['continuation']['base_run'])
    adapter = next((path for path in (
        base_run / 'train-correction/checkpoint/weights/adapter_model.safetensors',
        base_run / 'train-teacher/checkpoint/weights/adapter_model.safetensors') if path.is_file()), None)
    if adapter is None:
        raise ValueError('completed base adapter is missing')
    return {'version': 'natlang.model_swap/1', 'pipeline_sha256': recipe_digest(config),
            'run_directory': str(run),
            'student': {'endpoint': student_endpoint, 'ready_url': student_endpoint.rstrip('/') + '/health',
                        'start': ['bash', str(repo / 'scripts/serve.sh'), student_gguf, str(student_port)],
                        'stop': ['docker', 'stop', student_container],
                        'artifacts': {str(repo / 'models' / student_gguf): digest_file(repo / 'models' / student_gguf),
                                      str(adapter): digest_file(adapter)},
                        'env': {'NATLANG_SERVER_NAME': student_container}, 'startup_timeout_seconds': 600},
            'teacher': {'endpoint': teacher_endpoint, 'ready_url': teacher_endpoint.rstrip('/') + '/health',
                        'start': ['bash', str(repo / 'scripts/serve_bonsai.sh'), str(teacher_port)],
                        'stop': ['docker', 'stop', teacher_container],
                        'env': {'BONSAI_SERVER_NAME': teacher_container}, 'startup_timeout_seconds': 600}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', required=True, type=Path)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--student-gguf', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    config = model_swap_config(args.recipe, args.run, args.student_gguf)
    if args.output.exists():
        if json.loads(args.output.read_text()) != config:
            raise ValueError('refusing to replace a different model-swap config')
    else:
        atomic_json(args.output, config)
    print(args.output)


if __name__ == '__main__':
    main()
