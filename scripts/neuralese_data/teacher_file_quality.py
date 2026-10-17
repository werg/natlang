"""Exact JSON file-format evaluation continuation; original executions stay intact."""
from __future__ import annotations
import copy
import json
import re
from pathlib import Path
import subprocess

from natlang_neuralese.common.hashing import canonical_json_sha256_hex, sha256_file_hex
from scripts.neuralese_data.ground_teacher_continuations import OUTPUT_CONTRACT

SCHEMA = 'natlang.teacher-json-file-quality-continuation/1'


def file_verdict(row):
    """Use the existing runtime comparator, including exact read-only file checks."""
    root = Path(__file__).resolve().parents[2]
    oracle = root / 'ts-host/dist/evaluation/oracles.js'
    source = root / 'ts-host/src/evaluation/oracles.ts'
    paths = [Path(__file__).resolve(), root / 'scripts/neuralese_data/ground_teacher_continuations.py', source,
             *sorted((root / 'ts-host/dist').rglob('*.js'))]
    bindings = {str(path.relative_to(root)): sha256_file_hex(path) for path in paths}
    semantics = row['task']['program_ir']['semantics']
    payload = {'actual': row['outcome']['files'], 'expected': semantics['expected_files'],
               'input': semantics['folder_files']}
    code = ('import {readFileSync} from "node:fs";'
            'const {checkFiles}=await import(process.argv[1]);'
            'const p=JSON.parse(readFileSync(0,"utf8"));'
            'console.log(JSON.stringify(checkFiles(p.actual,p.expected,p.input,'
            '{compare:"json-string-record",threshold:1})));')
    completed = subprocess.run(['node', '--input-type=module', '-e', code, oracle.as_uri()],
        input=json.dumps(payload), capture_output=True, text=True, check=True)
    if bindings != {str(path.relative_to(root)): sha256_file_hex(path) for path in paths}:
        raise ValueError('file quality verifier changed during execution')
    return json.loads(completed.stdout), {'files': len(bindings),
        'inventory_sha256': canonical_json_sha256_hex(bindings)}


def json_file_program(program):
    """Correct only the evaluator metadata for an explicit JSON string-map task."""
    semantics = program['semantics']
    task = json.loads(semantics['folder_files']['task.json'])
    expected = semantics['expected']
    if (program.get('split') != 'train' or not program.get('source_groups')
        or program.get('family') != 'curriculum_port_record_calls'
        or task.get('output_contract') != OUTPUT_CONTRACT
        or semantics.get('files_oracle', {}).get('compare', 'exact') != 'exact'
        or not isinstance(expected, dict) or not expected
        or any(not isinstance(value, str) for value in expected.values())
        or len(task.get('items', [])) != len(expected)
        or {Path(path).stem for path in task.get('items', [])} != set(expected)
        or json.loads(semantics['expected_files'][task['output_file']]) != expected):
        raise ValueError('quality continuation requires the declared JSON string-map contract')
    result = copy.deepcopy(program)
    result['semantics']['files_oracle'] = {'compare': 'json-string-record', 'threshold': 1}
    return result


def continue_quality(original, original_binding):
    if 'source_quality_continuation' in original:
        raise ValueError('quality continuation requires the original captured execution')
    program = original['task']['program_ir']; semantics = program['semantics']; outcome = original['outcome']
    json_file_program(program)
    expected = semantics['expected']
    checks = outcome.get('checks', {})
    if (program.get('split') != 'train' or not program.get('source_groups')
        or program.get('family') != 'curriculum_port_record_calls'
        or semantics.get('files_oracle', {}).get('compare', 'exact') != 'exact'
        or not isinstance(expected, dict) or not expected
        or any(not isinstance(value, str) for value in expected.values())
        or canonical_json_sha256_hex(outcome.get('value')) != canonical_json_sha256_hex(expected)
        or outcome.get('status') != 'done' or outcome.get('accepted') is not False
        or outcome.get('oracle', {}).get('accepted') is not True
        or outcome.get('quality_pending') != []
        or outcome.get('rejection_reasons') != ['files']
        or checks.get('files') is not False
        or any(value is not True for key, value in checks.items() if key != 'files')
        or not checks or outcome.get('files_check', {}).get('accepted') is not False):
        raise ValueError('quality continuation requires only a declared JSON file-format failure')
    verdict, verifier = file_verdict(original)
    if verdict.get('accepted') is not True or verdict.get('failed') or verdict.get('errors') or verdict.get('pending'):
        raise ValueError('full JSON file contract verification failed')
    result = copy.deepcopy(original)
    result['outcome'].update(accepted=True, files_check=verdict, rejection_reasons=[])
    result['outcome']['checks']['files'] = True
    result['source_quality_continuation'] = {'schema': SCHEMA,
        'original': original_binding, 'original_row_sha256': canonical_json_sha256_hex(original),
        'verifier': verifier, 'comparator': {'compare': 'json-string-record', 'threshold': 1},
        'claim': 'evaluation-only JSON formatting correction; original expected answer and execution unchanged'}
    return result


def verify_quality_continuation(row, original, original_binding):
    expected = continue_quality(original, original_binding)
    # Historical runtime identity records custody, never admission authority.
    # A current canonical comparator must independently reproduce the outcome.
    historical = row.get('source_quality_continuation', {}).get('verifier')
    if (not isinstance(historical, dict) or set(historical) != {'files', 'inventory_sha256'}
        or type(historical.get('files')) is not int or historical['files'] <= 0
        or not isinstance(historical.get('inventory_sha256'), str)
        or not re.fullmatch(r'[0-9a-f]{64}', historical['inventory_sha256'])):
        raise ValueError('teacher quality continuation has invalid historical verifier provenance')
    expected['source_quality_continuation']['verifier'] = historical
    if canonical_json_sha256_hex(row) != canonical_json_sha256_hex(expected):
        raise ValueError('teacher quality continuation differs from exact original evaluation view')
