"""Hash-bound, explicit outcome projection for independently judged Luna files.

The raw collector result is never edited. A derived source row carries a
verifiable projection whose sole authority is an independent file-oracle
judgment over the exact emitted files and original source fixture.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

from natlang_neuralese.common.hashing import canonical_json_sha256_hex, sha256_file_hex

SCHEMA = 'natlang.provider-judged-outcome-regrade/1'


def canonical_sha(value) -> str:
    return canonical_json_sha256_hex(value)


def _read_jsonl(path: Path, row_id: str) -> dict:
    found = []
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get('id') == row_id:
                found.append(row)
    if len(found) != 1:
        raise ValueError('regrade source world row is missing or ambiguous')
    return found[0]


def _node(runtime: Path, actual: dict, expected: dict, inputs: dict, spec: dict, judgment: dict) -> dict:
    node = os.environ.get('NATLANG_NODE') or shutil.which(
        'node', path=str(Path.home() / '.local/bin') + os.pathsep + os.environ.get('PATH', ''))
    if not node:
        raise ValueError('Node is required to revalidate the frozen runtime file oracle')
    module = (runtime / 'dist/teacher/oracle.js').resolve(strict=True)
    script = ('import fs from "node:fs";'
        'const {checkFiles}=await import(process.argv[1]);'
        'const p=JSON.parse(fs.readFileSync(0,"utf8"));'
        'process.stdout.write(JSON.stringify(checkFiles(p.actual,p.expected,p.inputs,p.spec,p.judgments)));')
    raw = json.dumps({'actual': actual, 'expected': expected, 'inputs': inputs, 'spec': spec,
        'judgments': {'reports/HDFS-window-05.json': judgment}}, ensure_ascii=False)
    result = subprocess.run([node, '--input-type=module', '-e', script, module.as_uri()],
        input=raw, text=True, capture_output=True, check=True, timeout=90)
    verdict = json.loads(result.stdout)
    return verdict


def project(raw_row: dict, program_row: dict, source_worlds: Path, qualification_path: Path,
            *, plan_path: Path, plan_sha256: str, claim_id: str, claim_index: int, queue_path: Path,
            result_path: Path, result_manifest_path: Path, runtime: Path) -> tuple[dict, dict]:
    """Verify one source-rich Sol file regrade and build its separate derived row."""
    qualification_path = qualification_path.resolve(strict=True)
    qualification = json.loads(qualification_path.read_text())
    proof_path = qualification_path.parent / 'file-regrade.sol.json'
    if qualification.get('schema') != 'natlang.saved-trajectory-files-qualification/1':
        raise ValueError('unsupported saved-trajectory file qualification')
    proof_bytes = proof_path.read_bytes()
    if sha256_file_hex(proof_path) != qualification.get('regrade_sha256'):
        raise ValueError('Sol regrade artifact differs from the qualification receipt')
    proof = json.loads(proof_bytes)
    raw_bytes = result_path.read_bytes()
    raw_sha = sha256_file_hex(result_path)
    if (qualification.get('result_path') != str(result_path.resolve())
            or qualification.get('result_sha256') != raw_sha
            or proof.get('original_result_path') != str(result_path.resolve())
            or proof.get('original_result_sha256') != raw_sha
            or raw_row.get('id') != qualification.get('case_id')
            or raw_row.get('id') != proof.get('case_id')
            or proof.get('source_case_id') != qualification.get('source_world_id')
            or raw_row.get('provenance', {}).get('model') != qualification.get('actor')
            or raw_row.get('provenance', {}).get('model') != 'gpt-6-luna'
            or raw_row.get('provenance', {}).get('provider') != 'openai-codex'
            or raw_row.get('provenance', {}).get('transport') != 'pi-provider'):
        raise ValueError('regrade does not bind the exact Luna provider result and actor')
    file_grade = (qualification.get('combined_verdict', {}).get('judgments') or {}).get(
        'reports/HDFS-window-05.json')
    if (proof.get('judge') != {'model_id': 'gpt-6-sol', 'provider': 'openai-codex',
                              'pi_options': {'reasoningEffort': 'low'}}
            or proof.get('grade', {}).get('accepted') is not True
            or proof.get('grade', {}).get('needs_review') is True
            or qualification.get('judge') != proof.get('judge')
            or file_grade != proof.get('grade')):
        raise ValueError('file regrade lacks a positive distinct Sol judgment')
    req = proof.get('judge_request')
    if not isinstance(req, dict) or hashlib.sha256(json.dumps(req, ensure_ascii=False,
            separators=(',', ':')).encode()).hexdigest() != proof.get('judge_request_sha256'):
        raise ValueError('Sol judgment request is not hash-bound')
    try:
        judge_payload = json.loads(req['messages'][1]['content'])
    except (TypeError, KeyError, IndexError, json.JSONDecodeError) as error:
        raise ValueError('Sol request omits its exact source-grounded judgment payload') from error
    original_outcome = raw_row.get('outcome') or {}
    checks = original_outcome.get('checks') or {}
    if (original_outcome.get('status') != 'done' or original_outcome.get('accepted') is not False
            or (original_outcome.get('oracle') or {}).get('accepted') is not True
            or (original_outcome.get('oracle') or {}).get('level') != 'judged'
            or checks.get('files') is not False
            or any(value is not True for key, value in checks.items() if key != 'files')
            or set(original_outcome.get('rejection_reasons') or []) != {'files'}
            or original_outcome.get('effects') != {}):
        raise ValueError('raw Luna result is not the precisely regradable files-only rejection')
    program = raw_row.get('task', {}).get('program_ir')
    if not isinstance(program, dict) or program_row != program:
        raise ValueError('regrade does not preserve the exact source-selected program row')
    source_worlds = source_worlds.resolve(strict=True)
    source_world = _read_jsonl(source_worlds, qualification['source_world_id'])
    if (sha256_file_hex(source_worlds) != qualification.get('source_world_sha256')
            or source_world.get('source_groups') != qualification.get('source_groups')
            or canonical_sha(program.get('semantics', {}).get('folder_files')) !=
               canonical_sha(source_world.get('folder_files'))
            or canonical_sha(program.get('semantics', {}).get('expected_files')) !=
               canonical_sha(source_world.get('expected_files'))
            or canonical_sha(program) != canonical_sha(program_row)):
        raise ValueError('source-world rubric, folder, expected output, and selected program do not agree')
    files_spec = source_world.get('files_oracle')
    if not isinstance(files_spec, dict):
        raise ValueError('source world omits its authored file oracle')
    task_text = source_world.get('task', '') + '\n\nFile rubric: ' + files_spec.get('rubric', '')
    rubric_digest = hashlib.sha256((task_text + files_spec.get('rubric', '')).encode()).hexdigest()
    if (files_spec.get('compare') != 'rewrite'
            or rubric_digest != proof.get('rubric_sha256')
            or judge_payload.get('candidate', {}).get('task') != task_text):
        raise ValueError('judge request is not bound to the authored task and file rubric')
    if not set(qualification.get('source_groups') or []) <= set(program.get('source_groups') or []):
        raise ValueError('regrade source groups differ from exact task source custody')
    if (proof.get('source_groups') != ['loghub-system:HDFS']
            or source_world.get('source_groups') != ['loghub-system:HDFS']):
        raise ValueError('regrade is not bound to the exact upstream LogHub HDFS group')
    report = (original_outcome.get('files') or {}).get('reports/HDFS-window-05.json')
    expected_report = source_world['expected_files']['reports/HDFS-window-05.json']
    input_report = source_world['folder_files'].get('reports/HDFS-window-05.json')
    if (report is None or json.loads(report) != judge_payload.get('candidate', {}).get('candidate_report')
            or judge_payload.get('candidate', {}).get('source_window') !=
                json.loads(source_world['folder_files']['windows/HDFS-window-05.json'])
            or judge_payload.get('reference', {}).get('reference_report') != json.loads(expected_report)
            or judge_payload.get('candidate', {}).get('original_catalog') !=
                json.loads(source_world['folder_files']['catalog.json'])):
        raise ValueError('Sol request does not judge the exact created report and original source evidence')
    if input_report is not None:
        raise ValueError('created report unexpectedly existed in the original folder')
    actual = original_outcome.get('files') or {}
    full_actual = {**source_world['folder_files'], **actual}
    mechanically = _node(runtime, full_actual, source_world['expected_files'], source_world['folder_files'],
                         files_spec, proof['grade'])
    if (mechanically.get('accepted') is not True
            or qualification.get('combined_verdict') != mechanically
            or qualification.get('current_protection_preflight', {}).get('related') != []):
        raise ValueError('full exact file map does not pass with the captured independent Sol grade')
    plan_path = plan_path.resolve(strict=True)
    queue_path = queue_path.resolve(strict=True)
    result_path = result_path.resolve(strict=True)
    result_manifest_path = result_manifest_path.resolve(strict=True)
    plan = json.loads(plan_path.read_text())
    claim_id_expected = f"{plan.get('campaign_id')}-case-{claim_index:06d}" if type(claim_index) is int else None
    manifest = json.loads(result_manifest_path.read_text())
    queue = [json.loads(line) for line in queue_path.read_text().splitlines() if line.strip()]
    if (sha256_file_hex(plan_path) != plan_sha256
            or proof.get('original_plan_path') != str(plan_path)
            or proof.get('original_plan_sha256') != plan_sha256
            or not isinstance(claim_id_expected, str) or claim_id != claim_id_expected
            or proof.get('campaign_id') != plan.get('campaign_id')
            or len(queue) != 1 or queue[0].get('index') != claim_index
            or queue[0].get('output') != str(result_path)
            or manifest.get('version') != 'natlang.teacher_batch.native/1'
            or manifest.get('source_sha256') != plan.get('source_sha256')
            or manifest.get('range') != {'start': claim_index, 'count': 1}
            or manifest.get('model') != plan.get('model_id')):
        raise ValueError('regrade is detached from the exact immutable v8 plan and claim')
    old_outcome_sha = canonical_sha(original_outcome)
    new_outcome = copy.deepcopy(original_outcome)
    new_outcome['checks'] = {**checks, 'files': True}
    new_outcome['files_check'] = mechanically
    new_outcome['accepted'] = True
    new_outcome['rejection_reasons'] = []
    projection = {'schema': SCHEMA, 'qualification_path': str(qualification_path),
        'qualification_sha256': sha256_file_hex(qualification_path), 'sol_regrade_path': str(proof_path),
        'sol_regrade_sha256': sha256_file_hex(proof_path), 'plan_path': str(plan_path),
        'plan_sha256': plan_sha256, 'claim_id': claim_id, 'queue_path': str(queue_path),
        'queue_sha256': sha256_file_hex(queue_path), 'raw_result_path': str(result_path),
        'raw_result_sha256': raw_sha, 'result_manifest_path': str(result_manifest_path),
        'result_manifest_sha256': sha256_file_hex(result_manifest_path),
        'source_worlds_path': str(source_worlds), 'source_worlds_sha256': sha256_file_hex(source_worlds),
        'source_world_id': source_world['id'], 'source_world_sha256': qualification['source_world_sha256'],
        'task_id': program['id'], 'task_sha256': canonical_sha(program),
        'program_fields_preserved': True, 'raw_outcome_sha256': old_outcome_sha,
        'effective_outcome': new_outcome, 'effective_outcome_sha256': canonical_sha(new_outcome),
        'file_verdict_sha256': canonical_sha(mechanically),
        'fresh_source_group_check': qualification['current_protection_preflight'],
        'judge_model': 'gpt-6-sol', 'judge_provider': 'openai-codex',
        'oracle_level': 'judged', 'exact_oracle_claimed': False}
    derived = copy.deepcopy(raw_row)
    derived['outcome'] = new_outcome
    return derived, projection


def validate_projection(raw_row: dict, derived_row: dict, projection: dict) -> None:
    """Recheck the only permitted row change; task, trajectory and actor stay exact."""
    if not isinstance(projection, dict) or projection.get('schema') != SCHEMA:
        raise ValueError('derived provider row lacks an authenticated judged regrade')
    raw_result_hash = projection.get('raw_result_sha256')
    if (not isinstance(raw_result_hash, str) or len(raw_result_hash) != 64
            or any(char not in '0123456789abcdef' for char in raw_result_hash)):
        raise ValueError('regrade raw result binding is malformed')
    if projection.get('raw_outcome_sha256') != canonical_sha(raw_row.get('outcome')):
        raise ValueError('regrade changes or misbinds the original outcome')
    if projection.get('effective_outcome_sha256') != canonical_sha(projection.get('effective_outcome')):
        raise ValueError('regrade effective outcome hash is malformed')
    if projection.get('effective_outcome') != derived_row.get('outcome'):
        raise ValueError('derived row outcome differs from the explicit regrade projection')
    if projection.get('task_sha256') != canonical_sha((raw_row.get('task') or {}).get('program_ir')):
        raise ValueError('regrade changes the original task/program hash')
    raw_comparison = copy.deepcopy(derived_row)
    raw_comparison.pop('source_regrade', None)
    raw_comparison['outcome'] = raw_row.get('outcome')
    if canonical_sha(raw_comparison) != canonical_sha(raw_row):
        raise ValueError('derived provider row changes fields outside its explicit outcome projection')
    if (derived_row.get('provenance') != raw_row.get('provenance')
            or derived_row.get('trajectory') != raw_row.get('trajectory')
            or derived_row.get('execution_identity') != raw_row.get('execution_identity')):
        raise ValueError('regrade changes the actual provider actor or captured execution graph')
