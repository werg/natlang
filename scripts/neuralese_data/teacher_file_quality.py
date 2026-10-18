"""Exact JSON file-format evaluation continuation; original executions stay intact."""
from __future__ import annotations
import copy
import hashlib
import json
import re
from pathlib import Path
import subprocess

from natlang_neuralese.common.hashing import canonical_json_sha256_hex, sha256_file_hex
from scripts.neuralese_data.ground_teacher_continuations import OUTPUT_CONTRACT

SCHEMA = 'natlang.teacher-json-file-quality-continuation/1'
AUTHORING_SCHEMA = 'natlang.teacher-executable-authoring-quality-continuation/1'
AUTHORING_REVIEWED_ALIAS_SCHEMA = 'natlang.teacher-reviewed-source-authoring-quality-continuation/1'
AUTHORING_ALIAS_SCHEMA = 'natlang.teacher-source-grounded-authoring-quality-continuation/1'
AUTHORING_REQUIREMENT_SCHEMA = 'natlang.teacher-authoring-construct-quality-continuation/1'
AUTHORING_COMPILER_SCHEMA = 'natlang.teacher-authoring-compiler-quality-continuation/1'
SPAN_REVIEW_SCHEMA = 'natlang.teacher-span-answer-quality-continuation/1'
HISTORICAL_AUTHORING_RUNTIME = '43e248a84b6faba6e793181fe593796dc03c878b05199091d95e824408de7ed1'


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
    schema = row.get('source_quality_continuation', {}).get('schema')
    continuation = (continue_authoring_requirement_quality if schema == AUTHORING_REQUIREMENT_SCHEMA else
                    continue_authoring_alias_quality if schema == AUTHORING_ALIAS_SCHEMA else
                    continue_authoring_quality if schema == AUTHORING_SCHEMA else continue_quality)
    if schema == SPAN_REVIEW_SCHEMA:
        expected = continue_span_answer_quality(original, original_binding,
            review_reference=row['source_quality_continuation']['source_review'])
    elif schema == AUTHORING_REVIEWED_ALIAS_SCHEMA:
        expected = continue_authoring_alias_quality(original, original_binding,
            source_review=row['source_quality_continuation']['source_review'])
    elif schema == AUTHORING_COMPILER_SCHEMA:
        expected = continue_authoring_compiler_quality(original, original_binding,
            row['source_quality_continuation']['historical_runtime'])
    else:
        expected = continuation(original, original_binding)
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


def reviewed_span_answer(original, original_binding, review_reference):
    """Authenticate an independent NL verdict, without deciding equivalence here."""
    from scripts.coreweave.teacher_tranche_admission import read_bound
    review = read_bound(review_reference)
    if (review.get('schema') != 'natlang.teacher-span-answer-semantic-review/1'
            or review.get('reviewer') != 'independent-root-source-meaning-review'
            or review.get('training_admission') is not False
            or not isinstance(review.get('reason_verbatim'), str)
            or not review['reason_verbatim'].strip()):
        raise ValueError('span answer requires independent source-rubric review')
    inventory = read_bound(review['inventory'])
    if (inventory.get('schema') != 'natlang.source-owned-answer-equivalence-review-inventory/1'
            or inventory.get('training_admission') is not False
            or inventory.get('parent_release') is not False):
        raise ValueError('span answer review inventory is not exact non-admission custody')
    if sha256_file_hex(original_binding['path']) != original_binding['sha256']:
        raise ValueError('span answer original raw file changed')
    from natlang_neuralese.common.jsonio import iter_jsonl_rows
    captured_rows = [value for _, value in iter_jsonl_rows(original_binding['path'])
                     if value.get('id') == original.get('id')]
    if (len(captured_rows) != 1
            or canonical_json_sha256_hex(captured_rows[0]) != canonical_json_sha256_hex(original)):
        raise ValueError('span answer differs from complete original captured row')
    matches = [entry for entry in inventory['records']
        if entry['source_row_id'] == original.get('id')
        and entry['raw_output']['sha256'] == original_binding['sha256']]
    approvals = [entry for entry in review.get('approved', [])
        if entry.get('row_id') == original.get('id')
        and entry.get('raw_output_sha256') == original_binding['sha256']]
    if len(matches) != 1 or len(approvals) != 1:
        raise ValueError('span answer needs one exact reviewed original and approval')
    entry, approval = matches[0], approvals[0]
    program = original['task']['program_ir']; semantics = program['semantics']
    outcome = original['outcome']
    expected = {'source_row_id': original['id'], 'program_id': program['id'],
        'program_ir_sha256': canonical_json_sha256_hex(program), 'family': program.get('family'),
        'source_groups': program['source_groups'], 'source_gold': program.get('gold_sources'),
        'question': semantics['inputs'].get('question'), 'expected': semantics['expected'],
        'declared_oracle': semantics['oracle'], 'declared_instruction_files': semantics['files'],
        'captured_actual': outcome['value'], 'captured_oracle': outcome['oracle'],
        'captured_checks': outcome['checks'], 'captured_file_check': outcome.get('files_check'),
        'training_admission': False}
    if (any(canonical_json_sha256_hex(entry.get(key)) != canonical_json_sha256_hex(value)
            for key, value in expected.items())
            or entry['raw_output']['bytes'] != Path(original_binding['path']).stat().st_size
            or approval.get('program_ir_sha256') != expected['program_ir_sha256']
            or approval.get('accepted') is not True
            or not isinstance(approval.get('reason'), str) or not approval['reason'].strip()):
        raise ValueError('reviewed span answer/source/rubric binding differs')
    return {'accepted': True, 'verdict': approval['reason']}


def span_answer_verdict(original, approval):
    """Use the existing span oracle and its exact, source-bound judge interface."""
    root = Path(__file__).resolve().parents[2]
    oracle = root / 'ts-host/dist/evaluation/oracles.js'
    paths = [Path(__file__).resolve(), root / 'ts-host/src/evaluation/oracles.ts',
             *sorted((root / 'ts-host/dist').rglob('*.js'))]
    bindings = {str(path.relative_to(root)): sha256_file_hex(path) for path in paths}
    semantics = original['task']['program_ir']['semantics']
    payload = {'actual': original['outcome']['value'], 'expected': semantics['expected'],
               'spec': semantics['oracle'], 'approval': approval}
    code = ('import {readFileSync} from "node:fs";import {isDeepStrictEqual} from "node:util";'
        'const {checkOracle}=await import(process.argv[1]);'
        'const p=JSON.parse(readFileSync(0,"utf8"));let calls=0;'
        'const legacy=await checkOracle(p.actual,p.expected,p.spec);'
        'const verdict=await checkOracle(p.actual,p.expected,p.spec,async input=>{'
        'if(!isDeepStrictEqual(input,{actual:p.actual,expected:{answer:p.expected,context:p.spec.context},'
        'rubric:p.spec.rubric}))throw Error("reviewed oracle input changed");'
        'calls++;return p.approval;});console.log(JSON.stringify({legacy,verdict,calls}));')
    completed = subprocess.run(['node', '--input-type=module', '-e', code, oracle.as_uri()],
        input=json.dumps(payload), capture_output=True, text=True, check=True)
    if bindings != {str(path.relative_to(root)): sha256_file_hex(path) for path in paths}:
        raise ValueError('span answer verifier changed during execution')
    return json.loads(completed.stdout), {'files': len(bindings),
        'inventory_sha256': canonical_json_sha256_hex(bindings)}


def continue_span_answer_quality(original, original_binding, *, review_reference):
    if 'source_quality_continuation' in original:
        raise ValueError('span answer continuation requires original captured execution')
    program = original['task']['program_ir']; semantics = program['semantics']
    outcome = original['outcome']; checks = outcome.get('checks'); spec = semantics.get('oracle')
    required = {'seeded_failure_requirement', 'expected_status', 'honest_stop',
                'effects', 'files', 'world', 'file_return_consistency', 'answer'}
    if (program.get('split') != 'train' or not program.get('source_groups')
            or not isinstance(spec, dict) or spec.get('level') != 'span'
            or not isinstance(spec.get('rubric'), str) or not spec['rubric'].strip()
            or outcome.get('status') != 'done' or outcome.get('accepted') is not False
            or outcome.get('oracle', {}).get('accepted') is not False
            or outcome.get('oracle', {}).get('level') != 'span'
            or outcome.get('oracle', {}).get('needs_review') is not True
            or outcome.get('quality_pending') != ['answer_needs_review']
            or outcome.get('rejection_reasons') != ['answer']
            or not isinstance(checks, dict) or not required <= set(checks)
            or checks.get('answer') is not False
            or any(value is not True for key, value in checks.items() if key != 'answer')):
        raise ValueError('span answer requires only a completed answer-equivalence review')
    approval = reviewed_span_answer(original, original_binding, review_reference)
    checked, verifier = span_answer_verdict(original, approval)
    if (reviewed_span_answer(original, original_binding, review_reference) != approval
            or canonical_json_sha256_hex(checked['legacy']) != canonical_json_sha256_hex(outcome['oracle'])
            or checked.get('calls') != 1 or checked['verdict'].get('accepted') is not True
            or checked['verdict'].get('level') != 'span' or checked['verdict'].get('needs_review')
            or checked['verdict'].get('score') != checked['legacy'].get('score')):
        raise ValueError('fresh original span comparison and bound semantic review failed')
    result = copy.deepcopy(original)
    result['outcome'].update(accepted=True, oracle=checked['verdict'],
        quality_pending=[], rejection_reasons=[])
    result['outcome']['checks']['answer'] = True
    result['source_quality_continuation'] = {'schema': SPAN_REVIEW_SCHEMA,
        'original': original_binding, 'original_row_sha256': canonical_json_sha256_hex(original),
        'source_review': review_reference, 'verifier': verifier,
        'comparator': {'kind': 'source-reviewed-span-answer/1'},
        'claim': 'Independent semantic review under unchanged source span rubric; original negative oracle, files, target answer and complete captured trajectory retained unchanged in original custody'}
    return result


def authoring_verdict(row, *, source_aliases=False, construct_requirements=False, reviewed_source=None):
    """Run the actual existing executable contract; no model generation."""
    root = Path(__file__).resolve().parents[2]
    paths = [Path(__file__).resolve(), root / 'ts-host/src/teacher/authoring.ts',
             *sorted((root / 'ts-host/dist').rglob('*.js'))]
    bindings = {str(path.relative_to(root)): sha256_file_hex(path) for path in paths}
    payload = {'files': row['outcome']['files'], 'spec': row['task']['program_ir']['semantics']['authoring']}
    code = ('import {readFileSync} from "node:fs";'
            'const {checkAuthoring}=await import(process.argv[1]);'
            'const p=JSON.parse(readFileSync(0,"utf8"));'
            'console.log(JSON.stringify(await checkAuthoring(p.files,p.spec)));')
    if source_aliases:
        code = ('import {readFileSync} from "node:fs";'
                'const {checkAuthoring,checkAuthoringLegacyRequirements,withSourceArgumentAliases}=await import(process.argv[1]);'
                'const p=JSON.parse(readFileSync(0,"utf8"));'
                'const legacy=await checkAuthoringLegacyRequirements(p.files,p.spec);'
                'const spec=withSourceArgumentAliases(p.spec);'
                'console.log(JSON.stringify({legacy,spec,verdict:await checkAuthoring(p.files,spec)}));')
    if construct_requirements:
        if source_aliases:
            raise ValueError('authoring continuations must independently bind their exact correction')
        code = ('import {readFileSync} from "node:fs";'
                'const {checkAuthoring,checkAuthoringLegacyRequirements}=await import(process.argv[1]);'
                'const p=JSON.parse(readFileSync(0,"utf8"));'
                'console.log(JSON.stringify({legacy:await checkAuthoringLegacyRequirements(p.files,p.spec),'
                'verdict:await checkAuthoring(p.files,p.spec)}));')
    if reviewed_source is not None:
        if source_aliases or construct_requirements:
            raise ValueError('reviewed source aliases require a separate original current verdict')
        payload['review'] = reviewed_source
        code = ('import {readFileSync} from "node:fs";'
                'const {checkAuthoring,withSourceArgumentAliases}=await import(process.argv[1]);'
                'const p=JSON.parse(readFileSync(0,"utf8"));'
                'const legacy=await checkAuthoring(p.files,p.spec);'
                'const spec=withSourceArgumentAliases({...p.spec,source_criterion_review:p.review},p.files);'
                'console.log(JSON.stringify({legacy,spec,verdict:await checkAuthoring(p.files,spec)}));')
    completed = subprocess.run(['node', '--input-type=module', '-e', code,
        (root / 'ts-host/dist/teacher/authoring.js').as_uri()],
        input=json.dumps(payload), capture_output=True, text=True, check=True)
    if bindings != {str(path.relative_to(root)): sha256_file_hex(path) for path in paths}:
        raise ValueError('authoring verifier changed during execution')
    return json.loads(completed.stdout), {'files': len(bindings),
        'inventory_sha256': canonical_json_sha256_hex(bindings)}


def continue_authoring_quality(original, original_binding):
    if 'source_quality_continuation' in original:
        raise ValueError('authoring continuation requires original captured execution')
    program = original['task']['program_ir']; semantics = program['semantics']; outcome = original['outcome']
    spec = semantics.get('authoring'); checks = outcome.get('checks')
    if (program.get('split') != 'train' or not program.get('source_groups')
        or 'expected' not in semantics or semantics['expected'] is not None
        or not isinstance(spec, dict) or not isinstance(spec.get('runs'), list) or not spec['runs']
        or not isinstance(outcome.get('files'), dict) or not outcome['files']
        or outcome.get('status') != 'done' or outcome.get('accepted') is not True
        or outcome.get('oracle') != {'accepted': False, 'level': 'exact'}
        or outcome.get('quality_pending', []) != [] or outcome.get('rejection_reasons') != []
        or not isinstance(checks, dict)
        or not {'seeded_failure_requirement','expected_status','honest_stop','effects','files','world','file_return_consistency','answer'} <= set(checks)
        or any(value is not True for value in checks.values())
        or checks.get('files') is not True or checks.get('answer') is not True
        or not isinstance(outcome.get('authoring'), dict)
        or outcome['authoring'].get('ok') is not True or outcome['authoring'].get('problems') != []):
        raise ValueError('authoring continuation requires only root-return oracle accounting mismatch')
    for run in spec['runs']:
        if not isinstance(run, dict) or not isinstance(run.get('args'), list) or 'expected' not in run:
            raise ValueError('authoring continuation lacks complete declared runs')
    verdict, verifier = authoring_verdict(original)
    if (verdict.get('ok') is not True or verdict.get('problems') != []
        or len(verdict.get('results', [])) != len(spec['runs'])
        or canonical_json_sha256_hex(verdict) != canonical_json_sha256_hex(outcome['authoring'])):
        raise ValueError('fresh executable authoring verification failed')
    result = copy.deepcopy(original)
    result['outcome']['oracle'] = {'accepted': True, 'level': 'exact', 'verdict': 'executable-authoring/1'}
    result['source_quality_continuation'] = {'schema': AUTHORING_SCHEMA,
        'original': original_binding, 'original_row_sha256': canonical_json_sha256_hex(original),
        'verifier': verifier, 'comparator': {'kind': 'executable-authoring/1'},
        'claim': 'evaluation-only root oracle accounting correction; original authored files, tests and execution unchanged'}
    return result


def continue_authoring_requirement_quality(original, original_binding):
    """Repair a reproduced syntax-presence false-negative, never executable failure."""
    if 'source_quality_continuation' in original:
        raise ValueError('construct continuation requires original captured execution')
    program = original['task']['program_ir']; semantics = program['semantics']; outcome = original['outcome']
    spec = semantics.get('authoring'); checks = outcome.get('checks'); captured = outcome.get('authoring')
    required = {'seeded_failure_requirement','expected_status','honest_stop','effects','files','world','file_return_consistency','answer'}
    if (program.get('split') != 'train' or not program.get('source_groups')
        or 'expected' not in semantics or semantics['expected'] is not None
        or not isinstance(spec, dict) or not isinstance(spec.get('runs'), list) or not spec['runs']
        or (spec.get('requires') or {}).get('nl') is not True
        or not isinstance(outcome.get('files'), dict) or not outcome['files']
        or outcome.get('status') != 'done' or outcome.get('accepted') is not False
        or outcome.get('oracle') != {'accepted': False, 'level': 'exact'}
        or outcome.get('quality_pending', []) != [] or outcome.get('rejection_reasons') != ['files']
        or not isinstance(checks, dict) or not required <= set(checks) or checks.get('files') is not False
        or any(value is not True for key, value in checks.items() if key != 'files')
        or not isinstance(captured, dict) or captured.get('ok') is not False
        or captured.get('problems') != [str(spec.get('module')) + ' creates no nl function']):
        raise ValueError('construct continuation requires only a captured nl-presence false-negative')
    checked, verifier = authoring_verdict(original, construct_requirements=True)
    verdict = checked['verdict']
    if (canonical_json_sha256_hex(checked['legacy']) != canonical_json_sha256_hex(captured)
        or verdict.get('ok') is not True or verdict.get('problems') != []
        or len(verdict.get('results', [])) != len(spec['runs'])
        or canonical_json_sha256_hex(verdict.get('results')) != canonical_json_sha256_hex(captured.get('results'))):
        raise ValueError('fresh complete authoring construct verification failed')
    result = copy.deepcopy(original)
    result['outcome'].update(accepted=True, authoring=verdict, rejection_reasons=[],
        oracle={'accepted': True, 'level': 'exact', 'verdict': 'executable-authoring-construct/1'})
    result['outcome']['checks']['files'] = True
    result['source_quality_continuation'] = {'schema': AUTHORING_REQUIREMENT_SCHEMA,
        'original': original_binding, 'original_row_sha256': canonical_json_sha256_hex(original),
        'verifier': verifier, 'comparator': {'kind': 'typescript-authoring-construct/1'},
        'original_spec_sha256': canonical_json_sha256_hex(spec),
        'claim': 'Syntax-presence evaluation correction only; original complete tests, files, targets and captured trajectory unchanged'}
    return result


def continue_authoring_alias_quality(original, original_binding, *, source_review=None):
    if 'source_quality_continuation' in original:
        raise ValueError('argument oracle continuation requires original captured execution')
    program=original['task']['program_ir']; semantics=program['semantics']; outcome=original['outcome']
    spec=semantics.get('authoring'); checks=outcome.get('checks'); captured=outcome.get('authoring')
    required={'seeded_failure_requirement','expected_status','honest_stop','effects','files','world','file_return_consistency','answer'}
    if (program.get('split')!='train' or not program.get('source_groups')
        or 'expected'not in semantics or semantics['expected']is not None
        or not isinstance(spec,dict) or 'argument_aliases'in spec or 'source_criterion_review'in spec
        or not isinstance(spec.get('runs'),list) or not spec['runs']
        or not isinstance(outcome.get('files'),dict) or not outcome['files']
        or outcome.get('status')!='done' or outcome.get('accepted')is not False
        or outcome.get('oracle') not in ([{'accepted':False,'level':'exact'},
             {'accepted':False,'level':'exact','verdict':'executable-authoring/1'}] if source_review is not None
             else [{'accepted':False,'level':'exact'}])
        or outcome.get('quality_pending',[])!=[] or outcome.get('rejection_reasons')!=['files']
        or not isinstance(checks,dict) or not required<=set(checks) or checks.get('files')is not False
        or any(value is not True for key,value in checks.items()if key!='files')
        or not isinstance(captured,dict) or captured.get('ok')is not False
        or not captured.get('problems') or any(not isinstance(item,str)or'The authoring oracle has no answer for this call.'not in item for item in captured['problems'])):
        raise ValueError('argument oracle continuation requires only captured child-oracle coverage failure')
    reviewed = None
    if source_review is not None:
        reviewed = reviewed_source_criteria(original, original_binding, source_review)
    checked,verifier=authoring_verdict(original,source_aliases=source_review is None,reviewed_source=reviewed)
    if source_review is not None and reviewed_source_criteria(original, original_binding, source_review) != reviewed:
        raise ValueError('source child review changed during executable verification')
    if canonical_json_sha256_hex(checked['legacy'])!=canonical_json_sha256_hex(captured):
        raise ValueError('original executable authoring verdict was not reproduced')
    derived_spec=checked['spec'];verdict=checked['verdict']
    without_aliases={key:value for key,value in derived_spec.items()if key not in ('argument_aliases','source_criterion_review')}
    if (canonical_json_sha256_hex(without_aliases)!=canonical_json_sha256_hex(spec)
        or not derived_spec.get('argument_aliases') or verdict.get('ok')is not True
        or verdict.get('problems')!=[] or len(verdict.get('results',[]))!=len(spec['runs'])
        or (source_review is not None and canonical_json_sha256_hex(verdict.get('results')) !=
            canonical_json_sha256_hex([run['expected'] for run in spec['runs']]))):
        raise ValueError('source-grounded executable authoring verification failed')
    result=copy.deepcopy(original);result['outcome'].update(accepted=True,authoring=verdict,rejection_reasons=[],
        oracle={'accepted':True,'level':'exact','verdict':'source-grounded-executable-authoring/1'})
    result['outcome']['checks']['files']=True
    result['source_quality_continuation']={'schema':AUTHORING_REVIEWED_ALIAS_SCHEMA if source_review is not None else AUTHORING_ALIAS_SCHEMA,'original':original_binding,
        'original_row_sha256':canonical_json_sha256_hex(original),'verifier':verifier,
        'comparator':{'kind':'source-run-argument-authoring/1'},
        'original_spec_sha256':canonical_json_sha256_hex(spec),
        'derived_spec_sha256':canonical_json_sha256_hex(derived_spec),
        'argument_aliases':derived_spec['argument_aliases'],
        'claim':'Source-derived child-oracle coverage and null-root accounting correction; original tests, expected results, files and captured trajectory unchanged'}
    if source_review is not None:
        result['source_quality_continuation'].update(source_review=source_review,source_criterion_review=reviewed)
    return result


def historical_authoring_verdict(row, runtime_reference):
    """Reproduce the exact captured generation epoch; arbitrary checkers are forbidden."""
    from natlang_neuralese.common.runtime_inventory import task_runtime_inventory
    if (not isinstance(runtime_reference, dict) or set(runtime_reference) != {'path','inventory_sha256'}
        or runtime_reference['inventory_sha256'] != HISTORICAL_AUTHORING_RUNTIME):
        raise ValueError('historical authoring runtime is not the captured compiler epoch')
    runtime = Path(runtime_reference['path']).resolve(strict=True)
    inventory = task_runtime_inventory(runtime)
    if canonical_json_sha256_hex(inventory) != HISTORICAL_AUTHORING_RUNTIME:
        raise ValueError('historical authoring runtime changed')
    code = ('import {readFileSync} from "node:fs";'
            'const {checkAuthoring}=await import(process.argv[1]);'
            'const p=JSON.parse(readFileSync(0,"utf8"));'
            'console.log(JSON.stringify(await checkAuthoring(p.files,p.spec)));')
    payload = {'files': row['outcome']['files'], 'spec': row['task']['program_ir']['semantics']['authoring']}
    completed = subprocess.run(['node','--input-type=module','-e',code,
        (runtime/'dist/teacher/authoring.js').as_uri()],input=json.dumps(payload),
        capture_output=True,text=True,check=True)
    if task_runtime_inventory(runtime) != inventory:
        raise ValueError('historical authoring runtime changed during execution')
    return json.loads(completed.stdout)


def continue_authoring_compiler_quality(original, original_binding, historical_runtime):
    """Recover a reproduced compiler rejection only after every unchanged run passes."""
    if 'source_quality_continuation' in original:
        raise ValueError('compiler continuation requires original captured execution')
    program=original['task']['program_ir'];semantics=program['semantics'];outcome=original['outcome']
    spec=semantics.get('authoring');checks=outcome.get('checks');captured=outcome.get('authoring')
    required={'seeded_failure_requirement','expected_status','honest_stop','effects','files','world','file_return_consistency','answer'}
    runs=spec.get('runs')if isinstance(spec,dict)else None
    marker='This `nl` expression was not compiled by the natlang compiler.'
    problems=captured.get('problems')if isinstance(captured,dict)else None
    if (isinstance(problems,list)and isinstance(spec,dict)
        and (spec.get('requires')or {}).get('nl')is True
        and problems[:1]==[str(spec.get('module'))+' creates no nl function']):
        problems=problems[1:]
    if (program.get('split')!='train' or not program.get('source_groups')
        or 'expected'not in semantics or semantics['expected']is not None
        or not isinstance(runs,list)or not runs
        or any(not isinstance(run,dict)or not isinstance(run.get('args'),list)or 'expected'not in run for run in runs)
        or not isinstance(outcome.get('files'),dict)or not outcome['files']
        or outcome.get('status')!='done' or outcome.get('accepted')is not False
        or outcome.get('oracle')!={'accepted':False,'level':'exact'}
        or outcome.get('quality_pending',[])!=[] or outcome.get('rejection_reasons')!=['files']
        or not isinstance(checks,dict)or not required<=set(checks)or checks.get('files')is not False
        or any(v is not True for k,v in checks.items()if k!='files')
        or not isinstance(captured,dict)or captured.get('ok')is not False
        or captured.get('results')!=[None]*len(runs)
        or not isinstance(problems,list)or len(problems)!=len(runs)
        or any(not isinstance(problem,str)or not problem.startswith(f'run {i}: authoring_check: quiesced: error: The authored export failed: '+marker)for i,problem in enumerate(problems))):
        raise ValueError('compiler continuation requires only captured raw-nl compiler failure')
    legacy=historical_authoring_verdict(original,historical_runtime)
    if canonical_json_sha256_hex(legacy)!=canonical_json_sha256_hex(captured):
        raise ValueError('historical compiler verdict differs from captured execution')
    verdict,verifier=authoring_verdict(original)
    if (verdict.get('ok')is not True or verdict.get('problems')!=[]
        or canonical_json_sha256_hex(verdict.get('results'))!=canonical_json_sha256_hex([run['expected']for run in runs])):
        raise ValueError('fresh complete authoring compiler verification failed')
    result=copy.deepcopy(original)
    result['outcome'].update(accepted=True,authoring=verdict,rejection_reasons=[],
        oracle={'accepted':True,'level':'exact','verdict':'executable-authoring-compiler/1'})
    result['outcome']['checks']['files']=True
    result['source_quality_continuation']={'schema':AUTHORING_COMPILER_SCHEMA,'original':original_binding,
        'original_row_sha256':canonical_json_sha256_hex(original),'verifier':verifier,
        'historical_runtime':copy.deepcopy(historical_runtime),'original_spec_sha256':canonical_json_sha256_hex(spec),
        'comparator':{'kind':'typescript-authoring-compiler/1'},
        'claim':'Reproduced historical compiler rejection; unchanged files and ALL exact original runs pass current compiler. Original actor, trajectory and targets unchanged.'}
    return result


def reviewed_source_criteria(original, original_binding, review_reference):
    from scripts.coreweave.teacher_tranche_admission import read_bound
    review = read_bound(review_reference)
    if (review.get('schema') != 'natlang.authored-child-source-semantic-review/1'
            or review.get('reviewer') != 'independent-root-source-meaning-review'
            or review.get('training_admission') is not False
            or review.get('proposition') != 'whether-source-notes-announce-these-failures-as-expected'):
        raise ValueError('source child aliases require explicit independent criterion review')
    inventory = read_bound(review['inventory'])
    if inventory.get('schema') != 'natlang.authored-source-criterion-custody/1' or inventory.get('training_admission') is not False:
        raise ValueError('source child review has no exact original criterion inventory')
    approved = review.get('approved_templates')
    templates = inventory.get('templates')
    if (not isinstance(approved, dict) or not approved or not isinstance(templates, dict)
            or set(approved) - set(templates)):
        raise ValueError('source child review approves unknown original criteria')
    if sha256_file_hex(original_binding['path']) != original_binding['sha256']:
        raise ValueError('source child original raw file changed')
    matches = [row for row in inventory['rows'] if row['original']['sha256'] == original_binding['sha256']
               and row['row_id'] == original.get('id')]
    if len(matches) != 1:
        raise ValueError('source child original needs exactly one reviewed inventory row')
    row = matches[0]; program = original['task']['program_ir']; module = program['semantics']['authoring']['module']
    source = original['outcome']['files'][module]
    if (row['program_sha256'] != canonical_json_sha256_hex(program)
            or canonical_json_sha256_hex(row['authoring_spec']) != canonical_json_sha256_hex(program['semantics']['authoring'])
            or row['captured_file']['content'] != source
            or row['captured_file']['sha256'] != hashlib.sha256(source.encode()).hexdigest()):
        raise ValueError('reviewed source child program or captured module differs')
    criteria = []; bindings = None
    for tag in row['ast']['tags']:
        descriptor = {key: tag[key] for key in ('tag', 'typeArguments', 'template', 'interpolated')}
        key = canonical_json_sha256_hex(descriptor)
        if key not in approved:
            continue
        if (key not in templates or templates[key]['criterion'] != descriptor
                or row['key'] not in templates[key]['rows']):
            raise ValueError('source child template does not bind the original AST inventory')
        entry = approved[key]
        if (not isinstance(entry, dict) or set(entry) - {'instruction', 'result_kind', 'literal_bindings'}
                or not isinstance(entry.get('instruction'), str) or not entry['instruction'].strip()
                or entry.get('result_kind') not in ('boolean', 'decision')):
            raise ValueError('source child criterion review is malformed')
        if entry.get('literal_bindings') is not None:
            if bindings is not None and bindings != entry['literal_bindings']:
                raise ValueError('source child interpolation bindings conflict')
            bindings = entry['literal_bindings']
        criteria.append({key: entry[key] for key in ('instruction', 'result_kind')})
    if not criteria:
        raise ValueError('source child has no independently reviewed exact criterion')
    return {'schema': 'natlang.reviewed-source-child-criteria/1',
            'module_sha256': row['captured_file']['sha256'], 'criteria': criteria,
            **({'literal_bindings': bindings} if bindings is not None else {})}
