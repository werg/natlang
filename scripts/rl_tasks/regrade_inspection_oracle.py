#!/usr/bin/env python3
"""Create an explicitly versioned derived row for an independently corrected BIRD inspection oracle.

The input trajectory is immutable and must remain rejected under its historical fixture. The derived
row is emitted only after joining exact raw-result, source-case, public-policy, dataset and SQL evidence.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sqlite3
from pathlib import Path
from typing import Any


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def _source_facts(database: Path, business_id: int) -> dict[str, Any]:
    connection = sqlite3.connect(f'file:{database}?mode=ro', uri=True)
    try:
        inspections = connection.execute(
            'SELECT date,type,score FROM inspections WHERE business_id=? ORDER BY date,type,score',
            (business_id,),
        ).fetchall()
        violations = connection.execute(
            'SELECT date,violation_type_id,risk_category,description FROM violations '
            'WHERE business_id=? ORDER BY date,violation_type_id,description',
            (business_id,),
        ).fetchall()
    finally:
        connection.close()
    if not inspections:
        raise ValueError('regrade requires at least one source inspection')
    latest = max(row[0] for row in inspections)
    scoreless = sorted({date for date, event_type, score in inspections
                        if score is None and re.search(r'reinspection|follow.?up', event_type, re.I)})
    dates_by_type: dict[str, set[str]] = {}
    for date, type_id, risk, _description in violations:
        if risk == 'High Risk':
            dates_by_type.setdefault(str(type_id), set()).add(str(date))
    recurring = sorted({date for dates in dates_by_type.values() if len(dates) >= 2 for date in dates})
    return {'latest_inspection_date': latest, 'scoreless_followup_dates': scoreless,
            'recurrent_high_risk_dates': recurring}


def regrade(*, raw_result_path: Path, case_path: Path, proof_path: Path,
            expected_proof_sha256: str) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_bytes = raw_result_path.read_bytes()
    raw = json.loads(raw_bytes)
    proof_bytes = proof_path.read_bytes()
    if not re.fullmatch(r'[0-9a-f]{64}', expected_proof_sha256) or _sha(proof_bytes) != expected_proof_sha256:
        raise ValueError('oracle regrade proof differs from the separately pinned proof digest')
    proof = json.loads(proof_bytes)
    cases_bytes = case_path.read_bytes()
    cases = [json.loads(line) for line in cases_bytes.decode('utf-8').splitlines() if line.strip()]
    if proof.get('schema') != 'natlang.oracle-regrade-proof/1':
        raise ValueError('unsupported oracle regrade proof schema')
    if proof.get('disposition') != ('retain original raw rejection unchanged; corrected independent regrade is not a training row and is only consumable by a converter that verifies this proof and all bindings'):
        raise ValueError('proof disposition is not the reviewed no-rewrite policy')
    binding = proof.get('source_row', {})
    if binding.get('path') != str(case_path.resolve()) or binding.get('sha256') != _sha(cases_bytes):
        raise ValueError('source case file path or digest differs from the proof')
    upstream_id = binding.get('upstream_id')
    program_id = raw.get('task', {}).get('program_ir', {}).get('id')
    matches = [row for row in cases if row.get('id') == program_id and
               row.get('curriculum', {}).get('external', {}).get('upstream_id') == upstream_id]
    if not isinstance(upstream_id, str) or len(matches) != 1:
        raise ValueError('exactly one source case row must match the proof-bound upstream and program IDs')
    case = matches[0]
    if _sha(_canonical(case)) != binding.get('row_sha256'):
        raise ValueError('source case row digest differs from the proof')
    if raw.get('task', {}).get('program_ir') != case:
        raise ValueError('raw task program differs from the pinned public case row')
    program = raw['task']['program_ir']
    if program.get('split') != 'train' or not {'sql-db:bird:food_inspection', 'ext-unit:bird:db:food_inspection'} <= set(program.get('source_groups', [])):
        raise ValueError('raw task does not preserve the BIRD train source groups')
    if program.get('curriculum', {}).get('external', {}).get('program_revision') != binding.get('program_revision'):
        raise ValueError('raw task program revision differs from the proof')
    original = program.get('semantics', {}).get('expected')
    old = proof.get('original_oracle', {})
    corrected = proof.get('corrected_oracle', {})
    if old.get('kind') != 'exact' or _sha(_canonical(original)) != old.get('expected_sha256') or original != old.get('expected'):
        raise ValueError('historical expected answer does not match its proof binding')
    if corrected.get('version') != 'inspection-review-date-union/1' or corrected.get('accepted_under_corrected_oracle') is not True:
        raise ValueError('corrected oracle is not the reviewed date-union version')
    corrected_expected = corrected.get('expected')
    if _sha(_canonical(corrected_expected)) != corrected.get('expected_sha256'):
        raise ValueError('corrected expected answer digest mismatch')
    outcome = raw.get('outcome', {})
    if (_sha(raw_bytes) != old.get('raw_result_sha256') or outcome.get('accepted') is not False or
        outcome.get('rejection_reasons') != ['answer'] or outcome.get('status') != 'done' or
        outcome.get('oracle', {}).get('accepted') is not False or outcome.get('oracle', {}).get('level') != 'exact'):
        raise ValueError('raw historical rejection identity or status does not match the proof')
    checks = outcome.get('checks')
    if not isinstance(checks, dict) or checks.get('answer') is not False or any(
        value is not True for key, value in checks.items() if key != 'answer'
    ) or not {'seeded_failure_requirement', 'expected_status', 'honest_stop', 'effects', 'files', 'world', 'file_return_consistency'} <= set(checks):
        raise ValueError('raw trajectory has failures other than the original answer mismatch')
    if raw.get('provenance', {}).get('model') != 'gpt-6-luna' or raw.get('provenance', {}).get('provider') != 'openai-codex' or raw.get('provenance', {}).get('transport') != 'pi-provider' or raw.get('provenance', {}).get('collection_role') != 'teacher':
        raise ValueError('raw trajectory actor/provider/transport is not the actual reviewed teacher')
    actual = outcome.get('value')
    if not isinstance(actual, dict) or any(actual.get(key) != value for key, value in original.items() if key != 'recommendationDates'):
        raise ValueError('raw answer differs from the historical expected answer outside the corrected field')

    source = proof.get('independent_source_check', {})
    database = Path(source.get('database_path', '')).resolve(strict=True)
    if _sha(database.read_bytes()) != source.get('database_sha256'):
        raise ValueError('pinned BIRD training database digest mismatch')
    if source.get('dataset') != 'BIRD food_inspection train' or source.get('business_id') != 489:
        raise ValueError('regrade proof is bound to another source or business')
    facts = _source_facts(database, 489)
    if facts != source.get('query_facts'):
        raise ValueError('independent SQL facts differ from the bound proof')
    expected_dates = sorted(set([facts['latest_inspection_date'], *facts['scoreless_followup_dates'],
                                 *facts['recurrent_high_risk_dates']]))
    if corrected_expected.get('recommendationDates') != expected_dates or actual.get('recommendationDates') != expected_dates:
        raise ValueError('corrected oracle or raw model answer differs from the independent source date union')
    if ({key: value for key, value in corrected_expected.items() if key != 'recommendationDates'} !=
        {key: value for key, value in original.items() if key != 'recommendationDates'}):
        raise ValueError('corrected oracle changes fields other than the reviewed date-union field')
    policy_binding = proof.get('policy_binding', {})
    policy_path = Path(policy_binding.get('policy_path', '')).resolve(strict=True)
    policy_bytes = policy_path.read_bytes()
    policy_text = policy_binding.get('policy_text', '')
    public_policy = policy_bytes.decode('utf-8')
    embedded_policy = program.get('semantics', {}).get('files', {}).get('reviewEstablishment/recommendReview/chooseNextQuestion.nl')
    if (_sha(policy_bytes) != policy_binding.get('policy_sha256') or not policy_text or policy_text not in public_policy or
        embedded_policy != public_policy):
        raise ValueError('public operation policy source differs from the independently reviewed proof')
    if program['semantics'].get('oracle') != 'exact':
        raise ValueError('source task is not exact-answer graded')
    reference = program.get('curriculum', {}).get('reference', {}).get('root')
    if not isinstance(reference, list) or len(reference) != 1 or reference[0][0] != 'return_result' or reference[0][1].get('value') != original:
        raise ValueError('historical reference answer differs from the original fixture')

    derived = copy.deepcopy(raw)
    derived_program = derived['task']['program_ir']
    derived_program['semantics']['expected'] = copy.deepcopy(corrected_expected)
    derived_program['curriculum']['reference']['root'][0][1]['value'] = copy.deepcopy(corrected_expected)
    regrade_id = 'inspection-review-date-union/1:' + old['raw_result_sha256']
    derived['historical_outcome'] = copy.deepcopy(outcome)
    derived['oracle_regrade'] = {
        'schema': 'natlang.oracle-regrade-derived-row/1', 'regrade_id': regrade_id,
        'proof_path': str(proof_path.resolve()), 'proof_sha256': _sha(proof_bytes),
        'raw_result_path': str(raw_result_path.resolve()), 'raw_result_sha256': _sha(raw_bytes),
        'historical_expected_sha256': old['expected_sha256'],
        'corrected_expected_sha256': corrected['expected_sha256'],
        'current_oracle_version': corrected['version'],
        'current_oracle_status': 'accepted-by-independent-regrade',
        'admission_decision': 'pending-native-graph-conversion',
    }
    new_outcome = copy.deepcopy(outcome)
    new_outcome['accepted'] = True
    new_outcome['oracle'] = {**new_outcome['oracle'], 'accepted': True, 'verification_epoch': corrected['version']}
    new_outcome['checks'] = {**new_outcome['checks'], 'answer': True}
    new_outcome['rejection_reasons'] = []
    derived['outcome'] = new_outcome
    receipt = {
        'schema': 'natlang.oracle-regrade-conversion-receipt/1', 'regrade_id': regrade_id,
        'raw_result_sha256': _sha(raw_bytes), 'source_case_row_sha256': binding['row_sha256'],
        'proof_sha256': _sha(proof_bytes), 'historical_outcome_preserved': derived['historical_outcome'],
        'current_outcome': {'accepted': True, 'oracle_version': corrected['version'],
                            'answer_sha256': corrected['expected_sha256']},
        'current_admission_decision': 'pending-native-graph-conversion',
    }
    return derived, receipt


def qualify_converted_regrade(*, raw_result_path: Path, case_path: Path, proof_path: Path,
                              expected_proof_sha256: str, derived_path: Path,
                              conversion_receipt_path: Path, native_turns_path: Path) -> dict[str, Any]:
    """Issue a separate current admission receipt after strict regrade and exact graph conversion.

    This never edits the original result, derived row, or historical outcome. The regrade proof is rerun
    from its source artifacts, and each converted turn must point back to this derived trajectory and
    carry the original BIRD train custody.
    """
    derived_bytes = derived_path.read_bytes()
    derived_lines = [line for line in derived_bytes.splitlines() if line.strip()]
    if len(derived_lines) != 1:
        raise ValueError('derived regrade artifact must contain exactly one trajectory row')
    derived = json.loads(derived_lines[0])
    expected_derived, conversion_receipt = regrade(
        raw_result_path=raw_result_path, case_path=case_path, proof_path=proof_path,
        expected_proof_sha256=expected_proof_sha256,
    )
    if derived != expected_derived:
        raise ValueError('derived row differs from the independently recomputed regrade')
    conversion_receipt_disk = _load(conversion_receipt_path)
    for key in ('schema', 'regrade_id', 'raw_result_sha256', 'source_case_row_sha256', 'proof_sha256',
                'historical_outcome_preserved', 'current_outcome'):
        if conversion_receipt_disk.get(key) != conversion_receipt.get(key):
            raise ValueError(f'conversion receipt binding differs for {key}')
    if conversion_receipt_disk.get('current_admission_decision') != 'pending-native-graph-conversion':
        raise ValueError('conversion receipt has an unexpected admission status')
    turns_bytes = native_turns_path.read_bytes()
    turns = [json.loads(line) for line in turns_bytes.splitlines() if line.strip()]
    expected_ids = [f"{derived['id']}:decision:{index:04d}" for index in range(13)]
    if [turn.get('id') for turn in turns] != expected_ids:
        raise ValueError('converted turn sequence differs from the reviewed 13-turn trajectory')
    required_groups = {'sql-db:bird:food_inspection', 'ext-unit:bird:db:food_inspection'}
    for turn in turns:
        if (turn.get('split') != 'train' or not required_groups <= set(turn.get('source_groups', [])) or
            turn.get('teacher_trajectory_id') != derived.get('id') or
            turn.get('training_admission', {}).get('approved') is not True or
            turn.get('trace_admission', {}).get('admitted') is not True or
            turn.get('outcome', {}).get('accepted') is not True or
            turn.get('outcome', {}).get('oracle', {}).get('verification_epoch') != 'inspection-review-date-union/1'):
            raise ValueError('converted turn is not an accepted exact train turn bound to the corrected oracle')
        if turn.get('source_ref', {}).get('trajectory_id') != derived.get('id'):
            raise ValueError('converted turn source reference differs from its derived trajectory')
    return {
        'schema': 'natlang.oracle-regrade-current-admission/1',
        'regrade_id': conversion_receipt['regrade_id'],
        'current_admission_decision': 'eligible-for-source-admission',
        'historical_outcome_preserved': derived.get('historical_outcome'),
        'bindings': {
            'raw_result_sha256': _sha(raw_result_path.read_bytes()),
            'source_case_file_sha256': _sha(case_path.read_bytes()),
            'source_case_row_sha256': conversion_receipt['source_case_row_sha256'],
            'proof_sha256': _sha(proof_path.read_bytes()),
            'derived_row_file_sha256': _sha(derived_bytes),
            'conversion_receipt_sha256': _sha(conversion_receipt_path.read_bytes()),
            'native_turns_file_sha256': _sha(turns_bytes),
        },
        'turn_count': len(turns),
        'turn_ids': expected_ids,
        'source_groups': sorted(required_groups),
        'oracle_version': 'inspection-review-date-union/1',
        'actor': {'model': 'gpt-6-luna', 'provider': 'openai-codex', 'transport': 'pi-provider',
                  'collection_role': 'teacher'},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-result', type=Path, required=True)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--proof', type=Path, required=True)
    parser.add_argument('--proof-sha256', required=True,
                        help='digest independently bound by the reviewed regrade manifest')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--qualify-derived', action='store_true',
                        help='also require converted native turns and write a separate current admission receipt')
    parser.add_argument('--conversion-receipt', type=Path)
    parser.add_argument('--native-turns', type=Path)
    parser.add_argument('--admission-receipt', type=Path)
    args = parser.parse_args()
    derived, receipt = regrade(raw_result_path=args.raw_result, case_path=args.case,
                               proof_path=args.proof, expected_proof_sha256=args.proof_sha256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(derived, ensure_ascii=False) + '\n')
    with args.receipt.open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2); stream.write('\n')
    if args.qualify_derived:
        if not (args.conversion_receipt and args.native_turns and args.admission_receipt):
            raise SystemExit('--qualify-derived requires conversion receipt, native turns, and admission receipt paths')
        admission = qualify_converted_regrade(
            raw_result_path=args.raw_result, case_path=args.case, proof_path=args.proof,
            expected_proof_sha256=args.proof_sha256, derived_path=args.output,
            conversion_receipt_path=args.conversion_receipt, native_turns_path=args.native_turns)
        with args.admission_receipt.open('x', encoding='utf-8') as stream:
            json.dump(admission, stream, ensure_ascii=False, indent=2); stream.write('\n')
    print(json.dumps({'regrade_id': receipt['regrade_id'], 'status': 'verified-derived-row-not-yet-admitted',
                      'output': str(args.output.resolve()), 'receipt': str(args.receipt.resolve())}, sort_keys=True))


if __name__ == '__main__':
    main()
