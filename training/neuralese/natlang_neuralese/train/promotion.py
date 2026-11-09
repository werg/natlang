"""Task-behaviour promotion of recipe stages and candidate parents (owner 2026-10-29, review 2026-11-03).

A stage, or a candidate parent checkpoint, is promoted on paired protected-development outcomes under identical
invocation limits, never on step counts or training loss. The arms are ordinary held-actor evaluation receipts
(``scripts/neuralese_data/evaluate_native_actor.py`` ``completed.json``, optionally with each run's
``task-score.json`` for per-case pairing). The same small fixed case subset is run once per arm; evaluations stay rare
and small (at most ``MAX_PROMOTION_CASES`` cases, owner 2026-10-30 "frugal eval"). Receipts are recorded evidence, never
training data.

Declaration (a recipe stage's ``promotion`` field)::

    {"schema": "natlang.behavior-promotion-declaration/1",
     "packet": {"path", "sha256"},       registered protected development packet
     "cases": [int, ...],                fixed case indices (<= MAX_PROMOTION_CASES)
     "compare_to": "parent" | "base",    the reference arm
     "min_delta": int >= 0,              candidate successes - reference successes
     "max_case_regressions": int | null, per-case losses allowed (needs task-score detail)
     "confirmation": {"packet", "cases"} | null}   optional second packet: no regression there

``compare`` writes one ``natlang.behavior-promotion/1`` receipt per phase. Arms match only when both evaluated the
declared packet and cases with the same task spec (limits), prompt rendering, serving configuration, precision,
reasoning-history policy, tokenizer and Python runtime; anything else is recorded as an unmatched comparison and
promotes nothing. ``attach`` binds the receipts to a recipe stage (``<out>/<stage>-promotion.json``), which the
recipe runner requires before a dependent stage starts. ``select-parent`` records the behaviour-based choice between
a candidate and its reference (``natlang.behavior-parent-selection/1``) that adaptation stages bind as their backbone.

    python -m natlang_neuralese.train.promotion compare --declaration D --phase development \
        --candidate C/completed.json [--candidate-detail C/task-score.json] \
        --reference R/completed.json [--reference-detail R/task-score.json] --out receipt.json
    python -m natlang_neuralese.train.promotion attach --recipe-out OUT --stage ID --receipt receipt.json [...]
    python -m natlang_neuralese.train.promotion select-parent --stage ID --receipt receipt.json [...] \
        --candidate-path CK --reference-path PARENT --out selection.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..common.hashing import canonical_json_sha256_hex, sha256_file_hex

DECLARATION_SCHEMA = 'natlang.behavior-promotion-declaration/1'
RECEIPT_SCHEMA = 'natlang.behavior-promotion/1'
STAGE_SCHEMA = 'natlang.stage-promotion/1'
SELECTION_SCHEMA = 'natlang.behavior-parent-selection/1'
EVALUATION_SUMMARY_SCHEMA = 'natlang.in-run-task-eval/1'
MAX_PROMOTION_CASES = 32
PHASES = ('development', 'confirmation')
# The arm identity that must agree for a comparison to be paired under identical limits.
MATCHED_FIELDS = ('spec_sha256', 'packet_sha256', 'case_indices', 'prompt_rendering', 'prompt_rendering_identity_sha256',
                  'serving_config_identity_sha256', 'python_runtime_sha256', 'serving_precision_sha256',
                  'history_reasoning_sha256', 'tokenizer_fingerprint')


def _cases(value, where):
    if (not isinstance(value, list) or not value or len(set(value)) != len(value)
            or any(type(item) is not int or item < 0 for item in value)):
        raise ValueError(f'{where} must be distinct nonnegative case indices')
    if len(value) > MAX_PROMOTION_CASES:
        raise ValueError(f'{where} declares {len(value)} cases; promotion evaluations stay small '
                         f'(at most {MAX_PROMOTION_CASES}, owner frugal-eval rule)')
    return list(value)


def _packet(value, where):
    if not isinstance(value, dict) or not isinstance(value.get('path'), str) or not isinstance(value.get('sha256'), str) \
            or len(value['sha256']) != 64:
        raise ValueError(f'{where} needs {{"path", "sha256"}}')
    return {'path': value['path'], 'sha256': value['sha256']}


def parse_declaration(value):
    """The normalized promotion declaration of a stage (raises on an invalid one)."""
    if not isinstance(value, dict) or value.get('schema') != DECLARATION_SCHEMA:
        raise ValueError(f'promotion must be a {DECLARATION_SCHEMA} declaration')
    unknown = set(value) - {'schema', 'packet', 'cases', 'compare_to', 'min_delta', 'max_case_regressions', 'confirmation',
                            'note'}
    if unknown:
        raise ValueError('unknown promotion field(s): ' + ', '.join(sorted(unknown)))
    compare_to = value.get('compare_to', 'parent')
    if compare_to not in ('parent', 'base'):
        raise ValueError('promotion.compare_to is parent or base')
    min_delta = value.get('min_delta', 0)
    if type(min_delta) is not int or min_delta < 0:
        raise ValueError('promotion.min_delta must be a nonnegative integer')
    regressions = value.get('max_case_regressions')
    if regressions is not None and (type(regressions) is not int or regressions < 0):
        raise ValueError('promotion.max_case_regressions must be a nonnegative integer or null')
    confirmation = value.get('confirmation')
    if confirmation is not None:
        if not isinstance(confirmation, dict) or set(confirmation) - {'packet', 'cases'}:
            raise ValueError('promotion.confirmation is {"packet", "cases"} or null')
        confirmation = {'packet': _packet(confirmation.get('packet'), 'promotion.confirmation.packet'),
                        'cases': _cases(confirmation.get('cases'), 'promotion.confirmation.cases')}
    declaration = {'schema': DECLARATION_SCHEMA, 'packet': _packet(value.get('packet'), 'promotion.packet'),
                   'cases': _cases(value.get('cases'), 'promotion.cases'), 'compare_to': compare_to,
                   'min_delta': min_delta, 'max_case_regressions': regressions, 'confirmation': confirmation}
    if confirmation is not None and confirmation['packet']['sha256'] == declaration['packet']['sha256'] \
            and set(confirmation['cases']) & set(declaration['cases']):
        raise ValueError('promotion confirmation cases must be disjoint from the development cases')
    return declaration


def declaration_sha256(declaration):
    return canonical_json_sha256_hex(parse_declaration(declaration))


def _hash_or_none(value):
    return None if value is None else canonical_json_sha256_hex(value)


def arm_identity(receipt):
    """The comparable identity and outcome of one held-actor evaluation receipt (``completed.json``)."""
    contract, info, summary = receipt.get('contract') or {}, receipt.get('actor_info') or {}, receipt.get('summary') or {}
    policy = info.get('rl_policy_binding') or {}
    runtime = policy.get('python_runtime') or {}
    return {
        'checkpoint_sha256': contract.get('checkpoint_sha256'),
        'step': contract.get('step'),
        'status': receipt.get('status'),
        'training_admission': receipt.get('training_admission'),
        'eligible': summary.get('eligible'),
        'schema': summary.get('schema'),
        'task_successes': summary.get('task_successes'),
        'cases': summary.get('cases'),
        'outcome_counts': summary.get('outcome_counts'),
        'spec_sha256': contract.get('spec_sha256'),
        'packet_sha256': summary.get('packet_sha256'),
        'case_indices': summary.get('case_indices'),
        'prompt_rendering': summary.get('prompt_rendering'),
        'prompt_rendering_identity_sha256': info.get('prompt_rendering_identity_sha256'),
        'serving_config_identity_sha256': info.get('serving_config_identity_sha256'),
        'python_runtime_sha256': runtime.get('sha256'),
        'serving_precision_sha256': _hash_or_none(policy.get('serving_precision')),
        'history_reasoning_sha256': _hash_or_none(policy.get('history_reasoning')),
        'tokenizer_fingerprint': (policy.get('native_identity') or {}).get('tokenizer_fingerprint'),
    }


def _execution_problems(candidate, reference):
    """Reasons two arms' recorded execution identities (the full runtime closure, TS task runtime, driver and image they
    were actually launched under: pop_behavior_eval.execution_identity) are not the same measurement environment.
    ``None`` for both means the caller did not supply them (a comparison of receipts alone, as before)."""
    if candidate is None and reference is None:
        return []
    if not isinstance(candidate, dict) or not isinstance(reference, dict):
        return ['execution identity (runtime closure, driver, image) is recorded for only one arm']
    missing = [name for name, value in (('candidate', candidate), ('reference', reference))
               if not isinstance(value.get('identity_sha256'), str) or value.get('verified') is not True]
    if missing:
        return ['execution identity is unverified against the launch record for: ' + ', '.join(missing)]
    if candidate['identity_sha256'] != reference['identity_sha256']:
        fields = sorted(key for key in set(candidate.get('fields') or {}) | set(reference.get('fields') or {})
                        if (candidate.get('fields') or {}).get(key) != (reference.get('fields') or {}).get(key))
        return ['arms were not run under the identical execution environment: ' + (', '.join(fields) or 'identity')]
    return []


def _arm_problems(arm, packet, cases, name):
    problems = []
    if arm['schema'] != EVALUATION_SUMMARY_SCHEMA:
        problems.append(f'{name}: not an {EVALUATION_SUMMARY_SCHEMA} evaluation summary')
    if arm['status'] != 'finished' or arm['eligible'] is not True or type(arm['task_successes']) is not int:
        problems.append(f'{name}: evaluation is not a finished, eligible, scored run')
    if arm['training_admission'] is not False:
        problems.append(f'{name}: evaluation receipt does not declare training_admission false')
    if arm['packet_sha256'] != packet['sha256']:
        problems.append(f'{name}: evaluated another packet')
    if arm['case_indices'] != cases:
        problems.append(f'{name}: evaluated other cases')
    if not isinstance(arm['checkpoint_sha256'], str):
        problems.append(f'{name}: no checkpoint identity')
    return problems


def _detail(score, arm, name):
    """Per-case outcomes from a run's ``task-score.json``, checked against its completed receipt."""
    if score is None or type(arm['task_successes']) is not int:
        # An unscored arm (ineligible run: unresolved/infrastructure cases) has no success count to check its
        # outcome labels against and is already reported unmatched by ``_arm_problems``; its labels are not paired.
        return None
    for key in ('packet_sha256', 'case_indices', 'task_successes', 'cases'):
        if score.get(key) != arm[key]:
            raise ValueError(f'{name} task-score detail does not belong to its completed receipt ({key})')
    rows = score.get('cases_detail')
    if not isinstance(rows, list) or len(rows) != len(arm['case_indices']):
        raise ValueError(f'{name} task-score detail lacks one outcome per case')
    if sum(row.get('outcome') == 'task_success' for row in rows) != arm['task_successes']:
        raise ValueError(f'{name} task-score detail disagrees with its success count')
    return [{'index': index, 'id': row.get('id'), 'success': row.get('outcome') == 'task_success',
             'outcome': row.get('outcome')} for index, row in zip(arm['case_indices'], rows)]


def compare(declaration, *, phase, candidate, reference, candidate_detail=None, reference_detail=None,
            sources=None, candidate_execution=None, reference_execution=None):
    """One ``natlang.behavior-promotion/1`` receipt (pure; never raises on a losing or unmatched comparison)."""
    declaration = parse_declaration(declaration)
    if phase not in PHASES:
        raise ValueError('phase is development or confirmation')
    if phase == 'confirmation' and declaration['confirmation'] is None:
        raise ValueError('this declaration has no confirmation packet')
    target = declaration if phase == 'development' else declaration['confirmation']
    cand, ref = arm_identity(candidate), arm_identity(reference)
    reasons = _arm_problems(cand, target['packet'], target['cases'], 'candidate') + \
        _arm_problems(ref, target['packet'], target['cases'], 'reference')
    mismatched = [field for field in MATCHED_FIELDS if cand[field] is None or cand[field] != ref[field]]
    if mismatched:
        reasons.append('arms were not run under identical limits/runtime: ' + ', '.join(mismatched))
    reasons += _execution_problems(candidate_execution, reference_execution)
    if cand['checkpoint_sha256'] is not None and cand['checkpoint_sha256'] == ref['checkpoint_sha256']:
        reasons.append('candidate and reference are the same checkpoint')
    matched = not reasons
    cand_cases = _detail(candidate_detail, cand, 'candidate')
    ref_cases = _detail(reference_detail, ref, 'reference')
    transitions = None
    if cand_cases is not None and ref_cases is not None:
        if [item['id'] for item in cand_cases] != [item['id'] for item in ref_cases]:
            raise ValueError('candidate and reference task-score details list different case identities')
        transitions = {'gained': [c['id'] for c, r in zip(cand_cases, ref_cases) if c['success'] and not r['success']],
                       'lost': [c['id'] for c, r in zip(cand_cases, ref_cases) if r['success'] and not c['success']],
                       'both': sum(c['success'] and r['success'] for c, r in zip(cand_cases, ref_cases)),
                       'neither': sum(not c['success'] and not r['success'] for c, r in zip(cand_cases, ref_cases))}
    delta = (cand['task_successes'] - ref['task_successes']) if matched else None
    required = declaration['min_delta'] if phase == 'development' else 0
    decision = []
    if matched and delta < required:
        decision.append(f'candidate successes {cand["task_successes"]} below reference {ref["task_successes"]} '
                        f'+ min_delta {required}')
    if phase == 'development' and declaration['max_case_regressions'] is not None:
        if transitions is None:
            decision.append('max_case_regressions needs both arms\' per-case task-score detail')
        elif len(transitions['lost']) > declaration['max_case_regressions']:
            decision.append(f'{len(transitions["lost"])} per-case regressions exceed '
                            f'{declaration["max_case_regressions"]}')
    return {'schema': RECEIPT_SCHEMA, 'phase': phase, 'declaration_sha256': canonical_json_sha256_hex(declaration),
            'compare_to': declaration['compare_to'],
            'candidate': {key: cand[key] for key in ('checkpoint_sha256', 'step', 'task_successes', 'cases', 'outcome_counts')},
            'reference': {key: ref[key] for key in ('checkpoint_sha256', 'step', 'task_successes', 'cases', 'outcome_counts')},
            'matched_identity': {field: cand[field] for field in MATCHED_FIELDS} if matched else None,
            **({'execution_identity_sha256': candidate_execution['identity_sha256']}
               if matched and isinstance(candidate_execution, dict) else {}),
            'matched': matched, 'unmatched_reasons': reasons, 'delta': delta, 'case_transitions': transitions,
            'promoted': matched and not decision, 'decision_reasons': decision,
            **({'sources': sources} if sources else {}),
            'training_admission': False,
            'scope': 'paired protected-development outcomes on a small fixed case subset; evidence only, '
                     'not training data and not a broad capability claim'}


def require_promotion(record, declaration, *, candidate_sha256):
    """Refuse advancement unless ``record`` (``natlang.stage-promotion/1``) promotes ``candidate_sha256`` under
    exactly ``declaration`` in every declared phase."""
    declaration = parse_declaration(declaration)
    digest = canonical_json_sha256_hex(declaration)
    if not isinstance(record, dict) or record.get('schema') != STAGE_SCHEMA:
        raise ValueError('stage promotion record missing or malformed')
    if record.get('candidate_sha256') != candidate_sha256:
        raise ValueError('stage promotion record promotes another checkpoint')
    phases = ('development',) + (('confirmation',) if declaration['confirmation'] is not None else ())
    for phase in phases:
        receipt = (record.get('receipts') or {}).get(phase)
        if not isinstance(receipt, dict) or receipt.get('schema') != RECEIPT_SCHEMA or receipt.get('phase') != phase:
            raise ValueError(f'stage awaits its {phase} behaviour-promotion receipt')
        if receipt.get('declaration_sha256') != digest:
            raise ValueError(f'{phase} promotion receipt was measured under another declaration')
        if (receipt.get('candidate') or {}).get('checkpoint_sha256') != candidate_sha256:
            raise ValueError(f'{phase} promotion receipt measured another candidate')
        if receipt.get('training_admission') is not False:
            raise ValueError('promotion receipts are evidence only')
        if receipt.get('promoted') is not True:
            raise ValueError(f'{phase} behaviour comparison did not promote the candidate: ' +
                             '; '.join((receipt.get('unmatched_reasons') or []) + (receipt.get('decision_reasons') or [])))
    return True


def stage_record_path(out, stage_id):
    return Path(out) / f'{stage_id}-promotion.json'


def require_stage_promotion(out, stage, report):
    """The recipe runner's advancement check: a stage declaring ``promotion`` advances only with its record."""
    if not stage.get('promotion'):
        return None
    path = stage_record_path(out, stage['id'])
    if not path.is_file():
        raise ValueError(f'stage {stage["id"]} awaits its behaviour promotion: run the declared paired development '
                         f'evaluation and `python -m natlang_neuralese.train.promotion attach` ({path})')
    require_promotion(json.loads(path.read_text()), stage['promotion'], candidate_sha256=report['artifact_sha256'])
    return path


def _load(path):
    return json.loads(Path(path).read_text()) if path else None


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + '.pending')
    pending.write_text(json.dumps(value, indent=1, sort_keys=True) + '\n')
    pending.replace(path)


def attach(out, stage_id, receipts):
    """Bind development (and confirmation) receipts of one candidate to a recipe stage."""
    by_phase = {}
    for receipt in receipts:
        if receipt.get('schema') != RECEIPT_SCHEMA or receipt.get('phase') in by_phase:
            raise ValueError('attach takes at most one behaviour-promotion receipt per phase')
        by_phase[receipt['phase']] = receipt
    candidates = {receipt['candidate']['checkpoint_sha256'] for receipt in by_phase.values()}
    if len(candidates) != 1:
        raise ValueError('promotion receipts of one stage must measure one candidate')
    plan = Path(out) / 'recipe-plan.json'
    if plan.is_file():
        recipe = json.loads(plan.read_text())['recipe']
        stage = next((item for item in recipe['stages'] if item['id'] == stage_id), None)
        if stage is None or not stage.get('promotion'):
            raise ValueError(f'{stage_id} is not a stage of this lineage with a promotion declaration')
        digest = declaration_sha256(stage['promotion'])
        if any(receipt['declaration_sha256'] != digest for receipt in by_phase.values()):
            raise ValueError('receipts were measured under another promotion declaration than the stage declares')
    record = {'schema': STAGE_SCHEMA, 'stage': stage_id, 'candidate_sha256': candidates.pop(), 'receipts': by_phase}
    path = stage_record_path(out, stage_id)
    if path.exists() and json.loads(path.read_text()) != record:
        raise ValueError(f'{path} already records another promotion; promotions are immutable evidence')
    _write(path, record)
    return record


def memetic_gate(history, *, candidate, reference, tolerance=0.05):
    """The memetic meta-learning verdict for a candidate parent (rl.memetic_stage.promote ``promotion/history.json``):
    held-family improvement gains of the candidate's supervisor, paired with the reference's on the same held
    programs. The candidate passes when it does not regress the mean held gain by more than ``tolerance``. A declared
    memetic stage without a paired verdict for exactly this candidate/reference pair blocks the candidate."""
    entry = next((item for item in reversed(history or []) if item.get('parent_sha256') == candidate
                  and reference in (item.get('comparisons') or {})), None)
    if entry is None:
        return {'passed': False, 'reason': 'no paired memetic held-program verdict for this candidate and reference'}
    decision = entry['comparisons'][reference]
    delta = decision.get('mean_gain_delta')
    if delta is None:
        return {'passed': False, 'reason': 'memetic verdict shares no held program with the reference', 'entry': entry}
    return {'passed': delta >= -tolerance, 'mean_gain_delta': delta, 'tolerance': tolerance,
            'programs': decision.get('programs'), 'gains': entry.get('gains')}


def select_parent(stage_id, receipts, *, candidate_path=None, reference_path=None, memetic_history=None,
                  memetic_tolerance=0.05):
    """The behaviour-based choice between one candidate and its reference: the candidate only when every phase
    promoted it (and, when the recipe declares memetic meta-learning, its held-program memetic verdict does not
    regress, ``memetic_gate``); otherwise the reference stays the parent. Unmatched comparisons select nothing new."""
    if not receipts or any(receipt.get('schema') != RECEIPT_SCHEMA for receipt in receipts):
        raise ValueError('select-parent needs behaviour-promotion receipts')
    pairs = {(receipt['candidate']['checkpoint_sha256'], receipt['reference']['checkpoint_sha256']) for receipt in receipts}
    if len(pairs) != 1:
        raise ValueError('select-parent compares one candidate with one reference')
    if 'development' not in {receipt['phase'] for receipt in receipts}:
        raise ValueError('select-parent needs the development comparison')
    candidate, reference = pairs.pop()
    promoted = all(receipt['promoted'] for receipt in receipts)
    memetic = (None if memetic_history is None else
               memetic_gate(memetic_history, candidate=candidate, reference=reference, tolerance=memetic_tolerance))
    if memetic is not None and not memetic['passed']:
        promoted = False
    selected = candidate if promoted else reference
    record = {'schema': SELECTION_SCHEMA, 'stage': stage_id, 'selected_sha256': selected,
              'selected_path': candidate_path if promoted else reference_path,
              'candidate_sha256': candidate, 'reference_sha256': reference, 'candidate_promoted': promoted,
              'comparisons': [{key: receipt[key] for key in ('phase', 'declaration_sha256', 'matched', 'delta', 'promoted',
                                                             'unmatched_reasons', 'decision_reasons', 'candidate',
                                                             'reference', 'case_transitions')} for receipt in receipts],
              **({'memetic': memetic} if memetic is not None else {}),
              'held_data_training_admission': False,
              'scope': 'selection by paired protected-development outcomes; not aggregate loss or step count'}
    return record


def require_parent_selection(selection, checkpoint_sha256):
    """An adaptation stage's backbone binding: the selection record must select exactly this checkpoint."""
    if not isinstance(selection, dict) or selection.get('schema') != SELECTION_SCHEMA:
        raise ValueError(f'backbone parent needs a {SELECTION_SCHEMA} record')
    if selection.get('selected_sha256') != checkpoint_sha256:
        raise ValueError('the bound backbone checkpoint is not the behaviour-selected parent')
    if not any(item.get('matched') for item in selection.get('comparisons') or []):
        raise ValueError('parent selection has no matched behaviour comparison')
    return selection


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest='command', required=True)
    one = commands.add_parser('compare')
    one.add_argument('--declaration', required=True, help='promotion declaration JSON file or a recipe stage id with --recipe')
    one.add_argument('--recipe', help='recipe JSON whose stage --declaration names')
    one.add_argument('--phase', choices=PHASES, default='development')
    one.add_argument('--candidate', required=True)
    one.add_argument('--candidate-detail')
    one.add_argument('--reference', required=True)
    one.add_argument('--reference-detail')
    one.add_argument('--out', required=True)
    two = commands.add_parser('attach')
    two.add_argument('--recipe-out', required=True)
    two.add_argument('--stage', required=True)
    two.add_argument('--receipt', action='append', required=True)
    three = commands.add_parser('select-parent')
    three.add_argument('--stage', required=True)
    three.add_argument('--receipt', action='append', required=True)
    three.add_argument('--candidate-path')
    three.add_argument('--reference-path')
    three.add_argument('--out', required=True)
    three.add_argument('--memetic-history', help='rl.memetic_stage promotion/history.json: gates the candidate on its '
                       'paired held-program memetic verdict (required when the recipe declares native_memetic_rl)')
    three.add_argument('--memetic-tolerance', type=float, default=0.05)
    args = parser.parse_args(argv)
    if args.command == 'compare':
        if args.recipe:
            from .recipe import load_recipe
            stage = next((item for item in load_recipe(Path(args.recipe))['stages'] if item['id'] == args.declaration), None)
            if stage is None or not stage.get('promotion'):
                parser.error(f'{args.declaration} is not a stage with a promotion declaration')
            declaration = stage['promotion']
        else:
            declaration = _load(args.declaration)
        sources = {name: {'path': str(Path(path).resolve()), 'sha256': sha256_file_hex(path)}
                   for name, path in (('candidate', args.candidate), ('candidate_detail', args.candidate_detail),
                                      ('reference', args.reference), ('reference_detail', args.reference_detail)) if path}
        receipt = compare(declaration, phase=args.phase, candidate=_load(args.candidate), reference=_load(args.reference),
                          candidate_detail=_load(args.candidate_detail), reference_detail=_load(args.reference_detail),
                          sources=sources)
        _write(args.out, receipt)
        print(json.dumps({'command': 'compare', 'phase': receipt['phase'], 'matched': receipt['matched'],
                          'delta': receipt['delta'], 'promoted': receipt['promoted'],
                          'reasons': receipt['unmatched_reasons'] + receipt['decision_reasons']}))
    elif args.command == 'attach':
        record = attach(args.recipe_out, args.stage, [_load(path) for path in args.receipt])
        print(json.dumps({'command': 'attach', 'stage': record['stage'], 'candidate_sha256': record['candidate_sha256'],
                          'phases': sorted(record['receipts'])}))
    else:
        record = select_parent(args.stage, [_load(path) for path in args.receipt],
                               candidate_path=args.candidate_path, reference_path=args.reference_path,
                               memetic_history=_load(args.memetic_history) if args.memetic_history else None,
                               memetic_tolerance=args.memetic_tolerance)
        _write(args.out, record)
        print(json.dumps({'command': 'select-parent', 'selected_sha256': record['selected_sha256'],
                          'candidate_promoted': record['candidate_promoted']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
