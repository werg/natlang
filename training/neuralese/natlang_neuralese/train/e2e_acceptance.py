"""Acceptance of the end-to-end Neuralese lineage: an exact verifier of what the stage produced.

The requirement (owner decisions 2026-10-11 and 2026-10-12; plans/neuralese/NEURALESE_DIRECT_V1.md "End-to-end
stage"): consumers reading the producer's Neuralese values (V) against the same consumers reading the producer's
tokens on the same held producer calls, on loss and on task success, with no drift stop. On loss the tokens are the
producer's actual emitted tokens on the identical producer states and lengths: V <= that matched token reader
within the declared margin (``max_v_minus_t_ce_nats``). Independently generated T_self (greedy and sampled at 0.6,
train/self_text.py) and gold crisp text are reported separately. Forced-history loss does not certify autonomous
generation or stopping; the task criteria below still require real execution.
Controls: the same consumers reading a shuffled (fixed factual-disjoint donor) block and a zeroed block. The 48-probe
behaviour gate is reported only (owner 2026-10-12, plans/REVIEW_20261012.md item 3).

The recipe declares the margins (``e2e_acceptance``); the trainer (train/trajectories.py ``evaluate_written``) reports
V, T_self (greedy, t0.6), gold T, shuffled and zeroed means on the paired held readers in summary.json ``written-trained``. A criterion whose
measurement the stage did not produce is ``pending`` and keeps the lineage unqualified: missing evidence never passes.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

SCHEMA = 'natlang.neuralese-e2e-certificate/1'
KEYS = {'description', 'paired_reader', 'task_success', 'drift', 'reported'}
PAIRED = {'max_v_minus_t_ce_nats', 'min_v_margin_over_shuffled_nats', 'min_v_margin_over_zeroed_nats',
          'min_paired_readers'}


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _same_task_cases(left, right):
    """Exact completed task reports of one packet and case sequence."""
    indices = left.get('case_indices')
    packet = left.get('packet_sha256')
    return (left.get('eligible') is True and right.get('eligible') is True
            and left.get('complete') is True and right.get('complete') is True
            and isinstance(packet, str) and len(packet) == 64 and packet == right.get('packet_sha256')
            and isinstance(indices, list) and bool(indices)
            and all(type(index) is int for index in indices) and len(set(indices)) == len(indices)
            and indices == right.get('case_indices')
            and left.get('cases') == right.get('cases') == len(indices))


def validate_declaration(declaration):
    if not isinstance(declaration, dict) or set(declaration) - KEYS or not {'paired_reader', 'task_success',
                                                                             'drift'} <= set(declaration):
        raise ValueError('e2e_acceptance is {paired_reader, task_success, drift, reported?, description?}')
    paired = declaration['paired_reader']
    if not isinstance(paired, dict) or set(paired) != PAIRED or not all(_number(v) for v in paired.values()):
        raise ValueError('e2e_acceptance.paired_reader declares ' + ', '.join(sorted(PAIRED)) + ' as numbers')
    task = declaration['task_success']
    if (not isinstance(task, dict) or set(task) != {'oracle', 'max_success_drop_vs_t', 'max_success_drop_vs_parent'}
            or task['oracle'] != 'exact' or not _number(task['max_success_drop_vs_t'])
            or not _number(task['max_success_drop_vs_parent'])):
        raise ValueError('e2e_acceptance.task_success is {oracle: "exact", max_success_drop_vs_t: number, '
                         'max_success_drop_vs_parent: number}')
    if declaration['drift'] != {'no_drift_stop': True}:
        raise ValueError('e2e_acceptance.drift is {"no_drift_stop": true}')
    reported = declaration.get('reported', [])
    if not isinstance(reported, list) or any(not isinstance(item, str) for item in reported):
        raise ValueError('e2e_acceptance.reported lists report-only measurements by name')


def evaluate(declaration, summary):
    """Criteria results for one stage summary (summary.json of train/trajectories.py)."""
    validate_declaration(declaration)
    criteria = {}
    paired, held = declaration['paired_reader'], summary.get('written-trained') or {}
    v, t = held.get('written'), held.get('crisp')
    # Free T_self has a different autoregressive history from the sized, gold-feedback
    # value writer. Only the exact emitted-token trace isolates representation loss.
    matched_complete = (held.get('token_reference') ==
                        'emitted-token-embeddings-on-identical-producer-states-and-lengths'
                        and held.get('n_token_matched') == held.get('n') and held.get('n', 0) > 0
                        and not held.get('token_trace_errors'))
    v_minus_t_self = held.get('v_minus_t_token_matched') if matched_complete else None
    shuffled, zeroed, n = held.get('shuffled'), held.get('zeroed'), held.get('n', 0)

    def criterion(name, measured, passed, **detail):
        criteria[name] = {'status': 'pending' if measured is None else ('pass' if passed else 'fail'),
                          'measured': measured, **detail}

    criterion('paired_readers', n, n >= paired['min_paired_readers'], required=paired['min_paired_readers'])
    criterion('v_minus_t_ce', v_minus_t_self,
              v_minus_t_self is not None and v_minus_t_self <= paired['max_v_minus_t_ce_nats'],
              max=paired['max_v_minus_t_ce_nats'], v=v, t_self=held.get('crisp_token_matched'),
              token_reference=held.get('token_reference'), n_token_matched=held.get('n_token_matched'),
              independent_self_greedy=held.get('crisp_self_greedy'),
              n_self=held.get('n_self'), t_self_t06=held.get('crisp_self_t06'),
              v_minus_t_self_t06=held.get('v_minus_t_self_t06'), t_gold=t,
              v_minus_t_gold=None if v is None or t is None else v - t)
    criterion('v_margin_over_shuffled', None if v is None or shuffled is None else shuffled - v,
              v is not None and shuffled is not None and shuffled - v >= paired['min_v_margin_over_shuffled_nats'],
              min=paired['min_v_margin_over_shuffled_nats'])
    criterion('v_margin_over_zeroed', None if v is None or zeroed is None else zeroed - v,
              v is not None and zeroed is not None and zeroed - v >= paired['min_v_margin_over_zeroed_nats'],
              min=paired['min_v_margin_over_zeroed_nats'])
    # Task success (eval/task_cadence.py, the trainer's in-run evaluation, exact host checks). V against T: the same
    # cases run twice on the final weights, once with fused producer->consumer hand-offs carried as Neuralese blocks
    # (``task_eval_neuralese``) and once as text (``task_eval``); only cases where a hand-off was fused count.
    task = declaration['task_success']
    t_arm, v_arm = summary.get('task_eval') or {}, summary.get('task_eval_neuralese') or {}
    fused = v_arm.get('fused_cases')
    v_ok, t_ok = v_arm.get('fused_case_successes') or {}, t_arm.get('case_successes') or {}
    fused_complete = (_same_task_cases(t_arm, v_arm) and isinstance(fused, list) and bool(fused)
                      and all(type(c) is int for c in fused) and len(set(fused)) == len(fused)
                      and set(fused) <= set(t_arm['case_indices'])
                      and isinstance(v_ok, dict) and isinstance(t_ok, dict)
                      and all(type(v_ok.get(str(c))) is bool and type(t_ok.get(str(c))) is bool for c in fused))
    if fused_complete:
        v_rate = sum(v_ok[str(c)] for c in fused) / len(fused)
        t_rate = sum(t_ok[str(c)] for c in fused) / len(fused)
        criterion('task_success', t_rate - v_rate, t_rate - v_rate <= task['max_success_drop_vs_t'],
                  max_drop=task['max_success_drop_vs_t'], oracle='exact', fused_cases=len(fused), v=v_rate, t=t_rate)
    else:
        criterion('task_success', None, False, max_drop=task['max_success_drop_vs_t'], oracle='exact',
                  pending_reason='no complete comparable V/T task evidence with explicit fused-case outcomes: '
                                 'the held task subset routes every '
                                 'child result to crisp root code, so no model reads another call\'s value; it needs '
                                 'held tasks with model-to-model hand-offs and the collector carrying fused edges as '
                                 'Neuralese (runtime/fusion.ts) against the stage\'s server')
    # The same cases on the parent's weights (``task_eval-init``) against the final weights: e2e training must not cost
    # task success on the runtime's text transport.
    parent = summary.get('task_eval-init') or {}
    measured = (_same_task_cases(t_arm, parent)
                and all(_number(arm.get('task_success_rate')) and 0 <= arm['task_success_rate'] <= 1
                        for arm in (t_arm, parent)))
    drop = parent['task_success_rate'] - t_arm['task_success_rate'] if measured else None
    criterion('task_success_vs_parent', drop, measured and drop <= task['max_success_drop_vs_parent'],
              max_drop=task['max_success_drop_vs_parent'],
              parent=parent.get('task_successes'), final=t_arm.get('task_successes'), cases=t_arm.get('cases'))
    guard = summary.get('drift_guard')
    criterion('no_drift_stop', None if guard is None else not guard.get('stopped', True),
              guard is not None and guard.get('stopped') is False)
    reported = {name: summary.get(name) for name in declaration.get('reported', []) if name in summary}
    return criteria, reported


def certificate(declaration, summary_path, *, recipe_sha256, stage, report_path):
    summary_path = Path(summary_path)
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    criteria, reported = evaluate(declaration, summary)
    from ..common.hashing import sha256_file_hex
    return {'schema': SCHEMA, 'qualified': bool(criteria) and all(c['status'] == 'pass' for c in criteria.values()),
            'criteria': criteria, 'reported': reported, 'recipe_sha256': recipe_sha256, 'stage': stage,
            'summary': str(summary_path), 'summary_sha256': sha256_file_hex(summary_path) if summary else None,
            'stage_report': str(report_path),
            'scope': 'consumers reading values against matched emitted-token embeddings on identical producer '
                     'states and lengths (independent T_self and gold text reported) on the paired held readers, '
                     'exact-oracle task success, no drift stop; behaviour gate report-only (owner 2026-10-12)'}


# The projection on the embedding manifold (owner 2026-10-13): the end-to-end lineage starts from the NeuraleseProjection
# fitted on its exact parent export (train/projection_align.py). The shared e2e recipe declares the thresholds once
# (its direct_runtime stage's ``projection_on_manifold`` parameter, the same on both lines through line parity); the
# direct_runtime stage applies them to the aligned projection's held measurements before any e2e update.
MANIFOLD_KEYS = {'min_nearest_chosen', 'min_cos_chosen', 'max_functional_kl'}


def validate_manifold(declaration):
    if (not isinstance(declaration, dict) or set(declaration) - {'description'} != MANIFOLD_KEYS or
            not all(_number(declaration[k]) for k in MANIFOLD_KEYS)):
        raise ValueError('projection_on_manifold is {' + ', '.join(sorted(MANIFOLD_KEYS)) + ': number, description?}')


def projection_on_manifold(declaration, report):
    """Criteria of the manifold gate on an alignment report (projection_align ``fit``/``verify``): held nearest-token
    accuracy and cosine to E[chosen] over all held positions, and the mean functional KL (the backbone reading a held
    span as P(h) values against E[chosen] embeddings). A measurement the report lacks is pending."""
    validate_manifold(declaration)
    held = ((report or {}).get('held') or {}).get('all') or {}
    kl = ((report or {}).get('functional_kl') or {}).get('all') or {}
    criteria = {}

    def criterion(name, measured, bound, at_least):
        passed = measured is not None and (measured >= bound if at_least else measured <= bound)
        criteria[name] = {'status': 'pending' if measured is None else ('pass' if passed else 'fail'),
                          'measured': measured, ('min' if at_least else 'max'): bound}
    criterion('nearest_chosen', held.get('nearest_chosen'), declaration['min_nearest_chosen'], True)
    criterion('cos_chosen', held.get('cos_chosen'), declaration['min_cos_chosen'], True)
    criterion('functional_kl', kl.get('kl_mean'), declaration['max_functional_kl'], False)
    return criteria


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description='End-to-end lineage: acceptance of a finished e2e stage.')
    sub = parser.add_subparsers(dest='command', required=True)
    accept = sub.add_parser('acceptance')
    accept.add_argument('--recipe', type=Path, required=True)
    accept.add_argument('--summary', type=Path, required=True, help='the e2e stage summary.json')
    args = parser.parse_args(argv)
    from .recipe import resolve_recipe_data
    recipe = resolve_recipe_data(args.recipe)
    criteria, reported = evaluate(recipe['e2e_acceptance'], json.loads(args.summary.read_text()))
    result = {'qualified': all(c['status'] == 'pass' for c in criteria.values()), 'criteria': criteria}
    print(json.dumps(result, indent=1))
    return 0 if result['qualified'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
