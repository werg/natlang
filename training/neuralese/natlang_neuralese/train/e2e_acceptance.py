"""Acceptance of the end-to-end Neuralese lineage: an exact verifier of what the stage produced.

The requirement (owner decisions 2026-10-11 and 2026-10-12; plans/neuralese/NEURALESE_DIRECT_V1.md "End-to-end
stage"): consumers reading the producer's Neuralese values (V) against the same consumers reading the producer's
tokens (T, the gold crisp text) on the same held producer calls, on loss and on task success, with no drift stop.
Controls: the same consumers reading a shuffled (fixed factual-disjoint donor) block and a zeroed block. The 48-probe
behaviour gate is reported only (owner 2026-10-12, plans/REVIEW_20261012.md item 3).

The recipe declares the margins (``e2e_acceptance``); the trainer (train/trajectories.py ``evaluate_written``) reports
V, T, shuffled and zeroed means on the paired held readers in summary.json ``written-trained``. A criterion whose
measurement the stage did not produce is ``pending`` and keeps the lineage unqualified: missing evidence never passes.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

SCHEMA = 'natlang.neuralese-e2e-certificate/1'
KEYS = {'description', 'paired_reader', 'task_success', 'drift', 'reported'}
READINESS_KEYS = {'description', 'held_ce_plateau', 'no_drift_stop', 'task_success'}
PAIRED = {'max_v_minus_t_ce_nats', 'min_v_margin_over_shuffled_nats', 'min_v_margin_over_zeroed_nats',
          'min_paired_readers'}


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_declaration(declaration):
    if not isinstance(declaration, dict) or set(declaration) - KEYS or not {'paired_reader', 'task_success',
                                                                             'drift'} <= set(declaration):
        raise ValueError('e2e_acceptance is {paired_reader, task_success, drift, reported?, description?}')
    paired = declaration['paired_reader']
    if not isinstance(paired, dict) or set(paired) != PAIRED or not all(_number(v) for v in paired.values()):
        raise ValueError('e2e_acceptance.paired_reader declares ' + ', '.join(sorted(PAIRED)) + ' as numbers')
    task = declaration['task_success']
    if (not isinstance(task, dict) or set(task) != {'oracle', 'max_success_drop_vs_t'} or task['oracle'] != 'exact'
            or not _number(task['max_success_drop_vs_t'])):
        raise ValueError('e2e_acceptance.task_success is {oracle: "exact", max_success_drop_vs_t: number}')
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
    shuffled, zeroed, n = held.get('shuffled'), held.get('zeroed'), held.get('n', 0)

    def criterion(name, measured, passed, **detail):
        criteria[name] = {'status': 'pending' if measured is None else ('pass' if passed else 'fail'),
                          'measured': measured, **detail}

    criterion('paired_readers', n, n >= paired['min_paired_readers'], required=paired['min_paired_readers'])
    criterion('v_minus_t_ce', None if v is None or t is None else v - t,
              v is not None and t is not None and v - t <= paired['max_v_minus_t_ce_nats'],
              max=paired['max_v_minus_t_ce_nats'], v=v, t=t)
    criterion('v_margin_over_shuffled', None if v is None or shuffled is None else shuffled - v,
              v is not None and shuffled is not None and shuffled - v >= paired['min_v_margin_over_shuffled_nats'],
              min=paired['min_v_margin_over_shuffled_nats'])
    criterion('v_margin_over_zeroed', None if v is None or zeroed is None else zeroed - v,
              v is not None and zeroed is not None and zeroed - v >= paired['min_v_margin_over_zeroed_nats'],
              min=paired['min_v_margin_over_zeroed_nats'])
    success = summary.get('task_success') or {}
    v_rate, t_rate = success.get('v_success'), success.get('t_success')
    drop = None if v_rate is None or t_rate is None else t_rate - v_rate
    criterion('task_success', drop, drop is not None and drop <= declaration['task_success']['max_success_drop_vs_t'],
              max_drop=declaration['task_success']['max_success_drop_vs_t'], oracle='exact',
              **({} if success else {'pending_reason': 'no exact-oracle consumer task-success evaluation in the stage '
                                                        'report yet (plans/REVIEW_20261012.md item 1)'}))
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
            'scope': 'consumers reading values against consumers reading tokens (gold T) on the paired held readers, '
                     'exact-oracle task success, no drift stop; behaviour gate report-only (owner 2026-10-12)'}


# Readiness of a direct phase to hand off into the end-to-end lineage (owner 2026-10-12: e2e replaces text-only SFT
# as soon as it is ready, without waiting for the direct phase's full epoch). Declared once in the shared base recipe
# (``parent_readiness``), the same for both lines, and read from the running direct phase's own logs.

def validate_readiness(declaration):
    if not isinstance(declaration, dict) or set(declaration) - READINESS_KEYS or \
            not {'held_ce_plateau', 'no_drift_stop', 'task_success'} <= set(declaration):
        raise ValueError('parent_readiness is {held_ce_plateau, no_drift_stop, task_success, description?}')
    plateau = declaration['held_ce_plateau']
    if (not isinstance(plateau, dict) or set(plateau) != {'evaluations', 'max_relative_improvement'} or
            type(plateau['evaluations']) is not int or plateau['evaluations'] < 1 or
            not _number(plateau['max_relative_improvement'])):
        raise ValueError('held_ce_plateau is {evaluations: int >= 1, max_relative_improvement: number}')
    if declaration['no_drift_stop'] is not True:
        raise ValueError('parent_readiness.no_drift_stop is true')
    task = declaration['task_success']
    if (not isinstance(task, dict) or set(task) != {'min_vs_base', 'until_measured'} or
            not _number(task['min_vs_base']) or task['until_measured'] not in ('blocks', 'reported')):
        raise ValueError('parent_readiness.task_success is {min_vs_base: number, until_measured: blocks|reported}')


def _jsonl(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []


def readiness(declaration, direct_stage):
    """Whether the direct phase in ``direct_stage`` (its eval.jsonl, drift.jsonl) is ready to hand off.

    Held CE plateau: over the last ``evaluations`` held evaluations on one held set (a data continuation re-measures
    the base on a new held set; only evaluations sharing the latest base_ce are compared), the relative held CE
    improvement from the first to the last is at most ``max_relative_improvement``. No drift stop: the latest drift
    record's guard is not ``stop``. Task success: the direct phase's held task-success rate (eval ``task_success``
    {rate, base_rate}) at least base + ``min_vs_base``; until the phase measures it, ``until_measured`` says whether
    its absence blocks or is reported."""
    validate_readiness(declaration)
    evals = [e for e in _jsonl(Path(direct_stage) / 'eval.jsonl') if _number(e.get('ce'))]
    criteria = {}
    plateau = declaration['held_ce_plateau']
    same_set = [e for e in evals if evals and e.get('base_ce') == evals[-1].get('base_ce')]
    window = same_set[-(plateau['evaluations'] + 1):]
    if len(window) < plateau['evaluations'] + 1:
        criteria['held_ce_plateau'] = {'status': 'pending', 'evaluations_on_held_set': len(same_set),
                                       'needed': plateau['evaluations'] + 1}
    else:
        first, last = window[0]['ce'], window[-1]['ce']
        improvement = (first - last) / first if first else 0.0
        criteria['held_ce_plateau'] = {'status': 'pass' if improvement <= plateau['max_relative_improvement'] else
                                       'fail', 'relative_improvement': improvement,
                                       'max': plateau['max_relative_improvement'],
                                       'steps': [e.get('step') for e in window], 'ce': [e['ce'] for e in window]}
    drift = _jsonl(Path(direct_stage) / 'drift.jsonl')
    guard = ((drift[-1].get('behaviour_drift') or {}).get('guard') if drift else None)
    criteria['no_drift_stop'] = {'status': 'pending' if not drift else ('fail' if guard == 'stop' else 'pass'),
                                 'guard': guard, 'step': drift[-1].get('step') if drift else None}
    task = declaration['task_success']
    measured = (evals[-1].get('task_success') or {}) if evals else {}
    if _number(measured.get('rate')) and _number(measured.get('base_rate')):
        criteria['task_success'] = {'status': 'pass' if measured['rate'] >= measured['base_rate'] + task['min_vs_base']
                                    else 'fail', **measured, 'min_vs_base': task['min_vs_base']}
    else:
        criteria['task_success'] = {'status': 'pending' if task['until_measured'] == 'blocks' else 'reported',
                                    'reason': 'the direct phase reports no held task-success rate yet '
                                              '(plans/REVIEW_20261012.md item 1)'}
    ready = all(c['status'] in ('pass', 'reported') for c in criteria.values())
    return {'schema': 'natlang.neuralese-e2e-readiness/1', 'ready': ready, 'direct_stage': str(direct_stage),
            'latest_step': evals[-1].get('step') if evals else None, 'criteria': criteria}


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description='End-to-end lineage: readiness of a direct phase to hand off '
                                                 '(parent_readiness) or acceptance of a finished e2e stage.')
    sub = parser.add_subparsers(dest='command', required=True)
    ready = sub.add_parser('readiness')
    ready.add_argument('--recipe', type=Path, required=True, help='an e2e recipe (base or line)')
    ready.add_argument('--direct-stage', type=Path, required=True, help='the running direct phase directory')
    accept = sub.add_parser('acceptance')
    accept.add_argument('--recipe', type=Path, required=True)
    accept.add_argument('--summary', type=Path, required=True, help='the e2e stage summary.json')
    args = parser.parse_args(argv)
    from .recipe import resolve_recipe_data
    recipe = resolve_recipe_data(args.recipe)
    if args.command == 'readiness':
        result = readiness(recipe['parent_readiness'], args.direct_stage)
    else:
        criteria, reported = evaluate(recipe['e2e_acceptance'], json.loads(args.summary.read_text()))
        result = {'qualified': all(c['status'] == 'pass' for c in criteria.values()), 'criteria': criteria}
    print(json.dumps(result, indent=1))
    return 0 if result.get('ready', result.get('qualified')) else 1


if __name__ == '__main__':
    raise SystemExit(main())
