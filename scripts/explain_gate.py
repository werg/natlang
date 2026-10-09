#!/usr/bin/env python3
"""After a gate failed: compute the facts of the failed gate report, then (optionally) ask the advisory explainer.

Plans/FAILURE_EXPLANATION_PROGRAM.md (review item P11). This script never imports the trainers or the evaluation
modules and never changes a gate: it reads a finished report, recomputes the gate's own arithmetic from the numbers in
the report, and fails loudly when its result differs from the report's recorded pass field. The facts go to a fresh
file beside the report; the explanation (written by ts-host/scripts/explain-advisory.mjs gate) goes to another. Nothing
the pipeline acts on reads either.

Supported reports (detected by their fields):
  natlang.neuralese-foundation-control/1   token_aligned_reference_passed, rows with three exact-zero metrics
  self-feedback evaluation                 thresholds + strata + proposed_gate_passed (possibly nested in the report)
  causal feedback bootstrap eval record    strata with agreement/kl + feedback_gate_passed
"""
import argparse, hashlib, json, math, pathlib, subprocess, sys

FACTS_SCHEMA = 'natlang.gate_facts/1'
IDENTITY_METRICS = ('raw_transport_max_abs', 'identity_readback_logits_max_abs', 'full_output_reference_max_abs')


def _entry(stratum, metric, observed, limit, direction):
    """margin is the signed distance past the limit: positive when the gate is violated."""
    margin = (limit - observed) if direction == 'min' else (observed - limit)
    return {'stratum': stratum, 'metric': metric, 'observed': observed, 'limit': limit, 'direction': direction, 'margin': margin}


def _violates(entry):
    return not (entry['observed'] >= entry['limit'] if entry['direction'] == 'min' else entry['observed'] <= entry['limit']) \
        or not math.isfinite(entry['observed'])


def find_self_feedback(report):
    """The dict holding thresholds, strata and proposed_gate_passed, wherever the report nests it."""
    stack = [report]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            if {'thresholds', 'strata', 'proposed_gate_passed'} <= set(item):
                return item
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
    return None


def _identity_facts(report):
    failed, passed_rows, scored = [], [], []
    for index, row in enumerate(report['rows']):
        name = 'row:%d:%s' % (index, str(row.get('source_sha256', ''))[:12])
        bad = False
        for metric in IDENTITY_METRICS:
            entry = _entry(name, metric, float(row[metric]), 0.0, 'max')
            scored.append({'id': name, 'metric': metric, 'observed': entry['observed']})
            if _violates(entry):
                failed.append(entry); bad = True
        if not bad:
            passed_rows.append(name)
    passed = not failed
    return 'natlang.neuralese-foundation-control/1', passed, report['token_aligned_reference_passed'], failed, passed_rows, scored


def _self_feedback_facts(node):
    limits = node['thresholds']
    failed, passed_strata = [], []
    for stratum, value in sorted(node['strata'].items()):
        entries = [_entry(stratum, 'argmax_agreement', value['argmax_agreement'], limits['min_argmax_agreement'], 'min'),
                   _entry(stratum, 'kl_plain_to_projected_nats', value['kl_plain_to_projected_nats'], limits['max_kl_nats'], 'max'),
                   _entry(stratum, 'quality_ce_gap', value['quality_ce_gap'], limits['max_quality_ce_gap_nats'], 'max')]
        bad = [e for e in entries if _violates(e)]
        failed.extend(bad)
        if not bad:
            passed_strata.append(stratum)
    scored = []
    for window in node.get('windows', []):
        if window.get('proposed_gate_passed') is False:
            scored.append({'id': '%s@%s' % (str(window.get('document_sha256', ''))[:12], window.get('offset')),
                           'metric': 'kl_plain_to_projected_nats', 'observed': window['kl_plain_to_projected_nats']})
    return 'natlang.self-feedback-evaluation', not failed, node['proposed_gate_passed'], failed, passed_strata, scored


def _causal_facts(report, agreement_gate, kl_gate):
    failed, passed_strata = [], []
    for stratum, value in sorted(report['strata'].items()):
        entries = [_entry(stratum, 'agreement', value['agreement'], agreement_gate, 'min'),
                   _entry(stratum, 'kl', value['kl'], kl_gate, 'max')]
        bad = [e for e in entries if _violates(e)]
        failed.extend(bad)
        if not bad:
            passed_strata.append(stratum)
    return 'natlang.causal-feedback-bootstrap-eval', not failed, report['feedback_gate_passed'], failed, passed_strata, []


def gate_facts(report, *, agreement_gate=0.9, kl_gate=0.25, run='', checkpoint=None, step=None, earlier=(), worst=5):
    """Facts of a gate report, by the gate's own arithmetic. Raises when the recomputed decision differs from the report's."""
    if 'token_aligned_reference_passed' in report and 'rows' in report:
        found = _identity_facts(report)
    elif find_self_feedback(report) is not None:
        found = _self_feedback_facts(find_self_feedback(report))
    elif 'feedback_gate_passed' in report and isinstance(report.get('strata'), dict):
        found = _causal_facts(report, agreement_gate, kl_gate)
    else:
        raise ValueError('not a known gate report: expected a foundation control, a self-feedback evaluation or a causal bootstrap eval record')
    schema, passed, recorded, failed, passed_strata, scored = found
    if passed != recorded:
        raise ValueError('recomputed gate decision (%s) differs from the report\'s own pass field (%s); not explaining a report whose '
                         'arithmetic this script does not reproduce' % (passed, recorded))
    scored.sort(key=lambda item: -item['observed'])
    return {'schema': schema, 'passed': passed, 'failed': failed, 'passed_strata': passed_strata, 'worst_windows': scored[:worst],
            'context': {'run': run, 'checkpoint': checkpoint, 'step': step if step is not None else report.get('step'),
                        'earlier_reports': [{'path': str(path), 'passed': earlier_passed(path)} for path in earlier]}}


def earlier_passed(path):
    report = json.loads(pathlib.Path(path).read_text())
    for key in ('token_aligned_reference_passed', 'proposed_gate_passed', 'feedback_gate_passed'):
        if key in report:
            return bool(report[key])
    node = find_self_feedback(report)
    return bool(node['proposed_gate_passed']) if node else False


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('report', type=pathlib.Path)
    p.add_argument('--facts-out', type=pathlib.Path, help='default: <report>.facts.json, created fresh')
    p.add_argument('--agreement-gate', type=float, default=0.9, help='causal bootstrap reports only (its default)')
    p.add_argument('--kl-gate', type=float, default=0.25, help='causal bootstrap reports only (its default)')
    p.add_argument('--run', default='')
    p.add_argument('--checkpoint')
    p.add_argument('--earlier', type=pathlib.Path, action='append', default=[], help='earlier reports of the same gate, for context')
    p.add_argument('--explain', action='store_true', help='then run the advisory explainer; the remaining arguments go to ts-host/scripts/explain-advisory.mjs gate')
    a, rest = p.parse_known_args(argv)
    report = json.loads(a.report.read_text())
    facts = gate_facts(report, agreement_gate=a.agreement_gate, kl_gate=a.kl_gate, run=a.run, checkpoint=a.checkpoint, earlier=a.earlier)
    if facts['passed']:
        print(json.dumps({'passed': True, 'explained': False}))
        return 0
    out = a.facts_out or a.report.with_name(a.report.name + '.facts.json')
    document = {'schema': FACTS_SCHEMA, 'report': str(a.report.resolve()), 'report_sha256': hashlib.sha256(a.report.read_bytes()).hexdigest(),
                'facts': facts}
    with out.open('x') as stream:
        json.dump(document, stream, indent=2); stream.write('\n')
    print(json.dumps({'passed': False, 'facts': str(out), 'failed_entries': len(facts['failed'])}))
    if a.explain:
        script = pathlib.Path(__file__).resolve().parents[1] / 'ts-host' / 'scripts' / 'explain-advisory.mjs'
        return subprocess.call(['node', str(script), 'gate', '--facts', str(out), '--report', str(a.report),
                                '--out', str(a.report.with_name(a.report.name + '.explanation.json')), *rest])
    return 0


if __name__ == '__main__':
    sys.exit(main())
