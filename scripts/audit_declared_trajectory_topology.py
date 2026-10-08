#!/usr/bin/env python3
"""Compare declared inline/iterate topology with observed call and trace evidence."""
from __future__ import annotations
import argparse, json
from pathlib import Path


def read_json(path: Path):
    return json.loads(path.read_text())


def flatten_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from flatten_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from flatten_strings(item)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True, type=Path)
    ap.add_argument('--result', required=True, action='append', type=Path)
    ap.add_argument('--output', required=True, type=Path)
    args = ap.parse_args()
    cases = {case['id']: case for case in map(json.loads, args.source.read_text().splitlines()) if case}
    reports = []
    for path in args.result:
        result = read_json(path)
        ids = result.get('task', {}).get('source_program_ids', [])
        case = next((cases[i] for i in ids if i in cases), None)
        if case is None:
            raise SystemExit(f'result has no source row: {path}')
        cur = case.get('curriculum', {})
        requires_inline = cur.get('inline') == 'required'
        requires_iterate = cur.get('iterate') == 'required'
        outcome = result.get('outcome', {})
        actual = outcome.get('value')
        expected_text = case.get('semantics', {}).get('expected_files', {}).get('decision.json')
        try:
            expected = json.loads(expected_text) if isinstance(expected_text, str) else None
        except json.JSONDecodeError:
            expected = None
        accepted = expected is not None and actual == expected
        ledger = outcome.get('invocation_ledger', [])
        invocations = {entry.get('invocation_id') for entry in ledger if entry.get('invocation_id')}
        root = next((e.get('invocation_id') for e in result.get('trajectory', []) if e.get('invocation_id')), None)
        child_ids = invocations - ({root} if root else set())
        trace_path = Path(str(path).removesuffix('.result.json') + '.trace.jsonl')
        events = [json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]
        steps = [e for e in events if e.get('kind') == 'iteration_step']
        contract = cur.get('trajectory_contract', {})
        min_child_calls = int(contract.get('min_child_invocations', 1 if requires_inline else 0))
        min_state_edges = int(contract.get('min_state_edges_from_children', 1 if requires_iterate else 0))
        min_iteration_steps = int(contract.get('min_iteration_steps', 1 if requires_iterate else 0))
        state_edge_child_ids = set()
        for event in steps:
            for item in event.get('inputs', []):
                if not isinstance(item, dict) or not str(item.get('port', '')).startswith('state.'):
                    continue
                node = item.get('node', '')
                if not isinstance(node, str):
                    continue
                source_id = node.removeprefix('call:').split('#', 1)[0]
                if source_id in child_ids:
                    state_edge_child_ids.add(source_id)
        referenced_children = sorted(state_edge_child_ids)
        role_counts = {}
        for role in contract.get('required_child_roles', []):
            count = 0
            for entry in ledger:
                if entry.get('invocation_id') not in child_ids:
                    continue
                site = entry.get('inline_instruction_site', {})
                instruction = '\n'.join([site.get('realized_instruction', ''), *site.get('template_segments', [])])
                return_type = site.get('returns', {}).get('natlang') or site.get('returns', {}).get('text', '')
                if role.get('instruction_contains', '') in instruction and (not role.get('return_type_contains') or
                    role['return_type_contains'] in return_type):
                    count += 1
            role_counts[role['role']] = {'observed': count, 'required': role['min_invocations'],
                'qualified': count >= role['min_invocations']}
        roles_ok = all(item['qualified'] for item in role_counts.values())
        topology_ok = len(child_ids) >= min_child_calls and len(state_edge_child_ids) >= min_state_edges and len(steps) >= min_iteration_steps and roles_ok
        reports.append({
            'source_id': case['id'], 'source_sha256': __import__('hashlib').sha256((args.source.read_bytes())).hexdigest(),
            'result_path': str(path), 'result_sha256': __import__('hashlib').sha256(path.read_bytes()).hexdigest(),
            'trace_path': str(trace_path), 'trace_sha256': __import__('hashlib').sha256(trace_path.read_bytes()).hexdigest(),
            'answer_accepted': accepted, 'expected_value': expected, 'actual_value': actual,
            'declared_inline': cur.get('inline'), 'declared_iterate': cur.get('iterate'),
            'observed_invocations': len(invocations), 'observed_child_invocations': len(child_ids),
            'observed_iteration_steps': len(steps), 'state_edges_referencing_child_invocations': referenced_children,
            'topology_contract': contract or {'min_child_invocations': min_child_calls, 'min_state_edges_from_children': min_state_edges, 'min_iteration_steps': min_iteration_steps},
            'observed_child_roles': role_counts,
            'topology_credit': bool(topology_ok),
            'topology_reason': 'observed child calls and iteration state edges meet declared minima' if topology_ok else 'observed evidence is below declared child-call/state-edge/iteration minima',
            'admission': False,
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'version':'declared-trajectory-topology-audit/1','source_path':str(args.source),'source_sha256':__import__('hashlib').sha256(args.source.read_bytes()).hexdigest(),'rows':reports,'training_admission':False},indent=2)+'\n')
    print(json.dumps({'rows':len(reports),'accepted':sum(r['answer_accepted'] for r in reports),'topology_credited':sum(r['topology_credit'] for r in reports),'output':str(args.output)}))

if __name__ == '__main__': main()
