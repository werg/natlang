#!/usr/bin/env python3
"""Verify registered preparation artifacts; never infer training admission."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

def inventory(root, registry):
    result = {'schema': 'natlang.self-improvement-task-inventory/1', 'corpora': [], 'backlog': registry['backlog'], 'errors': []}
    for row in registry['corpora']:
        report = {'id': row['id'], 'state': row['state']}
        try:
            body = (root / row['tasks']).read_bytes()
            manifest = json.loads((root / row['manifest']).read_text())
            actual = hashlib.sha256(body).hexdigest()
            manifest_hash = manifest.get('sha256', manifest.get('outputs',manifest.get('artifacts',{})).get(Path(row['tasks']).name, {}).get('sha256'))
            if actual != row['sha256'] or actual != manifest_hash:
                raise ValueError('task hash differs from registry/manifest')
            rows = [json.loads(line) for line in body.split(b'\n') if line.strip()]
            declared_episodes = manifest.get('episodes', manifest.get('audit', {}).get('episodes'))
            expected_count = declared_episodes if declared_episodes is not None else manifest.get('tasks', manifest.get('task_count', manifest.get('counts', {}).get('candidate_rows')))
            if len(rows) != expected_count:
                raise ValueError('manifest task/episode count differs')
            actual_cases = sum(len(packet.get(role, {}).get('cases', [])) for packet in rows for role in ['support', 'query', 'transfer']) if declared_episodes is not None else len(rows)
            declared_cases = manifest.get('cases', manifest.get('candidate_rows'))
            if declared_cases is not None and actual_cases != declared_cases:
                raise ValueError('manifest case count differs')
            report.update(sha256=actual, episodes=declared_episodes or 0, cases=actual_cases)
            if row.get('audit'):
                audit = json.loads((root / row['audit']).read_text())
                audit_hash = audit.get('input_sha256')
                if audit_hash is None:
                    packets = audit.get('packets', [])
                    if len(packets) != 1:
                        raise ValueError('audit must bind exactly one packet')
                    audit_hash = packets[0].get('sha256')
                if audit_hash != actual or audit.get('errors') or audit.get('passed') is False:
                    raise ValueError('audit has errors or different input hash')
                report['audit'] = 'passed'
            else:
                report['audit'] = 'executor_pending'
        except (OSError, ValueError, KeyError) as error:
            report['state'] = 'held_artifact_unavailable_or_invalid'
            report['error'] = str(error)
            result['errors'].append({'id': row['id'], 'error': str(error)})
        result['corpora'].append(report)
    active = [report for report in result['corpora'] if not report['state'].startswith(('excluded_', 'held_', 'source_backing_'))]
    result['totals'] = {'active_prepared_episodes': sum(report.get('episodes', 0) for report in active),
        'audited_problem_instances': sum(report.get('cases', 0) for report in active if report['audit'] == 'passed'),
        'executor_pending_source_tasks': sum(report.get('cases', 0) for report in active if report['audit'] == 'executor_pending'),
        'admitted_training_trajectories': None}
    result['held'] = [{'id': report['id'], 'state': report['state'], 'episodes': report.get('episodes', 0), 'cases': report.get('cases', 0)} for report in result['corpora'] if report['state'].startswith('held_')]
    result['collections'] = []
    for row in registry.get('collections', []):
        report = {'id': row['id'], 'declared_state': row['state'], 'publication': 'Candidates only; admission not inferred'}
        try:
            directory = root / row['path']
            identity = json.loads((directory / 'queue.json').read_text())
            if identity['input_sha256'] != row['input_sha256'] or identity['runtime_manifest_sha256'] != row['runtime_manifest_sha256']:
                raise ValueError('collection identity differs from registry')
            ids = set(identity['episode_ids'])
            states = [json.loads(path.read_text()) for path in (directory / 'tasks').glob('*/state.json')]
            if len({state['episode'] for state in states}) != len(states) or any(state['episode'] not in ids for state in states):
                raise ValueError('collection state is duplicated or outside pinned input')
            terminal = [state for state in states if state.get('terminal') is True]
            report.update(episodes=len(ids), terminal=len(terminal), unfinished=len(ids)-len(terminal),
                dispositions=dict(Counter(state.get('disposition', 'unknown') for state in terminal)),
                positive_candidates=sum(state.get('positive') is True for state in terminal))
        except (OSError, ValueError, KeyError) as error:
            report['observed_state'] = 'unavailable_or_invalid'
            report['error'] = str(error)
        result['collections'].append(report)
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--registry', default='training/self_improvement_tasks.json')
    args = parser.parse_args()
    result = inventory(args.root, json.loads((args.root / args.registry).read_text()))
    print(json.dumps(result, indent=2))
    raise SystemExit(bool(result['errors']))
