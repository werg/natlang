#!/usr/bin/env python3
"""Verify registered preparation artifacts; never infer training admission."""
import argparse
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
            if actual != row['sha256'] or actual != manifest['sha256']:
                raise ValueError('task hash differs from registry/manifest')
            rows = [json.loads(line) for line in body.splitlines() if line.strip()]
            expected_count = manifest.get('episodes', manifest.get('tasks'))
            if len(rows) != expected_count:
                raise ValueError('manifest task/episode count differs')
            report.update(sha256=actual, episodes=manifest.get('episodes', 0), cases=manifest.get('cases', manifest.get('tasks', 0)))
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
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--registry', default='training/self_improvement_tasks.json')
    args = parser.parse_args()
    result = inventory(args.root, json.loads((args.root / args.registry).read_text()))
    print(json.dumps(result, indent=2))
    raise SystemExit(bool(result['errors']))
