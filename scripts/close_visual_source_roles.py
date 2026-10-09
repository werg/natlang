#!/usr/bin/env python3
"""Close a review-only visual role proposal over every declared overlap edge.

Produces no native episode, registration, model call or admission. Unsupported
members stay held within their component's role instead of discarding good peers.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'training' / 'neuralese'))
from natlang_neuralese.common.hashing import sha256_file_hex as sha  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('proposal', 'grouping', 'screen', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    proposal = json.loads(args.proposal.read_text())
    grouping = json.loads(args.grouping.read_text())
    screen = json.loads(args.screen.read_text())
    if screen.get('scorer_unchanged') is not True or screen['remaining']:
        raise ValueError('screen must be complete and scorer-stable')
    if screen['tasks_sha256'] != grouping['source']['tasks_sha256']:
        raise ValueError('screen/grouping source identity mismatch')
    nodes = {r['task_id']: r for r in grouping['rows']}
    measured = {r['id']: r for r in screen['results']}
    if len(nodes) != len(grouping['rows']) or set(nodes) != set(measured):
        raise ValueError('grouping/screen must cover exactly the same unique tasks')
    parent = {n: n for n in nodes}
    roles = {}

    def find(n):
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    def union(a, b):
        parent[find(a)] = find(b)

    edges = []
    for cluster in proposal['cluster_assignments']:
        if cluster['role'] not in ('support', 'query', 'transfer'):
            raise ValueError('unknown role')
        for n in cluster['task_ids']:
            if n not in nodes or n in roles:
                raise ValueError('proposal has unknown or repeated member')
            roles[n] = cluster['role']
            union(cluster['task_ids'][0], n)
    if set(roles) != set(nodes):
        raise ValueError('proposal omits tasks')
    for n, row in nodes.items():
        for other in row['likely_template_or_topic_overlaps']:
            if other not in nodes:
                raise ValueError('overlap references unknown task')
            edges.append((n, other))
            union(n, other)
    components = defaultdict(list)
    for n in sorted(nodes):
        components[find(n)].append(n)
    closed = []
    final_roles = {}
    for members in components.values():
        # Preserve support membership in mixed components; remaining ties use query.
        votes = Counter(roles[n] for n in members)
        role = 'support' if votes['support'] else 'query' if votes['query'] else 'transfer'
        final_roles.update({n: role for n in members})
        rows = []
        for n in members:
            result = measured[n]
            placeholder = nodes[n]['topic_archetype'] == 'restaurant_hospitality_placeholder'
            if placeholder:
                disposition = 'excluded_placeholder_content'
            elif result['status'] != 'measured':
                disposition = 'held_unsupported_reference'
            elif not all(result['baseline']['gates'].values()):
                disposition = 'held_requires_host_repair_proof'
            elif result['baseline']['quality'] < 1:
                disposition = 'candidate_source_headroom_model_baseline_pending'
            else:
                disposition = 'control_no_source_headroom_model_baseline_pending'
            rows.append({'task_id': n, 'title': nodes[n]['title'], 'disposition': disposition,
                         'baseline': result.get('baseline')})
        identity = hashlib.sha256(json.dumps(members, separators=(',', ':')).encode()).hexdigest()
        closed.append({'id': 'websight-component/' + identity, 'role': role,
                       'prior_role_counts': dict(votes), 'task_ids': members, 'members': rows})
    errors = [(a, b) for a, b in edges if final_roles[a] != final_roles[b]]
    if errors:
        raise ValueError('overlap closure failed')
    report = {'schema': 'natlang.visual-browser-closed-role-proposal/1',
              'status': 'review_only_not_registered_not_collectible',
              'inputs': {key: {'path': str(getattr(args, key)), 'sha256': sha(getattr(args, key))}
                         for key in ('proposal', 'grouping', 'screen')},
              'recipe_sha256': sha(Path(__file__)), 'browser_image': screen['image'],
              'scorer_sha256': screen['scorer_sha256'], 'cluster_assignments': closed,
              'task_count': len(nodes), 'component_count': len(closed),
              'role_counts': dict(Counter(final_roles.values())),
              'prior_cross_role_overlap_edges': sum(roles[a] != roles[b] for a, b in edges),
              'cross_role_overlap_edges': len(errors), 'provider_calls': 0,
              'holds': ['paint-design-review', 'host-repair-proofs-for-gate-failures',
                        'native-episode-contract', 'actual-model-headroom', 'training-admission']}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'cluster_assignments'}))


if __name__ == '__main__':
    main()
