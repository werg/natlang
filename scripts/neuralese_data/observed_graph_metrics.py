"""Descriptive telemetry of retained invocation graphs; never an admission criterion."""
from collections import Counter
import posixpath
import re


def graph_metrics(records):
    nodes, duplicates = {}, 0
    for record in records:
        ident = record.get('call_id')
        if not ident:
            continue
        if ident in nodes:
            duplicates += 1
        else:
            nodes[ident] = record
    def function(record):
        definition = record.get('definition') or {}
        source = definition.get('source') or '(unknown-source)'
        root = record.get('program_root') or ''
        path = source if source.startswith(('skills/', 'packages/', '.mounts/')) else posixpath.join(root, source)
        return path + '#' + (definition.get('name') or '(unknown-function)')
    paths = {ident: function(record) for ident, record in nodes.items()}
    def domain(path):
        namespaces = re.findall(r'(?:^|/)skills/([^/]+)', path.split('#')[0])
        return namespaces[-1] if namespaces else '(unscoped)'
    domains = {ident: domain(path) for ident, path in paths.items()}
    branches, edges, composition = Counter(), Counter(), Counter(domains.values())
    orphans = []
    for ident, record in nodes.items():
        parent = record.get('parent_call_id')
        if parent in nodes:
            branches[parent] += 1
            edges[(paths[parent], paths[ident])] += 1
        elif parent is not None:
            orphans.append(ident)
    depths, cycles = {}, set()
    for ident in nodes:
        trail, current = [], ident
        while current in nodes and current not in depths and current not in trail:
            trail.append(current)
            current = nodes[current].get('parent_call_id')
        if current in trail:
            cycles.update(trail[trail.index(current):])
            continue
        if current in nodes:
            depth = depths[current]
        elif current is None:
            depth = -1
        else:
            continue  # Missing ancestry cannot establish rooted depth.
        for child in reversed(trail):
            depth += 1
            depths[child] = depth
    return {'schema': 'natlang.observed-graph-metrics/1', 'invocations': len(nodes),
            'roots': sum(row.get('parent_call_id') is None for row in nodes.values()),
            'max_depth': max(depths.values(), default=None), 'depth_convention': 'root=0; unresolved ancestry excluded',
            'max_branching': max(branches.values(), default=0), 'unique_function_paths': sorted(set(paths.values())),
            'parent_child_edges': [{'parent': parent, 'child': child, 'count': count}
                                   for (parent, child), count in sorted(edges.items())],
            'domain_composition': dict(sorted(composition.items())),
            'domain_convention': 'observed skill source namespace, not inferred task semantics',
            'cross_domain_edges': sum(count for (parent, child), count in edges.items() if domain(parent) != domain(child)),
            'anomalies': {'duplicate_call_ids': duplicates, 'orphan_call_ids': sorted(orphans),
                          'cycle_call_ids': sorted(cycles), 'unresolved_depth_calls': len(nodes) - len(depths)}}


def aggregate_graph_metrics(captures):
    """Keep preserved failures separate from completed/partial supervised captures."""
    grouped = {}
    for bucket in ('completed', 'partial', 'preserved_failure'):
        rows = [row['metrics'] for row in captures if row['category'] == bucket]
        paths, edges, domains = set(), Counter(), Counter()
        for row in rows:
            paths.update(row['unique_function_paths'])
            domains.update(row['domain_composition'])
            for edge in row['parent_child_edges']:
                edges[edge['parent'], edge['child']] += edge['count']
        depths = [row['max_depth'] for row in rows if row['max_depth'] is not None]
        grouped[bucket] = {'captures': len(rows), 'invocations': sum(row['invocations'] for row in rows),
            'max_depth': max(depths, default=None), 'max_branching': max((row['max_branching'] for row in rows), default=0),
            'unique_function_paths': sorted(paths), 'domain_composition': dict(sorted(domains.items())),
            'parent_child_edges': [{'parent': parent, 'child': child, 'count': count} for (parent, child), count in sorted(edges.items())],
            'captures_without_graph': sum(not row['invocations'] for row in rows),
            'captures_with_graph_anomalies': sum(bool(row['anomalies']['duplicate_call_ids'] or row['anomalies']['unresolved_depth_calls']) for row in rows)}
    return {'schema': 'natlang.observed-graph-summary/1', 'categories': grouped}
