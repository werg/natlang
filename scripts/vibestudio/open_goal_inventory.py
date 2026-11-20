#!/usr/bin/env python3
"""Pin Vibestudio's actual case registry, template capabilities, and skill sources."""
import argparse
import json
from pathlib import Path
import subprocess

AGENTIC_BASELINE_ORIGINS = frozenset({'direct-agent', 'agent-orchestration', 'managed-teacher'})


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True)


def json_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def repository(url):
    return url.removeprefix('git+').removesuffix('.git').lower()


def case_origin(case, registered):
    if registered and registered.get('interaction') == 'open-user-goal':
        return 'open-user-goal'
    if registered or case['panelTeacher']:
        return 'managed-teacher'
    if not case['orchestrated']:
        return 'direct-agent'
    if case['validation'] == 'agent-goal' or case['sendsAgentTurns']:
        return 'agent-orchestration'
    if case['category'] == 'deterministic':
        return 'testkit'
    return 'harness-review'


def inventory(host, templates, catalog, teacher_registry):
    import yaml

    host_pin = git(host, 'rev-parse', 'HEAD').strip()
    registry = json.loads(git(host, 'show', f'{host_pin}:templates/registry.json'))['templates']
    templates_by_repo = {repository(row['url']): row['id'] for row in registry}
    dependencies, files, directories, pins = {}, {}, {}, {'host': host_pin}
    for row in registry:
        name = row['id']
        root = templates / name
        pin = pins[name] = git(root, 'rev-parse', 'HEAD').strip()
        manifest = yaml.safe_load(git(root, 'show', f'{pin}:meta/vibestudio.yml'))
        required_urls = [repository(dep['url']) for dep in manifest.get('template', {}).get('dependencies', [])]
        dependencies[name] = [templates_by_repo[url] for url in required_urls if url in templates_by_repo]
        files[name] = git(root, 'ls-tree', '-r', '--name-only', '-z', pin).rstrip('\0').split('\0')
        directories[name] = set(git(root, 'ls-tree', '-d', '-r', '--name-only', '-z', pin).rstrip('\0').split('\0'))

    for row in registry:
        for consumer in row.get('consumers', []):
            dependencies[consumer].append(row['id'])

    def sources_for(name):
        resolved, pending = set(), [name]
        while pending:
            source = pending.pop()
            if source not in resolved:
                resolved.add(source)
                pending.extend(dependencies[source])
        return sorted(resolved)

    source_sets = {row['id']: sources_for(row['id']) for row in registry if row['role'] != 'development'}
    workspace_directories = {name: set().union(*(directories[source] for source in sources))
                             for name, sources in source_sets.items()}
    skills = []
    for name, paths in files.items():
        for source in paths:
            if not source.endswith('/SKILL.md') and source != 'SKILL.md':
                continue
            prefix = source.removesuffix('SKILL.md')
            skills.append({'kind': 'skill', 'id': f'{name}:{source}', 'template': name, 'source': source,
                           'workspaces': [workspace for workspace, sources in source_sets.items() if name in sources],
                           'local_natlang': [file for file in paths if file.startswith(prefix) and file.endswith('.nl')]})

    registrations = {row['scenario']: row for row in teacher_registry}
    names = {case['name'] for case in catalog}
    if len(names) != len(catalog) or len(registrations) != len(teacher_registry):
        raise ValueError('Duplicate scenario name in the system-test catalog or teacher registry')
    if registrations.keys() - names:
        raise ValueError(f'Teacher scenarios missing from the executed catalog: {sorted(registrations.keys() - names)}')
    cases = []
    for case in sorted(catalog, key=lambda row: row['name']):
        name = case['name']
        registered = registrations.get(name)
        origin = case_origin(case, registered)
        required = set(case['requiresUnits'])
        cases.append({'kind': 'system-test', 'id': name, 'category': case['category'], 'origin': origin,
                      'capability': registered['family'] if registered else case['category'],
                      'description': case['description'], 'requires_units': sorted(required),
                      'candidate_workspaces': [workspace for workspace, paths in workspace_directories.items()
                                               if 'system-testing' in source_sets[workspace] and required <= paths],
                      **({'declared_workspace': registered['workspace'], 'source_groups': registered['sourceGroups']}
                         if registered else {})})
    baseline = sum(case['origin'] in AGENTIC_BASELINE_ORIGINS for case in cases)
    header = {'kind': 'snapshot', 'schema': 'vibestudio.open-goal-inventory/1', 'pins': pins,
              'workspace_sources': source_sets, 'skill_sources': len(skills), 'system_tests': len(cases),
              'agentic_baseline': baseline, 'minimum_distinct_open_goals': 2 * baseline,
              'harness_review': sum(case['origin'] == 'harness-review' for case in cases),
              'declared_open_goals': sum(case['origin'] == 'open-user-goal' for case in cases)}
    return [header, *sorted(skills, key=lambda row: row['id']), *cases]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ('host', 'templates', 'catalog', 'teacher-registry', 'out'):
        parser.add_argument(f'--{argument}', type=Path, required=True)
    args = parser.parse_args()
    rows = inventory(args.host, args.templates, json.loads(args.catalog.read_text()),
                     json_lines(args.teacher_registry))
    args.out.write_text(''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n' for row in rows))
    print(json.dumps(rows[0], indent=2))


if __name__ == '__main__':
    main()
