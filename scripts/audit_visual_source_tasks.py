#!/usr/bin/env python3
"""Audit preparation identities and blind inputs; does not approve task semantics."""
import argparse
from collections import Counter
import json
import re
from pathlib import Path, PurePosixPath

from acquire_visual_sources import digest, safe_path, write_json


def assets(value):
    if isinstance(value, dict):
        if set(('asset', 'sha256', 'bytes')) <= value.keys():
            yield value
        for item in value.values():
            yield from assets(item)
    elif isinstance(value, list):
        for item in value:
            yield from assets(item)


def audit(directory):
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    tasks_path = directory / 'artifact-source-tasks.jsonl'
    errors, groups, ids, rows = [], {}, set(), []
    try:
        held_path = directory / 'held-rows.json'
        if digest(held_path) != manifest['held_rows_sha256'] or len(json.loads(held_path.read_text())) != manifest['held_unsupported_rows']:
            raise ValueError('held-row dispositions differ from manifest')
        if digest(tasks_path) != manifest['sha256']:
            raise ValueError('task digest differs from manifest')
        adapter = directory / manifest['adapter_artifact']
        helper = directory / manifest['helper_artifact']
        if digest(helper) != manifest['helper_sha256']:
            raise ValueError('frozen helper identity differs')
        if digest(adapter) != manifest['adapter_sha256']:
            raise ValueError('frozen adapter identity differs')
        with tasks_path.open() as stream:
            for line in stream:
                row = json.loads(line)
                ident = row['id']
                if ident in ids:
                    raise ValueError('duplicate source task ID')
                ids.add(ident)
                if row['schema'] != 'natlang.artifact-source-task/1' or row['training_admitted'] is not False or row['provider_calls'] != 0:
                    raise ValueError('unsupported preparation/admission claim')
                group = row['group']
                if row.get('partition_assignment') != 'unassigned' or 'candidate_partition' in row:
                    raise ValueError('source preparation must not assign training/evaluation partitions')
                groups[group] = 'unassigned'
                source = row['source']
                if source['id'] != manifest['source']['id'] or source['revision'] != manifest['source']['revision'] or source['url'] != manifest['source']['upstream_url']:
                    raise ValueError('source identity differs from manifest')
                if not isinstance(row['input'], dict) or any(k in row['input'] for k in ('host_oracle', 'cadtests', 'expected', 'dst_code', 'score', 'quality')):
                    raise ValueError('oracle metadata in visible input')
                if not row['blockers'] or 'independent_artifact_executor_pending' not in row['blockers']:
                    raise ValueError('artifact executor pending state lost')
                expected_status = 'prepared_requires_artifact_executor' if row['blockers'] == ['independent_artifact_executor_pending'] else 'held_source_or_evaluator_review'
                if row['status'] != expected_status:
                    raise ValueError('task status differs from explicit blockers')
                for key, ref in [('host_oracle', row['host_oracle']), ('source record', {'path': source['record_path'], 'sha256': source['record_sha256']})]:
                    path = directory / safe_path(ref['path'])
                    if digest(path) != ref['sha256']:
                        raise ValueError(key + ' digest differs')
                record = json.loads((directory / source['record_path']).read_text())
                for asset in assets(record):
                    path = directory / safe_path(asset['asset'])
                    if path.stat().st_size != asset['bytes'] or digest(path) != asset['sha256']:
                        raise ValueError('source image/binary asset changed')
                if manifest['source']['adapter'] == 'cadtests':
                    if row['input']['prompt'] != record['prompt'] or any('cadtest' in key for key in row['input']):
                        raise ValueError('CAD prompt/check separation changed')
                    if group != 'cadprompt/program/' + str(record['sample_id']):
                        raise ValueError('CAD variants do not share source program group')
                if manifest['source']['adapter'] in ('designbench', 'designbench-frameworks'):
                    kind, framework = source['file'].split('/')[:2]
                    original = record['src_code'] if kind == 'edit' else record['code']
                    visible = row['input'].get('files', {})
                    if isinstance(original, dict):
                        mapped = {{'html': 'new.component.html', 'ts': 'new.component.ts', 'css': 'new.component.css'}[k]: v for k, v in original.items()}
                    else:
                        mapped = {{'vanilla': 'index.html', 'react': 'App.jsx', 'vue': 'App.vue'}[framework]: original}
                    if visible != mapped or (kind == 'edit' and row['input'].get('prompt') != record['prompt']):
                        raise ValueError('DesignBench input differs from original prompt/code')
                    if framework == 'angular':
                        refs = []
                        for code in visible.values():
                            refs.extend(re.findall(r'(?:templateUrl|styleUrl)\s*:\s*[\"\']([^\"\']+)', code))
                            for style_list in re.findall(r'styleUrls\s*:\s*\[([^\]]*)\]', code):
                                refs.extend(re.findall(r'[\"\']([^\"\']+)[\"\']', style_list))
                        missing = [ref for ref in refs if PurePosixPath(ref).as_posix() not in visible]
                        if missing and 'unresolved_relative_component_files' not in row['blockers']:
                            raise ValueError('unresolved Angular component files are untracked')
                rows.append(row)
        if len(rows) != manifest['tasks'] or len(groups) != manifest['groups']:
            raise ValueError('task/group count differs from manifest')
        if dict(Counter(row['status'] for row in rows)) != manifest['statuses']:
            raise ValueError('task status counts differ')
    except (OSError, ValueError, KeyError, TypeError) as error:
        errors.append(str(error))
    return {'schema': 'natlang.visual-source-preparation-audit/1', 'input_sha256': digest(tasks_path),
        'manifest_sha256': digest(manifest_path), 'passed': not errors, 'errors': errors,
        'tasks': len(rows), 'groups': len(groups), 'statuses': dict(Counter(row['status'] for row in rows)),
        'training_admitted': False, 'semantics_or_renderer_verified': False,
        'publication': 'Hash/group/input-separation audit only; rights, semantic rewards and runtime admission are not established.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', type=Path, nargs='+')
    args = parser.parse_args()
    reports = []
    for directory in args.directories:
        report = audit(directory)
        write_json(directory / 'audit.json', report)
        reports.append(report)
        print(json.dumps({'source': directory.name, **report}), flush=True)
    raise SystemExit(1 if any(not row['passed'] for row in reports) else 0)


if __name__ == '__main__':
    main()
