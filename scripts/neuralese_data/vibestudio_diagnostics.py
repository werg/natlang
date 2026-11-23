"""Retain diagnostics for every exact collector-owned instance, without live RPCs.

Collector source.json owns instance selection. Repository-keyed instanceRegistry
paths and serverLogStore's state/logs path own log lookup. Never choose a latest
collector line, an unrelated workspace, or overwrite another group's evidence.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile


def _digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def _copy(source, target):
    if not source.is_file():
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=target.name + '.', dir=target.parent)
    try:
        with os.fdopen(descriptor, 'wb') as output, source.open('rb') as stream:
            # A finite prefix of a live append-only log, without buffering the
            # entire transcript or chasing concurrent writes indefinitely.
            remaining = os.fstat(stream.fileno()).st_size
            while remaining:
                chunk = stream.read(min(remaining, 1024 * 1024))
                if not chunk:
                    break  # log rotation may truncate the source
                output.write(chunk)
                remaining -= len(chunk)
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return {'source': str(source), 'retained': str(target), 'sha256': _digest(target)}


def retain_instance_diagnostics(capture, host, output, profile):
    capture, host, output, profile = map(Path, (capture, host, output, profile))
    repo_key = hashlib.sha256(str(host.resolve()).encode()).hexdigest()[:16]
    groups = {}
    for pins in sorted((capture / 'cases').glob('*/source.json')):
        source = json.loads(pins.read_text())
        instance = source.get('instance')
        if not isinstance(instance, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', instance):
            raise ValueError(f'Invalid captured instance: {pins}')
        group = groups.setdefault(instance, {'instance': instance, 'sources': [], 'logs': []})
        group['sources'].append({'case': pins.parent.name, 'path': str(pins),
                                'sha256': hashlib.sha256(pins.read_bytes()).hexdigest(),
                                'workspace': source.get('workspace', 'system'),
                                'run_id': source.get('run_id')})
    for instance, group in groups.items():
        destination = output / 'instances' / instance
        supervisor = profile / 'instance-state' / repo_key / 'system-test-logs' / f'{instance}.log'
        retained = _copy(supervisor, destination / 'supervisor.log')
        if retained:
            group['logs'].append(retained)
        record = profile / 'dev-instances' / repo_key / f'{instance}.json'
        if not record.is_file():
            group['record_missing'] = True
            continue
        record_value = json.loads(record.read_text())
        root = record_value.get('root')
        if not isinstance(root, str) or not Path(root).is_absolute():
            raise ValueError(f'Invalid instance root: {record}')
        workspaces = Path(root) / 'workspaces'
        # Every native workspace owns its state/logs stream; preserve all, with
        # its actual workspace directory identity instead of first-find-wins.
        for log in sorted(workspaces.glob('*/state/logs/server-log.jsonl*')):
            if log.name not in {'server-log.jsonl', 'server-log.jsonl.1'}:
                continue
            retained = _copy(log, destination / 'workspaces' / log.parent.parent.parent.name / log.name)
            if retained:
                group['logs'].append(retained)
    # Retired live instances can disappear between periodic samples. Keep their
    # already copied bytes indexed rather than silently losing them from view.
    for instance, group in groups.items():
        destination = output / 'instances' / instance
        group['retained'] = [{'path': str(log), 'sha256': _digest(log)}
                             for log in sorted(destination.rglob('*')) if log.is_file()]
    result = {'schema': 'natlang.vibestudio-instance-diagnostics/1', 'capture': str(capture),
              'host': str(host.resolve()), 'repo_key': repo_key, 'producer_sha256': _digest(Path(__file__)),
              'instances': list(groups.values())}
    output.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='instances.', dir=output)
    try:
        with os.fdopen(descriptor, 'w') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')
        os.replace(temporary, output / 'instances.json')
    finally:
        Path(temporary).unlink(missing_ok=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ('capture', 'host', 'output'):
        parser.add_argument('--' + argument, type=Path, required=True)
    parser.add_argument('--profile', type=Path, default=Path(os.environ.get('XDG_CONFIG_HOME', Path.home() / '.config')) / 'vibestudio')
    args = parser.parse_args()
    result = retain_instance_diagnostics(args.capture, args.host, args.output, args.profile)
    print(json.dumps({'instances': len(result['instances']), 'logs': sum(len(group['logs']) for group in result['instances'])}))


if __name__ == '__main__':
    main()
