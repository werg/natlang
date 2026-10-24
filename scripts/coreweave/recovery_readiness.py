#!/usr/bin/env python3
"""Reconcile checkpoint-bound input identities against local and archived bytes.

This read-only audit distinguishes a local usable file from a sealed archive
member and from an unresolved dependency. It never grants source admission,
capacity qualification, restored-runtime credit or new training exposure.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time

try:
    from .recovery_inputs import digest, durable_json, inventory_archive
except ImportError:
    from recovery_inputs import digest, durable_json, inventory_archive


def required_bindings(metadata):
    required = {}

    def add(path, sha, kind):
        if not isinstance(path, str) or not re.fullmatch('[0-9a-f]{64}', str(sha)):
            return
        key = (path, sha)
        required.setdefault(key, {'path': path, 'sha256': sha, 'kinds': []})
        if kind not in required[key]['kinds']:
            required[key]['kinds'].append(kind)

    def source(value):
        add(value.get('path'), value.get('sha256'), 'source-manifest' if value.get('sources') else 'source-payload')
        for child in value.get('sources', []):
            source(child)

    for receipt in metadata['training'].get('source_receipts', []):
        source(receipt)

    def control(value):
        if isinstance(value, dict):
            if value.get('checkpoint_resident_value'):
                return
            add(value.get('path'), value.get('sha256', value.get('file_sha256')), 'control-binding')
            for key, child in value.items():
                if key not in ('by_source_group', 'sources'):
                    control(child)
        elif isinstance(value, list):
            for child in value:
                control(child)
        elif isinstance(value, str) and value[:1] in ('{', '['):
            try:
                control(json.loads(value))
            except (ValueError, RecursionError):
                pass

    control(metadata['training'].get('options', {}))
    config = metadata['model_config']
    for name, sha in config.get('base_inventory', {}).items():
        add(str(Path(config['base']) / name), sha, 'base-model-file')
    return list(required.values())


def reconcile(required, candidates, archived):
    local_cache, results = {}, []
    for binding in required:
        sha = binding['sha256']
        candidates_for_sha = [binding['path'], *candidates.get(sha, [])]
        local = None
        for candidate in dict.fromkeys(candidates_for_sha):
            if candidate not in local_cache:
                try:
                    path = Path(candidate)
                    before = path.stat()
                    value = digest(path) if path.is_file() else None
                    after = path.stat()
                    if (before.st_ino, before.st_size, before.st_mtime_ns) != (
                            after.st_ino, after.st_size, after.st_mtime_ns):
                        value = None
                    local_cache[candidate] = value
                except OSError:
                    local_cache[candidate] = None
            if local_cache[candidate] == sha:
                local = candidate
                break
        result = dict(binding)
        if local:
            result.update(status='local-exact', local=local)
        elif archived.get(sha):
            result.update(status='verified-archive-member', archives=archived[sha])
        else:
            result.update(status='unresolved')
        results.append(result)
    return {'at': time.time(), 'required': len(results),
            'local_exact': sum(r['status'] == 'local-exact' for r in results),
            'archive_members': sum(r['status'] == 'verified-archive-member' for r in results),
            'unresolved': sum(r['status'] == 'unresolved' for r in results),
            'bindings': results, 'runtime_restored': False, 'training_admission': False,
            'new_training_exposure': False, 'launch_ready': False,
            'scope': 'Checkpoint-declared sources and explicit option/base bindings only; '
                     'complete recipe/controller closure and actual device checks remain required.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--candidates', type=Path, required=True,
                        help='JSON mapping SHA256 to candidate local file paths')
    parser.add_argument('--archive-objects', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    archived = {}
    for path in args.archive_objects.glob('*.inventory.json'):
        inventory = json.loads(path.read_text())
        archive = args.archive_objects / (inventory['archive']['sha256'] + '.tar.zst')
        # Recheck the actual container, not an old inventory's existence.
        if (not archive.is_file() or archive.stat().st_size != inventory['archive']['bytes']
                or digest(archive) != inventory['archive']['sha256']):
            raise ValueError('recovery archive differs from its member inventory')
        checked = inventory_archive(archive, inventory['archive'])
        if checked['members'] != inventory['members']:
            raise ValueError('recovery archive member inventory differs from actual bytes')
        for member in inventory['members']:
            if member.get('sha256'):
                archived.setdefault(member['sha256'], []).append({
                    'archive': str(archive), 'archive_identity': inventory['archive'],
                    'member': member['path'], 'bytes': member['bytes']})
    metadata = json.loads(args.metadata.read_text())
    result = reconcile(required_bindings(metadata), json.loads(args.candidates.read_text()), archived)
    result['metadata'] = {'path': str(args.metadata), 'sha256': digest(args.metadata)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    durable_json(args.out, result)
    print(json.dumps({k: v for k, v in result.items() if k != 'bindings'}))


if __name__ == '__main__':
    main()
