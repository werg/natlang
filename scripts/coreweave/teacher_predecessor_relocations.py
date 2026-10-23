"""Create exact, non-admission relocation bindings for registered teacher corpora."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from natlang_neuralese.common.paths import resolve
from scripts.coreweave.teacher_tranche_admission import file_binding, read_bound


SCHEMA = 'natlang.teacher-predecessor-relocations/1'


def _matches(directory: Path, manifest: dict) -> list[dict]:
    members = []
    for reference in manifest['files']:
        path = directory / reference['path']
        try:
            actual = file_binding(path)
        except OSError:
            return []
        if any(actual.get(key) != reference.get(key) for key in ('sha256', 'bytes')):
            return []
        members.append({'path': reference['path'], 'manifest_member': reference})
    return members


def build(root: Path, candidate_roots: list[Path], *, namespace: str) -> dict:
    root = root.resolve(strict=True)
    registry_ref = file_binding(root / 'training/neuralese_corpora.json')
    registry = read_bound(registry_ref)
    corpora = []
    missing = []
    seen = set()
    for entry in registry['corpora']:
        ident = entry['id']
        if not ident.startswith(namespace + '-') or entry.get('training_admission') is not True:
            continue
        if ident in seen:
            raise ValueError(f'duplicate admitted corpus in registry: {ident}')
        seen.add(ident)
        manifest_ref = file_binding(root / 'training/corpus-manifests' / f'{ident}.json')
        manifest = read_bound(manifest_ref)
        if manifest.get('id') != ident or manifest.get('path') != entry['path']:
            raise ValueError(f'manifest does not bind registry entry: {ident}')
        if len({member['path'] for member in manifest['files']}) != len(manifest['files']):
            raise ValueError(f'manifest repeats member paths: {ident}')
        canonical = root / entry['path']
        if _matches(canonical, manifest):
            continue
        candidates = []
        for candidate_root in candidate_roots:
            # Materializer directory labels preserve the exact source batch/tranche.
            # Compare every immutable manifest member; labels alone never match.
            for directory in candidate_root.glob('batch*-tr*-*'):
                members = _matches(directory, manifest)
                if members:
                    candidates.append((directory.resolve(strict=True), members))
        unique = {str(directory): members for directory, members in candidates}
        if len(unique) != 1:
            missing.append({'id': ident, 'candidate_count': len(unique), 'reason': 'no unique complete byte-exact relocation'})
            continue
        directory, members = next(iter(unique.items()))
        corpora.append({'id': ident, 'registry_entry': entry, 'manifest': manifest_ref,
                        'directory': directory, 'members': members})
    if file_binding(root / 'training/neuralese_corpora.json') != registry_ref:
        raise ValueError('registry changed while building relocation declaration')
    return {'schema': SCHEMA, 'training_admission': False,
            'purpose': 'Exact byte relocation for already registered corpora; grants no admission or exposure.',
            'registry': registry_ref, 'namespace': namespace,
            'corpora': corpora, 'missing': missing}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--repository', type=Path, default=resolve('repo'))
    parser.add_argument('--candidate-root', action='append', type=Path, required=True)
    parser.add_argument('--namespace', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = build(args.repository, args.candidate_root, namespace=args.namespace)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(result, sort_keys=True, indent=2) + '\n').encode()
    tmp = args.output.with_suffix(args.output.suffix + '.tmp')
    tmp.write_bytes(raw)
    tmp.replace(args.output)
    print(json.dumps({'output': str(args.output), 'sha256': hashlib.sha256(raw).hexdigest(),
                      'relocated': len(result['corpora']), 'missing': len(result['missing'])}))


if __name__ == '__main__':
    main()
