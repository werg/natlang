"""Publish complete connected-source protection before native conversion/admission."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

from natlang_neuralese.common.hashing import sha256_file_hex
from natlang_neuralese.common.jsonio import write_json_durable
from scripts.neuralese_data.conversion_output_lease import acquire_output_lease
from scripts.neuralese_data.prepare_acquired_session_admission import (
    acquired_source_closure, verify_global_source_closure, bound)
from scripts.neuralese_data.build_verified_teacher_trajectory_graphs import current_protection


def project(base, output, *, exports=()):
    base, output = Path(base).resolve(strict=True), Path(output).resolve()
    original = json.loads(base.read_text()); protection = current_protection(base)
    base_sha = sha256_file_hex(base)
    with acquire_output_lease(output):
        target = output / 'protection-manifest.json'
        if target.exists():
            raise ValueError('connected protection output is immutable; select a new output directory')
        value = dict(original); inventory = {}
        replaced = {'source-group-components.jsonl', 'held-union/acquired-connected-source-groups.jsonl'}
        for item in original['files']:
            if item['path'] in replaced:
                continue
            source = bound(base.parent/item['path'], item)
            destination = output/item['path']; destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            inventory[item['path']] = dict(item)
        if exports:
            # A qualified component inventory carries earlier sources across
            # machines without copying their complete normalized text again.
            if not original.get('source_group_components'):
                raise ValueError('extending sources requires a qualified parent component projection')
            inherited = {}
            for role, source in [('components', base.parent/original['source_group_components']), ('projection', base)]:
                destination = output/('inherited-'+role+('.jsonl' if role=='components' else '.json'))
                shutil.copyfile(source, destination)
                inherited[role] = {'path': destination.name, 'sha256': sha256_file_hex(destination),
                    'bytes': destination.stat().st_size}
                inventory[destination.name] = inherited[role]
            value['inherited_source_components'] = inherited
            bindings = []
            for directory in exports:
                directory = Path(directory).resolve(strict=True)
                complete = json.loads((directory/'complete.json').read_text())
                declared = {Path(row['receipt']).resolve() for row in complete['files']
                            if row['status'] in {'exported', 'reused'}}
                actual = {path.resolve() for path in directory.glob('*/export.json')}
                if declared != actual or not actual:
                    raise ValueError('new source closure requires the complete normalized export inventory')
                for receipt in sorted(actual):
                    exported = json.loads(receipt.read_text())
                    source = bound(receipt.parent/'full-trajectories.jsonl.zst',
                        {'sha256': exported['outputs']['full-trajectories.jsonl.zst']})
                    bindings.append({'path': str(receipt), 'sha256': sha256_file_hex(receipt),
                        'source_sha256': sha256_file_hex(source)})
            value['normalized_export_bindings'] = bindings
        candidate = output/'protection-manifest.pending.json'
        # Reconstruction reads only exact bound normalized sources and any
        # independently qualified inherited component inventory.
        write_json_durable(candidate, value)
        accounting, closed, components = acquired_source_closure(candidate, protection)
        files = {
            'source-group-components.jsonl': [{'source_groups': group} for group in components],
            'held-union/acquired-connected-source-groups.jsonl': [{'source_groups': sorted(closed['held'])}],
            'quality-groups.jsonl': [{'source_groups': sorted(closed['quality'])}],
        }
        for relative, rows in files.items():
            path = output/relative; path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('w', encoding='utf-8') as stream:
                for row in rows: stream.write(json.dumps(row, sort_keys=True)+'\n')
                stream.flush(); os.fsync(stream.fileno())
            inventory[relative] = {'path': relative, 'sha256': sha256_file_hex(path), 'bytes': path.stat().st_size}
        value.update(files=list(inventory.values()), source_group_components='source-group-components.jsonl',
            held_groups=len(closed['held']), quality_groups=len(closed['quality']),
            derived_from={'path': str(base), 'sha256': base_sha},
            acquired_source_group_closure={'schema': 'natlang.acquired-source-group-closure/1', 'passed': True,
                'accounting': accounting, 'generator_sha256': sha256_file_hex(__file__)})
        write_json_durable(candidate, value)
        projected = current_protection(candidate)
        if any(projected[key] != closed[key] for key in closed):
            raise ValueError('published component projection differs from reconstructed source protection')
        verify_global_source_closure(candidate, projected)
        if sha256_file_hex(base) != base_sha or current_protection(base) != protection:
            raise ValueError('parent protection changed while closing acquired sources')
        # Copy durability precedes publication. Manifest publication is the only
        # point at which a consumer may treat this directory as complete.
        for path in output.rglob('*'):
            if path.is_file():
                with path.open('rb') as stream: os.fsync(stream.fileno())
        os.replace(candidate, target)
        for directory in sorted([output, *(path for path in output.rglob('*') if path.is_dir())],
                                key=lambda path: len(path.parts), reverse=True):
            descriptor = os.open(directory, os.O_DIRECTORY)
            try: os.fsync(descriptor)
            finally: os.close(descriptor)
        return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-protection', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--exports', type=Path, action='append', default=[])
    args = parser.parse_args(argv)
    path = project(args.base_protection, args.output, exports=args.exports)
    print(json.dumps({'path': str(path), 'sha256': sha256_file_hex(path)}))


if __name__ == '__main__': main()
