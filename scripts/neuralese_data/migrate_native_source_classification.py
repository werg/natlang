"""Build explicitly selected source-classification caches without graph compilation.

Run in a bounded CPU ledger unit, using immutable old/new runtime packages and
an exact current manifest leaf inventory. This never publishes a student pointer.
"""
from pathlib import Path
import argparse
import json

from natlang_neuralese.common.jsonio import write_json_durable
from natlang_neuralese.train.native_source_classification import (
    SCHEMA, _binding, _checked, migrate_index)


def manifest_leaves(reference, *, kind='native-graphs', model=None, parents=()):
    path = _checked({'path': reference['path'], 'sha256': reference['sha256']}).resolve()
    if path in parents:
        raise ValueError('classification inventory manifest cycle')
    if path.suffix == '.json' and kind == 'native-graphs':
        value = json.loads(path.read_text())
        if value.get('schema') != 'natlang.native-graph-sources/1' or not value.get('sources'):
            raise ValueError('classification inventory source manifest is invalid')
        rows = []
        for child in value['sources']:
            child_kind = child.get('kind')
            if child_kind not in ('native-graphs', 'semantic-pairs'):
                raise ValueError('classification inventory needs declared source kinds')
            child_model = child.get('model') if child_kind == 'semantic-pairs' else None
            if child_kind == 'semantic-pairs' and not isinstance(child_model, str):
                raise ValueError('classification inventory needs the selected semantic-pair model')
            target = Path(child['path'])
            if not target.is_absolute(): target = path.parent / target
            rows.extend(manifest_leaves({'path': str(target), 'sha256': child['sha256']},
                kind=child_kind, model=child_model, parents=(*parents, path)))
        return rows
    return [(str(path), reference['sha256'], kind, model)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--old-package', type=Path, required=True)
    parser.add_argument('--new-package', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    inventory = json.loads(args.inventory.read_text())
    proof = {'schema': SCHEMA, 'train_sha256': inventory['train']['sha256'],
             'train': inventory['train'], 'held': inventory['held'], 'inventory': _binding(args.inventory),
             'indexes': [], 'held_indexes': [], 'graph_position_expansions': 0,
             'model_allocations': 0, 'optimizer_updates': 0, 'student_pointer_writes': 0}
    for split, name in (('train', 'indexes'), ('test', 'held_indexes')):
        reference = inventory['train' if split == 'train' else 'held']
        expected = manifest_leaves(reference)
        supplied = inventory[name]
        actual = [(str(_checked(item['source']).resolve()), item['source']['sha256'],
                   item['kind'], item.get('model')) for item in supplied]
        if len(set(expected)) != len(expected) or len(set(actual)) != len(actual) or set(actual) != set(expected):
            raise ValueError('classification inventory must cover every unique current source leaf exactly once')
        for item in supplied:
            part = migrate_index(_checked(item['source']), _checked(item['old_index']),
                old_package=args.old_package, new_package=args.new_package,
                ordinary_partition=item.get('ordinary_partition'))
            from natlang_neuralese.train.native_source_classification import _index
            key = _index(part['old_index'])['key']
            if key['split'] != split or key['kind'] != item['kind'] or key['pair_model'] != item.get('model'):
                raise ValueError('classification inventory changed split/source model view')
            proof[name].append(part)
    write_json_durable(args.out, proof)
    print(json.dumps({'passed': True, 'proof': _binding(args.out),
        'train_leaves': len(proof['indexes']), 'held_leaves': len(proof['held_indexes']),
        'graph_position_expansions': 0, 'model_allocations': 0,
        'optimizer_updates': 0, 'student_pointer_writes': 0}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
