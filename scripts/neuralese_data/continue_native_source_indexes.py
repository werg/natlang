#!/usr/bin/env python3
"""Exact cache continuation for separately qualified native runtime extensions."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'training/neuralese'))
from natlang_neuralese.common import row_index
from natlang_neuralese.common.hashing import canonical_json_sha256_hex as digest, sha256_file_hex
from natlang_neuralese.common.jsonio import write_json_durable

POLICY_FILES = ('data/native_sources.py', 'data/native_graph_schema.py',
    'model/native_graph_plan.py', 'data/native_pair_schema.py',
    'common/row_index.py', 'common/jsonio.py', 'common/hashing.py')
# Reviewed complete-runtime extension, deliberately separate from prove_extension.
# These are exact source bytes, not permission to accept any future parser.
_UNCHANGED = {
    'data/native_graph_schema.py': '8ad9a94a5429080b3e6e407982517c3bb8cd2f83d2b2bf436e9ce72662e575c3',
    'data/native_pair_schema.py': '839df4000876a6c0f9ac1283f14fbae07cd66188e170d618b4e6008b8715f91d',
    'common/row_index.py': 'e592aa380417424c3018f6d1672a2b961e45c1571a7e49bdec9bc4be11123544',
    'common/jsonio.py': '0716177d8690b53f3ea39f6e6dd3c1ebe5ff7407f02a7d095e9e59d4a1631114',
}
_OLD_COMMON = {**_UNCHANGED,
    'data/native_sources.py': '15b43272e79acf2b73f3793416a87a1b844850f8df22c2c2fe27870c719abe97',
    'common/hashing.py': '736421674d6c1d9505f14f996e151ef320962bcef80bb0cf741dd9f686dce100'}
_OLD_COMPILERS = {
    'lfm-14ca1758': '233e05ae34b68700a1a7e5c973b71b95cfa7b051e27015e2e7abc2752bb69aae',
    'mellum-b50e8823': '80fa45cadbb455effcbf55260d7a2c31fae0c1cedb5b5dc52ac497e0527a01cc',
}
_NEW_COMPLETE = {**_UNCHANGED,
    'data/native_sources.py': 'e104cda0b7f6cd862bcf5a08bb8f76ee6f99c138b65a4a65dc2b354df5a1aedf',
    'model/native_graph_plan.py': '1c6d9c826364519fb61869282bf4d786c351dc391a07adc8e6cc9f3299fbcfcc',
    'common/hashing.py': 'fcb84e31bc6a4d0e681c22d57854d5f7646392325cff486216e01732797018da'}
_COMPLETE_HELPERS = {
    'data/native_source_tree.py': 'c8b0afe770bc02bc987d5a9d218556189152e897f9c509fe6eb309f7b6e08cf0'}


def prove_reviewed_runtime_extension(old_package, new_package):
    """Certify only the reviewed old-valid-row extension into canonical64ec.

    Old validated leaves cannot contain a stop_gradient field (the old constructor
    rejected it). False defaults are omitted from new row bindings. The manifest
    helper is checked separately because manifests themselves are never cached.
    Actual execution and geometry qualification remain separate requirements.
    """
    old, new = (Path(old_package), Path(new_package))
    inventories = [{name: sha256_file_hex(root / name) for name in POLICY_FILES}
                   for root in (old, new)]
    matched = [name for name, compiler in _OLD_COMPILERS.items()
               if inventories[0] == {**_OLD_COMMON, 'model/native_graph_plan.py': compiler}]
    if len(matched) != 1 or inventories[1] != _NEW_COMPLETE:
        raise ValueError('source policy is not the exact reviewed complete-runtime extension')
    helpers = {name: sha256_file_hex(new / name) for name in _COMPLETE_HELPERS}
    if helpers != _COMPLETE_HELPERS:
        raise ValueError('complete-runtime manifest helper differs from its reviewed bytes')
    return {'qualification': 'complete-native-runtime-64ec-20261016',
            'parent': matched[0], 'policy_files': inventories,
            'new_helpers': helpers,
            'old_policy_sha256': digest(inventories[0]),
            'new_policy_sha256': digest(inventories[1])}


GUARDS = (
    """if type(node.cache_prefix) is not bool or (node.cache_prefix and
            (node.producing or node.logical_producing or len(node.imports) > 1)):
        raise ValueError('raw prefix caches require unadapted tokens and at most one complete prefix import')""",
    """if source_node.cache_prefix:
        if not edge.logical or edge.at != 0 or lo != 0 or hi != len(sequences[source_index]):
            raise ValueError('raw prefix cache imports must include the complete causal prefix at position zero')
        declared = ((0, len(sequences[source_index])),)""",
    """if node.cache_prefix and (edge.source in external_sizes or not source_node.cache_prefix):
        raise ValueError('raw prefix caches can extend only another raw complete prefix cache')""")
HELPER = '''def invocation_dict(node):
    """Stable semantic serialization, preserving pre-cache graph identities."""
    value = asdict(node)
    if not node.cache_prefix:
        value.pop('cache_prefix')
    return value
'''


def tree(value):
    return ast.dump(value, include_attributes=False)


def prove_extension(old_package, new_package):
    roots = [Path(old_package), Path(new_package)]
    inventories = [{name: sha256_file_hex(root / name) for name in POLICY_FILES} for root in roots]
    for name in POLICY_FILES:
        if name != 'model/native_graph_plan.py' and inventories[0][name] != inventories[1][name]:
            raise ValueError(f'validation policy changed: {name}')
    parser = ast.parse((roots[1] / 'data/native_graph_schema.py').read_text())
    calls = [node for node in ast.walk(parser) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == 'Invocation']
    if not calls or any(len(call.args) > 5 or any(isinstance(arg, ast.Starred) for arg in call.args)
            or any(kw.arg in (None, 'cache_prefix') for kw in call.keywords) for call in calls):
        raise ValueError('raw parser can set internal prefix state')
    old, new = [ast.parse((root / 'model/native_graph_plan.py').read_text()) for root in roots]
    expected = {tree(ast.parse(code).body[0]): 0 for code in GUARDS}
    counts = {'field': 0, 'helper': 0, 'import': 0}

    class Normalize(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            if node.module == 'dataclasses' and any(alias.name == 'asdict' for alias in node.names):
                if tree(node) != tree(ast.parse('from dataclasses import asdict, dataclass').body[0]):
                    raise ValueError('unexpected dataclass import extension')
                counts['import'] += 1
                node.names = [alias for alias in node.names if alias.name != 'asdict']
            return node

        def visit_ClassDef(self, node):
            if node.name == 'Invocation':
                retained = []
                for field in node.body:
                    if isinstance(field, ast.AnnAssign) and isinstance(field.target, ast.Name) and field.target.id == 'cache_prefix':
                        if tree(field) != tree(ast.parse('cache_prefix: bool = False').body[0]):
                            raise ValueError('internal prefix field must default to False')
                        counts['field'] += 1
                    else:
                        retained.append(field)
                node.body = retained
            return self.generic_visit(node)

        def visit_FunctionDef(self, node):
            if node.name == 'invocation_dict':
                if tree(node) != tree(ast.parse(HELPER).body[0]):
                    raise ValueError('unexpected invocation serializer')
                counts['helper'] += 1
                return None
            return self.generic_visit(node)

        def visit_If(self, node):
            key = tree(node)
            if key in expected:
                expected[key] += 1
                return None
            return self.generic_visit(node)

    normalized = Normalize().visit(new)
    if counts != {'field': 1, 'helper': 1, 'import': 1} or any(count != 1 for count in expected.values()):
        raise ValueError('compiler is not the exact supported prefix extension')
    if tree(old) != tree(normalized):
        raise ValueError('ordinary compiler semantics changed')
    return {'policy_files': inventories, 'old_policy_sha256': digest(inventories[0]),
            'new_policy_sha256': digest(inventories[1])}


def continue_indexes(proof, source_hashes):
    wanted = set(source_hashes)
    if not wanted or any(len(value) != 64 or any(c not in '0123456789abcdef' for c in value) for value in wanted):
        raise ValueError('explicit source SHA-256 hashes required')
    results = []
    for path in sorted(row_index.cache_dir().glob('native-graphs-*.pkl')):
        value = row_index.load(path)
        key = value.get('key') if value else None
        if not isinstance(key, dict) or key.get('policy_sha256') != proof['old_policy_sha256'] or key.get('source_sha256') not in wanted:
            continue
        if set(key) != {'source_sha256', 'tokenizer', 'split', 'policy_sha256', 'kind', 'pair_model', 'role_ids'}:
            raise ValueError('unexpected source index key')
        if key['kind'] not in {'native-graphs', 'semantic-pairs'}:
            raise ValueError('unexpected source index kind')
        if path != row_index.entry_path('native-graphs', digest(key)):
            raise ValueError('source index filename does not bind its key')
        entries = value.get('entries')
        if not isinstance(entries, list) or not entries or value.get('metadata_sha256') != digest(entries):
            raise ValueError('source index metadata is corrupt')
        new_key = {**key, 'policy_sha256': proof['new_policy_sha256']}
        target = row_index.entry_path('native-graphs', digest(new_key))
        new_value = {**value, 'key': new_key}
        if target.exists():
            if row_index.load(target) != new_value:
                raise ValueError('existing destination index conflicts')
        else:
            row_index.save(target, new_value)
        if row_index.load(target) != new_value:
            raise ValueError('index continuation write failed')
        results.append({'source_sha256': key['source_sha256'], 'split': key['split'],
            'kind': key['kind'], 'rows': len(entries), 'metadata_sha256': value['metadata_sha256'],
            'old_index': str(path), 'old_index_sha256': sha256_file_hex(path),
            'new_index': str(target), 'new_index_sha256': sha256_file_hex(target)})
    return {'schema': 'natlang.native-source-index-continuation/1', **proof,
            'indexes': results, 'missing_source_sha256': sorted(wanted - {item['source_sha256'] for item in results})}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-package', type=Path, required=True)
    parser.add_argument('--new-package', type=Path, required=True)
    parser.add_argument('--source-sha256', action='append', required=True)
    parser.add_argument('--qualification', choices=('prefix-extension', 'complete-native-runtime'),
                        default='prefix-extension')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prove = (prove_extension if args.qualification == 'prefix-extension'
             else prove_reviewed_runtime_extension)
    receipt = continue_indexes(prove(args.old_package, args.new_package), args.source_sha256)
    write_json_durable(args.output, receipt)
    print(json.dumps({'indexes': len(receipt['indexes']), 'missing_sources': len(receipt['missing_source_sha256']),
                      'receipt': str(args.output)}))


if __name__ == '__main__':
    main()
