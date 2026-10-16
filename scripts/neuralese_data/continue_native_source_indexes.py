#!/usr/bin/env python3
"""Exact cache continuation for the default-false internal prefix extension."""
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


def sha(path):
    return sha256_file_hex(path)


def prove_extension(old_package, new_package):
    roots = [Path(old_package), Path(new_package)]
    inventories = [{name: sha(root / name) for name in POLICY_FILES} for root in roots]
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
            'old_index': str(path), 'old_index_sha256': sha(path),
            'new_index': str(target), 'new_index_sha256': sha(target)})
    return {'schema': 'natlang.native-source-index-continuation/1', **proof,
            'indexes': results, 'missing_source_sha256': sorted(wanted - {item['source_sha256'] for item in results})}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-package', type=Path, required=True)
    parser.add_argument('--new-package', type=Path, required=True)
    parser.add_argument('--source-sha256', action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    receipt = continue_indexes(prove_extension(args.old_package, args.new_package), args.source_sha256)
    write_json_durable(args.output, receipt)
    print(json.dumps({'indexes': len(receipt['indexes']), 'missing_sources': len(receipt['missing_source_sha256']),
                      'receipt': str(args.output)}))


if __name__ == '__main__':
    main()
