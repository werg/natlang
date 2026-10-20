#!/usr/bin/env python3
"""Exact cache continuation for separately qualified native runtime extensions."""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import pickle
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'training/neuralese'))
from natlang_neuralese.common import row_index
from natlang_neuralese.common.hashing import canonical_json_sha256_hex as digest, sha256_file_hex, sha256_file_hex_cached
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


METADATA_STATE_HELPER = 'def _checked_file_state(self, previous, state, *, context, strict=False):\n        """A metadata change is safe only after stable, fresh whole-file hashing."""\n        fields = (\'st_dev\', \'st_ino\', \'st_size\', \'st_mtime_ns\', \'st_ctime_ns\')\n        def signature(value):\n            return tuple(getattr(value, name) for name in fields)\n        def binding(value):\n            return (value.st_size, value.st_mtime_ns, value.st_ino)\n        expected = (getattr(self, \'_file_stat_signature\', None) if type(previous) is tuple\n                    else signature(previous))\n        same = signature(state) == expected and (type(previous) is not tuple or binding(state) == previous)\n        if not same:\n            if strict:\n                raise ValueError(f\'{self.path}: \' + context)\n            before = state\n            actual = sha256_file_hex(self.path)\n            state = self.path.stat()\n            if signature(before) != signature(state) or actual != self.file_sha256:\n                raise ValueError(f\'{self.path}: \' + context)\n        self._file_stat_signature = signature(state)\n        self.file_binding = binding(state)\n        return state\n'

def prove_metadata_revalidation_extension(old_package, new_package, reference_package):
    """An exact reviewed parser followed only by the declared file-state guard repair."""
    old,new,reference=map(Path,(old_package,new_package,reference_package))
    inventories=[{name:sha256_file_hex(root/name) for name in POLICY_FILES} for root in (old,new,reference)]
    if (inventories[2]!=_NEW_COMPLETE or
            {name:sha256_file_hex(reference/name) for name in _COMPLETE_HELPERS}!=_COMPLETE_HELPERS):
        raise ValueError('metadata continuation requires the exact reviewed canonical reference')
    if inventories[0]==_NEW_COMPLETE:
        parent='canonical64ec'
    else:
        parent=prove_reviewed_runtime_extension(old,reference)['parent']
    if any(inventories[1][name]!=inventories[2][name] for name in POLICY_FILES if name!='data/native_sources.py'):
        raise ValueError('metadata continuation changed parser dependencies')
    checks={
        "after = self._checked_file_state(before, after, context='source changed while loading graph index')":
            "if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):\n    raise ValueError(f'{self.path}: source changed while loading graph index')",
        "after = self._checked_file_state(before, after, context='source changed during streaming graph validation', strict=True)":
            "if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):\n    raise ValueError(f'{self.path}: source changed during streaming graph validation')",
        "state = self._checked_file_state(self.file_binding, state, context='source changed after index validation')":
            "if (state.st_size, state.st_mtime_ns, state.st_ino) != self.file_binding:\n    raise ValueError(f'{self.path}: source changed after index validation')",
        "after = self._checked_file_state(self.file_binding, after, context='source changed while streaming examples')":
            "if (after.st_size, after.st_mtime_ns, after.st_ino) != self.file_binding:\n    raise ValueError(f'{self.path}: source changed while streaming examples')"}
    checks={tree(ast.parse(key).body[0]):ast.parse(value).body[0] for key,value in checks.items()}
    post=tree(ast.parse("self._checked_file_state(self.file_binding, self.path.stat(), context='source changed while streaming examples')").body[0])
    expected_helper=tree(ast.parse(METADATA_STATE_HELPER).body[0])
    counts={'helper':0,'checks':0,'post':0,'import':0}
    class Normalize(ast.NodeTransformer):
        def visit_FunctionDef(self,node):
            if node.name=='_checked_file_state':
                if tree(node)!=expected_helper:raise ValueError('metadata continuation changed the exact file verifier')
                counts['helper']+=1;return None
            return self.generic_visit(node)
        def visit_ImportFrom(self,node):
            expected=ast.parse('from ..common.hashing import sha256_file_hex_cached, sha256_file_hex').body[0]
            if tree(node)==tree(expected):
                counts['import']+=1;node.names=node.names[:1]
            return node
        def visit_Assign(self,node):
            key=tree(node)
            if key in checks:counts['checks']+=1;return checks[key]
            return self.generic_visit(node)
        def visit_If(self,node):
            if tree(node.test)==tree(ast.parse("self.kind == 'manifest'",mode='eval').body):
                if len(node.body)<2 or not isinstance(node.body[-1],ast.Return) or tree(node.body[-2])!=post:
                    raise ValueError('metadata continuation lost the post-stream manifest guard')
                counts['post']+=1;node.body.pop(-2)
            return self.generic_visit(node)
    normalized=Normalize().visit(ast.parse((new/'data/native_sources.py').read_text()))
    if (counts!={'helper':1,'checks':4,'post':1,'import':1}
            or tree(normalized)!=tree(ast.parse((reference/'data/native_sources.py').read_text()))):
        raise ValueError('metadata continuation changed source admission or streaming semantics')
    helper=ast.parse((new/'data/native_source_tree.py').read_text())
    old_helper=ast.parse((reference/'data/native_source_tree.py').read_text())
    code="state = source._checked_file_state(before, state, context='native graph manifest changed during validation', strict=True)"
    guard=ast.parse("if (before.st_size, before.st_mtime_ns, before.st_ino) != (state.st_size, state.st_mtime_ns, state.st_ino):\n    raise ValueError('native graph manifest changed during validation')").body[0]
    key=tree(ast.parse(code).body[0]);hits=0
    function=next(node for node in helper.body if isinstance(node,ast.FunctionDef) and node.name=='initialize_manifest')
    for index,node in enumerate(function.body):
        if tree(node)==key:function.body[index]=guard;hits+=1
    assignment=ast.parse('source.file_sha256 = sha256_file_hex_cached(source.path)').body[0]
    matches=[node for node in function.body if tree(node)==tree(assignment)]
    # Hash-before-read closes the old read/hash replacement race. Apart from
    # that ordering and the exact guard call, the manifest parser is unchanged.
    if len(matches)!=1 or tree(function.body[1])!=tree(assignment):
        raise ValueError('metadata manifest must bind its hash before parsing bytes')
    function.body.remove(matches[0])
    kind=tree(ast.parse("source.kind = 'manifest'").body[0])
    positions=[index for index,node in enumerate(function.body) if tree(node)==kind]
    if len(positions)!=1:raise ValueError('metadata manifest kind changed')
    function.body.insert(positions[0]+1,matches[0])
    if hits!=1 or tree(helper)!=tree(old_helper):
        raise ValueError('metadata continuation changed manifest admission')
    return {'qualification':'source-metadata-revalidation-20261016','parent':parent,
        'policy_files':inventories[:2],'reference_policy_files':inventories[2],
        'new_helpers':{name:sha256_file_hex(new/name) for name in _COMPLETE_HELPERS},
        'old_policy_sha256':digest(inventories[0]),'new_policy_sha256':digest(inventories[1])}


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


def _publish_index_without_replacement(target, value):
    """Retain a concurrent trainer's destination, accepting only exact equality."""
    if target.exists():
        if row_index.load(target) != value:
            raise ValueError('existing destination index conflicts')
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=target.name + '.', suffix='.tmp', dir=target.parent)
    try:
        with os.fdopen(descriptor, 'wb') as handle:
            pickle.dump({**value, 'schema': 'natlang.row-index/1'}, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            if row_index.load(target) != value:
                raise ValueError('existing destination index conflicts')
        directory = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        os.unlink(temporary)


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
        _publish_index_without_replacement(target, new_value)
        if row_index.load(target) != new_value:
            raise ValueError('index continuation write failed')
        results.append({'source_sha256': key['source_sha256'], 'split': key['split'],
            'kind': key['kind'], 'rows': len(entries), 'metadata_sha256': value['metadata_sha256'],
            'old_index': str(path), 'old_index_sha256': sha256_file_hex(path),
            'new_index': str(target), 'new_index_sha256': sha256_file_hex(target)})
    return {'schema': 'natlang.native-source-index-continuation/1', **proof,
            'indexes': results, 'missing_source_sha256': sorted(wanted - {item['source_sha256'] for item in results})}



_CE_SETTER = '\n    def set_ce_policy(self, policy):\n        """Select an explicit objective without changing raw indexes/credits."""\n        self.ce_policy = policy\n        for child in getattr(self, \'children\', ()):\n            child.set_ce_policy(policy)\n'
_CE_YIELD = """                    from ..train.native_supervision import apply_policy
                    graph = row['models'][self.pair_model]['graph'] if self.kind == 'semantic-pairs' else row
                    yield apply_policy(example, graph, source_sha256=self.file_sha256,
                                       policy=getattr(self, 'ce_policy', None))"""


def prove_ce_objective_extension(old_package, new_package):
    """Certify only the exact iteration-time CE adapter; index semantics stay fixed."""
    old, new = map(Path, (old_package, new_package))
    inventories = [{name: sha256_file_hex(root / name) for name in POLICY_FILES}
                   for root in (old, new)]
    if digest(inventories[0]) != 'e6fce2c3752397ef71dc09a8b8689345583b2d99c2a79d8a6731f0cd7b3064e7':
        raise ValueError('CE continuation requires the exact reviewed classified predecessor policy')
    if any(inventories[0][name] != inventories[1][name]
           for name in POLICY_FILES if name != 'data/native_sources.py'):
        raise ValueError('CE continuation changed another intrinsic policy file')
    before = (old / 'data/native_sources.py').read_text()
    after = (new / 'data/native_sources.py').read_text()
    if (after.count(_CE_SETTER) != 1 or after.count(_CE_YIELD) != 1
            or after.replace(_CE_SETTER, '', 1).replace(_CE_YIELD, '                    yield example', 1) != before):
        raise ValueError('source adapter is not the exact reviewed CE-only extension')
    producer = 'train/native_source_classification.py'
    if not (sha256_file_hex(old / producer) == sha256_file_hex(new / producer)
            == '636944d3ea0efb87dc657dab9caed66f05d32777f0ea944480ba0eed7bd261d9'):
        raise ValueError('CE continuation changed classification verifier')
    return {'qualification': 'source-bound-ce-objective-only-20261020',
            'policy_files': inventories, 'old_policy_sha256': digest(inventories[0]),
            'new_policy_sha256': digest(inventories[1]),
            'old_package': str(old), 'new_package': str(new),
            'classification_producer_sha256': sha256_file_hex(new / producer)}


def continue_classified_objective_index(source, old_index, *, old_package, new_package):
    """Rekey one explicit authenticated classified witness without reading graph JSON."""
    def binding(path):
        path = Path(path)
        fields = ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')
        state = path.stat()
        before = tuple(getattr(state, k) for k in fields)
        sha = sha256_file_hex_cached(path, trust_archive_receipt=False)
        state = path.stat()
        after = tuple(getattr(state, k) for k in fields)
        if before != after:raise ValueError('objective continuation file changed while hashing')
        return {'path': str(path), 'sha256': sha, 'bytes': after[2]}
    def checked(ref):
        actual = binding(ref['path'])
        if actual != ref:
            raise ValueError('objective continuation witness changed')
        return Path(ref['path'])
    qualification = prove_ce_objective_extension(old_package, new_package)
    path, index_path = checked(source), checked(old_index)
    value = row_index.load(index_path)
    if value is None:
        raise ValueError('objective continuation needs a valid own index')
    key, entries = value.get('key'), value.get('entries')
    if (not isinstance(key, dict) or set(key) != {'source_sha256', 'tokenizer', 'split',
            'policy_sha256', 'kind', 'pair_model', 'role_ids'}
            or key['source_sha256'] != source['sha256']
            or key['policy_sha256'] != qualification['old_policy_sha256']
            or key['kind'] not in ('native-graphs', 'semantic-pairs')
            or key['split'] not in ('train', 'held', 'test', 'validation')
            or (key['kind'] == 'native-graphs' and key['pair_model'] is not None)
            or (key['kind'] == 'semantic-pairs' and not isinstance(key['pair_model'], str))
            or not isinstance(entries, list) or not entries
            or value.get('metadata_sha256') != digest(entries)):
        raise ValueError('objective continuation source/key/metadata differs')
    seen = set()
    for entry in entries:
        if (not isinstance(entry.get('id'), str) or entry['id'] in seen
                or any(type(entry.get(field)) is not bool for field in
                    ('communication', 'producer_consumer', 'external_state_consumption'))
                or entry['communication'] != (entry['producer_consumer'] or entry['external_state_consumption'])):
            raise ValueError('objective continuation needs complete exact classification')
        seen.add(entry['id'])
    new_key = {**key, 'policy_sha256': qualification['new_policy_sha256']}
    result = {**value, 'key': new_key}
    target = row_index.entry_path('native-graphs', digest(new_key))
    # Recheck all authority immediately before publishing; source bytes are hashed, never parsed.
    checked(source); checked(old_index)
    if prove_ce_objective_extension(old_package, new_package) != qualification:
        raise ValueError('objective continuation package changed')
    _publish_index_without_replacement(target, result)
    if row_index.load(target) != result:
        raise ValueError('objective continuation destination differs')
    checked(source); checked(old_index)
    if prove_ce_objective_extension(old_package, new_package) != qualification:
        raise ValueError('objective continuation package changed after publication')
    return {'source': source, 'old_index': old_index, 'new_index': binding(target),
            'rows': len(entries), 'producer': binding(Path(new_package) / 'train/native_source_classification.py'),
            'old_policy': {name: binding(Path(old_package) / name) for name in POLICY_FILES},
            'graph_position_expansions': 0, 'optimizer_updates': 0, 'ordinary_partition': None,
            'objective_policy_continuation': {'qualification': qualification,
                'producer': binding(__file__), 'entries_unchanged': True,
                'metadata_sha256': value['metadata_sha256'], 'raw_graph_parses': 0}}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-package', type=Path, required=True)
    parser.add_argument('--new-package', type=Path, required=True)
    parser.add_argument('--source-sha256', action='append', required=True)
    parser.add_argument('--qualification', choices=('prefix-extension', 'complete-native-runtime', 'source-metadata-revalidation'),
                        default='prefix-extension')
    parser.add_argument('--reference-package', type=Path, help='exact canonical64ec package required for metadata-only continuation')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prove = (prove_extension if args.qualification == 'prefix-extension'
             else prove_reviewed_runtime_extension)
    if args.qualification == 'source-metadata-revalidation':
        if args.reference_package is None:parser.error('metadata continuation requires --reference-package')
        proof=prove_metadata_revalidation_extension(args.old_package,args.new_package,args.reference_package)
    else:proof=prove(args.old_package,args.new_package)
    receipt = continue_indexes(proof, args.source_sha256)
    write_json_durable(args.output, receipt)
    print(json.dumps({'indexes': len(receipt['indexes']), 'missing_sources': len(receipt['missing_source_sha256']),
                      'receipt': str(args.output)}))


if __name__ == '__main__':
    main()
