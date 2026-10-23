"""Exact metadata reuse for the staged-path reader extension, without row replay.

The supervisor selects a continuation. This verifier accepts only unchanged row
parsers, one exact path-routing extension and a hash-reference helper unrelated to row classification.
Every cache entry, topology class and source byte binding remains unchanged.
"""
from __future__ import annotations

import ast
import copy
from pathlib import Path

from natlang_neuralese.common.hashing import canonical_json_sha256_hex as digest, sha256_file_hex
from natlang_neuralese.common import row_index
from natlang_neuralese.train import native_source_classification as classification
from scripts.neuralese_data.continue_native_source_indexes import _publish_index_without_replacement

# AST identities of the reviewed staged-path mechanism; formatting is immaterial.
_HELPERS = {'_staged_path': 'a2a0a7a3ff30283e42eb481817d16f8d96bccd1a5c6097d3309258ed6858885e', '_source_identity': '0523f6823add08e3ac221fc45ded0538c4fece04621702d05ce896164387f88d', 'sha256_reference': 'b645f1e1b181829eb9a3b70da2ab5ba61b2b01975a792036f24983153793f214'}


def _tree(node):
    return ast.dump(node, include_attributes=False)


def _module(path):
    return ast.parse(Path(path).read_text())


def prove_source_routing_extension(old_package, new_package):
    old, new = map(Path, (old_package, new_package))
    policy = [{name: sha256_file_hex(p / name) for name in classification.POLICY_FILES}
              for p in (old, new)]
    changed = {'data/native_sources.py', 'common/hashing.py'}
    if any(policy[0][name] != policy[1][name] for name in classification.POLICY_FILES if name not in changed):
        raise ValueError('source routing continuation changed row parser or index semantics')
    counts = {'import': 0, 'identity': 0, 'receipt': 0}
    import_node = _tree(ast.parse('from .native_source_tree import _source_identity').body[0])
    assignment = _tree(ast.parse('self.source_identity_path = _source_identity(self.path)').body[0])
    receipt = _tree(ast.parse('self.source_identity_path', mode='eval').body)

    class NormalizeSource(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            if _tree(node) == import_node:
                counts['import'] += 1
                return None
            return node

        def visit_Assign(self, node):
            if _tree(node) == assignment:
                counts['identity'] += 1
                return None
            return self.generic_visit(node)

        def visit_Attribute(self, node):
            if _tree(node) == receipt:
                counts['receipt'] += 1
                return ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='path', ctx=ast.Load())
            return self.generic_visit(node)

    normalized = NormalizeSource().visit(_module(new / 'data/native_sources.py'))
    if counts != {'import': 1, 'identity': 1, 'receipt': 1} or _tree(normalized) != _tree(_module(old / 'data/native_sources.py')):
        raise ValueError('source routing continuation changed source parsing, masks or row identities')

    def remove_helpers(module, names):
        retained, found = [], set()
        for node in module.body:
            if isinstance(node, ast.FunctionDef) and node.name in names:
                if node.name in found or digest(_tree(node)) != _HELPERS[node.name]:
                    raise ValueError('source routing continuation changed a reviewed helper')
                found.add(node.name)
            else:
                retained.append(node)
        if found != set(names):
            raise ValueError('source routing continuation lost an exact reviewed helper')
        module.body = retained
        return module

    hashing = remove_helpers(_module(new / 'common/hashing.py'), {'sha256_reference'})
    if _tree(hashing) != _tree(_module(old / 'common/hashing.py')):
        raise ValueError('source routing continuation changed hashing semantics')
    routes = remove_helpers(_module(new / 'data/native_source_tree.py'), {'_staged_path', '_source_identity'})
    routed = _tree(ast.parse("_staged_path(source.path, binding['path'])", mode='eval').body)
    path = ast.parse("Path(binding['path'])", mode='eval').body
    replacements = []

    class NormalizeRouting(ast.NodeTransformer):
        def visit_Call(self, node):
            if _tree(node) == routed:
                replacements.append(True)
                return copy.deepcopy(path)
            return self.generic_visit(node)

    routes = NormalizeRouting().visit(routes)
    if len(replacements) != 1 or _tree(routes) != _tree(_module(old / 'data/native_source_tree.py')):
        raise ValueError('source routing continuation changed manifest admission')
    return {'schema': 'natlang.native-source-routing-qualification/1',
            'old_policy_sha256': digest(policy[0]), 'new_policy_sha256': digest(policy[1]),
            'policy_files': policy, 'row_metadata_unchanged': True,
            'routing_helpers': dict(_HELPERS),
            'source_tree': [classification._binding(p / 'data/native_source_tree.py') for p in (old, new)],
            'verifier': classification._binding(__file__)}


def continue_classified_index(source_reference, prior_classification, *, old_package, new_package, split='indexes'):
    """Reuse only the exact index authenticated by the caller's prior recipe proof.

    The production operator supplies the existing resident plan's immutable proof
    reference. A cache path or a newly computed cache hash is never authority.
    """
    qualification = prove_source_routing_extension(old_package, new_package)
    if split not in ('indexes', 'held_indexes'):
        raise ValueError('source routing continuation requires the exact train/held member role')
    source = classification._checked(source_reference)
    source_ref = classification._binding(source)
    proof_path = classification._checked(prior_classification)
    prior_ref = classification._binding(proof_path)
    import json
    prior = json.loads(proof_path.read_bytes())
    if prior.get('schema') != classification.SCHEMA:
        raise ValueError('source routing continuation requires the authenticated prior classification proof')
    matches = [part for part in prior.get(split, [])
               if part.get('source') == source_ref]
    if len(matches) != 1:
        raise ValueError('source routing continuation source is not one exact prior proof member')
    previous = matches[0]
    producer = previous.get('producer') or {}
    expected_producer = classification._binding(Path(old_package) / 'train/native_source_classification.py')
    if any(producer.get(k) != expected_producer[k] for k in ('sha256', 'bytes')):
        raise ValueError('source routing continuation prior producer differs from its exact old package')
    classification._checked(producer)
    old_ref = previous['new_index']
    old_index = classification._checked(old_ref)
    value = classification._index(old_ref)
    if len(value['entries']) != previous['rows']:
        raise ValueError('source routing continuation index differs from prior proof row accounting')
    key = value['key']
    if (set(key) != {'source_sha256', 'tokenizer', 'split', 'policy_sha256', 'kind', 'pair_model', 'role_ids'}
            or key['source_sha256'] != source_ref['sha256']
            or key['policy_sha256'] != qualification['old_policy_sha256']
            or key['kind'] not in ('native-graphs', 'semantic-pairs')
            or not value['entries']
            or any(any(type(entry.get(k)) is not bool for k in classification.CLASS_FIELDS)
                   or entry['communication'] != (entry['producer_consumer'] or entry['external_state_consumption'])
                   for entry in value['entries'])):
        raise ValueError('source routing continuation lacks a complete current classified source index')
    new_key = {**key, 'policy_sha256': qualification['new_policy_sha256']}
    if new_key == key:
        raise ValueError('source routing continuation needs its distinct reader policy')
    target = row_index.entry_path('native-graphs', digest(new_key))
    result = {**value, 'key': new_key}
    _publish_index_without_replacement(target, result)
    if (row_index.load(target) != result or classification._binding(source) != source_ref
            or classification._binding(old_index) != old_ref
            or classification._binding(proof_path) != prior_ref
            or prove_source_routing_extension(old_package, new_package) != qualification):
        raise ValueError('source routing continuation changed during publication')
    return {'source': source_ref, 'old_index': old_ref, 'new_index': classification._binding(target),
            'rows': len(value['entries']), 'producer': classification._binding(Path(new_package) / 'train/native_source_classification.py'),
            'old_policy': {name: classification._binding(Path(old_package) / name) for name in classification.POLICY_FILES},
            'graph_position_expansions': 0, 'optimizer_updates': 0, 'ordinary_partition': None,
            'source_routing_continuation': {'qualification': qualification,
                'metadata_sha256': value['metadata_sha256'], 'entries_unchanged': True,
                'raw_graph_parses': 0, 'prior_classification': prior_ref, 'prior_split': split}}
