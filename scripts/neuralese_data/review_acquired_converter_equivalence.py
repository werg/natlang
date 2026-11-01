"""Prove the narrowly reviewed full-session converter transition without running old code."""
import ast
import copy
from pathlib import Path


def _function(tree, name):
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def _dump(node):
    return ast.dump(node, include_attributes=False)


def _without_twin_assignment(function, *, historical, whole):
    function = copy.deepcopy(function)
    if historical:
        selected = "row.get('source_twin_id') or source_records" if whole else "row.get('source_twin_id') or row.get('source_record_ids') or [identifier]"
        statements = f"twin = {selected}\ntwin_id = canonical_json_sha256_hex(sorted(str(value) for value in twin))"
    else:
        fallback = 'source_records' if whole else "row.get('source_record_ids') or [identifier]"
        statements = f"twin = _normalized_twin_identity(row.get('source_twin_id'), fallback={fallback})\ntwin_id = canonical_json_sha256_hex(twin)"
    expected = [_dump(node) for node in ast.parse(statements).body]
    matches = [i for i in range(len(function.body)-1) if [_dump(node) for node in function.body[i:i+2]] == expected]
    if len(matches) != 1: raise ValueError('converter twin assignment differs from reviewed transition')
    i = matches[0]; del function.body[i:i+2]
    return function


def review(root, registry=None):
    from scripts.neuralese_data.build_acquired_streaming_successor import bind, checked, read
    root = Path(root)
    registry = Path(registry) if registry is not None else root/'training/source-reviews/acquired-converter-full-session-equivalence-20261101.json'
    if not registry.is_file(): return None
    value = read(registry)
    if value.get('schema') != 'natlang.acquired-converter-full-session-equivalence/1':
        raise ValueError('unrecognized converter equivalence review')
    files, trees = {}, {}
    for key in ('builder', 'conversion'):
        refs = value['files'][key]; old = checked(refs['historical'], 'historical converter source')
        current_ref = {**refs['current'], 'path': str(root/refs['current']['path'])}
        current = checked(current_ref, 'reviewed current converter source')
        files[key] = {'historical': bind(old), 'current': bind(current)}
        trees[key] = (ast.parse(old.read_text()), ast.parse(current.read_text()))
    a, b = [_function(tree, 'session_graphs') for tree in trees['builder']]
    if _dump(a) != _dump(b) or any(isinstance(node, ast.Constant) and node.value == 'source_twin_id' for node in ast.walk(b)):
        raise ValueError('full-session renderer changed or supplies a twin identity override')
    for name in ('text_row_graph', 'whole_trajectory_text_row_graph'):
        a, b = [_function(tree, name) for tree in trees['conversion']]
        if _dump(_without_twin_assignment(a, historical=True, whole=name.startswith('whole_'))) != _dump(
                _without_twin_assignment(b, historical=False, whole=name.startswith('whole_'))):
            raise ValueError('converter changes more than the reviewed absent-twin normalization')
    return {'registry': bind(registry), 'files': files,
            'proof': 'session_graphs AST identical and no twin override; both graph converters AST identical except exact reviewed twin assignments',
            'scope': 'committed full-session conversion receipts only; current graph parsing, screening and effective-loss verification remain required'}
