"""Exact AST identities for named source laws, independent of unrelated CLI code."""
from __future__ import annotations

import ast
import builtins
import hashlib
import json
import symtable
from pathlib import Path


def scoped_python_policy(path, functions):
    """Bind selected functions and their recursively resolved module globals.

    This is an identity, not a file binding. Callers retain physical provenance
    separately. Imported modules must also be authenticated by their existing
    external dependency bindings; this records the exact alias/import authority.
    """
    path = Path(path)
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    stamp = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if stamp(before) != stamp(after):
        raise ValueError('source policy changed while reading')
    return scoped_python_policy_bytes(raw, functions, filename=str(path))


def scoped_python_policy_bytes(raw, functions, *, filename='<bound-source-policy>'):
    tree = ast.parse(raw, filename=filename)
    symbols = symtable.symtable(raw.decode(), filename, 'exec')
    functions = tuple(sorted(set(functions)))
    definitions = {}
    tables = {child.get_name(): child for child in symbols.get_children()}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            definitions.setdefault(node.name, []).append(node)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                name = alias.asname or (alias.name.split('.')[0] if isinstance(node, ast.Import) else alias.name)
                # Only the selected alias is relevant, not unrelated names on
                # the same import line. Preserve module/level semantics.
                binding = ast.Import(names=[alias]) if isinstance(node, ast.Import) else ast.ImportFrom(module=node.module, names=[alias], level=node.level)
                definitions.setdefault(name, []).append(binding)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                for child in ast.walk(target):
                    if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                        definitions.setdefault(child.id, []).append(node)
    pending = list(functions)
    selected = {}
    built_in = set(dir(builtins))
    used_builtins = set()

    def global_names(table):
        names = {s.get_name() for s in table.get_symbols() if s.is_referenced() and s.is_global()}
        for child in table.get_children():
            names.update(global_names(child))
        return names

    effects = {}
    while True:
        while pending:
            name = pending.pop()
            if name in selected:
                continue
            nodes = definitions.get(name)
            if nodes is None:
                if name in built_in:
                    used_builtins.add(name)
                    continue
                if name in {'__name__', '__file__', '__package__'}:
                    continue
                raise ValueError('unbound source policy global: ' + name)
            selected[name] = [ast.dump(n, include_attributes=False) for n in nodes]
            for node in nodes:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    if name not in tables:
                        raise ValueError('missing source policy symbol table: ' + name)
                    pending.extend(global_names(tables[name]))
                    # Decorators/defaults execute in the module scope.
                    extra = list(node.decorator_list)
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        extra += list(node.args.defaults) + [n for n in node.args.kw_defaults if n is not None]
                        extra += [a.annotation for a in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs) if a.annotation is not None]
                        if node.returns is not None: extra.append(node.returns)
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    extra = []
                else:
                    extra = [node.value] if getattr(node, 'value', None) is not None else []
                for expression in extra:
                    pending.extend(n.id for n in ast.walk(expression) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load))
        discovered = False
        for index, node in enumerate(tree.body):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)):
                continue
            if isinstance(node, ast.If) and ast.dump(node.test, include_attributes=False) == ast.dump(ast.parse("__name__ == '__main__'", mode='eval').body, include_attributes=False):
                continue
            names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
            loads = {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
            stores = {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
            calls = [n for n in ast.walk(node) if isinstance(n, ast.Call)]
            if any(isinstance(n.func, ast.Name) and n.func.id in {'globals', 'locals', 'exec', 'eval'} for n in calls):
                raise ValueError('dynamic module namespace is not a scoped source policy')
            control = isinstance(node, (ast.If, ast.Try, ast.For, ast.While, ast.With, ast.Match))
            if control and (names & selected.keys() or stores & used_builtins):
                raise ValueError('conditional module mutation is not a scoped source policy')
            if control:
                continue
            # All module-level calls are observable initialization. Traverse
            # their callees/arguments too, including indirect mutator functions.
            # Simple aliases are followed when they touch the selected graph.
            if not calls and not names & selected.keys():
                continue
            dumped = ast.dump(node, include_attributes=False)
            effects[index] = dumped
            for name in loads | (stores if names & selected.keys() else set()):
                if name not in selected and name not in used_builtins and name not in {'__name__', '__file__', '__package__'}:
                    pending.append(name); discovered = True
        if not discovered:
            break
    for name in functions:
        if name not in selected or not any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in definitions[name]):
            raise ValueError('source policy entry must be a function: ' + name)
    value = {'schema': 'natlang.scoped-python-source-policy/1', 'functions': list(functions),
             'definitions': dict(sorted(selected.items())), 'module_effects': [effects[index] for index in sorted(effects)]}
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
    return {'schema': value['schema'], 'functions': list(functions),
            'policy_sha256': hashlib.sha256(encoded).hexdigest()}


def raw_protection_policy(path):
    return scoped_python_policy(path, ('current_protection', 'file_rows'))
