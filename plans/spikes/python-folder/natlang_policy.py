"""Spike: natlang's code policy for Python. Source is checked and lowered before it runs:
- no `while`, no function that calls itself (directly or through another in the same module);
- no exec/eval/compile/__import__/importlib, and no modules that reach the host or load code around the policy;
- every `for` and comprehension iterates through __natlang_finite: concrete collections pass, anything else is taken
  up to a cap and raises beyond it;
- endless itertools iterators and iter(callable, sentinel) are Python generators, so the timeout can stop C code that
  drains them.
Local modules imported from the folder get the same treatment through an import hook."""
import ast, builtins, importlib.abc, importlib.util, itertools, sys

CAP = 100_000
FORBIDDEN_CALLS = {'exec', 'eval', 'compile', '__import__', 'breakpoint'}
# Modules that reach the host or load code around the policy. Network is not among the reasons: it follows the
# runtime's network setting, as fetch in eval does.
FORBIDDEN_MODULES = {'js', 'pyodide_js', 'pyodide.ffi', 'micropip', 'importlib', 'ctypes'}
CONCRETE = (list, tuple, str, bytes, bytearray, dict, set, frozenset, range, type({}.keys()), type({}.values()), type({}.items()))

class PolicyError(SyntaxError): pass

def finite(iterable):
    if isinstance(iterable, CONCRETE) or type(iterable).__module__.split('.')[0] in ('numpy', 'pandas'):
        return iterable
    items = list(itertools.islice(iterable, CAP + 1))
    if len(items) > CAP:
        raise RuntimeError(f'this loop would take more than {CAP} items from a {type(iterable).__name__}; '
                           'repeat an open-ended process with iterate_on(step, initial).until(done)')
    return items

def check(tree, filename):
    problems = []
    report = (lambda node, message: problems.append(f'{filename}:{node.lineno}: {message}'))
    calls = {}                                  # function name -> names it calls
    class Visitor(ast.NodeVisitor):
        def __init__(self): self.function = []
        def visit_While(self, node): report(node, '`while` loops are not allowed; use `for` over a collection, or iterate_on(step, initial).until(done) for an open-ended process')
        def visit_FunctionDef(self, node):
            self.function.append(node.name); calls.setdefault(node.name, set()); self.generic_visit(node); self.function.pop()
        visit_AsyncFunctionDef = visit_FunctionDef
        def visit_Call(self, node):
            if isinstance(node.func, ast.Name):
                if node.func.id in FORBIDDEN_CALLS: report(node, f'`{node.func.id}` is not allowed')
                if self.function: calls[self.function[-1]].add(node.func.id)
            self.generic_visit(node)
        def visit_Import(self, node):
            for alias in node.names: self.module(node, alias.name)
        def visit_ImportFrom(self, node): self.module(node, node.module or '')
        def module(self, node, name):
            if any(name == m or name.startswith(m + '.') for m in FORBIDDEN_MODULES): report(node, f'module `{name}` is not available')
    Visitor().visit(tree)
    # Recursion: a cycle in the call graph among this module's functions.
    def reaches(start, target, seen):
        return any(callee == target or (callee in calls and callee not in seen and reaches(callee, target, seen | {callee})) for callee in calls.get(start, ()))
    for name in calls:
        if reaches(name, name, {name}):
            problems.append(f'{filename}: `{name}` calls itself; recursion is not allowed, use a loop over a work list')
    if problems: raise PolicyError('\n'.join(problems))

class Lower(ast.NodeTransformer):
    def wrap(self, node): return ast.Call(ast.Name('__natlang_finite', ast.Load()), [node], [])
    def visit_For(self, node): self.generic_visit(node); node.iter = self.wrap(node.iter); return node
    visit_AsyncFor = visit_For
    def visit_comprehension(self, node): self.generic_visit(node); node.iter = self.wrap(node.iter); return node

def prepare(source, filename='<cell>'):
    tree = ast.parse(source, filename)
    check(tree, filename)
    return ast.fix_missing_locations(Lower().visit(tree))

class LocalModules(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Modules found under the working folder are checked and lowered like cells."""
    def __init__(self, root): self.root = root
    def find_spec(self, name, path, target=None):
        import os
        candidate = os.path.join(self.root, *name.split('.')) + '.py'
        return importlib.util.spec_from_file_location(name, candidate, loader=self) if os.path.exists(candidate) else None
    def create_module(self, spec): return None
    def exec_module(self, module):
        module.__dict__.update(__builtins__=USER_BUILTINS, __natlang_finite=finite)
        _exec(_compile(prepare(open(module.__spec__.origin).read(), module.__spec__.origin), module.__spec__.origin, 'exec'), module.__dict__)

# The policy binds user code only: library code (the standard library, pandas) keeps the real builtins, which it uses
# (dataclasses runs exec, for one), and Pyodide's own modules stay importable for the libraries that need them.
_exec, _compile = builtins.exec, builtins.compile

# Endless iterators written in C can be drained by C code without running a line of Python (sum(itertools.count()),
# any(itertools.repeat(0))), so the timeout's interrupt, which Python checks between instructions, never arrives and
# the interpreter hangs. User code gets the same iterators as Python generators, which the timeout stops.
def _count(start=0, step=1):
    while True:
        yield start
        start += step

def _cycle(iterable):
    saved = []
    for item in iterable:
        yield item
        saved.append(item)
    while saved:
        for item in saved: yield item

def _repeat(object, times=None):
    if times is not None: return itertools.repeat(object, times)
    return _forever(object)

def _forever(item):
    while True: yield item

def _until(call, sentinel):
    while True:
        value = call()
        if value == sentinel: return
        yield value

def _iter(source, *sentinel):
    if not sentinel: return builtins.iter(source)
    if not callable(source): raise TypeError('iter(v, w): v must be callable')
    return _until(source, *sentinel)

USER_ITERTOOLS = type(itertools)('itertools', itertools.__doc__)
USER_ITERTOOLS.__dict__.update({name: value for name, value in itertools.__dict__.items() if not name.startswith('__')},
                               count=_count, cycle=_cycle, repeat=_repeat)

def _import(name, globals=None, locals=None, fromlist=(), level=0):
    module = builtins.__import__(name, globals, locals, fromlist, level)
    return USER_ITERTOOLS if module is itertools else module

USER_BUILTINS = {name: value for name, value in builtins.__dict__.items() if name not in FORBIDDEN_CALLS - {'__import__'}}
USER_BUILTINS.update(__import__=_import, iter=_iter)

def fresh_namespace():
    return {'__builtins__': USER_BUILTINS, '__natlang_finite': finite}

def install(root):
    sys.meta_path.insert(0, LocalModules(root))

def run_cell(source, namespace):
    _exec(_compile(prepare(source), '<cell>', 'exec'), namespace)
