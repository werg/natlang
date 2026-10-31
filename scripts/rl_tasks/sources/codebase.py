"""Codebase exploration tasks over real Python packages on disk (pinned installed versions; one split unit per
package, so related questions about one codebase stay on one split).

Natlang framing: the package is a `repo` file service over a snapshot asset bound by SHA-256 (`files()`, `read(path)`,
`search(pattern, flags?)`); the root function explores it with eval and may hand files to natural-language children.
The Neuralese framing adds a callable helper `read_file(question, path, text): Neuralese<string>`, a private per-file
view the root fans in. A package has tens to hundreds of files, so reading everything into one context is impractical.

Every answer is computed by a crisp analyzer at build time (Python `ast` over the same snapshot):
  callers      functions/methods whose body calls something named X (`X(...)` or `obj.X(...)`)
  importers    files that import module M (absolute, or relative resolved against the package)
  definition   the file that assigns the module-level constant NAME (unique in the package)
  subclasses   classes whose bases name C directly
  raisers      functions/methods whose body raises exception class E (`raise E` or `raise E(...)`)
  effects      functions/methods that directly perform an effect of kind K (file writes/deletes, subprocess,
               network, environment changes), from a closed call/attribute list
  warnings     functions/methods that issue a DeprecationWarning (`warnings.warn(..., DeprecationWarning ...)`)
  mutation     the one function whose comparison operator was flipped in this copy (the service serves the altered
               file); answer "path:name"
Functions are named "path:qualname" (path relative to the package root; qualname joins enclosing classes and
functions with dots). Oracle: answer-list (order-free) for lists, normalized exact for single answers. Wrong-answer
controls: the gold with one item dropped, with a near-miss added (a definition of X, a function that only mentions
it), and other questions' answers."""
from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import json
import os
import random
import sysconfig
from collections import defaultdict
from pathlib import Path

from ..common import make_case, pick, resolve_str

V3_ASSETS = resolve_str('repo', 'runs', 'natlang-rl-broad-external-20261031-v3', 'assets')
BUILD_V3_ASSETS = resolve_str('data_nvme', 'broad-rl-tasks-20261031', 'assets-v3')
SITE_ROOTS = [sysconfig.get_paths()['stdlib'],
              resolve_str('repo', '.venv-neuralese', 'lib', 'python3.12', 'site-packages'),
              resolve_str('data_nvme', 'external-rl-sources', '.venv', 'lib', 'python3.12', 'site-packages')]
SKIP_DIRS = {'tests', 'test', 'testing', '_vendor', 'vendor', '__pycache__', 'benchmarks', 'examples'}
EFFECTS = {
    'file-write': {'calls': {'os.remove', 'os.unlink', 'os.rename', 'os.replace', 'os.rmdir', 'os.removedirs',
                             'os.makedirs', 'os.mkdir', 'shutil.rmtree', 'shutil.move', 'shutil.copy', 'shutil.copy2',
                             'shutil.copyfile', 'shutil.copytree', 'os.truncate', 'os.chmod'},
                   'methods': {'write_text', 'write_bytes', 'unlink', 'mkdir', 'rmdir', 'touch'},
                   'open_write': True,
                   'text': 'writes, creates, moves or deletes files or directories on disk'},
    'subprocess': {'calls': {'subprocess.run', 'subprocess.Popen', 'subprocess.call', 'subprocess.check_call',
                             'subprocess.check_output', 'os.system', 'os.popen', 'os.execv', 'os.execvp', 'os.spawnv'},
                   'methods': set(), 'open_write': False, 'text': 'starts another process or runs a shell command'},
    'network': {'calls': {'socket.socket', 'socket.create_connection', 'urllib.request.urlopen', 'urlopen',
                          'http.client.HTTPConnection', 'http.client.HTTPSConnection', 'requests.get', 'requests.post',
                          'requests.request', 'httpx.get', 'httpx.post'},
                'methods': set(), 'open_write': False, 'text': 'opens a network connection or makes an HTTP request'},
    'environment': {'calls': {'os.putenv', 'os.unsetenv', 'os.chdir', 'sys.exit', 'os._exit'},
                    'methods': set(), 'open_write': False,
                    'text': 'changes the process environment or working directory, or exits the process'},
}
INSTRUCTIONS = {
    'callers': ['repo is a Python codebase served file by file. Find every function or method whose own body calls '
                'something named {x} (as {x}(...) or as obj.{x}(...)). Return them as "path:name" strings, path relative '
                'to the codebase root and name prefixed by enclosing classes or functions with dots (Class.method).',
                'Which functions in repo call {x}? Count a call written {x}(...) or something.{x}(...) inside the '
                'function\'s own body (not in a nested function). Return each as "path:qualified.name".'],
    'importers': ['repo is a Python codebase. List every file that imports the module {x} (an import statement '
                  'naming it, absolutely or relatively). Return the file paths relative to the codebase root.',
                  'Which files of repo import {x}? Return their paths (relative to the root), one per file.'],
    'definition': ['repo is a Python codebase. Find the file that defines the module-level name {x} by assigning it '
                   '(for example {x} = ...). Return that file\'s path relative to the codebase root.',
                   'Where in repo is {x} assigned at module level? Return the path of that file.'],
    'subclasses': ['repo is a Python codebase. List every class whose base classes directly name {x}. Return each as '
                   '"path:ClassName" (with enclosing class names joined by dots for a nested class).',
                   'Which classes in repo inherit directly from {x}? Return them as "path:ClassName" strings.'],
    'raisers': ['repo is a Python codebase. Find every function or method whose own body raises {x} (raise {x} or '
                'raise {x}(...)). Return them as "path:qualified.name" strings.',
                'List the functions of repo that raise {x} themselves (not only through a call). Name each '
                '"path:Class.method" or "path:function".'],
    'effects': ['repo is a Python codebase. Find every function or method whose own body {x}. Return them as '
                '"path:qualified.name" strings.',
                'Which functions in repo directly {x}? Count only effects the function performs itself, not through '
                'calls to other functions of repo. Return "path:qualified.name" for each.'],
    'warnings': ['repo is a Python codebase. Find every function or method that issues a DeprecationWarning itself '
                 '(a warnings.warn call with that category). Return them as "path:qualified.name" strings.'],
    'mutation': ['repo is a Python codebase in which exactly one function was altered: one comparison operator in it '
                 'was replaced by another (for example < became >=), so it no longer does what its name, docstring '
                 'and callers expect. Find that function and return it as "path:qualified.name".'],
}
NEURALESE = {
    'helper': 'read_file', 'args': {'question': 'string', 'path': 'string', 'text': 'string'},
    'returns': 'Neuralese<string>',
    'instruction': 'Read text, the file at path in a codebase, for what question needs, and write a private note: the '
                   'relevant definitions, calls or lines with their qualified names, or that the file has none.',
    'root_note': 'read_file gives a private note on one file. You may narrow the candidate files in eval, read each '
                 'with read_file and combine the notes with your own exact checks before answering.',
}
SERVICE = r'''import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
const FILE = __FILE__;
const SHA256 = __SHA__;
const PATCH: Record<string, string> = __PATCH__;
function snapshot(): Record<string, string> {
  const cache = ((globalThis as any).__natlangRlRepos ??= new Map());
  if (!cache.has(SHA256)) {
    const bytes = readFileSync((process.env.NATLANG_RL_TASK_ASSETS_V3 ?? __ROOT__) + '/' + FILE);
    if (createHash('sha256').update(bytes).digest('hex') !== SHA256) throw new Error('repo asset changed: ' + FILE);
    cache.set(SHA256, JSON.parse(bytes.toString('utf8')));
  }
  return { ...cache.get(SHA256), ...PATCH };
}
/** Paths of every file in the codebase, relative to its root, sorted. */
export function files(): string[] { return Object.keys(snapshot()).sort(); }
/** The full text of one file. */
export function read(path: string): string {
  const text = snapshot()[path];
  if (text === undefined) throw new Error(`no file ${path}; files() lists the paths`);
  return text;
}
/** Every line matching a regular expression (JavaScript syntax; flags such as "i"), with its file and 1-based line. */
export function search(pattern: string, flags = ''): { path: string; line: number; text: string }[] {
  const re = new RegExp(pattern, flags.replace(/g/g, ''));
  const out: { path: string; line: number; text: string }[] = [];
  const all = snapshot();
  for (const path of Object.keys(all).sort()) {
    const lines = all[path]!.split('\n');
    for (let i = 0; i < lines.length; i++) if (re.test(lines[i]!)) out.push({ path, line: i + 1, text: lines[i]! });
  }
  return out;
}
'''


def _packages():
    """Distinct packages (first root wins), 10..400 Python files outside tests, under 6 MB of source."""
    seen, out = set(), []
    for root in SITE_ROOTS:
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            path = os.path.join(root, name)
            if (not os.path.isdir(path) or name.startswith(('_', '.')) or name.endswith(('dist-info', 'egg-info'))
                    or name in ('site-packages', 'idlelib', 'lib2to3', 'encodings', 'cuda', 'tzdata') or name in seen):
                continue
            files = _files(path)
            size = sum(len(t) for t in files.values())
            if 10 <= len(files) <= 400 and size < 6_000_000:
                seen.add(name)
                out.append((name, root, path, files))
    return out


def _files(path):
    out = {}
    for dirpath, dirnames, filenames in os.walk(path):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith('.'))
        for filename in sorted(filenames):
            if filename.endswith('.py'):
                full = os.path.join(dirpath, filename)
                try:
                    text = Path(full).read_text(encoding='utf-8')
                except (UnicodeDecodeError, OSError):
                    continue
                out[os.path.relpath(full, path)] = text
    return out


def _version(name, root):
    """(version, license) of the installed distribution that owns the package directory."""
    if root == SITE_ROOTS[0]:
        import platform
        return f'python-stdlib-{platform.python_version()}', 'PSF-2.0'
    for dist in importlib.metadata.distributions(path=[root]):
        top = (dist.read_text('top_level.txt') or '').split()
        if name in top or any(str(f).startswith(name + '/') for f in (dist.files or [])[:50]):
            meta = dist.metadata
            license_ = meta.get('License-Expression') or (meta.get('License') or '').split('\n')[0][:80] or next(
                (c.split('::')[-1].strip() for c in meta.get_all('Classifier') or [] if c.startswith('License ::')), '')
            return dist.version, license_ or 'not stated in package metadata'
    return 'unknown', 'not stated in package metadata'


def _dotted(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return '.'.join(reversed(parts))
    return None


class _Scan(ast.NodeVisitor):
    """Per function (own body only, nested defs excluded): called names, raised names, effect calls, deprecations;
    per class: base names; per module: imports and top-level assignments."""

    def __init__(self, path, package):
        self.path, self.package, self.stack = path, package, []
        self.functions, self.classes, self.imports, self.assigns = {}, {}, set(), set()
        self.duplicate = set()

    def _qual(self, name):
        return '.'.join([*self.stack, name])

    def _function(self, node):
        qual = self._qual(node.name)
        key = f'{self.path}:{qual}'
        if key in self.functions:
            self.duplicate.add(key)
        info = {'calls': set(), 'raises': set(), 'effects': set(), 'deprecation': False, 'node': node}
        self.functions[key] = info
        self.stack.append(node.name)
        for child in ast.iter_child_nodes(node):
            self._own(child, info)
        self.stack.pop()

    def _own(self, node, info):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            self.visit(node)
            return
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            self.visit(node) if not isinstance(node, ast.Lambda) else None
            return
        if isinstance(node, ast.Call):
            dotted = _dotted(node.func)
            name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else None
            if name:
                info['calls'].add(name)
            for kind, spec in EFFECTS.items():
                if dotted in spec['calls'] or (isinstance(node.func, ast.Attribute) and node.func.attr in spec['methods']
                                               and dotted not in ('str.' + node.func.attr,)):
                    info['effects'].add(kind)
                if spec['open_write'] and dotted in ('open', 'io.open', 'os.open', 'codecs.open'):
                    mode = node.args[1] if len(node.args) > 1 else next((k.value for k in node.keywords if k.arg == 'mode'), None)
                    if isinstance(mode, ast.Constant) and isinstance(mode.value, str) and set(mode.value) & set('wax+'):
                        info['effects'].add(kind)
            if dotted in ('warnings.warn', 'warn') and any(
                    _dotted(a) in ('DeprecationWarning', 'builtins.DeprecationWarning') for a in
                    list(node.args[1:]) + [k.value for k in node.keywords if k.arg == 'category']):
                info['deprecation'] = True
        if isinstance(node, ast.Raise) and node.exc is not None:
            target = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
            dotted = _dotted(target)
            if dotted:
                info['raises'].add(dotted.split('.')[-1])
        for child in ast.iter_child_nodes(node):
            self._own(child, info)

    def visit_FunctionDef(self, node):
        self._function(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        key = f'{self.path}:{self._qual(node.name)}'
        if key in self.classes:
            self.duplicate.add(key)
        self.classes[key] = {_dotted(b).split('.')[-1] for b in node.bases if _dotted(b)}
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_Import(self, node):
        for alias in node.names:
            self.imports.add(alias.name)

    def visit_ImportFrom(self, node):
        base = node.module or ''
        if node.level:
            parts = self.package.split('.')[: len(self.package.split('.')) - node.level + 1]
            base = '.'.join(p for p in parts + ([node.module] if node.module else []) if p)
        if base:
            self.imports.add(base)
        for alias in node.names:
            self.imports.add(f'{base}.{alias.name}' if base else alias.name)


def _analyze(name, files):
    scans = {}
    for path, text in files.items():
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError):
            continue
        module = '.'.join([name, *path[:-3].split(os.sep)]).removesuffix('.__init__')
        scan = _Scan(path, module if path.endswith('__init__.py') else module.rsplit('.', 1)[0])
        for node in tree.body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for t in targets:
                    if isinstance(t, ast.Name) and t.id.isupper() and len(t.id) > 3:
                        scan.assigns.add(t.id)
        scan.visit(tree)
        scans[path] = (scan, tree, module)
    return scans


def _questions(name, files, scans, rng):
    """(kind, x, answer, near_misses) for one package."""
    functions = {k: v for scan, _, _ in scans.values() for k, v in scan.functions.items() if k not in scan.duplicate}
    classes = {k: v for scan, _, _ in scans.values() for k, v in scan.classes.items() if k not in scan.duplicate}
    defined = defaultdict(set)
    for key in functions:
        defined[key.rsplit(':', 1)[1].split('.')[-1]].add(key)
    out = []
    callers = defaultdict(set)
    for key, info in functions.items():
        for called in info['calls']:
            callers[called].add(key)
    for x, keys in callers.items():
        if x in defined and 2 <= len(keys) <= 10 and not x.startswith('__') and len(x) > 3:
            out.append(('callers', x, sorted(keys), sorted(defined[x] - keys)[:1]))
    importers = defaultdict(set)
    modules = {module for _, _, module in scans.values()}
    for path, (scan, _, _) in scans.items():
        for imported in scan.imports:
            if imported in modules and imported != name:
                importers[imported].add(path)
    for x, paths in importers.items():
        if 2 <= len(paths) <= 10:
            out.append(('importers', x, sorted(paths), []))
    owners = defaultdict(set)
    for path, (scan, _, _) in scans.items():
        for assigned in scan.assigns:
            owners[assigned].add(path)
    for x, paths in owners.items():
        if len(paths) == 1:
            mentions = [p for p, t in files.items() if x in t and p not in paths]
            if mentions:
                out.append(('definition', x, sorted(paths)[0], mentions[:1]))
    subclasses = defaultdict(set)
    for key, bases in classes.items():
        for base in bases:
            subclasses[base].add(key)
    for x, keys in subclasses.items():
        if 2 <= len(keys) <= 10 and x not in ('object', 'Exception', 'Enum', 'NamedTuple', 'Protocol', 'Generic'):
            out.append(('subclasses', x, sorted(keys), []))
    raisers = defaultdict(set)
    for key, info in functions.items():
        for raised in info['raises']:
            raisers[raised].add(key)
    for x, keys in raisers.items():
        if 2 <= len(keys) <= 10:
            out.append(('raisers', x, sorted(keys), sorted((callers.get(x, set()) - keys))[:1]))
    for kind, spec in EFFECTS.items():
        keys = sorted(k for k, info in functions.items() if kind in info['effects'])
        if 1 <= len(keys) <= 12:
            out.append(('effects', spec['text'], keys, []))
    keys = sorted(k for k, info in functions.items() if info['deprecation'])
    if 1 <= len(keys) <= 12:
        out.append(('warnings', 'DeprecationWarning', keys, []))
    rng.shuffle(out)
    return out


FLIP = {ast.Lt: '>=', ast.LtE: '>', ast.Gt: '<=', ast.GtE: '<', ast.Eq: '!=', ast.NotEq: '=='}


def _mutations(files, scans, rng, count):
    """Flip one comparison in a documented function; returns (path, qualname, patched text, siblings)."""
    candidates = []
    for path, (scan, tree, _) in scans.items():
        for key, info in scan.functions.items():
            node = info['node']
            if key in scan.duplicate or not ast.get_docstring(node):
                continue
            compares = [c for c in ast.walk(node) if isinstance(c, ast.Compare) and len(c.ops) == 1
                        and type(c.ops[0]) in FLIP and c.lineno == c.end_lineno]
            if compares:
                candidates.append((path, key, compares))
    rng.shuffle(candidates)
    out = []
    for path, key, compares in candidates[:count]:
        compare = compares[0]
        lines = files[path].split('\n')
        line = lines[compare.lineno - 1]
        segment = line.encode()[compare.left.end_col_offset:compare.comparators[0].col_offset].decode(errors='ignore')
        op = segment.strip()
        if op not in ('<', '<=', '>', '>=', '==', '!='):
            continue
        start = len(line.encode()[:compare.left.end_col_offset].decode(errors='ignore'))
        new_segment = segment.replace(op, FLIP[type(compare.ops[0])], 1)
        lines[compare.lineno - 1] = line[:start] + new_segment + line[start + len(segment):]
        patched = '\n'.join(lines)
        try:
            ast.parse(patched)
        except SyntaxError:
            continue
        siblings = [k for k in scans[path][0].functions if k != key][:2]
        out.append((path, key, patched, siblings))
    return out


def _asset(name, files, out_assets):
    rel = f'repos/{name}.json'
    target = Path(out_assets) / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(files, ensure_ascii=False, sort_keys=True)
    if not target.exists() or target.read_text() != text:
        target.write_text(text)
    return rel, hashlib.sha256(target.read_bytes()).hexdigest()


def _service(rel, sha, patch=None):
    return SERVICE.replace('__FILE__', json.dumps(rel)).replace('__SHA__', json.dumps(sha)) \
        .replace('__ROOT__', json.dumps(V3_ASSETS)).replace('__PATCH__', json.dumps(patch or {}))


def _split(name):
    bucket = int(hashlib.sha256(f'broad-rl-codebase|{name}'.encode()).hexdigest()[:8], 16) % 100
    return 'train' if bucket < 86 else 'dev' if bucket < 93 else 'confirmation'


def python_packages(per_package=160, mutations=12, seed=20261031, out_assets=BUILD_V3_ASSETS):
    out = []
    for name, root, path, files in _packages():
        rng = random.Random(f'{seed}:{name}')
        scans = _analyze(name, files)
        rel, sha = _asset('py-' + name, files, out_assets)
        version, license_ = _version(name, root)
        split = _split(name)
        unit = f'codebase:py:{name}'
        service = _service(rel, sha)
        questions = _questions(name, files, scans, rng)[:per_package]
        pool = [q[2] for q in questions if isinstance(q[2], list)]
        for index, (kind, x, answer, near) in enumerate(questions):
            item = f'{name}:{kind}:{x}'
            if isinstance(answer, list):
                wrong = [answer[:-1] if len(answer) > 1 else answer + [a for a in pool[index % max(1, len(pool))] if a not in answer][:1]]
                wrong += [answer + [n] for n in near if n not in answer]
                wrong += [p for p in pool[index + 1:index + 3] if sorted(p) != answer][:1]
                returns, oracle = 'string[]', {'level': 'normalized', 'normalization': 'answer-list'}
                aliases = [list(reversed(answer))]
                level = 1 + (len(answer) > 3) + (kind in ('effects', 'raisers', 'callers'))
            else:
                wrong = list(near) + [p for p in files if p != answer][:1]
                returns, oracle, aliases, level = 'string', 'normalized', [f'./{answer}'], 1
            if not wrong:
                continue
            case = make_case(
                source='codebase-py', family=f'codebase_{kind}', shape=f'codebase-py-{kind}', upstream=f'pypi:{name}',
                upstream_id=item, upstream_split=split, unit=unit, groups=[unit], root_name='explore_codebase',
                args={}, returns=returns, instruction=pick(INSTRUCTIONS[kind], item).format(x=x), inputs={},
                expected=answer, oracle=oracle, framing=pick(['plain', 'neuralese-helper'], item + '|framing'),
                level=min(3, level), license_=license_, services={'repo': service},
                domain='relational', slice_='inline_placement', wrong=wrong, aliases=aliases,
                revision=version, neuralese=NEURALESE, split=split,
                neuralese_room='per-file notes from read_file fanned in to the answer')
            case['curriculum']['external']['codebase'] = {'package': name, 'version': version, 'files': len(files),
                                                          'bytes': sum(len(t) for t in files.values()),
                                                          'asset': rel, 'asset_sha256': sha, 'kind': kind}
            out.append(case)
        for path_, key, patched, siblings in _mutations(files, scans, rng, mutations):
            item = f'{name}:mutation:{key}'
            case = make_case(
                source='codebase-py', family='codebase_mutation', shape='codebase-py-mutation', upstream=f'pypi:{name}',
                upstream_id=item, upstream_split=split, unit=unit, groups=[unit], root_name='find_altered_function',
                args={}, returns='string', instruction=INSTRUCTIONS['mutation'][0], inputs={}, expected=key,
                oracle='normalized', level=3, framing=pick(['plain', 'neuralese-helper'], item + '|framing'),
                license_=license_, services={'repo': _service(rel, sha, {path_: patched})},
                domain='relational', slice_='inline_placement', wrong=siblings, aliases=[key.upper()],
                revision=version, neuralese=NEURALESE, split=split,
                neuralese_room='per-file notes from read_file fanned in to the answer')
            case['curriculum']['external']['codebase'] = {'package': name, 'version': version, 'files': len(files),
                                                          'bytes': sum(len(t) for t in files.values()),
                                                          'asset': rel, 'asset_sha256': sha, 'kind': 'mutation'}
            out.append(case)
    return out


# --- natlang's own TypeScript/natlang modules (pinned at the repository's HEAD commit) -----------------------------

REPO = Path(resolve_str('repo'))
TS_INSTRUCTIONS = {
    'callers': ['repo is a TypeScript and natlang codebase served file by file. Find every function or method whose '
                'own body calls something named {x} (as {x}(...) or obj.{x}(...)). Return them as "path:name" strings, '
                'path relative to the codebase root and name prefixed by enclosing classes or functions with dots.',
                'Which functions in repo call {x} in their own body (not in a nested function)? Return each as '
                '"path:qualified.name" (Class.method for a method).'],
    'importers': ['repo is a TypeScript codebase. List every file that imports from {x} (an import or re-export whose '
                  'relative specifier resolves to that file). Return the importing files\' paths relative to the root.',
                  'Which files of repo import {x}? Return their paths, one per file.'],
    'definition': ['repo is a TypeScript codebase. Find the file that declares the module-level constant {x} '
                   '(const {x} = ...). Return that file\'s path relative to the codebase root.',
                   'Where in repo is the constant {x} declared? Return the path of that file.'],
    'subclasses': ['repo is a TypeScript codebase. List every class whose extends or implements clause directly names '
                   '{x}. Return each as "path:ClassName".',
                   'Which classes in repo extend or implement {x} directly? Return them as "path:ClassName" strings.'],
    'raisers': ['repo is a TypeScript codebase. Find every function or method whose own body throws a new {x} '
                '(throw new {x}(...)). Return them as "path:qualified.name" strings.',
                'List the functions of repo that themselves throw new {x}. Name each "path:Class.method" or '
                '"path:function".'],
    'effects': ['repo is a TypeScript codebase. Find every function or method whose own body {x}. Return them as '
                '"path:qualified.name" strings.',
                'Which functions in repo directly {x}? Count only what the function itself does, not calls to other '
                'functions of repo. Return "path:qualified.name" for each.'],
    'exported-effects': ['repo is a TypeScript codebase. List every exported function whose own body {x}. Return them '
                         'as "path:name" strings.'],
}
TS_EFFECT_TEXT = {'file-write': 'writes, creates, renames or deletes files through the fs module',
                  'subprocess': 'starts a child process', 'network': 'makes a network request or opens a server or socket',
                  'environment': 'exits, kills or changes the working directory of a process'}
TS_UNITS = ['ts-host/src', 'applications', 'codebases']


def _ts_units():
    import subprocess
    listed = subprocess.run(['git', '-C', str(REPO), 'ls-files', '-s', *TS_UNITS], capture_output=True, text=True,
                            check=True).stdout.splitlines()
    commit = subprocess.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], capture_output=True, text=True,
                            check=True).stdout.strip()
    units = defaultdict(dict)
    for line in listed:
        meta, path = line.split('\t', 1)
        if not path.endswith(('.ts', '.nl')) or path.endswith('.d.ts') or '/test' in path or '.test.' in path:
            continue
        parts = path.split('/')
        unit = '/'.join(parts[:3]) if parts[0] == 'ts-host' else '/'.join(parts[:2])
        if len(parts) <= (3 if parts[0] == 'ts-host' else 2):
            continue
        units[unit][path[len(unit) + 1:]] = meta.split()[1]
    out = []
    for unit, blobs in sorted(units.items()):
        if not 10 <= len(blobs) <= 400:
            continue
        texts = {}
        for rel, blob in blobs.items():
            texts[rel] = subprocess.run(['git', '-C', str(REPO), 'cat-file', 'blob', blob], capture_output=True,
                                        check=True).stdout.decode('utf-8', 'replace')
        out.append((unit, commit, texts))
    return out


def _ts_questions(facts, files, rng):
    functions, classes = facts['functions'], facts['classes']
    defined = defaultdict(set)
    for key in functions:
        defined[key.rsplit(':', 1)[1].split('.')[-1]].add(key)
    out = []
    callers = defaultdict(set)
    for key, info in functions.items():
        for called in info['calls']:
            callers[called].add(key)
    for x, keys in callers.items():
        if x in defined and 2 <= len(keys) <= 10 and len(x) > 3:
            out.append(('callers', x, sorted(keys), sorted(defined[x] - keys)[:1]))
    importers = defaultdict(set)
    for path, info in facts['files'].items():
        for target in info['imports']:
            importers[target].add(path)
    for x, paths in importers.items():
        if 2 <= len(paths) <= 10:
            out.append(('importers', x, sorted(paths), []))
    owners = defaultdict(set)
    for path, info in facts['files'].items():
        for name in info['consts']:
            owners[name].add(path)
    for x, paths in owners.items():
        mentions = [p for p, t in files.items() if x in t and p not in paths]
        if len(paths) == 1 and mentions:
            out.append(('definition', x, sorted(paths)[0], mentions[:1]))
    subclasses = defaultdict(set)
    for key, bases in classes.items():
        for base in bases:
            subclasses[base].add(key)
    for x, keys in subclasses.items():
        if 2 <= len(keys) <= 10:
            out.append(('subclasses', x, sorted(keys), []))
    throwers = defaultdict(set)
    for key, info in functions.items():
        for thrown in info['throws']:
            throwers[thrown].add(key)
    for x, keys in throwers.items():
        if 2 <= len(keys) <= 10 and x != 'Error':
            out.append(('raisers', x, sorted(keys), sorted(callers.get(x, set()) - keys)[:1]))
    for kind, text in TS_EFFECT_TEXT.items():
        keys = sorted(k for k, info in functions.items() if kind in info['effects'])
        if 1 <= len(keys) <= 12:
            out.append(('effects', text, keys, []))
        exported = sorted(k for k in keys if functions[k]['exported'] and '.' not in k.rsplit(':', 1)[1])
        if 1 <= len(exported) < len(keys):
            out.append(('exported-effects', text, exported, [k for k in keys if k not in exported][:1]))
    rng.shuffle(out)
    return out


def natlang_modules(per_unit=200, seed=20261031, out_assets=BUILD_V3_ASSETS):
    import subprocess
    import tempfile
    out = []
    for unit_dir, commit, files in _ts_units():
        name = unit_dir.replace('/', '-')
        rng = random.Random(f'{seed}:{name}')
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, 'in.json').write_text(json.dumps(files))
            subprocess.run(['node', str(REPO / 'scripts/rl_tasks/ts_analyze.mjs'), str(REPO / 'ts-host'),
                            f'{tmp}/in.json', f'{tmp}/out.json'], check=True)
            facts = json.loads(Path(tmp, 'out.json').read_text())
        rel, sha = _asset(name, files, out_assets)
        split = _split(name)
        unit = f'codebase:natlang:{unit_dir}'
        service = _service(rel, sha)
        questions = _ts_questions(facts, files, rng)[:per_unit]
        pool = [q[2] for q in questions if isinstance(q[2], list)]
        for index, (kind, x, answer, near) in enumerate(questions):
            item = f'{unit_dir}:{kind}:{x}'
            if isinstance(answer, list):
                wrong = [answer[:-1] if len(answer) > 1 else answer + [a for a in pool[index % max(1, len(pool))] if a not in answer][:1]]
                wrong += [answer + [n] for n in near if n not in answer]
                wrong += [p for p in pool[index + 1:index + 3] if sorted(p) != answer][:1]
                returns, oracle, aliases = 'string[]', {'level': 'normalized', 'normalization': 'answer-list'}, [list(reversed(answer))]
                level = 1 + (len(answer) > 3) + (kind in ('effects', 'exported-effects', 'raisers', 'callers'))
            else:
                wrong = list(near) + [p for p in files if p != answer][:1]
                returns, oracle, aliases, level = 'string', 'normalized', [f'./{answer}'], 1
            wrong = [w for w in wrong if w and w != answer]
            if not wrong:
                continue
            case = make_case(
                source='codebase-natlang', family=f'codebase_ts_{kind.replace("-", "_")}', shape=f'codebase-ts-{kind}',
                upstream='natlang', upstream_id=item, upstream_split=split, unit=unit, groups=[unit],
                root_name='explore_codebase', args={}, returns=returns,
                instruction=pick(TS_INSTRUCTIONS[kind], item).format(x=x), inputs={}, expected=answer, oracle=oracle,
                framing=pick(['plain', 'neuralese-helper'], item + '|framing'), level=min(3, level),
                license_='natlang repository (owner)', services={'repo': service}, domain='relational',
                slice_='inline_placement', wrong=wrong, aliases=aliases, revision=commit, neuralese=NEURALESE, split=split,
                neuralese_room='per-file notes from read_file fanned in to the answer')
            case['curriculum']['external']['codebase'] = {'package': unit_dir, 'version': commit, 'files': len(files),
                                                          'bytes': sum(len(t) for t in files.values()),
                                                          'asset': rel, 'asset_sha256': sha, 'kind': kind}
            out.append(case)
    return out
