"""Natlang codebase tasks: work inside sizable natlang packages (named `.nl` functions with callable folders, TypeScript
children, `types.ts` aliases, `uses` entries) from the repository's codebases/ (the 100 design-brief applications and
the named teacher programs), pinned at the repository HEAD commit. One split unit per package; design-brief
applications keep their own release split (first three applications of a domain train, the fourth dev, the fifth
test, which becomes confirmation here); other packages are bucketed by name.

Kinds (all checked exactly):
  explore  The package becomes the companion folder of a new root function, so its operations and every nested
           function are callable and readable (read_code, eval). Questions about the package's own structure are
           answered from its sources: the direct callable items of a function, the functions whose declared
           parameters or result use a type, the functions whose instructions name a field, the leaf functions under
           an operation, the TypeScript children, the fields of a type alias and which are optional. Oracle:
           answer-list over function paths ("operation.child.grandchild"), normalized exact for single answers.
           The Neuralese framing adds `read_function(question, name, source): Neuralese<string>`, a private
           per-function view the root fans in.
  repair   A directory reducer over the package folder whose one planted fault makes the package fail to load with
           the actual natlang loader: a misspelled type name in a function's declared signature, a misspelled `uses`
           path, a syntax error in a TypeScript child, or a missing child function that another function `uses`.
           Judged by `semantics.codebase_check` (ts-host/src/teacher/codebase-check.ts): the edited package loads,
           every other file is byte-identical, and the declared signatures/uses/headers equal the originals.
           Negative controls: leaving it unchanged, deleting the broken file, a signature-changing "fix".
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import subprocess
from collections import defaultdict
from pathlib import Path

from ..common import make_case, pick, resolve_str

REPO = Path(resolve_str('repo'))
DESIGN = 'codebases/design_briefs'
PACKET = Path(resolve_str('data_nvme', 'luna-generation-20261026', 'design-briefs-pr4-review', 'packet-v3'))
NEURALESE = {
    'helper': 'read_function', 'args': {'question': 'string', 'name': 'string', 'source': 'string'},
    'returns': 'Neuralese<string>',
    'instruction': 'Read source, the code of the function name in a natlang program, for what question needs, and '
                   'write a private note: its declared parameters and result, the items it calls, the fields and '
                   'types it relies on, or that it has nothing relevant.',
    'root_note': 'read_function gives a private note on one function of the program. You may read the relevant '
                 'functions with read_code, hand each to read_function and combine the notes before answering.',
}
EXPLORE = {
    'children': ['The program in your callable folder is a natlang application. Which items can its function {x} call '
                 'directly (the functions and TypeScript children in its own companion folder)? Return their names.',
                 'List the callable items of {x}: the named functions and TypeScript modules in the folder that '
                 'belongs to it, not their own children. Return their names.'],
    'param-type': ['Which named functions of the program in your callable folder declare a parameter whose type '
                   'mentions {x}? Return each as its dotted path from the program root (operation.child.grandchild).',
                   'Find every function of the callable program that takes an argument typed with {x} (directly or '
                   'inside an array, union or record type). Return their dotted paths.'],
    'return-type': ['Which named functions of the program in your callable folder declare a result type that '
                    'mentions {x}? Return their dotted paths (operation.child.grandchild).'],
    'mentions': ['Which named functions of the callable program mention the field or identifier {x} in their '
                 'instructions? Return their dotted paths.'],
    'leaves': ['Under the operation {x} of the program in your callable folder, which named functions call nothing '
               'further (they have no callable items of their own)? Return their dotted paths.'],
    'ts-children': ['Which items of the program in your callable folder are TypeScript modules rather than '
                    'natural-language functions? Return their dotted paths.'],
    'optional-fields': ['In the program in your callable folder, which fields of the type {x} are optional? Return the '
                        'field names.'],
    'fields': ['List every field name of the type {x} declared by the program in your callable folder.'],
    'field-type': ['In the program in your callable folder, what is the declared type of the field {x}? Return the '
                   'type exactly as declared.'],
}
REPAIR = ('The folder is a natlang program that no longer loads: {hint} Repair it so that the program loads again. '
          'Change only what is broken; keep every function\'s declared parameters, result and uses as intended, and '
          'leave all other files unchanged. Read the loader\'s complaint by inspecting the files, fix the file, and '
          'return a one-line description of the fix.')
HINTS = {
    'type-typo': 'one function declares a parameter or result type with a misspelled name.',
    'uses-typo': 'one function lists a `uses` entry whose path is misspelled.',
    'ts-syntax': 'one TypeScript child has a syntax error.',
    'missing-child': 'a function that another function lists in its `uses` was removed. Recreate it with the '
                     'parameters and result its callers need: {contract}.',
}


def _git(*args):
    return subprocess.run(['git', '-C', str(REPO), *args], capture_output=True, check=True).stdout


def _packages():
    commit = _git('rev-parse', 'HEAD').decode().strip()
    listed = _git('ls-files', '-s', 'codebases').decode().splitlines()
    groups = defaultdict(dict)
    for line in listed:
        meta, path = line.split('\t', 1)
        parts = path.split('/')
        if parts[1] == 'design_briefs':
            if len(parts) < 4 or not re.match(r'\d{3}-', parts[2]):
                continue
            unit, rel = '/'.join(parts[:3]), '/'.join(parts[3:])
        else:
            if len(parts) < 3:
                continue
            unit, rel = '/'.join(parts[:2]), '/'.join(parts[2:])
        if rel.endswith(('.nl', '.ts')) or rel == 'natlang.json':
            if '.test.' in rel or rel.endswith('.d.ts'):
                continue
            groups[unit][rel] = meta.split()[1]
    out = []
    for unit, blobs in sorted(groups.items()):
        files = {rel: _git('cat-file', 'blob', blob).decode('utf-8', 'replace') for rel, blob in sorted(blobs.items())}
        files.setdefault('natlang.json', '{}\n')
        if sum(p.endswith('.nl') for p in files) >= 4:
            out.append((unit, commit, files))
    return out


def _design_splits():
    splits = {}
    for name, split in (('train', 'train'), ('dev', 'dev'), ('test', 'confirmation')):
        with open(PACKET / f'{name}.gates.ir.jsonl') as handle:
            for line in handle:
                row = json.loads(line)
                app = row['curriculum']['world_group'].removeprefix('design-brief-app-')
                splits[app] = split
    return splits


def frontmatter(text):
    match = re.match(r'---\n([\s\S]*?)\n---', text)
    out, key = {}, ''
    if not match:
        return out
    for line in match.group(1).split('\n'):
        head = re.match(r'^([A-Za-z_]+):\s?(.*)$', line)
        if head:
            key = head.group(1)
            out[key] = head.group(2)
        elif key:
            out[key] += '\n' + line
    return out


def _uses(value):
    if not value:
        return []
    return sorted(x.strip().lstrip('-').strip() for x in re.split(r'[,\n]', value.strip().strip('[]')) if x.strip().lstrip('-').strip())


def _dotted(rel):
    return rel.rsplit('.', 1)[0].replace('/', '.')


def _structure(files):
    nl = {p: frontmatter(t) for p, t in files.items() if p.endswith('.nl')}
    ts_children = [p for p in files if p.endswith('.ts') and not p.endswith('types.ts')]
    items = defaultdict(list)
    for p in list(nl) + ts_children:
        parent = p.rsplit('/', 1)[0] if '/' in p else ''
        items[parent].append(p)
    return nl, ts_children, items


def _type_aliases(files):
    """Top-level `export type Name = { ... }` record aliases in types.ts files: {name: {field: (optional, type)}}."""
    out = {}
    for path, text in files.items():
        if not path.endswith('types.ts'):
            continue
        for match in re.finditer(r'export type (\w+)\s*=\s*\{', text):
            start, depth, i = match.end(), 1, match.end()
            while i < len(text) and depth:
                depth += {'{': 1, '}': -1}.get(text[i], 0)
                i += 1
            body = text[start:i - 1]
            fields, depth, current = {}, 0, ''
            for ch in body:
                depth += {'{': 1, '(': 1, '[': 1, '<': 1, '}': -1, ')': -1, ']': -1, '>': -1}.get(ch, 0)
                if ch in ';\n,' and depth == 0:
                    current = current.strip()
                    field = re.match(r'^(?:readonly\s+)?(\w+)(\??):\s*(.+)$', re.sub(r'/\*[\s\S]*?\*/|//.*', '', current).strip())
                    if field:
                        fields[field.group(1)] = (field.group(2) == '?', ' '.join(field.group(3).split()))
                    current = ''
                else:
                    current += ch
            if fields and match.group(1) not in out:
                out[match.group(1)] = fields
    return out


def _explore_questions(files, rng):
    nl, ts_children, items = _structure(files)
    functions = sorted(nl)
    out = []
    for parent, children in items.items():
        if parent and parent + '.nl' in nl and 2 <= len(children) <= 12:
            out.append(('children', _dotted(parent + '.nl'), sorted(c.rsplit('/', 1)[-1].rsplit('.', 1)[0] for c in children)))
    words = set(re.findall(r'\b[A-Z][A-Za-z0-9]{3,}\b', ' '.join(f.get('args', '') + ' ' + f.get('returns', '') for f in nl.values())))
    for word in sorted(words):
        params = sorted(_dotted(p) for p in functions if re.search(rf'\b{word}\b', nl[p].get('args', '')))
        results = sorted(_dotted(p) for p in functions if re.search(rf'\b{word}\b', nl[p].get('returns', '')))
        if 2 <= len(params) <= 12:
            out.append(('param-type', word, params))
        if 2 <= len(results) <= 12:
            out.append(('return-type', word, results))
    aliases = _type_aliases(files)
    field_names = defaultdict(set)
    for alias, fields in aliases.items():
        for field in fields:
            field_names[field].add(alias)
        optional = sorted(f for f, (opt, _) in fields.items() if opt)
        if 3 <= len(fields) <= 16:
            out.append(('fields', alias, sorted(fields)))
        if 1 <= len(optional) < len(fields):
            out.append(('optional-fields', alias, optional))
        for field, (_, type_text) in fields.items():
            if len(field_names[field]) == 1 and len(type_text) <= 60:
                out.append(('field-type', f'{field} of {alias}', type_text))
    for field in sorted(field_names):
        if len(field) < 5:
            continue
        hits = sorted(_dotted(p) for p in functions if re.search(rf'\b{field}\b', files[p].split('---', 2)[-1]))
        if 2 <= len(hits) <= 10:
            out.append(('mentions', field, hits))
    for op in [p for p in functions if '/' not in p]:
        prefix = op[:-3] + '/'
        under = [p for p in functions if p.startswith(prefix)]
        leaves = sorted(_dotted(p) for p in under if not items.get(p[:-3]))
        if 2 <= len(leaves) <= 12 and len(leaves) < len(under):
            out.append(('leaves', _dotted(op), leaves))
    if 1 <= len(ts_children) <= 12:
        out.append(('ts-children', '', sorted(_dotted(p) for p in ts_children)))
    rng.shuffle(out)
    return out


def _typo(word, rng):
    i = rng.randrange(1, len(word) - 1)
    return word[:i] + word[i + 1:] if rng.random() < 0.5 else word[:i] + word[i] + word[i:]


def _repairs(files, rng, count):
    """(kind, broken path, broken text or None when deleted, constraint spec extras, wrong-fix text, contract)."""
    nl = {p: frontmatter(t) for p, t in files.items() if p.endswith('.nl')}
    aliases = set(_type_aliases(files)) | set(re.findall(r'export type (\w+)', ''.join(t for p, t in files.items() if p.endswith('types.ts'))))
    out = []
    for path in sorted(nl):
        fm, text = nl[path], files[path]
        for field in ('args', 'returns'):
            names = [w for w in re.findall(r'\b([A-Z]\w{3,})\b', fm.get(field, '')) if w in aliases]
            if names:
                word = rng.choice(names)
                bad = _typo(word, rng)
                if bad in aliases:
                    continue
                head, body = text.split('\n---', 1)
                broken_head = re.sub(rf'^({field}:.*?)\b{word}\b', lambda m: m.group(1) + bad, head, count=1, flags=re.M | re.S)
                if broken_head == head:
                    continue
                neutered = re.sub(rf'^{field}:.*$', f'{field}: unknown', head, count=1, flags=re.M)
                out.append(('type-typo', path, broken_head + '\n---' + body,
                            {'signatures': [{'path': path, 'args': fm.get('args'), 'returns': fm.get('returns')}]},
                            neutered + '\n---' + body, None))
                break
        uses = _uses(fm.get('uses'))
        if uses:
            entry = rng.choice(uses)
            last = entry.rsplit('/', 1)[-1]
            if len(last) > 4:
                bad = entry[:-len(last)] + _typo(last, rng)
                out.append(('uses-typo', path, text.replace(entry, bad, 1),
                            {'signatures': [{'path': path, 'uses': uses}]},
                            re.sub(r'^uses:.*$', 'uses: []', text, count=1, flags=re.M), None))
                target = entry + '.nl'
                if target in files and frontmatter(files[target]).get('args') is not None:
                    tfm = frontmatter(files[target])
                    contract = f'{entry} with args {" ".join(tfm["args"].split())} and returns {" ".join(tfm.get("returns", "").split())}'
                    out.append(('missing-child', target, None,
                                {'signatures': [{'path': target, 'args': tfm['args'], 'returns': tfm.get('returns')}],
                                 'require': [target], 'allow_new_prefixes': [target]},
                                '---\nargs: {}\nreturns: string\n---\nDo it.\n', contract))
    for path in sorted(p for p in files if p.endswith('.ts') and not p.endswith('types.ts')):
        text = files[path]
        header = re.search(r'export (?:default )?(?:async )?function [^{]+', text)
        closers = [m.start() for m in re.finditer(r'\)', text)]
        if header and closers:
            cut = rng.choice(closers)
            out.append(('ts-syntax', path, text[:cut] + text[cut + 1:],
                        {'contains': [{'path': path, 'text': header.group(0)}]},
                        f'export default function stub(): unknown {{ return null; }}\n', None))
    rng.shuffle(out)
    return out[:count]


def _split(unit, design_splits):
    if unit.startswith(DESIGN):
        app = unit.rsplit('/', 1)[-1][:3]
        return design_splits.get(app, 'train')
    bucket = int(hashlib.sha256(f'broad-rl-natcode|{unit}'.encode()).hexdigest()[:8], 16) % 100
    return 'train' if bucket < 86 else 'dev' if bucket < 93 else 'confirmation'


def natlang_codebases(per_package=60, repairs=24, seed=20261031):
    design_splits = _design_splits()
    out = []
    for unit_dir, commit, files in _packages():
        rng = random.Random(f'{seed}:{unit_dir}')
        split = _split(unit_dir, design_splits)
        unit = f'natlang-codebase:{unit_dir}'
        upstream_split = {'train': 'train', 'dev': 'validation', 'confirmation': 'validation'}[split]
        size = {'files': len(files), 'bytes': sum(len(t) for t in files.values()),
                'nl_functions': sum(p.endswith('.nl') for p in files)}
        questions = _explore_questions(files, rng)[:per_package]
        pool = [q[2] for q in questions if isinstance(q[2], list)]
        # The package becomes the root's companion folder; package-root `uses` paths gain that folder's prefix.
        program = {f'explore_program/{p}': (re.sub(r'^uses:.*?(?=^\w+:|^---)', lambda m: re.sub(
            r'([\[\s,-])([A-Za-z_][\w/]*)', lambda u: u.group(1) + 'explore_program/' + u.group(2), m.group(0)[5:]).join(['uses:', '']),
            t, count=1, flags=re.M | re.S) if p.endswith('.nl') else t) for p, t in files.items() if p != 'natlang.json'}
        for index, (kind, x, answer) in enumerate(questions):
            item = f'{unit_dir}:{kind}:{x}'
            text = pick(EXPLORE[kind], item).format(x=x)
            if isinstance(answer, list):
                wrong = [answer[:-1] if len(answer) > 1 else answer + ['index']]
                wrong += [p for p in pool[index + 1:index + 3] if sorted(p) != sorted(answer)][:1]
                returns, oracle, aliases = 'string[]', {'level': 'normalized', 'normalization': 'answer-list'}, [list(reversed(answer))]
            else:
                wrong = ['unknown', 'string']
                returns, oracle, aliases = 'string', 'normalized', [f' {answer} ']
            wrong = [w for w in wrong if w != answer]
            case = make_case(
                source='natlang-codebase', family=f'natlang_codebase_{kind.replace("-", "_")}',
                shape=f'natlang-codebase-explore-{kind}', upstream='natlang:codebases', upstream_id=item,
                upstream_split=upstream_split, unit=unit, groups=[unit], root_name='explore_program', args={},
                returns=returns, instruction=text, inputs={}, expected=answer, oracle=oracle, split=split,
                framing=pick(['plain', 'neuralese-helper'], item + '|framing'), level=2 if isinstance(answer, list) else 1,
                license_='natlang repository (owner)', extra_files=program, domain='relational',
                slice_='nested_scoped', wrong=wrong, aliases=aliases, revision=commit, neuralese=NEURALESE,
                neuralese_room='per-function notes from read_function fanned in to the answer')
            case['curriculum']['external']['codebase'] = {'package': unit_dir, 'version': commit, 'kind': f'explore-{kind}', **size}
            out.append(case)
        for kind, path, broken, extras, wrong_fix, contract in _repairs(files, rng, repairs):
            item = f'{unit_dir}:repair:{kind}:{path}'
            folder = dict(files)
            if broken is None:
                del folder[path]
            else:
                folder[path] = broken
            preserve = {p: hashlib.sha256(t.encode()).hexdigest() for p, t in files.items() if p != path}
            check = {'load': True, 'preserve': preserve, 'require': [path], **extras}
            write = lambda text: (f'await folder.file({json.dumps(path)}).writeText({json.dumps(text)}); '
                                  f'return "repaired {path}";')
            hint = HINTS[kind].format(contract=contract or '')
            case = make_case(
                source='natlang-codebase', family=f'natlang_codebase_repair_{kind.replace("-", "_")}',
                shape=f'natlang-codebase-repair-{kind}', upstream='natlang:codebases', upstream_id=item,
                upstream_split=upstream_split, unit=unit, groups=[unit], root_name='repair_program', args={},
                returns='string', instruction=REPAIR.format(hint=hint), inputs={}, expected=None, oracle='exact',
                split=split, framing='plain', level=3, license_='natlang repository (owner)', domain='relational',
                slice_='folder_failure', reference_root=[['eval', {'code': write(files[path]), 'finish': True}]],
                revision=commit, extra_semantics={'folder_files': folder, 'codebase_check': check})
            root = case['semantics']['root']
            case['semantics']['files'][root] = case['semantics']['files'][root].replace('---\nargs:', '---\nkind: directory-reducer\nargs:', 1)
            delete = f'await folder.file({json.dumps(path)}).remove(); return "removed";' if broken is not None else 'return "nothing";'
            case['curriculum']['external']['checks'] = {
                'wrong_code': ['return "looked, nothing to fix";', write(wrong_fix)] + ([delete] if broken is not None else []),
                'wrong_answers': [], 'alias_answers': []}
            case['curriculum']['external']['codebase'] = {'package': unit_dir, 'version': commit, 'kind': f'repair-{kind}', **size}
            out.append(case)
    return out
