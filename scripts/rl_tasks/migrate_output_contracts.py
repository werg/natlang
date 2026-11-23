#!/usr/bin/env python3
"""Move authored tasks' prose ``Output contract:`` paragraphs into declared types (DECISIONS 2026-11-23 "typed output
contracts").

Exact desks since authoring round 6 restate their output format in a root paragraph that starts ``Output contract:``.
The runtime already shows a call its signature and the type aliases it names; a directory reducer now also declares the
files it writes (`writes:`, ts-host runtime/loader.ts), so the contract moves there:

* the root gets ``writes: <Root>Files``, a record whose fields are the paths (or ``dir/*.ext`` globs) it may write, each
  typed ``string``; a contract sentence about one file becomes the doc comment of that file's field, and the sentences
  about the folder as a whole (line endings, files left unchanged) the doc comment of the alias;
* the sentences about the return value become the doc comment of ``<Root>Result``, an alias of the declared return type;
* the paragraph leaves the instructions.

The aliases go in the package's ``types.ts`` (the root's directory), which the loader applies to the root and its folder.

Which files the root may write comes from the task, never from its outcome alone (a list of only the files the reference
writes would tell a solver which conditional files stay unwritten): the files the reference creates or changes, every
path the instructions name that is not an input file, every folder they name that the input lacks, and the input files
named in a sentence that writes to them, and the input folders named as a destination (`to done/`). A file named in the instructions keeps its exact path; files the instructions do
not name (per-item outputs, moved files) are covered by a glob of their folder.

    migrate_output_contracts.py IN.jsonl... --out OUT.jsonl [--report REPORT.json]

Rows without a root ``Output contract:`` paragraph are copied unchanged. A row that cannot be migrated (already declares
``writes:``, an unreadable ``returns:``) is left out and listed in the report with its reason. Qualification (reference
replay, negatives) is ``scripts/rl_tasks/verify.mjs`` on OUT.jsonl with a runtime that has ``writes:``.
"""
from __future__ import annotations

import argparse
import json
import posixpath
import re
import sys
from pathlib import Path

CONTRACT = re.compile(r'(?:^|\n)\s*\**Output contract:?\**:?(.*?)(?=\n\s*\n|$)', re.S | re.I)
FRONT = re.compile(r'^---\r?\n(.*?)\r?\n---\r?\n?(.*)$', re.S)
EXT = r'(?:md|json|txt|csv|tsv|ya?ml|log|ini|toml|xml|html)'
FILE_PATH = re.compile(rf'(?<![\w/.<>-])((?:[A-Za-z0-9_][\w.-]*/)*[A-Za-z0-9_][\w.-]*\.{EXT})(?![\w/-])')
DIR_PATH = re.compile(r'(?<![\w/.<>-])((?:[A-Za-z0-9_][\w.-]*/)+)(?![\w<*])')
RETURN = re.compile(r'\b(return|returns|returned|result)\b', re.I)
RETURN_LEAD = re.compile(r'^\W*(?:the\s+)?(?:return|returns|returned|result)\b', re.I)
WRITE_VERB = re.compile(r'\b(write|writes|written|rewrite|rewrites|update|updates|updated|append|appends|appended|add|adds|'
                        r'replace|replaces|set|sets|change|changes|edit|edits|correct|corrects|record|records|mark|marks|'
                        r'create|creates|overwrite|overwrites|fill|fills|move|moves|moved|file it|files it)\b', re.I)
SENTENCE = re.compile(r'(?<=[.!?])\s+(?=[A-Z`"(<])|\n+')


def sentences(text, wrapped=False):
    """The sentences of ``text``; a line break ends one too unless ``wrapped`` (a paragraph wrapped across lines)."""
    if wrapped:
        text = re.sub(r'[ \t]*\n[ \t]*', ' ', text)
    return [part.strip() for part in SENTENCE.split(text.strip()) if part.strip()]


def pascal(name):
    return ''.join(part[:1].upper() + part[1:] for part in re.split(r'[^A-Za-z0-9]+', name) if part) or 'Root'


def doc(lines, indent=''):
    text = ' '.join(lines).replace('*/', '* /').strip()
    return f'{indent}/** {text} */\n' if text else ''


def frontmatter_value(front, key):
    """(start, end, text) of a top-level `key:` entry with its indented continuation lines, or None."""
    lines = front.split('\n')
    for index, line in enumerate(lines):
        if re.match(rf'^{key}:', line):
            end = index + 1
            while end < len(lines) and re.match(r'^(\s|$)', lines[end]) and lines[end].strip():
                end += 1
            return index, end, ' '.join([line[len(key) + 1:].strip()] + [item.strip() for item in lines[index + 1:end]]).strip()
    return None


def unquote(text):
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] == '"':
        return json.loads(text)
    if len(text) >= 2 and text[0] == text[-1] == "'":
        return text[1:-1].replace("''", "'")
    return text


def written_paths(folder_files, expected_files):
    if expected_files is None:
        return set()
    return {path for path, text in expected_files.items() if folder_files.get(path) != text}


def output_fields(semantics, root_body, contract_text):
    """{field name: [paths it covers]}: the files the root may write (module docstring)."""
    inputs = semantics.get('folder_files') or {}
    written = written_paths(inputs, semantics.get('expected_files'))
    instructions = root_body + '\n' + contract_text
    named_files = set(FILE_PATH.findall(instructions))
    input_dirs = {posixpath.dirname(path) + '/' for path in inputs if '/' in path}
    for path in list(inputs):
        parts = path.split('/')[:-1]
        for depth in range(1, len(parts)):
            input_dirs.add('/'.join(parts[:depth]) + '/')
    exact, globs = set(), set()
    for path in written:
        if path in named_files or posixpath.basename(path) in {posixpath.basename(f) for f in named_files if '/' not in f} and '/' not in path:
            exact.add(path)
        elif '/' in path:
            globs.add((posixpath.dirname(path), posixpath.splitext(path)[1]))
        else:
            exact.add(path)
    for path in named_files:
        if path in inputs:
            if any(path in s and WRITE_VERB.search(s) for s in sentences(instructions)):
                exact.add(path)
        elif not any(path.startswith(d + '/') for d, _ in globs):
            exact.add(path)
    for folder in set(DIR_PATH.findall(instructions)):
        name = folder.rstrip('/')
        if not name or any(d == name for d, _ in globs):
            continue
        # An input folder is written only when a sentence sends something to it (a move's source is not written).
        into = re.compile(rf'\b(?:to|into)\s+`?{re.escape(folder)}')
        if folder not in input_dirs or any(into.search(s) for s in sentences(instructions)):
            exts = {posixpath.splitext(p)[1] for p in inputs if p.startswith(folder)} or {''}
            globs.add((name, exts.pop() if len(exts) == 1 else ''))
    # A glob covers the exact paths inside its folder.
    exact = {path for path in exact if not any(posixpath.dirname(path) == d for d, _ in globs)}
    fields = {path: [path] for path in sorted(exact)}
    for folder, ext in sorted(globs):
        fields[f'{folder}/*{ext}'] = sorted(p for p in written if posixpath.dirname(p) == folder)
    return fields, written


def attribute(contract_text, fields, has_return):
    """(folder sentences, {field: sentences}, return sentences)."""
    general, per_field, returned = [], {name: [] for name in fields}, []
    for sentence in sentences(contract_text, wrapped=True):
        mentioned = [name for name in fields if (name.split('/*')[0] + '/' if '/*' in name else name) in sentence]
        if RETURN_LEAD.search(sentence) or (RETURN.search(sentence) and not mentioned):
            if has_return:
                returned.append(sentence)
                continue
        if len(mentioned) == 1:
            per_field[mentioned[0]].append(sentence)
        else:
            general.append(sentence)
    return general, per_field, returned


def migrate(row):
    """(new row, None) or (None, reason)."""
    semantics = row['semantics']
    root = semantics['root']
    text = semantics['files'][root]
    match = FRONT.match(text)
    if not match:
        return None, 'root has no frontmatter'
    front, body = match.group(1), match.group(2)
    contract = CONTRACT.search(body)
    if not contract:
        return row, None
    if frontmatter_value(front, 'writes'):
        return None, 'root already declares writes:'
    contract_text = contract.group(1).strip()
    new_body = (body[:contract.start()] + body[contract.end():]).rstrip('\n') + '\n'
    new_body = re.sub(r'\n{3,}', '\n\n', new_body)
    kind = frontmatter_value(front, 'kind')
    reducer = bool(kind and unquote(kind[2]) == 'directory-reducer')
    fields, written = output_fields(semantics, body[:contract.start()] + body[contract.end():], contract_text) if reducer else ({}, set())
    if written and not reducer:
        return None, 'expected files without a directory-reducer root'
    returns = frontmatter_value(front, 'returns')
    if not returns:
        return None, 'root has no returns:'
    return_type = unquote(returns[2])
    general, per_field, returned = attribute(contract_text, fields, True)
    if not fields:
        returned = general + returned
        general = []
    base = pascal(posixpath.splitext(posixpath.basename(root))[0])
    files = dict(semantics['files'])
    types_path = posixpath.join(posixpath.dirname(root), 'types.ts') if posixpath.dirname(root) else 'types.ts'
    existing = files.get(types_path, '')
    taken = set(re.findall(r'\btype\s+([A-Za-z_$][\w$]*)', existing)) | set(re.findall(r'^\s+([A-Za-z_$][\w$]*)\s*:', front, re.M))
    def fresh(name):
        while name in taken:
            name += '_'
        taken.add(name)
        return name
    declarations, lines = [], front.split('\n')
    if returned:
        result_name = fresh(base + 'Result')
        declarations.append(doc(returned) + f'export type {result_name} = {return_type};\n')
        start, end, _ = returns
        lines[start:end] = [f'returns: {result_name}']
    if fields:
        files_name = fresh(base + 'Files')
        members = ''.join(doc(per_field[name], '  ') + f'  {json.dumps(name)}: string;\n' for name in fields)
        declarations.append(doc(general) + f'export type {files_name} = {{\n{members}}};\n')
        at = next((i for i, line in enumerate(lines) if line.startswith('returns:')), len(lines) - 1)
        while at + 1 < len(lines) and re.match(r'^\s+\S', lines[at + 1]):
            at += 1
        lines.insert(at + 1, f'writes: {files_name}')
    if not declarations:
        return None, 'contract paragraph has no sentence to declare'
    files[root] = '---\n' + '\n'.join(lines) + '\n---\n' + new_body
    files[types_path] = (existing.rstrip('\n') + '\n\n' if existing.strip() else '') + '\n'.join(declarations)
    out = json.loads(json.dumps(row))
    out['semantics']['files'] = files
    migration = {'schema': 'natlang.typed-output-contract/1', 'contract': contract_text,
                 'writes': {name: covered for name, covered in fields.items()},
                 'unattributed': []}
    out.setdefault('curriculum', {}).setdefault('external', {})['typed_output_contract'] = migration
    return out, None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('inputs', nargs='+')
    parser.add_argument('--out', required=True)
    parser.add_argument('--report')
    parser.add_argument('--only-contract', action='store_true', help='write only the migrated rows')
    args = parser.parse_args(argv)
    kept, dropped, migrated, copied = [], [], 0, 0
    for path in args.inputs:
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            has = 'Output contract' in line
            new, reason = migrate(row)
            if new is None:
                dropped.append({'id': row['id'], 'source': path, 'reason': reason})
                continue
            if new is row:
                if args.only_contract:
                    continue
                copied += 1
            else:
                migrated += 1
            kept.append(new)
    Path(args.out).write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in kept))
    report = {'migrated': migrated, 'copied': copied, 'dropped': dropped}
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=1))
    print(json.dumps({'migrated': migrated, 'copied': copied, 'dropped': len(dropped)}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
