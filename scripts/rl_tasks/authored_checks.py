#!/usr/bin/env python3
"""Static construct and contract checks for authored RL tasks (plans/neuralese/RL_TASK_AUTHORING_KIT.md section 4).

Qualification replays the reference, the negatives and the scripted smoke; none of that executes the program the way
a model would, so a task can pass every replay while its construct tags, folders, readouts, skills or helper calls
are fiction. These checks read the authored task and its program files and reject (errors) or flag (warnings):

- ``folder_reducer`` without ``folders`` (the root must reduce the task's world folder, not a folder it builds), and a
  ``kind: directory-reducer`` root without one;
- a ``readout: decision`` helper that calls helpers or owns a callable folder: readouts run in one pass with no tools,
  so sub-helper calls inside them never run;
- ``skills`` files bound to a function that does not exist, or read by a function other than the one owning the
  folder (the runtime binds ``<folder>/skills/<name>/SKILL.md`` only to the call that owns ``<folder>``);
- ``multi_operation_codebase`` on a world whose tasks run a single root operation, or whose roots share no helper file
  (same name, identical text, in the folders of two or more roots) (batch-level);
- ``iterateOn`` without an explicit stopping condition in its instructions, and an ``iterate`` tag without ``iterateOn``;
- ``call_depth_3`` without a root -> helper -> sub-helper chain, or whose middle helper never calls its sub-helper;
  warning when the middle helper looks like a pass-through or the sub-helper like a one-line comparison;
- helper calls spelled out in the instructions (backticked code) with a different number of arguments than the
  helper declares.

Round 6 adds the static counterparts of the blind-solve rejection modes of rounds 4-5 (``blind_failure_modes``): a helper that
reads a labelled line or quotes a phrase no note contains, a helper instruction that quotes a world entry (the reference child
matcher is first-match on fragments of the call opening), a move or write whose destination file name is not stated, a sort
whose tie order is not stated, a reopened / corrected / re-fitted item whose earlier record is not said to be cleared or kept,
and an ``Output contract:`` paragraph that is missing or whose templates and sort key contradict the numbered steps. Tasks by
workers numbered ``ROUND6_FIRST_WORKER`` or higher get an error for a missing contract (and the
scripted-match fragment inside a helper instruction, which is an error for every task); everything else is a ``heuristic:``
warning, so the old batches can be measured without being failed.

The call_depth_3 warnings are heuristics: pass-through means identical argument and return types to its sub-helper, thin
means under ``THIN_WORDS`` words of its own besides the call; they point the auditor at a task, they do not decide it.
Smoke and reference scripts get the same argument-count check in structure_smoke.mjs (it parses the code).

  authored_checks.py TASKS.jsonl     print {task id: {errors, warnings}} for every task with findings
"""
from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

THIN_WORDS = 20
NEURALESE_CONSTRUCTS = frozenset({'typed_neuralese', 'neuralese_relay', 'neuralese_fan_in', 'iterate_neuralese'})
ONE_LINE_WORDS = 25
STOP_WORD = re.compile(r'\b(stop|stops|stopping|stopped|until|halt|halts|terminate|terminates|ends? when)\b', re.I)
CODE = re.compile(r'```.*?```|`[^`\n]+`', re.S)
CALL = re.compile(r'\b([A-Za-z_]\w*)\s*\(')
SKILL_PATH = re.compile(r'^(.*)/skills/([^/]+)/')
HEAD = re.compile(r'^---\r?\n(.*?)\r?\n---\r?\n?(.*)$', re.S)


def split_file(text):
    match = HEAD.match(text)
    return (match.group(1), match.group(2)) if match else ('', text)


def split_top(text, separators):
    """Split at separators that are outside brackets, braces, parentheses, angle brackets and quotes."""
    parts, depth, quote, start, i = [], 0, None, 0, 0
    while i < len(text):
        char = text[i]
        if quote:
            if char == '\\':
                i += 1
            elif char == quote:
                quote = None
        elif char in '"\'`':
            quote = char
        elif char in '([{<':
            depth += 1
        elif char in ')]}>':
            if not (char == '>' and text[i - 1:i] == '='):
                depth -= 1
        elif char in separators and depth == 0:
            parts.append(text[start:i])
            start = i + 1
        i += 1
    parts.append(text[start:])
    return [part.strip() for part in parts if part.strip()]


def declared_args(header):
    """(required, total) argument counts from the frontmatter ``args`` field (block or inline form)."""
    lines = header.splitlines()
    for index, line in enumerate(lines):
        match = re.match(r'^args:(.*)$', line)
        if not match:
            continue
        inline = match.group(1).strip()
        block = []
        while index + 1 < len(lines) and re.match(r'^(\s|$)', lines[index + 1]):
            index += 1
            block.append(lines[index])
        text = inline + '\n' + '\n'.join(block)
        if inline.startswith('{'):
            members = split_top(inline.strip()[1:-1] if inline.strip().endswith('}') else inline[1:], ',;')
        else:
            indent = min((len(b) - len(b.lstrip()) for b in block if b.strip()), default=0)
            members = [b.strip() for b in block if b.strip() and len(b) - len(b.lstrip()) == indent]
        names = [re.match(r'^([A-Za-z_$][\w$]*)(\?)?\s*:', member) for member in members]
        names = [m for m in names if m]
        return sum(1 for m in names if not m.group(2)), len(names)
    return 0, 0


def call_arguments(text, open_paren):
    """The argument strings of the call whose '(' is at open_paren, or None when it is not closed."""
    depth, quote, i = 0, None, open_paren
    while i < len(text):
        char = text[i]
        if quote:
            if char == '\\':
                i += 1
            elif char == quote:
                quote = None
        elif char in '"\'`':
            quote = char
        elif char in '([{':
            depth += 1
        elif char in ')]}':
            depth -= 1
            if depth == 0:
                return split_top(text[open_paren + 1:i], ',')
        i += 1
    return None


def arity_problems(code, children):
    """Calls to direct helpers (``children``: name -> (required, total)) with the wrong number of arguments."""
    problems = []
    for match in CALL.finditer(code):
        name = match.group(1)
        arguments = call_arguments(code, match.end() - 1)
        if arguments is None:
            continue
        if name == 'decide' and arguments and arguments[0] in children:
            name, arguments = arguments[0], arguments[1:]
        if name not in children:
            continue
        required, total = children[name]
        if not required <= len(arguments) <= total:
            problems.append(f'{name} is called with {len(arguments)} argument(s) but declares '
                            + (f'{total}' if required == total else f'{required} to {total}'))
    return problems


def functions_of(task):
    """path -> (header, body) for every .nl file."""
    return {path: split_file(text) for path, text in task['files'].items() if path.endswith('.nl')}


def children_of(functions, path):
    """Direct helpers of the function at path: name -> declared (required, total) argument counts."""
    folder = path[:-3] + '/'
    return {other[len(folder):-3]: declared_args(functions[other][0]) for other in functions
            if other.startswith(folder) and '/' not in other[len(folder):]}


def lint_task(task):
    errors, warnings = [], []
    constructs = set(task.get('constructs') or [])
    functions = functions_of(task)
    root = task['root']
    folders = task.get('folders')
    root_header = functions.get(root, ('', ''))[0]
    reducer_root = bool(re.search(r'^kind:\s*directory-reducer\s*$', root_header, re.M))

    if 'folder_reducer' in constructs and not folders:
        errors.append('folder_reducer needs `folders` (path -> file text): the root reduces the task\'s world folder, '
                      'it must not create its own items')
    if reducer_root and not folders:
        errors.append('a `kind: directory-reducer` root needs `folders`')

    for path, (header, body) in functions.items():
        name = path.rsplit('/', 1)[-1][:-3]
        if re.search(r'^readout:\s*decision', header, re.M):
            others = {p.rsplit('/', 1)[-1][:-3] for p in functions if p != path}
            called = sorted({n for n in CALL.findall(body) if n in others})
            if called:
                errors.append(f'readout helper {path} calls {", ".join(called)}: a decision readout is scored in '
                              'one pass with no tools, so calls inside it never run')
            if any(other.startswith(path[:-3] + '/') for other in task['files']):
                errors.append(f'readout helper {path} owns a callable folder: a readout is a leaf')
        if 'iterateOn' in body and not STOP_WORD.search(body):
            errors.append(f'{path} uses iterateOn without an explicit stopping condition (say when to stop)')
        for code in CODE.findall(body):
            for problem in arity_problems(code, children_of(functions, path)):
                errors.append(f'{path}: {problem}')

    smoke = task.get('smoke') or {}
    if constructs & NEURALESE_CONSTRUCTS and not smoke.get('replay'):
        errors.append('Neuralese constructs need `smoke.replay: true`: the smoke code is then the whole root script '
                      '(helpers answer with scripted values) and its returned value is judged by the oracle, so the '
                      'exact steps behind `expected` are checked by a real run, not only by the blind solve')
    if smoke.get('replay'):
        tail = re.search(r'return\s+([^;]*?);?\s*$', str(smoke.get('code', '')).strip(), re.S)
        if not tail:
            errors.append('smoke.replay code must end with `return <answer>`')
        elif re.fullmatch(r'"[^"]*"|\'[^\']*\'|-?\d[\d.]*|true|false|null', tail.group(1).strip()):
            warnings.append('heuristic: smoke.replay returns a bare literal: the replay checks no code step; compute '
                            'the answer from the helpers\' scripted values')
    if {'iterate', 'iterate_neuralese'} & constructs and not any('iterateOn' in b for _, b in functions.values()):
        errors.append('iterate/iterate_neuralese tag but no function uses iterateOn')

    for path in task['files']:
        match = SKILL_PATH.match(path)
        if not match:
            continue
        folder, skill = match.groups()
        owner = folder + '.nl'
        if owner not in functions:
            errors.append(f'skill {path}: no function {owner} owns folder {folder}/, so no call can see it')
            continue
        readers = sorted(p for p, (_, body) in functions.items()
                         if p != owner and re.search(rf'skills[./]{re.escape(skill)}\b', body))
        if readers:
            errors.append(f'skill {skill} is bound to {owner} (the function owning {folder}/) but read by '
                          f'{", ".join(readers)}; put it in the folder of the function that reads it')

    if 'call_depth_3' in constructs:
        stem = root[:-3]
        middles = [(p[:p.rindex('/')] + '.nl', p) for p in functions if p.startswith(stem + '/') and p.count('/') >= 2]
        middles = [(m, s) for m, s in middles if m in functions and m != root]
        if not middles:
            errors.append('call_depth_3 needs a root -> helper -> sub-helper chain (a file two folders below the root)')
        for middle, sub in middles:
            sub_name = sub.rsplit('/', 1)[-1][:-3]
            (m_header, m_body), (s_header, s_body) = functions[middle], functions[sub]
            if not re.search(rf'\b{re.escape(sub_name)}\b', m_body):
                errors.append(f'call_depth_3: {middle} never calls its sub-helper {sub_name}')
                continue
            own = len(re.sub(rf'[^.\n]*\b{re.escape(sub_name)}\b[^.\n]*', ' ', m_body).split())
            same = (declared_args(m_header) == declared_args(s_header)
                    and re.sub(r'\s+', '', re.sub(r'^.*?returns:', '', m_header, flags=re.S))
                    == re.sub(r'\s+', '', re.sub(r'^.*?returns:', '', s_header, flags=re.S)))
            if same:
                warnings.append(f'heuristic: {middle} has the same arguments and return type as {sub_name}: '
                                'a pass-through is not a distinct subproblem')
            elif own < THIN_WORDS:
                warnings.append(f'heuristic: {middle} adds under {THIN_WORDS} words of its own work around '
                                f'{sub_name}: does it own a different subproblem?')
            if (re.search(r'returns:\s*"?boolean', s_header) or re.search(r'readout:\s*decision', s_header)) \
                    and len(s_body.split()) < ONE_LINE_WORDS:
                warnings.append(f'heuristic: sub-helper {sub_name} is a short yes/no comparison: exact comparisons '
                                'belong in code, not in a helper')
    new_errors, new_warnings = blind_failure_modes(task, functions)
    return errors + new_errors, warnings + new_warnings


ROUND6_FIRST_WORKER = 136
ID_TOKEN = re.compile(r'\b[A-Za-z]{1,6}[-_]\d{1,5}\b')
QUOTED = re.compile(r'["\u201c`]([^"\u201d`\n]{3,40})["\u201d`]')
CODEISH = re.compile(r'[{}()=;<>\[\]|\\$]|=>|\.\w+\(')
MOVE_WORDS = re.compile(r'\b(move|moves|moved|moving|file it|files it|archive|archives|refile|refiles)\b', re.I)
NAME_RULE = re.compile(r'(same (file )?name|keeps? (its|their) (file )?name|<same|<name>|<file|<id>|file name|filename|renamed? to)', re.I)
SORT_WORDS = re.compile(r'\b(sort|sorted|sorts|sorting|ordered by|in order of|ranked|ranks?)\b', re.I)
TIE_WORDS = re.compile(r'\b(ties|tie|tied|then by|then the|break|breaks|equal|same (time|date|key|value|count|score)|identical)\b', re.I)
REOPEN_WORDS = re.compile(r'\b(reopen\w*|re-open\w*|re-?fit\w*|re-?issue\w*|re-?submit\w*|reinstat\w*|reactivat\w*|'
                          r'restor(?:e|es|ed)|retract\w*|supersed\w*|reversal|reverse[sd]?|undo\w*|corrected|corrections?)\b', re.I)
CLEAR_WORDS = re.compile(r'\b(clear\w*|reset\w*|keeps?|kept|retain\w*|replac\w*|overwrit\w*|supersed\w*|remov\w*|drop\w*|'
                         r'discard\w*|no longer|still (counts?|stands?|applies)|stands?|stays?|erase\w*|cancel\w*|ignor\w*|'
                         r'wipe\w*|restart\w*|starts? (over|fresh|again)|latest|last (one|entry|line|note) wins)\b', re.I)
SORT_KEY = re.compile(r'(?:sorted|sort|ordered|order|ranked)\s+(?:ascending|descending|alphabetically|numerically)?\s*by\s+([^.;,\n]+)', re.I)
CONTRACT = re.compile(r'(?:^|\n)\s*\**Output contract:?\**:?(.*?)(?=\n\s*\n|$)', re.S | re.I)


def world_corpus(task):
    """All text of the task's world: folder files and input strings (flattened), lower-cased pieces kept for lookups."""
    parts = list((task.get('folders') or {}).values()) + list((task.get('folders') or {}).keys())

    def walk(value):
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, dict):
            for key, item in value.items():
                parts.append(str(key))
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
    walk(task.get('inputs'))
    return '\n'.join(parts)


def sentences(text):
    return [part for part in re.split(r'(?<=[.!?])\s+|\n+', text) if part.strip()]


def round6_worker(task):
    match = re.fullmatch(r'w(\d+)', str((task.get('author') or {}).get('worker', '')))
    return bool(match) and int(match.group(1)) >= ROUND6_FIRST_WORKER


def blind_failure_modes(task, functions):
    """(errors, warnings) for the static blind-solve failure modes (see the module docstring)."""
    errors, warnings = [], []
    strict = round6_worker(task)
    corpus = world_corpus(task)
    low = corpus.lower()
    root = task['root']
    root_body = functions.get(root, ('', ''))[1]
    helpers = {p: hb for p, hb in functions.items() if p != root}

    def mechanical(message):
        warnings.append('heuristic: ' + message)

    # 1. A helper reads a labelled line or quotes a phrase that no note / input / the root contains.
    for path, (header, body) in helpers.items():
        for phrase in sorted(set(QUOTED.findall(body))):
            clean = phrase.strip()
            if CODEISH.search(clean) or len(clean) < 4 or clean.lower() in header.lower():
                continue
            labelled = clean.endswith(':')
            if not (labelled or ' ' in clean):
                continue
            if clean.lower() in low or clean.lower() in root_body.lower():
                continue
            if labelled or re.search(r'\b(line|lines|note|notes|starts?|begins?|reads?|says?|saying|contains?|mentions?)\b',
                                     body[max(0, body.find(phrase) - 80):body.find(phrase)], re.I):
                mechanical(f'{path} reads "{clean}" but no note, folder file or input of the world contains it: the helper '
                           'describes wording the notes do not use (a solver must guess the mapping)')
    # 2. A helper instruction quotes a world entry (id or long note line): first-match child scripting breaks.
    ids = set(ID_TOKEN.findall(corpus))
    lines = [line.strip() for line in corpus.splitlines() if len(line.strip()) >= 24]
    for path, (header, body) in helpers.items():
        quoted_ids = sorted(token for token in ids if re.search(rf'(?<![\w-]){re.escape(token)}(?![\w-])', body))
        quoted_lines = [line for line in lines if line in body]
        if quoted_ids or quoted_lines:
            mechanical(f'{path} quotes world entries ({", ".join((quoted_ids + [l[:30] + "..." for l in quoted_lines])[:3])}): '
                       'put no example ids or note lines in a helper instruction, because reference children match fragments of the '
                       'call opening and a quoted entry makes every call match the first scripted child')
    # 2b. The reference matcher's own symptom: a scripted `$match` fragment that occurs in the helper's instruction text matches
    # EVERY call of that helper (the correct value with a wrong world). Applies to every task, not only round-6 workers.
    scripted = (task.get('smoke') or {}).get('children') or {}
    for name, value in scripted.items():
        entries = value.get('$match') if isinstance(value, dict) else None
        files = [(p_, hb) for p_, hb in helpers.items() if p_.rsplit('/', 1)[-1][:-3] == name]
        for entry in entries or []:
            fragment = str(entry.get('match', '')).strip().lower()
            if len(fragment) < 6:
                continue
            for path, (header, body) in files:
                if fragment in (header + '\n' + body).lower():
                    errors.append(f'smoke child {name}: the match fragment "{entry["match"][:40]}" occurs in the instruction text of '
                                  f'{path}, so every call of the helper opens with it and the first scripted entry answers all '
                                  'calls (a correct value with a wrong world): match on text only the call arguments contain')
    # 3. Destination file name unstated when the root moves files.
    if task.get('folders') and MOVE_WORDS.search(root_body) and not NAME_RULE.search(root_body):
        warnings.append('heuristic: the root moves files but no sentence says the destination file name (state e.g. '
                        '`filed/<same name>`: a moved file keeps its exact name and text)')
    # 4. Sorting without a tie order.
    for path, (header, body) in functions.items():
        if SORT_WORDS.search(body) and not TIE_WORDS.search(body):
            warnings.append(f'heuristic: {path} sorts or ranks but states no tie order (name the tie-break key, or say that '
                            'equal keys keep file-name order)')
    # 5. Reopen / correction rule silent on clearing the earlier record.
    for path, (header, body) in functions.items():
        for number, sentence in enumerate(sentences(body)):
            if REOPEN_WORDS.search(sentence):
                window = ' '.join(sentences(body)[number:number + 2])
                if not CLEAR_WORDS.search(window):
                    warnings.append(f'heuristic: {path}: "{sentence.strip()[:70]}..." names a reopened / corrected / reversed item '
                                    'without saying whether its earlier record is cleared or kept')
                    break
    # 6. Output contract present and consistent with the numbered steps.
    contract = CONTRACT.search(root_body)
    if not contract:
        if strict:
            errors.append('the root has no `Output contract:` paragraph (paths, line templates, separators, number format, '
                          'empty case, sort key and tie-break, counted cases)')
    else:
        text = contract.group(1)
        steps = root_body[:contract.start()] + root_body[contract.end():]
        for template in sorted(set(re.findall(r'`([^`\n]*(?:\||<[^>]+>)[^`\n]*)`', text))):
            if template.strip() and template not in steps and template not in root_body.replace(text, ''):
                warnings.append(f'heuristic: the output contract template `{template[:50]}` is not the template of any numbered step: '
                                'contract and steps must give the same line format')
        keys_contract = [re.sub(r'\W+', ' ', k.lower()).split() for k in SORT_KEY.findall(text)]
        keys_steps = [re.sub(r'\W+', ' ', k.lower()).split() for k in SORT_KEY.findall(steps)]
        if keys_contract and keys_steps and not any(set(a) & set(b) for a in keys_contract for b in keys_steps):
            warnings.append('heuristic: the output contract sorts by a different key than the numbered steps')
    return errors, warnings


def helper_signature(task):
    return {(path.rsplit('/', 1)[-1], re.sub(r'\s+', ' ', text)) for path, text in task['files'].items()
            if path.endswith('.nl') and path != task['root'] and '/' in path}


def lint_batch(tasks):
    """{task id: {'errors': [...], 'warnings': [...]}} for every task, with the batch-level world checks."""
    report = {}
    for task in tasks:
        errors, warnings = lint_task(task)
        report[task['id']] = {'errors': errors, 'warnings': warnings}
    worlds = collections.defaultdict(list)
    for task in tasks:
        if 'multi_operation_codebase' in (task.get('constructs') or []):
            worlds[task['world']].append(task)
    for world, group in worlds.items():
        roots = {task['root'] for task in group}
        if len(roots) < 2:
            for task in group:
                report[task['id']]['errors'].append(
                    f'multi_operation_codebase: world {world} has one root operation ({task["root"]}) among its tagged '
                    'tasks; several root operations over shared helper files are required')
            continue
        by_helper = collections.Counter(sig for root in roots
                                        for sig in set().union(*(helper_signature(t) for t in group if t['root'] == root)))
        if not any(count >= 2 for count in by_helper.values()):
            for task in group:
                report[task['id']]['errors'].append(
                    f'multi_operation_codebase: world {world} has {len(roots)} roots but no helper file (same name, identical '
                    'text) in the folders of two or more of them; one reusable program means shared helper files')
    return report


def main(argv=None):
    argv = argv or sys.argv[1:]
    if len(argv) != 1:
        raise SystemExit(__doc__)
    tasks = [json.loads(line) for line in Path(argv[0]).read_text().splitlines() if line.strip()]
    findings = {k: v for k, v in lint_batch(tasks).items() if v['errors'] or v['warnings']}
    print(json.dumps(findings, indent=1))
    return 1 if any(v['errors'] for v in findings.values()) else 0


if __name__ == '__main__':
    sys.exit(main())
