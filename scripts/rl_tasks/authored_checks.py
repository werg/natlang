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

Verifier tasks (``oracle: verifier``, judged by executing their outcome) get direct controls of their verifier module
(``verifier_controls.mjs`` through the runtime's trusted-verifier binding): the reference and every alias accepted,
every wrong answer, degenerate values (null, {}, [], "", 0) and the keyword baseline rejected, the reference verdict
deterministic. A verifier that accepts a wrong answer or the keyword baseline does not discriminate, and fails the task.
Their instructions may not spell a judgment as a keyword rule ("contains the word ...", "by whole words"): the cue list
is what the keyword baseline already does, and the task exists to need the meaning.

  authored_checks.py TASKS.jsonl [--runtime TS_HOST]     print {task id: {errors, warnings}} for every task with findings
"""
from __future__ import annotations

import collections
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CUE_RULE = re.compile(r'\bcontains?\s+(?:the|a|any|one|either)?\s*(?:of\s+the\s+)?(?:words?|phrases?|strings?|keywords?)\b'
                      r'|\bby\s+whole\s+words?\b|\bkeywords?\b', re.I)

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
    errors += semantic_fp_problems(task, constructs, functions)

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


def _scripts(task):
    """Every root script a task carries: the smoke code, the reference evals and the wrong-code variants."""
    reference = task.get('reference')
    actions = reference if isinstance(reference, list) else (reference or {}).get('root') or []
    codes = [str((task.get('smoke') or {}).get('code') or '')]
    codes += [str(args.get('code') or '') for tool, args in actions if tool == 'eval' and isinstance(args, dict)]
    return '\n'.join(codes)


# A semantic-FP construct is claimed only when the program and the scripts that solve it really use it
# (plans/semantic-fp/05 §3): (pattern, where, message). `where` is 'scripts' (smoke/reference code), 'program' (the
# .nl/.ts files) or 'both' (either).
SEMANTIC_FP_CHECKS = {
    'inline_nl': (r'\bnl(?:\.with)?(?:<[^`]*>)?(?:\([^)]*\))?`', 'scripts',
                  'inline_nl: the smoke or reference script never creates an inline nl`...` lambda'),
    'nl_with_context': (r'\bnl\.with\b', 'scripts', 'nl_with_context: no script uses nl.with({...})`...`'),
    'nl_convergence': (r'\.until\(\s*nl\b', 'both', 'nl_convergence: no iterateOn(...).until(nl`...`) stop'),
    'context_lens': (r'\.in\(', 'both', 'context_lens: no script or function uses .in(context)'),
    'folder_transaction': (r'\.propose\(', 'both', 'folder_transaction: nothing calls folder.propose(...)'),
    'untrusted_contract': (r'\bUntrusted<', 'program', 'untrusted_contract: no argument or type is Untrusted<T>'),
    'refinement_contract': (r'\bIs<|^ensures:|(?<![.\w$])refine\s*[<(]', 'both',
                            'refinement_contract: no Is<T, P> type, refine(...) call or ensures: contract'),
    'ensures_contract': (r'^ensures:', 'program', 'ensures_contract: no .nl function declares ensures: postconditions'),
    'structural_recursion': (r'^(?:recursive:\s*structural|measure:)', 'program',
                             'structural_recursion: no .nl function declares recursive: structural or measure:'),
    # @natlang/std combinators: the program imports the combinator from the standard library and calls it.
    **{construct: (rf'@natlang/(?:node|browser)/std[\s\S]*\b{name}(?:<[^>()]*>)?\(|\b{name}(?:<[^>()]*>)?\([\s\S]*@natlang/(?:node|browser)/std', 'program',
                   f'{construct}: no program file imports {name} from @natlang/node/std and calls it')
       for construct, name in (('semantic_map', 'semanticMap'), ('semantic_filter', 'semanticFilter'),
                               ('semantic_partition', 'semanticPartition'), ('semantic_reduce', 'semanticReduce'),
                               ('semantic_match', 'semanticMatch'), ('semantic_rank', 'semanticRank'),
                               ('semantic_merge', 'semanticMerge'), ('semantic_repair', 'semanticRepair'),
                               ('fixed_point', 'fixedPoint'), ('escalate', 'escalate'),
                               ('with_context', 'withContext'), ('compile_rule', 'compileRule'),
                               ('propose_and_verify', 'proposeAndVerify'), ('branch_and_select', 'branchAndSelect'))},
}


def semantic_fp_problems(task, constructs, functions):
    errors = []
    scripts = _scripts(task)
    program = '\n'.join(task['files'].values())
    for construct, (pattern, where, message) in SEMANTIC_FP_CHECKS.items():
        if construct not in constructs:
            continue
        text = {'scripts': scripts, 'program': program, 'both': scripts + '\n' + program}[where]
        if not re.search(pattern, text, re.M):
            errors.append(message)
    if 'folder_transaction' in constructs and not re.search(r'\.(accept|select)\(', scripts + '\n' + program):
        errors.append('folder_transaction: a proposal is never accepted or selected')
    if 'library_discovery' in constructs:
        root = task['root']
        stem = root[:-3] + '/'
        helpers = sorted(p.rsplit('/', 1)[-1].rsplit('.', 1)[0] for p in task['files']
                         if p.startswith(stem) and p.endswith(('.nl', '.ts')) and p.count('/') == stem.count('/'))
        if len(helpers) < 10:
            errors.append(f'library_discovery: the root library has {len(helpers)} helpers; it needs at least 10')
        named = [h for h in helpers if re.search(rf'\b{re.escape(h)}\b', functions.get(root, ('', ''))[1])]
        if named:
            errors.append('library_discovery: the root instruction names its helpers ' + ', '.join(named) +
                          '; the task is to find them')
    if 'higher_order' in constructs:
        returns_fn = any(re.search(r'returns:\s*"[^"\n]*=>', header) for header, _ in functions.values())
        if not returns_fn and not re.search(r'=>\s*Promise<', program) and not re.search(r'\bnl(?:\.with)?\b', scripts):
            errors.append('higher_order: no function returns or takes a callable, and no script builds one')
    return errors


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
    # 6. Output contract present and consistent with the numbered steps (legacy exact desks only; verifier tasks specify
    # their contract via types, ensures:, and executable verifiers rather than a prose Output contract block).
    contract = CONTRACT.search(root_body)
    if not contract:
        if strict and task.get('oracle') != 'verifier':
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


def node():
    for candidate in sorted(Path.home().glob('.local/share/natlang-dev/node-v24*/bin/node'), reverse=True):
        return str(candidate)
    return 'node'


def verifier_controls(tasks, runtime=None):
    """{task id: {'errors': [...], 'verdicts': {...}}} of ``verifier_controls.mjs`` for the batch's verifier tasks."""
    chosen = [task for task in tasks if task.get('oracle') == 'verifier']
    if not chosen:
        return {}
    runtime = runtime or os.environ.get('NATLANG_VERIFIER_RUNTIME') or str(REPO / 'ts-host')
    text = ''.join(json.dumps(task) + '\n' for task in chosen)
    done = subprocess.run([node(), str(Path(__file__).resolve().parent / 'verifier_controls.mjs'), runtime, '/dev/stdin'],
                          input=text, capture_output=True, text=True,
                          cwd=os.environ.get('NATLANG_REPO_ROOT') or str(REPO))
    if done.returncode:
        raise RuntimeError('verifier_controls.mjs failed: ' + done.stderr[-2000:])
    return json.loads(done.stdout)


def cue_rules(task):
    """Instructions of a verifier task that spell a judgment as a keyword rule."""
    found = []
    for path, text in task['files'].items():
        if path.endswith('.nl'):
            match = CUE_RULE.search(split_file(text)[1])
            if match:
                found.append(f'{path} spells a judgment as a keyword rule ("{match.group(0)}"): a verifier task states '
                             'what the judgment means and leaves surface cues to the keyword baseline it must beat')
    return found


# A custody group names one real shared source item: `<dataset>:<item id>` (e.g. `strategyqa:18689be402e9`). A group is
# a custody identity, so one held world withholds every world that shares it (sfp1: family-wide `invented:<theme>`
# groups tied 28 of 36 train rows to held worlds).
REAL_GROUP = re.compile(r'^[a-z0-9][a-z0-9_.-]*:[^\s:/]{4,}$')


def group_problems(tasks):
    """{task id: [errors]}: groups that are not one real source item (invented, family-wide or theme-wide)."""
    found = collections.defaultdict(list)
    for task in tasks:
        groups = task.get('groups') or []
        if not isinstance(groups, list) or not all(isinstance(group, str) for group in groups):
            found[task['id']].append('groups must be a list of strings')
            continue
        for group in groups:
            if group.lower().startswith('invented') or not REAL_GROUP.match(group):
                found[task['id']].append(
                    f'group {group!r} is not one real source item (`<dataset>:<item id>`): a group is a custody identity '
                    'and joins every world that carries it, so invented worlds leave groups empty')
            elif task.get('oracle') == 'verifier' and group.split(':', 1)[0] == task.get('family'):
                found[task['id']].append(f'group {group!r} names the family, which ties every world of it together')
    return found


def lint_batch(tasks, runtime=None):
    """{task id: {'errors': [...], 'warnings': [...]}} for every task, with the batch-level world checks."""
    report = {}
    for task in tasks:
        errors, warnings = lint_task(task)
        if task.get('oracle') == 'verifier':
            errors.extend(cue_rules(task))
        report[task['id']] = {'errors': errors, 'warnings': warnings}
    for ident, found in verifier_controls(tasks, runtime).items():
        report[ident]['errors'].extend(found['errors'])
    for ident, found in group_problems(tasks).items():
        report[ident]['errors'].extend(found)
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
    runtime = None
    if len(argv) == 3 and argv[1] == '--runtime':
        argv, runtime = argv[:1], argv[2]
    if len(argv) != 1:
        raise SystemExit(__doc__)
    tasks = [json.loads(line) for line in Path(argv[0]).read_text().splitlines() if line.strip()]
    findings = {k: v for k, v in lint_batch(tasks, runtime).items() if v['errors'] or v['warnings']}
    print(json.dumps(findings, indent=1))
    return 1 if any(v['errors'] for v in findings.values()) else 0


if __name__ == '__main__':
    sys.exit(main())
