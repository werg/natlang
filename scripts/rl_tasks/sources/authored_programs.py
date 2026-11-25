#!/usr/bin/env python3
"""Authored programs (`natlang.authored-program/1`): semantic functional programs with authored execution trajectories.

Owner 2026-11-25: "why don't we just author entire trajectories directly for the IR without having to run it through
a model? Just have the author write the entire execution trajectory to train on and then all we need to do is collect
the reasoning in between the actions." "We very, very strongly need to limit the amount of crisp TypeScript code ...
It absolutely has to be a deep code base of purely natural language functions able to call each other ... use a lot
of the NatLang specific functionality ... we need statistics not checks on our then generated trajectories ...
feeding analysis back into top of funnel prompting." "Remember we want to do semantic functional programming."

A builder (applications/program-inventor/BUILD_BRIEF.md) writes a program (a deep tree of typed `.nl` functions,
composed with the semantic-FP constructs), two or three literal input instances, and for each instance the whole
execution trajectory: the root's actions and every other call's turns or answer, open generative answers included.
The trajectory is replayed through the collector's own execution path with no model (scripts/rl_tasks/
authored_replay.mjs over curriculum.replayReference): the only check is that it is consistent with the program (every
call answered, every action runs, every value has its declared type). The reasoning between actions is filled later
(filled_reasoning.py); the per-call SFT rows are rendered by the authored-trajectory exporter
(scripts/rl_tasks/reference_episodes.mjs, scripts/neuralese_data/authored_trajectories.py). An instance that declares
an exact oracle (`rl`) is also an RL task, an optional by-product.

  authored_programs.py check SUBMISSION [--runtime TS_HOST] [--no-natlang-check]
      static shape + `natlang check --json` + the replay; prints each instance's call tree, the entry that answered
      each call, failures, and the program's statistics. Writes nothing into SUBMISSION. Seconds.
  authored_programs.py admit SUBMISSION [--root ROOT] [--idea FILE]
      check, then ROOT/admitted/<slug>/: trajectories.jsonl (every instance as a natlang.program/2 case whose
      curriculum.reference is the authored trajectory and whose expected value is what it returns), cases.jsonl (the
      instances with an `rl` oracle, for `invented.py publish`), stats.json, admission.json; the idea enters
      ROOT/accepted.jsonl and its claim closes as accepted.
  authored_programs.py stats SUBMISSION [--runtime TS_HOST]       the statistics of one program (JSON)
  authored_programs.py wave-report WAVE_DIR [--runtime TS_HOST] [--previous REPORT.json]
      the statistics of every submission under WAVE_DIR/sub/, per program and for the wave, against the campaign's
      targets: WAVE_DIR/REPORT.md, WAVE_DIR/report.json and WAVE_DIR/EMPHASIS.draft.md (the next wave's emphasis,
      which the coordinator edits and installs as NEXT_WAVE/EMPHASIS.md: BUILD_BRIEF.md "Read first").

Trajectory entries (`instances[].trajectory`):
  root:  [[tool, args], ...]        the root call's turns in order (eval, return_result, ...)
  calls: [{fn?, match?, value? | turns? | call?}, ...]
         one entry per call below the root, or one entry for every call of a function whose turns are the same code
         (eval code reads the call's own parameters). `fn` names the function (`You are inside this call: fn(` in the
         call's opening); `match` fragments must all appear in the call's opening (its instructions and arguments); an
         inline `nl` call is matched by a fragment of its instructions. The entry with the most fragments that matches
         answers (author order breaks ties). `value` answers in one turn (a decision readout's choice, or
         return_result); `turns` plays several turns ([tool, args] each, e.g. an eval that calls this function's own
         callable items, then return_result).
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from scripts.rl_tasks.sources import invented  # noqa: E402
from scripts.rl_tasks.common import CURRICULUM_VERSION, slug, text_digest  # noqa: E402

SCHEMA = 'natlang.authored-program/1'
GENERATOR = 'natlang.authored-programs/1'
ENTRIES = frozenset({'program', 'submission.json', 'idea.json', 'NOTES.md', 'review.json'})
ANCHOR = 'You are inside this call: {}('

# Campaign targets: statistics the wave report compares against, never admission checks (owner 2026-11-25:
# "statistics not checks"). Revisable between waves; each report records the targets it used.
TARGETS = {
    'crisp_line_share_max': 0.15,      # TypeScript logic lines / (TypeScript logic + .nl instruction lines)
    'nl_depth_min': 4,                 # NL composition depth (root -> ... -> leaf) of the deepest chain, median instance
    'interior_functions_min': 3,       # .nl functions that call other natural-language functions in the trajectory
    'construct_kinds_min': 6,          # distinct semantic-FP construct kinds the trajectories use
    'higher_order_min': 1,             # higher-order uses per instance (functions passed, bound, returned)
    'open_output_share_min': 0.35,     # share of natural-language calls whose answer is open text
    'nl_call_share_min': 0.85,         # natural-language calls / (natural-language + crisp helper calls)
    'literals_from_data_min': 1,       # NL literals per instance whose text is built from runtime data (synthesis)
    'literal_tree_share_min': 0.2,     # share of NL calls that run through an NL literal (it or its descendants)
}
# Semantic-FP construct kinds, in the campaign's vocabulary (plans/semantic-fp/README.md, skills/natlang-authoring)
CONSTRUCTS = ('map', 'filter', 'partition', 'reduce', 'match', 'rank', 'repair', 'fixed-point', 'merge',
              'propose-verify', 'branch-select', 'escalate', 'with-context', 'compile-rule',
              'higher-order', 'partial', 'context-rebind', 'function-result', 'iterate-measure', 'iterate-nl-until',
              'structural-recursion', 'folder-reducer', 'folder-transaction', 'decision', 'decide', 'ensures',
              'refine', 'untrusted', 'refinement-type', 'neuralese', 'inline-nl', 'pluggable')
STD_KIND = {'semanticMap': 'map', 'semanticFilter': 'filter', 'semanticPartition': 'partition',
            'semanticReduce': 'reduce', 'semanticMatch': 'match', 'semanticRank': 'rank', 'semanticRepair': 'repair',
            'fixedPoint': 'fixed-point', 'semanticMerge': 'merge', 'proposeAndVerify': 'propose-verify',
            'branchAndSelect': 'branch-select', 'escalate': 'escalate', 'withContext': 'with-context',
            'compileRule': 'compile-rule'}
COMBINATOR_KINDS = frozenset(STD_KIND.values()) | {'iterate-measure', 'iterate-nl-until', 'folder-reducer',
                                                   'folder-transaction'}
CODE_PATTERNS = {
    'partial': r'\.partial\s*\(',
    'context-rebind': r'\.in\s*\(\s*(NatlangContext|Context)\b',
    'with-context': r'\bwithContext\s*\(',
    'decide': r'\bdecide\s*\(',
    'refine': r'\b(refine|assume)\s*\(',
    'folder-transaction': r'\bfolder\.(apply|propose|accept|iterateOn|snapshot|select)\s*\(|\bproposeAndVerify\s*\(|\bbranchAndSelect\s*\(',
    'iterate-measure': r'\.withMeasure\s*\(|\.withLimit\s*\(',
}
WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)?")


# ---- loading and packaging ---------------------------------------------------------------------------------------

def load(submission: Path) -> tuple[dict, dict[str, str]]:
    return invented.load(submission)


def children(trajectory: dict) -> list[dict]:
    """The authored `calls` as curriculum.reference children: `fn` becomes the opening's call anchor; the entry with the
    most fragments comes first (Python's sort is stable, so author order breaks ties)."""
    out = []
    for entry in trajectory.get('calls') or []:
        match = entry.get('match') or []
        match = [match] if isinstance(match, str) else list(match)
        if entry.get('fn'):
            match = [ANCHOR.format(entry['fn'])] + match
        child: dict = {'match': match}
        if 'turns' in entry:
            child['calls'] = entry['turns']
        elif 'call' in entry:
            child['call'] = entry['call']
        else:
            child['value'] = entry.get('value')
        out.append(child)
    order = sorted(range(len(out)), key=lambda i: -len(out[i]['match']))
    return [out[i] for i in order]


def entry_order(trajectory: dict) -> list[int]:
    """Author index of each reference child (children() order)."""
    calls = trajectory.get('calls') or []
    sizes = [len(([e['match']] if isinstance(e.get('match'), str) else list(e.get('match') or []))) + bool(e.get('fn'))
             for e in calls]
    return sorted(range(len(calls)), key=lambda i: -sizes[i])


def package(data: dict, files: dict[str, str], returned: dict[str, object] | None = None) -> list[dict]:
    """One natlang.program/2 case per instance. Its reference is the authored trajectory; its expected value is the
    instance's declared `rl.expected`, else what the trajectory returns (`returned`, from a first replay)."""
    family = 'auth_' + re.sub(r'[^a-z0-9]+', '_', data['slug'].lower()).strip('_')
    root = data['root']
    root_body = invented.frontmatter(files[root])[1]
    author = data.get('author') or {}
    cases = []
    for inst in data['instances']:
        trajectory = inst['trajectory']
        rl = inst.get('rl') or None
        verifier = invented.verifier_binding({'oracle': rl.get('oracle'), 'verifier': rl.get('verifier')}) if rl else None
        if rl:
            aliases = rl.get('alias_answers') or []
            oracle = 'exact' if verifier else ({'level': rl['oracle'], 'alternates': aliases} if aliases else rl['oracle'])
            expected = rl.get('expected')
        else:
            oracle, expected = 'exact', (returned or {}).get(inst['name'])
        payload = json.dumps({'files': files, 'instance': inst}, sort_keys=True)
        case_id = f'auth:{slug(data["slug"], 48)}:{slug(inst["name"], 24)}:{text_digest(payload)[:8]}'
        cases.append({
            'version': 'natlang.program/2', 'id': case_id, 'kind': 'lambda_source', 'family': family,
            'source': 'authored:' + str(author.get('builder', 'unknown')), 'split': invented.split_of(data),
            'source_ids': [f'authored:{data["slug"]}:{inst["name"]}'], 'source_groups': [invented.world(data)],
            'source_revisions': [GENERATOR], 'license': 'authored (project-authored by frontier models)',
            'gold_sources': ['authored-trajectory' + ('-with-oracle' if rl else '')],
            'generation': {'generator': GENERATOR, 'teacher_generation': True, 'author': author,
                           'method': 'a new program idea, its program, literal instances and the whole execution '
                                     'trajectory of each, written by a frontier model; the trajectory is replayed '
                                     'through the program with no model and must be consistent with it; reasoning '
                                     'between actions is filled later'},
            'curriculum': {
                'version': CURRICULUM_VERSION, 'family': family, 'family_version': 1, 'shape': data['slug'],
                'variant': 'authored', 'pair_group': None, 'split_group': invented.world(data),
                'slice': 'nested_scoped', 'domain': 'other', 'mode': 'single_call', 'inline': 'optional',
                'named': 'required', 'evidence': {'world': [], 'retrieved': [], 'background': []}, 'assumptions': [],
                'decisive': [], 'plausible_actions': [], 'minimum_sequence': [],
                'reference': {'root': trajectory['root'], 'children': children(trajectory),
                              **({'failures': int(inst['expected_failures'])} if inst.get('expected_failures') else {})},
                'rl': {'level': int(inst.get('level', 3)), 'family': family,
                       'basis': ('authored-trajectory-' + ('verifier' if verifier else 'exact') + '-oracle') if rl
                       else 'authored-trajectory-only'},
                'external': {'source': 'authored', 'upstream': 'authored:' + str(author.get('builder', 'unknown')),
                             'upstream_id': f'{data["slug"]}:{inst["name"]}', 'upstream_split': 'train-carve',
                             'unit': invented.world(data), 'shape': data['slug'],
                             'instruction_sha256': text_digest(root_body)[:16],
                             'instruction_id': text_digest(root_body)[:12], 'neuralese_room': '',
                             'framing': 'authored', 'subject_domain': 'other', 'oracle_depth': 'answer',
                             'rl_task': bool(rl),
                             'checks': {'wrong_answers': [], 'wrong_mutants': [], 'alias_answers': [],
                                        'keyword_baseline_code': [], 'ablations': []}}},
            'semantics': {'root': root, 'files': files, 'inputs': inst['inputs'], 'expected': expected,
                          'oracle': oracle, **({'trusted_exact_verifier': verifier} if verifier else {}),
                          **({'folder_files': inst['folder_files']} if inst.get('folder_files') else {})},
        })
    return cases


# ---- the check: shape, natlang check, replay -----------------------------------------------------------------------

def static_problems(submission: Path, data: dict, files: dict[str, str]) -> list[str]:
    problems = []
    if data.get('schema') != SCHEMA:
        problems.append(f'schema must be {SCHEMA}')
    root = data.get('root', '')
    if root not in files or '/' in root or not root.endswith('.nl'):
        return problems + ['root must name a top-level .nl file of program/']
    stem = root[:-3]
    for path in files:
        if path not in (root, 'types.ts') and not path.startswith(stem + '/'):
            problems.append(f'program/{path} must live in the root callable folder {stem}/ (or be types.ts)')
    for entry in submission.iterdir():
        if entry.name not in ENTRIES and not entry.name.startswith('.'):
            problems.append(f'{entry.name} is not part of a submission (program/, submission.json, idea.json, NOTES.md); '
                            'keep logs and scratch beside SUBMISSION')
    instances = data.get('instances') or []
    if not 1 <= len(instances) <= 3:
        problems.append('one to three instances')
    names = [i.get('name') for i in instances]
    if len(set(names)) != len(names) or not all(isinstance(n, str) and n for n in names):
        problems.append('every instance has its own name')
    for inst in instances:
        label = f'instance {inst.get("name")}'
        trajectory = inst.get('trajectory') or {}
        if not trajectory.get('root'):
            problems.append(f'{label}: trajectory.root (the root call\'s turns) is missing')
        for number, entry in enumerate(trajectory.get('calls') or [], 1):
            if not isinstance(entry, dict) or not (entry.get('fn') or entry.get('match')):
                problems.append(f'{label}: trajectory.calls[{number}] names no call: give fn, match or both')
            elif sum(k in entry for k in ('value', 'turns', 'call')) != 1:
                problems.append(f'{label}: trajectory.calls[{number}] needs exactly one of value, turns, call')
        rl = inst.get('rl')
        if rl is not None:
            if rl.get('oracle') not in invented.ORACLES:
                problems.append(f'{label}: rl.oracle must be exact, normalized or verifier')
            if rl.get('oracle') == 'verifier':
                path = (rl.get('verifier') or {}).get('path', '')
                if not path.startswith(invented.VERIFIER_ROOT) or not (REPO / path).is_file():
                    problems.append(f'{label}: rl.verifier must be an existing module under {invented.VERIFIER_ROOT}')
            elif 'expected' not in rl:
                problems.append(f'{label}: rl.expected is missing')
    return problems


def replay(cases: list[dict], runtime: Path) -> dict:
    with tempfile.TemporaryDirectory(prefix='authored-replay-') as scratch:
        source, target = Path(scratch) / 'cases.jsonl', Path(scratch) / 'replay.json'
        source.write_text(''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in cases))
        run = subprocess.run([invented.node(), str(REPO / 'scripts/rl_tasks/authored_replay.mjs'), str(runtime),
                              str(source), str(target)], capture_output=True, text=True, timeout=3600)
        try:
            return json.loads(target.read_text())
        except (OSError, ValueError):
            return {'error': (run.stderr or run.stdout)[-2000:], 'results': []}


def capture_cause(fragments: list[str], captured: list[str]) -> str:
    """When a fragment matched a variable an inline `nl` call captured (authored_replay.mjs `captured`), say so: a
    capture's declaration is part of every opening of that literal, so it matches calls the author meant apart."""
    for fragment in fragments:
        for text in captured:
            if fragment in text:
                line = next((ln for ln in text.splitlines() if fragment in ln), '')
                name = re.match(r'\s*(?:const|let)\s+([A-Za-z_$][\w$]*)', line)
                var = f'`{name.group(1)}`' if name else 'a variable'
                return (f'; the fragment {fragment!r} matched {var}, a variable of the calling code that the inline nl '
                        'captured (an inline nl captures each visible variable its instructions name, as a word too). '
                        'Match a fragment of its instructions or arguments, or list the captures with nl.with({ … }).')
    return ''


def fragments_of(entry: dict) -> list[str]:
    """An entry's match fragments as the replay tests them: the call anchor of `fn` first, then `match`."""
    match = entry.get('match') or []
    match = [match] if isinstance(match, str) else list(match)
    return ([ANCHOR.format(entry['fn'])] if entry.get('fn') else []) + match


def shows(opening: str, fragment: str) -> bool:
    """curriculum.ts `shows`: the fragment as written, or as its JSON-escaped text."""
    return fragment in opening or json.dumps(fragment, ensure_ascii=False)[1:-1] in opening


def answer_of(entry: dict):
    """What an entry answers: its value, or the value of its last return_result turn (None for other turns)."""
    if 'value' in entry:
        return entry['value']
    for turn in reversed(entry.get('turns') or []):
        if isinstance(turn, list) and turn and turn[0] == 'return_result':
            return (turn[1] or {}).get('value')
    return None


def call_label(answer: dict, limit: int = 160) -> str:
    shown = ' '.join(str(answer.get('shown', '')).split())
    return f'{answer.get("name")}({shown[:limit]}{"..." if len(shown) > limit else ""})'


def consistency(data: dict, rows: list[dict]) -> tuple[list[str], list[str]]:
    """The only check: the authored trajectory is consistent with the program. Warnings name entries that may answer
    calls they were not written for: the silent wrong answers (a fragment that also appears in a later call quoting an
    earlier record, an entry without `fn` reaching another function's call)."""
    problems, warnings = [], []
    for inst, row in zip(data['instances'], rows):
        label = f'instance {inst["name"]}'
        if row.get('error'):
            problems.append(f'{label}: the replay did not run: {row["error"][:800]}')
            continue
        missing = [a for a in row.get('answers') or [] if a.get('entry') == -1]
        for a in missing:
            problems.append(f'{label}: NO ENTRY for a call of {call_label(a)}'
                            + (' (answered by a placeholder for the rest of this replay)' if a.get('placeholder') else ''))
        if row.get('status') != 'done':
            problems.append(f'{label}: the trajectory does not finish the root call: {str(row.get("detail"))[:1200]}')
        failed = row.get('failed_actions') or []
        allowed = int(inst.get('expected_failures') or 0)
        if len(failed) != allowed:
            listed = '; '.join(f'{f["tool"]}: {f["result"][:600]}' for f in failed)
            problems.append(f'{label}: {len(failed)} actions failed where the trajectory declares {allowed}'
                            f'{" (expected_failures)" if allowed else ""}'
                            f'{" (some may follow from placeholder answers to missing entries)" if missing else ""}: {listed}'[:3000])
        if inst.get('rl') is not None and row.get('status') == 'done' and not row.get('accepted'):
            problems.append(f'{label}: rl: the trajectory returns {json.dumps(row.get("value"))[:300]}, which the '
                            'declared oracle rejects')
        trajectory = inst.get('trajectory') or {}
        entries = trajectory.get('calls') or []
        order = entry_order(trajectory)
        answered = collections.defaultdict(list)
        for a in row.get('answers') or []:
            if 0 <= a.get('entry', -1) < len(order):
                answered[order[a['entry']]].append(a)
        for index, calls in answered.items():
            entry = entries[index]
            names = sorted({a.get('name') for a in calls})
            if len(names) > 1:
                warnings.append(f'{label}: trajectory.calls[{index + 1}] ({entry.get("fn") or entry.get("match")}) answers '
                                f'calls of {len(names)} functions ({", ".join(names)}): its fragments also appear in other '
                                'functions\' openings; give it `fn`')
            shown = {a.get('shown', '') for a in calls}
            if 'value' in entry and len(shown) > 1 and len(fragments_of(entry)) >= 2:
                warnings.append(f'{label}: trajectory.calls[{index + 1}] ({entry.get("fn") or entry.get("match")}) '
                                f'answers {len(shown)} calls with different arguments with one value, e.g. '
                                f'{call_label(calls[0], 100)} and {call_label(calls[-1], 100)}: its fragments match more '
                                'than its item; take the start of the argument\'s line (`message: Message = { messageId: '
                                '"m3"`) or a passage only this call shows')
        # A call that several written-for-an-item entries match is answered by the one with the most fragments (author
        # order breaks ties): when they answer differently, one of them is answering a call it was not written for.
        for a in row.get('answers') or []:
            opening = a.get('opening') or ''
            if not opening or a.get('entry', -1) < 0:
                continue
            fits = [i for i, e in enumerate(entries) if e.get('match')
                    and all(shows(opening, f) for f in fragments_of(e))]
            if len(fits) > 1:
                answers = {json.dumps(answer_of(entries[i]), sort_keys=True, ensure_ascii=False)
                           if answer_of(entries[i]) is not None else f'turns:{i}' for i in fits}
                if len(answers) > 1:
                    won = order[a['entry']] if a['entry'] < len(order) else None
                    warnings.append(f'{label}: the call {call_label(a, 100)} is matched by trajectory.calls '
                                    f'{", ".join(str(i + 1) for i in fits)}, which answer differently; '
                                    f'{won + 1 if won is not None else "?"} answered it. A fragment of another entry '
                                    'probably quotes this call (a later call showing an earlier record): make each '
                                    'entry\'s fragments specific to its own call')
        for index, entry in enumerate(entries):
            if index not in answered:
                warnings.append(f'{label}: trajectory.calls[{index + 1}] ({entry.get("fn") or entry.get("match")}) '
                                'answered no call')
    return problems, warnings


NUMBER = re.compile(r'(?<![\w.])\d+(?:[.,]\d+)?(?![\w])')
NAME = re.compile(r'(?<=[a-z,;:] )[A-Z][a-z]+(?:[\'’][a-z]+)?')


def ungrounded(row: dict, entries: list[dict], order: list[int]) -> list[dict]:
    """Open answers that state numbers or names their call's opening does not show: the writer used what the author
    knew about the instance rather than what the call received (a statistic and a list for the builder, not a check)."""
    out = []
    for a in row.get('answers') or []:
        if not 0 <= a.get('entry', -1) < len(order) or not a.get('opening'):
            continue
        texts = [t for t in leaves(answer_of(entries[order[a['entry']]]), []) if len(WORD.findall(t)) >= 8]
        if not texts:
            continue
        opening = a['opening']
        lowered = opening.lower()
        tokens = {m.group(0) for t in texts for m in NUMBER.finditer(t)} | \
                 {m.group(0) for t in texts for m in NAME.finditer(t)}
        absent = sorted(t for t in tokens if t not in opening and t.lower() not in lowered
                        and t.replace(',', '.') not in opening)
        if absent:
            out.append({'call': call_label(a, 80), 'entry': order[a['entry']] + 1, 'absent': absent[:8]})
    return out


def check(submission: Path, runtime: Path, natlang: bool = True) -> dict:
    data, files = load(submission)
    problems = static_problems(submission, data, files) + invented.runtime_problems(runtime)
    if problems and any(p.startswith(('schema', 'root must')) for p in problems):
        return {'ok': False, 'problems': problems, 'warnings': [], 'rows': [], 'cases': []}
    if natlang:
        program = submission / 'program'
        problems += invented.natlang_check(program)
        # `natlang check` writes declarations (`*.d.nl.ts`) beside the sources; load() never reads them as program files
        for generated in program.rglob('*.d.nl.ts'):
            generated.unlink(missing_ok=True)
    try:
        first = package(data, files)
    except (KeyError, TypeError, ValueError) as error:
        return {'ok': False, 'problems': problems + [f'package: {error}'], 'warnings': [], 'rows': [], 'cases': []}
    result = replay(first, runtime)
    if result.get('error'):
        problems.append('authored_replay.mjs did not finish: ' + result['error'][-1200:])
    rows = result.get('results') or []
    found, warnings = consistency(data, rows)
    problems += found
    returned = {inst['name']: row.get('value') for inst, row in zip(data['instances'], rows)}
    cases = package(data, files, returned)
    for case, row in zip(cases, rows):
        if row.get('model_calls'):
            case['semantics']['episode_model_calls'] = invented.REFERENCE_CALL_FACTOR * int(row['model_calls'])
    return {'ok': not problems, 'problems': problems, 'warnings': warnings, 'rows': rows, 'cases': cases,
            'stats': program_stats(data, files, rows)}


# ---- statistics ----------------------------------------------------------------------------------------------------

def body_lines(text: str) -> list[str]:
    return [line for line in invented.frontmatter(text)[1].splitlines() if line.strip()]


def ts_logic_lines(text: str) -> int:
    return sum(1 for line in invented.strip_comments(text).splitlines()
               if line.strip() and not line.strip().startswith(('import ', 'export type', 'type ', '}')))


def front(text: str) -> dict[str, str]:
    head = invented.frontmatter(text)[0]
    out, key = {}, None
    for line in head.splitlines():
        m = re.match(r'^([A-Za-z_]+):\s*(.*)$', line)
        if m:
            key = m.group(1)
            out[key] = m.group(2)
        elif key:
            out[key] += '\n' + line
    return out


def source_stats(data: dict, files: dict[str, str]) -> dict:
    nl = {p: t for p, t in files.items() if p.endswith('.nl')}
    ts = {p: t for p, t in files.items() if p.endswith(('.ts', '.js', '.mjs')) and p != 'types.ts'
          and not p.endswith('.d.nl.ts')}
    nl_lines = sum(len(body_lines(t)) for t in nl.values())
    nl_words = sum(len(WORD.findall(' '.join(body_lines(t)))) for t in nl.values())
    crisp = {p: ts_logic_lines(t) for p, t in ts.items()}
    fronts = {p: front(t) for p, t in nl.items()}
    declared = collections.Counter()
    for f in fronts.values():
        if f.get('readout', '').strip() in ('decision', 'instant'):
            declared['decision'] += 1
        if 'recursive' in f or 'measure' in f:
            declared['structural-recursion'] += 1
        if 'ensures' in f:
            declared['ensures'] += 1
        if f.get('kind', '').strip() == 'directory-reducer':
            declared['folder-reducer'] += 1
        if '=>' in f.get('args', ''):
            declared['higher-order'] += 1
        if '=>' in f.get('returns', ''):
            declared['function-result'] += 1
    every = '\n'.join(files.values())
    for name, pattern in (('untrusted', r'\bUntrusted<'), ('refinement-type', r'\bIs<'), ('neuralese', r'\bNeuralese<'),
                          ('pluggable', r'\bpluggable\s*\(')):
        if re.search(pattern, every):
            declared[name] += 1
    depth = max((p.count('/') + 1 for p in nl), default=0)
    justified = {c.get('file') for c in data.get('crisp') or [] if isinstance(c, dict) and c.get('why')}
    return {'nl_functions': len(nl), 'nl_instruction_lines': nl_lines, 'nl_instruction_words': nl_words,
            'ts_files': len(ts), 'ts_logic_lines': sum(crisp.values()), 'ts_lines_by_file': crisp,
            'ts_unjustified': sorted(p for p in ts if p not in justified),
            'types_lines': len([l for l in files.get('types.ts', '').splitlines() if l.strip()]),
            'crisp_line_share': round(sum(crisp.values()) / max(1, sum(crisp.values()) + nl_lines), 3),
            'folder_depth': depth, 'declared': dict(declared),
            'directory_reducers': sorted(p for p, f in fronts.items() if f.get('kind', '').strip() == 'directory-reducer'),
            'ensures_functions': sorted(p for p, f in fronts.items() if 'ensures' in f)}


def kind_of(definition: str) -> str:
    if definition.startswith('nl:'):
        rest = definition[3:]
        return 'nl' if rest.endswith('.nl') or '/' in rest else 'inline'
    for prefix, kind in (('std:', 'std'), ('iterateOn:', 'iterate'), ('pluggable:', 'pluggable'), ('tiers:', 'tiers'),
                         ('refinement:', 'judge')):
        if definition.startswith(prefix):
            return kind
    return 'other'


def leaves(value, out: list[str]) -> list[str]:
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, list):
        for item in value:
            leaves(item, out)
    elif isinstance(value, dict):
        for key, item in value.items():
            if not key.startswith('$'):
                leaves(item, out)
    return out


def squash(text: str) -> str:
    return ' '.join(text.split()).lower()


def output_kind(node: dict) -> str:
    """closed: a decision or a value with no text; open: some text of eight words or more that the call wrote;
    selection: every such text is copied from the call's arguments (a filter or pick that returns the items it was
    given writes nothing new); structured: the rest."""
    if node.get('decision'):
        return 'closed'
    texts = leaves(node.get('value'), [])
    long = [t for t in texts if len(WORD.findall(t)) >= 8]
    if long:
        given = squash('\n'.join(leaves(node.get('args'), [])))
        if given and all(squash(t) in given for t in long):
            return 'selection'
        return 'open'
    if not texts or all(len(WORD.findall(t)) <= 1 for t in texts) and not isinstance(node.get('value'), (dict, list)):
        return 'closed'
    return 'structured'


NL_LITERAL = re.compile(r'(?<![\w.$"\'])nl\s*(?:\.with\s*(?:<[^`]*?>)?\s*\([^`]*?\))?\s*(?:<[^`]*?>)?\s*`((?:[^`\\]|\\.)*)`')


def nl_literals(code: str) -> list[str]:
    """The text of every NL literal (`nl`...``, `nl<T>`...``, `nl.with(...)`...``) in eval code."""
    return [m.group(1) for m in NL_LITERAL.finditer(code)]


NL_KINDS = ('nl', 'inline', 'judge')
INTERPOLATION = re.compile(r'\$\{([^}]*)\}')


def interpolated_names(literal: str) -> set[str]:
    """The leading identifiers the `${...}` interpolations of a literal read (`${rubric.text}` -> rubric)."""
    return {m.group(0) for e in INTERPOLATION.findall(literal) for m in [re.match(r'[A-Za-z_$][\w$]*', e.strip())] if m}


def caller_of(tree: list[dict], by_id: dict) -> None:
    """A call runs in the frame its function was bound to, so a function made in one call and passed to another
    records its calls under its maker (often the root). Place each such call under the call whose eval was running
    when it started: a call holding a function argument, below the recorded parent, with an eval window around the
    start (the innermost when several hold). Rewrites `parent` in place."""
    holders = [n for n in tree if n.get('functional') and n.get('eval_windows')]
    if not holders:
        return

    def ancestors(i):
        seen = []
        while i in by_id and i not in seen:
            seen.append(i)
            i = by_id[i].get('parent')
        return seen

    for n in tree:
        start = n.get('started')
        if start is None or n['id'] == '$root':
            continue
        best = None
        for h in holders:
            if h['id'] == n['id'] or n['parent'] == h['id']:
                continue
            windows = [w for w in h['eval_windows'] if w[0] <= start <= w[1]]
            if not windows or n['parent'] not in ancestors(h['id']) or h['id'] in ancestors(n['id']):
                continue
            opened = max(w[0] for w in windows)
            if best is None or opened > best[0] or (opened == best[0] and len(ancestors(h['id'])) > best[1]):
                best = (opened, len(ancestors(h['id'])), h['id'])
        if best:
            n['parent'] = best[2]
            n['moved_from_maker'] = True


def instance_stats(row: dict, files: dict[str, str], item_kinds: dict[str, str], src: dict,
                   root_actions: list | None = None, root: str = '', root_inputs: list[str] | None = None) -> dict:
    # The root call is not among the recorded invocations (its trace is the run's own): it is added as the tree's root,
    # with the authored root actions as its turns.
    root_node = {'id': '$root', 'parent': None, 'definition': 'nl:' + root, 'name': Path(root).stem, 'outcome': 'done',
                 'decision': None, 'functional': False, 'value': row.get('value'),
                 'actions': [{'tool': a[0], 'code': str((a[1] or {}).get('code', ''))} if a[0] == 'eval' else {'tool': a[0]}
                             for a in root_actions or [] if isinstance(a, list) and a]}
    recorded = row.get('tree') or []
    ids = {n['id'] for n in recorded}
    tree = [root_node] + [{**n, 'parent': n['parent'] if n['parent'] in ids else '$root'} for n in recorded]
    by_id = {n['id']: n for n in tree}
    nl_nodes = [n for n in tree if kind_of(n['definition']) in NL_KINDS]  # the root included
    caller_of(tree, by_id)
    parent_nl = {}
    for n in tree:
        p = n.get('parent')
        while p in by_id and kind_of(by_id[p]['definition']) not in NL_KINDS:
            p = by_id[p].get('parent')
        parent_nl[n['id']] = p if p in by_id else None

    def depth(n):
        d, p = 1, parent_nl.get(n['id'])
        while p:
            d, p = d + 1, parent_nl.get(p)
        return d

    nl_depth = max((depth(n) for n in nl_nodes), default=0)
    with_nl_children = collections.Counter(parent_nl[n['id']] for n in nl_nodes if parent_nl[n['id']])
    interior = {by_id[i]['definition'] for i in with_nl_children if i in by_id}
    constructs = collections.Counter()
    # recursion: an NL call below a call of its own definition
    for n in nl_nodes:
        p = parent_nl.get(n['id'])
        while p:
            if by_id[p]['definition'] == n['definition']:
                constructs['structural-recursion'] += 1
                break
            p = parent_nl.get(p)
    for n in tree:
        kind = kind_of(n['definition'])
        if kind == 'std':
            constructs[STD_KIND.get(n.get('combinator') or n['definition'][4:], n['definition'][4:])] += 1
        elif kind == 'iterate':
            # semanticRepair and fixedPoint run their own iterateOn: that loop is theirs, not an authored iterateOn
            if kind_of(by_id.get(n.get('parent'), {}).get('definition', '')) == 'std':
                continue
            predicate = (n.get('iteration') or {}).get('predicate') or ''
            constructs['iterate-nl-until' if predicate.startswith('nl:') else 'iterate-measure'] += 1
        elif kind == 'pluggable':
            constructs['pluggable'] += 1
        elif kind == 'inline':
            constructs['inline-nl'] += 1
        if n.get('decision'):
            constructs['decision'] += 1
        if n.get('functional'):
            constructs['higher-order'] += 1
        if kind == 'nl':
            path = n['definition'][3:]
            if path in src['directory_reducers']:
                constructs['folder-reducer'] += 1
            if path in src['ensures_functions']:
                constructs['ensures'] += 1
            if '=>' in (front(files.get(path, '')).get('returns', '')):
                constructs['function-result'] += 1
    code = [a.get('code', '') for n in tree for a in n.get('actions') or [] if a.get('tool') == 'eval']
    joined = '\n'.join(code)
    literals = nl_literals(joined)
    # Built from data: an interpolation that reads something the program derived. A literal whose interpolations read
    # only the root's own inputs (`${unit}`) restates the input; it is counted apart.
    root_code = '\n'.join(a.get('code', '') for a in root_node['actions'] if a.get('tool') == 'eval')
    from_inputs = [t for t in nl_literals(root_code) if '${' in t and interpolated_names(t)
                   and interpolated_names(t) <= set(root_inputs or ())]
    dynamic = [t for t in literals if '${' in t]
    for t in from_inputs:
        if t in dynamic:
            dynamic.remove(t)
    inline_nodes = [n for n in nl_nodes if kind_of(n['definition']) == 'inline']
    through = set()
    for n in nl_nodes:
        p = n['id']
        while p:
            if kind_of(by_id[p]['definition']) == 'inline':
                through.add(n['id'])
                break
            p = parent_nl.get(p)
    if dynamic:
        constructs['nl-literal-from-data'] += len(dynamic)
    for name, pattern in CODE_PATTERNS.items():
        hits = len(re.findall(pattern, joined))
        if hits:
            constructs[name] = max(constructs[name], hits) if name == 'with-context' else constructs[name] + hits
    if re.search(r'\bUntrusted<|\buntrusted\s*\(', joined) or 'untrusted' in src['declared']:
        constructs['untrusted'] += 1
    if 'refinement-type' in src['declared']:
        constructs['refinement-type'] += 1
    if re.search(r'\bNeuralese<', joined) or 'neuralese' in src['declared']:
        constructs['neuralese'] += 1
    crisp_names = [n for n, k in item_kinds.items() if k == 'ts']
    crisp_calls = sum(len(re.findall(rf'(?<![\w.]){re.escape(n)}\s*\(', joined)) for n in crisp_names)
    kinds = collections.Counter(output_kind(n) for n in nl_nodes)
    answer_words = [sum(len(WORD.findall(t)) for t in leaves(n.get('value'), [])) for n in nl_nodes]
    eval_lines = sum(len([l for l in c.splitlines() if l.strip()]) for c in code)
    higher = sum(constructs[k] for k in ('higher-order', 'partial', 'context-rebind', 'function-result'))
    higher += sum(constructs[k] for k in STD_KIND.values())
    higher += constructs['iterate-measure'] + constructs['iterate-nl-until'] + constructs['nl-literal-from-data']
    # Judgments the runtime asks (refinements and `ensures:` predicates as `holds`, the shipped std judgments) and what
    # they answered: a refinement that never fails trains only the happy path.
    judgments = collections.defaultdict(collections.Counter)
    for n in nl_nodes:
        shipped = n['definition'].startswith('nl:/@natlang/std/') or kind_of(n['definition']) == 'judge'
        if shipped and isinstance(n.get('value'), bool):
            judgments[n['name']][str(n['value']).lower()] += 1
    refusals = sum(1 for n in tree for a in n.get('actions') or [] if a.get('outcome') in ('rejected', 'refused'))
    return {
        'status': row.get('status'), 'model_calls': row.get('model_calls'), 'nl_calls': len(nl_nodes),
        'judgments': {k: dict(v) for k, v in judgments.items()}, 'refused_actions': refusals,
        'repair_fixes': sum(1 for n in tree if kind_of(n['definition']) == 'std' and n.get('combinator', n['definition'][4:])
                            in ('semanticRepair', 'fixedPoint') for c in tree if c.get('parent') == n['id']
                            and kind_of(c['definition']) in NL_KINDS),
        'nl_calls_by_function': dict(collections.Counter(n['name'] for n in nl_nodes).most_common()),
        'nl_depth': nl_depth, 'interior_functions': len(interior), 'crisp_helper_calls': crisp_calls,
        'nl_call_share': round(len(nl_nodes) / max(1, len(nl_nodes) + crisp_calls), 3),
        'constructs': dict(constructs), 'construct_kinds': sorted(k for k in constructs if constructs[k]),
        'higher_order_uses': higher,
        'combinator_uses': sum(v for k, v in constructs.items() if k in COMBINATOR_KINDS),
        'outputs': dict(kinds), 'open_output_share': round(kinds['open'] / max(1, len(nl_nodes)), 3),
        'answer_words': {'median': statistics.median(answer_words) if answer_words else 0,
                         'p90': sorted(answer_words)[int(0.9 * (len(answer_words) - 1))] if answer_words else 0,
                         'total': sum(answer_words)},
        'nl_literals': {'created': len(literals), 'built_from_data': len(dynamic), 'from_inputs_only': len(from_inputs),
                        'calls': len(inline_nodes),
                        'call_tree_share': round(len(through) / max(1, len(nl_nodes)), 3)},
        'eval_turns': len(code), 'eval_code_lines': eval_lines,
        'eval_lines_per_turn': round(eval_lines / max(1, len(code)), 1),
    }


def program_stats(data: dict, files: dict[str, str], rows: list[dict]) -> dict:
    src = source_stats(data, files)
    item_kinds = {}
    for path in files:
        name = Path(path).name
        if path.endswith('.nl'):
            item_kinds[name[:-3]] = 'nl'
        elif path.endswith('.ts') and path != 'types.ts' and not path.endswith('.d.nl.ts'):
            item_kinds[name[:-3]] = 'ts'
            for m in re.finditer(r'^export\s+(?:async\s+)?function\s+(\w+)', files[path], re.M):
                item_kinds[m.group(1)] = 'ts'
    per = [instance_stats(r, files, item_kinds, src, (inst.get('trajectory') or {}).get('root'), data.get('root', ''),
                          list((inst.get('inputs') or {}).keys()))
           for inst, r in zip(data.get('instances') or [], rows)]
    for inst, r, p in zip(data.get('instances') or [], rows, per):
        trajectory = inst.get('trajectory') or {}
        found = ungrounded(r, trajectory.get('calls') or [], entry_order(trajectory))
        p['ungrounded_open_answers'] = {'count': len(found), 'examples': found[:12]}
    kinds = sorted({k for p in per for k in p['construct_kinds']})
    judgments = collections.defaultdict(collections.Counter)
    for p in per:
        for name, counts in p['judgments'].items():
            judgments[name].update(counts)
    return {'slug': data.get('slug'), 'source': src, 'instances': per, 'construct_kinds': kinds,
            'construct_kind_count': len(kinds),
            'nl_depth': statistics.median([p['nl_depth'] for p in per]) if per else 0,
            'interior_functions': max((p['interior_functions'] for p in per), default=0),
            'open_output_share': round(sum(p['outputs'].get('open', 0) for p in per) / max(1, sum(p['nl_calls'] for p in per)), 3),
            'nl_call_share': round(sum(p['nl_calls'] for p in per) / max(1, sum(p['nl_calls'] + p['crisp_helper_calls'] for p in per)), 3),
            'higher_order_per_instance': round(sum(p['higher_order_uses'] for p in per) / max(1, len(per)), 2),
            'combinators_per_instance': round(sum(p['combinator_uses'] for p in per) / max(1, len(per)), 2),
            'nl_calls_per_instance': round(sum(p['nl_calls'] for p in per) / max(1, len(per)), 1),
            'judgments': {k: dict(v) for k, v in judgments.items()},
            'refused_actions': sum(p['refused_actions'] for p in per),
            'repair_fixes': sum(p['repair_fixes'] for p in per),
            'selection_outputs': sum(p['outputs'].get('selection', 0) for p in per),
            'ungrounded_open_answers': sum(p['ungrounded_open_answers']['count'] for p in per),
            'ungrounded_share': round(sum(p['ungrounded_open_answers']['count'] for p in per)
                                      / max(1, sum(p['outputs'].get('open', 0) for p in per)), 3),
            'nl_literals': {'created': sum(p['nl_literals']['created'] for p in per),
                            'built_from_data': sum(p['nl_literals']['built_from_data'] for p in per),
                            'from_inputs_only': sum(p['nl_literals']['from_inputs_only'] for p in per),
                            'calls': sum(p['nl_literals']['calls'] for p in per),
                            'call_tree_share': round(sum(p['nl_literals']['call_tree_share'] * p['nl_calls'] for p in per)
                                                     / max(1, sum(p['nl_calls'] for p in per)), 3)}}


def shortfalls(stats: dict, targets: dict = TARGETS) -> list[str]:
    s, out = stats['source'], []
    if s['crisp_line_share'] > targets['crisp_line_share_max']:
        out.append(f'crisp share {s["crisp_line_share"]:.2f} > {targets["crisp_line_share_max"]}')
    if stats['nl_depth'] < targets['nl_depth_min']:
        out.append(f'NL depth {stats["nl_depth"]} < {targets["nl_depth_min"]}')
    if stats['interior_functions'] < targets['interior_functions_min']:
        out.append(f'interior NL functions {stats["interior_functions"]} < {targets["interior_functions_min"]}')
    if stats['construct_kind_count'] < targets['construct_kinds_min']:
        out.append(f'construct kinds {stats["construct_kind_count"]} < {targets["construct_kinds_min"]}')
    if stats['higher_order_per_instance'] < targets['higher_order_min']:
        out.append(f'higher-order uses {stats["higher_order_per_instance"]} < {targets["higher_order_min"]}')
    if stats['open_output_share'] < targets['open_output_share_min']:
        out.append(f'open outputs {stats["open_output_share"]:.2f} < {targets["open_output_share_min"]}')
    if stats['nl_call_share'] < targets['nl_call_share_min']:
        out.append(f'NL call share {stats["nl_call_share"]:.2f} < {targets["nl_call_share_min"]}')
    lit, n = stats['nl_literals'], max(1, len(stats['instances']))
    if lit['built_from_data'] / n < targets['literals_from_data_min']:
        out.append(f'NL literals built from data per instance {lit["built_from_data"] / n:.1f} < {targets["literals_from_data_min"]}')
    if lit['call_tree_share'] < targets['literal_tree_share_min']:
        out.append(f'call-tree share through NL literals {lit["call_tree_share"]:.2f} < {targets["literal_tree_share_min"]}')
    return out


def render_tree(row: dict) -> list[str]:
    """The call tree of one replay, indented, for the builder: each NL call with what answered it."""
    tree = row.get('tree') or []
    kids = collections.defaultdict(list)
    ids = {n['id'] for n in tree}
    for n in tree:
        kids[n['parent'] if n['parent'] in ids else None].append(n)
    lines = []

    def walk(n, depth):
        kind = kind_of(n['definition'])
        label = {'std': f'[{n.get("combinator") or n["definition"][4:]}]', 'iterate': '[iterateOn]',
                 'pluggable': '[pluggable]'}.get(kind, n['name'])
        extra = ''
        if kind in ('nl', 'inline'):
            extra = f' -> {json.dumps(n.get("value"), ensure_ascii=False)[:140]}'
            if n.get('decision'):
                extra = ' (decision)' + extra
        elif kind == 'iterate':
            extra = f' {n.get("detail", "")}'
        lines.append(f'{"  " * depth}{label}{"" if n["outcome"] == "done" else " [" + str(n["outcome"]) + "]"}{extra}')
        for child in kids.get(n['id'], []):
            walk(child, depth + 1)

    for n in kids.get(None, []):
        walk(n, 0)
    return lines


# ---- admission and the wave report ---------------------------------------------------------------------------------

def admit(submission: Path, *, root: Path, runtime: Path, idea_path: Path | None = None, natlang: bool = True) -> dict:
    result = check(submission, runtime, natlang=natlang)
    if not result['ok']:
        return {'admitted': False, 'stage': 'check', 'problems': result['problems']}
    data, _ = load(submission)
    sys.path.insert(0, str(REPO / 'scripts' / 'rl_tasks'))
    import program_atlas
    states = program_atlas.claim_states(root)
    idea_path = idea_path or (submission / 'idea.json' if (submission / 'idea.json').exists() else None)
    idea = json.loads(idea_path.read_text()) if idea_path else (states.get(data['slug']) or {}).get('idea')
    if not idea:
        return {'admitted': False, 'stage': 'idea', 'problems': ['no idea card: pass --idea, write idea.json or claim it']}
    idea = idea.get('idea', idea)
    split = invented.split_of(data)
    home = root / 'admitted' / data['slug']
    trajectories = result['cases']
    rl_cases = [c for c in trajectories if c['curriculum']['external']['rl_task']]
    invented.save_rows(home / 'trajectories.jsonl', trajectories)
    invented.save_rows(home / 'cases.jsonl', rl_cases)
    invented.save_rows(home / 'stages.jsonl', [])
    (home / 'stats.json').write_text(json.dumps(result['stats'], ensure_ascii=False, indent=1) + '\n')
    admission = {'slug': data['slug'], 'schema': SCHEMA, 'split': split, 'submission': str(submission),
                 'digest': invented.digest(submission), 'trajectories': len(trajectories), 'cases': len(rl_cases),
                 'stages': 0, 'reference_calls': {c['id']: c['semantics'].get('episode_model_calls', 0) // 2
                                                  for c in trajectories},
                 'reviewed': False, 'at': time.time(), 'published': None}
    (home / 'admission.json').write_text(json.dumps(admission, indent=1) + '\n')
    ledger = root / 'accepted.jsonl'
    listed = {row['idea']['slug'] for row in invented.rows_of(ledger)}
    if data['slug'] not in listed:
        with open(ledger, 'a') as handle:
            handle.write(json.dumps({'idea': idea, 'submission': str(submission), 'split': split, 'at': time.time()},
                                    ensure_ascii=False) + '\n')
    if data['slug'] in states:
        program_atlas.release(root, data['slug'], 'accepted', str(submission))
    return {'admitted': True, **admission, 'warnings': result['warnings']}


def wave_report(wave: Path, runtime: Path, previous: Path | None = None) -> dict:
    programs = []
    for submission in sorted((wave / 'sub').glob('*/submission.json')):
        sub = submission.parent
        try:
            data, files = load(sub)
            if data.get('schema') != SCHEMA:
                continue
            result = check(sub, runtime, natlang=False)
        except (OSError, ValueError, KeyError) as error:
            programs.append({'slug': sub.name, 'error': str(error)[:300]})
            continue
        stats = result.get('stats') or {}
        programs.append({'slug': data.get('slug', sub.name), 'consistent': result['ok'],
                         'problems': result['problems'][:5], 'stats': stats,
                         'shortfalls': shortfalls(stats) if stats else []})
    ok = [p for p in programs if p.get('stats')]
    wave_stats = aggregate([p['stats'] for p in ok])
    before = json.loads(previous.read_text()).get('wave') if previous and previous.is_file() else None
    report = {'wave': wave_stats, 'targets': TARGETS, 'programs': programs, 'previous': before,
              'at': time.strftime('%Y-%m-%d %H:%M')}
    (wave / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=1) + '\n')
    (wave / 'REPORT.md').write_text(report_markdown(wave.name, report))
    (wave / 'EMPHASIS.draft.md').write_text(emphasis(report))
    return report


def aggregate(stats: list[dict]) -> dict:
    if not stats:
        return {'programs': 0}
    mean = lambda xs: round(sum(xs) / len(xs), 3)  # noqa: E731
    kinds = collections.Counter(k for s in stats for k in s['construct_kinds'])
    outputs = collections.Counter()
    for s in stats:
        for inst in s['instances']:
            outputs.update(inst['outputs'])
    total = sum(outputs.values())
    return {'programs': len(stats),
            'crisp_line_share': mean([s['source']['crisp_line_share'] for s in stats]),
            'ts_logic_lines': mean([s['source']['ts_logic_lines'] for s in stats]),
            'nl_functions': mean([s['source']['nl_functions'] for s in stats]),
            'nl_depth': mean([s['nl_depth'] for s in stats]),
            'interior_functions': mean([s['interior_functions'] for s in stats]),
            'construct_kinds': mean([s['construct_kind_count'] for s in stats]),
            'higher_order_per_instance': mean([s['higher_order_per_instance'] for s in stats]),
            'combinators_per_instance': mean([s['combinators_per_instance'] for s in stats]),
            'nl_calls_per_instance': mean([s['nl_calls_per_instance'] for s in stats]),
            'nl_call_share': mean([s['nl_call_share'] for s in stats]),
            'literals_per_instance': mean([s['nl_literals']['created'] / max(1, len(s['instances'])) for s in stats]),
            'literals_from_data_per_instance': mean([s['nl_literals']['built_from_data'] / max(1, len(s['instances']))
                                                     for s in stats]),
            'literal_calls_per_instance': mean([s['nl_literals']['calls'] / max(1, len(s['instances'])) for s in stats]),
            'literal_tree_share': mean([s['nl_literals']['call_tree_share'] for s in stats]),
            'open_output_share': round(outputs['open'] / max(1, total), 3),
            'output_mix': {k: round(v / max(1, total), 3) for k, v in outputs.items()},
            'programs_using': {k: kinds[k] for k in CONSTRUCTS if kinds[k]},
            'programs_never_using': [k for k in CONSTRUCTS if not kinds[k]]}


def report_markdown(name: str, report: dict) -> str:
    w, t = report['wave'], report['targets']
    lines = [f'# Wave report: {name} ({report["at"]})', '',
             'Statistics of the authored trajectories (scripts/rl_tasks/sources/authored_programs.py wave-report).',
             'Targets are where the campaign steers the top of the funnel; they are not admission checks.', '']
    if w.get('programs'):
        rows = [('crisp line share (mean)', w['crisp_line_share'], f'<= {t["crisp_line_share_max"]}'),
                ('TypeScript logic lines (mean)', w['ts_logic_lines'], ''),
                ('.nl functions (mean)', w['nl_functions'], ''),
                ('NL composition depth (mean of medians)', w['nl_depth'], f'>= {t["nl_depth_min"]}'),
                ('interior NL functions (mean)', w['interior_functions'], f'>= {t["interior_functions_min"]}'),
                ('semantic-FP construct kinds (mean)', w['construct_kinds'], f'>= {t["construct_kinds_min"]}'),
                ('higher-order uses per instance', w['higher_order_per_instance'], f'>= {t["higher_order_min"]}'),
                ('combinator uses per instance', w['combinators_per_instance'], ''),
                ('NL calls per instance', w['nl_calls_per_instance'], ''),
                ('NL call share', w['nl_call_share'], f'>= {t["nl_call_share_min"]}'),
                ('open output share', w['open_output_share'], f'>= {t["open_output_share_min"]}'),
                ('NL literals created per instance', w['literals_per_instance'], ''),
                ('NL literals built from data per instance', w['literals_from_data_per_instance'],
                 f'>= {t["literals_from_data_min"]}'),
                ('calls through NL literals per instance', w['literal_calls_per_instance'], ''),
                ('call-tree share through NL literals', w['literal_tree_share'], f'>= {t["literal_tree_share_min"]}')]
        prev = report.get('previous') or {}
        lines += ['| statistic | wave | previous | target |', '|---|---|---|---|']
        keys = ['crisp_line_share', 'ts_logic_lines', 'nl_functions', 'nl_depth', 'interior_functions', 'construct_kinds',
                'higher_order_per_instance', 'combinators_per_instance', 'nl_calls_per_instance', 'nl_call_share',
                'open_output_share', 'literals_per_instance', 'literals_from_data_per_instance',
                'literal_calls_per_instance', 'literal_tree_share']
        for (label, value, target), key in zip(rows, keys):
            lines.append(f'| {label} | {value} | {prev.get(key, "")} | {target} |')
        lines += ['', f'Output mix of NL calls: {json.dumps(w["output_mix"])}', '',
                  'Programs using each construct kind: ' + ', '.join(f'{k} {v}' for k, v in w['programs_using'].items()),
                  '', 'Never used this wave: ' + (', '.join(w['programs_never_using']) or 'none'), '']
    lines += ['## Programs', '', '| program | consistent | crisp share | depth | interior | kinds | higher-order/inst | '
              'open share | NL calls/inst | literals (from data) | literal tree share | short of target |',
              '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for p in report['programs']:
        s = p.get('stats')
        if not s:
            lines.append(f'| {p["slug"]} | error: {p.get("error", "")} | | | | | | | | | | |')
            continue
        lines.append(f'| {p["slug"]} | {"yes" if p["consistent"] else "no"} | {s["source"]["crisp_line_share"]} | '
                     f'{s["nl_depth"]} | {s["interior_functions"]} | {s["construct_kind_count"]} | '
                     f'{s["higher_order_per_instance"]} | {s["open_output_share"]} | {s["nl_calls_per_instance"]} | '
                     f'{s["nl_literals"]["created"]} ({s["nl_literals"]["built_from_data"]}) | '
                     f'{s["nl_literals"]["call_tree_share"]} | {"; ".join(p["shortfalls"]) or "-"} |')
    lines += ['', 'The next wave\'s emphasis is drafted in EMPHASIS.draft.md: the coordinator edits it and installs it as '
              'the next wave\'s EMPHASIS.md, which BUILD_BRIEF.md has every builder read first.', '']
    return '\n'.join(lines)


GUIDANCE = {
    'crisp': 'Keep TypeScript to what exactness or the outside world requires. Composition, ordering of stages and '
             'every judgment are `.nl` functions calling each other; the glue between them is the few lines of eval '
             'code the calling function writes in its trajectory.',
    'depth': 'Grow the tree downwards: a function that reads a whole item delegates the parts it reads to functions '
             'in its own callable folder, which delegate again where a part has parts.',
    'interior': 'Give more of the middle of the tree its own callable folder: an interior function receives a whole '
                'item, decides which of its helpers the item needs, and composes their results.',
    'kinds': 'Reach for the constructs this wave did not use (listed below): each has a short positive example in the '
             'build brief.',
    'higher': 'Pass judgments as values: `semanticFilter(items, meets.partial(rule))`, a function that returns the '
              'judgment written for its case, `fn.in(NatlangContext.empty().with({ policy }))`.',
    'open': 'More of the answers are open text the function writes: summaries, replies, rewritten passages, plans, '
            'explanations. Closed decisions stay where a choice among a few values is the honest answer.',
    'literals': 'Synthesize more functions at run time: build an NL literal whose instructions come from the data '
                '(a criterion per category the program found, a rubric an earlier call wrote, the house style read '
                'from examples), then map, pass and compose it like any other function.',
    'share': 'Replace helper calls into TypeScript by calls of `.nl` functions; arithmetic and lookups the eval does '
             'in a line need no helper at all.',
}


def emphasis(report: dict) -> str:
    w, t = report['wave'], report['targets']
    if not w.get('programs'):
        return '# Emphasis for the next wave\n\nNo program of this wave had statistics.\n'
    keyed = [('crisp', w['crisp_line_share'] > t['crisp_line_share_max']), ('depth', w['nl_depth'] < t['nl_depth_min']),
             ('interior', w['interior_functions'] < t['interior_functions_min']),
             ('kinds', w['construct_kinds'] < t['construct_kinds_min']),
             ('higher', w['higher_order_per_instance'] < t['higher_order_min']),
             ('open', w['open_output_share'] < t['open_output_share_min']),
             ('share', w['nl_call_share'] < t['nl_call_share_min']),
             ('literals', w['literals_from_data_per_instance'] < t['literals_from_data_min']
              or w['literal_tree_share'] < t['literal_tree_share_min'])]
    best = sorted((p for p in report['programs'] if p.get('stats')),
                  key=lambda p: (len(p['shortfalls']), -p['stats']['construct_kind_count']))[:2]
    lines = ['# Emphasis for the next wave', '',
             f'Drafted from the previous wave\'s statistics ({w["programs"]} programs). The coordinator edits this and '
             'installs it as the wave\'s EMPHASIS.md.', '']
    for key, short in keyed:
        if short:
            lines.append(f'- {GUIDANCE[key]}')
    if w['programs_never_using']:
        lines.append('- Constructs no program of the last wave used: ' + ', '.join(w['programs_never_using']) + '.')
    if best:
        lines += ['', 'Programs of the last wave closest to the targets, worth reading as examples: ' +
                  ', '.join(p['slug'] for p in best) + '.']
    return '\n'.join(lines) + '\n'


# ---- CLI -----------------------------------------------------------------------------------------------------------

def suggest_fragment(answer: dict, answers: list[dict]) -> str | None:
    """The start of the first argument line this call shows and no other call of its function shows: a fragment that
    names this call's item (BUILD_BRIEF "The trajectory")."""
    others = [a.get('shown', '') for a in answers if a is not answer and a.get('name') == answer.get('name')]
    for line in str(answer.get('shown', '')).splitlines():
        head = line[:80].rstrip()
        if len(head) >= 6 and not head.startswith('Declared ') and not any(head in o for o in others):
            return head
    return None


def no_entry_blocks(data: dict, rows: list[dict]) -> list[str]:
    """Every call no entry answered, with the whole opening (instructions and arguments) and a suggested fragment."""
    lines = []
    for inst, row in zip(data.get('instances') or [], rows):
        answers = row.get('answers') or []
        for a in answers:
            if a.get('entry') != -1:
                continue
            fragment = suggest_fragment(a, answers)
            lines += [f'NO ENTRY in instance {inst.get("name")} for a call of {a["name"]}'
                      + (' (a placeholder answered it so the replay could go on)' if a.get('placeholder') else ''),
                      '  suggested entry: ' + json.dumps({'fn': a['name'], 'match': [fragment] if fragment else []},
                                                         ensure_ascii=False),
                      '  opening:']
            lines += ['    ' + line for line in str(a.get('opening') or a.get('shown') or '').splitlines()]
    return lines


def dump(submission: Path, runtime: Path, out: Path, natlang: bool = False) -> dict:
    """The whole replay of every instance, as JSON: each call's opening and the entry that answered it, the call tree
    (calls through function arguments placed under the call that made them, as the statistics do) with arguments,
    eval code and values in full, failed actions in full, and the statistics."""
    result = check(submission, runtime, natlang=natlang)
    data, files = load(submission)
    instances = []
    for inst, row in zip(data.get('instances') or [], result['rows']):
        trajectory = inst.get('trajectory') or {}
        order = entry_order(trajectory)
        tree = [dict(n) for n in row.get('tree') or []]
        by_id = {n['id']: n for n in tree}
        caller_of(tree, by_id)
        instances.append({'name': inst.get('name'), 'status': row.get('status'), 'detail': row.get('detail'),
                          'value': row.get('value'), 'model_calls': row.get('model_calls'),
                          'failed_actions': row.get('failed_actions'), 'error': row.get('error'),
                          'answers': [{**a, 'entry': order[a['entry']] + 1 if 0 <= a.get('entry', -1) < len(order) else None}
                                      for a in row.get('answers') or []],
                          'tree': tree})
    payload = {'submission': str(submission), 'ok': result['ok'], 'problems': result['problems'],
               'warnings': result['warnings'], 'instances': instances, 'stats': result.get('stats')}
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + '\n')
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    c = sub.add_parser('check')
    c.add_argument('submission', type=Path)
    c.add_argument('--runtime', type=Path, default=REPO / 'ts-host')
    c.add_argument('--no-natlang-check', action='store_true')
    s = sub.add_parser('stats')
    s.add_argument('submission', type=Path)
    s.add_argument('--runtime', type=Path, default=REPO / 'ts-host')
    d = sub.add_parser('dump', help='the whole replay as JSON (default: SUBMISSION.dump.json beside SUBMISSION)')
    d.add_argument('submission', type=Path)
    d.add_argument('--out', type=Path)
    d.add_argument('--runtime', type=Path, default=REPO / 'ts-host')
    a = sub.add_parser('admit')
    a.add_argument('submission', type=Path)
    a.add_argument('--root', type=Path, default=invented.INVENTOR_ROOT)
    a.add_argument('--runtime', type=Path, default=REPO / 'ts-host')
    a.add_argument('--idea', type=Path)
    w = sub.add_parser('wave-report')
    w.add_argument('wave', type=Path)
    w.add_argument('--runtime', type=Path, default=REPO / 'ts-host')
    w.add_argument('--previous', type=Path)
    args = parser.parse_args(argv)
    runtime = args.runtime.resolve()
    if args.command == 'wave-report':
        report = wave_report(args.wave.resolve(), runtime, args.previous)
        print(json.dumps(report['wave'], indent=1))
        return 0
    submission = args.submission.resolve()
    if not (submission / 'submission.json').is_file():
        print(json.dumps({'ok': False, 'problems': [
            f'{submission}/submission.json is missing: SUBMISSION is the directory that holds submission.json and '
            'program/ (a build script run from elsewhere may have written submission.json one directory up)']}, indent=1))
        return 1
    if args.command == 'admit':
        result = admit(submission, root=args.root, runtime=runtime, idea_path=args.idea)
        print(json.dumps(result, ensure_ascii=False, indent=1))
        return 0 if result['admitted'] else 1
    if args.command == 'dump':
        out = args.out or submission.with_name(submission.name + '.dump.json')
        payload = dump(submission, runtime, out)
        print(json.dumps({'ok': payload['ok'], 'out': str(out), 'problems': len(payload['problems']),
                          'warnings': len(payload['warnings'])}))
        return 0
    result = check(submission, runtime, natlang=args.command == 'check' and not args.no_natlang_check)
    if args.command == 'stats':
        print(json.dumps(result.get('stats'), ensure_ascii=False, indent=1))
        return 0
    data, _ = load(submission)
    blocks = no_entry_blocks(data, result['rows'])
    if blocks:
        print(f'{sum(1 for b in blocks if b.startswith("NO ENTRY"))} calls have no entry (all of them, in every stage: '
              'missing ones were answered by placeholders so the replay could reach later stages)')
        print('\n'.join(blocks))
    for index, (inst, row) in enumerate(zip(data.get('instances') or [], result['rows'])):
        print(f'== instance {inst.get("name")}: {row.get("status")}, {row.get("model_calls")} model calls')
        print('returned: ' + json.dumps(row.get('value'), ensure_ascii=False))
        for line in render_tree(row):
            print(line)
        for f in row.get('failed_actions') or []:
            print(f'FAILED {f["tool"]}: {f["result"]}')
        per = (result.get('stats') or {}).get('instances') or []
        if index < len(per):
            p = per[index]
            print('instance statistics: ' + json.dumps({
                'nl_calls': p['nl_calls'], 'nl_depth': p['nl_depth'], 'interior_functions': p['interior_functions'],
                'outputs': p['outputs'], 'open_output_share': p['open_output_share'], 'judgments': p['judgments'],
                'nl_literals': p['nl_literals'], 'construct_kinds': p['construct_kinds']}, ensure_ascii=False))
            for u in p['ungrounded_open_answers']['examples']:
                print(f'  open answer states what its call does not show: entry {u["entry"]} {u["call"]}: {", ".join(u["absent"])}')
    stats = result.get('stats')
    if stats:
        print('statistics: ' + json.dumps({k: stats[k] for k in ('nl_depth', 'interior_functions', 'construct_kinds',
                                                                  'open_output_share', 'nl_call_share',
                                                                  'higher_order_per_instance', 'nl_calls_per_instance',
                                                                  'judgments', 'ungrounded_open_answers')},
                                          ensure_ascii=False))
        print('NL literals: ' + json.dumps(stats['nl_literals']))
        print('source: ' + json.dumps({k: stats['source'][k] for k in ('nl_functions', 'ts_logic_lines',
                                                                       'crisp_line_share', 'ts_unjustified')}))
        print('short of the campaign targets (statistics, not checks): ' + ('; '.join(shortfalls(stats)) or 'none'))
    print(json.dumps({'ok': result['ok'], 'problems': result['problems'][:60], 'warnings': result['warnings'][:60]},
                     ensure_ascii=False, indent=1))
    return 0 if result['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
