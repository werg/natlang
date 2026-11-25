#!/usr/bin/env python3
"""Invented programs (`natlang.invented-program/1`) -> crisp requirement checks, `natlang.program/2` cases, qualification.

The program inventor (`applications/program-inventor/`, plans/NOVEL_PROGRAMS_20261124.md) builds one complete natlang
program per new idea, with two or three literal input instances, exact checkers and controls
(`applications/program-inventor/BUILD_BRIEF.md`). This module is the crisp mechanism around it:

  invented.py check SUBMISSION            static crisp requirements (requirements.json, check "crisp"), metrics
  invented.py package SUBMISSION OUT      one `natlang.program/2` case per instance (the collector's case format)
  invented.py qualify SUBMISSION [--runtime TS_HOST]
      check + `natlang check --json` (F1) + package + scripts/rl_tasks/verify.mjs (reference replay F3, wrong answers
      F4, keyword baselines S3) + scripts/rl_tasks/ablate.mjs (judgment ablations S2) + scripts/rl_tasks/
      reference_trace.mjs (the reference run call by call: each case declares `semantics.episode_model_calls`, twice
      its reference run's model calls, and its stage tasks are cut); writes SUBMISSION/qualification/{cases.jsonl,
      stages.jsonl,trace.json,report.json}. Run through the memory ledger on the DGX.
  invented.py dry-run SUBMISSION [--controls]
      the program's TypeScript run with the scripted judgment answers (no model, seconds): the reference against
      `expected`, the entry that answered each call, the static checks; --controls replays every wrong answer,
      ablation and keyword baseline too. Writes nothing into the submission.
  invented.py stages SUBMISSION [--runtime TS_HOST]
      only the stage tasks of an already qualified submission (qualification/stages.jsonl), verified by replay.

The per-submission pipeline of a generation campaign (ROOT: data_nvme:natlang-novel-programs, or
NATLANG_INVENTOR_ROOT):

  invented.py digest SUBMISSION        the digest of program/ + submission.json a review names (REVIEW_BRIEF.md)
  invented.py admit SUBMISSION [--review FILE] [--idea FILE] [--root ROOT] [--publish] [--skip-qualify]
      qualify (above: crisp requirements, natlang check, replay, ablations, keyword baselines, reference trace,
      episode budgets, stage cut) admits; a review is optional (default SUBMISSION/review.json,
      `natlang.invented-review/1`), and one naming this digest blocks only by its `wrong_expected` items
      (NATLANG_INVENTED_STRICT_REVIEW=1: every judged requirement must be met); then the idea's split (70/15/15 by world):
      train ideas stage their cases and stage tasks for the pool, dev and confirmation ideas their cases for the held
      slice (ROOT/admitted/<slug>/). The idea enters ROOT/accepted.jsonl (the atlas every later builder reads) and its
      claim is closed as accepted. --publish publishes at once.
  invented.py sample-reviews N [--seed S]   N admitted submissions drawn at random for a review (REVIEW_BRIEF.md)
  invented.py publish [--root ROOT] [--dry-run]
      every admitted, unpublished idea, as one batch (a program whose sampled review of the admitted version lists
      `wrong_expected` is withheld from this and every later batch): train cases + stage tasks -> ROOT/pool/<stamp>/cases.jsonl,
      repetition measured against the pool (measure_repetition.py), registry dataset natlang-rl-invented-<stamp>
      promoted to grpo-pool; dev + confirmation programs of every admitted idea so far -> the held slice
      ops/held-inv/<stamp>/ (held_eval pin --all-rows, also excluding held-v3; held_eval verify against every pool row
      training selects from), registry dataset held-inv-<stamp> with role held-eval-invented, so
      held_eval.training_exclusion_ids keeps it out of training. held-v3 stays untouched.

Stage tasks (owner 2026-11-25: "also cut programs into per-stage tasks with checked intermediate outputs, so a small
model gets dense signal"): every judgment call of an instance's reference run is an intermediate output the reference
checks, so each distinct leaf `.nl` call (one without a callable folder of its own and with no function among its
arguments) becomes a task of its own: the same program files with that judgment as the root, the call's own arguments
as inputs and the reference's answer as the exact expected value. They share the program's split and world.

It never invents gold, wrong answers or references: everything it judges with comes from the submission. Each idea is
its own family (`inv_<slug>`) and world (`invented:<slug>`): all instances of a program share one split.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import shutil
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(os.environ.get('NATLANG_REPO_ROOT') or Path(__file__).resolve().parents[3])
sys.path.insert(0, str(REPO))
from scripts.rl_tasks.common import CURRICULUM_VERSION, slug, text_digest  # noqa: E402

SCHEMA = 'natlang.invented-program/1'
GENERATOR = 'natlang.invented-programs/1'
ORACLES = frozenset({'exact', 'normalized', 'verifier'})
VERIFIER_ROOT = 'training/semantic_fp_tasks/verifiers/'
# The bar a wave of builders qualifies against: BAR_VERSION and the requirements' digest go into every report.json, and
# they change only between waves (coordinator 2026-11-25), never while builders iterate.
BAR_VERSION = '2026-11-25.4'
# The bar (requirements.json; coordinator 2026-11-25 after the calibration review): size floors, code depth, notes.
MIN_WORDS, MIN_TEXT_ITEMS, MIN_TS_LINES, MIN_HARDEST = 800, 15, 100, 3
SUBMISSION_ENTRIES = frozenset({'program', 'submission.json', 'NOTES.md', 'qualification', 'idea.json', 'review.json'})
PROGRAM_EXTRA = frozenset({'natlang.json', 'package.json', 'tsconfig.json', 'examples.json', '.natlang', 'node_modules'})
STD = ('semanticMap', 'semanticFilter', 'semanticPartition', 'semanticReduce', 'semanticMatch', 'semanticRank',
       'semanticMerge', 'semanticRepair', 'fixedPoint', 'proposeAndVerify', 'branchAndSelect', 'escalate',
       'withContext', 'compileRule')
# Semantic-FP constructs found statically (D2). Each pattern must appear in program source, not in comments only.
CONSTRUCT_PATTERNS = {
    'decide': r'\bdecide\s*\(|^readout:\s*(decision|instant)\s*$',
    'std-combinator': r'\b(' + '|'.join(STD) + r')\s*\(',
    'higher-order': r'\.partial\s*\(|^\s+\w+\??:\s*["\']?\(.*\)\s*=>|=>\s*Promise<|returns:\s*["\']?\(',
    'iterate-until': r'\biterateOn\s*\([\s\S]*?\.until\s*\(',
    'structural-recursion': r'^(recursive|measure):',
    'ensures': r'^ensures:',
    'refinement': r'\bIs<|\bUntrusted<|\buntrusted\s*\(',
    'neuralese': r'\bNeuralese<',
    'folder-transaction': r'\.(propose|accept|apply)\s*\(',
    'pluggable': r'\bpluggable\s*\(',
    'inline-nl': r'\bnl\s*(<[^`]*>)?\s*`',
}
INLINE_NL = re.compile(r'\bnl\s*(?:<[^`]*>)?\s*`([^`]*)`')
TS_EXPORT = re.compile(r'^export\s+(?:default\s+)?(?:async\s+)?function\b', re.M)


def load(submission: Path) -> tuple[dict, dict[str, str]]:
    data = json.loads((submission / 'submission.json').read_text())
    program = submission / 'program'
    files = {}
    for path in sorted(program.rglob('*')):
        rel = path.relative_to(program).as_posix()
        # `natlang check` writes generated declarations (`*.d.nl.ts`) beside the sources; they are not program files
        if (path.is_file() and rel.split('/')[0] not in PROGRAM_EXTRA and not rel.startswith('.')
                and not rel.endswith('.d.nl.ts')):
            files[rel] = path.read_text()
    return data, files


def frontmatter(text: str) -> tuple[str, str]:
    if not text.startswith('---'):
        return '', text
    _, head, body = text.split('---', 2)
    return head, body


def strip_comments(text: str) -> str:
    return re.sub(r'//[^\n]*|/\*[\s\S]*?\*/', '', text)


def words(value) -> int:
    if isinstance(value, str):
        return len(value.split())
    if isinstance(value, dict):
        return sum(words(v) for v in value.values())
    if isinstance(value, list):
        return sum(words(v) for v in value)
    return 0


def longest_list(value) -> int:
    if isinstance(value, list):
        return max([len(value), *(longest_list(v) for v in value)])
    if isinstance(value, dict):
        return max([0, *(longest_list(v) for v in value.values())])
    return 0


PROSE_WORD = re.compile(r"[^\W\d_]{2,}(?:['\u2019][^\W\d_]+)?")


def prose_words(value, seen: set | None = None) -> int:
    """Words of prose in value: alphabetic words of the string leaves holding at least three of them, each distinct
    string counted once (ids, codes, timecodes and numbers are not prose)."""
    seen = set() if seen is None else seen
    if isinstance(value, str):
        found = PROSE_WORD.findall(value)
        if len(found) < 3 or value in seen:
            return 0
        seen.add(value)
        return len(found)
    if isinstance(value, dict):
        return sum(prose_words(v, seen) for v in value.values())
    if isinstance(value, list):
        return sum(prose_words(v, seen) for v in value)
    return 0


def string_leaves(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in string_leaves(v)]
    if isinstance(value, list):
        return [s for v in value for s in string_leaves(v)]
    return []


def text_items(value) -> int:
    """The most elements of one list in value that each hold text (at least three words): items a judgment can read."""
    if isinstance(value, list):
        return max([sum(1 for v in value if words(v) >= 3), *(text_items(v) for v in value)])
    if isinstance(value, dict):
        return max([0, *(text_items(v) for v in value.values())])
    return 0


def declared_words(card: dict) -> int | None:
    """The instance size in words the idea card declares (`inputs`: "... about 1,300 words"), its lower bound."""
    match = re.search(r'(\d[\d,]*)(?:\s*(?:-|\u2013|to)\s*\d[\d,]*)?\s+words', str(card.get('inputs') or ''))
    return int(match.group(1).replace(',', '')) if match else None


# S6: a plain `string` a judgment reads is text people wrote unless it is an identifier (its name or doc says so).
ID_NAME = re.compile(r'^(id|ids|key|keys|slug|code|codes|ref|refs|uuid|url|urls|path|paths|email|date|time|timestamp|'
                     r'language|lang|locale|currency|unit|units|iso)$|(Id|Ids|Key|Keys|Code|Codes|Ref|Refs|Url|Path|Date|'
                     r'Time|At|Iso)$')
ID_DOC = re.compile(r'\b(identifier|ids?|ISO|code|date|timestamp|timecode|URL|path|digits|computed)\b', re.I)
TYPE_ALIAS = re.compile(r'(?:/\*\*((?:(?!\*/)[\s\S])*)\*/\s*)?export\s+type\s+(\w+)\s*=\s*')


def top_level(text: str, end: str) -> int:
    """Index of the first `end` character outside brackets and strings in text (len(text) when there is none)."""
    depth, quote, skip = 0, '', -1
    for i, ch in enumerate(text):
        if i < skip:
            continue
        if not quote and text.startswith('/*', i):  # a doc comment's apostrophes are not quotes
            skip = text.find('*/', i) + 2 or len(text)
            continue
        if quote:
            quote = '' if ch == quote and text[i - 1] != '\\' else quote
        elif ch in '"\'`':
            quote = ch
        elif ch in '{([<':
            depth += 1
        elif ch in '})]>' and not (ch == '>' and text[i - 1] == '='):
            depth -= 1
        elif depth == 0 and ch in end:
            return i
    return len(text)


def type_aliases(files: dict[str, str]) -> dict[str, str]:
    aliases = {}
    for path, text in files.items():
        if path.split('/')[-1] != 'types.ts':
            continue
        text = re.sub(r'//[^\n]*', '', text)
        for match in TYPE_ALIAS.finditer(text):
            rest = text[match.end():]
            aliases[match.group(2)] = rest[:top_level(rest, ';')].strip()
    return aliases


def record_fields(text: str) -> list[tuple[str, str, str]]:
    """(name, type, doc) of each member of an object type literal `{ ... }`."""
    body = text.strip()[1:-1]
    fields, doc = [], ''
    while body.strip():
        body = body.lstrip()
        if body.startswith('/**'):
            close = body.find('*/')
            doc, body = body[3:close], body[close + 2:]
            continue
        cut = top_level(body, ';,\n')
        member, body = body[:cut], body[cut + 1:]
        found = re.match(r'\s*(?:readonly\s+)?["\']?(\w+)["\']?\??\s*:\s*([\s\S]+)$', member)
        if found:
            fields.append((found.group(1), found.group(2).strip(), doc))
            doc = ''
    return fields


def plain_strings(type_text: str, aliases: dict, at: str, seen: set) -> list[str]:
    """Paths under type_text that are a bare `string` (alone, in an array, or with null), following type aliases."""
    text = type_text.strip()
    if re.fullmatch(r'\(?\s*string\s*\)?(\[\])*(\s*\|\s*(null|undefined))*', text):
        return [at]
    if re.match(r'^(Untrusted|Is|Neuralese)\s*<', text):
        return []
    out = []
    if text.startswith('{') and text.endswith('}'):
        for name, field_type, doc in record_fields(text):
            if not (ID_NAME.search(name) or ID_DOC.search(doc)):
                out += plain_strings(field_type, aliases, f'{at}.{name}', seen)
        return out
    for name in re.findall(r'\b[A-Z]\w*\b', text):
        if name in aliases and name not in seen:
            out += plain_strings(aliases[name], aliases, f'{at} ({name})', seen | {name})
    return out


def nl_args(head: str) -> dict[str, str]:
    """The `args:` of a .nl frontmatter, name to type text, in inline or block form."""
    lines = head.splitlines()
    for index, line in enumerate(lines):
        found = re.match(r'^args:\s*(.*)$', line)
        if not found:
            continue
        if found.group(1).strip().startswith('{'):
            return {n: t for n, t, _ in record_fields(found.group(1).strip())}
        args = {}
        for item in lines[index + 1:]:
            entry = re.match(r'^\s+(\w+)\??\s*:\s*(.+)$', item)
            if not entry:
                break
            args[entry.group(1)] = entry.group(2).strip().strip('"\'')
        return args
    return {}


def untrusted_problems(files: dict[str, str]) -> list[str]:
    aliases, problems = type_aliases(files), []
    for path, text in sorted(files.items()):
        if not path.endswith('.nl'):
            continue
        for name, type_text in nl_args(frontmatter(text)[0]).items():
            if ID_NAME.search(name):
                continue
            for where in dict.fromkeys(plain_strings(type_text, aliases, name, set())):
                problems.append(f'S6: program/{path}: {where} is a plain string this judgment reads; type it Untrusted<string> '
                                'when people wrote it (names, titles, nicknames, labels, notes), a literal union when its values '
                                'are a closed set, or, for an identifier, name it so (id, ...Id, code, date) or say so in its doc '
                                'comment')
    return problems


# Every quoted span in turn (a short one too, so the next quote pairs its own marks); quote_problems keeps those of three
# words or more.
QUOTED = re.compile(r'["\u201c]([^"\u201c\u201d]{1,400}?)["\u201d]')   # a quote may wrap onto the next line


def normalized(text: str) -> str:
    return re.sub(r'\s+', ' ', text.replace('\u2019', "'")).strip().lower()


def quote_problems(submission: Path, data: dict, files: dict[str, str]) -> list[str]:
    """Fragments NOTES.md and wrong-answer `mistake` texts quote (three words or more) still occur in what they describe:
    the instances or the program. A quote of removed text is a stale note."""
    instances = data.get('instances') or []
    program = normalized(' '.join(files.values()))
    every = normalized(json.dumps([i.get('inputs') for i in instances], ensure_ascii=False)
                       .replace('\\n', ' ').replace('\\"', '"')) + ' ' + normalized(' '.join(
                           s for i in instances for s in string_leaves(i.get('inputs'))))
    def missing(text: str, where: str) -> list[str]:
        # A quote with an ellipsis ("DMSO ... makes 3%") quotes each of its parts.
        def absent(part: str) -> bool:
            part = normalized(part).strip(' .,;:')
            return len(part.split()) >= 3 and part not in where and part not in program
        return [q for q in QUOTED.findall(text) if any(absent(part) for part in re.split(r'\.\.\.|\u2026', q))]
    problems = []
    notes = submission / 'NOTES.md'
    for quote in missing(notes.read_text() if notes.is_file() else '', every):
        problems.append(f'C4: NOTES.md quotes "{quote[:80]}", which no instance or program file holds now; quote the text '
                        'as it stands or drop the quote')
    for inst in instances:
        own = normalized(' '.join(string_leaves(inst.get('inputs'))))
        for w in inst.get('wrong_answers') or []:
            if isinstance(w, dict):
                for quote in missing(str(w.get('mistake', '')), own):
                    problems.append(f'F4: instance {inst.get("name")}: a wrong answer\'s mistake quotes "{quote[:80]}", which '
                                    'the instance does not hold now; describe the mistake on the text as it stands')
    return problems


def hardest_items(notes: str) -> int:
    """List entries under the NOTES.md heading `Hardest items`."""
    section = re.search(r'^#+\s*Hardest items[^\n]*\n([\s\S]*?)(?=^#|\Z)', notes, re.M | re.I)
    return len(re.findall(r'^\s*(?:[-*]|\d+\.)\s+\S', section.group(1), re.M)) if section else 0


# ---- what the reference run shows: sizes, copies of answers, identity, the children each call took ------------------
STOP_VALUES = frozenset({'none', 'other', 'same', 'yes', 'no', 'true', 'false', 'first', 'last', 'only', 'neither',
                         'both', 'all', 'unknown', 'unclear', 'null', 'before', 'after', 'unrelated', 'nothing'})
ITEM_TOKEN = re.compile(r'[\w.:-]*\d[\w.:-]*')


def stem(word: str) -> str:
    word = word.lower()
    for suffix in ('ing', 'ed', 'es', 's'):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[:-len(suffix)]
    return word


def answer_leaves(value) -> list[str]:
    """The string leaves of an answer that may name a relation: not those under an identifier or unit field
    (`unit: 'knots'` names a unit the text may well use, not what the item does)."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [leaf for key, item in value.items() if not ID_NAME.search(str(key)) for leaf in answer_leaves(item)]
    if isinstance(value, list):
        return [leaf for item in value for leaf in answer_leaves(item)]
    return []


def value_words(value, args) -> list[str]:
    """The words of a finite answer (a literal of three words at most that is not copied from the arguments)."""
    leaves = set(string_leaves(args))
    return [word for leaf in answer_leaves(value) if leaf not in leaves and len(leaf.split()) <= 3
            for word in re.split(r'[-_\s]+', leaf) if len(word) >= 4 and word.lower() not in STOP_VALUES]


def deciding_words(calls: list[dict]) -> list[str]:
    """S7: a relation word of an answer that settles a judgment by itself. For each such word, the calls whose own
    argument text holds it: when every one of them has that answer, a reader answers from the word alone (the item states
    its answer). A word that also appears in items with other answers settles nothing."""
    found = []
    # Only relation words (a verb in the third person: replaces, cancels, knows): a noun value (a solvent, a household)
    # names a domain term the text may well use; a text saying "this replaces" states the relation the judgment reads.
    words = {w.lower() for call in calls for w in value_words(call.get('value'), call.get('args'))
             if re.fullmatch(r'[a-z]{3,}[^su\W]s', w.lower()) and not w.lower().endswith(('ous', 'ss', 'is'))}
    # A text several calls share (a datasheet, a style guide) names its terms for all of them: a word it holds settles nothing.
    counts = collections.Counter(t for call in calls for t in set(string_leaves(call.get('args'))))
    shared = [t for t, n in counts.items() if n > 1]
    for word in sorted(words):
        pattern = re.compile(r'\b' + re.escape(stem(word)) + r'\w*', re.I)
        if any(pattern.search(t) for t in shared):
            continue
        holding = [call for call in calls if any(pattern.search(t) for t in string_leaves(call.get('args'))
                                                   if len(PROSE_WORD.findall(t)) >= 3)]
        answered = [call for call in holding if word in {w.lower() for w in value_words(call.get('value'), call.get('args'))}]
        if holding and len(answered) == len(holding):
            name = holding[0].get('name') or holding[0].get('path')
            texts = [t for t in string_leaves(holding[0].get('args')) if pattern.search(t)]
            hit = pattern.search(texts[0]) if texts else None
            excerpt = texts[0][max(0, hit.start() - 40):hit.end() + 40] if hit else ''
            found.append(f'{name}: {"the one call" if len(holding) == 1 else f"every one of the {len(holding)} calls"} whose '
                         f'text says "{word}" answers {word} (e.g. "...{excerpt}..."), so the word alone settles it')
    return found


def identical_arguments(args) -> list[str]:
    if not isinstance(args, dict):
        return []
    keys = list(args)
    same = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            left, right = args[a], args[b]
            if left == right and (isinstance(left, (dict, list)) or (isinstance(left, str) and len(left) >= 12)):
                same.append(f'{a} and {b}')
    return same


def hardest_entries(notes: str) -> list[str]:
    section = re.search(r'^#+\s*Hardest items[^\n]*\n([\s\S]*?)(?=^#|\Z)', notes, re.M | re.I)
    return re.findall(r'^\s*(?:[-*]|\d+\.)\s+(.+)$', section.group(1), re.M) if section else []


def own_entries(row: dict) -> dict[int, dict]:
    """The children entries that answer exactly one call of the reference run, with that call's judgment name and shown text."""
    per_entry: dict = {}
    for answer in row.get('answers', []):
        per_entry.setdefault(answer['entry'], []).append(answer)
    return {entry: answers[0] for entry, answers in per_entry.items() if entry >= 0 and len(answers) == 1}


def calls_of_own_entries(row: dict, entries: set[int]) -> list[dict]:
    """The trace's judgment calls answered by one of `entries`: a call is matched to the answer of the same judgment whose
    shown arguments hold all its prose, and counts only when every such answer is one of those entries."""
    out = []
    for call in row.get('children', []):
        leaves = [leaf[:30] for leaf in string_leaves(call.get('args')) if len(PROSE_WORD.findall(leaf)) >= 3]
        if not leaves:
            continue
        matching = [a for a in row.get('answers', []) if a['name'] == call.get('name') and all(l in a['shown'] for l in leaves)]
        if matching and all(a['entry'] in entries for a in matching):
            out.append(call)
    return out


def dependence_probes(cases: list[dict], trace: dict) -> list[dict]:
    """For each case, one probe per own entry with a finite value: another answer its judgment gives elsewhere."""
    by_id = {row['id']: row for row in trace.get('results', [])}
    probes = []
    for case in cases:
        row = by_id.get(case['id'], {})
        children = case['curriculum']['reference'].get('children') or []
        names = {a['entry']: a['name'] for a in row.get('answers', []) if a['entry'] >= 0}
        values: dict = {}
        for entry, name in names.items():
            if 'value' in children[entry]:
                values.setdefault(name, []).append(children[entry]['value'])
        mine = []
        for entry in own_entries(row):
            if 'value' not in children[entry]:
                continue
            own = json.dumps(children[entry]['value'], sort_keys=True)
            other = next((v for v in values.get(names[entry], []) if json.dumps(v, sort_keys=True) != own), None)
            if other is None and isinstance(children[entry]['value'], bool):
                other = not children[entry]['value']
            if other is not None:
                mine.append({'entry': entry, 'value': other})
        probes.append({'case': case, 'probes': mine[:300]})
    return probes


def reference_report(data: dict, cases: list[dict], trace: dict, notes: str, declared: int | None,
                     dependence: dict | None = None) -> tuple[list[str], list[str], str]:
    """Problems, warnings and the readable reference report (qualification/reference.txt) from the reference trace."""
    problems, warnings, lines = [], [], []
    names = {case['id']: inst.get('name') for case, inst in zip(cases, data.get('instances') or [])}
    by_id = {case['id']: case for case in cases}
    shown_all = []
    for row in trace.get('results', []):
        case = by_id.get(row['id'])
        if not case:
            continue
        label = f'instance {names.get(row["id"])}'
        children = case['curriculum']['reference'].get('children') or []
        lines += [f'== {label} ({row["id"]}): {row.get("status")}, {row.get("calls")} model calls',
                  'returned: ' + json.dumps(row.get('value'), ensure_ascii=False)]
        if row.get('detail'):
            lines.append('detail: ' + row['detail'])
        if row.get('error') or row.get('status') != 'done':
            problems.append(f'F3: {label}: the reference trace did not run to an answer: '
                            f'{(row.get("error") or row.get("detail") or row.get("status"))!s:.600}')
        # C2 counts the prose of calls answered by an entry of their own that the result depends on (changing the entry's
        # answer changes what the reference returns): text a scripted reading decides.
        per_entry: dict = {}
        for answer in row.get('answers', []):
            per_entry.setdefault(answer['entry'], []).append(answer['shown'])
        own = own_entries(row)
        matters = ((dependence or {}).get(row['id']) or {})
        decisive = {entry for entry in own if matters.get(str(entry), matters.get(entry, dependence is None))}
        for entry in sorted(set(own) - decisive):
            item = next(iter(ITEM_TOKEN.findall(own[entry]['shown'])), '')
            warnings.append(f'{label}: no answer depends on children entry {entry} ({own[entry]["name"]} {item}): another '
                            'reading of it returns the same result, so its text does not count for C2')
        judged_all = prose_words([c.get('args') for c in row.get('children', [])])
        judged = prose_words([c.get('args') for c in calls_of_own_entries(row, decisive)])
        lines.append(f'prose words reaching judgment calls: {judged_all}; in items read by an entry of their own that the '
                     f'result depends on: {judged}')
        if row.get('status') == 'done':
            if judged < MIN_WORDS:
                problems.append(f'C2: {label}: {judged} words of prose in items a judgment reads with an entry of their own '
                                f'that the result depends on ({judged_all} reach judgment calls in all); at least {MIN_WORDS}')
            # An entry may be there for calls only an ablation or a mutant makes; a reference whose run takes few of its
            # entries did not run the program the entries were written for (a stale runtime, a broken fragment).
            unused = [index for index in range(len(children)) if index not in per_entry]
            for index in unused:
                warnings.append(f'{label}: children entry {index} {json.dumps(children[index].get("match"), ensure_ascii=False)[:160]} '
                                'answers no call of the reference run (fine when an ablation or a mutant needs it)')
            if children and len(unused) * 2 > len(children):
                problems.append(f'F3: {label}: the reference run took {len(children) - len(unused)} of its {len(children)} '
                                f'children entries ({row.get("calls")} model calls); it does not run the program they '
                                'were written for: check the fragments against the openings in reference.txt')
        by_name: dict = {}
        for call in row.get('children', []):
            by_name.setdefault(call.get('name') or call.get('path'), []).append(call)
        for calls in by_name.values():
            for copy in deciding_words(calls):
                problems.append(f'S7: {label}: {copy}; let the item describe what happens and the judgment name it')
        for call in row.get('children', []):
            for pair in identical_arguments(call.get('args')):
                problems.append(f'S8: {label}: {call.get("name")} is asked about {pair}, which are the same value; code '
                                'settles exact identity before it asks a judgment')
        uses: dict = {}
        for answer in row.get('answers', []):
            uses.setdefault(answer['name'], {}).setdefault(answer['entry'], []).append(answer['shown'])
            shown_all.append(answer['shown'])
        for name, entries in sorted(uses.items()):
            lines.append(f'-- {name}: {sum(len(v) for v in entries.values())} calls')
            for entry, shown in sorted(entries.items()):
                child = children[entry] if 0 <= entry < len(children) else {}
                items = [next(iter(ITEM_TOKEN.findall(text)), oneline(text)[:40]) for text in shown]
                # An entry with an item fragment besides the instruction's is written for one item; when it answers
                # several different calls, its item fragment also occurs in their openings (a list of the same records).
                if len(child.get('match') or []) >= 2 and len(set(shown)) > 1:
                    warnings.append(f'{label}: children entry {entry} {json.dumps(child.get("match"), ensure_ascii=False)[:160]} '
                                    f'answered {len(set(shown))} different {name} calls ({", ".join(dict.fromkeys(items))[:200]}): '
                                    'an entry with an item fragment is meant for one item; when that fragment also occurs in '
                                    "other calls' arguments, take it from the argument's own line "
                                    '(`event: Event = { eventId: "E15"`), and write a default as an entry with the '
                                    'instruction fragment alone')
                what = (f'entry {entry} {json.dumps(child.get("match"), ensure_ascii=False)} -> '
                        f'{json.dumps(child.get("value", child.get("call")), ensure_ascii=False)[:120]}') if child else 'no entry'
                lines.append(f'   {what}: {len(shown)} call(s): {", ".join(items)[:600]}')
    shown_normalized = [normalized(text) for text in shown_all]
    for number, entry in enumerate(hardest_entries(notes), 1):
        tokens = [t.strip('.,:;') for t in ITEM_TOKEN.findall(entry)]
        # An item whose call carries no id (a phrase a landmark judgment resolves) is named by quoting its text.
        quoted = [normalized(q).strip(' .,;:') for q in QUOTED.findall(entry) if len(q.split()) >= 2]
        if not (any(re.search(rf'(?<![\w]){re.escape(t)}(?![\w])', text) for t in tokens if t for text in shown_all)
                or any(q and q in text for q in quoted for text in shown_normalized)):
            problems.append(f'C4: hardest item {number} ("{entry[:60]}") names no item a judgment is called on; name the '
                            'item by an id its call carries, or quote a phrase of its text in double quotes')
    return problems, warnings, '\n'.join(lines) + '\n'


def oneline(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()


def metrics(data: dict, files: dict[str, str]) -> dict:
    root = data['root']
    stem = root[:-3]
    nl_files = [p for p in files if p.endswith('.nl')]
    ts_files = [p for p in files if p.endswith('.ts') and p != 'types.ts' and p.startswith(stem + '/')]
    inline = sorted({m.group(1).strip()[:80] for p in ts_files for m in INLINE_NL.finditer(files[p])})
    exported = sum(len(TS_EXPORT.findall(files[p])) for p in ts_files)
    ts_lines = sum(1 for p in ts_files for line in strip_comments(files[p]).splitlines()
                   if line.strip() and not line.strip().startswith(('import ', 'export type', 'type ')))
    # Call depth: root (1) -> an item of its callable folder (2) -> a judgment that item calls: a .nl in the item's own
    # folder, a sibling .nl a TypeScript helper imports, an inline nl in it, or a std combinator it runs (3).
    depth = 1 + bool([p for p in files if p.startswith(stem + '/')])
    for path in files:
        if not path.startswith(stem + '/') or path.count('/') != 1:
            continue
        own = path.rsplit('.', 1)[0] + '/'
        text = files[path]
        if any(p.startswith(own) for p in files) or (path.endswith('.ts') and (
                re.search(r"from\s+'\./[^']+\.nl'", text) or INLINE_NL.search(text)
                or re.search(r'\b(' + '|'.join(STD) + r')\s*\(', text))):
            depth = 3
    for path in files:
        if path.startswith(stem + '/') and path.count('/') >= 2:
            depth = max(depth, path.count('/') + 1)
    source = '\n'.join(strip_comments(text) if not path.endswith('.nl') else text for path, text in files.items())
    constructs = sorted(name for name, pattern in CONSTRUCT_PATTERNS.items() if re.search(pattern, source, re.M))
    judgment_files = [p for p in nl_files if p != root]
    instances = data.get('instances', [])
    return {'nl_functions': len(nl_files), 'judgment_files': len(judgment_files), 'inline_nl': len(inline),
            'ts_helpers': len(ts_files), 'ts_exported_functions': exported, 'ts_logic_lines': ts_lines,
            'functions': len(nl_files) + exported, 'call_depth': depth, 'constructs': constructs,
            'instances': [{'name': i.get('name'), 'words': words(i.get('inputs')), 'prose_words': prose_words(i.get('inputs')),
                           'longest_list': longest_list(i.get('inputs')), 'text_items': text_items(i.get('inputs'))}
                          for i in instances]}


def static_problems(submission: Path, data: dict, files: dict[str, str], m: dict) -> list[str]:
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
    # N2: no generator anywhere in the submission
    for entry in submission.iterdir():
        if entry.name not in SUBMISSION_ENTRIES and not entry.name.startswith('.'):
            problems.append(f'N2: {entry.name} is not part of a submission (program/, submission.json, NOTES.md); '
                            'instances are literal data, not generated: move logs and notes beside SUBMISSION')
    instances = data.get('instances') or []
    card_path = submission / 'idea.json'
    try:
        declared = declared_words(json.loads(card_path.read_text())) if card_path.is_file() else None
    except ValueError:
        declared = None
        problems.append('C2: idea.json does not parse')
    if not card_path.is_file():
        problems.append('C2: SUBMISSION/idea.json is missing; keep the idea card you built there (its instance size binds)')
    if not 2 <= len(instances) <= 3:
        problems.append('F2: two or three instances')
    canonical = lambda value: json.dumps(value, sort_keys=True)  # noqa: E731
    helpers = {Path(p).stem for p in files if p.startswith(stem + '/') and p.count('/') == 1}
    judgments = data.get('judgments') or []
    names = {j.get('name') for j in judgments}
    for index, inst in enumerate(instances):
        label = f'instance {inst.get("name", index)}'
        if inst.get('oracle') not in ORACLES:
            problems.append(f'F2: {label}: oracle must be exact, normalized or verifier')
        if inst.get('oracle') == 'verifier':
            path = (inst.get('verifier') or {}).get('path', '')
            if not path.startswith(VERIFIER_ROOT) or not (REPO / path).is_file():
                problems.append(f'F2: {label}: verifier must be an existing module under {VERIFIER_ROOT}')
        if 'expected' not in inst:
            problems.append(f'F2: {label}: expected is missing')
        reference = inst.get('reference') or {}
        root_actions = reference.get('root') or []
        code = ' '.join(str(a[1].get('code', '')) for a in root_actions if isinstance(a, list) and a and a[0] == 'eval')
        if not code or not any(re.search(rf'\b{re.escape(h)}\s*\(', code) for h in helpers):
            problems.append(f'F3: {label}: the reference root eval must call the program\'s own helpers')
        if not reference.get('children'):
            problems.append(f'F3: {label}: the reference needs scripted children for the judgment calls')
        wrong = inst.get('wrong_answers') or []
        if len(wrong) < 3 or not all(isinstance(w, dict) and w.get('mistake') for w in wrong):
            problems.append(f'F4: {label}: at least three wrong answers, each with its mistake')
        for number, w in enumerate(wrong, 1):
            if isinstance(w, dict) and not (isinstance(w.get('children'), list) or isinstance(w.get('code'), str)):
                problems.append(f'F4: {label}: wrong answer {number} declares no mistake to run: give "children" (the '
                                'judgment answers the mistake changes) or "code" (the root script with the exact step '
                                'done the wrong way); qualify replays it and the checker must reject what it returns')
        if any(isinstance(w, dict) and 'value' in w and canonical(w['value']) == canonical(inst.get('expected')) for w in wrong):
            problems.append(f'F4: {label}: a wrong answer equals expected')
        ablated = {a.get('judgment') for a in inst.get('ablations') or [] if a.get('children')}
        if names - ablated:
            problems.append(f'S2: {label}: no ablation for judgment(s) {", ".join(sorted(n for n in names - ablated if n))}')
        baseline = inst.get('keyword_baseline')
        if not isinstance(baseline, str) or 'return' not in baseline:
            problems.append(f'S3: {label}: keyword_baseline is a root eval script ending in return')
        elif INLINE_NL.search(baseline) or re.search(r'\bdecide\s*\(', baseline):
            problems.append(f'S3: {label}: the keyword baseline makes a natural-language call')
        size = m['instances'][index]
        if size['prose_words'] < MIN_WORDS:
            problems.append(f'C2: {label}: {size["prose_words"]} words of prose in the inputs; at least {MIN_WORDS} words '
                            'of text people wrote reach the judgments (ids, codes and timecodes do not count)')
        if declared and size['prose_words'] < 0.9 * declared:
            problems.append(f'C2: {label}: {size["prose_words"]} words of prose, but the idea card declares {declared}; '
                            "the card's size is binding: write the instance at the size the card names")
    if len(judgments) < 3 or m['judgment_files'] + m['inline_nl'] < 3:
        problems.append('S1: at least three distinct judgments (judgment .nl files or inline nl calls), listed in judgments')
    if m['functions'] < 5:
        problems.append(f'C1: {m["functions"]} functions; at least five (.nl files and exported TypeScript functions)')
    if m['call_depth'] < 3:
        problems.append(f'C1: call depth {m["call_depth"]}; at least three (root, helper, judgment)')
    if m['ts_logic_lines'] < MIN_TS_LINES:
        problems.append(f'D1: {m["ts_logic_lines"]} lines of TypeScript logic in the callable folder; at least {MIN_TS_LINES}')
    problems += untrusted_problems(files)
    problems += quote_problems(submission, data, files)
    notes = submission / 'NOTES.md'
    listed = hardest_items(notes.read_text()) if notes.is_file() else 0
    if listed < MIN_HARDEST:
        problems.append(f'C4: NOTES.md lists {listed} hardest items; give a section "## Hardest items" with at least '
                        f'{MIN_HARDEST} entries, each the item and why a careful reader could get it wrong')
    found = [c for c in m['constructs'] if c != 'inline-nl']
    if len(found) < 3:
        problems.append(f'D2: semantic-FP constructs found: {", ".join(found) or "none"}; at least three')
    head, body = frontmatter(files[root])
    returns = re.search(r'^returns:\s*["\']?([A-Za-z_]\w*)["\']?\s*$', head, re.M)
    types = files.get('types.ts', '')
    if not returns or not re.search(rf'/\*\*[\s\S]*?\*/\s*export\s+type\s+{returns.group(1)}\b', types):
        problems.append('D4: the root returns a named type exported from types.ts with a doc comment')
    if re.search(r'output contract', body, re.I):
        problems.append('D4: the output format is declared in types.ts, not in an instruction paragraph')
    return problems


def world(data: dict) -> str:
    return 'invented:' + data['slug']


def split_of(data: dict) -> str:
    bucket = int(text_digest('invented-20261124|' + world(data))[:8], 16) % 100
    return 'train' if bucket < 70 else ('dev' if bucket < 85 else 'confirmation')


def verifier_binding(inst: dict) -> dict | None:
    if inst.get('oracle') != 'verifier':
        return None
    spec = inst['verifier']
    return {'path': spec['path'], 'sha256': hashlib.sha256((REPO / spec['path']).read_bytes()).hexdigest(),
            'oracle': spec['oracle']}


def package(data: dict, files: dict[str, str]) -> list[dict]:
    family = 'inv_' + re.sub(r'[^a-z0-9]+', '_', data['slug'].lower()).strip('_')
    root = data['root']
    root_body = frontmatter(files[root])[1]
    author = data.get('author') or {}
    cases = []
    for inst in data['instances']:
        verifier = verifier_binding(inst)
        aliases = inst.get('alias_answers') or []
        oracle = 'exact' if verifier else ({'level': inst['oracle'], 'alternates': aliases} if aliases else inst['oracle'])
        payload = json.dumps({'files': files, 'instance': inst}, sort_keys=True)
        case_id = f'inv:{slug(data["slug"], 48)}:{slug(inst["name"], 24)}:{text_digest(payload)[:8]}'
        constructs = sorted(set(data.get('constructs') or []))
        cases.append({
            'version': 'natlang.program/2', 'id': case_id, 'kind': 'lambda_source', 'family': family,
            'source': 'invented:' + str(author.get('builder', 'unknown')), 'split': split_of(data),
            'source_ids': [f'invented:{data["slug"]}:{inst["name"]}'], 'source_groups': [world(data)],
            'source_revisions': [GENERATOR], 'license': 'invented',
            'gold_sources': ['invented-verifier-oracle' if verifier else 'invented-exact-oracle'],
            'generation': {'generator': GENERATOR, 'teacher_generation': True, 'author': author,
                           'method': 'a new program idea (checked against the atlas of existing ideas), its program, '
                                     'literal instances, exact checker and controls written by frontier models; '
                                     'qualified by reference replay through the program, wrong answers, judgment '
                                     'ablations and keyword baselines; no sampled trajectory'},
            'curriculum': {
                'version': CURRICULUM_VERSION, 'family': family, 'family_version': 1, 'shape': data['slug'],
                'variant': 'base', 'pair_group': None, 'split_group': world(data), 'slice': 'nested_scoped',
                'domain': 'other', 'mode': 'single_call', 'inline': 'optional', 'named': 'required',
                'evidence': {'world': [], 'retrieved': [], 'background': []}, 'assumptions': [], 'decisive': [],
                'plausible_actions': [], 'minimum_sequence': [],
                'reference': {'root': inst['reference']['root'], 'children': inst['reference']['children']},
                'rl': {'level': int(inst.get('level', 3)), 'family': family,
                       'basis': 'invented-verifier-oracle-reference-verified' if verifier
                       else 'invented-exact-oracle-reference-verified'},
                'external': {'source': 'invented', 'upstream': 'invented:' + str(author.get('inventor', 'unknown')),
                             'upstream_id': f'{data["slug"]}:{inst["name"]}', 'upstream_split': 'train-carve',
                             'unit': world(data), 'shape': data['slug'],
                             'instruction_sha256': text_digest(root_body)[:16], 'instruction_id': text_digest(root_body)[:12],
                             'neuralese_room': '', 'framing': 'invented', 'constructs': constructs,
                             'subject_domain': 'other', 'oracle_depth': 'answer',
                             'checks': {'wrong_answers': [w['value'] for w in inst['wrong_answers']
                                                          if 'value' in w and 'children' not in w and 'code' not in w],
                                        'wrong_mutants': [w for w in inst['wrong_answers'] if 'children' in w or 'code' in w],
                                        'alias_answers': aliases,
                                        'keyword_baseline_code': [inst['keyword_baseline']],
                                        'ablations': inst.get('ablations') or []}}},
            'semantics': {'root': root, 'files': files, 'inputs': inst['inputs'], 'expected': inst['expected'],
                          'oracle': oracle, **({'trusted_exact_verifier': verifier} if verifier else {})},
        })
    return cases


REFERENCE_CALL_FACTOR = 2   # owner 2026-11-25: an episode's model-call budget is about twice the reference run's calls


def with_reference_calls(cases: list[dict], trace: dict) -> list[dict]:
    """Each case declares its episode budget: REFERENCE_CALL_FACTOR x its reference run's model calls (the collector
    raises the run's budget to it, never lowers it: ts-host teacher/collector.ts caseModelRequestBudget)."""
    calls = {row['id']: row.get('calls') for row in trace.get('results', [])}
    for case in cases:
        if calls.get(case['id']):
            case['semantics']['episode_model_calls'] = REFERENCE_CALL_FACTOR * int(calls[case['id']])
    return cases


def stage_cases(cases: list[dict], trace: dict) -> list[dict]:
    """The stage tasks of packaged cases: one per distinct leaf judgment call of each reference run (module docstring)."""
    by_id = {row['id']: row for row in trace.get('results', [])}
    out = []
    for case in cases:
        files = case['semantics']['files']
        seen = set()
        for child in (by_id.get(case['id']) or {}).get('children') or []:
            path = child.get('path') or ''
            if (path not in files or not path.endswith('.nl') or child.get('functional') or not child.get('returned')
                    or child.get('status') != 'success' or not isinstance(child.get('args'), dict)
                    or any(p.startswith(path[:-3] + '/') for p in files)):
                continue
            key = json.dumps([path, child['args']], sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            judgment = Path(path).stem
            stage = json.loads(json.dumps(case))
            stage['id'] = f'{case["id"]}:stage:{slug(judgment, 32)}:{text_digest(key)[:8]}'
            stage['source_ids'] = [f'{sid}:stage:{judgment}:{text_digest(key)[:8]}' for sid in case['source_ids']]
            stage['semantics'] = {'root': path, 'files': files, 'inputs': child['args'], 'expected': child['value'],
                                  'oracle': 'exact'}
            curriculum = stage['curriculum']
            curriculum['shape'] = f'{curriculum["shape"]}:{judgment}'
            curriculum['variant'] = 'stage'
            curriculum['reference'] = {'root': [['return_result', {'status': 'success', 'value': child['value']}]]}
            curriculum['rl'] = {**curriculum['rl'], 'level': 1,
                                'basis': 'invented-stage-reference-answer-of-a-qualified-program'}
            external = curriculum['external']
            external['shape'] = curriculum['shape']
            external['stage'] = {'of': case['id'], 'judgment': judgment, 'path': path}
            body = frontmatter(files[path])[1]
            external['instruction_sha256'] = text_digest(body)[:16]
            external['instruction_id'] = text_digest(body)[:12]
            external['checks'] = {'wrong_answers': [], 'alias_answers': [], 'keyword_baseline_code': [], 'ablations': []}
            stage['generation'] = {**stage['generation'],
                                   'method': 'a judgment call of a qualified invented program\'s reference run, as its own '
                                             'task: the call\'s arguments and the reference\'s answer; no sampled trajectory'}
            out.append(stage)
    return out


def run_node(script: str, runtime: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([node(), str(REPO / 'scripts/rl_tasks' / script), str(runtime), *args],
                          capture_output=True, text=True, timeout=3600)


def qualify_stages(out: Path, runtime: Path, cases: list[dict], trace: dict) -> dict:
    """Cut and replay the stage tasks; only those whose reference replays to an accepted answer are kept."""
    stages = stage_cases(cases, trace)
    (out / 'stages-input.jsonl').write_text(''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in stages))
    if not stages:
        (out / 'stages.jsonl').write_text('')
        return {'cut': 0, 'kept': 0}
    verified = out / 'stages-verified'
    run = run_node('verify.mjs', runtime, str(out / 'stages-input.jsonl'), str(verified), '4')
    try:
        kept_ids = {row['id'] for row in map(json.loads, (verified / 'cases.jsonl').read_text().splitlines()) if row}
    except OSError:
        return {'cut': len(stages), 'kept': 0, 'error': (run.stderr or run.stdout)[-600:]}
    kept = [c for c in stages if c['id'] in kept_ids]
    (out / 'stages.jsonl').write_text(''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in kept))
    per = collections.Counter(c['curriculum']['external']['stage']['judgment'] for c in kept)
    return {'cut': len(stages), 'kept': len(kept), 'by_judgment': dict(per)}


def node() -> str:
    for candidate in sorted(Path.home().glob('.local/share/natlang-dev/node-v24*/bin/node'), reverse=True):
        return str(candidate)
    return 'node'


def newest(path: Path) -> float:
    if path.is_file():
        return path.stat().st_mtime
    return max((f.stat().st_mtime for f in path.rglob('*') if f.is_file()), default=0.0) if path.is_dir() else 0.0


def stale_runtime(runtime: Path) -> bool:
    """Whether the runtime's built dist is older than its sources (the replays would run old code)."""
    built = min((runtime / 'dist' / rel).stat().st_mtime if (runtime / 'dist' / rel).is_file() else 0.0
                for rel in ('cli/main.js', 'teacher/curriculum.js'))
    return built < max(newest(runtime / 'src'), newest(runtime / 'prelude.js'), newest(runtime / 'package.json'))


def runtime_problems(runtime: Path) -> list[str]:
    """Rebuild a stale dist (scripts/dev-bootstrap.mjs, as the natlang CLI does) and fail when it stays stale."""
    if not stale_runtime(runtime):
        return []
    bootstrap = runtime.parent / 'scripts' / 'dev-bootstrap.mjs'
    if bootstrap.is_file():
        subprocess.run([node(), '--input-type=module', '-e',
                        f'const m = await import({json.dumps(bootstrap.as_uri())}); await m.ensureNodeBuild({json.dumps(str(runtime.parent))});'],
                       capture_output=True, text=True, timeout=1800)
    return [f'F1: {runtime}/dist is older than its sources and could not be rebuilt; the replays would run old code '
            '(rebuild with npm --prefix ts-host run build:node and qualify again)'] if stale_runtime(runtime) else []


def natlang_check(program: Path) -> list[str]:
    natlang = os.environ.get('NATLANG_BIN') or str(Path.home() / '.local/bin/natlang')
    run = subprocess.run([natlang, 'check', '--json'], cwd=program, capture_output=True, text=True, timeout=900)
    try:
        report = json.loads(run.stdout[run.stdout.index('{'):])
    except ValueError:
        return [f'F1: natlang check gave no JSON (exit {run.returncode}): {(run.stderr or run.stdout)[-600:]}']
    diagnostics = report.get('diagnostics') or []
    errors = [d for d in diagnostics if d.get('severity', 'error') == 'error']
    out = [f'F1: {d.get("file", "")}:{d.get("line", "")} {d.get("code", "")} {d.get("message", "")}'[:400] for d in errors]
    if run.returncode != 0 and not out:
        out.append(f'F1: natlang check exited {run.returncode}: {json.dumps(report)[:400]}')
    return out


def bar() -> dict:
    requirements = REPO / 'applications' / 'program-inventor' / 'requirements.json'
    return {'version': BAR_VERSION, 'requirements_sha256': hashlib.sha256(requirements.read_bytes()).hexdigest()[:16]
            if requirements.is_file() else None}


def write_report(out: Path, report: dict) -> None:
    """report.json appears only whole, at the end of a run (a temp file moved in)."""
    temp = out / f'.report.json.{os.getpid()}'
    temp.write_text(json.dumps(report, ensure_ascii=False, indent=1) + '\n')
    os.replace(temp, out / 'report.json')


def qualify(submission: Path, runtime: Path, print_reference: bool = False) -> dict:
    data, files = load(submission)
    m = metrics(data, files)
    problems = static_problems(submission, data, files, m)
    out = submission / 'qualification'
    out.mkdir(exist_ok=True)
    # The previous run's report goes at once: a run stopped part way (preempted, killed) leaves no report rather than
    # an old one that reads as this run's result.
    (out / 'report.json').unlink(missing_ok=True)
    report: dict = {'slug': data.get('slug'), 'bar': bar(), 'digest': digest(submission), 'metrics': m}
    problems += runtime_problems(runtime)
    problems += natlang_check(submission / 'program')
    try:
        cases = package(data, files)
    except (KeyError, TypeError, ValueError) as error:
        problems.append(f'package: {error}')
        cases = []
    (out / 'cases.jsonl').write_text(''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in cases))
    if cases:
        verified = out / 'verified'
        run = subprocess.run([node(), str(REPO / 'scripts/rl_tasks/verify.mjs'), str(runtime), str(out / 'cases.jsonl'),
                              str(verified), '2'], capture_output=True, text=True, timeout=3600)
        try:
            verification = json.loads((verified / 'verification.json').read_text())
        except (OSError, ValueError):
            verification = {'error': (run.stderr or run.stdout)[-1500:]}
            problems.append('verify.mjs did not finish: ' + verification['error'][-300:])
        report['verification'] = {key: verification.get(key) for key in ('summary', 'failures', 'error')}
        broken = {row.get('id') for row in verification.get('results', []) if not row.get('reference_ok')}
        for row in verification.get('results', []):
            name = row.get('id', '?')
            if not row.get('reference_ok'):
                problems.append(f'F3: {name}: reference replay not accepted: {json.dumps(row.get("problems"))[:400]}')
            for wrong in row.get('wrong', []):
                if wrong.get('accepted') is not False:
                    problems.append(f'F4: {name}: wrong answer not rejected: {json.dumps(wrong)[:300]}')
            for number, mutant in enumerate(row.get('mutants', []), 1):
                if not mutant.get('ran'):
                    problems.append(f'F4: {name}: wrong answer "{mutant.get("mistake")}" {mutant.get("error", "did not run")}'[:900])
                elif mutant.get('accepted') is not False:
                    problems.append(f'F4: {name}: wrong answer "{mutant.get("mistake")}" returns an answer the checker '
                                    f'accepts: {json.dumps(mutant.get("value"))[:300]}')
                elif mutant.get('value_matches') is False:
                    problems.append(f'F4: {name}: wrong answer "{mutant.get("mistake")}" declares a value, but its replay '
                                    f'returns {json.dumps(mutant.get("value"))[:300]}; drop "value" or make it that')
            for baseline in row.get('keyword_baseline', []):
                if baseline.get('accepted') is not False:
                    problems.append(f'S3: {name}: keyword baseline not rejected: {json.dumps(baseline)[:900]}')
        run = subprocess.run([node(), str(REPO / 'scripts/rl_tasks/ablate.mjs'), str(runtime), str(out / 'cases.jsonl'),
                              str(out / 'ablations.json')], capture_output=True, text=True, timeout=3600)
        try:
            ablations = json.loads((out / 'ablations.json').read_text())
        except (OSError, ValueError):
            ablations = {'error': (run.stderr or run.stdout)[-1500:]}
            problems.append('ablate.mjs did not finish: ' + ablations['error'][-300:])
        report['ablations'] = ablations
        for row in ablations.get('results', []):
            if row.get('rejected') is not True and row.get('id') in broken:
                # An ablation replays the reference with one judgment changed: while the reference fails, it shows nothing.
                problems.append(f'S2: {row["id"]}: ablating {row["judgment"]} not judged: the reference replay of '
                                f'{row["id"]} fails (F3 above); fix that first')
            elif row.get('rejected') is not True:
                problems.append(f'S2: {row["id"]}: ablating {row["judgment"]} did not change the answer to a rejected one'
                                f' ({row.get("error") or json.dumps(row.get("value"))[:200]})')
        run = run_node('reference_trace.mjs', runtime, str(out / 'cases.jsonl'), str(out / 'trace.json'))
        try:
            trace = json.loads((out / 'trace.json').read_text())
        except (OSError, ValueError):
            trace = {'error': (run.stderr or run.stdout)[-1500:]}
            problems.append('reference_trace.mjs did not finish: ' + trace['error'][-300:])
        report['reference_calls'] = {row['id']: row.get('calls') for row in trace.get('results', [])}
        card = submission / 'idea.json'
        try:
            declared = declared_words(json.loads(card.read_text())) if card.is_file() else None
        except ValueError:
            declared = None
        notes = submission / 'NOTES.md'
        (out / 'dependence-probes.json').write_text(json.dumps(dependence_probes(cases, trace), ensure_ascii=False))
        run = run_node('entry_dependence.mjs', runtime, str(out / 'dependence-probes.json'), str(out / 'dependence.json'))
        try:
            dependence = {row['id']: row['entries'] for row in json.loads((out / 'dependence.json').read_text())['results']}
        except (OSError, ValueError, KeyError):
            dependence = {}
            problems.append('entry_dependence.mjs did not finish: ' + (run.stderr or run.stdout)[-300:])
        found, warnings, text = reference_report(data, cases, trace, notes.read_text() if notes.is_file() else '', declared,
                                                 dependence)
        problems += found
        report['warnings'] = warnings
        (out / 'reference.txt').write_text(text)
        if print_reference:
            print(text, flush=True)
        cases = with_reference_calls(cases, trace)
        (out / 'cases.jsonl').write_text(''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in cases))
        if not problems:
            report['stages'] = qualify_stages(out, runtime, cases, trace)
    report['problems'] = problems
    report['ok'] = not problems
    write_report(out, report)
    return report


def dry_run(submission: Path, runtime: Path, controls: bool = False) -> dict:
    """The program's TypeScript run with the scripted judgment answers, no model, in seconds: what the reference returns
    against `expected`, which entry answered which call, the static checks, and with `controls` what each wrong
    answer, ablation and keyword baseline returns. It writes nothing into the submission (qualification/ is qualify's)."""
    data, files = load(submission)
    m = metrics(data, files)
    problems = static_problems(submission, data, files, m) + runtime_problems(runtime)
    try:
        cases = package(data, files)
    except (KeyError, TypeError, ValueError) as error:
        return {'ok': False, 'problems': problems + [f'package: {error}'], 'warnings': [], 'text': ''}
    with tempfile.TemporaryDirectory(prefix='invented-dry-run-') as scratch:
        source, target = Path(scratch) / 'cases.jsonl', Path(scratch) / 'dry-run.json'
        source.write_text(''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in cases))
        run = run_node('dry_run.mjs', runtime, str(source), str(target), '1' if controls else '0')
        try:
            trace = json.loads(target.read_text())
        except (OSError, ValueError):
            return {'ok': False, 'problems': problems + ['dry_run.mjs did not finish: ' + (run.stderr or run.stdout)[-1500:]],
                    'warnings': [], 'text': ''}
    notes = submission / 'NOTES.md'
    card = submission / 'idea.json'
    try:
        declared = declared_words(json.loads(card.read_text())) if card.is_file() else None
    except ValueError:
        declared = None
    found, warnings, text = reference_report(data, cases, trace, notes.read_text() if notes.is_file() else '', declared)
    problems += found
    lines = ['(dry run: C2 counts every entry of its own as decisive; qualify counts only those the result depends on)']
    for inst, row in zip(data.get('instances') or [], trace.get('results', [])):
        label = f'instance {inst.get("name")}'
        lines.append(f'== {label}: reference {"accepted" if row.get("accepted") else "NOT accepted"}')
        lines.append('expected: ' + json.dumps(inst.get('expected'), ensure_ascii=False))
        lines.append('returned: ' + json.dumps(row.get('value'), ensure_ascii=False))
        if row.get('status') == 'done' and not row.get('accepted'):
            problems.append(f'F3: {label}: the reference returns an answer the checker rejects')
        for mutant in row.get('mutants', []):
            state = ('did not run: ' + mutant.get('error', '') if not mutant.get('ran') else
                     'ACCEPTED (must be rejected)' if mutant.get('accepted') else 'rejected')
            if mutant.get('value_matches') is False:
                state += ', declared value differs'
            lines.append(f'wrong answer "{mutant.get("mistake")}": {state}: {json.dumps(mutant.get("value"), ensure_ascii=False)[:300]}')
            if not (mutant.get('ran') and mutant.get('accepted') is False and mutant.get('value_matches') is not False):
                problems.append(f'F4: {label}: wrong answer "{mutant.get("mistake")}": {state}'[:600])
        for ablation in row.get('ablations', []):
            state = ('did not run: ' + ablation.get('error', '') if not ablation.get('ran') else
                     'ACCEPTED (must be rejected)' if ablation.get('accepted') else 'rejected')
            lines.append(f'ablation {ablation.get("judgment")}: {state}: {json.dumps(ablation.get("value"), ensure_ascii=False)[:300]}')
            if not (ablation.get('ran') and ablation.get('accepted') is False):
                problems.append(f'S2: {label}: ablating {ablation.get("judgment")}: {state}'[:600])
        for baseline in row.get('baselines', []):
            state = ('did not run: ' + baseline.get('error', '') if not baseline.get('ran') else
                     'ACCEPTED (must be rejected)' if baseline.get('accepted') else 'rejected')
            lines.append(f'keyword baseline: {state}: {json.dumps(baseline.get("value"), ensure_ascii=False)[:300]}')
            if not (baseline.get('ran') and baseline.get('accepted') is False):
                problems.append(f'S3: {label}: keyword baseline: {state}'[:600])
    return {'ok': not problems, 'problems': problems, 'warnings': warnings, 'text': '\n'.join(lines) + '\n' + text}


# ---- campaign pipeline: admit one submission, publish batches --------------------------------------------------------
sys.path.insert(0, str(REPO / 'training' / 'neuralese'))
from natlang_neuralese.common.paths import resolve as data_path  # noqa: E402

INVENTOR_ROOT = Path(os.environ.get('NATLANG_INVENTOR_ROOT') or data_path('data_nvme', 'natlang-novel-programs'))
HELD_ROOT = data_path('data_nvme', 'lfm3-rl-dgx-20261029', 'ops', 'held-inv')
HELD_V3 = data_path('data_nvme', 'lfm3-rl-dgx-20261029', 'ops', 'held-v3')
REVIEW_SCHEMA = 'natlang.invented-review/1'
JUDGED = ('S4', 'S5', 'C3', 'D3', 'D5', 'N1')
PYTHON = REPO / '.venv-neuralese/bin/python'


def digest(submission: Path) -> str:
    """sha256 over program/ files (as load() reads them) and submission.json: what a review is a review of."""
    data, files = load(submission)
    return hashlib.sha256(json.dumps({'files': files, 'submission': data}, sort_keys=True).encode()).hexdigest()


def review_problems(review: dict, submission: Path) -> list[str]:
    """What in a review keeps a qualified submission out (owner 2026-11-25: volume over per-program perfection;
    reviews sample quality and feed the brief and skill). Only an expected answer the reviewer shows is wrong
    (``wrong_expected``: item, why) blocks, and only a review of this version; judged requirements are recorded,
    not gates. ``NATLANG_INVENTED_STRICT_REVIEW=1`` restores the all-requirements-met gate."""
    if review.get('schema') != REVIEW_SCHEMA:
        return [f'review: schema must be {REVIEW_SCHEMA}']
    if review.get('digest') != digest(submission):
        return []   # a review of an older version neither blocks nor stands for this one
    problems = [f'wrong expected answer: {str(item)[:400]}' for item in review.get('wrong_expected') or []]
    if os.environ.get('NATLANG_INVENTED_STRICT_REVIEW') == '1':
        if not review.get('reviewer'):
            problems.append('review: reviewer (model id) is missing')
        for rid in JUDGED:
            verdict = (review.get('requirements') or {}).get(rid) or {}
            if verdict.get('met') is not True:
                problems.append(f'{rid}: judged unmet: {"; ".join(map(str, verdict.get("findings") or ["no verdict"]))[:400]}')
    return problems


def sample_reviews(root: Path, count: int, seed: str | None = None, reviewed: bool = False) -> list[dict]:
    """Draw admitted submissions at random for a review (REVIEW_BRIEF.md: one sampled pass, owner 2026-11-25: "reviews
    should only be on a drawn sample"). By default only those without a review of their current version."""
    pool = []
    for path in sorted((root / 'admitted').glob('*/admission.json')):
        admission = json.loads(path.read_text())
        submission = Path(admission.get('submission') or '')
        if not (submission / 'submission.json').is_file():
            continue
        current = digest(submission)
        review = submission / 'review.json'
        done = review.is_file() and json.loads(review.read_text()).get('digest') == current
        if reviewed or not done:
            pool.append({'slug': admission['slug'], 'submission': str(submission), 'split': admission.get('split'),
                         'digest': current, 'reviewed': done})
    return random.Random(seed).sample(pool, min(count, len(pool)))


def withheld(admission: dict, home: Path) -> list[str]:
    """Wrong expected answers a sampled review found in the admitted version: that program stays out of the batches
    published from now on (a review written after admission sits in the submission; admit copies one it saw)."""
    for review_path in (Path(admission.get('submission') or '') / 'review.json', home / 'review.json'):
        if review_path.is_file():
            review = json.loads(review_path.read_text())
            if review.get('digest') == admission.get('digest') and review.get('wrong_expected'):
                return [str(item)[:400] for item in review['wrong_expected']]
    return []


def rows_of(path: Path) -> list[dict]:
    from natlang_neuralese.common import jsonio
    return jsonio.read_jsonl(path) if path.exists() else []


def save_rows(path: Path, rows: list[dict]) -> None:
    from natlang_neuralese.common.jsonio import write_bytes_durable
    path.parent.mkdir(parents=True, exist_ok=True)
    write_bytes_durable(path, ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows).encode())


def admit(submission: Path, *, review_path: Path | None, idea_path: Path | None, root: Path, runtime: Path,
          skip_qualify: bool = False) -> dict:
    out = submission / 'qualification'
    stored = json.loads((out / 'report.json').read_text()) if (out / 'report.json').is_file() else {}
    # A stored report stands only for this submission under the current bar; anything else is qualified again.
    current = skip_qualify and stored.get('bar') == bar() and stored.get('digest') == digest(submission)
    report = stored if current else qualify(submission, runtime)
    if not report.get('ok'):
        return {'admitted': False, 'stage': 'qualify', 'problems': report.get('problems', [])}
    review_path = review_path or submission / 'review.json'
    # A review is optional: qualification admits; a sampled review blocks only on a wrong expected answer.
    review = json.loads(review_path.read_text()) if review_path.exists() else None
    if review is None and os.environ.get('NATLANG_INVENTED_STRICT_REVIEW') == '1':
        return {'admitted': False, 'stage': 'review', 'problems': [f'no review at {review_path} (REVIEW_BRIEF.md)']}
    problems = review_problems(review, submission) if review is not None else []
    if problems:
        return {'admitted': False, 'stage': 'review', 'problems': problems}
    data, _ = load(submission)
    sys.path.insert(0, str(REPO / 'scripts' / 'rl_tasks'))
    import program_atlas
    states = program_atlas.claim_states(root)
    if idea_path is None and (submission / 'idea.json').exists():
        idea_path = submission / 'idea.json'
    idea = json.loads(idea_path.read_text()) if idea_path else (states.get(data['slug']) or {}).get('idea')
    if not idea:
        return {'admitted': False, 'stage': 'idea', 'problems': ['no idea card: pass --idea, write idea.json or claim it']}
    idea = idea.get('idea', idea)
    cases = rows_of(out / 'cases.jsonl')
    stages = rows_of(out / 'stages.jsonl')
    split = split_of(data)
    home = root / 'admitted' / data['slug']
    save_rows(home / 'cases.jsonl', cases)
    save_rows(home / 'stages.jsonl', stages)
    if review is not None:
        shutil.copy2(review_path, home / 'review.json')
    admission = {'slug': data['slug'], 'split': split, 'submission': str(submission), 'digest': digest(submission),
                 'cases': len(cases), 'stages': len(stages), 'reference_calls': report.get('reference_calls'),
                 'reviewed': review is not None and review.get('digest') == digest(submission),
                 'at': time.time(), 'published': None}
    (home / 'admission.json').write_text(json.dumps(admission, indent=1) + '\n')
    ledger = root / 'accepted.jsonl'
    listed = {row['idea']['slug'] for row in rows_of(ledger)}
    if data['slug'] not in listed:
        with open(ledger, 'a') as handle:
            handle.write(json.dumps({'idea': idea, 'submission': str(submission), 'split': split, 'at': time.time()},
                                    ensure_ascii=False) + '\n')
    if data['slug'] in states:
        program_atlas.release(root, data['slug'], 'accepted', str(submission))
    return {'admitted': True, **admission}


def pool_avoid(directory: Path) -> tuple[Path, Path]:
    """Ids and split/source groups of every row training selects from (registry roles grpo-pool, grpo-memetic-tasks):
    what a held slice must not share."""
    from natlang_neuralese.common.data_registry import role_paths
    ids, groups = set(), set()
    for path in [*role_paths('grpo-pool'), *role_paths('grpo-memetic-tasks')]:
        for line in open(path, encoding='utf-8'):
            row = json.loads(line)
            ids.add(row.get('id'))
            groups.add((row.get('curriculum') or {}).get('split_group') or '')
            groups.update(row.get('source_groups') or [])
    ids.discard(None)
    groups.discard('')
    (directory / 'avoid-ids.json').write_text(json.dumps(sorted(ids)))
    (directory / 'avoid-groups.json').write_text(json.dumps(sorted(groups)))
    return directory / 'avoid-ids.json', directory / 'avoid-groups.json'


def registry(*args: str) -> None:
    subprocess.run([str(PYTHON), str(REPO / 'scripts/data_registry.py'), *args], check=True, cwd=REPO)


def publish(root: Path, *, dry_run: bool = False, no_check: bool = False, authored: bool = True) -> dict:
    stamp = time.strftime('%Y%m%d-%H%M%S')
    admitted = [(json.loads((d / 'admission.json').read_text()), d) for d in sorted((root / 'admitted').glob('*'))
                if (d / 'admission.json').exists()]
    for admission, d in admitted:
        reasons = withheld(admission, d)
        if reasons and admission.get('withheld') != reasons and not dry_run:
            admission['withheld'] = reasons
            (d / 'admission.json').write_text(json.dumps(admission, indent=1) + '\n')
        admission['withheld'] = reasons
    admitted = [(a, d) for a, d in admitted if not a['withheld']]
    fresh = [(a, d) for a, d in admitted if not a.get('published')]
    train = [(a, d) for a, d in fresh if a['split'] == 'train']
    held = [(a, d) for a, d in admitted if a['split'] != 'train']
    summary = {'stamp': stamp, 'fresh': [a['slug'] for a, _ in fresh], 'train_ideas': len(train),
               'held_ideas': len(held), 'new_held_ideas': sum(1 for a, _ in fresh if a['split'] != 'train')}
    if dry_run or not fresh:
        return summary
    env = {**os.environ, 'PYTHONPATH': f'{REPO}/training/neuralese:{REPO}'}
    if train:
        pool = root / 'pool' / stamp
        rows = [r for _, d in train for r in rows_of(d / 'cases.jsonl') + rows_of(d / 'stages.jsonl')]
        save_rows(pool / 'cases.jsonl', rows)
        from natlang_neuralese.common.data_registry import role_paths
        against = [a for p in role_paths('grpo-pool') for a in ('--against', p)]
        subprocess.run([str(PYTHON), str(REPO / 'scripts/rl_tasks/measure_repetition.py'), '--cases',
                        str(pool / 'cases.jsonl'), *against, '--label', f'invented-{stamp}', '-o',
                        str(pool / 'repetition.json')], check=True, env=env, cwd=REPO)
        dataset = f'natlang-rl-invented-{stamp}'
        registry('record', '--id', dataset, '--path', str(pool), '--kind', 'rl-tasks', '--include', 'cases.jsonl',
                 '--include', 'repetition.json', '--license', 'invented (project-authored by frontier models)',
                 '--rows', str(len(rows)),
                 '--format', f'natlang.program/2 cases of {len(train)} invented programs (train split): '
                             f'{sum(a["cases"] for a, _ in train)} whole-program tasks and '
                             f'{sum(a["stages"] for a, _ in train)} stage tasks (scripts/rl_tasks/sources/invented.py)',
                 '--note', 'program inventor campaign (plans/NOVEL_PROGRAMS_20261124.md); ideas: '
                           + ', '.join(a['slug'] for a, _ in train))
        registry('promote', dataset, '--role', 'grpo-pool', *(['--no-check'] if no_check else []))
        summary['pool'] = {'dataset': dataset, 'rows': len(rows), 'path': str(pool)}
    if held:
        directory = HELD_ROOT / stamp
        (directory / 'source').mkdir(parents=True, exist_ok=True)
        save_rows(directory / 'source' / 'cases.jsonl', [r for _, d in held for r in rows_of(d / 'cases.jsonl')])
        avoid_ids, avoid_groups = pool_avoid(directory / 'source')
        held_eval = [str(PYTHON), '-m', 'natlang_neuralese.rl.held_eval']
        subprocess.run([*held_eval, 'pin', '--cases', str(directory / 'source' / 'cases.jsonl'), '--dir', str(directory),
                        '--all-rows', '--seed', f'held-inv-{stamp}', '--avoid-ids', str(avoid_ids),
                        '--avoid-groups', str(avoid_groups), '--also-exclude', str(HELD_V3 / 'manifest.json')],
                       check=True, env=env, cwd=REPO)
        shutil.copy2(HELD_V3 / 'engine.json', directory / 'engine.json')
        verify = subprocess.run([*held_eval, 'verify', '--manifest', str(directory / 'manifest.json'),
                                 '--avoid-ids', str(avoid_ids), '--avoid-groups', str(avoid_groups)],
                                capture_output=True, text=True, env=env, cwd=REPO)
        (directory / 'verify.json').write_text(verify.stdout)
        if verify.returncode:
            raise SystemExit(f'held_eval verify found held-inv tasks training sees: {verify.stdout[-1500:]}')
        previous = sorted(p.name for p in HELD_ROOT.glob('*') if p.is_dir() and p.name < stamp and
                          (p / 'manifest.json').exists())
        dataset = f'held-inv-{stamp}'
        tasks = len(json.loads((directory / 'manifest.json').read_text())['tasks'])
        registry('record', '--id', dataset, '--path', str(directory), '--kind', 'held', '--protected',
                 *[a for f in ('manifest.json', 'cases.jsonl', 'engine.json', 'exclusion-ids.json', 'verify.json',
                               'source/cases.jsonl', 'source/avoid-ids.json', 'source/avoid-groups.json')
                   for a in ('--include', f)],
                 '--license', 'invented (project-authored by frontier models)', '--rows', str(tasks),
                 *(['--supersedes', f'held-inv-{previous[-1]}'] if previous else []),
                 '--format', f'held_eval pinned set: {tasks} whole-program tasks of the {len(held)} invented ideas '
                             'whose world drew dev or confirmation (the new-programs slice; held-v3 untouched)',
                 '--note', 'evaluation-only-never-training: role held-eval-invented; training excludes every '
                           'held-eval* role (held_eval.training_exclusion_ids); ideas: ' + ', '.join(a['slug'] for a, _ in held))
        registry('promote', dataset, '--role', 'held-eval-invented', '--role-file', 'manifest.json',
                 *(['--no-check'] if no_check else []))
        summary['held'] = {'dataset': dataset, 'tasks': tasks, 'path': str(directory)}
    for admission, d in fresh:
        admission['published'] = stamp
        (d / 'admission.json').write_text(json.dumps(admission, indent=1) + '\n')
    with open(root / 'publications.jsonl', 'a') as handle:
        handle.write(json.dumps(summary) + '\n')
    if authored:
        summary['authored'] = authored_feed(root, no_check=no_check)
    return summary


def authored_feed(root: Path, *, no_check: bool = False) -> dict:
    """The admitted train programs' reference runs as authored trajectories (scripts/neuralese_data/
    authored_trajectories.py feed: teacher pool, registry role ei-authored-teacher, the fill WORK). A failure is
    reported, never undoes the publication."""
    run = subprocess.run([str(PYTHON), str(REPO / 'scripts/neuralese_data/authored_trajectories.py'), 'feed',
                          '--root', str(root), *(['--no-check'] if no_check else [])],
                         capture_output=True, text=True, cwd=REPO,
                         env={**os.environ, 'PYTHONPATH': f'{REPO}/training/neuralese:{REPO}'})
    if run.returncode:
        return {'error': (run.stderr or run.stdout)[-1200:]}
    try:
        return json.loads(run.stdout)
    except ValueError:
        return {'output': run.stdout[-1200:]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    c = sub.add_parser('check')
    c.add_argument('submission', type=Path)
    p = sub.add_parser('package')
    p.add_argument('submission', type=Path)
    p.add_argument('out', type=Path)
    q = sub.add_parser('qualify')
    q.add_argument('submission', type=Path)
    q.add_argument('--runtime', type=Path, default=REPO / 'ts-host')
    q.add_argument('--print-reference', action='store_true',
                   help="print each instance's reference run: the value it returns, the prose words its judgment calls "
                        'read, and for each judgment which children entry answered which item (also in '
                        'qualification/reference.txt)')
    dg = sub.add_parser('digest')
    dg.add_argument('submission', type=Path)
    ad = sub.add_parser('admit')
    ad.add_argument('submission', type=Path)
    ad.add_argument('--review', type=Path)
    ad.add_argument('--idea', type=Path)
    ad.add_argument('--root', type=Path, default=INVENTOR_ROOT)
    ad.add_argument('--runtime', type=Path, default=REPO / 'ts-host')
    ad.add_argument('--publish', action='store_true')
    ad.add_argument('--skip-qualify', action='store_true', help='use the existing qualification/report.json')
    pb = sub.add_parser('publish')
    pb.add_argument('--root', type=Path, default=INVENTOR_ROOT)
    pb.add_argument('--dry-run', action='store_true')
    pb.add_argument('--no-authored', action='store_true',
                    help='no authored-trajectory export after the batch (authored_trajectories.py feed: the admitted '
                         'train programs\' reference runs as an expert-iteration teacher tier)')
    pb.add_argument('--no-check', action='store_true',
                    help="promote without the registry's check-main --quick (when another strand's uncommitted change "
                         'turns it red; the promotion commits only registry files)')
    dr = sub.add_parser('dry-run', help="run the program's TypeScript with the scripted judgment answers, no model, in "
                                         'seconds: the reference against expected, which entry answered which call, the '
                                         'static checks; --controls also replays every wrong answer, ablation and baseline')
    dr.add_argument('submission', type=Path)
    dr.add_argument('--runtime', type=Path, default=REPO / 'ts-host')
    dr.add_argument('--controls', action='store_true')
    sr = sub.add_parser('sample-reviews', help='draw N admitted submissions at random for a review (REVIEW_BRIEF.md)')
    sr.add_argument('count', type=int)
    sr.add_argument('--root', type=Path, default=INVENTOR_ROOT)
    sr.add_argument('--seed')
    sr.add_argument('--reviewed', action='store_true', help='also draw submissions whose current version has a review')
    st = sub.add_parser('stages')
    st.add_argument('submission', type=Path)
    st.add_argument('--runtime', type=Path, default=REPO / 'ts-host')
    args = parser.parse_args(argv)
    if args.command == 'publish':
        print(json.dumps(publish(args.root, dry_run=args.dry_run, no_check=args.no_check,
                                 authored=not args.no_authored), indent=1))
        return 0
    if args.command == 'sample-reviews':
        print(json.dumps(sample_reviews(args.root, args.count, seed=args.seed, reviewed=args.reviewed), indent=1))
        return 0
    submission = args.submission.resolve()
    if args.command == 'digest':
        print(digest(submission))
        return 0
    if args.command == 'admit':
        result = admit(submission, review_path=args.review, idea_path=args.idea, root=args.root,
                       runtime=args.runtime.resolve(), skip_qualify=args.skip_qualify)
        if result['admitted'] and args.publish:
            result['publication'] = publish(args.root)
        print(json.dumps(result, ensure_ascii=False, indent=1))
        return 0 if result['admitted'] else 1
    if args.command == 'stages':
        out = submission / 'qualification'
        report = json.loads((out / 'report.json').read_text())
        if not report.get('ok'):
            raise SystemExit('stages: the submission is not qualified')
        cases = [json.loads(line) for line in (out / 'cases.jsonl').read_text().splitlines() if line.strip()]
        run = run_node('reference_trace.mjs', args.runtime.resolve(), str(out / 'cases.jsonl'), str(out / 'trace.json'))
        if run.returncode:
            raise SystemExit('reference_trace.mjs failed: ' + (run.stderr or run.stdout)[-600:])
        trace = json.loads((out / 'trace.json').read_text())
        cases = with_reference_calls(cases, trace)
        (out / 'cases.jsonl').write_text(''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in cases))
        report['reference_calls'] = {row['id']: row.get('calls') for row in trace.get('results', [])}
        report['stages'] = qualify_stages(out, args.runtime.resolve(), cases, trace)
        write_report(out, report)
        print(json.dumps({'reference_calls': report['reference_calls'], 'stages': report['stages']}, indent=1))
        return 0
    if args.command == 'dry-run':
        result = dry_run(submission, args.runtime.resolve(), controls=args.controls)
        print(result['text'])
        print(json.dumps({'ok': result['ok'], 'problems': result['problems'][:60], 'warnings': result['warnings']},
                         ensure_ascii=False, indent=1))
        return 0 if result['ok'] else 1
    if args.command == 'check':
        data, files = load(submission)
        m = metrics(data, files)
        problems = static_problems(submission, data, files, m)
        print(json.dumps({'ok': not problems, 'problems': problems, 'metrics': m}, indent=1))
        return 0 if not problems else 1
    if args.command == 'package':
        data, files = load(submission)
        cases = package(data, files)
        args.out.write_text(''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in cases))
        print(json.dumps({'cases': len(cases)}))
        return 0
    report = qualify(submission, args.runtime.resolve(), print_reference=args.print_reference)
    print(json.dumps({'ok': report['ok'], 'problems': report['problems'][:60], 'warnings': report.get('warnings', [])},
                     ensure_ascii=False, indent=1))
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
