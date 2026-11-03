"""Reference notes for the child calls that produce Neuralese (plans/neuralese/NEURALESE_SPAN_BASIS_AUDIT_20261103.md, S1).

A reader or composed RL case declares helper functions whose result is ``Neuralese<...>`` (a private note). Its
oracle grades only the root's final answer, so without a reference for those helpers a producer child gets no
JOLT+ privileged context and no target note. The upstream datasets hold the intermediates (per-hop answers, supporting
sentences, evidence triples, gold cells); this module turns them into ``curriculum.reference.children`` entries
``{match, value, provenance}`` in the shape ``rl/privileged_context.case_facts`` consumes:

* ``match``: strings of the child's prompt text, all of which must occur in it: the call's signature opening
  (``You are inside this call: gather(``) and the fragments of its own inputs that name which call site it is
  (the question, the document, the part). ``rl/privileged_context.child_matches`` applies the same rule at run time.
* ``value``: the gold note, in the child's declared type (a string, or the typed record/list).
* ``provenance`` (one implementation of the vocabulary in ``natlang_neuralese.data.operator_seeds``):
  ``gold`` the dataset's own annotation verbatim; ``checked`` derived from annotations by code that verified the
  value against the inputs (a supporting sentence located by its gold answer, cells located by derivation numbers);
  ``mechanical:<method>`` extracted from the child's own inputs by the shared deterministic seed methods, ranked
  by the question and the gold answer, where the dataset has no intermediates.

Only a child whose note is known is emitted. An unknown one is left out (never invented), and ``coverage`` reports it.
"""
from __future__ import annotations

import json
import re
from collections import Counter

from natlang_neuralese.data import operator_seeds as seeds

CALL = 'You are inside this call: '
NOTE_CHARS = 1600
ORDER = ('gold', 'checked')


def rendered(text):
    """``text`` as the runtime prints a string value in a call's opening (JSON string escapes, no outer quotes)."""
    return json.dumps(text, ensure_ascii=False)[1:-1]


def fragment(text, limit=60, minimum=6):
    """A prefix of ``text``, at most ``limit`` characters cut at a word boundary, written as the runtime renders a string
    value (escaped quotes, newlines), so that it occurs verbatim in the child's prompt; None when it is too short to
    identify anything."""
    text = str(text).lstrip()
    if len(text) > limit:
        cut = text.rfind(' ', limit // 2, limit)
        text = text[:cut if cut > 0 else limit]
    text = text.strip()
    return rendered(text) if len(text) >= minimum else None


def document_key(title):
    """The rendering of a ``{ title, text }`` argument's opening: its title pins which document the call reads."""
    return '{ title: "' + rendered(title) + '", text: "'


def string_key(name, value):
    """The rendering of a string argument ``name`` holding ``value``."""
    return f'{name}: string = "{rendered(value)}"'


def child(function, keys, value, provenance):
    """One reference child entry, or None when a needed key fragment cannot be identified safely."""
    keys = list(keys)
    if any(key is None for key in keys) or value is None:
        return None
    if provenance not in ORDER and not provenance.startswith('mechanical:'):
        raise ValueError(f'unknown note provenance {provenance}')
    return {'match': [f'{CALL}{function}(', *keys], 'value': value, 'provenance': provenance}


def weakest(*provenances):
    """The least authoritative of several provenances (gold > checked > mechanical)."""
    ranked = [p for p in provenances if p]
    return next((p for p in ranked if p.startswith('mechanical:')), 'checked' if 'checked' in ranked else 'gold')


def clip(text, limit=NOTE_CHARS):
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text.rfind('\n', 0, limit)
    return text[:cut if cut > limit // 2 else limit].rstrip()


def sentence_at(text, offset):
    """The sentence of ``text`` that contains character ``offset``."""
    start = 0
    for found in seeds._SENTENCE.finditer(text):
        if found.start() > offset:
            return text[start:found.start()].strip()
        start = found.end()
    return text[start:].strip()


def answer_sentences(text, answers, limit=500):
    return seeds.answer_lines(text, answers, limit)


def overlap(text, reference, limit=NOTE_CHARS):
    """Mechanical: the sentences of ``text`` overlapping ``reference`` most, in source order."""
    return seeds.overlap_sentences(text, reference, limit)


def add(out, entry):
    if entry is not None:
        out.append(entry)


def attach(case, children):
    """Set a training case's reference children (a held case stays exactly as built)."""
    if case['split'] != 'train':
        return case
    children = [c for c in children if c is not None]
    case['curriculum']['reference']['children'] = children
    return case


def coverage(cases):
    """Per producer call site: how many cases declare it and which provenance its note carries.

    A call site is a helper the case's structure (or neuralese framing) declares as a Neuralese producer; it is
    covered by a note when some reference child matches its signature opening. Returns counters keyed
    ``covered|uncovered`` and, for the covered ones, the weakest provenance among that helper's entries.
    """
    by_status, by_provenance, total = Counter(), Counter(), 0
    for case in cases:
        sites = declared_sites(case)
        notes = {}
        for entry in (case['curriculum'].get('reference') or {}).get('children') or []:
            match = entry.get('match')
            match = [match] if isinstance(match, str) else match or []
            found = re.match(re.escape(CALL) + r'([\w$]+)\(', match[0]) if match else None
            if found:
                notes.setdefault(found.group(1), []).append(entry.get('provenance') or 'unlabelled')
        for site in sites:
            total += 1
            if site in notes:
                by_status['covered'] += 1
                by_provenance[weakest(*notes[site]).split(':')[0]] += 1
            else:
                by_status['uncovered'] += 1
    return {'sites': total, **by_status, 'provenance': dict(by_provenance)}


def declared_sites(case):
    """Helper names whose declared result is Neuralese<...>, read from the case's callable-folder files."""
    sites = set()
    for path, text in ((case.get('semantics') or {}).get('files') or {}).items():
        head = text.split('---')[1] if text.startswith('---') else ''
        returns = re.search(r'^returns:\s*(.*)$', head, re.M)
        if returns and 'Neuralese<' in returns.group(1) and path != (case['semantics'].get('root')):
            sites.add(path.rsplit('/', 1)[-1].removesuffix('.nl'))
    return sorted(sites)
