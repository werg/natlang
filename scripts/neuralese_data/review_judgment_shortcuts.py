#!/usr/bin/env python3
"""Confirm keyword/regex substitutes for semantic judgment in supervised TRAIN targets.

Generalizes the 2026-11-02 semantic-ticket review to every family. A candidate is a
supervised eval block of a typed natural-language judgment invocation (a call whose
declared result is a boolean, a closed label set or a decision/label record) that
contains a crisp text test. The TypeScript parser (``judgment_code_check.mjs``)
supplies structural evidence; this policy holds a candidate only when that test is a
boolean keyword/regex test over prose words. Extraction (``exec``/``match``/
``replace``/``split``), field-label lookups (``Status:``), numeric/date formats and
format tokens are exact parsing and stay clear. A candidate whose code does not parse
is ``review``, never held automatically. This is a quality screen, not a semantic
oracle: holds are reviewed exact rows, and a spot-checked precision is recorded by
the caller.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

CHECKER = Path(__file__).with_name('judgment_code_check.mjs')
TYPESCRIPT = Path(__file__).resolve().parents[2] / 'ts-host/node_modules/typescript/lib/typescript.js'
JUDGMENT_FIELDS = re.compile(
    r'\b(?:decision|label|category|intent|verdict|relevant|relevance|sentiment|priority|urgency|severity|'
    r'class|classification|route|routing|action|outcome|status|risk|harm|match|same|eligible|approved|keep)\b\s*\??\s*:', re.I)
FORMAT_WORDS = {'true', 'false', 'yes', 'null', 'none', 'undefined', 'nan', 'http', 'https', 'www', 'json', 'csv',
                'txt', 'md', 'nl', 'utf', 'eof', 'todo'}
ESCAPES = re.compile(r'\\[dDsSwWbBnrtfv]|\\u[0-9a-fA-F]{4}|\\x[0-9a-fA-F]{2}|\\.')
WORD = re.compile(r'[A-Za-z]{3,}')


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def judgment_signature(signature: str) -> bool:
    """A typed natural-language call whose result is a judgment, not an extraction."""
    if not re.match(r'(?:nl@|nl\b|nl<)', signature or ''):
        return False
    tail = signature.rsplit(')', 1)[-1]
    result = tail.split(':', 1)[1].strip() if ':' in tail else ''
    if not result:
        return False
    if re.fullmatch(r'boolean|Promise<boolean>', result):
        return True
    if re.fullmatch(r'(?:"[^"]*"\s*\|\s*)+"[^"]*"', result) or re.fullmatch(r"(?:'[^']*'\s*\|\s*)+'[^']*'", result):
        return True
    return bool(result.startswith('{') and JUDGMENT_FIELDS.search(result))


def prose_words(pattern: str) -> list[str]:
    stripped = ESCAPES.sub(' ', pattern or '')
    return [word for word in WORD.findall(stripped) if word.lower() not in FORMAT_WORDS]


def shortcut(item: dict) -> bool:
    """A boolean test of prose keywords (semantic substitute), as opposed to exact parsing."""
    if item.get('use') != 'test':
        return False
    if re.match(r'\s*(?:\w+\.)*(?:options|choices|labels|categories|allowed\w*|valid\w*|names|keys)\s*\.', item.get('code') or '', re.I):
        return False  # membership in a declared option/label list, not a test of the prose

    pattern = item.get('pattern') or ''
    if re.search(r'[A-Za-z_]\w*\s*:', pattern) or pattern.rstrip().endswith(':'):
        return False  # a field label lookup in structured text
    if re.fullmatch(r'[\s.\-_/\\]*\w{1,5}', pattern) and pattern.startswith('.'):
        return False  # a file extension
    return bool(prose_words(pattern))


class JudgmentCodeChecker:
    """One bounded serial Node parser for a complete review."""

    def __enter__(self):
        self.dependencies = [{'path': str(path.resolve()), 'sha256': digest(path)}
                             for path in (Path(__file__), CHECKER, TYPESCRIPT)]
        self.process = subprocess.Popen(['node', str(CHECKER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        return self

    def __call__(self, code: str) -> dict:
        self.process.stdin.write(json.dumps({'code': code}) + '\n')
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError('judgment code checker exited without a result')
        return json.loads(line)

    def __exit__(self, *_):
        self.process.stdin.close()
        status = self.process.wait(timeout=30)
        self.process.stdout.close()
        if status:
            raise RuntimeError(f'judgment code checker failed: {status}')
        if any(digest(Path(ref['path'])) != ref['sha256'] for ref in self.dependencies):
            raise RuntimeError('judgment reviewer dependencies changed during review')


def review(candidates, checker):
    """``candidates``: dicts with ``signature`` and ``codes``; returns a disposition per candidate."""
    for candidate in candidates:
        if not judgment_signature(candidate['signature']):
            yield {**candidate, 'disposition': 'clear', 'evidence': [], 'parse_errors': 0,
                   'reason': 'not a typed natural-language judgment invocation'}
            continue
        evidence, errors = [], 0
        for code in candidate['codes']:
            inspected = checker(code)
            errors += inspected['parse_errors']
            evidence.extend(item for item in inspected['evidence'] if shortcut(item))
        if evidence and not errors:
            disposition, reason = 'hold', 'semantic judgment replaced by a boolean keyword/regex test in supervised eval code'
        elif evidence:
            disposition, reason = 'review', 'keyword/regex test found but the supervised code does not parse cleanly'
        else:
            disposition, reason = 'clear', 'only exact parsing or no crisp prose test in parsed eval code'
        yield {**candidate, 'disposition': disposition, 'evidence': evidence, 'parse_errors': errors, 'reason': reason}
