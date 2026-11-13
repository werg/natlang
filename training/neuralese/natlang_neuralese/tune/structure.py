"""Structural shape of a tune program: what makes program families actually differ for adaptation (review 2026-11-03).

Nominal family names are not structure: the 51-program supply has 18 names over one shape (a crisp helper plus an
ordered natural-language rule list returning a category). The program-adaptation stage's exit criterion counts
families whose programs differ in *structure*: operations (distinct root functions with examples), natural-language
functions, crisp helpers, the root result kind, iteration (``iterateOn``), host services, Neuralese values and whether
the decision is an ordered rule list. Read from the program's own files; an exact verifier of the stage's inputs, not a
semantic judgment of task difficulty.

    python -m natlang_neuralese.tune.structure PROGRAM_DIR [...]   prints each program's shape and its key
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from ..common.hashing import canonical_json_sha256_hex

SCHEMA = 'natlang.tune-program-structure/1'
RULE_ARROW = re.compile(r'->')
IN_ORDER = re.compile(r'\bin order\b', re.IGNORECASE)
OTHERWISE = re.compile(r'\botherwise\b', re.IGNORECASE)
# Features that separate shapes. A crisp helper's presence is recorded but does not make a new shape: "exact helper
# plus ordered rules returning a category" is one shape with or without the helper (review 2026-11-03).
SHAPE_FEATURES = ('operations', 'nl_functions', 'result', 'iterates', 'services', 'neuralese', 'ordered_rules')


def _frontmatter(text):
    if not text.startswith('---'):
        return {}, text
    _, head, body = text.split('---', 2)
    fields = {}
    for line in head.strip().splitlines():
        key, _, value = line.partition(':')
        fields[key.strip()] = value.strip()
    return fields, body.strip()


def _bucket(count, edges=(1, 3)):
    return '0' if count == 0 else '1' if count <= edges[0] else f'2-{edges[1]}' if count <= edges[1] else f'{edges[1] + 1}+'


def _result_kind(returns, types_text):
    returns = (returns or '').strip()
    if not returns:
        return 'unknown'
    if returns.startswith('Neuralese<'):
        return 'neuralese'
    if returns.endswith('[]') or returns.startswith('Array<'):
        return 'list'
    if returns.startswith('{'):
        return 'record'
    if returns in ('number', 'string', 'boolean'):
        return returns
    match = re.search(r'\btype\s+' + re.escape(returns) + r'\s*=\s*([^;]+);', types_text)
    if match:
        body = match.group(1).strip()
        if re.fullmatch(r"('[^']*'|\"[^\"]*\")(\s*\|\s*('[^']*'|\"[^\"]*\"))*", body):
            return 'category'
        return _result_kind(body, types_text)
    return 'named'


def _ordered_rules(body):
    """An ordered decision list: "Apply in order: a -> x; b -> y; ..." or "... in order: if ... otherwise if ..."."""
    arrows, otherwise = len(RULE_ARROW.findall(body)), len(OTHERWISE.findall(body))
    return arrows >= 3 or (bool(IN_ORDER.search(body)) and arrows + otherwise >= 2)


def program_structure(root):
    """``{schema, program, features, key}`` for one program directory."""
    root = Path(root)
    nl_files = sorted(path for path in root.rglob('*.nl') if '.natlang' not in path.parts and 'node_modules' not in path.parts)
    ts_files = sorted(path for path in root.rglob('*.ts') if '.natlang' not in path.parts and 'node_modules' not in path.parts
                      and path.name != 'types.ts' and not path.name.endswith('.d.ts'))
    types_text = '\n'.join(path.read_text() for path in root.rglob('types.ts') if '.natlang' not in path.parts)
    texts = {path: path.read_text() for path in nl_files}
    main = root / 'main.nl'
    fields, _ = _frontmatter(main.read_text()) if main.is_file() else ({}, '')
    operations = set()
    examples = root / 'examples.json'
    if examples.is_file():
        operations.update(example.get('function') for example in json.loads(examples.read_text()))
    everything = '\n'.join(texts.values()) + '\n' + '\n'.join(path.read_text() for path in ts_files)
    ordered_rules = any(_ordered_rules(_frontmatter(text)[1]) for text in texts.values())
    features = {
        'operations': _bucket(len(operations) or 1),
        'nl_functions': _bucket(len(nl_files)),
        'crisp_helpers': _bucket(len(ts_files), (1, 2)),
        'result': _result_kind(fields.get('returns'), types_text),
        'iterates': 'iterateOn' in everything,
        'services': (root / 'tune.services.mjs').is_file(),
        'neuralese': 'Neuralese<' in everything or 'Neuralese<' in types_text,
        'ordered_rules': ordered_rules,
    }
    return {'schema': SCHEMA, 'program': root.name, 'features': features,
            'key': canonical_json_sha256_hex({name: features[name] for name in SHAPE_FEATURES})[:16]}


def distinct_shapes(structures):
    return sorted({item['key'] for item in structures})


def main(argv=None):
    for path in (sys.argv[1:] if argv is None else argv):
        print(json.dumps(program_structure(path), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
