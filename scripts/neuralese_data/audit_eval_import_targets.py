#!/usr/bin/env python3
"""Streaming audit of supervised `eval` action targets in native-graph sources.

Finds eval code that starts with `import`, contains Python-only syntax, or opens with prose instead of
TypeScript. Verdicts come from the real scope compiler (`eval_code_shape_check.mjs`), so a flagged action is one
the runtime rejects, not a lookalike. Read-only: sources are streamed one row at a time, and rows that cannot
contain an eval header are skipped on their raw bytes without JSON decoding.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'training' / 'neuralese'))
from natlang_neuralese.common.jsonio import open_jsonl_text  # noqa: E402

# `<function=` `eval` `>\n` `<parameter=` `code` `>\n` in the LFM3 tokenizer; `</parameter>` closes the code.
FN_OPEN, EVAL, NL_GT, PARAM_OPEN, CODE, PARAM_CLOSE = 125021, 95890, 703, 125023, 3454, 125024
CALL_OPEN = re.compile(r'[\[,]125021,')
HEADER = (FN_OPEN, EVAL, NL_GT, PARAM_OPEN, CODE, NL_GT)
RAW_HEADER = re.compile(r'125021,\s*95890,\s*703,\s*125023,\s*3454,\s*703')

_KEYWORDS = {'declare', 'const', 'let', 'var', 'function', 'return', 'await', 'if', 'else', 'for', 'while', 'try', 'catch',
             'throw', 'new', 'import', 'export', 'class', 'type', 'interface', 'async', 'yield', 'do', 'switch',
             'case', 'default', 'break', 'continue', 'typeof', 'delete', 'void', 'in', 'of'}
PY_IMPORT = re.compile(r'^\s*(?:import\s+[\w.]+(?:\s+as\s+\w+)?(?:\s*,\s*[\w.]+(?:\s+as\s+\w+)?)*|'
                       r'from\s+[\w.]+\s+import\s+[^\'"]+)\s*$')
JS_IMPORT = re.compile(r'^\s*import\b[^\n]*[\'"]')
IMPORT_START = re.compile(r'^\s*import\b')
PY_MARKERS = (
    ('py-def', re.compile(r'^\s*def\s+\w+\s*\(.*\)\s*(?:->[^:]*)?:\s*$', re.M)),
    ('py-block', re.compile(r'^\s*(?:if|elif|else|for|while|with|try|except|finally|class)\b[^{;\n]*:\s*(?:#.*)?$', re.M)),
    ('py-import', re.compile(r'^\s*(?:from\s+[\w.]+\s+import\s+\w|import\s+(?:os|sys|re|json|math|pathlib|subprocess|shutil|'
                             r'collections|itertools|typing|datetime|csv|glob|random|time)\b)', re.M)),
    ('py-print', re.compile(r'(?<![\w.])print\s*\(')),
    ('py-constant', re.compile(r'(?<![\w.\'"])(?:True|False|None)\b(?![\'"])')),
    ('py-hash-comment', re.compile(r'^\s*#(?!!)', re.M)),
    ('py-fstring', re.compile(r'(?<![\w$])f"[^"\n]*\{[^}\n]*\}')),
    ('py-with-open', re.compile(r'\bwith\s+open\s*\(')),
    ('py-len-range', re.compile(r'\bfor\s+\w+\s+in\s+(?:range|enumerate)\s*\(')),
)
PROSE = re.compile(r"^[A-Za-z][A-Za-z'’,;:.!?-]*(?:\s+[A-Za-z'’,;:.!?-]+){3,}\s*$")


def eval_spans(ids):
    """(code_start, code_stop) token spans of every eval code argument in one node."""
    spans, position, size = [], 0, len(ids)
    while True:
        try:
            position = ids.index(FN_OPEN, position)
        except ValueError:
            return spans
        if tuple(ids[position:position + 6]) == HEADER:
            start = position + 6
            try:
                stop = ids.index(PARAM_CLOSE, start)
            except ValueError:
                stop = size
            spans.append((start, stop))
            position = stop
        else:
            position += 1


def first_line(code):
    for line in code.split('\n'):
        if line.strip():
            return line
    return ''


def classify(code):
    """Cheap lexical flags; the compiler verdict decides whether a flagged action can run at all."""
    head = first_line(code)
    flags = []
    if PY_IMPORT.match(head):
        flags.append('start-python-import')
    elif JS_IMPORT.match(head):
        flags.append('start-js-import')
    elif IMPORT_START.match(head):
        flags.append('start-import-other')
    for name, pattern in PY_MARKERS:
        if pattern.search(code):
            flags.append(name)
    words = head.strip().split()
    if PROSE.match(head.strip()) and words[0].rstrip(':,') not in _KEYWORDS:
        flags.append('start-prose')
    return flags


class Compiler:
    """One serial Node process; verdicts of repeated code are cached."""
    def __init__(self):
        self.process = subprocess.Popen(['node', str(ROOT / 'scripts/neuralese_data/eval_code_shape_check.mjs')],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        self.cache = {}

    def __call__(self, code):
        key = hashlib.sha1(code.encode()).digest()
        if key not in self.cache:
            if len(self.cache) > 200_000:
                self.cache.clear()
            self.process.stdin.write(json.dumps({'code': code}) + '\n')
            self.process.stdin.flush()
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError('eval code checker exited')
            self.cache[key] = json.loads(line)
        return self.cache[key]

    def close(self):
        self.process.stdin.close()
        self.process.wait(timeout=30)


IM_START = 124899
FOREIGN = {'start-python-import', 'start-prose', 'py-def', 'py-block', 'py-print', 'py-import', 'py-with-open', 'py-len-range'}
_CALL = re.compile(r'<function=([\w.-]+)>\s*(?:<parameter=(\w+)>\n)?([\s\S]*?)(?:</parameter>|</function>|$)')
_COMPILE_ERROR = re.compile(r'^\s*\d+:\d+ [a-z-]+:')
_ERROR_HEAD = re.compile(r'^\s*(?:\w*Error\b|eval (?:failed|repair)|Error\b)')


def _shape(text):
    return re.sub(r'\d+', '#', text.strip().split('\n', 1)[0])[:60]


def census_node(ids, weights, tokenizer, totals, examples, row_id, pairs=None):
    """Message-level census of one node: tool-call names with import-first code, and what follows a failed eval."""
    starts = [i for i, token in enumerate(ids) if token == IM_START]
    messages = []
    for number, begin in enumerate(starts):
        end = starts[number + 1] if number + 1 < len(starts) else len(ids)
        role = tokenizer.decode(ids[begin + 1:begin + 2]) if begin + 1 < end else ''
        supervised = any(w > 0 for w in weights[begin:end])
        stop = min(end, begin + 2 + 400) if role == 'tool' else end
        text = tokenizer.decode([i for i in ids[begin + 2:stop] if i < len(tokenizer)]) if role in ('assistant', 'tool') else ''
        messages.append((role, supervised, text, begin, end))
    pending = None  # (code_hash, supervised) of the last eval whose result has not been classified
    failure = None  # (kind, code_hash, supervised) of the eval whose failing result is awaiting the next action
    for role, supervised, text, begin, end in messages:
        if role == 'assistant':
            calls = [m for m in _CALL.finditer(text)]
            for m in calls:
                name, param, body = m.group(1), m.group(2), m.group(3)
                kind = 'supervised' if supervised else 'context'
                totals[f'tool:{name}:{kind}'] += 1
                if param in ('code', 'command', 'script') and name != 'eval':
                    head = first_line(body)
                    if PY_IMPORT.match(head) or re.match(r'^\s*(?:import|from)\s', head):
                        totals[f'tool_import_start:{name}:{kind}'] += 1
            if failure is not None:
                bad_kind, bad_hash, bad_supervised, bad_span = failure
                failure = None
                eval_calls = [m for m in calls if m.group(1) == 'eval']
                if not calls:
                    outcome = 'reply-only'
                elif eval_calls:
                    new_hash = hashlib.sha1(eval_calls[0].group(3).encode()).digest()
                    outcome = 'same-eval' if new_hash == bad_hash else 'different-eval'
                else:
                    outcome = 'other-tool:' + calls[0].group(1)
                key = f'recovery:{bad_kind}:{outcome}:{"supervised" if supervised else "context"}'
                totals[key] += 1
                if pairs is not None:
                    pairs.append((key, bad_span, (begin, end)))
                if len(examples.setdefault(key, [])) < 5:
                    examples[key].append(row_id)
            eval_calls = [m for m in calls if m.group(1) == 'eval']
            pending = (hashlib.sha1(eval_calls[-1].group(3).encode()).digest(), supervised,
                       bool(FOREIGN & set(classify(eval_calls[-1].group(3)))), (begin, end)) if eval_calls else None
        elif role == 'tool' and pending is not None:
            body = text.split('<tool_response>', 1)[-1].lstrip('\n')
            totals['resp_shape:' + _shape(body)] += 1
            kind = ('compile' if _COMPILE_ERROR.match(body) else 'runtime-error' if _ERROR_HEAD.match(body) else
                    'repeat-warning' if 'failed the same way' in body[:600] else None)
            if kind:
                totals[f'eval_failure:{kind}{"+foreign" if pending[2] else ""}:{"supervised" if pending[1] else "context"}'] += 1
                failure = (kind + ('+foreign' if pending[2] else ''), pending[0], pending[1], pending[3])
            pending = None


def audit_leaf(path, tokenizer, compiler, record, limit=None, census=None, shard=(0, 1)):
    """Stream one source file. `record(event)` receives each flagged occurrence; returns the aggregate counters."""
    totals = Counter()
    with open_jsonl_text(path) as stream:
        for number, line in enumerate(stream):
            if number % shard[1] != shard[0]:
                continue
            totals['rows'] += 1
            if limit and totals['rows'] > limit:
                break
            if (CALL_OPEN if census is not None else RAW_HEADER).search(line) is None:
                continue
            row = json.loads(line)
            row_has_eval = False
            for node_index, node in enumerate(row['nodes']):
                ids, weights = node['token_ids'], node['target_weights']
                if census is not None:
                    census_node(ids, weights, tokenizer, totals, census, row['id'])
                for start, stop in eval_spans(ids):
                    code = tokenizer.decode([i for i in ids[start:stop] if i < len(tokenizer)])
                    supervised = sum(1 for w in weights[start:stop] if w > 0)
                    kind = 'supervised' if supervised else 'context'
                    totals['eval_' + kind] += 1
                    row_has_eval = True
                    flags = classify(code)
                    if flags:
                        verdict = compiler(code)
                        syntax_fail = any(d['code'] == 'typescript-syntax' for d in verdict['diagnostics'])
                        totals[f'flagged_{kind}'] += 1
                        totals[f'flagged_syntax_fail_{kind}'] += syntax_fail
                        for flag in flags:
                            totals[f'flag:{flag}:{kind}'] += 1
                        record({'id': row['id'], 'node': node_index, 'node_name': node.get('name'), 'start': start,
                                'stop': stop, 'kind': kind, 'supervised_tokens': supervised, 'tokens': stop - start,
                                'flags': flags, 'compile_ok': verdict['ok'],
                                'syntax_fail': syntax_fail, 'diagnostics': verdict['diagnostics'], 'code': code[:1200]})
            totals['rows_with_eval'] += row_has_eval
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--leaf', type=Path, action='append', required=True)
    parser.add_argument('--out', type=Path, required=True, help='JSONL of flagged occurrences')
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--shard', default='0/1', help='K/N: handle rows whose index is K modulo N')
    parser.add_argument('--census', action='store_true', help='also count tool calls and what follows failed evals')
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    compiler = Compiler()
    summary = {}
    try:
        with args.out.open('x') as sink:
            for leaf in args.leaf:
                before = leaf.stat()
                def record(event, leaf=leaf):
                    event['leaf'] = str(leaf)
                    sink.write(json.dumps(event, ensure_ascii=False) + '\n')
                examples = {} if args.census else None
                totals = audit_leaf(leaf, tokenizer, compiler, record, args.limit, examples, tuple(map(int, args.shard.split('/'))))
                totals = Counter({k: v for k, v in totals.items() if not k.startswith('resp_shape:') or v >= 20})
                if examples:
                    totals['examples'] = examples
                after = leaf.stat()
                if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
                    raise ValueError(f'source changed while auditing: {leaf}')
                summary[str(leaf)] = dict(totals)
                sink.flush()
                args.summary.write_text(json.dumps(summary, indent=1, sort_keys=True))
                print(json.dumps({'event': 'leaf_done', 'leaf': str(leaf), **totals}), flush=True)
    finally:
        compiler.close()
    args.summary.write_text(json.dumps(summary, indent=1, sort_keys=True))


if __name__ == '__main__':
    main()
