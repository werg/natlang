#!/usr/bin/env python3
"""Neuralese-output spans of a direct union rendering: the sidecar the mixed direct+e2e run reads.

Design: plans/neuralese/NEURALESE_DIRECT_V1.md, "The mixed run" and "Neuralese outputs in direct samples" (owner
2026-10-13). A Neuralese output is a value a record's target writes (conversion /16 producers, ``target_writes``); its
text is the producer's handover source (``handover_notes``), the text the e2e writer is sized from. A direct row (one
document of the union's ``text-data.jsonl``) names its native records in ``source_record_ids``, the same ids the e2e
cohort uses, so the join is exact.

Output, one JSON row per direct row that has at least one span:
    {"id": row id, "spans": [[begin, end, write_name, has_edge], ...]}
``begin``/``end`` are positions in the row's own ``token_ids`` (end exclusive) covering the written value inside the
row's supervised part; ``has_edge`` says whether the e2e cohort has a consumer edge for that write. The mixed run
makes edged spans context-only in direct updates (they train through e2e) and reads the others through the
TokenChoiceProjection with h detached.

    neuralese_spans.py --rows <text-data.jsonl> --cohort <e2e cohort dir> --tokenizer <dir> --out <neuralese-spans.jsonl>
"""
from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'training' / 'neuralese'))


def value_forms(value: str) -> list[str]:
    """Equivalent JSON rendering spacings, raw or escaped inside a string argument."""
    forms = [value]
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        pass
    else:
        for ascii_only in (False, True):
            for separators in ((',', ':'), (', ', ': ')):
                forms.append(json.dumps(parsed, ensure_ascii=ascii_only, separators=separators))
    return list(dict.fromkeys(form for raw in forms
                             for form in (raw, json.dumps(raw, ensure_ascii=False)[1:-1])))


def token_offsets(token_ids, text, tokenizer):
    """Character (start, end) of each token in ``text``. Re-tokenising ``text`` must reproduce ``token_ids`` (then
    the tokenizer's own offset mapping is exact); otherwise each token's decoded piece is laid end to end, which is
    exact for byte-level vocabularies whose pieces decode to whole characters, and the caller checks the result."""
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if list(encoded['input_ids']) == list(token_ids):
        return [tuple(o) for o in encoded['offset_mapping']], text
    pieces = [tokenizer.decode([t]) for t in token_ids]
    offsets, at = [], 0
    for piece in pieces:
        offsets.append((at, at + len(piece)))
        at += len(piece)
    return offsets, ''.join(pieces)


def return_value_literals(text, start, value_path=('value',), tool='return_result'):
    """Literal value arguments in the native call rendering; None for another dialect."""
    # The native envelope separates prose from the call expression. Apostrophes
    # in assistant prose must not become unterminated code-string delimiters.
    if '<|tool_call_start|>' in text[start:]:
        values = []
        pattern = re.compile(re.escape('<|tool_call_start|>') + r'(.*?)' +
                             re.escape('<|tool_call_end|>'), re.S)
        for match in pattern.finditer(text, start):
            literals = return_value_literals(match.group(1), 0, value_path, tool)
            values.extend((match.start(1) + begin, match.start(1) + end, value)
                          for begin, end, value in literals or ())
        return values
    # Mellum renders real tool calls as JSON objects inside explicit tool tags.
    # Read member boundaries with the decoder, never substring-search status/code.
    if '<tool_call>' in text[start:]:
        values = []
        decoder = json.JSONDecoder()
        def members(begin):
            at = begin
            while at < len(text) and text[at].isspace():
                at += 1
            if text[at:at + 1] != '{':
                raise ValueError('tool arguments must be an object')
            at += 1
            result = {}
            while True:
                while at < len(text) and text[at].isspace():
                    at += 1
                if text[at:at + 1] == '}':
                    return result
                key, at = decoder.raw_decode(text, at)
                if not isinstance(key, str) or key in result:
                    raise ValueError('tool arguments need unique string keys')
                while at < len(text) and text[at].isspace():
                    at += 1
                if text[at:at + 1] != ':':
                    raise ValueError('missing member colon')
                at += 1
                while at < len(text) and text[at].isspace():
                    at += 1
                begin = at
                value, at = decoder.raw_decode(text, at)
                result[key] = (begin, at, value)
                while at < len(text) and text[at].isspace():
                    at += 1
                if text[at:at + 1] == '}':
                    return result
                if text[at:at + 1] != ',':
                    raise ValueError('missing member comma')
                at += 1
        recognized = False
        for match in re.finditer(re.escape('<tool_call>'), text[start:]):
            try:
                call = members(start + match.end())
                if call.get('name', (None, None, None))[2] != tool:
                    continue
                recognized = True
                arguments = call.get('arguments')
                if arguments is not None and isinstance(arguments[2], dict):
                    value = members(arguments[0]).get(value_path[0])
                    for part in value_path[1:]:
                        if value is None:
                            break
                        if isinstance(part, str) and isinstance(value[2], dict):
                            value = members(value[0]).get(part)
                        elif type(part) is int and isinstance(value[2], list) and 0 <= part < len(value[2]):
                            at = value[0] + 1
                            items = []
                            for _ in value[2]:
                                while text[at].isspace():
                                    at += 1
                                begin = at
                                item, at = decoder.raw_decode(text, at)
                                items.append((begin, at, item))
                                while text[at].isspace():
                                    at += 1
                                at += 1  # verified JSON comma or closing bracket
                            value = items[part]
                        else:
                            value = None
                    if value is not None:
                        values.append(value)
            except (ValueError, IndexError):
                continue
        return values if recognized else None
    candidates = {start + match.start(): match for match in re.finditer(r'\b' + re.escape(tool) + r'\s*\(', text[start:])}
    calls, quote, escaped = [], None, False
    for index in range(start, len(text)):
        char = text[index]
        if quote:
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == quote:
                quote = None
        elif char in ('"', "'"):
            quote = char
        elif index in candidates:
            calls.append(candidates[index])
    if not calls:
        return None
    values = []
    for match in calls:
        begin = start + match.start()
        opening = start + match.end() - 1
        depth, quote, escaped, end = 1, None, False, None
        for index in range(opening + 1, len(text)):
            char = text[index]
            if quote:
                if escaped:
                    escaped = False
                elif char == '\\':
                    escaped = True
                elif char == quote:
                    quote = None
            elif char in ('"', "'"):
                quote = char
            elif char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
                if depth == 0:
                    end = index + 1
                    break
        if end is None:
            continue
        rendered = text[begin:end]
        try:
            call = ast.parse(rendered, mode='eval').body
            node = next((kw.value for kw in call.keywords if kw.arg == value_path[0]),
                        call.args[0] if len(call.args) == 1 and value_path[0] == 'value' else None)
            for part in value_path[1:]:
                if isinstance(node, ast.Dict):
                    node = next((v for k, v in zip(node.keys, node.values)
                                 if isinstance(k, ast.Constant) and k.value == part), None)
                elif isinstance(node, (ast.List, ast.Tuple)) and type(part) is int and 0 <= part < len(node.elts):
                    node = node.elts[part]
                else:
                    node = None
            if node is None:
                continue
            class JsonConstants(ast.NodeTransformer):
                def visit_Name(self, item):
                    if item.id in ('true', 'false', 'null'):
                        return ast.copy_location(ast.Constant({'true': True, 'false': False, 'null': None}[item.id]), item)
                    return item
            literal = ast.literal_eval(JsonConstants().visit(node))
            lines = rendered.split('\n')
            def position(line, column):
                return begin + sum(len(s) + 1 for s in lines[:line - 1]) + len(lines[line - 1].encode()[:column].decode())
            left = position(node.lineno, node.col_offset)
            right = position(node.end_lineno, node.end_col_offset)
        except (SyntaxError, ValueError, TypeError, AttributeError, UnicodeDecodeError):
            continue
        values.append((left, right, literal))
    return values


def quoted_content_offsets(rendered, literal):
    """Raw source bounds of each decoded character in a JSON/Python quoted literal."""
    if len(rendered) < 2 or rendered[0] not in ('"', "'") or rendered[-1] != rendered[0]:
        return None
    try:
        json_mode = json.loads(rendered) == literal
    except (ValueError, TypeError):
        json_mode = False
    decoder = json.loads if json_mode else ast.literal_eval
    offsets, decoded, at = [], '', 1
    while at < len(rendered) - 1:
        begin = at
        if rendered[at] != '\\':
            piece = rendered[at]
            at += 1
        else:
            match = re.match(r'''\\(?:["'\\/abfnrtv]|x[0-9a-fA-F]{2}|u[0-9a-fA-F]{4}|U[0-9a-fA-F]{8})''', rendered[at:])
            if match is None:
                return None
            at += len(match.group())
            fragment = rendered[begin:at]
            # JSON combines a high/low surrogate pair into one Unicode scalar.
            if json_mode and fragment.startswith('\\u') and 0xD800 <= int(fragment[2:], 16) <= 0xDBFF:
                following = re.match(r'\\u[dD][c-fC-F][0-9a-fA-F]{2}', rendered[at:])
                if following is not None:
                    at += len(following.group())
                    fragment = rendered[begin:at]
            try:
                piece = decoder(rendered[0] + fragment + rendered[0])
            except (ValueError, SyntaxError):
                return None
        decoded += piece
        offsets.extend([(begin, at)] * len(piece))
    return offsets if decoded == literal else None


def spans_for_row(row: dict, writes: list[tuple[str, str, bool]], tokenizer) -> tuple[list, int]:
    """(spans, writes not found) of one direct row. ``writes``: (write name, value text, has edge) of the row's source
    records. A value is searched in the row's supervised part only (it also appears in later prompts as an input)."""
    token_ids = list(row['token_ids'])
    offsets, text = token_offsets(token_ids, row['text'], tokenizer)
    if text != row['text']:
        # A byte fragment decoded in isolation can become a replacement
        # character. Its invented character offsets cannot locate gold tokens.
        return [], len(writes)
    start_token = int(row.get('supervised_suffix_start') or 0)
    supervised_from = offsets[start_token][0] if start_token < len(offsets) else len(text)
    target_start = supervised_from + len(text[supervised_from:]) - len(text[supervised_from:].lstrip())
    if text.startswith('<think>', target_start):
        end_think = text.find('</think>', target_start + len('<think>'))
        if end_think < 0:
            return [], len(writes)
        supervised_from = end_think + len('</think>')
    spans, missing = [], 0
    literal_cache = {}
    for write in writes:
        name, value, has_edge = write[:3]
        site = write[3] if len(write) > 3 else {}
        if site.get('source_kind') == 'inline-code-sidecar' and site.get('source_span') is not None:
            path, tool = tuple(site['value_path']), site['tool']
            key = (tool, path)
            if key not in literal_cache:
                literal_cache[key] = return_value_literals(text, supervised_from, path, tool)
            found = None
            for begin, end, code in literal_cache[key] or ():
                if not isinstance(code, str) or code != site.get('code'):
                    continue
                left, right = site['source_span']
                if not 0 <= left < right <= len(code):
                    continue
                bounds = quoted_content_offsets(text[begin:end], code)
                if bounds is not None:
                    found = (begin + bounds[left][0], begin + bounds[right - 1][1])
                    break
            covered = [] if found is None else [i for i, (b, e) in enumerate(offsets) if b < found[1] and e > found[0]]
            if covered:
                spans.append([covered[0], covered[-1] + 1, name, bool(has_edge)])
            else:
                missing += 1
            continue
        path = tuple(site.get('value_path', ('value',)))
        tool = site.get('tool', 'return_result')
        key = (tool, path)
        if key not in literal_cache:
            literal_cache[key] = return_value_literals(text, supervised_from, path, tool)
        literals = literal_cache[key]
        if not isinstance(value, str) or not value:
            missing += 1
            continue
        found = None
        if literals is not None:
            try:
                expected = json.loads(value)
            except json.JSONDecodeError:
                expected = value
            for begin, end, literal in literals:
                if isinstance(literal, str):
                    if literal not in (value, expected):
                        continue
                    rendered = text[begin:end]
                    if len(rendered) >= 2 and rendered[0] in ('"', "'") and rendered[-1] == rendered[0]:
                        found = (begin + 1, end - 1)
                        break
                elif json.dumps(literal, sort_keys=True, ensure_ascii=False) == json.dumps(expected, sort_keys=True, ensure_ascii=False):
                    found = (begin, end)
                    break
        else:
            for form in value_forms(value):
                at = text.find(form, supervised_from)
                if at >= 0:
                    found = (at, at + len(form))
                    break
        if found is None:
            missing += 1
            continue
        covered = [i for i, (b, e) in enumerate(offsets) if b < found[1] and e > found[0]]
        if not covered:
            missing += 1
            continue
        spans.append([covered[0], covered[-1] + 1, name, bool(has_edge)])
    return sorted(spans), missing


def cohort_writes(cohort: Path) -> dict[str, list[tuple[str, str, bool]]]:
    """Record id -> (write name, value text, has edge) of every record in the cohort whose target writes."""
    from natlang_neuralese.train import trajectories as T
    from natlang_neuralese.common import row_index
    from natlang_neuralese.common.hashing import sha256_file_hex, sha256_file_hex_cached
    # Reuse the shared exact-input cache. The code inventory covers write_sites
    # and its transitive helpers, not only this caller, so a changed extractor
    # cannot reuse a previous meaning of the same source records.
    package = ROOT / 'training/neuralese/natlang_neuralese'
    code = hashlib.sha256('\n'.join(str(p.relative_to(package)) + ':' + sha256_file_hex(p)
                                   for p in sorted(package.rglob('*.py'))).encode()).hexdigest()
    cache = row_index.entry_path('neuralese-cohort-writes-v2', code, sha256_file_hex(Path(__file__)),
                                 sha256_file_hex_cached(cohort / 'records.jsonl'),
                                 sha256_file_hex_cached(cohort / 'edges.jsonl'))
    stored = row_index.load(cache)
    if stored is not None and isinstance(stored.get('writers'), dict):
        return stored['writers']
    edged = collections.defaultdict(set)
    for line in open(cohort / 'edges.jsonl'):
        edge = json.loads(line)
        edged[edge['writer_record_id']].add(edge['write_name'])
    writers = {}
    for line in open(cohort / 'records.jsonl'):
        record = json.loads(line)
        names = T.target_writes(record)
        if names:
            notes = T.handover_notes(record)
            sites = {site.name: site for site in T.extract_write_sites(record)}
            writers[record['id']] = []
            for n in sorted(names):
                value = (n, notes.get(n), n in edged.get(record['id'], ()))
                if n in sites:
                    site = sites[n]
                    value += ({'tool': site.tool, 'value_path': list(site.value_path),
                               'source_kind': site.source_kind},)
                    if site.source_kind == 'inline-code-sidecar':
                        value[-1].update(source_span=list(site.source_span), code=site.arguments['code'])
                writers[record['id']].append(value)
    row_index.save(cache, {'writers': writers})
    return writers


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--rows', type=Path, required=True)
    parser.add_argument('--cohort', type=Path, action='append', required=True, help='e2e cohort dir (repeatable)')
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--out', type=Path, required=True)
    a = parser.parse_args(argv)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(a.tokenizer)
    writers = {}
    for cohort in a.cohort:
        for rid, writes in cohort_writes(cohort).items():
            writers.setdefault(rid, writes)
    stats = collections.Counter()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    partial = a.out.with_name(a.out.name + '.partial')
    from natlang_neuralese.common.jsonio import open_jsonl_text
    with open_jsonl_text(a.rows) as rows, open(partial, 'w') as out:
        for line in rows:
            row = json.loads(line)
            stats['rows'] += 1
            writes = [w for rid in (row.get('source_record_ids') or [row.get('id')]) for w in writers.get(rid, ())]
            if not writes:
                continue
            spans, missing = spans_for_row(row, writes, tokenizer)
            stats['writer_rows'] += 1
            stats['writes'] += len(writes)
            stats['writes_not_found'] += missing
            stats['spans'] += len(spans)
            stats['span_tokens'] += sum(e - b for b, e, _, _ in spans)
            stats['span_tokens_with_edge'] += sum(e - b for b, e, _, edge in spans if edge)
            if spans:
                out.write(json.dumps({'id': row['id'],
                                      'document_sha256': hashlib.sha256(row['text'].encode()).hexdigest(),
                                      'token_count': len(row['token_ids']), 'spans': spans}) + '\n')
    partial.replace(a.out)
    receipt = {'schema': 'natlang.neuralese-output-spans/1', 'rows': str(a.rows), 'cohorts': [str(c) for c in a.cohort],
               'tokenizer': a.tokenizer, 'out': str(a.out),
               'out_sha256': hashlib.sha256(a.out.read_bytes()).hexdigest(), **stats}
    a.out.with_name(a.out.stem + '.receipt.json').write_text(json.dumps(receipt, indent=1) + '\n')
    print(json.dumps(receipt), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
