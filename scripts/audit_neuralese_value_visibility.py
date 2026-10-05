#!/usr/bin/env python3
"""Audit recurrent controls against canonical values and model-visible answer literals.

This is a diagnostic: visible text may be a legitimate input. Metadata, oracle targets
and read-source annotations are not counted as input. No admission is granted here.
"""
import argparse, collections, hashlib, json
from pathlib import Path
from audit_neuralese_recurrence import names


def write_sources(value):
    result = {}
    if isinstance(value, dict):
        if '$write' in value:
            site = value['$write']; result[site['name']] = site.get('source', '')
        else:
            for v in value.values(): result.update(write_sources(v))
    elif isinstance(value, list):
        for v in value: result.update(write_sources(v))
    elif isinstance(value, str):
        try: result.update(write_sources(json.loads(value)))
        except (ValueError, RecursionError): pass
    return result


def facts(value, path=()):
    if isinstance(value, dict):
        for k, v in value.items(): yield from facts(v, path + (k,))
    elif isinstance(value, list):
        for i, v in enumerate(value): yield from facts(v, path + (i,))
    elif isinstance(value, str) and path and path[-1] in {'quote', 'description'}:
        yield path, value


def argument_strings(value):
    if isinstance(value, dict):
        if '$write' not in value:
            for v in value.values(): yield from argument_strings(v)
    elif isinstance(value, list):
        for v in value: yield from argument_strings(v)
    elif isinstance(value, str): yield value


def visible_segments(row, pieces):
    for i, m in enumerate(row['messages']):
        content = m.get('content')
        if isinstance(content, str): yield f'message:{i}:{m["role"]}', 'text', content
        elif isinstance(content, list):
            for j, part in enumerate(content):
                if part['type'] == 'text': yield f'message:{i}:part:{j}', 'text', part['text']
                elif part['type'] == 'soft': yield f'message:{i}:part:{j}', 'soft-source', pieces[part['name']]
                elif part['type'] == 'digest': yield f'message:{i}:part:{j}', 'digest-preview', part['preview']
        for j, call in enumerate(m.get('tool_calls', [])):
            value = call['function']['arguments']
            if isinstance(value, str): value = json.loads(value)
            for text in argument_strings(value): yield f'message:{i}:call:{j}', 'tool-argument', text


def canonical(value):
    try: return json.dumps(json.loads(value), sort_keys=True, ensure_ascii=False)
    except ValueError: return value


def audit(rows, pieces, limit):
    sources = {}
    for row in rows:
        for name, value in write_sources(row.get('target')).items():
            if name in sources: raise ValueError('ambiguous producer ' + name)
            sources[name] = value
    held = [r for r in rows if r.get('split') == 'test'][:limit]
    readers = [r for r in held if names(r['messages'], 'read')]
    reports = []
    for i, row in enumerate(readers):
        read_names = names(row['messages'], 'read')
        donor = next((r for k in range(1, len(readers)) for r in [readers[(i+k) % len(readers)]]
                      if names(r['messages'], 'read') != read_names), None)
        own_values = collections.Counter(canonical(sources[n]) for n in read_names)
        donor_values = collections.Counter(canonical(sources[n]) for n in names(donor['messages'], 'read')) if donor else None
        literals = set()
        for name in read_names:
            try: value = json.loads(sources[name])
            except ValueError: continue
            literals.update((str(path[-1]), fact) for path, fact in facts(value))
        segments = list(visible_segments(row, pieces)); hits = []
        for field, literal in sorted(literals):
            if len(literal) < 8: continue
            for location, kind, text in segments:
                if literal in text or json.dumps(literal, ensure_ascii=False)[1:-1] in text:
                    hits.append({'field': field, 'literal_sha256': hashlib.sha256(literal.encode()).hexdigest(), 'location': location, 'kind': kind})
        reports.append({'id': row['id'], 'donor': donor['id'] if donor else None,
                        'same_canonical_value_multiset': own_values == donor_values, 'visible_literals': hits})
    return {'schema': 'natlang.recurrent-value-visibility-audit/1', 'held_records': len(held), 'readers': len(readers),
            'identical_donor_values': sum(r['same_canonical_value_multiset'] for r in reports),
            'readers_with_visible_quotes': sum(any(h['field'] == 'quote' for h in r['visible_literals']) for r in reports),
            'readers_with_visible_descriptions': sum(any(h['field'] == 'description' for h in r['visible_literals']) for r in reports),
            'results': reports, 'scope': 'literal visibility and canonical source comparison, not payload causal-use proof; legitimate catalog/input text is distinguished from masked reads'}


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--records', type=Path, required=True); p.add_argument('--pieces', type=Path, required=True); p.add_argument('--out', type=Path, required=True); p.add_argument('--eval', type=int, default=28); a = p.parse_args()
    if a.eval < 1 or a.out.exists(): raise ValueError('positive eval count and fresh report required')
    rows = [json.loads(line) for line in a.records.open()]; pieces = {r['name']: r['text'] for r in map(json.loads, a.pieces.open())}
    report = audit(rows, pieces, a.eval); report['inputs'] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in [a.records, a.pieces]}
    a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(report, indent=2)+'\n'); print(json.dumps({k:v for k,v in report.items() if k not in {'results','inputs'}}))
if __name__ == '__main__': main()
