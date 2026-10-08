#!/usr/bin/env python3
"""Report answers already present in a task's initial files; never mutate sources.

Matches require review: an unchanged file or an explicitly supplied draft can be
intentional. An exact final answer at the declared output path is stronger
evidence than a substring match. No admission decision is inferred here.
"""
import argparse
import hashlib
import json
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def parsed(text):
    try:
        return True, json.loads(text)
    except (ValueError, TypeError):
        return False, None


def same_content(left, right):
    a, x = parsed(left)
    b, y = parsed(right)
    # Python considers True == 1; canonical JSON preserves this distinction.
    if a and b:
        return json.dumps(x, sort_keys=True, ensure_ascii=False) == json.dumps(
            y, sort_keys=True, ensure_ascii=False)
    return left == right


def audit_record(record):
    semantics = record.get('semantics', {})
    files = semantics.get('folder_files', {})
    expected_files = semantics.get('expected_files', {})
    findings = []
    if not isinstance(files, dict):
        return findings
    declared_outputs = set()
    for text in files.values():
        ok, value = parsed(text)
        if ok and isinstance(value, dict) and isinstance(value.get('output_path'), str):
            declared_outputs.add(value['output_path'])
    expected = semantics.get('expected')
    # Empty placeholders, primitive booleans and tiny incidental strings are
    # not sufficient evidence of answer exposure.
    substantive = isinstance(expected, (dict, list)) and bool(expected)
    for name, text in files.items():
        if not isinstance(text, str):
            continue
        reasons = []
        ok, value = parsed(text)
        if (substantive and name in declared_outputs and ok
                and same_content(text, json.dumps(expected, ensure_ascii=False))):
            reasons.append('declared-output-initially-contains-exact-final-result')
        if (isinstance(expected_files, dict)
                and isinstance(expected_files.get(name), str)
                and same_content(text, expected_files[name])):
            reasons.append('initial-file-already-matches-expected-final-file')
        if reasons:
            findings.append({'path': name, 'initial_file_sha256': digest(text.encode()),
                             'reasons': reasons, 'disposition': 'requires-source-review'})
    return findings


def audit_file(path):
    raw = path.read_bytes()
    rows = []
    for index, line in enumerate(raw.splitlines()):
        if not line.strip():
            continue
        record = json.loads(line)
        findings = audit_record(record)
        if findings:
            rows.append({'physical_row_index': index, 'id': record.get('id'),
                         'row_sha256': digest(line), 'split': record.get('split'),
                         'source_groups': record.get('source_groups', []),
                         'findings': findings})
    return {'path': str(path), 'sha256': digest(raw), 'flagged_rows': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sources', nargs='+', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    report = {'schema': 'natlang.source-answer-exposure-audit/1',
              'admission_decision': 'none; findings require independent source review',
              'sources': [audit_file(path) for path in args.sources]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'sources': len(report['sources']), 'flagged_rows': sum(
        len(source['flagged_rows']) for source in report['sources'])}))


if __name__ == '__main__':
    main()
