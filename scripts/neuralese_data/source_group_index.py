#!/usr/bin/env python3
"""Pin source groups already present in admitted rendered training data.

Generation selectors can use this compact index to avoid paying a teacher for
another rendering of an existing source world. The index includes train and
held groups; it does not change corpus admission or splits.
"""
import argparse
import hashlib
import json
from pathlib import Path


def build(inputs, output):
    if output.exists() or output.with_suffix(output.suffix + '.manifest.json').exists():
        raise ValueError('source-group index already exists')
    groups = set()
    sources = []
    for path in inputs:
        digest = hashlib.sha256()
        rows = 0
        with path.open('rb') as stream:
            for line in stream:
                digest.update(line)
                if not line.strip():
                    continue
                row = json.loads(line)
                groups.update(row.get('source_groups') or ())
                rows += 1
        sources.append({'path': str(path.resolve()), 'sha256': digest.hexdigest(), 'rows': rows})
    output.parent.mkdir(parents=True, exist_ok=True)
    data = ''.join(json.dumps(group, ensure_ascii=False) + '\n' for group in sorted(groups)).encode()
    output.write_bytes(data)
    manifest = {'schema': 'natlang.training-source-group-index/1',
                'inputs': sources, 'groups': len(groups),
                'index_path': str(output.resolve()),
                'index_sha256': hashlib.sha256(data).hexdigest(),
                'purpose': 'exclude previously trained or held source worlds from new teacher queues'}
    output.with_suffix(output.suffix + '.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, action='append', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.input, args.out), sort_keys=True))


if __name__ == '__main__':
    main()
