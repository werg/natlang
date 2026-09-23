#!/usr/bin/env python3
"""Join execution-verified student successes and teacher corrections immutably."""
import argparse
import json
import os
from pathlib import Path
import tempfile

from create_improvement_round import validate_correction_turns
from run_training_pipeline import atomic_json, digest_file


def combine(student, teacher, output):
    student, teacher, output = map(Path, (student, teacher, output))
    rows, seen = [], set()
    for source in (student, teacher):
        for line in source.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get('id') in seen:
                raise ValueError(f'duplicate verified decision: {row.get("id")}')
            seen.add(row.get('id'))
            rows.append(row)
    data = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if output.read_text() != data:
            raise ValueError('refusing to overwrite changed verified turns')
        approved = validate_correction_turns(output)
    else:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf8', dir=output.parent,
                                         prefix=output.name + '.pending-', delete=False) as stream:
            pending = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            approved = validate_correction_turns(pending)
            pending.replace(output)
        finally:
            pending.unlink(missing_ok=True)
    manifest = {'version': 'natlang.verified_improvement_turns/1',
                'student_sha256': digest_file(student), 'teacher_sha256': digest_file(teacher),
                'output_sha256': digest_file(output), 'rows': len(rows),
                'approved_decisions': approved}
    manifest_path = Path(str(output) + '.manifest.json')
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('refusing to overwrite changed verified turns manifest')
    else:
        atomic_json(manifest_path, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--student', required=True, type=Path)
    parser.add_argument('--teacher', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(combine(args.student, args.teacher, args.output)))


if __name__ == '__main__':
    main()
