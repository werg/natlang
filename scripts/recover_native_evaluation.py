#!/usr/bin/env python3
"""Recover a held-out evaluation export from durable completed native jobs.

Partial checkpoints remain missing in the score. This never calls a model.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def recover(directory, runtime):
    directory = Path(directory).resolve()
    runtime = Path(runtime).resolve()
    planned = [json.loads(line) for line in (directory / 'heldout.ir.jsonl').read_text().splitlines() if line.strip()]
    ids = [row['id'] for row in planned]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate planned evaluation IDs')
    rows, sources = {}, []
    for path in sorted((directory / 'heldout.jobs').glob('*.result.json')):
        data = path.read_bytes()
        row = json.loads(data)
        program = row.get('task', {}).get('program_ir', {})
        key = program.get('id')
        if key not in ids or key in rows:
            raise ValueError(f'unplanned or duplicate result: {path}')
        if program.get('split') != 'test' or row.get('provenance', {}).get('collection_role') != 'student':
            raise ValueError(f'not a held-out student evaluation: {path}')
        rows[key] = row
        sources.append({'path': str(path), 'sha256': hashlib.sha256(data).hexdigest(), 'program_id': key})
    destination = directory / 'heldout.recovered.results.jsonl'
    temporary = destination.with_suffix('.tmp')
    temporary.write_text(''.join(json.dumps(rows[key], separators=(',', ':')) + '\n' for key in ids if key in rows))
    # The native scorer checks exact IR and provenance digests before publication.
    score_path = directory / 'heldout.recovered.score.json'
    subprocess.run(['node', str(runtime / 'scripts/inline-curriculum/score.mjs'),
                    '--ir', str(directory / 'heldout.ir.jsonl'), '--results', str(temporary),
                    '--out', str(score_path)], check=True)
    temporary.replace(destination)
    score = json.loads(score_path.read_text())
    score['results'] = str(destination)
    score_path.write_text(json.dumps(score, indent=2) + '\n')
    report = json.loads((directory / 'report.json').read_text())
    report['heldout'] = score
    report['recovery'] = {'reason': 'single-case collector exports overwrote the aggregate',
                          'sources': sources, 'results_sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
                          'original_report_retained': str(directory / 'report.json'),
                          'provider_calls': 0, 'partial_checkpoints_count_as_missing': True}
    recovered = directory / 'report.recovered.json'
    recovered.write_text(json.dumps(report, indent=2) + '\n')
    return recovered


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    parser.add_argument('--runtime', required=True)
    args = parser.parse_args()
    print(recover(args.directory, args.runtime))
