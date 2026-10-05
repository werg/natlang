#!/usr/bin/env python3
"""Compare completed teacher outcomes against the current native corpus by identity.

Identity presence is coverage evidence, not admission or byte/digest equivalence.
Uncovered successes are a review/import backlog, never silently admitted.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    native = set()
    with args.native.open() as stream:
        for line in stream:
            row = json.loads(line)
            identity = row.get('teacher_trajectory_id')
            if identity:
                native.add(identity)
    groups, uncovered, malformed = {}, defaultdict(list), []
    for path in sorted(set(args.evidence.glob('campaign*/**/results.jsonl')) |
                       set(args.evidence.glob('campaign*/**/*.results.jsonl'))):
        counts, accepted_ids = Counter(), set()
        with path.open() as stream:
            for number, line in enumerate(stream, 1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    malformed.append({'file': str(path.relative_to(args.evidence)), 'line': number})
                    continue
                outcome = row.get('outcome', {})
                accepted = outcome.get('accepted') is True
                counts['accepted' if accepted else 'nonaccepted'] += 1
                identity = row.get('id')
                if accepted and identity:
                    accepted_ids.add(identity)
                    counts['identity_present' if identity in native else 'identity_missing'] += 1
                    if identity not in native:
                        uncovered[identity].append(str(path.relative_to(args.evidence)))
        groups[str(path.relative_to(args.evidence))] = {**dict(counts), 'unique_accepted_ids': len(accepted_ids)}
    report = {'schema': 'natlang.teacher-corpus-coverage/1', 'native_path': str(args.native),
              'evidence_path': str(args.evidence), 'native_teacher_ids': len(native), 'files': groups,
              'uncovered_accepted_ids': dict(uncovered), 'malformed': malformed,
              'policy': 'Missing accepted IDs require provenance/import, source-policy and protected/split review. Presence does not prove same revision/digest. Negatives retained for repair/preference audit, not counted as successful SFT.'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'files': len(groups), 'native_teacher_ids': len(native),
                      'uncovered_accepted_ids': len(uncovered), 'malformed': len(malformed)}))


if __name__ == '__main__':
    main()
