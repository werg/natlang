#!/usr/bin/env python3
"""Combine admitted teacher tracks with bounded rehearsal from earlier stages."""
import argparse
import hashlib
import json
from pathlib import Path

from run_training_pipeline import atomic_json, digest_file


def read_rows(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def assemble(output, general, coding, teacher_path, tracks):
    if not tracks or len({name for name, _ in tracks}) != len(tracks):
        raise ValueError('at least one distinctly named teacher track is required')
    output = Path(output)
    inputs = {str(Path(path).resolve()): digest_file(path)
              for path in [general, coding, teacher_path, *(path for _, path in tracks)]}
    teacher = []
    counts = {}
    seen = set()
    origin = {}
    for name, path in tracks:
        rows = list(read_rows(path))
        if not rows:
            raise ValueError(f'track {name} has no admitted turns: {path}')
        for row in rows:
            row_id = row['id']
            if row_id in origin:
                raise ValueError(f'duplicate teacher decision id: {row_id}')
            origin[row_id] = name
    for row in read_rows(teacher_path):
        row_id = row['id']
        if row_id not in origin:
            raise ValueError(f'prepared teacher row has no discovered track: {row_id}')
        seen.add(row_id)
        name = origin[row_id]
        counts.setdefault(name, {'rows': 0, 'train': 0})
        counts[name]['rows'] += 1
        counts[name]['train'] += row.get('split') == 'train'
        teacher.append({**row, 'training_track': name})
    for name, _ in tracks:
        if counts.get(name, {}).get('train', 0) < 1:
            raise ValueError(f'track {name} has no prepared training rows')
    # The final adapter sees every admitted teacher decision and a bounded sample
    # from each earlier stage. The coding phase already contains some general
    # rehearsal; identity deduplication prevents that overlap being multiplied.
    limit = len(teacher)
    def rehearsal(path, name):
        rows = sorted(read_rows(path), key=lambda row: hashlib.sha256(
            f'{name}:{row["id"]}'.encode()).hexdigest())
        selected = []
        for row in rows:
            if row['id'] in seen:
                continue
            seen.add(row['id'])
            selected.append({**row, 'training_track': name})
            if len(selected) == limit:
                break
        counts[name] = {'rows': len(selected), 'train': sum(row.get('split') == 'train' for row in selected)}
        return selected
    coding_rows = rehearsal(coding, 'coding_rehearsal')
    general_rows = rehearsal(general, 'general_rehearsal')
    rows = [*teacher, *coding_rows, *general_rows]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.pending')
    with temporary.open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    temporary.replace(output)
    manifest = {'version': 'natlang.joint_curriculum/1', 'inputs': inputs,
                'tracks': counts, 'rows': len(rows), 'sha256': digest_file(output)}
    atomic_json(output.with_suffix(output.suffix + '.manifest.json'), manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--general', required=True, type=Path)
    parser.add_argument('--coding', required=True, type=Path)
    parser.add_argument('--teacher', required=True, type=Path)
    parser.add_argument('--track', action='append', nargs=2, metavar=('NAME', 'PATH'), required=True)
    args = parser.parse_args()
    print(json.dumps(assemble(args.output, args.general, args.coding, args.teacher, args.track)['tracks']))


if __name__ == '__main__':
    main()
