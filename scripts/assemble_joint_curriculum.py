#!/usr/bin/env python3
"""Combine admitted teacher tracks with bounded rehearsal from earlier stages."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import shutil
import tempfile

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


def assemble_streaming(output, general, coding, teacher_path, tracks):
    """SQLite-backed assembler preserving teacher order and hashed rehearsal order."""
    if not tracks or len({name for name, _ in tracks}) != len(tracks):
        raise ValueError('at least one distinctly named teacher track is required')
    output = Path(output)
    input_paths = [general, coding, teacher_path, *(path for _, path in tracks)]
    inputs = {str(Path(path).resolve()): digest_file(path) for path in input_paths}
    output.parent.mkdir(parents=True, exist_ok=True)
    input_bytes = sum(Path(path).stat().st_size for path in input_paths)
    required_bytes = input_bytes * 2 + 1024 ** 3
    free_bytes = shutil.disk_usage(output.parent).free
    if free_bytes < required_bytes:
        raise OSError(f"streaming joint assembly requires about {required_bytes} free bytes "
                      f"(2x {input_bytes} input bytes plus 1 GiB reserve); only {free_bytes} available")
    fd, db_name = tempfile.mkstemp(prefix='joint-curriculum-', suffix='.sqlite', dir=output.parent)
    os.close(fd)
    conn = sqlite3.connect(db_name)
    conn.execute('PRAGMA journal_mode=OFF')
    conn.execute('PRAGMA synchronous=OFF')
    conn.execute('PRAGMA temp_store=FILE')
    conn.execute('PRAGMA cache_size=-16384')
    conn.executescript('''
      CREATE TABLE track_ids(id TEXT PRIMARY KEY, name TEXT);
      CREATE TABLE teacher(seq INTEGER PRIMARY KEY, body TEXT, name TEXT, split TEXT);
      CREATE TABLE seen(id TEXT PRIMARY KEY);
      CREATE TABLE candidates(lane TEXT, seq INTEGER, id TEXT, rank TEXT, body TEXT, PRIMARY KEY(lane,seq));
      CREATE INDEX candidate_order ON candidates(lane,rank,seq);
      CREATE TABLE chosen(seq INTEGER, lane TEXT, rank TEXT, body TEXT, split TEXT, PRIMARY KEY(lane,seq));
    ''')
    try:
        nonempty = set()
        for name, path in tracks:
            for row in read_rows(path):
                nonempty.add(name)
                try:
                    conn.execute('INSERT INTO track_ids VALUES(?,?)', (row['id'], name))
                except sqlite3.IntegrityError as exc:
                    raise ValueError(f'duplicate teacher decision id: {row["id"]}') from exc
        if len(nonempty) != len(tracks):
            missing = sorted({name for name, _ in tracks} - nonempty)
            raise ValueError(f'track {missing[0]} has no admitted turns')
        counts = {}
        teacher_count = 0
        for seq, row in enumerate(read_rows(teacher_path)):
            name_row = conn.execute('SELECT name FROM track_ids WHERE id=?', (row['id'],)).fetchone()
            if name_row is None:
                raise ValueError(f'prepared teacher row has no discovered track: {row["id"]}')
            name = name_row[0]
            try:
                conn.execute('INSERT INTO seen VALUES(?)', (row['id'],))
            except sqlite3.IntegrityError:
                # Existing assembler keeps duplicate prepared teacher rows in
                # teacher order, while using a set only for later rehearsal dedup.
                pass
            row = {**row, 'training_track': name}
            conn.execute('INSERT INTO teacher VALUES(?,?,?,?)',
                         (seq, json.dumps(row, ensure_ascii=False), name, row.get('split')))
            stat = counts.setdefault(name, {'rows': 0, 'train': 0})
            stat['rows'] += 1
            stat['train'] += row.get('split') == 'train'
            teacher_count += 1
        for name, _ in tracks:
            if counts.get(name, {}).get('train', 0) < 1:
                raise ValueError(f'track {name} has no prepared training rows')
        if teacher_count < 1:
            raise ValueError('prepared teacher corpus has no rows')

        def choose(path, lane):
            # Keep only rank/id metadata plus the raw normalized JSON body in sqlite.
            conn.execute('DELETE FROM candidates WHERE lane=?', (lane,))
            for seq, row in enumerate(read_rows(path)):
                conn.execute('INSERT INTO candidates VALUES(?,?,?,?,?)',
                             (lane, seq, row['id'], hashlib.sha256(f'{lane}:{row["id"]}'.encode()).hexdigest(),
                              json.dumps(row, ensure_ascii=False)))
            conn.commit()
            selected = 0
            for (seq, row_id, rank, body) in conn.execute(
                    'SELECT seq,id,rank,body FROM candidates WHERE lane=? ORDER BY rank,seq', (lane,)):
                try:
                    conn.execute('INSERT INTO seen VALUES(?)', (row_id,))
                except sqlite3.IntegrityError:
                    continue
                row = {**json.loads(body), 'training_track': lane}
                conn.execute('INSERT INTO chosen VALUES(?,?,?,?,?)',
                             (seq, lane, rank, json.dumps(row, ensure_ascii=False), row.get('split')))
                selected += 1
                if selected == teacher_count:
                    break
            conn.commit()
            counts[lane] = {'rows': selected, 'train': conn.execute(
                "SELECT COUNT(*) FROM chosen WHERE lane=? AND split='train'", (lane,)).fetchone()[0]}
            conn.execute('DELETE FROM candidates WHERE lane=?', (lane,))
            conn.commit()

        choose(coding, 'coding_rehearsal')
        choose(general, 'general_rehearsal')
        temporary = output.with_suffix(output.suffix + '.pending')
        row_count = 0
        with temporary.open('w', encoding='utf-8') as stream:
            for (body,) in conn.execute('SELECT body FROM teacher ORDER BY seq'):
                stream.write(json.dumps(json.loads(body), ensure_ascii=False) + '\n')
                row_count += 1
            for lane in ('coding_rehearsal', 'general_rehearsal'):
                for (body,) in conn.execute('SELECT body FROM chosen WHERE lane=? ORDER BY rank,seq', (lane,)):
                    stream.write(json.dumps(json.loads(body), ensure_ascii=False) + '\n')
                    row_count += 1
            stream.flush(); os.fsync(stream.fileno())
        temporary.replace(output)
        manifest = {'version': 'natlang.joint_curriculum/1', 'inputs': inputs, 'tracks': counts,
                    'rows': row_count, 'sha256': digest_file(output),
                    'builder_mode': 'sqlite-streaming/1'}
        atomic_json(output.with_suffix(output.suffix + '.manifest.json'), manifest)
        return manifest
    finally:
        conn.close()
        Path(db_name).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--general', required=True, type=Path)
    parser.add_argument('--coding', required=True, type=Path)
    parser.add_argument('--teacher', required=True, type=Path)
    parser.add_argument('--track', action='append', nargs=2, metavar=('NAME', 'PATH'), required=True)
    parser.add_argument('--streaming', action='store_true', help='use SQLite-backed out-of-core assembly')
    args = parser.parse_args()
    builder = assemble_streaming if args.streaming else assemble
    print(json.dumps(builder(args.output, args.general, args.coding, args.teacher, args.track)['tracks']))


if __name__ == '__main__':
    main()
