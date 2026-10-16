"""Stream an acquired MiroVerse zip against its sibling JSONL SFT files."""
from __future__ import annotations

import argparse
import codecs
import hashlib
import json
from pathlib import Path
import sqlite3
import zipfile

from natlang_neuralese.common.hashing import canonical_json_sha256_hex, sha256_file_hex
from natlang_neuralese.common.jsonio import open_jsonl_text, write_json_durable


def _file_sha(path: Path) -> tuple[int, str]:
    size = 0
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _identity(row):
    if not isinstance(row, dict):
        return None
    for key in ('id', '_id', 'example_id', 'sample_id', 'record_id', 'uid'):
        value = row.get(key)
        if isinstance(value, (str, int)) and not isinstance(value, bool):
            return key + ':' + str(value)
    return None


def _read_rows(stream, insert, side: str, key: str, *, raw_digest=None) -> int:
    count = 0
    for ordinal, line in enumerate(stream):
        if raw_digest is not None:
            raw_digest.update(line)
        if not line.strip():
            continue
        try:
            if isinstance(line, bytes):
                line = line.decode('utf-8')
            row = json.loads(line)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError(f'{side} {key} row {ordinal} is not JSON: {error}') from error
        insert((side, key, ordinal, canonical_json_sha256_hex(row), _identity(row)))
        count += 1
    return count


def _read_json_array_rows(stream, insert, key: str, *, raw_digest=None, chunk_bytes=1024 * 1024) -> int:
    """Read a top-level JSON array with bounded buffering and retain no row payloads."""
    decoder = json.JSONDecoder()
    utf8 = codecs.getincrementaldecoder('utf-8')('strict')
    buffer = ''
    position = 0
    eof = False
    ordinal = 0

    def fill():
        nonlocal buffer, eof
        raw = stream.read(chunk_bytes)
        if raw:
            if raw_digest is not None:
                raw_digest.update(raw)
            buffer += utf8.decode(raw)
        else:
            eof = True
            buffer += utf8.decode(b'', final=True)

    def compact():
        nonlocal buffer, position
        if position:
            buffer = buffer[position:]
            position = 0

    def need_more():
        compact()
        if not eof:
            fill()

    while True:
        while position < len(buffer) and buffer[position].isspace():
            position += 1
        if position < len(buffer) or eof:
            break
        need_more()
    if position >= len(buffer) or buffer[position] != '[':
        raise ValueError(f'{key} is not a top-level JSON array')
    position += 1
    first = True
    while True:
        while True:
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if position < len(buffer) or eof:
                break
            need_more()
        if position >= len(buffer):
            raise ValueError(f'{key} ended before closing JSON array')
        if buffer[position] == ']':
            position += 1
            while True:
                if position < len(buffer) and buffer[position:].strip():
                    raise ValueError(f'{key} has trailing content after JSON array')
                if eof:
                    break
                need_more()
            return ordinal
        if not first:
            if buffer[position] != ',':
                raise ValueError(f'{key} expected comma between array rows')
            position += 1
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer) or eof:
                    break
                need_more()
            if position >= len(buffer) or buffer[position] == ']':
                raise ValueError(f'{key} has a trailing/missing array value')
        while True:
            try:
                row, end = decoder.raw_decode(buffer, position)
                break
            except json.JSONDecodeError as error:
                if eof:
                    raise ValueError(f'{key} JSON array row {ordinal} is invalid: {error}') from error
                need_more()
        if not isinstance(row, dict):
            raise ValueError(f'{key} JSON array row {ordinal} is not an object')
        insert(('zip', key, ordinal, canonical_json_sha256_hex(row), _identity(row)))
        ordinal += 1
        position = end
        first = False
        if position > chunk_bytes:
            compact()


def audit(acquisition_receipt: Path, source_root: Path, output: Path, *, sqlite_path: Path | None = None):
    acquisition_receipt = acquisition_receipt.resolve(strict=True)
    source_root = source_root.resolve(strict=True)
    output = output.resolve()
    receipt = json.loads(acquisition_receipt.read_text())
    if receipt.get('schema') != 'natlang.external-source-acquisition/1' or receipt.get('ref') != 'D24':
        raise ValueError('expected pinned D24 acquisition receipt')
    receipt_sha = sha256_file_hex(acquisition_receipt)
    files = {item['path']: item for item in receipt.get('files', [])}
    archive_rel = 'zip_sft/MiroVerse-v0_1-SFT.zip'
    archive_bound = files.get(archive_rel)
    if not archive_bound:
        raise ValueError('pinned acquisition receipt has no MiroVerse SFT archive')
    source_items = {name: item for name, item in files.items()
                    if name.startswith('jsonl_sft/') and name.endswith('.jsonl')}
    if not source_items:
        raise ValueError('pinned acquisition receipt has no sibling JSONL SFT files')

    relevant = {archive_rel: archive_bound, **source_items}
    verified = {}
    for name, item in relevant.items():
        path = (source_root / name).resolve(strict=True)
        if not path.is_relative_to(source_root):
            raise ValueError(f'acquired path escapes source root: {name}')
        size, digest = _file_sha(path)
        if size != item['bytes'] or digest != item['sha256']:
            raise ValueError(f'acquired source differs from immutable receipt: {name}')
        verified[name] = {'path': str(path), 'bytes': size, 'sha256': digest}

    archive_path = Path(verified[archive_rel]['path'])
    sources_by_basename: dict[str, list[str]] = {}
    for name in source_items:
        sources_by_basename.setdefault(Path(name).name, []).append(name)
    db_path = sqlite_path or output.with_suffix('.sqlite')
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    connection = sqlite3.connect(db_path)
    connection.execute('PRAGMA journal_mode=OFF')
    connection.execute('PRAGMA synchronous=OFF')
    connection.execute('PRAGMA temp_store=FILE')
    connection.execute('PRAGMA cache_size=-16384')
    connection.execute('CREATE TABLE rows(side TEXT NOT NULL, file_key TEXT NOT NULL, ordinal INTEGER NOT NULL, digest TEXT NOT NULL, identity TEXT)')
    pairs = []
    unmatched_archive_members = []
    loaded_source_names = set()
    with connection:
        def insert(values):
            connection.execute('INSERT INTO rows VALUES(?,?,?,?,?)', values)
        with zipfile.ZipFile(archive_path) as archive:
            infos = [item for item in archive.infolist() if not item.is_dir()]
            archive_jsonl = [item for item in infos if item.filename.endswith('.jsonl')]
            archive_json = [item for item in infos if item.filename.endswith('.json')]
            basenames: dict[str, list[zipfile.ZipInfo]] = {}
            member_hashes = {}
            member_counts = {}
            for info in archive_jsonl:
                basenames.setdefault(Path(info.filename).name, []).append(info)
                member_digest = hashlib.sha256()
                with archive.open(info, 'r') as stream:
                    count = _read_rows(stream,
                        lambda values, key=info.filename: insert(('zip', key, *values[2:])),
                        'zip', info.filename, raw_digest=member_digest)
                if count == 0:
                    raise ValueError(f'empty JSONL archive member: {info.filename}')
                # Store the exact uncompressed member hash even when no sibling
                # JSONL exists; the archive-level digest alone cannot identify
                # which member produced a later coverage decision.
                member_hashes[id(info)] = member_digest.hexdigest()
                member_counts[id(info)] = count
            for info in archive_json:
                member_digest = hashlib.sha256()
                with archive.open(info, 'r') as stream:
                    count = _read_json_array_rows(stream,
                        lambda values: insert(values), info.filename, raw_digest=member_digest)
                if count == 0:
                    raise ValueError(f'empty JSON array archive member: {info.filename}')
                member_hashes[id(info)] = member_digest.hexdigest()
                member_counts[id(info)] = count
            for info in archive_jsonl:
                match = sources_by_basename.get(Path(info.filename).name, [])
                if len(match) != 1 or len(basenames[Path(info.filename).name]) != 1:
                    unmatched_archive_members.append({'member': info.filename, 'reason': 'ambiguous-or-unmatched-basename',
                        'source_candidates': match, 'archive_members_with_basename': [x.filename for x in basenames[Path(info.filename).name]],
                        'archive_member_sha256': member_hashes[id(info)]})
                    continue
                source_name = match[0]
                source_path = Path(verified[source_name]['path'])
                if source_name not in loaded_source_names:
                    with open_jsonl_text(source_path) as stream:
                        source_count = _read_rows(stream, lambda values, key=source_name: insert(('source', key, *values[2:])), 'source', source_name)
                    loaded_source_names.add(source_name)
                else:
                    source_count = connection.execute('SELECT COUNT(*) FROM rows WHERE side=? AND file_key=?',
                        ('source', source_name)).fetchone()[0]
                zip_count = connection.execute('SELECT COUNT(*) FROM rows WHERE side=? AND file_key=?', ('zip', info.filename)).fetchone()[0]
                if source_count != zip_count:
                    pairs.append({'member': info.filename, 'source': source_name, 'status': 'row-count-mismatch',
                                  'archive_rows': zip_count, 'source_rows': source_count})
                    continue
                differences = connection.execute('''
                    WITH z AS (SELECT digest,COUNT(*) n FROM rows WHERE side='zip' AND file_key=? GROUP BY digest),
                         s AS (SELECT digest,COUNT(*) n FROM rows WHERE side='source' AND file_key=? GROUP BY digest)
                    SELECT d.digest,COALESCE(z.n,0),COALESCE(s.n,0)
                    FROM (SELECT digest FROM z UNION SELECT digest FROM s) d
                    LEFT JOIN z ON z.digest=d.digest LEFT JOIN s ON s.digest=d.digest
                    WHERE COALESCE(z.n,0)!=COALESCE(s.n,0)
                    LIMIT 20
                ''', (info.filename, source_name)).fetchall()
                identity_conflicts = connection.execute('''
                    WITH z AS (SELECT identity,digest FROM rows WHERE side='zip' AND file_key=? AND identity IS NOT NULL GROUP BY identity,digest),
                         s AS (SELECT identity,digest FROM rows WHERE side='source' AND file_key=? AND identity IS NOT NULL GROUP BY identity,digest)
                    SELECT z.identity,z.digest,s.digest FROM z JOIN s USING(identity)
                    WHERE z.digest!=s.digest LIMIT 20
                ''', (info.filename, source_name)).fetchall()
                row_sha = lambda side, key: canonical_json_sha256_hex(connection.execute(
                    'SELECT digest,COUNT(*) AS n FROM rows WHERE side=? AND file_key=? GROUP BY digest ORDER BY digest',
                    (side, key)).fetchall())
                pairs.append({'member': info.filename, 'source': source_name,
                    'archive_uncompressed_bytes': info.file_size, 'archive_compressed_bytes': info.compress_size,
                    'archive_member_crc32': f'{info.CRC:08x}',
                    'archive_member_sha256': member_hashes[id(info)],
                    'source_sha256': verified[source_name]['sha256'],
                    'archive_rows': zip_count, 'source_rows': source_count,
                    'archive_row_multiset_sha256': row_sha('zip', info.filename),
                    'source_row_multiset_sha256': row_sha('source', source_name),
                    'exact_row_multiset_match': not differences,
                    'row_differences_sample': differences,
                    'same_identity_different_payload_sample': identity_conflicts})
        # Compare split JSON arrays and all acquired JSONL sources as one exact
        # multiset. The archive's split ranges do not correspond to source filenames.
        for source_name in sorted(source_items):
            if source_name in loaded_source_names:
                continue
            source_path = Path(verified[source_name]['path'])
            with open_jsonl_text(source_path) as stream:
                _read_rows(stream, lambda values, key=source_name: insert(('source', key, *values[2:])),
                           'source', source_name)
        archive_count = connection.execute("SELECT COUNT(*) FROM rows WHERE side='zip'").fetchone()[0]
        source_count = connection.execute("SELECT COUNT(*) FROM rows WHERE side='source'").fetchone()[0]
        overall_diff = connection.execute('''
            WITH z AS (SELECT digest,COUNT(*) n FROM rows WHERE side='zip' GROUP BY digest),
                 s AS (SELECT digest,COUNT(*) n FROM rows WHERE side='source' GROUP BY digest)
            SELECT d.digest,COALESCE(z.n,0),COALESCE(s.n,0)
            FROM (SELECT digest FROM z UNION SELECT digest FROM s) d
            LEFT JOIN z ON z.digest=d.digest LEFT JOIN s ON s.digest=d.digest
            WHERE COALESCE(z.n,0)!=COALESCE(s.n,0) LIMIT 50
        ''').fetchall()
        overall_identity_conflicts = connection.execute('''
            WITH z AS (SELECT identity,digest FROM rows WHERE side='zip' AND identity IS NOT NULL GROUP BY identity,digest),
                 s AS (SELECT identity,digest FROM rows WHERE side='source' AND identity IS NOT NULL GROUP BY identity,digest)
            SELECT z.identity,z.digest,s.digest FROM z JOIN s USING(identity)
            WHERE z.digest!=s.digest LIMIT 50
        ''').fetchall()
        def global_row_sha(side):
            return canonical_json_sha256_hex(connection.execute(
                'SELECT digest,COUNT(*) AS n FROM rows WHERE side=? GROUP BY digest ORDER BY digest', (side,)).fetchall())
    source_not_in_archive = sorted(set(source_items) - {item['source'] for item in pairs})
    unhandled = [item.filename for item in infos if not item.filename.endswith(('.json', '.jsonl', '.md', '.txt'))]
    output_receipt = {
        'schema': 'natlang.miroverse-zip-jsonl-coverage/2',
        'acquisition_receipt': {'path': str(acquisition_receipt), 'sha256': receipt_sha,
                                'repo': receipt['repo'], 'revision': receipt['revision']},
        'source_root': str(source_root), 'verified_acquired_files': verified,
        'archive': archive_rel, 'archive_jsonl_member_count': len(archive_jsonl),
        'archive_json_array_member_count': len(archive_json),
        'archive_json_array_members': [{'path': item.filename, 'rows': member_counts[id(item)],
            'uncompressed_bytes': item.file_size, 'compressed_bytes': item.compress_size,
            'member_sha256': member_hashes[id(item)]} for item in archive_json],
        'pairs': pairs, 'unmatched_archive_members': unmatched_archive_members,
        'jsonl_sources_without_unique_archive_pair': source_not_in_archive,
        'all_paired_rows_exact': bool(pairs) and all(item.get('exact_row_multiset_match') is True for item in pairs),
        'global_archive_vs_jsonl_source': {'archive_rows': archive_count, 'source_rows': source_count,
            'archive_row_multiset_sha256': global_row_sha('zip'),
            'source_row_multiset_sha256': global_row_sha('source'),
            'exact_row_multiset_match': archive_count == source_count and not overall_diff,
            'row_differences_sample': overall_diff,
            'same_identity_different_payload_sample': overall_identity_conflicts},
        'complete_jsonl_coverage': (bool(source_items) and bool(archive_json) and not unhandled
            and archive_count == source_count and not overall_diff),
        'zip_non_jsonl_members': [{'path': item.filename, 'bytes': item.file_size, 'compressed_bytes': item.compress_size}
                                  for item in infos if not item.filename.endswith('.jsonl')],
        'unhandled_archive_payloads': [{'path': name, 'bytes': next(item.file_size for item in infos if item.filename == name)}
            for name in unhandled],
        'all_archive_payloads_accounted_for': not unhandled,
    }
    write_json_durable(output, output_receipt)
    connection.close()
    if sqlite_path is None:
        db_path.unlink(missing_ok=True)
    return output_receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--acquisition-receipt', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sqlite', type=Path, help='retain SQLite scratch at this exact path for diagnosis')
    args = parser.parse_args(argv)
    result = audit(args.acquisition_receipt, args.source_root, args.output, sqlite_path=args.sqlite)
    print(json.dumps({'output': str(args.output), 'pairs': len(result['pairs']),
                      'exact_rows': sum(x.get('archive_rows', 0) for x in result['pairs'] if x.get('exact_row_multiset_match')),
                      'all_paired_rows_exact': result['all_paired_rows_exact'],
                      'unmatched_archive_members': len(result['unmatched_archive_members']),
                      'jsonl_sources_without_unique_archive_pair': len(result['jsonl_sources_without_unique_archive_pair'])}, sort_keys=True))


if __name__ == '__main__':
    main()
