import hashlib
import json
import sqlite3
import struct

import pytest

from scripts.coreweave.teacher_catalog_storage import copy_catalog, decode_body, encode_body


def test_body_restores_exact_unicode_and_rejects_corrupt_or_noncanonical_storage():
    text = '{"id":"文😀","text":"' + 'a\\n\\\"' * 400 + '"}\n'
    sha = hashlib.sha256(text.encode()).hexdigest()
    packed = encode_body(text)
    assert len(packed) < len(text.encode())
    assert decode_body(packed, sha) == decode_body(text, sha) == text
    for corrupt in (packed[:-1], packed + b"extra", b"bad", packed[:8] + struct.pack(">Q", 1) + packed[16:]):
        with pytest.raises(ValueError):
            decode_body(corrupt, sha)
    with pytest.raises(ValueError, match="canonical bytes"):
        decode_body(packed, "0" * 64)


def make_catalog(path):
    db = sqlite3.connect(path)
    db.executescript('''
      CREATE TABLE files(path TEXT PRIMARY KEY,sha TEXT,bytes INTEGER,cursor INTEGER,complete INTEGER,reason TEXT);
      CREATE TABLE programs(sha TEXT PRIMARY KEY,id TEXT,body TEXT,family TEXT);
      CREATE INDEX program_ids ON programs(id);
      CREATE TABLE occurrences(file TEXT,line INTEGER,row_sha TEXT,program_sha TEXT,reason TEXT,groups_json TEXT,PRIMARY KEY(file,line));
      CREATE INDEX occurrence_programs ON occurrences(program_sha);
      CREATE TABLE omissions(file TEXT,line INTEGER,reason TEXT,PRIMARY KEY(file,line));
    ''')
    db.execute('INSERT INTO files VALUES(?,?,?,?,?,?)', ('source', 'source-sha', 700, 11, 0, None))
    # Deliberately sparse, non-lexical rowids: selection follows original row order.
    for rowid, name in ((41, "z"), (7, "a")):
        body = json.dumps({"id": name, "text": "日本語" * 1000}, ensure_ascii=False)
        sha = hashlib.sha256(body.encode()).hexdigest()
        db.execute('INSERT INTO programs(rowid,sha,id,body,family)VALUES(?,?,?,?,?)',
                   (rowid, sha, name, body, "family"))
        db.execute('INSERT INTO occurrences VALUES(?,?,?,?,?,?)',
                   ('source', rowid, 'row-sha', sha, 'held_group' if rowid == 7 else None, '["g"]'))
    db.execute('INSERT INTO omissions VALUES(?,?,?)', ('source', 3, 'no_program_ir'))
    db.commit()
    db.close()


def test_verified_copy_preserves_selection_rowids_omissions_and_unfinished_cursors(tmp_path):
    source, target, proof = (tmp_path / n for n in ('old.sqlite', 'new.sqlite', 'copy.json'))
    make_catalog(source)
    original = source.read_bytes()
    result = copy_catalog(source, target, proof, minimum_free_bytes=0)
    assert result == json.loads(proof.read_text())
    assert result['independent_target_readback'] and not result['training_admission']
    assert source.read_bytes() == original
    a, b = sqlite3.connect(source), sqlite3.connect(target)
    for table in ('files', 'occurrences', 'omissions'):
        assert a.execute(f'SELECT rowid,* FROM {table}').fetchall() == b.execute(f'SELECT rowid,* FROM {table}').fetchall()
    left = a.execute('SELECT rowid,sha,id,body,family FROM programs ORDER BY rowid').fetchall()
    right = b.execute('SELECT rowid,sha,id,body,family FROM programs ORDER BY rowid').fetchall()
    assert [r[0] for r in right] == [7, 41]
    assert [(r[0], r[1], r[2], decode_body(r[3], r[1]), r[4]) for r in right] == left
    assert all(isinstance(r[3], bytes) for r in right)
    a.close()
    b.close()
    with pytest.raises(ValueError):
        copy_catalog(source, target, proof, minimum_free_bytes=0)


def test_failed_source_authentication_never_replaces_source_or_publishes_target(tmp_path):
    source, target, proof = (tmp_path / n for n in ('old.sqlite', 'new.sqlite', 'copy.json'))
    make_catalog(source)
    db = sqlite3.connect(source)
    db.execute("UPDATE programs SET body='changed' WHERE rowid=7")
    db.commit()
    db.close()
    original = source.read_bytes()
    with pytest.raises(ValueError, match="canonical bytes"):
        copy_catalog(source, target, proof, minimum_free_bytes=0)
    assert source.read_bytes() == original and not target.exists() and not proof.exists()


def test_copy_cannot_overwrite_its_own_proof_or_follow_a_dangling_target(tmp_path):
    source = tmp_path / 'old.sqlite'
    make_catalog(source)
    target = tmp_path / 'new.sqlite'
    with pytest.raises(ValueError, match='distinct'):
        copy_catalog(source, target, target, minimum_free_bytes=0)
    target.symlink_to(tmp_path / 'missing')
    with pytest.raises(ValueError, match='must be new'):
        copy_catalog(source, target, tmp_path / 'proof.json', minimum_free_bytes=0)


def test_disk_floor_stops_before_allocating_a_partial_catalog(tmp_path):
    source = tmp_path / 'old.sqlite'
    make_catalog(source)
    target = tmp_path / 'new.sqlite'
    with pytest.raises(RuntimeError, match='reserve'):
        copy_catalog(source, target, tmp_path / 'proof.json', minimum_free_bytes=2**100)
    assert not target.exists() and not target.with_name(target.name + '.incomplete').exists()
