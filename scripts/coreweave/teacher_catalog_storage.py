"""Lossless catalog body storage and independently verified SQLite copying."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import struct
import zlib
from pathlib import Path

try:
    from .teacher_generation_guardian import binding, write
except ImportError:
    from teacher_generation_guardian import binding, write

_MAGIC = b"NTLGCAT\x01"
_TABLES = {"files", "programs", "occurrences", "omissions"}


def encode_body(body: str) -> bytes:
    if not isinstance(body, str):
        raise ValueError("catalog body must be original UTF-8 text")
    raw = body.encode("utf-8")
    return _MAGIC + struct.pack(">Q", len(raw)) + zlib.compress(raw, 3)


def decode_body(body: str | bytes, sha256: str | None = None) -> str:
    if isinstance(body, str):
        raw = body.encode("utf-8")
    elif isinstance(body, bytes) and body.startswith(_MAGIC) and len(body) >= 16:
        size = struct.unpack(">Q", body[8:16])[0]
        if size >= 2**63 - 1:
            raise ValueError("invalid catalog body length")
        decoder = zlib.decompressobj()
        try:
            raw = decoder.decompress(body[16:], size + 1)
        except zlib.error as exc:
            raise ValueError("corrupt compressed catalog body") from exc
        if (len(raw) != size or not decoder.eof or decoder.unused_data
                or decoder.unconsumed_tail):
            raise ValueError("compressed catalog body length or stream differs")
    else:
        raise ValueError("unsupported catalog body representation")
    if sha256 is not None and hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError("catalog program differs from original canonical bytes")
    return raw.decode("utf-8")


def _schema(db):
    rows = db.execute("SELECT type,name,tbl_name,sql FROM sqlite_master "
                      "WHERE name NOT LIKE 'sqlite_%' ORDER BY type,name").fetchall()
    if {r[1] for r in rows if r[0] == "table"} != _TABLES:
        raise ValueError("unexpected catalog tables")
    if any(r[0] not in ("table", "index") or r[3] is None for r in rows):
        raise ValueError("unexpected catalog schema object")
    return rows


def _rows(db, table):
    columns = [row[1] for row in db.execute(f'PRAGMA table_info("{table}")')]
    if table == "programs" and columns != ["sha", "id", "body", "family"]:
        raise ValueError("unexpected program table columns")
    for row in db.execute(f'SELECT rowid,* FROM "{table}" ORDER BY rowid'):
        values = list(row)
        if table == "programs":
            values[3] = decode_body(values[3], values[1])
        yield values


def _update(digest, row):
    digest.update(json.dumps(row, ensure_ascii=False, separators=(",", ":"),
                             allow_nan=False).encode("utf-8") + b"\n")


def _logical(db, table):
    digest = hashlib.sha256()
    count = 0
    for row in _rows(db, table):
        _update(digest, row)
        count += 1
    return {"rows": count, "logical_sha256": digest.hexdigest()}


def copy_catalog(source: Path, target: Path, proof: Path, *, minimum_free_bytes=20 * 1024**3):
    """Copy a quiescent catalog; never publish pointers or delete source data."""
    mechanism = binding(Path(__file__))
    source, target, proof = map(Path, (source, target, proof))
    if not source.is_file() or os.path.lexists(target) or os.path.lexists(proof):
        raise ValueError("source must exist and target/proof must be new")
    if len({source.resolve(), target.resolve(), proof.resolve()}) != 3:
        raise ValueError("catalog copy needs a distinct target")
    target.parent.mkdir(parents=True, exist_ok=True)
    def floor():
        if any(shutil.disk_usage(path).free < minimum_free_bytes
               for path in (source.parent, target.parent)):
            raise RuntimeError("catalog copy disk reserve reached; source remains intact")
    floor()
    partial = target.with_name(target.name + ".incomplete")
    # Exclusive creation prevents adopting an unrelated partial copy.
    fd = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(fd)
    src = sqlite3.connect(source.resolve().as_uri() + "?mode=rw", uri=True, timeout=5)
    dst = sqlite3.connect(partial)
    try:
        # The supervisor has stopped the writer; this also excludes an accidental
        # concurrent writer for the entire source snapshot, including WAL mode.
        src.execute("BEGIN EXCLUSIVE")
        schema = _schema(src)
        dst.execute("PRAGMA synchronous=FULL")
        for kind, _, _, sql in schema:
            if kind == "table":
                dst.execute(sql)
        original = {}
        for table in sorted(_TABLES):
            digest, count = hashlib.sha256(), 0
            columns = [r[1] for r in dst.execute(f'PRAGMA table_info("{table}")')]
            quoted = ",".join('"' + c.replace('"', '""') + '"' for c in columns)
            marks = ",".join("?" for _ in range(len(columns) + 1))
            insert = f'INSERT INTO "{table}"(rowid,{quoted}) VALUES({marks})'
            for row in _rows(src, table):
                _update(digest, row)
                stored = list(row)
                if table == "programs":
                    stored[3] = encode_body(stored[3])
                dst.execute(insert, stored)
                count += 1
                if count % 1000 == 0:
                    dst.commit()
                    floor()
            original[table] = {"rows": count, "logical_sha256": digest.hexdigest()}
            dst.commit()
        for kind, _, _, sql in schema:
            if kind == "index":
                dst.execute(sql)
        dst.commit()
        dst.close()
        with partial.open("rb") as stream:
            os.fsync(stream.fileno())
        # Reopen after close/fsync: comparison uses a fresh SQLite connection,
        # not the writer's page cache or its in-memory source rows.
        readback = sqlite3.connect(partial.resolve().as_uri() + "?mode=ro", uri=True)
        try:
            if _schema(readback) != schema:
                raise ValueError("catalog schema changed during storage copy")
            restored = {table: _logical(readback, table) for table in sorted(_TABLES)}
            if restored != original:
                raise ValueError("catalog logical rows changed during storage copy")
            if readback.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
                raise ValueError("copied catalog integrity check failed")
        finally:
            readback.close()
        floor()
        # Publish the closed, fsynced immutable file without overwriting a
        # destination created during the copy. Remove the temporary link before
        # any receipt can authorize opening the target as a mutable database.
        os.link(partial, target)
        partial.unlink()
        fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        result = {"schema": "natlang.teacher-catalog-storage-copy/1",
                  "source_path": str(source), "target": binding(target),
                  "tables": original, "schema_sha256": hashlib.sha256(
                      json.dumps(schema, separators=(",", ":")).encode()).hexdigest(),
                  "mechanism": mechanism,
                  "independent_target_readback": True,
                  "training_admission": False,
                  "scope": "lossless storage only; no pointer switch or source deletion"}
        write(proof, result)
        return result
    finally:
        dst.close()
        src.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--proof", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(copy_catalog(args.source, args.target, args.proof)))


if __name__ == "__main__":
    main()
