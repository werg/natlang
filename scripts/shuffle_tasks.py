#!/usr/bin/env python3
"""Place normalized JSONL tasks in a reproducible, source-mixed order.

This disk-backed sort avoids loading a full external dataset into memory. It
preserves task records byte-for-byte; only their order changes. The classifier
labeler then consumes the next uncompleted IDs on each quota window.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path


def shuffle(source: Path, output: Path, seed: str) -> dict:
    if source.resolve() == output.resolve():
        raise ValueError("input and output paths must differ")
    output.parent.mkdir(parents=True, exist_ok=True)
    database = output.with_suffix(output.suffix + ".sort.sqlite")
    if database.exists():
        raise ValueError(f"temporary sort database already exists: {database}")
    connection = sqlite3.connect(database)
    digest = hashlib.sha256()
    counts = Counter()
    try:
        connection.execute("CREATE TABLE tasks (sort_key TEXT PRIMARY KEY, task_id TEXT UNIQUE, line TEXT NOT NULL)")
        with source.open("rb") as stream:
            for number, raw in enumerate(stream, 1):
                digest.update(raw)
                if not raw.strip():
                    continue
                row = json.loads(raw)
                rid = row.get("id")
                if not isinstance(rid, str) or not rid:
                    raise ValueError(f"line {number}: missing task id")
                key = hashlib.sha256(f"{seed}\0{rid}".encode()).hexdigest()
                try:
                    connection.execute("INSERT INTO tasks VALUES (?, ?, ?)",
                                       (key, rid, raw.decode("utf-8").rstrip("\n")))
                except sqlite3.IntegrityError as exc:
                    raise ValueError(f"line {number}: duplicate task id {rid}") from exc
                counts[row.get("split", "unknown")] += 1
        connection.commit()
        out_digest = hashlib.sha256()
        with output.open("wb") as target:
            for (line,) in connection.execute("SELECT line FROM tasks ORDER BY sort_key"):
                raw = (line + "\n").encode("utf-8")
                target.write(raw)
                out_digest.update(raw)
        manifest = {"input": str(source), "input_sha256": digest.hexdigest(),
                    "output": str(output), "output_sha256": out_digest.hexdigest(),
                    "seed": seed, "counts": dict(counts)}
        output.with_suffix(output.suffix + ".manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return manifest
    finally:
        connection.close()
        database.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", default="natlang-external-v1")
    args = parser.parse_args()
    print(json.dumps(shuffle(args.input, args.output, args.seed)["counts"]))


if __name__ == "__main__":
    main()
