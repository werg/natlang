"""Shared plumbing for the S1 converters: rejections, sampling readers and JSONL output."""
from __future__ import annotations

import json
import os
import random
import tempfile
from collections import Counter
from pathlib import Path

DEFAULT_OUTPUT_ROOT = Path(os.environ.get("NATLANG_NEURALESE_DATA", "/mnt/external/natlang-development-data/data/neuralese"))


class Reject(Exception):
    """A source row that cannot become a valid record; the message is the rejection reason."""


def parquet_rows(path: Path, columns=None, limit: int | None = None, seed: int = 0):
    """Yield (row_index, row) from a parquet file, sampling whole row groups in a seeded order when limited."""
    import pyarrow.parquet as pq  # requires pyarrow; run with an environment that has it

    f = pq.ParquetFile(str(path))
    groups = list(range(f.metadata.num_row_groups))
    starts, offset = [], 0
    for g in groups:
        starts.append(offset)
        offset += f.metadata.row_group(g).num_rows
    if limit is not None:
        random.Random(seed).shuffle(groups)
    produced = 0
    for g in groups:
        table = f.read_row_group(g, columns=columns)
        rows = table.to_pylist()
        if limit is not None:
            # Spread the sample inside a row group too, so one large group does not dominate.
            step = max(1, len(rows) // max(1, min(len(rows), limit - produced)))
            picks = range(0, len(rows), step)
        else:
            picks = range(len(rows))
        for i in picks:
            yield starts[g] + i, rows[i]
            produced += 1
            if limit is not None and produced >= limit:
                return


def jsonl_rows(path: Path, limit: int | None = None):
    with open(path, encoding="utf-8") as stream:
        for i, line in enumerate(stream):
            if limit is not None and i >= limit:
                return
            if line.strip():
                yield i, json.loads(line)


class Sink:
    """Write accepted records and rejections atomically, with per-reason counts."""

    def __init__(self, out_dir: Path, name: str):
        self.out_dir = out_dir
        self.name = name
        out_dir.mkdir(parents=True, exist_ok=True)
        self._fds = {}
        self.accepted = 0
        self.rejected: Counter = Counter()
        self.by_split: Counter = Counter()
        self.examples: dict[str, str] = {}
        self._open("records", f"{name}.port-records.jsonl")
        self._open("rejects", f"{name}.rejects.jsonl")

    def _open(self, key: str, filename: str):
        fd, tmp = tempfile.mkstemp(prefix=filename + ".", suffix=".pending", dir=self.out_dir)
        self._fds[key] = (os.fdopen(fd, "w", encoding="utf-8"), tmp, self.out_dir / filename)

    def accept(self, record: dict):
        self._fds["records"][0].write(json.dumps(record, ensure_ascii=False) + "\n")
        self.accepted += 1
        self.by_split[record["split"]] += 1

    def reject(self, store: str, row, reason: str):
        key = reason.split(":")[0]
        self.rejected[key] += 1
        self.examples.setdefault(key, f"{store}#{row}: {reason}")
        self._fds["rejects"][0].write(json.dumps({"store": store, "row": row, "reason": reason}, ensure_ascii=False) + "\n")

    def close(self) -> dict:
        for stream, tmp, final in self._fds.values():
            stream.close()
            os.chmod(tmp, 0o644)
            os.replace(tmp, final)
        return {"name": self.name, "accepted": self.accepted, "by_split": dict(self.by_split),
                "rejected": dict(self.rejected), "reject_examples": self.examples}
