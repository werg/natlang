"""Canonical JSON serialisation shared by manifest and identity hashing.

Dominant repo convention: sorted keys, compact separators, ensure_ascii=False,
UTF-8 bytes, no trailing newline. Variants that differ (no sort_keys, indent,
trailing newline, ASCII escaping) are deliberately NOT provided here.
"""
from __future__ import annotations

import json
import os
import io
import subprocess
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any
from pathlib import Path


def canonical_json_str(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_json_bytes(value: Any) -> bytes:
    return canonical_json_str(value).encode("utf-8")


def utc_now_iso() -> str:
    """Timezone-aware UTC timestamp, `datetime.isoformat()` form (+00:00 suffix)."""
    return datetime.now(timezone.utc).isoformat()


def write_canonical_jsonl(path, rows) -> int:
    """Write rows as canonical JSON lines (one `canonical_json_str` per line) to a new file; refuses to overwrite,
    because published outputs are immutable. Returns the row count."""
    count = 0
    with open(path, "x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(canonical_json_str(row) + "\n")
            count += 1
    return count


def write_json_durable(path, value) -> None:
    """Replace `path` with indented, key-sorted JSON (a state file, not a canonical form): write a `.pending`
    sibling, fsync it and rename it over `path`, so a crash leaves the old or the new file, never a torn one."""
    pending = path.with_suffix(path.suffix + ".pending")
    with pending.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(path)


@contextmanager
def open_jsonl_text(path: str | Path):
    """Exact UTF-8/LF text, plain or zstd; complete reads verify decompressor success."""
    path = Path(path)
    if path.suffix != '.zst':
        with path.open(encoding='utf-8', newline='\n') as stream:
            yield stream
        return
    import tempfile
    with tempfile.TemporaryFile() as errors:
        proc = subprocess.Popen(['zstd', '-dcq', '--long=27', '--', str(path)],
                                stdout=subprocess.PIPE, stderr=errors)
        complete = False
        try:
            with io.TextIOWrapper(proc.stdout, encoding='utf-8', newline='\n') as stream:
                yield stream
                complete = stream.read(1) == ''
        finally:
            if proc.stdout is not None:
                proc.stdout.close()
            if not complete and proc.poll() is None:
                proc.terminate()
            result = proc.wait()
        if complete and result:
            errors.seek(0)
            detail = errors.read(4096).decode('utf-8', errors='replace')
            raise RuntimeError(f'zstd failed on {path} (exit {result}): {detail}')


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with open_jsonl_text(path) as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected an object at {path}:{number}")
            rows.append(value)
    return rows

