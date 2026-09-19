#!/usr/bin/env python3
"""Sample complete CSV rows across the pinned Hugging Face sales file.

The source has large embedding columns. Byte-range sampling spans the whole
file without downloading those columns for all 100,000 records. Every retained
row is parsed and checked against the source header. For exhaustive ingestion,
download the full CSV and pass it to prepare_recent_tasks.py instead.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REVISION = "714f4544cdbc3f192e7f8ea93053815c8e5479cf"
URL = ("https://huggingface.co/datasets/DeepMostInnovations/saas-sales-conversations/"
       f"resolve/{REVISION}/cleaned_custom_dataset.csv")
SOURCE_BYTES = 7166710929


def complete_records(data: bytes, start: int) -> list[bytes]:
    # This source's first column is a synthetic company id (saas-N). Quoted
    # text fields may contain physical newlines, so a plain splitlines() can
    # tear records. A candidate start is checked by the full CSV parser below.
    starts = [match.start() + (1 if data[match.start():match.start() + 1] == b"\n" else 0)
              for match in re.finditer(rb"(?:^|\n)saas-[0-9]+,", data)]
    records = [data[a:b] for a, b in zip(starts, starts[1:])]
    if starts and data.endswith(b"\n"):
        records.append(data[starts[-1]:])
    if start == 0:
        first_newline = data.find(b"\n")
        if first_newline < 0:
            raise RuntimeError("CSV header exceeds one range")
        records.insert(0, data[:first_newline + 1])
    return records


def get_chunk(index: int, chunks: int, chunk_bytes: int, url: str, total: int):
    start = (index * total) // chunks
    end = min(start + chunk_bytes - 1, total - 1)
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}",
                                                   "User-Agent": "natlang-sales-sampler/1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        if response.status != 206:
            raise RuntimeError(f"range {index}: expected HTTP 206, got {response.status}")
        content_range = response.headers.get("Content-Range", "")
        if content_range != f"bytes {start}-{end}/{total}":
            raise RuntimeError(f"range {index}: unexpected Content-Range {content_range!r}")
        data = response.read()
    if len(data) != end - start + 1:
        raise RuntimeError(f"range {index}: incomplete download")
    return index, start, end, complete_records(data, start)


def sample(output: Path, chunks: int, chunk_bytes: int, workers: int,
           url: str = URL, total: int = SOURCE_BYTES) -> dict:
    if chunks < 1 or chunk_bytes < 1024 or workers < 1:
        raise ValueError("chunks and workers must be positive; chunk bytes must be at least 1024")
    output.parent.mkdir(parents=True, exist_ok=True)
    ranges = []
    count = 0
    seen = set()
    with ThreadPoolExecutor(max_workers=workers) as pool, output.open("wb") as out:
        chunks_data = pool.map(lambda i: get_chunk(i, chunks, chunk_bytes, url, total), range(chunks))
        header = None
        width = None
        for index, start, end, records in chunks_data:
            ranges.append({"start": start, "end": end, "complete_records": len(records)})
            for position, raw in enumerate(records):
                try:
                    fields = next(csv.reader(io.StringIO(raw.decode("utf-8-sig"))))
                except (UnicodeDecodeError, csv.Error) as exc:
                    raise RuntimeError(f"range {index}: invalid CSV record") from exc
                if index == 0 and position == 0:
                    header = raw
                    width = len(fields)
                    if "conversation" not in fields or "conversation_id" not in fields:
                        raise RuntimeError("source header lacks conversation columns")
                    out.write(raw)
                    continue
                if len(fields) != width:
                    raise RuntimeError(f"range {index}: expected {width} CSV fields, got {len(fields)}")
                digest = hashlib.sha256(raw).digest()
                if digest in seen:
                    continue
                seen.add(digest)
                out.write(raw)
                count += 1
    output_hash = hashlib.sha256()
    with output.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            output_hash.update(block)
    manifest = {"source_url": url, "source_revision": REVISION,
                "source_bytes": total, "chunks": chunks, "chunk_bytes": chunk_bytes,
                "workers": workers, "ranges": ranges,
                "sample_rows": count, "output_sha256": output_hash.hexdigest()}
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--chunks", type=int, default=80)
    parser.add_argument("--chunk-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    result = sample(args.out, args.chunks, args.chunk_bytes, args.workers)
    print(json.dumps({"sample_rows": result["sample_rows"], "chunks": result["chunks"]}))


if __name__ == "__main__":
    main()
