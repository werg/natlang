"""Append-only, source-bound JSONL output frames for resumable data selection.

This is storage plumbing, not a row-selection policy. Completed frames are
independent zstd frames, so a caller can durably checkpoint after an input
source and truncate an interrupted tail without decoding or rewriting the
committed prefix.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path


def _hash_range(stream, start: int, length: int) -> str:
    stream.seek(start)
    digest = hashlib.sha256()
    remaining = length
    while remaining:
        chunk = stream.read(min(1 << 20, remaining))
        if not chunk:
            raise ValueError('committed output frame is truncated')
        digest.update(chunk)
        remaining -= len(chunk)
    return digest.hexdigest()


class ResumableJsonlFrames:
    """Write per-source independent frames into stable output-shard files.

    ``frames`` is the checkpoint's per-shard list of ``{offset, bytes,
    sha256, source_index}`` records. Opening verifies every committed frame,
    truncates only bytes beyond that frontier, and refuses missing/corrupt
    committed data.
    """

    def __init__(self, directory: str | Path, shards: int, *, compressed: bool,
                 frames: list[list[dict]] | None = None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.shards = shards
        self.compressed = compressed
        self.paths = [self.directory / (f'{index:04d}.jsonl' + ('.zst' if compressed else ''))
                      for index in range(shards)]
        self.frames = frames if frames is not None else [[] for _ in range(shards)]
        if len(self.frames) != shards:
            raise ValueError('checkpoint frame shard count changed')
        self.files = []
        self.offsets = []
        self.active = {}
        self.starts = {}
        self.closed = False
        for index, path in enumerate(self.paths):
            records = self.frames[index]
            offset = 0
            if not path.exists() and records:
                raise ValueError(f'committed shard is missing: {path}')
            if not path.exists():
                path.touch()
            stream = path.open('r+b')
            for record in records:
                if (record.get('offset') != offset or not isinstance(record.get('bytes'), int)
                        or record['bytes'] <= 0 or not isinstance(record.get('sha256'), str)):
                    raise ValueError(f'invalid committed frame table for shard {index}')
                if _hash_range(stream, offset, record['bytes']) != record['sha256']:
                    raise ValueError(f'committed output frame checksum mismatch: {path}@{offset}')
                offset += record['bytes']
            if path.stat().st_size < offset:
                raise ValueError(f'committed output shard is truncated: {path}')
            if path.stat().st_size != offset:
                stream.truncate(offset)
                stream.flush()
                os.fsync(stream.fileno())
            stream.seek(offset)
            self.files.append(stream)
            self.offsets.append(offset)

    def write(self, shard: int, text: str) -> None:
        if self.closed:
            raise RuntimeError('selection frame writer is closed')
        if not 0 <= shard < self.shards:
            raise ValueError(f'invalid output shard: {shard}')
        stream = self.files[shard]
        if self.compressed:
            writer = self.active.get(shard)
            if writer is None:
                import zstandard
                writer = zstandard.ZstdCompressor(level=3, write_checksum=True).stream_writer(
                    stream, closefd=False, write_size=1 << 16)
                self.active[shard] = writer
                self.starts[shard] = self.offsets[shard]
            writer.write(text.encode('utf-8'))
        else:
            if shard not in self.starts:
                self.starts[shard] = self.offsets[shard]
            stream.write(text.encode('utf-8'))

    def finish_source(self, source_index: int) -> list[dict]:
        """Close this source's frames, fsync, and return newly committed ranges."""
        if self.closed:
            raise RuntimeError('selection frame writer is closed')
        for shard, writer in list(self.active.items()):
            # Each source uses its own stream_writer.  Closing that writer
            # finalizes a complete zstd frame; flushing FLUSH_FRAME first and
            # then closing the same writer emitted a malformed/truncated tail
            # with the supported python-zstandard version.
            writer.close()
            del self.active[shard]
        committed = []
        for shard, start in sorted(self.starts.items()):
            stream = self.files[shard]
            stream.flush()
            os.fsync(stream.fileno())
            end = stream.seek(0, os.SEEK_END)
            size = end - start
            if size <= 0:
                raise ValueError(f'empty output frame for shard {shard}')
            record = {'offset': start, 'bytes': size, 'sha256': _hash_range(stream, start, size),
                      'source_index': source_index}
            self.frames[shard].append(record)
            self.offsets[shard] = end
            committed.append({'shard': shard, **record})
            stream.seek(end)
        self.starts.clear()
        return committed

    def close(self) -> None:
        if self.closed:
            return
        if self.active or self.starts:
            raise RuntimeError('cannot close with an uncommitted source frame')
        for stream in self.files:
            if self.compressed and stream.seek(0, os.SEEK_END) == 0:
                # Keep empty output shards valid JSONL.zst members. Consumers
                # still open every declared shard, including shards receiving
                # no rows under the stable hash partition.
                import zstandard
                stream.write(zstandard.ZstdCompressor(level=3, write_checksum=True).compress(b''))
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
        self.closed = True

    def abort(self) -> None:
        """Close handles without committing; the next open truncates this tail."""
        if self.closed:
            return
        for writer in self.active.values():
            try:
                writer.close()
            except Exception:
                pass
        for stream in self.files:
            try:
                stream.close()
            except Exception:
                pass
        self.closed = True

