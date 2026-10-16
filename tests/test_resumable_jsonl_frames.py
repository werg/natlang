from __future__ import annotations

import json

import pytest

from natlang_neuralese.common.jsonio import open_jsonl_text
from natlang_neuralese.common.resumable_jsonl_frames import ResumableJsonlFrames


def _read(path):
    with open_jsonl_text(path) as stream:
        return list(stream)


def test_resume_truncates_uncommitted_tail_and_preserves_source_frames(tmp_path):
    directory = tmp_path / 'selected'
    first = ResumableJsonlFrames(directory, 2, compressed=True)
    first.write(0, '{"id":"a"}\n')
    first.write(1, '{"id":"b"}\n')
    first.finish_source(0)
    saved_frames = json.loads(json.dumps(first.frames))
    first.write(0, '{"id":"partial"}\n')
    first.abort()

    # Simulate an interrupted tail that did not reach the atomic selector state.
    with (directory / '0000.jsonl.zst').open('ab') as stream:
        stream.write(b'uncommitted')
    resumed = ResumableJsonlFrames(directory, 2, compressed=True, frames=saved_frames)
    resumed.write(0, '{"id":"c"}\n')
    resumed.write(1, '{"id":"d"}\n')
    resumed.finish_source(1)
    resumed.close()

    assert _read(directory / '0000.jsonl.zst') == ['{"id":"a"}\n', '{"id":"c"}\n']
    assert _read(directory / '0001.jsonl.zst') == ['{"id":"b"}\n', '{"id":"d"}\n']


def test_resume_rejects_corrupt_committed_frame(tmp_path):
    directory = tmp_path / 'selected'
    writer = ResumableJsonlFrames(directory, 1, compressed=True)
    writer.write(0, '{"id":"a"}\n')
    writer.finish_source(0)
    frames = json.loads(json.dumps(writer.frames))
    writer.close()
    path = directory / '0000.jsonl.zst'
    data = bytearray(path.read_bytes())
    data[0] ^= 1
    path.write_bytes(data)

    with pytest.raises(ValueError, match='checksum mismatch'):
        ResumableJsonlFrames(directory, 1, compressed=True, frames=frames)


def test_uncompressed_selection_uses_the_same_durable_frontier(tmp_path):
    directory = tmp_path / 'selected'
    writer = ResumableJsonlFrames(directory, 1, compressed=False)
    writer.write(0, '{"id":"a"}\n')
    writer.finish_source(0)
    frames = json.loads(json.dumps(writer.frames))
    writer.close()
    resumed = ResumableJsonlFrames(directory, 1, compressed=False, frames=frames)
    resumed.write(0, '{"id":"b"}\n')
    resumed.finish_source(1)
    resumed.close()

    assert (directory / '0000.jsonl').read_text() == '{"id":"a"}\n{"id":"b"}\n'
