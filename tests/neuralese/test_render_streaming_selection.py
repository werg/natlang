"""Selected scratch stays compressed without changing admitted record bytes."""
import json
from contextlib import contextmanager

import pytest

from test_render_held_split_alias import Args, load_renderer


def test_compressed_selection_matches_plain_bytes_and_all_selection_metadata(tmp_path):
    render = load_renderer()
    common = {'version': 'natlang.teacher_training_turn.native/1', 'split': 'train',
              'source_groups': ['group'], 'program_id': 'program', 'teacher_trajectory_id': 'trajectory',
              'training_admission': {'approved': True}, 'messages': [{'role': 'user', 'content': 'café\u2028line'}]}
    rows = [{**common, 'id': 'a'}, {**common, 'id': 'b', 'split': 'validation'},
            {**common, 'id': 'c', 'training_admission': {'approved': False}}, {**common, 'id': 'a'}]
    source = tmp_path/'source.jsonl'
    source.write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in rows))
    results = []
    for compressed in (False, True):
        directory = tmp_path/str(compressed);directory.mkdir()
        args = Args(source=[source], limit=None, train_only=False, registry_admits_reason=None,
                    exclude_programs=[], reserved=None, protected=None, shards=3, full_sample=1.0,
                    compress_text=compressed)
        metadata = render.select(args, directory)
        decoded = {}
        for path in directory.iterdir():
            with render.open_jsonl_text(path) as stream:
                decoded[path.name.removesuffix('.zst')] = stream.read()
        if compressed:
            assert all(path.name.endswith('.jsonl.zst') for path in directory.iterdir())
        results.append((metadata, decoded))
    assert results[0] == results[1]
    assert sum(len(text.splitlines()) for text in results[0][1].values()) == 4  # U+2028 is JSON content
    assert sum(text.count('\n') for text in results[0][1].values()) == 2


def test_selection_does_not_succeed_when_streaming_compressor_close_fails(tmp_path, monkeypatch):
    render = load_renderer()
    source = tmp_path/'empty.jsonl';source.write_text('')
    args = Args(source=[source], include_ids=[], limit=None, train_only=False, registry_admits_reason=None,
                exclude_programs=[], reserved=None, protected=None, shards=1, full_sample=1.0, compress_text=True)
    @contextmanager
    def failed_writer(path):
        with path.open('w') as stream:
            yield stream
        raise RuntimeError('compression failed')
    monkeypatch.setattr(render, 'write_jsonl_text', failed_writer)
    with pytest.raises(RuntimeError, match='compression failed'):
        render.select(args, tmp_path)
