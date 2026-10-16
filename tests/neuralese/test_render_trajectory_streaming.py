from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from natlang_neuralese.common.jsonio import open_jsonl_text, write_jsonl_text

ROOT = Path(__file__).resolve().parents[2]
RENDERER = ROOT / 'scripts/neuralese_data/render_native_turn_corpus.py'
spec = importlib.util.spec_from_file_location('streaming_trajectory_renderer', RENDERER)
renderer = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = renderer
spec.loader.exec_module(renderer)


def _source(path: Path):
    rows = [
        {'id': 'a0', 'teacher_trajectory_id': 'trajectory-a', 'source_ref': {'trajectory_id': 'trajectory-a', 'invocation_id': 'i0'},
         'split': 'train', 'source_groups': ['g-a'], 'messages': [], 'target': {'role': 'assistant', 'content': 'one'}},
        {'id': 'b0', 'teacher_trajectory_id': 'trajectory-b', 'source_ref': {'trajectory_id': 'trajectory-b', 'invocation_id': 'i0'},
         'split': 'test', 'source_groups': ['g-b'], 'messages': [], 'target': {'role': 'assistant', 'content': 'held'}},
        {'id': 'a1', 'teacher_trajectory_id': 'trajectory-a', 'source_ref': {'trajectory_id': 'trajectory-a', 'invocation_id': 'i1'},
         'split': 'train', 'source_groups': ['g-a'], 'messages': [], 'target': {'role': 'assistant', 'content': 'two'}},
    ]
    with write_jsonl_text(path) as stream:
        for row in rows:
            stream.write(renderer.canonical(row) + '\n')
    return rows


def _read(path):
    with open_jsonl_text(path) as stream:
        return [json.loads(line) for line in stream if line.strip()]


def test_streamed_trajectory_groups_are_exact_and_resume_after_interruption(tmp_path, monkeypatch):
    selected = tmp_path / 'selected.jsonl.zst'
    source_rows = _source(selected)
    calls = []

    def materialized(job):
        shard_path, out_dir, tokenizer, _blocked, _corpus, _held, _decision, _header, _full, modes, compressed, _target = job
        group_source = Path(shard_path)
        group_root = Path(out_dir)
        group_id = group_source.name.removesuffix('.jsonl')
        rows = _read(group_source)
        calls.append([row['id'] for row in rows])
        if len(calls) == 2:
            raise RuntimeError('simulated worker interruption before trajectory commit')
        result = {'shard': group_id, 'tokenizer': tokenizer, 'records': len(rows), 'whole-trajectory': len(rows),
                  'parity': {'equal': len(rows)}}
        for mode in modes:
            output = group_root / mode / tokenizer / (f'{group_id}.jsonl.zst' if compressed else f'{group_id}.jsonl')
            output.parent.mkdir(parents=True, exist_ok=True)
            with write_jsonl_text(output) as stream:
                for row in rows:
                    stream.write(renderer.canonical({'id': row['id'], 'split': row['split'],
                                                     'source_groups': row['source_groups']}) + '\n')
        omission = group_root / 'omissions' / tokenizer / f'{group_id}.jsonl'
        omission.parent.mkdir(parents=True, exist_ok=True)
        omission.write_text('')
        return result

    monkeypatch.setattr(renderer, '_render_shard_materialized', materialized)
    work = tmp_path / 'work'
    job = (str(selected), str(work), 'lfm25', {}, 'fixture', set(), set(), False, None,
           ('whole-trajectory',), True, set())
    import pytest
    with pytest.raises(RuntimeError, match='simulated worker interruption'):
        renderer.render_shard(job)
    assert len(list((work / 'trajectory-stream' / 'lfm25' / 'selected').glob('receipts/*.json'))) == 1

    calls.clear()
    result = renderer.render_shard(job)
    assert result['records'] == len(source_rows)
    assert result['trajectory_groups'] == 2
    # The first trajectory was committed before the interruption and must be
    # reused; the other group is rendered exactly once after restart.
    assert len(calls) == 1
    output = work / 'whole-trajectory' / 'lfm25' / 'selected.jsonl.zst'
    assert [row['id'] for row in _read(output)] == ['a0', 'b0', 'a1']
    assert not list(output.parent.glob('*.tmp'))


def test_streamed_trajectory_binding_rejects_changed_source(tmp_path, monkeypatch):
    selected = tmp_path / 'selected.jsonl'
    rows = _source(selected)

    def materialized(job):
        group_id = Path(job[0]).stem
        root = Path(job[1])
        path = root / 'whole-trajectory' / job[2] / f'{group_id}.jsonl'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('')
        omission = root / 'omissions' / job[2] / f'{group_id}.jsonl'
        omission.parent.mkdir(parents=True, exist_ok=True)
        omission.write_text('')
        return {'shard': group_id, 'tokenizer': job[2], 'records': 1, 'whole-trajectory': 0}

    monkeypatch.setattr(renderer, '_render_shard_materialized', materialized)
    job = (str(selected), str(tmp_path / 'work'), 'lfm25', {}, 'fixture', set(), set(), False, None,
           ('whole-trajectory',), False, set())
    renderer.render_shard(job)
    changed = rows[0]
    changed['target']['content'] = 'changed target'
    with write_jsonl_text(selected) as stream:
        for row in rows:
            stream.write(renderer.canonical(row) + '\n')
    with __import__('pytest').raises(ValueError, match='committed trajectory binding changed'):
        renderer.render_shard(job)
