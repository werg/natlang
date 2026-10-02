import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from scripts.assemble_joint_curriculum import assemble, assemble_streaming


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    return path


def test_sqlite_joint_assembly_preserves_order_dedup_and_rehearsal(tmp_path):
    track1 = write_rows(tmp_path / 'track-one.jsonl', [
        {'id': 't1', 'split': 'train', 'target': {'content': 'a'}},
        {'id': 't2', 'split': 'test', 'target': {'content': 'b'}},
    ])
    track2 = write_rows(tmp_path / 'track-two.jsonl', [
        {'id': 't3', 'split': 'train', 'target': {'content': 'c'}},
    ])
    teacher = write_rows(tmp_path / 'teacher.jsonl', [
        {'id': 't1', 'split': 'train', 'target': {'content': 'a'}},
        {'id': 't2', 'split': 'test', 'target': {'content': 'b'}},
        {'id': 't3', 'split': 'train', 'target': {'content': 'c'}},
    ])
    coding = write_rows(tmp_path / 'coding.jsonl', [
        {'id': 't1', 'split': 'train'}, {'id': 'c1', 'split': 'train'},
        {'id': 'c2', 'split': 'test'}, {'id': 'c3', 'split': 'train'},
        {'id': 'c4', 'split': 'train'},
    ])
    general = write_rows(tmp_path / 'general.jsonl', [
        {'id': 'g1', 'split': 'train'}, {'id': 'g2', 'split': 'test'},
        {'id': 'g3', 'split': 'train'}, {'id': 'g4', 'split': 'train'},
    ])
    tracks = [('one', track1), ('two', track2)]
    batch = assemble(tmp_path / 'batch.jsonl', general, coding, teacher, tracks)
    streaming = assemble_streaming(tmp_path / 'streaming.jsonl', general, coding, teacher, tracks)
    assert batch['tracks'] == streaming['tracks']
    assert batch['rows'] == streaming['rows']
    assert (tmp_path / 'batch.jsonl').read_bytes() == (tmp_path / 'streaming.jsonl').read_bytes()
