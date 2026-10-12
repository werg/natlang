"""A later copy of a selected source row ID (duplicate sources, replacement overlays) is recorded as an omission but
does not block its decision chain; a genuinely unselected decision still does."""
import json

from test_render_held_split_alias import Args, load_renderer


def row(rid, n, approved=True, marker='orig'):
    return {'version': 'natlang.teacher_training_turn.native/1', 'id': rid, 'split': 'train', 'source_groups': ['g'],
            'program_id': 'p', 'teacher_trajectory_id': 't', 'training_admission': {'approved': approved},
            'messages': [{'role': 'user', 'content': f'{marker}{i}'} for i in range(n)], 'family': 'f'}


def run(tmp_path, sources):
    render = load_renderer()
    paths = []
    for i, rows in enumerate(sources):
        path = tmp_path / f's{i}.jsonl'
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        paths.append(path)
    shards = tmp_path / 'shards'
    shards.mkdir()
    args = Args(source=paths, limit=None, train_only=False, registry_admits_reason=None, exclude_programs=[],
                reserved=None, protected=None, shards=1, full_sample=1.0)
    omissions, counts, blocked, *_ = render.select(args, shards)
    written = [json.loads(line) for line in (shards / '0000.jsonl').read_text().splitlines()]
    return omissions, blocked, written


def test_overlay_copy_wins_and_does_not_block(tmp_path):
    overlay = [row('d1', 2, marker='fixed')]
    original = [row('d1', 2), row('d2', 4)]
    omissions, blocked, written = run(tmp_path, [overlay, original])
    assert [r['id'] for r in written] == ['d1', 'd2']
    assert written[0]['messages'][0]['content'] == 'fixed0'
    assert [o['reason'] for o in omissions] == ['duplicate_source_row_id']
    assert blocked == {}


def test_unselected_decision_still_blocks(tmp_path):
    omissions, blocked, written = run(tmp_path, [[row('d1', 2), row('d2', 4, approved=False)]])
    assert [r['id'] for r in written] == ['d1']
    assert list(blocked.values()) == [4]
