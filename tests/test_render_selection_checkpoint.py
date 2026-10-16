from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.neuralese_data.render_native_turn_corpus as renderer
from natlang_neuralese.common.jsonio import open_jsonl_text
from natlang_neuralese.common.resumable_jsonl_frames import ResumableJsonlFrames
from scripts.neuralese_data.render_native_turn_corpus import (
    _selection_state,
    select,
    selection_input_binding,
)


def _row(identifier, program, split='train'):
    return {
        'id': identifier,
        'version': 'natlang.teacher_training_turn.native/1',
        'split': split,
        'program_id': program,
        'family': 'fixture',
        'source_groups': [f'group-{program}'],
        'teacher_trajectory_id': program,
        'training_admission': {'approved': True},
        'messages': [{'role': 'user', 'content': 'prompt'}],
        'target': {'role': 'assistant', 'content': 'answer'},
    }


def _args(source, *, protected=None):
    return SimpleNamespace(
        source=source, include_ids=[], exclude_programs=[], decision_exclude_ids=[], target_exclude_ids=[],
        held_text=[], reserved=None, protected=protected, corpus_id='fixture', include_admits=False,
        train_only=False, registry_admits_reason='', limit=None, shards=4, full_sample=1.0,
        candidate_selection_admission=None, compress_text=True,
    )


def _decode(path):
    with open_jsonl_text(path) as stream:
        return [json.loads(line) for line in stream if line.strip()]


def test_select_checkpoint_resume_matches_uninterrupted_order_and_state(tmp_path):
    source_a, source_b = tmp_path / 'a.jsonl', tmp_path / 'b.jsonl'
    rows_a = [_row('a0', 'pa'), _row('a1', 'pa')]
    rows_b = [_row('b0', 'pb'), _row('a0', 'pa')]
    source_a.write_text(''.join(json.dumps(row) + '\n' for row in rows_a))
    source_b.write_text(''.join(json.dumps(row) + '\n' for row in rows_b))
    args = _args([source_a, source_b])
    binding = selection_input_binding(args)

    uninterrupted_dir = tmp_path / 'uninterrupted' / 'selected'
    uninterrupted_state = tmp_path / 'uninterrupted' / 'selection-state.json'
    uninterrupted_dir.mkdir(parents=True)
    expected = select(args, uninterrupted_dir, uninterrupted_state, binding)

    resumed_dir = tmp_path / 'resumed' / 'selected'
    resumed_state = tmp_path / 'resumed' / 'selection-state.json'
    resumed_dir.mkdir(parents=True)
    exclude = {}
    reserved_cache, reserved_groups = {}, set()
    protected_cache = {}
    counter = Counter()
    strata = {'fixture': {'pa'}}
    blocked = {}
    seen_ids = {'a0', 'a1'}
    test_groups = set()
    frames = ResumableJsonlFrames(resumed_dir, 4, compressed=True)
    for row in rows_a:
        shard = int(__import__('hashlib').sha256(row['teacher_trajectory_id'].encode()).hexdigest()[:8], 16) % 4
        frames.write(shard, json.dumps(row) + '\n')
        counter['source_rows'] += 1
        counter['selected'] += 1
        counter['selected:train'] += 1
    frames.finish_source(0)
    checkpoint = _selection_state(binding, 1, [], counter, strata, blocked, protected_cache,
                                  seen_ids, test_groups, reserved_groups, reserved_cache, frames)
    from natlang_neuralese.common.jsonio import write_json_durable
    write_json_durable(resumed_state, checkpoint)
    frames.write(0, '{"id":"uncommitted"}\n')
    frames.abort()

    actual = select(args, resumed_dir, resumed_state, binding)
    assert actual[1] == expected[1]
    assert actual[2:6] == expected[2:6]
    for index in range(4):
        filename = f'{index:04d}.jsonl.zst'
        assert _decode(uninterrupted_dir / filename) == _decode(resumed_dir / filename)


def test_selection_checkpoint_rejects_changed_source_bytes(tmp_path):
    source = tmp_path / 'input.jsonl'
    source.write_text(json.dumps(_row('a0', 'pa')) + '\n')
    args = _args([source])
    binding = selection_input_binding(args)
    source.write_text(json.dumps(_row('a0', 'changed')) + '\n')
    checkpoint_dir = tmp_path / 'selected'
    checkpoint_dir.mkdir()
    frames = ResumableJsonlFrames(checkpoint_dir, 4, compressed=True)
    state = _selection_state(binding, 0, [], Counter(), {}, {}, {}, set(), set(), set(), {}, frames)
    from natlang_neuralese.common.jsonio import write_json_durable
    write_json_durable(tmp_path / 'selection-state.json', state)
    frames.close()

    with pytest.raises(ValueError, match='inputs or selector semantics changed'):
        select(args, checkpoint_dir, tmp_path / 'selection-state.json', selection_input_binding(args))


def test_legacy_whole_renderer_hash_migrates_only_when_all_selection_inputs_match(tmp_path):
    source = tmp_path / 'input.jsonl'
    source.write_text(json.dumps(_row('a0', 'pa')) + '\n')
    args = _args([source])
    binding = selection_input_binding(args)
    shard_dir, state_path = tmp_path / 'selected', tmp_path / 'selection-state.json'
    shard_dir.mkdir()
    select(args, shard_dir, state_path, binding)

    state = json.loads(state_path.read_text())
    legacy = dict(state['binding'])
    legacy_options = dict(legacy['options'])
    legacy_options['selector_code_sha256'] = '797e534147c258b5453c146365e53763133ac45ec468193b66fba442419bdccf'
    legacy['options'] = legacy_options
    legacy.pop('sha256', None)
    from natlang_neuralese.common.jsonio import canonical_json_str
    from natlang_neuralese.common.hashing import sha256_hex
    legacy['sha256'] = sha256_hex(canonical_json_str(legacy).encode())
    state['binding'] = legacy
    state_path.write_text(json.dumps(state))

    select(args, shard_dir, state_path, binding)
    migrated = json.loads(state_path.read_text())
    assert migrated['binding'] == binding
    selected = [row for path in sorted(shard_dir.glob('*.jsonl.zst')) for row in _decode(path)]
    assert selected == [_row('a0', 'pa')]


def test_selector_binding_includes_split_aliases_and_legacy_migration_is_semantic_pinned(tmp_path, monkeypatch):
    source = tmp_path / 'input.jsonl'
    source.write_text(json.dumps(_row('a0', 'pa')) + '\n')
    args = _args([source])
    original = selection_input_binding(args)
    monkeypatch.setattr(renderer, 'HELD_SPLIT_ALIASES', (*renderer.HELD_SPLIT_ALIASES, 'qa'))
    changed = selection_input_binding(args)
    assert changed['options']['selector_code_sha256'] != original['options']['selector_code_sha256']

    monkeypatch.setattr(renderer, 'HELD_SPLIT_ALIASES', ('validation', 'valid', 'val', 'dev'))
    binding = selection_input_binding(args)
    monkeypatch.setattr(renderer, 'LEGACY_SELECTOR_SEMANTICS_SHA256', '0' * 64)
    shard_dir, state_path = tmp_path / 'selected', tmp_path / 'selection-state.json'
    shard_dir.mkdir()
    select(args, shard_dir, state_path, binding)
    state = json.loads(state_path.read_text())
    legacy = dict(state['binding'])
    legacy_options = dict(legacy['options'])
    legacy_options['selector_code_sha256'] = '797e534147c258b5453c146365e53763133ac45ec468193b66fba442419bdccf'
    legacy['options'] = legacy_options
    legacy.pop('sha256', None)
    from natlang_neuralese.common.jsonio import canonical_json_str
    from natlang_neuralese.common.hashing import sha256_hex
    legacy['sha256'] = sha256_hex(canonical_json_str(legacy).encode())
    state['binding'] = legacy
    state_path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match='inputs or selector semantics changed'):
        select(args, shard_dir, state_path, binding)
