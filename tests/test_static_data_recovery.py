import hashlib
import json
import pytest

from scripts.create_training_pipeline import recipe as build_recipe, file_project_has_subfunctions


def recipe(repo, **kwargs):
    # These fixtures exercise static/reference wiring, not generated-job discovery.
    kwargs.setdefault('teacher_results_override', [path for path in
        (repo / 'runs/inline-curriculum').glob('*.results.jsonl')])
    return build_recipe(repo, **kwargs)


@pytest.fixture(autouse=True)
def isolated_catalog(tmp_path):
    (tmp_path / 'training').mkdir()
    (tmp_path / 'training/data_sources.json').write_text(json.dumps({
        'decisions': [], 'replacements': {}, 'required_default_inputs': []}))


def registry(monkeypatch, tmp_path):
    monkeypatch.setattr('scripts.create_training_pipeline.subprocess.check_output', lambda argv, **k:
                        json.dumps({'results': str(tmp_path / 'empty.results.jsonl'),
                                    'failure_candidates': str(tmp_path / 'empty.failures.jsonl'),
                                    'failure_manifest': str(tmp_path / 'empty.failures.manifest.json')}
                                   if 'snapshot-generated-training.mjs' in argv[1] else
                                   [{'id': 'fixture', 'generated_families': ['fixture'], 'source_families': []}]))


def test_existing_references_are_admitted_materialized_and_joined(tmp_path, monkeypatch):
    registry(monkeypatch, tmp_path)
    path = tmp_path / 'runs/inline-curriculum/ref-v1.results.jsonl'
    path.parent.mkdir(parents=True)
    path.write_text('{}\n')
    config = recipe(tmp_path)
    stages = {stage['id']: stage for stage in config['stages']}
    assert str(path) in stages['admit-existing-curriculum']['inputs']
    assert '${run}/runtime-host/frozen-runtime.json' in stages['admit-existing-curriculum']['inputs']
    assert '/runtime-host/scripts/' in stages['materialize-existing-curriculum']['command'][1]
    assert '${run}/existing-curriculum.turns.jsonl' in stages['prepare-teacher']['command']
    assert 'existing-curriculum' in stages['assemble-joint']['command']
    assert not any('--model-id' in stages[name]['command'] for name in
                   ('admit-existing-curriculum', 'materialize-existing-curriculum'))
    assert not recipe(tmp_path, teacher_results_override=[])['existing_teacher_results']


def test_helper_project_requires_current_files():
    ir = {'version': 'natlang.program/2', 'source_layout': {'root': 'f.nl', 'subfunctions': {'g': 'f/g.ts'}},
          'semantics': {'root': 'f.nl', 'files': {'f.nl': 'root', 'f/g.ts': 'helper'}}}
    assert file_project_has_subfunctions(ir)
    del ir['semantics']['files']['f/g.ts']
    assert not file_project_has_subfunctions(ir)
    assert not file_project_has_subfunctions({'semantics': {'root': {'$lambda': {'codebase': {'g': {}}}}}})


def test_retired_default_turn_snapshots_are_quarantined(tmp_path, monkeypatch):
    registry(monkeypatch, tmp_path)
    path = tmp_path / 'data/direct-code-2026-09-23/chunk-native-final.jsonl.turns.jsonl'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'task': {'program_ir': {'version': 'natlang.program/1'}}}) + '\n')
    config = recipe(tmp_path)
    prepare = next(stage for stage in config['stages'] if stage['id'] == 'prepare')
    assert str(path) not in prepare['inputs']
    assert config['excluded_legacy_turns'][0]['path'] == str(path)
    with pytest.raises(ValueError, match='must be replayed'):
        recipe(tmp_path, verified_turns_override=[path])
    path.write_text(json.dumps({'task': {'program_ir': {'version': 'natlang.program/2'}}}) + '\n')
    config = recipe(tmp_path)
    assert str(path) in next(stage for stage in config['stages'] if stage['id'] == 'prepare')['inputs']


def test_current_helper_capture_enters_recipe(tmp_path, monkeypatch):
    registry(monkeypatch, tmp_path)
    folder = tmp_path / 'data/direct-code-2026-09-23/unit-test-corpus/helper'
    folder.mkdir(parents=True)
    turns = folder / 'native-replay.jsonl.turns.jsonl'
    ir = {'version': 'natlang.program/2', 'source_layout': {'root': 'f.nl', 'subfunctions': {'g': 'f/g.ts'}},
          'semantics': {'root': 'f.nl', 'files': {'f.nl': 'root', 'f/g.ts': 'helper'}}}
    row = {'task': {'program_ir': ir}, 'outcome': {'accepted': True}}
    turns.write_text(json.dumps(row) + '\n')
    (folder / 'native-replay.jsonl').write_text(json.dumps(row) + '\n')
    (folder / 'tasks.jsonl').write_text(json.dumps({'function': {'helpers': ['helper']}}) + '\n')
    (folder / 'manifest.json').write_text(json.dumps({'native_replay': {'accepted': 1,
        'turns_sha256': hashlib.sha256(turns.read_bytes()).hexdigest()}}))
    config = recipe(tmp_path)
    assert str(turns) in next(stage for stage in config['stages'] if stage['id'] == 'prepare')['inputs']


def test_saved_self_contained_capture_is_replayed_before_preparation(tmp_path, monkeypatch):
    registry(monkeypatch, tmp_path)
    folder = tmp_path / 'data/direct-code-2026-09-23/unit-test-corpus/fixture'
    folder.mkdir(parents=True)
    turns = folder / 'native-replay.jsonl.turns.jsonl'
    turns.write_text(json.dumps({'task': {'program_ir': {'version': 'natlang.program/1'}}}) + '\n')
    (folder / 'tasks.jsonl').write_text(json.dumps({'function': {}}) + '\n')
    (folder / 'captures.jsonl').write_text('{}\n')
    (folder / 'native-replay.jsonl').write_text('{}\n')
    (folder / 'manifest.json').write_text(json.dumps({'native_replay': {'accepted': 1,
        'turns_sha256': hashlib.sha256(turns.read_bytes()).hexdigest()}}))
    config = recipe(tmp_path)
    stages = {stage['id']: stage for stage in config['stages']}
    replay = stages['replay-saved-source-0000']
    assert '/runtime-host/scripts/' in replay['command'][1]
    assert str(folder / 'captures.jsonl') in replay['inputs']
    assert '${run}/saved-source-0000.jsonl.turns.jsonl' in stages['prepare']['inputs']
    assert str(turns) not in stages['prepare']['inputs']
