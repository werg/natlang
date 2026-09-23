import hashlib
import json
import pytest

from scripts.create_training_pipeline import recipe
from scripts.create_student_improvement_pipeline import improvement_pipeline
from scripts.run_training_pipeline import run_pipeline


def base(tmp_path):
    config = recipe(tmp_path, python='python')
    base_recipe = tmp_path / 'base-recipe.json'
    base_recipe.write_text(json.dumps(config))
    run = tmp_path / 'base-run'
    checkpoint = run / 'train-teacher/checkpoint'
    (checkpoint / 'weights').mkdir(parents=True)
    (checkpoint / 'weights/adapter_model.safetensors').write_bytes(b'adapter')
    (checkpoint / 'state.json').write_text(json.dumps({'trained_examples': 1,
                                                       'corpus': {'target_examples': 1}}))
    (run / 'prepared').mkdir()
    (run / 'prepared/splits.json').write_text('{"groups":{}}')
    (run / 'pipeline-state.json').write_text(json.dumps({
        'config_sha256': hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        'stages': {'train-teacher': {'status': 'complete'}}}))
    programs = tmp_path / 'programs.jsonl'
    programs.write_text('{}\n')
    return base_recipe, run, programs


def test_round_is_single_durable_pipeline_with_frozen_runtime(tmp_path):
    base_recipe, base_run, programs = base(tmp_path)
    run = tmp_path / 'round-1'
    config = improvement_pipeline(base_recipe, base_run, programs, run,
                                  'http://student:8080', 'student-1',
                                  'http://teacher:8081', 'teacher-1')
    stages = {stage['id']: stage for stage in config['stages']}
    assert list(stages) == ['freeze-runtime', 'collect-student', 'build-hard-states',
                            'collect-corrections', 'materialize-student', 'materialize-teacher',
                            'build-preferences', 'combine-verified', 'validate-corrections', 'prepare-correction',
                            'render-correction', 'audit-correction', 'train-correction']
    assert config['run_directory'] == str(run)
    for name in ('collect-student', 'build-hard-states', 'collect-corrections',
                 'materialize-student', 'materialize-teacher', 'build-preferences'):
        assert '${run}/runtime-host/frozen-runtime.json' in stages[name]['inputs']
        assert any('/runtime-host/' in item for item in stages[name]['command'])
    assert '--collection-role' in stages['collect-student']['command']
    assert '--handoff-queue' in stages['collect-corrections']['command']
    assert '--all' in stages['collect-corrections']['command']
    assert str(run / 'verified-turns.jsonl') in stages['validate-corrections']['inputs']
    assert str(base_run / 'prepared/splits.json') in stages['prepare-correction']['inputs']
    assert str(base_run / 'train-teacher/checkpoint/weights') in stages['train-correction']['command']
    config_path = tmp_path / 'round-recipe.json'
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='different run directory'):
        run_pipeline(config_path, tmp_path / 'wrong-run', until='freeze-runtime')
