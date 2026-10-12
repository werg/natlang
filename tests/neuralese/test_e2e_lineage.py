"""The end-to-end lineage (NEURALESE_DIRECT_V1.md "End-to-end stage"): recipe validation, acceptance and readiness."""
import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'training' / 'neuralese'))
from natlang_neuralese.common.hashing import sha256_file_hex  # noqa: E402
from natlang_neuralese.train import e2e_acceptance  # noqa: E402
from natlang_neuralese.train.recipe import load_recipe  # noqa: E402

RECIPES = Path(__file__).resolve().parents[2] / 'training' / 'neuralese' / 'recipes'


def launch_recipe(tmp_path, line='mellum', **extra):
    for name in ('neuralese-e2e-v1', 'neuralese-e2e-' + line + '-v1'):
        shutil.copy(RECIPES / (name + '.json'), tmp_path / (name + '.json'))
    manifest = tmp_path / 'export' / 'manifest.json'
    manifest.parent.mkdir()
    manifest.write_text('{}\n')
    recipe = {'schema': 'natlang.neuralese-training-recipe/1', 'id': 'launch', 'extends': 'neuralese-e2e-' + line + '-v1',
              'overrides': {'input_bindings': {'direct_artifact': {'path': str(manifest),
                                                                  'sha256': sha256_file_hex(manifest)}}, **extra}}
    path = tmp_path / 'launch.json'
    path.write_text(json.dumps(recipe))
    return path


@pytest.mark.parametrize('line', ['mellum', 'lfm'])
def test_a_launch_recipe_binding_an_export_validates(tmp_path, line):
    recipe = load_recipe(launch_recipe(tmp_path, line))
    assert [s['kind'] for s in recipe['stages']] == ['direct_runtime_qualification', 'raw_recurrence_training']
    e2e = recipe['stages'][1]['parameters']
    assert e2e['latent_optimizer'] == 'muonsr' and e2e['latent_step_in_backward'] is True and e2e['monitor']['gate']
    assert ('quantization' in recipe) == (line == 'mellum')


def test_a_mid_lineage_snapshot_holds_its_qat_progress(tmp_path):
    recipe = load_recipe(launch_recipe(tmp_path, quantization={'stage_progress': {'e2e': [0.02, 0.02]}}))
    assert recipe['quantization']['stage_progress'] == {'e2e': [0.02, 0.02]}


def test_an_e2e_lineage_needs_its_acceptance_and_runtime_stage(tmp_path):
    with pytest.raises(ValueError, match='e2e_acceptance'):
        load_recipe(launch_recipe(tmp_path, e2e_acceptance={'$delete': True}))
    (tmp_path / 'b').mkdir()
    path = launch_recipe(tmp_path / 'b', stages={'e2e': {'requires': []}})
    with pytest.raises(ValueError, match='direct_runtime_qualification'):
        load_recipe(path)


def test_acceptance_compares_values_with_tokens_and_keeps_missing_evidence_pending():
    declaration = json.loads((RECIPES / 'neuralese-e2e-v1.json').read_text())['e2e_acceptance']
    held = {'written': 1.0, 'crisp': 1.1, 'shuffled': 1.2, 'zeroed': 1.3, 'n': 40}
    criteria, _ = e2e_acceptance.evaluate(declaration, {'written-trained': held, 'drift_guard': {'stopped': False}})
    assert {k: c['status'] for k, c in criteria.items()} == {
        'paired_readers': 'pass', 'v_minus_t_ce': 'pass', 'v_margin_over_shuffled': 'pass',
        'v_margin_over_zeroed': 'pass', 'task_success': 'pending', 'no_drift_stop': 'pass'}
    criteria, _ = e2e_acceptance.evaluate(declaration, {'written-trained': {**held, 'written': 1.15},
                                                        'task_success': {'v_success': 0.5, 't_success': 0.5},
                                                        'drift_guard': {'stopped': False}})
    assert criteria['v_minus_t_ce']['status'] == 'fail' and criteria['task_success']['status'] == 'pass'


def test_readiness_needs_a_held_ce_plateau_on_one_held_set(tmp_path):
    declaration = json.loads((RECIPES / 'neuralese-e2e-v1.json').read_text())['parent_readiness']
    stage = tmp_path / 'direct_sft'
    stage.mkdir()
    rows = [{'step': 0, 'ce': 2.0, 'base_ce': 2.0}, {'step': 400, 'ce': 1.0, 'base_ce': 1.5},
            {'step': 800, 'ce': 0.95, 'base_ce': 1.5}]
    (stage / 'eval.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    (stage / 'drift.jsonl').write_text(json.dumps({'step': 800, 'behaviour_drift': {'guard': 'warn'}}) + '\n')
    result = e2e_acceptance.readiness(declaration, stage)
    assert not result['ready'] and result['criteria']['held_ce_plateau']['status'] == 'pending'
    rows += [{'step': 1200, 'ce': 0.94, 'base_ce': 1.5}, {'step': 1600, 'ce': 0.935, 'base_ce': 1.5}]
    (stage / 'eval.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    result = e2e_acceptance.readiness(declaration, stage)
    assert not result['ready'] and result['criteria']['held_ce_plateau']['status'] == 'fail'  # 1.0 -> 0.935: 6.5%
    rows += [{'step': 2000, 'ce': 0.932, 'base_ce': 1.5}]
    (stage / 'eval.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    result = e2e_acceptance.readiness(declaration, stage)
    assert result['ready'] and result['criteria']['task_success']['status'] == 'reported'
    (stage / 'drift.jsonl').write_text(json.dumps({'step': 2000, 'behaviour_drift': {'guard': 'stop'}}) + '\n')
    assert not e2e_acceptance.readiness(declaration, stage)['ready']
