"""The end-to-end lineage (NEURALESE_DIRECT_V1.md "End-to-end stage"): recipe validation and acceptance."""
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
    aligned = tmp_path / 'aligned' / 'aligned-projection.pt'
    aligned.parent.mkdir()
    aligned.write_bytes(b'aligned P fitted on the export')
    recipe = {'schema': 'natlang.neuralese-training-recipe/1', 'id': 'launch', 'extends': 'neuralese-e2e-' + line + '-v1',
              'overrides': {'input_bindings': {'direct_artifact': {'path': str(manifest),
                                                                  'sha256': sha256_file_hex(manifest)},
                                               'aligned_projection': {'path': str(aligned),
                                                                      'sha256': sha256_file_hex(aligned)}}, **extra}}
    path = tmp_path / 'launch.json'
    path.write_text(json.dumps(recipe))
    return path


@pytest.mark.parametrize('line', ['mellum', 'lfm'])
def test_a_launch_recipe_binding_an_export_validates(tmp_path, line):
    recipe = load_recipe(launch_recipe(tmp_path, line))
    assert [s['kind'] for s in recipe['stages']] == ['direct_runtime_qualification', 'raw_recurrence_training']
    e2e = recipe['stages'][1]['parameters']
    assert e2e['latent_optimizer'] == 'muonsr' and e2e['latent_step_in_backward'] is True and e2e['monitor']['gate']
    assert 'quantization' in recipe  # lockstep QAT in the shared base for both lines (1f1042c3)


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
    arm = {'eligible': True, 'case_indices': [0, 3], 'cases': 2}
    summary = {'written-trained': held, 'drift_guard': {'stopped': False},
               'task_eval-init': {**arm, 'task_successes': 1, 'task_success_rate': 0.5},
               'task_eval': {**arm, 'task_successes': 1, 'task_success_rate': 0.5,
                             'case_successes': {'0': True, '3': False}}}
    criteria, _ = e2e_acceptance.evaluate(declaration, summary)
    assert {k: c['status'] for k, c in criteria.items()} == {
        'paired_readers': 'pass', 'v_minus_t_ce': 'pass', 'v_margin_over_shuffled': 'pass',
        'v_margin_over_zeroed': 'pass', 'task_success': 'pending', 'task_success_vs_parent': 'pass',
        'no_drift_stop': 'pass'}
    summary['task_eval'] = {**summary['task_eval'], 'task_successes': 0, 'task_success_rate': 0.0}
    summary['task_eval_neuralese'] = {'fused_cases': [0], 'fused_case_successes': {'0': True}}
    summary['written-trained'] = {**held, 'written': 1.15}
    criteria, _ = e2e_acceptance.evaluate(declaration, summary)
    assert criteria['v_minus_t_ce']['status'] == 'fail' and criteria['task_success_vs_parent']['status'] == 'fail'
    assert criteria['task_success']['status'] == 'pass'  # V 1/1 against T 1/1 on the fused case


def test_family_interleaved_order_spreads_families_and_is_seeded():
    from natlang_neuralese.train.trajectories import family_interleaved_order, record_family
    records = ([{'id': f'a{i}', 'task_family': 'a'} for i in range(60)] +
               [{'id': f'b{i}', 'task_family': 'b'} for i in range(20)] +
               [{'id': f'c{i}', 'task_family': 'c'} for i in range(4)])
    order = family_interleaved_order(records, 7, 0)
    assert sorted(r['id'] for r in order) == sorted(r['id'] for r in records)
    assert order == family_interleaved_order(records, 7, 0) and order != family_interleave_other(records)
    for start in range(0, len(order) - 20, 10):  # every stretch of 21 updates sees family b, no run is one family
        assert 'b' in {record_family(r) for r in order[start:start + 21]}
    assert family_interleaved_order(records, 7, 1) != order  # a new permutation per epoch


def family_interleave_other(records):
    from natlang_neuralese.train.trajectories import family_interleaved_order
    return family_interleaved_order(records, 8, 0)

def test_the_manifold_gate_checks_the_aligned_projection_of_the_exact_export(tmp_path):
    """Owner 2026-10-13: no line enters e2e with an unaligned projection. The shared recipe declares the thresholds on
    its direct_runtime stage (both lines through line parity); the stage reads the aligned P with its e2e stage."""
    from natlang_neuralese.train.recipe import require_gate
    recipe = load_recipe(launch_recipe(tmp_path))
    runtime, e2e = recipe['stages']
    declaration = runtime['parameters']['projection_on_manifold']
    assert runtime['inputs']['aligned_projection'] == e2e['inputs']['aligned_projection'] == 'aligned_projection'
    report = {'held': {'all': {'nearest_chosen': 0.995, 'cos_chosen': 0.985}},
              'functional_kl': {'all': {'kl_mean': 0.001}}}
    statuses = lambda r: {k: c['status'] for k, c in e2e_acceptance.projection_on_manifold(declaration, r).items()}
    assert set(statuses(report).values()) == {'pass'}
    assert statuses({**report, 'held': {'all': {'nearest_chosen': 0.9, 'cos_chosen': 0.985}}})['nearest_chosen'] == 'fail'
    assert statuses({'held': report['held']})['functional_kl'] == 'pending'  # missing evidence never passes
    qualified = {'schema': 'natlang.neuralese-direct-runtime-qualification/1', 'runtime_qualified': True}
    with pytest.raises(ValueError, match='manifold'):
        require_gate({**qualified, 'projection_on_manifold': {'passed': False, 'criteria': {}}},
                     'direct_runtime_qualification')
    require_gate({**qualified, 'projection_on_manifold': {'passed': True}}, 'direct_runtime_qualification')
    (tmp_path / 'c').mkdir()
    with pytest.raises(ValueError, match='aligned_projection'):
        load_recipe(launch_recipe(tmp_path / 'c', stages={'e2e': {'inputs': {'aligned_projection': {'$delete': True}}}}))
