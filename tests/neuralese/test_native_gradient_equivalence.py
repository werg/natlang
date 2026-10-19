"""Carried measurements retain their old runtime and reject changed numerics."""
import copy
import json
from argparse import ArgumentParser
from pathlib import Path

import pytest
import torch
from natlang_neuralese.train import native_gradient_equivalence as eq
from natlang_neuralese.train import native_resident as res
from natlang_neuralese.train.native_source_classification import _binding
from natlang_neuralese.eval.native_gradients import gradient_metrics, SCHEMA


def fixture(tmp_path, monkeypatch):
    old = {'model/kernel.py': 'a', 'train/metadata.py': 'b'}
    new = {**old, 'train/metadata.py': 'c'}
    changes = {'train/metadata.py': {'old': 'b', 'new': 'c'}}
    monkeypatch.setattr(eq, 'APPROVED', {(res.digest(old), res.digest(new)): changes})
    package = tmp_path / 'new' / 'training/neuralese/natlang_neuralese'
    monkeypatch.setattr(res, 'code_inventory', lambda p: new if Path(p) == package else old)
    parent = {'path': '/immutable/parent.pt', 'sha256': 'f' * 64, 'bytes': 99}
    names = ['w', 'adapter', 'input_rows', 'output_rows']
    inventory = [{'name': name, 'shape': [1], 'dtype': 'torch.float32'} for name in names]
    roles = dict(zip(('backbone', 'adapter', 'learned_input', 'learned_output'), [[n] for n in names]))
    identity = {'checkpoint': parent['sha256'], 'model': 'same'}
    comparison = gradient_metrics(torch.ones(1), torch.ones(1), atol=1e-5, rtol=1e-4)
    complete = {'schema': SCHEMA, 'passed': True, 'native_identity': identity,
        'inventory': inventory, 'inventory_sha256': res.digest(inventory),
        'parameter_grads_preserved': True, 'optimizer_updates': 0, 'atol': 1e-5, 'rtol': 1e-4,
        'trainable_parameter_count': 4, 'derivative_comparison_count': 5,
        'comparisons': {name: dict(comparison) for name in ['input_embeddings', *names]},
        'roles': roles, 'case': {'fixture': 'same-gradient-program'},
        'model_config': {'dtype': 'fp32'}}
    complete['case_sha256'] = res.digest(complete['case'])
    complete['model_config_sha256'] = res.digest(complete['model_config'])
    parser = ArgumentParser()
    parser.add_argument('--forward-atol', type=float); parser.add_argument('--gradient-atol', type=float)
    parser.add_argument('--out'); parser.add_argument('--recipe-binding')
    parser.add_argument('--source-classification-migration'); parser.add_argument('--resident-data-directory')
    parser.add_argument('--checkpoint-input-offload-directory'); parser.add_argument('--checkpoint-publication-readonly-input-roots')
    parser.add_argument('--native-linear-precision'); parser.add_argument('--continue-from')
    args = ['--forward-atol', '0.001', '--gradient-atol', '0.00001', '--native-linear-precision', 'bf16x6',
            '--continue-from', parent['path']]
    prior = {'runtime': str(tmp_path / 'old'), 'runtime_code_sha256': res.digest(old),
             'parent': parent, 'out': str(tmp_path / 'q1/learner'), 'trainer_arguments': [*args, '--out', str(tmp_path / 'q1/learner')]}
    current = {**prior, 'runtime': str(tmp_path / 'new'), 'runtime_code_sha256': res.digest(new),
               'out': str(tmp_path / 'q2/learner'), 'trainer_arguments': [*args, '--out', str(tmp_path / 'q2/learner')]}
    docs = {'gate_spec': prior,
        'performance': {'passed': True, 'parent': parent, 'runtime_code_sha256': res.digest(old), 'optimizer_updates_from_measurement': 0},
        'inventory': {'parent': parent, 'runtime_code_sha256': res.digest(old), 'inventory': inventory,
                      'inventory_sha256': res.digest(inventory), 'actual_count': 4, 'roles': roles},
        'actor': {'passed': True, 'parent': parent, 'candidate_identity': identity, 'optimizer_updates': 0},
        'precision': {'parent_weights_sha256': parent['sha256'], 'runtime_reference': {
            'checkpoint_sha256': parent['sha256'], 'passed': True, 'native_identity': identity,
            'gradient_scope': 'complete-production-trainables', 'complete_gradient_qualification': complete, 'forward_atol': 0.001}},
        'ffn': {'passed': True}, 'pre_capacity': {'passed': True, 'parent': parent, 'new_optimizer_updates': 0},
        'failure': {'error': 'geometry evidence source prefix/count/metadata differs'}}
    def request():
        refs = {}
        for name, value in docs.items():
            p = tmp_path / (name + '.json'); p.write_text(json.dumps(value)); refs[name] = _binding(p)
        p = tmp_path / 'request.json'; p.write_text(json.dumps({'schema': 'natlang.gradient-metadata-equivalence-request/1', **refs}))
        return _binding(p)
    return old, new, package, inventory, roles, identity, parser, current, docs, request


def test_carried_proof_preserves_original_runtime_and_report(tmp_path, monkeypatch):
    old, new, package, inventory, roles, identity, parser, current, docs, request = fixture(tmp_path, monkeypatch)
    original = copy.deepcopy(docs['precision'])
    result = eq.carry_qualification(request(), current, package=package, inventory=inventory,
                                   roles=roles, actor_identity=identity, parser=parser)
    assert result['precision_transition'] == original
    assert result['equivalence']['qualified_runtime_code'] == res.digest(old)
    assert result['equivalence']['executing_runtime_code'] == res.digest(new)
    assert result['numerical_measurement_performed'] is False and result['optimizer_updates'] == 0


@pytest.mark.parametrize('change', ['model-code', 'metadata-code', 'parent', 'inventory', 'precision',
    'failed-phase', 'failed-comparison', 'role', 'actor', 'count'])
def test_changes_cannot_relabel_old_pass_as_current_qualification(tmp_path, monkeypatch, change):
    old, new, package, inventory, roles, identity, parser, current, docs, request = fixture(tmp_path, monkeypatch)
    if change == 'model-code': new['model/kernel.py'] = 'changed'
    elif change == 'metadata-code': new['train/metadata.py'] = 'unreviewed'
    elif change == 'parent': current['parent'] = {**current['parent'], 'sha256': '0' * 64}
    elif change == 'inventory': inventory[0] = {**inventory[0], 'dtype': 'torch.bfloat16'}
    elif change == 'precision': current['trainer_arguments'][current['trainer_arguments'].index('bf16x6')] = 'ieee'
    elif change == 'failed-phase': docs['performance']['passed'] = False
    elif change == 'failed-comparison': docs['precision']['runtime_reference']['complete_gradient_qualification']['comparisons']['w']['passed'] = False
    elif change == 'role': roles['backbone'] = []
    elif change == 'actor': identity = {**identity, 'model': 'different'}
    elif change == 'count': docs['precision']['runtime_reference']['complete_gradient_qualification']['derivative_comparison_count'] = 4
    with pytest.raises(ValueError):
        eq.carry_qualification(request(), current, package=package, inventory=inventory,
                               roles=roles, actor_identity=identity, parser=parser)


def baseline_fixture(tmp_path, monkeypatch):
    values = fixture(tmp_path, monkeypatch)
    old, new, package, inventory, roles, identity, parser, current, docs, _ = values
    parser.add_argument('--ffn-chunk-tokens', type=int)
    docs['gate_spec']['trainer_arguments'].extend(['--ffn-chunk-tokens', '4096'])
    current['trainer_arguments'].extend(['--ffn-chunk-tokens', '0'])
    baseline = copy.deepcopy(docs['precision']['runtime_reference'])
    baseline.update(consumer_ce_abs=0., forward_max_abs=0., channel_restore_exact=True,
                    backbone_optimizer_updates=0)
    baseline['complete_gradient_qualification']['model_config']['ffn_chunk_tokens'] = 0
    baseline['complete_gradient_qualification']['model_config_sha256'] = res.digest(baseline['complete_gradient_qualification']['model_config'])
    candidate = copy.deepcopy(baseline)
    candidate['complete_gradient_qualification']['model_config']['ffn_chunk_tokens'] = 4096
    candidate['complete_gradient_qualification']['model_config_sha256'] = res.digest(candidate['complete_gradient_qualification']['model_config'])
    docs['ffn'] = {'schema': 'natlang.native-ffn-chunk-transition-complete/1',
        'execution_setting': 'ffn_chunk_tokens', 'previous_mode': 0, 'candidate_mode': 4096,
        'passed': False, 'cross_mode_passed': False, 'cached_reference_passed': True,
        'parent_checkpoint_sha256': current['parent']['sha256'], 'optimizer_updates': 0,
        'source_state_preserved': True, 'baseline_cached_reference': baseline,
        'candidate_cached_reference': candidate}
    docs['performance']['passed'] = False
    docs['failure']['error'] = 'actual-parent FFN boundary qualification failed'
    def request():
        refs = {}
        for name in ('gate_spec', 'performance', 'inventory', 'actor', 'ffn', 'failure'):
            p = tmp_path / ('baseline-' + name + '.json'); p.write_text(json.dumps(docs[name])); refs[name] = _binding(p)
        p = tmp_path / 'baseline-request.json'
        p.write_text(json.dumps({'schema': 'natlang.qualified-baseline-gradient-selection/1',
            'selection': {'execution_setting': 'ffn_chunk_tokens', 'mode': 0,
                          'certificate': 'baseline_cached_reference'}, **refs}))
        return _binding(p)
    return values[:-1], request


def test_measured_baseline_selection_keeps_rejected_transition_failed(tmp_path, monkeypatch):
    values, request = baseline_fixture(tmp_path, monkeypatch)
    _, _, package, inventory, roles, identity, parser, current, docs = values
    result = eq.select_qualified_baseline(request(), current, package=package, inventory=inventory,
        roles=roles, actor_identity=identity, config={'dtype': 'fp32'}, parser=parser)
    assert result['runtime_certificate'] == docs['ffn']['baseline_cached_reference']
    assert result['rejected_transition_passed'] is False and result['numerical_measurement_performed'] is False
    assert result['selection']['mode'] == 0 and result['optimizer_updates'] == 0
    assert docs['ffn']['passed'] is False


@pytest.mark.parametrize('change', ['mode', 'other-option', 'config', 'model-code', 'parent',
                                   'inventory', 'failed-baseline', 'failed-coordinate', 'actor'])
def test_baseline_selection_rejects_unmeasured_execution(tmp_path, monkeypatch, change):
    values, request = baseline_fixture(tmp_path, monkeypatch)
    old, new, package, inventory, roles, identity, parser, current, docs = values
    config = {'dtype': 'fp32'}
    if change == 'mode': current['trainer_arguments'][-1] = '4096'
    elif change == 'other-option': current['trainer_arguments'][current['trainer_arguments'].index('bf16x6')] = 'ieee'
    elif change == 'config': config['dtype'] = 'bf16'
    elif change == 'model-code': new['model/kernel.py'] = 'changed'
    elif change == 'parent': current['parent'] = {**current['parent'], 'sha256': '0' * 64}
    elif change == 'inventory': inventory[0] = {**inventory[0], 'dtype': 'torch.bfloat16'}
    elif change == 'failed-baseline': docs['ffn']['baseline_cached_reference']['passed'] = False
    elif change == 'failed-coordinate': docs['ffn']['baseline_cached_reference']['complete_gradient_qualification']['comparisons']['w']['passed'] = False
    elif change == 'actor': identity = {**identity, 'model': 'other'}
    with pytest.raises(ValueError):
        eq.select_qualified_baseline(request(), current, package=package, inventory=inventory,
            roles=roles, actor_identity=identity, config=config, parser=parser)
