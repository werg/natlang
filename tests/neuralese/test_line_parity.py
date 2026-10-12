"""Line parity (plans/neuralese/LINE_PARITY.md): the declared lines match the shared base up to declared differences."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'training' / 'neuralese'))
from natlang_neuralese.train import line_parity  # noqa: E402
from natlang_neuralese.train.recipe import main as recipe_main  # noqa: E402

RECIPES = Path(__file__).resolve().parents[2] / 'training' / 'neuralese' / 'recipes'


def test_declared_lines_match_the_shared_base():
    assert line_parity.check() == []


def test_an_undeclared_difference_fails():
    parity = line_parity.load_parity()
    kind, config = line_parity.effective_config({'recipe': 'neuralese-direct-lfm-v3'})
    config = {**config, 'parameters.tokens': 8192}
    errors, _ = line_parity.compare(parity, 'direct', 'lfm', kind, config)
    assert len(errors) == 1 and 'parameters.tokens' in errors[0]


def test_unequal_supervision_fails():
    parity = line_parity.load_parity()
    kind, config = line_parity.effective_config({'recipe': 'neuralese-direct-lfm-v3'})
    config = {**config, 'stages.direct_sft.derived.supervision': '3.0 epochs'}
    errors, _ = line_parity.compare(parity, 'direct', 'lfm', kind, config)
    assert errors and 'supervision' in errors[0]


def test_direct_phases_are_compared_stage_by_stage_in_epochs():
    kind, config = line_parity.effective_config({'recipe': 'neuralese-direct-mellum-v4'})
    assert config['stages.direct_warmup.derived.supervision'] == '0.5 epochs'
    assert config['stages.direct_sft.derived.supervision'] == '1.0 epochs'
    assert config['stages.direct_trajectory.derived.supervision'] == '1.0 epochs'
    assert 'stages.direct_sft.parameters.steps' not in config


def test_a_data_substitution_needs_a_registered_twin():
    parity = line_parity.load_parity()
    kind, config = line_parity.effective_config({'recipe': 'neuralese-direct-lfm-v3'})
    errors, _ = line_parity.compare(parity, 'direct', 'mellum', kind, {**config, 'inputs.text_data': '0' * 64,
                                                                         'parameters.monitor.gate.reference.gate_pass': 1})
    assert any('tokenizer twin' in e for e in errors)


def test_fresh_direct_stage_launch_requires_a_parity_declaration():
    with pytest.raises(ValueError, match='parity'):
        line_parity.check_declared_stage({'schema': 'natlang.neuralese-declared-direct-stage/1'})
    line_parity.check_declared_stage({'parity': 'diagnostic'})


def test_cli_cannot_replace_declared_data(tmp_path, capsys):
    with pytest.raises(SystemExit):
        recipe_main(['--recipe', str(RECIPES / 'neuralese-direct-lfm-v2.json'), '--heads', str(tmp_path / 'h.pt'),
                     '--records', str(tmp_path / 'r.jsonl'), '--out', str(tmp_path / 'out')])
    assert 'declares its data inputs' in capsys.readouterr().err


def test_both_lines_progress_in_lockstep():
    """Lockstep (plans/neuralese/TRAINING_RECIPE.md): every stage of the progression (direct, then e2e) declares a
    recipe for both lines; a pending line or a stage only one line runs fails CI."""
    parity = line_parity.load_parity()
    assert list(parity['stages']) == ['direct', 'e2e']
    for stage, declaration in parity['stages'].items():
        assert set(declaration['lines']) == set(line_parity.LINES), stage
        assert all(spec.get('recipe') for spec in declaration['lines'].values()), stage


def test_e2e_lines_share_acceptance_and_readiness():
    parity = line_parity.load_parity()
    for line in line_parity.LINES:
        kind, config = line_parity.effective_config(parity['stages']['e2e']['lines'][line])
        assert kind == 'direct_runtime_qualification+raw_recurrence_training'
        assert config['parent_readiness.held_ce_plateau.evaluations'] == 3
        assert config['e2e_acceptance.paired_reader.max_v_minus_t_ce_nats'] == 0.0
    kind, config = line_parity.effective_config({'recipe': 'neuralese-e2e-lfm-v1'})
    config = {**config, 'parent_readiness.held_ce_plateau.max_relative_improvement': 0.1}
    errors, _ = line_parity.compare(parity, 'e2e', 'lfm', kind, config)
    assert len(errors) == 1 and 'parent_readiness' in errors[0]
