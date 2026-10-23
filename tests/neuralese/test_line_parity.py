"""Exact Mellum/LFM parity for native production SFT recipes."""
import json
import shutil
from pathlib import Path

import pytest
from natlang_neuralese.train import line_parity

RECIPES = Path(__file__).resolve().parents[2] / 'training' / 'neuralese' / 'recipes'


def test_declared_native_lines_match_the_shared_base():
    assert line_parity.check() == []


def test_undeclared_native_training_change_fails():
    parity = line_parity.load_parity()
    _, config = line_parity.effective_config({'recipe': 'native-state-production-lfm-20261015-v1'})
    config = {**config, 'parameters.batch_tokens': 65536}
    errors, _ = line_parity.compare(parity, 'lfm', config)
    assert len(errors) == 1 and 'parameters.batch_tokens' in errors[0]


def test_model_specific_graphs_require_the_shared_admitted_source_and_split():
    parity = line_parity.load_parity()
    _, config = line_parity.effective_config({'recipe': 'native-state-production-mellum-20261015-v1'})
    config = {**config, 'inputs.native_graphs': '0' * 64}
    errors, _ = line_parity.compare(parity, 'mellum', config)
    assert any('shared admitted source membership and exact train/held split' in error for error in errors)


def test_both_native_production_lines_are_declared():
    declaration = line_parity.load_parity()['stages']['native_state_sft']
    assert set(declaration['lines']) == set(line_parity.LINES)
    assert all(spec.get('recipe') for spec in declaration['lines'].values())


def test_recipe_resolver_rejects_projected_stage_in_parity(tmp_path):
    (tmp_path / 'training/neuralese/recipes').mkdir(parents=True)
    shutil.copy(RECIPES / 'native-state-production-lfm-20261015-v1.json',
                tmp_path / 'training/neuralese/recipes/native.json')
    with pytest.raises(ValueError, match='native SFT stages only'):
        line_parity._stage_view({'stages': [{'kind': 'neuralese_direct'}]})
