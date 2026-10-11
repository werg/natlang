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
    kind, config = line_parity.effective_config({'recipe': 'neuralese-direct-lfm-v2'})
    config = {**config, 'parameters.tokens': 8192}
    errors, _ = line_parity.compare(parity, 'direct', 'lfm', kind, config)
    assert len(errors) == 1 and 'parameters.tokens' in errors[0]


def test_unequal_supervision_fails():
    parity = line_parity.load_parity()
    kind, config = line_parity.effective_config({'recipe': 'neuralese-direct-lfm-v2'})
    config = {**config, 'derived.supervised_positions': 256 * 1000}
    errors, _ = line_parity.compare(parity, 'direct', 'lfm', kind, config)
    assert errors and 'supervised_positions' in errors[0]


def test_a_data_substitution_needs_a_registered_twin():
    parity = line_parity.load_parity()
    kind, config = line_parity.effective_config({'recipe': 'neuralese-direct-lfm-v2'})
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
