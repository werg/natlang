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


def test_e2e_lines_share_acceptance():
    parity = line_parity.load_parity()
    for line in line_parity.LINES:
        kind, config = line_parity.effective_config(parity['stages']['e2e']['lines'][line])
        assert kind == 'direct_runtime_qualification+raw_recurrence_training'
        assert config['e2e_acceptance.paired_reader.max_v_minus_t_ce_nats'] == 0.0
        assert not any(key.startswith('parent_readiness') for key in config)
    kind, config = line_parity.effective_config({'recipe': 'neuralese-e2e-lfm-v1'})
    config = {**config, 'e2e_acceptance.paired_reader.max_v_minus_t_ce_nats': 0.1}
    errors, _ = line_parity.compare(parity, 'e2e', 'lfm', kind, config)
    assert len(errors) == 1 and 'e2e_acceptance' in errors[0]


def _repo_copy(tmp_path):
    import shutil
    repo = Path(__file__).resolve().parents[2]
    (tmp_path / 'training' / 'neuralese').mkdir(parents=True)
    shutil.copytree(RECIPES, tmp_path / 'training' / 'neuralese' / 'recipes')
    shutil.copy(repo / 'training' / 'neuralese' / 'line-parity.json', tmp_path / 'training' / 'neuralese')
    shutil.copy(repo / 'training' / 'neuralese_corpora.json', tmp_path / 'training')
    return tmp_path


def test_both_lines_declare_their_running_chain():
    parity = line_parity.load_parity()
    for line in line_parity.LINES:
        assert parity['stages']['direct']['lines'][line].get('active'), line


def test_a_key_added_to_one_running_chain_fails(tmp_path):
    """Owner 2026-10-13: a setting Pop added only to its running LFM handoff chain (task_eval.identical_turn_limit)
    went unnoticed because only next-launch recipes were compared. The running chains are checked too."""
    repo = _repo_copy(tmp_path)
    parity_file = repo / 'training' / 'neuralese' / 'line-parity.json'
    parity = json.loads(parity_file.read_text())
    leaf = line_parity._recipe_file(parity['stages']['direct']['lines']['lfm']['active'], repo)
    recipe = json.loads(leaf.read_text())  # one more handoff setting, on the LFM chain only
    recipe['overrides'].setdefault('stage_parameter_defaults', {}).setdefault('neuralese_direct', {}).setdefault(
        'task_eval', {})['identical_turn_limit'] = 3
    leaf.write_text(json.dumps(recipe))
    errors = line_parity.check(repo)
    assert len(errors) == 1, errors
    assert errors[0].startswith(recipe['id'] + ': direct: stages.*.parameters.task_eval.identical_turn_limit differs '
                                'between lines (mellum=')
    assert 'lfm=3' in errors[0] and 'move it to the shared recipe or declare it in line-parity' in errors[0]


def test_every_leaf_setting_is_compared():
    kind, config = line_parity.effective_config({'recipe': parity_active('lfm')})
    assert any(key.startswith('runtime.package_code.') for key in config)
    assert 'outputs.directory' in config


def parity_active(line):
    return line_parity.load_parity()['stages']['direct']['lines'][line]['active']


def test_line_specific_entries_expire():
    import datetime
    errors = line_parity.check(today=datetime.date(2099, 1, 1))
    assert errors and all('expired on' in e for e in errors)


def test_line_specific_entries_need_a_date_and_expiry(tmp_path):
    repo = _repo_copy(tmp_path)
    parity_file = repo / 'training' / 'neuralese' / 'line-parity.json'
    parity = json.loads(parity_file.read_text())
    parity['line_specific'].append({'stage': 'direct', 'lines': ['lfm'], 'key': 'parameters.seed', 'reason': 'x'})
    parity_file.write_text(json.dumps(parity))
    with pytest.raises(ValueError, match='expires'):
        line_parity.load_parity(repo)


def test_e2e_carries_the_direct_base_qat_on_both_lines():
    """QAT in every stage, in lockstep: the e2e stage of both lines trains the direct base's shared points, held at
    the parent's progress; a line drops or changes them only by a declared difference."""
    from natlang_neuralese.train.recipe import resolve_recipe_data
    direct = resolve_recipe_data(RECIPES / 'neuralese-direct-v3.json')['quantization']
    shared = {p['name']: p for p in direct['points']}
    for line in line_parity.LINES:
        q = resolve_recipe_data(RECIPES / f'neuralese-e2e-{line}-v1.json')['quantization']
        assert q['stage_progress'] == {'e2e': [1.0, 1.0]} and q['loss'] == direct['loss']
        points = {p['name']: p for p in q['points']}
        assert all(points.get(name) == point for name, point in shared.items()), line
    parity = line_parity.load_parity()
    kind, config = line_parity.effective_config({'recipe': 'neuralese-e2e-lfm-v1'})
    errors, _ = line_parity.compare(parity, 'e2e', 'lfm', kind,
                                    {k: v for k, v in config.items() if '.quantization.' not in k})
    assert errors and all('quantization' in e for e in errors)
