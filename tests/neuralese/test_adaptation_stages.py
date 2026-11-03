import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

from natlang_neuralese.train import adaptation_stages, recipe
from natlang_neuralese.tune import structure
from natlang_neuralese.tune.stage import entry_state, paired_gain


def bound(path):
    return {'path': str(path), 'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def write_recipe(tmp_path, *, adaptation_parameters=None, optimizer_parameters=None, backbone_inputs=True):
    files = {}
    for name in ('graphs', 'held', 'programs', 'protection') + (('checkpoint', 'selection') if backbone_inputs else ()):
        files[name] = tmp_path / (f'{name}.jsonl' if name in ('graphs', 'held') else f'{name}.json')
        files[name].write_text(json.dumps({'file': name}))
    adaptation_inputs = {'programs': 'programs', 'protection': 'protection'}
    if backbone_inputs:
        adaptation_inputs.update(continue_from='checkpoint', parent_selection='selection')
    value = {'schema': 'natlang.neuralese-training-recipe/1', 'id': 'adaptation-test',
             'identity_qualification': recipe.NATIVE_IDENTITY_QUALIFICATION,
             'input_bindings': {name: bound(path) for name, path in files.items()},
             'stages': [
                 {'id': 'joint', 'kind': 'native_state_sft', 'requires': [],
                  'inputs': {'native_graphs': 'graphs', 'held_native_graphs': 'held'}, 'parameters': {'epochs': 1}},
                 {'id': 'adapt', 'kind': 'program_adaptation', 'requires': ['joint'], 'inputs': adaptation_inputs,
                  'parameters': adaptation_parameters or {}},
                 {'id': 'optimizer', 'kind': 'learned_optimizer', 'requires': ['adapt'],
                  'inputs': {'protection': 'protection', **({'continue_from': 'checkpoint', 'parent_selection': 'selection'}
                                                           if backbone_inputs else {})},
                  'parameters': optimizer_parameters or {}}]}
    path = tmp_path / 'recipe.json'
    path.write_text(json.dumps(value))
    return path


def test_recipe_declares_adaptation_and_optimizer_stages_after_backbone_training(tmp_path):
    loaded = recipe.load_recipe(write_recipe(tmp_path, adaptation_parameters={'exit_min_programs': 12, 'program_names': ['a']},
                                             optimizer_parameters={'train_bank': True, 'held_fraction': .25}))
    assert [stage['kind'] for stage in loaded['stages']] == ['native_state_sft', 'program_adaptation', 'learned_optimizer']
    args = recipe.stage_parameter_args({'program_names': ['a'], 'confirmation': True, 'tuner': {'teacher': 'self'}})
    assert args == ['--program-names', '["a"]', '--confirmation', '--tuner', '{"teacher":"self"}']


@pytest.mark.parametrize('parameters,message', [
    ({'train_bank': False}, 'explicitly enabled'),
    ({'train_basis': False}, 'explicitly enabled'),
    ({'held_fraction': 0}, 'held families'),
])
def test_optimizer_stage_refuses_disabled_bank_training_or_no_held_side(tmp_path, parameters, message):
    with pytest.raises(ValueError, match=message):
        recipe.load_recipe(write_recipe(tmp_path, optimizer_parameters=parameters))


def test_adaptation_stage_requires_confirmation_and_valid_program_names(tmp_path):
    with pytest.raises(ValueError, match='confirmation'):
        recipe.load_recipe(write_recipe(tmp_path, adaptation_parameters={'confirmation': False}))
    with pytest.raises(ValueError, match='program_names'):
        recipe.load_recipe(write_recipe(tmp_path, adaptation_parameters={'program_names': []}))
    with pytest.raises(ValueError, match='unknown stage parameters'):
        recipe.load_recipe(write_recipe(tmp_path, adaptation_parameters={'learning_rate': 1e-3}))


def test_stage_promotion_declaration_is_validated_at_load(tmp_path):
    path = write_recipe(tmp_path)
    value = json.loads(path.read_text())
    value['stages'][0]['promotion'] = {'schema': 'natlang.behavior-promotion-declaration/1',
                                       'packet': {'path': '/p', 'sha256': 'a' * 64}, 'cases': list(range(64))}
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='stay small'):
        recipe.load_recipe(path)


def adaptation_report(**changes):
    report = {'schema': adaptation_stages.PROGRAM_ADAPTATION_REPORT, 'completed': True, 'production_exposure_advanced': False,
              'entry': {'met': True}, 'exit': {'met': True, 'gained_programs': 12, 'min_programs': 12,
                                               'structural_shapes': 6, 'min_structural_shapes': 6,
                                               'confirmations_missing': []}}
    report.update(changes)
    return report


def test_program_adaptation_gate_reads_behaviour_criteria_not_completion_alone():
    recipe.require_gate(adaptation_report(), 'program_adaptation')
    with pytest.raises(ValueError, match='entry criterion not met: 1 of 4'):
        recipe.require_gate(adaptation_report(completed=False, entry={'met': False, 'reason': '1 of 4'}), 'program_adaptation')
    with pytest.raises(ValueError, match='exit criterion not met'):
        recipe.require_gate(adaptation_report(exit={'met': False, 'reason': 'one shape'}), 'program_adaptation')
    with pytest.raises(ValueError, match='inconsistent'):
        recipe.require_gate(adaptation_report(exit={**adaptation_report()['exit'], 'structural_shapes': 1}),
                            'program_adaptation')
    with pytest.raises(ValueError, match='exposure'):
        recipe.require_gate(adaptation_report(production_exposure_advanced=True), 'program_adaptation')


def test_learned_optimizer_gate_needs_bank_training_and_held_promotion():
    report = {'schema': adaptation_stages.LEARNED_OPTIMIZER_REPORT, 'completed': True, 'production_exposure_advanced': False,
              'outer_training': {'steps': 400, 'train_basis': True, 'train_bank': True, 'train_keys': True},
              'held': {'met': True}}
    recipe.require_gate(report, 'learned_optimizer')
    with pytest.raises(ValueError, match='basis, bank and keys'):
        recipe.require_gate({**report, 'outer_training': {**report['outer_training'], 'train_bank': False}}, 'learned_optimizer')
    with pytest.raises(ValueError, match='held-family promotion not met: trails'):
        recipe.require_gate({**report, 'held': {'met': False, 'reason': 'trails'}}, 'learned_optimizer')


def program(tmp_path, name, *, child, returns='Decision', helper=True, extra_files=None, examples=('main.nl',)):
    root = tmp_path / name
    (root / 'main').mkdir(parents=True)
    (root / 'main.nl').write_text(f'---\nargs: {{ item: Item }}\nreturns: {returns}\n---\nDecide the item.\n')
    (root / 'main' / 'decide.nl').write_text(f'---\nargs: {{ item: Item }}\nreturns: {returns}\n---\n{child}\n')
    (root / 'types.ts').write_text("export type Decision='a'|'b'|'c';\nexport type Item={x:number};\n")
    if helper:
        (root / 'main' / 'total.ts').write_text('export function total(item) { return item.x; }\n')
    for relative, text in (extra_files or {}).items():
        (root / relative).write_text(text)
    (root / 'tune.examples.jsonl').write_text(''.join(json.dumps({'function': function, 'input': {}}) + '\n'
                                                      for function in examples))
    return root


def test_structural_shape_ignores_names_and_helpers_but_separates_iteration_and_operations(tmp_path):
    rules = 'Apply in order: item.x>3 -> a; item.x>1 -> b; otherwise -> c.'
    one = structure.program_structure(program(tmp_path, 'refund', child=rules))
    renamed = structure.program_structure(program(tmp_path, 'access', child=rules, helper=False))
    prose = structure.program_structure(program(tmp_path, 'inventory', child='Apply in order: if x return a; otherwise if y '
                                                                             'return b; otherwise return c.'))
    assert one['features']['ordered_rules'] and one['key'] == renamed['key'] == prose['key']
    iterating = structure.program_structure(program(tmp_path, 'scan', child='Use iterateOn over the items until done.',
                                                    returns='string[]'))
    operations = structure.program_structure(program(tmp_path, 'app', child=rules, examples=('main.nl', 'ops/a.nl', 'ops/b.nl')))
    assert len(structure.distinct_shapes([one, renamed, prose, iterating, operations])) == 3


def arm(results, execution=None):
    return {'arm': 'x', 'release': None, 'execution': execution or {'maxTurns': 12},
            'roots': [{'id': f'r{i}', 'oracle': True, 'correct': bool(ok), 'ok': True} for i, ok in enumerate(results)]}


def test_development_gain_pairs_roots_under_identical_limits():
    gain = paired_gain(arm([0, 0, 1, 1]), arm([1, 1, 1, 0]))
    assert gain['paired'] and gain['gain'] == 1 and gain['gained'] == ['r0', 'r1'] and gain['lost'] == ['r3']
    assert not paired_gain(arm([0]), arm([1], {'maxTurns': 24}))['paired']


def test_entry_counts_programs_with_headroom_both_ways(tmp_path):
    programs = []
    for name, results in (('zero', [0, 0]), ('some', [1, 0]), ('more', [0, 1]), ('all', [1, 1])):
        directory = tmp_path / name / '.natlang' / 'tune' / 'evaluations' / 'development'
        directory.mkdir(parents=True)
        (directory / 'untuned.json').write_text(json.dumps({**arm(results), 'arm': 'untuned'}))
        programs.append({'program': name, 'path': str(tmp_path / name)})
    entry = entry_state(tmp_path, programs, {'entry_min_headroom_programs': 3})
    assert not entry['met'] and entry['headroom_programs'] == ['more', 'some'] and '2 of 4' in entry['reason']
    assert entry_state(tmp_path, programs, {'entry_min_headroom_programs': 2})['met']


FAKE_STAGE = '''
import json, sys
from pathlib import Path
args = sys.argv[1:]
value = lambda flag: args[args.index(flag) + 1]
out = Path(value('--out')); out.mkdir(parents=True, exist_ok=True)
(out / 'argv.json').write_text(json.dumps(args))
(out / 'adaptation-manifest.json').write_text('{}')
(out / 'report.json').write_text(json.dumps({REPORT}))
'''


def test_runner_runs_one_adaptation_stage_on_an_explicit_selected_backbone(tmp_path, monkeypatch):
    modules = tmp_path / 'modules'
    modules.mkdir()
    (modules / 'fake_adaptation_stage.py').write_text(FAKE_STAGE.replace('{REPORT}', repr(adaptation_report())))
    monkeypatch.setitem(recipe.HANDLERS['program_adaptation'], 'module', 'fake_adaptation_stage')
    monkeypatch.setenv('PYTHONPATH', str(modules) + os.pathsep + os.environ.get('PYTHONPATH', ''))
    monkeypatch.setenv('NATLANG_HARDWARE_PROFILE', 'none')
    path = write_recipe(tmp_path)
    out = tmp_path / 'lineage'
    assert recipe.main(['--recipe', str(path), '--out', str(out), '--device', 'cpu', '--stage', 'adapt']) == 0
    argv = json.loads((out / 'adapt' / 'argv.json').read_text())
    flags = {argv[i]: argv[i + 1] for i in range(len(argv) - 1) if argv[i].startswith('--')}
    assert flags['--continue-from'] == str(tmp_path / 'checkpoint.json')
    assert flags['--parent-selection'] == str(tmp_path / 'selection.json')
    assert flags['--programs'] == str(tmp_path / 'programs.json') and '--protection' in flags
    assert json.loads((out / 'adapt-report.json').read_text())['kind'] == 'program_adaptation'
    assert not (out / 'joint').exists()  # the continuous backbone stage was neither run nor required
    with pytest.raises(SystemExit):
        recipe.main(['--recipe', str(path), '--out', str(out), '--device', 'cpu', '--stage', 'joint'])


def test_runner_refuses_adaptation_without_a_selected_backbone(tmp_path, monkeypatch):
    monkeypatch.setenv('NATLANG_HARDWARE_PROFILE', 'none')
    path = write_recipe(tmp_path, backbone_inputs=False)
    with pytest.raises(ValueError, match='requires completed stage joint'):
        recipe.main(['--recipe', str(path), '--out', str(tmp_path / 'lineage'), '--device', 'cpu', '--stage', 'adapt'])
