"""Real file bindings and ownership gates; these fixtures do not qualify CUDA."""
import json
from pathlib import Path
import pytest
from natlang_neuralese.train.native_deployment import prepare, binding
from natlang_neuralese.train.native_readiness import validate_ready, ready_participants
from test_native_readiness import ready


def test_terminal_trainer_required_before_transfer_of_optimizer_ownership(tmp_path):
    path, _, requirement = ready(tmp_path, 'lfm')
    requirement['trainer_exit'] = str(tmp_path/'exit.json')
    assert validate_ready(path, requirement) is None
    Path(requirement['trainer_exit']).write_text(json.dumps({'returncode': 1}))
    with pytest.raises(ValueError, match='did not drain'):
        validate_ready(path, requirement)
    Path(requirement['trainer_exit']).write_text(json.dumps({'returncode': 0}))
    assert validate_ready(path, requirement)['model_id'] == 'lfm'
    launch=tmp_path/'launch.json';launch.write_text(json.dumps({'at':10,'argv':['actual-recipe']}))
    requirement.update(trainer_launch=binding(launch),trainer_recipe='actual-recipe')
    Path(requirement['trainer_exit']).write_text(json.dumps({'returncode':0,'at':9}))
    with pytest.raises(ValueError,match='different trainer launch'):
        validate_ready(path,requirement)
    Path(requirement['trainer_exit']).write_text(json.dumps({'returncode':0,'at':11}))
    assert validate_ready(path,requirement)['model_id']=='lfm'


def test_prepare_actual_file_bound_two_sandbox_controller_before_foundation_finishes(tmp_path):
    runtime=tmp_path/'runtime';(runtime/'ts-host/dist/teacher').mkdir(parents=True)
    (runtime/'ts-host/dist/teacher/cli.js').write_text('// built collector fixture')
    train=tmp_path/'train';train.write_text('{}\n')
    held=tmp_path/'held';held.write_text('{}\n')
    cases=tmp_path/'cases';cases.write_text('{"split":"train"}\n')
    curriculum=tmp_path/'semantic.json'
    curriculum.write_text(json.dumps({'schema':'natlang.native-semantic-training/1',
        'pairs':binding(train),'held_pairs':binding(held)}))
    records=[]
    for role in ('lfm','mellum'):
        base=tmp_path/role/'base';base.mkdir(parents=True)
        recipe=tmp_path/role/'recipe.json'
        sources={'train':binding(train),'held':binding(held),'foundation':binding(train)}
        recipe.write_text(json.dumps({'schema':'natlang.neuralese-training-recipe/1','id':role,
            'input_bindings':sources,'stages':[{'id':'native_full_sft','kind':'native_state_sft',
                'requires':[],'inputs':{'native_graphs':'train','held_native_graphs':'held',
                    'foundation_native_graphs':'foundation'},
                'parameters':{'epochs':1,'base':str(base),'communication_foundation':True,
                    'task_eval':{'packet':binding(cases)},'task_eval_every':5000}}]}))
        owner=tmp_path/role/'owner';owner.mkdir()
        (owner/'launch.json').write_text(json.dumps({'at':1,'argv':['python','--recipe',str(recipe)]}))
        records.append({'role':role,'recipe_path':str(recipe),'recipe_sha256':binding(recipe)['sha256'],
            'foundation_manifest':binding(train),'sandbox_id':role+'-existing',
            'control':str(tmp_path/role/'owner')})
    release=tmp_path/'release.json';release.write_text(json.dumps(records))
    controller=prepare(release,curriculum,tmp_path/'deployment',participants={'lfm':'lfm2:actual','mellum':'mellum:actual'},
        cases=cases,runtime=runtime,namespace=str(tmp_path))
    config=json.loads(controller.read_text())
    assert len(config['participants'])==2
    assert all(item['readiness_boundary']=='communication-foundation' and item['trainer_exit']
        for item in config['participants'])
    bootstrap=json.loads(Path(config['rl_bootstrap']['path']).read_text())
    assert {item['provider']['sandbox_id'] for item in bootstrap['models']}=={'lfm-existing','mellum-existing'}
    assert bootstrap['refresh_provider']['sandbox_id']=='mellum-existing'
    assert all(item['production_stage']=='native_full_sft' for item in bootstrap['models'])
    # Preparation may run early; actual controller waits, without launching models.
    selected,pending=ready_participants(config)
    assert not selected and len(pending)==2
    train.write_text('changed bytes\n')
    with pytest.raises(ValueError,match='artifact changed'):
        prepare(release,curriculum,tmp_path/'invalid',participants={'lfm':'a','mellum':'b'},
            cases=cases,runtime=runtime,namespace=str(tmp_path))
