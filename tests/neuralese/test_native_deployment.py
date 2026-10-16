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


@pytest.mark.parametrize('routine_cadence', [0, 5000])
@pytest.mark.parametrize('separate_refresh', [False, True])
def test_prepare_actual_file_bound_two_sandbox_controller_before_foundation_finishes(tmp_path, routine_cadence, separate_refresh):
    runtime=tmp_path/'runtime';(runtime/'ts-host/dist/teacher').mkdir(parents=True)
    (runtime/'ts-host/dist/teacher/cli.js').write_text('// built collector fixture')
    train=tmp_path/'train';train.write_text('{}\n')
    held=tmp_path/'held';held.write_text('{}\n')
    cases=tmp_path/'cases';cases.write_text('{"split":"train"}\n')
    curriculum=tmp_path/'semantic.json'
    curriculum.write_text(json.dumps({'schema':'natlang.native-semantic-training/1',
        'pairs':binding(train),'held_pairs':binding(held)}))
    refresh_path = None
    if separate_refresh:
        expanded = tmp_path/'expanded-pairs'; expanded.write_text('{}\n{}\n')
        refresh_path = tmp_path/'expanded-semantic.json'
        refresh_path.write_text(json.dumps({'schema':'natlang.native-semantic-training/1',
            'pairs':binding(expanded),'held_pairs':binding(held)}))
    records=[]; participants={}
    for role in ('lfm','mellum'):
        base=tmp_path/role/'base';base.mkdir(parents=True)
        (base/'config.json').write_text(json.dumps({'architecture':role}))
        (base/'weights.safetensors').write_bytes(b'pinned weight fixture '+role.encode())
        from natlang_neuralese.model.native_model import base_inventory
        from natlang_neuralese.common.hashing import canonical_json_sha256_hex
        participants[role]=role+':'+canonical_json_sha256_hex(base_inventory(base))
        recipe=tmp_path/role/'recipe.json'
        sources={'train':binding(train),'held':binding(held),'foundation':binding(train)}
        recipe.write_text(json.dumps({'schema':'natlang.neuralese-training-recipe/1','id':role,
            'input_bindings':sources,'stages':[{'id':'native_full_sft','kind':'native_state_sft',
                'requires':[],'inputs':{'native_graphs':'train','held_native_graphs':'held',
                    'foundation_native_graphs':'foundation'},
                'parameters':{'epochs':1,'base':str(base),'communication_foundation':True,
                    'task_eval':{'packet':binding(cases)},'task_eval_every':routine_cadence}}]}))
        owner=tmp_path/role/'owner';owner.mkdir()
        (owner/'launch.json').write_text(json.dumps({'at':1,'argv':['python','--recipe',str(recipe)]}))
        records.append({'role':role,'recipe_path':str(recipe),'recipe_sha256':binding(recipe)['sha256'],
            'foundation_manifest':binding(train),'sandbox_id':role+'-existing',
            'control':str(tmp_path/role/'owner')})
    release=tmp_path/'release.json';release.write_text(json.dumps(records))
    controller=prepare(release,curriculum,tmp_path/'deployment',participants=participants,
        cases=cases,runtime=runtime,namespace=str(tmp_path), refresh_curriculum_path=refresh_path)
    config=json.loads(controller.read_text())
    assert controller == tmp_path/'deployment/controller.json'
    assert config['out'] == str(tmp_path/'shared/native_shared_stage')
    assert config['rl_out'] == str(tmp_path/'shared/native_rl')
    assert len(config['participants'])==2
    assert all(item['readiness_boundary']=='communication-foundation' and item['trainer_exit']
        for item in config['participants'])
    bootstrap=json.loads(Path(config['rl_bootstrap']['path']).read_text())
    assert {item['provider']['sandbox_id'] for item in bootstrap['models']}=={'lfm-existing','mellum-existing'}
    assert bootstrap['refresh_provider']['sandbox_id']=='mellum-existing'
    assert all(item['production_stage']=='native_full_sft' for item in bootstrap['models'])
    raw=json.loads(Path(config['raw_rl_bootstrap']['path']).read_text())
    assert raw['schema']=='natlang.native-raw-rl-bootstrap/1'
    assert {item['model'] for item in raw['models']}=={'lfm','mellum'}
    assert all(item['handoff_requirement']==requirement for item,requirement in
               zip(raw['models'],config['participants']))
    assert 'semantic_refresh_config' not in raw
    # Production auto-bootstrap defaults to the integrity-bound 128-task union
    # for both participants; the older explicit fixture above remains supported.
    admitted_manifest = (Path(__file__).resolve().parents[2]
        / 'runs/decision-rich-train-20261016-v2-curriculum/task-source-manifest.json')
    admitted = prepare(release, curriculum, tmp_path/'deployment-admitted',
        participants=participants, runtime=runtime, namespace=str(tmp_path))
    admitted_bootstrap = json.loads(Path(json.loads(admitted.read_text())['rl_bootstrap']['path']).read_text())
    assert len(admitted_bootstrap['models']) == 2
    assert {row['cases']['sha256'] for row in admitted_bootstrap['models']} == {
        'bfb1b595d013616cfca5c4bc4aab44a218e2398ffc505b1954405dbb56bfd53a'}
    assert all(row['task_source']['manifest'] == str(admitted_manifest.resolve())
        and row['task_source']['rows'] == 128 for row in admitted_bootstrap['models'])
    first = json.loads(Path(config['curriculum']).read_text())
    refresh = json.loads(Path(bootstrap['semantic_refresh_config']['path']).read_text())
    assert first['pairs'] == binding(train)
    assert refresh['pairs'] == binding(expanded if separate_refresh else train)
    assert first['models'] == refresh['models']
    assert first['task_eval'] == refresh['task_eval']
    assert first['policy_schedule'] == refresh['policy_schedule'] == {
        'id':'policy','epochs':1,'every_semantic_updates':1,'updates_per_refresh':16}
    assert 'policy_training' not in first and 'policy_training' not in refresh
    if separate_refresh:
        assert bootstrap['semantic_refresh_config']['path'] != config['curriculum']
        assert config['deployment_inputs']['refresh_curriculum'] == binding(refresh_path)
        before = expanded.read_bytes(); expanded.write_bytes(b'changed expanded source')
        with pytest.raises(ValueError, match='artifact changed'):
            prepare(release,curriculum,tmp_path/'bad-refresh',participants=participants,
                cases=cases,runtime=runtime,namespace=str(tmp_path),refresh_curriculum_path=refresh_path)
        expanded.write_bytes(before)
    else:
        assert bootstrap['semantic_refresh_config']['path'] == config['curriculum']
    # Preparation may run early; actual controller waits, without launching models.
    selected,pending=ready_participants(config)
    assert not selected and len(pending)==2
    release.write_text(json.dumps([{**row,'base':{'path':str(tmp_path/row['role']/'base'),
        'inventory_sha256':'0'*64}} for row in records]))
    with pytest.raises(ValueError,match='resident base inventory'):
        prepare(release,curriculum,tmp_path/'wrong-base',participants=participants,
            cases=cases,runtime=runtime,namespace=str(tmp_path))
    release.write_text(json.dumps(records))
    cases.write_text('{"split":"test"}\n')
    with pytest.raises(ValueError,match='train-only'):
        prepare(release,curriculum,tmp_path/'wrong-cases',participants=participants,
            cases=cases,runtime=runtime,namespace=str(tmp_path))
    cases.write_text('{"split":"train"}\n')
    train.write_text('changed bytes\n')
    with pytest.raises(ValueError,match='artifact changed'):
        prepare(release,curriculum,tmp_path/'invalid',participants=participants,
            cases=cases,runtime=runtime,namespace=str(tmp_path))


def test_semantic_loader_relocates_exact_pinned_base_without_resetting_optimizer(tmp_path):
    import shutil
    import torch
    from test_native_model import tiny_source
    from natlang_neuralese.model.native_model import load_native_model, model_state, native_identity, CHECKPOINT_SCHEMA
    from natlang_neuralese.train.native_state_sft import make_native_optimizer
    from natlang_neuralese.train.native_semantic_stage import load_participant
    base=tmp_path/'original';tiny_source(base)
    model,_,_,config=load_native_model(base,dtype=torch.float32,rank=4)
    optimizer,_=make_native_optimizer(model,lr=.001,in_backward=False)
    loss=sum(parameter.square().mean() for parameter in model.parameters() if parameter.requires_grad)
    loss.backward();optimizer.step()
    checkpoint=tmp_path/'trained.pt'
    torch.save({'schema':CHECKPOINT_SCHEMA,'model_config':config,**model_state(model),
        'training':{'optimizer':optimizer.state_dict(),'step':1}},checkpoint)
    identity=native_identity(model,config,checkpoint_sha256=binding(checkpoint)['sha256'])
    relocated=tmp_path/'resident-base';shutil.move(base,relocated)
    loaded=load_participant('student',{'checkpoint':binding(checkpoint),'native_identity':identity},
        {'backbone_lr':.001,'models':{'student':{'base':str(relocated)}}},'cpu')
    assert loaded.identity==identity and loaded.config['base']==str(relocated)
    def exact(left,right):
        if isinstance(left,torch.Tensor):assert torch.equal(left,right)
        elif isinstance(left,dict):
            assert left.keys()==right.keys()
            for key in left:exact(left[key],right[key])
        elif isinstance(left,(tuple,list)):
            assert len(left)==len(right)
            for a,b in zip(left,right):exact(a,b)
        else:assert left==right
    exact(loaded.optimizer.state_dict(),optimizer.state_dict())
    assert loaded.global_step==1
    for name,value in loaded.model.state_dict().items():
        assert torch.equal(value,model.state_dict()[name])
