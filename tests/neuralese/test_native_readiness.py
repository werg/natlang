"""Readiness is exact trained-state evidence."""
import json
from pathlib import Path

import pytest

from natlang_neuralese.common.hashing import sha256_file_hex
from natlang_neuralese.train.native_readiness import validate_ready
from natlang_neuralese.model.native_model import CHECKPOINT_SCHEMA


def ready(tmp_path, name):
    root = tmp_path/name; root.mkdir()
    def artifact(filename, value):
        path = root/filename
        path.write_text(json.dumps(value))
        return {'path': str(path), 'sha256': sha256_file_hex(path)}
    checkpoint = artifact('checkpoint', {'test': 'artifact-byte binding, not a model qualification'})
    checkpoint['schema'] = CHECKPOINT_SCHEMA
    identity = {'checkpoint_sha256': checkpoint['sha256'], 'tokenizer_fingerprint': '2'*64,
                'embedding_weight_sha256': '3'*64, 'architecture': name}
    recipe = artifact('recipe', {'native_stages': ['full', 'graph']})
    prior = artifact('prior-exposure', {'completed': True})
    communication = {'graphs': 4, 'required_graph_visits': 4, 'graph_visits': 4, 'complete': True}
    from natlang_neuralese.common.hashing import canonical_json_sha256_hex
    from natlang_neuralese.eval.native_gradients import SCHEMA
    roles={role:[role+'.weight']for role in('backbone','adapter','learned_input','learned_output')}
    inventory=[{'name':names[0],'shape':[1],'dtype':'torch.float32'}for names in roles.values()]
    comparison={'passed':True,'finite':True,'max_abs':0.,'relative_l2':0.,'cosine':1.,
        'reference_l2':1.,'actual_l2':1.,'mismatched_elements':0,'elements':1,
        'shape':[1],'dtype':'torch.float32','atol':1e-5,'rtol':1e-4}
    case={'fixture':'exact artifact checks; not a model qualification'}
    model_config={'architecture':name}
    profile={'model_config_sha256':canonical_json_sha256_hex(model_config),'fp8':{'kind':'off'}}
    report = artifact('report', {'completed': True, 'checkpoint_sha256': checkpoint['sha256'],
                                 'communication_exposure': communication,'qualification_profile':profile})
    complete={'schema':SCHEMA,'passed':True,'native_identity':identity,'inventory':inventory,
        'inventory_sha256':canonical_json_sha256_hex(inventory),'parameter_grads_preserved':True,
        'optimizer_updates':0,'atol':1e-5,'rtol':1e-4,'roles':roles,'case':case,
        'case_sha256':canonical_json_sha256_hex(case),
        'model_config':model_config,'model_config_sha256':profile['model_config_sha256'],
        'training_backend_execution':{'configuration':{'enabled':False},'passed':True},
        'comparisons':{name:dict(comparison)for name in('input_embeddings',*(row['name']for row in inventory))}}
    runtime = artifact('runtime', {'schema': 'natlang.native-runtime-certificate/1', 'passed': True,
        'native_identity': identity, 'backbone_optimizer_tested': True, 'backbone_optimizer_updates': 20,
        'forward_atol': 1e-4, 'gradient_atol': 1e-5,
        'forward_max_abs': 0., 'incremental_max_abs': 0., 'consumer_ce_abs': 0., 'consumer_gradient_max_abs': 1e-6,
        'complete_gradient_qualification':complete,'qualification_profile':profile})
    task = artifact('tasks', {'schema': 'natlang.native-task-execution-receipt/1', 'passed': True,
        'native_identity': identity, 'complete': True, 'eligible': True, 'task_successes': 0})
    receipt = {'schema': 'natlang.native-training-readiness/1', 'model_id': name,
        'completed': True, 'stage_id': 'graph', 'native_identity': identity, 'checkpoint': checkpoint,
        'recipe': recipe, 'prerequisite_stages': [{'id': 'full', 'completed': True, 'exposure_receipt': prior}],
        'training_receipt': report, 'runtime_certificate': runtime, 'task_execution_receipt': task,
        'communication_exposure': communication,
        'exposure': {'consumed_updates': 20, 'supervised_token_visits': 800}}
    path = root/'readiness.json'; path.write_text(json.dumps(receipt))
    requirement = {'model_id': name, 'readiness': str(path), 'recipe_sha256': recipe['sha256'],
                   'stage_ids': ['full', 'graph'], 'forward_atol': 1e-4, 'gradient_atol': 1e-5}
    return path, receipt, requirement


@pytest.mark.parametrize('corrupt',['missing','backbone','profile','gates','zero-backbone','fp8-unused'])
def test_channel_only_or_incomplete_derivatives_cannot_certify_readiness(tmp_path,corrupt):
    path,receipt,requirement=ready(tmp_path,'lfm')
    runtime_path=Path(receipt['runtime_certificate']['path'])
    runtime=json.loads(runtime_path.read_text());proof=runtime['complete_gradient_qualification']
    if corrupt=='missing':runtime.pop('complete_gradient_qualification')
    elif corrupt=='backbone':proof['comparisons'].pop('backbone.weight')
    elif corrupt=='profile':proof['model_config']['architecture']='different'
    elif corrupt=='gates':proof['comparisons']['backbone.weight']['rtol']=.01
    elif corrupt=='zero-backbone':proof['comparisons']['backbone.weight'].update(reference_l2=0.,actual_l2=0.)
    else:
        profile=runtime['qualification_profile'];profile['fp8']={'kind':'triton-experts','recipe':'current-bf16-dx'}
        report_path=Path(receipt['training_receipt']['path']);report=json.loads(report_path.read_text())
        report['qualification_profile']=profile;report_path.write_text(json.dumps(report))
        receipt['training_receipt']['sha256']=sha256_file_hex(report_path)
    runtime_path.write_text(json.dumps(runtime));receipt['runtime_certificate']['sha256']=sha256_file_hex(runtime_path)
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match='complete|communication|FP8'):
        validate_ready(path,requirement)


@pytest.mark.parametrize('communication', [None,
    {'graphs': 0, 'required_graph_visits': 0, 'graph_visits': 0, 'complete': True},
    {'graphs': 4, 'required_graph_visits': 4, 'graph_visits': 3, 'complete': False},
    {'graphs': 4, 'required_graph_visits': 2, 'graph_visits': 2, 'complete': True}])
def test_ordinary_or_partial_communication_training_cannot_certify_readiness(tmp_path, communication):
    path, receipt, requirement = ready(tmp_path, 'lfm')
    report_path = Path(receipt['training_receipt']['path'])
    report = json.loads(report_path.read_text())
    report['communication_exposure'] = communication
    report_path.write_text(json.dumps(report))
    receipt['training_receipt']['sha256'] = sha256_file_hex(report_path)
    receipt['communication_exposure'] = communication
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='genuine producer/consumer'):
        validate_ready(path, requirement)


def test_readiness_communication_exposure_must_match_bound_training_report(tmp_path):
    path, receipt, requirement = ready(tmp_path, 'lfm')
    receipt['communication_exposure'] = {**receipt['communication_exposure'], 'graph_visits': 5}
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='genuine producer/consumer'):
        validate_ready(path, requirement)


    # The once inspection must never accidentally start a GPU process.


def test_changed_checkpoint_or_missing_stage_does_not_count_ready(tmp_path):
    path, receipt, requirement = ready(tmp_path, 'lfm')
    assert validate_ready(path, requirement)['model_id'] == 'lfm'
    with pytest.raises(ValueError, match='missing required'):
        validate_ready(path, {**requirement, 'stage_ids': ['full', 'graph', 'whole']})
    Path(receipt['checkpoint']['path']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='artifact changed'):
        validate_ready(path, requirement)


def test_inherited_recipe_readiness_binds_file_bytes_and_resolved_recipe(tmp_path):
    from natlang_neuralese.train.recipe import recipe_identity_sha256
    path, receipt, requirement = ready(tmp_path, 'lfm')
    recipe_dir = tmp_path/'recipes'; recipe_dir.mkdir()
    parent = recipe_dir/'base.json'
    child = recipe_dir/'line.json'
    parent.write_text(json.dumps({'schema':'natlang.neuralese-training-recipe/1',
        'id':'base', 'stages':[], 'runtime':{'precision':'bf16'}, 'description':'base policy'}))
    child.write_text(json.dumps({'schema':'natlang.neuralese-training-recipe/1',
        'id':'line', 'extends':'base', 'description':'line policy', 'stages':[]}))
    identity = recipe_identity_sha256(child)
    file_sha = sha256_file_hex(child)
    receipt['recipe'] = {'path':str(child), 'sha256':identity, 'file_sha256':file_sha}
    path.write_text(json.dumps(receipt))
    requirement['recipe_sha256'] = identity
    assert validate_ready(path, requirement)['model_id'] == 'lfm'
    parent.write_text(json.dumps({'schema':'natlang.neuralese-training-recipe/1',
        'id':'base', 'stages':[], 'runtime':{'precision':'fp32'}, 'description':'changed inherited policy'}))
    with pytest.raises(ValueError, match='production recipe inheritance changed'):
        validate_ready(path, requirement)


def test_handed_off_recipe_readiness_binds_current_policy_and_preserves_lineage(tmp_path):
    from natlang_neuralese.train.recipe import native_recipe_binding, recipe_identity_sha256
    path, receipt, requirement = ready(tmp_path, 'lfm')
    parent = path.parent / 'parent-recipe.json'
    parent.write_text(json.dumps({'schema': 'natlang.neuralese-training-recipe/1', 'id': 'old',
        'stages': [{'id': 'graph', 'kind': 'native_state_sft', 'parameters': {'epochs': 1}}]}))
    derivative = path.parent / 'current-recipe.json'
    derivative.write_text(json.dumps({'schema': 'natlang.neuralese-training-recipe/1',
        'id': 'current', 'extends': 'old',
        'overrides': {'stages': {'graph': {'parameters': {'epochs': 2}}}}}))
    lineage = recipe_identity_sha256(parent)
    binding = native_recipe_binding(derivative, lineage)
    assert binding['sha256'] != lineage
    assert binding['lineage_sha256'] == lineage
    assert binding['file_sha256'] == sha256_file_hex(derivative)
    receipt['recipe'] = binding
    requirement['recipe_sha256'] = binding['sha256']
    path.write_text(json.dumps(receipt))
    assert validate_ready(path, requirement)['recipe']['sha256'] == binding['sha256']
    # Changing inherited policy remains detectable even when the derivative's
    # own file bytes and the original lineage label are unchanged.
    parent_data = json.loads(parent.read_text())
    parent_data['stage_parameter_defaults'] = {'native_state_sft': {'weight_decay': .2}}
    parent.write_text(json.dumps(parent_data))
    with pytest.raises(ValueError, match='inheritance changed'):
        validate_ready(path, requirement)


def test_current_policy_readiness_cannot_be_replaced_by_equal_lineage(tmp_path):
    from natlang_neuralese.train.recipe import native_recipe_binding
    path,receipt,requirement=ready(tmp_path,'lfm')
    recipe=Path(receipt['recipe']['path'])
    receipt['recipe']=native_recipe_binding(recipe,'a'*64)
    requirement['recipe_lineage_sha256']='a'*64
    path.write_text(json.dumps(receipt))
    assert validate_ready(path,requirement)
    with pytest.raises(ValueError,match='preserved recipe lineage'):
        validate_ready(path,{**requirement,'recipe_lineage_sha256':'b'*64})
    newer=tmp_path/'current-policy';newer.write_text('actual changed recipe policy')
    with pytest.raises(ValueError,match='declared production recipe'):
        validate_ready(path,{**requirement,'recipe_sha256':sha256_file_hex(newer)})
