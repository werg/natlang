"""Readiness is exact trained-state evidence; the second model triggers work."""
import json
from pathlib import Path

import pytest

from natlang_neuralese.common.hashing import sha256_file_hex
from natlang_neuralese.train.native_readiness import ready_participants, run, validate_ready
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
    report = artifact('report', {'completed': True, 'checkpoint_sha256': checkpoint['sha256'],
                                 'communication_exposure': communication})
    runtime = artifact('runtime', {'schema': 'natlang.native-runtime-certificate/1', 'passed': True,
        'native_identity': identity, 'backbone_optimizer_tested': True, 'backbone_optimizer_updates': 20,
        'forward_atol': 1e-4, 'gradient_atol': 1e-5,
        'forward_max_abs': 0., 'incremental_max_abs': 0., 'consumer_ce_abs': 0., 'consumer_gradient_max_abs': 1e-6})
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


@pytest.mark.parametrize('communication', [None,
    {'graphs': 0, 'required_graph_visits': 0, 'graph_visits': 0, 'complete': True},
    {'graphs': 4, 'required_graph_visits': 4, 'graph_visits': 3, 'complete': False},
    {'graphs': 4, 'required_graph_visits': 2, 'graph_visits': 2, 'complete': True}])
def test_ordinary_or_partial_communication_training_cannot_start_vae(tmp_path, communication):
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


def test_second_ready_model_proceeds_without_waiting_for_all_models(tmp_path):
    a, _, first = ready(tmp_path, 'lfm')
    b, _, second = ready(tmp_path, 'mellum')
    third = {**first, 'model_id': 'later', 'readiness': str(tmp_path/'not-ready')}
    config = {'schema': 'natlang.native-shared-stage-controller/1', 'participants': [first, third, second],
              'command': ['python', 'actual-shared-stage.py', '--participants', '{participants}'],
              'out': str(tmp_path/'shared')}
    selected, pending = ready_participants(config)
    assert [item['model_id'] for item in selected] == ['lfm', 'mellum']
    assert len(pending) == 1
    path = tmp_path/'controller.json'; path.write_text(json.dumps(config))
    result = run(path, once=True)
    assert result['ready_models'] == 2 and result['child_pid'] is None
    assert len(json.loads(Path(result['participants']).read_text())['models']) == 2
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


def test_corrupt_third_model_is_a_repair_without_blocking_two_ready_models(tmp_path):
    _, _, first = ready(tmp_path, 'lfm')
    _, _, second = ready(tmp_path, 'mellum')
    _, receipt, third = ready(tmp_path, 'broken')
    Path(receipt['runtime_certificate']['path']).write_text('{}')
    selected, pending = ready_participants({'participants': [third, first, second]})
    assert len(selected) == 2 and pending[0]['repair_required'] is True


@pytest.mark.parametrize('qualified', [False, True])
def test_optional_rl_launch_waits_for_shared_validation_and_uses_same_child_supervision(tmp_path, monkeypatch, qualified):
    """Control-only child fixtures: never claims a completed model qualification."""
    from natlang_neuralese.train import native_readiness as controller, loop
    from natlang_neuralese.common.hashing import canonical_json_sha256_hex
    out = tmp_path/'controller'; out.mkdir()
    curriculum = tmp_path/'curriculum.json'; curriculum.write_text('{}')
    plan = tmp_path/'rl-plan.json'; plan.write_text(json.dumps({
        'schema': 'natlang.native-rl-bootstrap/1', 'models': [{'model': 'lfm'}, {'model': 'mellum'}]}))
    config = {'schema': 'natlang.native-shared-stage-controller/1', 'out': str(out),
        'command': ['semantic-fixture', '{participants}'], 'curriculum': str(curriculum),
        'participants': [], 'poll_seconds': .001,
        'rl_bootstrap': {'path': str(plan), 'sha256': sha256_file_hex(plan)}}
    path = tmp_path/'config.json'; path.write_text(json.dumps(config))
    ready = [{'model_id': 'lfm'}, {'model_id': 'mellum'}]
    monkeypatch.setattr(controller, 'ready_participants', lambda config: (ready, []))
    class Stop:
        requested = False
        def install(self):
            pass
        def restore(self):
            pass
    stop = Stop()
    monkeypatch.setattr(loop, 'StopSignal', lambda: stop)
    launched, validated = [], []
    proof = {'schema': 'control-only-fixture', 'checkpoint': {'path': 'fixture', 'sha256': 'hash'},
        'adapter': {'path': 'fixture-adapter', 'sha256': 'adapter'},
        'participants_signature': canonical_json_sha256_hex(ready),
        'curriculum_signature': canonical_json_sha256_hex({})}
    def validate(path, published):
        validated.append(True)
        if not qualified:
            raise ValueError('shared qualification missing')
        return proof, 'completed-semantic-fixture'
    monkeypatch.setattr(controller, 'validate_completion', validate)
    class Child:
        pid = 12345
        def __init__(self, semantic):
            self.returncode = 0 if semantic else None
        def poll(self):
            return self.returncode
        def terminate(self):
            self.returncode = 0
    def launch(argv, **kwargs):
        launched.append(argv)
        if len(launched) == 1:
            (out/'shared-stage-completion.json').write_text(json.dumps(proof))
            return Child(True)
        assert validated and qualified  # no early RL despite zero semantic exit
        stop.requested = True
        return Child(False)
    monkeypatch.setattr(controller.subprocess, 'Popen', launch)
    # This is a controller-only child fixture, not an actual /proc process.
    monkeypatch.setattr(controller, '_owned_process_identity', lambda pid, launch, **kwargs: {
        'pid': pid, 'start_ticks': 1, 'boot_id': 'control-fixture',
        'argv': launch['argv'], 'cwd': launch['cwd']} if pid == 12345 else None)
    def pause(seconds):
        if not qualified:
            stop.requested = True
    monkeypatch.setattr(controller.time, 'sleep', pause)
    if not qualified:
        result = controller.run(path)
        assert len(launched) == 1
        assert result['last_successful_participants'] is None
        assert result['retry']['reason'].startswith('shared completion rejected:')
    else:
        result = controller.run(path)
        assert len(launched) == 2
        assert 'natlang_neuralese.rl.native_rounds' in launched[1]
        assert '--bootstrap' in launched[1] and str(plan) in launched[1]
        assert result['status'] == 'stopped-after-checkpoint-request'


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
