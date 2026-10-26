"""Joint path-routed program adapter training on the real native engine (plans/HIERARCHICAL_LORA_TUNING.md §5)."""
import copy
import json

import pytest
import torch

from test_native_serving_adapters import engine_fixture
from natlang_neuralese.model.native_adapters import (create_adapter, resolve_adapters, run_native, target_parameters)
from natlang_neuralese.model.shared_state import Invocation, compile_graph
from natlang_neuralese.tune.trainer import JointTrainer, Recipe

IDENTITY_KEYS = ('architecture', 'checkpoint_sha256', 'embedding_weight_sha256', 'tokenizer_fingerprint')


def manifest_for(engine, families=(), memberships=()):
    tree = [{'id': 'tree:', 'level': 'tree', 'path': '', 'parent': None},
            {'id': 'folder:a', 'level': 'folder', 'path': 'a', 'parent': 'tree:'},
            {'id': 'folder:b', 'level': 'folder', 'path': 'b', 'parent': 'tree:'},
            {'id': 'lambda:x', 'level': 'lambda', 'path': 'a/x.nl', 'parent': 'folder:a'},
            {'id': 'lambda:y', 'level': 'lambda', 'path': 'b/y.nl', 'parent': 'folder:b'}]
    binding = lambda ident, folder, lam: {'id': ident, 'component': 'p::'+lam, 'definitionId': lam, 'contextId': None,
        'origin': 'named', 'ancestry': ['tree:', folder, lam], 'semanticHash': 'h'+ident}
    return {'schema': 'natlang.adapters/v1', 'program': {'id': 'p', 'buildHash': 'b'},
            'base': {key: engine.native_identity[key] for key in IDENTITY_KEYS}, 'tree': tree,
            'bindings': [binding('bx', 'folder:a', 'lambda:x'), binding('by', 'folder:b', 'lambda:y')],
            'artifacts': [], 'families': list(families), 'memberships': list(memberships), 'scales': {},
            'validatedReductions': [], 'fallback': 'base', 'receipts': {}}


def rows_for(engine, binding, count, offset=0):
    fingerprint = engine.native_identity['tokenizer_fingerprint']
    return [{'binding': binding, 'evidence_weight': 1. + index % 2, 'occurrence': f'{binding}-{index}',
             'graph': {'schema': 'natlang.neuralese-native-graph/1', 'id': f'{binding}-{index}', 'split': 'train',
                       'source_groups': [f'{binding}-group'], 'tokenizer_fingerprint': fingerprint,
                       'nodes': [{'name': 'answer', 'token_ids': [4 + (index + offset) % 5, 5, 6 + index % 3, 7, 8],
                                  'target_weights': [0, .1, 1, 1, 1]}]}} for index in range(count)]


def tensors(engine, artifact):
    return engine.store.get_native_tensor_artifact(artifact).artifact.tensors


@pytest.mark.parametrize('parameterization', ['native-subspace', 'free-lora'])
def test_fresh_stack_is_exactly_the_base_model(tmp_path, parameterization):
    engine = engine_fixture(tmp_path, 'lfm')
    trainer = JointTrainer(engine, manifest_for(engine), rows_for(engine, 'bx', 2),
                           Recipe(parameterization=parameterization, steps=0))
    plan = compile_graph([Invocation('answer', (4, 5, 6, 7))])
    model = engine.shared_model
    base = model.logits(model(plan)['h_final']).detach()
    entries = [{'id': trainer.state['node_artifacts'][node], 'scale': 1.} for node in ('tree:', 'folder:a', 'lambda:x')]
    adapted = run_native(engine, lambda: model.logits(model(plan)['h_final']).detach(), entries)
    assert torch.equal(adapted, base)
    if parameterization == 'native-subspace':
        # One canonical basis: every node slices the same artifact, rank by level.
        sources = {engine.store.get_native_tensor_artifact(trainer.state['node_artifacts'][node]).artifact.source_artifact_id
                   for node in trainer.state['node_artifacts']}
        assert len(sources) == 1
        resolved = {node: resolve_adapters(engine, [trainer.state['node_artifacts'][node]])[0] for node in ('tree:', 'lambda:x')}
        name = trainer.target_names[0]
        tree_left, lambda_left = resolved['tree:'][4]['left.'+name], resolved['lambda:x'][4]['left.'+name]
        assert tree_left.shape[-1] == 8 and lambda_left.shape[-1] == 2 and torch.equal(tree_left[..., :2], lambda_left)


@pytest.mark.parametrize('parameterization', ['native-subspace', 'free-lora'])
def test_step_updates_exactly_the_active_path(tmp_path, parameterization):
    engine = engine_fixture(tmp_path, 'lfm')
    trainer = JointTrainer(engine, manifest_for(engine), rows_for(engine, 'bx', 3),
                           Recipe(parameterization=parameterization, batch_rows=3,
                                  optimizer={'name': 'adam', 'learning_rate': 1e-2}))
    before = dict(trainer.state['node_artifacts'])
    record = trainer.step()
    assert record['active_nodes'] == ['folder:a', 'lambda:x', 'tree:'] == record['updated_nodes']
    for node in ('folder:b', 'lambda:y'):
        assert trainer.state['node_artifacts'][node] == before[node] and trainer.state['optimizer_states'][node] is None
    for node in ('tree:', 'folder:a', 'lambda:x'):
        old, new = tensors(engine, before[node]), tensors(engine, trainer.state['node_artifacts'][node])
        changed = [key for key in new if not torch.equal(old[key], new[key])]
        if parameterization == 'free-lora':
            # A gets no gradient while B is zero; B moves.
            assert changed and all(key.startswith('lora_B.') for key in changed)
        else:
            assert len(changed) == len(new)
    assert trainer.state['exposure'] == {f'bx-{index}': 1 for index in range(3)}


def test_one_snapshot_lambda_balanced_gradient_matches_reference(tmp_path):
    engine = engine_fixture(tmp_path, 'lfm')
    rows = rows_for(engine, 'bx', 2) + rows_for(engine, 'by', 1, offset=3)
    recipe = Recipe(batch_rows=3, optimizer={'name': 'sgd', 'learning_rate': 1.},
                    depth_penalty={'tree': 0., 'folder': 0., 'lambda': 0.})
    trainer = JointTrainer(engine, manifest_for(engine), rows, recipe)
    ids = dict(trainer.state['node_artifacts'])
    start = {node: tensors(engine, artifact) for node, artifact in ids.items()}
    trainer.step()
    # Reference: independent per-row gradients at the same starting values, weighted mean over lambdas of the
    # mean evidence-weighted row loss.
    from natlang_neuralese.data.native_graph import graph_loss
    from natlang_neuralese.data.native_graph_schema import parse_graph
    expected = {node: {key: torch.zeros_like(value) for key, value in values.items()} for node, values in start.items()}
    for row in rows:
        path = ['tree:', 'folder:a' if row['binding'] == 'bx' else 'folder:b', 'lambda:x' if row['binding'] == 'bx' else 'lambda:y']
        leaves = {ids[node]: {key: value.clone().requires_grad_(True) for key, value in start[node].items()} for node in path}
        example = parse_graph(row['graph'], tokenizer_fingerprint=engine.model_config['tokenizer_fingerprint'])
        weight = row['evidence_weight'] / (2 * (2 if row['binding'] == 'bx' else 1))
        def execute():
            loss, _ = graph_loss(engine.shared_model, example)
            flat = [value for node in path for value in leaves[ids[node]].values()]
            return torch.autograd.grad(loss * weight, flat)
        grads = iter(run_native(engine, execute, [{'id': ids[node], 'scale': 1.} for node in path],
                                leaves=leaves, gradient_precision='fp32'))
        for node in path:
            for key in start[node]:
                expected[node][key] += next(grads)
    for node, artifact in trainer.state['node_artifacts'].items():
        for key, value in tensors(engine, artifact).items():
            torch.testing.assert_close(start[node][key] - value, expected[node][key], rtol=1e-5, atol=1e-7)


def family_setup(engine):
    names = sorted(target_parameters(engine.shared_model))
    family = create_adapter(engine, {'kind': 'xs', 'rank': 2, 'parameter_names': names})
    from dataclasses import replace
    nonzero = engine.store.put_native_tensor_artifact(replace(family.artifact, tensors={
        key: torch.full_like(value, .05) for key, value in family.artifact.tensors.items()}))
    families = [{'family': 'classify', 'version': 'v1', 'artifact': nonzero.id, 'amplitudeHex': (1.).hex()}]
    memberships = [{'binding': 'bx', 'family': 'classify', 'version': 'v1', 'gate': .5}]
    return nonzero, families, memberships


def test_family_gate_trains_projected_while_family_tensors_stay_frozen(tmp_path):
    engine = engine_fixture(tmp_path, 'lfm')
    family, families, memberships = family_setup(engine)
    frozen = {key: value.clone() for key, value in family.artifact.tensors.items()}
    recipe = Recipe(batch_rows=2, tau=.5, a_max=.6, gate_learning_rate=.05)
    trainer = JointTrainer(engine, manifest_for(engine, families, memberships), rows_for(engine, 'bx', 2), recipe)
    key = json.dumps(['bx', 'classify', 'v1'])
    values = []
    for _ in range(3):
        trainer.step()
        values.append(float.fromhex(trainer.state['gates'][key]))
    assert values[0] != .5 and all(0 <= value <= .6 for value in values)
    assert all(torch.equal(frozen[name], value) for name, value in tensors(engine, family.id).items())
    assert trainer.manifest_fragment()['memberships'][0]['gate'] == values[-1]


def test_resume_from_serialized_state_reproduces_the_uninterrupted_run(tmp_path):
    engine = engine_fixture(tmp_path, 'lfm')
    family, families, memberships = family_setup(engine)
    manifest = manifest_for(engine, families, memberships)
    rows = rows_for(engine, 'bx', 3) + rows_for(engine, 'by', 2, offset=1)
    recipe = Recipe(batch_rows=2, optimizer={'name': 'adam', 'learning_rate': 1e-2})
    continuous = JointTrainer(engine, manifest, rows, recipe)
    continuous.run(4)
    interrupted = JointTrainer(engine, manifest, rows, recipe)
    interrupted.run(2)
    resumed = JointTrainer(engine, manifest, rows, recipe, state=json.loads(json.dumps(interrupted.state)))
    resumed.run(2)
    assert resumed.state['node_artifacts'] == continuous.state['node_artifacts']
    assert resumed.state['optimizer_states'] == continuous.state['optimizer_states']
    assert resumed.state['gates'] == continuous.state['gates']
    assert (resumed.state['epoch'], resumed.state['position'], resumed.state['exposure']) == \
           (continuous.state['epoch'], continuous.state['position'], continuous.state['exposure'])
    with pytest.raises(ValueError, match='another manifest'):
        JointTrainer(engine, manifest, rows[:-1], recipe, state=copy.deepcopy(interrupted.state))


def test_build_entry_resumes_from_its_durable_artifacts(tmp_path):
    from natlang_neuralese.tune import build
    (tmp_path / 'fixture').mkdir()
    engine = engine_fixture(tmp_path / 'fixture', 'lfm')
    manifest = manifest_for(engine)
    (tmp_path / 'rows.jsonl').write_text('\n'.join(json.dumps(row) for row in rows_for(engine, 'bx', 2) + rows_for(engine, 'by', 2)))
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    def plan(steps):
        (tmp_path / 'plan.json').write_text(json.dumps({'manifest': 'manifest.json', 'rows': 'rows.jsonl',
            'engine': {'checkpoint': str(tmp_path / 'fixture' / 'native.pt')}, 'recipe': {'steps': steps, 'batch_rows': 2}}))
        return str(tmp_path / 'plan.json')
    assert build.main(['--plan', plan(1), '--out', str(tmp_path / 'resumed')]) == 0
    assert build.main(['--plan', plan(2), '--out', str(tmp_path / 'resumed')]) == 0
    assert build.main(['--plan', plan(2), '--out', str(tmp_path / 'continuous')]) == 0
    results = [json.loads((tmp_path / out / 'result.json').read_text())['receipt'] for out in ('resumed', 'continuous')]
    assert results[0]['artifacts'] == results[1]['artifacts'] and results[0]['step'] == 2
    lambda_x = next(item for item in results[0]['artifacts'] if item['node'] == 'lambda:x')
    tree, folder = (next(item['artifact'] for item in results[0]['artifacts'] if item['node'] == node) for node in ('tree:', 'folder:a'))
    assert lambda_x['requires'] == [tree, folder]
    assert results[0]['exposure']['rows_seen'] == 4


def test_exported_stack_equals_the_native_effective_weights(tmp_path):
    from natlang_neuralese.tune.export import export_peft, load_peft_dense
    from natlang_neuralese.tune.stack import stack_factors
    engine = engine_fixture(tmp_path, 'lfm')
    family, families, memberships = family_setup(engine)
    trainer = JointTrainer(engine, manifest_for(engine, families, memberships), rows_for(engine, 'bx', 2),
                           Recipe(batch_rows=2, optimizer={'name': 'adam', 'learning_rate': 5e-2}))
    trainer.run(2)
    members, bases = trainer.stack_members('bx')
    available = target_parameters(engine.shared_model)
    targets = {name: tuple(available[name].shape) for name in trainer.target_names}
    stack = stack_factors(members, bases, targets)
    # The release stack the runtime sends: family with its gate as scale, then the ancestry.
    entries = [{'id': family.id, 'scale': members[0].scale}] + [
        {'id': trainer.state['node_artifacts'][node], 'scale': 1.} for node in ('tree:', 'folder:a', 'lambda:x')]
    effective = run_native(engine, lambda: {name: value.detach().clone() for name, value in target_parameters(engine.shared_model).items()}, entries)
    for name in trainer.target_names:
        torch.testing.assert_close(stack[name].dense(targets[name]).double(), (effective[name] - available[name]).double(),
                                   rtol=1e-4, atol=1e-6)
    # Nested subspace nodes sum in one basis: rank at most the largest node's; the separately based family concatenates.
    assert stack[trainer.target_names[0]].rank == 8 + 2
    receipt = export_peft(stack, trainer.targets, tmp_path / 'peft', base_identity=dict(engine.native_identity),
                          base_model_name_or_path='fixture')
    deltas = load_peft_dense(tmp_path / 'peft')
    for name, target in trainer.targets.items():
        torch.testing.assert_close(deltas[target.hf_module].float(), stack[name].dense(targets[name]), rtol=1e-4, atol=1e-6)
    assert set(receipt['module_map']) == set(trainer.target_names)


def test_student_target_tables_cover_the_real_port_inventories(tmp_path):
    from natlang_neuralese.tune.export import _modules
    from natlang_neuralese.tune.stack import TargetFactors
    from natlang_neuralese.tune.targets import module_map, student_family
    for family, student, classes in (('lfm', 'lfm2', {'attention', 'conv-projection', 'mlp'}),
                                     ('mellum', 'mellum', {'attention', 'experts'})):
        (tmp_path / family).mkdir()
        engine = engine_fixture(tmp_path / family, family)
        names = sorted(target_parameters(engine.shared_model))
        assert student_family(names) == student
        targets, kinds = module_map(names)
        assert set(kinds.values()) == classes
        if student == 'mellum':
            # The router is mapped but trained only on explicit request.
            assert any(name.endswith('mlp.gate.weight') for name in names) and not any(name.endswith('mlp.gate.weight') for name in targets)
            name = next(name for name in names if name.endswith('experts.gate_up'))
            assert targets[name].hf_module[1].endswith('.mlp.experts.{expert}.up_proj')
            experts, rows, cols = target_parameters(engine.shared_model)[name].shape
            B, A = torch.randn(experts, rows, 3), torch.randn(experts, 3, cols)
            parts = {module: (a, b) for module, a, b in _modules(name, TargetFactors(B, A, 3, 'concatenated', ()), targets[name])}
            for expert in range(experts):
                gate, up = (parts[template.format(expert=expert)] for template in targets[name].hf_module)
                assert torch.equal(torch.cat([gate[1] @ gate[0], up[1] @ up[0]]), B[expert] @ A[expert])
        engine.store.close()
