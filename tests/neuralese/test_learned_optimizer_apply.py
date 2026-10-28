"""Applying the learned optimizer with the tune.build plan/result contract, and the adaptation services."""
import dataclasses
import json

from test_learned_optimizer_bank import _component
from test_tune_soft_states import soft_engine
from test_tune_trainer import manifest_for, rows_for
from natlang_neuralese.serve.store import TensorStore
from natlang_neuralese.tune.learned.mixture import MixtureFormat
from natlang_neuralese.tune.learned.apply import main, service
from natlang_neuralese.tune.learned.bank import new_bank, save_bank, weight_svd_basis
from natlang_neuralese.tune.learned.bundle import verify_bundle
from natlang_neuralese.tune.learned.trainer import OuterRecipe, OuterTrainer
from natlang_neuralese.model.native_adapters import target_parameters


def operation(tmp_path):
    engine = soft_engine(tmp_path)
    store = tmp_path / 'store'
    engine.store.close()
    engine.store = TensorStore(store)
    names = sorted(target_parameters(engine.shared_model))[:6]
    bank = new_bank(engine, weight_svd_basis(engine, names, 4), MixtureFormat())
    components = [dataclasses.replace(_component(bank, seed), key=engine.native_write_text(f't{40 + seed} t{50 + seed}')['id'])
                  for seed in (1, 2, 3)]
    bank = bank.with_components(components, reason='planted')
    (tmp_path / 'bank.json').write_text(json.dumps({'bank': save_bank(engine, bank)}))
    trainer = OuterTrainer(engine, bank, OuterRecipe(writer={'kind': 'free-lora', 'rank': 2, 'alpha': 2.}), [])
    (tmp_path / 'state.json').write_text(json.dumps(trainer.state))
    manifest = manifest_for(engine)
    engine.store.close()
    # The fixture's chat template renders no reply target for teacher turns (as in test_tune_soft_states), so the
    # plan carries already-lowered training graphs; soft-state writing is covered by the trainer/proposal tests.
    rows = rows_for(engine, 'bx', 3) + rows_for(engine, 'by', 2, offset=2)
    (tmp_path / 'turns.jsonl').write_text(''.join(json.dumps(item) + '\n' for item in rows))
    plan = {'schema': 'natlang.tune-plan/v1', 'manifest': manifest, 'rows': 'turns.jsonl',
            'student': {'checkpoint': str(tmp_path / 'native.pt'), 'engine': {}}, 'softStates': False, 'exports': [],
            'learned': {'bank': str(tmp_path / 'bank.json'), 'state': str(tmp_path / 'state.json'), 'store': str(store),
                        'scopes': {'lambda': 3, 'folder': 3}, 'steps': 3, 'patience': 3, 'seed': 1}}
    (tmp_path / 'plan.json').write_text(json.dumps(plan))
    return str(tmp_path / 'plan.json'), bank, store


def test_apply_writes_a_build_result_with_the_complete_episode(tmp_path):
    plan, bank, _ = operation(tmp_path)
    assert main(['--plan', plan, '--out', str(tmp_path / 'out')]) == 0
    result = json.loads((tmp_path / 'out' / 'result.json').read_text())
    assert result['schema'] == 'natlang.tune.build-result/1'
    receipt = result['receipt']
    assert receipt['bank'] == bank.snapshot and len(receipt['history']) >= 1
    assert receipt['history'][0]['behavior']['choose'] is None  # the first proposal executes the mixture itself
    if receipt['selected']:
        assert {item['node'] for item in result['artifacts']} == {'folder:a', 'folder:b', 'lambda:x', 'lambda:y'}
        assert all((tmp_path / 'out' / 'artifacts' / f"{item['artifact']}.nnt").exists() for item in result['artifacts'])
    else:
        assert result['artifacts'] == [] and result['softStates'] == []


def test_services_propose_evaluate_refine_retrieve_and_release(tmp_path):
    plan, bank, store = operation(tmp_path)
    common = {'plan': plan, 'out': str(tmp_path / 'svc')}
    evidence = service({**common, 'op': 'evidence'})
    assert evidence['view']['items'] == [] and evidence['scopes'] == ['folder:a', 'folder:b', 'lambda:x', 'lambda:y']
    proposed = service({**common, 'op': 'propose', 'sample': True, 'seed': 3, 'focus': ['bx']})
    assert proposed['bundle']['representation']['bank'] == bank.snapshot
    evaluated = service({**common, 'op': 'evaluate', 'parent': proposed['bundle']})
    assert abs(evaluated['supportLoss'] - proposed['supportLoss']) < 1e-5
    retrieved = service({**common, 'op': 'retrieve', 'parent': None})
    assert retrieved['supportLoss'] <= evidence['incomingLoss'] + 1e-6
    refined = service({**common, 'op': 'refine', 'parent': proposed['bundle'], 'steps': 2})
    released = service({**common, 'op': 'release', 'parent': refined['bundle']})
    assert {item['node'] for item in released['artifacts']} == set(evidence['scopes'])
    controller = {'root': '/controller', 'release': 'earned-release'}
    service({**common, 'op': 'finish', 'parent': refined['bundle'], 'history': [{'accepted': True}], 'controller': controller})
    result = json.loads((tmp_path / 'svc' / 'result.json').read_text())
    assert result['receipt']['controller'] == controller
    assert result['receipt']['operatorPolicy'] == 'natlang-controller-v1'
    assert result['receipt']['selected'] == refined['bundle']['id']
    assert {item['node'] for item in result['artifacts']} == set(evidence['scopes'])
    import torch
    from natlang_neuralese.serve import load_native_engine
    engine = load_native_engine(str(tmp_path / 'native.pt'), device='cpu', dtype=torch.float32)
    engine.store.close()
    engine.store = TensorStore(store)
    try:
        for bundle in (proposed['bundle'], retrieved['bundle'], refined['bundle']):
            verify_bundle(engine, bundle, bank)
    finally:
        engine.store.close()


def test_tune_build_emits_free_tune_boundaries_after_each_commit(tmp_path):
    from test_native_serving_adapters import engine_fixture
    from natlang_neuralese.tune.build import main as build
    engine = engine_fixture(tmp_path, 'lfm')
    rows = rows_for(engine, 'bx', 2)
    (tmp_path / 'rows.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    plan = tmp_path / 'plan.json'
    plan.write_text(json.dumps({'manifest': manifest_for(engine), 'rows': 'rows.jsonl', 'emitBoundaries': True, 'label': 'fit',
                                'student': {'name': 'lfm', 'checkpoint': str(tmp_path / 'native.pt')},
                                'recipe': {'steps': 2, 'batch_rows': 2, 'prompt_weight': 0}}))
    assert build(['--plan', str(plan), '--out', str(tmp_path / 'out')]) == 0
    boundaries = [json.loads(line) for line in (tmp_path / 'out' / 'boundaries.jsonl').read_text().splitlines()]
    assert [item['stepAfter'] for item in boundaries] == [1, 2]
    assert boundaries[0]['kind'] == 'final-residual' and boundaries[1]['kind'] == 'incremental'
    assert set(boundaries[0]['nodesAfter']) == {'tree:', 'folder:a', 'lambda:x'}
    assert boundaries[1]['nodesBefore'] == boundaries[0]['nodesAfter']


def test_revision_plans_select_what_each_operator_changes(tmp_path):
    import pytest
    from natlang_neuralese.tune.learned import LearnedOptimizerError
    from natlang_neuralese.tune.learned.apply import Context
    plan, _, _ = operation(tmp_path)
    context = Context(plan, tmp_path / 'ctx')
    try:
        ancestry = {binding['id']: set(binding['ancestry']) & set(context.scopes) for binding in context.manifest['bindings']}
        assert context.selection({'bx': 'weights', 'by': 'keep'}) == (set(), ancestry['bx'])
        assert context.selection(None) == (set(context.soft), set(context.scopes))
        with pytest.raises(LearnedOptimizerError, match='not a binding of this program'):
            context.selection({'bz': 'weights'})
        with pytest.raises(LearnedOptimizerError, match='is not one of'):
            context.selection({'bx': 'rewrite'})
        with pytest.raises(LearnedOptimizerError, match='choose the operator "keep"'):
            context.selection({'bx': 'keep'})
        incoming = context.incoming_bundle()
        # Every operator changes exactly the selection; every other scope keeps its parent's adapter and mixture.
        plan_ops = {'bx': 'weights'}
        for bundle in (context.refine(None, steps=1, operations=plan_ops), context.retrieve(None, operations=plan_ops)[0],
                       context.propose(None, sample=False, seed=0, operations=plan_ops)[0]):
            assert set(bundle['adapters']) == set(context.scopes)
            for scope in set(context.scopes) - ancestry['bx']:
                assert bundle['adapters'][scope] == incoming['adapters'][scope]
                assert bundle['loraMixtures'][scope] == incoming['loraMixtures'][scope]
            assert bundle['instructions'] == incoming['instructions'] and bundle['skills'] == incoming['skills']
    finally:
        context.close()


def test_the_persistent_service_keeps_contexts_and_runs_commands(tmp_path):
    import io
    from natlang_neuralese.tune.learned.apply import serve
    plan, _, _ = operation(tmp_path)
    out = str(tmp_path / 'svc')
    calls = []
    lines = [{'id': 1, 'plan': plan, 'out': out, 'op': 'evidence'},
             {'id': 2, 'plan': plan, 'out': out, 'op': 'refine', 'parent': None, 'steps': 1, 'operations': {'bx': 'weights'}},
             {'id': 3, 'plan': plan, 'out': out, 'op': 'refine', 'parent': None, 'operations': {'bx': 'keep'}},
             {'id': 4, 'command': 'ladder', 'spec': {'x': 1}},
             {'id': 5, 'plan': plan, 'out': out, 'op': 'nonsense'}]
    output = io.StringIO()
    def commands(name, spec):
        calls.append((name, spec))
        print('a command writing to stdout does not corrupt the replies')
        return {'ran': name}
    import natlang_neuralese.tune.learned.apply as apply
    opened = []
    original = apply.Context.__init__
    def counting(self, *args, **kwargs):
        opened.append(args)
        original(self, *args, **kwargs)
    apply.Context.__init__ = counting
    try:
        serve(io.StringIO(''.join(json.dumps(line) + '\n' for line in lines)), output, commands=commands)
    finally:
        apply.Context.__init__ = original
    replies = [json.loads(line) for line in output.getvalue().splitlines()]
    assert [reply['id'] for reply in replies] == [1, 2, 3, 4, 5]
    assert len(opened) == 1  # one loaded context served every call of this plan
    assert replies[0]['result']['incomingLoss'] > 0
    refined = replies[1]['result']
    measured = json.loads((tmp_path / 'svc' / 'bundles' / f"{refined['bundle']['id']}.json").read_text())
    assert measured['supportLoss'] == refined['supportLoss']  # what a root verifier checks an episode's result against
    assert 'choose the operator "keep"' in replies[2]['error']
    assert replies[3]['result'] == {'ran': 'ladder'} and calls == [('ladder', {'x': 1})]
    assert 'unknown adaptation service' in replies[4]['error']
