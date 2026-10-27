"""Applying the learned optimizer with the tune.build plan/result contract, and the adaptation services."""
import dataclasses
import json

from test_learned_optimizer_bank import _component
from test_learned_optimizer_writer import GRID
from test_tune_soft_states import soft_engine
from test_tune_trainer import manifest_for, rows_for
from natlang_neuralese.serve.store import TensorStore
from natlang_neuralese.tune.learned import numbers as N
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
    bank = new_bank(engine, weight_svd_basis(engine, names, 4), N.NumberFormat(.08))
    components = [dataclasses.replace(_component(bank, seed), key=engine.native_write_text(f't{40 + seed} t{50 + seed}')['id'])
                  for seed in (1, 2, 3)]
    bank = bank.with_components(components, reason='planted')
    (tmp_path / 'bank.json').write_text(json.dumps({'bank': save_bank(engine, bank)}))
    trainer = OuterTrainer(engine, bank, GRID, OuterRecipe(writer={'kind': 'free-lora', 'rank': 2, 'alpha': 2.}), [])
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
                        'scopes': {'lambda': 3, 'folder': 3}, 'steps': 3, 'patience': 3, 'seed': 1,
                        'grid_tokens': {**GRID.__dict__, 'digits': list(GRID.digits)}}}
    (tmp_path / 'plan.json').write_text(json.dumps(plan))
    return str(tmp_path / 'plan.json'), bank, store


def test_apply_writes_a_build_result_with_the_complete_episode(tmp_path):
    plan, bank, _ = operation(tmp_path)
    assert main(['--plan', plan, '--out', str(tmp_path / 'out')]) == 0
    result = json.loads((tmp_path / 'out' / 'result.json').read_text())
    assert result['schema'] == 'natlang.tune.build-result/1'
    receipt = result['receipt']
    assert receipt['bank'] == bank.snapshot and len(receipt['history']) >= 1
    assert receipt['history'][0]['behavior']['sampled'] is False  # the first proposal decodes deterministically
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
