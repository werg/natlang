import json
import shutil

import pytest
import torch

from test_native_serving_adapters import engine_fixture
from test_tune_trainer import manifest_for, rows_for
from natlang_neuralese.serve import load_native_engine
from natlang_neuralese.tune.build import main as build
from natlang_neuralese.tune.learned import lane


def test_relocated_checkpoint_runs_actual_tune_fit_and_rejects_wrong_snapshot(tmp_path):
    engine = engine_fixture(tmp_path, 'lfm')
    engine.store.close()
    checkpoint = tmp_path / 'native.pt'
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    state['model_config']['base'] = '/missing/original/base'
    torch.save(state, checkpoint)
    with pytest.raises(FileNotFoundError):
        load_native_engine(str(checkpoint), device='cpu', dtype=torch.float32)
    engine = load_native_engine(str(checkpoint), base=str(tmp_path / 'base'), device='cpu', dtype=torch.float32)
    manifest = manifest_for(engine)
    rows = rows_for(engine, 'bx', 2)
    engine.store.close()
    (tmp_path / 'rows.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    plan = {'manifest': manifest, 'rows': 'rows.jsonl', 'softStates': False, 'exports': [],
            'student': {'checkpoint': str(checkpoint), 'engine': {'base': str(tmp_path / 'base'), 'device': 'cpu'}},
            'recipe': {'steps': 1, 'batch_rows': 2, 'prompt_weight': 0}}
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps(plan))
    assert build(['--plan', str(path), '--out', str(tmp_path / 'out')]) == 0
    assert json.loads((tmp_path / 'out' / 'result.json').read_text())['artifacts']
    wrong = tmp_path / 'wrong'
    shutil.copytree(tmp_path / 'base', wrong)
    with (wrong / 'config.json').open('a') as stream:
        stream.write(' ')
    with pytest.raises(ValueError, match='base inventory differs'):
        load_native_engine(str(checkpoint), base=str(wrong), device='cpu', dtype=torch.float32)


def test_lane_binds_relocated_base_into_training_and_bank_transition(tmp_path):
    op = tmp_path / 'op'
    (op / 'tunes').mkdir(parents=True)
    for name in ('one', 'two'):
        (op / 'tunes' / f'{name}.json').write_text(json.dumps({
            'schema': lane.RESULT_SCHEMA, 'program': name, 'root': f'/{name}', 'family': name,
            'sft_rows': 1, 'sft_plan': f'/{name}/plan.json', 'description': name}))
    checkpoint = tmp_path / 'checkpoint.pt'
    checkpoint.write_bytes(b'checkpoint')
    base = tmp_path / 'base'
    base.mkdir()
    templates = __import__('pathlib').Path(__file__).resolve().parents[2] / 'training/neuralese/learned-optimizer'
    lane.specs(op, checkpoint, templates, base=base, held_fraction=0)
    for name in ('bank-init-spec.json', 'corpus-spec.json', 'train-template.json'):
        assert json.loads((op / name).read_text())['base'] == str(base)
    state = op / 'optimizer/state.json'
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(json.dumps({'step': 1, 'labels_stale': True}))
    action = lane.outer_next(op)
    assert json.loads(__import__('pathlib').Path(action['adopt']).read_text())['base'] == str(base)
