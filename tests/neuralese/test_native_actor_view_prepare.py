"""natlang pins prepare a compact actor view in one call; the server loads it under the full checkpoint's identity."""
import json
import subprocess
import sys

import pytest
import torch

from natlang_neuralese.model.native_bundle import load_actor_view, prepare_actor_view
from natlang_neuralese.serve import load_native_engine
from natlang_neuralese.train.checkpoint_policy import CheckpointPolicy
from natlang_neuralese.train.native_continuation import TRAINING_SCHEMA
from natlang_neuralese.train.loop import capture_training_rng_state


def complete(tmp_path):
    from test_native_serving_adapters import engine_fixture
    engine = engine_fixture(tmp_path, 'lfm')
    state = CheckpointPolicy.load(engine.native_checkpoint)
    state.update(step=12, training={'schema': TRAINING_SCHEMA, 'global_step': 12,
        'optimizer': {'state': {'moment': torch.arange(7.)}}, 'rng': capture_training_rng_state('cpu'),
        'source_receipts': {'source': 'retained'}})
    path = tmp_path/'full.pt'
    torch.save(state, path)
    engine.store.close()
    return path


def test_prepared_view_serves_the_full_checkpoint_identity(tmp_path):
    path = complete(tmp_path)
    reference = prepare_actor_view(path, tmp_path/'views')
    assert prepare_actor_view(path, tmp_path/'views') == reference, 'a second preparation reuses the immutable view'
    view = load_actor_view(path, reference)
    assert 'training' not in view and view['actor_view']['training_resume'] is False
    full = load_native_engine(str(path), device='cpu')
    compact = load_native_engine(str(path), device='cpu', actor_view=reference)
    assert compact.native_inference_only and not full.native_inference_only
    prompt = list(range(5, 17))
    with torch.no_grad():
        a = full._native_logits(full._native_forward(prompt, None)['h_final'])
        b = compact._native_logits(compact._native_forward(prompt, None)['h_final'])
    assert torch.equal(a, b)


def test_cli_prints_the_receipt_and_refuses_incomplete_sources(tmp_path):
    path = complete(tmp_path)
    out = subprocess.run([sys.executable, '-m', 'natlang_neuralese.model.native_bundle', 'actor-view',
                          '--checkpoint', str(path), '--out', str(tmp_path/'views')], capture_output=True, text=True, check=True)
    reference = json.loads(out.stdout.strip().splitlines()[-1])
    assert set(reference) == {'path', 'sha256', 'bytes'}
    state = CheckpointPolicy.load(path)
    del state['training']['optimizer']
    torch.save(state, tmp_path/'partial.pt')
    with pytest.raises(ValueError, match='optimizer'):
        prepare_actor_view(tmp_path/'partial.pt', tmp_path/'views')
