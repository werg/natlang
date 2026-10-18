"""Relative production quanta retain actual optimizer, RNG and exposure progress."""
import json

import pytest
import torch

from natlang_neuralese.train import native_state_sft as trainer


@pytest.mark.parametrize('count', ['0', '-2'])
def test_invalid_relative_quantum_fails_before_model_load(tmp_path, monkeypatch, count):
    monkeypatch.setattr(trainer, 'load_native_model', lambda *a, **kw: pytest.fail('model allocation'))
    with pytest.raises(ValueError, match='refresh-after-updates must be a positive integer'):
        trainer.main(['--native-graphs', 'unused', '--held-native-graphs', 'unused',
                      '--out', str(tmp_path/'out'), '--refresh-after-updates', count])


def _same(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _same(a, b)
    else:
        # NumPy RNG state stores its integer stream as an array.
        if hasattr(left, 'shape'):
            assert (left == right).all()
        else:
            assert left == right


def test_actual_nonaligned_resume_quantum_matches_uninterrupted_state(tmp_path, monkeypatch):
    from test_native_model import tiny_source
    from natlang_neuralese.model.native_model import load_native_model
    from natlang_neuralese.data.native_graph import SCHEMA
    base = tmp_path/'base'
    tiny_source(base)
    _, _, roles, config = load_native_model(base, dtype=torch.float32, rank=4)
    op, close, consume, end_consume = roles.ids
    row = {'schema': SCHEMA, 'id': 'train', 'split': 'train', 'source_groups': ['train'],
           'tokenizer_fingerprint': config['tokenizer_fingerprint'], 'nodes': [
               {'name': 'writer', 'token_ids': [1, op, 5, 6, close],
                'target_weights': [0, 1, 1, 1, 1], 'producing': [[2, 4]]},
               {'name': 'reader', 'token_ids': [3, consume, end_consume, 11, 2],
                'target_weights': [0, 1, 0, 1, 1],
                'imports': [{'at': 2, 'source': 'writer', 'start': 2, 'stop': 4}]}]}
    train = tmp_path/'train.jsonl'
    held = tmp_path/'held.jsonl'
    train.write_text(json.dumps(row)+'\n')
    held.write_text(json.dumps({**row, 'id': 'held', 'split': 'test', 'source_groups': ['held']})+'\n')
    args = ['--base', str(base), '--dtype', 'float32', '--rank', '4', '--device', 'cpu',
            '--native-graphs', str(train), '--held-native-graphs', str(held),
            '--epochs', '8', '--eval-every', '1', '--seed', '1']
    run = tmp_path/'run'
    assert trainer.main([*args, '--out', str(run), '--checkpoint-every', '1', '--refresh-at-checkpoint']) == 0
    saved = torch.load(run/'checkpoint.pt', weights_only=False)
    assert saved['training']['step'] == 1
    parent = tmp_path/'parent.pt'
    parent.write_bytes((run/'checkpoint.pt').read_bytes())
    calls = []
    original_save = trainer._save
    def observe(policy, *a, **kw):
        calls.append(kw['step'])
        return original_save(policy, *a, **kw)
    monkeypatch.setattr(trainer, '_save', observe)
    # Absolute checkpoint 2 must save but must NOT end three new updates from 1.
    assert trainer.main([*args, '--out', str(run), '--checkpoint-every', '2',
                         '--refresh-at-checkpoint', '--refresh-after-updates', '3']) == 0
    assert 2 in calls and 4 in calls
    state4 = torch.load(run/'checkpoint.pt', weights_only=False)
    report = json.loads((run/'report.json').read_text())
    assert state4['training']['step'] == state4['training']['global_step'] == 4
    assert state4['training']['exposure_counts'] == {'train': 4}
    assert report['continuation_boundary']['new_optimizer_updates'] == 3
    assert report['continuation_boundary']['reached'] and not report['completed']
    # A subsequent quantum has a fresh relative clock and forces an off-cadence save.
    calls.clear()
    assert trainer.main([*args, '--out', str(run), '--checkpoint-every', '2',
                         '--refresh-after-updates', '3']) == 0
    assert 6 in calls and 7 in calls
    resumed = torch.load(run/'checkpoint.pt', weights_only=False)
    assert resumed['training']['step'] == 7
    assert resumed['training']['exposure_counts'] == {'train': 7}
    baseline = tmp_path/'baseline'
    baseline.mkdir()
    (baseline/'checkpoint.pt').write_bytes(parent.read_bytes())
    assert trainer.main([*args, '--out', str(baseline), '--checkpoint-every', '2',
                         '--refresh-after-updates', '6']) == 0
    uninterrupted = torch.load(baseline/'checkpoint.pt', weights_only=False)
    # Every parameter/moment and the resumed RNG stream match the same six updates.
    for key in resumed:
        if key != 'training':
            _same(resumed[key], uninterrupted[key])
    for key in ('optimizer', 'rng', 'step', 'global_step', 'exposure_counts',
                'epoch', 'cursor', 'source_row_bindings', 'source_row_order'):
        _same(resumed['training'][key], uninterrupted['training'][key])
