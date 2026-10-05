import pytest
import torch

from natlang_neuralese.train.trajectory_state import atomic_checkpoint, trajectory_optimizer, validate_resume


@pytest.mark.parametrize('policy', ['adamw', 'muon'])
def test_optimizer_state_roundtrip_and_parameter_group_rates(tmp_path, policy):
    torch.manual_seed(0)
    params = {'prompt': torch.nn.Parameter(torch.randn(3, 8))}
    lora = [torch.nn.Parameter(torch.randn(2, 8))]
    heads = [torch.nn.Parameter(torch.randn(13, 8)), torch.nn.Parameter(torch.randn(8, 8))]
    options = dict(vocab_size=13, lr=.01, lora_lr=.002, heads_lr=.003)
    optimizer = trajectory_optimizer(policy, params, lora, heads, **options)
    for g in optimizer.param_groups:
        expected = .01 if any(q is params['prompt'] for q in g['params']) else (.002 if any(q is lora[0] for q in g['params']) else .003)
        assert g['lr'] == expected
    values = list(params.values()) + lora + heads
    sum(q.square().sum() for q in values).backward()
    optimizer.step()
    path = tmp_path / 'checkpoint.pt'
    identity = {'input': 'fixed', 'optimizer': policy}
    atomic_checkpoint(path, {'schema': 'natlang.neuralese_recurrence_checkpoint/1', 'identity': identity,
                             'optimizer': optimizer.state_dict(), 'weights': [q.detach().clone() for q in values]})
    saved = torch.load(path, weights_only=False)
    validate_resume(saved, identity)
    with pytest.raises(ValueError, match='changed'):
        validate_resume(saved, {**identity, 'input': 'other'})
    other = trajectory_optimizer(policy, params, lora, heads, **options)
    other.load_state_dict(saved['optimizer'])
    assert len(other.param_groups) == len(optimizer.param_groups)
    assert not path.with_suffix('.pending').exists()


def test_evaluation_restores_training_rng_and_baseline_even_on_error():
    import random
    from natlang_neuralese.train.trajectory_state import evaluation_state
    rng = random.Random(12)
    stop = torch.Generator().manual_seed(13)
    baseline = {'value': 1.25}
    state = rng.getstate(), stop.get_state(), torch.get_rng_state(), random.getstate()
    with pytest.raises(RuntimeError):
        with evaluation_state(rng, stop, baseline):
            rng.random()
            torch.rand(3, generator=stop)
            torch.rand(3)
            random.random()
            baseline['value'] = 999
            raise RuntimeError('probe failure')
    assert rng.getstate() == state[0]
    assert torch.equal(stop.get_state(), state[1])
    assert torch.equal(torch.get_rng_state(), state[2])
    assert random.getstate() == state[3]
    assert baseline == {'value': 1.25}
