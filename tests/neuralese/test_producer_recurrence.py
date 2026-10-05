import torch

from natlang_neuralese.train.recurrence import ProducerMemo, is_acyclic


def test_dag_check_rejects_cycles_and_accepts_shared_children():
    assert is_acyclic({'a': {'c'}, 'b': {'c'}, 'c': set()})
    assert not is_acyclic({'a': {'b'}, 'b': {'a'}})
    assert not is_acyclic({'a': {'a'}})


def test_shared_producer_retains_all_reader_gradients_and_depth_identity():
    x = torch.tensor(2., requires_grad=True)
    memo = ProducerMemo()
    calls = []
    def compute():
        calls.append(1)
        return x.square()
    a, b = memo.write('child', 1, compute), memo.write('child', 1, compute)
    assert a is b and len(calls) == 1 and memo.hits == 1
    (a * 3 + b * 5).backward()
    assert x.grad == 32
    memo.write('child', 2, compute)
    assert len(calls) == 2


def test_disabled_memo_keeps_stochastic_writes_independent():
    memo = ProducerMemo(False)
    a = memo.write('child', 1, lambda: object())
    b = memo.write('child', 1, lambda: object())
    assert a is not b


def test_probe_local_memo_matches_fresh_writes_and_does_not_cross_updates():
    weight = torch.tensor(2.)
    calls = []
    def probe(shared):
        memo = ProducerMemo(shared)
        def child():
            calls.append('child')
            return weight.square()
        def parent():
            calls.append('parent')
            return memo.write('child', 2, child) + weight
        with torch.no_grad():
            return torch.stack([memo.write('parent', 1, parent) * reader
                                for reader in (1., 3., 5.)])
    expected = probe(False)
    calls.clear()
    actual = probe(True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert calls == ['parent', 'child']
    weight.fill_(4.)
    assert not torch.equal(probe(True), actual)
    torch.testing.assert_close(probe(True), probe(False), rtol=0, atol=0)


def test_named_curriculum_preserves_joint_and_bounds_sampled_chains():
    import pytest
    from natlang_neuralese.train.recurrence import curriculum_max_writes
    assert curriculum_max_writes('joint', 0) == 0
    assert curriculum_max_writes('joint', 2) == 2
    assert curriculum_max_writes('sampled-chain', 0) == 1
    assert curriculum_max_writes('sampled-chain', 1) == 1
    with pytest.raises(ValueError, match='sampled-chain'):
        curriculum_max_writes('sampled-chain', 2)
