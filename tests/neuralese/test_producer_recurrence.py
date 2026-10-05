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
