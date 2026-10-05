import pytest
import torch

from natlang_neuralese.train.staging import GraphBudgetExceeded, StagedWrites, graph_memory_budget


@pytest.mark.parametrize('scale', [1., .25])
def test_staged_branching_dag_matches_joint_inputs_parameters_and_local_losses(scale):
    torch.manual_seed(1)
    x = torch.randn(3, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(3, 3, dtype=torch.float64, requires_grad=True)
    def write(value):
        y = torch.tanh(weight @ value)
        return y, [y.square().mean()]
    child, child_terms = write(x)
    left, left_terms = write(child)
    right, right_terms = write(child + x)
    objective = (left * right).sum() + .3 * sum(child_terms + left_terms + right_terms) / 3
    expected = torch.autograd.grad(objective * scale, (x, weight))
    staged = StagedWrites()
    child = staged.add(lambda: write(x))
    left = staged.add(lambda: write(child.value))
    right = staged.add(lambda: write(child.value + x))
    loss = (left.value * right.value).sum() + staged.penalty_loss(.3)
    torch.testing.assert_close(loss, objective)
    (loss * scale).backward()
    staged.backward(penalty_weight=.3, scale=scale)
    torch.testing.assert_close(x.grad, expected[0])
    torch.testing.assert_close(weight.grad, expected[1])
    assert staged.replay_max_abs_error == 0
    staged.clear()


def test_budget_retains_joint_graph_below_limit_and_stops_above():
    x = torch.tensor(2., requires_grad=True)
    with graph_memory_budget(100, measure=lambda: 99):
        x.square().backward()
    assert x.grad == 4
    with pytest.raises(GraphBudgetExceeded):
        with graph_memory_budget(100, measure=lambda: 101):
            x.square()


@pytest.mark.parametrize('scale', [1., .25])
@pytest.mark.parametrize('core_penalties', [False, True])
def test_separate_auxiliary_graph_matches_joint_branching_gradients(scale, core_penalties):
    torch.manual_seed(7)
    x = torch.randn(3, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(3, 3, dtype=torch.float64, requires_grad=True)
    def write(value):
        y = torch.tanh(weight @ value)
        return y, [y.square().mean()] if core_penalties else []
    def auxiliary(value):
        # Independent producer supervision still differentiates into the
        # actual ancestors, and shares trainable writer weights.
        return (weight @ value - 1).square().mean()
    child, child_terms = write(x)
    left, left_terms = write(child)
    right, right_terms = write(child + x)
    terms = sum(child_terms + left_terms + right_terms) + auxiliary(x) + auxiliary(child) + auxiliary(child + x)
    objective = (left * right).sum() + .3 * terms / 3
    expected = torch.autograd.grad(objective * scale, (x, weight))
    staged = StagedWrites()
    child = staged.add(lambda: write(x), auxiliary=lambda: auxiliary(x))
    left = staged.add(lambda: write(child.value), auxiliary=lambda: auxiliary(child.value))
    right = staged.add(lambda: write(child.value + x), auxiliary=lambda: auxiliary(child.value + x))
    loss = (left.value * right.value).sum() + staged.penalty_loss(.3)
    torch.testing.assert_close(loss, objective)
    assert staged.penalty_count == 3
    (loss * scale).backward()
    staged.backward(penalty_weight=.3, scale=scale)
    torch.testing.assert_close(x.grad, expected[0])
    torch.testing.assert_close(weight.grad, expected[1])
    assert staged.replay_max_abs_error == 0
    staged.clear()


def test_split_auxiliary_observation_predicts_joint_tape_and_releases_writer():
    import weakref
    parameter = torch.tensor(2., requires_grad=True)
    references = []
    observed = []
    measurements = iter([10, 35, 12, 52])
    def compute():
        value = parameter.square()[None]
        references.append(weakref.ref(value))
        return value, [value.mean()]
    def auxiliary():
        assert references[-1]() is None
        return (parameter - 1).square()
    staged = StagedWrites(observe=lambda value, size: observed.append(size), measure=lambda: next(measurements))
    node = staged.add(compute, auxiliary=auxiliary)
    assert observed == [65]  # 25 writer + 40 auxiliary, not the split maximum40
    assert node.penalty_values == (5.,)
    node.value.sum().backward()
    staged.backward(penalty_weight=.3)
    assert parameter.grad == pytest.approx(5.8)  # caller4 + local(writer4 + auxiliary2)*.3
