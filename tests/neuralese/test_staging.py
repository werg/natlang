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
