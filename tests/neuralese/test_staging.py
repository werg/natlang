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


def test_local_staging_collector_runs_per_node_and_full_collector_runs_on_clear():
    events = []
    source = torch.tensor(1., requires_grad=True)
    weight = torch.tensor(2., requires_grad=True)
    staged = StagedWrites(
        collect=lambda: events.append('full'),
        collect_local=lambda: events.append('local'))
    first = staged.add(lambda: (source * weight, []))
    second = staged.add(lambda: (first.value * weight, []))
    second.value.sum().backward()
    staged.backward()
    torch.testing.assert_close(source.grad, torch.tensor(4.))
    torch.testing.assert_close(weight.grad, torch.tensor(4.))
    assert events == ['local', 'local', 'local', 'local']
    staged.clear()
    assert events == ['local', 'local', 'local', 'local', 'full']


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


@pytest.mark.parametrize('scale', [1., .25])
@pytest.mark.parametrize('core_penalties', [False, True])
def test_batched_frontiers_match_joint_shared_child_and_auxiliary_adjoints(scale, core_penalties):
    torch.manual_seed(13)
    x = torch.randn(3, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(3, 3, dtype=torch.float64, requires_grad=True)
    calls = []
    def batch(inputs):
        calls.append(len(inputs))
        output = torch.tanh(torch.stack(inputs) @ weight.T)
        values = list(output.unbind())
        return values, [[v.square().mean()] if core_penalties else [] for v in values]
    def aux(value):
        return (weight @ value - 1).square().mean()
    children, cp = batch([x, x * 2])
    parents, pp = batch([children[0] + children[1], children[0] + x])
    penalties = sum(sum(p) for p in cp + pp) + aux(x) + aux(x * 2) + aux(children[0] + children[1]) + aux(children[0] + x)
    expected_loss = (parents[0] * parents[1]).sum() + .3 * penalties / 4
    expected = torch.autograd.grad(expected_loss * scale, (x, weight))
    calls.clear()
    staged = StagedWrites()
    children = staged.add_batch(lambda: batch([x, x * 2]), auxiliaries=[lambda: aux(x), lambda: aux(x * 2)])
    parents = staged.add_batch(lambda: batch([children[0].value + children[1].value, children[0].value + x]),
        auxiliaries=[lambda: aux(children[0].value + children[1].value), lambda: aux(children[0].value + x)])
    loss = (parents[0].value * parents[1].value).sum() + staged.penalty_loss(.3)
    torch.testing.assert_close(loss, expected_loss)
    (loss * scale).backward()
    staged.backward(penalty_weight=.3, scale=scale)
    for actual, reference in zip((x.grad, weight.grad), expected):
        torch.testing.assert_close(actual, reference)
    assert calls == [2, 2, 2, 2]  # two primal frontiers and two batched replays
    assert staged.replay_max_abs_error == 0
    staged.clear()
    assert all(n.batch is None for n in children + parents)


def test_batched_frontier_rejects_changed_replay_membership():
    parameter = torch.tensor(2., requires_grad=True)
    count = [0]
    def compute():
        count[0] += 1
        values = [parameter[None]] * (2 if count[0] == 1 else 1)
        return values, [[] for _ in values]
    staged = StagedWrites()
    rows = staged.add_batch(compute, auxiliaries=[None, None])
    sum(n.value.sum() for n in rows).backward()
    with pytest.raises(RuntimeError, match='changed batch membership'):
        staged.backward()
    staged.clear()


def test_batch_admission_observes_writer_tape_without_released_gold_tapes():
    p = torch.tensor(1., requires_grad=True)
    measurements = iter([100, 125])
    observations = []
    staged = StagedWrites(measure=lambda: next(measurements))
    nodes = staged.add_batch(lambda: ([p * 2, p * 3], [[], []]),
        auxiliaries=[lambda: p.square() * 4, lambda: p.square() * 5],
        observe=lambda values, size: observations.append(size))
    assert observations == [25]
    assert nodes[0].penalty_values == (4.,) and nodes[1].penalty_values == (5.,)
    staged.clear()
