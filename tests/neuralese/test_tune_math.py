import pytest
import torch

from natlang_neuralese.tune import TuneMathError
from natlang_neuralese.tune.bases import BasisSlice, canonical_basis, verify_nested
from natlang_neuralese.tune.gates import family_norm, gate_l1, normalize_family, project_gates, q_weights
from natlang_neuralese.tune.objective import depth_penalty, program_loss, total_loss, trajectory_losses
from natlang_neuralese.tune.parameterization import FreeLoRADelta, SubspaceDelta, dense, init_free, init_subspace
from natlang_neuralese.tune.similarity import delta_cosine
from natlang_neuralese.tune.stack import StackMember, stack_dense, stack_factors

IDENTITY = {'architecture': 'test', 'checkpoint_sha256': '0'*64}


def weight(shape, seed):
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float64).float()


def core(rank, seed, leading=()):
    return torch.randn((*leading, rank, rank), generator=torch.Generator().manual_seed(seed), dtype=torch.float32)


def test_nested_slices_are_views_of_one_canonical_basis():
    basis = canonical_basis(IDENTITY, 'w', weight((12, 10), 1), 8)
    left, right = basis.slice(3)
    assert left.data_ptr() == basis.left.data_ptr() and left.shape == (12, 3) and right.shape == (10, 3)
    assert torch.allclose(basis.left.T @ basis.left, torch.eye(8), atol=1e-5)
    with pytest.raises(TuneMathError):
        basis.slice(9)
    other = canonical_basis(IDENTITY, 'w', weight((12, 10), 2), 8)
    assert other.basis_id == basis.basis_id  # same declared key, different content
    assert canonical_basis({**IDENTITY, 'checkpoint_sha256': '1'*64}, 'w', weight((12, 10), 1), 8).basis_id != basis.basis_id
    with pytest.raises(TuneMathError):
        verify_nested([BasisSlice(basis.basis_id, 2), BasisSlice('nb1:other', 2)], {basis.basis_id: basis})
    forged = type(basis)(basis.basis_id, 'w', 8, other.left, other.right, basis.content_sha256)
    with pytest.raises(TuneMathError):
        verify_nested([BasisSlice(basis.basis_id, 2)], {basis.basis_id: forged})


def test_common_basis_sum_is_exact_and_rank_bounded_by_max():
    basis = canonical_basis(IDENTITY, 'w', weight((16, 14), 3), 8)
    bases = {basis.basis_id: basis}
    members = [StackMember('family:a', 'family', 0.7, {'w': SubspaceDelta(basis.basis_id, 8, core(8, 4))}),
               StackMember('tree:root', 'tree', 1., {'w': SubspaceDelta(basis.basis_id, 4, core(4, 5))}),
               StackMember('lambda:x', 'lambda', 1., {'w': SubspaceDelta(basis.basis_id, 2, core(2, 6))})]
    factors = stack_factors(members, bases, {'w': (16, 14)})['w']
    assert factors.mode == 'common-basis' and factors.rank == 8
    expected = sum(member.scale * dense(member.deltas['w'], bases).double() for member in members)
    assert torch.allclose(factors.dense((16, 14)).double(), expected, atol=1e-5)
    assert factors.B.shape[-1] == 8 and torch.linalg.matrix_rank(factors.dense((16, 14)), rtol=1e-5) <= 8


def test_free_and_unrelated_bases_concatenate_with_scaling_once():
    basis = canonical_basis(IDENTITY, 'w', weight((9, 7), 7), 4)
    bases = {basis.basis_id: basis}
    free = init_free((9, 7), 3, alpha=6., seed=1)
    free.B = weight((9, 3), 8)
    members = [StackMember('folder:f', 'folder', 1., {'w': SubspaceDelta(basis.basis_id, 2, core(2, 9))}),
               StackMember('lambda:y', 'lambda', 0.5, {'w': free})]
    factors = stack_factors(members, bases, {'w': (9, 7), 'v': (5, 4)})
    assert factors['w'].mode == 'concatenated' and factors['w'].rank == 5
    assert factors['v'].mode == 'zero' and factors['v'].rank == 0
    assert torch.linalg.matrix_rank(factors['w'].dense((9, 7)), rtol=1e-5) <= 2 + 3
    expected = dense(members[0].deltas['w'], bases) + 0.5 * (6./3) * free.B @ free.A
    assert torch.allclose(factors['w'].dense((9, 7)), expected, atol=1e-5)
    with pytest.raises(TuneMathError):
        stack_factors([StackMember('n', 'lambda', 1., {'missing': free})], bases, {'w': (9, 7)})
    with pytest.raises(TuneMathError):
        stack_factors([members[1], members[1]], bases, {'w': (9, 7)})


def test_expert_banks_compose_per_expert():
    bank = weight((3, 8, 6), 10)
    basis = canonical_basis(IDENTITY, 'experts', bank, 4)
    bases = {basis.basis_id: basis}
    members = [StackMember('tree:t', 'tree', 1., {'e': SubspaceDelta(basis.basis_id, 4, core(4, 11, (3,)))}),
               StackMember('lambda:l', 'lambda', 1., {'e': FreeLoRADelta(weight((3, 8, 2), 12), weight((3, 2, 6), 13), 2.)})]
    deltas = stack_dense(members, bases, {'e': (3, 8, 6)})
    for expert in range(3):
        left, right = basis.left[expert, :, :4], basis.right[expert, :, :4]
        expected = left @ members[0].deltas['e'].core[expert] @ right.T + members[1].deltas['e'].B[expert] @ members[1].deltas['e'].A[expert]
        assert torch.allclose(deltas['e'][expert], expected, atol=1e-5)


def test_zero_init_free_lora_has_zero_delta_and_live_b_gradient():
    delta = init_free((6, 5), 2, seed=3)
    assert torch.count_nonzero(dense(delta)) == 0 and torch.count_nonzero(delta.A) > 0
    delta.B.requires_grad_(True)
    x = weight((4, 5), 14)
    (x @ (weight((6, 5), 15) + dense(delta)).T).sum().backward()
    assert torch.count_nonzero(delta.B.grad) > 0
    sub = init_subspace(canonical_basis(IDENTITY, 'w', weight((6, 5), 16), 3), 3)
    assert sub.core.dtype == torch.float32 and torch.count_nonzero(sub.core) == 0


def test_gates_projection_l1_and_frozen_family_amplitude():
    gates = {'lambda:a': torch.tensor([1.4, -0.2]), 'lambda:b': torch.tensor([0.3]), 'lambda:none': torch.tensor([])}
    project_gates(gates, 1.)
    assert gates['lambda:a'].tolist() == [1.0, 0.0]
    assert torch.isclose(gate_l1(gates, 0.5), torch.tensor(0.5 * (1.0 + 0.3) / 2))
    factors = {'w': (weight((6, 2), 17), weight((2, 5), 18)), 'v': (weight((2, 4, 1), 19), weight((2, 1, 3), 20))}
    q = q_weights({'w': (6, 5), 'v': (2, 4, 3)})
    direct = (q['w'] * (factors['w'][0] @ factors['w'][1]).square().sum() + q['v'] * (factors['v'][0] @ factors['v'][1]).square().sum()).sqrt()
    assert torch.isclose(family_norm(factors, q), direct, rtol=1e-5)
    scaled, _ = normalize_family(factors, 0.25, q)
    assert torch.isclose(family_norm(scaled, q), torch.tensor(0.25), rtol=1e-5)
    with pytest.raises(TuneMathError):
        normalize_family({'w': (torch.zeros(6, 2), weight((2, 5), 21))}, 0.25, q)


def test_gauge_invariant_delta_similarity():
    B, A = weight((7, 3), 22).double(), weight((3, 5), 23).double()
    Q = weight((3, 3), 24).double() + 3 * torch.eye(3, dtype=torch.float64)
    q = {'w': 1/35}
    same = delta_cosine({'w': (B, A)}, {'w': (B @ Q, torch.linalg.solve(Q, A))}, q)
    assert torch.isclose(same, torch.tensor(1., dtype=torch.float64))
    C, D = weight((7, 2), 25).double(), weight((2, 5), 26).double()
    dense_cos = ((B @ A) * (C @ D)).sum() / ((B @ A).norm() * (C @ D).norm())
    assert torch.isclose(delta_cosine({'w': (B, A)}, {'w': (C, D)}, q), dense_cos)


def test_objective_normalization_rejections_and_terms():
    logprobs = torch.log(torch.tensor([[0.5, 0.25, 1.], [0.5, 1., 1.]]))
    weights = torch.tensor([[1., 0.5, 0.], [1., 0., 0.]])
    losses = trajectory_losses(logprobs, weights)
    assert torch.allclose(losses, torch.tensor([-(1*torch.log(torch.tensor(.5)) + .5*torch.log(torch.tensor(.25)))/1.5, torch.log(torch.tensor(2.))]))
    with pytest.raises(TuneMathError):
        trajectory_losses(logprobs, torch.tensor([[1., 0, 0], [0, 0, 0]]))
    rows = torch.tensor([1., 3., 5.])
    assert torch.isclose(program_loss(rows, ['a', 'a', 'b'], [1., 1., 0.5]), torch.tensor(((1+3)/2 + 2.5)/2))
    with pytest.raises(TuneMathError):
        program_loss(rows, ['a', 'a', 'b'], [1., 1., 1.], occurrences=['o1', 'o1', 'o2'])
    factors = {'lambda:x': {'w': (weight((4, 2), 27), weight((2, 3), 28))}}
    q = {'w': 1/12}
    penalty = depth_penalty(factors, {'lambda:x': 0.1}, q)
    assert torch.isclose(penalty, 0.1/12 * (factors['lambda:x']['w'][0] @ factors['lambda:x']['w'][1]).square().sum(), rtol=1e-5)
    with pytest.raises(TuneMathError):
        depth_penalty(factors, {}, q)
    total, parts = total_loss(rows, ['a', 'a', 'b'], [1., 1., 0.5], factors, {'lambda:x': 0.1}, q,
                              {'a': torch.tensor([0.5])}, 0.2)
    assert torch.isclose(total, parts['task'] + parts['depth'] + parts['gate'])
