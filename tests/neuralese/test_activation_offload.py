import pytest
import torch

from natlang_neuralese.train.memory import offload_attention_tensors


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_saved_activation_offload_preserves_gradients_under_allocator_reuse():
    torch.manual_seed(0)
    x = torch.randn(1024, 256, device='cuda', requires_grad=True)
    def loss():
        value = x.square().mean()
        for i in range(24):
            # Temporary tensors are released during the forward; CUDA allocator
            # addresses can recur even though the saved values are different.
            part = torch.sin(x * (i + 1))
            value = value + part.square().mean()
        return value
    expected = loss()
    grad = torch.autograd.grad(expected, x)[0]
    with offload_attention_tensors(16*2**20, activations=True) as stats:
        actual = loss()
    got = torch.autograd.grad(actual, x)[0]
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(got, grad, atol=1e-7, rtol=1e-6)
    assert 0 < stats['offloaded_bytes'] <= 16*2**20
    assert stats['peak_offloaded_bytes'] <= 16*2**20


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_persistent_weights_are_not_offloaded():
    x = torch.randn(1024, 256, device='cuda', requires_grad=True)
    with offload_attention_tensors(4*2**20, activations=True, persistent_tensors=[x]) as stats:
        loss = x.square().mean()
    loss.backward()
    assert stats['offloaded_bytes'] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_staged_offload_budget_is_reused_and_replay_gradients_are_exact():
    from natlang_neuralese.train.staging import StagedWrites
    x = torch.randn(1024, 256, device='cuda', requires_grad=True)
    def compute():
        y = torch.sin(x * 2)
        return y.mean(dim=0), [y.square().mean()]
    value, terms = compute()
    expected = torch.autograd.grad(value.sum() + .3 * terms[0], x)[0]
    with offload_attention_tensors(4*2**20, activations=True, persistent_tensors=[x]) as stats:
        staged = StagedWrites()
        node = staged.add(compute)
        assert stats['live_offloaded_bytes'] == 0
        (node.value.sum() + staged.penalty_loss(.3)).backward()
        staged.backward(penalty_weight=.3)
        staged.clear()
        assert stats['live_offloaded_bytes'] == 0
    torch.testing.assert_close(x.grad, expected, atol=1e-7, rtol=1e-6)
    assert stats['offloaded_bytes'] > stats['peak_offloaded_bytes']
    assert 0 < stats['peak_offloaded_bytes'] <= 4*2**20


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_small_nonoffloaded_saved_tensors_do_not_retain_staged_graphs():
    import weakref
    from natlang_neuralese.train.staging import StagedWrites
    x = torch.randn(8, device='cuda', requires_grad=True)
    references = []
    def compute():
        value = x.sin()
        references.append(weakref.ref(value))
        return value.square(), []
    with offload_attention_tensors(2**20, activations=True):
        staged = StagedWrites()
        node = staged.add(compute)
        assert references[-1]() is None
        node.value.sum().backward()
        staged.backward()
        assert references[-1]() is None
        staged.clear()


def test_nonoffloaded_intermediate_storage_is_released_after_staged_forward():
    # Python Tensor wrappers can disappear while a C++ autograd cycle still
    # owns their storage. Inspect actual storage lifetime, not only wrappers.
    from natlang_neuralese.train.staging import StagedWrites
    x = torch.randn(128, 64, requires_grad=True)
    handles = []
    def compute():
        state = x
        for _ in range(16):
            state = (state * 1.01).sin()
            handles.append(state.untyped_storage()._weak_ref())
        return state.mean(dim=0), []
    try:
        with offload_attention_tensors(2**20, activations=True):
            staged = StagedWrites()
            node = staged.add(compute)
            assert all(torch.UntypedStorage._expired(handle) for handle in handles)
            node.value.sum().backward()
            staged.backward()
            assert all(torch.UntypedStorage._expired(handle) for handle in handles)
            staged.clear()
    finally:
        for handle in handles:
            torch.UntypedStorage._free_weak_ref(handle)
