import torch

from natlang_neuralese.train.projection_anchor import gold_aligned_projection_errors, scale_gradient


class _Heads(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.content_map = torch.nn.Linear(3, 3, bias=False)
        self.feedback_map = torch.nn.Linear(3, 3, bias=False)

    def content(self, sketch, top):
        return self.content_map(top) + sketch

    def feedback(self, shallow):
        return self.feedback_map(shallow)


def test_gold_aligned_projection_anchor_trains_both_heads_and_detaches_raw_targets():
    torch.manual_seed(9)
    heads = _Heads()
    top = torch.randn(1, 4, 3, requires_grad=True)
    shallow = torch.randn(1, 4, 3, requires_grad=True)
    target = torch.randn(1, 4, 3, requires_grad=True)

    full_projection = heads.content(torch.zeros_like(top), top)
    shallow_projection = heads.feedback(shallow)
    full, feedback = gold_aligned_projection_errors(full_projection, shallow_projection, target)
    loss = (full.mean() + feedback.mean()) * 0.5
    loss.backward()

    assert full.shape == feedback.shape == (1, 4)
    assert heads.content_map.weight.grad is not None
    assert heads.feedback_map.weight.grad is not None
    assert top.grad is not None and shallow.grad is not None
    assert target.grad is None


def test_projection_anchor_can_keep_forward_values_and_reduce_backbone_gradient():
    value = torch.tensor([1.0, -2.0], requires_grad=True)
    scaled = scale_gradient(value, 0.05)
    assert torch.equal(scaled.detach(), value.detach())
    scaled.square().sum().backward()
    torch.testing.assert_close(value.grad, torch.tensor([0.1, -0.2]))
