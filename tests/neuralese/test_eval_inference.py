import torch

from natlang_neuralese.eval.inference import evaluation_inference


class _Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ffn_chunk_tokens = 0


def test_evaluation_inference_disables_autograd_and_restores_state():
    backbone = _Backbone().train()
    heads = torch.nn.Linear(2, 2).train()

    with evaluation_inference(backbone, heads, ffn_owner=backbone, ffn_chunk_tokens=1024):
        assert not backbone.training
        assert not heads.training
        assert backbone.ffn_chunk_tokens == 1024
        assert not torch.is_grad_enabled()
        assert torch.is_inference_mode_enabled()

    assert backbone.training
    assert heads.training
    assert backbone.ffn_chunk_tokens == 0
    assert torch.is_grad_enabled()


def test_evaluation_inference_restores_state_after_exception():
    backbone = _Backbone().train()

    try:
        with evaluation_inference(backbone, ffn_owner=backbone, ffn_chunk_tokens=1024):
            raise RuntimeError("evaluation failed")
    except RuntimeError:
        pass

    assert backbone.training
    assert backbone.ffn_chunk_tokens == 0
