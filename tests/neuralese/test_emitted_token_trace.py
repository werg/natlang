"""The token reference must follow actual value choices, not gold feedback or another generation."""
import torch
from natlang_neuralese.serve.direct_write import DirectWriter, write_direct
from natlang_neuralese.write import Opened


class CausalReference:
    num_layers = 1
    embedding_weight = torch.eye(3)
    def embed(self, ids):
        return self.embedding_weight[ids]
    def logits(self, hidden):
        return hidden * 10
    def run_layers(self, inputs, layers, cache):
        return inputs.roll(1, -1), cache


class Correction(torch.nn.Module):
    config = {'blocks': 1, 'kernel': 1}
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3, bias=False)
        torch.nn.init.zeros_(self.linear.weight)
    def forward(self, hidden, segments=None, last_only=False, anchor=None):
        if last_only:
            hidden = hidden[:, -1:]
        return anchor + self.linear(hidden)


def opened(backbone):
    top = torch.tensor([[1., 0., 0.]])
    return Opened(None, top, backbone.logits(top), top)


def test_identity_uses_emitted_choices_even_when_feedback_is_gold():
    backbone, correction = CausalReference(), Correction()
    writer = DirectWriter(backbone, correction, opened(backbone), 2, [2], length=2,
                          tokens=[2, 2]).forced_sequence()
    payload = writer.complete()[0]
    assert writer.emitted_token_ids().tolist() == [[0, 0]]
    assert torch.equal(payload, backbone.embed(writer.emitted_token_ids()))
    payload.sum().backward()
    assert correction.linear.weight.grad.abs().sum() > 0


def test_incremental_identity_and_sequence_replay_share_the_emitted_trace():
    backbone, correction = CausalReference(), Correction()
    greedy = write_direct(backbone, correction, opened(backbone), 2, [2], length=2)
    assert greedy.emitted_token_ids().tolist() == [[0, 1]]
    replay = DirectWriter(backbone, correction, opened(backbone), 2, [2], length=2,
                          tokens=[0, 1]).forced_sequence()
    assert torch.equal(greedy.complete()[0], replay.complete()[0])
    assert torch.equal(greedy.complete()[0], backbone.embed(greedy.emitted_token_ids()))
