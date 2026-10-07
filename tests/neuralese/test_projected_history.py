"""Projection diagnostics consume next-token payloads at previous-token inputs."""
from types import SimpleNamespace

import pytest
import torch

from natlang_neuralese.eval.projected_history import projected_history_metrics


class CausalCycle(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_weight = torch.eye(16)
        self.controls = SimpleNamespace(close_id=15)
        self.inputs = []

    def embed(self, ids):
        return self.embedding_weight[ids]

    def forward_ids(self, ids, **kwargs):
        states = self.embed(ids)
        return {'h_cut':states, 'h_final':states}

    def forward_embeds(self, inputs, **kwargs):
        self.inputs.append(inputs.clone())
        return {'h_final':inputs}

    def logits(self, states):
        return states.roll(1, -1)*10


class CycleContent:
    def __init__(self, shift=1):
        self.shift = shift

    def __call__(self, sketch, top):
        return top.roll(self.shift, -1)

    def reference(self, top):
        return top.roll(1, -1)


class CycleHeads:
    def __init__(self, shift=1):
        self.cutoff = 1
        self.read_markers = False
        self.content = CycleContent(shift)
        self.read_calls = []

    def feedback(self, top):
        return top.roll(1, -1)

    def read_embeddings(self, backbone, payload):
        self.read_calls.append(payload.clone())
        return payload


def test_projected_history_uses_production_reader_and_causal_shift():
    backbone = CausalCycle(); heads = CycleHeads()
    prefix = torch.tensor([[1,2]]); span = torch.tensor([[3,4,5]])
    report = projected_history_metrics(backbone, heads, prefix, span)
    assert len(heads.read_calls)==4
    for history in backbone.inputs:
        assert torch.equal(history, backbone.embed(torch.tensor([[1,2,3,4]])))
    for scores in report.values():
        assert scores['whole']['tokens']==3
        assert scores['whole']['argmax_agreement_with_gold']==1
        assert scores['whole']['ce_delta_from_gold']==pytest.approx(0)


def test_wrong_full_projection_is_visible_without_harming_reference_control():
    report = projected_history_metrics(CausalCycle(), CycleHeads(shift=2),
                                      torch.tensor([[1,2]]), torch.tensor([[3,4,5]]))
    assert report['reference']['whole']['argmax_agreement_with_gold']==1
    assert report['full_projection']['whole']['argmax_agreement_with_gold']==pytest.approx(1/3)
    assert report['full_projection']['whole']['ce_delta_from_gold']>0


def test_marked_read_profile_cannot_silently_change_positions():
    heads = CycleHeads(); heads.read_markers=True
    with pytest.raises(ValueError, match='raw read profile'):
        projected_history_metrics(CausalCycle(), heads, torch.tensor([[1]]), torch.tensor([[2]]))
