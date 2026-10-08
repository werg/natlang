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
        self.forward_id_calls = 0

    def embed(self, ids):
        return self.embedding_weight[ids]

    def forward_ids(self, ids, **kwargs):
        self.forward_id_calls += 1
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
    assert len(heads.read_calls)==5
    for history in backbone.inputs:
        assert torch.equal(history, backbone.embed(torch.tensor([[1,2,3,4]])))
    assert report['producer_controls']['reference_equal_live_greedy']
    for name, scores in report.items():
        if name=='producer_controls': continue
        assert scores['whole']['tokens']==3
        assert scores['whole']['argmax_agreement_with_gold']==1
        assert scores['whole']['ce_delta_from_gold']==pytest.approx(0)
        assert scores['whole']['argmax_agreement_with_live_greedy']==1
        assert scores['whole']['ce_delta_from_live_greedy']==pytest.approx(0)


def test_wrong_full_projection_is_visible_without_harming_reference_control():
    report = projected_history_metrics(CausalCycle(), CycleHeads(shift=2),
                                      torch.tensor([[1,2]]), torch.tensor([[3,4,5]]))
    assert report['reference']['whole']['argmax_agreement_with_gold']==1
    assert report['full_projection']['whole']['argmax_agreement_with_gold']==pytest.approx(1/3)
    assert report['full_projection']['whole']['ce_delta_from_gold']>0
    assert report['full_projection']['whole']['argmax_agreement_with_live_greedy']==pytest.approx(1/3)
    assert report['full_projection']['whole']['ce_delta_from_live_greedy']>0


def test_marked_read_profile_cannot_silently_change_positions():
    heads = CycleHeads(); heads.read_markers=True
    with pytest.raises(ValueError, match='raw read profile'):
        projected_history_metrics(CausalCycle(), heads, torch.tensor([[1]]), torch.tensor([[2]]))


def test_matched_consumers_reuse_pass_zero_and_report_read_history_distance():
    backbone=CausalCycle();heads=CycleHeads(shift=2)
    prefix=torch.tensor([[1,2]]);span=torch.tensor([[3,4,5]])
    completion={'top':backbone.embed(span),'sketches':backbone.embed(span)}
    live_tokens=torch.tensor([[4,5,6]])
    report=projected_history_metrics(backbone,heads,prefix,span,completion=completion,
        live_tokens=live_tokens,consumers=('full_projection','live_greedy'),per_window=True)
    assert backbone.forward_id_calls==0, 'the held evaluator supplied the existing pass-zero producer'
    assert len(heads.read_calls)==2, 'only the matched full-projection and crisp consumers run'
    assert len(report['windows'])==1
    projected=report['windows'][0]['full_projection']
    assert projected['whole']['tokens']==3
    assert projected['whole']['history_positions']==2
    assert projected['whole']['argmax_agreement_with_live_greedy']<1
    assert isinstance(projected['whole']['ce_delta_from_live_greedy'],float)
    assert projected['whole']['read_history_mse_vs_live_greedy']>0
    assert projected['last256']==projected['whole']
    assert report['live_greedy']['whole']['read_history_mse_vs_live_greedy']==0
