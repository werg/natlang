"""Projection diagnostics consume next-token payloads at previous-token inputs."""
from types import SimpleNamespace

import pytest
import torch

from natlang_neuralese.train.execution import causal_gold_prefix_mask
from natlang_neuralese.eval.projected_history import (
    _first_divergence_details, _gold_reference_survival, _token_id_window_fingerprints,
    projected_history_metrics,
)


def test_shared_causal_prefix_mask_keeps_first_mismatch_for_each_batch_row():
    gold = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
    generated = torch.tensor([[9, 2, 3, 4], [1, 2, 9, 4], [1, 2, 3, 4]])
    assert causal_gold_prefix_mask(generated, gold).tolist() == [
        [True, False, False, False], [True, True, True, False], [True, True, True, True]]


@pytest.mark.parametrize('generated,gold', [
    (torch.empty(1, 0, dtype=torch.long), torch.empty(1, 0, dtype=torch.long)),
    (torch.ones(2, dtype=torch.long), torch.ones(2, dtype=torch.long)),
    (torch.ones(1, 2, dtype=torch.long), torch.ones(1, 3, dtype=torch.long)),
])
def test_shared_causal_prefix_mask_rejects_unaligned_or_empty_tokens(generated, gold):
    with pytest.raises(ValueError, match='nonempty aligned'):
        causal_gold_prefix_mask(generated, gold)


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


def test_greedy_gold_reference_metrics_stop_survival_at_first_divergence():
    gold=torch.tensor([[4,5,6,7]])
    predicted=torch.tensor([[4,9,6,7]])
    losses=torch.tensor([[.1,.2,8.,9.]])
    metrics=_gold_reference_survival(predicted,losses,gold)
    assert metrics['first_token_accuracy']==1
    assert metrics['first_token_ce']==pytest.approx(.1)
    assert metrics['first_divergence_index_by_window']==[1]
    # Target 1 is still scored on the exact prefix through target 0; later
    # gold targets follow a prefix that the rollout never generated.
    assert metrics['exact_prefix_survival_tokens']==2
    assert metrics['exact_prefix_survival_ce']==pytest.approx(.15)
    assert metrics['exact_prefix_survival_accuracy']==pytest.approx(.5)


def test_greedy_gold_reference_reports_full_survival_without_zero_case_ambiguity():
    gold=torch.tensor([[4,5]])
    metrics=_gold_reference_survival(gold,torch.tensor([[.2,.3]]),gold)
    assert metrics['first_divergence_index_by_window']==[None]
    assert metrics['exact_prefix_survival_tokens']==2
    assert metrics['exact_prefix_survival_ce']==pytest.approx(.25)
    assert metrics['exact_prefix_survival_accuracy']==1


def test_first_divergence_details_bind_tokens_logits_and_payload_without_extra_forward():
    class EmbeddingBackbone:
        def embed(self, ids):
            return torch.nn.functional.one_hot(ids, num_classes=4).float()

    span=torch.tensor([[1,2,3]])
    generated={'ar_projection':torch.tensor([[1,0,3]]),'ar_greedy':torch.tensor([[1,2,3]])}
    losses=torch.tensor([[.1,.8,.9]])
    survival=_gold_reference_survival(generated['ar_projection'],losses,span)
    assert survival['first_divergence_index_by_window']==[1]
    details=_first_divergence_details(
        EmbeddingBackbone(),'ar_projection',survival,generated,
        torch.tensor([[1,0,3]]),
        {'ar_projection':(torch.tensor([[0.,2.,0.]]),torch.tensor([[3.,4.,2.]]),torch.tensor([[1.,1.5,1.]]))},
        {'ar_projection':torch.tensor([[[0.,1.,0.,0.],[0.,0.,0.,2.],[0.,0.,1.,0.]]])},span,[2])
    row=details[0]
    assert row['target_index']==1
    assert (row['gold_token_id'],row['generated_token_id'],row['rescored_token_id'])==(2,0,0)
    assert row['ar_greedy_control_token_id']==2
    assert row['generated_logit_minus_gold_logit']==pytest.approx(2)
    assert row['top1_logit_margin']==pytest.approx(1.5)
    assert row['emitted_payload_l2_norm']==pytest.approx(2)
    assert row['gold_embedding_l2_norm']==pytest.approx(1)
    assert row['emitted_payload_minus_gold_embedding_l2_norm']==pytest.approx(5**.5)
    assert row['emitted_payload_gold_embedding_cosine']==pytest.approx(0)
    assert row['preceding_feedback_index']==0
    assert row['preceding_feedback_l2_norm']==pytest.approx(1)
    assert row['preceding_feedback_minus_gold_embedding_l2_norm']==pytest.approx(0)
    assert row['preceding_feedback_gold_embedding_cosine']==pytest.approx(1)
    assert row['ar_greedy_control_first_divergence_index']==2
    assert row['ar_greedy_control_gold_prefix_valid_at_target'] is True
    assert 'through prior targets' in row['ar_greedy_control_comparability_scope']


def test_autoregressive_input_fingerprints_are_stable_and_bind_target_ids():
    prefix=torch.tensor([[1,2],[3,4]])
    span=torch.tensor([[5,6,7],[8,9,10]])
    first=_token_id_window_fingerprints(prefix,span)
    assert _token_id_window_fingerprints(prefix.clone(),span.clone())==first
    assert first[0]['prefix_token_count']==2
    assert first[0]['target_token_count']==3
    changed_span=span.clone();changed_span[0,0]=11
    changed=_token_id_window_fingerprints(prefix,changed_span)
    assert changed[0]['prefix_token_ids_sha256']==first[0]['prefix_token_ids_sha256']
    assert changed[0]['target_token_ids_sha256']!=first[0]['target_token_ids_sha256']
    assert changed[1]==first[1]
