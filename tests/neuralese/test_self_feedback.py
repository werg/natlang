from types import SimpleNamespace

import pytest
import torch

from natlang_neuralese.eval import self_feedback


class TinyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_weight = torch.nn.Parameter(torch.eye(3), requires_grad=False)
        self.controls = SimpleNamespace(close_id=2)

    def embed(self, ids):
        return torch.nn.functional.embedding(ids, self.embedding_weight)

    def forward_embeds(self, embeds, logits=False):
        return {'h_final': embeds, 'logits': None}

    def logits(self, states):
        return 5.0 * states @ self.embedding_weight.T


class TinyHeads:
    read_markers = False

    def __init__(self, *, perturb=False):
        self.perturb = perturb
        self.read_calls = 0

    def read_embeddings(self, backbone, payload):
        self.read_calls += 1
        return payload + (torch.tensor([0.0, 0.0, 2.0]) if self.perturb else 0.0)


def _fixed_rollouts(monkeypatch, projection=(1, 1, 1), crisp=(0, 0, 0)):
    def payloads(backbone, heads, prefix, steps, kinds, *, return_generated_tokens=False):
        assert tuple(kinds) == ('ar_greedy', 'ar_projection')
        assert return_generated_tokens is True
        projection_ids = torch.tensor([projection[:steps]], device=prefix.device)
        crisp_ids = torch.tensor([crisp[:steps]], device=prefix.device)
        result = {
            'ar_projection': backbone.embed(projection_ids),
            'ar_greedy': backbone.embed(crisp_ids),
        }
        generated = {'ar_projection': projection_ids, 'ar_greedy': crisp_ids}
        return result, generated
    monkeypatch.setattr(self_feedback, 'autoregressive_payloads', payloads)


def test_exact_projection_feedback_matches_plain_history_but_quality_gap_is_separate(monkeypatch):
    _fixed_rollouts(monkeypatch)
    backbone = TinyBackbone()
    heads = TinyHeads()
    result = self_feedback.self_feedback_window_metrics(
        backbone, heads, torch.tensor([[0]]), steps=3,
        logit_chunk_tokens=1, readout_chunk_tokens=1)
    assert result['kl_plain_to_projected_nats'] == pytest.approx(0, abs=1e-7)
    assert result['argmax_agreement'] == 1
    assert result['quality_ce_gap'] > .05
    assert heads.read_calls == 1
    assert result['max_live_logit_positions'] == 1
    assert result['projection_first_close_index'] is None
    assert result['stopping_qualified'] is False


def test_perturbed_projection_feedback_fails_channel_fidelity(monkeypatch):
    _fixed_rollouts(monkeypatch)
    result = self_feedback.self_feedback_window_metrics(
        TinyBackbone(), TinyHeads(perturb=True), torch.tensor([[0]]), steps=3,
        logit_chunk_tokens=2)
    assert result['argmax_agreement'] < 1
    assert result['kl_plain_to_projected_nats'] > .02


def test_channel_fidelity_does_not_depend_on_an_alternate_gold_continuation(monkeypatch):
    _fixed_rollouts(monkeypatch, projection=(0, 0, 0), crisp=(0, 0, 0))
    backbone = TinyBackbone()
    heads = TinyHeads()
    # The diagnostic compares each projected rollout against its own ordinary
    # token replay; no gold continuation is accepted as an input to this API.
    result = self_feedback.self_feedback_window_metrics(
        backbone, heads, torch.tensor([[0]]), steps=3)
    assert result['kl_plain_to_projected_nats'] == pytest.approx(0, abs=1e-7)
    assert result['argmax_agreement'] == 1


def test_fixed_span_reports_close_without_claiming_early_stopping(monkeypatch):
    _fixed_rollouts(monkeypatch, projection=(1, 2, 1), crisp=(0, 2, 0))
    result = self_feedback.self_feedback_window_metrics(
        TinyBackbone(), TinyHeads(), torch.tensor([[0]]), steps=3)
    assert result['projection_first_close_index'] == 1
    assert result['crisp_first_close_index'] == 1
    assert result['fixed_span_includes_post_close_positions'] is True
    assert result['stopping_qualified'] is False


def test_chunked_distribution_comparison_matches_single_block_exactly():
    backbone = TinyBackbone()
    plain = torch.tensor([[[1., 0., 0.], [0., 1., 0.], [1., 0., 0.], [0., 0., 1.]]])
    projected = plain.clone()
    projected[:, 1] = torch.tensor([.2, .7, .1])
    whole_kl, whole_agree = self_feedback._distribution_comparison(
        backbone, plain, projected, logit_chunk_tokens=4)
    chunk_kl, chunk_agree = self_feedback._distribution_comparison(
        backbone, plain, projected, logit_chunk_tokens=1)
    assert torch.allclose(whole_kl, chunk_kl, atol=1e-7)
    assert torch.equal(whole_agree, chunk_agree)


def test_stratum_aggregation_is_token_weighted_and_applies_proposed_thresholds():
    thresholds = {'min_argmax_agreement': .99, 'max_kl_nats': .02,
                  'max_quality_ce_gap_nats': .05}
    rows = [
        {'tokens': 1, 'argmax_agreement': 0.0, 'kl_plain_to_projected_nats': .2,
         'projection_generated_plain_history_ce': 1.0,
         'crisp_generated_plain_history_ce': .1},
        {'tokens': 9, 'argmax_agreement': 1.0, 'kl_plain_to_projected_nats': 0.0,
         'projection_generated_plain_history_ce': .1,
         'crisp_generated_plain_history_ce': .1},
    ]
    aggregate = self_feedback._aggregate(rows, thresholds=thresholds)
    assert aggregate['tokens'] == 10
    assert aggregate['argmax_agreement'] == pytest.approx(.9)
    assert aggregate['kl_plain_to_projected_nats'] == pytest.approx(.02)
    assert aggregate['quality_ce_gap'] == pytest.approx(.09)
    assert aggregate['passed'] is False


@pytest.mark.parametrize('thresholds', [
    {},
    {'min_argmax_agreement': .9, 'max_kl_nats': .02},
    {'min_argmax_agreement': float('nan'), 'max_kl_nats': .02,
     'max_quality_ce_gap_nats': .05},
    {'min_argmax_agreement': 1.1, 'max_kl_nats': .02,
     'max_quality_ce_gap_nats': .05},
    {'min_argmax_agreement': .9, 'max_kl_nats': -.1,
     'max_quality_ce_gap_nats': .05},
])
def test_threshold_api_requires_complete_finite_in_range_values(thresholds):
    with pytest.raises(ValueError):
        self_feedback._validate_thresholds(thresholds)


def test_evaluator_uses_declared_strata_not_extra_per_window_gate(monkeypatch):
    def fake_metrics(backbone, heads, prefix, *, steps, logit_chunk_tokens, readout_chunk_tokens):
        # First window of the small document fails; the aggregate first stratum
        # still passes because the other selected document contributes more tokens.
        if int(prefix[0, 0]) == 1:
            return {'tokens': 1, 'kl_plain_to_projected_nats': .2,
                    'argmax_agreement': 0., 'projection_generated_plain_history_ce': .2,
                    'crisp_generated_plain_history_ce': .1, 'quality_ce_gap': .1}
        return {'tokens': 100, 'kl_plain_to_projected_nats': 0.,
                'argmax_agreement': 1., 'projection_generated_plain_history_ce': .1,
                'crisp_generated_plain_history_ce': .1, 'quality_ce_gap': 0.}
    monkeypatch.setattr(self_feedback, 'self_feedback_window_metrics', fake_metrics)
    windows = [
        {'ids': [1, 2, 0], 'prefix': 1, 'document': 'a', 'groups': ['g'], 'offset': 0, 'strata': ['first']},
        {'ids': [2, 2, 0], 'prefix': 1, 'document': 'b', 'groups': ['g'], 'offset': 0, 'strata': ['first']},
        {'ids': [2, 2, 0], 'prefix': 1, 'document': 'c', 'groups': ['g'], 'offset': 9, 'strata': ['last']},
    ]
    result = self_feedback.evaluate_windows(TinyBackbone(), TinyHeads(), windows)
    first = result['strata']['first']
    assert first['passed'] is True
    assert result['windows'][0]['proposed_gate_passed'] is False
    assert result['proposed_gate_passed'] is True


def test_actual_requested_length_strata_are_reported(monkeypatch):
    def fake_metrics(backbone, heads, prefix, *, steps, logit_chunk_tokens, readout_chunk_tokens):
        return {'tokens': steps, 'kl_plain_to_projected_nats': 0., 'argmax_agreement': 1.,
                'projection_generated_plain_history_ce': .1,
                'crisp_generated_plain_history_ce': .1, 'quality_ce_gap': 0.}
    monkeypatch.setattr(self_feedback, 'self_feedback_window_metrics', fake_metrics)
    windows = [
        {'ids': [0] + [0] * 16, 'prefix': 1, 'document': 'short', 'groups': ['g'],
         'offset': 0, 'strata': ['first', 'last']},
        {'ids': [0] + [0] * 100, 'prefix': 1, 'document': 'medium', 'groups': ['g'],
         'offset': 0, 'strata': ['first', 'last']},
        {'ids': [0] + [0] * 129, 'prefix': 1, 'document': 'long', 'groups': ['g'],
         'offset': 0, 'strata': ['first', 'last']},
    ]
    result = self_feedback.evaluate_windows(TinyBackbone(), TinyHeads(), windows, steps=256)
    assert [row['length_band'] for row in result['windows']] == ['short', 'medium', 'long']
    assert [row['actual_steps'] for row in result['windows']] == [16, 100, 129]
    assert result['strata']['length_band:short']['windows'] == 1
    assert result['strata']['length_band:medium']['windows'] == 1
    assert result['strata']['length_band:long']['windows'] == 1


def test_requested_steps_are_clamped_to_window_target_span(monkeypatch):
    received = []
    def fake_metrics(backbone, heads, prefix, *, steps, logit_chunk_tokens, readout_chunk_tokens):
        received.append(steps)
        return {'tokens': steps, 'kl_plain_to_projected_nats': 0., 'argmax_agreement': 1.,
                'projection_generated_plain_history_ce': .1,
                'crisp_generated_plain_history_ce': .1, 'quality_ce_gap': 0.}
    monkeypatch.setattr(self_feedback, 'self_feedback_window_metrics', fake_metrics)
    windows = [{'ids': list(range(7)), 'prefix': 3, 'document': 'a', 'groups': ['g'], 'offset': 0,
                'strata': ['first', 'last']}]
    result = self_feedback.evaluate_windows(TinyBackbone(), TinyHeads(), windows, steps=256)
    assert received == [4]
    assert result['windows'][0]['requested_steps'] == 256
    assert result['windows'][0]['actual_steps'] == 4


@pytest.mark.parametrize('mask_system_prompt', [True, False])
def test_held_windows_use_shared_trainer_builder_and_system_mask(mask_system_prompt):
    class Tokenizer:
        unk_token_id = 999
        ids = {'<|im_start|>': 10, 'system': 11, 'user': 12,
               'assistant': 13, 'tool': 14}

        def convert_tokens_to_ids(self, token):
            return self.ids.get(token, self.unk_token_id)

    engine = SimpleNamespace(
        tokenizer=Tokenizer(),
        backbone=SimpleNamespace(controls=SimpleNamespace(open_id=1, close_id=2)),
        _tokens=lambda text: [],
    )
    row = {'split': 'test', 'text': 'document', 'source_groups': ['group-a'],
           'token_ids': [10, 11, 20, 21, 22, 10, 12, 23, 24, 25, 26, 27],
           'supervised_suffix_start': 5}
    windows, selection, masking = self_feedback.prepare_held_test_windows(
        engine, [row], tokens=10, prefix_tokens=4, target_tokens=3,
        held_documents=1, mask_system_prompt=mask_system_prompt)
    assert selection['selected_documents'][0]['window_count'] >= 1
    assert masking['enabled'] is mask_system_prompt
    assert masking['requested'] is mask_system_prompt
    assert masking['role_start_id'] == 10
    assert (masking['masked_system_tokens'] > 0) is mask_system_prompt
    assert all('roles' in window for window in windows)
    assert all('supervised_suffix_start' in window for window in windows)
    assert any('first' in window['strata'] for window in windows)
    assert any('last' in window['strata'] for window in windows)
