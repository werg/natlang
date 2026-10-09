import pytest
import torch

from natlang_neuralese.train.execution import causal_gold_prefix_mask, causal_prefix_metrics
from natlang_neuralese.train.text_warmup import _normalize_context_valid_strata


def test_causal_prefix_metrics_reports_context_valid_and_tail_intersection():
    gold = torch.tensor([[1, 2, 3, 4, 5],
                         [1, 2, 3, 4, 5],
                         [1, 2, 3, 4, 5]])
    predicted = torch.tensor([[9, 2, 3, 4, 5],
                              [1, 2, 9, 4, 5],
                              [1, 2, 3, 4, 5]])
    mask = causal_gold_prefix_mask(predicted, gold)
    losses = torch.arange(1, 16, dtype=torch.float32).reshape(3, 5)
    reference_losses = torch.zeros_like(losses)

    metrics = causal_prefix_metrics(
        predicted, gold, losses, mask,
        reference_prediction=gold, reference_token_losses=reference_losses,
        tail_tokens=2, include_windows=True, producer_predictions=predicted)

    assert metrics['context_valid_tokens'].item() == 9
    assert metrics['context_valid_fraction'].item() == pytest.approx(9 / 15)
    # The first mismatching decision remains valid; later gold targets do not.
    assert metrics['context_valid_ce'].item() == pytest.approx(87 / 9)
    assert metrics['context_valid_accuracy'].item() == pytest.approx(7 / 9)
    assert metrics['context_valid_text_argmax_agreement'].item() == pytest.approx(7 / 9)
    assert metrics['context_valid_ce_delta'].item() == pytest.approx(87 / 9)

    # The first two valid prefixes do not reach the right-aligned last two
    # targets, while the all-correct row contributes both.
    assert metrics['tail_target_tokens'].item() == 6
    assert metrics['tail_context_valid_tokens'].item() == 2
    assert metrics['tail_context_valid_fraction'].item() == pytest.approx(1 / 3)
    assert metrics['tail_context_valid_ce'].item() == pytest.approx(14.5)
    assert metrics['tail_context_valid_accuracy'].item() == 1
    assert [row['first_divergence_index'] for row in metrics['windows']] == [0, 2, None]
    assert [row['tail_context_valid_tokens'] for row in metrics['windows']] == [0, 0, 2]
    assert [row['context_valid_tokens'] for row in metrics['windows']] == [1, 3, 5]


def test_causal_prefix_metrics_handles_zero_coverage_short_windows_and_all_correct():
    gold = torch.tensor([[2, 3], [4, 5]])
    predicted = torch.tensor([[8, 3], [4, 5]])
    mask = torch.tensor([[False, False], [True, True]])
    losses = torch.tensor([[5.0, 7.0], [2.0, 4.0]])

    metrics = causal_prefix_metrics(
        predicted, gold, losses, mask,
        reference_prediction=gold, reference_token_losses=torch.zeros_like(losses),
        tail_tokens=256, include_windows=True, producer_predictions=predicted)

    assert metrics['context_valid_tokens'].item() == 2
    assert metrics['context_valid_fraction'].item() == pytest.approx(0.5)
    assert metrics['context_valid_ce'].item() == pytest.approx(3.0)
    assert metrics['tail_target_tokens'].item() == 4
    assert metrics['tail_context_valid_tokens'].item() == 2
    assert metrics['tail_context_valid_fraction'].item() == pytest.approx(0.5)
    assert metrics['windows'][0]['context_valid_tokens'] == 0
    assert metrics['windows'][0]['first_divergence_index'] == 0
    assert metrics['windows'][1]['first_divergence_index'] is None
    assert metrics['windows'][1]['context_valid_accuracy'] == 1


def test_causal_prefix_metrics_reports_actual_final_token_divergence():
    gold = torch.tensor([[1, 2, 3, 4]])
    predicted = torch.tensor([[1, 2, 3, 9]])
    mask = causal_gold_prefix_mask(predicted, gold)
    losses = torch.ones_like(gold, dtype=torch.float32)
    metrics = causal_prefix_metrics(
        predicted, gold, losses, mask,
        reference_prediction=gold, reference_token_losses=losses,
        producer_predictions=predicted, include_windows=True)

    assert metrics['context_valid_tokens'].item() == 4
    assert metrics['windows'][0]['first_divergence_index'] == 3
    # A gold-mapped/control pass has no producer rollout and must not invent AR divergence.
    control = causal_prefix_metrics(
        gold, gold, losses, torch.ones_like(gold, dtype=torch.bool),
        reference_prediction=gold, reference_token_losses=losses,
        include_windows=True)
    assert control['windows'][0]['first_divergence_index'] is None


def test_context_valid_regional_strata_normalize_for_long_windows():
    # Mirrors the evaluator's last256 region for a source window longer than 256.
    strata = {'pass-1-length-long-tail-last256': {
        'tokens': 256,
        'ce': 384.,
        'context_valid_gold_tokens': 17,
        'context_valid_gold_fraction': 17,
        'context_valid_last256_target_tokens': 256,
        'context_valid_last256_gold_tokens': 17,
        'context_valid_gold_ce_weighted_sum': 34.,
        'context_valid_gold_accuracy_weighted_sum': 17.,
        'context_valid_text_argmax_agreement_weighted_sum': 16.,
        'context_valid_ce_delta_weighted_sum': 3.4,
        'context_valid_last256_gold_ce_weighted_sum': 34.,
        'context_valid_last256_gold_accuracy_weighted_sum': 17.,
        'context_valid_last256_text_argmax_agreement_weighted_sum': 16.,
        'context_valid_last256_ce_delta_weighted_sum': 3.4,
    }}
    _normalize_context_valid_strata(strata)
    row = strata['pass-1-length-long-tail-last256']
    assert row['ce'] == pytest.approx(1.5)
    assert row['context_valid_gold_tokens'] == 17
    assert row['context_valid_gold_fraction'] == pytest.approx(17 / 256)
    assert row['context_valid_gold_ce'] == pytest.approx(2.)
    assert row['context_valid_last256_gold_fraction'] == pytest.approx(17 / 256)
    assert row['context_valid_last256_gold_ce'] == pytest.approx(2.)


@pytest.mark.parametrize('mask', [
    torch.tensor([[True, False, True]]),
    torch.tensor([[1, 1, 0]], dtype=torch.int64),
])
def test_causal_prefix_metrics_rejects_nonprefix_or_nonboolean_mask(mask):
    gold = torch.tensor([[1, 2, 3]])
    losses = torch.ones_like(gold, dtype=torch.float32)
    with pytest.raises(ValueError):
        causal_prefix_metrics(gold, gold, losses, mask,
                              reference_prediction=gold,
                              reference_token_losses=losses,
                              include_windows=True)
