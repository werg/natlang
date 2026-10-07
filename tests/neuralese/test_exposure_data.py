import pytest
import torch

from natlang_neuralese.train.exposure_data import (
    EXPOSURE_DOCUMENT_SCHEMA,
    align_generated_targets,
    build_exposure_window,
    generated_target_mask,
    stop_supervision_masks,
    validate_exposure_document,
    validate_exposure_splits,
)


H = 'a' * 64


def row(*, groups=None, split='train'):
    return {
        'schema': EXPOSURE_DOCUMENT_SCHEMA,
        'split': split,
        'source_groups': groups or ['factual-source-1'],
        'source_provenance': {
            'corpus_id': 'reviewed-corpus',
            'corpus_manifest_sha256': H,
            'source_record_sha256': 'b' * 64,
            'source_locator': 'record/17',
        },
        'tokenizer_sha256': 'c' * 64,
        'token_ids': [10, 11, 12, 13, 14],
        'semantic_spans': [{
            'start_token': 1, 'end_token': 5,
            'category': 'causal-explanation', 'difficulty': 'hard',
            'annotation_id': 'ann-1', 'label_source': 'review-round-2',
        }, {
            'start_token': 0, 'end_token': 2,
            'category': 'definition', 'difficulty': 'easy',
            'annotation_id': 'ann-2', 'label_source': 'review-round-2',
        }],
    }


def test_contract_requires_explicit_provenance_and_difficulty():
    doc = row()
    assert validate_exposure_document(doc) is doc
    del doc['semantic_spans'][0]['difficulty']
    with pytest.raises(ValueError, match='explicitly labeled'):
        validate_exposure_document(doc)


def test_split_groups_cannot_leak():
    with pytest.raises(ValueError, match='cross train/test'):
        validate_exposure_splits([row(), row(split='test')])
    assert validate_exposure_splits([row(), row(groups=['independent'], split='test')])['disjoint']


def test_window_preserves_actual_start_close_and_truncation():
    doc = row()
    interior = doc['semantic_spans'][0]
    window = build_exposure_window(doc, interior, open_id=1, close_id=2,
                                   prefix_tokens=3, max_capacity_tokens=5)
    assert window.prefix_ids == (1, 10)
    assert window.real_document_start_in_prefix
    assert window.target_ids == (11, 12, 13, 14, 2)
    assert window.target_body_end == 5
    assert window.real_document_close_target_position == 5
    assert not window.hard_cap_truncated

    at_start = doc['semantic_spans'][1]
    first = build_exposure_window(doc, at_start, open_id=1, close_id=2,
                                  prefix_tokens=3, max_capacity_tokens=2)
    assert first.prefix_ids == (1,)
    assert first.real_document_start_in_prefix
    assert first.target_ids == (10, 11)
    assert first.target_body_end == 2
    assert first.real_document_close_target_position is None
    assert not first.hard_cap_truncated

    truncated = build_exposure_window(doc, interior, open_id=1, close_id=2,
                                      prefix_tokens=3, max_capacity_tokens=3)
    assert truncated.target_ids == (11, 12, 13)
    assert truncated.target_body_end == 4
    assert truncated.hard_cap_truncated
    assert truncated.real_document_close_target_position is None


def test_generated_alignment_masks_invalid_divergent_positions():
    generated = torch.tensor([[[1.], [2.], [3.]], [[4.], [5.], [6.]]])
    gold = torch.tensor([[11, 12, 13], [21, 22, 23]])
    aligned_gen, aligned_gold, mask = align_generated_targets(generated, gold, [2, 1])
    assert aligned_gen.shape == generated.shape
    assert aligned_gold.equal(gold)
    assert mask.tolist() == [[True, True, False], [True, False, False]]
    assert generated_target_mask([1, 2], [3, 1], width=3).tolist() == [
        [True, False, False], [True, False, False],
    ]


def test_stop_targets_distinguish_real_close_from_capacity_cut():
    labels, mask = stop_supervision_masks(generated_lengths=[5, 3],
        max_capacity_tokens=5, real_close_target_position=4)
    assert labels.tolist() == [[False, False, False, True, False],
                               [False, False, False, False, False]]
    assert mask.tolist() == [[True, True, True, True, False],
                             [True, True, True, False, False]]

    labels, mask = stop_supervision_masks(generated_lengths=[5],
        max_capacity_tokens=5, hard_cap_truncated=True)
    assert labels.tolist() == [[False] * 5]
    assert mask.tolist() == [[True, True, True, True, False]]

    labels, mask = stop_supervision_masks(generated_lengths=[3],
        max_capacity_tokens=5)
    assert not labels.any()
    assert not mask.any()
