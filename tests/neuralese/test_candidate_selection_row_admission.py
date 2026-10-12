"""Registry approval must bind each exact child action and preserve explicit row rejections."""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load_renderer():
    path = ROOT / 'scripts/neuralese_data/render_native_turn_corpus.py'
    spec = importlib.util.spec_from_file_location('candidate_selection_renderer_under_test', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_registry_approval_is_per_row_and_cannot_override_rejection():
    render = load_renderer()
    row = {
        'id': 'turn-1', 'item_id': 'item-1', 'split': 'train', 'source_groups': ['group-1'],
        'sampling_group': 'source-item:item-1', 'target': {'value': 'answer'},
        'messages': [{'role': 'user', 'content': 'question'}],
    }
    approved = {
        'id': 'turn-1', 'item_id': 'item-1', 'split': 'train', 'source_groups': ['group-1'],
        'target_sha256': render.sha(render.canonical(row['target']).encode()),
        'messages_sha256': render.sha(render.canonical(row['messages']).encode()),
        'decision': 'admit-exact-selected-native-action-SFT-only', 'training_admission': True,
    }
    pending = {
        **row,
        'decision': {'training_approved': True},
        'training_admission': {
            'kind': 'candidate-selection-pending-corpus-admission',
            'approved': False,
            'reason': 'source quality remains held; exact child target is materialized for review only; '
                      'corpus admission remains false',
        },
    }

    # The source-quality-pending marker is promoted only with the exact verifier/owner row.
    render.validate_candidate_selection_row(pending, approved)

    rejected = {**pending, 'training_admission': {'approved': False, 'reason': 'reviewer rejected target'}}
    with pytest.raises(ValueError, match='explicit training rejection'):
        render.validate_candidate_selection_row(rejected, approved)

    with pytest.raises(ValueError, match='exact native-action SFT approval'):
        render.validate_candidate_selection_row(pending, {
            **approved,
            'decision': 'admit-reviewed-candidate-selection', 'training_admission': True,
        })
    with pytest.raises(ValueError, match='exact native-action SFT approval'):
        render.validate_candidate_selection_row(pending, {
            **approved,
            'decision': 'admit-exact-selected-native-action-SFT-only', 'training_admission': False,
        })
    with pytest.raises(ValueError, match='exact native-action SFT approval'):
        render.validate_candidate_selection_row(pending, {})

    for field, value in (
        ('id', 'turn-2'),
        ('target', {'value': 'changed'}),
        ('messages', [{'role': 'user', 'content': 'changed'}]),
        ('split', 'test'),
        ('source_groups', ['other-group']),
    ):
        tampered = {**pending, field: value}
        with pytest.raises(ValueError, match='changed after packet verification'):
            render.validate_candidate_selection_row(tampered, approved)

    unsupported_pending = {
        **pending,
        'training_admission': {
            'kind': 'candidate-selection-pending-corpus-admission', 'approved': False,
            'reason': 'some other pending state',
        },
    }
    with pytest.raises(ValueError, match='explicit training rejection'):
        render.validate_candidate_selection_row(unsupported_pending, approved)


def test_correction_pin_must_be_absent_on_both_sides_or_match_exactly():
    render = load_renderer()
    assert render.candidate_selection_correction_ref({}, {}) is None
    assert render.candidate_selection_correction_ref(
        {'selection_review_pin_correction': None}, {'selection_review_pin_correction_sha256': None}) is None

    digest = 'a' * 64
    packet = {'selection_review_pin_correction': {'path': 'selection-review-pin-correction.json', 'sha256': digest}}
    assert render.candidate_selection_correction_ref(packet, {
        'selection_review_pin_correction_sha256': digest,
    }) == packet['selection_review_pin_correction']

    for bad_packet, bad_admission in (
        (packet, {}),
        ({}, {'selection_review_pin_correction_sha256': digest}),
        (packet, {'selection_review_pin_correction_sha256': 'b' * 64}),
        ({'selection_review_pin_correction': 'not-a-reference'}, {}),
    ):
        with pytest.raises(ValueError, match='correction pin mismatch|reference is malformed'):
            render.candidate_selection_correction_ref(bad_packet, bad_admission)


def test_owner_item_digest_is_checked_when_declared_and_optional_under_pinned_receipt():
    render = load_renderer()
    verified = 'a' * 64
    # Older /1 receipts omit this row field; packet verifier + registry pins bind it.
    assert render.candidate_selection_owner_item_sha({}, verified) == verified
    assert render.candidate_selection_owner_item_sha({'item_sha256': verified}, verified) == verified
    with pytest.raises(ValueError, match='owner item hash differs'):
        render.candidate_selection_owner_item_sha({'item_sha256': 'b' * 64}, verified)
    with pytest.raises(ValueError, match='owner item hash differs'):
        render.candidate_selection_owner_item_sha({'item_sha256': None}, verified)
