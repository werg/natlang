"""Contracts for semantic-span greedy exposure and honest target alignment.

Rows must carry explicit annotation and source provenance. Difficulty is never
inferred from length or model confidence. A generated-history state may be
supervised against the gold token at the same source offset, but it is not
reported as having the crisp teacher-forced continuation history.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

import torch


EXPOSURE_DOCUMENT_SCHEMA = 'natlang.semantic-exposure-document/1'
DIFFICULTIES = frozenset({'easy', 'medium', 'hard'})
_SHA256 = re.compile(r'^[0-9a-f]{64}$')


@dataclass(frozen=True)
class ExposureWindow:
    prefix_ids: tuple[int, ...]
    target_ids: tuple[int, ...]
    target_body_start: int
    target_body_end: int
    source_body_start: int
    source_body_end: int
    real_document_start_in_prefix: bool
    real_document_close_target_position: int | None
    desired_target_tokens: int
    max_capacity_tokens: int
    hard_cap_truncated: bool
    split: str
    source_groups: tuple[str, ...]
    semantic_category: str
    difficulty: str
    annotation_id: str

    @property
    def has_real_close(self):
        return self.real_document_close_target_position is not None


def validate_exposure_document(row):
    """Validate an annotated tokenized document without deriving annotations."""
    if not isinstance(row, dict) or row.get('schema') != EXPOSURE_DOCUMENT_SCHEMA:
        raise ValueError('semantic exposure document schema is missing or unsupported')
    if row.get('split') not in ('train', 'test'):
        raise ValueError('exposure split must be train or test')
    groups = row.get('source_groups')
    if (not isinstance(groups, list) or not groups or
            any(not isinstance(group, str) or not group.strip() for group in groups) or
            len(groups) != len(set(groups))):
        raise ValueError('exposure source_groups must be unique nonempty strings')
    provenance = row.get('source_provenance')
    required_provenance = ('corpus_id', 'corpus_manifest_sha256',
                           'source_record_sha256', 'source_locator')
    if not isinstance(provenance, dict) or any(not provenance.get(key) for key in required_provenance):
        raise ValueError('exposure source provenance is incomplete')
    for key in ('corpus_manifest_sha256', 'source_record_sha256'):
        if not isinstance(provenance[key], str) or not _SHA256.fullmatch(provenance[key]):
            raise ValueError('exposure provenance hashes must be lowercase SHA-256')
    if not isinstance(row.get('tokenizer_sha256'), str) or not _SHA256.fullmatch(row['tokenizer_sha256']):
        raise ValueError('exposure tokenizer fingerprint must be lowercase SHA-256')
    token_ids = row.get('token_ids')
    if (not isinstance(token_ids, list) or not token_ids or
            any(type(token) is not int or token < 0 for token in token_ids)):
        raise ValueError('exposure token_ids must be a nonempty list of nonnegative integers')
    annotations = row.get('semantic_spans')
    if not isinstance(annotations, list) or not annotations:
        raise ValueError('exposure requires explicit semantic span annotations')
    for annotation in annotations:
        if not isinstance(annotation, dict):
            raise ValueError('semantic span annotation must be an object')
        start, end = annotation.get('start_token'), annotation.get('end_token')
        if (type(start) is not int or type(end) is not int or
                start < 0 or end <= start or end > len(token_ids)):
            raise ValueError('semantic span offsets must be a nonempty body-token interval')
        if annotation.get('difficulty') not in DIFFICULTIES:
            raise ValueError('semantic difficulty must be explicitly labeled easy, medium, or hard')
        for key in ('category', 'annotation_id', 'label_source'):
            if not isinstance(annotation.get(key), str) or not annotation[key].strip():
                raise ValueError('semantic category and annotation provenance are required')
    return row


def validate_exposure_splits(rows):
    """Reject factual source groups shared by train and test exposure documents."""
    seen = {'train': set(), 'test': set()}
    for row in rows:
        validate_exposure_document(row)
        seen[row['split']].update(row['source_groups'])
    overlap = seen['train'] & seen['test']
    if overlap:
        raise ValueError('semantic exposure source groups cross train/test: ' + ', '.join(sorted(overlap)))
    return {'train_groups': sorted(seen['train']), 'test_groups': sorted(seen['test']),
            'disjoint': True}


def build_exposure_window(row, annotation, *, open_id, close_id,
                          prefix_tokens, max_capacity_tokens):
    """Build one source-offset preserving prefix/span with no synthetic close.

    Offset 0 in the augmented sequence is the actual document open marker;
    offset ``len(token_ids)+1`` is the actual document close marker. Interior
    windows do not receive a replacement open marker. A natural close is
    included only when the annotated span reaches the actual document end.
    """
    validate_exposure_document(row)
    if annotation not in row['semantic_spans']:
        raise ValueError('semantic annotation must be an exact declared row member')
    if type(open_id) is not int or type(close_id) is not int or open_id < 0 or close_id < 0:
        raise ValueError('open and close IDs must be nonnegative integers')
    if type(prefix_tokens) is not int or prefix_tokens < 1:
        raise ValueError('positive prefix capacity required')
    if type(max_capacity_tokens) is not int or max_capacity_tokens < 1:
        raise ValueError('positive generated span capacity required')
    body = row['token_ids']
    start, end = annotation['start_token'], annotation['end_token']
    augmented = [open_id, *body, close_id]
    augmented_start = start + 1
    contains_document_close = end == len(body)
    augmented_end = end + 1 + int(contains_document_close)
    desired = augmented[augmented_start:augmented_end]
    target = desired[:max_capacity_tokens]
    hard_truncated = len(desired) > max_capacity_tokens
    prefix_start = max(0, augmented_start - prefix_tokens)
    prefix = augmented[prefix_start:augmented_start]
    real_close = None
    if contains_document_close and not hard_truncated:
        real_close = len(target)
    target_body_tokens = len(target) - int(real_close is not None)
    return ExposureWindow(
        prefix_ids=tuple(prefix), target_ids=tuple(target),
        target_body_start=start, target_body_end=start + target_body_tokens,
        source_body_start=start, source_body_end=end,
        real_document_start_in_prefix=(prefix_start == 0 and bool(prefix) and prefix[0] == open_id),
        real_document_close_target_position=real_close,
        desired_target_tokens=len(desired), max_capacity_tokens=max_capacity_tokens,
        hard_cap_truncated=hard_truncated, split=row['split'],
        source_groups=tuple(row['source_groups']), semantic_category=annotation['category'],
        difficulty=annotation['difficulty'], annotation_id=annotation['annotation_id'])


def generated_target_mask(generated_lengths, target_lengths, *, width, device=None):
    """Return a position mask for valid generated vectors aligned to gold offsets.

    This mask supports per-position gold CE/embedding losses. It does not imply
    that a later generated-history state has the crisp teacher-forced context.
    """
    generated_lengths = torch.as_tensor(generated_lengths, dtype=torch.long, device=device)
    target_lengths = torch.as_tensor(target_lengths, dtype=torch.long, device=generated_lengths.device)
    if generated_lengths.ndim != 1 or target_lengths.shape != generated_lengths.shape:
        raise ValueError('generated and target lengths must be aligned vectors')
    if type(width) is not int or width < 0:
        raise ValueError('an explicit nonnegative target width is required')
    positions = torch.arange(width, device=generated_lengths.device)[None]
    mask = (positions < generated_lengths[:, None]) & (positions < target_lengths[:, None])
    return mask


def align_generated_targets(generated_values, gold_targets, generated_lengths):
    """Return only same-offset generated/gold positions and their explicit mask."""
    if generated_values.ndim < 2 or gold_targets.ndim < 2:
        raise ValueError('generated values and gold targets need batch and time axes')
    if generated_values.shape[0] != gold_targets.shape[0]:
        raise ValueError('generated and gold target batches differ')
    lengths = torch.as_tensor(generated_lengths, dtype=torch.long, device=generated_values.device)
    if lengths.shape != (generated_values.shape[0],):
        raise ValueError('one generated length is required per batch row')
    width = min(generated_values.shape[1], gold_targets.shape[1])
    mask = generated_target_mask(lengths, torch.full_like(lengths, width),
                                 width=width, device=generated_values.device)
    mask = mask[:, :width]
    return generated_values[:, :width], gold_targets[:, :width], mask


def stop_supervision_masks(*, generated_lengths, max_capacity_tokens,
                           real_close_target_position=None, hard_cap_truncated=False,
                           device=None):
    """Build valid masked BCE targets with distinct close and capacity semantics.

    Positions are 1-based vector counts. A real close gets a positive target at
    its true source position and negatives before it. If the target reaches the
    hard cap without a real close, prior positions are negatives and the final
    hard-cap position is unlabelled. An interior semantic span with no close and
    no capacity truncation has no stop labels.
    """
    if type(max_capacity_tokens) is not int or max_capacity_tokens < 1:
        raise ValueError('positive hard capacity required')
    if type(hard_cap_truncated) is not bool:
        raise TypeError('hard-cap truncation must be boolean')
    if real_close_target_position is not None and (
            type(real_close_target_position) is not int or
            not 1 <= real_close_target_position <= max_capacity_tokens):
        raise ValueError('real close position must be within generated capacity')
    lengths = torch.as_tensor(generated_lengths, dtype=torch.long, device=device)
    if lengths.ndim != 1:
        raise ValueError('generated lengths must be a vector')
    width = max_capacity_tokens
    counts = torch.arange(1, width + 1, device=lengths.device)[None]
    valid = counts <= lengths[:, None]
    if real_close_target_position is not None:
        supervised = counts <= real_close_target_position
        labels = counts == real_close_target_position
    elif hard_cap_truncated:
        supervised = counts < max_capacity_tokens
        labels = torch.zeros_like(supervised)
    else:
        supervised = torch.zeros_like(valid)
        labels = torch.zeros_like(valid)
    mask = valid & supervised
    return labels & mask, mask
