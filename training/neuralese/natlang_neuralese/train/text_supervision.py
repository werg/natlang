"""Shared document sampling and whole-trajectory text supervision.

Corpus admission happens before this module: an explicit cohort name or weight
does not admit a source. Assistant reasoning has the same weight as replies.
"""
from __future__ import annotations

import math

ROLE_CODES = ('other', 'system', 'user', 'tool', 'assistant_reasoning', 'assistant_reply')
TEXT_POSITION_WEIGHT_POLICY = {
    'all_positions_fraction': .5,
    'observed_suffix_fraction': .5,
    'unannotated_or_no_suffix_window': 'role-weighted-all-positions',
    'feedback_weight': .25,
    'reasoning_weight': 1.,
    'qualification': 'unweighted full-history complete-window and last256 strata',
}


def balanced_position_weights(span, suffix_starts, *, roles=None,
                              context_weight=1., feedback_weight=.25):
    """Half context, half observed suffix; normalize each document separately.

Tool feedback is reduced only inside the context term. Neither assistant
reasoning nor the observed target suffix is reduced. Missing role metadata is
ordinary text, not an inferred tool observation.
"""
    import torch
    if len(suffix_starts) != span.shape[0]:
        raise ValueError('one suffix coordinate per document required')
    if not math.isfinite(context_weight) or context_weight <= 0:
        raise ValueError('context weight must be finite and positive')
    if not math.isfinite(feedback_weight) or not 0 <= feedback_weight <= 1:
        raise ValueError('feedback weight must be finite and between zero and one')
    importance = torch.ones_like(span, dtype=torch.float32)
    if roles is not None:
        if roles.shape != span.shape:
            raise ValueError('role labels must align with supervised positions')
        importance = torch.where(roles == ROLE_CODES.index('tool'), feedback_weight, importance)
    # A tool-only window with zero feedback weight has no useful supervision.
    # Keep a small positive declared weight instead of silently dropping it.
    if (importance.sum(1) <= 0).any():
        raise ValueError('a complete window has zero context supervision; use positive feedback weight')
    context = importance * span.shape[1] / importance.sum(1, keepdim=True)
    starts = torch.tensor([span.shape[1] if start is None else start for start in suffix_starts],
                          device=span.device)
    mask = torch.arange(span.shape[1], device=span.device)[None] >= starts[:, None]
    counts = mask.sum(1, keepdim=True)
    weighted = (.5 * context_weight * context + .5 * mask.float() * span.shape[1] / counts.clamp_min(1))
    weighted = torch.where(counts > 0, weighted, context)
    return weighted * span.shape[1] / weighted.sum(1, keepdim=True)


class CohortDocumentSampler:
    """Shared text/trajectory sampler, using the caller's checkpointed RNG.

    Keys explicitly identify the cohort and source document. Several windows or
    target records from one document do not multiply its sampling probability.
    """
    def __init__(self, items, cohort_weights=None, *, cohort_key, document_key):
        self.documents = {}
        self.cohort_key = cohort_key
        for item in items:
            cohort = cohort_key(item)
            if not isinstance(cohort, str) or not cohort:
                raise ValueError('text cohort names must be nonempty strings')
            document = document_key(item)
            if not isinstance(document, str) or not document:
                raise ValueError('document keys must be explicit nonempty strings')
            self.documents.setdefault(cohort, {}).setdefault(document, []).append(item)
        if not self.documents:
            raise ValueError('no training documents')
        self.cohorts = sorted(self.documents)
        if cohort_weights is None:
            if len(self.cohorts) != 1:
                raise ValueError('multiple text cohorts require explicit cohort_weights')
            cohort_weights = {self.cohorts[0]: 1.}
        if not isinstance(cohort_weights, dict) or set(cohort_weights) != set(self.cohorts):
            raise ValueError('cohort weights must name exactly the loaded training cohorts')
        if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
               for v in cohort_weights.values()) or not math.isclose(sum(cohort_weights.values()), 1., abs_tol=1e-8):
            raise ValueError('cohort fractions must be positive, finite, and sum to one')
        self.weights = [cohort_weights[c] for c in self.cohorts]
        self.documents = {c: list(d.values()) for c, d in self.documents.items()}

    @staticmethod
    def _window(documents, rng):
        return rng.choice(rng.choice(documents))

    def sample(self, rng):
        cohort = rng.choices(self.cohorts, weights=self.weights, k=1)[0]
        return self._window(self.documents[cohort], rng)

    def receipt(self):
        return {'policy': 'cohort-then-document-then-item/1',
                'cohort_weights': dict(zip(self.cohorts, self.weights)),
                'documents': {c: len(d) for c, d in self.documents.items()},
                'admission_granted': False}


class DocumentWindowSampler(CohortDocumentSampler):
    """Text batches with document-uniform companions conditional on shape.

    Batch companions keep the first draw's cohort and exact shape. They are
    document-uniform within that shape, rather than window-count weighted.
    """
    def __init__(self, windows, cohort_weights=None):
        super().__init__(windows, cohort_weights,
                         cohort_key=lambda w: w.get('cohort', 'native'),
                         document_key=lambda w: w['document'])
        shapes = {}
        for w in windows:
            shape = (self.cohort_key(w), w['prefix'], len(w['ids']))
            shapes.setdefault(shape, {}).setdefault(w['document'], []).append(w)
        self.by_shape = {k: list(d.values()) for k, d in shapes.items()}

    def batch(self, size, rng):
        if size < 1:
            raise ValueError('batch size must be positive')
        first = self.sample(rng)
        cohort = self.cohort_key(first)
        companions = self.by_shape[(cohort, first['prefix'], len(first['ids']))]
        return [first] + [self._window(companions, rng) for _ in range(size - 1)]

    def receipt(self):
        return {**super().receipt(), 'policy': 'cohort-then-document-then-window/1',
                'batch_companions': 'same cohort and shape; uniform document then window'}
