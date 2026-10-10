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


class DocumentWindowSampler:
    """Draw cohort, document, then window, using the caller's checkpointed RNG.

Batch companions keep the first draw's cohort and exact shape. They are also
document-uniform within that shape, rather than window-count weighted.
"""
    def __init__(self, windows, cohort_weights=None):
        self.documents = {}
        self.by_shape = {}
        for window in windows:
            cohort = window.get('cohort', 'native')
            if not isinstance(cohort, str) or not cohort:
                raise ValueError('text cohort names must be nonempty strings')
            document = window['document']
            self.documents.setdefault(cohort, {}).setdefault(document, []).append(window)
            shape = (cohort, window['prefix'], len(window['ids']))
            self.by_shape.setdefault(shape, {}).setdefault(document, []).append(window)
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
        self.by_shape = {k: list(d.values()) for k, d in self.by_shape.items()}

    @staticmethod
    def _window(documents, rng):
        return rng.choice(rng.choice(documents))

    def batch(self, size, rng):
        if size < 1:
            raise ValueError('batch size must be positive')
        cohort = rng.choices(self.cohorts, weights=self.weights, k=1)[0]
        first = self._window(self.documents[cohort], rng)
        companions = self.by_shape[(cohort, first['prefix'], len(first['ids']))]
        return [first] + [self._window(companions, rng) for _ in range(size - 1)]

    def receipt(self):
        return {'policy': 'cohort-then-document-then-window/1',
                'cohort_weights': dict(zip(self.cohorts, self.weights)),
                'documents': {c: len(d) for c, d in self.documents.items()},
                'batch_companions': 'same cohort and shape; uniform document then window',
                'admission_granted': False}
