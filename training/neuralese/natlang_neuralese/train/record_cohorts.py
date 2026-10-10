"""Record cohorts for the trajectory trainer: per-cohort, per-document sampling and earlier-record coverage.

A record (task, whole earlier trajectory, one target turn) is one document. Records carry an optional ``cohort``
label (default ``native``) set by an explicit assembly; the label and the fractions admit nothing. Sampling reuses the
shared cohort/document sampler (train/text_supervision.CohortDocumentSampler) with the record as document, and
draws each record from a generator seeded by the run seed and the record cursor, so resume and step rollback replay
the same records without extra checkpoint state.
"""
from __future__ import annotations

import hashlib
import json
import random

DEFAULT_COHORT = 'native'


def record_cohort(record: dict) -> str:
    cohort = record.get('cohort', DEFAULT_COHORT)
    if not isinstance(cohort, str) or not cohort:
        raise ValueError('record cohort labels must be nonempty strings: ' + str(record.get('id')))
    return cohort


class RecordCohortSampler:
    """Cohort by the declared fraction, then a uniform document (record) of that cohort."""

    def __init__(self, records, cohort_weights, *, seed: int):
        from .text_supervision import CohortDocumentSampler

        self._sampler = CohortDocumentSampler(records, cohort_weights, cohort_key=record_cohort,
                                              document_key=lambda record: record['id'])
        self.seed = seed

    def record(self, cursor: int) -> dict:
        return self._sampler.sample(random.Random(f'record-cohorts:{self.seed}:{cursor}'))

    def receipt(self) -> dict:
        receipt = self._sampler.receipt()
        receipt.update({'document': 'one record (task, whole earlier trajectory, one target turn)',
                        'draw_rng': 'random.Random(f"record-cohorts:{seed}:{cursor}") per draw'})
        return receipt


def _digest(value) -> bytes:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).digest()


def _prefix_chain(messages):
    """Chained digests: element j identifies messages[:j] exactly."""
    chain, current = [hashlib.sha256(b'').digest()], hashlib.sha256(b'').digest()
    for message in messages:
        current = hashlib.sha256(current + _digest(message)).digest()
        chain.append(current)
    return chain


def trajectory_coverage(records) -> dict:
    """``{record id: covered_replies}``: the assistant replies of each record's prompt that an earlier record of the same
    set supervises, as its prompt or target.

    Record A covers record B's first ``len(A.messages) + 1`` messages when they are exactly A's messages followed by A's
    target (B continues A's trajectory). The longest such A counts; with none, nothing is covered and B's whole prompt
    is whole-trajectory supervision (instructions, replies and, at the feedback weight, tool results). This is the
    ``covered_replies`` argument of GradSession._context_weights.
    """
    covers = set()
    for record in records:
        if record.get('target'):
            covers.add(_prefix_chain(list(record['messages']) + [record['target']])[-1])
    coverage = {}
    for record in records:
        messages = record['messages']
        chain = _prefix_chain(messages)
        covered = 0
        for length in range(len(messages), 0, -1):
            if chain[length] in covers:
                covered = sum(1 for message in messages[:length] if message.get('role') == 'assistant')
                break
        coverage[record['id']] = covered
    return coverage
