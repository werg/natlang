"""Cohort policy from COHORT_FAIR_SCHEDULING_DECOMPOSITION_20261017.md."""
from __future__ import annotations

from collections import deque
import heapq
import hashlib
import json


def source_cohort_ids(source):
    """Stable graph-file leaf identities outside authenticated row metadata."""
    def walk(current):
        if current.kind != 'manifest':
            return [current.file_sha256] * len(current)
        raw = current.path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != current.file_sha256:
            raise ValueError('cohort source changed after validation')
        manifest = json.loads(raw)
        declarations = manifest['sources']
        if len(declarations) != len(current.children):
            raise ValueError('cohort source declarations differ from validated children')
        result = []
        for binding, child in zip(declarations, current.children):
            if binding.get('sha256') != child.file_sha256:
                raise ValueError('cohort source declaration changed after validation')
            result.extend(walk(child))
        return result
    result = walk(source)
    if len(result) != len(source.entries):
        raise ValueError('cohort identities do not cover source rows')
    return result


def balance_urgent_batches(entries, batches, exposure_counts, cohort_ids, *, foundation_ids=()):
    """Choose least served pending cohort; saved row visits are the history."""
    queues = {}
    for batch in batches:
        ids = {cohort_ids[index] for index in batch[1]}
        if len(ids) != 1:
            raise ValueError('urgent batch mixes source cohorts')
        key = next(iter(ids))
        queues.setdefault(key, deque()).append(batch)
    service = dict.fromkeys(queues, 0)
    for index, entry in enumerate(entries):
        key = cohort_ids[index]
        if key in service and entry.get('training_priority', 0) > 0 and entry['id'] not in foundation_ids:
            credit = exposure_counts.get(entry['id'], 0)
            if type(credit) is not int or credit < 0:
                raise ValueError('cohort service requires nonnegative integer row credits')
            service[key] += credit * entry['forward_tokens']
    heap = []
    for ordinal, (key, queue) in enumerate(queues.items()):
        priority = entries[queue[0][1][0]].get('training_priority', 0)
        heapq.heappush(heap, (service[key], -priority, ordinal, key))
    result = []
    while heap:
        served, priority, ordinal, key = heapq.heappop(heap)
        batch = queues[key].popleft()
        result.append(batch)
        served += sum(entries[index]['forward_tokens'] for index in batch[1])
        if queues[key]:
            heapq.heappush(heap, (served, priority, ordinal, key))
    return result
