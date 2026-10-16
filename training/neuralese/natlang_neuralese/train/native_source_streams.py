"""Independent monotone readers for the existing interleaved curriculum."""
from __future__ import annotations


def scheduling_rank(entry):
    return (not bool(entry.get('communication')), -entry.get('training_priority', 0))


def stream_pending_batches(source, pending_batches, *, priority_ids=None):
    """Keep exact batch order without rescanning sources at every rank switch.

    Each rank/visit is already monotone in the production scheduler. Separate
    its source cursor from other ranks; retain the existing immutable source,
    parser and row-binding checks rather than caching parsed graph tensors.
    """
    priority_ids = set(priority_ids or ())
    def stream_key(visit, rank, indices):
        return (visit, rank, bool(priority_ids) and any(source.entries[index]['id'] in priority_ids for index in indices))
    requests = {}
    for visit, indices, _ in pending_batches:
        if not indices:
            raise ValueError('native pending source batch is empty')
        rank = scheduling_rank(source.entries[indices[0]])
        if any(scheduling_rank(source.entries[index]) != rank for index in indices):
            raise ValueError('native pending source batch mixes curriculum ranks')
        requests.setdefault(stream_key(visit, rank, indices), []).extend(indices)
    if any(indices != sorted(set(indices)) for indices in requests.values()):
        raise ValueError('native pending source stream must be monotone and distinct')
    streams = {}
    try:
        for visit, indices, pass_end in pending_batches:
            key = stream_key(visit, scheduling_rank(source.entries[indices[0]]), indices)
            if key not in streams:
                streams[key] = source.iter_indices(requests[key])
            try:
                batch = [next(streams[key]) for _ in indices]
            except StopIteration as error:
                raise ValueError('native graph source ended before its validated row count') from error
            if [item.id for item in batch] != [source.entries[index]['id'] for index in indices]:
                raise ValueError('native graph stream differs from scheduled optimizer-credit order')
            yield visit, indices, pass_end, batch
        # Drain each reader once so full-source count/decompression/stat checks
        # run when its complete requested schedule finishes.
        for stream in streams.values():
            try:
                next(stream)
            except StopIteration:
                continue
            raise ValueError('native graph stream yielded beyond its requested schedule')
    finally:
        for stream in streams.values():
            stream.close()
