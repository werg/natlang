"""Explicit write bounds, independent of unused constant-stop position rows."""
import math


def set_write_capacity(heads, capacity):
    if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < 1:
        raise ValueError('write capacity must be a positive integer')
    if heads.stop.use_position and capacity >= heads.stop.position.weight.shape[0]:
        raise ValueError('position-dependent stop head lacks learned rows for this write capacity')
    # Constant-position stops only ever read row zero. Changing the bound needs
    # no new weights or optimizer moments, nor a parameter schema migration.
    heads.max_length = capacity
    heads.stop.max_length = capacity


def checkpoint_write_capacity(heads_state, metadata, requested=None):
    if metadata.get('profile') == 'latent-sketch-v2':
        if 'stop.position.weight' in heads_state:
            raise ValueError('autoregressive close-token stop cannot have position rows')
        table_capacity = None
        capacity = metadata.get('max_length')
    else:
        table_capacity = int(heads_state['stop.position.weight'].shape[0]) - 1
        capacity = metadata.get('max_length', table_capacity)
    if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < 1:
        raise ValueError('invalid checkpoint write capacity')
    if table_capacity is not None and metadata.get('stop_position', True) and capacity > table_capacity:
        raise ValueError('checkpoint write capacity exceeds position-dependent stop rows')
    if requested is not None and requested != capacity:
        raise ValueError(f'max_block must match checkpoint length {capacity}')
    return capacity, table_capacity


def source_vector_length(token_count, tokens_per_vector, capacity):
    if token_count < 0 or not math.isfinite(tokens_per_vector) or tokens_per_vector <= 0 or capacity < 1:
        raise ValueError('invalid supervised write sizing controls')
    wanted = max(1, math.ceil(token_count / tokens_per_vector))
    if wanted > capacity:
        raise ValueError(f'supervised write needs {wanted} vectors, exceeds capacity {capacity}; '
                         'declare a sufficient --max-write-vectors instead of clipping the gold value')
    return wanted
