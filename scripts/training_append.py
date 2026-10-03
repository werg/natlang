"""Pure helpers for source-safe, checkpoint-preserving corpus append transitions."""
from __future__ import annotations

import hashlib
import math
from typing import Sequence


def ids_digest(ids: Sequence[str]) -> str:
    import json
    return hashlib.sha256(json.dumps(list(ids), sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def append_train_order(base_train, candidate_rows, combined_rows, *,
                       base_order_sha256: str, candidate_order_sha256: str):
    """Return combined offset rows as exact old permutation followed by candidate order.

    The input rows are lightweight ``index_pairs`` records. The identity checks
    prevent a same-count replacement, reorder, duplicate ID, or row omission.
    """
    base_ids = [row["id"] for row in base_train]
    candidate_ids = [row["id"] for row in candidate_rows]
    if ids_digest(base_ids) != base_order_sha256:
        raise ValueError("reconstructed base training permutation differs from append receipt")
    if ids_digest(candidate_ids) != candidate_order_sha256:
        raise ValueError("candidate physical row order differs from append receipt")
    if len(set(base_ids)) != len(base_ids) or len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("append input contains duplicate row IDs")
    if set(base_ids).intersection(candidate_ids):
        raise ValueError("candidate row IDs overlap the old training rows")
    train_rows = [row for row in combined_rows if row.get("split") != "test"]
    by_id = {row["id"]: row for row in train_rows}
    if len(by_id) != len(train_rows):
        raise ValueError("combined corpus contains duplicate row IDs")
    expected = base_ids + candidate_ids
    if set(by_id) != set(expected):
        raise ValueError("combined train rows do not match base plus candidate IDs")
    return [by_id[row_id] for row_id in expected]


def cosine_extension_multiplier(step: int, *, start_step: int, end_step: int) -> float:
    """Continuous post-checkpoint cosine factor, exactly 1 at its anchor."""
    if start_step < 0 or end_step <= start_step:
        raise ValueError("invalid scheduler extension interval")
    if step <= start_step:
        return 1.0
    progress = min(1.0, (step - start_step) / (end_step - start_step))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def extended_target_examples(old_target: int, appended_rows: int) -> int:
    if old_target < 1 or appended_rows < 1:
        raise ValueError("append target and row count must be positive")
    return old_target + appended_rows
