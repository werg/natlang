"""Exact typed block and soft-state read reference counters."""
from __future__ import annotations

import json


def context_ref_count(value, block_id: str, expected_read_name: str | None = None) -> int:
    """Count a typed Neuralese marker or a read with its exact registered name."""
    if isinstance(value, dict):
        count = int((value.get("type") == "neuralese" and value.get("id") == block_id)
                    or (expected_read_name is not None and value.get("type") == "read"
                        and value.get("name") == expected_read_name))
        for key, child in value.items():
            if key == "arguments" and isinstance(child, str):
                try:
                    count += context_ref_count(json.loads(child), block_id, expected_read_name)
                except json.JSONDecodeError:
                    pass
            else:
                count += context_ref_count(child, block_id, expected_read_name)
        return count
    if isinstance(value, list):
        return sum(context_ref_count(child, block_id, expected_read_name) for child in value)
    return 0


def typed_neuralese_ref_count(value, block_id: str) -> int:
    """Count only explicit typed Neuralese message markers."""
    if isinstance(value, dict):
        count = int(value.get("type") == "neuralese" and value.get("id") == block_id)
        for key, child in value.items():
            if key == "arguments" and isinstance(child, str):
                try:
                    count += typed_neuralese_ref_count(json.loads(child), block_id)
                except json.JSONDecodeError:
                    pass
            else:
                count += typed_neuralese_ref_count(child, block_id)
        return count
    if isinstance(value, list):
        return sum(typed_neuralese_ref_count(child, block_id) for child in value)
    return 0
