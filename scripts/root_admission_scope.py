"""Shared normalization for fields that encode zero world credit."""
from __future__ import annotations


def no_new_world_credit(value: object) -> bool:
    """Accept schema encodings of no credit without accepting truthy/nonzero values.

    Existing per-action receipts use both JSON ``false`` and integer ``0`` for
    the same zero-credit scope.  Python treats ``bool`` as an ``int`` subclass,
    so the exact type check deliberately keeps ``true`` out of the zero case.
    """
    return value is False or (type(value) is int and value == 0)
