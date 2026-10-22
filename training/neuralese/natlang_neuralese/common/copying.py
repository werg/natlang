"""Independent metadata snapshots with bulk paths for large atomic containers."""
from __future__ import annotations

import copy


_ATOMIC = frozenset((type(None), bool, int, float, complex, str, bytes))


def _clone_metadata(item, memo):
    kind = type(item)
    if kind in _ATOMIC:
        return item
    address = id(item)
    if address in memo:
        return memo[address]
    if kind is dict:
        if all(type(key) in _ATOMIC and type(child) in _ATOMIC
               for key, child in item.items()):
            result = item.copy()
        else:
            result = {}
            memo[address] = result
            for key, child in item.items():
                result[_clone_metadata(key, memo)] = _clone_metadata(child, memo)
    elif kind is list:
        if all(type(child) in _ATOMIC for child in item):
            result = item.copy()
        else:
            result = []
            memo[address] = result
            result.extend(_clone_metadata(child, memo) for child in item)
    else:
        return copy.deepcopy(item, memo)
    memo[address] = result
    # Match deepcopy's keep-alive convention, including custom objects that
    # use the same memo through their standard deepcopy/reduction protocol.
    memo.setdefault(id(memo), []).append(item)
    return result


def deepcopy_metadata(value):
    """Preserve deepcopy semantics while copying common row tables in bulk.

    Only exact builtin dictionaries/lists get fast paths. Subclasses and other
    objects retain Python's custom deepcopy/reduction behavior. Mutable results
    never borrow source containers; immutable atoms retain their original identity.
    The memo has no closure cycle, so setup-only source metadata releases promptly.
    """
    return _clone_metadata(value, {})
