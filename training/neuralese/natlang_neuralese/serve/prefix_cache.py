"""Exact reuse of prompt-prefix model state across requests (serve/THROUGHPUT_DECOMPOSITION.md, prefix reuse).

Natlang calls of one function share a long rendered prefix (system instructions, signature, tools) and differ in a
short tail. With a prefix cache the engine prefills a fresh prompt on a fixed grid of `block` tokens and keeps the
state after each block, keyed by the exact token prefix and the active adapter binding. A later prompt resumes from
its longest stored prefix that ends on the grid (or from its whole prompt), so the computation is the one a fresh
prefill on that grid performs: reuse changes nothing but the work done. `PortCache` states are immutable (a
reference is a snapshot; lfm2_port.PortCache), so stored states are shared, never copied.

The grid itself splits prefill differently from the default single chunk; it is therefore opt-in per server
(`--prefix-cache-gb`), and actors whose exact numerics are bound elsewhere keep the default.
"""
from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict


def _storages(value, out):
    """Every distinct tensor storage a cache state holds, as {data_ptr: nbytes}."""
    import torch
    if isinstance(value, torch.Tensor):
        storage = value.untyped_storage()
        out[(value.device.type, storage.data_ptr())] = storage.nbytes()
    elif isinstance(value, (tuple, list)):
        for item in value:
            _storages(item, out)
    elif hasattr(value, '__dataclass_fields__'):
        for name in value.__dataclass_fields__:
            _storages(getattr(value, name), out)
    elif hasattr(value, '__slots__'):
        for name in value.__slots__:
            _storages(getattr(value, name, None), out)
    return out


class PrefixCache:
    def __init__(self, budget_bytes: int, block: int):
        if budget_bytes <= 0 or block <= 0:
            raise ValueError('prefix cache needs a positive budget and block')
        self.budget, self.block = int(budget_bytes), int(block)
        self.entries: OrderedDict = OrderedDict()   # digest -> (cache, h_last, storages)
        self.refs: dict = {}                         # storage -> [nbytes, entries holding it]
        self.bytes = self.hits = self.misses = self.reused_tokens = 0
        self.lock = threading.Lock()

    def digests(self, key, ids) -> dict[int, str]:
        """Digest of the prefix ending at every grid boundary and at the end of `ids`."""
        out, digest, ends = {}, hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode()).digest(), []
        ends = list(range(self.block, len(ids), self.block)) + [len(ids)]
        start = 0
        for end in ends:
            digest = hashlib.sha256(digest + b'|' + ','.join(map(str, ids[start:end])).encode()).digest()
            out[end] = digest.hex()
            start = end
        return out

    def lookup(self, digests: dict[int, str]):
        """(end, cache, h_last) of the longest stored prefix, or None."""
        with self.lock:
            for end in sorted(digests, reverse=True):
                entry = self.entries.get(digests[end])
                if entry is not None:
                    self.entries.move_to_end(digests[end])
                    self.hits += 1
                    self.reused_tokens += end
                    return end, entry[0], entry[1]
            self.misses += 1
            return None

    def store(self, digest: str, cache, h_last):
        storages = _storages((cache, h_last), {})
        with self.lock:
            # A snapshot that cannot fit alone must not evict useful shorter
            # prefixes, especially the final grid block of a long prompt.
            if sum(storages.values()) > self.budget:
                return
            if digest in self.entries:
                self.entries.move_to_end(digest)
                return
            for ref, size in storages.items():
                held = self.refs.setdefault(ref, [size, 0])
                if held[1] == 0:
                    self.bytes += size
                held[1] += 1
            self.entries[digest] = (cache, h_last, storages)
            while self.bytes > self.budget and self.entries:
                _, (_, _, old) = self.entries.popitem(last=False)
                for ref in old:
                    held = self.refs[ref]
                    held[1] -= 1
                    if held[1] == 0:
                        self.bytes -= held[0]
                        del self.refs[ref]

    def metrics(self):
        with self.lock:
            return {'natlang_prefix_cache_bytes': self.bytes, 'natlang_prefix_cache_entries': len(self.entries),
                    'natlang_prefix_cache_hits': self.hits, 'natlang_prefix_cache_misses': self.misses,
                    'natlang_prefix_cache_reused_tokens': self.reused_tokens}
