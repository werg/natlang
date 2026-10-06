"""Bounded host-only token IDs for a fixed tokenizer, never tensor activations."""
from collections import OrderedDict
import threading


class TokenCache:
    def __init__(self, tokenizer, budget_bytes):
        if budget_bytes < 0:
            raise ValueError('negative token cache budget')
        self.tokenizer = tokenizer
        self.budget = budget_bytes
        self.bytes = self.hits = self.misses = 0
        self.entries = OrderedDict()
        self.lock = threading.Lock()

    def tokens(self, text, *, escaped=None):
        # None preserves the tokenizer's default; False and True are distinct.
        key = (text, escaped)
        with self.lock:
            if key in self.entries:
                self.hits += 1
                self.entries.move_to_end(key)
                return list(self.entries[key][0])
            self.misses += 1
        options = {} if escaped is None else {'split_special_tokens': escaped}
        ids = tuple(self.tokenizer(text, add_special_tokens=False, **options)['input_ids'])
        # Conservative Python-object estimate, including Unicode/key/LRU overhead.
        size = 512 + 4 * len(text) + 48 * len(ids)
        if size <= self.budget:
            with self.lock:
                if key not in self.entries:
                    while self.entries and self.bytes + size > self.budget:
                        _, (_, old_size) = self.entries.popitem(last=False)
                        self.bytes -= old_size
                    self.entries[key] = (ids, size)
                    self.bytes += size
        return list(ids)

    def stats(self):
        with self.lock:
            return {'hits': self.hits, 'misses': self.misses,
                    'estimated_bytes': self.bytes, 'budget_bytes': self.budget,
                    'entries': len(self.entries)}
