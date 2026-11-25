"""Exact memoized tokenization of chat-template text by added-token segments (``Engine._token_cache``).

A fast tokenizer first cuts its input at every added token (special or not) and then normalizes, pre-tokenizes and
encodes each piece between them on its own (``tokenizers`` AddedVocabulary). With no added token that strips
neighbouring whitespace, matches a single word only or matches normalized text, and no normalizer, the ids of a text are
therefore the concatenation of each piece's ids, and a piece's ids can be remembered. Every call of an episode renders
the whole history again (one call's prompt is all earlier messages plus the new ones), so without the memo the
tokenizer ran over the history once per call: 90% of re-rendering an episode for training rows (filled_reasoning.py
build, sft_sources build). Escaped runs (content holding special-token text, tokenized as plain text) are remembered
whole. ``NATLANG_SEGMENT_TOKEN_CACHE=0`` opts out.
"""
from __future__ import annotations

import os
import re
import threading
from array import array
from collections import OrderedDict


class SegmentTokenCache:
    """``tokens(text, escaped=False)``: the ids ``tokenizer(text, add_special_tokens=False,
    split_special_tokens=escaped)`` returns, from remembered pieces. ``budget_chars`` bounds the remembered text
    (least recently used pieces go first); ids are kept as int32 arrays (a tuple of ints took ~36 bytes per token)."""

    def __init__(self, tokenizer, budget_chars=32 << 20):
        self.tokenizer = tokenizer
        added = sorted({t.content for t in tokenizer.added_tokens_decoder.values()}, key=len, reverse=True)
        self.pattern = re.compile('(' + '|'.join(map(re.escape, added)) + ')') if added else None
        self.memo, self.chars, self.budget = OrderedDict(), 0, budget_chars
        self.hits = self.misses = 0
        self.lock = threading.Lock()     # renderers are shared by threads (filled_reasoning generate, grpo)

    @classmethod
    def for_tokenizer(cls, tokenizer, **kwargs):
        """A cache for ``tokenizer``, or None when its tokenization is not exact by pieces (a slow tokenizer, an added
        token with lstrip/rstrip/single_word/normalized, a normalizer, ``split_special_tokens`` on) or the opt-out is
        set."""
        if os.environ.get('NATLANG_SEGMENT_TOKEN_CACHE', '1') == '0' or not cls.exact(tokenizer):
            return None
        return cls(tokenizer, **kwargs)

    @staticmethod
    def exact(tokenizer):
        backend = getattr(tokenizer, 'backend_tokenizer', None)
        if backend is None or getattr(tokenizer, 'split_special_tokens', False) or backend.normalizer is not None:
            return False
        return all(not (t.lstrip or t.rstrip or t.single_word or t.normalized)
                   for t in tokenizer.added_tokens_decoder.values())

    def _encode(self, text, escaped):
        return array('i', self.tokenizer(text, add_special_tokens=False, split_special_tokens=escaped)['input_ids'])

    def _piece(self, text, escaped):
        key = (text, escaped)
        with self.lock:
            ids = self.memo.get(key)
            if ids is not None:
                self.memo.move_to_end(key)
                self.hits += 1
                return ids
            self.misses += 1
        ids = self._encode(text, escaped)
        with self.lock:
            if key not in self.memo:
                self.memo[key] = ids
                self.chars += len(text)
                while self.chars > self.budget and len(self.memo) > 1:
                    (old, _), _ids = self.memo.popitem(last=False)
                    self.chars -= len(old)
        return ids

    def tokens(self, text, escaped=False):
        if escaped or self.pattern is None:
            return self._piece(text, escaped).tolist()
        ids = []
        for piece in self.pattern.split(text):
            if piece:
                ids.extend(self._piece(piece, False))
        return ids
