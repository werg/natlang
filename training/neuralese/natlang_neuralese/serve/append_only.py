"""Append-only prompts for multi-turn episodes (token-in, token-out).

A harness re-sends the whole conversation every call; the actor re-renders it and re-tokenizes the earlier assistant
replies. Byte-level BPE does not round-trip: the re-rendered text is identical but its token ids differ from the ids the
model sampled at the merge boundaries (w5 round 4: 25% of continuation calls diverge, typically at the first tokens of the
previous reply or inside its tool-call text). A diverging prompt is no longer an extension of the cached sequence, so the
radix cache (on a hybrid model it can only resume where a conv/SSM state was saved) re-prefills the previous reply and
everything after the nearest saved state.

`TokenHistory` keeps the token sequences this actor produced (prompt ids + the ids it generated). For a new prompt whose ids
start with a stored prompt, the stored continuation is spliced in token for token as far as it decodes to the same bytes,
then the new prompt's ids resume at the next shared byte boundary. The prompt text is unchanged byte for byte (checked);
only the tokenization of the model's own earlier output follows what it sampled. The capture records the ids actually
sent, so learners see exactly what the actor saw.
"""
from __future__ import annotations

import threading
import time
from array import array
from collections import OrderedDict

_MOD = (1 << 61) - 1
_BASE = 1_000_003


def _bytes_to_unicode():
    """GPT-2 byte-level BPE alphabet: printable byte -> unicode character."""
    bs = list(range(ord('!'), ord('~') + 1)) + list(range(ord('¡'), ord('¬') + 1)) + list(range(ord('®'), ord('ÿ') + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


def token_bytes_table(tokenizer):
    """id -> the bytes it decodes to (byte-level BPE; added/special tokens as their UTF-8 content), or None when the
    tokenizer is not byte-level (then nothing is spliced)."""
    decoder = {char: byte for byte, char in _bytes_to_unicode().items()}
    added = {}
    for token_id, token in (getattr(tokenizer, 'added_tokens_decoder', None) or {}).items():
        added[int(token_id)] = str(token).encode('utf-8')
    size = len(tokenizer)
    table = [None] * size
    pieces = tokenizer.convert_ids_to_tokens(list(range(size)))
    for token_id, piece in enumerate(pieces):
        if token_id in added:
            table[token_id] = added[token_id]
        elif piece is None:
            table[token_id] = b''
        else:
            try:
                table[token_id] = bytes(decoder[ch] for ch in piece)
            except KeyError:
                return None
    return table


def _prefix_hashes(ids):
    out = [0] * (len(ids) + 1)
    h = 0
    for i, token in enumerate(ids):
        h = (h * _BASE + int(token) + 1) % _MOD
        out[i + 1] = h
    return out


class TokenHistory:
    """Bounded LRU: (prompt length, prompt hash) -> the replies the actor generated after that prompt (the samples of
    one task group send the same first prompt, so a prompt keeps its last `max_replies` distinct replies and a splice
    takes the one whose bytes the new prompt continues furthest).

    Only the hash identifies the prompt: a collision can only change the tokenization of an identical text, because a
    splice is accepted only when its bytes equal the rendered prompt's bytes."""

    def __init__(self, tokenizer, max_entries=32768, max_replies=16):
        self.table = token_bytes_table(tokenizer)
        self.max_entries, self.max_replies = max_entries, max_replies
        # (length, hash) -> [(array('i') of generated ids, time it was stored, the ids actually sent for this prompt
        # when they differ from these (a spliced prompt, stored also under its re-tokenized key), else None)], newest last
        self.entries = OrderedDict()
        self.lengths = {}                 # prompt length -> entry count
        # A spliced prompt's key -> the key of the ids it was spliced from: the next call re-tokenizes the same text the
        # same way, so its prefix matches the re-tokenized ids, never the spliced ones (without this a splice applied
        # only to the first continuation of an episode).
        self.spliced_from = OrderedDict()
        self.lock = threading.Lock()
        self.stats = {'append_only_lookups': 0, 'append_only_spliced': 0, 'append_only_tokens_kept': 0,
                      'append_only_continuations': 0, 'append_only_gap_seconds': 0.}

    @property
    def enabled(self):
        return self.table is not None

    def _put(self, key, value):
        replies = self.entries.pop(key, None)
        if replies is None:
            replies = []
            self.lengths[key[0]] = self.lengths.get(key[0], 0) + 1
        replies = [reply for reply in replies if reply[0] != value[0]] + [value]
        self.entries[key] = replies[-self.max_replies:]

    def store(self, prompt, generated):
        if not self.enabled or not len(generated):
            return
        key = (len(prompt), _prefix_hashes(prompt)[-1])
        generated, now = array('i', (int(t) for t in generated)), time.monotonic()
        with self.lock:
            self._put(key, (generated, now, None))
            raw = self.spliced_from.pop(key, None)
            if raw is not None:
                self._put(raw, (generated, now, array('i', (int(t) for t in prompt))))
            while len(self.entries) > self.max_entries:
                (length, _), _ = self.entries.popitem(last=False)
                self.lengths[length] -= 1
                if not self.lengths[length]:
                    del self.lengths[length]

    def splice(self, ids):
        """`ids` with the longest stored continuation of one of its prefixes spliced in (same bytes); else `ids`."""
        if not self.enabled or not self.entries:
            return ids
        hashes = _prefix_hashes(ids)
        with self.lock:
            self.stats['append_only_lookups'] += 1
            for length in sorted((n for n in self.lengths if n < len(ids)), reverse=True):
                key = (length, hashes[length])
                if key in self.entries:
                    replies, start = list(self.entries[key]), length
                    self.entries.move_to_end(key)
                    break
            else:
                return ids
        new = ids[start:]
        old, stored, sent = max(replies, key=lambda reply: self._shared_bytes(reply[0], new)) if len(replies) > 1 else replies[0]
        with self.lock:
            # A continuation of a stored reply: the time between the reply and this request is the client's
            # (harness, tool execution, queueing) share of the episode.
            self.stats['append_only_continuations'] += 1
            self.stats['append_only_gap_seconds'] += time.monotonic() - stored
        # Skip the common token prefix, then align on bytes.
        j = 0
        while j < len(old) and j < len(new) and old[j] == new[j]:
            j += 1
        base = list(sent) if sent is not None else list(ids[:start])
        if j == len(old):
            return ids if sent is None else self._accept(ids, base + list(ids[start:]), j)   # already append-only
        table = self.table
        a_bytes, b_bytes = b''.join(table[t] for t in old[j:]), b''.join(table[t] for t in new[j:])
        common, limit = 0, min(len(a_bytes), len(b_bytes))
        while common < limit and a_bytes[common] == b_bytes[common]:
            common += 1
        # Token boundaries (byte offsets from position j) of both sequences inside the common bytes.
        b_bounds, offset = {0: 0}, 0
        for m, token in enumerate(new[j:], 1):
            offset += len(table[token])
            if offset > common:
                break
            b_bounds[offset] = m
        best_i = best_m = 0
        offset = 0
        for i, token in enumerate(old[j:], 1):
            offset += len(table[token])
            if offset > common:
                break
            if offset in b_bounds:
                best_i, best_m = i, b_bounds[offset]
        if best_i == 0:
            return ids if sent is None else self._accept(ids, base + list(ids[start:]), j)
        return self._accept(ids, base + list(new[:j]) + list(old[j:j + best_i]) + list(new[j + best_m:]), j + best_i)

    def _shared_bytes(self, old, new):
        """Length of the common byte prefix of two id sequences."""
        table = self.table
        j = 0
        while j < len(old) and j < len(new) and old[j] == new[j]:
            j += 1
        a, b = b''.join(table[t] for t in old[j:]), b''.join(table[t] for t in new[j:j + len(old)])
        common, limit = 0, min(len(a), len(b))
        while common < limit and a[common] == b[common]:
            common += 1
        return sum(len(table[t]) for t in old[:j]) + common

    def _accept(self, ids, result, kept):
        """`result` when it is the same text as `ids` (never change the text), recorded so its continuation is found."""
        table = self.table
        if result == list(ids) or b''.join(table[t] for t in result) != b''.join(table[t] for t in ids):
            return ids
        with self.lock:
            self.stats['append_only_spliced'] += 1
            self.stats['append_only_tokens_kept'] += kept
            self.spliced_from[(len(result), _prefix_hashes(result)[-1])] = (len(ids), _prefix_hashes(ids)[-1])
            while len(self.spliced_from) > self.max_entries:
                self.spliced_from.popitem(last=False)
        return result
