"""Malformed reply forms that no learner reinforces, shared by every learner that trains sampled replies (grpo, exp0_sft,
program tune's rl/native_replay). The runtime answers the same forms with a one-sentence error at serving time
(ts-host ``native/tool-markup.ts`` ``secondThinkClose``)."""
from __future__ import annotations

import numpy as np

THINK_CLOSE = '</think>'
# Where a reply's action starts after its reasoning: a tool call, or the end of the turn.
SECOND_CLOSE_STOPS = ('<tool_call>', '<|im_end|>', '<|endoftext|>')


def second_close_mask(sample, close_id, stop_ids):
    """Per target of one reply: 0 on every ``</think>`` after the reply's first one and on the tokens after it up to the
    action (the next of ``stop_ids``: ``<tool_call>``, the end of turn; exclusive), else 1; None when the reply closes
    at most once. A reply that closed its reasoning, wrote visible text and closed again is malformed output: its
    second close must not be reinforced (grpo-v4-run2 put 0.6-0.96 on a second ``</think>`` after short answers by step
    12; DECISIONS 2026-11-19). The first close, the visible text and the action keep their weight."""
    sample = np.asarray(sample)
    closes = np.flatnonzero(sample == close_id)
    if len(closes) < 2:
        return None
    mask = np.ones(len(sample), np.float32)
    stops = np.isin(sample, list(stop_ids))
    for k in closes[1:]:
        after = np.flatnonzero(stops[k + 1:])
        mask[k:k + 1 + (after[0] if len(after) else len(sample) - k - 1)] = 0
    return mask


def second_close_ids(tokenizer):
    """(``</think>`` id, ids of ``SECOND_CLOSE_STOPS`` the vocabulary has) of a Hugging Face tokenizer; None when it has
    no ``</think>`` (a backbone without reasoning blocks has no second close)."""
    vocab = tokenizer.get_vocab()
    if THINK_CLOSE not in vocab:
        return None
    return int(vocab[THINK_CLOSE]), frozenset(int(vocab[token]) for token in SECOND_CLOSE_STOPS if token in vocab)
