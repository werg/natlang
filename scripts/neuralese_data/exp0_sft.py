#!/usr/bin/env python3
"""Experiment 0 (plans/neuralese/APPROACH_REVIEW_20261111.md section 4): a plain FP32-master AdamW SFT of a native
LFM3 checkpoint's backbone on per-call rows lowered from verified-successful student episodes
(`lower_rollout_turns.py`), written back as a native checkpoint the existing actor serves.

The native backbone state is a plain HF state dict, so the loop is the vendored HF LFM3 (`third_party/lfm3`) in FP32
with bf16 autocast, assistant (sampled output) tokens only carrying loss. The Neuralese interface (marker rows, token
override rows, producer adapter) is frozen and copied unchanged: the 6 sparse embedding rows are merged into the
embedding for the forward (as `export/native_hf.merged_state` does) and their gradient is masked.

A call's prompt extends the previous call's (prompt + sampled output) of the same invocation in most cases, so such a
chain is one sequence whose sampled segments are all supervised: equal to per-call rows (causal), 10x cheaper.

    exp0_sft.py parity --turns T.pkl --checkpoint C --base B           # HF forward vs the actor's recorded logprobs
    exp0_sft.py train  --turns T.pkl --checkpoint C --base B --out-dir D
    exp0_sft.py bench  --turns T.pkl --checkpoint C --base B           # stock vs fast fwd+bwd on fixed chains: loss, grads, tok/s
    exp0_sft.py train  --chains S.pkl [--val-chains V.pkl] ...          # prebuilt sequences (SFT v2: lower_sft_sources.py select)

A prebuilt chain's segments may carry per-token loss weights (`row['weight']`, e.g. the own-output objective's reasoning
share); the step loss is then the weighted mean over its targets. Each `--val-chains` set reports its weighted loss every
`--val-every` steps and at the end (the run's held-out training curve).

A chain with a ``decision`` (a decision readout row with its option set, `sft_sources build`) trains the readout's
option-normalized log score (`decision_backward`, one target unit of loss mass per decision);
`--no-decision-objective` trains it as the token CE of the chosen reply instead. A reasoned readout's decision (its
options scored after the student's own sampled reasoning, `reasoning_span`) also trains that reasoning as token CE, one
unit of loss mass per token (`--no-decision-reasoning`: the choice only, still conditioned on the reasoning).

Speed (GB10, 11k-token sequences): the stock HF modelling code runs at about 1.9k fwd+bwd tokens/s because half of the GPU
time is unfused elementwise work and a fifth dense sliding-window attention; `lfm3_fast_train` (compiled per-kind layers,
varlen flash attention with the sliding window, shifted-add conv, fused linear NLL) plus partial activation checkpointing
sized to `--activation-gb` runs about 3x faster with the same BF16 arithmetic (`--stock-model` opts out). A step's chains
are packed into rows up to the length the activation budget holds unrecomputed (`--pack-tokens`).
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'training/neuralese'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from natlang_neuralese.train import lfm3_fast as lfm3_fast_train
from natlang_neuralese.train.learner_state import (
    export_merged_checkpoint,
    load_learner_state,
    save_learner_state,
)
from natlang_neuralese.train.lfm3_learner import (
    DEFAULT_SFT_LR,
    AdapterSpec,
    Lfm3Learner,
    OOMRetryPlan,
    load_base_model,
    run_step_with_oom_retry,
)
# One implementation for every learner (grpo, exp0_sft, program tune).
from natlang_neuralese.train.reply_format import SECOND_CLOSE_STOPS, second_close_mask

MAX_LEN = 32000


def row_decisions(row):
    """A row's decisions: its joint ``decision``, or the field steps ``decisions`` of a field-mode readout
    (``sft_sources.render_decision``); [] for a plain row."""
    if row.get('decision') is not None:
        return [row['decision']]
    return list(row.get('decisions') or ())


def chains_of(episodes, max_len=MAX_LEN):
    """Sequences: [{'ids': int32[], 'segs': [(start, length, row)], 'imports': [(start, key)], 'sources': {key:
    producer document}, 'task', 'round'}]; calls of one invocation whose prompt extends the previous call's full text,
    with the same imported native state at the same positions (``row['imports']``), share a sequence. ``sources``: the
    pass-through producer documents of the imports (``lower_rollout_turns.write_sources``)."""
    chains = []
    for episode in episodes:
        by_invocation = collections.defaultdict(list)
        for row in episode['rows']:
            by_invocation[row['invocation']].append(row)
        for rows in by_invocation.values():
            rows.sort(key=lambda row: row['index'])
            current = None
            for row in rows:
                ids, sample = row['prompt'], row['sample']
                decisions = row_decisions(row)
                if decisions:
                    # a decision readout is one choice over its option set: a chain of its own (``decision_backward``);
                    # a field-mode readout is one choice per field step, each its own chain on the same prompt
                    if current is not None:
                        chains.append(current)
                    current = None
                    for step, decision in enumerate(decisions):
                        chains.append({'ids': np.concatenate([ids, sample]), 'segs': [(len(ids), len(sample), row)],
                                       'imports': [], 'sources': {}, 'task': episode['task_id'],
                                       'round': episode['round'], 'decision': decision,
                                       **({'decision_step': step} if step else {})})
                    continue
                if len(ids) + len(sample) > max_len:
                    if current is not None:
                        chains.append(current)
                    current = None
                    continue
                imports = [tuple(i) for i in row.get('imports') or ()]
                if current is not None:
                    full = current['ids']
                    if (len(ids) >= len(full) and np.array_equal(ids[:len(full)], full)
                            and [i for i in imports if i[0] < len(full)] == current['imports']):
                        current['ids'] = np.concatenate([ids, sample])
                        current['segs'].append((len(ids), len(sample), row))
                        current['imports'] = imports
                        current['sources'].update(row.get('sources') or {})
                        continue
                    chains.append(current)
                current = {'ids': np.concatenate([ids, sample]), 'segs': [(len(ids), len(sample), row)],
                           'imports': imports, 'sources': dict(row.get('sources') or {}),
                           'task': episode['task_id'], 'round': episode['round']}
            if current is not None:
                chains.append(current)
    return chains


def load_model(checkpoint, base, device='cuda', dtype=torch.bfloat16):
    """Load an HF LFM3 backbone from a native checkpoint via the shared learner core."""
    model, state, special, _markers, _overrides = load_base_model(checkpoint, base, device=device, dtype=dtype)
    return model, state, special


def hidden(model, ids, imports=None):
    """Final hidden states (T, D) through the stock HF modules. ``imports`` (``lfm3_fast_train.import_tensors``): at those
    rows each attention layer's normed pre-RoPE keys and values and each conv layer's B*x are the imported state (hooks
    on ``k_norm``/``v_proj`` and on ``in_proj``, where B = 1 and x = the state, as the SGLang fork splices it)."""
    if imports is None:
        return model.model(input_ids=ids[None], use_cache=False).last_hidden_state[0]
    if torch.is_grad_enabled() and getattr(model, 'is_gradient_checkpointing', False):
        raise ValueError('stock-path imports are hooks removed after the forward: no layer recompute (use lfm3_fast_train)')
    rows, handles = imports['rows'], []

    def put(value, last):
        def hook(module, args, output):
            output = output.clone()
            output[0, rows] = value(output).to(output.dtype) if callable(value) else value.reshape(len(rows), *last).to(output.dtype)
            return output
        return hook
    try:
        for layer, state in zip(model.model.layers, imports['layers']):
            if layer.is_attention_layer:
                k, v = state
                handles.append(layer.self_attn.k_norm.register_forward_hook(put(k, k.shape[1:])))
                handles.append(layer.self_attn.v_proj.register_forward_hook(put(v, (v[0].numel(),))))
            else:
                (bx,) = state
                width = bx.shape[-1]
                ones = torch.ones_like(bx)
                handles.append(layer.conv.in_proj.register_forward_hook(
                    put(lambda output, bx=bx, ones=ones: torch.cat([ones, output[0, rows, width:2 * width], bx], -1), None)))
        return model.model(input_ids=ids[None], use_cache=False).last_hidden_state[0]
    finally:
        for handle in handles:
            handle.remove()


# the activation budget's arithmetic (one implementation, shared with the prefix tree)
ACTIVATION_BYTES_PER_TOKEN_LAYER = lfm3_fast_train.ACTIVATION_BYTES_PER_TOKEN_LAYER
resident_tokens = lfm3_fast_train.resident_tokens
set_checkpointing = lfm3_fast_train.set_checkpointing


def _nll(weight, h, target, allowed=None):
    logits = F.linear(h, weight).float()
    if allowed is not None:
        logits = logits.masked_fill(~allowed, float('-inf'))
    return F.cross_entropy(logits, target, reduction='none')


def token_nll(model, h, positions, targets, chunk=2048, *, trainable_rows=None, mask=None):
    """NLL (fp32) of ``targets`` predicted from hidden rows ``positions`` (fused chunked linear NLL; ``mask``: the
    sampling-law masks of those rows, ``law_mask``)."""
    rows = getattr(model, '_lfm3_trainable_rows', None) if trainable_rows is None else trainable_rows
    if getattr(model, '_lfm3_fast', None) is not None:
        return lfm3_fast_train.linear_nll(h[positions], model.lm_head.weight, targets, trainable_rows=rows, mask=mask)
    out = []
    weight = model.lm_head.weight
    for start in range(0, len(positions), chunk):
        h_rows = h[positions[start:start + chunk]]
        target = targets[start:start + chunk]
        allowed = (mask.allowed_rows(start, start + len(target), weight.shape[0], h.device)
                   if mask is not None and len(mask) else None)
        if torch.is_grad_enabled():
            out.append(torch_checkpoint(_nll, weight, h_rows, target, allowed, use_reentrant=False))
        else:
            out.append(_nll(weight, h_rows, target, allowed))
    return torch.cat(out)


def token_nll_nucleus(model, h, positions, targets, top_p, *, trainable_rows=None, mask=None):
    """(nll, nll under each row's own top-p nucleus, target outside the nucleus) of ``targets`` at hidden rows
    ``positions`` (`lfm3_fast.linear_nll_nucleus`; prepared models only)."""
    if getattr(model, '_lfm3_fast', None) is None:
        raise ValueError('the nucleus NLL needs a lfm3_fast.prepare()d model')
    rows = getattr(model, '_lfm3_trainable_rows', None) if trainable_rows is None else trainable_rows
    return lfm3_fast_train.linear_nll_nucleus(h[positions], model.lm_head.weight, targets, top_p, trainable_rows=rows,
                                              mask=mask)


# The tokenizer a row's tool grammar is replayed over (`common.tool_grammar.info_from_base`): the learner's --base
# (its tokenizer with the Neuralese role markers is the served export's). Set by the trainers (exp0_sft, grpo).
LAW_BASE = {'path': None}


def law_mask(chains, vocab_size):
    """``lfm3_fast_train.HeadMask`` of the targets of ``chains`` packed back to back (``row_tensors``' order): the
    sampling-law masks their calls were drawn under (`lower_rollout_turns.sampled_call`: a tool grammar, guidance bans;
    `common.tool_grammar.row_masks`), or None when no call was. The learner then scores each such target under the law
    the engine drew it from, for the policy, the reference and the behaviour ratio alike."""
    from natlang_neuralese.common import tool_grammar
    rows, which, tables, seen, offset = [], [], [], {}, 0
    grammar = None
    for chain in chains:
        for _, length, row in chain['segs']:
            if row.get('tool_grammar') is not None or row.get('bans'):
                if grammar is None and row.get('tool_grammar') is not None:
                    if LAW_BASE['path'] is None:
                        raise ValueError('a call drawn under a tool grammar needs exp0_sft.LAW_BASE (the --base tokenizer)')
                    grammar = tool_grammar.info_from_base(LAW_BASE['path'])
                info, sha = grammar or (None, None)
                table, index = tool_grammar.row_masks(row, info, sha, vocab_size)
                slots = []
                for mask in table:
                    key = mask.tobytes()
                    if key not in seen:
                        seen[key] = len(tables)
                        tables.append(mask)
                    slots.append(seen[key])
                hit = np.nonzero(index >= 0)[0]
                rows.extend((offset + hit).tolist())
                which.extend(slots[int(index[i])] for i in hit)
            offset += length
    if not rows:
        return None
    return lfm3_fast_train.HeadMask(np.asarray(rows, np.int64), np.stack(tables), np.asarray(which, np.int64))


# Ambient SFT (Daras et al., arXiv 2611.23229, Algorithm 1; DECISIONS 2026-11-20): with probability ``--ambient-p-clean``
# an optimizer step trains the plain SFT loss; otherwise it trains Temperature Matching, the cross-entropy from the frozen
# start checkpoint's temperature-T next-token distribution to the student's temperature-T distribution over the full
# vocabulary, at the same supervised positions and weights (KL(p_T,teacher || p_T,student) up to the teacher's entropy;
# no T^2 factor). In expectation: SFT + lambda * TM with lambda = (1 - p_c) / p_c. T -> inf is SFT, T = 1 is KD.
AMBIENT_TEMPERATURE = 2.5
AMBIENT_P_CLEAN = 0.75
AMBIENT_MODE = 'additive'


def ambient_clean(seed, step, p_clean):
    """The step's coin (one per optimizer step, Algorithm 1): True for a plain SFT step. Deterministic in (seed, step),
    so a rerun or a resume draws the same sequence."""
    if p_clean >= 1.:
        return True
    return random.Random(f'{seed}:ambient:{step}').random() < p_clean


def _tm_ce(weight, h, teacher_h, teacher_weight, temperature, sparse_ids=None, sparse_rows=None,
           teacher_sparse_ids=None, teacher_sparse_rows=None, allowed=None):
    # Preserve FP32 head logits (matching lfm3_fast._head_logits_chunk & serving engines) so tiny tm_kl
    # differences are not quantized into BF16 staircases near |logit| ~ 20.
    s_logits = lfm3_fast_train._head_logits_chunk(h, weight, sparse_ids, sparse_rows)
    s = s_logits.float() / temperature
    if allowed is not None:      # rows drawn under a sampling-law mask: both laws over the allowed tokens (law_mask)
        s = s.masked_fill(~allowed, float('-inf'))
    with torch.no_grad():
        t_logits = lfm3_fast_train._head_logits_chunk(teacher_h, teacher_weight,
                                                      teacher_sparse_ids, teacher_sparse_rows)
        t = t_logits.float() / temperature
        if allowed is not None:
            t = t.masked_fill(~allowed, float('-inf'))
        q = torch.softmax(t, -1)
        entropy = -torch.where(q > 0, q * torch.log_softmax(t, -1), 0.).sum(-1)
    return -torch.where(q > 0, q * torch.log_softmax(s, -1), 0.).sum(-1), entropy


def tm_ce(weight, h_rows, teacher_rows, teacher_weight, temperature, chunk=512, *,
          trainable_rows=None, teacher_trainable_rows=None, mask=None):
    """(per-row cross-entropy from softmax(teacher logits / T) to log_softmax(student logits / T), per-row teacher
    entropy (no grad)) over the full vocabulary (over the allowed tokens at the rows of ``mask``, a
    ``lfm3_fast_train.HeadMask``), with optional tied trainable sparse rows."""
    ces, entropies = [], []
    s_ids = trainable_rows.token_ids.to(h_rows.device) if trainable_rows is not None else None
    s_rows = trainable_rows.rows if trainable_rows is not None else None
    t_ids = teacher_trainable_rows.token_ids.to(teacher_rows.device) if teacher_trainable_rows is not None else None
    t_rows = teacher_trainable_rows.rows if teacher_trainable_rows is not None else None
    for start in range(0, len(h_rows), chunk):
        allowed = (mask.allowed_rows(start, min(len(h_rows), start + chunk), weight.shape[0], h_rows.device)
                   if mask is not None else None)
        args = (weight, h_rows[start:start + chunk], teacher_rows[start:start + chunk], teacher_weight, temperature,
                s_ids, s_rows, t_ids, t_rows, allowed)
        ce, entropy = (torch_checkpoint(_tm_ce, *args, use_reentrant=False) if torch.is_grad_enabled()
                       else _tm_ce(*args))
        ces.append(ce)
        entropies.append(entropy.detach())
    if not ces:
        empty = h_rows.new_zeros(0, dtype=torch.float32)
        return empty, empty
    return torch.cat(ces), torch.cat(entropies)


STUDENT_SOURCES = ('rl', 'rl-neuralese-graph')


def student_sampled(chain):
    """Whether a chain's targets are the student's own top-p samples (own GRPO successes, own Neuralese graph rows):
    a chain's ``sampled`` flag, else its source (``chain_source``)."""
    return bool(chain.get('sampled', chain_source(chain) in STUDENT_SOURCES))


def targets_nll(model, h, positions, targets, sampled, top_p, mask=None):
    """Per-target NLL: under each row's top-p nucleus (``token_nll_nucleus``) where ``sampled`` (bool per target), the
    full softmax elsewhere. The student sampled its own replies at top-p ``top_p``, so their likelihood is the
    nucleus-renormalized one; the full-softmax NLL of such samples is not zero in expectation on the student's own
    distribution: it drains the tail into the nucleus and so raises every token that sits inside it, ``</think>`` at
    the points where reasoning may end among them (sft-v4-run1's behaviour probe: hazard 0.546 -> 1.03; one fresh-Adam
    step on the base's own samples raised it 5x under the full NLL and lowered it under the nucleus NLL)."""
    law = {} if mask is None else {'mask': mask}
    if top_p >= 1. or sampled is None or not bool(sampled.any()):
        return token_nll(model, h, positions, targets, **law)
    nll, nucleus, _ = token_nll_nucleus(model, h, positions, targets, top_p, **law)
    return torch.where(sampled, nucleus, nll)


def sampled_targets(chain):
    """Per target of ``chain`` (bool array): a row's own ``sampled`` flags where it declares them (a bool, or one per
    target: filled-reasoning rows mix the student's sampled think block with the teacher action after it,
    ``filled_reasoning.py``), else the chain's ``student_sampled``."""
    default = student_sampled(chain)
    parts = []
    for _, length, row in chain['segs']:
        flags = row.get('sampled', default)
        flags = np.asarray(flags, bool)
        parts.append(np.broadcast_to(flags, (length,)) if flags.ndim == 0 else flags)
        if len(parts[-1]) != length:
            raise ValueError(f"row {row.get('id')}: {len(parts[-1])} sampled flags for {length} targets")
    return np.concatenate(parts) if parts else np.zeros(0, bool)


def sampled_mask(chains, device):
    """Per-target ``sampled_targets`` flags of chains packed back to back (``row_tensors`` order)."""
    return torch.cat([torch.from_numpy(np.ascontiguousarray(sampled_targets(chain))).to(device) for chain in chains])


def chain_tensors(chain, device):
    ids = torch.from_numpy(chain['ids'].astype(np.int64)).to(device)
    positions = np.concatenate([np.arange(start - 1, start + length - 1) for start, length, _ in chain['segs']])
    targets = np.concatenate([row['sample'] for _, _, row in chain['segs']])
    return ids, torch.from_numpy(positions).to(device), torch.from_numpy(targets.astype(np.int64)).to(device)


def chain_weights(chain, device):
    """Per-target loss weights (ones unless the segments declare them)."""
    weights = np.concatenate([row['weight'] if 'weight' in row else np.ones(length, np.float32)
                              for _, length, row in chain['segs']])
    return torch.from_numpy(weights.astype(np.float32)).to(device)


def empty_think_lead(sample, close_id, blank):
    """Leading tokens of ``sample`` that are the template's empty think block: ``</think>`` as the first token (the
    generation prompt opens the block) and the whitespace tokens after it (``blank(id)``); 0 when the reply reasons."""
    if not len(sample) or int(sample[0]) != close_id:
        return 0
    end = 1
    while end < len(sample) and blank(int(sample[end])):
        end += 1
    return end


def mask_empty_think(chains, tokenizer):
    """Zero the loss weight of every row's leading empty think block in ``chains`` (in place); returns the rows changed.

    A reply that was not sampled by the student and carries no reasoning (a written or dataset target: operator seeds,
    operator consumers' exact answers) renders the template's empty think block, ``</think>`` at the first target
    position. Training it teaches the student to close its reasoning at once on ordinary calls; no verified outcome
    selects that (the same rule as ``sft_sources.think_context`` for teacher replies). So is the first token after it,
    predicted at the ``</think>`` position the student never reaches (``serve.targets.empty_think_first_end``,
    opt-out ``NATLANG_EMPTY_THINK_FIRST=train``). The rest of the action trains."""
    import os
    first = os.environ.get('NATLANG_EMPTY_THINK_FIRST', 'context') != 'train'
    close_id = tokenizer.convert_tokens_to_ids('</think>')
    cache = {}

    def blank(token):
        if token not in cache:
            cache[token] = not tokenizer.decode([token]).strip()
        return cache[token]

    changed = 0
    for chain in chains:
        for _, length, row in chain['segs']:
            lead = empty_think_lead(row['sample'], close_id, blank)
            if not lead:
                continue
            lead = min(lead + 1, length) if first else lead
            weight = np.array(row['weight'], np.float32) if 'weight' in row else np.ones(length, np.float32)
            if weight[:lead].any():
                weight[:lead] = 0
                row['weight'] = weight
                changed += 1
    return changed


def vocabulary_ids(tokenizer, tokens):
    """Ids of those ``tokens`` the vocabulary has (never the unknown-token id a tokenizer gives for a missing one)."""
    vocab = tokenizer.get_vocab() if hasattr(tokenizer, 'get_vocab') else None
    ids = set()
    for token in tokens:
        if vocab is not None:
            if token in vocab:
                ids.add(int(vocab[token]))
            continue
        try:
            index = tokenizer.convert_tokens_to_ids(token)
        except (ValueError, KeyError):
            continue
        if isinstance(index, int) and index >= 0:
            ids.add(index)
    return ids


def mask_after_end(chains, tokenizer):
    """Zero the loss weight of every target after a row's first end-of-turn token (in place); returns the rows changed.

    The actor stops at the end token; what a chat template writes after it (LFM3: the newline before the next turn) is
    never sampled. ``serve.chat.assistant_reply`` returned that separator as part of the reply, so all 535 operator
    chains of sft-warmup-20261118-v1 trained ``<|im_end|>`` followed by ``\n``."""
    ends = vocabulary_ids(tokenizer, ('<|im_end|>', '<|endoftext|>'))
    changed = 0
    for chain in chains:
        for _, length, row in chain['segs']:
            hits = np.flatnonzero(np.isin(row['sample'], list(ends)))
            if not len(hits) or hits[0] + 1 >= length:
                continue
            weight = np.array(row['weight'], np.float32) if 'weight' in row else np.ones(length, np.float32)
            if weight[hits[0] + 1:].any():
                weight[hits[0] + 1:] = 0
                row['weight'] = weight
                changed += 1
    return changed


def mask_second_close(chains, tokenizer):
    """Zero the loss weight of every row's second and later ``</think>`` and of what follows each up to the action
    (``second_close_mask``; in place); returns the rows changed."""
    close_id = tokenizer.convert_tokens_to_ids('</think>')
    stops = vocabulary_ids(tokenizer, SECOND_CLOSE_STOPS)
    changed = 0
    for chain in chains:
        for _, length, row in chain['segs']:
            mask = second_close_mask(row['sample'], close_id, stops)
            if mask is None:
                continue
            weight = np.array(row['weight'], np.float32) if 'weight' in row else np.ones(length, np.float32)
            if (weight * (1 - mask)).any():
                row['weight'] = weight * mask
                changed += 1
    return changed


def context_masks(chains, tokenizer, second_close=True):
    """The load-time target rules every learner applies to prebuilt chains, whatever built them (in place): a leading
    empty think block is context (``mask_empty_think``) and nothing after the end of turn trains (``mask_after_end``).
    A second ``</think>`` in a reply and what follows it up to the action are context too (``mask_second_close``;
    ``second_close`` False opts out). Returns the rows each changed."""
    return {'empty_think_rows': mask_empty_think(chains, tokenizer), 'after_end_rows': mask_after_end(chains, tokenizer),
            **({'second_close_rows': mask_second_close(chains, tokenizer)} if second_close else {})}


def packed_rows(chains, max_tokens):
    """Greedy, order-preserving packing of ``chains`` into rows of at most ``max_tokens`` tokens (a longer chain is a row
    alone; ``max_tokens`` 0: one chain per row). Returns lists of chains."""
    rows, row, size = [], [], 0
    for chain in chains:
        n = len(chain['ids']) + source_tokens(chain)
        if row and (not max_tokens or size + n > max_tokens):
            rows.append(row)
            row, size = [], 0
        row.append(chain)
        size += n
    if row:
        rows.append(row)
    return rows


def row_tensors(row, device, starts=None):
    """(ids, positions, targets, weights, lengths) of chains packed back to back (`lfm3_fast_train.hidden` lengths).
    ``starts``: each chain from this row on (its rows before are a prefix tree's past; its targets all after it)."""
    starts = starts or [0] * len(row)
    parts = [chain_tensors(chain, device) for chain in row]
    lengths = [len(chain['ids']) - start for chain, start in zip(row, starts)]
    offsets = np.cumsum([0] + lengths[:-1])
    ids = torch.cat([p[0][start:] for p, start in zip(parts, starts)])
    positions = torch.cat([p[1] + int(o - start) for p, o, start in zip(parts, offsets, starts)])
    targets = torch.cat([p[2] for p in parts])
    weights = torch.cat([chain_weights(chain, device) for chain in row])
    return ids, positions, targets, weights, lengths


def source_tokens(chain):
    """Tokens of the pass-through producer documents a chain's imports add to its forward."""
    sources = chain.get('sources') or {}
    return sum(len(sources[key]['ids']) for key in {key for _, key in chain.get('imports') or ()} if key in sources)


def row_inputs(row, device, passthrough=True, fixed=None, starts=None):
    """(ids, positions, targets, weights, lengths, imports) of chains packed back to back (``row_tensors``, ``starts``:
    continuations of a prefix tree's past, which carry no imports) with their imports' producer documents appended as
    more documents and ``hidden``'s pass-through ``imports`` (``lfm3_fast_train.passthrough_plan``; ``passthrough``
    False: the same state detached). ``fixed``: (rows) -> ``import_tensors`` of imports without a producer document (the
    actor's stored state); without it they are refused."""
    if starts and any(start and chain.get('imports') for chain, start in zip(row, starts)):
        raise ValueError('a chain that imports Neuralese state is never a continuation of a shared prefix')
    ids, positions, targets, weights, lengths = row_tensors(row, device, starts)
    offsets = np.cumsum([0] + lengths[:-1])
    plan = lfm3_fast_train.passthrough_plan([(int(o), chain.get('imports') or (), chain.get('sources'))
                                             for o, chain in zip(offsets, row)], len(ids), frozen_all=not passthrough)
    if plan['fixed'] and fixed is None:
        raise ValueError(f'{len(plan["fixed"])} imports without a producer document and no stored-state reader')
    if plan['documents']:
        ids = torch.cat([ids] + [torch.from_numpy(np.asarray(d, np.int64)).to(device) for d in plan['documents']])
        lengths = lengths + [len(d) for d in plan['documents']]
    stored = fixed(plan['fixed']) if plan['fixed'] else None
    return ids, positions, targets, weights, lengths, lfm3_fast_train.passthrough_tensors(plan, device, stored)


def chain_mass(chain):
    if chain.get('decision_objective'):
        return DECISION_MASS + chain.get('decision_reasoning', 0)
    return float(sum(row['weight'].sum() if 'weight' in row else length for _, length, row in chain['segs']))


# The decision objective (``--no-decision-objective`` opts out; plans/semantic-fp/05 O3). A decision readout (natlang
# ``decide``, a ``readout: decision`` call) is scored, not sampled: the runtime reads softmax(s) over the options'
# sequence log-probabilities s_o = sum_t log p(c_{o,t} | prompt, c_{o,<t}) (``serve.native_learning._score_rendered``,
# the SGLang actor's ``sglang_score``). Its chain trains the proper log score of that readout,
# -sum_o y_o log softmax(s)_o (``common.decision_objective.decision_loss``, the native-graph trainer's
# ``train.decision_supervision``), with y the verified choice (own and teacher successes; a teacher distribution where
# its endpoint returned option log-probabilities). It raises the chosen option and lowers the competing ones; the
# token CE of the chosen reply alone cannot lower a wrongly preferred option. One decision counts as one target unit of
# the step's loss mass, like one invocation of the own-output objective.
DECISION_MASS = 1.
DECISION_RULE = 'logLoss'


def mark_decision_objective(chains, on=True, reasoning=True):
    """Mark the decision chains (``chains_of``: a row with its ``decision`` option set) of ``chains`` for the decision
    objective (in place; ``on`` False: they train as the plain token CE of the chosen reply, once per row: the later
    field steps of a field-mode row, ``decision_step``, are removed). A reasoned readout's decision (``reasoning_span``,
    the student's own sampled reasoning in its prompt, ``sft_sources.render_decision``) also trains that span as token
    CE (``decision_reasoning``: its token count, its loss mass besides the decision's); ``reasoning`` False: the choice
    only. With ``on`` False the plain row holds the reasoning and the reply and trains them once. Returns the count."""
    if not on:
        chains[:] = [chain for chain in chains if not chain.get('decision_step')]
    count = 0
    for chain in chains:
        if chain.get('decision') is not None:
            if on and chain.get('imports'):
                raise ValueError('a decision chain with imported Neuralese state trains in the native-graph trainer')
            chain['decision_objective'] = bool(on)
            span = chain['decision'].get('reasoning_span')
            chain['decision_reasoning'] = int(span[1] - span[0]) if on and reasoning and span is not None else 0
            count += 1
    return count


def decision_parts(chain):
    """(prompt ids, option continuation ids, target probabilities) of a decision chain; the prompt ends with the tokens
    every option shares ('prefix')."""
    start, _, row = chain['segs'][0]
    decision = chain['decision']
    options = [np.asarray(ids, np.int64) for ids in decision['options']]
    if len(options) < 2 or any(not len(ids) for ids in options):
        raise ValueError('a decision needs at least two nonempty option continuations')
    if 'probabilities' in decision:
        target = [float(p) for p in decision['probabilities']]
    else:
        target = [1. if index == decision['label'] else 0. for index in range(len(options))]
    # the tokens every option shares are prompt (serve.native_learning.decision_items, sft_sources 'prefix')
    prompt = np.asarray(chain['ids'][:start], np.int64)
    if decision.get('prefix') is not None and len(decision['prefix']):
        prompt = np.concatenate([prompt, np.asarray(decision['prefix'], np.int64)])
    return prompt, options, target


def decision_alternatives(chain):
    """Per option its other token sequences (a split-tokenized readout, ``decision_items`` split_tokenization; the
    option scores as the logsumexp over all its sequences), empty lists when the row carries none."""
    decision = chain['decision']
    alternatives = decision.get('alternatives')
    if alternatives is None:
        return [[] for _ in decision['options']]
    if len(alternatives) != len(decision['options']) or any(not len(ids) for alts in alternatives for ids in alts):
        raise ValueError('decision alternatives are nonempty sequences, one list per option')
    return [[np.asarray(ids, np.int64) for ids in alts] for alts in alternatives]


def decision_reasoning_span(chain):
    """The (start, stop) of a decision chain's reasoning in its decision prompt that trains as token CE, or None
    (no reasoned readout, or ``mark_decision_objective`` turned it off)."""
    if not chain.get('decision_reasoning'):
        return None
    start, stop = (int(v) for v in chain['decision']['reasoning_span'])
    if not 0 < start < stop or stop - start != chain['decision_reasoning']:
        raise ValueError('a decision reasoning span lies in its prompt after the first token')
    return start, stop


def decision_tokens(chain):
    prompt, options, _ = decision_parts(chain)
    return sum(len(prompt) + len(ids) for ids in options + [a for alts in decision_alternatives(chain) for a in alts])


def decision_backward(model, forward, chains, denominator, pack_tokens, activation_gb, fast=True, mark=None,
                      backward=True, student_top_p=1.):
    """Forward and backward of the decision objective over ``chains`` (decision chains), the loss
    sum(DECISION_MASS * CE + reasoning NLL) / ``denominator``. Every option is its prompt followed by its closed
    continuation; a decision's options (and, on the fast path, several decisions up to ``pack_tokens``) run as packed
    documents of one forward, so its option scores share one graph. Scores use the full softmax, as the readout does.
    A reasoned readout's chain (``decision_reasoning`` > 0, ``mark_decision_objective``) also takes the token NLL of
    its reasoning span (under ``student_top_p`` nucleus NLL when < 1 on the fast path, matching ``targets_nll``).
    Returns (sum of the weighted loss, tokens computed, {'decisions', 'ce', 'correct', 'reasoning_tokens', 'reasoning_nll'});
    ``backward`` False: no backward (validation, under ``torch.no_grad``)."""
    from natlang_neuralese.common.decision_objective import decision_loss, option_sequence_scores, sequence_owner
    mark = mark or (lambda name, since: time.time())
    device = model.lm_head.weight.device
    total, computed = 0., 0
    report = {'decisions': 0, 'ce': 0., 'correct': 0, 'reasoning_tokens': 0, 'reasoning_nll': 0.}
    groups, group, size = [], [], 0
    for chain in chains:
        n = decision_tokens(chain)
        if group and (not fast or not pack_tokens or size + n > pack_tokens):
            groups.append(group)
            group, size = [], 0
        group.append(chain)
        size += n
    if group:
        groups.append(group)
    for group in groups:
        t = time.time()
        parts = [decision_parts(chain) for chain in group]
        # each option's sequences: its canonical continuation, then (after all options) its split-form alternatives
        alternatives = [decision_alternatives(chain) for chain in group]
        owners = [sequence_owner(len(options), alts) for (_, options, _), alts in zip(parts, alternatives)]
        sequences = [options + [a for alt in alts for a in alt] for (_, options, _), alts in zip(parts, alternatives)]
        reasoning = [decision_reasoning_span(chain) for chain in group]
        # per document: its local target positions and tokens (spans in order: a reasoned decision's reasoning span
        # on its first option's document, then each option's continuation)
        documents, local, targets, spans, span_is_reasoning = [], [], [], [], []
        for (prompt, _, _), options, span in zip(parts, sequences, reasoning):
            if span is not None and span[1] > len(prompt):
                raise ValueError('a decision reasoning span lies in its prompt')
            for k, ids in enumerate(options):
                at_rows, at_ids = [], []
                if k == 0 and span is not None:
                    at_rows.append(np.arange(span[0] - 1, span[1] - 1))
                    at_ids.append(prompt[span[0]:span[1]])
                    spans.append(span[1] - span[0])
                    span_is_reasoning.append(True)
                at_rows.append(np.arange(len(prompt) - 1, len(prompt) - 1 + len(ids)))
                at_ids.append(ids)
                spans.append(len(ids))
                span_is_reasoning.append(False)
                documents.append(np.concatenate([prompt, ids]))
                local.append(np.concatenate(at_rows))
                targets.append(np.concatenate(at_ids))
        if fast:
            ids = torch.from_numpy(np.concatenate(documents)).to(device)
            if getattr(model, '_lfm3_fast', None) is not None:
                set_checkpointing(model, len(ids), activation_gb)
            h = forward(model, ids, [len(d) for d in documents])
            offsets = np.cumsum([0] + [len(d) for d in documents[:-1]])
            pos_t = torch.from_numpy(np.concatenate([o + r for o, r in zip(offsets, local)])).to(device)
            tgt_t = torch.from_numpy(np.concatenate(targets)).to(device)
            if student_top_p < 1. and any(span_is_reasoning) and getattr(model, '_lfm3_fast', None) is not None:
                sampled_t = torch.from_numpy(np.concatenate([
                    np.full(n, is_r, dtype=bool) for n, is_r in zip(spans, span_is_reasoning)])).to(device)
                nll = targets_nll(model, h, pos_t, tgt_t, sampled_t, student_top_p)
            else:
                nll = token_nll(model, h, pos_t, tgt_t)
        else:                        # the stock path: one document per forward
            nll = torch.cat([token_nll(model, forward(model, torch.from_numpy(d).to(device)),
                                       torch.from_numpy(r).to(device), torch.from_numpy(c).to(device))
                             for d, r, c in zip(documents, local, targets)])
        sums = -torch.stack([part.sum() for part in nll.split(spans)])
        t = mark('decision_forward', t)
        loss, at = sums.new_zeros((), dtype=torch.float32), 0
        for (_, options, target), owner, span in zip(parts, owners, reasoning):
            if span is not None:     # the student's own reasoning tokens: token CE, weight 1 each
                loss = loss - sums[at]
                report['reasoning_tokens'] += int(span[1] - span[0])
                report['reasoning_nll'] += float(-sums[at].detach())
                at += 1
            scores = option_sequence_scores(sums[at:at + len(owner)], owner, len(options))
            at += len(owner)
            ce = decision_loss(scores, rule=DECISION_RULE, probabilities=target)
            loss = loss + DECISION_MASS * ce
            report['ce'] += float(ce.detach())
            report['correct'] += int(int(scores.detach().argmax()) == max(range(len(target)), key=target.__getitem__))
            report['decisions'] += 1
        if backward:
            (loss / denominator).backward()
            mark('decision_backward', t)
        total += float(loss.detach())
        computed += sum(len(d) for d in documents)
    return total, computed, report


OPEN_GOAL_SCOPES = ('vibestudio', 'all')


def in_open_goal_scope(chain, scope):
    """Whether a chain counts toward the open-goal shares: ``vibestudio`` (a build's default) counts the Vibestudio
    tasks' chains (open goals and historical assignments; a mixed expert-iteration build's other parts are outside the
    contract), ``all`` (and bundles from before the scope) every chain."""
    return scope in (None, 'all') or str(chain.get('task') or '').startswith('vibestudio:')


def open_goal_exposure(chains, scope=None):
    """Count distinct verified goals and training weight in the selected chains (those in ``scope``,
    ``in_open_goal_scope``)."""
    goals, trajectories = set(), set()
    open_mass = total_mass = 0.
    for chain in chains:
        if not in_open_goal_scope(chain, scope):
            continue
        mass = chain_mass(chain)
        if mass <= 0:
            continue
        trajectories.add((chain.get('part'), chain.get('task'), chain.get('round')))
        total_mass += mass
        if goal := chain.get('open_goal_fingerprint'):
            goals.add(goal)
            open_mass += mass
    return {
        'distinct_goals': len(goals),
        'trajectories': len(trajectories),
        'weighted_target_share': open_mass / (total_mass or 1.),
    }


def open_goal_depth_report(chains):
    """Report supervised open-goal weight by distance below the original root invocation."""
    weights = collections.Counter()
    goals = collections.defaultdict(set)
    for chain in chains:
        goal = chain.get('open_goal_fingerprint')
        if not goal:
            continue
        weight = chain_mass(chain)
        if weight > 0:
            depth = str(chain.get('open_goal_depth', 'unknown'))
            weights[depth] += weight
            goals[depth].add(goal)
    return {depth: {'distinct_goals': len(goals[depth]), 'weighted_target_tokens': weight}
            for depth, weight in weights.items()}


def check_open_goal_exposure(report, minimum):
    report['distinct_goal_share'] = report['distinct_goals'] / max(1, report['trajectories'])
    report['required_share'] = minimum
    if min(report['distinct_goal_share'], report['weighted_target_share']) < minimum:
        raise SystemExit(f'Open user goal minimum not met: {json.dumps(report, sort_keys=True)}')
    return report


def chain_loss_tokens(chain):
    return sum(length for _, length, _ in chain['segs'])


def chain_source(chain):
    """The source a chain came from: its round's prefix (``rl``, ``teacher``, ``rl-neuralese-graph``, ...)."""
    return str(chain.get('round')).split(':')[0]


def stratified_epoch(chains, rng):
    """One epoch's chain order with every source spread evenly over it: episodes (task, round; their chains stay in
    order and together) shuffled within each source, each placed at the midpoint of its share of its source's
    weighted tokens (``chain_mass``), all sources merged by that position. A cut into steps then gives every step about
    the source mix of the whole set. sft-v4-run1 (random order) drew 92 own chains of ~6k loss tokens among 16k: by
    chance steps 1-11 held 4% own tokens against 19% overall, at peak LR, and val-own NLL rose 0.532 -> 0.587; it fell
    back once own-rich steps came (steps 11-20: 34%)."""
    by_source = collections.defaultdict(lambda: collections.defaultdict(list))
    for index, chain in enumerate(chains):
        by_source[chain_source(chain)][(chain['task'], chain['round'])].append(index)
    placed = []
    for source in sorted(by_source):
        episodes = list(by_source[source].values())
        rng.shuffle(episodes)
        masses = [sum(chain_mass(chains[i]) for i in members) for members in episodes]
        total = sum(masses) or 1.
        done = 0.
        for members, mass in zip(episodes, masses):
            placed.append(((done + mass / 2) / total, rng.random(), members))
            done += mass
    placed.sort(key=lambda item: item[:2])
    return [index for _, _, members in placed for index in members]


def step_groups(chains, epochs, batch_loss_tokens, seed, grouped, stratified=False):
    """Optimizer steps (lists of chain indices) of at least ``batch_loss_tokens`` loss tokens each, counted as the loss
    counts them (``chain_mass``: zero-weight context tokens such as a teacher's think block do not fill a batch).
    Random order (``grouped`` False): one shuffled list over all epochs, cut in order; a last group under half a batch
    is dropped. Stratified (``grouped`` False, ``stratified``): the same cut over ``stratified_epoch`` orders. Grouped: per epoch, episodes (task, round) in lexicographic order of their first chain's tokens (so the
    chains that share a system prompt, task prompt or earlier calls are neighbours; ties and the cut offset random) with
    each episode's chains in order, cut into steps (a remainder under half a batch joins the epoch's last step) whose
    order is then shuffled: every chain exactly once per epoch, reproducible from ``seed``."""
    rng = random.Random(seed)
    if not grouped:
        order = []
        for _ in range(epochs):
            if stratified:
                order += stratified_epoch(chains, rng)
                continue
            epoch = list(range(len(chains)))
            rng.shuffle(epoch)
            order += epoch
        steps, group, tokens = [], [], 0
        for index in order:
            group.append(index)
            tokens += chain_mass(chains[index])
            if tokens >= batch_loss_tokens:
                steps.append(group)
                group, tokens = [], 0
        if group and tokens >= batch_loss_tokens // 2:
            steps.append(group)
        return steps
    episodes = collections.defaultdict(list)
    for index, chain in enumerate(chains):
        episodes[(chain['task'], chain['round'])].append(index)
    first = {key: chains[members[0]]['ids'].astype('>i4').tobytes() for key, members in episodes.items()}
    steps = []
    for _ in range(epochs):
        keys = list(episodes)
        rng.shuffle(keys)
        keys.sort(key=first.__getitem__)          # stable: identical first chains stay in random order
        shift = rng.randrange(len(keys)) if keys else 0
        keys = keys[shift:] + keys[:shift]         # batch boundaries differ between epochs
        epoch, group, tokens = [], [], 0
        for key in keys:
            for index in episodes[key]:
                group.append(index)
                tokens += chain_mass(chains[index])
                if tokens >= batch_loss_tokens:
                    epoch.append(group)
                    group, tokens = [], 0
        if group:
            if tokens >= batch_loss_tokens // 2 or not epoch:
                epoch.append(group)
            else:
                epoch[-1] += group
        rng.shuffle(epoch)
        steps += epoch
    return steps


def step_tree(chains, share_prefix_tokens):
    """The prefix tree of one step's chains (``lfm3_fast_train.prefix_tree``; 0: every chain whole): chains that import
    Neuralese state stay whole (they carry their producer documents in the same forward)."""
    plain = [i for i, chain in enumerate(chains) if not chain.get('imports')]
    nodes = lfm3_fast_train.remap_tree(lfm3_fast_train.prefix_tree(
        [chains[i]['ids'] for i in plain], [chains[i]['segs'][0][0] - 1 for i in plain], share_prefix_tokens), plain)
    return nodes + [{'start': 0, 'index': i} for i, chain in enumerate(chains) if chain.get('imports')]


def add_prefix_tree_args(parser):
    """The prefix tree's flags, shared by both learners (exp0_sft, grpo)."""
    parser.add_argument('--prefix-tree-mode', choices=lfm3_fast_train.PREFIX_TREE_MODES, default='levels',
                        help='shared prefixes: tree-mask (each batch one flattened forward, every token attending and '
                        'convolving along its own path), levels (level-by-level packed continuations over cached state), '
                        'off (every chain whole; as --share-prefix-tokens 0)')
    parser.add_argument('--prefix-retained-gb', type=float, default=-1.,
                        help='levels: what the shared prefixes keep until their backward (layer inputs, FP32 history keys '
                        'and values and their gradients), at most this; a larger step runs its prefix tree in parts '
                        '(lfm3_fast_train.prefix_parts, exact, shared ancestors recomputed per part). grpo-v4-run1 step 1 '
                        'held 229k prefix tokens (~31 GB) at once. It comes out of --activation-gb, so the default (-1) '
                        'picks the bound with the least estimated compute, prefix copies vs leaf layer recompute '
                        '(lfm3_fast_train.choose_retained_gb; 8 without an activation budget); N: a fixed bound (the '
                        'earlier default was 8); 0: unbounded')
    parser.add_argument('--tree-batch-tokens', type=int, default=-1,
                        help='tree-mask batch cap (prefixes once plus their continuations); default: the cap with the least '
                        'estimated compute, prefix copies vs layer recompute (lfm3_fast_train.plan_tree_batches)')


def prefix_tree_settings(args):
    """(share_prefix_tokens, mode) of the learner flags (``off``: no shared prefix)."""
    return (0, 'levels') if args.prefix_tree_mode == 'off' else (args.share_prefix_tokens, args.prefix_tree_mode)


def step_backward(model, forward, chains, denominator, pack_tokens, activation_gb, share_prefix_tokens, fast=True,
                  mark=None, passthrough=True, mode='levels', tree_batch_tokens=None, retained_gb=None, student_top_p=1.,
                  ambient=None):
    """Forward and backward of one optimizer step's chains, the loss sum(weight * nll) / ``denominator``. Prefixes of at
    least ``share_prefix_tokens`` tokens that several chains share are computed once and every batch is packed into
    varlen rows of ``pack_tokens`` (`lfm3_fast_train.run_prefix_tree` in ``mode``, exact gradients); 0: every chain
    whole, packed.
    Chains that import Neuralese state stay whole and carry their producer documents in the same forward (``row_inputs``:
    the consumer's loss reaches the producer's state; ``passthrough`` False: detached). ``student_top_p`` < 1: the
    student's own sampled chains take the nucleus NLL (``targets_nll``).
    ``ambient`` (a Temperature Matching step, ``ambient_teacher_rows``): {'temperature', 'head' (the teacher's output
    weight), 'rows' {chain index: teacher hidden rows of its targets}}; the loss is then sum(weight * ``tm_ce``) /
    ``denominator`` instead, and ``ambient['kl']`` collects sum(weight * KL) (telemetry).
    Returns (sum of the weighted loss, tokens computed)."""
    mark = mark or (lambda name, since: time.time())
    total_nll = 0.
    device = model.lm_head.weight.device

    def loss_of(h, positions, targets, weights, indices, sampled):
        mask = law_mask([chains[index] for index in indices], model.lm_head.weight.shape[0])
        if ambient is None:
            return targets_nll(model, h, positions, targets, sampled, student_top_p, mask) * weights
        teacher = torch.cat([ambient['rows'][index] for index in indices])
        ce, entropy = tm_ce(model.lm_head.weight, h[positions], teacher, ambient['head'], ambient['temperature'],
                            trainable_rows=getattr(model, '_lfm3_trainable_rows', None),
                            teacher_trainable_rows=ambient.get('teacher_trainable_rows'), mask=mask)
        ambient['kl'] = ambient.get('kl', 0.) + ((ce.detach() - entropy) * weights).sum().item()
        ambient['tm_nll'] = ambient.get('tm_nll', 0.) + (ce.detach() * weights).sum().item()
        if ambient.get('mode') == 'additive':
            # Keep 1.0x SFT supervision on every unique batch (preventing 1-epoch + max_prior_exposures=1 from
            # permanently discarding 25% of unique data) while adding weighted TM to preserve exact Eq. 4 balance.
            sft_loss = targets_nll(model, h, positions, targets, sampled, student_top_p, mask) * weights
            ambient['sft_nll'] = ambient.get('sft_nll', 0.) + sft_loss.detach().sum().item()
            tm_weight = float(ambient.get('weight', 1.0))
            return sft_loss + (tm_weight * ce * weights)
        return ce * weights
    if not fast:                     # the stock HF path: one chain per forward, no imports
        for index, chain in enumerate(chains):
            t = time.time()
            ids, positions, targets, weights, _, imports = row_inputs([chain], device, passthrough)
            if imports is not None:
                raise ValueError('imported Neuralese state needs the fast path (lfm3_fast_train)')
            h = forward(model, ids)
            t = mark('forward', t)
            nll = loss_of(h, positions, targets, weights, [index], sampled_mask([chain], device))
            (nll.sum() / denominator).backward()
            mark('loss_backward', t)
            total_nll += nll.sum().item()
            del h, nll
        return total_nll, sum(len(chain['ids']) for chain in chains)
    nodes = step_tree(chains, share_prefix_tokens)
    cuda_ids = {}

    def ids_of(index, start, end):
        if index not in cuda_ids:
            cuda_ids[index] = torch.from_numpy(chains[index]['ids'].astype(np.int64)).to(device)
        return cuda_ids[index][start:end]

    def tokens(index, start):
        return len(chains[index]['ids']) - start + source_tokens(chains[index])

    clock = {'leaves': None, 'end': None}

    def leaf(items, run):
        nonlocal total_nll
        if clock['leaves'] is None:
            clock['leaves'] = mark('prefix_forward', clock['start'])
        t = time.time()
        ids, positions, targets, weights, lengths, imports = row_inputs(
            [chains[index] for index, _ in items], device, passthrough, starts=[start for _, start in items])
        h = run(ids, lengths, imports)
        t = mark('forward', t)
        sampled = sampled_mask([chains[index] for index, _ in items], device)
        nll = loss_of(h, positions, targets, weights, [index for index, _ in items], sampled)
        (nll.sum() / denominator).backward()
        clock['end'] = mark('loss_backward', t)
        total_nll += nll.sum().item()
    clock['start'] = time.time()
    lfm3_fast_train.run_prefix_tree(model, nodes, ids_of, leaf, tokens, pack_tokens or 1, activation_gb,
                                    mode=mode if share_prefix_tokens > 0 else 'levels', batch_tokens=tree_batch_tokens,
                                    retained_gb=retained_gb)
    if clock['end'] is not None:
        mark('prefix_backward', clock['end'])
    return total_nll, lfm3_fast_train.tree_tokens(nodes, [tokens(i, 0) for i in range(len(chains))])


def ambient_teacher_rows_lora(learner: Lfm3Learner, chains, ref_bundles, ref_rows, pack_tokens, activation_gb,
                              share_prefix_tokens, passthrough=True, mode='levels', tree_batch_tokens=None,
                              retained_gb=None):
    """Frozen teacher hidden rows for an Ambient SFT Temperature Matching step using the learner's immutable
    reference view (zero 3.4 GB base weight copy, zero live weight swap)."""
    model = learner.model
    device = model.lm_head.weight.device
    rows = {}
    nodes = step_tree(chains, share_prefix_tokens)
    cuda_ids = {}

    def ids_of(index, start, end):
        if index not in cuda_ids:
            cuda_ids[index] = torch.from_numpy(chains[index]['ids'].astype(np.int64)).to(device)
        return cuda_ids[index][start:end]

    def tokens(index, start):
        return len(chains[index]['ids']) - start + source_tokens(chains[index])

    def leaf(items, run):
        ids, positions, _targets, _, lengths, imports = row_inputs(
            [chains[index] for index, _ in items], device, passthrough, starts=[start for _, start in items])
        h = run(ids, lengths, imports)[positions].detach()
        counts = [chain_loss_tokens(chains[index]) for index, _ in items]
        for (index, _), part in zip(items, h.split(counts)):
            rows[index] = part

    with learner.reference_view(ref_bundles, ref_rows) as ref_kw, torch.no_grad():
        lfm3_fast_train.run_prefix_tree(
            model, nodes, ids_of, leaf, tokens, pack_tokens or 1, activation_gb,
            backward=False, mode=mode if share_prefix_tokens > 0 else 'levels',
            batch_tokens=tree_batch_tokens, retained_gb=retained_gb, **ref_kw.kwargs,
        )
    learner.zero_grad(set_to_none=True)
    return rows


def ambient_teacher_rows(model, forward, chains, live, teacher, restore, pack_tokens, activation_gb, share_prefix_tokens,
                         fast=True, passthrough=True, mode='levels', tree_batch_tokens=None, retained_gb=None):
    """Legacy helper kept for standalone unit tests (`test_ambient_sft.py`); production SFT uses ``ambient_teacher_rows_lora``."""
    device = model.lm_head.weight.device
    rows = {}
    with torch.no_grad():
        torch._foreach_copy_([live[name] for name in teacher], list(teacher.values()))
    try:
        if not fast:
            with torch.no_grad():
                for index, chain in enumerate(chains):
                    ids, positions, _, _, _, imports = row_inputs([chain], device, passthrough)
                    rows[index] = forward(model, ids)[positions].detach()
            return rows
        nodes = step_tree(chains, share_prefix_tokens)
        cuda_ids = {}

        def ids_of(index, start, end):
            if index not in cuda_ids:
                cuda_ids[index] = torch.from_numpy(chains[index]['ids'].astype(np.int64)).to(device)
            return cuda_ids[index][start:end]

        def tokens(index, start):
            return len(chains[index]['ids']) - start + source_tokens(chains[index])

        def leaf(items, run):
            ids, positions, targets, _, lengths, imports = row_inputs(
                [chains[index] for index, _ in items], device, passthrough, starts=[start for _, start in items])
            h = run(ids, lengths, imports)[positions].detach()
            counts = [chain_loss_tokens(chains[index]) for index, _ in items]
            for (index, _), part in zip(items, h.split(counts)):
                rows[index] = part
        with torch.enable_grad():
            lfm3_fast_train.run_prefix_tree(model, nodes, ids_of, leaf, tokens, pack_tokens or 1, activation_gb,
                                            backward=False, mode=mode if share_prefix_tokens > 0 else 'levels',
                                            batch_tokens=tree_batch_tokens, retained_gb=retained_gb)
        return rows
    finally:
        restore()
        for p in live.values():
            p.grad = None


def load_chain_bundle(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle)


def load_chains(path):
    return load_chain_bundle(path)['chains']


def validation_loss(model, forward, chains, pack_tokens=0, student_top_p=1.):
    """Weighted mean NLL over prebuilt validation chains (no gradient; packed rows on the fast path), under the
    training objective (``targets_nll``). Excludes decision-objective chains (scored separately by
    ``decision_validation``) so decision rows are neither scored as token CE nor double-counted across field steps."""
    token_chains = [c for c in chains if not c.get('decision_objective')]
    total, mass = 0., 0.
    with torch.no_grad():
        for row in packed_rows(token_chains, pack_tokens):
            ids, positions, targets, weights, lengths, imports = row_inputs(row, 'cuda')
            h = forward(model, ids, lengths, imports) if imports is not None or len(lengths) > 1 else forward(model, ids)
            nll = targets_nll(model, h, positions, targets, sampled_mask(row, 'cuda'), student_top_p,
                              law_mask(row, model.lm_head.weight.shape[0]))
            total += (nll * weights).sum().item()
            mass += weights.sum().item()
    return total / max(mass, 1e-9)


def resumed_decision_objective(args, wanted, key='decision_objective'):
    """Whether this run trains the decision objective (``key`` 'decision_reasoning': a reasoned readout's reasoning
    tokens): ``wanted``, except that a ``--resume`` keeps the setting its learner state trained with (the objective
    changes decision chains' loss mass, so the step list a state's batch cursor indexes; a state saved before the
    setting existed trained without it)."""
    state_dir = Path(args.state_dir) if args.state_dir else Path(args.out_dir) / 'learner-state'
    if not (args.resume and (state_dir / 'learner_manifest.json').is_file()):
        return wanted
    saved = None
    if (state_dir / 'trainer_state.pt').is_file():
        saved = (torch.load(state_dir / 'trainer_state.pt', map_location='cpu', weights_only=False) or {}).get(key)
    saved = bool(saved)
    if saved != wanted:
        print(json.dumps({'event': 'decision objective kept from the resumed learner state', key: saved,
                          'state_dir': str(state_dir)}), flush=True)
    return saved


def resumed_ambient_mode(args, wanted):
    """The Ambient mode this run trains with: ``wanted``, except that a ``--resume`` keeps its learner state's mode (a
    state saved before ``ambient_mode`` was recorded trained with ``replace``: ddd97a765 made ``additive`` the default,
    which changes the objective a resumed round continues; review 2026-11-26)."""
    state_dir = Path(args.state_dir) if args.state_dir else Path(args.out_dir) / 'learner-state'
    if not (args.resume and (state_dir / 'learner_manifest.json').is_file()):
        return wanted
    saved = 'replace'
    if (state_dir / 'trainer_state.pt').is_file():
        saved = (torch.load(state_dir / 'trainer_state.pt', map_location='cpu', weights_only=False) or {}).get(
            'ambient_mode') or 'replace'
    if saved != wanted:
        print(json.dumps({'event': 'ambient mode kept from the resumed learner state', 'ambient_mode': saved,
                          'state_dir': str(state_dir)}), flush=True)
    return saved


def decision_validation(model, forward, val_sets, pack_tokens=0, activation_gb=0.):
    """Per validation set with decision chains: the mean decision log score and the share of decisions whose top
    option is the target (``decision_backward`` without backward)."""
    record = {}
    with torch.no_grad():
        for name, chains in val_sets.items():
            decided = [chain for chain in chains if chain.get('decision') is not None]
            if decided:
                _, _, report = decision_backward(model, forward, decided, 1., pack_tokens, activation_gb,
                                                 backward=False)
                record[f'val_decision_ce:{name}'] = report['ce'] / report['decisions']
                record[f'val_decision_accuracy:{name}'] = report['correct'] / report['decisions']
    return record


def parity(args):
    episodes = pickle.load(open(args.turns, 'rb'))['episodes']
    chains = [c for c in chains_of(episodes) if len(c['ids']) < args.max_tokens][:args.samples]
    model, _, _ = load_model(args.checkpoint, args.base, dtype=torch.bfloat16 if args.bf16_params else torch.float32)
    model.eval()
    diffs, nll = [], []
    for chain in chains:
        ids, positions, targets = chain_tensors(chain, 'cuda')
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=not args.bf16_params):
            h = hidden(model, ids)
            value = -token_nll(model, h, positions, targets, mask=law_mask([chain], model.lm_head.weight.shape[0]))
        recorded = torch.from_numpy(np.concatenate([row['logp'] for _, _, row in chain['segs']])).cuda()
        diffs.append((value.float() - recorded))
        nll.append(-value.float().mean().item())
        d = diffs[-1]
        print(json.dumps({'tokens': len(ids), 'sampled': len(d), 'mean_abs': d.abs().mean().item(), 'mean_signed': d.mean().item(),
                          'p99_abs': d.abs().quantile(0.99).item(), 'nll_hf': nll[-1], 'nll_actor': -recorded.mean().item()}), flush=True)
    d = torch.cat(diffs)
    print(json.dumps({'parity_tokens': len(d), 'mean_abs': d.abs().mean().item(), 'mean_signed': d.mean().item(),
                      'frac_over_0.5': (d.abs() > 0.5).float().mean().item()}))


def train(args):
    run_started = time.time()
    # what the run trains on, recorded in the data registry before it is read (data no entry holds stops it here;
    # NATLANG_ALLOW_UNREGISTERED_DATA=1 opts out)
    from natlang_neuralese.common import data_registry
    data_registry.consume([args.chains or args.turns], run=data_registry.run_name('sft', args.out_dir),
                          how='SFT training chains' if args.chains else 'SFT training turns')
    if args.val_chains:
        data_registry.consume(args.val_chains, run=data_registry.run_name('sft', args.out_dir),
                              how='SFT validation chains')
    if not 0. < args.ambient_p_clean <= 1. or args.ambient_temperature <= 0.:
        raise SystemExit('--ambient-p-clean must be in (0, 1] and --ambient-temperature > 0')
    episodes = [] if args.chains else pickle.load(open(args.turns, 'rb'))['episodes']
    if args.max_per_task:
        # compute-bound (GB10: about 1k tokens/s of fwd+bwd at 10k context): the shortest verified episodes of each task
        by_task = collections.defaultdict(list)
        for episode in episodes:
            by_task[episode['task_id']].append((sum(len(c['ids']) for c in chains_of([episode])), episode['round'], episode['sample'], episode))
        episodes = [item[-1] for task in sorted(by_task) for item in sorted(by_task[task], key=lambda x: x[:3])[:args.max_per_task]]
    bundle = load_chain_bundle(args.chains) if args.chains else {}
    chains = bundle['chains'] if args.chains else chains_of(episodes)
    loss_tokens = sum(length for c in chains for _, length, _ in c['segs'])
    val_sets = {Path(path).stem: load_chains(path) for path in args.val_chains}
    student_top_p = args.student_top_p
    if args.context_masks == 'on':
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=False)
        second = args.second_close == 'context'
        print(json.dumps({'context_masks': context_masks(chains, tokenizer, second),
                          **{f'context_masks:{name}': context_masks(c, tokenizer, second) for name, c in val_sets.items()}}),
              flush=True)
    decision_on = resumed_decision_objective(args, not args.no_decision_objective)
    reasoning_on = resumed_decision_objective(args, not args.no_decision_reasoning, 'decision_reasoning')
    ambient_mode = resumed_ambient_mode(args, getattr(args, 'ambient_mode', AMBIENT_MODE))
    decisions = mark_decision_objective(chains, decision_on, reasoning_on)
    for vc in val_sets.values():
        mark_decision_objective(vc, decision_on, reasoning_on)
    print(json.dumps({'decision_objective': {'on': decision_on, 'chains': decisions, 'rule': DECISION_RULE,
                                             'mass_per_decision': DECISION_MASS,
                                             'reasoning': {'on': reasoning_on,
                                                           'reasoned': sum(c['decision'].get('reasoning_span') is not None
                                                                           for c in chains if c.get('decision') is not None),
                                                           'trained': sum(bool(c.get('decision_reasoning')) for c in chains),
                                                           'target_tokens': int(sum(c.get('decision_reasoning', 0)
                                                                                    for c in chains))},
                                             'options': int(sum(len(c['decision']['options']) for c in chains
                                                                if c.get('decision') is not None)),
                                             'forward_tokens': int(sum(decision_tokens(c) for c in chains
                                                                       if c.get('decision_objective'))),
                                             **{f'val:{name}': sum(c.get('decision') is not None for c in v)
                                                for name, v in val_sets.items()}}}), flush=True)
    print(json.dumps({'episodes': len(episodes), 'sequences': len(chains), 'sequence_tokens': int(sum(len(c['ids']) for c in chains)),
                      'loss_tokens': int(loss_tokens), 'epochs': args.epochs}), flush=True)
    order = 'stratified' if args.batch_order == 'auto' else args.batch_order
    steps = step_groups(chains, args.epochs, args.batch_loss_tokens, args.seed, order == 'grouped',
                        stratified=order == 'stratified')
    total = len(steps)
    if args.max_steps:
        steps = steps[:args.max_steps]
    minimum = bundle.get('minimum_open_goal_share')
    if minimum is not None and args.open_goal_mix_check == 'on':
        if type(minimum) not in (float, int) or not 0. <= minimum <= 1.:
            raise SystemExit('Invalid minimum open user goal share in the training chains')
        scope = bundle.get('open_goal_scope')
        if scope not in (None, *OPEN_GOAL_SCOPES):
            raise SystemExit(f'Invalid open user goal scope in the training chains: {scope!r}')
        report = open_goal_exposure((chains[index] for step in steps for index in step), scope)
        print(json.dumps({'planned_open_user_goals': check_open_goal_exposure(report, minimum)}), flush=True)
        depths = open_goal_depth_report(chains[index] for step in steps for index in step)
        print(json.dumps({'planned_open_user_goal_depths': depths}), flush=True)

    from natlang_neuralese.common.hashing import sha256_file_hex_cached
    source_sha = sha256_file_hex_cached(Path(args.checkpoint))
    chains_sha = sha256_file_hex_cached(Path(args.chains)) if args.chains else None
    spec = AdapterSpec.for_sft(
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        trainable_rows=not args.no_trainable_rows,
        gate_rank=args.gate_rank,
    )
    learner = Lfm3Learner.from_checkpoint(
        args.checkpoint, args.base, spec, device='cuda', dtype=torch.bfloat16,
        compile_layers=not args.no_compile, fp8_base=False if args.no_fp8_base else None,
        seed=args.seed, base_weights_sha256=source_sha, marker_init=args.marker_init,
    )
    model, state = learner.model, learner.state
    if learner.marker_init:
        print(json.dumps({'event': 'role-marker rows initialised from text', **learner.marker_init}), flush=True)
    model.train()
    forward = lfm3_fast_train.hidden
    pack_tokens = args.pack_tokens if args.pack_tokens >= 0 else resident_tokens(model, args.activation_gb)
    ref_bundles, ref_rows = learner.snapshot_reference()
    optimizer = learner.build_optimizer(lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=args.weight_decay,
                                        row_lr_scale=args.row_lr_scale)
    parameters = learner.trainable_parameters()

    start_step = 0
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state_dir = Path(args.state_dir) if args.state_dir else out_dir / 'learner-state'
    trainer_state = None
    trained_rounds: list[str] = []
    round_id = getattr(args, 'round_id', None)
    if args.resume and (state_dir / 'learner_manifest.json').is_file():
        restored = load_learner_state(state_dir, learner, optimizer, verify_base_sha=source_sha)
        trained_rounds = list(restored.get('trained_rounds') or ())
        if round_id and round_id in trained_rounds:
            print(json.dumps({'event': 'round already trained (dedup ledger)', 'round_id': round_id,
                              'trained_rounds': trained_rounds}), flush=True)
            return
        # the state's batch cursor indexes this run's step list: a state of other chains or another schedule (epochs,
        # loss tokens per step, seed, order) would resume at a wrong step of a different list
        saved_chains = restored['manifest'].get('data_manifest_sha256')
        if saved_chains and saved_chains != chains_sha:
            raise SystemExit(f'--resume: {state_dir} trained chains {saved_chains}, not {chains_sha}')
        saved_total = (restored.get('batch_cursor') or {}).get('total_steps')
        saved_schedule = (restored.get('trainer_state') or {}).get('schedule') or {}
        if saved_total is not None and int(saved_total) != total and not saved_schedule.get('replanned'):
            raise SystemExit(f'--resume: {state_dir} has a {saved_total}-step schedule, these arguments make {total} '
                             '(--epochs/--batch-loss-tokens/--seed/--batch-order must match the run it resumes)')
        start_step = int((restored.get('batch_cursor') or {}).get('next_step', restored['manifest']['learner_step']))
        if restored['ref_bundles'] or restored['ref_rows'] is not None:
            ref_bundles, ref_rows = restored['ref_bundles'], restored['ref_rows']
        trainer_state = restored.get('trainer_state') or {}
        # the --hours replan cut the schedule: the resume trains the same cut schedule with the same cosine
        schedule = trainer_state.get('schedule')
        if schedule:
            total = int(schedule['total'])
            del steps[int(schedule['steps']):]
        print(json.dumps({'resumed_from': str(state_dir), 'start_step': start_step,
                          'effective_policy_sha256': learner.effective_policy_sha256(source_sha)}), flush=True)

    share_prefix_tokens, tree_mode = prefix_tree_settings(args)
    importing = sum(1 for chain in chains if chain.get('imports'))
    adapter = sorted(k for k in state['interface_state'] if k.startswith('adapter.'))
    if importing and adapter:
        raise SystemExit(f'{importing} chains import Neuralese state: needs a checkpoint without a '
                         f'producer adapter (has {adapter})')
    print(json.dumps({'optimizer_steps': total, 'start_step': start_step, 'lr': args.lr,
                      'batch_loss_tokens': args.batch_loss_tokens, 'pack_tokens': pack_tokens,
                      'activation_gib': args.activation_gb,
                      'share_prefix_tokens': share_prefix_tokens, 'prefix_tree_mode': tree_mode, 'batch_order': order,
                      'learner': learner.parameter_summary(),
                      'ambient': {'p_clean': args.ambient_p_clean, 'temperature': args.ambient_temperature}}),
          flush=True)
    log = open(out_dir / 'train.jsonl', 'a')
    if val_sets and start_step == 0:
        model.eval()
        record = {'step': 0, **{f'val_loss:{name}': validation_loss(model, forward, c, pack_tokens, student_top_p) for name, c in val_sets.items()},
                  **decision_validation(model, forward, val_sets, pack_tokens, args.activation_gb)}
        model.train()
        print(json.dumps(record), flush=True)
        log.write(json.dumps(record) + '\n')
    started = time.time()
    last_state_save = started
    forward_tokens = 0
    phase = collections.Counter()
    compile_health = lfm3_fast_train.CompileHealthTracker()
    oom_plan = OOMRetryPlan(pack_tokens=pack_tokens or 1, activation_gib=float(args.activation_gb))
    best = None
    if trainer_state:
        if trainer_state.get('oom_plan'):
            oom_plan = OOMRetryPlan.from_dict(trainer_state['oom_plan'])   # a shrunk plan stays shrunk
        best = trainer_state.get('best')                                   # --keep-best-val's earlier best
    if args.keep_best_val and start_step == 0:
        if not val_sets or f'val_loss:{args.keep_best_val}' not in record:
            raise SystemExit(f'--keep-best-val {args.keep_best_val}: no such --val-chains set')
        best = {'loss': record[f'val_loss:{args.keep_best_val}'], 'step': 0,
                'leaves': {name: t.detach().cpu().clone() for name, t in learner.named_trainable_tensors().items()}}

    peak_by_phase = {}

    def mark(name, since):
        torch.cuda.synchronize()
        now = time.time()
        phase[name] += now - since
        # the CUDA peak of the phase that just ended (each mark starts a new peak window)
        peak = torch.cuda.max_memory_allocated() / 2**30
        peak_by_phase[name] = round(max(peak_by_phase.get(name, 0.), peak), 2)
        torch.cuda.reset_peak_memory_stats()
        return now

    def save_state_now(next_step_idx: int, *, mark_round_done: bool = False):
        nonlocal last_state_save
        t_save = time.time()
        rounds_now = list(trained_rounds)
        if mark_round_done and round_id and round_id not in rounds_now:
            rounds_now.append(round_id)
        save_learner_state(
            state_dir, learner, optimizer, learner_step=next_step_idx,
            source_checkpoint_id=str(args.checkpoint), source_step=int(state['step']),
            base_weights_sha256=source_sha, objective_version='sft_ambient_v1',
            optimizer_config={'lr': args.lr, 'betas': [0.9, 0.95], 'eps': 1e-8, 'weight_decay': args.weight_decay,
                              'clip': args.clip, 'warmup': args.warmup, 'min_lr_fraction': args.min_lr_fraction},
            ambient_state={'seed': args.seed, 'step': next_step_idx, 'p_clean': args.ambient_p_clean,
                           'temperature': args.ambient_temperature},
            data_manifest_sha256=chains_sha,
            batch_cursor={'next_step': next_step_idx, 'total_steps': total},
            trained_rounds=rounds_now,
            ref_bundles=ref_bundles, ref_rows=ref_rows,
            trainer_state={'oom_plan': oom_plan.to_dict(), 'best': best, 'decision_objective': decision_on,
                           'decision_reasoning': reasoning_on, 'ambient_mode': ambient_mode,
                           'schedule': {'total': total, 'steps': len(steps), 'replanned': replanned}},
        )
        last_state_save = time.time()
        phase['state_save'] += last_state_save - t_save

    replanned = bool((trainer_state or {}).get('schedule', {}).get('replanned'))
    completed_step = start_step
    stopped_early = False
    step = start_step
    while step < len(steps):
        group = steps[step]
        if args.stop_file and Path(args.stop_file).exists():
            save_state_now(step)
            print(json.dumps({'stopped_by': args.stop_file, 'at_step': step}), flush=True)
            stopped_early = True
            break
        fraction = step / max(1, total - 1)
        scale = min(1., (step + 1) / max(1, args.warmup)) * (args.min_lr_fraction + (1 - args.min_lr_fraction) * 0.5 * (1 + math.cos(math.pi * fraction)))
        for g in optimizer.param_groups:
            g['lr'] = args.lr * scale * g.get('lr_scale', 1.0)
        denominator = sum(chain_mass(chains[i]) for i in group)
        step_started = time.time()
        torch.cuda.reset_peak_memory_stats()
        peak_by_phase.clear()
        # decision chains take the decision objective in every step (``decision_backward``); the rest the step's
        # token loss (plain SFT or Temperature Matching)
        step_decisions = [chains[i] for i in group if chains[i].get('decision_objective')]
        step_chains = [chains[i] for i in group if not chains[i].get('decision_objective')]
        is_clean = ambient_clean(args.seed, step, args.ambient_p_clean)

        def run_step_body(cur_pack_tokens, cur_act_gib):
            ambient, decided = None, (0., 0, {'decisions': 0, 'ce': 0., 'correct': 0, 'reasoning_tokens': 0,
                                              'reasoning_nll': 0.})
            if step_decisions:
                decided = decision_backward(model, forward, step_decisions, denominator, cur_pack_tokens, cur_act_gib,
                                            mark=mark, student_top_p=student_top_p)
            if not step_chains:
                return decided[0], decided[1], ambient, decided[2]
            if not is_clean:
                t_amb = time.time()
                rows = ambient_teacher_rows_lora(
                    learner, step_chains, ref_bundles, ref_rows, cur_pack_tokens, cur_act_gib,
                    share_prefix_tokens, passthrough=not args.no_neuralese_passthrough, mode=tree_mode,
                    tree_batch_tokens=args.tree_batch_tokens, retained_gb=args.prefix_retained_gb,
                )
                mark('ambient_teacher', t_amb)
                ambient = {'temperature': args.ambient_temperature, 'head': model.lm_head.weight,
                           'teacher_trainable_rows': ref_rows, 'rows': rows,
                           'mode': ambient_mode,
                           'weight': (1.0 / args.ambient_p_clean) if ambient_mode == 'additive' else 1.0}
            total_nll_val, computed_val = step_backward(
                model, forward, step_chains, denominator, cur_pack_tokens, cur_act_gib, share_prefix_tokens,
                fast=True, mark=mark, passthrough=not args.no_neuralese_passthrough, mode=tree_mode,
                tree_batch_tokens=args.tree_batch_tokens, retained_gb=args.prefix_retained_gb,
                student_top_p=student_top_p, ambient=ambient,
            )
            if ambient is not None:
                ambient['decision_nll'] = decided[0]      # the additive log line adds it to the SFT part
            return total_nll_val + decided[0], computed_val + decided[1], ambient, decided[2]

        total_nll, computed, ambient, decided = run_step_with_oom_retry(run_step_body, learner, oom_plan)
        forward_tokens += computed
        t = time.time()
        norm = torch.nn.utils.clip_grad_norm_(parameters, args.clip).item()
        fam_grads = learner.family_grad_norms()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        mark('optimizer', t)
        completed_step = step + 1
        step_s = time.time() - step_started
        chain_denom = sum(chain_mass(c) for c in step_chains) if step_chains else denominator
        if ambient is None:
            logged_loss = total_nll / denominator
            loss_kind = 'sft'
            ambient_fields = {}
        elif ambient.get('mode') == 'additive':
            # ``decided`` is the decision report dict here (run_step_body's 4th value): ``decided[0]`` raised KeyError
            # on the first non-clean step of every additive round (aei1 r0014 2026-11-26)
            logged_loss = (ambient.get('sft_nll', 0.) + ambient.get('decision_nll', 0.)) / denominator
            loss_kind = 'sft+tm'
            ambient_fields = {'tm_kl': ambient.get('kl', 0.) / max(1e-9, chain_denom),
                              'tm_loss': ambient.get('tm_nll', 0.) / max(1e-9, chain_denom)}
        else:
            logged_loss = total_nll / denominator
            loss_kind = 'tm'
            ambient_fields = {'tm_kl': ambient.get('kl', 0.) / max(1e-9, chain_denom)}
        record = {'step': completed_step, 'of': total, 'loss': logged_loss, 'lr': args.lr * scale,
                  'grad_norm': norm, 'family_grad_norms': fam_grads,
                  'loss_kind': loss_kind,
                  **ambient_fields,
                  **({'decisions': decided['decisions'], 'decision_ce': decided['ce'] / decided['decisions'],
                      'decision_accuracy': decided['correct'] / decided['decisions'],
                      'decision_loss_share': DECISION_MASS * decided['ce'] / max(1e-9, total_nll)}
                     if decided['decisions'] else {}),
                  **({'decision_reasoning_tokens': decided['reasoning_tokens'],
                      'decision_reasoning_nll': decided['reasoning_nll'] / decided['reasoning_tokens']}
                     if decided['reasoning_tokens'] else {}),
                  'sequences': len(group), 'loss_tokens': denominator, 'elapsed_s': round(time.time() - started, 1),
                  'chain_tokens': int(sum(len(chains[i]['ids']) for i in group)), 'computed_tokens': computed,
                  'forward_tokens_per_s': round(forward_tokens / max(1e-6, time.time() - started)),
                  'step_seconds': round(step_s, 1),
                  'phase_seconds': {k: round(v, 1) for k, v in phase.items()},
                  'cuda_peak_gib': round(max([torch.cuda.max_memory_allocated() / 2**30, *peak_by_phase.values()]), 1),
                  'cuda_peak_by_phase_gb': dict(peak_by_phase),
                  'oom_plan': {'pack_tokens': oom_plan.pack_tokens, 'activation_gib': oom_plan.activation_gib},
                  'oom_retries': oom_plan.retries,
                  'compile_health': compile_health.record_step(step_s, is_warmup=(step == start_step))}
        if args.state_save_minutes > 0 and (time.time() - last_state_save) >= args.state_save_minutes * 60:
            save_state_now(completed_step)
            record['saved_state_step'] = completed_step
        if val_sets and (completed_step % args.val_every == 0 or completed_step == len(steps)):
            model.eval()
            record.update({f'val_loss:{name}': validation_loss(model, forward, c, oom_plan.pack_tokens, student_top_p) for name, c in val_sets.items()})
            record.update(decision_validation(model, forward, val_sets, oom_plan.pack_tokens, oom_plan.activation_gib))
            model.train()
            if args.keep_best_val:
                current = record[f'val_loss:{args.keep_best_val}']
                if best is None or current < best['loss']:
                    best = {'loss': current, 'step': completed_step,
                            'leaves': {name: t.detach().cpu().clone() for name, t in learner.named_trainable_tensors().items()}}
                    if args.save_best:
                        exported = export_merged_checkpoint(learner, state, out_dir / 'checkpoint.pt',
                                                            source_sha256=source_sha, learner_step=completed_step)
                        record['saved_best'] = {'step': completed_step, 'val_loss': current,
                                                'checkpoint_sha256': exported['checkpoint_sha256']}
                record['best_val'] = {'loss': best['loss'], 'step': best['step']}
        if args.hours and completed_step == args.replan_step and completed_step < len(steps):
            rate = forward_tokens / max(1e-6, time.time() - started)
            left = args.hours * 3600 - (time.time() - run_started) - args.reserve_minutes * 60
            mean = np.mean([sum(len(chains[i]['ids']) for i in g) for g in steps[completed_step:]])
            fit = completed_step + max(0, int(left * rate / mean))
            if fit < len(steps):
                del steps[fit:]
                total = fit
                replanned = True
            record['replanned_steps'] = total
        print(json.dumps(record), flush=True)
        log.write(json.dumps(record) + '\n')
        log.flush()
        step += 1
    save_state_now(completed_step)
    if stopped_round(stopped_early, round_id):
        return
    if args.max_steps and not args.checkpoint_at_max_steps:
        return
    out_path = out_dir / 'checkpoint.pt'
    export_step = completed_step
    if best is not None:
        with torch.no_grad():
            for name, leaf in learner.named_trainable_tensors().items():
                leaf.copy_(best['leaves'][name].to(device=leaf.device, dtype=leaf.dtype))
        export_step = best['step']
        print(json.dumps({'kept_best_val': {'loss': best['loss'], 'step': best['step']}}), flush=True)

    def factors():
        from natlang_neuralese.train.learner_state import export_adapter_factors
        info = export_adapter_factors(learner, args.export_factors_dir, base_checkpoint_sha256=source_sha,
                                      learner_step=export_step, round_id=round_id,
                                      source_checkpoint_id=str(args.checkpoint), objective_version='sft_ambient_v1')
        print(json.dumps({'adapter_factors': info}), flush=True)

    def checkpoint():
        exported = export_merged_checkpoint(learner, state, out_path, source_sha256=source_sha, learner_step=export_step)
        print(json.dumps({'checkpoint': str(out_path), **exported}), flush=True)

    export_round(factors=factors if args.export_factors_dir else None, checkpoint=checkpoint,
                 factors_only=args.factors_only, round_id=None if stopped_early else round_id,
                 mark_trained=lambda: save_state_now(completed_step, mark_round_done=True))


def stopped_round(stopped_early, round_id):
    """A round stopped via --stop-file keeps its resumable state and exports nothing: no partial round, no factors
    labelled with its round id (with --keep-best-val too: its best step is of an unfinished round), no round marked done.
    The resumed run finishes the round and exports it."""
    return bool(stopped_early and round_id)


def export_round(*, factors, checkpoint, factors_only, round_id, mark_trained):
    """The end of a run after its final learner state: the factor export (``factors``, an async round's, shipped and
    merged on the DGX), the merged checkpoint unless ``factors_only``, and only then the round in the learner's ledger
    (``mark_trained``): a preemption during an export leaves the round unmarked, so the resumed run (cursor at the
    end) exports again instead of skipping a round that has no export."""
    if factors is not None:
        factors()
    if factors is None or not factors_only:
        checkpoint()
    if round_id:
        mark_trained()
        print(json.dumps({'event': 'round trained', 'round_id': round_id}), flush=True)


def bench(args):
    """Eager vs compiled fast forward+backward on the shared LoRA+rows learner core: per-chain loss,
    trainable-leaf gradient agreement and tokens/s."""
    from natlang_neuralese.train.lfm3_learner import AdapterSpec, Lfm3Learner
    episodes = pickle.load(open(args.turns, 'rb'))['episodes']
    chains = [c for c in chains_of(episodes) if args.min_tokens <= len(c['ids']) <= args.max_tokens][:args.chains]
    spec = AdapterSpec.for_sft()

    def run_variant(compile_layers, label):
        learner = Lfm3Learner.from_checkpoint(
            args.checkpoint, args.base, spec, device='cuda',
            fp8_base=False if getattr(args, 'no_fp8_base', False) else None, compile_layers=compile_layers,
        )
        model = learner.model
        params = learner.trainable_parameters()
        grads = [torch.zeros_like(p, dtype=torch.float32) for p in params]
        losses, seconds, tokens = [], 0., 0
        for repeat in range(args.repeats + 1):
            if repeat == 1:
                seconds, tokens = 0., 0
            for chain in chains:
                ids, positions, targets = chain_tensors(chain, 'cuda')
                set_checkpointing(model, len(ids), args.activation_gb, parameterization='lora')
                torch.cuda.synchronize()
                started = time.time()
                h = lfm3_fast_train.hidden(model, ids, adapters=learner.bundles,
                                           trainable_rows=learner.trainable_rows, fp8_base=learner.fp8_base)
                nll = token_nll(model, h, positions, targets, trainable_rows=learner.trainable_rows)
                (nll.sum() / len(targets)).backward()
                torch.cuda.synchronize()
                seconds += time.time() - started
                tokens += len(ids)
                if repeat == 0:
                    losses.append(nll.mean().item())
                    for g, p in zip(grads, params):
                        if p.grad is not None:
                            g.add_(p.grad.float())
                for p in params:
                    p.grad = None
        print(json.dumps({'impl': label, 'chain_tokens': [len(c['ids']) for c in chains], 'loss': [round(x, 5) for x in losses],
                          'fwd_bwd_tokens_per_s': round(tokens / max(seconds, 1e-6)),
                          'cuda_peak_gib': round(torch.cuda.max_memory_allocated() / 2**30, 1)}), flush=True)
        return losses, grads

    eager_losses, eager_grads = run_variant(False, 'lora-eager')
    fast_losses, fast_grads = run_variant(True, 'lora-compiled')

    def agreement(x, y):
        dot = sum((a * b).sum().item() for a, b in zip(x, y))
        na = math.sqrt(sum((a * a).sum().item() for a in x))
        nb = math.sqrt(sum((b * b).sum().item() for b in y))
        diff = math.sqrt(sum(((a - b) ** 2).sum().item() for a, b in zip(x, y)))
        return {'cosine': dot / max(na * nb, 1e-12), 'rel_l2': diff / max(na, 1e-12), 'norm_a': na, 'norm_b': nb}

    print(json.dumps({'eager_vs_compiled': agreement(eager_grads, fast_grads)}), flush=True)
    print(json.dumps({'loss_abs_diff_max': max(abs(a - b) for a, b in zip(eager_losses, fast_losses)),
                      'loss_mean_eager': float(np.mean(eager_losses)), 'loss_mean_compiled': float(np.mean(fast_losses))}),
          flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('parity', 'train', 'bench'):
        p = sub.add_parser(name)
        p.add_argument('--turns', required=name != 'train')
        p.add_argument('--checkpoint', required=True)
        p.add_argument('--base', required=True)
    p = sub.choices['parity']
    p.add_argument('--samples', type=int, default=6)
    p.add_argument('--max-tokens', type=int, default=14000)
    p.add_argument('--bf16-params', action='store_true')
    p = sub.choices['bench']
    p.add_argument('--chains', type=int, default=6)
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--min-tokens', type=int, default=9000)
    p.add_argument('--max-tokens', type=int, default=12500)
    p.add_argument('--activation-gb', type=float, default=20.)
    p.add_argument('--fp32-reference', action='store_true')
    p = sub.choices['train']
    p.add_argument('--activation-gb', '--activation-gib', dest='activation_gb', type=float, default=56.,
                   help='saved-activation budget (GiB) per sequence; layers beyond it are recomputed (G4: 80, GB10: 56)')
    p.add_argument('--pack-tokens', type=int, default=-1, help='pack a step\'s chains back to back into rows of at most this many tokens; default: the longest row --activation-gib holds without recompute; 0: one chain per forward')
    p.add_argument('--lora-rank', type=int, default=128, help='LoRA rank on all 9 LFM3 projection families (default 128)')
    p.add_argument('--lora-alpha', type=float, default=None, help='LoRA alpha (default: equal to --lora-rank)')
    p.add_argument('--gate-rank', type=int, default=None, help='optional attention gate_proj rank (<= 16; default frozen)')
    p.add_argument('--no-trainable-rows', action='store_true', help='freeze Neuralese marker/override embedding rows (default trainable)')
    p.add_argument('--marker-init', choices=('text-v1', 'none'), default='text-v1',
                   help="role-marker rows still equal to the base's reserved row start from text (lfm3_learner.MARKER_PHRASES); "
                        "'none' keeps them")
    p.add_argument('--row-lr-scale', type=float, default=10.0,
                   help="the trainable Neuralese rows' LR as a multiple of --lr (their own optimizer group; 1 = one group's LR)")
    p.add_argument('--no-fp8-base', action='store_true', help='disable Blackwell FP8 frozen-base GEMMs (BF16 base opt-out)')
    p.add_argument('--no-compile', action='store_true', help='disable torch.compile on fast layers')
    p.add_argument('--resume', action='store_true', help='resume from --state-dir (default OUT_DIR/learner-state)')
    p.add_argument('--state-dir', help='resumable learner state directory (default OUT_DIR/learner-state)')
    p.add_argument('--state-save-minutes', type=float, default=15.,
                   help='save resumable learner state atomically every N minutes (default 15 for Colab preemption safety; 0: end only)')
    p.add_argument('--round-id', help='async-train round identifier recorded in learner_state dedup ledger')
    p.add_argument('--export-factors-dir', help='write FP32 LoRA factors + trainable rows + policy_manifest.json (WAN shipping)')
    p.add_argument('--factors-only', action='store_true', help='with --export-factors-dir: skip local 3.4 GB merged checkpoint.pt write (DGX merges)')
    p.add_argument('--max-steps', type=int, default=0, help='stop after this many optimizer steps without writing a checkpoint')
    p.add_argument('--share-prefix-tokens', type=int, default=256,
                   help='compute a prefix that several of a step\'s chains share (system prompt, task prompt, earlier calls) '
                   'once when it is at least this long (exact gradients, lfm3_fast.prefix_tree); 0: every chain whole')
    add_prefix_tree_args(p)
    p.add_argument('--no-neuralese-passthrough', action='store_true',
                   help='imported Neuralese state as fixed context: the same in-graph state from the producer document, '
                   'detached (default: the consumer\'s loss reaches the producer\'s state, NEURALESE_PASSTHROUGH_20261117)')
    p.add_argument('--batch-order', choices=('auto', 'stratified', 'grouped', 'random'), default='auto',
                   help='stratified: every step carries about the whole set\'s source mix (own, teacher, graph, operator '
                   'rows; exp0_sft.stratified_epoch), episodes kept together; grouped: steps from neighbouring episodes '
                   'in token order (shared prefixes in one step), step order shuffled; random: chains shuffled; auto '
                   '(default): stratified')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--lr', type=float, default=DEFAULT_SFT_LR)
    p.add_argument('--weight-decay', type=float, default=0.)
    p.add_argument('--warmup', type=int, default=8)
    p.add_argument('--min-lr-fraction', type=float, default=0.1)
    p.add_argument('--clip', type=float, default=1.0)
    p.add_argument('--batch-loss-tokens', type=int, default=30000)
    p.add_argument('--seed', type=int, default=20261111)
    p.add_argument('--max-per-task', type=int, default=0)
    p.add_argument('--chains', help='prebuilt sequences pickle ({"chains": [...]}) instead of --turns')
    p.add_argument('--open-goal-mix-check', choices=('on', 'off'), default='on',
                   help='enforce the chain bundle minimum on the post-mask optimizer schedule')
    p.add_argument('--val-chains', action='append', default=[],
                   help='prebuilt validation sequences (repeatable, named by file stem): weighted NLL every --val-every steps')
    p.add_argument('--val-every', type=int, default=50)
    p.add_argument('--checkpoint-at-max-steps', action='store_true',
                   help='with --max-steps: write checkpoint.pt after the last step (short ablations; smokes write none)')
    p.add_argument('--student-top-p', type=float, default=0.95,
                   help='the student\'s own sampled chains (own successes, own Neuralese graph rows; exp0_sft.student_sampled) '
                        'train the NLL under each row\'s top-p nucleus, the law they were sampled with (default 0.95, the '
                        'actor\'s; exp0_sft.targets_nll); 1: the full-softmax NLL for every chain')
    p.add_argument('--second-close', choices=('context', 'train'), default='context',
                   help="with --context-masks on: a reply's second and later </think> and the tokens after each up to the action "
                        'are context (default; exp0_sft.second_close_mask); train: they train like the rest (before 2026-11-19)')
    p.add_argument('--no-decision-objective', action='store_true',
                   help='train decision readout rows (a chain with its option set, sft_sources build) as the plain token '
                   'CE of the chosen reply instead of the option-normalized log score of the readout (default: the '
                   'decision objective, exp0_sft.decision_backward)')
    p.add_argument('--no-decision-reasoning', action='store_true',
                   help='with the decision objective: do not train a reasoned readout\'s sampled reasoning as token CE '
                   '(its options are still scored after it; default: the reasoning span trains, '
                   'exp0_sft.mark_decision_objective)')
    p.add_argument('--context-masks', choices=('on', 'off'), default='on',
                   help='on (default): at load, a target row\'s leading empty think block and anything after its end '
                        'token are context in the training and validation chains (exp0_sft.context_masks); off: the '
                        'chains\' own weights')
    p.add_argument('--save-best', action='store_true', help='with --keep-best-val: write checkpoint.pt at every new best validation step')
    p.add_argument('--stop-file', help='stop before the next optimizer step when this file exists (the best-validation checkpoint is written)')
    p.add_argument('--keep-best-val', help='name (file stem) of a --val-chains set: write the checkpoint of the step with its lowest NLL (step 0 counts)')
    p.add_argument('--ambient-p-clean', type=float, default=AMBIENT_P_CLEAN,
                   help='Ambient SFT (arXiv 2611.23229, exp0_sft.ambient_clean): each optimizer step trains the plain SFT '
                        'loss with this probability (default 0.75, the paper\'s), else Temperature Matching to the frozen '
                        'start checkpoint at --ambient-temperature (one extra no-grad forward of the step, 3.4 GB of '
                        'resident teacher weights); 1 opts out (plain SFT)')
    p.add_argument('--ambient-temperature', type=float, default=AMBIENT_TEMPERATURE,
                   help='Temperature Matching temperature T (default 2.5, the paper\'s; large T approaches SFT, 1 is KD)')
    p.add_argument('--ambient-mode', choices=('additive', 'replace'), default=AMBIENT_MODE,
                   help="additive (default): non-clean steps train SFT + (1/p_clean)*TM jointly so 100%% of unique "
                        "1-epoch batches receive full SFT gradient under exact Eq. 4 balance; replace: legacy Algorithm 1 "
                        "coin flip (replaces SFT with TM-only on 1-p_clean of steps)")
    p.add_argument('--hours', type=float, default=0., help='wall-clock budget: at --replan-step the schedule (cosine '
                   'included) is cut to the steps that fit at the measured rate, keeping --reserve-minutes for eval and save')
    p.add_argument('--replan-step', type=int, default=20)
    p.add_argument('--reserve-minutes', type=float, default=25.)
    args = parser.parse_args()
    LAW_BASE['path'] = getattr(args, 'base', None)
    from natlang_neuralese.common.cuda_context import ensure_context
    ensure_context()   # GB10: a context can fail to create under host-memory fragmentation; retried
    {'parity': parity, 'train': train, 'bench': bench}[args.command](args)


if __name__ == '__main__':
    main()
