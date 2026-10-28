"""Numerically stable softmax, cross-entropy and divergence gradients (one shared module).

Barron, Gnanapragasam & Tschannen 2026 (https://jonbarron.info/softmaxfix/; plan:
plans/neuralese/TRAINING_INSIGHTS_20261028.md section 8). Two failure modes of the
standard formulation are repaired:

* softmax collapse (hard targets): once the target leads by a margin of ~5.5 (BF16) /
  16.6 (FP32) logits its gradient ``q_t - 1`` rounds to zero while the non-target
  gradients stay positive. Here every log-probability is formed from logits shifted
  so the maximum is exactly zero plus ``softplus(-m)`` with the margin
  ``m = -LSE(other shifted logits)``; the gradient at a selected id uses
  ``1 - q = -expm1(log q)``, exact down to exponent underflow.
* softmax cancellation (soft targets near agreement): ``q - p`` loses all precision
  when student and teacher nearly agree. The forward-KL/soft-CE, reverse-KL and
  generalized-JSD gradients are formed from logit differences centred under the
  appropriate distribution, so equal logits give an exactly zero gradient.

Forward values equal the standard definitions (the KL/JSD values are evaluated in
the same cancellation-free form, so they are closer to the exact value, not
different definitions). Gradients are the fixed ones.

Memory: inputs of any shape ``(..., V)`` are flattened to rows and processed in row
chunks (``chunk_elements`` elements of the compute dtype, default FP32 for 16-bit
inputs, never a whole-tensor upcast). Backward passes recompute per chunk from the
saved input logits and per-row statistics, so no vocabulary-sized FP32 tensor
outlives a chunk. The teacher side of every divergence is a constant (no gradient
flows to it); pass a detached tensor.

Policy: ``POLICY`` is the default for every trainer, RL loss and distillation path.
``LEGACY_POLICY`` reproduces the previous plain PyTorch formulation and exists only
for replays/re-verification of completed work and as the explicit opt-out
(``NATLANG_SOFTMAX_NUMERICS=torch-native-v0`` or ``use_numerics_policy``). Trainers
record ``numerics_record()`` in their checkpoint/recipe metadata.
"""
from __future__ import annotations

import contextlib
import contextvars
import math
import os

import torch
import torch.nn.functional as F

POLICY = 'stable-softmax-barron2026-v1'
LEGACY_POLICY = 'torch-native-v0'
POLICIES = (POLICY, LEGACY_POLICY)
ENV = 'NATLANG_SOFTMAX_NUMERICS'
DEFAULT_CHUNK_ELEMENTS = 1 << 23
REFERENCE = 'https://jonbarron.info/softmaxfix/'

_GLOBAL = [None]
_OVERRIDE: contextvars.ContextVar = contextvars.ContextVar('natlang_softmax_numerics', default=None)


def _check(policy):
    if policy not in POLICIES:
        raise ValueError(f'unknown softmax numerics policy {policy!r}; expected one of {POLICIES}')
    return policy


def active_policy() -> str:
    """The numerics policy in effect: context override, then process setting, then env, then default."""
    policy = _OVERRIDE.get() or _GLOBAL[0] or os.environ.get(ENV) or POLICY
    return _check(policy)


def set_numerics_policy(policy: str | None) -> None:
    """Set the process-wide policy (``None`` restores the environment/default)."""
    _GLOBAL[0] = None if policy is None else _check(policy)


@contextlib.contextmanager
def use_numerics_policy(policy: str | None):
    """Scope a policy (e.g. a replay of completed work recorded under ``LEGACY_POLICY``)."""
    token = _OVERRIDE.set(None if policy is None else _check(policy))
    try:
        yield
    finally:
        _OVERRIDE.reset(token)


def numerics_record(policy: str | None = None) -> dict:
    """The identity trainers record in checkpoints/recipes."""
    return {'softmax_numerics': _check(policy or active_policy()), 'reference': REFERENCE}


def numerics_transition(previous: str | None, current: str | None = None) -> dict | None:
    """A recorded transition when a continuation adopts the current policy.

    ``previous`` is the parent's recorded policy (``None`` = recorded before this module,
    i.e. ``LEGACY_POLICY``). Returns ``None`` when nothing changes.
    """
    before = _check(previous or LEGACY_POLICY)
    after = _check(current or active_policy())
    if before == after:
        return None
    return {'kind': 'softmax-numerics-transition-v1', 'from': before, 'to': after, 'reference': REFERENCE}


def _stable(policy):
    return _check(policy or active_policy()) == POLICY


def _compute_dtype(dtype, compute_dtype):
    if compute_dtype is not None:
        return compute_dtype
    return torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype


def _rows(n_cols, chunk_elements):
    return max(1, int(chunk_elements or DEFAULT_CHUNK_ELEMENTS) // max(1, n_cols))


def _shifted(z):
    """(shifted logits with an exact-zero max, softplus(-margin), max, argmax) per row.

    ``log q = shifted - sp``; the argmax entry is exactly ``-sp`` with
    ``sp = softplus(LSE(other shifted logits))``. Functional (no in-place ops), so
    a create_graph backward can differentiate through it.
    """
    zmax, imax = z.detach().max(-1, keepdim=True)
    finite = torch.isfinite(zmax)
    safe_max = torch.where(finite, zmax, torch.zeros_like(zmax))
    # Exact zero at the argmax: x - x == 0 for finite x (also through autograd).
    shifted = z - z.gather(-1, imax).where(finite, safe_max)
    sp = F.softplus(torch.logsumexp(shifted.scatter(-1, imax, float('-inf')), -1, keepdim=True))
    if not bool(finite.all()):
        # An all -inf row has no distribution; keep it -inf (standard gives NaN).
        shifted = torch.where(finite, shifted, torch.full_like(shifted, float('-inf')))
    return shifted, sp, zmax, imax


def stable_log_softmax_rows(z):
    """Log-softmax over the last dim with an exact-zero max (no autograd customization)."""
    shifted, sp, _, _ = _shifted(z)
    return shifted - sp


# ---------------------------------------------------------------------------- log_softmax / softmax


def log_softmax(logits, dim=-1, *, policy=None, compute_dtype=None):
    """Full log-softmax. Autograd follows the shifted/softplus-margin formulation.

    Returns ``compute_dtype`` (FP32 for 16-bit inputs). For training losses over
    a large vocabulary prefer the selected-id/divergence functions below, which
    never materialize the full gradient graph.
    """
    if not _stable(policy):
        return torch.log_softmax(logits.to(_compute_dtype(logits.dtype, compute_dtype)), dim)
    z = logits.to(_compute_dtype(logits.dtype, compute_dtype)).movedim(dim, -1)
    zmax, imax = z.detach().max(-1, keepdim=True)
    is_max = torch.zeros_like(z, dtype=torch.bool).scatter_(-1, imax, True)
    z_at_max = z.gather(-1, imax)
    shifted = torch.where(is_max, torch.zeros_like(z), z - z_at_max)
    others = shifted.masked_fill(is_max, float('-inf'))
    has_other = torch.isfinite(others).any(-1, keepdim=True)
    safe_others = torch.where(has_other, others, torch.zeros_like(others))
    sp = torch.where(has_other, F.softplus(torch.logsumexp(safe_others, -1, keepdim=True)), torch.zeros_like(z_at_max))
    return (shifted - sp).movedim(-1, dim)


def softmax(logits, dim=-1, *, policy=None, compute_dtype=None):
    return log_softmax(logits, dim, policy=policy, compute_dtype=compute_dtype).exp()


# ---------------------------------------------------------------------------- selected ids


class _SelectedLogSoftmax(torch.autograd.Function):
    """log q[ids] with the zero-sum gradient; ids (N, k) may repeat.

    The first-order path works in place on one owned compute-dtype copy per chunk:
    ``sp = log1p(sum_{j != argmax} exp(z_j - z_max))`` (accurate to exp underflow).
    A create_graph backward recomputes through the functional ``_shifted``.
    """

    @staticmethod
    def forward(ctx, z, ids, cdt, rows):
        n = z.shape[0]
        out = torch.empty(ids.shape, dtype=cdt, device=z.device)
        stats = torch.empty((n, 1), dtype=cdt, device=z.device)
        maxima = torch.empty((n, 1), dtype=cdt, device=z.device)
        for start in range(0, n, rows):
            stop = start + rows
            chunk = z[start:stop].to(cdt, copy=True)
            zmax, imax = chunk.max(-1, keepdim=True)
            safe_max = torch.where(torch.isfinite(zmax), zmax, torch.zeros_like(zmax))
            chunk.sub_(safe_max)
            picked = chunk.gather(-1, ids[start:stop])
            chunk.scatter_(-1, imax, float('-inf'))
            sp = torch.log1p(chunk.exp_().sum(-1, keepdim=True))
            out[start:stop] = picked - sp
            stats[start:stop], maxima[start:stop] = sp, safe_max
            del chunk
        ctx.save_for_backward(z, ids, stats, maxima)
        ctx.rows = rows
        return out

    @staticmethod
    def backward(ctx, grad):
        z, ids, stats, maxima = ctx.saved_tensors
        result = _Rows(z)
        grad = grad.to(stats.dtype)
        differentiable = torch.is_grad_enabled()  # create_graph: recompute row statistics through autograd
        k = ids.shape[-1]
        for start in range(0, z.shape[0], ctx.rows):
            stop = start + ctx.rows
            w, index = grad[start:stop], ids[start:stop]
            total = w.sum(-1, keepdim=True)
            if differentiable:
                shifted, sp, _, _ = _shifted(z[start:stop].to(stats.dtype))
                logq = shifted - sp
                weight = torch.zeros_like(logq).scatter_add(-1, index, w)
                # w - q W, with 1 - q = -expm1(log q) exact at a dominant id.
                g = weight * (-torch.expm1(logq)) - logq.exp() * (total - weight)
                result.add(g.to(z.dtype))
                continue
            chunk = z[start:stop].to(stats.dtype, copy=True)
            chunk.sub_(maxima[start:stop])
            # Selected entries from the exact-zero shift (precise at the argmax).
            logq_ids = chunk.gather(-1, index) - stats[start:stop]
            chunk.sub_(stats[start:stop]).exp_().mul_(-total)  # -q W everywhere
            mine = w if k == 1 else ((index[:, :, None] == index[:, None, :]) * w[:, None, :]).sum(-1)
            precise = mine * (-torch.expm1(logq_ids)) - logq_ids.exp() * (total - mine)
            chunk.scatter_(-1, index, precise)
            result.add(chunk.to(z.dtype))
            del chunk
        return result.result(), None, None, None


class _Rows:
    """Backward output assembly: in place normally, functional under create_graph."""

    def __init__(self, like):
        self.functional = torch.is_grad_enabled()
        self.parts = [] if self.functional else None
        self.out = None if self.functional else torch.empty_like(like)
        self.at = 0

    def add(self, value):
        if self.functional:
            self.parts.append(value)
        else:
            self.out[self.at:self.at + value.shape[0]] = value
        self.at += value.shape[0]

    def result(self):
        return torch.cat(self.parts) if self.functional else self.out


def _flatten(logits):
    return logits.reshape(-1, logits.shape[-1])


def selected_log_softmax(logits, ids, *, policy=None, compute_dtype=None, chunk_elements=None):
    """``log_softmax(logits)[..., ids]`` for ids of shape ``logits.shape[:-1] + (k,)``.

    The building block for token log-probabilities (RL policy terms, score
    centering head sets, top-k teacher alignment) and hard-target CE.
    """
    cdt = _compute_dtype(logits.dtype, compute_dtype)
    ids = ids.long()
    if not _stable(policy):
        return torch.log_softmax(logits.to(cdt), -1).gather(-1, ids)
    lead = logits.shape[:-1]
    if ids.shape[:-1] != lead:
        raise ValueError(f'ids shape {tuple(ids.shape)} does not match logits rows {tuple(lead)}')
    z = _flatten(logits)
    flat_ids = ids.reshape(z.shape[0], -1)
    out = _SelectedLogSoftmax.apply(z, flat_ids, cdt, _rows(z.shape[-1], chunk_elements))
    return out.reshape(ids.shape)


def token_logprobs(logits, labels, *, policy=None, compute_dtype=None, chunk_elements=None):
    """``log_softmax(logits)[..., label]`` for labels of shape ``logits.shape[:-1]``."""
    return selected_log_softmax(logits, labels.long().unsqueeze(-1), policy=policy,
                                compute_dtype=compute_dtype, chunk_elements=chunk_elements).squeeze(-1)


def _reduce(values, weights, reduction):
    if weights is not None:
        values = values * weights.to(values.dtype)
    if reduction == 'none':
        return values
    if reduction == 'sum':
        return values.sum()
    if reduction == 'mean':
        return values.mean()
    raise ValueError(f'unknown reduction {reduction!r}')


def cross_entropy(logits, labels, *, weights=None, ignore_index=-100, reduction='none', policy=None,
                  compute_dtype=None, chunk_elements=None):
    """Hard-target CE per position (``softplus(-margin)``) with the exact zero-sum gradient.

    ``reduction='none'`` returns per-position losses (0 at ``ignore_index``);
    ``'mean'`` averages over non-ignored positions like ``F.cross_entropy``;
    ``'sum'`` sums. ``weights`` multiply per position before reduction.
    """
    labels = labels.long()
    keep = labels != ignore_index
    safe = torch.where(keep, labels, torch.zeros_like(labels))
    if not _stable(policy):
        cdt = _compute_dtype(logits.dtype, compute_dtype)
        values = F.cross_entropy(_flatten(logits.to(cdt)), safe.reshape(-1), reduction='none').reshape(labels.shape)
    else:
        values = -token_logprobs(logits, safe, policy=POLICY, compute_dtype=compute_dtype,
                                 chunk_elements=chunk_elements)
    values = values * keep.to(values.dtype)
    if reduction == 'mean' and weights is None:
        return values.sum() / keep.sum().clamp_min(1).to(values.dtype)
    return _reduce(values, weights, reduction)


# ---------------------------------------------------------------------------- divergences

_KINDS = ('forward_kl', 'soft_ce', 'reverse_kl', 'jsd')


def _divergence_chunk(zs, zt, kind, beta, need_grad, need_value=None):
    """Per-row value and (optionally) the fixed gradient w.r.t. student logits.

    zs, zt: (n, V) in the compute dtype; zt may be -inf outside the teacher support.
    """
    logq = stable_log_softmax_rows(zs)
    logp = stable_log_softmax_rows(zt)
    q, p = logq.exp(), logp.exp()
    support = torch.isfinite(logp)
    student = torch.isfinite(logq)
    zero = torch.zeros((), dtype=zs.dtype, device=zs.device)
    both = support & student
    delta = torch.where(both, zs - zt, zero)
    value = grad = None
    need_value = (not need_grad) if need_value is None else need_value
    need_r = kind != 'reverse_kl' or need_value
    if kind in ('forward_kl', 'soft_ce', 'jsd') or need_r:
        # r = log(q / p) on the support, centred under p (cancellation-free).
        mu = (p * delta).sum(-1, keepdim=True)
        d = torch.where(both, delta - mu, zero)
        em1 = (p * torch.where(both, torch.expm1(d), zero)).sum(-1, keepdim=True)
        # Student mass outside the teacher support, relative to the teacher normalizer.
        lse_t = torch.logsumexp(zt, -1, keepdim=True)
        off = torch.where(support, torch.full_like(zs, float('-inf')), zs - lse_t - mu)
        log_tail = torch.logsumexp(off, -1, keepdim=True)
        s = em1 + log_tail.exp()
        exact = torch.log1p(s)
        fallback = torch.logaddexp(torch.logsumexp(torch.where(both, logp + d, torch.full_like(d, float('-inf'))),
                                                   -1, keepdim=True), log_tail)
        c = torch.where(torch.isfinite(s) & (s < 1e30), exact, fallback)
        # Teacher mass where the student has none: KL is infinite there.
        r = torch.where(both, d - c, torch.where(support, torch.full_like(d, float('-inf')),
                                                 torch.full_like(d, float('inf'))))
    if kind in ('forward_kl', 'soft_ce'):
        if not need_value:
            pass
        elif kind == 'forward_kl':
            value = c - (p * d).sum(-1, keepdim=True)
            value = torch.where((support & ~student).any(-1, keepdim=True), torch.full_like(value, float('inf')), value)
        else:
            value = -(p * torch.where(support, logq, zero)).sum(-1, keepdim=True)
            value = torch.where((support & ~student).any(-1, keepdim=True), torch.full_like(value, float('inf')), value)
        if need_grad:
            u = torch.where(support, torch.where(r.abs() < 1, p * torch.expm1(r), q - p), q)
            grad = u - q * u.sum(-1, keepdim=True)
    elif kind == 'reverse_kl':
        if need_grad:
            mean = (q * delta).sum(-1, keepdim=True)
            grad = q * torch.where(both, delta - mean, zero)
        if need_r:
            value = (q * torch.where(student, r, zero)).sum(-1, keepdim=True)
    else:  # jsd: beta KL(p||m) + (1-beta) KL(q||m), m = beta p + (1-beta) q
        h = _jsd_h(r, beta)  # log(q / m); -log(1-beta) off the teacher support
        h = torch.where(student, h, zero)
        hp = torch.where(both, h - r, zero)  # log(p / m) where both exist
        hp = torch.where(support & ~student, torch.full_like(hp, -math.log(beta)), hp)
        if need_value:
            value = beta * (p * hp).sum(-1, keepdim=True) + (1 - beta) * (q * h).sum(-1, keepdim=True)
        if need_grad:
            grad = (1 - beta) * q * (h - (q * h).sum(-1, keepdim=True))
    return (None if value is None else value[:, 0]), grad


def _rowdot(a, b):
    return torch.linalg.vecdot(a, b).unsqueeze(-1)


def _log_normalizer(p, d, zt):
    """c = log E_p[exp d] = log1p(E_p[expm1 d]) for d centred under p; log-domain fallback far from agreement."""
    c = torch.log1p(_rowdot(p, torch.expm1(d)))
    if not bool(torch.isfinite(c).all()):
        fallback = torch.logsumexp(torch.log_softmax(zt, -1) + d, -1, keepdim=True)
        c = torch.where(torch.isfinite(c), c, fallback)
    return c


def _jsd_h(r, beta):
    """log(q / m) = -log1p(beta expm1(-r)), m = beta p + (1-beta) q; log-domain where q << p."""
    h = -torch.log1p(beta * torch.expm1(-r))
    far = -torch.logaddexp(math.log(beta) - r, torch.full_like(r, math.log1p(-beta)))
    return torch.where(r < -1, far, h)


def _divergence_fast(zs, zt, kind, beta, need_grad, need_value):
    """The same law as ``_divergence_chunk`` for finite student/teacher logits, first order only.

    Fused log-softmax/softmax kernels and in-place updates on owned buffers; no
    masking passes. Never modifies ``zs``/``zt``.
    """
    value = grad = None
    p = torch.softmax(zt, -1)
    d = zs - zt
    mu = _rowdot(p, d)
    d.sub_(mu)  # d: delta centred under p
    if kind == 'reverse_kl':
        q = torch.softmax(zs, -1)
        if need_value:
            # KL(q||p) = E_q[d] - log1p(E_p[expm1 d]) without cancellation.
            value = _rowdot(q, d) - _log_normalizer(p, d, zt)
        if need_grad:
            mean = _rowdot(q, d)
            grad = d.sub_(mean).mul_(q)
        return (None if value is None else value[:, 0]), grad
    c = _log_normalizer(p, d, zt)
    if need_value and kind == 'forward_kl':
        value = c - _rowdot(p, d)
    elif need_value and kind == 'soft_ce':
        value = -_rowdot(p, torch.log_softmax(zs, -1))
    r = d.sub_(c)  # log(q / p)
    q = torch.softmax(zs, -1) if need_grad or kind == 'jsd' else None
    if kind == 'jsd':
        h = _jsd_h(r, beta)  # log(q / m)
        if need_value:
            value = beta * _rowdot(p, h - r) + (1 - beta) * _rowdot(q, h)
        if need_grad:
            grad = h.sub_(_rowdot(q, h)).mul_(q).mul_(1 - beta)
        return (None if value is None else value[:, 0]), grad
    if need_grad:
        far = r.abs() >= 1
        u = torch.expm1(r).mul_(p)
        u = torch.where(far, q - p, u)
        grad = u.addcmul_(q, -u.sum(-1, keepdim=True))
    return (None if value is None else value[:, 0]), grad


def _divergence_rows(zs, zt, kind, beta, need_grad, need_value, fast):
    if fast and not torch.is_grad_enabled():
        return _divergence_fast(zs, zt, kind, beta, need_grad, need_value)
    return _divergence_chunk(zs, zt, kind, beta, need_grad, need_value)


def _finite_logits(*tensors):
    """True when no -inf masks are present (one pass each, once per call)."""
    return not any(bool(torch.isinf(t).any()) for t in tensors)


class _Divergence(torch.autograd.Function):
    @staticmethod
    def forward(ctx, zs, zt, kind, beta, cdt, rows):
        n = zs.shape[0]
        fast = _finite_logits(zs, zt)
        out = torch.empty(n, dtype=cdt, device=zs.device)
        for start in range(0, n, rows):
            value, _ = _divergence_rows(zs[start:start + rows].to(cdt), zt[start:start + rows].to(cdt), kind, beta,
                                        False, True, fast)
            out[start:start + rows] = value
        ctx.save_for_backward(zs, zt)
        ctx.kind, ctx.beta, ctx.cdt, ctx.rows, ctx.fast = kind, beta, cdt, rows, fast
        return out

    @staticmethod
    def backward(ctx, grad):
        zs, zt = ctx.saved_tensors
        result = _Rows(zs)
        grad = grad.to(ctx.cdt)
        for start in range(0, zs.shape[0], ctx.rows):
            stop = start + ctx.rows
            _, g = _divergence_rows(zs[start:stop].to(ctx.cdt), zt[start:stop].to(ctx.cdt), ctx.kind, ctx.beta,
                                    True, False, ctx.fast)
            result.add((g * grad[start:stop, None]).to(zs.dtype))
        return result.result(), None, None, None, None, None


def _legacy_divergence(zs, zt, kind, beta):
    logq = torch.log_softmax(zs, -1)
    logp = torch.log_softmax(zt, -1)
    p, q = logp.exp(), logq.exp()
    if kind == 'soft_ce':
        return -(torch.where(p > 0, p * logq, torch.zeros_like(p))).sum(-1)
    if kind == 'forward_kl':
        return torch.where(p > 0, p * (logp - logq), torch.zeros_like(p)).sum(-1)
    if kind == 'reverse_kl':
        return torch.where(q > 0, q * (logq - logp), torch.zeros_like(q)).sum(-1)
    logm = torch.logaddexp(math.log(beta) + logp, math.log1p(-beta) + logq)
    return (beta * torch.where(p > 0, p * (logp - logm), torch.zeros_like(p)).sum(-1)
            + (1 - beta) * torch.where(q > 0, q * (logq - logm), torch.zeros_like(q)).sum(-1))


def _scatter_teacher(teacher_logits, teacher_ids, width):
    full = teacher_logits.new_full(teacher_logits.shape[:-1] + (width,), float('-inf'))
    return full.scatter(-1, teacher_ids.long(), teacher_logits)


def divergence(student_logits, teacher_logits, kind, *, beta=0.5, teacher_ids=None, weights=None,
               reduction='none', policy=None, compute_dtype=None, chunk_elements=None):
    """Per-position divergence between ``q = softmax(student)`` and ``p = softmax(teacher)``.

    kind: ``'forward_kl'`` KL(p||q), ``'soft_ce'`` -sum p log q (same gradient),
    ``'reverse_kl'`` KL(q||p), ``'jsd'`` beta KL(p||m) + (1-beta) KL(q||m) with
    ``m = beta p + (1-beta) q``. Teacher logits may be log-probabilities and may be
    ``-inf`` outside its support (truncated teachers; forward KL / soft CE / JSD).
    ``teacher_ids`` (``(..., k)``) gives a top-k teacher: ``teacher_logits`` are then its
    k logits/log-probs (renormalized over the k), the student is over the full vocab.
    Gradients flow to the student only.
    """
    if kind not in _KINDS:
        raise ValueError(f'unknown divergence {kind!r}; expected one of {_KINDS}')
    if kind == 'jsd' and not 0 < beta < 1:
        raise ValueError('generalized JSD needs 0 < beta < 1')
    if teacher_logits.requires_grad:
        raise ValueError('divergence teacher must be detached (gradients flow to the student only)')
    width = student_logits.shape[-1]
    cdt = _compute_dtype(student_logits.dtype, compute_dtype)
    lead = student_logits.shape[:-1]
    zs = _flatten(student_logits)
    if teacher_ids is not None:
        if teacher_ids.shape != teacher_logits.shape or teacher_ids.shape[:-1] != lead:
            raise ValueError('teacher_ids/teacher_logits must share shape (..., k) with the student rows')
        zt_rows, ids_rows = _flatten(teacher_logits), _flatten(teacher_ids)
    else:
        if teacher_logits.shape != student_logits.shape:
            raise ValueError('teacher and student logits must have the same shape')
        zt_rows, ids_rows = _flatten(teacher_logits), None
    if not _stable(policy):
        zt = zt_rows.to(cdt) if ids_rows is None else _scatter_teacher(zt_rows.to(cdt), ids_rows, width)
        values = _legacy_divergence(zs.to(cdt), zt, kind, beta)
    else:
        rows = _rows(width, chunk_elements)
        if ids_rows is None:
            values = _Divergence.apply(zs, zt_rows, kind, float(beta), cdt, rows)
        else:
            # Scatter the k teacher logits into -inf rows chunk-by-chunk inside the Function.
            values = _TopKDivergence.apply(zs, zt_rows, ids_rows.long(), kind, float(beta), cdt, rows)
    return _reduce(values.reshape(lead), weights, reduction)


class _TopKDivergence(torch.autograd.Function):
    @staticmethod
    def forward(ctx, zs, zt, ids, kind, beta, cdt, rows):
        n, width = zs.shape
        out = torch.empty(n, dtype=cdt, device=zs.device)
        for start in range(0, n, rows):
            teacher = _scatter_teacher(zt[start:start + rows].to(cdt), ids[start:start + rows], width)
            value, _ = _divergence_chunk(zs[start:start + rows].to(cdt), teacher, kind, beta, False)
            out[start:start + rows] = value
        ctx.save_for_backward(zs, zt, ids)
        ctx.kind, ctx.beta, ctx.cdt, ctx.rows = kind, beta, cdt, rows
        return out

    @staticmethod
    def backward(ctx, grad):
        zs, zt, ids = ctx.saved_tensors
        width = zs.shape[-1]
        result = _Rows(zs)
        grad = grad.to(ctx.cdt)
        for start in range(0, zs.shape[0], ctx.rows):
            stop = start + ctx.rows
            teacher = _scatter_teacher(zt[start:stop].to(ctx.cdt), ids[start:stop], width)
            _, g = _divergence_chunk(zs[start:stop].to(ctx.cdt), teacher, ctx.kind, ctx.beta, True)
            result.add((g * grad[start:stop, None]).to(zs.dtype))
        return result.result(), None, None, None, None, None, None


def forward_kl(student_logits, teacher_logits, **kw):
    """KL(teacher || student)."""
    return divergence(student_logits, teacher_logits, 'forward_kl', **kw)


def soft_cross_entropy(student_logits, teacher_logits, **kw):
    """-sum_v p_v log q_v."""
    return divergence(student_logits, teacher_logits, 'soft_ce', **kw)


def reverse_kl(student_logits, teacher_logits, **kw):
    """KL(student || teacher); gradient q (delta - E_q delta). Needs a full-support teacher."""
    return divergence(student_logits, teacher_logits, 'reverse_kl', **kw)


def generalized_jsd(student_logits, teacher_logits, beta=0.5, **kw):
    """beta KL(p||m) + (1-beta) KL(q||m), m = beta p + (1-beta) q."""
    return divergence(student_logits, teacher_logits, 'jsd', beta=beta, **kw)


def topk_forward_kl(student_logits, teacher_ids, teacher_logits, **kw):
    """KL over a teacher's top-k (renormalized over the k) against the full-vocab student."""
    return divergence(student_logits, teacher_logits, 'forward_kl', teacher_ids=teacher_ids, **kw)


# ---------------------------------------------------------------------------- score centering

IMPORTANCE = ('plain', 'tis', 'mis')


def _importance(log_ratio, mode, cap, low, high):
    """f(p/q) and its log; f = 1 (plain), min(r, cap) (TIS), r on [low, high] else 0 (MIS)."""
    if mode == 'plain':
        return torch.zeros_like(log_ratio)
    if mode == 'tis':
        return torch.minimum(log_ratio, torch.full_like(log_ratio, math.log(cap)))
    if mode == 'mis':
        inside = (log_ratio >= math.log(low)) & (log_ratio <= math.log(high))
        return torch.where(inside, log_ratio, torch.full_like(log_ratio, float('-inf')))
    raise ValueError(f'unknown importance mode {mode!r}; expected one of {IMPORTANCE}')


@torch.no_grad()
def score_centering_coefficients(trainer_logits, head_ids, sampler_head_logprobs, *, importance='plain',
                                 tis_cap=2.0, mis_range=(0.5, 5.0), policy=None, compute_dtype=None,
                                 chunk_elements=None):
    """Stable score-centering coefficients ``c_v = q_v w_v - alpha p_v`` over the head set H.

    q = sampler probabilities (logged top-k log-probs ``sampler_head_logprobs`` at
    ``head_ids``), p = trainer probabilities from ``trainer_logits``;
    ``rho = (1 - sum_H q) / (1 - sum_H p)`` uses the exact trainer tail mass
    ``LSE(off-H logits) - LSE(all)`` and ``1 - sum_H q = -expm1(LSE(log q_H))``;
    ``alpha = rho f(1/rho)``, ``w_v = f(p_v / q_v)``. ``c_v`` is formed as
    ``p_v alpha expm1(log w_v + log q_v - log p_v - log alpha)`` so it does not
    cancel near agreement. Returns a dict of detached FP32 tensors:
    ``coefficients`` (..., k), ``trainer_logprobs`` (..., k), ``log_rho`` (...),
    ``alpha`` (...). A vanished tail on both sides gives rho = 1.
    """
    cdt = _compute_dtype(trainer_logits.dtype, compute_dtype)
    lead = trainer_logits.shape[:-1]
    z = _flatten(trainer_logits)
    ids = head_ids.long().reshape(z.shape[0], -1)
    logq_h = sampler_head_logprobs.to(cdt).reshape(z.shape[0], -1)
    stable = _stable(policy)
    rows = _rows(z.shape[-1], chunk_elements)
    logp_h = torch.empty_like(logq_h)
    log_tail_p = torch.empty((z.shape[0],), dtype=cdt, device=z.device)
    for start in range(0, z.shape[0], rows):
        chunk = z[start:start + rows].to(cdt)
        logp = stable_log_softmax_rows(chunk) if stable else torch.log_softmax(chunk, -1)
        logp_h[start:start + rows] = logp.gather(-1, ids[start:start + rows])
        if stable:
            log_tail_p[start:start + rows] = torch.logsumexp(
                logp.scatter(-1, ids[start:start + rows], float('-inf')), -1)
        else:
            log_tail_p[start:start + rows] = torch.log1p(-logp_h[start:start + rows].exp().sum(-1).clamp(max=1.))
    if stable:
        # k head entries: FP64 resolves a tiny sampler tail from its logged log-probs at no real cost.
        sampler_mass = torch.logsumexp(sampler_head_logprobs.double().reshape(logq_h.shape), -1).clamp(max=0.)
        log_tail_q = torch.log(-torch.expm1(sampler_mass)).to(cdt)
    else:
        log_tail_q = torch.log1p(-logq_h.exp().sum(-1).clamp(max=1.))
    both_empty = torch.isneginf(log_tail_p) & torch.isneginf(log_tail_q)
    log_rho = torch.where(both_empty, torch.zeros_like(log_tail_p), log_tail_q - log_tail_p)
    low, high = mis_range
    log_alpha = log_rho + _importance(-log_rho, importance, tis_cap, low, high)
    log_w = _importance(logp_h - logq_h, importance, tis_cap, low, high)
    if stable:
        exponent = log_w + logq_h - logp_h - log_alpha[:, None]
        coefficients = logp_h.exp() * log_alpha.exp()[:, None] * torch.expm1(exponent)
        # alpha = 0: c = q w; w = 0: c = -alpha p (expm1(-inf) = -1 already covers it).
        coefficients = torch.where(torch.isneginf(log_alpha)[:, None], (logq_h + log_w).exp(), coefficients)
    else:
        coefficients = (logq_h + log_w).exp() - log_alpha.exp()[:, None] * logp_h.exp()
    return {'coefficients': coefficients.reshape(head_ids.shape), 'trainer_logprobs': logp_h.reshape(head_ids.shape),
            'log_rho': log_rho.reshape(lead), 'alpha': log_alpha.exp().reshape(lead)}


def score_centered_surrogate(trainer_logits, sampled_ids, sampled_sampler_logprobs, head_ids, sampler_head_logprobs,
                             *, importance='plain', tis_cap=2.0, mis_range=(0.5, 5.0), policy=None,
                             compute_dtype=None, chunk_elements=None):
    """Per-position ``sg[w_y] log p_y - sum_{v in H} sg[c_v] log p_v``.

    The score-centered policy loss is ``-(R * surrogate * mask).sum()`` with R the
    centred advantage. Log-probabilities use the zero-sum selected-id gradient.
    Returns ``(surrogate, info)`` where info holds the detached coefficients.
    """
    info = score_centering_coefficients(trainer_logits.detach(), head_ids, sampler_head_logprobs,
                                        importance=importance, tis_cap=tis_cap, mis_range=mis_range, policy=policy,
                                        compute_dtype=compute_dtype, chunk_elements=chunk_elements)
    ids = torch.cat([sampled_ids.long().unsqueeze(-1), head_ids.long()], -1)
    logp = selected_log_softmax(trainer_logits, ids, policy=policy, compute_dtype=compute_dtype,
                                chunk_elements=chunk_elements)
    logp_y, logp_h = logp[..., 0], logp[..., 1:]
    low, high = mis_range
    w_y = _importance(logp_y.detach() - sampled_sampler_logprobs.to(logp_y.dtype), importance, tis_cap, low, high).exp()
    surrogate = w_y * logp_y - (info['coefficients'].to(logp_h.dtype) * logp_h).sum(-1)
    info['sampled_weight'] = w_y
    return surrogate, info



# ---------------------------------------------------------------------------- fused-kernel helpers


@torch.no_grad()
def cross_entropy_value_and_grad(logits, labels, *, policy=None):
    """For hand-written fused kernels: per-row CE and its logit gradient for (n, V) rows.

    Computes in ``logits.dtype`` (callers pass their compute-dtype chunk). The
    gradient is ``q - onehot`` with ``1 - q_t = -expm1(log q_t)`` under ``POLICY``.
    """
    labels = labels.long()
    if not _stable(policy):
        logp = torch.log_softmax(logits, -1)
        grad = logp.exp()
        grad[torch.arange(labels.numel(), device=labels.device), labels] -= 1
        return -logp.gather(-1, labels[:, None])[:, 0], grad
    loss = logits.new_empty(labels.shape)
    grad = torch.empty_like(logits)
    rows = _rows(logits.shape[-1], None)
    for start in range(0, logits.shape[0], rows):
        stop = start + rows
        shifted, sp, _, _ = _shifted(logits[start:stop])
        logq = shifted - sp
        del shifted
        picked = logq.gather(-1, labels[start:stop, None])
        grad[start:stop] = logq.exp_().scatter_(-1, labels[start:stop, None], torch.expm1(picked))
        loss[start:stop] = -picked[:, 0]
    return loss, grad


@torch.no_grad()
def divergence_value_and_grad(student_logits, teacher_logits, kind, *, beta=0.5, policy=None):
    """For hand-written fused kernels: per-row divergence value and student-logit gradient for (n, V) rows."""
    if kind not in _KINDS:
        raise ValueError(f'unknown divergence {kind!r}; expected one of {_KINDS}')
    if not _stable(policy):
        zs = student_logits.detach().requires_grad_(True)
        with torch.enable_grad():
            value = _legacy_divergence(zs, teacher_logits, kind, beta)
            (grad,) = torch.autograd.grad(value.sum(), zs)
        return value.detach(), grad
    value = student_logits.new_empty(student_logits.shape[:-1])
    grad = torch.empty_like(student_logits)
    fast = _finite_logits(student_logits, teacher_logits)
    rows = _rows(student_logits.shape[-1], None)
    for start in range(0, student_logits.shape[0], rows):
        stop = start + rows
        value[start:stop], grad[start:stop] = _divergence_rows(student_logits[start:stop], teacher_logits[start:stop],
                                                               kind, beta, True, True, fast)
    return value, grad
