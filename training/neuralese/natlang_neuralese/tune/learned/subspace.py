"""Gradient subspaces for the shared basis and the gradient probe for bank views (plan "Sources", "Bank views").

*Subspaces.* For each source (task family, program or binding with training rows) accumulate the reply-weighted loss
gradient ``G_m`` of every target matrix on the executor's exact base, weighting rows by their current loss so rows the
executor already fits contribute little. Each source's leading left/right singular subspaces are pooled with equal
weight per source, together with the canonical weight-SVD basis, and the leading pooled directions initialize
``U_m, V_m`` (`pooled_basis`). Development and confirmation rows never contribute (`require_training_rows`).

*Probe.* With the bundle's composed core ``C_m`` as a live leaf, ``dL/dC_m = U_m^T G_m V_m`` exactly, so one backward
pass gives every component's first-order effect ``dL/da_k = sum_m <U_m^T G_m V_m, C_km>``.
"""
from __future__ import annotations

import torch

from . import LearnedOptimizerError
from .bank import SharedBasis, orthonormalize


def require_training_rows(rows):
    for row in rows:
        if row['graph'].get('split') != 'train':
            raise LearnedOptimizerError('only training-split rows contribute gradients, fits or components')


def row_loss_weights(losses, floor=0.):
    """Current-loss weights normalized to mean one; already-fit rows contribute little."""
    values = torch.as_tensor(losses, dtype=torch.float64).clamp_min(floor)
    total = float(values.sum())
    return (values * len(values) / total).tolist() if total > 0 else [0.] * len(values)


def source_gradients(engine, rows, names, *, entries=None, weights=None):
    """Dense FP32 ``{m: G_m}`` summed over one source's rows (``weights`` scales each row; default 1)."""
    from ...data.native_graph import graph_loss
    from ...data.native_graph_schema import parse_graph
    from ...model.native_adapters import run_native, target_parameters
    require_training_rows(rows)
    available = target_parameters(engine.shared_model)
    params = [available[name] for name in names]
    totals = {name: torch.zeros(available[name].shape, dtype=torch.float32) for name in names}
    fingerprint = engine.model_config['tokenizer_fingerprint']
    restore = {param: param.requires_grad for param in params}
    try:
        for param in params:
            param.requires_grad_(True)
        for index, row in enumerate(rows):
            weight = 1. if weights is None else float(weights[index])
            if weight == 0:
                continue
            example = parse_graph(row['graph'], tokenizer_fingerprint=fingerprint)
            def execute():
                loss, _ = graph_loss(engine.shared_model, example, checkpoint_layers=False)
                return torch.autograd.grad(loss * weight, params, allow_unused=True)
            for name, grad in zip(names, run_native(engine, execute, entries)):
                if grad is not None:
                    totals[name] += grad.detach().float().cpu()
    finally:
        for param, enabled in restore.items():
            param.requires_grad_(enabled)
    return totals


def row_losses(engine, rows, *, entries=None):
    from ...data.native_graph import graph_loss
    from ...data.native_graph_schema import parse_graph
    from ...model.native_adapters import run_native
    fingerprint = engine.model_config['tokenizer_fingerprint']
    out = []
    with torch.no_grad():
        for row in rows:
            example = parse_graph(row['graph'], tokenizer_fingerprint=fingerprint)
            out.append(float(run_native(engine, lambda: graph_loss(engine.shared_model, example, checkpoint_layers=False)[0], entries)))
    return out


def leading_subspaces(gradient, rank):
    """Leading left/right singular vectors of a (possibly batched) gradient."""
    u, _, vh = torch.linalg.svd(gradient.float(), full_matrices=False)
    rank = min(rank, u.shape[-1])
    return u[..., :rank], vh[..., :rank, :].transpose(-2, -1)


def pooled_basis(native_identity, sources, rank, *, per_source_rank=None, weight_svd=None, descriptors=None):
    """Equal-weight pooling of each source's leading subspaces (and the weight-SVD basis when given).

    ``sources`` is a list of ``{m: G_m}``. For each matrix the pooled projector ``sum_s Q_s Q_s^T`` is
    diagonalized through the SVD of the column-stacked orthonormal blocks; its leading ``rank`` directions form the
    basis. A matrix whose pooled span has fewer than ``rank`` directions is completed from the weight-SVD basis.
    """
    per_source_rank = per_source_rank or rank
    names = sorted(set().union(*[set(source) for source in sources]) | (set(weight_svd.left) if weight_svd else set()))
    left, right = {}, {}
    for name in names:
        lefts, rights = [], []
        for source in sources:
            if name in source and float(source[name].abs().max()) > 0:
                u, v = leading_subspaces(source[name], per_source_rank)
                lefts.append(u)
                rights.append(v)
        if weight_svd is not None and name in weight_svd.left:
            lefts.append(weight_svd.left[name])
            rights.append(weight_svd.right[name])
        if not lefts:
            raise LearnedOptimizerError(f'no source gradient or weight-SVD direction for {name}')
        left[name] = _pool(lefts, rank)
        right[name] = _pool(rights, rank)
    method = {'kind': 'gradient-subspace', 'per_source_rank': per_source_rank, 'sources': list(descriptors or []),
              'weight_svd': weight_svd is not None}
    return SharedBasis(dict(native_identity), rank, left, right, method).check()


def _pool(blocks, rank):
    stacked = torch.cat([block.float() for block in blocks], -1)
    u, s, _ = torch.linalg.svd(stacked, full_matrices=False)
    if u.shape[-1] < rank:
        raise LearnedOptimizerError(f'pooled span has {u.shape[-1]} directions, fewer than rank {rank}')
    q, _ = orthonormalize(u[..., :rank])
    return q.contiguous()


def probe(engine, bank, slots, adapter_id, cores, support_loss):
    """``dL/da_k`` for each view slot at the current composed cores.

    ``support_loss() -> scalar tensor`` computes the support objective inside the active native execution, where the
    bundle's composed adapter ``adapter_id`` executes the live core leaves. Returns ``(probe values, loss)``."""
    from ...model.native_adapters import run_native
    leaves = {name: value.detach().clone().float().reshape(*value.shape[:-2], -1).requires_grad_(True) for name, value in cores.items()}
    def execute():
        loss = support_loss()
        grads = torch.autograd.grad(loss, [leaves[name] for name in bank.basis.names], allow_unused=True)
        return loss.detach(), grads
    loss, grads = run_native(engine, execute, [{'id': adapter_id, 'scale': 1.}], leaves={adapter_id: leaves}, gradient_precision='fp32')
    R = bank.basis.rank
    gradient = {name: (g if g is not None else torch.zeros_like(leaves[name])).reshape(*leaves[name].shape[:-1], R, R)
                for name, g in zip(bank.basis.names, grads)}
    values = []
    for k in slots:
        total = torch.zeros((), dtype=torch.float64)
        for name in bank.components[k].factors:
            total += (gradient[name].double() * bank.components[k].core(name).double()).sum()
        values.append(float(total))
    return values, float(loss)


def probe_integers(values, max_int=9):
    """Descent direction as small integers for the optimizer's view: ``-sign(g) * round(9 |g| / max|g|)``."""
    peak = max((abs(v) for v in values), default=0.)
    if peak == 0:
        return [0] * len(values)
    return [int(-round(max_int * v / peak)) for v in values]
