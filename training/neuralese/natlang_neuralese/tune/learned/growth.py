"""Filling, growing and sizing the bank (plan "A bank large enough to move the needle": sources, sizing rules).

* `source_component`: for one task family, program or binding with training rows, a short subspace fit in the shared
  basis yields one component (a program fit under the existing rules; it earns no production exposure).
* `free_tune_component`: a projected free tune; its out-of-basis residual drives basis growth.
* `residual_growth`: at publication boundaries, add a basis direction where residuals recur across independent
  sources.
* `merge_or_retire`: merge components whose behavioral effects are nearly identical; retire negligible ones.
* Coverage: weight-space capture, effective components in use, saturation over sub-banks and basis slices.
"""
from __future__ import annotations

import math
import random

import torch

from . import LearnedOptimizerError
from .bank import Component, component_id, factor_core, normalize, orthonormalize


def source_component(engine, bank, rows, *, source, description, rank=8, steps=20, learning_rate=1e-2,
                     levels=('tree', 'folder', 'lambda'), owner=None):
    """Fit one full ``R x R`` core per matrix on ``rows`` through a placeholder bank adapter, then keep its best
    rank-``rank`` factorization; the key is written from ``description`` on the exact base."""
    from .fit import placeholder_adapters
    from .subspace import require_training_rows
    from ...data.native_graph import graph_loss
    from ...data.native_graph_schema import parse_graph
    from ...model.native_adapters import run_native
    require_training_rows(rows)
    adapter = placeholder_adapters(engine, bank, [f'source:{source["name"]}'], owner)[f'source:{source["name"]}']
    R = bank.basis.rank
    cores = {name: torch.zeros(*bank.basis.left[name].shape[:-2], R, R, requires_grad=True) for name in bank.basis.names}
    optimizer = torch.optim.Adam(list(cores.values()), lr=learning_rate)
    fingerprint = engine.model_config['tokenizer_fingerprint']
    examples = [parse_graph(row['graph'], tokenizer_fingerprint=fingerprint) for row in rows]
    losses = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        total = 0.
        for example in examples:
            def execute(example=example):
                loss, _ = graph_loss(engine.shared_model, example, checkpoint_layers=False)
                (loss / len(examples)).backward()
                return float(loss.detach())
            total += run_native(engine, execute, [{'id': adapter, 'scale': 1.}],
                                leaves={adapter: bank.adapter_tensors(cores)}, gradient_precision='fp32') / len(examples)
        losses.append(total)
        optimizer.step()
    factors = {name: factor_core(core.detach(), min(rank, R)) for name, core in cores.items() if float(core.detach().abs().max()) > 0}
    if not factors:
        raise LearnedOptimizerError(f'source {source["name"]} produced no update; record it as absent')
    key = engine.native_write_text(description)['id']
    component = Component(component_id(source, factors), factors, key, description, {**source, 'kind': 'source-fit'}, tuple(levels))
    component, scale = normalize(component, bank.q)
    return component, {'losses': losses, 'normalization': scale}


def free_tune_component(bank, cores, *, source, description, key, rank=8, levels=('tree', 'folder', 'lambda')):
    """A projected free tune (basis coordinates from `ComponentBank.project_factors`) as a normalized component."""
    factors = {name: factor_core(core, min(rank, bank.basis.rank)) for name, core in cores.items() if float(core.abs().max()) > 0}
    if not factors:
        raise LearnedOptimizerError('the free tune projects to nothing inside the basis')
    component = Component(component_id(source, factors), factors, key, description, {**source, 'kind': 'free-tune'}, tuple(levels))
    return normalize(component, bank.q)[0]


def basis_residual(bank, factors):
    """Per matrix, the out-of-basis parts ``(I - U U^T) B A`` and ``B A (I - V V^T)`` as factor pairs."""
    out = {}
    for name, (B, A) in factors.items():
        U, V = bank.basis.left[name].to(B), bank.basis.right[name].to(A)
        left = B - U @ (U.transpose(-2, -1) @ B)
        right = A - (A @ V) @ V.transpose(-2, -1)
        out[name] = (left, right)
    return out


def residual_growth(bank, residuals, *, add, min_sources=2, tolerance=.5):
    """New basis columns where out-of-basis residual directions recur across ``min_sources`` independent sources.
    ``residuals``: list of per-source ``{m: (left_residual (rows, r), right_residual (r, cols))}``. Returns the grown
    basis left/right dictionaries (new rank ``R + add``) and the per-matrix support counts; existing core coordinates
    stay valid because the old columns are kept as the leading block."""
    from .bank import SharedBasis
    left, right, support = {}, {}, {}
    for name in bank.basis.names:
        lefts = [r[name][0] for r in residuals if name in r and float(r[name][0].abs().max()) > 0]
        rights = [r[name][1].transpose(-2, -1) for r in residuals if name in r and float(r[name][1].abs().max()) > 0]
        support[name] = len(lefts)
        U, V = bank.basis.left[name], bank.basis.right[name]
        if len(lefts) < min_sources:
            left[name], right[name] = _extend(U, None, add), _extend(V, None, add)
            continue
        left[name] = _extend(U, torch.cat(lefts, -1), add)
        right[name] = _extend(V, torch.cat(rights, -1), add)
    grown = SharedBasis(bank.basis.native_identity, bank.basis.rank + add, left, right,
                        {**bank.basis.method, 'grown_from': bank.basis_artifact, 'added': add})
    return grown.check(), support


def _extend(Q, candidates, add):
    """Append ``add`` orthonormal columns: leading residual directions when given, else deterministic fill."""
    if candidates is None:
        generator = torch.Generator().manual_seed(Q.shape[-2] * 7919 + Q.shape[-1])
        candidates = torch.randn(*Q.shape[:-1], add, generator=generator)
    projected = candidates.float() - Q @ (Q.transpose(-2, -1) @ candidates.float())
    u, _, _ = torch.linalg.svd(projected, full_matrices=False)
    extra = u[..., :add]
    if extra.shape[-1] < add:
        raise LearnedOptimizerError('not enough independent residual directions to grow the basis')
    q, _ = orthonormalize(torch.cat([Q, extra], -1))
    return q


def pad_component(component, old_rank, new_rank):
    """Re-express a component after basis growth: the old coordinates occupy the leading block."""
    factors = {name: (torch.nn.functional.pad(P, (0, 0, 0, new_rank - old_rank)),
                      torch.nn.functional.pad(Q, (0, 0, 0, new_rank - old_rank))) for name, (P, Q) in component.factors.items()}
    return Component(component.id, factors, component.key, component.description, component.source, component.levels)


def merge_or_retire(bank, effects, *, similar=.98, negligible=1e-3):
    """``effects``: component ID -> behavioral effect vector (e.g. per-probe-row log-likelihood changes of a single
    component). Retire negligible components; merge near-identical ones (keep the first). Returns the decisions."""
    retire = [cid for cid, vector in effects.items() if float(torch.as_tensor(vector).norm()) < negligible]
    kept = [cid for cid in bank.ids if cid in effects and cid not in retire]
    merges = []
    for i, a in enumerate(kept):
        for b in kept[i + 1:]:
            va, vb = torch.as_tensor(effects[a], dtype=torch.float64), torch.as_tensor(effects[b], dtype=torch.float64)
            cosine = float(va @ vb / (va.norm() * vb.norm()))
            if cosine > similar and b not in [m[1] for m in merges]:
                merges.append((a, b, cosine))
    return {'retire': retire, 'merge': merges}


def effective_components(utilization):
    """Exponentiated entropy of component use (mass of ``|weight|`` per component across proposals or fits)."""
    values = torch.as_tensor(list(utilization), dtype=torch.float64).abs()
    total = float(values.sum())
    if total == 0:
        return 0.
    p = values[values > 0] / total
    return float(torch.exp(-(p * p.log()).sum()))


def weight_space_coverage(bank, held_out):
    """Share of held-out free-tune update energy inside the basis span and inside the component span (diagnostic)."""
    observed = inside_basis = inside_bank = 0.
    for factors in held_out:
        factors = {name: value for name, value in factors.items() if name in bank.basis.left}
        cores, energy = bank.project_factors(factors)
        observed += sum(bank.q[n] * e[0] for n, e in energy.items())
        inside_basis += sum(bank.q[n] * e[1] for n, e in energy.items())
        if bank.components:
            _, residual, target = bank.fit_mixture(cores)
            inside_bank += target - residual
    return {'basisShare': inside_basis / observed if observed else None,
            'componentShare': inside_bank / observed if observed and bank.components else None}


def saturation(measure, bank, *, sizes, repeats=3, seed=0):
    """Recovered gain (``measure(sub_bank) -> float``) over random sub-banks of each size; the last doubling's
    increment says whether to grow before training the optimizer harder."""
    rng = random.Random(seed)
    curve = {}
    for size in sizes:
        values = []
        for _ in range(repeats):
            chosen = sorted(rng.sample(range(len(bank.components)), min(size, len(bank.components))))
            values.append(measure(bank.with_components([bank.components[k] for k in chosen], reason=f'saturation sub-bank {size}')))
        curve[size] = sum(values) / len(values)
    ordered = sorted(curve)
    increment = (curve[ordered[-1]] - curve[ordered[-2]]) if len(ordered) > 1 else None
    return {'curve': curve, 'lastIncrement': increment}
