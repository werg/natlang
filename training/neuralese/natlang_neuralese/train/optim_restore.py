"""Named, safe optimizer-state restore for the neuralese trainers.

Checkpoints record ``optimizer_param_names``: the parameter name of every optimizer slot, in the order of the
optimizer's flattened ``param_groups`` (for ``PortMuonAdamW`` that is the Muon child followed by the AdamW child).
This lets a later run with a different trainable set map each saved moment to its parameter instead of guessing by
position, and it replaces the old silent "ValueError -> fresh optimizer" fallback:

* unchanged layout (same names in the same order): ``optimizer.load_state_dict(saved)`` exactly as before,
  including restoring the saved group hyperparameters;
* changed layout: parameters present in both keep their exact state (moments, step counts, any per-parameter
  state); parameters in ``declared_added`` that the checkpoint lacks start fresh, explicitly; anything else
  (undeclared new parameter, vanished parameter, optimizer-kind change such as Muon <-> AdamW, state-shape change)
  is refused with an error naming the parameters. Group hyperparameters (lr, betas, weight decay, ...) come from the
  *current* optimizer on this path, because the saved groups no longer line up with the current ones;
* checkpoints without names (written before this change) load only when ``load_state_dict`` accepts them, as before;
  otherwise they are refused rather than silently reset;
* ``fresh=True`` (``--optimizer-state fresh``) is the explicit, declared way to start the optimizer from scratch.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

SCHEMA = "natlang.optimizer-restore/1"


class OptimizerRestoreError(ValueError):
    """The saved optimizer state cannot be mapped onto the current optimizer without an undeclared reset."""


def _children(optimizer):
    """``[(kind, plain optimizer)]``: PortMuonAdamW splits into its Muon and AdamW children."""
    if hasattr(optimizer, "muon"):
        return [("muon", optimizer.muon)] + ([("adamw", optimizer.auxiliary)] if optimizer.auxiliary is not None else [])
    return [("plain", optimizer)]


def _id_names(named_params):
    items = named_params.items() if hasattr(named_params, "items") else named_params
    return {id(p): n for n, p in items}


def optimizer_param_names(optimizer, named_params):
    """Names of the optimizer's slots in flattened ``param_groups`` order, or None if any parameter is unnamed
    or names repeat (the checkpoint then simply stays positional)."""
    names = _id_names(named_params)
    flat = []
    for _, child in _children(optimizer):
        for group in child.param_groups:
            for p in group["params"]:
                if id(p) not in names:
                    return None
                flat.append(names[id(p)])
    return flat if len(set(flat)) == len(flat) else None


def declared_added_names(named_params, prefixes):
    """Names (from ``named_params``) starting with any declared prefix."""
    prefixes = tuple(prefixes or ())
    items = named_params.items() if hasattr(named_params, "items") else named_params
    return {n for n, _ in items if prefixes and n.startswith(prefixes)}


def _saved_children(saved):
    if "muon" in saved:  # PortMuonAdamW
        return [("muon", saved["muon"])] + ([("adamw", saved["adamw"])] if saved.get("adamw") is not None else [])
    return [("plain", saved)]


def _layout_matches(optimizer, saved):
    if hasattr(optimizer, "muon") != ("muon" in saved):
        return False
    kids, skids = _children(optimizer), _saved_children(saved)
    return len(kids) == len(skids) and all(
        [len(g["params"]) for g in c.param_groups] == [len(g["params"]) for g in s["param_groups"]]
        for (_, c), (_, s) in zip(kids, skids))


def restore_optimizer_state(optimizer, saved, named_params, declared_added=(), *, saved_names=None, fresh=False):
    report = _restore_optimizer_state(optimizer, saved, named_params, declared_added, saved_names=saved_names,
                                      fresh=fresh)
    own_host_tensors(optimizer)
    return report


def own_host_tensors(optimizer):
    """Replace every host tensor left in the optimizer's state and groups by an owned copy. torch's loader keeps CPU
    values as given (AdamW's ``step`` counters, group values), and one tensor still viewing an ``mmap=True``
    checkpoint keeps the whole file mapping alive, with every page the device copies dirtied (Mellum's resume:
    ~38-46 GB of host memory for the rest of the run)."""
    def own(value):
        return value.clone() if torch.is_tensor(value) and value.device.type == "cpu" else value
    children = [child for _, child in _children(optimizer)]
    if getattr(optimizer, "latent", None) is not None:
        children.append(optimizer.latent)
    for child in children:
        for state in child.state.values():
            for key, value in list(state.items()):
                state[key] = own(value)
        for group in child.param_groups:
            for key, value in list(group.items()):
                if key != "params":
                    group[key] = own(value)


def _restore_optimizer_state(optimizer, saved, named_params, declared_added=(), *, saved_names=None, fresh=False):
    """Restore ``saved`` (an ``optimizer.state_dict()``) into ``optimizer``; returns a JSON-able report.

    ``named_params``: ``{name: Parameter}`` covering every optimizer parameter. ``saved_names``: the checkpoint's
    ``optimizer_param_names`` (None for old checkpoints). ``declared_added``: names allowed to start fresh when the
    checkpoint lacks them. Raises ``OptimizerRestoreError`` for every undeclared difference."""
    declared_added = set(declared_added or ())
    current = optimizer_param_names(optimizer, named_params)
    kids = _children(optimizer)
    if fresh:
        return _report("fresh", kids, current, kept=[], initialized=current or [],
                       note="explicit --optimizer-state fresh")
    if saved_names is None:
        try:
            optimizer.load_state_dict(saved)
        except ValueError as error:
            raise OptimizerRestoreError(
                "checkpoint has no optimizer parameter names and its positional optimizer state does not match the "
                f"current parameter groups ({str(error)[:200]}); restore needs a checkpoint written with "
                "optimizer_param_names, or pass an explicit --optimizer-state fresh") from error
        return _report("positional", kids, current, kept=current or [], initialized=[], all_kept=True)
    saved_names = list(saved_names)
    if current is None:
        raise OptimizerRestoreError("named_params does not cover every optimizer parameter or has duplicate names")
    if saved_names == current and _layout_matches(optimizer, saved):
        try:
            optimizer.load_state_dict(saved)
        except ValueError as error:
            raise OptimizerRestoreError(
                f"optimizer state for identical parameter names was rejected: {str(error)[:300]}") from error
        return _report("exact", kids, current, kept=current, initialized=[])

    if hasattr(optimizer, "muon") != ("muon" in saved):
        raise OptimizerRestoreError("optimizer kind changed: saved and current optimizers differ in Muon/AdamW partitioning")
    saved_info, offset = {}, 0  # name -> (kind, state or None)
    for kind, sd in _saved_children(saved):
        order = [i for g in sd["param_groups"] for i in g["params"]]
        for slot, i in enumerate(order):
            if offset + slot >= len(saved_names):
                raise OptimizerRestoreError("checkpoint optimizer_param_names is shorter than its optimizer state")
            saved_info[saved_names[offset + slot]] = (kind, sd["state"].get(i))
        offset += len(order)
    if offset != len(saved_names):
        raise OptimizerRestoreError("checkpoint optimizer_param_names length does not match its optimizer state")

    by_id = _id_names(named_params)
    kind_changed, undeclared, shape_bad, kept, initialized = [], [], [], [], []
    aligned = {}
    for kind, child in kids:
        state, groups, local = {}, [], 0
        for group in child.param_groups:
            idx = []
            for p in group["params"]:
                name = by_id[id(p)]
                if name in saved_info:
                    skind, sstate = saved_info[name]
                    if skind != kind and not (skind == "plain" and kind == "plain"):
                        kind_changed.append(f"{name} ({skind} -> {kind})")
                    else:
                        bad = [k for k, v in (sstate or {}).items()
                               if torch.is_tensor(v) and v.numel() > 1 and tuple(v.shape) != tuple(p.shape)]
                        if bad:
                            shape_bad.append(f"{name} {bad}")
                        else:
                            if sstate:
                                state[local] = sstate
                            kept.append(name)
                elif name in declared_added:
                    initialized.append(name)
                else:
                    undeclared.append(name)
                idx.append(local)
                local += 1
            groups.append({**{k: v for k, v in group.items() if k != "params"}, "params": idx})
        aligned[kind] = {"state": state, "param_groups": groups}
    removed = sorted(set(saved_info) - set(current))
    problems = []
    if undeclared:
        problems.append("new parameters not declared as added: " + ", ".join(sorted(undeclared)))
    if removed:
        problems.append("saved parameters missing from the current optimizer: " + ", ".join(removed))
    if kind_changed:
        problems.append("optimizer kind/group changed: " + ", ".join(kind_changed))
    if shape_bad:
        problems.append("saved state shape differs from parameter: " + ", ".join(shape_bad))
    if problems:
        raise OptimizerRestoreError("refusing optimizer-state restore (" + "; ".join(problems) +
                                    "). Declare additions explicitly, or pass --optimizer-state fresh to reset deliberately.")
    if hasattr(optimizer, "muon"):
        payload = {"format": saved["format"], "schema": optimizer.schema, "muon": aligned["muon"],
                   "adamw": aligned.get("adamw")}
    else:
        payload = aligned["plain"]
    optimizer.load_state_dict(payload)
    return _report("by-name", kids, current, kept=kept, initialized=initialized)


def _report(mode, kids, current, *, kept, initialized, note=None, all_kept=False):
    by_group, flat = {}, 0
    kept_set, init_set = set(kept), set(initialized)
    names = current or []
    for kind, child in kids:
        for gi, group in enumerate(child.param_groups):
            row = by_group.setdefault(f"{kind}:{gi}", {"kept": 0, "initialized": 0, "lr": group.get("lr")})
            for _ in group["params"]:
                name = names[flat] if flat < len(names) else None
                flat += 1
                if mode == "fresh" or name in init_set:
                    row["initialized"] += 1
                elif all_kept or name in kept_set:
                    row["kept"] += 1
    report = {"schema": SCHEMA, "mode": mode, "kept": sum(r["kept"] for r in by_group.values()),
              "initialized": sorted(initialized), "initialized_count": len(initialized), "by_group": by_group}
    if note:
        report["note"] = note
    return report


def record_restore_report(report, out_dir, *, source=None):
    """Print the restore event and persist it beside the run's other receipts (``optimizer-restore.jsonl``)."""
    event = {"event": "optimizer_state_restore", **report, "initialized": report["initialized"][:50]}
    if source is not None:
        event["source"] = str(source)
    print(json.dumps(event), flush=True)
    with (Path(out_dir) / "optimizer-restore.jsonl").open("a") as stream:
        stream.write(json.dumps({**report, **({"source": str(source)} if source is not None else {})}) + "\n")


def add_optimizer_restore_arguments(parser):
    parser.add_argument("--optimizer-state", choices=["restore", "fresh"], default="restore",
                        help="restore: named, refusing restore of the saved optimizer state; fresh: explicitly start "
                             "the optimizer from scratch (a declared change, never the silent default)")
    parser.add_argument("--optimizer-added", action="append", default=[], metavar="PREFIX",
                        help="parameter-name prefix declared as newly added to the optimizer (starts fresh; repeatable)")
