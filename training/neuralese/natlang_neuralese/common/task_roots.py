"""Pure task-root transformations shared by preparation and custody verification."""
from __future__ import annotations
from typing import Any
import hashlib


def student_root_row(row: dict[str, Any]) -> dict[str, Any]:
    """Run the student at the root as well as in children, without reference-root replay."""
    if row.get("split") != "train":
        raise ValueError(f"refusing non-train row for RL rollouts: {row.get('id')} (split={row.get('split')!r})")
    curriculum = dict(row.get("curriculum") or {})
    contract = dict(curriculum.get("trajectory_contract") or {})
    if contract.get("authored_root_eval"):
        contract.pop("authored_root_eval", None)
        contract.pop("authored_root_sha256", None)
        curriculum["trajectory_contract"] = contract
    return {**row, "curriculum": curriculum}


def authored_root_row(row: dict[str, Any]) -> dict[str, Any]:
    """The held-task packet's derivation applied to a train row: the root eval is the row's authored reference root,
    so rollouts exercise the model inside the child calls exactly as the held-task eval does."""
    if row.get("split") != "train":
        raise ValueError(f"refusing non-train row for RL rollouts: {row.get('id')} (split={row.get('split')!r})")
    curriculum = dict(row.get("curriculum") or {})
    root = next((args for kind, args in ((curriculum.get("reference") or {}).get("root") or []) if kind == "eval"), None)
    if not isinstance(root, dict) or not isinstance(root.get("code"), str) or not root["code"].strip():
        raise ValueError(f"row has no authored reference root eval: {row.get('id')}")
    curriculum["trajectory_contract"] = {
        "version": "natlang.trajectory_contract/1", "authored_root_eval": True,
        "authored_root_sha256": hashlib.sha256(root["code"].encode()).hexdigest(),
        "min_child_invocations": 0, "min_state_edges_from_children": 0, "min_iteration_steps": 0,
    }
    return {**row, "curriculum": curriculum}
