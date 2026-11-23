"""Pure task-root transformations shared by preparation and custody verification."""
from __future__ import annotations
from typing import Any
import hashlib


def student_root_row(row: dict[str, Any], *, held: bool = False) -> dict[str, Any]:
    """Run the student at the root as well as in children, without reference-root replay. ``held``: the row is a pinned
    held-evaluation case (``held_eval measure``), measured whatever its split; a set authored for holding (held-v4) keeps
    its rows' dev/confirmation splits. Rollouts that may train keep the train-only guard."""
    if not held and row.get("split") != "train":
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
    if row.get('kind') == 'evaluation_fixture':
        # The fixture already declares its real authored module/NL entry. A
        # synthetic reference eval would fabricate another root and erase the
        # distinction between an entrypoint and a semantic anchor.
        spec = (row.get('semantics') or {}).get('evaluation_fixture') or {}
        if spec.get('schema') != 'natlang.evaluation-fixture/1' or not spec.get('entry'):
            raise ValueError('authored fixture mode requires its actual declared entry')
        contract = dict(curriculum.get('trajectory_contract') or {})
        contract.pop('authored_root_eval', None)
        contract.pop('authored_root_sha256', None)
        contract.update(version='natlang.trajectory_contract/1', authored_fixture_entry=True)
        curriculum['trajectory_contract'] = contract
        return {**row, 'curriculum': curriculum}
    root = next((args for kind, args in ((curriculum.get("reference") or {}).get("root") or []) if kind == "eval"), None)
    if not isinstance(root, dict) or not isinstance(root.get("code"), str) or not root["code"].strip():
        raise ValueError(f"row has no authored reference root eval: {row.get('id')}")
    curriculum["trajectory_contract"] = {
        "version": "natlang.trajectory_contract/1", "authored_root_eval": True,
        "authored_root_sha256": hashlib.sha256(root["code"].encode()).hexdigest(),
        "min_child_invocations": 0, "min_state_edges_from_children": 0, "min_iteration_steps": 0,
    }
    return {**row, "curriculum": curriculum}
