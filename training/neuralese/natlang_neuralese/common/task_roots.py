"""Pure task-root transformations shared by preparation and custody verification."""
from __future__ import annotations
from typing import Any


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
