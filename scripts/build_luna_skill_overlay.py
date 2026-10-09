#!/usr/bin/env python3
"""Create a skill-only overlay from exact selected Luna source rows."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def build(source: Path, selections: list[dict[str, Any]], skill_path: Path,
          output: Path, receipt: Path, *, replace_existing_skill_sha256: str | None = None,
          skill_name: str = "judge-against-criteria") -> dict[str, Any]:
    source = source.resolve(strict=True)
    skill_path = skill_path.resolve(strict=True)
    source_bytes = source.read_bytes()
    source_hash = sha(source_bytes)
    if (not isinstance(skill_name, str) or not skill_name
            or any(not (char.islower() or char.isdigit() or char == "-") for char in skill_name)
            or skill_name[0] == "-" or skill_name[-1] == "-"):
        raise ValueError("skill_name must be a lowercase hyphenated companion name")
    skill = skill_path.read_text(encoding="utf-8")
    skill_bytes = skill.encode("utf-8")
    source_lines = source_bytes.splitlines(keepends=True)
    if any(not line.strip() for line in source_lines):
        raise ValueError("source JSONL must not contain blank lines; indices use physical line numbers")
    source_rows = [json.loads(line) for line in source_lines]
    if not selections or len({s.get("index") for s in selections}) != len(selections):
        raise ValueError("selection must be nonempty and use unique row indices")
    output_rows, proofs, used_groups, used_seeds = [], [], set(), set()
    for selection in selections:
        index, seed = selection.get("index"), selection.get("seed")
        label = selection.get("label")
        if type(index) is not int or not 0 <= index < len(source_rows):
            raise ValueError(f"invalid source row index: {index}")
        if type(seed) is not int or seed < 0 or seed in used_seeds:
            raise ValueError("each selected row requires a unique nonnegative dispatcher seed")
        if not isinstance(label, str) or not label:
            raise ValueError("each selected row requires a descriptive label")
        row = source_rows[index]
        source_line = source_lines[index]
        groups = row.get("source_groups")
        if row.get("split") != "train" or not isinstance(groups, list) or len(groups) != 1:
            raise ValueError(f"selected row {index} must be one train row in one source group")
        if groups[0] in used_groups:
            raise ValueError(f"duplicate source group: {groups[0]}")
        used_groups.add(groups[0]); used_seeds.add(seed)
        overlay = copy.deepcopy(row)
        program_files = overlay.get("semantics", {}).get("files", {})
        nl_files = [name for name in program_files if name.endswith(".nl")]
        if len(nl_files) != 1:
            raise ValueError(f"source row {index} must contain exactly one .nl program file")
        module = nl_files[0].removesuffix(".nl")
        companion = f"{module}/skills/{skill_name}/SKILL.md"
        original = copy.deepcopy(overlay)
        prior_skill = program_files.get(companion)
        if prior_skill is not None:
            prior_sha = sha(prior_skill.encode("utf-8"))
            if replace_existing_skill_sha256 is None:
                raise ValueError(f"source row {index} already contains {companion}; replacement must be explicitly hash-bound")
            if prior_sha != replace_existing_skill_sha256:
                raise ValueError(f"existing skill hash mismatch for {companion}: {prior_sha}")
            program_files[companion] = skill
            skill_change = {"kind": "replace-existing-skill", "prior_sha256": prior_sha}
        else:
            if replace_existing_skill_sha256 is not None:
                raise ValueError(f"source row {index} has no existing {companion} to replace")
            program_files[companion] = skill
            skill_change = {"kind": "add-skill-companion", "prior_sha256": None}
        reconstructed = copy.deepcopy(overlay)
        if prior_skill is None:
            del reconstructed["semantics"]["files"][companion]
        else:
            reconstructed["semantics"]["files"][companion] = prior_skill
        if reconstructed != original:
            raise ValueError("overlay changed source content beyond the exact skill companion")
        output_row = canonical(overlay) + b"\n"
        output_rows.append(output_row)
        proofs.append({
            "label": label, "source_index": index, "seed": seed,
            "program_id": row.get("id"), "source_group": groups[0], "split": row.get("split"),
            "source_revision": row.get("source_revisions"),
            "source_row_raw_sha256_including_lf": sha(source_line),
            "source_row_canonical_sha256": sha(canonical(row)),
            "overlay_row_sha256_including_lf": sha(output_row),
            "root_code_sha256": sha(row["curriculum"]["reference"]["root"][0][1]["code"].encode("utf-8")),
            "skill_path": companion, "skill_name": skill_name,
            "skill_sha256": sha(skill_bytes), "skill_bytes": len(skill_bytes),
            "skill_change": skill_change,
            "expected_sha256_before": sha(canonical(row["semantics"]["expected"])),
            "expected_sha256_after": sha(canonical(overlay["semantics"]["expected"])),
            "folder_files_sha256_before": sha(canonical(row["semantics"]["folder_files"])),
            "folder_files_sha256_after": sha(canonical(overlay["semantics"]["folder_files"])),
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(b"".join(output_rows))
    cases = {"cases": [{"index": i, "seed": s["seed"]} for i, s in enumerate(selections)]}
    receipt_value = {
        "schema": "natlang.luna-skill-only-source-overlay/1",
        "training_admission": False,
        "source_path": str(source), "source_sha256": source_hash,
        "skill_path": str(skill_path), "skill_sha256": sha(skill_bytes),
        "overlay_path": str(output), "overlay_sha256": sha(output.read_bytes()),
        "cases": cases,
        "transform": "At the JSON object level, only the selected program-bound companion SKILL.md entry is added or explicitly hash-bound replaced. Rows are emitted in canonical key order, so output line bytes are not claimed identical to source line bytes. Facts, gold, other files, split, group, source revision, and root code are verified unchanged.",
        "rows": proofs,
    }
    with receipt.open("x", encoding="utf-8") as stream:
        json.dump(receipt_value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return receipt_value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--selections", type=Path, required=True, help="JSON array of {label,index,seed}")
    parser.add_argument("--skill", type=Path, required=True)
    parser.add_argument("--replace-existing-skill-sha256",
                        help="allow replacement only when the existing program-bound skill has this exact SHA-256")
    parser.add_argument("--skill-name", default="judge-against-criteria",
                        help="companion directory name under the program's skills/ directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    selections = json.loads(args.selections.read_text(encoding="utf-8"))
    result = build(args.source, selections, args.skill, args.output, args.receipt,
                   replace_existing_skill_sha256=args.replace_existing_skill_sha256,
                   skill_name=args.skill_name)
    print(json.dumps({"status": "built", "overlay": result["overlay_path"],
                      "overlay_sha256": result["overlay_sha256"],
                      "receipt": str(args.receipt), "rows": len(result["rows"])}, sort_keys=True))


if __name__ == "__main__":
    main()
