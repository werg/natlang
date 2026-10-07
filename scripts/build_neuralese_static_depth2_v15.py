#!/usr/bin/env python3
"""Create a fresh V15 static depth variant without modifying the reviewed source.

The twelve nested source worlds keep their facts, decisions, IDs, groups, and
splits. Their reference program changes from a crisp folder reducer to a
per-folder Neuralese reducer that creates a semantic per-item Neuralese judge.
The twelve iterate worlds are copied as controls. This creates source/runtime
fixtures only; it does not admit data or generate teacher targets.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from neuralese_source_world_builder import build_proof, validate_world_rows

BASE = ROOT / "runs/neuralese-successor-v15-prepared-20261007-candidate-v9/source.cases.jsonl"
OUT = ROOT / "runs/neuralese-static-depth2-v15-20261007-v8"
VARIANT = "static-per-folder-nl-reducer-plus-item-semantic-judge/1"

ROOT_CODE = (
    "const task=await folder.file('task.json').readJson(); const teams=await folder.folders('teams/*'); "
    "const nested=await Promise.all(teams.map(async (team:Folder)=>{ "
    "const inheritedRule=JSON.parse(await team.file('policy.json').readText()).inherited_rule; "
    "const taskInstruction:string=task.instruction; "
    "const reduceTeam:Neuralese<(team:Folder)=>Promise<string[]>>=nl.with({inheritedRule,taskInstruction})`<|neuralese|>For the supplied folder, read its local override and list its complete item files. For each item, create a semantic judgment using the inherited rule, local override, task instruction, and that item's complete FileHandle. Keep only items satisfying every applicable clause. Return this folder's selected complete item IDs sorted lexicographically.<|/neuralese|>`; "
    "return await reduceTeam(team); })); console.log('Per-folder reducer results',nested); "
    "const selected=nested.flat().sort(); await folder.file(task.output_path).writeText(JSON.stringify(selected)); "
    "const saved=await folder.file(task.output_path).readJson(); if(JSON.stringify(saved)!==JSON.stringify(selected)) throw Error('readback mismatch'); return saved;"
)

OUTER_CHILD_EVAL = (
    "const localRule=JSON.parse(await team.file('override.json').readText()).local_rule; "
    "const items=await team.files('items/*.md'); "
    "const judge:Neuralese<(item:FileHandle)=>Promise<boolean>>=nl.with({inheritedRule,localRule,taskInstruction})`<|neuralese|>Read this complete item's evidence. Apply the inherited rule, this folder's local override, and all applicable item-specific facts. Decide only this item; do not borrow evidence from a sibling folder or item. Return true only when every applicable clause is supported.<|/neuralese|>`; "
    "const pairs=await Promise.all(items.map(async item=>{const itemId=item.name.replace(/\\.md$/,''); return [itemId,await judge(item)] as const;})); "
    "return pairs.filter(([,keep])=>keep).map(([id])=>id).sort();"
)

HUMAN_SIGNATURES = {
    ("archive_sound_rights", "ARC-201"): ("Tern Street Radio", "Mara Velin"),
    ("archive_sound_rights", "ARC-215"): ("Dovetail Oral History Circle", "Eli Wren"),
    ("archive_sound_rights", "ARC-236"): ("Lantern Market Archive", "Rina Sol"),
    ("community_energy_match", "COM-208"): ("Hill Lantern Co-op", "Nia Bell"),
    ("community_energy_match", "COM-222"): ("Tideglass Housing", "Oren Vale"),
    ("community_energy_match", "COM-236"): ("South Arcade Association", "Sana Dey"),
    ("habitat_field_permits", "HAB-208"): ("Kelp Edge Survey", "Ranger Ila Moss"),
    ("habitat_field_permits", "HAB-222"): ("Salt Grass Watch", "Ranger D. Vale"),
    ("habitat_field_permits", "HAB-236"): ("Moon Snail Project", "Ranger S. Rafi"),
    ("neighborhood_tree_work", "NEI-215"): ("Willow Steps Council", "wildlife monitor L. Arden"),
    ("neighborhood_tree_work", "NEI-229"): ("Maple Yard Association", "wildlife monitor M. Sato"),
    ("school_accessible_trip", "SCH-208"): ("Class 6A", "N. Reed"),
    ("school_accessible_trip", "SCH-222"): ("Class 5D", "L. Ames"),
    ("school_accessible_trip", "SCH-236"): ("Class 7B", "V. Sato"),
    ("translation_editorial_clearance", "TRA-208"): ("Ruma clinic guide edition 2", "A. Lema"),
    ("translation_editorial_clearance", "TRA-222"): ("Ruma ferry notices edition 1", "K. Dala"),
    ("translation_editorial_clearance", "TRA-236"): ("Ruma food-safety card edition 6", "V. Neri"),
}


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def update_signature_facts(row: dict) -> list[dict]:
    changes = []
    slug = row["curriculum"]["shape"].removeprefix("v15-").split("-nested-reducer", 1)[0]
    for path, value in list(row["semantics"]["folder_files"].items()):
        if "/items/" not in path:
            continue
        item_id = Path(path).stem
        role_change = HUMAN_SIGNATURES.get((slug, item_id))
        if role_change:
            old_person, new_person = role_change
            old = value
            if item_id.startswith("ARC-"):
                old_fragment = f"Speaker release SR-{item_id} signed by {old_person} names recording revision"
                new_fragment = f"Speaker release SR-{item_id} signed by {new_person}, the recorded speaker, names recording revision"
            elif item_id.startswith("COM-"):
                old_fragment = f"Engineer {old_person} signed roof capacity note"
                new_fragment = f"Engineer {new_person} signed roof capacity note"
            elif item_id.startswith("HAB-"):
                old_fragment = f"Ranger {old_person} accepted rope-access plan"
                new_fragment = f"{new_person} accepted rope-access plan"
            elif item_id.startswith("NEI-"):
                old_fragment = f"Monitor {old_person} accepted buffer plan"
                new_fragment = f"{new_person} accepted buffer plan"
            elif item_id.startswith("SCH-"):
                old_fragment = f"Quiet-room contact {old_person} is assigned"
                new_fragment = f"Quiet-room contact {new_person} is assigned"
            else:
                tail = "has not approved the register choice for clinic guide" if item_id == "TRA-208" else (
                    "approved the register choice for ferry notices" if item_id == "TRA-222" else
                    "approved the register choice for food-safety card")
                old_fragment = f"Community reviewer {old_person} {tail}"
                new_fragment = f"Community reviewer {new_person} {tail}"
            if old_fragment not in old:
                raise ValueError(f"{item_id}: expected organization signer not found in {path}")
            value = old.replace(old_fragment, new_fragment)
            row["semantics"]["folder_files"][path] = value
            changes.append({"item_id": item_id, "path": path, "previous": old_fragment, "replacement": new_fragment})
    return changes


def make_reference_children(row: dict) -> tuple[list[dict], list[dict]]:
    old_children = row["curriculum"]["reference"].get("children", [])
    item_children, outer_children = [], []
    for child in old_children:
        match = child["match"]
        if not isinstance(match, str) or not match.endswith(".md"):
            raise ValueError(f"unexpected nested reference child match: {match!r}")
        final = next((call for call in reversed(child.get("calls", [])) if call[0] == "return_result"), None)
        if not final:
            raise ValueError(f"{match}: no source-reference item judgment")
        item_id = match[:-3]
        # The item is now read by the folder reducer and passed as a FileHandle
        # to this child, so the child reference only supplies the judgment.
        item_children.append({"match": match, "value": final[1]["value"]})

    team_paths = sorted({path.split("/items/", 1)[0] for path in row["semantics"]["folder_files"] if "/items/" in path})
    values_by_id = {child["match"][:-3]: child["value"] for child in item_children}
    for team_path in team_paths:
        first_item_path = next(path for path in sorted(row["semantics"]["folder_files"])
                               if path.startswith(team_path + "/items/") and path.endswith(".md"))
        # The reducer invocation receives its scoped folder as `folder`, so
        # its opening lists this item path rather than the original root path.
        # Matching the scoped first item distinguishes sibling reducers while
        # remaining disjoint from item children, whose handle path is a leaf.
        outer_children.append({
            "match": first_item_path[len(team_path) + 1 :],
            "calls": [["eval", {"code": OUTER_CHILD_EVAL}],
                      ["return_result", {"status": "success", "value": sorted(
                          item_id for item_id, keep in values_by_id.items()
                          if keep and f"{team_path}/items/{item_id}.md" in row["semantics"]["folder_files"])}]],
        })
    return outer_children, item_children


def build() -> tuple[bytes, dict]:
    source_bytes = BASE.read_bytes()
    rows = [json.loads(line) for line in source_bytes.decode().splitlines() if line]
    changes = []
    nested_count = 0
    for row in rows:
        shape = row["curriculum"]["shape"]
        if "-nested-reducer" not in shape:
            continue
        nested_count += 1
        row["source_revisions"] = ["authored-semantic-source-worlds-v15/9", VARIANT]
        row["generation"]["generator"] = VARIANT
        row["generation"]["source_base_sha256"] = hashlib.sha256(source_bytes).hexdigest()
        row["generation"]["reference_kind"] = "deterministic-authored-static-reference; no teacher/provider calls"
        changes.extend({"source_group": row["source_groups"][0], **change} for change in update_signature_facts(row))
        row["semantics"]["expected_files"] = {
            **row["semantics"]["folder_files"],
            row["semantics"]["expected_path"] if "expected_path" in row["semantics"] else "selection.json": json.dumps(
                row["semantics"]["expected"], ensure_ascii=False, separators=(",", ":")),
        }
        root_eval = next((call for call in row["curriculum"]["reference"]["root"] if call[0] == "eval"), None)
        if not root_eval:
            raise ValueError(f"{row['id']}: missing root eval source")
        root_eval[1]["code"] = ROOT_CODE
        outer, items = make_reference_children(row)
        row["curriculum"]["reference"]["children"] = outer + items
        row["curriculum"]["variant"] = VARIANT
        row["curriculum"]["shape"] += "-depth2-static"
        row["semantics"]["files"]["review_nested_items.nl"] = (
            "---\nargs: {}\nreturns: string[]\nkind: directory-reducer\n---\n"
            "For each team folder, create a folder-scoped natural-language reducer with the inherited policy as a typed snapshot. "
            "That reducer reads the local override and uses a semantic per-item lambda on each complete matching item handle. "
            "Inspect the returned array from each folder reducer before combining them. Return sorted IDs of all and only items "
            "satisfying every applicable clause. Write, read back, verify, and return.\n"
        )
        row["semantics"]["expected_files"].update({
            path: text for path, text in row["semantics"]["folder_files"].items()
        })
        row["semantics"]["expected_files"]["selection.json"] = json.dumps(
            row["semantics"]["expected"], ensure_ascii=False, separators=(",", ":"))
    validate_world_rows(rows)
    raw = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows).encode()
    proof = build_proof(rows, raw)
    proof.update({
        "variant": VARIANT,
        "base_source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "nested_depth_variant_worlds": nested_count,
        "unchanged_iterate_control_worlds": len(rows) - nested_count,
        "signature_fact_changes": changes,
        "world_audits": "carried from source base; no training targets generated",
    })
    return raw, proof


def main(output_dir: Path = OUT) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    raw, proof = build()
    atomic_write(output_dir / "source.cases.jsonl", raw)
    proof["source_cases_sha256"] = hashlib.sha256(raw).hexdigest()
    atomic_write(output_dir / "source-variant-proof.json", (json.dumps(proof, indent=2, ensure_ascii=False) + "\n").encode())
    manifest = {
        "schema": "natlang.neuralese-static-depth-variant-source/1",
        "variant": VARIANT,
        "base_source": str(BASE.relative_to(ROOT)),
        "base_source_sha256": hashlib.sha256(BASE.read_bytes()).hexdigest(),
        "source_cases": "source.cases.jsonl",
        "source_cases_sha256": hashlib.sha256(raw).hexdigest(),
        "worlds": 24,
        "nested_depth_variant_worlds": 12,
        "unchanged_iterate_control_worlds": 12,
        "same_source_groups_and_splits": True,
        "static_reference_only": True,
        "model_calls": 0,
        "provider_calls": 0,
        "teacher_trajectories": 0,
        "training_admission_granted": False,
        "registry_changed": False,
        "source_realism_role_revisions": 17,
    }
    atomic_write(output_dir / "source-variant-manifest.json", (json.dumps(manifest, indent=2) + "\n").encode())
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
