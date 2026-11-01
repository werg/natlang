#!/usr/bin/env python3
"""Stage an authenticated positive acquired-session graph append for native SFT.

This creates a source/CE/recipe/geometry continuation only. It does not publish
the source to a learner or claim optimizer exposure.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "training/neuralese"))
sys.path.insert(0, str(REPO_ROOT))

from natlang_neuralese.common.custody import ProtectedWorlds, world_identities
from natlang_neuralese.common.hashing import (
    canonical_json_sha256_hex, sha256_file_hex,
)
from natlang_neuralese.common.jsonio import open_jsonl_text, write_json_durable
from natlang_neuralese.data.native_graph_schema import parse_graph, communication_classification
from natlang_neuralese.data.native_sources import NativeGraphSource
from natlang_neuralese.train import native_packed_capacity as capacity
from natlang_neuralese.train.native_resident import binding
from natlang_neuralese.train.native_resident_ce import extension_policy
from natlang_neuralese.train.native_supervision import load_policy
from natlang_neuralese.train.own_output_supervision import prepare_policy, continuation_identity
from natlang_neuralese.train.recipe import load_recipe, resolve_stage_inputs, _data_continuation


EXPECTED_ROWS = 189394
EXPECTED_NEW_ROWS = 760
EXPECTED_PROTECTION = "8ed08f067c6a9715e5f48d73bf1d5ddf5d3c07501ee100c83b361c7ae7407ef7"
SCHEMA = "natlang.acquired-positive-graph-successor/1"


def read_json(path):
    return json.loads(Path(path).read_bytes())


def bind(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": sha256_file_hex(path), "bytes": path.stat().st_size}


def manifest_bind(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": sha256_file_hex(path)}


def check_binding(path, ref):
    path = Path(path).resolve(strict=True)
    return (isinstance(ref, dict) and Path(ref.get("path", "")).resolve() == path
            and ref.get("sha256") == sha256_file_hex(path)
            and ("bytes" not in ref or ref["bytes"] == path.stat().st_size))


def load_positive_rows(screen_path, expected_protection):
    screen_path = Path(screen_path).resolve(strict=True)
    screen = read_json(screen_path)
    if screen.get("schema") != "natlang.acquired-current-protection-screen/1":
        raise ValueError("source screen receipt schema is not recognized")
    protection = screen.get("protected_worlds") or {}
    if (protection.get("identity_set_sha256") != expected_protection
            or protection.get("matches_expected") is not True):
        raise ValueError("source screen was not checked against the current expected protection union")
    coverage_ref = screen["coverage_audit"]
    coverage_path = Path(coverage_ref["path"]).resolve(strict=True)
    if sha256_file_hex(coverage_path) != coverage_ref["sha256"]:
        raise ValueError("source coverage audit changed after the protection screen")
    coverage = read_json(coverage_path)
    if (coverage.get("schema") != "natlang.acquired-session-conversion-coverage-audit/1"
            or coverage.get("exports_manifest_sha256") != coverage_ref.get("source_manifest_sha256")
            or coverage.get("complete_receipts") != coverage_ref.get("complete_receipts")):
        raise ValueError("screen does not bind the current complete conversion receipt inventory")
    screen_output = screen["output"]
    graph_path = Path(screen_output["path"]).resolve(strict=True)
    if sha256_file_hex(graph_path) != screen_output["sha256"]:
        raise ValueError("screened positive graph file changed")
    if screen_output.get("rows") != EXPECTED_NEW_ROWS:
        raise ValueError("screened positive graph row count changed")

    # The screen receipt binds the complete per-export receipt/output list and
    # its current-protection pass. Cross-check its published positive counts
    # against the coverage summary and the exact source-family split below;
    # this avoids rereading hundreds of already hash-verified external files.
    per_source = coverage.get("per_source") or {}
    positive_by_source = {name: item.get("reason:graphs:train", 0)
        for name, item in per_source.items() if item.get("reason:graphs:train", 0)}
    if positive_by_source != {"D13": 604, "D14": 156}:
        raise ValueError("coverage positive-train inventory is not the screened D13/D14 set")
    if any(per_source[name].get("reason:positive_sessions", 0) != count
            for name, count in positive_by_source.items()):
        raise ValueError("source-positive session totals differ from their output graph totals")
    if sum(entry.get("counts", {}).get("graphs:train", 0) for entry in coverage.get("receipts", [])) != EXPECTED_NEW_ROWS:
        raise ValueError("receipt inventory's total positive train graph count differs")
    input_receipts = screen.get("input_receipts") or {}
    if (input_receipts.get("count") != sum(len(item.get("outputs", []))
            for item in coverage.get("receipts", []))
            or not isinstance(input_receipts.get("sha256"), str)
            or len(input_receipts["sha256"]) != 64):
        raise ValueError("protection screen does not bind the full input receipt/output inventory")
    builder_path = Path("/tmp/screen_acquired_current_protection.py")
    if not builder_path.is_file():
        raise ValueError("the exact current-protection screen builder source is unavailable")

    protected = ProtectedWorlds.load()
    if protected.digest != expected_protection:
        raise ValueError("current ProtectedWorlds changed while preparing the append")
    rows = []
    ids = set()
    groups = {}
    tokenizer = screen_output["tokenizer_fingerprint"]
    with open_jsonl_text(graph_path) as stream:
        for line in stream:
            if not line.strip():
                continue
            graph = json.loads(line)
            ident = graph.get("id")
            if not isinstance(ident, str) or not ident or ident in ids:
                raise ValueError("screened graph has missing or repeated identity")
            ids.add(ident)
            if (graph.get("schema") != "natlang.neuralese-native-graph/1"
                    or graph.get("split") != "train"
                    or graph.get("tokenizer_fingerprint") != tokenizer):
                raise ValueError("screened graph schema, split or tokenizer changed")
            lineage = ((graph.get("conversion") or {}).get("acquired_session") or {}).get("source_lineage") or {}
            notes = lineage.get("notes") or {}
            original_context = notes.get("original_cooperation_context") or {}
            store = lineage.get("store")
            origin_domain = "D13" if ":D13:" in ident else "D14" if ":D14:" in ident else None
            expected_store = {"D13": "CooperBench/team-coop",
                "D14": "CooperBench/qwen9b-coop-mini-swe-agent"}.get(origin_domain)
            if (store != expected_store or not isinstance(original_context.get("repo"), str)
                    or not original_context["repo"] or type(original_context.get("task_id")) not in (str, int)
                    or not any(group.startswith("cooper-task:") for group in graph.get("source_groups", []))):
                raise ValueError("positive source does not have source-backed coding-domain provenance")
            example = parse_graph(graph, tokenizer_fingerprint=tokenizer)
            cls = communication_classification(example)
            if cls["communication"] or cls["producer_consumer"] or cls["external_state_consumption"]:
                raise ValueError("acquired text trajectory was misclassified as native communication")
            if sum(weight > 0 for weight in example.target_weights) <= 0:
                raise ValueError("screened positive graph has no supervised target tokens")
            case = {"id": ident, "split": "train", "source_ids": [
                (graph.get("conversion") or {}).get("source_row_id", "")],
                "source_groups": graph.get("source_groups") or [],
                "split_groups": graph.get("source_groups") or []}
            matches = sorted(world_identities(case, evidence_items=False) & protected.identities)
            if any(kind in {"id", "group", "binding"} for kind, _ in matches):
                raise ValueError("screened positive source now intersects current protection")
            for group in graph.get("source_groups") or []:
                previous = groups.setdefault(group, "train")
                if previous != "train":
                    raise ValueError("one related world crosses split boundaries")
            origin = {"id": ident, "graph_row_sha256": canonical_json_sha256_hex(graph),
                "source_lineage_store": store, "source_domain": origin_domain,
                "upstream_revision": lineage.get("upstream_revision"),
                "original_source_file": original_context.get("path"),
                "repo": original_context["repo"], "task_id": original_context["task_id"],
                "source_row_id": (graph.get("conversion") or {}).get("source_row_id"),
                "source_groups": graph.get("source_groups") or [], "family": "coding",
                "native_communication": False}
            rows.append((graph, origin, example, cls))
    source_domains = Counter(row[1]["source_domain"] for row in rows)
    if len(rows) != EXPECTED_NEW_ROWS or source_domains != Counter({"D13": 604, "D14": 156}):
        raise ValueError("screened graph rows do not match complete positive source-domain totals")
    return screen_path, screen, coverage_path, coverage, rows, protected, builder_path


def build(predecessor_dir: Path, screen_path: Path, output_dir: Path):
    predecessor_dir = Path(predecessor_dir).resolve(strict=True)
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise ValueError("successor outputs are immutable; choose a fresh directory")
    previous_receipt_path = predecessor_dir / "receipt.json"
    previous = read_json(previous_receipt_path)
    previous_source_path = predecessor_dir / "train-sources.json"
    if (previous.get("rows") != EXPECTED_ROWS
            or not check_binding(previous_source_path, previous.get("source"))):
        raise ValueError("predecessor is not the exact bound 189394-row source")
    if not check_binding(predecessor_dir / "receipt.json", bind(previous_receipt_path)):
        raise ValueError("predecessor receipt could not be freshly verified")
    old_map_path = Path(previous["family_map"]["path"]).resolve(strict=True)
    old_geometry_path = Path(previous["geometry"]["path"]).resolve(strict=True)
    old_ce_path = Path(previous["ce_policy"]["path"]).resolve(strict=True)
    old_recipe_path = Path(previous["recipe"]["path"]).resolve(strict=True)
    for ref, path, name in ((previous["family_map"], old_map_path, "family map"),
                            (previous["geometry"], old_geometry_path, "packed geometry"),
                            (previous["ce_policy"], old_ce_path, "CE policy"),
                            (previous["recipe"], old_recipe_path, "recipe")):
        if not check_binding(path, ref):
            raise ValueError(f"predecessor {name} changed")
    protection = ProtectedWorlds.load()
    if (protection.digest != EXPECTED_PROTECTION
            or previous.get("current_protection_digest") != protection.digest):
        raise ValueError("predecessor and current full protection union do not match")

    screen_path, screen, coverage_path, coverage, qualified_rows, _, screen_builder_path = load_positive_rows(
        screen_path, protection.digest)
    graph_path = Path(screen["output"]["path"]).resolve(strict=True)
    graph_binding = bind(graph_path)
    graph_binding["bytes"] = graph_path.stat().st_size
    output_dir.mkdir(parents=True)
    source_path = output_dir / "train-sources.json"
    preserved_screen_builder = output_dir / "screen-builder-source.py"
    shutil.copyfile(screen_builder_path, preserved_screen_builder)
    source_manifest = read_json(previous_source_path)
    if source_manifest.get("schema") != "natlang.native-graph-sources/1":
        raise ValueError("predecessor source manifest schema is unsupported")
    source_manifest["sources"].append({"kind": "native-graphs", **graph_binding, "training_priority": 1})
    write_json_durable(source_path, source_manifest)

    prior_map = read_json(old_map_path)
    if (prior_map.get("schema") != "natlang.logical-family-map/1"
            or prior_map.get("source_sha256") != previous["source"]["sha256"]
            or len(prior_map.get("rows", [])) != EXPECTED_ROWS):
        raise ValueError("predecessor family map is not bound to its exact source")
    prior_geometry = read_json(old_geometry_path)
    tokenizer = prior_geometry["identity"]["tokenizer"]
    roles = prior_geometry["identity"]["roles"]
    if tokenizer != screen["output"]["tokenizer_fingerprint"]:
        raise ValueError("acquired source tokenizer differs from current training source")

    new_source, _, geometry_ref, geometry_scan = capacity.prepare_source(
        source_path, tokenizer, roles, output_dir, prior=previous["geometry"])
    if len(new_source.entries) != EXPECTED_ROWS + EXPECTED_NEW_ROWS:
        raise ValueError("successor source does not contain exactly the 760 screened positive graphs")
    prior_pairs = [(row["id"], row["binding"]) for row in prior_map["rows"]]
    if [(entry["id"], entry["binding"]) for entry in new_source.entries[:EXPECTED_ROWS]] != prior_pairs:
        raise ValueError("source extension changed the exact predecessor row prefix")

    existing = {row["id"] for row in prior_map["rows"]}
    graph_source = NativeGraphSource(graph_path, tokenizer, required_split="train", role_ids=roles)
    if len(graph_source.entries) != EXPECTED_NEW_ROWS:
        raise ValueError("positive graph artifact does not independently index to 760 rows")
    family_rows, origin_rows = [], []
    for graph, origin, example, classification in qualified_rows:
        if graph["id"] in existing:
            raise ValueError("acquired graph ID already exists in the predecessor source")
        existing.add(graph["id"])
        binding_value = graph_source.bindings[graph["id"]]
        family_rows.append({"id": graph["id"], "binding": binding_value, "family": "coding"})
        origin_rows.append(origin)
    append_pairs = [(entry["id"], entry["binding"]) for entry in new_source.entries[EXPECTED_ROWS:]]
    if append_pairs != [(row["id"], row["binding"]) for row in family_rows]:
        raise ValueError("family-map additions differ from the indexed appended graph rows")

    family_map_path = output_dir / "families.json"
    family_map = {"schema": "natlang.logical-family-map/1",
        "source_sha256": sha256_file_hex(source_path), "rows": prior_map["rows"] + family_rows}
    write_json_durable(family_map_path, family_map)

    ce_policy = read_json(old_ce_path)
    ce_policy["inventory"] = [manifest_bind(source_path)]
    ce_policy["objective"]["family_balance"]["map"] = manifest_bind(family_map_path)
    prior_objective = previous["objective_transition"]["to"]
    ce_policy["objective_transition"] = {"from": prior_objective["sha256"]}
    graph_digest = graph_binding["sha256"]
    if any(item.get("sha256") == graph_digest for item in ce_policy["sources"]):
        raise ValueError("acquired graph source is already present in the CE declaration")
    ce_policy["sources"].append({"sha256": graph_digest, "encoding": "native-row-objectives-v1",
        "context_weight": ce_policy["source_contract"]["default_raw_context_weight"]})
    ce_policy["sources"].sort(key=lambda row: row["sha256"])
    ce_policy["source_contract"]["inventory"] = [{**bind(source_path), "bytes": source_path.stat().st_size}]
    active = ce_policy["source_contract"]["active_sources"]
    active.append({**graph_binding, "bytes": graph_binding["bytes"]})
    ce_policy["source_contract"]["active_sources"] = active
    ce_policy["source_contract"]["active_source_count"] = len(active)
    ce_path = output_dir / "ce-policy.json"
    write_json_durable(ce_path, ce_policy)
    policy = prepare_policy(load_policy(bind(ce_path)), new_source)

    recipe = read_json(old_recipe_path)
    recipe["input_bindings"]["v4_previous_train_graphs"] = manifest_bind(previous_source_path)
    recipe["input_bindings"]["v4_train_graphs"] = manifest_bind(source_path)
    recipe["input_bindings"]["v4_ce_policy"] = manifest_bind(ce_path)
    recipe_path = output_dir / "recipe-continuation.json"
    write_json_durable(recipe_path, recipe)
    old_recipe, new_recipe = load_recipe(old_recipe_path), load_recipe(recipe_path)
    stage = lambda value: next(item for item in value["stages"] if item["id"] == "native_joint_sft_v4")
    old_roles = resolve_stage_inputs(old_recipe, stage(old_recipe), {})
    new_roles = resolve_stage_inputs(new_recipe, stage(new_recipe), {})
    if not _data_continuation({k: v for k, v in old_roles.items() if k != "ce_supervision_policy"},
                              {k: v for k, v in new_roles.items() if k != "ce_supervision_policy"}):
        raise ValueError("candidate changes the authenticated SFT role continuation")
    for key in ("native_graphs", "foundation_native_graphs"):
        if (key not in old_roles or key not in new_roles
                or Path(old_roles[key]["path"]).resolve() != previous_source_path
                or Path(new_roles[key]["path"]).resolve() != source_path):
            raise ValueError(f"shared train/foundation role {key} did not move together")
    if old_roles.get("held_native_graphs") != new_roles.get("held_native_graphs"):
        raise ValueError("candidate changes the current protected held source")
    old_train = NativeGraphSource(old_roles["native_graphs"]["path"], tokenizer,
        required_split="train", role_ids=roles)
    extension = extension_policy(old_roles, new_roles, prior_train=old_train, candidate_train=new_source)
    _, objective_transition = continuation_identity(
        {"supervision_objective": prior_objective}, extension)

    descriptor_path = output_dir / "sft-capacity-descriptor.json"
    write_json_durable(descriptor_path, {"schema": "natlang.native-sft-capacity-cases/1",
        "source": bind(source_path), "geometry": geometry_ref})
    new_source.set_ce_policy(policy)
    zero_rows, weighted_mass = [], 0.0
    target_tokens = 0
    for index, example in enumerate(new_source.iter_from(EXPECTED_ROWS), EXPECTED_ROWS):
        if example.id != new_source.entries[index]["id"]:
            raise ValueError("source changed while computing appended CE mass")
        amount = sum(example.target_weights) * example.loss_scale
        if not math.isfinite(amount) or amount <= 0:
            zero_rows.append(example.id)
        weighted_mass += amount
        target_tokens += sum(weight > 0 for weight in example.target_weights)
    if zero_rows or target_tokens != screen["accounting"]["positive_target_tokens"]:
        raise ValueError("one or more appended rows lacks positive supervised CE mass")

    qualification_path = output_dir / "source-positive-qualification.json"
    qualification = {"schema": "natlang.acquired-positive-graph-qualification/1",
        "training_admission": False, "current_protection_digest": protection.digest,
        "source_screen": bind(screen_path), "screen_builder": bind(preserved_screen_builder),
        "screen_input_receipts": screen["input_receipts"], "coverage_audit": bind(coverage_path),
        "exports_manifest_sha256": coverage["exports_manifest_sha256"],
        "verified_source_receipts": len(coverage["receipts"]),
        "positive_sessions_by_source": {name: per_source[name]["reason:positive_sessions"]
            for name in ("D13", "D14")},
        "selected_positive_graphs": len(origin_rows),
        "selection_rule": "the screen bound the complete conversion receipt/output inventory; its train graph IDs biject to the source-positive D13/D14 totals, and no D32 success-only status is treated as semantic correctness",
        "source_success_semantics": "positive_candidate on normalized whole trajectories; converter applies exact source-reported positive label and rejects failed outcomes; positive candidate here is not an independently re-run environment result",
        "family_assignment": {"family": "coding", "evidence": "each row's source_lineage.store is CooperBench/team-coop (D13) or CooperBench/qwen9b-coop-mini-swe-agent (D14), with original_cooperation_context.repo and task_id; source-positive sessions are exact full coding trajectories; native communication classification is false for every row"},
        "selected_rows": origin_rows,
        "counts": {"selected_graphs": len(origin_rows), "coding": len(family_rows),
            "native_communication": 0, "forward_tokens": screen["accounting"]["node_forward_tokens"],
            "positive_target_tokens": target_tokens, "effective_weighted_mass": weighted_mass}}
    write_json_durable(qualification_path, qualification)

    result = {"schema": SCHEMA, "predecessor": bind(previous_receipt_path),
        "predecessor_source": bind(previous_source_path), "source": bind(source_path),
        "appended_graphs": graph_binding, "source_positive_qualification": bind(qualification_path),
        "family_map": bind(family_map_path), "ce_policy": bind(ce_path), "recipe": bind(recipe_path),
        "geometry": geometry_ref, "capacity_descriptor": bind(descriptor_path),
        "prior_ce_reuse": bind(Path(previous["prior_ce_reuse"]["path"])),
        "rows": len(new_source.entries), "predecessor_rows": EXPECTED_ROWS,
        "new_rows": EXPECTED_NEW_ROWS, "new_families": ["coding"] * EXPECTED_NEW_ROWS,
        "new_effective_weighted_mass": weighted_mass, "positive_target_tokens": target_tokens,
        "zero_effective_rows": zero_rows, "geometry_scan": geometry_scan,
        "family_census": policy["_family_census"], "objective_transition": objective_transition,
        "family_balance": ce_policy["objective"]["family_balance"],
        "staging_code": [bind(Path(__file__).resolve()), *[
            bind(Path(__file__).resolve().parents[2] / rel) for rel in (
                "training/neuralese/natlang_neuralese/common/custody.py",
                "training/neuralese/natlang_neuralese/common/hashing.py",
                "training/neuralese/natlang_neuralese/common/jsonio.py",
                "training/neuralese/natlang_neuralese/data/native_source_tree.py",
                "training/neuralese/natlang_neuralese/data/native_sources.py",
                "training/neuralese/natlang_neuralese/data/native_graph_schema.py",
                "training/neuralese/natlang_neuralese/data/decision_graph.py",
                "training/neuralese/natlang_neuralese/model/native_graph_plan.py",
                "training/neuralese/natlang_neuralese/train/native_packed_capacity.py",
                "training/neuralese/natlang_neuralese/train/native_resident.py",
                "training/neuralese/natlang_neuralese/train/native_supervision.py",
                "training/neuralese/natlang_neuralese/train/own_output_supervision.py",
                "training/neuralese/natlang_neuralese/train/native_resident_ce.py",
                "training/neuralese/natlang_neuralese/train/recipe.py")]],
        "builder_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
            check=True, capture_output=True, text=True).stdout.strip(),
        "python_version": sys.version.split()[0],
        "current_protection_digest": protection.digest,
        "roles": {"native_graphs": new_roles["native_graphs"],
            "foundation_native_graphs": new_roles["foundation_native_graphs"],
            "held_native_graphs": new_roles.get("held_native_graphs")},
        "status": "source-ce-recipe-geometry-qualified-not-published",
        "training_admission": False,
        "limitations": ["This is not a learner update or exposure receipt.",
            "Positive labels remain source-reported test outcomes; the environment was not rerun during this append."]}
    receipt_path = output_dir / "receipt.json"
    write_json_durable(receipt_path, result)
    return {"receipt": bind(receipt_path), **result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predecessor-dir", required=True)
    parser.add_argument("--screen-receipt", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(build(Path(args.predecessor_dir), Path(args.screen_receipt), Path(args.output_dir)),
        indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
