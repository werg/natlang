#!/usr/bin/env python3
"""Stage an authenticated positive acquired-session graph append for native SFT.

This creates a source/CE/recipe/geometry continuation only. It does not publish
the source to a learner or claim optimizer exposure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
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

    # The protection screen's aggregate binds every receipt and output artifact;
    # reread only the positive train artifacts to bind each included row to its
    # exact per-export outcome and current converter receipt.
    entries = coverage.get("receipts") or []
    positive_receipts = {}
    # screen receipt v1 records an aggregate over sorted output bindings. Match
    # positive rows using the complete coverage inventory, which the screen
    # verified and hash-bound, rather than trusting an unbound list here.
    for entry in entries:
        if not entry.get("counts", {}).get("positive_sessions"):
            continue
        receipt_path = Path(entry["receipt"]).resolve(strict=True)
        raw = receipt_path.read_bytes()
        receipt_sha = hashlib.sha256(raw).hexdigest()
        if receipt_sha != entry.get("receipt_sha256"):
            raise ValueError("a positive conversion receipt changed after the screen")
        receipt = json.loads(raw)
        if receipt.get("training_admission") is not False:
            raise ValueError("an acquired conversion receipt claims training admission")
        export_ref = receipt.get("export_receipt") or {}
        export_path = Path(export_ref.get("path", "")).resolve(strict=True)
        if sha256_file_hex(export_path) != export_ref.get("sha256"):
            raise ValueError("source export receipt changed")
        outputs = [out for out in receipt.get("outputs", []) if out.get("rows", 0) > 0]
        if sum(out["rows"] for out in outputs) != receipt["counts"].get("graphs:train", 0):
            raise ValueError("positive source receipt output count differs from its train graph count")
        if receipt["counts"].get("positive_sessions", 0) <= 0:
            raise ValueError("train graphs lack a source-positive session count")
        for output in outputs:
            path = Path(output["path"]).resolve(strict=True)
            if (path.parts[-2] != "train" or path.stat().st_size != output["bytes"]
                    or sha256_file_hex(path) != output["sha256"]):
                raise ValueError("positive source output differs from its receipt")
            with open_jsonl_text(path) as stream:
                found = 0
                for line in stream:
                    if not line.strip():
                        continue
                    graph = json.loads(line)
                    found += 1
                    ident = graph.get("id")
                    if ident in positive_receipts:
                        raise ValueError("positive graph identity occurs in two acquired outputs")
                    positive_receipts[ident] = {
                        "source_receipt": {"path": str(receipt_path), "sha256": receipt_sha},
                        "source_export_receipt": export_ref,
                        "source": receipt.get("source"),
                        "output": output,
                        "row_sha256": canonical_json_sha256_hex(graph),
                        "source_reported_positive_sessions": receipt["counts"]["positive_sessions"],
                        "source_graphs_train": receipt["counts"]["graphs:train"],
                        "source_dataset": receipt.get("source", {}).get("source_dataset"),
                    }
                if found != output["rows"]:
                    raise ValueError("positive source output row count differs from receipt")

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
            source = positive_receipts.get(ident)
            if source is None or source["row_sha256"] != canonical_json_sha256_hex(graph):
                raise ValueError("screened graph is not bound to a source-reported-positive output row")
            if (graph.get("schema") != "natlang.neuralese-native-graph/1"
                    or graph.get("split") != "train"
                    or graph.get("tokenizer_fingerprint") != tokenizer):
                raise ValueError("screened graph schema, split or tokenizer changed")
            lineage = ((graph.get("conversion") or {}).get("acquired_session") or {}).get("source_lineage") or {}
            if (lineage.get("store") != "CooperBench/qwen9b-coop-mini-swe-agent"
                    or not any(group.startswith("repo:") for group in graph.get("source_groups", []))
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
            rows.append((graph, source, example, cls))
    if len(rows) != EXPECTED_NEW_ROWS or ids != set(positive_receipts):
        raise ValueError("screened graphs are not exactly the complete source-positive output set")
    return screen_path, screen, coverage_path, coverage, rows, protected


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

    screen_path, screen, coverage_path, coverage, qualified_rows, _ = load_positive_rows(
        screen_path, protection.digest)
    graph_path = Path(screen["output"]["path"]).resolve(strict=True)
    graph_binding = bind(graph_path)
    graph_binding["bytes"] = graph_path.stat().st_size
    output_dir.mkdir(parents=True)
    source_path = output_dir / "train-sources.json"
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
        origin_rows.append({"id": graph["id"], "graph_row_sha256": origin["row_sha256"],
            "source_receipt": origin["source_receipt"], "source_export_receipt": origin["source_export_receipt"],
            "source_output": origin["output"], "source_positive_sessions": origin["source_reported_positive_sessions"],
            "source_graphs_train": origin["source_graphs_train"],
            "source_lineage_store": "CooperBench/qwen9b-coop-mini-swe-agent",
            "family": "coding", "native_communication": False})
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
        "source_screen": bind(screen_path), "coverage_audit": bind(coverage_path),
        "exports_manifest_sha256": coverage["exports_manifest_sha256"],
        "verified_source_receipts": len(coverage["receipts"]),
        "selected_positive_graphs": len(origin_rows),
        "selection_rule": "only full train graphs from conversion receipts with source-reported positive_sessions; failed/non-positive rows and held outputs are excluded",
        "source_success_semantics": "positive_candidate on normalized whole trajectories; converter applies exact source-reported positive label and rejects failed outcomes; positive candidate here is not an independently re-run environment result",
        "family_assignment": {"family": "coding", "evidence": "every selected graph binds CooperBench/qwen9b-coop-mini-swe-agent source lineage, repo and cooper-task source groups, and a source-reported-positive full session; native communication classification was false for every row"},
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
