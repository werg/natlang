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
    expected_new_rows = screen_output.get("rows")
    if type(expected_new_rows) is not int or expected_new_rows <= 0:
        raise ValueError("screened positive graph row count is absent or invalid")

    # The screen receipt binds the complete per-export receipt/output list and
    # its current-protection pass. Cross-check its published positive counts
    # against the coverage summary and the exact source-family split below;
    # this avoids rereading hundreds of already hash-verified external files.
    per_source = coverage.get("per_source") or {}
    positive_by_source = {name: item.get("reason:graphs:train", 0)
        for name, item in per_source.items() if item.get("reason:graphs:train", 0)}
    if (not positive_by_source or any(name not in {"D13", "D14"} for name in positive_by_source)
            or sum(positive_by_source.values()) != expected_new_rows):
        raise ValueError("coverage positive-train inventory does not match screened D13/D14 rows")
    if any(per_source[name].get("reason:positive_sessions", 0) != count
            for name, count in positive_by_source.items()):
        raise ValueError("source-positive session totals differ from their output graph totals")
    if sum(entry.get("counts", {}).get("graphs:train", 0) for entry in coverage.get("receipts", [])) != expected_new_rows:
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
    if len(rows) != expected_new_rows or source_domains != Counter(positive_by_source):
        raise ValueError("screened graph rows do not match complete positive source-domain totals")
    return screen_path, screen, coverage_path, coverage, rows, protected, builder_path


def load_prepared_rows(derivation_receipt_path: Path, expected_protection: str, *, derivation_program: Path | None = None):
    """Load an independently verified, immutable acquired-graph derivation."""
    derivation_receipt_path = Path(derivation_receipt_path).resolve(strict=True)
    derivation = read_json(derivation_receipt_path)
    if (derivation.get("schema") != "natlang.acquired-bash-context-derivation/1"
            or derivation.get("status") != "derived; not source admission or learner exposure"
            or derivation.get("training_admission") is True):
        raise ValueError("prepared source is not a supported non-admitted derivation")
    derivation_program = (derivation_program or Path(derivation["derivation_program"]["path"])).resolve(strict=True)
    if sha256_file_hex(derivation_program) != derivation["derivation_program"]["sha256"]:
        raise ValueError("prepared source derivation program differs from its receipt")
    verification_path = derivation_receipt_path.parent / "independent-verification.json"
    verification = read_json(verification_path)
    if (verification.get("schema") != "natlang.acquired-bash-context-independent-verification/1"
            or verification.get("status") != "verified derivation integrity; not source admission or learner exposure"
            or verification.get("derivation_receipt", {}).get("sha256") != sha256_file_hex(derivation_receipt_path)
            or verification.get("current_protected_worlds_digest") != expected_protection):
        raise ValueError("prepared source lacks the exact current independent verification")
    graph_path = Path(derivation["output"]["path"]).resolve(strict=True)
    if (sha256_file_hex(graph_path) != derivation["output"]["sha256"]
            or verification.get("derived_graph_artifact", {}).get("sha256") != sha256_file_hex(graph_path)):
        raise ValueError("prepared graph source differs from its derivation/verification receipt")
    source_screen_ref = derivation["source_screen"]
    screen_path = Path(source_screen_ref["path"]).resolve(strict=True)
    if sha256_file_hex(screen_path) != source_screen_ref["sha256"]:
        raise ValueError("prepared source's original screen receipt changed")
    screen = read_json(screen_path)
    if (screen.get("schema") != "natlang.acquired-current-protection-screen/1"
            or screen.get("protected_worlds", {}).get("identity_set_sha256") != expected_protection):
        raise ValueError("prepared source screen no longer binds current protection")
    screened_input = Path(derivation["source_graph_artifact"]["path"]).resolve(strict=True)
    if (sha256_file_hex(screened_input) != derivation["source_graph_artifact"]["sha256"]
            or screen.get("output", {}).get("sha256") != derivation["source_graph_artifact"]["sha256"]):
        raise ValueError("prepared source's screened graph input changed")
    protection = ProtectedWorlds.load()
    if protection.digest != expected_protection:
        raise ValueError("current full protection union changed")
    tokenizer = derivation.get("tokenizer_fingerprint")
    if tokenizer != screen.get("output", {}).get("tokenizer_fingerprint"):
        raise ValueError("prepared source tokenizer differs from its screen")
    if (derivation.get("annotation_contract", {}).get("claimed_original_actor_prompt") is not False
            or derivation.get("annotation_contract", {}).get("claimed_natlang_runtime_tool") is not False
            or derivation.get("annotation_contract", {}).get("function") != "bash"
            or derivation.get("annotation_contract", {}).get("arguments", {}).get("command") !=
                {"required": True, "type": "string"}):
        raise ValueError("prepared source training-only action context is misdeclared")
    rows, ids, groups = [], set(), {}
    source_domains, positive_target_tokens, forward_tokens = Counter(), 0, 0
    with open_jsonl_text(graph_path) as stream:
        for line in stream:
            if not line.strip():
                continue
            graph = json.loads(line)
            ident = graph.get("id")
            if not isinstance(ident, str) or not ident or ident in ids:
                raise ValueError("prepared graph source has missing/repeated ID")
            ids.add(ident)
            if (graph.get("schema") != "natlang.neuralese-native-graph/1"
                    or graph.get("split") != "train" or graph.get("tokenizer_fingerprint") != tokenizer
                    or len(graph.get("nodes", [])) != 1):
                raise ValueError("prepared graph schema, split, tokenizer or structure changed")
            training_context = (graph.get("conversion") or {}).get("training_context_augmentation") or {}
            if (training_context.get("schema") != "natlang.acquired-action-format-annotation/1"
                    or training_context.get("source_prompt_claim") is not False
                    or training_context.get("natlang_tool_declaration") is not False
                    or training_context.get("annotation_sha256") != derivation.get("annotation_sha256")):
                raise ValueError("prepared graph lacks the exact static-context annotation")
            lineage = ((graph.get("conversion") or {}).get("acquired_session") or {}).get("source_lineage") or {}
            notes = lineage.get("notes") or {}
            original_context = notes.get("original_cooperation_context") or {}
            store = lineage.get("store")
            domain = "D13" if ":D13:" in ident else "D14" if ":D14:" in ident else None
            expected_store = {"D13": "CooperBench/team-coop",
                "D14": "CooperBench/qwen9b-coop-mini-swe-agent"}.get(domain)
            if (store != expected_store or not isinstance(original_context.get("repo"), str)
                    or not original_context["repo"] or type(original_context.get("task_id")) not in (str, int)
                    or not any(group.startswith("cooper-task:") for group in graph.get("source_groups", []))):
                raise ValueError("prepared graph lacks source-backed coding-domain provenance")
            example = parse_graph(graph, tokenizer_fingerprint=tokenizer)
            classification = communication_classification(example)
            if classification["communication"] or classification["producer_consumer"] or classification[
                    "external_state_consumption"]:
                raise ValueError("acquired text trajectory is misclassified as native communication")
            row_target_tokens = sum(weight > 0 for weight in example.target_weights)
            if row_target_tokens <= 0:
                raise ValueError("prepared graph has no positive supervised targets")
            positive_target_tokens += row_target_tokens
            forward_tokens += sum(len(node.get("token_ids", [])) for node in graph.get("nodes", []))
            case = {"id": ident, "split": "train", "source_ids": [
                (graph.get("conversion") or {}).get("source_row_id", "")],
                "source_groups": graph.get("source_groups") or [],
                "split_groups": graph.get("source_groups") or []}
            if world_identities(case, evidence_items=False) & protection.identities:
                raise ValueError(f"prepared graph now intersects current protection: {ident}")
            for group in graph.get("source_groups") or []:
                prior_split = groups.setdefault(group, "train")
                if prior_split != "train":
                    raise ValueError("a related invocation group crosses splits")
            origin = {"id": ident, "graph_row_sha256": canonical_json_sha256_hex(graph),
                "source_lineage_store": store, "source_domain": domain,
                "upstream_revision": lineage.get("upstream_revision"),
                "original_source_file": original_context.get("path"),
                "repo": original_context["repo"], "task_id": original_context["task_id"],
                "source_row_id": (graph.get("conversion") or {}).get("source_row_id"),
                "source_groups": graph.get("source_groups") or [], "family": "coding",
                "native_communication": False,
                "training_context_annotation_sha256": training_context.get("annotation_sha256"),
                "source_graph_derivation_sha256": derivation["output"]["sha256"]}
            rows.append((graph, origin, example, classification))
            source_domains[domain] += 1
    if (len(rows) != derivation.get("derived_graph_rows")
            or len(rows) != verification.get("derived_rows")
            or source_domains != Counter(derivation.get("source_counts") or {})
            or sum(source_domains.values()) != verification.get("derived_rows")
            or verification.get("validated_strict_calls") != derivation.get("validated_source_calls")):
        raise ValueError("prepared source inventory differs from its exact derivation receipts")
    coverage_path = Path(screen["coverage_audit"]["path"]).resolve(strict=True)
    if sha256_file_hex(coverage_path) != screen["coverage_audit"]["sha256"]:
        raise ValueError("prepared source's complete conversion coverage audit changed")
    coverage = read_json(coverage_path)
    builder_path = None
    source_evidence = {"mode": "prequalified-training-derivation",
        "derivation_receipt": bind(derivation_receipt_path),
        "independent_verification": bind(verification_path),
        "source_graphs": bind(graph_path), "source_counts": dict(source_domains),
        "strict_calls": verification["validated_strict_calls"],
        "source_action_profiles": derivation.get("call_profiles"),
        "target_projection": derivation.get("training_target_projection")}
    return screen_path, screen, coverage_path, coverage, rows, protection, builder_path, graph_path, tokenizer, source_evidence, positive_target_tokens, forward_tokens


def build(predecessor_dir: Path, screen_path: Path | None, output_dir: Path,
          prepared_graph_derivation: Path | None = None, *, derivation_program: Path | None = None):
    predecessor_dir = Path(predecessor_dir).resolve(strict=True)
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise ValueError("successor outputs are immutable; choose a fresh directory")
    previous_receipt_path = predecessor_dir / "receipt.json"
    previous = read_json(previous_receipt_path)
    previous_source_path = predecessor_dir / "train-sources.json"
    predecessor_rows = previous.get("rows")
    if (type(predecessor_rows) is not int or predecessor_rows <= 0
            or not check_binding(previous_source_path, previous.get("source"))):
        raise ValueError("predecessor is not an exact, bound positive-row source")
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

    if prepared_graph_derivation is not None:
        if screen_path is not None:
            raise ValueError("choose either a source screen or a prepared derivation")
        (screen_path, screen, coverage_path, coverage, qualified_rows, _, screen_builder_path,
            graph_path, tokenizer_fingerprint, source_evidence, source_target_tokens,
            source_forward_tokens) = load_prepared_rows(prepared_graph_derivation, protection.digest, derivation_program=derivation_program)
    else:
        if screen_path is None:
            raise ValueError("a source screen or prequalified derivation is required")
        (screen_path, screen, coverage_path, coverage, qualified_rows, _, screen_builder_path) = load_positive_rows(
            screen_path, protection.digest)
        graph_path = Path(screen["output"]["path"]).resolve(strict=True)
        tokenizer_fingerprint = screen["output"]["tokenizer_fingerprint"]
        source_evidence = {"mode": "current-protection-screen", "screen": bind(screen_path),
            "source_graphs": bind(graph_path)}
        source_target_tokens = sum(sum(weight > 0 for weight in row[2].target_weights)
            for row in qualified_rows)
        source_forward_tokens = sum(sum(len(node.get("token_ids", [])) for node in row[0].get("nodes", []))
            for row in qualified_rows)
    new_rows = len(qualified_rows)
    if new_rows <= 0:
        raise ValueError("source derivation has no qualified graphs")
    graph_binding = bind(graph_path)
    graph_binding["bytes"] = graph_path.stat().st_size
    output_dir.mkdir(parents=True)
    source_path = output_dir / "train-sources.json"
    preserved_screen_builder = None
    if screen_builder_path is not None:
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
            or len(prior_map.get("rows", [])) != predecessor_rows):
        raise ValueError("predecessor family map is not bound to its exact source")
    prior_geometry = read_json(old_geometry_path)
    tokenizer = prior_geometry["identity"]["tokenizer"]
    roles = prior_geometry["identity"]["roles"]
    if tokenizer != tokenizer_fingerprint:
        raise ValueError("acquired source tokenizer differs from current training source")

    new_source, _, geometry_ref, geometry_scan = capacity.prepare_source(
        source_path, tokenizer, roles, output_dir, prior=previous["geometry"])
    if len(new_source.entries) != predecessor_rows + new_rows:
        raise ValueError("successor source row total differs from predecessor plus qualified source graphs")
    prior_pairs = [(row["id"], row["binding"]) for row in prior_map["rows"]]
    if [(entry["id"], entry["binding"]) for entry in new_source.entries[:predecessor_rows]] != prior_pairs:
        raise ValueError("source extension changed the exact predecessor row prefix")

    existing = {row["id"] for row in prior_map["rows"]}
    graph_source = NativeGraphSource(graph_path, tokenizer, required_split="train", role_ids=roles)
    if len(graph_source.entries) != new_rows:
        raise ValueError("positive graph artifact does not independently index to its qualified row count")
    family_rows, origin_rows = [], []
    for graph, origin, example, classification in qualified_rows:
        if graph["id"] in existing:
            raise ValueError("acquired graph ID already exists in the predecessor source")
        existing.add(graph["id"])
        binding_value = graph_source.bindings[graph["id"]]
        family_rows.append({"id": graph["id"], "binding": binding_value, "family": "coding"})
        origin_rows.append(origin)
    append_pairs = [(entry["id"], entry["binding"]) for entry in new_source.entries[predecessor_rows:]]
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
    for index, example in enumerate(new_source.iter_from(predecessor_rows), predecessor_rows):
        if example.id != new_source.entries[index]["id"]:
            raise ValueError("source changed while computing appended CE mass")
        amount = sum(example.target_weights) * example.loss_scale
        if not math.isfinite(amount) or amount <= 0:
            zero_rows.append(example.id)
        weighted_mass += amount
        target_tokens += sum(weight > 0 for weight in example.target_weights)
    if zero_rows or source_target_tokens <= 0 or target_tokens <= 0:
        raise ValueError(
            "appended source lacks positive raw supervision or a positive CE-masked target inventory: "
            f"raw_positive_tokens={source_target_tokens}, ce_positive_tokens={target_tokens}, "
            f"zero_effective_rows={len(zero_rows)}")

    qualification_path = output_dir / "source-positive-qualification.json"
    selected_by_source = Counter(row[1]["source_domain"] for row in qualified_rows)
    coverage_by_source = coverage.get("per_source") or {}
    qualification = {"schema": "natlang.acquired-positive-graph-qualification/1",
        "training_admission": False, "current_protection_digest": protection.digest,
        "source_screen": bind(screen_path),
        "screen_builder": bind(preserved_screen_builder) if preserved_screen_builder else None,
        "screen_input_receipts": screen["input_receipts"], "coverage_audit": bind(coverage_path),
        "exports_manifest_sha256": coverage["exports_manifest_sha256"],
        "verified_source_receipts": len(coverage["receipts"]),
        "screened_positive_rows": screen["output"].get("rows"),
        "screen_positive_sessions_by_source": {name: coverage_by_source[name].get("reason:positive_sessions", 0)
            for name in sorted(coverage_by_source) if name in {"D13", "D14"}},
        "selected_graphs_by_source": dict(selected_by_source),
        "prepared_source_evidence": source_evidence,
        "selected_positive_graphs": len(origin_rows),
        "selection_rule": ("append the exact rows from the independently verified training-only action-format derivation; preserve all original context and targets except the 25 mapped empty extra parameters"
            if prepared_graph_derivation else "the screen bound the complete conversion receipt/output inventory and its train graph IDs biject to source-positive D13/D14 totals"),
        "source_success_semantics": "positive_candidate on normalized whole trajectories; converter applies exact source-reported positive label and rejects failed outcomes; positive candidate here is not an independently re-run environment result",
        "family_assignment": {"family": "coding", "evidence": "each row's source_lineage.store is CooperBench/team-coop (D13) or CooperBench/qwen9b-coop-mini-swe-agent (D14), with original_cooperation_context.repo and task_id; source-positive sessions are full coding trajectories; native communication classification is false for every row"},
        "selected_rows": origin_rows,
        "counts": {"selected_graphs": len(origin_rows), "coding": len(family_rows),
            "native_communication": 0, "forward_tokens": source_forward_tokens,
            "raw_positive_target_tokens": source_target_tokens,
            "positive_target_tokens": target_tokens, "effective_weighted_mass": weighted_mass}}
    write_json_durable(qualification_path, qualification)

    result = {"schema": SCHEMA, "predecessor": bind(previous_receipt_path),
        "predecessor_source": bind(previous_source_path), "source": bind(source_path),
        "appended_graphs": graph_binding, "source_positive_qualification": bind(qualification_path),
        "family_map": bind(family_map_path), "ce_policy": bind(ce_path), "recipe": bind(recipe_path),
        "geometry": geometry_ref, "capacity_descriptor": bind(descriptor_path),
        "prior_ce_reuse": bind(Path(previous["prior_ce_reuse"]["path"])),
        "rows": len(new_source.entries), "predecessor_rows": predecessor_rows,
        "new_rows": new_rows, "new_families": ["coding"] * new_rows,
        "new_effective_weighted_mass": weighted_mass,
        "raw_positive_target_tokens": source_target_tokens, "positive_target_tokens": target_tokens,
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
    parser.add_argument("--screen-receipt")
    parser.add_argument("--prepared-graph-derivation")
    parser.add_argument("--derivation-program", type=Path, help="Exact archived program for a historical derivation; recorded SHA stays mandatory")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if bool(args.screen_receipt) == bool(args.prepared_graph_derivation):
        parser.error("provide exactly one of --screen-receipt or --prepared-graph-derivation")
    print(json.dumps(build(Path(args.predecessor_dir),
        Path(args.screen_receipt) if args.screen_receipt else None,
        Path(args.output_dir),
        Path(args.prepared_graph_derivation) if args.prepared_graph_derivation else None,
        derivation_program=args.derivation_program),
        indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
