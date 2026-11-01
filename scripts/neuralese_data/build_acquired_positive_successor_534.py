#!/usr/bin/env python3
"""Stage an authenticated positive acquired-session graph append for native SFT.

This creates a source/CE/recipe/geometry continuation only. It does not publish
the source to a learner or claim optimizer exposure.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import os
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
EXPECTED_SCREEN_ROWS = 760
EXPECTED_NEW_ROWS = 534
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


def load_positive_rows(screen_path, expected_protection, filtered_path, source_tokenizer):
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
    if screen_output.get("rows") != EXPECTED_SCREEN_ROWS:
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
    if sum(entry.get("counts", {}).get("graphs:train", 0) for entry in coverage.get("receipts", [])) != EXPECTED_SCREEN_ROWS:
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
    ids = set()
    groups = {}
    screened_domains = Counter()
    tokenizer = screen_output["tokenizer_fingerprint"]
    marker_ids = source_tokenizer.encode("<tool_call>", add_special_tokens=False)
    if not marker_ids:
        raise ValueError("the pinned source tokenizer does not encode the tool-call marker")
    from natlang_neuralese.train.own_output_supervision import label_roles, role_tokens_from_tokenizer
    role_tokens = role_tokens_from_tokenizer(source_tokenizer, tokenizer_fingerprint=tokenizer)
    assistant_roles = {4, 5}  # text_supervision.ROLE_CODES assistant reasoning/reply
    user_pattern = re.compile(r"<\|im_start\|>user\n(.*?)(?=<\|im_start\||\Z)", re.S)
    shell_fence_pattern = re.compile(r"```(?:bash|sh|shell)\b", re.I)
    selected_origins, quarantined_rows = [], []
    context_audit, selected_context_audit = Counter(), Counter()
    selected_by_source = Counter()
    selected_forward_tokens = selected_raw_positive_weight_positions = 0
    selected_assistant_target_tokens = selected_failed_masked_tokens = 0
    total_rows = 0
    filtered_path = Path(filtered_path)
    with filtered_path.open("xb") as output, open_jsonl_text(graph_path) as stream:
        for line in stream:
            if not line.strip():
                continue
            graph = json.loads(line)
            total_rows += 1
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
            screened_domains[origin_domain] += 1
            if origin_domain not in ("D13", "D14"):
                raise ValueError("positive source graph lacks an authenticated D13/D14 origin")
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
            token_ids = graph["nodes"][0]["token_ids"]
            labels = label_roles(token_ids, role_tokens)
            call_offsets = [i for i in range(len(token_ids) - len(marker_ids) + 1)
                if token_ids[i:i + len(marker_ids)] == marker_ids and labels[i] in assistant_roles]
            rendered = source_tokenizer.decode(token_ids, skip_special_tokens=False,
                clean_up_tokenization_spaces=False)
            user_messages = user_pattern.findall(rendered)
            if not user_messages or not any(message.strip() for message in user_messages):
                raise ValueError(f"source graph {ident} has no rendered user task context")
            user_text = "\n".join(user_messages)
            user_context = {"user_message_count": len(user_messages),
                "user_text_sha256": __import__("hashlib").sha256(user_text.encode("utf-8")).hexdigest(),
                "source_bash_fenced_context_count": len(shell_fence_pattern.findall(user_text)),
                "has_submission_path": "/workspace/repo/patch.txt" in user_text,
                "has_source_shell_harness_text": bool(re.search(
                    r"available shell commands|submission protocol|\bbash\b|\bshell\b|terminal", user_text, re.I))}
            context_audit["graphs_with_user_task_context"] += 1
            context_audit["graphs_with_source_shell_harness_text"] += user_context["has_source_shell_harness_text"]
            context_audit["graphs_with_bash_fenced_context"] += user_context["source_bash_fenced_context_count"] > 0
            context_audit["user_bash_fenced_contexts"] += user_context["source_bash_fenced_context_count"]
            if call_offsets:
                quarantined_rows.append({"id": ident, "graph_row_sha256": origin["graph_row_sha256"],
                    "source_domain": origin["source_domain"],
                    "reason": "assistant-tool-call-target-without-source-rendered-tool-schema",
                    "assistant_tool_call_marker_offsets": call_offsets,
                    "assistant_tool_call_marker_raw_weights": [graph["nodes"][0]["target_weights"][i]
                        for i in call_offsets]})
            else:
                origin["source_user_context"] = user_context
                selected_origins.append(origin)
                selected_by_source[origin["source_domain"]] += 1
                selected_forward_tokens += len(token_ids)
                selected_raw_positive_weight_positions += sum(weight > 0 for weight in example.target_weights)
                source_supervision = (graph.get("conversion", {}).get("acquired_session", {})
                    .get("supervision_policy") or {})
                selected_assistant_target_tokens += source_supervision.get("assistant_target_tokens", 0)
                selected_failed_masked_tokens += source_supervision.get("failed_assistant_tokens_masked", 0)
                selected_context_audit["graphs_with_user_task_context"] += 1
                selected_context_audit["graphs_with_source_shell_harness_text"] += user_context["has_source_shell_harness_text"]
                selected_context_audit["graphs_with_bash_fenced_context"] += user_context["source_bash_fenced_context_count"] > 0
                selected_context_audit["user_bash_fenced_contexts"] += user_context["source_bash_fenced_context_count"]
                output.write((line if line.endswith("\n") else line + "\n").encode("utf-8"))
        output.flush()
        os.fsync(output.fileno())
    if total_rows != EXPECTED_SCREEN_ROWS or screened_domains != Counter({"D13": 604, "D14": 156}):
        raise ValueError("screened graph source counts differ from the exact source domains")
    if len(selected_origins) != EXPECTED_NEW_ROWS or len(quarantined_rows) != EXPECTED_SCREEN_ROWS - EXPECTED_NEW_ROWS:
        raise ValueError("source tool-protocol filter changed from the audited 534/226 split")
    if selected_context_audit["graphs_with_source_shell_harness_text"] != EXPECTED_NEW_ROWS:
        raise ValueError("an action-free graph lacks source-rendered coding/shell task context")
    if context_audit["graphs_with_user_task_context"] != EXPECTED_SCREEN_ROWS:
        raise ValueError("screened graph rows do not match complete positive source-domain totals")
    return (screen_path, screen, coverage_path, coverage, selected_origins, quarantined_rows,
        protected, builder_path, per_source, marker_ids, selected_by_source,
        selected_forward_tokens, selected_raw_positive_weight_positions,
        selected_assistant_target_tokens, selected_failed_masked_tokens,
        context_audit, selected_context_audit)


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

    from scripts.neuralese_data.build_acquired_full_session_graphs import prepare_tokenizer
    tokenizer_root = "/home/werg/.cache/huggingface/hub/models--LiquidAI--LFM3-1.7B/snapshots/40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd"
    source_tokenizer = prepare_tokenizer(tokenizer_root, history_reasoning_policy="keep").tokenizer
    output_dir.mkdir(parents=True)
    filtered_graph_path = output_dir / "native-graphs.jsonl"
    (screen_path, screen, coverage_path, coverage, qualified_rows, quarantined_rows,
        _, screen_builder_path, per_source, marker_ids, selected_by_source,
        selected_forward_tokens, selected_raw_positive_weight_positions,
        selected_assistant_target_tokens, selected_failed_masked_tokens,
        context_audit, selected_context_audit) = load_positive_rows(
            screen_path, protection.digest, filtered_graph_path, source_tokenizer)
    screened_graph_path = Path(screen["output"]["path"]).resolve(strict=True)
    screened_graph_binding = bind(screened_graph_path)
    screened_graph_binding["bytes"] = screened_graph_path.stat().st_size
    graph_path = filtered_graph_path.resolve(strict=True)
    graph_binding = bind(graph_path)
    protocol_selection_path = output_dir / "source-protocol-selection.json"
    protocol_selection = {
        "schema": "natlang.acquired-positive-protocol-selection/1",
        "current_protection_digest": protection.digest,
        "source_screen": bind(screen_path), "screened_graph_artifact": screened_graph_binding,
        "tokenizer_fingerprint": screen["output"]["tokenizer_fingerprint"],
        "excluded_marker": "<tool_call>", "excluded_marker_token_ids": marker_ids,
        "selection_rule": "exclude whole graphs with an assistant-role <tool_call> marker because rendered source conversations have no tool schema; preserve all other original graph values including full prompts and target weights",
        "selected_graphs": len(qualified_rows), "quarantined_graphs": len(quarantined_rows),
        "selected_by_source": dict(selected_by_source),
        "quarantined_by_source": dict(Counter(row["source_domain"] for row in quarantined_rows)),
        "selected_rows": [{"id": row["id"], "graph_row_sha256": row["graph_row_sha256"],
            "source_user_context": row["source_user_context"]} for row in qualified_rows],
        "quarantined_rows": quarantined_rows,
        "prompt_context_check": {
            "source_harness": "source-rendered user prompts with coding-task and shell/submission instructions retained verbatim",
            "all_screened_have_user_task_context": context_audit["graphs_with_user_task_context"] == EXPECTED_SCREEN_ROWS,
            "all_selected_have_user_task_context": selected_context_audit["graphs_with_user_task_context"] == EXPECTED_NEW_ROWS,
            "selected_per_graph_audit": dict(selected_context_audit),
            "all_selected_have_source_shell_harness_text": selected_context_audit["graphs_with_source_shell_harness_text"] == EXPECTED_NEW_ROWS,
            "no_assistant_tool_calls_in_selected_rows": True,
            "no_synthetic_natlang_tool_schema": True}}
    write_json_durable(protocol_selection_path, protocol_selection)
    del source_tokenizer
    import gc
    gc.collect()
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
        raise ValueError("successor source does not contain exactly the 534 protocol-qualified graphs")
    prior_pairs = [(row["id"], row["binding"]) for row in prior_map["rows"]]
    if [(entry["id"], entry["binding"]) for entry in new_source.entries[:EXPECTED_ROWS]] != prior_pairs:
        raise ValueError("source extension changed the exact predecessor row prefix")

    existing = {row["id"] for row in prior_map["rows"]}
    graph_source = NativeGraphSource(graph_path, tokenizer, required_split="train", role_ids=roles)
    if len(graph_source.entries) != EXPECTED_NEW_ROWS:
        raise ValueError("protocol-qualified graph artifact does not independently index to 534 rows")
    family_rows, origin_rows = [], []
    for origin in qualified_rows:
        if origin["id"] in existing:
            raise ValueError("acquired graph ID already exists in the predecessor source")
        existing.add(origin["id"])
        binding_value = graph_source.bindings[origin["id"]]
        family_rows.append({"id": origin["id"], "binding": binding_value, "family": "coding"})
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
    if zero_rows:
        raise ValueError("one or more appended rows lacks positive supervised CE mass")

    qualification_path = output_dir / "source-positive-qualification.json"
    qualification = {"schema": "natlang.acquired-positive-graph-qualification/1",
        "training_admission": False, "current_protection_digest": protection.digest,
        "source_screen": bind(screen_path), "screen_builder": bind(preserved_screen_builder),
        "protocol_selection": bind(protocol_selection_path),
        "screen_input_receipts": screen["input_receipts"], "coverage_audit": bind(coverage_path),
        "exports_manifest_sha256": coverage["exports_manifest_sha256"],
        "verified_source_receipts": len(coverage["receipts"]),
        "positive_sessions_by_source": dict(selected_by_source),
        "screened_positive_sessions_by_source": {name: per_source[name]["reason:positive_sessions"]
            for name in ("D13", "D14")},
        "selected_positive_graphs": len(origin_rows), "quarantined_for_missing_tool_schema": len(quarantined_rows),
        "screened_positive_graphs": EXPECTED_SCREEN_ROWS,
        "selection_rule": "screen binds complete conversion receipt inventory; positive source graphs with assistant tool-call actions are preserved in screen artifact but quarantined here because the source renderer omitted a tool schema; remaining 534 original graphs are intact",
        "source_success_semantics": "positive_candidate on normalized whole trajectories; converter applies exact source-reported positive label and rejects failed outcomes; positive candidate here is not an independently re-run environment result",
        "family_assignment": {"family": "coding", "evidence": "each row's source_lineage.store is CooperBench/team-coop (D13) or CooperBench/qwen9b-coop-mini-swe-agent (D14), with original_cooperation_context.repo and task_id; source-positive sessions are exact full coding trajectories; native communication classification is false for every row"},
        "selected_rows": origin_rows,
        "counts": {"selected_graphs": len(origin_rows), "coding": len(family_rows),
            "native_communication": 0,
            "forward_tokens": selected_forward_tokens,
            "failed_assistant_tokens_masked": selected_failed_masked_tokens,
            "positive_ce_weight_positions": target_tokens,
            "source_assistant_target_tokens": selected_assistant_target_tokens,
            "source_raw_positive_weight_positions": selected_raw_positive_weight_positions,
            "effective_weighted_mass": weighted_mass,
            "quarantined_graphs": len(quarantined_rows)}}
    write_json_durable(qualification_path, qualification)

    result = {"schema": SCHEMA, "predecessor": bind(previous_receipt_path),
        "predecessor_source": bind(previous_source_path), "source": bind(source_path),
        "appended_graphs": graph_binding, "protocol_selection": bind(protocol_selection_path),
        "source_positive_qualification": bind(qualification_path),
        "family_map": bind(family_map_path), "ce_policy": bind(ce_path), "recipe": bind(recipe_path),
        "geometry": geometry_ref, "capacity_descriptor": bind(descriptor_path),
        "prior_ce_reuse": bind(Path(previous["prior_ce_reuse"]["path"])),
        "rows": len(new_source.entries), "predecessor_rows": EXPECTED_ROWS,
        "new_rows": EXPECTED_NEW_ROWS, "screened_positive_rows": EXPECTED_SCREEN_ROWS,
        "quarantined_tool_call_rows": len(quarantined_rows), "new_families": ["coding"] * EXPECTED_NEW_ROWS,
        "new_effective_weighted_mass": weighted_mass,
        "positive_ce_weight_positions": target_tokens,
        "source_assistant_target_tokens": selected_assistant_target_tokens,
        "source_raw_positive_weight_positions": selected_raw_positive_weight_positions,
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
