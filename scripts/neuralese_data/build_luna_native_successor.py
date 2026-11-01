#!/usr/bin/env python3
"""Stage exact whole-episode Luna graphs as a native SFT source continuation.

The output is a source/CE/geometry successor only. It never publishes a learner
update or marks the graphs admitted by itself.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from natlang_neuralese.common.custody import ProtectedWorlds
from natlang_neuralese.common.hashing import sha256_file_hex, sha256_hex
from natlang_neuralese.common.jsonio import write_json_durable
from natlang_neuralese.data.native_graph_schema import communication_classification
from natlang_neuralese.data.native_sources import NativeGraphSource
from natlang_neuralese.train import native_packed_capacity as capacity
from natlang_neuralese.train.native_supervision import load_policy
from natlang_neuralese.train.own_output_supervision import prepare_policy, continuation_identity
from natlang_neuralese.train.native_resident_ce import extension_policy
from natlang_neuralese.train.recipe import load_recipe, resolve_stage_inputs, _data_continuation


SCHEMA = "natlang.luna-native-source-successor/1"


def _read(path):
    return json.loads(Path(path).read_bytes())


def _bind(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": sha256_file_hex(path), "bytes": path.stat().st_size}


def _manifest_bind(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": sha256_file_hex(path)}


def _matches_binding(path, binding):
    """Verify a predecessor binding without requiring optional byte-count fields."""
    path = Path(path).resolve(strict=True)
    if not isinstance(binding, dict):
        return False
    try:
        bound_path = Path(binding["path"]).resolve(strict=True)
    except (KeyError, OSError, TypeError):
        return False
    return (bound_path == path and binding.get("sha256") == sha256_file_hex(path)
            and ("bytes" not in binding or binding["bytes"] == path.stat().st_size))


def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _load_episode(receipt_path: Path, protection: ProtectedWorlds):
    receipt_path = Path(receipt_path).resolve(strict=True)
    receipt = _read(receipt_path)
    if (receipt.get("schema") != "natlang.luna-native-episode-assembly/1"
            or receipt.get("training_admission") is not False
            or receipt.get("oracle_level") != "exact"
            or receipt.get("exact_oracle_claimed") is not True
            or receipt.get("partial_turns_complete") is not True):
        raise ValueError("candidate episode does not carry a complete exact-oracle assembly receipt")
    graph_path = Path(receipt["graph_path"]).resolve(strict=True)
    graph_bytes = graph_path.read_bytes()
    if sha256_hex(graph_bytes) != receipt.get("graph_sha256"):
        raise ValueError("assembled episode graph changed after its receipt")
    graph_rows = [json.loads(line) for line in graph_bytes.splitlines() if line.strip()]
    if len(graph_rows) != 1:
        raise ValueError("an episode assembly must contain exactly one whole-episode graph")
    graph = graph_rows[0]
    audit_path = Path(receipt["audit_path"]).resolve(strict=True)
    capture_audit_path = Path(receipt["capture_audit_path"]).resolve(strict=True)
    audit = _read(audit_path)
    capture_audit = _read(capture_audit_path)
    if (sha256_file_hex(audit_path) != receipt.get("audit_sha256")
            or sha256_file_hex(capture_audit_path) != receipt.get("capture_audit_sha256")
            or audit.get("training_admission") is not False
            or audit.get("oracle_level") != "exact"
            or audit.get("exact_oracle_claimed") is not True):
        raise ValueError("episode validation/capture audit differs from its exact source binding")
    raw_path = Path(audit["original_result"]["path"]).resolve(strict=True)
    if _bind(raw_path) != audit["original_result"]:
        raise ValueError("exact raw Luna result differs from its provider-source audit")
    raw_rows = _read_jsonl(raw_path)
    if len(raw_rows) != 1 or raw_rows[0].get("id") != audit.get("task_id"):
        raise ValueError("episode audit does not bind one exact provider result")
    raw = raw_rows[0]
    outcome = raw.get("outcome") or {}
    oracle = outcome.get("oracle") or {}
    if (outcome.get("status") != "done" or outcome.get("accepted") is not True
            or oracle.get("accepted") is not True or oracle.get("level") != "exact"):
        raise ValueError("provider source task was not actually accepted by its exact oracle")
    program = (raw.get("task") or {}).get("program_ir") or {}
    custody = protection.check(program)
    if (program.get("split") != "train" or custody.get("split") != "train"
            or graph.get("split") != "train"
            or set(graph.get("source_groups") or []) != set((program.get("source_groups") or [])) | {program.get("id")}):
        raise ValueError("exact provider world and native graph do not share a current protected train disposition")
    if (graph.get("schema") != "natlang.neuralese-native-graph/1"
            or graph.get("id") != "luna-native-episode:" + raw["id"]
            or len(graph.get("nodes") or []) != receipt.get("native_graph_invocations")
            or len(graph.get("teacher_source_receipts") or []) != receipt.get("source_turn_count")):
        raise ValueError("assembled graph omits turns or differs from its exact task identity")
    actor = {"checkpoint": receipt.get("native_checkpoint_sha256"),
        "tokenizer": receipt.get("tokenizer_fingerprint"),
        "runtime": receipt.get("actor_runtime_identity"),
        "prompt": receipt.get("prompt_rendering_identity_sha256"),
        "serving_config": receipt.get("serving_config_identity_sha256")}
    if any(value is None for value in actor.values()):
        raise ValueError("episode receipt lacks its actual actor/runtime/tokenizer identity")
    return {"receipt_path": str(receipt_path), "receipt_binding": _bind(receipt_path),
        "receipt": receipt, "graph_path": str(graph_path), "graph_binding": _bind(graph_path),
        "graph": graph, "audit_binding": _bind(audit_path),
        "capture_audit_binding": _bind(capture_audit_path), "raw_binding": _bind(raw_path),
        "raw": raw, "task_id": audit["task_id"], "program": program,
        "custody": custody, "actor": actor}


def build(*, predecessor_dir: Path, episode_receipts: list[Path], output_dir: Path) -> dict:
    predecessor_dir = Path(predecessor_dir).resolve(strict=True)
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise ValueError("successor output is immutable; choose a fresh directory")
    previous_receipt_path = predecessor_dir / "receipt.json"
    previous = _read(previous_receipt_path)
    previous_source_path = predecessor_dir / "train-sources.json"
    if (not _matches_binding(previous_source_path, previous.get("source"))
            or previous.get("rows") != 189390):
        raise ValueError("predecessor must be the exact current 189390-row source frontier")
    previous_map_path = Path(previous["family_map"]["path"]).resolve(strict=True)
    if not _matches_binding(previous_map_path, previous["family_map"]):
        raise ValueError("predecessor family map changed")
    previous_geometry_path = Path(previous["geometry"]["path"]).resolve(strict=True)
    if not _matches_binding(previous_geometry_path, previous["geometry"]):
        raise ValueError("predecessor packed geometry changed")

    protection = ProtectedWorlds.load()
    if protection.digest != previous.get("protection"):
        raise ValueError("current protection union changed after the predecessor receipt")
    episodes = [_load_episode(path, protection) for path in episode_receipts]
    if len(episodes) != 2 or len({episode["raw"]["id"] for episode in episodes}) != 2:
        raise ValueError("this append requires two distinct complete Luna episode receipts")
    if len({json.dumps(episode["actor"], sort_keys=True) for episode in episodes}) != 1:
        raise ValueError("episodes were not captured by the same actual native student actor/runtime")
    groups = [set(episode["graph"]["source_groups"]) for episode in episodes]
    shared_groups = set.intersection(*groups) if groups and all(groups) else set()
    shared_codebases = {group for group in shared_groups if group.startswith("codebase:")}
    if (not shared_groups or not shared_codebases
            or not any(group.startswith("sql-db:bird:food_inspection") for group in shared_groups)
            or not any(group.startswith("world:luna-relational-inspection-operations-20261101-c-v7:")
                       for group in shared_groups)):
        raise ValueError("related invocation worlds must retain shared codebase, upstream database and train-world custody")

    output_dir.mkdir(parents=True)
    graph_path = output_dir / "luna-v7-native-episodes.jsonl"
    graph_data = b"".join(Path(item["graph_path"]).read_bytes() for item in episodes)
    graph_path.write_bytes(graph_data)
    with graph_path.open("rb") as stream:
        import os
        os.fsync(stream.fileno())
    source_graph = _bind(graph_path)
    qualification = {"schema": "natlang.luna-v7-whole-episode-qualification/1",
        "training_admission": False, "oracle_level": "exact", "episodes": [{
            "trajectory_id": item["raw"]["id"], "task_id": item["task_id"],
            "raw_result": item["raw_binding"], "validation_audit": item["audit_binding"],
            "capture_audit": item["capture_audit_binding"],
            "capture_receipt": item["receipt_binding"], "graph": item["graph_binding"],
            "graph_id": item["graph"]["id"], "nodes": len(item["graph"]["nodes"]),
            "invocations": item["receipt"]["native_graph_invocations"],
            "supervised_turns": item["receipt"]["target_supervised_turns"],
            "context_only_turns": item["receipt"]["context_only_turns"],
            "encoder_count": item["receipt"]["teacher_text_encoder_count"],
            "consumer_actions": item["receipt"]["consumer_action_count"],
            "split": "train", "source_groups": item["graph"]["source_groups"],
            "protected_worlds_receipt": item["custody"],
            "oracle_result": {"status": item["raw"]["outcome"]["status"],
                "accepted": item["raw"]["outcome"]["accepted"],
                "level": item["raw"]["outcome"]["oracle"]["level"],
                "checks": item["raw"]["outcome"]["oracle"].get("checks")}}
            for item in episodes], "native_student_actor": episodes[0]["actor"],
        "current_protection_digest": protection.digest}
    qualification_path = output_dir / "whole-episode-qualification.json"
    write_json_durable(qualification_path, qualification)

    source = _read(previous_source_path)
    source["sources"].append({"kind": "native-graphs", **source_graph, "training_priority": 1})
    source_path = output_dir / "train-sources.json"
    write_json_durable(source_path, source)

    old_map = _read(previous_map_path)
    if len(old_map.get("rows") or []) != 189390:
        raise ValueError("prior logical-family map does not cover the exact 189390-row source")
    prior_geometry = _read(previous_geometry_path)
    tokenizer = prior_geometry["identity"]["tokenizer"]
    roles = prior_geometry["identity"]["roles"]
    new_source, slots, geometry_ref, geometry_scan = capacity.prepare_source(
        source_path, tokenizer, roles, output_dir, prior=previous["geometry"])
    if len(new_source.entries) != 189392:
        raise ValueError("new source does not extend the exact predecessor by two complete graphs")
    old_pairs = [(row["id"], row["binding"]) for row in old_map["rows"]]
    if [(entry["id"], entry["binding"]) for entry in new_source.entries[:189390]] != old_pairs:
        raise ValueError("candidate changes or reorders the existing training source prefix")

    existing = {row["id"] for row in old_map["rows"]}
    additions = []
    graph_source = NativeGraphSource(graph_path, tokenizer, required_split="train", role_ids=roles)
    if len(graph_source.entries) != 2:
        raise ValueError("combined graph file does not contain exactly two whole-episode rows")
    for example in graph_source:
        if example.id in existing:
            raise ValueError("whole-episode graph duplicates an existing logical-family ID")
        existing.add(example.id)
        classification = communication_classification(example)
        family = "native-communication" if classification["communication"] else "natlang"
        additions.append({"id": example.id, "binding": graph_source.bindings[example.id], "family": family})
    if [(row["id"], row["binding"]) for row in new_source.entries[189390:]] != [
            (row["id"], row["binding"]) for row in additions]:
        raise ValueError("appended whole-episode rows differ from the graph source’s indexed identities")

    family_map_path = output_dir / "families.json"
    write_json_durable(family_map_path, {"schema": "natlang.logical-family-map/1",
        "source_sha256": sha256_file_hex(source_path), "rows": old_map["rows"] + additions})
    ce_policy = _read(Path(previous["ce_policy"]["path"]))
    ce_policy["inventory"] = [_manifest_bind(source_path)]
    ce_policy["objective"]["family_balance"]["map"] = _manifest_bind(family_map_path)
    ce_policy["objective_transition"] = {"from": previous["objective_transition"]["to"]["sha256"]}
    ce_policy["sources"].append({"sha256": source_graph["sha256"],
        "encoding": "native-row-objectives-v1",
        "context_weight": ce_policy["source_contract"]["default_raw_context_weight"]})
    ce_policy["sources"].sort(key=lambda row: row["sha256"])
    ce_policy["source_contract"]["inventory"] = [{**_bind(source_path), "bytes": source_path.stat().st_size}]
    active = ce_policy["source_contract"]["active_sources"]
    active.append({**source_graph, "bytes": source_graph["bytes"]})
    ce_policy["source_contract"]["active_sources"] = active
    ce_policy["source_contract"]["active_source_count"] = len(active)
    ce_policy_path = output_dir / "ce-policy.json"
    write_json_durable(ce_policy_path, ce_policy)
    policy = prepare_policy(load_policy(_bind(ce_policy_path)), new_source)

    recipe = _read(predecessor_dir / "recipe-continuation.json")
    recipe["input_bindings"]["v4_previous_train_graphs"] = _manifest_bind(previous_source_path)
    recipe["input_bindings"]["v4_train_graphs"] = _manifest_bind(source_path)
    recipe["input_bindings"]["v4_ce_policy"] = _manifest_bind(ce_policy_path)
    recipe_path = output_dir / "recipe-continuation.json"
    write_json_durable(recipe_path, recipe)
    old_recipe = load_recipe(predecessor_dir / "recipe-continuation.json")
    new_recipe = load_recipe(recipe_path)
    stage = lambda value: next(item for item in value["stages"] if item["id"] == "native_joint_sft_v4")
    old_roles = resolve_stage_inputs(old_recipe, stage(old_recipe), {})
    new_roles = resolve_stage_inputs(new_recipe, stage(new_recipe), {})
    if not _data_continuation({key: value for key, value in old_roles.items() if key != "ce_supervision_policy"},
                              {key: value for key, value in new_roles.items() if key != "ce_supervision_policy"}):
        raise ValueError("candidate source does not preserve the authenticated role continuation")
    for key in ("native_graphs", "foundation_native_graphs"):
        if key in old_roles and key in new_roles:
            if (Path(old_roles[key]["path"]).resolve() != previous_source_path
                    or Path(new_roles[key]["path"]).resolve() != source_path):
                raise ValueError(f"shared train/foundation role {key} did not move to the same successor source")
    if old_roles.get("held_native_graphs") != new_roles.get("held_native_graphs"):
        raise ValueError("candidate changes the held graph source")
    old_train = NativeGraphSource(old_roles["native_graphs"]["path"], tokenizer,
        required_split="train", role_ids=roles)
    extension = extension_policy(old_roles, new_roles, prior_train=old_train, candidate_train=new_source)
    _, objective_transition = continuation_identity({"supervision_objective": previous["objective_transition"]["to"]},
        extension)

    descriptor_path = output_dir / "sft-capacity-descriptor.json"
    write_json_durable(descriptor_path, {"schema": "natlang.native-sft-capacity-cases/1",
        "source": _bind(source_path), "geometry": geometry_ref})
    new_source.set_ce_policy(policy)
    zero_rows, weighted_mass = [], 0.0
    for index, example in enumerate(new_source.iter_from(189390), 189390):
        if example.id != new_source.entries[index]["id"]:
            raise ValueError("candidate source changed while evaluating its appended CE rows")
        amount = sum(example.target_weights) * example.loss_scale
        if not math.isfinite(amount) or amount <= 0:
            zero_rows.append(example.id)
        weighted_mass += amount
    if zero_rows:
        raise ValueError("one or more whole-episode rows receive zero effective CE weight")

    result = {"schema": SCHEMA, "predecessor": _bind(previous_receipt_path),
        "source": _bind(source_path), "source_graphs": source_graph,
        "whole_episode_qualification": _bind(qualification_path),
        "family_map": _bind(family_map_path), "ce_policy": _bind(ce_policy_path),
        "recipe": _bind(recipe_path), "geometry": geometry_ref,
        "capacity_descriptor": _bind(descriptor_path), "prior_ce_reuse": _bind(predecessor_dir / "ce-reuse-proof.json"),
        "rows": len(new_source.entries), "predecessor_rows": 189390, "new_rows": 2,
        "new_families": [row["family"] for row in additions], "new_effective_weighted_mass": weighted_mass,
        "zero_effective_rows": zero_rows, "geometry_scan": geometry_scan,
        "family_census": policy["_family_census"], "objective_transition": objective_transition,
        "current_protection_digest": protection.digest,
        "status": "source-ce-recipe-geometry-qualified-not-published",
        "training_admission": False,
        "limitations": ["Source publication and learner adoption remain separate; this stage is not an update receipt."]}
    receipt_path = output_dir / "receipt.json"
    write_json_durable(receipt_path, result)
    return {"receipt": _bind(receipt_path), **result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predecessor-dir", required=True)
    parser.add_argument("--episode-receipt", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(build(predecessor_dir=Path(args.predecessor_dir),
        episode_receipts=[Path(item) for item in args.episode_receipt],
        output_dir=Path(args.output_dir)), indent=2))


if __name__ == "__main__":
    main()
