#!/usr/bin/env python3
"""Convert exact historical Luna train episodes into complete ordinary graphs.

This is a CPU-only, read-only source conversion. It groups all approved action
targets for each accepted episode into one graph, keeps the actual invocation
and effect tree as bound provenance, uses the current serving tokenizer, and
does not infer Neuralese producers or imports.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training/neuralese"))

from natlang_neuralese.common.paths import resolve as machine_path
from natlang_neuralese.common.custody import ProtectedWorlds
from natlang_neuralese.common.hashing import canonical_json_bytes, sha256_file_hex, sha256_hex
from natlang_neuralese.common.jsonio import write_json_durable
from natlang_neuralese.data.native_graph_schema import parse_graph
from natlang_neuralese.data.text_corpus import native_gold_packet, tokenizer_fingerprint


SCHEMA = "natlang.historical-luna-full-episode-graphs/1"
EXPECTED_TOKENIZER = "133504d493a77e3121f62cc8c83699cb222ee993c46959d604c4b683a2ffdaf3"
from scripts.neuralese_data.student_tokenizers import STUDENTS

DEFAULT_BASE_TOKENIZER = Path(STUDENTS["lfm3"]["path"])


def prior_geometry_path() -> Path:
    return machine_path(
        "data_nvme", "loghub-graph-successor-20261031", "terra-ce-stage",
        "telecom-v27-plus1", "packed-capacity-geometry",
        "fe23368e8588bcf585030d5d62c4b2b9a90ba74bef0ba824907f999aecc9e35b.json")


def read(path: Path):
    return json.loads(Path(path).read_bytes())


def rows(path: Path):
    return [json.loads(line) for line in Path(path).read_bytes().splitlines() if line.strip()]


def binding(path: Path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file_hex(path)}


def check_binding(actual: dict, declared: dict, label: str):
    if actual != declared:
        raise ValueError(f"{label} differs from its exact bound file")


def _verify_source(continuity_path: Path, recovery_path: Path, selected_path: Path):
    continuity_path = Path(continuity_path).resolve(strict=True)
    recovery_path = Path(recovery_path).resolve(strict=True)
    selected_path = Path(selected_path).resolve(strict=True)
    continuity, recovery = read(continuity_path), read(recovery_path)
    if (continuity.get("schema") != "natlang.historical-provider-source-continuity/1"
            or continuity.get("dispatch_authorized") is not False
            or continuity.get("training_admission") is not False
            or continuity.get("training_exposure") is not False):
        raise ValueError("historical continuity receipt is not the read-only exact-source form")
    if (recovery.get("schema") != "natlang.relational-v5-current-protection-recovery/1"
            or recovery.get("training_admission") is not False
            or recovery.get("training_exposure") is not False):
        raise ValueError("v5 recovery receipt is not the immutable non-admission selection record")
    check_binding(binding(selected_path), recovery.get("source_artifact"), "selected full-context turn source")
    selected_derivation = recovery.get("source_derivation") or {}
    if selected_derivation.get("raw_provider_trajectories") != continuity.get("raw_trajectory_source"):
        raise ValueError("selected turns do not bind the same authenticated historical raw provider source")
    if selected_derivation.get("provider_source_manifest") != continuity.get("provider_source_manifest"):
        raise ValueError("selected turns and continuity receipt name different Luna provider manifests")

    raw_binding = continuity["raw_trajectory_source"]
    raw_path = Path(raw_binding["path"]).resolve(strict=True)
    check_binding(binding(raw_path), raw_binding, "raw historical provider trajectory source")
    raw_rows = rows(raw_path)
    if len(raw_rows) != len(continuity.get("included_accepted_results") or []):
        raise ValueError("historical raw source row count differs from the closed continuity receipt")
    raw_by_id = {row.get("id"): row for row in raw_rows}
    if len(raw_by_id) != len(raw_rows):
        raise ValueError("historical source repeats an episode identity")
    for result in continuity["included_accepted_results"]:
        result_path = Path(result["result"]["path"]).resolve(strict=True)
        check_binding(binding(result_path), result["result"], "historical collector result")
        if result.get("collector_manifest"):
            manifest_path = Path(result["collector_manifest"]["path"]).resolve(strict=True)
            check_binding(binding(manifest_path), result["collector_manifest"], "historical collector manifest")
        result_rows = rows(result_path)
        if len(result_rows) != 1 or result_rows[0] != raw_by_id.get(result.get("trajectory_id")):
            raise ValueError("historical source row differs from its closed collector result")

    protection = ProtectedWorlds.load()
    if (protection.digest != recovery["current_protection"]["protected_worlds_digest"]
            or protection.digest != continuity["current_protection"]["identity_digest"]):
        raise ValueError("current protected-world union changed since source qualification")
    accepted_ids = {item["trajectory_id"] for item in continuity["included_accepted_results"]}
    if accepted_ids != set(recovery["selection"]["accepted_trajectory_ids"]):
        raise ValueError("selected episode set differs from the closed exact-result source")
    selected_rows = rows(selected_path)
    selected_receipts = {item["row_id"]: item for item in recovery["selection"]["selected_rows"]}
    if len(selected_rows) != recovery["selection"].get("selected_turns"):
        raise ValueError("selected turn artifact count differs from its immutable recovery receipt")
    if len(selected_rows) != len(selected_receipts):
        raise ValueError("selected turn source does not have a one-to-one exact recovery receipt")
    if recovery["selection"].get("oracle_level") != "exact":
        raise ValueError("historical exact Luna outcomes were downgraded or changed in the recovery receipt")

    included = {item["trajectory_id"]: item for item in continuity["included_accepted_results"]}
    rows_by_episode: dict[str, list[dict]] = {key: [] for key in accepted_ids}
    for turn in selected_rows:
        rid = turn.get("id")
        source_ref = turn.get("source_ref") or {}
        trajectory_id = source_ref.get("trajectory_id")
        record = selected_receipts.get(rid)
        raw = raw_by_id.get(trajectory_id)
        if not record or raw is None or trajectory_id not in included:
            raise ValueError("selected turn does not resolve to its exact accepted source episode")
        if (record.get("trajectory_id") != trajectory_id
                or record.get("source_row_sha256") != included[trajectory_id].get("source_row_sha256")
                or source_ref.get("source_row_sha256") != record.get("source_row_sha256")
                or turn.get("training_admission", {}).get("approved") is not True
                or turn.get("oracle_level") != "exact"
                or turn.get("split") != "train"):
            raise ValueError("selected Luna action lacks its exact train approval and source-row custody")
        if sha256_hex(canonical_json_bytes(turn.get("target"))) != record.get("turn_target_sha256"):
            raise ValueError("selected assistant action differs from its exact per-turn target digest")
        decision_index = (turn.get("decision") or {}).get("index")
        trace = raw.get("trajectory") or []
        if type(decision_index) is not int or not 0 <= decision_index < len(trace):
            raise ValueError("selected assistant action does not map to a captured trajectory index")
        source_turn = trace[decision_index]
        if (turn.get("messages") != source_turn.get("context")
                or source_ref.get("invocation_id") != source_turn.get("invocation_id")):
            raise ValueError("selected full prompt or invocation differs from the captured Luna trajectory")
        # The selected target must be the exact action from this captured provider
        # response. This prevents a correct context/index from being paired with a
        # separately substituted target.
        target = turn.get("target") or {}
        response = source_turn.get("model_response") or {}
        target_calls = target.get("tool_calls") or []
        response_calls = response.get("calls") or []
        if target_calls:
            if len(target_calls) != len(response_calls):
                raise ValueError("selected tool action count differs from the captured provider response")
            for selected_call, captured_call in zip(target_calls, response_calls):
                function = selected_call.get("function") or {}
                try:
                    selected_arguments = json.loads(function.get("arguments", ""))
                except (TypeError, json.JSONDecodeError) as exc:
                    raise ValueError("selected tool action arguments are not canonical captured JSON") from exc
                if (function.get("name") != captured_call[0]
                        or selected_arguments != captured_call[1]):
                    raise ValueError("selected tool action differs from the captured provider response")
        elif target.get("content") != response.get("text") or response_calls:
            raise ValueError("selected assistant text differs from the captured provider response")
        program = (raw.get("task") or {}).get("program_ir") or {}
        expected_groups = set(program.get("source_groups") or []) | {program.get("id")}
        if (program.get("split") != "train" or program.get("id") != source_ref.get("program_ir_id")
                or set(turn.get("source_groups") or []) != expected_groups):
            raise ValueError("selected turn changed its actual program or related-world source groups")
        custody = protection.check(program)
        if custody.get("split") != "train":
            raise ValueError("a selected historical operation is no longer protected train data")
        if any(turn.get("neuralese_conversion", {}).get(key) for key in (
                "external_context_inputs", "teacher_text_spans", "native_imports")):
            raise ValueError("ordinary historical conversion unexpectedly contains claimed native imports")
        rows_by_episode[trajectory_id].append(turn)
    if set(rows_by_episode) != accepted_ids or any(not value for value in rows_by_episode.values()):
        raise ValueError("current train source lost one of the two whole accepted episodes")
    for trajectory_id, episode_rows in rows_by_episode.items():
        episode_rows.sort(key=lambda row: row["decision"]["index"])
        indices = [row["decision"]["index"] for row in episode_rows]
        if len(indices) != len(set(indices)):
            raise ValueError("episode contains duplicate source action indices")
        program = raw_by_id[trajectory_id]["task"]["program_ir"]
        for turn in episode_rows:
            if turn["program_id"] != program["id"]:
                raise ValueError("selected turn program changed within one episode")

    return {"continuity_path": continuity_path, "continuity": continuity,
        "recovery_path": recovery_path, "recovery": recovery,
        "raw_path": raw_path, "raw_binding": raw_binding, "raw_by_id": raw_by_id,
        "selected_path": selected_path, "selected_binding": binding(selected_path),
        "selected_rows": selected_rows, "rows_by_episode": rows_by_episode,
        "protection": protection}


def _bind_code(paths):
    return [binding(path) for path in paths]


def build(*, continuity_path: Path, recovery_path: Path, selected_path: Path,
          tokenizer_path: Path, output_dir: Path):
    source = _verify_source(continuity_path, recovery_path, selected_path)
    tokenizer_path = Path(tokenizer_path).resolve(strict=True)
    from transformers import AutoTokenizer
    from scripts.neuralese_data.build_native_semantic_pairs_e2e import provision
    from natlang_neuralese.data.native_graph_schema import communication_classification

    prior_geometry = read(prior_geometry_path())
    expected_tokenizer = prior_geometry["identity"]["tokenizer"]
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=False, local_files_only=True)
    provision(tokenizer)
    fingerprint = tokenizer_fingerprint(tokenizer)
    if fingerprint != EXPECTED_TOKENIZER or fingerprint != expected_tokenizer:
        raise ValueError("current expanded serving tokenizer differs from the active native source tokenizer")

    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise ValueError("historical full-episode output is immutable; choose a new directory")
    output_dir.mkdir(parents=True)
    graph_rows, graph_summaries = [], []
    raw_by_id = source["raw_by_id"]
    for trajectory_id in sorted(source["rows_by_episode"]):
        raw = raw_by_id[trajectory_id]
        outcome = raw.get("outcome") or {}
        oracle = outcome.get("oracle") or {}
        provenance = raw.get("provenance") or {}
        if (outcome.get("status") != "done" or outcome.get("accepted") is not True
                or oracle.get("accepted") is not True or oracle.get("level") != "exact"
                or provenance.get("model") != "gpt-6-luna"
                or provenance.get("provider") != "openai-codex"
                or provenance.get("transport") != "pi-provider"
                or provenance.get("collection_role") != "teacher"):
            raise ValueError("historical Luna episode lost its exact actual-teacher outcome identity")
        program = raw["task"]["program_ir"]
        episode_rows = source["rows_by_episode"][trajectory_id]
        nodes, node_receipts, action_names = [], [], set()
        for turn in episode_rows:
            index = turn["decision"]["index"]
            tools = turn.get("tools") or []
            text, token_ids, boundary = native_gold_packet(tokenizer,
                turn["messages"], turn["target"], tools, prompt_rendering="full")
            if (not isinstance(text, str) or not token_ids or type(boundary) is not int
                    or not 0 < boundary < len(token_ids)):
                raise ValueError("current full serving renderer did not preserve an exact action target boundary")
            # native_gold_packet renders the complete serving context through
            # serve.chat.render_messages before applying the template. Raw
            # provider history has OpenAI JSON-string tool arguments, while the
            # backbone template requires mappings; calling the template directly
            # on that raw history would bypass the serving normalization path.
            name = f"action:{index}:{sha256_hex(turn['id'].encode())[:12]}"
            if name in action_names:
                raise ValueError("whole episode contains duplicate source action node identity")
            action_names.add(name)
            weights = [0.0] * boundary + [1.0] * (len(token_ids) - boundary)
            weights[0] = 0.0
            nodes.append({"name": name, "token_ids": list(token_ids), "target_weights": weights})
            node_receipts.append({"node": name, "turn_id": turn["id"],
                "turn_index": index, "target_sha256": sha256_hex(canonical_json_bytes(turn["target"])),
                "source_row_sha256": turn["source_ref"]["source_row_sha256"],
                "provider_context_sha256": sha256_hex(canonical_json_bytes(turn["messages"])),
                "invocation_id": turn["source_ref"]["invocation_id"],
                "tokens": len(token_ids), "prompt_tokens": boundary,
                "supervised_tokens": len(token_ids) - boundary})

        full_trace = raw.get("trajectory") or []
        selected_indices = {turn["decision"]["index"] for turn in episode_rows}
        invocation_ledger = outcome.get("invocation_ledger") or []
        execution_graph = outcome.get("execution_graph") or []
        action_ledger = outcome.get("action_ledger") or []
        if not invocation_ledger or not execution_graph or not action_ledger:
            raise ValueError("full raw execution graph, invocation ledger, and action ledger are required")
        invocation_ids = {item.get("invocation_id") for item in invocation_ledger}
        if any(receipt["invocation_id"] not in invocation_ids for receipt in node_receipts):
            raise ValueError("selected action turn does not resolve to a historical invocation receipt")
        groups = sorted(set(program.get("source_groups") or []) | {program["id"]})
        graph = {"schema": "natlang.neuralese-native-graph/1",
            "id": "luna-historical-whole-episode:" + trajectory_id,
            "split": "train", "source_groups": groups,
            "tokenizer_fingerprint": fingerprint,
            "nodes": nodes,
            "conversion": {"kind": "historical-luna-whole-episode-static/1",
                "trajectory_id": trajectory_id,
                "source_row_sha256": source["continuity"]["included_accepted_results"][
                    next(i for i, item in enumerate(source["continuity"]["included_accepted_results"])
                         if item["trajectory_id"] == trajectory_id)]["source_row_sha256"],
                "actor": {"model": provenance["model"], "provider": provenance["provider"],
                    "transport": provenance["transport"], "collection_role": provenance["collection_role"],
                    "historical_runtime": provenance.get("runtime"),
                    "neuralese_transport": provenance.get("neuralese_transport")},
                "upstream_oracle": {"level": "exact", "accepted": True,
                    "checks": outcome.get("checks")},
                "current_protection_digest": source["protection"].digest,
                "whole_episode_actions": node_receipts,
                "unselected_source_indices": sorted(set(range(len(full_trace))) - selected_indices),
                "actual_call_tree": {"schema": "natlang.historical-execution-call-tree/1",
                    "invocation_ledger": invocation_ledger,
                    "execution_graph": execution_graph,
                    "action_ledger": action_ledger},
                "actual_call_tree_sha256": sha256_hex(canonical_json_bytes({
                    "invocation_ledger": invocation_ledger,
                    "execution_graph": execution_graph,
                    "action_ledger": action_ledger})),
                "no_neuralese_imports_claimed": True,
                "source_full_trajectory_sha256": sha256_hex(canonical_json_bytes(full_trace))}}
        example = parse_graph(graph, tokenizer_fingerprint=fingerprint)
        classification = communication_classification(example)
        if classification["communication"] or any(node.imports for node in example.plan.invocations):
            raise ValueError("ordinary v5 full-episode graph unexpectedly creates a Neuralese import edge")
        graph_rows.append(graph)
        graph_summaries.append({"id": graph["id"], "trajectory_id": trajectory_id,
            "program_id": program["id"], "source_groups": groups,
            "selected_action_count": len(nodes), "source_action_count": len(full_trace),
            "context_only_source_indices": graph["conversion"]["unselected_source_indices"],
            "invocation_count": len(invocation_ledger), "execution_graph_nodes": len(execution_graph),
            "supervised_tokens": sum(sum(node["target_weights"]) for node in nodes),
            "communication": classification["communication"]})

    output_path = output_dir / "full-episodes.jsonl"
    data = b"".join(canonical_json_bytes(graph) + b"\n" for graph in graph_rows)
    output_path.write_bytes(data)
    with output_path.open("rb") as stream:
        import os
        os.fsync(stream.fileno())
    directory_fd = __import__("os").open(output_dir, __import__("os").O_RDONLY)
    try:
        __import__("os").fsync(directory_fd)
    finally:
        __import__("os").close(directory_fd)

    files = _bind_code([tokenizer_path / name for name in (
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja")
        if (tokenizer_path / name).is_file()])
    source_code = _bind_code([
        Path(__file__), ROOT / "training/neuralese/natlang_neuralese/data/text_corpus.py",
        ROOT / "training/neuralese/natlang_neuralese/data/native_graph_schema.py",
        ROOT / "training/neuralese/natlang_neuralese/data/native_sources.py",
        ROOT / "training/neuralese/natlang_neuralese/model/native_graph_plan.py",
        ROOT / "scripts/neuralese_data/build_native_semantic_pairs_e2e.py",
        ROOT / "training/neuralese/natlang_neuralese/serve/chat.py"])
    receipt = {"schema": SCHEMA, "source_artifact": binding(output_path),
        "source_artifact_rows": len(graph_rows),
        "historical_continuity": binding(source["continuity_path"]),
        "v5_selection_receipt": binding(source["recovery_path"]),
        "selected_action_source": source["selected_binding"],
        "raw_provider_source": source["raw_binding"],
        "current_protection_digest": source["protection"].digest,
        "tokenizer": {"path": str(tokenizer_path), "fingerprint": fingerprint,
            "expected_active_fingerprint": EXPECTED_TOKENIZER, "files": files},
        "source_code": source_code,
        "episodes": graph_summaries,
        "supervision": {"selected_action_targets": sum(item["selected_action_count"] for item in graph_summaries),
            "context_only_source_actions": sum(len(item["context_only_source_indices"]) for item in graph_summaries),
            "teacher_native_writers": 0, "neuralese_import_edges": 0,
            "execution_call_tree_preserved": True},
        "training_admission": False,
        "limitations": ["Static ordinary-action distillation; no teacher-created Neuralese vector or learned native import edge.",
            "Upstream exact task acceptance is source evidence, not a new model capability claim."]}
    receipt_path = output_dir / "receipt.json"
    write_json_durable(receipt_path, receipt)
    return {"receipt": binding(receipt_path), **receipt}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continuity", required=True, type=Path)
    parser.add_argument("--recovery", required=True, type=Path)
    parser.add_argument("--selected-turns", required=True, type=Path)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_BASE_TOKENIZER)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build(continuity_path=args.continuity, recovery_path=args.recovery,
        selected_path=args.selected_turns, tokenizer_path=args.tokenizer,
        output_dir=args.output), indent=2))


if __name__ == "__main__":
    main()
