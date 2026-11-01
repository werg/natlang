#!/usr/bin/env python3
"""Independently verify the acquired bash-context derivation against its input."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training/neuralese"))
sys.path.insert(0, str(ROOT))

from natlang_neuralese.common.custody import ProtectedWorlds, world_identities
from natlang_neuralese.common.hashing import canonical_json_sha256_hex, sha256_file_hex
from natlang_neuralese.common.jsonio import open_jsonl_text, write_json_durable
from scripts.neuralese_data.build_acquired_full_session_graphs import prepare_tokenizer
from scripts.neuralese_data.derive_acquired_bash_context_226_v4 import ANNOTATION, CALL_RE, FUNCTION_RE, PARAMETER_RE
from natlang_neuralese.train.own_output_supervision import label_roles, role_tokens_from_tokenizer


TOKENIZER_ROOT = Path("/home/werg/.cache/huggingface/hub/models--LiquidAI--LFM3-1.7B/snapshots/40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd")
EXPECTED_PROTECTION = "8ed08f067c6a9715e5f48d73bf1d5ddf5d3c07501ee100c83b361c7ae7407ef7"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("derivation_dir", type=Path)
    args = ap.parse_args()
    derivation_dir = args.derivation_dir.resolve(strict=True)
    receipt_path = derivation_dir / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("schema") != "natlang.acquired-bash-context-derivation/1":
        raise ValueError("unexpected derivation receipt schema")
    if receipt.get("status") != "derived; not source admission or learner exposure":
        raise ValueError("derivation makes an invalid admission/exposure claim")
    source_path = Path(receipt["source_graph_artifact"]["path"]).resolve(strict=True)
    out_path = Path(receipt["output"]["path"]).resolve(strict=True)
    if sha(source_path) != receipt["source_graph_artifact"]["sha256"]:
        raise ValueError("source screened graph bytes changed")
    if sha(out_path) != receipt["output"]["sha256"]:
        raise ValueError("derived graph bytes changed")
    if sha(receipt_path) != hashlib.sha256(receipt_path.read_bytes()).hexdigest():
        raise ValueError("receipt changed during read")
    program = Path(receipt["derivation_program"]["path"]).resolve(strict=True)
    if sha(program) != receipt["derivation_program"]["sha256"]:
        raise ValueError("derivation program differs from receipt")
    protection = ProtectedWorlds.load()
    if protection.digest != EXPECTED_PROTECTION or protection.digest != receipt[
            "current_protected_worlds"]["identity_set_sha256"]:
        raise ValueError("current protected-world union differs")
    if receipt["annotation_contract"].get("claimed_original_actor_prompt") is not False or receipt[
            "annotation_contract"].get("claimed_natlang_runtime_tool") is not False:
        raise ValueError("annotation is misrepresented as source prompt or Natlang runtime schema")
    prefix_len = receipt["annotation_tokens"]
    prefix_text = "<|im_start|>system\n" + ANNOTATION + "<|im_end|>\n"
    tokenizer = prepare_tokenizer(str(TOKENIZER_ROOT), history_reasoning_policy="keep").tokenizer
    fp = receipt["tokenizer_fingerprint"]
    role_tokens = role_tokens_from_tokenizer(tokenizer, tokenizer_fingerprint=fp)
    marker_ids = tokenizer.encode("<tool_call>", add_special_tokens=False)
    projection = receipt["training_target_projection"]
    projection_by_id = {item["graph_id"]: item for item in projection["per_graph_position_and_token_hash_map"]}
    if len(projection_by_id) != len(projection["per_graph_position_and_token_hash_map"]):
        raise ValueError("target projection has duplicate graph entries")
    derived_iter = iter(opened_json_rows(out_path))
    current_derived = next(derived_iter, None)
    source_rows = derived_rows = output_calls = 0
    call_profile = Counter()
    ordered_ids = hashlib.sha256()
    ordered_derived_rows = hashlib.sha256()
    ordered_calls = hashlib.sha256()
    source_order_seen = 0
    for original in opened_json_rows(source_path):
        source_rows += 1
        if current_derived is None:
            continue
        if original["id"] != current_derived["id"]:
            continue
        derived = current_derived
        current_derived = next(derived_iter, None)
        derived_rows += 1
        source_order_seen += 1
        ident = original["id"]
        if original.get("split") != "train" or derived.get("split") != "train":
            raise ValueError(f"{ident}: unexpected split")
        if original.get("source_groups") != derived.get("source_groups"):
            raise ValueError(f"{ident}: source groups changed")
        case = {"id": ident, "split": "train", "source_ids": [
            (original.get("conversion") or {}).get("source_row_id", "")],
            "source_groups": original.get("source_groups") or [],
            "split_groups": original.get("source_groups") or []}
        if world_identities(case, evidence_items=False) & protection.identities:
            raise ValueError(f"{ident}: derived source intersects current protection")
        if canonical_json_sha256_hex(original) != projection_by_id.get(ident, {}).get(
                "source_graph_row_sha256", canonical_json_sha256_hex(original)):
            raise ValueError(f"{ident}: projection's source graph row hash differs")
        source_node = original["nodes"][0]
        derived_node = derived["nodes"][0]
        source_ids = list(source_node["token_ids"])
        source_weights = list(source_node["target_weights"])
        if len(source_ids) != len(source_weights):
            raise ValueError(f"{ident}: source token/weight lengths differ")
        map_row = projection_by_id.get(ident)
        removed = [] if map_row is None else map_row["removed_empty_bash_fields"]
        if map_row is not None and map_row["source_graph_row_sha256"] != canonical_json_sha256_hex(original):
            raise ValueError(f"{ident}: original row hash differs from target projection mapping")
        last = 0
        retained_ids, retained_weights = [], []
        for span in sorted(removed, key=lambda item: item["token_start"]):
            start, stop = span["token_start"], span["token_stop"]
            if start < last or stop <= start or stop > len(source_ids):
                raise ValueError(f"{ident}: overlapping or out-of-range projection span")
            segment_ids, segment_weights = source_ids[start:stop], source_weights[start:stop]
            if segment_ids != span["removed_token_ids"]:
                raise ValueError(f"{ident}: mapped removed token IDs differ from source")
            if hashlib.sha256(json.dumps(segment_ids, separators=(",", ":")).encode()).hexdigest() != span[
                    "removed_token_ids_sha256"]:
                raise ValueError(f"{ident}: removed token hash differs")
            if hashlib.sha256(json.dumps(segment_weights, separators=(",", ":")).encode()).hexdigest() != span[
                    "removed_target_weights_sha256"]:
                raise ValueError(f"{ident}: removed weight hash differs")
            segment_text = tokenizer.decode(segment_ids, skip_special_tokens=False,
                clean_up_tokenization_spaces=False).strip()
            if re.fullmatch(r'<parameter=bash>\s*""\s*</parameter>', segment_text) is None:
                raise ValueError(f"{ident}: projected text is not exactly the empty extra field")
            retained_ids.extend(source_ids[last:start])
            retained_weights.extend(source_weights[last:start])
            last = stop
        retained_ids.extend(source_ids[last:])
        retained_weights.extend(source_weights[last:])
        original_rendered = tokenizer.decode(source_ids, skip_special_tokens=False,
            clean_up_tokenization_spaces=False)
        original_calls = list(CALL_RE.finditer(original_rendered))
        original_labels = label_roles(source_ids, role_tokens)
        original_marker_offsets = [i for i in range(len(source_ids) - len(marker_ids) + 1)
            if source_ids[i:i + len(marker_ids)] == marker_ids and original_labels[i] in {4, 5}]
        if len(original_calls) != len(original_marker_offsets):
            raise ValueError(f"{ident}: original source call marker count differs")
        for ordinal, match in enumerate(original_calls):
            fn = FUNCTION_RE.match(match.group(1))
            if fn is None or fn.group(1) != "bash":
                raise ValueError(f"{ident}: source function isn't bash")
            fields = PARAMETER_RE.findall(fn.group(2))
            if tuple(k for k, _ in fields) not in {("command",), ("command", "bash")}:
                raise ValueError(f"{ident}: original source parameter sequence differs")
            decoded_source_args = {k: json.loads(v) for k, v in fields}
            if any(not isinstance(value, str) for value in decoded_source_args.values()):
                raise ValueError(f"{ident}: original source argument is not a string")
            if tuple(decoded_source_args) not in {("command",), ("command", "bash")}:
                raise ValueError(f"{ident}: original source argument set differs")
            if "bash" in decoded_source_args and decoded_source_args["bash"] != "":
                raise ValueError(f"{ident}: projected source argument is not the known empty string")
            ordered_calls.update(json.dumps({"graph_id": ident, "ordinal": ordinal,
                "function": "bash", "arguments": {k: hashlib.sha256(v.encode()).hexdigest()
                    for k, v in decoded_source_args.items()}}, sort_keys=True,
                separators=(",", ":")).encode())
            ordered_calls.update(b"\n")
        prefix_ids = derived_node["token_ids"][:prefix_len]
        if tokenizer.decode(prefix_ids, skip_special_tokens=False,
                clean_up_tokenization_spaces=False) != prefix_text:
            raise ValueError(f"{ident}: inserted annotation text differs")
        if any(weight != 0 for weight in derived_node["target_weights"][:prefix_len]):
            raise ValueError(f"{ident}: training context is supervised as a target")
        if derived_node["token_ids"][prefix_len:] != retained_ids or derived_node[
                "target_weights"][prefix_len:] != retained_weights:
            raise ValueError(f"{ident}: derived target/token content differs outside declared projection")
        if len(derived_node["token_ids"]) != len(derived_node["target_weights"]):
            raise ValueError(f"{ident}: derived token/weight lengths differ")
        # Every graph-level field and all source conversion metadata must be
        # unchanged except for the two declared training annotations.
        original_without_arrays = copy.deepcopy(original)
        derived_without_arrays = copy.deepcopy(derived)
        for candidate in (original_without_arrays, derived_without_arrays):
            for node in candidate["nodes"]:
                node.pop("token_ids", None)
                node.pop("target_weights", None)
        derived_conv = derived_without_arrays["conversion"]
        annotation = derived_conv.pop("training_context_augmentation", None)
        target_transform = derived_conv.pop("training_target_projection", None)
        if target_transform is None and removed:
            raise ValueError(f"{ident}: missing projection metadata")
        if annotation is None or annotation.get("annotation_sha256") != receipt["annotation_sha256"]:
            raise ValueError(f"{ident}: annotation metadata is not receipt-bound")
        if derived_without_arrays != original_without_arrays:
            raise ValueError(f"{ident}: unexpected graph/source metadata mutation")
        if target_transform and target_transform.get("source_graph_row_sha256") != canonical_json_sha256_hex(original):
            raise ValueError(f"{ident}: target projection metadata is not source-row bound")
        rendered = tokenizer.decode(derived_node["token_ids"], skip_special_tokens=False,
            clean_up_tokenization_spaces=False)
        calls = list(CALL_RE.finditer(rendered))
        labels = label_roles(derived_node["token_ids"], role_tokens)
        marker_offsets = [i for i in range(len(derived_node["token_ids"]) - len(marker_ids) + 1)
            if derived_node["token_ids"][i:i + len(marker_ids)] == marker_ids and labels[i] in {4, 5}]
        if len(calls) != len(marker_offsets):
            raise ValueError(f"{ident}: transformed call marker count differs")
        for ordinal, match in enumerate(calls):
            fn = FUNCTION_RE.match(match.group(1))
            if fn is None or fn.group(1) != "bash":
                raise ValueError(f"{ident}: transformed function isn't bash")
            fields = PARAMETER_RE.findall(fn.group(2))
            if tuple(k for k, _ in fields) != ("command",) or re.sub(PARAMETER_RE, "", fn.group(2)).strip():
                raise ValueError(f"{ident}: transformed call isn't strict bash(command:string)")
            command = json.loads(fields[0][1])
            if not isinstance(command, str) or not command.strip():
                raise ValueError(f"{ident}: transformed command is not a nonempty string")
            output_calls += 1
            call_profile["bash(command:string)"] += 1
        ordered_ids.update((ident + "\n").encode())
        ordered_derived_rows.update((canonical_json_sha256_hex(derived) + "\n").encode())
    if current_derived is not None:
        raise ValueError("derived graph row was not found in source order")
    if (source_rows != receipt["screened_graph_rows"] or derived_rows != receipt["derived_graph_rows"]
            or output_calls != receipt["validated_source_calls"]
            or derived_rows != 226 or output_calls != 8550):
        raise ValueError(f"verified rows/calls differ: {source_rows}/{derived_rows}/{output_calls}")
    if len(projection_by_id) != 7 or sum(len(row["removed_empty_bash_fields"])
            for row in projection_by_id.values()) != 25:
        raise ValueError("target projection map count differs")
    if ordered_ids.hexdigest() != receipt["ordered_source_graph_ids_sha256"]:
        raise ValueError("derived graph ID digest differs")
    if ordered_derived_rows.hexdigest() != receipt["ordered_derived_graph_row_sha256"]:
        raise ValueError("derived row digest differs")
    if ordered_calls.hexdigest() != receipt["ordered_call_identity_sha256"]:
        raise ValueError("canonicalized action digest differs")
    verification = {
        "schema": "natlang.acquired-bash-context-independent-verification/1",
        "status": "verified derivation integrity; not source admission or learner exposure",
        "derivation_receipt": {"path": str(receipt_path), "sha256": sha(receipt_path)},
        "verifier_program": {"path": str(Path(__file__).resolve()), "sha256": sha(Path(__file__).resolve())},
        "source_graph_artifact": {"path": str(source_path), "sha256": sha(source_path)},
        "derived_graph_artifact": {"path": str(out_path), "sha256": sha(out_path)},
        "current_protected_worlds_digest": protection.digest,
        "screened_source_rows": source_rows, "derived_rows": derived_rows,
        "validated_strict_calls": output_calls, "call_profile": dict(call_profile),
        "source_target_projection_rows": len(projection_by_id),
        "removed_empty_parameter_occurrences": 25,
        "preserved_other_tokens_and_weights": True,
    }
    write_json_durable(derivation_dir / "independent-verification.json", verification)
    print(json.dumps(verification, sort_keys=True))


def opened_json_rows(path: Path):
    with open_jsonl_text(path) as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


if __name__ == "__main__":
    main()
