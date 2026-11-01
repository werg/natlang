#!/usr/bin/env python3
"""Derive training-context-annotated graphs for acquired bash-action rows.

This is a separate, immutable source derivation. It never edits the upstream
screened graphs or conversion receipts. The inserted annotation describes the
normalized corpus call representation; it is explicitly not an original actor
prompt or a Natlang runtime tool declaration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training/neuralese"))
sys.path.insert(0, str(ROOT))

from natlang_neuralese.common.custody import ProtectedWorlds, world_identities
from natlang_neuralese.common.hashing import canonical_json_sha256_hex, sha256_file_hex
from natlang_neuralese.common.jsonio import open_jsonl_text, write_json_durable, write_jsonl_text
from scripts.neuralese_data.build_acquired_full_session_graphs import prepare_tokenizer
from natlang_neuralese.train.own_output_supervision import label_roles, role_tokens_from_tokenizer


SCREEN = Path("/home/werg/data/dgxops/acquired-current-protection-screen-20261101/receipt.json")
TOKENIZER_ROOT = Path("/home/werg/.cache/huggingface/hub/models--LiquidAI--LFM3-1.7B/snapshots/40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd")
EXPECTED_PROTECTION = "8ed08f067c6a9715e5f48d73bf1d5ddf5d3c07501ee100c83b361c7ae7407ef7"
EXPECTED_SCREEN_ROWS = 760
EXPECTED_TOOL_GRAPHS = 226
EXPECTED_CALLS = 8550
EXPECTED_CALL_PROFILES = {("D14", ("command",)): 4895,
    ("D13", ("command",)): 3630, ("D13", ("command", "bash")): 25}
ANNOTATION = (
    "Training-only corpus-format annotation; this text was added during conversion and was not in the source actor prompt. "
    "These imported CooperBench trajectories are represented with one action function named bash and one required argument named command. "
    "The command argument is a string containing the shell command. In this normalized training record it is rendered as "
    "<function=bash><parameter=command> followed by a JSON string and the matching closing tags. "
    "This training-only annotation describes the imported action representation; it is not an original actor prompt or a Natlang runtime tool declaration."
)
CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
FUNCTION_RE = re.compile(r"^\s*<function=([A-Za-z0-9_-]+)>\s*(.*?)\s*</function>\s*$", re.S)
PARAMETER_RE = re.compile(r"<parameter=([A-Za-z0-9_-]+)>\s*(.*?)\s*</parameter>", re.S)


def file_ref(path: Path) -> dict:
    path = path.resolve(strict=True)
    return {"path": str(path), "sha256": sha256_file_hex(path), "bytes": path.stat().st_size}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    screen_path = SCREEN.resolve(strict=True)
    screen = json.loads(screen_path.read_text())
    if screen.get("schema") != "natlang.acquired-current-protection-screen/1":
        raise ValueError("unexpected screening receipt schema")
    if screen["protected_worlds"].get("identity_set_sha256") != EXPECTED_PROTECTION:
        raise ValueError("screen protection digest differs from the current full union")
    protected = ProtectedWorlds.load()
    if protected.digest != EXPECTED_PROTECTION:
        raise ValueError("current ProtectedWorlds changed")
    source_path = Path(screen["output"]["path"]).resolve(strict=True)
    if sha256_file_hex(source_path) != screen["output"]["sha256"]:
        raise ValueError("screened source graph artifact changed")
    if screen["output"].get("rows") != EXPECTED_SCREEN_ROWS:
        raise ValueError("screened source graph count changed")
    tokenizer = prepare_tokenizer(str(TOKENIZER_ROOT), history_reasoning_policy="keep").tokenizer
    fp = screen["output"]["tokenizer_fingerprint"]
    role_tokens = role_tokens_from_tokenizer(tokenizer, tokenizer_fingerprint=fp)
    marker_ids = tokenizer.encode("<tool_call>", add_special_tokens=False)
    if not marker_ids:
        raise ValueError("source tokenizer does not encode the call marker")
    # A single explicit call-marker offset is enough to find candidate rows;
    # calls themselves are parsed from the exact tokenizer-decoded source text.
    selected = 0
    total = 0
    calls_total = 0
    removed_fields = 0
    projection_rows = []
    source_counts: Counter[str] = Counter()
    function_counts: Counter[str] = Counter()
    parameter_counts: Counter[str] = Counter()
    call_profiles: Counter = Counter()
    call_digest = hashlib.sha256()
    graph_ids = hashlib.sha256()
    graph_row_digest = hashlib.sha256()
    added_token_count = None
    row_path = out / "native-graphs.jsonl.zst"
    with write_jsonl_text(row_path) as dst, open_jsonl_text(source_path) as src:
        for line in src:
            if not line.strip():
                continue
            graph = json.loads(line)
            total += 1
            if graph.get("schema") != "natlang.neuralese-native-graph/1" or graph.get("split") != "train":
                raise ValueError("source graph schema or split changed")
            if graph.get("tokenizer_fingerprint") != fp or len(graph.get("nodes", [])) != 1:
                raise ValueError("source graph tokenizer or one-node structure changed")
            node = graph["nodes"][0]
            ids = node["token_ids"]
            weights = node["target_weights"]
            if len(ids) != len(weights):
                raise ValueError("source token/weight lengths differ")
            # Only source assistant call markers qualify. This preserves rows
            # with incidental literal marker text in user-provided code.
            role_ids = label_roles(ids, role_tokens)
            assistant_roles = {4, 5}
            candidate_offsets = [i for i in range(len(ids) - len(marker_ids) + 1)
                if ids[i:i + len(marker_ids)] == marker_ids and role_ids[i] in assistant_roles]
            if not candidate_offsets:
                continue
            if len(candidate_offsets) <= 0:
                raise ValueError("call candidate unexpectedly has no source markers")
            rendered = tokenizer.decode(ids, skip_special_tokens=False,
                clean_up_tokenization_spaces=False)
            matches = list(CALL_RE.finditer(rendered))
            if len(matches) != len(candidate_offsets):
                raise ValueError(f"{graph['id']}: decoded action count differs from source assistant call markers")
            domain = "D13" if ":D13:" in graph["id"] else "D14" if ":D14:" in graph["id"] else None
            if domain not in {"D13", "D14"}:
                raise ValueError("source action graph has no recognized D13/D14 lineage")
            source_row_sha256 = canonical_json_sha256_hex(graph)
            extra_occurrences = 0
            parsed_call_shapes = []
            for ordinal, match in enumerate(matches):
                parsed_function = FUNCTION_RE.match(match.group(1))
                if parsed_function is None:
                    raise ValueError(f"{graph['id']}: malformed normalized function call")
                function, body = parsed_function.groups()
                fields = PARAMETER_RE.findall(body)
                if re.sub(PARAMETER_RE, "", body).strip():
                    raise ValueError(f"{graph['id']}: unparsed content in normalized function arguments")
                shape = tuple(name for name, _ in fields)
                parsed_call_shapes.append(shape)
                call_profiles[(domain, shape)] += 1
                decoded = {}
                for argument, encoded_value in fields:
                    try:
                        value = json.loads(encoded_value)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"{graph['id']}: {argument} is not a serialized JSON value") from exc
                    if not isinstance(value, str):
                        raise ValueError(f"{graph['id']}: {argument} value is not a string")
                    decoded[argument] = value
                    parameter_counts[argument] += 1
                if function != "bash" or "command" not in decoded:
                    raise ValueError(f"{graph['id']}: unsupported source call signature {function}{shape}")
                if shape not in {("command",), ("command", "bash")}:
                    raise ValueError(f"{graph['id']}: unrecognized source argument set {shape}")
                if "bash" in decoded and (domain != "D13" or decoded["bash"] != ""):
                    raise ValueError(f"{graph['id']}: extra bash field is not the known empty D13 serialization artifact")
                if "bash" in decoded:
                    extra_occurrences += 1
                command = decoded["command"]
                if not command.strip():
                    raise ValueError(f"{graph['id']}: command argument is not a nonempty string")
                function_counts[function] += 1
                call_digest.update(json.dumps({"graph_id": graph["id"], "ordinal": ordinal,
                    "function": function, "arguments": {k: hashlib.sha256(v.encode()).hexdigest()
                        for k, v in decoded.items()}},
                    sort_keys=True, separators=(",", ":")).encode())
                call_digest.update(b"\n")
            if extra_occurrences:
                end_tag = tokenizer.encode("</parameter>", add_special_tokens=False)
                original_ids = node["token_ids"]
                original_weights = node["target_weights"]
                closes = [i for i in range(len(original_ids) - len(end_tag) + 1)
                    if original_ids[i:i + len(end_tag)] == end_tag]
                field_count = sum(len(shape) for shape in parsed_call_shapes)
                if len(closes) != field_count or any(role_ids[i] not in {4, 5} for i in closes):
                    raise ValueError(f"{graph['id']}: parameter close tags do not exactly cover assistant call fields")
                removed = []
                field_cursor = 0
                for shape in parsed_call_shapes:
                    command_close = closes[field_cursor]
                    field_cursor += 1
                    if len(shape) != 2:
                        continue
                    extra_close = closes[field_cursor]
                    field_cursor += 1
                    start = command_close + len(end_tag)
                    stop = extra_close + len(end_tag)
                    field_text = tokenizer.decode(original_ids[start:stop],
                        skip_special_tokens=False, clean_up_tokenization_spaces=False).strip()
                    if re.fullmatch(r'<parameter=bash>\s*""\s*</parameter>', field_text) is None:
                        raise ValueError(f"{graph['id']}: source extra bash parameter is not exactly empty")
                    span_ids = original_ids[start:stop]
                    span_weights = original_weights[start:stop]
                    if not span_ids or not all(weight > 0 for weight in span_weights):
                        raise ValueError(f"{graph['id']}: projected field is not fully supervised in source target")
                    removed.append({"token_start": start, "token_stop": stop,
                        "removed_token_ids": span_ids,
                        "removed_token_ids_sha256": hashlib.sha256(
                            json.dumps(span_ids, separators=(",", ":")).encode()).hexdigest(),
                        "removed_target_weights_sha256": hashlib.sha256(
                            json.dumps(span_weights, separators=(",", ":")).encode()).hexdigest(),
                        "field_value_sha256": hashlib.sha256(b"\"\"").hexdigest()})
                # Delete right-to-left so each recorded span is an exact source
                # position; unrelated target IDs and weights remain unchanged.
                ids = list(original_ids)
                weights = list(original_weights)
                for span in sorted(removed, key=lambda item: item["token_start"], reverse=True):
                    del ids[span["token_start"]:span["token_stop"]]
                    del weights[span["token_start"]:span["token_stop"]]
                node["token_ids"] = ids
                node["target_weights"] = weights
                projection_rows.append({"graph_id": graph["id"],
                    "source_graph_row_sha256": source_row_sha256,
                    "removed_empty_bash_fields": removed})
                removed_fields += len(removed)
            lineage = ((graph.get("conversion") or {}).get("acquired_session") or {}).get("source_lineage") or {}
            origin = lineage.get("notes") or {}
            if domain not in {"D13", "D14"}:
                raise ValueError("source action graph has no recognized D13/D14 lineage")
            task_case = {"id": graph["id"], "split": "train",
                "source_ids": [(graph.get("conversion") or {}).get("source_row_id", "")],
                "source_groups": graph.get("source_groups") or [],
                "split_groups": graph.get("source_groups") or []}
            matches_protected = world_identities(task_case, evidence_items=False) & protected.identities
            if any(kind in {"id", "group", "binding"} for kind, _ in matches_protected):
                raise ValueError(f"source action graph now intersects protection: {graph['id']}")
            if added_token_count is None:
                prefix = tokenizer.encode("<|im_start|>system\n" + ANNOTATION + "<|im_end|>\n",
                    add_special_tokens=False)
                if not prefix:
                    raise ValueError("training annotation encoded to no tokens")
                added_token_count = len(prefix)
            else:
                prefix = annotation_token_ids
            # Cache the prefix ids after the first tokenizer call.
            annotation_token_ids = prefix
            node["token_ids"] = prefix + ids
            node["target_weights"] = [0.0] * len(prefix) + weights
            conv = graph.setdefault("conversion", {})
            conv["training_context_augmentation"] = {
                "schema": "natlang.acquired-action-format-annotation/1",
                "annotation_sha256": hashlib.sha256(ANNOTATION.encode()).hexdigest(),
                "annotation_tokens": len(prefix),
                "source_prompt_claim": False,
                "natlang_tool_declaration": False,
                "basis": "exact normalized assistant action serialization validated against this source graph",
                "source_domain": domain,
                "source_revision": lineage.get("upstream_revision"),
                "source_file_sha256": origin.get("sha256"),
                "source_row_id": (conv.get("source_row_id") or ""),
            }
            if extra_occurrences:
                conv["training_target_projection"] = {
                    "schema": "natlang.acquired-empty-argument-projection/1",
                    "operation": "remove only the empty optional parameter-bash span from the normalized target; preserve all other source token IDs and weights",
                    "source_graph_row_sha256": source_row_sha256,
                    "removed_occurrences": extra_occurrences,
                }
            graph_ids.update((graph["id"] + "\n").encode())
            graph_row_digest.update((canonical_json_sha256_hex(graph) + "\n").encode())
            source_counts[domain] += 1
            calls_total += len(matches)
            dst.write(json.dumps(graph, ensure_ascii=False, separators=(",", ":")) + "\n")
            selected += 1
    if total != EXPECTED_SCREEN_ROWS or selected != EXPECTED_TOOL_GRAPHS or calls_total != EXPECTED_CALLS:
        raise ValueError(f"screen/call totals changed: source={total}, action_graphs={selected}, calls={calls_total}")
    if function_counts != Counter({"bash": EXPECTED_CALLS}) or parameter_counts != Counter({"command": EXPECTED_CALLS, "bash": 25}):
        raise ValueError("call function/argument inventory differs from the audited single-string bash interface")
    if dict(call_profiles) != EXPECTED_CALL_PROFILES:
        raise ValueError(f"source call profile totals changed: {dict(call_profiles)}")
    if removed_fields != 25 or len(projection_rows) == 0:
        raise ValueError("empty parameter projection count differs from the exact 25 audited D13 occurrences")
    # Flush durable output and hash only after the compressed stream has closed.
    fd = os.open(row_path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    output_ref = file_ref(row_path)
    receipt = {
        "schema": "natlang.acquired-bash-context-derivation/1",
        "status": "derived; not source admission or learner exposure",
        "created_utc": "2026-11-01",
        "source_screen": file_ref(screen_path),
        "source_graph_artifact": file_ref(source_path),
        "current_protected_worlds": {"identity_set_sha256": protected.digest,
            "registry_path": str(ROOT / "training/protected-worlds.json"),
            "registry_sha256": sha256_file_hex(ROOT / "training/protected-worlds.json")},
        "tokenizer_fingerprint": fp,
        "annotation": ANNOTATION,
        "annotation_sha256": hashlib.sha256(ANNOTATION.encode()).hexdigest(),
        "annotation_tokens": added_token_count,
        "annotation_contract": {
            "added_during_training_derivation": True,
            "claimed_original_actor_prompt": False,
            "claimed_source_structured_tools_field": False,
            "claimed_natlang_runtime_tool": False,
            "function": "bash", "arguments": {"command": {"type": "string", "required": True}},
            "validation": "all normalized assistant call spans parsed from source target tokens; exact D13/D14 argument shapes checked and all values decoded as strings; the 25 observed empty extra fields are removed by the explicitly recorded target projection",
        },
        "source_counts": dict(source_counts),
        "screened_graph_rows": total,
        "derived_graph_rows": selected,
        "validated_source_calls": calls_total,
        "function_counts": dict(function_counts),
        "parameter_counts": dict(parameter_counts),
        "call_profiles": {f"{domain}:{','.join(shape)}": count
            for (domain, shape), count in call_profiles.items()},
        "training_target_projection": {"schema": "natlang.acquired-empty-argument-projection/1",
            "removed_empty_bash_parameter_occurrences": removed_fields,
            "per_graph_position_and_token_hash_map": projection_rows},
        "ordered_call_identity_sha256": call_digest.hexdigest(),
        "ordered_source_graph_ids_sha256": graph_ids.hexdigest(),
        "ordered_derived_graph_row_sha256": graph_row_digest.hexdigest(),
        "output": output_ref,
        "derivation_program": file_ref(Path(__file__)),
    }
    write_json_durable(out / "receipt.json", receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
