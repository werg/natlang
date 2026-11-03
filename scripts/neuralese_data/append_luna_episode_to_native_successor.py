#!/usr/bin/env python3
"""Append one fully qualified Luna episode to a prepared native source successor.

The builder writes a source/CE/geometry continuation only. It preserves the
predecessor tree byte-for-byte, the complete teacher episode graph, and the
existing held sources. It never publishes data or claims optimizer exposure.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training/neuralese"))

from natlang_neuralese.common.custody import ProtectedWorlds
from natlang_neuralese.common.hashing import sha256_file_hex, sha256_hex
from natlang_neuralese.common.jsonio import write_json_durable
from natlang_neuralese.data.native_graph_schema import communication_classification, parse_graph
from natlang_neuralese.data.provider_static_graph import STATIC_PROVENANCES
from natlang_neuralese.data.native_sources import NativeGraphSource
from natlang_neuralese.train import native_packed_capacity as capacity
from natlang_neuralese.train.native_resident_ce import extension_policy
from natlang_neuralese.train.native_supervision import load_policy
from natlang_neuralese.train.own_output_supervision import (
    continuation_identity, identity, policy_objective, prepare_policy,
)
from natlang_neuralese.train.recipe import (
    _data_continuation, load_recipe, resolve_stage_inputs,
)



def _encoder_weights(graph, node):
    """The exact encoder-node CE weights a graph's conversion policy declares: all zero (historical rows), or only the
    closing marker under producer-close supervision (data/provider_native_graph.PRODUCER_CLOSE_SUPERVISION)."""
    from natlang_neuralese.data.provider_native_graph import PRODUCER_CLOSE_SUPERVISION, encoder_target_weights
    policy = graph.get("producer_close_supervision")
    if policy is None:
        return [0.0] * len(node["token_ids"])
    if policy != PRODUCER_CLOSE_SUPERVISION:
        raise ValueError("unknown producer-close supervision policy")
    return encoder_target_weights(len(node["token_ids"]))

def _read(path):
    return json.loads(Path(path).read_bytes())


def _bind(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": sha256_file_hex(path), "bytes": path.stat().st_size}


def _manifest_bind(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": sha256_file_hex(path)}


def _write_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _teacher_actor_binding(raw, raw_binding):
    """Bind the actual external teacher separately from the student encoder."""
    provenance = raw.get("provenance") or {}
    if (provenance.get("model") != "gpt-6-luna"
            or provenance.get("collection_role") != "teacher"
            or not isinstance(provenance.get("provider"), str)
            or not isinstance(provenance.get("transport"), str)):
        raise ValueError("raw result lacks authenticated Luna teacher provenance")
    return {
        "model": provenance["model"],
        "provider": provenance["provider"],
        "transport": provenance["transport"],
        "collection_role": provenance["collection_role"],
        "reasoning_effort": (provenance.get("pi_options") or {}).get("reasoningEffort"),
        "raw_result": raw_binding,
    }


def _verified_oracle_claim(audit, raw):
    """Validate and preserve the actual exact-vs-judged source qualification."""
    level = audit.get("oracle_level")
    exact = audit.get("exact_oracle_claimed")
    outcome = raw.get("outcome") or {}
    oracle = outcome.get("oracle") or {}
    if outcome.get("accepted") is not True or oracle.get("accepted") is not True:
        raise ValueError("source result is not accepted by its recorded oracle")
    if level == "exact":
        if exact is not True or oracle.get("level") != "exact":
            raise ValueError("exact oracle claim does not match the raw result")
        return {"oracle_level": "exact", "exact_oracle_claimed": True}
    if level == "judged":
        judge = audit.get("oracle_judge") or {}
        provenance_judge = (raw.get("provenance") or {}).get("judge") or {}
        if (exact is not False or oracle.get("level") != "judged"
                or not isinstance(judge.get("model_id"), str)
                or not isinstance(judge.get("provider"), str)
                or provenance_judge.get("model") != judge.get("model_id")
                or provenance_judge.get("provider") != judge.get("provider")
                or provenance_judge.get("transport") != "pi-provider"):
            raise ValueError("judged oracle claim lacks matching independent judge provenance")
        return {"oracle_level": "judged", "exact_oracle_claimed": False,
            "oracle_judge": {"model_id": judge["model_id"], "provider": judge["provider"]}}
    raise ValueError("source audit declares an unsupported oracle level")


def _verify_episode(receipt_path, protection, tokenizer):
    receipt = _read(receipt_path)
    if receipt.get("schema") == "natlang.luna-static-episode-build/1":
        return _verify_static_episode(receipt_path, protection, tokenizer)
    return _verify_captured_episode(receipt_path, protection, tokenizer)


def _verify_captured_episode(receipt_path, protection, tokenizer):
    receipt_path = Path(receipt_path).resolve(strict=True)
    receipt_binding = _bind(receipt_path)
    receipt = _read(receipt_path)
    if (receipt.get("schema") != "natlang.luna-native-episode-assembly/1"
            or receipt.get("training_admission") is not False
            or receipt.get("oracle_level") != "exact"
            or receipt.get("exact_oracle_claimed") is not True
            or receipt.get("partial_turns_complete") is not True):
        raise ValueError("episode must have a complete exact-oracle Luna assembly receipt")
    graph_path = Path(receipt["graph_path"]).resolve(strict=True)
    graph_bytes = graph_path.read_bytes()
    if sha256_hex(graph_bytes) != receipt.get("graph_sha256"):
        raise ValueError("episode graph changed after its assembly receipt")
    graph_rows = [json.loads(line) for line in graph_bytes.splitlines() if line.strip()]
    if len(graph_rows) != 1:
        raise ValueError("one episode receipt must bind exactly one full graph row")
    graph = graph_rows[0]
    audit_path = Path(receipt["audit_path"]).resolve(strict=True)
    audit = _read(audit_path)
    if (sha256_file_hex(audit_path) != receipt.get("audit_sha256")
            or audit.get("training_admission") is not False
            or audit.get("oracle_level") != "exact"
            or audit.get("exact_oracle_claimed") is not True):
        raise ValueError("episode source audit changed or does not preserve exact source qualification")
    raw_path = Path(audit["raw_results"]["path"]).resolve(strict=True)
    raw_bytes = raw_path.read_bytes()
    if sha256_hex(raw_bytes) != audit["raw_results"].get("sha256"):
        raise ValueError("raw Luna result changed after source audit")
    raw_rows = [json.loads(line) for line in raw_bytes.splitlines() if line.strip()]
    if len(raw_rows) != 1:
        raise ValueError("source audit must bind one actual Luna task result")
    raw = raw_rows[0]
    outcome, oracle = raw.get("outcome") or {}, (raw.get("outcome") or {}).get("oracle") or {}
    program = (raw.get("task") or {}).get("program_ir") or {}
    if (raw.get("id") != audit.get("task_id") or outcome.get("status") != "done"
            or outcome.get("accepted") is not True or oracle.get("accepted") is not True
            or oracle.get("level") != "exact" or program.get("split") != "train"):
        raise ValueError("raw Luna result is not an accepted exact-oracle train task")
    disposition = protection.check(program)
    if disposition.get("split") != "train" or graph.get("split") != "train":
        raise ValueError("episode is not currently protected as a train world")
    expected_groups = set(program.get("source_groups") or []) | {program.get("id")}
    if not expected_groups <= set(graph.get("source_groups") or []):
        raise ValueError("episode graph dropped its task or upstream source groups")
    if (graph.get("schema") != "natlang.neuralese-native-graph/1"
            or graph.get("id") != "luna-native-episode:" + raw["id"]
            or graph.get("tokenizer_fingerprint") != tokenizer
            or len(graph.get("nodes") or []) != receipt.get("native_graph_invocations")
            or len(graph.get("teacher_source_receipts") or []) != receipt.get("source_turn_count")):
        raise ValueError("episode graph identity, tokenizer or full-turn count differs from its receipt")
    parsed = parse_graph(graph, tokenizer_fingerprint=tokenizer)
    communication = communication_classification(parsed)
    encoders = graph.get("teacher_text_encoders") or []
    actions = graph.get("teacher_episode_actions") or []
    if (not communication["producer_consumer"] or len(encoders) != 1
            or encoders[0].get("source_kind") != "authenticated-same-run-host-result"
            or sum(bool(action.get("consumer_native_block_ids")) for action in actions)
               != receipt.get("consumer_action_count")):
        raise ValueError("episode lacks its authenticated native producer and actual consumers")
    encoder_nodes = [node for node in graph["nodes"]
                     if node.get("name") == encoders[0].get("source_occurrence")]
    if (len(encoder_nodes) != 1 or encoder_nodes[0]["target_weights"] != _encoder_weights(graph, encoder_nodes[0])
            or not encoder_nodes[0].get("producing")):
        raise ValueError("teacher-text encoder must be an actual in-graph producer with zero body CE")
    action_nodes = [node for node in graph["nodes"] if node not in encoder_nodes]
    imported = [(node, edge) for node in action_nodes for edge in node.get("imports", [])
                if edge.get("source") == encoders[0]["source_occurrence"]]
    if (len(imported) != receipt.get("consumer_action_count")
            or any(not any(weight > 0 for weight in node["target_weights"])
                   for node, _ in imported)):
        raise ValueError("producer/read graph does not retain supervised consumer actions")
    return {"receipt_path": str(receipt_path), "receipt_binding": receipt_binding,
        "receipt": receipt, "graph_path": str(graph_path), "graph_bytes": graph_bytes,
        "graph": graph, "graph_binding": _bind(graph_path), "raw_binding": _bind(raw_path),
        "audit_binding": _bind(audit_path), "raw": raw, "program": program,
        "protection_disposition": disposition, "classification": communication,
        "parsed": parsed}



def _verify_static_episode(receipt_path, protection, tokenizer):
    """Revalidate a complete training-only graph without claiming model capture."""
    from natlang_neuralese.common.runtime_inventory import immutable_native_runtime_tree
    from natlang_neuralese.data.provider_teacher_source import load_verified_luna_teacher_turn, _jsonl_by_id, _read_bound
    receipt_path=Path(receipt_path).resolve(strict=True);receipt=_read(receipt_path)
    if receipt.get('training_admission') is not False or receipt.get('schema')!='natlang.luna-static-episode-build/1':
        raise ValueError('static episode receipt is malformed')
    def checked(ref):
        path=Path(ref['path']).resolve(strict=True)
        if sha256_file_hex(path)!=ref.get('sha256'):raise ValueError('static episode bound artifact changed')
        return path
    graph_path=checked(receipt['graph']); audit_path=checked(receipt['audit']); audit=_read(audit_path)
    if audit.get('training_admission') is not False:
        raise ValueError('static episode source audit must remain unadmitted')
    runtime=receipt.get('runtime') or {}; actual=immutable_native_runtime_tree(runtime.get('root',''))
    if actual!=runtime:raise ValueError('static compiler execution runtime identity changed')
    graph_rows=_jsonl(graph_path)
    if len(graph_rows)!=1:raise ValueError('static receipt must bind one whole graph')
    graph=graph_rows[0]; provenance=graph.get('static_training_provenance') or {}
    if (provenance.get('schema') not in STATIC_PROVENANCES or provenance.get('audit_sha256')!=receipt['audit']['sha256']
        or provenance.get('execution_runtime')!=runtime or graph.get('student_native_identity') is not None
        or graph.get('tokenizer_fingerprint')!=tokenizer or graph.get('split')!='train'):
        raise ValueError('static graph compiler provenance, tokenizer or split differs')
    raw_path=checked(audit['raw_results']);raw_rows=_jsonl(raw_path)
    if len(raw_rows)!=1:raise ValueError('static source must bind one complete teacher result')
    raw=raw_rows[0];outcome=raw.get('outcome') or {};oracle=outcome.get('oracle') or {};program=(raw.get('task') or {}).get('program_ir') or {}
    oracle_claim = _verified_oracle_claim(audit, raw)
    if (raw.get('id')!=audit.get('task_id') or outcome.get('status')!='done'
        or program.get('split')!='train'):
        raise ValueError('static teacher result is not an accepted train task')
    _teacher_actor_binding(raw,_bind(raw_path));disposition=protection.check(program)
    if disposition.get('split')!='train' or not (set(program.get('source_groups') or [])|{program.get('id')})<=set(graph.get('source_groups') or []):
        raise ValueError('static graph differs from current protected train world')
    ref=audit['python_corpus_conversion'];data=_read_bound(ref['path'],ref['sha256']); rows=[json.loads(line) for line in data.splitlines() if line.strip()]
    source_receipts=graph.get('teacher_source_receipts') or []
    if len(rows)!=receipt.get('source_turn_count') or len(rows)!=len(source_receipts) or graph.get('id')!='luna-native-episode:'+raw['id']:
        raise ValueError('static graph omits part of the actual decision trajectory')
    actions=(graph.get('nodes') or [])[-len(rows):]
    for row,saved,node in zip(rows,source_receipts,actions,strict=True):
        opening=(row.get('native_source_conversion') or {}).get('source_messages')
        actual_turn=load_verified_luna_teacher_turn(str(audit_path),row['id'],replay_messages=opening,replay_tools=row.get('tools') or [],allow_context_only=True)
        if actual_turn['source_receipt']!=saved:raise ValueError('static graph source turn receipt changed')
        positive=any(weight>0 for weight in node.get('target_weights') or [])
        if positive != saved.get('target_supervision'):raise ValueError('static graph supervised/context disposition changed')
    parsed=parse_graph(graph,tokenizer_fingerprint=tokenizer);classification=communication_classification(parsed)
    if (len(parsed.plan.invocations)!=receipt.get('nodes') or sum(len(x.imports) for x in parsed.plan.invocations)!=receipt.get('imports')
        or sum(w>0 for w in parsed.target_weights)!=receipt.get('supervised_tokens') or not classification['producer_consumer']):
        raise ValueError('static graph node, positive-target or import census differs')
    encoders=graph.get('teacher_text_encoder_sources') or [];nodes={n['name']:n for n in graph['nodes']}
    for item in encoders:
        node=nodes.get(item.get('producer'))
        if node is None or item.get('producer_span') not in node.get('producing',[]):raise ValueError('static teacher source producer span differs')
        if item.get('source_kind')=='authenticated-host-text-encoder' and node['target_weights']!=_encoder_weights(graph,node):
            raise ValueError('host text encoder must keep zero body CE (only a declared producer close is supervised)')
    return {'kind':'static','receipt_path':str(receipt_path),'receipt_binding':_bind(receipt_path),'receipt':receipt,
        'graph_path':str(graph_path),'graph_bytes':graph_path.read_bytes(),'graph':graph,'graph_binding':_bind(graph_path),
        'raw_binding':_bind(raw_path),'audit_binding':_bind(audit_path),'raw':raw,'program':program,
        'protection_disposition':disposition,'classification':classification,'parsed':parsed,
        'oracle_claim':oracle_claim}


def _normalize_base_receipt(receipt):
    """Accept a sealed Big2 append using its actual published field names."""
    if receipt.get("schema") == "natlang.big2-source-objective-recipe-successor/1":
        required = ("successor_source", "ce_policy", "recipe", "geometry", "family_map",
            "objective_census")
        if (receipt.get("training_admission") is not False
                or receipt.get("optimizer_exposure") is not False
                or not isinstance(receipt.get("successor_rows"), int)
                or receipt["successor_rows"] < 1
                or not isinstance(receipt.get("current_protection_digest"), str)):
            raise ValueError("Big2 prepared successor is not source-only and unexposed")
        for key in required:
            ref = receipt.get(key)
            if not isinstance(ref, dict) or not ref.get("path") or not ref.get("sha256"):
                raise ValueError("Big2 successor lacks bound " + key)
            path = Path(ref["path"]).resolve(strict=True)
            if (sha256_file_hex(path) != ref["sha256"]
                    or ("bytes" in ref and path.stat().st_size != ref["bytes"])):
                raise ValueError("Big2 successor binding changed: " + key)
        predecessor = receipt.get("predecessor_source")
        if not isinstance(predecessor, dict) or not predecessor.get("path") or not predecessor.get("sha256"):
            raise ValueError("Big2 successor lacks its exact source predecessor")
        predecessor_path = Path(predecessor["path"]).resolve(strict=True)
        if (sha256_file_hex(predecessor_path) != predecessor["sha256"]
                or ("bytes" in predecessor and predecessor_path.stat().st_size != predecessor["bytes"])):
            raise ValueError("Big2 predecessor source binding changed")
        return {**receipt, "schema": "natlang.native-quality-successor/1",
            "source": receipt["successor_source"], "rows_after": receipt["successor_rows"],
            "protection_digest": receipt["current_protection_digest"],
            "status": "source-ce-recipe-geometry-qualified-not-published"}
    if receipt.get('schema')!='natlang.big2-sft-source-append/1':return receipt
    if (receipt.get('training_admission') is not False or receipt.get('optimizer_exposure') is not False
        or not isinstance(receipt.get('rows'),int) or receipt['rows']<1
        or not isinstance(receipt.get('objective_census'),dict)):
        raise ValueError('Big2 predecessor is not fully qualified and unadopted')
    for key in ('source_predecessor','source_successor','ce_policy','recipe','geometry','objective_census','family_map'):
        ref=receipt.get(key)
        if not isinstance(ref,dict) or not ref.get('path') or not ref.get('sha256'):
            raise ValueError('Big2 predecessor lacks bound '+key)
        path=Path(ref['path']).resolve(strict=True)
        if sha256_file_hex(path)!=ref['sha256'] or ('bytes' in ref and path.stat().st_size!=ref['bytes']):
            raise ValueError('Big2 predecessor binding changed: '+key)
    return {**receipt,'schema':'natlang.luna-native-successor-append/1',
            'source':receipt['source_successor'],'status':'source-ce-recipe-geometry-qualified-not-published'}

def build(base_dir, episode_receipt, output_dir):
    base_dir = Path(base_dir).resolve(strict=True)
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise ValueError("successor output is immutable; choose a fresh path")
    base_receipt_path = base_dir / "receipt.json"
    base_receipt_binding = _bind(base_receipt_path)
    base_receipt = _normalize_base_receipt(_read(base_receipt_path))
    if (base_receipt.get("schema") not in {"natlang.native-quality-successor/1", "natlang.luna-native-successor-append/1"}
            or base_receipt.get("training_admission") is not False
            or base_receipt.get("status") != "source-ce-recipe-geometry-qualified-not-published"):
        raise ValueError("base must be the exact reviewed source-only native quality successor")
    source_path = Path(base_receipt["source"]["path"]).resolve(strict=True)
    if _bind(source_path) != base_receipt["source"]:
        raise ValueError("base train-source manifest changed after its quality receipt")
    row_count = base_receipt.get("rows_after", base_receipt.get("rows"))
    if type(row_count) is not int or row_count < 1:
        raise ValueError("base receipt lacks its exact predecessor row count")
    map_path = Path(base_receipt["family_map"]["path"]).resolve(strict=True)
    if _bind(map_path) != base_receipt["family_map"]:
        raise ValueError("base logical-family map changed")
    prior_map = _read(map_path)
    if (prior_map.get("schema") != "natlang.logical-family-map/1"
            or prior_map.get("source_sha256") != base_receipt["source"]["sha256"]
            or len(prior_map.get("rows") or []) != row_count):
        raise ValueError("base family map does not cover the exact candidate source")
    geometry_path = Path(base_receipt["geometry"]["path"]).resolve(strict=True)
    if any(_bind(geometry_path).get(key) != value for key,value in base_receipt["geometry"].items()):
        raise ValueError("base packed-capacity geometry changed")
    prior_geometry = _read(geometry_path)
    tokenizer = prior_geometry["identity"]["tokenizer"]
    roles = prior_geometry["identity"]["roles"]

    protection = ProtectedWorlds.load()
    if protection.digest != base_receipt.get("protection_digest", base_receipt.get("current_protection_digest")):
        raise ValueError("current protection union differs from the reviewed base candidate")
    episode = _verify_episode(episode_receipt, protection, tokenizer)
    old_ce_path = Path(base_receipt["ce_policy"]["path"]).resolve(strict=True)
    if _bind(old_ce_path) != base_receipt["ce_policy"]:
        raise ValueError("base CE policy changed after the quality receipt")
    old_recipe_path = Path(base_receipt["recipe"]["path"]).resolve(strict=True)
    if _bind(old_recipe_path) != base_receipt["recipe"]:
        raise ValueError("base recipe changed after the quality receipt")

    output_dir.mkdir(parents=True)
    graph_path = output_dir / "luna-native-episode.jsonl"
    _write_bytes(graph_path, episode["graph_bytes"])
    graph_binding = _bind(graph_path)
    source_manifest = _read(source_path)
    if source_manifest.get("schema") != "natlang.native-graph-sources/1":
        raise ValueError("base source manifest schema is unsupported")
    if any(item.get("sha256") == graph_binding["sha256"] for item in source_manifest.get("sources", [])):
        raise ValueError("episode graph bytes already occur in the base source tree")
    source_manifest["sources"].append({"kind": "native-graphs", **graph_binding,
        "training_priority": 1})
    candidate_source_path = output_dir / "train-sources.json"
    write_json_durable(candidate_source_path, source_manifest)

    indexed_graph = NativeGraphSource(graph_path, tokenizer, required_split="train", role_ids=roles)
    if len(indexed_graph.entries) != 1:
        raise ValueError("episode graph must independently index as one train row")
    new_source, _, geometry_ref, geometry_scan = capacity.prepare_source(
        candidate_source_path, tokenizer, roles, output_dir, prior=base_receipt["geometry"])
    if len(new_source.entries) != row_count + 1:
        raise ValueError("candidate source does not append exactly one whole graph")
    if [(row["id"], row["binding"]) for row in new_source.entries[:row_count]] != [
            (row["id"], row["binding"]) for row in prior_map["rows"]]:
        raise ValueError("source extension changed the exact quality-corrected predecessor prefix")
    addition = {"id": indexed_graph.entries[0]["id"],
        "binding": indexed_graph.entries[0]["binding"],
        "family": "native-communication" if episode["classification"]["communication"] else "natlang"}
    if (new_source.entries[row_count]["id"], new_source.entries[row_count]["binding"]
            ) != (addition["id"], addition["binding"]):
        raise ValueError("indexed successor addition differs from the authenticated episode graph")
    if addition["id"] in {row["id"] for row in prior_map["rows"]}:
        raise ValueError("episode graph ID already occurs in predecessor source")

    family_path = output_dir / "families.json"
    write_json_durable(family_path, {"schema": "natlang.logical-family-map/1",
        "source_sha256": _bind(candidate_source_path)["sha256"],
        "rows": [*prior_map["rows"], addition]})

    ce_policy = _read(old_ce_path)
    old_loaded_policy = load_policy(_bind(old_ce_path))
    prior_objective = identity(policy_objective(old_loaded_policy))["sha256"]
    ce_policy["inventory"] = [_manifest_bind(candidate_source_path)]
    ce_policy["objective"]["family_balance"]["map"] = _manifest_bind(family_path)
    ce_policy["objective_transition"] = {"from": prior_objective}
    if any(item.get("sha256") == graph_binding["sha256"] for item in ce_policy["sources"]):
        raise ValueError("episode graph is already present in the base CE declarations")
    ce_policy["sources"].append({"sha256": graph_binding["sha256"],
        "encoding": "native-row-objectives-v1",
        "context_weight": ce_policy["source_contract"]["default_raw_context_weight"]})
    ce_policy["sources"].sort(key=lambda row: row["sha256"])
    ce_policy["source_contract"]["inventory"] = [_bind(candidate_source_path)]
    ce_policy["source_contract"]["active_sources"].append(graph_binding)
    ce_policy["source_contract"]["active_sources"].sort(key=lambda row: row["sha256"])
    ce_policy["source_contract"]["active_source_count"] = len(ce_policy["source_contract"]["active_sources"])
    ce_path = output_dir / "ce-policy.json"
    write_json_durable(ce_path, ce_policy)
    policy = prepare_policy(load_policy(_bind(ce_path)), new_source)

    recipe = _read(old_recipe_path)
    recipe["input_bindings"]["v4_previous_train_graphs"] = _manifest_bind(source_path)
    recipe["input_bindings"]["v4_train_graphs"] = _manifest_bind(candidate_source_path)
    recipe["input_bindings"]["v4_ce_policy"] = _manifest_bind(ce_path)
    recipe_path = output_dir / "recipe-continuation.json"
    write_json_durable(recipe_path, recipe)
    old_recipe, new_recipe = load_recipe(old_recipe_path), load_recipe(recipe_path)
    stage = lambda value: next(item for item in value["stages"] if item["id"] == "native_joint_sft_v4")
    old_roles = resolve_stage_inputs(old_recipe, stage(old_recipe), {})
    new_roles = resolve_stage_inputs(new_recipe, stage(new_recipe), {})
    if not _data_continuation({key: value for key, value in old_roles.items()
                               if key != "ce_supervision_policy"},
                              {key: value for key, value in new_roles.items()
                               if key != "ce_supervision_policy"}):
        raise ValueError("candidate does not preserve the complete named-role data continuation")
    for key in ("native_graphs", "foundation_native_graphs"):
        if (Path(old_roles[key]["path"]).resolve() != source_path
                or Path(new_roles[key]["path"]).resolve() != candidate_source_path):
            raise ValueError(f"shared train/foundation role {key} did not advance together")
    if old_roles.get("held_native_graphs") != new_roles.get("held_native_graphs"):
        raise ValueError("candidate changed the protected held source")
    prior_train = NativeGraphSource(source_path, tokenizer, required_split="train", role_ids=roles)
    extension = extension_policy(old_roles, new_roles, prior_train=prior_train, candidate_train=new_source)
    objective, objective_transition = continuation_identity(
        {"supervision_objective": identity(policy_objective(old_loaded_policy))}, extension)

    descriptor_path = output_dir / "sft-capacity-descriptor.json"
    write_json_durable(descriptor_path, {"schema": "natlang.native-sft-capacity-cases/1",
        "source": _bind(candidate_source_path), "geometry": geometry_ref})
    new_source.set_ce_policy(policy)
    new_rows, weighted_mass, supervised_tokens = [], 0.0, 0
    for index, example in enumerate(new_source.iter_from(row_count), row_count):
        if example.id != new_source.entries[index]["id"]:
            raise ValueError("candidate source changed while evaluating appended graph")
        amount = sum(example.target_weights) * example.loss_scale
        if not math.isfinite(amount) or amount <= 0:
            raise ValueError("appended whole graph has no positive effective supervised CE mass")
        new_rows.append(example.id)
        weighted_mass += amount
        supervised_tokens += sum(weight > 0 for weight in example.target_weights)
    if new_rows != [addition["id"]]:
        raise ValueError("CE verification did not visit exactly the appended graph")

    qualification_path = output_dir / "episode-qualification.json"
    is_static=episode.get('kind')=='static'
    oracle_claim = episode.get("oracle_claim") or {"oracle_level": "exact", "exact_oracle_claimed": True}
    qualification = {"schema": "natlang.luna-native-append-qualification/1",
        "training_admission": False, **oracle_claim,
        "base_successor": base_receipt_binding, "episode_source_kind":'static-training-graph' if is_static else 'captured-native-episode',
        "episode_assembly": episode["receipt_binding"], "episode_audit": episode["audit_binding"], "raw_result": episode["raw_binding"],
        "graph": graph_binding, "current_protection_digest": protection.digest, "train_disposition": episode["protection_disposition"],
        "source_groups": episode["graph"]["source_groups"],
        "whole_graph": {"rows": 1, "invocations": len(episode["parsed"].plan.invocations),
            "producer_consumer": episode["classification"]["producer_consumer"],
            "encoder_count": len(episode["graph"].get("teacher_text_encoder_sources",[]) if is_static else episode["graph"]["teacher_text_encoders"]),
            "consumer_actions": sum(bool(n.get('imports')) for n in episode['graph']['nodes']) if is_static else episode["receipt"]["consumer_action_count"],
            "supervised_turns": sum(bool(r.get('target_supervision')) for r in episode['graph']['teacher_source_receipts']) if is_static else episode["receipt"]["target_supervised_turns"],
            "supervised_tokens": supervised_tokens,
            # 0: historical all-zero encoders; else the declared close-only policy (body stays zero).
            "encoder_target_weights": episode["graph"].get("producer_close_supervision") or 0},
        "teacher_actor": _teacher_actor_binding(episode["raw"], episode["raw_binding"]),
        **({'static_compiler_runtime':episode['receipt']['runtime'], 'tokenizer_fingerprint':tokenizer,
            'scope':'Training-only token graph; no captured student state or checkpoint is claimed.'} if is_static else
           {'student_encoder': {"checkpoint_sha256": episode["receipt"]["native_checkpoint_sha256"],
            "tokenizer_fingerprint": episode["receipt"]["tokenizer_fingerprint"],
            "serving_config_identity_sha256": episode["receipt"]["serving_config_identity_sha256"]}})}
    write_json_durable(qualification_path, qualification)

    source_rows = _read(candidate_source_path).get("sources") or []
    result = {"schema": "natlang.luna-native-successor-append/1",
        "predecessor": base_receipt_binding, "predecessor_source": _bind(source_path),
        "source": _bind(candidate_source_path), "source_graph": graph_binding,
        "episode_qualification": _bind(qualification_path),
        "family_map": _bind(family_path), "ce_policy": _bind(ce_path),
        "recipe": _bind(recipe_path), "geometry": geometry_ref,
        "capacity_descriptor": _bind(descriptor_path),
        "staging_code": [_bind(path) for path in (Path(__file__).resolve(),
            ROOT / "training/neuralese/natlang_neuralese/common/custody.py",
            ROOT / "training/neuralese/natlang_neuralese/common/hashing.py",
            ROOT / "training/neuralese/natlang_neuralese/common/jsonio.py",
            ROOT / "training/neuralese/natlang_neuralese/data/native_graph_schema.py",
            ROOT / "training/neuralese/natlang_neuralese/data/native_sources.py",
            ROOT / "training/neuralese/natlang_neuralese/data/decision_graph.py",
            ROOT / "training/neuralese/natlang_neuralese/model/native_graph_plan.py",
            ROOT / "training/neuralese/natlang_neuralese/train/native_packed_capacity.py",
            ROOT / "training/neuralese/natlang_neuralese/train/native_supervision.py",
            ROOT / "training/neuralese/natlang_neuralese/train/own_output_supervision.py",
            ROOT / "training/neuralese/natlang_neuralese/train/native_resident_ce.py",
            ROOT / "training/neuralese/natlang_neuralese/train/recipe.py")],
        "predecessor_rows": row_count, "rows": len(new_source.entries), "new_rows": len(new_rows),
        "new_graph_ids": new_rows, "new_families": [addition["family"]],
        "source_manifest_entries": len(source_rows), "supervised_tokens": supervised_tokens,
        "new_effective_weighted_mass": weighted_mass, "zero_effective_rows": [],
        "geometry_scan": geometry_scan, "family_census": policy["_family_census"],
        "objective": objective, "objective_transition": objective_transition,
        "current_protection_digest": protection.digest,
        "base_quality_mask_receipts": {key: base_receipt[key] for key in (
            "masked_graph", "qwen15_review", "qwen36_review", "predecessor_v7",
            "predecessor_source", "checks") if key in base_receipt},
        "status": "source-ce-recipe-geometry-qualified-not-published",
        "training_admission": False,
        "limitations": ["Future-only source adoption remains separate; no learner exposure is claimed."]}
    result_path = output_dir / "receipt.json"
    write_json_durable(result_path, result)
    # Recheck all immutable base bindings after the full append validation.
    if (_bind(base_receipt_path) != base_receipt_binding or _bind(source_path) != base_receipt["source"]
            or _bind(map_path) != base_receipt["family_map"]
            or _bind(old_ce_path) != base_receipt["ce_policy"]
            or _bind(old_recipe_path) != base_receipt["recipe"]):
        raise ValueError("base successor changed while staging this append")
    return {"receipt": _bind(result_path), **result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--episode-receipt", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.base_dir, args.episode_receipt, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
