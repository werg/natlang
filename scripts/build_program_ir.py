#!/usr/bin/env python3
"""Compile labeled decisions or SCONE state sequences to durable program IR."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from generate_external import build_task_groups, read_tasks, select_tasks
from program_ir import VERSION, decision_record, digest, validate
from program_ir import scene_node_type


def decisions(args):
    tasks, input_hashes = read_tasks(args.inputs)
    tasks, skipped = select_tasks(tasks, args.allow_unknown_license, set(args.splits))
    for number, (program_id, source_tasks) in enumerate(
            build_task_groups(tasks, args.families, args.batch_size)):
        if args.max_programs is not None and number >= args.max_programs:
            break
        kind = {"leaf": "decision_leaf", "map": "decision_map_report",
                "case": "decision_case_report"}[program_id.split(":", 1)[0]]
        yield decision_record(kind, program_id, source_tasks)
    args.source_manifest = {"input_sha256": input_hashes, "selected_tasks": len(tasks),
                            "skipped": dict(skipped), "families": args.families,
                            "batch_size": args.batch_size}


def scone(args):
    input_hash = hashlib.sha256(args.archive.read_bytes()).hexdigest()
    args.source_manifest = {"archive": str(args.archive), "archive_sha256": input_hash,
                            "domains": args.domains, "splits": args.splits}
    emitted = 0
    with zipfile.ZipFile(args.archive) as zf:
        for split in args.splits:
            for domain in args.domains:
                name = f"rlong/{domain}-{split}.tsv"
                with zf.open(name) as stream:
                    for line_no, raw in enumerate(stream, 1):
                        if args.max_programs is not None and emitted >= args.max_programs:
                            return
                        fields = raw.decode("utf-8").rstrip("\r\n").split("\t")
                        if len(fields) < 4 or len(fields) % 2:
                            raise ValueError(f"{name}:{line_no}: malformed state sequence")
                        source_id = fields[0]
                        steps = [{"utterance": fields[i], "before": fields[i-1], "after": fields[i+1]}
                                 for i in range(2, len(fields), 2)]
                        record = {"version": VERSION, "id": f"scone:{domain}:{source_id}",
                                  "kind": "state_sequence", "source": "SCONE",
                                  "split": split, "source_ids": [source_id],
                                  "source_groups": [f"{domain}:{source_id}"],
                                  "source_revisions": [input_hash], "license": "cc-by-sa-4.0",
                                  "gold_sources": ["dataset-intermediate-states"],
                                  "semantics": {"domain": domain, "initial_state": fields[1],
                                                "steps": steps}}
                        yield validate(record)
                        emitted += 1


def sgd(args):
    root = args.dataset
    emitted = 0
    source_files = []
    for split in args.splits:
        schema_path = root / split / "schema.json"
        schema = {item["service_name"]: item for item in json.loads(schema_path.read_text())}
        source_files.append(str(schema_path))
        for path in sorted((root / split).glob("dialogues_*.json")):
            source_files.append(str(path))
            revision = hashlib.sha256(path.read_bytes()).hexdigest()
            for dialogue in json.loads(path.read_text()):
                if args.max_programs is not None and emitted >= args.max_programs:
                    args.source_manifest = {"dataset": str(root), "files_read": source_files,
                                            "splits": args.splits}
                    return
                for service in dialogue["services"]:
                    if args.max_programs is not None and emitted >= args.max_programs:
                        args.source_manifest = {"dataset": str(root), "files_read": source_files,
                                                "splits": args.splits}
                        return
                    service_schema = schema[service]
                    state = {"active_intent": "NONE", "requested_slots": [], "slot_values": {}}
                    steps = []
                    cursor = 0
                    for index, turn in enumerate(dialogue["turns"]):
                        if turn["speaker"] != "USER":
                            continue
                        frame = next((f for f in turn["frames"] if f["service"] == service), None)
                        if frame is None or "state" not in frame:
                            continue
                        utterance = "\n".join(t["speaker"] + ": " + t["utterance"]
                                              for t in dialogue["turns"][cursor:index+1])
                        after = frame["state"]
                        steps.append({"utterance": utterance, "before": state, "after": after})
                        state, cursor = after, index + 1
                    if not steps:
                        continue
                    slot_text = "; ".join(s["name"] + ": " + s["description"]
                                          for s in service_schema["slots"])
                    intent_text = "; ".join(i["name"] + ": " + i["description"]
                                            for i in service_schema["intents"])
                    context = (f"Service {service}: {service_schema['description']}. "
                               f"Intents: {intent_text}. Slots: {slot_text}. "
                               "Track active_intent, requested_slots and slot_values from the dialogue.")
                    record = {"version": VERSION,
                              "id": f"sgd:{split}:{dialogue['dialogue_id']}:{service}",
                              "kind": "state_sequence", "source": "SGD", "split": split,
                              "source_ids": [dialogue["dialogue_id"]],
                              "source_groups": [f"{split}:{dialogue['dialogue_id']}"],
                              "source_revisions": [revision], "license": "cc-by-sa-4.0",
                              "gold_sources": ["annotated-dialogue-state"],
                              "semantics": {"domain": service, "task_context": context,
                                            "state_type": "{ active_intent: Text, requested_slots: Text[], slot_values: Dict<Text[]> }",
                                            "initial_state": {"active_intent": "NONE", "requested_slots": [],
                                                              "slot_values": {}}, "steps": steps}}
                    yield validate(record)
                    emitted += 1
    args.source_manifest = {"dataset": str(root), "files_read": source_files,
                            "splits": args.splits}


def parse_numeric_program(source):
    operations = {"add", "subtract", "multiply", "divide", "exp", "greater"}
    chunks = re.findall(r"([a-z_]+)\(([^()]*)\)", source)
    if not chunks or len(chunks) != source.count("),") + 1:
        return None
    steps = []
    values = []
    try:
        for op, raw_args in chunks:
            if op not in operations:
                return None
            fields = [field.strip() for field in raw_args.split(",")]
            if len(fields) != 2:
                return None
            operands = []
            args = []
            for field in fields:
                if field.startswith("#"):
                    index = int(field[1:])
                    if index >= len(values):
                        return None
                    args.append(index)
                    operands.append(values[index])
                else:
                    numeric = field.removeprefix("const_")
                    percent = numeric.endswith("%")
                    value = float(numeric.rstrip("%")) / (100 if percent else 1)
                    args.append(value)
                    operands.append(value)
            a, b = operands
            result = {"add": lambda: a + b, "subtract": lambda: a - b,
                      "multiply": lambda: a * b, "divide": lambda: a / b,
                      "exp": lambda: a ** b,
                      "greater": lambda: "yes" if a > b else "no"}[op]()
            if not isinstance(result, (int, float, str)) or isinstance(result, float) and not math.isfinite(result):
                return None
            steps.append({"op": op, "args": args})
            values.append(result)
    except (ValueError, ZeroDivisionError, OverflowError):
        return None
    return steps, values[-1]


def finqa(args):
    counts = Counter()
    emitted = 0
    input_hashes = {}
    for split in args.splits:
        path = args.dataset / "dataset" / f"{split}.json"
        input_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        for row in json.loads(path.read_text()):
            if args.max_programs is not None and emitted >= args.max_programs:
                args.source_manifest = {"input_sha256": input_hashes, "skipped": dict(counts)}
                return
            qa = row["qa"]
            parsed = parse_numeric_program(qa["program"])
            if parsed is None:
                counts["unsupported_program"] += 1
                continue
            steps, computed = parsed
            answer = qa["exe_ans"]
            if isinstance(computed, str):
                if computed != answer:
                    counts["answer_mismatch"] += 1
                    continue
                answer_type = "Text"
            else:
                try:
                    numeric_answer = float(answer)
                except (ValueError, TypeError):
                    counts["answer_mismatch"] += 1
                    continue
                if abs(computed - numeric_answer) > max(0.0001, abs(numeric_answer) * 1e-7):
                    counts["answer_mismatch"] += 1
                    continue
                answer, answer_type = numeric_answer, "Num"
            evidence = "\n".join(qa.get("gold_inds", {}).values())
            if row.get("table"):
                evidence += "\nTable:\n" + "\n".join(" | ".join(map(str, cells)) for cells in row["table"])
            record = {"version": VERSION, "id": f"finqa:{split}:{row['id']}",
                      "kind": "numeric_program", "source": "FinQA", "split": split,
                      "source_ids": [row["id"]], "source_groups": [row["id"]],
                      "source_revisions": [input_hashes[str(path)]], "license": "mit",
                      "gold_sources": ["dataset-executable-program"],
                      "semantics": {"question": qa["question"], "evidence": evidence,
                                    "steps": steps, "answer_type": answer_type, "answer": answer}}
            yield validate(record)
            emitted += 1
    args.source_manifest = {"input_sha256": input_hashes, "skipped": dict(counts)}


def execute_scene_nodes(nodes, scene):
    objects = scene["objects"]
    outputs = []
    for node in nodes:
        op = node["function"]
        scene_node_type(op)
        inputs = [outputs[index] for index in node["inputs"]]
        values = node.get("value_inputs", [])
        if op == "scene":
            result = list(range(len(objects)))
        elif op.startswith("filter_"):
            attr = op.removeprefix("filter_")
            result = [i for i in inputs[0] if objects[i][attr] == values[0]]
        elif op == "unique":
            if len(inputs[0]) != 1:
                raise ValueError("nonunique selection")
            result = inputs[0][0]
        elif op == "relate":
            result = scene["relationships"][values[0]][inputs[0]]
        elif op.startswith("same_"):
            attr = op.removeprefix("same_")
            result = [i for i, obj in enumerate(objects)
                      if i != inputs[0] and obj[attr] == objects[inputs[0]][attr]]
        elif op == "union":
            result = sorted(set(inputs[0]) | set(inputs[1]))
        elif op == "intersect":
            result = [i for i in inputs[0] if i in inputs[1]]
        elif op == "count":
            result = len(inputs[0])
        elif op == "exist":
            result = bool(inputs[0])
        elif op.startswith("query_"):
            result = objects[inputs[0]][op.removeprefix("query_")]
        elif op.startswith("equal_"):
            result = inputs[0] == inputs[1]
        elif op == "greater_than":
            result = inputs[0] > inputs[1]
        elif op == "less_than":
            result = inputs[0] < inputs[1]
        else:
            raise ValueError(op)
        outputs.append(result)
    return outputs[-1]


def clevr(args):
    counts = Counter()
    emitted = 0
    archive_hash = hashlib.sha256(args.archive.read_bytes()).hexdigest()
    args.source_manifest = {"archive": str(args.archive), "archive_sha256": archive_hash,
                            "splits": args.splits, "every": args.every, "skipped": {}}
    with zipfile.ZipFile(args.archive) as zf:
        for split in args.splits:
            prefix = "CLEVR_v1.0/"
            with zf.open(prefix + f"scenes/CLEVR_{split}_scenes.json") as stream:
                scenes = {scene["image_index"]: scene for scene in json.load(stream)["scenes"]}
            with zf.open(prefix + f"questions/CLEVR_{split}_questions.json") as stream:
                questions = json.load(stream)["questions"]
            for row in questions:
                if row["question_index"] % args.every:
                    continue
                if args.max_programs is not None and emitted >= args.max_programs:
                    args.source_manifest["skipped"] = dict(counts)
                    return
                source_scene = scenes[row["image_index"]]
                scene = {"objects": [{key: obj[key] for key in ("color", "size", "shape", "material")}
                                     for obj in source_scene["objects"]],
                         "relationships": source_scene["relationships"]}
                try:
                    answer = execute_scene_nodes(row["program"], scene)
                except (ValueError, IndexError, KeyError, TypeError):
                    counts["unsupported_or_invalid_program"] += 1
                    continue
                target = row["answer"]
                displayed = ("yes" if answer else "no") if isinstance(answer, bool) else str(answer)
                if displayed != target:
                    counts["answer_mismatch"] += 1
                    continue
                answer_type = scene_node_type(row["program"][-1]["function"])
                if answer_type not in {"Num", "Bool", "Text"}:
                    counts["non_scalar_answer"] += 1
                    continue
                record = {"version": VERSION, "id": f"clevr:{split}:{row['question_index']}",
                          "kind": "scene_program", "source": "CLEVR", "split": split,
                          "source_ids": [str(row["question_index"])],
                          "source_groups": [f"{split}:image:{row['image_index']}"],
                          "source_revisions": [archive_hash], "license": "cc-by-4.0",
                          "gold_sources": ["dataset-functional-program"],
                          "semantics": {"question": row["question"], "scene": scene,
                                        "nodes": row["program"], "answer_type": answer_type,
                                        "answer": answer}}
                yield validate(record)
                emitted += 1
            del questions, scenes
    args.source_manifest["skipped"] = dict(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="adapter", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out", type=Path, required=True)
    common.add_argument("--max-programs", type=int)
    dec = sub.add_parser("decisions", parents=[common])
    dec.add_argument("inputs", type=Path, nargs="+")
    dec.add_argument("--families", nargs="+", choices=["leaf", "map_report", "case_report"],
                     default=["leaf", "map_report", "case_report"])
    dec.add_argument("--batch-size", type=int, default=4)
    dec.add_argument("--splits", nargs="+", default=["train"])
    dec.add_argument("--allow-unknown-license", action="store_true")
    sc = sub.add_parser("scone", parents=[common])
    sc.add_argument("--archive", type=Path, required=True)
    sc.add_argument("--domains", nargs="+", default=["alchemy", "scene", "tangrams"])
    sc.add_argument("--splits", nargs="+", default=["train"])
    gd = sub.add_parser("sgd", parents=[common])
    gd.add_argument("--dataset", type=Path, required=True)
    gd.add_argument("--splits", nargs="+", default=["train"])
    fq = sub.add_parser("finqa", parents=[common])
    fq.add_argument("--dataset", type=Path, required=True)
    fq.add_argument("--splits", nargs="+", default=["train"])
    cv = sub.add_parser("clevr", parents=[common])
    cv.add_argument("--archive", type=Path, required=True)
    cv.add_argument("--splits", nargs="+", default=["train"])
    cv.add_argument("--every", type=int, default=1, help="sample every Kth question across scenes")
    args = parser.parse_args()
    if args.max_programs is not None and args.max_programs < 1:
        parser.error("max-programs must be positive")
    if args.adapter == "decisions" and args.batch_size < 2:
        parser.error("batch-size must be at least 2")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    content_hash = hashlib.sha256()
    seen = set()
    if args.adapter == "clevr" and args.every < 1:
        parser.error("every must be positive")
    adapter = {"decisions": decisions, "scone": scone, "sgd": sgd,
               "finqa": finqa, "clevr": clevr}[args.adapter]
    with args.out.open("w") as stream:
        for record in adapter(args):
            if record["id"] in seen:
                raise ValueError(f"duplicate IR program id: {record['id']}")
            seen.add(record["id"])
            line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            stream.write(line)
            content_hash.update(line.encode())
            counts[record["kind"]] += 1
    manifest = {"version": VERSION, "adapter": args.adapter, "source": args.source_manifest,
                "programs": sum(counts.values()), "kinds": dict(counts),
                "ir_sha256": content_hash.hexdigest()}
    args.out.with_suffix(args.out.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
