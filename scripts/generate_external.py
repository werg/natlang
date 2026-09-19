#!/usr/bin/env python3
"""Legacy entry point for normalized decisions; CLI routes through program IR.

Input schema is documented in DATASET_TRAJECTORIES_PLAN.md. This script never
calls a model: the supplied gold is the semantic leaf oracle, while the
reference policy generates every interpreter action through the real harness.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from natlang.gen.programs import Plan, Program
from natlang.native import _strip_private
from generate import run_program


APPROVED_LICENSES = {"apache-2.0", "cc0-1.0", "mit", "cc-by-4.0"}


def _state_text(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)


def _labels(task):
    values = task.get("labels")
    if not isinstance(values, list) or len(values) < 2 or any(not isinstance(v, str) or not v for v in values):
        raise ValueError("labels must contain at least two nonempty strings")
    if len(values) != len(set(values)):
        raise ValueError("labels must be unique")
    if task.get("kind") == "boolean" and set(values) != {"false", "true"}:
        raise ValueError("boolean labels must be false and true")
    if task.get("kind") not in {"choice", "boolean", "score"}:
        raise ValueError("kind must be choice, boolean, or score")
    return values


def _gold(task):
    value = task.get("gold")
    if isinstance(value, bool):
        value = str(value).lower()
    if isinstance(value, int):
        value = str(value)
    if value not in _labels(task):
        raise ValueError(f"gold {value!r} is not an allowed label")
    return value


def _typed_value(task):
    value = _gold(task)
    return value == "true" if task["kind"] == "boolean" else value


def _type_parts(task):
    if task["kind"] == "boolean":
        return "Bool", {}
    union = " | ".join(json.dumps(value, ensure_ascii=False) for value in _labels(task))
    return "Label", {"Label": union}


def _rubric(task):
    criteria = task.get("criteria") or {}
    return "\n".join(f"{name}: {criteria.get(name) or name}" for name in _labels(task))


def leaf_program(task):
    result_type, types = _type_parts(task)
    body = (
        "Read `args/state` and answer this question: " + task["instruction"].strip() +
        "\nUse the choices and descriptions in `args/rubric`. Return one allowed value."
    )
    root = {"$lambda": {"type": f"Lambda<{{ state: Text, rubric: Text }}, {result_type}>",
                        "types": types, "instructions": body}}
    expected = _typed_value(task)
    return Program("external_leaf", root,
                   {"state": _state_text(task["state"]), "rubric": _rubric(task)}, expected,
                   {body: Plan("leaf", gold=lambda args, value=expected: value,
                               note="Applied the question to the state.")})


def map_report_program(tasks):
    first = tasks[0]
    labels = _labels(first)
    result_type, types = _type_parts(first)
    types = {**types, "Report": f"{{ labels: {result_type}[], counts: Dict<Num> }}"}
    body = (
        "For each item in `args/items`, call `classify` using `args/rubric`. "
        "Keep the labels in item order, then count every allowed label exactly "
        "and return the labels and counts."
    )
    child_body = (
        "Read `args/state` and answer this question: " + first["instruction"].strip() +
        "\nUse the choices and descriptions in `args/rubric`. Return one allowed value."
    )
    codebase = {"classify": {"description": "Classify one item with the stated rubric.",
                             "args": {"state": "Text", "rubric": "Text"},
                             "returns": result_type, "instructions": child_body}}
    root = {"$lambda": {"type": "Lambda<{ items: Text[], rubric: Text, label_names: Text[] }, Report>",
                        "types": types, "instructions": body, "codebase": codebase}}
    states = [_state_text(task["state"]) for task in tasks]
    if len(set(states)) != len(states):
        raise ValueError("map report requires distinct states")
    gold_by_state = dict(zip(states, (_typed_value(task) for task in tasks)))
    values = [gold_by_state[state] for state in states]
    counts = {label: sum(value == (label == "true" if first["kind"] == "boolean" else label)
                         for value in values) for label in labels}
    code = ("({ labels: locals.labels, counts: Object.fromEntries("
            "args.label_names.map(k => [k, locals.labels.filter(x => "
            "String(x) === k).length])) })")
    plans = {
        body: Plan("calls", steps=[
            ("call", {"function": "classify", "to": "let/labels", "over": "args/items",
                      "inputs": {"rubric": "args/rubric"}}),
            ("glue", code, "return", "Report")], note="Classified and counted the items."),
        "classify": Plan("leaf", gold=lambda args: gold_by_state[args["state"]],
                         note="Applied the question to the item."),
    }
    return Program("external_map_report", root,
                   {"items": states, "rubric": _rubric(first), "label_names": labels},
                   {"labels": values, "counts": counts}, plans)


def case_report_program(tasks):
    """Several typed questions about the same source state become subroutine calls."""
    if len(tasks) < 2 or len({_state_text(task["state"]) for task in tasks}) != 1:
        raise ValueError("case report needs several questions about one state")
    root_types = {}
    fields = []
    codebase = {}
    plans = {}
    args = {"state": _state_text(tasks[0]["state"])}
    expected = {}
    steps = []
    for index, task in enumerate(tasks):
        raw_name = str(task.get("source_meta", {}).get("question_key") or task["id"].rsplit("/", 1)[-1])
        field = re.sub(r"\W", "_", raw_name)
        if not field or field[0].isdigit() or field in fields:
            field = f"question_{index}"
        fields.append(field)
        result_type, aliases = _type_parts(task)
        if aliases:
            alias = f"Label{index}"
            root_types[alias] = aliases["Label"]
            result_type = alias
        child = f"decide_{field}"
        child_body = ("Read `args/state` and answer: " + task["instruction"].strip() +
                      "\nUse `args/rubric` for the allowed answers.")
        codebase[child] = {"description": f"Decide {field} from the state.",
                           "args": {"state": "Text", "rubric": "Text"},
                           "returns": result_type, "instructions": child_body}
        value = _typed_value(task)
        plans[child] = Plan("leaf", gold=lambda child_args, v=value: v,
                            note=f"Decided {field}.")
        args[f"rubric_{field}"] = _rubric(task)
        expected[field] = value
        steps.append(("call", {"function": child, "to": f"return/{field}",
                               "inputs": {"state": "args/state", "rubric": f"args/rubric_{field}"}}))
    root_types["Report"] = "{ " + ", ".join(
        f"{field}: {codebase['decide_' + field]['returns']}" for field in fields) + " }"
    input_type = "{ state: Text, " + ", ".join(f"rubric_{field}: Text" for field in fields) + " }"
    body = ("For this state, answer each question through its named decision function: " +
            "; ".join(f"{field} = decide_{field}(state, rubric_{field})" for field in fields) +
            ". Return a record with every answer.")
    plans[body] = Plan("calls", steps=steps, note="Completed the typed decision report.")
    root = {"$lambda": {"type": f"Lambda<{input_type}, Report>", "types": root_types,
                        "instructions": body, "codebase": codebase}}
    return Program("external_case_report", root, args, expected, plans)


def read_tasks(paths):
    tasks_by_id = {}
    hashes = {}
    group_splits = {}
    for path in paths:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for line_number, raw in enumerate(stream, 1):
                digest.update(raw)
                if not raw.strip():
                    continue
                task = json.loads(raw)
                for key in ("id", "source", "group_id", "split", "state", "instruction", "kind", "labels"):
                    if key not in task:
                        raise ValueError(f"{path}:{line_number}: missing {key}")
                _labels(task)
                prior = tasks_by_id.get(task["id"])
                if prior is not None:
                    prior_jev, new_jev = prior.get("jev") or {}, task.get("jev") or {}
                    retried = (prior_jev.get("status") == "error"
                               and prior_jev.get("input_hash")
                               and prior_jev.get("input_hash") == new_jev.get("input_hash"))
                    if not retried:
                        raise ValueError(f"duplicate task id: {task['id']}")
                group = (task["source"], str(task["group_id"]))
                prior_split = group_splits.setdefault(group, task["split"])
                if prior_split != task["split"]:
                    raise ValueError(f"source group crosses splits: {group!r}")
                tasks_by_id[task["id"]] = task
        hashes[str(path)] = digest.hexdigest()
    return list(tasks_by_id.values()), hashes


def _group_key(task):
    return (task["split"], task["source"], task["kind"], task["instruction"],
            tuple(task["labels"]), json.dumps(task.get("criteria") or {}, sort_keys=True))


def select_tasks(tasks, allow_unknown_license, splits):
    selected, skipped = [], Counter()
    for task in tasks:
        if task["split"] not in splits:
            skipped["split"] += 1
            continue
        if "gold" not in task:
            skipped["unlabeled"] += 1
            continue
        if not allow_unknown_license and str(task.get("license", "unknown")).lower() not in APPROVED_LICENSES:
            skipped["license"] += 1
            continue
        _gold(task)
        selected.append(task)
    return selected, skipped


def build_task_groups(tasks, families, batch_size):
    if "leaf" in families:
        for task in tasks:
            yield "leaf:" + task["id"], [task]
    if "map_report" in families:
        groups = defaultdict(list)
        for task in tasks:
            groups[_group_key(task)].append(task)
        for grouped in groups.values():
            for start in range(0, len(grouped), batch_size):
                batch = grouped[start:start + batch_size]
                if len(batch) < 2:
                    continue
                states = set()
                unique = []
                for task in batch:
                    state = _state_text(task["state"])
                    if state not in states:
                        states.add(state)
                        unique.append(task)
                if len(unique) >= 2:
                    yield "map:" + ":".join(t["id"] for t in unique), unique
    if "case_report" in families:
        cases = defaultdict(list)
        for task in tasks:
            parent = task["id"].rsplit("/", 1)[0]
            cases[(task["split"], task["source"], parent, _state_text(task["state"]))].append(task)
        for case_tasks in cases.values():
            if len(case_tasks) >= 2:
                yield "case:" + case_tasks[0]["id"].rsplit("/", 1)[0], case_tasks


def build_programs(tasks, families, batch_size):
    constructors = {"leaf": leaf_program, "map": map_report_program,
                    "case": case_report_program}
    for program_id, source_tasks in build_task_groups(tasks, families, batch_size):
        yield program_id, source_tasks, constructors[program_id.split(":", 1)[0]](
            source_tasks[0] if program_id.startswith("leaf:") else source_tasks)


def generate(tasks, output, families=("leaf", "map_report"), batch_size=4, max_programs=None):
    # Kept for callers of the Python API; the CLI persists IR before calling
    # the shared materializer. Both paths use the same semantic lowering.
    from program_ir import decision_record, lower
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    with output.open("w") as stream:
        for number, (program_id, source_tasks) in enumerate(build_task_groups(tasks, families, batch_size)):
            if max_programs is not None and number >= max_programs:
                break
            kind = {"leaf": "decision_leaf", "map": "decision_map_report",
                    "case": "decision_case_report"}[program_id.split(":", 1)[0]]
            record = decision_record(kind, program_id, source_tasks)
            program = lower(record)
            samples, _ = run_program(program)
            metadata = {"program_id": program_id, "family": program.family,
                        "source": source_tasks[0]["source"], "split": source_tasks[0]["split"],
                        "source_ids": [task["id"] for task in source_tasks],
                        "source_groups": sorted({str(task["group_id"]) for task in source_tasks}),
                        "source_revisions": sorted({str(task.get("source_revision", "unknown")) for task in source_tasks}),
                        "license": source_tasks[0].get("license", "unknown"),
                        "gold_sources": sorted({str(task.get("gold_source", "unknown")) for task in source_tasks})}
            for turn, sample in enumerate(samples):
                sample = {**sample, "tools": _strip_private(sample["tools"])}
                stream.write(json.dumps({"id": f"external-{number}-{turn}", **metadata, **sample},
                                        ensure_ascii=False, default=str) + "\n")
                counts[sample["skill"]] += 1
            counts["programs"] += 1
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tasks", type=Path, nargs="+", help="normalized task JSONL files, direct pairs or Jev-labeled")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--families", nargs="+", choices=["leaf", "map_report", "case_report"],
                        default=["leaf", "map_report", "case_report"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-programs", type=int)
    parser.add_argument("--splits", nargs="+", default=["train"],
                        help="source splits to include; defaults to train only")
    parser.add_argument("--allow-unknown-license", action="store_true",
                        help="local research only; provenance retains the source license")
    args = parser.parse_args()
    if args.batch_size < 2 or args.max_programs is not None and args.max_programs < 1:
        parser.error("batch-size must be >=2 and max-programs must be positive")
    import subprocess
    ir_path = args.out.with_suffix(".ir.jsonl")
    command = [sys.executable, str(Path(__file__).with_name("build_program_ir.py")),
               "decisions", *(str(path) for path in args.tasks), "--out", str(ir_path),
               "--families", *args.families, "--batch-size", str(args.batch_size),
               "--splits", *args.splits]
    if args.allow_unknown_license:
        command.append("--allow-unknown-license")
    if args.max_programs is not None:
        command.extend(["--max-programs", str(args.max_programs)])
    subprocess.run(command, check=True)
    subprocess.run([sys.executable, str(Path(__file__).with_name("materialize_ir.py")),
                    str(ir_path), str(args.out)], check=True)


if __name__ == "__main__":
    main()
