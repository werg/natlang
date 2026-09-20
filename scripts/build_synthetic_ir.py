#!/usr/bin/env python3
"""Freeze serializable synthetic source tasks from a pinned generator revision."""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from generate import make_program
from natlang.corpus import digest, file_digest
from natlang.gen import codebases as codebase_references
from natlang.gen.programs import BLOCKED
from program_ir import VERSION, validate

SUPPORTED = {"judge", "classify", "extract", "crisp_scalar", "map_leaf", "map_then_count",
             "tiny", "composed", "ticket_report", "review_digest", "expense_audit", "guarded_call",
             "nested_assessment", "per_item_condition", "fold_with_steps", "cb_reconciliation", "cb_dependency_plan",
             "cb_nlprolog", "cb_legal_move", "cb_moderation", "cb_highlighter", "cb_mail_rules",
             "cb_order_saga", "cb_shopkeeper", "cb_webserver"}
EXACT = {"sum(args.numbers)": "sum",
         "count(args.numbers, x => x > 100)": "count_over_100",
         "max(args.numbers)": "max",
         "Math.round(mean(args.numbers) * 100) / 100": "mean_round_2"}


def generator_fingerprint(repo):
    tracked = sorted(path for folder in ("natlang", "codebases", "examples")
                     for path in (repo / folder).rglob("*")
                     if path.is_file() and path.suffix in (".py", ".js", ".nl", ".ts", ".md"))
    return {"sources": digest([(str(path.relative_to(repo)), file_digest(path)) for path in tracked]),
            "phrase_bank": file_digest(repo / "data/phrases.json"),
            "leaf_references": file_digest(repo / "data/leaf_references.jsonl")}


def freeze(program, program_id, provenance):
    family = program.family
    if family in {"cb_reconciliation", "cb_dependency_plan", "cb_nlprolog", "cb_legal_move", "cb_moderation", "cb_highlighter", "cb_mail_rules", "cb_order_saga", "cb_shopkeeper", "cb_webserver"}:
        from natlang.codebase import load_function
        facts = program.source_semantics
        if facts.get("fold"):
            from natlang.values import dump_state
            root = dump_state(program.loader())
        else:
            fn = load_function(facts["codebase_file"])
            root = {"$lambda": {**fn.to_lambda_doc(), "function": fn.name,
                                "codebase": {name: child.to_inline() for name, child in fn.codebase.items()}}}
        if "capture_leaf_oracles" in facts:
            gold, observations, effects = capture_observations(
                program, facts["capture_leaf_oracles"], capture_effect_args=facts.get("capture_effect_args", False))
            facts = {**facts, "expected": gold, "leaf_oracles": observations, "effects": effects,
                     "contains_templates": any(row.get("template") for oracle in observations.values()
                                               for row in oracle["cases"])}
        expected = facts.get("expected")
        if expected is None:
            expected = capture_gold(program) if callable(program.expected) else program.expected
        sem = {"root": root, "inputs": facts["inputs"], "expected": expected,
               "operations": facts["operations"], "source_lines": [],
               "leaf_oracles": facts.get("leaf_oracles", {})}
        for key in ("functions", "leaf_rules", "effects", "events", "contains_templates"):
            if key in facts:
                sem[key] = facts[key]
        return validate({"version": VERSION, "id": program_id, "kind": "lambda_graph",
                         "family": family, "source": "natlang-synthetic", "split": "train",
                         "source_ids": [program_id], "source_groups": [program_id],
                         "source_revisions": [provenance["sources"]], "license": "project-generated",
                         "gold_sources": ["synthetic-generator"], "generation": provenance,
                         "semantics": sem})
    if family not in SUPPORTED or program.loader is not None or callable(program.expected):
        raise ValueError("synthetic family needs a different semantic adapter")
    body = program.root["$lambda"]["instructions"]
    plan = program.plans.get(body) or program.plans.get(program.root["$lambda"].get("function"))
    sem = {"root": program.root, "inputs": program.inputs, "expected": program.expected}
    kind = "lambda_source"
    if program.source_semantics:
        from natlang.gen.synth import LEAVES
        facts = program.source_semantics
        names = facts["leaf_names"]
        root_name = program.root["$lambda"].get("function")
        nested = facts.get("nested")
        allowed = {nested["function"]} if nested else set()
        if set(program.plans) - set(names.values()) - {root_name} - allowed:
            raise ValueError(f"{family} contains nested scripts requiring a separate adapter")
        operations = []
        for step in facts["steps"]:
            if step[0] == "call":
                args = step[1]
                op = {"op": "invoke", "function": args["function"], "target": args["to"]}
                for source, dest in (("inputs", "arguments"), ("over", "foreach"),
                                     ("init", "initial"), ("until", "until"), ("max", "max_steps")):
                    if source in args:
                        op[dest] = args[source]
            elif step[0] == "glue":
                _, expression, target, value_type = step
                op = {"op": "compute", "expression": expression, "target": target,
                      "value_type": value_type}
            elif step[0] == "write":
                args = step[1]
                op = {"op": "assign", "target": args["path"], "value_type": args["type"]}
                if "source" in args:
                    op["from"] = args["source"]
                else:
                    op["value"] = args["value"]
            else:
                raise ValueError(f"unsupported generator operation: {step[0]}")
            operations.append(op)
        oracles = {}
        for canon, alias in names.items():
            param = next(iter(LEAVES[canon][1]))
            rows = []
            for item, hidden in facts["hidden"].items():
                row = {"input": item}
                if canon == "classify" and hidden.get("uncovered"):
                    row["blocked_reason"] = "The rubric has no label for this ticket: it is about " + hidden["uncovered"] + "."
                else:
                    row["output"] = LEAVES[canon][4](hidden)
                rows.append(row)
            oracles[alias] = {"parameter": param, "cases": rows}
        sem.update(operations=operations, source_lines=facts["line_meta"], leaf_oracles=oracles)
        if nested:
            sem["nested"] = nested
        kind = "lambda_graph"
    elif plan.kind == "leaf":
        sem["operation"] = "leaf"
    elif plan.kind == "blocked":
        sem.update(operation="blocked", blocked_reason=plan.note)
    elif plan.kind == "crisp":
        sem.update(operation="exact", formula=EXACT[plan.code])
    elif family in {"map_leaf", "map_then_count"}:
        fn = "classify" if family == "map_leaf" else "is_urgent"
        child = program.plans[fn]
        if child.kind != "leaf":
            raise ValueError("map child is not a leaf oracle")
        sem["operation"] = "map" if family == "map_leaf" else "map_count"
        sem["leaf_oracles"] = [{"input": {"ticket": ticket,
                                           **({"rubric": program.inputs["rubric"]} if fn == "classify" else {})},
                                "output": child.gold({"ticket": ticket})}
                               for ticket in program.inputs["tickets"]]
    else:
        raise ValueError(f"unsupported plan: {family}/{plan.kind}")
    record = {"version": VERSION, "id": program_id, "kind": kind,
              "family": family, "source": "natlang-synthetic", "split": "train",
              "source_ids": [program_id], "source_groups": [program_id],
              "source_revisions": [provenance["sources"]], "license": "project-generated",
              "gold_sources": ["synthetic-generator"], "generation": provenance,
              "semantics": sem}
    return validate(record)


def capture_gold(program):
    """Freeze the full verified value where a legacy generator checks only a predicate."""
    from natlang.gen.policy import ReferenceAgent
    from natlang.runtime import Runtime
    from natlang.values import dump

    def agent(lam):
        plan = program.plans.get(lam.fn_name)
        if plan is None:
            raise ValueError(f"no original plan for {lam.fn_name}")
        return ReferenceAgent(plan, [], check_grammar=False)
    outcome, value = Runtime(agent, max_episodes=2000, capabilities=program.capabilities).run_root(program.loader())
    if outcome.kind != "done":
        raise ValueError(f"cannot capture gold: {outcome.kind}: {outcome.detail}")
    gold = dump(value)
    if not program.expected(gold):
        raise ValueError("original program's expected predicate rejected captured gold")
    return gold


def capture_observations(program, leaf_names, *, capture_effect_args=False):
    """Freeze an observed oracle table and effect contract for a generated case."""
    from natlang.gen.policy import ReferenceAgent
    from natlang.runtime import Runtime
    from natlang.values import dump

    rows = {name: {} for name in leaf_names}
    effects = {name: [] for name in program.capabilities}
    wrapped = {}
    for name, original in program.capabilities.items():
        def callback(args, name=name, original=original):
            effects[name].append(dump(args) if capture_effect_args else dump(args[0]))
            return original(args)
        wrapped[name] = callback

    def factory(lam):
        plan = program.plans.get(lam.fn_name)
        if plan is None:
            raise ValueError(f"no original plan for {lam.fn_name}")
        agent = ReferenceAgent(plan, [], check_grammar=False)
        if lam.fn_name not in rows:
            return agent
        def run(session, agent=agent, plan=plan, name=lam.fn_name):
            args = dump(session.lam.in_)
            result = agent.run(session)
            value = dump(session.lam.ret)
            key = json.dumps(args, sort_keys=True, ensure_ascii=False)
            row = {"input": args, "output": value, "template": bool(plan.template)}
            prior = rows[name].setdefault(key, row)
            if prior != row:
                raise ValueError("same source leaf inputs produced different gold")
            return result
        return type("RecordingAgent", (), {"run": staticmethod(run)})()

    outcome, value = Runtime(factory, max_episodes=2000, capabilities=wrapped).run_root(program.loader())
    if outcome.kind != "done":
        raise ValueError(f"cannot capture case: {outcome.kind}: {outcome.detail}")
    gold = dump(value)
    if callable(program.expected) and not program.expected(gold):
        raise ValueError("original program rejected captured gold")
    return gold, {name: {"arg_mode": "all", "cases": list(cases.values())}
                  for name, cases in rows.items()}, (
        {name: {"kind": "record_args", "expected": values} for name, values in effects.items()}
        if capture_effect_args else effects)


def build_one(job):
    seed, index, families, provenance = job
    family, program = make_program(seed, index, families)
    if family not in SUPPORTED:
        return "unadapted:" + family, None, False
    record = freeze(program, f"{seed}:{family}:{index}", {**provenance, "index": index})
    return ("migrated:" + family, json.dumps(record, ensure_ascii=False) + "\n",
            bool(record["semantics"].get("contains_templates")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="manifest from the original synthetic reference run")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-programs", type=int)
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 2)))
    parser.add_argument("--allow-source-drift", action="store_true",
                        help="create new tasks from current code instead of claiming historical replay")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    original = json.loads(args.manifest.read_text())
    if original.get("version") != "generation/2":
        parser.error("requires a generation/2 manifest with seed and source hashes")
    repo = Path(__file__).resolve().parent.parent
    bank_path = repo / "data/leaf_references.jsonl"
    bank_before = file_digest(bank_path)
    codebase_references.load_references()
    if file_digest(bank_path) != bank_before:
        raise RuntimeError("reference bank changed while loading generator references")
    current = generator_fingerprint(repo)
    if current["leaf_references"] != bank_before:
        raise RuntimeError("reference bank changed before generation started")
    changes = [key for key in current if original.get(key) != current[key]]
    if changes and not args.allow_source_drift:
        parser.error("historical generator inputs differ: " + ", ".join(changes) +
                     "; use the matching code revision or --allow-source-drift for a new corpus")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    provisional_programs = 0
    content_hash = hashlib.sha256()
    seed = original["seed"]
    start = original["start_index"]
    n = original["n"] if args.max_programs is None else min(original["n"], args.max_programs)
    provenance = {"seed": seed, "manifest_sha256": file_digest(args.manifest),
                  "historical_match": not changes, **current}
    staging = args.out.with_suffix(args.out.suffix + ".building")
    jobs = ((seed, index, original["families"], provenance)
            for index in range(start, start + n))
    with staging.open("w") as stream:
        if args.workers == 1:
            results = map(build_one, jobs)
            pool = None
        else:
            pool = multiprocessing.Pool(args.workers)
            results = pool.imap(build_one, jobs, chunksize=8)
        try:
            for position, (count_key, line, provisional) in enumerate(results, 1):
                counts[count_key] += 1
                provisional_programs += provisional
                if line is not None:
                    # Field order matters to native writes of typed records.
                    stream.write(line)
                    content_hash.update(line.encode())
                if position % 1000 == 0:
                    print(f"serialized {position}/{n} source tasks", flush=True)
        finally:
            if pool is not None:
                pool.terminate()
                pool.join()
    if generator_fingerprint(repo) != current:
        raise RuntimeError("generator sources or references changed during build; staged output is not published")
    staging.replace(args.out)
    manifest = {"version": VERSION, "historical_match": not changes,
                "generator_changes": changes, "input_manifest": str(args.manifest),
                "programs_considered": n, "counts": dict(counts),
                "provisional_programs": provisional_programs,
                "ir_sha256": content_hash.hexdigest()}
    args.out.with_suffix(args.out.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
