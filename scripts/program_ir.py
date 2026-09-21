"""Versioned, template-independent training program records.

Adapters produce these records. Only this module lowers them to the current
natlang harness; traces and model-specific SFT text are disposable builds.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from natlang.gen.programs import Plan, Program
from generate_external import case_report_program, leaf_program, map_report_program

VERSION = "natlang.program/1"
KINDS = {"decision_leaf", "decision_map_report", "decision_case_report", "state_sequence",
         "numeric_program", "scene_program", "lambda_source", "lambda_graph", "lambda_scenario",
         "proposal_review"}


def canonical(record):
    return json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def digest(record):
    return hashlib.sha256(canonical(record).encode()).hexdigest()


def decision_record(kind, program_id, tasks):
    if kind not in {"decision_leaf", "decision_map_report", "decision_case_report"} or not tasks:
        raise ValueError("invalid decision program")
    source, split = tasks[0]["source"], tasks[0]["split"]
    if any(t["source"] != source or t["split"] != split for t in tasks):
        raise ValueError("decision program mixes sources or splits")
    semantic_keys = ("id", "source", "group_id", "split", "state", "instruction",
                     "kind", "labels", "criteria", "gold")
    decisions = []
    for task in tasks:
        decision = {key: task[key] for key in semantic_keys if key in task}
        question_key = task.get("source_meta", {}).get("question_key")
        if question_key is not None:
            decision["source_meta"] = {"question_key": question_key}
        decisions.append(decision)
    return validate({"version": VERSION, "id": program_id, "kind": kind,
                     "source": source, "split": split,
                     "source_ids": [t["id"] for t in tasks],
                     "source_groups": sorted({str(t["group_id"]) for t in tasks}),
                     "source_revisions": sorted({str(t.get("source_revision", "unknown")) for t in tasks}),
                     "license": tasks[0].get("license", "unknown"),
                     "gold_sources": sorted({str(t.get("gold_source", "unknown")) for t in tasks}),
                     "semantics": {"decisions": decisions}})


def validate(record):
    if record.get("version") != VERSION or record.get("kind") not in KINDS:
        raise ValueError("unsupported program IR version or kind")
    for key in ("id", "source", "split", "source_ids", "source_groups", "license", "semantics"):
        if key not in record:
            raise ValueError(f"missing IR field: {key}")
    if not isinstance(record["semantics"], dict) or not record["id"]:
        raise ValueError("invalid IR semantics or id")
    if record["kind"] == "proposal_review":
        sem = record["semantics"]
        if not isinstance(sem.get("base_program_id"), str) or not isinstance(sem.get("turn_index"), int):
            raise ValueError("review needs a linked program and turn")
        if not isinstance(sem.get("proposal"), list) or sem.get("verdict") not in {
            "approve", "withdraw", "error", "blocker"}:
            raise ValueError("review needs a proposal and verdict")
        if not isinstance(sem.get("reason"), str) or not isinstance(sem.get("lesson_ids"), list):
            raise ValueError("review needs rationale and lessons")
    elif record["kind"] == "lambda_scenario":
        sem = record["semantics"]
        if not isinstance(sem.get("root"), dict) or not isinstance(sem.get("operations"), list):
            raise ValueError("lambda scenario needs source and operations")
        contract = sem.get("contract")
        if not isinstance(contract, dict) or contract.get("kind") not in {"done", "blocked", "error"}:
            raise ValueError("unsupported scenario outcome")
        if not isinstance(contract.get("effects"), list) or not isinstance(contract.get("required_actions"), list):
            raise ValueError("scenario needs expected effects")
        if not isinstance(contract.get("constrained_calls"), list):
            raise ValueError("scenario needs call constraints")
        if any(op.get("op") not in {"invoke", "assign", "fail", "block"} for op in sem["operations"]):
            raise ValueError("unsupported scenario operation")
    elif record["kind"] == "lambda_graph":
        sem = record["semantics"]
        if not isinstance(sem.get("root"), dict) or not isinstance(sem.get("inputs"), dict):
            raise ValueError("lambda graph needs source and inputs")
        if "expected" not in sem or not isinstance(sem.get("operations"), list) or not sem["operations"]:
            raise ValueError("lambda graph needs expected and operations")
        if not isinstance(sem.get("source_lines"), list) or not isinstance(sem.get("leaf_oracles"), dict):
            raise ValueError("lambda graph needs source line mapping and leaf oracles")
        def check_ops(ops):
            if not isinstance(ops, list):
                raise ValueError("lambda graph operations must be a list")
            for op in ops:
                if op.get("op") not in {"invoke", "compute", "assign", "branch", "clone_function", "edit_function"}:
                    raise ValueError("unsupported lambda graph operation")
                if op["op"] == "branch":
                    if op.get("test") not in {"truthy", "empty", "zero", "equals"} or not (
                        op.get("test_path") or op.get("test_variable")):
                        raise ValueError("invalid lambda graph branch")
                    check_ops(op.get("then"))
                    check_ops(op.get("else"))
        check_ops(sem["operations"])
        for ops in sem.get("functions", {}).values():
            check_ops(ops)
        nested = sem.get("nested")
        if nested is not None and nested.get("kind") not in {"assessment", "conditional_flag", "conditional_increment"}:
            raise ValueError("unsupported nested lambda graph operation")
    elif record["kind"] == "lambda_source":
        sem = record["semantics"]
        if sem.get("operation") not in {"leaf", "blocked", "exact", "algorithm", "map", "map_count"}:
            raise ValueError("unsupported lambda source operation")
        if not isinstance(sem.get("root"), dict) or not isinstance(sem.get("inputs"), dict):
            raise ValueError("lambda source needs root and inputs")
        if "expected" not in sem:
            raise ValueError("lambda source needs expected")
        if sem["operation"] in {"map", "map_count"} and not sem.get("leaf_oracles"):
            raise ValueError("map source needs leaf oracles")
    elif record["kind"] == "scene_program":
        sem = record["semantics"]
        if not isinstance(sem.get("nodes"), list) or not sem["nodes"] or not isinstance(sem.get("scene"), dict):
            raise ValueError("scene program needs nodes and scene")
        if sem.get("answer_type") not in {"Num", "Bool", "Text"}:
            raise ValueError("scene program needs typed answer")
    elif record["kind"] == "numeric_program":
        sem = record["semantics"]
        if not isinstance(sem.get("steps"), list) or not sem["steps"]:
            raise ValueError("numeric program needs steps")
        if sem.get("answer_type") not in {"Num", "Text"} or "answer" not in sem:
            raise ValueError("numeric program needs typed answer")
    elif record["kind"] == "state_sequence":
        sem = record["semantics"]
        steps = sem.get("steps")
        if not isinstance(sem.get("initial_state"), (str, dict)) or not isinstance(steps, list) or not steps:
            raise ValueError("state sequence requires initial state and steps")
        state = sem["initial_state"]
        for step in steps:
            if not isinstance(step.get("utterance"), str) or not isinstance(step.get("after"), (str, dict)):
                raise ValueError("invalid state transition")
            if step.get("before") != state or step["after"] == "?":
                raise ValueError("broken or missing intermediate state")
            state = step["after"]
    else:
        tasks = record["semantics"].get("decisions")
        if not isinstance(tasks, list) or not tasks:
            raise ValueError("decision program has no decisions")
        if record["kind"] == "decision_leaf" and len(tasks) != 1:
            raise ValueError("leaf must have one decision")
        if record["kind"] != "decision_leaf" and len(tasks) < 2:
            raise ValueError("composite requires multiple decisions")
        if any("gold" not in task for task in tasks):
            raise ValueError("decision without gold")
    return record


def lower(record):
    """Compile semantic IR to today's harness. No model template enters here."""
    validate(record)
    kind, sem = record["kind"], record["semantics"]
    if kind == "proposal_review":
        raise ValueError("proposal reviews materialize through materialize_review_ir.py")
    if kind == "decision_leaf":
        return leaf_program(sem["decisions"][0])
    if kind == "decision_map_report":
        return map_report_program(sem["decisions"])
    if kind == "decision_case_report":
        return case_report_program(sem["decisions"])
    if kind == "numeric_program":
        return numeric_program(record)
    if kind == "scene_program":
        return scene_program(record)
    if kind == "lambda_source":
        return lambda_source_program(record)
    if kind == "lambda_graph":
        return lambda_graph_program(record)
    if kind == "lambda_scenario":
        return lambda_scenario_program(record)
    return state_sequence_program(record)


def lambda_scenario_program(record):
    sem = record["semantics"]
    root = sem["root"]
    body = root["$lambda"]["instructions"]
    steps = []
    for op in sem["operations"]:
        if op["op"] == "invoke":
            args = {"function": op["function"], "to": op["target"]}
            for source, dest in (("foreach", "over"), ("arguments", "inputs"),
                                 ("initial", "init"), ("until", "until"),
                                 ("max_steps", "max")):
                if source in op:
                    args[dest] = op[source]
            if "completion" in op:
                args["done"] = op["completion"]
            steps.append(("call", args))
        elif op["op"] == "assign":
            args = {"path": op["target"], "type": op["value_type"]}
            args["source" if "from" in op else "value"] = op.get("from", op.get("value"))
            if "completion" in op:
                args["done"] = op["completion"]
            steps.append(("write", args))
        elif op["op"] == "fail":
            steps.append(("report_error", {"message": op["message"]}))
        else:
            steps.append(("report_blocker", {"missing": op["missing"]}))
    fault = sem.get("injected_fault")
    injected = ((fault["action"], fault["arguments"], fault["expected_result"])
                if fault else None)
    contract = sem["contract"]
    return Program("lambda_scenario", root, {}, contract.get("value"),
                   {body: Plan("calls", steps=steps, note="Completed the source task.")},
                   outcome=contract["kind"], expected_effects=contract["effects"],
                   injected_fault=injected)


def lambda_graph_program(record):
    """Choose today's tool policy for source-level invoke, compute, assign operations."""
    from natlang.gen.synth import build_script
    from natlang.types import format_type

    sem = record["semantics"]
    root = sem["root"]
    fold = "$fold" in root
    name = (root["$fold"]["step"]["$lambda"]["function"] if fold else
            root["$lambda"]["function"])
    steps = []
    for op in sem["operations"]:
        if op["op"] == "invoke":
            args = {"function": op["function"], "to": op["target"]}
            fields = (("foreach", "over"), ("initial", "init"), ("until", "until"),
                      ("max_steps", "max"), ("arguments", "inputs")) if "initial" in op else (
                      ("foreach", "over"), ("arguments", "inputs"), ("until", "until"),
                      ("max_steps", "max"))
            for source, dest in fields:
                if source in op:
                    args[dest] = op[source]
            steps.append(("call", args))
        elif op["op"] == "compute":
            steps.append(("glue", op["expression"], op["target"], op["value_type"]))
        elif op["op"] == "assign":
            args = {"path": op["target"], "type": op["value_type"]}
            args["source" if "from" in op else "value"] = op.get("from", op.get("value"))
            steps.append(("write", args))
    if "functions" in sem or any(op["op"] == "branch" for op in sem["operations"]):
        root_script = graph_script(sem["operations"])
    else:
        root_script = build_script(sem["source_lines"], steps, "grouped")
    plans = {name: Plan("script", script=root_script,
                        note="Carried out the program.")}
    for fn, ops in sem.get("functions", {}).items():
        plans[fn] = Plan("script", script=graph_script(ops), note="Carried out the source function.")
    for alias, oracle in sem["leaf_oracles"].items():
        cases = {canonical(row["input"]): row for row in oracle["cases"]}
        param = oracle.get("parameter")
        all_args = oracle.get("arg_mode") == "all"
        lookup_path = oracle.get("lookup_path", [])
        variant_marker = oracle.get("variant_marker")
        def leaf_script(lam, cases=cases, param=param, all_args=all_args, lookup_path=lookup_path,
                        variant_marker=variant_marker, alias=alias):
            key = lam.in_ if all_args else lam.in_[param]
            for part in lookup_path:
                key = key[part]
            row = cases[canonical(key)]
            plans[alias].template = bool(row.get("template"))
            if "blocked_reason" in row:
                yield [("report_blocker", {"missing": row["blocked_reason"]})]
            else:
                value = row["variant_output"] if variant_marker and variant_marker in lam.body else row["output"]
                yield [("write", {"path": "return", "type": format_type(lam.type.returns),
                                  "value": value})]
        plans[alias] = Plan("script", script=leaf_script, note="Answered the source function.")
    nested = sem.get("nested")
    if nested:
        plans[nested["function"]] = Plan("script", script=nested_graph_script(nested),
                                          note="Carried out the nested function.")
    for alias, rule in sem.get("leaf_rules", {}).items():
        if rule["kind"] == "min_priority_then_id":
            priorities = rule["priorities"]
            plans[alias] = Plan("leaf", gold=lambda args, priorities=priorities:
                                min(args["ready"], key=lambda task:
                                    (priorities[task["id"]], task["id"]))["id"])
        elif rule["kind"] == "equal_fields":
            plans[alias] = Plan("leaf", gold=lambda args, left=rule["left"], right=rule["right"]:
                                args[left] == args[right])
        elif rule["kind"] == "forward_conclusions":
            rules, people = rule["rules"], rule["people"]
            def conclusions(args, rules=rules, people=people):
                conclusion, conditions = rules[args["rule"]]
                known = set(args["known"])
                return [f"{person} is {conclusion}." for person in people
                        if all(f"{person} is {condition}." in known for condition in conditions)
                        and f"{person} is {conclusion}." not in known]
            plans[alias] = Plan("leaf", gold=conclusions)
        else:
            raise ValueError("unsupported source leaf rule")
    effects = sem.get("effects", {})
    observed = {name: [] for name in effects}
    capabilities = {}
    for effect_name, spec in effects.items():
        if isinstance(spec, list):
            capabilities[effect_name] = lambda args, bucket=observed[effect_name]: bucket.append(args[0])
        elif spec.get("kind") == "record_args":
            capabilities[effect_name] = lambda args, bucket=observed[effect_name]: bucket.append(args)
        elif spec.get("kind") == "deliver_once_ack_loss":
            delivered_keys, failed = set(), set()
            def send(args, bucket=observed[effect_name], delivered_keys=delivered_keys,
                     failed=failed, fail_key=spec["fail_key"]):
                command = args[0]
                if command["key"] not in delivered_keys:
                    delivered_keys.add(command["key"])
                    bucket.append(command)
                if command["key"] == fail_key and fail_key not in failed:
                    failed.add(fail_key)
                    raise RuntimeError("delivery succeeded but its acknowledgement was lost")
            capabilities[effect_name] = send
        else:
            raise ValueError("unsupported effect contract")
    expected = sem["expected"]
    if effects:
        expected_effects = {name: (spec if isinstance(spec, list) else
                                   spec["expected"] if spec["kind"] == "record_args" else
                                   spec["expected_delivered"])
                            for name, spec in effects.items()}
        expected = lambda value, expected=expected, effects=expected_effects, observed=observed: (
            value == expected and observed == effects)
    if fold:
        from natlang.runtime import OpenList
        from natlang.values import load_program
        def loader(root=root, events=sem["events"]):
            node = load_program(root)
            node.over = OpenList(iter(events))
            return node
        return Program(record.get("family", "lambda_graph"), {}, {}, expected,
                       plans, loader=loader, capabilities=capabilities)
    return Program(record.get("family", "lambda_graph"), root, sem["inputs"], expected,
                   plans, capabilities=capabilities)


def graph_script(operations):
    """Interpret a source operation graph using the currently available tools."""
    def run(ops, bindings):
        for op in ops:
            if op["op"] == "invoke":
                args = {"function": op["function"], "to": op["target"]}
                fields = (("foreach", "over"), ("initial", "init"), ("until", "until"),
                          ("max_steps", "max"), ("arguments", "inputs")) if "initial" in op else (
                          ("foreach", "over"), ("arguments", "inputs"), ("until", "until"),
                          ("max_steps", "max"))
                for source, dest in fields:
                    if source in op:
                        args[dest] = op[source]
                result = yield [("call", args)]
                for _ in range(op.get("retry_on_quiesced", 0)):
                    if result.kind != "quiesced":
                        break
                    result = yield [("call", {"function": op["function"], "to": op["target"]})]
                if result.kind == "quiesced" and "on_quiesced" in op:
                    yield [("report_blocker", {"missing": op["on_quiesced"]})]
                    return
            elif op["op"] == "compute":
                result = yield [("run_code", {"code": op["expression"]})]
                yield [("write", {"path": op["target"], "type": op["value_type"],
                                  "value": result.value})]
            elif op["op"] == "assign":
                args = {"path": op["target"], "type": op["value_type"]}
                args["source" if "from" in op else "value"] = op.get("from", op.get("value"))
                yield [("write", args)]
            elif op["op"] == "clone_function":
                yield [("write", {"path": op["target"], "type": f"Function<{op['function']}>"})]
            elif op["op"] == "edit_function":
                yield [("edit", {"path": op["target"], "old": op["old"], "new": op["new"]})]
            else:
                if "test_variable" in op:
                    value = bindings[op["test_variable"]]
                else:
                    result = yield [("read", {"path": op["test_path"]})]
                    value = result.value
                    if "bind" in op:
                        bindings[op["bind"]] = value
                take_then = (bool(value) if op["test"] == "truthy" else
                             value == 0 if op["test"] == "zero" else
                             value == op["value"] if op["test"] == "equals" else
                             not bool(value))
                yield from run(op["then"] if take_then else op["else"], bindings)
    def script(_lam):
        yield from run(operations, {})
    return script


def nested_graph_script(spec):
    """Compile a nested source function, branching on the value it just computed."""
    from natlang.types import format_type

    def script(lam):
        kind = spec["kind"]
        if kind == "assessment":
            label = ("call", {"function": spec["label_function"], "to": "return/label",
                              "inputs": {"ticket": "args/ticket", "rubric": "args/rubric"}})
            flag = ("call", {"function": spec["flag_function"], "to": "return/urgent",
                             "inputs": {"ticket": "args/ticket"}})
            yield [label]
            yield [flag]
        elif kind == "conditional_flag":
            args = {spec["label_parameter"]: "args/item"}
            if spec["rubric"]:
                args["rubric"] = "args/rubric"
            result = yield [("call", {"function": spec["label_function"],
                                      "to": "return/label", "inputs": args})]
            if result.kind == "quiesced":
                yield [("report_blocker", {"missing": "The label function did not finish."})]
                return
            if result.value == spec["target_label"]:
                yield [("call", {"function": spec["flag_function"], "to": "return/flag",
                                 "inputs": {spec["flag_parameter"]: "args/item"}})]
            else:
                yield [("write", {"path": "return/flag", "type": "Bool", "value": False})]
        else:
            result = yield [("call", {"function": spec["flag_function"], "to": "let/hit",
                                      "inputs": {spec["flag_parameter"]: "args/item"}})]
            if result.kind == "quiesced":
                yield [("report_blocker", {"missing": "The flag function did not finish."})]
                return
            if result.value:
                value = yield [("run_code", {"code": "args.acc + 1"})]
                yield [("write", {"path": "return", "type": format_type(lam.type.returns),
                                  "value": value.value})]
            else:
                yield [("write", {"path": "return", "type": format_type(lam.type.returns),
                                  "source": "args/acc"})]
    return script


def lambda_source_program(record):
    sem = record["semantics"]
    root = sem["root"]
    body = root["$lambda"]["instructions"]
    operation = sem["operation"]
    plans = {}
    if operation == "leaf":
        value = sem["expected"]
        plans[body] = Plan("leaf", gold=lambda _args: value, note="Answered the task.")
    elif operation == "blocked":
        plans[body] = Plan("blocked", note=sem["blocked_reason"])
    elif operation == "exact":
        formula = sem["formula"]
        expressions = {"sum": "sum(args.numbers)",
                       "count_over_100": "count(args.numbers, x => x > 100)",
                       "max": "max(args.numbers)",
                       "mean_round_2": "Math.round(mean(args.numbers) * 100) / 100"}
        plans[body] = Plan("crisp", code=expressions[formula], note="Computed the result.")
    elif operation == "algorithm":
        plans[body] = Plan("crisp", code=sem["expression"], note="Executed the exact algorithm.")
    else:
        oracle = {canonical(row["input"]): row["output"] for row in sem["leaf_oracles"]}
        def answer(args):
            return oracle[canonical(args)]
        if operation == "map":
            plans[body] = Plan("calls", steps=[("call", {"function": "classify", "to": "return",
                                                       "over": "args/tickets",
                                                       "inputs": {"rubric": "args/rubric"}})],
                               note="Classified the tickets.")
            plans["classify"] = Plan("leaf", gold=answer, note="Classified one ticket.")
        else:
            plans[body] = Plan("calls", steps=[
                ("call", {"function": "is_urgent", "to": "let/flags", "over": "args/tickets"}),
                ("call", {"function": "count_true", "to": "return",
                          "inputs": {"flags": "let/flags"}})],
                note="Judged and counted the urgent tickets.")
            plans["is_urgent"] = Plan("leaf", gold=answer, note="Judged one ticket.")
    return Program(record.get("family", "lambda_source"), root, sem["inputs"],
                   sem["expected"], plans)


def scene_node_type(op):
    if op in {"scene", "relate", "union", "intersect"} or op.startswith(("filter_", "same_")):
        return "Num[]"
    if op in {"unique", "count"}:
        return "Num"
    if op == "exist" or op.startswith("equal_") or op in {"greater_than", "less_than"}:
        return "Bool"
    if op.startswith("query_"):
        return "Text"
    raise ValueError(f"unsupported scene operation: {op}")


def scene_program(record):
    sem = record["semantics"]
    body = ("Answer `args/question` from the structured objects and spatial "
            "relationships in `args/scene`. Use exact code for each selection, "
            "relation, count, and comparison; keep intermediate results in locals.")
    root = {"$lambda": {"type": f"Lambda<{{ question: Text, scene: Text }}, {sem['answer_type']}>",
                        "instructions": body}}
    steps = []
    long_program = len(sem["nodes"]) > 16
    scene = "S" if long_program else "JSON.parse(args.scene)"
    statements = []
    for index, node in enumerate(sem["nodes"]):
        op = node["function"]
        refs = [f"n_{j}" if long_program else f"locals.node_{j}" for j in node["inputs"]]
        values = node.get("value_inputs", [])
        if op == "scene":
            code = f"Array.from({{length: {scene}.objects.length}}, (_, i) => i)"
        elif op.startswith("filter_"):
            attr = op.removeprefix("filter_")
            code = f"{refs[0]}.filter(i => {scene}.objects[i][{json.dumps(attr)}] === {json.dumps(values[0])})"
        elif op == "unique":
            code = f"{refs[0]}[0]"
        elif op == "relate":
            code = f"{scene}.relationships[{json.dumps(values[0])}][{refs[0]}]"
        elif op.startswith("same_"):
            attr = op.removeprefix("same_")
            code = (f"{scene}.objects.map((o,i) => i).filter(i => i !== {refs[0]} && "
                    f"{scene}.objects[i][{json.dumps(attr)}] === "
                    f"{scene}.objects[{refs[0]}][{json.dumps(attr)}])")
        elif op == "union":
            code = f"[...new Set([...{refs[0]}, ...{refs[1]}])].sort((a,b) => a-b)"
        elif op == "intersect":
            code = f"{refs[0]}.filter(i => {refs[1]}.includes(i))"
        elif op == "count":
            code = f"{refs[0]}.length"
        elif op == "exist":
            code = f"({refs[0]}.length > 0)"
        elif op.startswith("query_"):
            attr = op.removeprefix("query_")
            code = f"{scene}.objects[{refs[0]}][{json.dumps(attr)}]"
        elif op.startswith("equal_"):
            code = f"({refs[0]} === {refs[1]})"
        elif op == "greater_than":
            code = f"({refs[0]} > {refs[1]})"
        elif op == "less_than":
            code = f"({refs[0]} < {refs[1]})"
        else:
            raise ValueError(op)
        final = index == len(sem["nodes"]) - 1
        if long_program:
            statements.append(f"const n_{index} = {code};")
        else:
            steps.append(("glue", code, "return" if final else f"let/node_{index}", scene_node_type(op)))
    if long_program:
        code = "(() => { const S = JSON.parse(args.scene); " + " ".join(statements) + (
            f" return n_{len(sem['nodes']) - 1}; }})()")
        plan = Plan("crisp", code=code, note="Executed the scene program.")
    else:
        plan = Plan("calls", steps=steps, note="Executed the scene program.")
    return Program("scene_program", root,
                   {"question": sem["question"], "scene": canonical(sem["scene"])},
                   sem["answer"], {body: plan})


def numeric_program(record):
    sem = record["semantics"]
    answer_type = sem["answer_type"]
    body = ("Answer `args/question` from `args/evidence`. Use exact arithmetic with "
            "`run_code` for every calculation; preserve intermediate results in locals. "
            "Return the computed answer.")
    root = {"$lambda": {"type": f"Lambda<{{ question: Text, evidence: Text }}, {answer_type}>",
                        "instructions": body}}

    def operand(value):
        if isinstance(value, int):
            return f"locals.step_{value}"
        return repr(float(value))

    steps = []
    for index, step in enumerate(sem["steps"]):
        a, b = (operand(value) for value in step["args"])
        op = step["op"]
        expr = {"add": f"({a} + {b})", "subtract": f"({a} - {b})",
                "multiply": f"({a} * {b})", "divide": f"({a} / {b})",
                "exp": f"Math.pow({a}, {b})",
                "greater": f"(({a} > {b}) ? 'yes' : 'no')"}[op]
        final = index == len(sem["steps"]) - 1
        steps.append(("glue", expr, "return" if final else f"let/step_{index}",
                      answer_type if final else "Num"))
    expected = sem["answer"]
    if answer_type == "Num":
        check = lambda value: isinstance(value, (int, float)) and abs(value - expected) <= max(0.0001, abs(expected) * 1e-7)
    else:
        check = expected
    return Program("numeric_program", root,
                   {"question": sem["question"], "evidence": sem["evidence"]},
                   check, {body: Plan("calls", steps=steps, note="Computed the answer.")})


def state_sequence_program(record):
    sem = record["semantics"]
    steps = sem["steps"]
    state_type = sem.get("state_type", "Text")
    input_types = [f"state: {state_type}"] + [f"instruction_{i}: Text, history_{i}: Text"
                                         for i in range(1, len(steps) + 1)]
    body = ("Starting at `args/state`, apply the numbered instructions in order. "
            "Call `transition` once for each instruction, passing the state from the previous call. "
            "Return the final state.")
    child_body = ("Apply `args/instruction` to `args/state`, using `args/history` "
                  "to resolve references to earlier actions. " +
                  sem.get("task_context", "Domain: " + sem["domain"] + ".") +
                  " Return the complete state after this one instruction.")
    root = {"$lambda": {"type": "Lambda<{ " + ", ".join(input_types) + " }, " + state_type + ">",
                        "instructions": body,
                        "codebase": {"transition": {
                            "description": "Apply one world instruction to a state.",
                            "args": {"state": state_type, "instruction": "Text", "history": "Text"},
                            "returns": state_type, "instructions": child_body}}}}
    inputs = {"state": sem["initial_state"]}
    gold = {}
    calls = []
    prior = "args/state"
    for i, step in enumerate(steps, 1):
        inputs[f"instruction_{i}"] = step["utterance"]
        history = "\n".join(f"{j}. {previous['utterance']}" for j, previous in enumerate(steps[:i-1], 1))
        inputs[f"history_{i}"] = history
        key = (canonical(step["before"]), step["utterance"], history)
        if key in gold and gold[key] != step["after"]:
            raise ValueError("same world and instruction have conflicting next states")
        gold[key] = step["after"]
        destination = "return" if i == len(steps) else f"let/state_{i}"
        calls.append(("call", {"function": "transition", "to": destination,
                               "inputs": {"state": prior, "instruction": f"args/instruction_{i}",
                                          "history": f"args/history_{i}"}}))
        prior = destination
    return Program("state_sequence", root, inputs, steps[-1]["after"],
                   {body: Plan("calls", steps=calls, note="Applied each instruction in sequence."),
                    "transition": Plan("leaf", gold=lambda a: gold[(canonical(a["state"]),
                                                                      a["instruction"], a["history"])],
                                       note="Updated the world state.")})


def read_jsonl(path):
    with Path(path).open() as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield validate(json.loads(line))
                except (ValueError, KeyError) as exc:
                    raise ValueError(f"{path}:{number}: {exc}") from exc
