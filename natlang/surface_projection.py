"""Explicit projection of historical engine-less tool calls onto tools-v3."""
from __future__ import annotations

import copy
import json
import re


def _alternative(tools: list | None, tool_name: str, function: str | None) -> dict:
    for tool in tools or []:
        fn = tool.get("function") or {}
        if fn.get("name") != tool_name:
            continue
        for alt in (fn.get("parameters") or {}).get("x-natlang-alternatives") or []:
            if (alt.get("function") or {}).get("const") == function:
                return alt
    return {}


def _literal_type(value) -> str:
    if isinstance(value, bool):
        return "Bool"
    if isinstance(value, (int, float)):
        return "Num"
    if isinstance(value, str):
        return "Text"
    if value is None:
        return "Null"
    if isinstance(value, list) and value:
        types = {_literal_type(item) for item in value}
        if len(types) == 1:
            return f"({types.pop()})[]"
    if isinstance(value, dict) and value:
        return "{ " + ", ".join(f"{key}: {_literal_type(item)}" for key, item in value.items()) + " }"
    raise ValueError("cannot infer the type of an empty or heterogeneous historical literal; regenerate this action")


def _ordered_paths(args: dict, mode: str, offered_tools: list | None) -> tuple[list[str], list[tuple[str, dict]]]:
    """Order historical named bindings by the selected function signature.

    Literal bindings become explicit typed locals, so the projected call remains
    path-only. Structural types are inferred only when the old action did not
    retain a named type; replay validation remains authoritative.
    """
    alt = _alternative(offered_tools, mode, args.get("function"))
    order = list(alt.get("x-natlang-parameters") or [])
    skip = {"run_function": 0, "for_each": 1, "fold": 2, "repeat": 1}[mode]
    inputs, values = dict(args.get("inputs") or {}), dict(args.get("values") or {})
    bound = {**inputs, **values}
    names = order[skip:] if order else list(bound)
    types = list(alt.get("x-natlang-types") or [])[skip:]
    writes, paths = [], []
    for position, name in enumerate(names):
        if name in inputs:
            paths.append(inputs[name])
        elif name in values:
            value = values[name]
            path = f"let/migrated_{name}"
            ty = types[position] if position < len(types) else _literal_type(value)
            writes.append(("write_value", {"destination": path, "type": ty, "value": value}))
            paths.append(path)
    return paths, writes


def project_actions_v4(name: str, args: dict, *, offered_tools: list | None = None) -> list[tuple[str, dict]]:
    """Project an old action into path-only tools-v4 actions."""
    done = args.get("done")
    action, prefix = project_action_v4(name, args, offered_tools=offered_tools)
    actions = [*prefix, action]
    if done is not None:
        values = done if isinstance(done, list) else [done]
        mark = {"start": values[0]}
        if len(values) > 1:
            mark["end"] = values[-1]
        actions.append(("mark_lines", mark))
    return actions


def project_action_v4(name: str, args: dict, *, offered_tools: list | None = None) -> tuple[tuple[str, dict], list[tuple[str, dict]]]:
    args = copy.deepcopy(args)
    args.pop("done", None)
    if name == "write":
        if "source" in args:
            return ("copy_value", {"source": args["source"], "destination": args["path"]}), []
        stated = str(args.get("type") or "")
        if stated.startswith("Function<") and stated.endswith(">") and "value" not in args:
            return ("copy_function", {"function": stated[9:-1], "save_as": args["path"]}), []
        out = {"destination": args["path"], "type": args.get("type"), "value": args.get("value")}
        return ("write_value", out), []
    if name in ("edit",):
        # Historical exact edits could select any unique substring. The v4
        # exact schema offers bounded existing spans; its fuzzy branch retains
        # the old runtime behavior and still prefers an exact hit first.
        return ("edit_text", {"path": args["path"], "find": args["old"], "fuzzy": True,
                              "replace_with": args.get("new", "")}), []
    if name in ("mark_done",):
        return ("mark_lines", args), []
    if name not in ("call", "call_function"):
        return (name, args), []
    if not any(key in args for key in ("inputs", "values", "over", "init", "until", "max")):
        for tool in offered_tools or []:
            fn = tool.get("function") or {}
            if fn.get("name") == "resume":
                paths = ((fn.get("parameters") or {}).get("properties") or {}).get("computation", {}).get("enum", [])
                if args.get("to") in paths:
                    return ("resume", {"computation": args["to"]}), []
    mode = "repeat" if "until" in args else "fold" if "over" in args and "init" in args else \
           "for_each" if "over" in args else "run_function"
    paths, writes = _ordered_paths(args, mode, offered_tools)
    out = {"function": args["function"]}
    if mode in ("for_each", "fold"):
        out["items"] = args["over"]
    if mode in ("fold", "repeat"):
        initial = args["init"]
        if not (isinstance(initial, str) and initial.startswith(("args/", "let/", "return"))):
            alt = _alternative(offered_tools, mode, args.get("function"))
            declared_types = list(alt.get("x-natlang-types") or [])
            initial_type = declared_types[0] if declared_types else _literal_type(initial)
            suffix = re.sub(r"[^a-z0-9_]+", "_", f"{args.get('function', '')}_{initial_type}".lower()).strip("_")
            initial_path = "let/initial_" + (suffix or "value")
            writes.append(("write_value", {"destination": initial_path,
                                             "type": initial_type,
                                             "value": initial}))
            initial = initial_path
        out["initial"] = initial
    if paths:
        out["inputs"] = paths
    if mode == "repeat":
        out.update(until=args["until"], at_most=args["max"])
    out["save_as"] = args["to"]
    return (mode, out), writes

def project_legacy_turn(row: dict, *, engine: str = "quickjs-isolated") -> dict:
    """Return a new row; retain the original alongside it in corpus storage.

    Only known historical run_code calls are changed. Tool schemas and model
    messages need rebuilding by the materializer for the selected surface.
    """
    projected = copy.deepcopy(row)

    def visit(value):
        if isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            if value.get("name") == "run_code" and isinstance(value.get("args"), dict):
                value["args"].setdefault("engine", engine)
            function = value.get("function")
            if isinstance(function, dict) and function.get("name") == "run_code" and "arguments" in function:
                args = json.loads(function["arguments"])
                if "engine" not in args:
                    args["engine"] = engine
                    function["arguments"] = json.dumps(args, ensure_ascii=False)
            for item in value.values():
                visit(item)

    visit(projected)
    projected["surface_projection"] = {"from": "tools-v2", "to": "tools-v3",
                                       "historical_engine": engine}
    return projected
