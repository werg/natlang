"""The mode-explicit, positional tools-v4 model surface.

Runtime internals retain named fields for readable program bodies. The model-facing
calling convention is positional and path-only: `inputs` is an ordered list of
workspace paths, bound to declared lambda parameters from left to right.
"""
from __future__ import annotations

import copy

from .nodes import MISSING, QUIESCED, UNREDUCED, is_pending
from .surface import ToolSurface, schema_of
from .types import fits, format_type, parse_type


FUTURE_LOCAL = {"type": "string", "pattern": "^let/[a-z_][a-z0-9_]*$",
                "description": "a local produced by another call in this batch"}
NEW_LOCAL = {"type": "string", "x-natlang": "new-local",
             "pattern": "^let/[a-z_][a-z0-9_]*$", "description": "a new local destination"}


def _tool(name, description, properties, required, alternatives=None):
    parameters = {"type": "object", "properties": properties, "required": required,
                  "additionalProperties": False}
    if alternatives:
        parameters["x-natlang-alternatives"] = alternatives
    return {"type": "function", "function": {"name": name, "description": description,
                                                "parameters": parameters}}


def _unique(shapes):
    out = []
    for shape in shapes:
        if shape and all(repr(shape) != repr(existing) for existing in out):
            out.append(copy.deepcopy(shape))
    return out


def _merged(shapes, fallback=None):
    shapes = _unique(shapes)
    return shapes[0] if len(shapes) == 1 else {"anyOf": shapes} if shapes else (fallback or {})


def _text_spans(text: str, maximum: int = 96) -> list[str]:
    lines, out = text.splitlines(keepends=True), []
    for width in range(1, 4):
        for start in range(0, len(lines) - width + 1):
            span = "".join(lines[start:start + width])
            if span.strip() and len(span) <= 600 and text.count(span) == 1:
                out.append(span)
                trimmed = span.rstrip("\r\n")
                if trimmed and trimmed != span and text.count(trimmed) == 1:
                    out.append(trimmed)
    return list(dict.fromkeys(sorted(out, key=lambda value: (len(value), text.find(value)))))[:maximum]


class ExplicitToolSurface(ToolSurface):
    name = "tools-v4"

    def tools(self, session) -> list:
        legacy = super().tools(session)
        by_name = {item["function"]["name"]: item for item in legacy}
        lam, slots = session.lam, self.slots(session)
        is_body = lambda slot: slot.ref.holder is not None and slot.ref.attr == "body"
        definitions = dict(lam.codebase)
        definitions.update({f"let/{name}": definition for name, definition in lam.fn_copies.items()})

        def env_for(definition):
            names = {name: parse_type(text) for name, text in definition.types.items()
                     if session.env.lookup(name) is None}
            return session.env.child(names)

        def references(type_text, definition):
            expected, env, paths = parse_type(type_text), env_for(definition), []
            for slot in slots:
                if (slot.value is MISSING or is_pending(slot.value) or slot.ref.type is None
                        or is_body(slot)):
                    continue
                try:
                    if fits(slot.ref.type, expected, env):
                        paths.append(slot.path)
                except Exception:
                    pass
            existing = {"enum": list(dict.fromkeys(paths))[:48]} if paths else None
            return {"anyOf": [existing, FUTURE_LOCAL]} if existing else copy.deepcopy(FUTURE_LOCAL)

        def destinations(type_text, definition):
            paths = []
            for slot in slots:
                if (slot.path.startswith("args") or slot.ref.deny or slot.ref.type is None
                        or is_body(slot)):
                    continue
                paths.append(slot.path)
            existing = [{"enum": list(dict.fromkeys(paths))[:48]}] if paths else []
            return {"anyOf": [*existing, NEW_LOCAL]}

        def positional_inputs(definition, skip):
            ordered = list(definition.args.items())[skip:]
            seen_optional = False
            for raw, _ in ordered:
                if raw.endswith("?"):
                    seen_optional = True
                elif seen_optional:
                    return None
            prefix = [references(text, definition) for _, text in ordered]
            required = sum(not raw.endswith("?") for raw, _ in ordered)
            # `maxItems` closes the tuple. Some OpenAI-compatible servers reject
            # the JSON Schema boolean form `items: false`.
            return {"type": "array", "prefixItems": prefix, "items": {},
                    "minItems": required, "maxItems": len(prefix)}

        result = [copy.deepcopy(by_name["read"])]
        code = by_name["run_code"]["function"]
        code_props = code["parameters"]["properties"]
        ordered_code = {name: copy.deepcopy(code_props[name]) for name in ("engine", "code") if name in code_props}
        result.append(_tool("run_code", code["description"], ordered_code, list(ordered_code)))

        value_alts, copy_alts, function_alts = [], [], []
        for raw in by_name["write"]["function"]["parameters"].get("x-natlang-alternatives") or []:
            if "value" in raw:
                if (raw.get("path") or {}).get("x-natlang") == "new-local":
                    continue
                alt = {"destination": raw["path"], "type": raw["type"]}
                alt["value"] = raw["value"]
                value_alts.append(alt)
            elif "source" in raw:
                copy_alts.append({"source": raw["source"], "destination": raw["path"]})
            elif str((raw.get("type") or {}).get("const", "")).startswith("Function<"):
                name = raw["type"]["const"][9:-1]
                function_alts.append({"function": {"const": name}, "save_as": raw["path"]})
        basic_types = ("Num", "Text", "Bool", "Null", "Num[]", "Text[]", "Bool[]",
                       "Dict<Num>", "Dict<Text>", "Dict<Bool>")
        local_shapes = [(name, schema_of(parse_type(name), session.env)) for name in basic_types]
        for slot in slots:
            if slot.ref.type is not None and not is_body(slot):
                shape = schema_of(slot.ref.type, slot.ref.env)
                if shape:
                    local_shapes.append((format_type(slot.ref.type), shape))
        for definition in definitions.values():
            env = env_for(definition)
            for type_text in [*definition.args.values(), definition.returns]:
                shape = schema_of(parse_type(type_text), env)
                if shape:
                    local_shapes.append((type_text, shape))
        seen_shapes = set()
        for type_text, shape in local_shapes:
            key = (type_text, repr(shape))
            if key in seen_shapes:
                continue
            seen_shapes.add(key)
            value_alts.append({"destination": NEW_LOCAL, "type": {"const": type_text}, "value": shape})
            if len(seen_shapes) >= 32:
                break
        for slot in slots:
            if (slot.value is not MISSING and not is_pending(slot.value)
                    and slot.ref.type is not None and not is_body(slot)):
                copy_alts.append({"source": {"const": slot.path}, "destination": NEW_LOCAL})

        result.append(_tool("write_value", "Write one literal value. Choose its destination and type before generating the value.",
                            {"destination": {"type": "string"}, "type": {"type": "string"}, "value": {}},
                            ["destination", "type", "value"], value_alts))
        if copy_alts:
            result.append(_tool("copy_value", "Copy a value between workspace paths.",
                                {"source": {"type": "string"}, "destination": {"type": "string"}},
                                ["source", "destination"], copy_alts))
        if function_alts:
            result.append(_tool("copy_function", "Make an editable local copy of a named function.",
                                {"function": {"enum": list(lam.codebase)}, "save_as": NEW_LOCAL},
                                ["function", "save_as"], function_alts))

        edit = by_name.get("edit")
        if edit:
            offered = edit["function"]["parameters"]["properties"]["path"].get("enum", [])
            exact, fuzzy = [], []
            for slot in slots:
                if slot.path not in offered or not isinstance(slot.value, str):
                    continue
                spans = _text_spans(slot.value)
                if spans:
                    exact.append({"path": {"const": slot.path}, "find": {"enum": spans},
                                  "replace_with": {"type": "string"}})
                fuzzy.append({"path": {"const": slot.path}, "find": {"type": "string"},
                              "fuzzy": {"const": True}, "replace_with": {"type": "string"}})
            result.append(_tool("edit_text", "Replace existing text. Use fuzzy only for one unambiguous inexact selection.",
                                {"path": {"enum": offered}, "find": {"type": "string"},
                                 "fuzzy": {"type": "boolean"}, "replace_with": {"type": "string"}},
                                ["path", "find", "replace_with"], exact + fuzzy))

        call_alts, map_alts, fold_alts, repeat_alts = [], [], [], []
        bool_checks = [(name, definition, list(definition.args.items())[0][1])
                       for name, definition in definitions.items()
                       if definition.returns.strip() == "Bool" and definition.args]
        for name, definition in definitions.items():
            args = list(definition.args.items())
            inputs = positional_inputs(definition, 0)
            if inputs is not None:
                alt = {"function": {"const": name},
                       "x-natlang-parameters": [raw.rstrip("?") for raw in definition.args],
                       "x-natlang-types": list(definition.args.values())}
                if inputs["maxItems"]:
                    alt["inputs"] = inputs
                    if inputs["minItems"] == 0: alt["x-optional"] = ["inputs"]
                alt["save_as"] = destinations(definition.returns, definition)
                call_alts.append(alt)

            if args:
                extra = positional_inputs(definition, 1)
                if extra is not None:
                    alt = {"function": {"const": name},
                           "x-natlang-parameters": [raw.rstrip("?") for raw, _ in args],
                           "x-natlang-types": [text for _, text in args],
                           "items": references(f"({args[0][1]})[]", definition)}
                    if extra["maxItems"]:
                        alt["inputs"] = extra
                        if extra["minItems"] == 0: alt["x-optional"] = ["inputs"]
                    alt["save_as"] = destinations(f"({definition.returns})[]", definition)
                    map_alts.append(alt)

            if len(args) >= 2:
                acc_type, item_type = args[0][1], args[1][1]
                try:
                    fold_ok = fits(parse_type(definition.returns), parse_type(acc_type), env_for(definition))
                except Exception:
                    fold_ok = False
                extra = positional_inputs(definition, 2)
                if fold_ok and extra is not None:
                    alt = {"function": {"const": name},
                           "x-natlang-parameters": [raw.rstrip("?") for raw, _ in args],
                           "x-natlang-types": [text for _, text in args],
                           "items": references(f"({item_type})[]", definition),
                           "initial": references(acc_type, definition)}
                    if extra["maxItems"]:
                        alt["inputs"] = extra
                        if extra["minItems"] == 0: alt["x-optional"] = ["inputs"]
                    alt["save_as"] = destinations(acc_type, definition)
                    fold_alts.append(alt)

            if args:
                state_type = args[0][1]
                try:
                    state_ok = fits(parse_type(definition.returns), parse_type(state_type), env_for(definition))
                except Exception:
                    state_ok = False
                checks = [check for check, check_def, check_type in bool_checks
                          if len(check_def.required()) == 1 and
                          fits(parse_type(state_type), parse_type(check_type), env_for(check_def))]
                extra = positional_inputs(definition, 1)
                if state_ok and checks and extra is not None:
                    alt = {"function": {"const": name},
                           "x-natlang-parameters": [raw.rstrip("?") for raw, _ in args],
                           "x-natlang-types": [text for _, text in args],
                           "initial": references(state_type, definition)}
                    if extra["maxItems"]:
                        alt["inputs"] = extra
                        if extra["minItems"] == 0: alt["x-optional"] = ["inputs"]
                    alt.update({"until": {"enum": checks}, "at_most": {"type": "integer"},
                                "save_as": destinations(state_type, definition)})
                    repeat_alts.append(alt)

        def add_mode(name, description, alternatives, fields):
            if not alternatives:
                return
            props = {field: _merged([alt.get(field) for alt in alternatives],
                                    {"type": "array", "items": {"type": "string"}}
                                    if field == "inputs" else {}) for field in fields}
            result.append(_tool(name, description, props,
                                [field for field in fields if field != "inputs"], alternatives))

        add_mode("run_function", "Run a function once. `inputs` lists workspace paths in parameter order.",
                 call_alts, ("function", "inputs", "save_as"))
        add_mode("for_each", "Run a function for every item in `items`. The item fills parameter 1; `inputs` fills the rest.",
                 map_alts, ("function", "items", "inputs", "save_as"))
        add_mode("fold", "Fold `items`. Accumulator fills parameter 1, item parameter 2, and `inputs` fills the rest.",
                 fold_alts, ("function", "items", "initial", "inputs", "save_as"))
        add_mode("repeat", "Repeat a state transition. State fills parameter 1 and `inputs` fills the rest.",
                 repeat_alts, ("function", "initial", "inputs", "until", "at_most", "save_as"))

        pending = [slot.path for slot in slots if is_pending(slot.value)
                   and slot.value.status in (UNREDUCED, QUIESCED)]
        if pending:
            result.append(_tool("resume", "Continue a pending computation with its retained inputs and progress.",
                                {"computation": {"enum": pending}}, ["computation"],
                                [{"computation": {"const": path}} for path in pending]))
        if "mark_done" in by_name:
            mark = copy.deepcopy(by_name["mark_done"])
            mark["function"]["name"] = "mark_lines"
            result.append(mark)
        for name in ("report_blocker", "report_error"):
            if name in by_name:
                result.append(copy.deepcopy(by_name[name]))
        return result
