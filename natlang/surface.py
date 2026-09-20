"""The model-facing tool surface.

A surface turns a Session into (a) a fixed list of native tool definitions whose
argument schemas are narrowed from the tree and its types each turn, (b) a
rendering of the state, and (c) a mapping from a tool call to a harness
operation. The tool LIST never changes between turns; only argument schemas do.
Surfaces are meant to be swapped while the architecture is being explored.

Surface "tools-v2" follows the ordinary agent loop these models are trained on: the user's
message is a request, inputs are things to read, work happens through tools, and the episode
ends when the assistant replies instead of calling a tool. The result is whatever was written
to `return`; the reply is never parsed as a result. Five tools, always the same:

  read  write  edit  run_code  run

`write(path, type, value)` is typed. A plain type puts a value into the slot. A sub-task type puts
a task there whose RESULT belongs in the slot: Task<T> (instructions), Code<T> (TypeScript),
Map<A, B>, Fold<A, S>, Iterate<S>. Every type is derivable (T, B, S from the slot; A from the
list chosen to map or fold over; a task's parameter types from the inputs wired into it), so
under constrained decoding the type tokens are forced: the model appears to have chosen the right
type, and cannot choose a wrong one.
"""
from __future__ import annotations

import os

import json
from typing import Any, Optional

from .slots import enumerate_slots
from .nodes import MISSING, QUIESCED, RUNNING, UNREDUCED, Lambda, is_pending
from .render import INLINE, PREVIEW_ITEMS, numbered, pending_line, scalar
from .types import (DictT, FoldT, IterateT, LambdaT, ListT, Lit, MapT, Prim, Record, TypeEnv, UnionT,
                    format_type, PENDING_TYPES)
from .values import problems

MAX_PATHS = 48
MARKS_DEFAULT = os.environ.get("NATLANG_MARKS", "1") != "0"
NOTHING_HIDDEN = "(nothing is hidden: every value is already shown in full)"


# --------------------------------------------------------------------------- types -> JSON Schema

def schema_of(t, env: TypeEnv, depth: int = 0) -> dict:
    """JSON Schema for a complete value of type `t`."""
    if depth > 5:
        return {}
    rt = env.resolve(t)
    if isinstance(rt, Prim):
        return {"Text": {"type": "string"}, "Blob": {"type": "string"}, "Num": {"type": "number"},
                "Bool": {"type": "boolean"}, "Null": {"type": "null"}}[rt.name]
    if isinstance(rt, Lit):
        return {"const": rt.value}
    if isinstance(rt, UnionT):
        if all(isinstance(env.resolve(m), Lit) for m in rt.members):
            return {"enum": [env.resolve(m).value for m in rt.members]}
        return {"anyOf": [schema_of(m, env, depth + 1) for m in rt.members]}
    if isinstance(rt, ListT):
        return {"type": "array", "items": schema_of(rt.elem, env, depth + 1)}
    if isinstance(rt, DictT):
        return {"type": "object", "additionalProperties": schema_of(rt.elem, env, depth + 1)}
    if isinstance(rt, Record):
        return {"type": "object",
                "properties": {n: schema_of(ft, env, depth + 1) for n, ft, _ in rt.fields},
                "required": [n for n, _, opt in rt.fields if not opt],
                "additionalProperties": False}
    return {}


def _enum_or_string(values: list, description: str) -> dict:
    values = list(dict.fromkeys(values))[:MAX_PATHS]
    out = {"type": "string", "description": description}
    if values:
        out["enum"] = values
    return out


# --------------------------------------------------------------------------- the surface

class ToolSurface:
    name = "tools-v2"

    def __init__(self, marks: Optional[bool] = None, *, error_tool: bool = True, state_view: bool = False):
        self.error_tool = error_tool
        self.state_view = state_view
        # numbered listing + the `mark` tool for functions that have a code base. NATLANG_MARKS=0 turns it off, to
        # evaluate models trained before marks existed.
        self.marks = MARKS_DEFAULT if marks is None else marks
        self.done_arg = self.marks and os.environ.get("NATLANG_DONE_ARG", "1") != "0"   # `done=` on write and call

    def marking(self, session) -> bool:
        return self.marks and bool(session.lam.codebase)

    def pending(self, session) -> list:
        """Lines still open once marking has begun; [] for unmarked legacy programs and leaves."""
        from .render import pending_lines
        lam = session.lam
        return pending_lines(lam.original_body or lam.body, lam.marks) if self.marking(session) and lam.marks else []
    TOOLS = ("read", "write", "edit", "run_code", "call", "report_blocker", "report_error")   # `call` only when there are functions

    def slots(self, session):
        return enumerate_slots(session.lam, session.outer_env)

    # -- tool definitions -------------------------------------------------------
    def tools(self, session) -> list:
        lam = session.lam
        slots = self.slots(session)
        existing = [s for s in slots if s.value is not MISSING]
        own_args = lambda s: s.ref.holder is lam and s.ref.attr == "in_"
        is_body = lambda s: s.ref.holder is not None and s.ref.attr == "body"

        def value_slot(s):
            if s.ref.deny or s.ref.type is None or own_args(s) or is_body(s):
                return False
            return not isinstance(s.ref.env.resolve(s.ref.type), PENDING_TYPES)

        # The inside of a sub-task is not a place to work: only the inputs it still waits for are offered.
        # (A Fold written into `return` otherwise exposes over/init/step/... and the schema grows tenfold.)
        pending_roots = [x.path for x in slots if x.value is not MISSING and is_pending(x.value)]

        def inside_subtask(x):
            for r in pending_roots:
                if x.path.startswith(r + "/"):
                    rest = x.path[len(r) + 1:].split("/")
                    if not (len(rest) == 2 and rest[0] == "args" and x.value is MISSING):
                        return True
            return False

        slots = [x for x in slots if not inside_subtask(x)]
        writable = [s for s in slots if value_slot(s) and s.path.count("/") <= 3][:MAX_PATHS]
        definable = [s for s in slots if not s.ref.deny and s.ref.type is not None and not own_args(s)
                     and not is_body(s) and s.path.count("/") <= 3][:MAX_PATHS]
        texts = [s for s in existing if isinstance(s.value, str) and not s.ref.deny
                 and (is_body(s) or format_type(s.ref.type) == "Text")
                 and not (self.marking(session) and s.ref.holder is lam and is_body(s))]   # own instructions are immutable
        pending = [s for s in existing if is_pending(s.value) and s.value.status in (UNREDUCED, QUIESCED)
                   and not s.ref.deny]
        retryable = [s for s in existing if not is_pending(s.value) and not s.ref.deny
                     and isinstance(session.rt.origins.get(s.ref.slot_key()) if session.rt else None, Lambda)]
        already_read = getattr(session, "reads_done", set())
        # named values before the elements of lists: with long lists the budget of paths must not be spent on items
        # and what the interpreter made itself (locals, then the result) before the inputs it was given
        zone = lambda p_: 0 if p_.startswith("let/") else 1 if p_.startswith("return") else 2
        named_first = lambda xs: sorted(xs, key=lambda x: (any(seg.isdigit() for seg in x.path.split("/")),
                                                           x.path.count("/") > 2, zone(x.path), x.path.count("/")))
        readable = named_first([s for s in existing if not is_body(s) and s.path not in already_read])

        schemas, seen = [], set()
        for s in writable:
            sch = schema_of(s.ref.type, s.ref.env)
            key = json.dumps(sch, sort_keys=True)
            if key not in seen:
                seen.add(key)
                schemas.append(sch)
        value_schema = schemas[0] if len(schemas) == 1 else ({"anyOf": schemas} if schemas else {})

        def tool(name, description, properties, required, alternatives=None):
            params = {"type": "object", "properties": properties, "required": required,
                      "additionalProperties": False}
            if alternatives:        # argument sets that belong together; used by natlang/native.py
                params["x-natlang-alternatives"] = alternatives
            return {"type": "function", "function": {"name": name, "description": description,
                                                     "parameters": params}}

        def positions(v):
            n = len(v.splitlines()) if isinstance(v, str) else len(v) if isinstance(v, list) else 0
            first = 1 if isinstance(v, str) else 0          # lines are numbered from 1, items from 0
            return list(range(first, first + n)) if 1 < n <= 60 else []

        read_alts = [{"path": {"const": "args"}}] if lam.in_ else []
        for sl in readable[:MAX_PATHS]:
            read_alts.append({"path": {"const": sl.path}})
            pos = positions(sl.value)
            if pos:                                           # a range can only name positions that exist
                read_alts.append({"path": {"const": sl.path}, "start": {"enum": pos}, "end": {"enum": pos}})
        NEW_LOCAL = {"type": "string", "x-natlang": "new-local",
                     "description": "let/<name>: a new local, created by this call"}
        plain = named_first([sl for sl in existing if not is_body(sl) and not is_pending(sl.value)])
        def fitting(type_text, env_types):
            """Paths of existing values that fit a parameter of this type."""
            from .types import fits, parse_type
            try:
                want = parse_type(type_text)
            except Exception:
                return [sl.path for sl in plain][:MAX_PATHS]
            out = []
            for sl in plain:
                env_ = sl.ref.env.child({n: parse_type(t) for n, t in (env_types or {}).items()
                                         if not _declared(sl.ref.env, n)})
                try:
                    if sl.ref.type is not None and fits(sl.ref.type, want, env_):
                        out.append(sl.path)
                except Exception:
                    pass
            return out[:MAX_PATHS]

        targets = [sl.path for sl in definable if not sl.path.startswith("args")][:MAX_PATHS]
        to_schema = {"anyOf": ([{"enum": targets}] if targets else []) + [NEW_LOCAL]}
        functions = dict(lam.codebase)
        for name, tpl in lam.let.items():                  # edited copies are callable too
            if isinstance(tpl, Lambda) and name in lam.fn_copies:
                functions[f"let/{name}"] = lam.fn_copies[name]
        checks = [n for n, f in lam.codebase.items() if f.returns.strip() == "Bool" and len(f.required()) == 1]
        call_alts = []
        for ref_name, f in functions.items():
            names = {n.rstrip("?"): t for n, t in f.args.items()}
            in_props = {n: {"enum": fitting(t, f.types)} for n, t in names.items()}
            in_props = {n: sch for n, sch in in_props.items() if sch["enum"]}
            inputs_schema = {"type": "object", "properties": in_props, "required": [], "additionalProperties": False}
            base = {"function": {"const": ref_name}, "to": to_schema}
            req_names = [n.rstrip("?") for n in f.required()]
            if all(n in in_props for n in req_names):                      # a plain call
                call_alts.append({**base, **({"inputs": {**inputs_schema, "required": req_names}} if names else {})})
            list_paths = [sl.path for sl in plain if isinstance(sl.value, list)]
            if list_paths and names:                                       # once per item of a list
                call_alts.append({**base, "over": {"enum": list_paths}, "inputs": inputs_schema,
                                  "x-optional": ["inputs"]})
                if "acc" in names and "item" in names:                     # carried along a list
                    rest = {n: sch for n, sch in in_props.items() if n not in ("acc", "item")}   # the harness binds those two
                    call_alts.append({**base, "over": {"enum": list_paths}, "init": {},
                                      **({"inputs": {**inputs_schema, "properties": rest}, "x-optional": ["inputs"]} if rest else {})})
            if checks and names:                                           # repeated until a check holds
                starts = list(dict.fromkeys(p_ for sch in in_props.values() for p_ in sch["enum"]))   # fits some parameter
                call_alts.append({**base, "init": {"enum": starts or [sl.path for sl in plain][:MAX_PATHS]},
                                  "until": {"enum": checks}, "max": {"type": "integer"},
                                  "inputs": inputs_schema, "x-optional": ["inputs"]})

        value_alts = [{"path": {"const": s.path}, "type": {"const": format_type(s.ref.type)},
                       "value": schema_of(s.ref.type, s.ref.env)} for s in writable]
        value_alts.append({"path": NEW_LOCAL, "type": {"type": "string"}, "value": {}})
        source_alts = []
        for s_ in writable:                                # copy an existing value that fits, instead of re-emitting it
            srcs = [p_ for p_ in fitting(format_type(s_.ref.type), {}) if p_ != s_.path and not p_.startswith(s_.path + "/")]
            if srcs:
                source_alts.append({"path": {"const": s_.path}, "type": {"const": format_type(s_.ref.type)},
                                    "source": {"enum": srcs}})
        copy_alts = [{"path": NEW_LOCAL, "type": {"const": f"Function<{n}>"}} for n in lam.codebase]
        write_alts = value_alts + source_alts + copy_alts
        shapes, seen_shapes = [], set()
        for a_ in value_alts:
            # Untyped schemas produced object wrappers through the teacher's
            # XML tool interface. Expose scalar alternatives explicitly.
            alternatives = [a_["value"]] if a_["value"] else [
                {"type": t} for t in ("string", "number", "boolean", "null", "object", "array")]
            for shape in alternatives:
                k_ = json.dumps(shape, sort_keys=True, default=str)
                if k_ not in seen_shapes:
                    seen_shapes.add(k_)
                    shapes.append(shape)
        any_value = {"description": "The value itself, complete (not wrapped in an object).",
                     "anyOf": shapes}              # explicit JSON shapes allow new locals and invalid proposals

        tools = [
            tool("read", "Read a value from the workspace. Optional line or item range for long ones. "
                         "`codebase/<function>` shows the text of a function.",
                 {"path": _enum_or_string((["args"] if lam.in_ else []) + [s.path for s in readable] +
                                          [f"codebase/{n}" for n in lam.codebase], "what to read"),
                  "start": {"type": "integer"}, "end": {"type": "integer"}}, ["path"],
                 alternatives=read_alts + [{"path": {"const": f"codebase/{n}"}} for n in lam.codebase]),
            tool("write", "Write a value into the workspace: into `return`, or into a local `let/<name>` (a new name "
                          "creates the local; `type` says what it holds). Supply `value` or `source`; a type alone is not a value. "
                          "The value must be complete; to reuse a value that "
                          "already exists, give `source` (its path) instead of `value`. Source copying preserves the value and type; "
                          "it does not wrap or convert. Use the destination requested by the program; "
                          "do not append a field name to make incompatible types fit. "
                          "To change how a function works, copy it first: type `Function<name>` with path "
                          "`let/<copy>`, then `edit` `let/<copy>/instructions`, then `call` it as `let/<copy>`.",
                 {"path": {"type": "string", "description": "`return`, a part of it, or let/<name>"},
                  "type": {"type": "string", "description": "the type of what is written, e.g. Bool[], Text[], Num, Text, "
                                                          "{ name: Text, count: Num }, or a type name of this task"},
                  "value": any_value,
                  "source": {"type": "string", "description": "instead of `value`: the path of an existing value to copy"}},
                 ["path", "type"], alternatives=write_alts),
            tool("edit", "Replace text: `old` must occur exactly once in the text at `path`. "
                         "Use it to delete finished steps from `instructions` (new = \"\"), to substitute a "
                         "result into them, or to adapt a copied function.",
                 {"path": _enum_or_string([s.path for s in texts], "a text"),
                  "old": {"type": "string"}, "new": {"type": "string"}}, ["path", "old", "new"]),
            tool("run_code", ("Run code" if session.rt.engine_selection else "Run TypeScript") +
                             " for exact work (counting, arithmetic, sorting, string operations). "
                             "Your inputs are in `args`, your locals in `locals`. The value of the last expression "
                             "comes back to you." + (" Select an available engine." if session.rt.engine_selection else ""),
                 {"code": {"type": "string"}, **({"engine": {"enum": sorted(session.rt.executors)}}
                                                    if session.rt.engine_selection else {})},
                 ["code", "engine"] if session.rt.engine_selection else ["code"]),
        ]
        if functions:
            tools.append(
                tool("call", "Call one of your functions and put its result at `to` (`return`, a part of it, or a "
                             "local `let/<name>`). `inputs` maps each parameter to the path of its value. With "
                             "`over`: call it once for every item of that list (the item goes to the one parameter "
                             "you left out). Do not put the mapped item in inputs. The result is the list of results. "
                             "With `over` and `init`, omit both item and acc from inputs: carry `acc` "
                             "through the list. With `init`, `until`, `max`: repeat from the value at `init` until "
                             "the function `until` says true, at most `max` times. Calling again with only "
                             "`function` and `to` retries what did not finish.",
                     {"function": {"enum": list(functions)}, "to": {"type": "string"},
                      "inputs": {"type": "object", "properties": {n.rstrip("?"): {"type": "string"}
                          for f in functions.values() for n in f.args}, "additionalProperties": False},
                      "over": {"type": "string"}, "init": {"anyOf": [{"type": t} for t in
                          ("string", "number", "boolean", "null", "object", "array")]}, "until": {"type": "string"},
                      "max": {"type": "integer"}}, ["function", "to"],
                     alternatives=call_alts + [{"function": {"const": n}, "to": {"enum": [p_]}}
                                               for n in functions for p_ in
                                               [sl.path for sl in existing if is_pending(sl.value)
                                                and sl.value.status in (UNREDUCED, QUIESCED)][:4]]))
        # A literal write must carry a value, a reference copy a source. Only
        # copying a named function can omit both. Keep the server schema as
        # explicit about this as the native grammar's alternatives already are.
        write_params = tools[1]["function"]["parameters"]
        write_params["anyOf"] = [{"required": ["value"]}, {"required": ["source"]}]
        if lam.codebase:
            write_params["anyOf"].append({"properties": {"type": {
                "enum": [f"Function<{n}>" for n in lam.codebase]}}, "required": ["type"]})
        if self.marking(session):
            from .render import pending_lines, program_lines
            open_ = pending_lines(lam.original_body or lam.body, lam.marks)
            if open_:
                # Legacy reference plans sometimes include a function declaration in
                # a mark range. It has a line number but no work to discharge.
                markable_numbers = sorted(set(open_) | {n for n, text, markable in
                                                      program_lines(lam.original_body or lam.body)
                                                      if not markable and text.strip().startswith("function ")})
                yes = {"const": True}
                tools.append(
                    tool("mark_done", "Mark lines of your program as finished. `start` alone for one line, `start` and `end` "
                                      "for an inclusive range (EVERY line between the endpoints). Add skipped=true when the lines did not apply, such as the branch of an "
                                      "`if` that was not taken. Mark a line only after everything it asks for is finished.",
                         {"start": {"type": "integer"}, "end": {"type": "integer"}, "skipped": {"type": "boolean"}}, ["start"],
                         alternatives=[{"start": {"enum": markable_numbers}, "skipped": yes, "x-optional": ["skipped"]},
                                       {"start": {"enum": markable_numbers}, "end": {"enum": markable_numbers}, "skipped": yes, "x-optional": ["skipped"]}]))
                if self.done_arg:                       # en passant: the same mark as an argument of the action that finishes the line
                    line = {"anyOf": [{"enum": markable_numbers}, {"type": "array", "items": {"enum": markable_numbers}, "minItems": 1, "maxItems": 2}]}
                    for t in tools:
                        if t["function"]["name"] in ("write", "call"):
                            t["function"]["parameters"]["properties"]["done"] = {
                                **line, "description": "line or inclusive [first, last] range that this action finishes; EVERY line in the range is marked done on success. Never include an untaken branch."}
                            for alt in t["function"]["parameters"].get("x-natlang-alternatives") or []:
                                alt["done"] = line
                                alt["x-optional"] = list(alt.get("x-optional") or []) + ["done"]
        tools.append(
            tool("report_blocker", "The task cannot be done as asked: the inputs do not determine the result, or a rule "
                                   "does not cover the case. Say exactly what is missing. This ends the task without "
                                   "a result; do not guess instead.",
                 {"missing": {"type": "string"}}, ["missing"]))
        if self.error_tool:
            tools.append(tool("report_error", "The executed instructions cannot be satisfied: a contradiction, invalid "
                              "operation, or incompatible required result prevents correct completion. Explain the "
                              "error. This ends the task without a result. Do not change the requirements to succeed. "
                              "Use report_blocker for missing information instead.",
                              {"message": {"type": "string"}}, ["message"]))
        return tools

    # -- what the model is shown ----------------------------------------------------
    def render_request(self, session) -> str:
        """The opening user message: the instructions as a request, and nothing else. Data never
        shares a channel with instructions: the workspace arrives as a tool result (`opening_read`)."""
        lam = session.lam
        body = lam.body.strip() or "(no instructions left)"
        if self.marking(session):
            from .render import listing
            body = (listing(lam.original_body or lam.body, lam.marks) +
                    "\n\nThe lines are numbered. [ ] is still to do, [x] is done, [-] did not apply. Mark lines done as you finish them.")
        fns = self.functions(session)
        return f"{body}\n\nWrite the result to `return` ({format_type(lam.type.returns)})." + (f"\n\n{fns}" if fns else "")

    def opening_read(self, session):
        """A first step the harness performs on the agent's behalf: read the workspace. Returns
        (tool name, arguments, result text), or None when there is nothing to read."""
        if not session.lam.in_ and not session.lam.type.params.fields and not self.state_view:
            return None
        return "read", {"path": "args"}, self.execution_state(session) if self.state_view else self.render_state(session)

    def functions(self, session) -> str:
        from .codebase import listing
        text = listing(session.lam.codebase)
        return ("Functions you can call:\n" + text) if text else ""

    def render_state(self, session) -> str:
        """The workspace. Small values are shown inline; long ones are listed for `read`.
        Anti-parroting rules: no value-like placeholders; filled values apart from what is missing."""
        lam, env = session.lam, session.env
        out = ["Workspace:"]
        for n, ft, _ in lam.type.params.fields:
            if n in lam.in_:
                out.append(f"  args/{n} ({format_type(ft)}, read-only): {_preview(lam.in_[n])}")
            else:
                out.append(f"  args/{n} ({format_type(ft)}, read-only): not supplied")
        if self.state_view and not lam.let_types:
            out.append("  locals: none computed")
        for n, t in lam.let_types.items():
            v = lam.let.get(n, MISSING)
            if v is not MISSING:
                kind = f"a copy of {lam.fn_copies[n].name}, editable" if n in lam.fn_copies else format_type(t)
                out.append(f"  let/{n} ({kind}): {'' if n in lam.fn_copies else _preview(v)}".rstrip(": ").rstrip())
        filled, todo, subs = [], [], []
        self._walk(lam.ret, lam.type.returns, env, "return", filled, todo, subs)
        out.append(f"  return ({format_type(lam.type.returns)}): " +
                   ("not written yet" if lam.ret is MISSING else "written" if not todo and not subs else "partly written"))
        if filled and (self.state_view or todo or subs):
            out.append("    written so far: " + "; ".join(filled))
        if todo and lam.ret is not MISSING:
            out.append("    still missing: " + ", ".join(todo))
        if subs:
            out.append("    sub-tasks: " + "; ".join(subs))
        return "\n".join(out)

    def execution_state(self, session) -> str:
        """Factual state only: no inferred next action or parsing of pseudocode."""
        from .render import listing
        parts = [self.render_state(session)]
        if self.marking(session):
            parts.append("Current program marks:\n" + listing(session.lam.original_body or session.lam.body, session.lam.marks))
        if self.functions(session):
            parts.append(self.functions(session))
        return "\n\n".join(parts)

    def missing(self, session) -> str:
        """One line saying what `return` still needs, or '' when complete. No data in it: this goes
        into a user message."""
        lam = session.lam
        if lam.ret is MISSING:
            return f"`return` has not been written yet. Write a {format_type(lam.type.returns)} to `return`."
        filled, todo, subs = [], [], []
        self._walk(lam.ret, lam.type.returns, session.env, "return", filled, todo, subs)
        if subs:
            return "A sub-task has not been run yet: " + ", ".join(x.split(" [")[0] for x in subs) + ". Run it."
        if todo:
            return "`return` is missing: " + ", ".join(todo) + "."
        return ""

    def _walk(self, value, t, env, path, filled, todo, subs, depth=0):
        if value is MISSING:
            todo.append(f"{path} ({format_type(t)})")
            return
        if is_pending(value):
            subs.append(f"{path} [{pending_line(value)}]")
            return
        rt = env.resolve(t)
        if isinstance(rt, Record) and isinstance(value, dict) and depth < 3:
            for n, ft, opt in rt.fields:
                if n in value:
                    self._walk(value[n], ft, env, f"{path}/{n}", filled, todo, subs, depth + 1)
                elif not opt:
                    todo.append(f"{path}/{n} ({format_type(ft)})")
            return
        if isinstance(value, list) and any(is_pending(x) for x in value):
            done = sum(1 for x in value if not is_pending(x))
            subs.append(f"{path} [{done} of {len(value)} items done]")
            return
        filled.append(f"{path} = {_preview(value)}")

    # -- tool call -> harness operation -----------------------------------------
    def apply(self, session, name: str, args: dict):
        result = session.apply(name, args or {})
        if self.state_view and result.kind in ("ok", "done", "quiesced"):
            result.text = result.text.rstrip() + "\n\n" + self.execution_state(session)
        return result


def _declared(env, name: str) -> bool:
    from .types import TypeSyntaxError, parse_type
    try:
        env.check_names(parse_type(name))
        return True
    except TypeSyntaxError:
        return False


FULL_TEXT, FULL_LINES = 400, 8     # a text up to this size is shown whole: a cut-off rubric reads like a complete one


def is_previewed(v) -> bool:
    """True when the workspace listing shows only part of `v`, so that it has to be read."""
    if isinstance(v, str):
        t = v.rstrip()
        return len(t) > FULL_TEXT or len(t.splitlines()) > FULL_LINES
    return isinstance(v, list) and len(v) > PREVIEW_ITEMS


def _preview(v) -> str:
    if isinstance(v, str):
        t = v.rstrip()
        if not is_previewed(v):
            return json.dumps(t, ensure_ascii=False) if "\n" not in t else \
                "\n" + "\n".join("      | " + l for l in t.splitlines())
        return (json.dumps(t.splitlines()[0][:INLINE], ensure_ascii=False) +
                f" … CUT OFF: only the beginning of {len(t.splitlines())} lines, {len(t)} characters. Read it before using it.")
    if is_pending(v):
        return f"[{pending_line(v)}]"
    if isinstance(v, list):
        head = ", ".join(_preview(x) for x in v[:PREVIEW_ITEMS])
        more = f", … {len(v) - PREVIEW_ITEMS} more (read to see)" if len(v) > PREVIEW_ITEMS else ""
        return f"{len(v)} items: [{head}{more}]"
    if isinstance(v, dict):
        return "{ " + ", ".join(f"{k}: {_preview(x)}" for k, x in list(v.items())[:6]) + " }"
    return scalar(v)
