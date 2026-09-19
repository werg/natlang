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

import json
from typing import Any, Optional

from .grammar import enumerate_slots
from .nodes import MISSING, QUIESCED, RUNNING, UNREDUCED, Lambda, is_pending
from .render import INLINE, PREVIEW_ITEMS, numbered, pending_line, scalar
from .types import (DictT, FoldT, IterateT, LambdaT, ListT, Lit, MapT, Prim, Record, TypeEnv, UnionT,
                    format_type, PENDING_TYPES)
from .values import problems

MAX_PATHS = 24
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
    TOOLS = ("read", "write", "edit", "run_code", "run", "report_blocker")

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

        writable = [s for s in slots if value_slot(s) and s.path.count("/") <= 3][:MAX_PATHS]
        definable = [s for s in slots if not s.ref.deny and s.ref.type is not None and not own_args(s)
                     and not is_body(s) and s.path.count("/") <= 3][:MAX_PATHS]
        texts = [s for s in existing if isinstance(s.value, str) and not s.ref.deny
                 and (is_body(s) or format_type(s.ref.type) == "Text")]
        pending = [s for s in existing if is_pending(s.value) and s.value.status in (UNREDUCED, QUIESCED)
                   and not s.ref.deny]
        retryable = [s for s in existing if not is_pending(s.value) and not s.ref.deny
                     and isinstance(session.rt.origins.get(s.ref.slot_key()) if session.rt else None, Lambda)]
        already_read = getattr(session, "reads_done", set())
        readable = [s for s in existing if not is_body(s) and s.path not in already_read]

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
        sources = [sl.path for sl in existing if not is_body(sl) and not is_pending(sl.value)
                   and sl.path.count("/") <= 2][:MAX_PATHS]
        inputs = {"type": "object", "additionalProperties": {"enum": sources}} if sources else None
        lists = [(sl.path, sl.ref.env.resolve(sl.ref.type).elem) for sl in existing
                 if isinstance(sl.value, list) and sl.ref.type is not None
                 and isinstance(sl.ref.env.resolve(sl.ref.type), ListT)]

        def body(required: dict, optional: dict = None) -> dict:
            props = {**required, **{k: v for k, v in (optional or {}).items() if v}}
            return {"type": "object", "properties": props, "required": list(required),
                    "additionalProperties": False}

        text = {"type": "string"}
        later = {"type": "object", "additionalProperties": {"type": "string"}}   # name -> type, produced by a sub-task
        task_alts = []
        # Sub-task forms are offered for named slots only: offering them for every element of a filled list
        # multiplies the tool schema by the list length (a four-item `return` cost 15k characters).
        is_element = lambda x: x.path.rsplit("/", 1)[-1].isdigit()
        for sl in [x for x in definable if x.path.count("/") <= 2 and not is_element(x)]:
            t, env_ = sl.ref.type, sl.ref.env
            rt, ft = env_.resolve(t), format_type(t)
            if isinstance(rt, PENDING_TYPES):
                continue
            alt_ = lambda ty, val: {"path": {"const": sl.path}, "type": {"const": ty}, "value": val}
            task_alts.append(alt_(f"Task<{ft}>", body({"instructions": text}, {"inputs": inputs, "params": later})))
            task_alts.append(alt_(f"Code<{ft}>", body({"code": text}, {"inputs": inputs, "params": later})))
            for over, a in lists:
                fa = format_type(a)
                if isinstance(rt, ListT):
                    task_alts.append(alt_(f"Map<{fa}, {format_type(rt.elem)}>",
                                          body({"over": {"const": over}, "instructions": text}, {"inputs": inputs})))
                task_alts.append(alt_(f"Fold<{fa}, {ft}>",
                                      body({"over": {"const": over}, "init": schema_of(t, env_), "instructions": text},
                                           {"inputs": inputs})))
            task_alts.append(alt_(f"Iterate<{ft}>",
                                  body({"init": schema_of(t, env_), "instructions": text, "until": text,
                                        "max": {"type": "integer"}}, {"inputs": inputs})))
        value_alts = [{"path": {"const": s.path}, "type": {"const": format_type(s.ref.type)},
                       "value": schema_of(s.ref.type, s.ref.env)} for s in writable]
        write_alts = value_alts + task_alts
        shapes, seen_shapes = [], set()              # for servers that read plain JSON Schema: every shape `value` may take
        for a_ in write_alts:
            k_ = json.dumps(a_["value"], sort_keys=True, default=str)
            if k_ not in seen_shapes:
                seen_shapes.add(k_)
                shapes.append(a_["value"])
        any_value = {"description": "For a plain type: the value itself (not wrapped in an object). "
                                    "For a sub-task type: an object with instructions or code.",
                     "anyOf": shapes} if shapes else {}
        all_types = list(dict.fromkeys(a["type"]["const"] for a in write_alts))

        sub = {"type": "object", "description": "A sub-task: its type, and either instructions or code.",
               "properties": {"type": {"type": "string", "description": "e.g. Lambda<{ item: Text }, Bool>"},
                              "instructions": {"type": "string"}, "code": {"type": "string"},
                              "args": {"type": "object"},
                              "args_from": {"type": "object", "additionalProperties": {"type": "string"}}},
               "required": ["type"]}
        paths = [s.path for s in existing]
        return [
            tool("read", "Read a value from the workspace. Optional line or item range for long ones.",
                 {"path": _enum_or_string((["args"] if lam.in_ else []) + [s.path for s in readable], "what to read"),
                  "start": {"type": "integer"}, "end": {"type": "integer"}}, ["path"], alternatives=read_alts),
            tool("write", "Write into the workspace. `type` says what you are putting at `path`. A plain type (as shown "
                          "for the slot): `value` is the value itself, complete. Task<T>: a sub-task whose result goes "
                          "there; `value` = {instructions, inputs}. Code<T>: the same with {code} in TypeScript, for "
                          "exact work. Map<A, B>: do the instructions once for every item of the list `over` (the item "
                          "is `args/item`). Fold<A, S>: carry `init` through the list item by item (`args/acc`, "
                          "`args/item`). Iterate<S>: repeat from `init` until `until` holds, at most `max` times. "
                          "`inputs` maps a name to a path whose value the sub-task receives as `args/<name>`; `params` "
                          "declares inputs (name -> type) that another sub-task, written into `<path>/args/<name>`, will produce. "
                          "After writing a sub-task, `run` it.",
                 {"path": _enum_or_string([s.path for s in definable], "where the value or the result belongs"),
                  "type": _enum_or_string(all_types, "what is being written"),
                  "value": any_value}, ["path", "type", "value"], alternatives=write_alts),
            tool("edit", "Replace text: `old` must occur exactly once in the text at `path`. "
                         "Use it to delete finished steps from `instructions` (new = \"\") or to substitute a "
                         "result into them.",
                 {"path": _enum_or_string([s.path for s in texts], "a text"),
                  "old": {"type": "string"}, "new": {"type": "string"}}, ["path", "old", "new"]),
            tool("run_code", "Run TypeScript for exact work (counting, arithmetic, sorting, string operations). "
                             "Your inputs are in `args`. The value of the last expression comes back to you.",
                 {"code": {"type": "string"}}, ["code"]),
            tool("run", "Run sub-tasks you defined and wait for their results.",
                 {"paths": {"type": "array", "minItems": 1,
                            "items": _enum_or_string([s.path for s in pending], "a sub-task")}}, ["paths"]),
            tool("report_blocker", "The task cannot be done as asked: the inputs do not determine the result, or a rule "
                                   "does not cover the case. Say exactly what is missing. This ends the task without "
                                   "a result; do not guess instead.",
                 {"missing": {"type": "string"}}, ["missing"]),
        ]

    # -- what the model is shown ----------------------------------------------------
    def render_request(self, session) -> str:
        """The opening user message: the instructions as a request, and nothing else. Data never
        shares a channel with instructions: the workspace arrives as a tool result (`opening_read`)."""
        lam = session.lam
        body = lam.body.strip() or "(no instructions left)"
        return f"{body}\n\nWrite the result to `return` ({format_type(lam.type.returns)})."

    def opening_read(self, session):
        """A first step the harness performs on the agent's behalf: read the workspace. Returns
        (tool name, arguments, result text), or None when there is nothing to read."""
        if not session.lam.in_:
            return None
        return "read", {"path": "args"}, self.render_state(session)

    def render_state(self, session) -> str:
        """The workspace. Small values are shown inline; long ones are listed for `read`.
        Anti-parroting rules: no value-like placeholders; filled values apart from what is missing."""
        lam, env = session.lam, session.env
        out = ["Workspace:"]
        for n, ft, _ in lam.type.params.fields:
            if n in lam.in_:
                out.append(f"  args/{n} ({format_type(ft)}, read-only): {_preview(lam.in_[n])}")
        filled, todo, subs = [], [], []
        self._walk(lam.ret, lam.type.returns, env, "return", filled, todo, subs)
        out.append(f"  return ({format_type(lam.type.returns)}): " +
                   ("not written yet" if lam.ret is MISSING else "written" if not todo and not subs else "partly written"))
        if filled and (todo or subs):
            out.append("    written so far: " + "; ".join(filled))
        if todo and lam.ret is not MISSING:
            out.append("    still missing: " + ", ".join(todo))
        if subs:
            out.append("    sub-tasks: " + "; ".join(subs))
        return "\n".join(out)

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
        return session.apply(name, args or {})


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
