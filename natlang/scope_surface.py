"""Persistent typed-scope model surface (``scope-eval-v1``)."""
from __future__ import annotations

import copy

from .nodes import MISSING
from .render import listing, pending_lines
from .surface import ToolSurface


def _tool(name, description, properties, required):
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required,
                           "additionalProperties": False}}}


class ScopeEvalSurface(ToolSurface):
    """Small stable interface: code operates on variables, tools close the task."""

    name = "scope-eval-v1"

    def __init__(self, *, state_view: bool = False):
        super().__init__(marks=True, error_tool=True, state_view=state_view)
        self.done_arg = False

    def marking(self, session) -> bool:
        return True

    def pending(self, session) -> list:
        return pending_lines(session.lam.original_body or session.lam.body, session.lam.marks)

    def tools(self, session) -> list:
        # Once the result is valid and every instruction line is closed there is
        # deliberately nothing left to call.  The following model response is the
        # natural end-of-turn signal.  This also prevents a small model from
        # repeatedly "returning" an already complete result and eventually
        # damaging an otherwise valid episode.
        if not self.missing(session) and not self.pending(session):
            return []
        tools = [
            _tool("eval", "Execute one TypeScript-like step in the persistent typed scope. Declarations persist; imported functions are called with await and positional values.",
                  {"code": {"type": "string"}}, ["code"]),
            _tool("read_value", "Inspect a variable or a field/index selection without executing code.",
                  {"expression": {"type": "string"}, "start": {"type": "integer"},
                   "end": {"type": "integer"}}, ["expression"]),
            _tool("write_value", "Transport an already supplied literal into a top-level scope variable. For normal program work, including literal decisions, prefer eval declarations. as_type is needed only when inference is ambiguous.",
                  {"name": {"type": "string", "pattern": "^[A-Za-z_$][A-Za-z0-9_$]*$"},
                   "value": {}, "as_type": {"type": "string"}}, ["name", "value"]),
            _tool("return_value", "Stage one existing variable as this function's typed result. End the turn naturally after all instruction lines are closed.",
                  {"variable": {"type": "string", "pattern": "^[A-Za-z_$][A-Za-z0-9_$]*$"}}, ["variable"]),
            _tool("mark_lines", "Close one instruction line or one inclusive contiguous range after its work succeeded. Use skipped only for an untaken branch.",
                  {"start": {"type": "integer"}, "end": {"type": "integer"},
                   "skipped": {"type": "boolean"}}, ["start"]),
            _tool("report_blocker", "End without a result because required information is missing. Do not guess.",
                  {"missing": {"type": "string"}}, ["missing"]),
            _tool("report_error", "End without a result because the instructions require an invalid or contradictory operation.",
                  {"message": {"type": "string"}}, ["message"]),
        ]
        return tools

    def render_request(self, session) -> str:
        lam = session.lam
        imports = []
        for name, fn in lam.codebase.items():
            imports.append(f"  {name}(" + ", ".join(
                f"{raw.rstrip('?')}: {typ}" for raw, typ in fn.args.items()) + f"): {fn.returns}")
        from .types import format_type
        inputs = [f"  {name}: {format_type(typ)} = {self._preview(lam.in_.get(name, MISSING))}"
                  for name, typ, _ in lam.type.params.fields]
        locals_ = [f"  {name}: {format_type(lam.let_types[name])} = {self._preview(value)}"
                   for name, value in lam.let.items() if name in lam.let_types]
        body = listing(lam.original_body or lam.body, lam.marks, compact=True, window=5)
        return "\n".join(["Execute the natural-language function line by line.", "", "Program:", body,
                           "", "Scope:", " inputs (immutable)", *(inputs or ["  (none)"]),
                           " imports (immutable live bindings)", *(imports or ["  (none)"]),
                           " locals", *(locals_ or ["  (none)"]),
                           f" result: {format_type(lam.type.returns)} — " +
                           ("not staged" if lam.ret is MISSING else "staged")])

    def opening_read(self, session):
        return None

    @staticmethod
    def _preview(value):
        if value is MISSING:
            return "missing"
        if isinstance(value, list):
            return f"{len(value)} items"
        if isinstance(value, dict):
            return "{ " + ", ".join(list(value)[:5]) + (" …" if len(value) > 5 else "") + " }"
        text = repr(value)
        return text if len(text) <= 100 else text[:97] + "…"

    def apply(self, session, name, args):
        session.surface_name = self.name
        return session.apply(name, args)

    def missing(self, session) -> str:
        if session.lam.ret is MISSING:
            return "No result is staged. Bind the correct value, call return_value, close every program line, then end the turn."
        open_ = self.pending(session)
        if open_:
            return "Lines still open: " + ", ".join(map(str, open_))
        from .values import problems
        holes, pending = problems(session.lam.ret, session.lam.type.returns, session.env, "return")
        if not holes and not pending:
            return ""
        return "The staged result is invalid or still contains pending work."
