"""The reference policy: canonical trajectories in the tools-v2 format.

It follows a Plan (it never reads the natural-language instructions), performs every call through the
real harness, and records the episode in exactly the message structure the ToolAgent uses. Each
assistant turn is also rendered in the native call text and checked against the turn's grammar, so the
data is decodable under the same constraints used at inference.
"""
from __future__ import annotations

import json
import random
from decimal import Decimal
from typing import Optional

from .. import gbnf
from ..native import CALL_OPEN, call_grammar
from ..nodes import MISSING
from ..render import INLINE, PREVIEW_ITEMS
from ..surface import ToolSurface, is_previewed
from ..surface_projection import project_actions_v4
from ..tool_agent import CHECKPOINT_REQUEST, TOOLS_PROMPT
from ..types import format_type
from ..types import parse_type, TypeSyntaxError
from ..diag import Reject


def _native_repr(value):
    # The native-call grammar accepts decimal literals, not Python's exponent
    # notation. Preserve the round-trippable decimal digits of the float.
    if isinstance(value, float):
        return format(Decimal(str(value)), "f")
    if isinstance(value, list):
        return "[" + ", ".join(_native_repr(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{_native_repr(k)}: {_native_repr(v)}" for k, v in value.items()) + "}"
    return repr(value)


def native_text(calls) -> str:
    return CALL_OPEN + "[" + ", ".join(
        f"{n}({', '.join(f'{k}={_native_repr(v)}' for k, v in a.items())})" for n, a in calls) + "]"


class ReferenceAgent:
    def __init__(self, plan, sink: list, *, surface: Optional[ToolSurface] = None, check_grammar: bool = True,
                 recovery_rng=None, recovery_rate: float = 0, system_prompt: str = TOOLS_PROMPT,
                 segment_turns: Optional[int] = 6, segment_messages: Optional[int] = 12):
        self.system_prompt = system_prompt
        self.plan, self.sink, self.s, self.check = plan, sink, surface or ToolSurface(), check_grammar
        self.recovery_rng = recovery_rng or random.Random(0)
        self.recovery_rate, self.recovered = recovery_rate, False
        self.segment_turns = segment_turns
        self.segment_messages = segment_messages

    def _opening_messages(self, session):
        messages = [{"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": self.s.render_request(session)}]
        opening = self.s.opening_read(session)
        if opening:
            name, args, text = opening
            call = {"id": "call_0", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
            messages += [{"role": "assistant", "content": "", "tool_calls": [call]},
                         {"role": "tool", "tool_call_id": "call_0", "content": text}]
        return messages

    def recovery_action(self, session, calls):
        """Choose a provably rejected destination or a side-effect-free code error."""
        for name, args in calls:
            if (name == 'call' and args.get('to', '').startswith('let/') and session.lam.ret is MISSING
                    and not any(k in args for k in ('over', 'init', 'until', 'max'))):
                fn = session._function(args['function'])
                try:
                    # The same pre-placement check the runtime uses. No call is run here.
                    session._check_fit(parse_type(fn.type_text), session.resolve('return')[1])
                except Reject as error:
                    if any(d.code == 'type-does-not-fit-slot' for d in error.diags):
                        return 'call', {k: v for k, v in dict(args, to='return').items() if k != 'done'}, 'rejected'
                except TypeSyntaxError:
                    pass  # Named types available only inside this callee: do not inject.
        if any(n == 'run_code' for n, _ in calls):
            bad = self.recovery_rng.choice(['locals.__missing_value.length', "JSON.parse('{')"])
            return 'run_code', {'code': bad}, 'error'
        return None

    # -- what a good interpreter does for each kind of plan
    def turns(self, session):
        lam, p = session.lam, self.plan
        rtype = format_type(lam.type.returns)
        hidden = [f"args/{n}" for n, v in lam.in_.items() if is_previewed(v)]
        if p.kind == "blocked":                              # the inputs do not determine the result
            if hidden:
                yield [("read", {"path": h}) for h in hidden]
            yield [("report_blocker", {"missing": p.note})]
        elif p.kind == "leaf":
            if hidden:                                       # read what the listing only previews
                yield [("read", {"path": h}) for h in hidden]
            yield [("write", {"path": "return", "type": rtype, "value": p.gold(lam.in_)})]
        elif p.kind == "crisp":
            result = yield [("run_code", {"code": p.code})]
            yield [("write", {"path": "return", "type": rtype, "value": result})]
        # A turn's grammar is built from the state BEFORE the turn, so a call that depends on what an
        # earlier call created goes into the next turn.
        elif p.kind == "script":                             # a hand-written interpreter for a hand-written code base
            gen, sent = p.script(lam), None
            while True:
                try:
                    turn = gen.send(sent)
                except StopIteration:
                    break
                if turn and turn[0][0] == "glue":            # exact work: compute, then keep the result
                    _, code, path, ty = turn[0]
                    r = yield [("run_code", {"code": code})]
                    sent = yield [("write", {"path": path, "type": ty, "value": r.value})]
                else:
                    sent = yield turn
        elif p.kind == "calls":
            for step in (p.steps(lam.in_) if callable(p.steps) else p.steps):
                if step[0] == "glue":                        # exact work no function covers: compute, then keep it
                    _, code, path, ty = step
                    result = yield [("run_code", {"code": code})]
                    yield [("write", {"path": path, "type": ty, "value": result})]
                else:
                    yield [(step[0], step[1])]
        else:
            raise ValueError(p.kind)

    def run(self, session) -> Optional[str]:
        s = self.s
        messages = self._opening_messages(session)
        segment_turns = 0
        gen, sent = self.turns(session), None
        while True:
            try:
                calls = gen.send(sent)
            except StopIteration:
                break
            tools = s.tools(session)
            planned_calls = calls
            if s.name == "tools-v4":
                calls = [projected for name, args in calls
                         for projected in project_actions_v4(name, args, offered_tools=tools)]
            recovery = self.recovery_action(session, planned_calls) if not self.recovered and self.recovery_rate else None
            if recovery and s.name == "tools-v4":
                projected = project_actions_v4(recovery[0], recovery[1], offered_tools=tools)
                recovery = (*projected[-1], recovery[2]) if len(projected) == 1 else None
            if (recovery and self.recovery_rng.random() < self.recovery_rate
                    and gbnf.accepts(call_grammar(tools), native_text([(recovery[0], recovery[1])]))):
                # A failed model action is history, never a supervised target. It has no
                # effects or tree mutation; the next target is the verified correction.
                name, args, expected_kind = recovery
                result = s.apply(session, name, args)
                assert result.kind == expected_kind, result.text
                call_id = f"recovery_{len(messages)}"
                messages += [{"role": "assistant", "content": "", "tool_calls": [{"id": call_id,
                              "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]},
                             {"role": "tool", "tool_call_id": call_id, "content": result.text}]
                self.recovered = True
            self._emit(session, messages, tools, calls=calls)
            raw, results = [], []
            for i, (name, args) in enumerate(calls):
                r = s.apply(session, name, args)
                if r.kind in ("rejected", "refused", "error", "budget"):
                    raise AssertionError(f"reference call failed: {name} {args}\n{r.text}")
                raw.append({"id": f"call_{len(messages)}_{i}", "type": "function",
                            "function": {"name": name, "arguments": json.dumps(args)}})
                results.append(r)
            messages.append({"role": "assistant", "content": "", "tool_calls": raw})
            for c, r in zip(raw, results):
                messages.append({"role": "tool", "tool_call_id": c["id"], "content": r.text})
            segment_turns += 1
            if (((self.segment_turns is not None and segment_turns >= self.segment_turns) or
                 (self.segment_messages is not None and len(messages) >= self.segment_messages))
                    and calls[-1][0] in ("write", "call", "edit", "mark_done", "write_value", "copy_value",
                                         "copy_function", "run_function", "for_each", "fold", "repeat",
                                         "edit_text", "mark_lines")
                    and (s.missing(session) or s.pending(session))):
                note = ("No hidden decision remains. Continue from the typed workspace and "
                        "complete the remaining open instruction lines in order.")
                checkpoint_messages = [*messages, {"role": "user", "content": CHECKPOINT_REQUEST}]
                self.sink.append({"messages": [dict(m) for m in checkpoint_messages], "tools": [],
                                  "target": {"role": "assistant", "content": note},
                                  "native_target": "", "kind": self.plan.kind,
                                  "recovery": self.recovered, "template": bool(self.plan.template),
                                  "skill": "checkpoint"})
                session.lam.continuation_note = note
                messages = self._opening_messages(session)
                segment_turns = 0
            work_results = [result for (name, _), result in zip(calls, results) if name != "mark_lines"]
            last = work_results[-1] if work_results else results[-1]
            sent = last if self.plan.kind == "script" else last.value
            if last.kind == "blocked":
                return last.text
            if last.kind == "quiesced" and self.plan.kind != "script":
                raise AssertionError(f"a reference call did not finish: {calls} -> {last.text}")
        open_lines = s.pending(session)
        if session.lam.ret is MISSING or open_lines or not session.finish():
            raise AssertionError(f"reference policy ended without a valid return and closed lines: {open_lines}; "
                                 f"body={session.lam.body!r}; marks={session.lam.marks!r}")
        self._emit(session, messages, s.tools(session), reply="")
        return None

    def _emit(self, session, messages, tools, calls=None, reply=None):
        target = ({"role": "assistant", "content": reply} if calls is None else
                  {"role": "assistant", "content": "", "tool_calls": [
                      {"type": "function", "function": {"name": n, "arguments": json.dumps(a)}} for n, a in calls]})
        native = "" if calls is None else native_text(calls)
        if self.check and not gbnf.accepts(call_grammar(tools), native):
            raise AssertionError(f"the turn's grammar refuses the reference turn:\n{native}")
        self.sink.append({"messages": [dict(m) for m in messages], "tools": tools, "target": target,
                          "native_target": native, "kind": self.plan.kind, "recovery": self.recovered, "template": bool(self.plan.template),
                          "skill": "reply" if calls is None else "+".join(n for n, _ in calls)})
