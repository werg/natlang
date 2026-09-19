"""The reference policy: canonical trajectories in the tools-v2 format.

It follows a Plan (it never reads the natural-language instructions), performs every call through the
real harness, and records the episode in exactly the message structure the ToolAgent uses. Each
assistant turn is also rendered in the native call text and checked against the turn's grammar, so the
data is decodable under the same constraints used at inference.
"""
from __future__ import annotations

import json
import random
from typing import Optional

from .. import gbnf
from ..native import CALL_OPEN, call_grammar
from ..nodes import MISSING
from ..render import INLINE, PREVIEW_ITEMS
from ..surface import ToolSurface, is_previewed
from ..tool_agent import TOOLS_PROMPT
from ..types import format_type


def native_text(calls) -> str:
    return CALL_OPEN + "[" + ", ".join(
        f"{n}({', '.join(f'{k}={v!r}' for k, v in a.items())})" for n, a in calls) + "]"


class ReferenceAgent:
    def __init__(self, plan, sink: list, *, surface: Optional[ToolSurface] = None, check_grammar: bool = True,
                 recovery_rng=None, recovery_rate: float = 0):
        self.plan, self.sink, self.s, self.check = plan, sink, surface or ToolSurface(), check_grammar
        self.recovery_rng = recovery_rng or random.Random(0)
        self.recovery_rate, self.recovered = recovery_rate, False

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
        messages = [{"role": "system", "content": TOOLS_PROMPT},
                    {"role": "user", "content": s.render_request(session)}]
        opening = s.opening_read(session)
        if opening:
            name, args, text = opening
            call = {"id": "call_0", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
            messages += [{"role": "assistant", "content": "", "tool_calls": [call]},
                         {"role": "tool", "tool_call_id": "call_0", "content": text}]
        gen, sent = self.turns(session), None
        while True:
            try:
                calls = gen.send(sent)
            except StopIteration:
                break
            tools = s.tools(session)
            if not self.recovered and any(n == "run_code" for n, _ in calls) and self.recovery_rng.random() < self.recovery_rate:
                # A failed model action is history, never a supervised target. It has no
                # effects or tree mutation; the next target is the verified correction.
                bad = self.recovery_rng.choice(["locals.__missing_value.length", "JSON.parse('{')"])
                args = {"code": bad}
                assert gbnf.accepts(call_grammar(tools), native_text([("run_code", args)]))
                result = s.apply(session, "run_code", args)
                assert result.kind == "error", result.text
                call_id = f"recovery_{len(messages)}"
                messages += [{"role": "assistant", "content": "", "tool_calls": [{"id": call_id,
                              "type": "function", "function": {"name": "run_code", "arguments": json.dumps(args)}}]},
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
            sent = results[-1] if self.plan.kind == "script" else results[-1].value
            if results[-1].kind == "blocked":
                return results[-1].text
            if results[-1].kind == "quiesced" and self.plan.kind != "script":
                raise AssertionError(f"a reference call did not finish: {calls} -> {results[-1].text}")
        if session.lam.ret is MISSING or not session.finish():
            raise AssertionError("reference policy ended without a valid `return`")
        self._emit(session, messages, s.tools(session), reply=self.plan.note or "Done.")
        return None

    def _emit(self, session, messages, tools, calls=None, reply=None):
        target = ({"role": "assistant", "content": reply} if calls is None else
                  {"role": "assistant", "content": "", "tool_calls": [
                      {"type": "function", "function": {"name": n, "arguments": json.dumps(a)}} for n, a in calls]})
        native = reply if calls is None else native_text(calls)
        if self.check and not gbnf.accepts(call_grammar(tools), native):
            raise AssertionError(f"the turn's grammar refuses the reference turn:\n{native}")
        self.sink.append({"messages": [dict(m) for m in messages], "tools": tools, "target": target,
                          "native_target": native, "kind": self.plan.kind, "recovery": self.recovered, "template": bool(self.plan.template),
                          "skill": "reply" if calls is None else "+".join(n for n, _ in calls)})
