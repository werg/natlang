"""An agent that works through native tool calls (surface tools-v2).

The loop is the ordinary one: request -> tool calls -> results -> ... -> reply. The reply ends
the episode. The result is what was written to `return`; the reply is kept as a note only.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from .decoder import Decoder
from .surface import ToolSurface

TOOLS_PROMPT = (Path(__file__).parent / "prompts" / "tools_small.md").read_text()
MAX_NUDGES = 2
FAILED = ("rejected", "refused", "error", "budget")


class ToolAgent:
    def __init__(self, decoder: Decoder, *, surface: Optional[ToolSurface] = None, temperature: float = 0.2,
                 system_prompt: str = TOOLS_PROMPT, log: Optional[list] = None, transcript: Optional[list] = None,
                 max_turns: int = 64, max_tokens: int = 4000, max_seconds: float = 900,
                 validation_feedback: str = "local"):
        if validation_feedback not in ("local", "caller"):
            raise ValueError("validation_feedback must be local or caller")
        self.validation_feedback = validation_feedback
        self.dec, self.surface = decoder, surface or ToolSurface()
        self.temperature, self.system = temperature, system_prompt
        self.log = log if log is not None else []
        self.max_turns, self.max_tokens, self.max_seconds = max_turns, max_tokens, max_seconds
        self.transcript = transcript          # if given, receives the final message list (for debugging)

    def run(self, session) -> Optional[str]:
        s = self.surface
        messages = [{"role": "system", "content": self.system},
                    {"role": "user", "content": s.render_request(session)}]
        opening = s.opening_read(session)
        if opening:                               # data reaches the model only through the tool channel
            name, args, text = opening
            call = {"id": "call_0", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
            messages += [{"role": "assistant", "content": "", "tool_calls": [call]},
                         {"role": "tool", "tool_call_id": "call_0", "content": text}]
        nudges = turns = tokens = 0
        previous_deadline = getattr(self.dec, "deadline", None)
        deadline = time.monotonic() + self.max_seconds
        if previous_deadline is not None:
            deadline = min(deadline, previous_deadline)
        previous_runtime_deadline = session.rt.deadline
        if previous_runtime_deadline is not None:
            deadline = min(deadline, previous_runtime_deadline)
        self.dec.deadline = session.rt.deadline = deadline
        try:
            while True:
                if turns >= self.max_turns or tokens >= self.max_tokens or time.monotonic() >= deadline:
                    return "episode turn, token, or wall-clock budget exhausted"
                allowance = min(700, self.max_tokens - tokens)
                turn = self.dec.chat(messages, s.tools(session), temperature=self.temperature,
                                     seed=0, max_tokens=allowance)
                turns += 1
                # A backend without usage is charged its entire requested allowance.
                used = getattr(turn, "completion_tokens", None)
                tokens += allowance if used is None else max(1, used)
                if time.monotonic() >= deadline or tokens > self.max_tokens:
                    return "episode token or wall-clock budget exhausted"
                first = None
                if turn.calls:
                    name, args = turn.calls[0]
                    first = s.apply(session, name, args)
                    self.log.append({"action": f"{name} {json.dumps(args)}", "kind": first.kind, "attempt": 0})
                    # Even a failed operation may have performed effects. Feed its result back;
                    # never silently replay it or refund the work/turn budget.

                if not turn.calls:                # the reply: the normal end of an agent episode
                    if self.validation_feedback == "caller" and (missing := s.missing(session)):
                        return "validation failed: " + missing
                    open_ = s.pending(session) if hasattr(s, "pending") else []
                    if open_ and nudges < MAX_NUDGES:             # no data in a nudge: line numbers only
                        nudges += 1
                        messages += [{"role": "assistant", "content": turn.text or "(no reply)"},
                                     {"role": "user", "content": "Lines still marked [ ]: " + ", ".join(map(str, open_)) +
                                                                 ". Mark completed work done and untaken work skipped (skipped=true). "
                                                                 "A done range marks EVERY line between its endpoints; do not include untaken work. "
                                                                 "Carry out any applicable unfinished work before marking it."}]
                        continue
                    if session.finish():
                        session.lam.note = turn.text
                        return None
                    if self.validation_feedback == "caller":
                        return "validation failed: " + s.missing(session)
                    nudges += 1
                    if nudges > MAX_NUDGES:
                        return "replied without writing `return`: " + turn.text[:280]
                    messages += [{"role": "assistant", "content": turn.text or "(no reply)"},
                                 {"role": "user", "content": s.missing(session)}]
                    continue

                results = [first]
                for name, args in turn.calls[1:]:     # several calls in one turn is the model's native habit
                    if results[-1].kind in FAILED or results[-1].kind == "blocked":
                        break
                    if time.monotonic() >= deadline:
                        return "episode wall-clock budget exhausted"
                    r = s.apply(session, name, args)
                    self.log.append({"action": f"{name} {json.dumps(args)}", "kind": r.kind, "attempt": 0})
                    results.append(r)
                if results[-1].kind == "budget":
                    return "budget exhausted"
                if results[-1].kind == "blocked":         # ends the episode; the lambda quiesces with this note
                    return results[-1].text
                raw = [c if c.get("id") else {**c, "id": f"call_{len(messages)}_{i}"}
                       for i, c in enumerate(turn.raw_calls or _raw(turn.calls))][: len(results)]
                messages.append({"role": "assistant", "content": "", "tool_calls": raw})
                for c, r in zip(raw, results):
                    messages.append({"role": "tool", "tool_call_id": c["id"], "content": r.text})
                if self.validation_feedback == "caller" and results[-1].kind in ("rejected", "refused"):
                    return "validation failed: " + results[-1].text
        except TimeoutError:
            return "episode wall-clock budget exhausted"
        finally:
            self.dec.deadline = previous_deadline
            session.rt.deadline = previous_runtime_deadline
            if self.transcript is not None:
                self.transcript[:] = messages


def _raw(calls):
    return [{"id": "", "type": "function", "function": {"name": n, "arguments": json.dumps(a)}} for n, a in calls]
