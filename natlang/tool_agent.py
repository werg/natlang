"""An agent that works through native tool calls (surface tools-v2).

The loop is the ordinary one: request -> tool calls -> results -> ... -> reply. The reply ends
the episode. The result is what was written to `return`; the reply is kept as a note only.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .decoder import Decoder
from .surface import ToolSurface

TOOLS_PROMPT = (Path(__file__).parent / "prompts" / "tools_small.md").read_text()
RESAMPLES, MAX_NUDGES = 2, 2
FAILED = ("rejected", "refused", "error", "budget")


class ToolAgent:
    def __init__(self, decoder: Decoder, *, surface: Optional[ToolSurface] = None, temperature: float = 0.2,
                 system_prompt: str = TOOLS_PROMPT, log: Optional[list] = None, transcript: Optional[list] = None):
        self.dec, self.surface = decoder, surface or ToolSurface()
        self.temperature, self.system = temperature, system_prompt
        self.log = log if log is not None else []
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
        nudges = 0
        try:
            while True:
                turn = first = None
                for attempt in range(RESAMPLES + 1):
                    turn = self.dec.chat(messages, s.tools(session),
                                         temperature=self.temperature if attempt == 0 else 0.7, seed=attempt)
                    if not turn.calls:
                        break
                    name, args = turn.calls[0]
                    first = s.apply(session, name, args)
                    self.log.append({"action": f"{name} {json.dumps(args)}", "kind": first.kind, "attempt": attempt})
                    if first.kind in ("rejected", "error") and attempt < RESAMPLES:
                        session.actions -= 1      # a discarded sample does not spend the budget
                        continue                  # and is never shown to the model
                    break

                if not turn.calls:                # the reply: the normal end of an agent episode
                    open_ = s.pending(session) if hasattr(s, "pending") else []
                    if open_ and nudges < MAX_NUDGES:             # no data in a nudge: line numbers only
                        nudges += 1
                        messages += [{"role": "assistant", "content": turn.text or "(no reply)"},
                                     {"role": "user", "content": "Lines still marked [ ]: " + ", ".join(map(str, open_)) +
                                                                 ". Finish them, or mark them done or skipped."}]
                        continue
                    if session.finish():
                        session.lam.note = turn.text
                        return None
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
        finally:
            if self.transcript is not None:
                self.transcript[:] = messages


def _raw(calls):
    return [{"id": "", "type": "function", "function": {"name": n, "arguments": json.dumps(a)}} for n, a in calls]
