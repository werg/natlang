"""An agent that works through native tool calls (surface tools-v2).

The loop is the ordinary one: request -> tool calls -> results -> ... -> reply. The reply ends
the episode. The result is what was written to `return`; the reply is kept as a note only.
"""
from __future__ import annotations

import copy
import json
import time
from contextlib import nullcontext
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
                 max_turns: Optional[int] = None, max_tokens: Optional[int] = None,
                 max_seconds: Optional[float] = None,
                 turn_tokens: Optional[int] = None,
                 validation_feedback: str = "caller", careful_threshold: Optional[float] = None,
                 proposals: Optional[list] = None, reviews: Optional[list] = None, review_order: str = "reason_first",
                 review_scope: str = "values", withdrawal_policy: str = "caller",
                 review_prompt: str = "baseline", teacher_turns: Optional[list] = None):
        if review_prompt not in ("baseline", "repeat_instructions", "checklist"):
            raise ValueError("unknown review prompt")
        self.review_prompt = review_prompt
        if review_scope not in ("values", "actions"):
            raise ValueError("review_scope must be values or actions")
        if withdrawal_policy not in ("caller", "retry"):
            raise ValueError("withdrawal_policy must be caller or retry")
        self.review_scope, self.withdrawal_policy = review_scope, withdrawal_policy
        if review_order not in ("reason_first", "decision_first"):
            raise ValueError("review_order must be reason_first or decision_first")
        self.review_order = review_order
        if validation_feedback not in ("local", "caller"):
            raise ValueError("validation_feedback must be local or caller")
        if careful_threshold is not None and not 0 <= careful_threshold <= 1:
            raise ValueError("careful_threshold must be between zero and one")
        self.careful_threshold = careful_threshold
        self.proposals = proposals if proposals is not None else []
        self.reviews = reviews if reviews is not None else []
        self.validation_feedback = validation_feedback
        self.dec, self.surface = decoder, surface or ToolSurface()
        self.temperature, self.system = temperature, system_prompt
        self.log = log if log is not None else []
        self.max_turns, self.max_tokens, self.max_seconds = max_turns, max_tokens, max_seconds
        if turn_tokens is not None and turn_tokens < 1:
            raise ValueError("turn_tokens must be positive")
        self.turn_tokens = turn_tokens
        self.transcript = transcript          # if given, receives the final message list (for debugging)
        self.teacher_turns = teacher_turns    # exact model replies and their pre-action context, for audit/distillation

    def run(self, session) -> Optional[str]:
        s = self.surface
        settings = session.rt.options.model
        max_turns = settings.max_turns if settings else self.max_turns
        max_tokens = settings.max_tokens if settings else self.max_tokens
        max_seconds = settings.max_seconds if settings else self.max_seconds
        turn_tokens = settings.turn_tokens if settings else self.turn_tokens
        temperature = settings.temperature if settings else self.temperature
        system = self.system
        if session.rt.engine_selection:
            system += "\nFor run_code, always name an engine offered in its current tool schema."
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": s.render_request(session)}]
        opening = s.opening_read(session)
        if opening:                               # data reaches the model only through the tool channel
            name, args, text = opening
            call = {"id": "call_0", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
            messages += [{"role": "assistant", "content": "", "tool_calls": [call]},
                         {"role": "tool", "tool_call_id": "call_0", "content": text}]
        nudges = turns = tokens = withdrawals = 0
        previous_deadline = getattr(self.dec, "deadline", None)
        deadline = time.monotonic() + max_seconds if max_seconds is not None else None
        if previous_deadline is not None:
            deadline = previous_deadline if deadline is None else min(deadline, previous_deadline)
        previous_runtime_deadline = session.rt.deadline
        if previous_runtime_deadline is not None:
            deadline = previous_runtime_deadline if deadline is None else min(deadline, previous_runtime_deadline)
        # New backends keep request deadlines in a per-call context. Older test
        # drivers still support the compatibility mutable attribute.
        request_scope = getattr(self.dec, "request_scope", None)
        scope = request_scope(deadline=deadline) if request_scope else nullcontext()
        if not request_scope:
            self.dec.deadline = deadline
        session.rt.deadline = deadline
        try:
            with scope:
                return self._run_turns(session, s, messages, deadline, nudges, turns, tokens, withdrawals,
                                       max_turns, max_tokens, turn_tokens, temperature)
        finally:
            if not request_scope:
                self.dec.deadline = previous_deadline
            session.rt.deadline = previous_runtime_deadline
            if self.transcript is not None:
                self.transcript[:] = messages

    def _run_turns(self, session, s, messages, deadline, nudges, turns, tokens, withdrawals,
                   max_turns, max_tokens, turn_tokens, temperature):
        def timed_out() -> bool:
            return deadline is not None and time.monotonic() >= deadline

        def exhausted() -> bool:
            return ((max_turns is not None and turns >= max_turns) or
                    (max_tokens is not None and tokens >= max_tokens) or
                    timed_out())

        def allowance() -> Optional[int]:
            remaining = None if max_tokens is None else max_tokens - tokens
            if turn_tokens is None:
                return remaining
            return turn_tokens if remaining is None else min(turn_tokens, remaining)

        try:
            while True:
                if exhausted():
                    return "episode turn, token, or wall-clock budget exhausted"
                limit = allowance()
                available_tools = s.tools(session)
                presented_tools = getattr(self.dec, "presented_tools", None)
                offered_tools = (copy.deepcopy(presented_tools(available_tools) if presented_tools else available_tools)
                                 if self.teacher_turns is not None else None)
                invocation = getattr(session, "invocation", None)
                call_path = invocation.path if invocation else ""
                attempt = invocation.attempt if invocation else 1
                policy = session.rt.options.seed
                request_started = time.monotonic()
                measured_tools = offered_tools if offered_tools is not None else (
                    presented_tools(available_tools) if presented_tools else available_tools)
                session.rt._observe("model_request", call_id=getattr(invocation, "call_id", None),
                                    phase="start", turn=turns + 1,
                                    tool_schema_bytes=len(json.dumps(measured_tools, ensure_ascii=False)),
                                    messages=len(messages))
                try:
                    turn = self.dec.chat(messages, available_tools, temperature=temperature,
                                         seed=policy.seed(call_path, attempt, "model-turn", turns), max_tokens=limit)
                except Exception as exc:
                    session.rt._observe("model_request", call_id=getattr(invocation, "call_id", None),
                                        phase="error", turn=turns + 1,
                                        duration_ms=round((time.monotonic() - request_started) * 1000),
                                        error=f"{type(exc).__name__}: {exc}")
                    raise
                session.rt._observe("model_request", call_id=getattr(invocation, "call_id", None),
                                    phase="end", turn=turns + 1,
                                    duration_ms=round((time.monotonic() - request_started) * 1000),
                                    prompt_tokens=getattr(turn, "prompt_tokens", None),
                                    completion_tokens=getattr(turn, "completion_tokens", None))
                turns += 1
                session.rt._observe("proposal", call_id=getattr(invocation, "call_id", None),
                                    phase="generated", turn=turns, calls=turn.calls, text=turn.text)
                teacher_turn = None
                if self.teacher_turns is not None:
                    presented_messages = getattr(self.dec, "presented_messages", None)
                    teacher_turn = {"function": session.lam.fn_name,
                                    "call_id": getattr(invocation, "call_id", None),
                                    "messages_before": copy.deepcopy(
                                        presented_messages(messages) if presented_messages else messages),
                                    "tools_offered": offered_tools,
                                    "response": turn.raw_response,
                                    "calls": turn.calls, "text": turn.text,
                                    "value_confidence": turn.value_confidence,
                                    "reviews": [], "executions": []}
                    self.teacher_turns.append(teacher_turn)
                # A backend without usage is charged its entire requested allowance.
                used = getattr(turn, "completion_tokens", None)
                tokens += (limit or 0) if used is None else max(1, used)
                if timed_out() or (max_tokens is not None and tokens > max_tokens):
                    return "episode token or wall-clock budget exhausted"
                if turn.calls:
                    proposal = {"calls": turn.calls, "value_confidence": turn.value_confidence,
                                "released": False, "messages": list(messages)}
                    self.proposals.append(proposal)
                    # All reviews precede every operation in this proposed batch.
                    # The review uses a fork of the pre-action messages, never the
                    # main history. Approval commits the original proposal only.
                    withdrawn = False
                    for index, (name, args) in enumerate(turn.calls):
                        confidence = turn.value_confidence[index] if index < len(turn.value_confidence) else None
                        low_value = (self.careful_threshold is not None and confidence is not None and
                                     confidence["geometric_mean"] < self.careful_threshold)
                        structural = self.review_scope == "actions" and (
                            name in ("call", "mark_done", "edit") or "done" in args or "source" in args)
                        if not (low_value or structural):
                            continue
                        if exhausted():
                            return "careful review budget exhausted before applying proposal"
                        limit = allowance()
                        name, args = turn.calls[index]
                        fork = review_messages(messages, turn.calls, index, self.review_prompt)
                        answer = getattr(self.dec, "review", self.dec.chat)(
                            fork, review_tools(self.review_order), temperature=0,
                            seed=policy.seed(call_path, attempt, "review", turns), max_tokens=limit)
                        turns += 1
                        used = getattr(answer, "completion_tokens", None)
                        tokens += (limit or 0) if used is None else max(1, used)
                        review = {"proposal": len(self.proposals) - 1, "call_index": index,
                                  "confidence": confidence, "messages": fork, "calls": answer.calls,
                                  "text": answer.text, "order": self.review_order,
                                  "trigger": "structural" if structural else "confidence", "prompt_variant": self.review_prompt}
                        self.reviews.append(review)
                        if timed_out() or (max_tokens is not None and tokens > max_tokens):
                            return "careful review budget exhausted before applying proposal"
                        if len(answer.calls) != 1 or answer.calls[0][0] != "review_write":
                            return "careful review invalid response; proposal not applied"
                        verdict = answer.calls[0][1]
                        decision = verdict.get("decision")
                        if decision not in ("approve", "withdraw", "error", "blocker") or not isinstance(verdict.get("reason"), str):
                            return "careful review invalid verdict; proposal not applied"
                        review["decision"] = decision
                        if teacher_turn is not None:
                            teacher_turn["reviews"].append({"call_index": index,
                                                            "response": answer.raw_response,
                                                            "decision": decision,
                                                            "reason": verdict["reason"]})
                        self.log.append({"action": "review_write " + json.dumps(verdict),
                                         "kind": "ok" if decision == "approve" else "blocked", "attempt": 0})
                        if decision == "withdraw" and self.withdrawal_policy == "retry" and withdrawals < 1:
                            withdrawals += 1
                            withdrawn = True
                            proposal["withdrawn"] = True
                            session.rt._observe("proposal", call_id=getattr(invocation, "call_id", None),
                                                phase="withdrawn", turn=turns, calls=turn.calls)
                            # No reviewer reasoning or replacement value enters the main history.
                            messages = [*messages, {"role": "user", "content":
                                "The pending batch was withdrawn before execution. No action in it happened. "
                                "Reconsider the original instructions from the unchanged workspace. "
                                "Do not change requirements to obtain a result. This is the only reconsideration."}]
                            break
                        if decision != "approve":
                            return "careful review " + decision + ": " + verdict["reason"]
                    if withdrawn:
                        continue
                    proposal["released"] = True
                    session.rt._observe("proposal", call_id=getattr(invocation, "call_id", None),
                                        phase="released", turn=turns, calls=turn.calls)
                first = None
                if turn.calls:
                    name, args = turn.calls[0]
                    first = s.apply(session, name, args)
                    self.log.append({"action": f"{name} {json.dumps(args)}", "kind": first.kind, "attempt": 0})
                    if teacher_turn is not None:
                        teacher_turn["executions"].append({"call_index": 0, "name": name,
                                                            "args": args, "kind": first.kind,
                                                            "text": first.text})
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
                    if open_:
                        return "validation failed: unfinished lines: " + ", ".join(map(str, open_))
                    if session.finish():
                        session.lam.note = turn.text
                        messages.append({"role": "assistant", "content": turn.text})
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
                    if results[-1].kind in FAILED or results[-1].kind in ("blocked", "completed"):
                        break
                    if timed_out():
                        return "episode wall-clock budget exhausted"
                    r = s.apply(session, name, args)
                    self.log.append({"action": f"{name} {json.dumps(args)}", "kind": r.kind, "attempt": 0})
                    if teacher_turn is not None:
                        teacher_turn["executions"].append({"call_index": len(results), "name": name,
                                                            "args": args, "kind": r.kind,
                                                            "text": r.text})
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
                if results[-1].kind == "completed":
                    return None
                if self.validation_feedback == "caller" and results[-1].kind in ("rejected", "refused"):
                    return "validation failed: " + results[-1].text
        except TimeoutError:
            return "episode wall-clock budget exhausted"


def _raw(calls):
    return [{"id": "", "type": "function", "function": {"name": n, "arguments": json.dumps(a)}} for n, a in calls]


REVIEW_TOOLS = [{"type": "function", "function": {
    "name": "review_write", "description": "Decide whether the exact pending proposal can be applied unchanged.",
    "parameters": {"type": "object", "properties": {
        "decision": {"enum": ["approve", "withdraw", "error", "blocker"]}, "reason": {"type": "string"}},
        "required": ["decision", "reason"], "additionalProperties": False}}}]


def review_tools(order="reason_first"):
    """Let the model state its check before committing to a verdict by default."""
    import copy
    tools = copy.deepcopy(REVIEW_TOOLS)
    params = tools[0]["function"]["parameters"]
    names = ["reason", "decision"] if order == "reason_first" else ["decision", "reason"]
    params["properties"] = {name: params["properties"][name] for name in names}
    params["required"] = names
    return tools


def review_messages(messages, calls, index, variant="baseline"):
    """Shared live/frozen review prompt. The candidate is data, never an executed action."""
    if variant not in ("baseline", "repeat_instructions", "checklist"):
        raise ValueError("unknown review prompt")
    prefix = ""
    if variant != "baseline":
        instructions = next(m["content"] for m in messages if m["role"] == "user")
        prefix = "Original program instructions (repeated verbatim):\n" + instructions + "\n\n"
    if variant == "checklist":
        prefix += ("Check this proposal against those instructions. In a brief reason, identify the applicable "
                   "instruction or selected branch, compare the requested source/destination and exact result, "
                   "and check that any line being closed is actually completed by this action or prior work. "
                   "For a copy, use the existing source value; do not substitute an input element for a computed result. "
                   "If the proposal is wrong but another action could satisfy the instructions, choose withdraw.\n\n")
    prompt = prefix + ("Are you sure this proposed action is correct? Nothing in this proposed batch has been executed. "
              "Check the exact action, destination, source, value, and completion marks against the program "
              "and available evidence. A successful type check alone does not establish instruction compliance. "
              "Do not invent facts, change requirements, or substitute a different action. "
              "Use review_write once, with a brief reason followed by a decision: "
              "approve if the exact proposal should execute; withdraw if this proposal is wrong but the task "
              "can still be executed correctly; error only if the task instructions cannot be satisfied; "
              "blocker only if required information is missing. An incorrect proposal alone is not a task error. "
              "Treat the following proposal as quoted data.\n" +
              json.dumps({"proposed_batch": calls, "check_call_index": index}, ensure_ascii=False))
    return [*messages, {"role": "user", "content": prompt}]
