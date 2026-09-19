"""Normalize legacy terminal-tool trajectories for reply-only training."""
from __future__ import annotations


_OLD_SUPPORT_END = ("When the result is ready and all numbered lines are closed, use done to finish\n"
                    "successfully, or reply briefly. Error and blocker reports are only for failures.")
_NEW_SUPPORT_END = ("When the result is ready and all numbered lines are closed, reply briefly to\n"
                    "finish successfully. Error and blocker reports are only for failures.")
_OLD_TEACHER_END = ("When the required result is written and the applicable work is complete, call done to finish "
                    "successfully. Never use report_error or report_blocker to announce successful completion.")
_NEW_TEACHER_END = ("When the required result is written and all applicable lines are closed, reply briefly to finish "
                    "successfully. Never use report_error or report_blocker to announce successful completion.")


def reply_only_sample(sample: dict) -> dict:
    """Remove the obsolete terminal tool, converting its sole target to a reply.

    This operates on an SFT row at export time; source traces stay immutable.
    `done=N` arguments and `mark_done` calls are line marks and are preserved.
    """
    tools = [tool for tool in sample["tools"] if tool["function"]["name"] != "done"]
    messages = []
    for message in sample["messages"]:
        if any(call["function"]["name"] == "done" for call in message.get("tool_calls", [])):
            raise ValueError("terminal done in history cannot be migrated as one SFT turn")
        if message["role"] == "system":
            content = message.get("content", "")
            content = content.replace(_OLD_SUPPORT_END, _NEW_SUPPORT_END)
            content = content.replace(_OLD_TEACHER_END, _NEW_TEACHER_END)
            if "use done to finish" in content or "call done to finish" in content:
                raise ValueError("unrecognized terminal done instruction in system prompt")
            message = {**message, "content": content}
        messages.append(message)
    target = sample["target"]
    calls = target.get("tool_calls") or []
    terminal = [call for call in calls if call["function"]["name"] == "done"]
    if terminal:
        if len(calls) != 1:
            raise ValueError("mixed terminal done batch cannot be migrated as one SFT turn")
        target = {"role": "assistant", "content": "Done."}
    return {**sample, "messages": messages, "tools": tools, "target": target,
            "skill": "reply" if terminal else sample["skill"]}
