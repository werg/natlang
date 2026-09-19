"""Normalize legacy terminal-tool trajectories for reply-only training."""
from __future__ import annotations


_OLD_SUPPORT_END = ("When the result is ready and all numbered lines are closed, use done to finish\n"
                    "successfully, or reply briefly. Error and blocker reports are only for failures.")
_PREVIOUS_SUPPORT_END = ("When the result is ready and all numbered lines are closed, reply briefly to\n"
                         "finish successfully. Error and blocker reports are only for failures.")
_NEW_SUPPORT_END = ("When the result is ready and all numbered lines are closed, end your turn with\n"
                    "no text. Error and blocker reports are only for failures.")
_OLD_TEACHER_END = ("When the required result is written and the applicable work is complete, call done to finish "
                    "successfully. Never use report_error or report_blocker to announce successful completion.")
_PREVIOUS_TEACHER_END = ("When the required result is written and all applicable lines are closed, reply briefly to "
                         "finish successfully. The reply is only a note; the result is what you wrote to return. "
                         "Never use report_error or report_blocker to announce successful completion.")
_NEW_TEACHER_END = ("When the required result is written and all applicable lines are closed, end your turn without "
                    "text. The result is what you wrote to return. Never use report_error or report_blocker to "
                    "announce successful completion.")
_OLD_SMALL_INTRO = ("Carry out the task step by step with the tools, then reply briefly to say what you did. "
                    "Your reply is only a note: the result is whatever you wrote to `return`.")
_NEW_SMALL_INTRO = ("Carry out the task step by step with the tools, then end your turn without text. "
                    "The result is whatever you wrote to `return`.")
_OLD_DELEGATE_END = ("When `return` holds the finished result, reply briefly; the reply is only a note.")
_NEW_DELEGATE_END = ("When `return` holds the finished result, end your turn without text.")


def reply_only_sample(sample: dict) -> dict:
    """Normalize a successful turn to an empty end-of-turn target.

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
            content = content.replace(_OLD_SMALL_INTRO, _NEW_SMALL_INTRO)
            content = content.replace(_OLD_DELEGATE_END, _NEW_DELEGATE_END)
            content = content.replace(_OLD_SUPPORT_END, _NEW_SUPPORT_END)
            content = content.replace(_PREVIOUS_SUPPORT_END, _NEW_SUPPORT_END)
            content = content.replace(_OLD_TEACHER_END, _NEW_TEACHER_END)
            content = content.replace(_PREVIOUS_TEACHER_END, _NEW_TEACHER_END)
            if any(phrase in content for phrase in ("use done to finish", "call done to finish", "reply briefly")):
                raise ValueError("unrecognized success-reply instruction in system prompt")
            message = {**message, "content": content}
        messages.append(message)
    target = sample["target"]
    calls = target.get("tool_calls") or []
    terminal = [call for call in calls if call["function"]["name"] == "done"]
    if terminal:
        if len(calls) != 1:
            raise ValueError("mixed terminal done batch cannot be migrated as one SFT turn")
        target = {"role": "assistant", "content": ""}
    reply = bool(terminal) or (sample["skill"] == "reply" and not calls)
    if reply:
        target = {**target, "content": ""}
    return {**sample, "messages": messages, "tools": tools, "target": target,
            "native_target": "" if reply else sample.get("native_target"),
            "skill": "reply" if reply else sample["skill"]}
