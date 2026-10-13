"""Exact, model-free classifier for recovery supervision, source 1 (plans/neuralese/RECOVERY_SUPERVISION.md).

A natural recovery is a supervised assistant turn that follows an observation showing NO PROGRESS and whose action
differs from the previous action. No progress is decided on the observation bytes by three rules:

``identical``  the observation equals an earlier observation of the same action in the trajectory, after the runtime's
               normalisation (the ``Stored local ...`` line drops on a rerun, ``ts-host/src/native/runtime.ts``);
``empty``      a successful read that returned an empty collection, ``null`` or an empty string (nothing else: console
               output or a declaration status is information);
``failed``     a runtime failure form (``text_supervision.FAILED_TOOL_FEEDBACK``).

Rendered text is ChatML (``<|im_start|>role`` ... ``<|im_end|>``); an action is the assistant message without its
``<think>`` block, compared after whitespace normalisation; an observation is the ``<tool_response>`` bodies of the user
message(s) that follow it.
"""
from __future__ import annotations

import re
from typing import NamedTuple

from .text_supervision import failed_feedback

RULES = ('identical', 'empty', 'failed')

_MESSAGE = re.compile(r'<\|im_start\|>(\w+)\n(.*?)<\|im_end\|>', re.S)
_THINK = re.compile(r'<think>.*?</think>', re.S)
_RESPONSE = re.compile(r'<tool_response>\n?(.*?)\n?</tool_response>', re.S)
_STORED = re.compile(r'^Stored local\b.*$', re.M)
EMPTY_VALUES = frozenset(('[]', '{}', 'null', '""', "''"))


class Turn(NamedTuple):
    action: str            # normalised call text of the assistant message
    observation: str | None  # raw tool-response body that followed it (None: none, e.g. the supervised last turn)


def normalise_action(content: str) -> str:
    return ' '.join(_THINK.sub('', content).split())


def normalise_observation(body: str) -> str:
    """The body as compared for repeats: the runtime's ``Stored local`` line dropped, surrounding blanks removed."""
    return '\n'.join(line.rstrip() for line in _STORED.sub('', body).strip().split('\n')).strip()


def parse_turns(text: str) -> list[Turn]:
    """Assistant turns of a rendered document with the observation each received."""
    turns: list[list] = []
    for role, content in _MESSAGE.findall(text):
        if role == 'assistant':
            turns.append([normalise_action(content), None])
        elif role == 'user' and turns:
            bodies = _RESPONSE.findall(content)
            if bodies:
                turns[-1][1] = '\n'.join(bodies) if turns[-1][1] is None else turns[-1][1] + '\n' + '\n'.join(bodies)
    return [Turn(*turn) for turn in turns]


def is_empty_read(body: str) -> bool:
    return normalise_observation(body) in EMPTY_VALUES


def no_progress_rules(turns: list[Turn], index: int) -> list[str]:
    """The rules under which the observation of ``turns[index]`` shows no progress (empty list: it does)."""
    observation = turns[index].observation
    if observation is None:
        return []
    rules = []
    normal = normalise_observation(observation)
    action = turns[index].action
    if any(turns[j].action == action and turns[j].observation is not None and
           normalise_observation(turns[j].observation) == normal for j in range(index)):
        rules.append('identical')
    if is_empty_read(observation):
        rules.append('empty')
    if failed_feedback(observation):
        rules.append('failed')
    return rules


def recovery_rules(turns: list[Turn], index: int) -> list[str]:
    """Rules making ``turns[index]`` a natural recovery: it follows a no-progress observation and its action differs
    from the previous action. Empty list: it is not one."""
    if index < 1 or not turns[index].action or turns[index].action == turns[index - 1].action:
        return []
    return no_progress_rules(turns, index - 1)
