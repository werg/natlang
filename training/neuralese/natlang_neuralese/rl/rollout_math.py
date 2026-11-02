"""Pure group statistics for scored native task rollouts."""
from __future__ import annotations

import math
import json
import re


DEFAULT_EXECUTION_EFFICIENCY_WEIGHT = 1.0
MAX_EXECUTION_EFFICIENCY_COST = 0.05


def group_advantages(rewards: list[float | None], eps: float = 1e-6) -> tuple[list[float | None], dict]:
    """Group-relative advantages of one task's rollouts; None for unscored rollouts. A group whose scored rewards
    have no spread gets all-zero advantages."""
    scored = [r for r in rewards if r is not None]
    if len(scored) < 2:
        return [None if r is None else 0.0 for r in rewards], {"scored": len(scored), "spread": False}
    mean = sum(scored) / len(scored)
    std = math.sqrt(sum((r - mean) ** 2 for r in scored) / len(scored))
    if std < eps:
        return [None if r is None else 0.0 for r in rewards], {"scored": len(scored), "spread": False, "mean": mean}
    return [None if r is None else (r - mean) / (std + eps) for r in rewards], \
        {"scored": len(scored), "spread": True, "mean": mean, "std": std}


def rollout_return(rollout: dict) -> float:
    """An admitted episode's return: its task-outcome ``quality`` plus any recorded Neuralese-use bonus
    (rl.task_rollouts.score). ``quality`` alone stays the success signal."""
    reward = rollout['reward']
    return float(reward['quality']) + float(reward.get('neuralese_bonus') or 0.)


def execution_efficiency_counters(turns: list[dict]) -> dict:
    """Recount whole-graph model calls, observed generated tokens and proven no-progress repeats from turns.

    A cross-turn repeat counts only when the exact same action in the same invocation has two linked tool responses
    with identical contents. Missing responses are unknown and do not count. Identical calls in one sampled response
    have no intervening observation and count as a repeat.
    """
    model_calls = len(turns)
    token_counts: list[int] = []
    token_complete = True
    prior_actions: dict[str, dict[str, list[str | None]]] = {}
    repeated_actions = 0
    for turn_index, turn in enumerate(turns):
        usage = turn.get('usage') or {}
        count = usage.get('completion_tokens')
        if type(count) is not int or count < 0:
            samples = (turn.get('sampling') or {}).get('samples')
            if isinstance(samples, list):
                count = len(samples)
            else:
                token_complete = False
        if token_complete:
            token_counts.append(count)
        target = turn.get('target') or {}
        invocation = turn.get('invocation_id')
        invocation_key = invocation if isinstance(invocation, str) and invocation else f'__turn_{turn_index}'
        action_history = prior_actions.setdefault(invocation_key, {})
        same_turn: set[str] = set()
        new_actions: list[tuple[str, str | None]] = []
        for call in target.get('tool_calls') or []:
            function = call.get('function') or {}
            arguments = function.get('arguments')
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except (ValueError, TypeError):
                    pass
            signature = json.dumps([function.get('name'), arguments], sort_keys=True, ensure_ascii=False,
                                   separators=(',', ':'))
            if signature in same_turn:
                repeated_actions += 1
            same_turn.add(signature)
            call_id = call.get('id')
            new_actions.append((signature, call_id if isinstance(call_id, str) and call_id else None))
        messages = turn.get('messages') or []
        for signature, _ in new_actions:
            previous_ids = action_history.get(signature) or []
            if len(previous_ids) >= 2 and previous_ids[-1] and previous_ids[-2]:
                def observation(call_id: str) -> str | None:
                    matches = [message.get('content') for message in messages if isinstance(message, dict)
                               and message.get('role') == 'tool' and message.get('tool_call_id') == call_id]
                    if len(matches) != 1:
                        return None
                    return json.dumps(matches[0], sort_keys=True, ensure_ascii=False, separators=(',', ':'))
                latest = observation(previous_ids[-1])
                earlier = observation(previous_ids[-2])
                if latest is not None and latest == earlier:
                    repeated_actions += 1
        for signature, call_id in new_actions:
            action_history.setdefault(signature, []).append(call_id)
    return {'model_calls': model_calls,
            'generated_tokens': sum(token_counts) if token_complete else None,
            'repeated_actions': repeated_actions}


def execution_efficiency_cost(*, model_calls: int, generated_tokens: int | None,
                              repeated_actions: int, weight: float = DEFAULT_EXECUTION_EFFICIENCY_WEIGHT) -> float:
    """Bounded cost from observed whole-episode counters; no elapsed-time or prompt-size terms.

    Missing generated-token telemetry contributes no token component and must remain explicitly unknown in the
    caller's receipt. Calls and repeated actions are independently observed from the captured execution graph and
    sampled actions. The maximum is 0.05, keeping it small beside binary task outcomes and the 0.1 native-use bonus.
    """
    if (type(model_calls) is not int or model_calls < 0 or type(repeated_actions) is not int or repeated_actions < 0
            or (generated_tokens is not None and (type(generated_tokens) is not int or generated_tokens < 0))
            or type(weight) not in (int, float) or not math.isfinite(weight) or not 0 <= weight <= 1):
        raise ValueError('invalid execution-efficiency counters or weight')
    call_cost = 0.015 * min(model_calls, 8) / 8
    token_cost = 0.0 if generated_tokens is None else 0.015 * min(generated_tokens, 2048) / 2048
    repeat_cost = 0.020 * min(repeated_actions, 4) / 4
    return min(MAX_EXECUTION_EFFICIENCY_COST, call_cost + token_cost + repeat_cost) * float(weight)


def grounded_failure_report_credit(turns: list[dict]) -> float:
    """Small credit for a terminal report quoting an observed, linked tool error.

    This proves accurate reporting of execution evidence, not that the task was
    impossible. Unverified claims of missing information receive nothing. Only
    failed task outcomes may use this credit; task quality remains unchanged.
    """
    if not turns:
        return 0.
    final = turns[-1]
    calls = (final.get('target') or {}).get('tool_calls') or []
    if len(calls) != 1:
        return 0.
    function = calls[0].get('function') or {}
    arguments = function.get('arguments')
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (ValueError, TypeError):
            return 0.
    if (function.get('name') != 'return_result' or not isinstance(arguments, dict)
            or arguments.get('status') not in ('blocked', 'failed')):
        return 0.
    reason = arguments.get('reason')
    invocation = final.get('invocation_id')
    if not isinstance(reason, str) or not invocation:
        return 0.
    prior_ids = {call.get('id') for turn in turns[:-1] if turn.get('invocation_id') == invocation
                 for call in (turn.get('target') or {}).get('tool_calls') or [] if call.get('id')}
    for message in final.get('messages') or []:
        content = message.get('content')
        if (message.get('role') != 'tool' or message.get('tool_call_id') not in prior_ids
                or not isinstance(content, str)):
            continue
        for line in content.splitlines():
            line = line.strip()
            if (24 <= len(line) <= 1200 and line in reason and re.match(
                    r'(?i)^(?:error\b|[A-Za-z]*Error:|invalid-binding\b|type-error\b|eval-error\b)', line)):
                return .02
    return 0.


def reporting_spread(rewards: list[dict]) -> bool:
    """Only outcome-equal failures may learn solely from grounded reporting."""
    return bool(rewards) and all(row.get('quality') == 0 for row in rewards) and len({
        float((row.get('execution_efficiency') or {}).get('failure_report_credit') or 0.)
        for row in rewards}) > 1
