"""S7 task-success rollouts: GRPO groups over train-split task rows, run by the collector, scored by classify_case.

The reward is the task-success eval's own outcome class (`eval.task_packet.classify_case`): 1 for `task_success`,
0 for every model-caused failure, None (unscored) for `infrastructure_error` and `unresolved`, which must not teach
the policy. Rollouts run through the same frozen-collector path as the held-task eval (`scripts/run_bonsai_queue.py`
against an OpenAI-compatible `natlang_neuralese.serve`), so RL trains and is judged on the same execution path.

  prepare  derive a train-only packet and a queue of `group` sampled single-case entries per row
           (distinct seeds, temperature > 0). The root mode is declared: authored for
           child-policy rollouts, student for complete student-policy rollouts.
  score    read the queue journal and results, classify every rollout and report per-group rewards, advantages
           (grpo.group_advantages) and the spread curriculum: groups with mixed outcomes are GRPO-ready; groups
           with zero spread go to rejection sampling (all fail: teacher seeding; all succeed: retire).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from ..eval.task_packet import classify_case, classification_binding, read_jsonl
from .rollout_math import (DEFAULT_EXECUTION_EFFICIENCY_WEIGHT, group_advantages,
                           execution_efficiency_cost, execution_efficiency_counters, grounded_failure_report_credit, reporting_spread)
from ..common.code_provenance import code_provenance
from ..common.jsonio import json_line
from .episode_files import present
from ..common.task_roots import student_root_row, authored_root_row

UNSCORED_OUTCOMES = {"infrastructure_error", "unresolved"}
# Owner 2026-10-31: reward Neuralese use. A correct student episode whose captured execution graph passes native
# state between calls (a real producer export that a consumer actually imported) earns this bonus on top of the
# success reward (1.0); outcome stays primary and failed episodes never earn it. 0 is the explicit opt-out.
DEFAULT_NEURALESE_USE_BONUS = 0.1


def neuralese_use_bonus(value: Any) -> float:
    """The configured bonus (None means the default); finite and nonnegative, at most the success reward."""
    value = DEFAULT_NEURALESE_USE_BONUS if value is None else value
    if type(value) not in (int, float) or not 0 <= value <= 1:
        raise ValueError("neuralese_use_bonus must be a number in [0, 1] (0 opts out)")
    return float(value)


def neuralese_exchanges(rollout: dict[str, Any], *, captured: bool = True) -> int:
    """Count the episode's real between-call Neuralese edges from its captured graph, never from text or markers.

    An edge is a ``block_read`` whose one exact input is an in-episode ``block_write`` of the same state, made by a
    model turn of a different invocation. ``validate_rollout_graph`` first binds every graph write/read to the
    turns' actual captured state exports (``sampling.writes``) and native-state imports (``sampling.prompt``), so a
    read with no captured import, or an import of an external/older state, is not an exchange. An episode whose
    graph or capture does not validate counts zero. ``captured=False`` (a run that requested no RL sampling capture,
    such as the held-out evaluation) counts the same edges over the runtime's execution graph after its model-turn
    coverage check: a measurement, never a reward (the capture-bound check refused every such episode, so held
    evaluations reported zero use while their relay/typed episodes passed state)."""
    from .episode_graph import rollout_turn_names, validate_model_turn_coverage, validate_rollout_graph
    try:
        if captured:
            _, nodes = validate_rollout_graph(rollout)
        else:
            nodes = validate_model_turn_coverage(rollout_turn_names(rollout.get('turns') or []),
                                                 rollout.get('execution_graph') or [])
    except (TypeError, ValueError, KeyError, AttributeError):
        return 0
    def invocation(turn: Any) -> Any:
        event = nodes.get(turn) or {}
        return event.get("call_id") or (turn.rsplit("#turn", 1)[0] if isinstance(turn, str) else None)
    count = 0
    for event in nodes.values():
        if event.get("kind") != "block_read":
            continue
        sources = [edge.get("node") for edge in event.get("inputs") or []
                   if isinstance(edge, dict) and edge.get("block") == event.get("block")]
        writer = nodes.get(sources[0]) if len(sources) == 1 else None
        if (writer is None or writer.get("kind") != "block_write" or writer.get("block") != event.get("block")
                or writer.get("turn") == event.get("turn")):
            continue
        if invocation(writer.get("turn")) != invocation(event.get("turn")):
            count += 1
    return count


def _packet_source_custody(task: dict[str, Any], *, packet_sha256: str, row_index: int,
                           row_bytes: bytes) -> dict[str, Any]:
    """Bind source labels to the exact task row scored by this collector packet.

    The packet artifact digest authenticates the containing task set; the raw
    JSONL line digest identifies the exact row without reserializing it. Source
    identifiers are copied from that row and validated before they can be used
    to label a rollout or native graph.
    """
    task_id = task.get("id")
    if not isinstance(task_id, str) or not task_id:
        raise ValueError(f"packet row {row_index} has no stable task id")
    result: dict[str, Any] = {}
    for field in ("source_ids", "source_groups", "source_revisions"):
        values = task.get(field)
        if (not isinstance(values, list) or not values or
                any(not isinstance(value, str) or not value.strip() for value in values) or
                len(values) != len(set(values))):
            raise ValueError(f"packet task {task_id} needs complete unique {field}")
        result[field] = list(values)
    if (not re.fullmatch(r"[0-9a-f]{64}", packet_sha256) or
            not isinstance(row_bytes, bytes) or not row_bytes.endswith(b"\n")):
        raise ValueError("packet source custody needs exact newline-terminated JSONL bytes")
    return {
        "schema": "natlang.rollout-source-custody/1",
        "task_id": task_id,
        "packet_sha256": packet_sha256,
        "packet_row_index": row_index,
        "packet_row_sha256": hashlib.sha256(row_bytes).hexdigest(),
        **result,
        **({"source": task["source"]} if isinstance(task.get("source"), str) else {}),
    }


def reward_of(outcome: str, finish: dict[str, Any] | None = None) -> float | None:
    """Binary task-success reward; infrastructure outcomes are unscored, never 0. A host wall-clock stop
    (`case_wall_budget`) is a collection resource limit that depends on server load, not on the policy (the journal
    says so: "not a language deadline or model-negative label"), so it is unscored too; a model-request budget stays 0."""
    if outcome == "budget_exhaustion" and (finish or {}).get("resource_limit_reason") == "case_wall_budget":
        return None
    if outcome in UNSCORED_OUTCOMES:
        return None
    return 1.0 if outcome == "task_success" else 0.0


# Resends of one failed actor request before its episode fails. An actor engine that dies is relaunched within about a
# minute (pop_actor_driver, serve_native_checkpoint's ENGINE_DIED_EXIT); at the remote 15 s retry delay and the 30 s
# cap, 6 resends wait about 2.5 minutes, so in-flight episodes ride out a relaunch instead of being lost (aei1 held,
# 2026-11-24: 32 episodes failed after the default single resend). A resend never spends call budget (ts-host
# collector). NATLANG_ACTOR_REQUEST_RETRIES overrides it (1: the collector default).
ACTOR_REQUEST_RETRIES = 6


def actor_request_retries() -> int:
    value = os.environ.get("NATLANG_ACTOR_REQUEST_RETRIES")
    if value is None:
        return ACTOR_REQUEST_RETRIES
    if not value.isdigit():
        raise ValueError("NATLANG_ACTOR_REQUEST_RETRIES must be a nonnegative integer")
    return int(value)


def infra_lost(row: dict[str, Any]) -> bool:
    """Whether a rollout row was lost to infrastructure (``infrastructure_error``, a held sample's ``infra_lost``
    label, or an unscored row with 0 model calls): it says nothing about the policy, so pass rates leave it out. Any other unscored row (a wall-clock stop, an
    unresolved run) counts as a failure in a pass rate, though it never trains (aei1 held re-judgement, 2026-11-24)."""
    if row.get("infra_lost"):
        return True
    if row.get("outcome") == "infrastructure_error":
        return True
    # An unscored row whose policy was never called measured nothing about it: rows written before the scorer
    # classified a collector's transport failure as infrastructure_error read 'unresolved' (aei1 r1's DGX collection:
    # 686 of 1,142 student rows, 0 model calls; the r0009 frontier counted 853 such rows as failures on 257 tasks).
    # NATLANG_ZERO_CALL_UNSCORED_LOST=0 counts them as failures again.
    calls = (row.get("execution_efficiency") or {}).get("model_calls")
    zero_call_stall = (calls == 0 and "memetic_success" not in row
                       and (row.get("reward") is None or row.get("outcome") == "budget_exhaustion"
                            or row.get("resource_limit_reason") == "model_request_budget"))
    return zero_call_stall and os.environ.get("NATLANG_ZERO_CALL_UNSCORED_LOST", "1") != "0"


def pass_rate_counts(rows) -> tuple[int, int]:
    """(successes, counted) of rollout rows for a task's pass rate: rows lost to infrastructure are left out, other
    unscored rows are failures; a success is a reward of 1 (a float, or a reward dict's ``quality``) or a memetic
    episode's ``memetic_success``."""
    successes = counted = 0
    for row in rows:
        if infra_lost(row):
            continue
        counted += 1
        if "memetic_success" in row:
            successes += bool(row["memetic_success"])
            continue
        reward = row.get("reward")
        reward = reward.get("quality") if isinstance(reward, dict) else reward
        successes += reward == 1.0
    return successes, counted


def entry_outcome(entry: dict[str, Any], task: dict[str, Any], finish: dict[str, Any] | None) -> tuple[str, float | None, float | None]:
    """(outcome class, binary reward, graded reward) of one finished queue entry, classified as ``score`` classifies it
    (memetic improvement tasks aside): read before its group is complete (grpo.py dynamic sampling)."""
    output = present(entry["output"])     # raw or compacted once scored (episode_files.compact_results)
    results = read_jsonl(output) if output else []
    result = results[0] if results else None
    outcome = classify_case(result, finish)
    reward = reward_of(outcome, finish)
    return outcome, reward, graded_reward(outcome, reward, result, task.get("semantics"))


# Checks that a graded reward may give partial credit for; any other failed check (status, honest stop, stop evidence,
# world, required actions, ...) leaves the episode at 0.
GRADED_CHECKS = frozenset({"answer", "files"})
# File compares whose checkFiles score is already over the paths the episode had to change or did change
# (``oracles.ts`` default branch: ``expected != input or actual != input``), one exact item per path or record key.
CHANGED_PATH_COMPARES = frozenset({"exact", "moves", "rewrite", "csv", "counts", "json-string-record", "qa-string-map"})


def _json_canonical(text: str) -> str | None:
    def pairs(items):
        keys = [key for key, _ in items]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate key")
        return dict(items)

    def plain(value):
        if isinstance(value, float) and value.is_integer() and abs(value) < 2 ** 53:
            return int(value)
        if isinstance(value, list):
            return [plain(item) for item in value]
        if isinstance(value, dict):
            return {key: plain(item) for key, item in value.items()}
        return value
    try:
        return json.dumps(plain(json.loads(text, object_pairs_hook=pairs)), sort_keys=True, ensure_ascii=False)
    except (ValueError, TypeError):
        return None


def file_content_equal(path: str, actual: str | None, expected: str | None) -> bool:
    """``oracles.ts`` ``fileContentEqual``: exact text, JSON files equal as data."""
    if actual is None or expected is None:
        return actual is expected
    if not path.lower().endswith(".json"):
        return actual == expected
    left = _json_canonical(actual)
    return left is not None and left == _json_canonical(expected)


def effect_fraction(outcome: dict[str, Any], semantics: dict[str, Any]) -> float | None:
    """The share of the file items the episode had to change or did change that end equal to the expected state: only
    paths where the expected or the final file differs from the starting one count, so a file that already matched and
    was left alone earns nothing. None when the task has no checked folder effect (no ``files_check`` or nothing to
    write)."""
    check = outcome.get("files_check")
    if not isinstance(check, dict):
        return None
    spec = semantics.get("files_oracle") or {"compare": "content"}
    compare = spec.get("compare") or "exact"
    if compare == "content":
        before = semantics.get("folder_files")
        after = outcome.get("files")
        if not isinstance(before, dict) or not isinstance(after, dict):
            return None
        want = semantics.get("expected_files")
        want = before if want is None else want
        # Items: every file the episode had to write (expected present and different from the start), which passes
        # when it ends equal; every file it had to remove but left, and every file it changed, created or removed that
        # should have stayed as it was, which fail. A correct removal earns nothing on its own (the source half of a
        # move): moving files into the wrong folder must not score the emptied sources.
        passed = items = 0
        for path in sorted(set(before) | set(want) | set(after)):
            start, goal, end = before.get(path), want.get(path), after.get(path)
            if not file_content_equal(path, goal, start):
                if goal is not None:
                    items += 1
                    passed += file_content_equal(path, end, goal)
                elif end is not None:
                    items += 1
            elif not file_content_equal(path, end, start):
                items += 1
        return passed / items if items else None
    if compare in CHANGED_PATH_COMPARES:
        items = check.get("items")
        return check.get("passed", 0) / items if items else None
    return 1.0 if check.get("accepted") else 0.0


def graded_reward(outcome_class: str, reward: float | None, result: dict[str, Any] | None,
                  semantics: dict[str, Any] | None) -> float | None:
    """The episode's reward in [0, 1] with partial credit for file effects (owner decision 2026-11-13): 1 for an accepted
    episode, None when unscored, and for a finished episode the oracle rejected (``semantic_mismatch``) whose only failed
    checks are the answer and the files, ``effect_fraction`` (halved when the answer is wrong too). The work done is
    the credit: a right answer with 1 of 43 files earns 0.02, not half (returning the value and skipping the effects is
    the base model's commonest failure and must not be rewarded), and an episode that achieved none of the effects it
    had to make earns 0 whatever it returned. A task without a checked folder effect stays binary. Budget stops, other outcomes
    and any other failed check are 0.

    A task judged by a trusted executable verifier (``semantics.trusted_exact_verifier``) whose verdict carries a graded
    ``score`` gets that score for its answer: the verifier's credit when the answer alone failed, and the factor on the
    effect fraction (in place of the 0.5) when the files failed too."""
    if reward is None or reward == 1.0:
        return reward
    if outcome_class != "semantic_mismatch" or not result:
        return 0.0
    outcome = result.get("outcome") if isinstance(result.get("outcome"), dict) else {}
    checks = outcome.get("checks") or {}
    failed = {name for name, passed in checks.items() if not passed}
    if not checks or not failed or not failed <= GRADED_CHECKS:
        return 0.0
    answer_credit = verifier_score(outcome, semantics or {})
    if failed == {"answer"} and answer_credit is not None:
        return round(answer_credit, 6)
    effects = effect_fraction(outcome, semantics or {})
    if not effects:
        return 0.0
    factor = 1.0 if checks.get("answer") else (0.5 if answer_credit is None else answer_credit)
    return round(effects * factor, 6)


def verifier_score(outcome: dict[str, Any], semantics: dict[str, Any]) -> float | None:
    """The graded answer credit in [0, 1] a trusted executable verifier returned (``trusted-exact-verifier.ts``), None
    when the task has no such verifier or its verdict carries no score."""
    if not isinstance(semantics.get("trusted_exact_verifier"), dict):
        return None
    oracle = outcome.get("oracle") if isinstance(outcome.get("oracle"), dict) else {}
    score = oracle.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= 1:
        return None
    return float(score)


def expand_captures(trajectory: list[Any]) -> list[Any]:
    """The trajectory with every compact RL capture (`trajectory.ts` `compactCaptures`) rebuilt exactly as the actor
    returned it: a `tokens_delta` prompt is the first `keep` tokens and `producing` flags of turn `turn`'s (expanded)
    prompt, then `token_ids`, producing exactly at `producing_at`; `samples_columns` are the ordinary `samples` and their
    first `generated_inputs` tokens. Other steps are returned as they are."""
    out, full = [], {}
    for index, step in enumerate(trajectory):
        sampling = ((step or {}).get("model_response") or {}).get("rl_sampling") if isinstance(step, dict) else None
        if not isinstance(sampling, dict):
            out.append(step)
            continue
        expanded = sampling
        columns = sampling.get("samples_columns")
        if columns is not None:
            ids, logps, start, terminal = (columns["token_id"], columns["logp"], columns["query_own_start"],
                                           set(columns["terminal_at"]))
            if len(ids) != len(logps) or not 0 <= columns["generated_inputs"] <= len(ids):
                raise ValueError(f"trajectory turn {index}: malformed samples_columns")
            expanded = {key: value for key, value in sampling.items() if key != "samples_columns"}
            expanded["samples"] = [{"index": i, "token_id": token, "logp": logp, "input_role": "ordinary",
                                    "terminal": i in terminal, "query_own_position": start + i}
                                   for i, (token, logp) in enumerate(zip(ids, logps))]
            expanded["generated_inputs"] = [{"sample_index": i, "token_id": ids[i], "input_role": "ordinary"}
                                            for i in range(columns["generated_inputs"])]
        prompt = expanded.get("prompt")
        if isinstance(prompt, list) and len(prompt) == 1 and isinstance(prompt[0], dict):
            segment = prompt[0]
            if segment.get("kind") == "tokens_delta":
                base = full.get(segment["turn"])
                if base is None or segment["turn"] >= index or len(base["token_ids"]) < segment["keep"]:
                    raise ValueError(f"trajectory turn {index}: prompt delta refers to turn {segment['turn']}, "
                                     "which has no earlier captured token prompt")
                rest = [False] * len(segment["token_ids"])
                for position in segment["producing_at"]:
                    rest[position] = True
                segment = {"kind": "tokens", "token_ids": base["token_ids"][:segment["keep"]] + list(segment["token_ids"]),
                           "producing": base["producing"][:segment["keep"]] + rest}
                expanded = {**expanded, "prompt": [segment]}
            if segment.get("kind") == "tokens" and isinstance(segment.get("token_ids"), list):
                full[index] = segment
        if expanded is not sampling:
            step = {**step, "model_response": {**step["model_response"], "rl_sampling": expanded}}
        out.append(step)
    return out


def rollout_turns(result: dict[str, Any] | None) -> list[dict[str, Any]]:
    """The policy's own model turns of a collector result as `natlang.rollout/1` turns ({messages, tools, target}),
    replayable as `logLikelihood` terms. Host-authored actions (`action_provenance`, e.g. the authored root eval) are
    not policy samples and are skipped.

    A reasoned decision readout's reasoning generations (phase ``decision_reasoning``) are never turns: their free reply
    after ``</think>`` was never used. Each scored entry (phase ``action``, ``decision_reasoning: {index, samples,
    action}``) keeps that metadata plus its paired generation's exact sampled ids (``reasoning_sample_ids``, the
    generation of the same invocation and sample index before it) and the hash of its scored prompt ids
    (``scored_prompt_sha256``, ``data.decision_graph.token_ids_sha256``), what ``sft_sources build`` trains the
    reasoning from and checks its rendering against (``data.decision_graph.reasoned_decision_plan``). The action entry
    (``action: true``, the sample whose choice the program used) also keeps that generation's whole exact capture
    (``reasoning_sampling``: prompt, sampled ids and behaviour log-probabilities, sampling law, finish reason), the
    sampled action GRPO trains (``lower_rollout_turns.decision_reasoning_turns``)."""
    turns = []
    invocations = {item.get('invocation_id'): item for item in ((result or {}).get('outcome') or {}).get('invocation_ledger') or []}
    reasoning_steps: dict[tuple[Any, Any], dict[str, Any]] = {}
    for step in expand_captures((result or {}).get("trajectory") or []):
        if step.get("phase") == "decision_reasoning":
            marker = step.get("decision_reasoning")
            if isinstance(marker, dict):
                reasoning_steps[(step.get("invocation_id"), marker.get("index"))] = step
            continue
        assistant = step.get("assistant")
        if step.get("phase") != "action" or not isinstance(assistant, dict) or assistant.get("action_provenance") \
                or step.get("action_provenance") or not isinstance(step.get("context"), list):
            continue
        target: dict[str, Any] = {"role": "assistant", "content": assistant.get("content") or ""}
        reasoning = assistant.get("reasoning")
        if isinstance(reasoning, str) and reasoning:
            target["reasoning_content"] = reasoning
        response = step.get("model_response") or {}
        # Raw calls are the sampled action. Normalized calls can rename tools,
        # reserialize JSON, or even infer a tool call from plain output text.
        # Those are runtime interpretations, not the policy's sampled tokens.
        raw_calls = response.get("raw_calls")
        native_xml = bool(response.get('tool_parse'))
        if native_xml:
            from ..data.native_xml_custody import derived_tool_calls
            target['tool_calls'] = derived_tool_calls(step)
        elif isinstance(raw_calls, list) and raw_calls:
            target["tool_calls"] = copy.deepcopy(raw_calls)
        elif "raw_calls" not in response:
            # Legacy trajectories did not retain raw calls. Keep them usable for
            # RFT; the receipt explicitly excludes exact on-policy replay.
            calls = [{"id": call.get("call_id") or f"call_{index}", "type": "function",
                      "function": {"name": call.get("source_tool") or call.get("tool"),
                                   "arguments": json.dumps(call.get("arguments") or {}, ensure_ascii=False)}}
                     for index, call in enumerate(assistant.get("calls") or [])]
            if calls:
                target["tool_calls"] = calls
        reasoned = _reasoned_decision(step, reasoning_steps)
        invocation = invocations.get(step.get('invocation_id')) or {}
        role = ('unknown' if 'parent_invocation_id' not in invocation else
                'root' if invocation['parent_invocation_id'] is None else 'child')
        turns.append({"messages": copy.deepcopy(step["context"]),
                      "tools": copy.deepcopy(step.get("tools_offered") or None), "target": target,
                      "invocation_id": step.get("invocation_id"),
                      'function': invocation.get('name') or 'unknown',
                      'definition_id': invocation.get('definition_id'),
                      'parent_invocation': invocation.get('parent_invocation_id'), 'invocation_role': role,
                      **({'response_tools':copy.deepcopy(step['response_tools']), 'native_xml_turn':copy.deepcopy(step)} if native_xml else {}),
                      **({'sampling': copy.deepcopy(response['rl_sampling'])} if response.get('rl_sampling') else {}),
                      **({'decision_reasoning': reasoned} if reasoned is not None else {}),
                      **({'usage': copy.deepcopy(response['usage'])} if isinstance(response.get('usage'), dict) else
                         {'usage': {'completion_tokens': response['completion_tokens']}}
                         if type(response.get('completion_tokens')) is int else
                         {'usage': copy.deepcopy(step['usage'])} if isinstance(step.get('usage'), dict) else {}),
                      **({'native_state_imports': copy.deepcopy(response['native_state_imports'])}
                         if response.get('native_state_imports') else {}),
                      "replay": {"raw_calls_retained": "raw_calls" in response,
                                 **({"assistant_content_xml_authenticated": True} if native_xml else {}),
                                 "request_sha256": step.get("request_sha256"),
                                 "raw_response_sha256": step.get("raw_response_sha256")}})
    return turns


def _reasoned_decision(step: dict[str, Any], reasoning_steps: dict[tuple[Any, Any], dict[str, Any]]) -> dict[str, Any] | None:
    """A reasoned decision entry's ``decision_reasoning`` turn field (``rollout_turns``), or None for any other step."""
    marker = step.get("decision_reasoning")
    if not isinstance(marker, dict):
        return None
    from ..data.decision_graph import capture_prompt_ids, token_ids_sha256
    out = {key: marker[key] for key in ("index", "samples", "action") if key in marker}
    out["action"] = marker.get("action") is True
    scored = capture_prompt_ids((step.get("model_response") or {}).get("rl_sampling"))
    if scored is not None:
        out["scored_prompt_sha256"] = token_ids_sha256(scored)
    paired = reasoning_steps.pop((step.get("invocation_id"), marker.get("index")), None)
    sampling = ((paired or {}).get("model_response") or {}).get("rl_sampling")
    if isinstance(sampling, dict) and isinstance(sampling.get("samples"), list):
        out["reasoning_sample_ids"] = [int(sample["token_id"]) for sample in sampling["samples"]]
        prompt = capture_prompt_ids(sampling)
        if prompt is not None:
            out["reasoning_prompt_sha256"] = token_ids_sha256(prompt)
        if out["action"]:
            # the action sample's reasoning is part of the sampled action: kept whole for GRPO
            out["reasoning_sampling"] = copy.deepcopy(sampling)
    return out


def execution_efficiency(rollout: dict[str, Any], weight: float | None = None) -> dict[str, Any]:
    """Summarize observed model work and repeated sampled actions for one whole graph.

    Generated-token count stays null unless every policy turn has a captured completion count or an exact native
    sampling-token list. No count is inferred from text length. Wall time, provider latency, and prompt length are
    never part of the cost.
    """
    weight = DEFAULT_EXECUTION_EFFICIENCY_WEIGHT if weight is None else weight
    from .rollout_math import EXECUTION_EFFICIENCY_POLICY
    counters = execution_efficiency_counters(rollout.get('turns') or [])
    cost = execution_efficiency_cost(**counters, weight=weight)
    return {**counters, 'cost': cost, 'weight': float(weight), 'policy': EXECUTION_EFFICIENCY_POLICY,
            'failure_report_credit': grounded_failure_report_credit(rollout.get('turns') or [])}


def _decision_frame(row: dict[str, Any], unmatched: list[Any]) -> dict[str, Any] | None:
    """The trajectory step of a sampled closed-set decision, rebuilt from its `decision_action_frame` evidence row
    exactly as the collector pushes it into a completed episode's trajectory (the readout is the chosen option, no
    tool calls, the native capture as `rl_sampling`). The row must equal an unused entry of the partial's separately
    saved decisions (same request hash and scores), so a decision turn is custody-checked like an action turn."""
    request, scores, request_sha = row.get("request"), row.get("scores"), row.get("request_sha256")
    if (not isinstance(request, dict) or not isinstance(scores, dict) or not isinstance(request_sha, str) or
            not re.fullmatch(r"[0-9a-f]{64}", request_sha) or not isinstance(request.get("messages"), list) or
            not isinstance(request.get("options"), list) or request.get("invocation_id") != row.get("invocation_id") or
            not isinstance(row.get("invocation_id"), str) or not row["invocation_id"]):
        return None
    capture, log_probs = scores.get("native_decision_capture"), scores.get("log_probs")
    if (not isinstance(capture, dict) or not isinstance(log_probs, list) or len(log_probs) != len(request["options"]) or
            not log_probs):
        return None
    chosen = scores.get("chosen")
    if chosen is None:
        chosen = max(range(len(log_probs)), key=lambda i: log_probs[i])
    if type(chosen) is not int or not 0 <= chosen < len(request["options"]) or \
            not isinstance(request["options"][chosen], str):
        return None
    match = next((i for i, saved in enumerate(unmatched) if isinstance(saved, dict) and
                  saved.get("request_sha256") == request_sha and saved.get("scores") == scores), None)
    if match is None:
        return None
    del unmatched[match]
    text = request["options"][chosen]
    imports = capture.get("native_state_imports")
    return {"phase": "action", "invocation_id": row["invocation_id"], "context": copy.deepcopy(request["messages"]),
            "request_sha256": request_sha, "tools_offered": [],
            "model_response": {"calls": [], "text": text, "raw_calls": [], "rl_sampling": copy.deepcopy(capture),
                               **({"native_state_imports": copy.deepcopy(imports)} if isinstance(imports, list) else {})},
            "assistant": {"content": text, "reasoning": capture.get("reasoning") if isinstance(capture.get("reasoning"), str)
                          else None, "calls": []},
            **({"decision_reasoning": copy.deepcopy(row["decision_reasoning"])}
               if isinstance(row.get("decision_reasoning"), dict) else {})}


def _graph_of_turn_prefix(graph: list[dict[str, Any]], kept_names: list[str]) -> list[dict[str, Any]]:
    """The execution graph of an episode cut to its first turns: drop the model turns after the cut, the block
    reads/writes they own, and (transitively) every event that takes one of those as an input."""
    kept = set(kept_names)
    dropped = {event["node"] for event in graph
               if (event.get("kind") == "model_turn" and event["node"] not in kept) or
               (event.get("kind") in {"block_write", "block_read"} and event.get("turn") not in kept)}
    changed = True
    while changed:
        changed = False
        for event in graph:
            if event["node"] not in dropped and any(isinstance(edge, dict) and edge.get("node") in dropped
                                                    for edge in event["inputs"]):
                dropped.add(event["node"])
                changed = True
    return [event for event in graph if event["node"] not in dropped]


def _budget_terminated_prefix(entry: dict[str, Any], task: dict[str, Any], finish: dict[str, Any] | None,
                              *, packet_source_path: Path, packet_source_sha256: str | None
                              ) -> dict[str, Any] | None:
    """Recover only a completely captured, exact model-request-budget prefix.

    The collector has no result row for this terminal path. It does, however,
    durably append the interrupted runtime snapshot and action frames to the
    latest partial-evidence sidecar before writing the bound terminal error.
    An incomplete or ambiguous capture is deliberately not a training sample.
    """
    if not isinstance(finish, dict) or finish.get("resource_limit_reason") != "model_request_budget":
        return None
    if task.get("split") != "train":
        return None
    accounting = finish.get("output_accounting")
    if not isinstance(accounting, dict) or packet_source_sha256 is None:
        return None
    job_index = entry.get("index")
    program_id = task.get("id")
    states = accounting.get("job_states")
    matching_states = [row for row in states or [] if isinstance(row, dict) and
                       type(row.get("index")) is int and row.get("index") == job_index]
    state = matching_states[0] if len(matching_states) == 1 else None
    if (not isinstance(state, dict) or state.get("program_id") != program_id or
            state.get("state") != "failed" or state.get("resource_limit_reason") != "model_request_budget"):
        return None
    jobs = Path(entry["jobs"])
    source_path = Path(entry.get("source", ""))
    try:
        if (source_path.is_symlink() or not source_path.is_file() or
                source_path.resolve(strict=True) != packet_source_path.resolve(strict=True)):
            return None
    except OSError:
        return None
    error_path = jobs / f"{int(job_index):06d}.error.json"
    digest = state.get("digest")
    if not isinstance(digest, str) or len(digest) != 64 or not all(c in "0123456789abcdef" for c in digest):
        return None
    partial_path = jobs / f"{int(job_index):06d}-{digest[:16]}.partial.json"
    try:
        error_bytes = error_path.read_bytes()
        error = json.loads(error_bytes)
        partial = json.loads(partial_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    source_sha = accounting.get("source_sha256")
    terminal_error = state.get("terminal_error") or {}
    if (terminal_error.get("sha256") != hashlib.sha256(error_bytes).hexdigest() or
            Path(str(terminal_error.get("path", ""))).name != error_path.name or
            terminal_error.get("error") != str(error.get("error", "")) or
            error.get("index") != job_index or error.get("program_id") != program_id or
            error.get("code") != "NATLANG_MODEL_REQUEST_BUDGET" or
            source_sha != packet_source_sha256 or error.get("source_sha256") != packet_source_sha256 or
            terminal_error.get("source_sha256") != packet_source_sha256 or
            terminal_error.get("record_binding_verified") is not True or
            error.get("program_ir_sha256") != state.get("digest") or
            (state.get("terminal_error") or {}).get("program_ir_sha256") != state.get("digest") or
            partial.get("version") != "natlang.teacher_partial.native/1" or partial.get("program_id") != program_id or
            (partial.get("provenance") or {}).get("program_ir_sha256") != state.get("digest") or
            not isinstance(partial.get("execution_attempt_id"), str)):
        return None

    snapshots = partial.get("evidence_snapshots")
    if not isinstance(snapshots, list) or not snapshots:
        return None
    request_attempt_ids = partial.get("request_attempt_ids")
    if (not isinstance(request_attempt_ids, list) or not request_attempt_ids or
            not all(isinstance(value, str) and value for value in request_attempt_ids)):
        return None
    manifest = snapshots[-1]
    if (not isinstance(manifest, dict) or manifest.get("schema") != "natlang.teacher_partial_evidence/1" or
            manifest.get("status") != "execution_interrupted" or
            manifest.get("attempt_id") != request_attempt_ids[-1] or
            not isinstance(manifest.get("path"), str) or Path(manifest["path"]).name != manifest["path"]):
        return None
    evidence_path = jobs / manifest["path"]
    try:
        if evidence_path.is_symlink() or not evidence_path.is_file():
            return None
        evidence_bytes = evidence_path.read_bytes()
    except OSError:
        return None
    if (len(evidence_bytes) != manifest.get("bytes") or
            hashlib.sha256(evidence_bytes).hexdigest() != manifest.get("sha256")):
        return None
    try:
        records = [json.loads(line) for line in evidence_bytes.splitlines() if line]
    except (ValueError, TypeError):
        return None
    if (len(records) != manifest.get("records") or
            any(row.get("version") != "natlang.teacher_partial_evidence_record/1" or
                row.get("attempt_id") != manifest.get("attempt_id") or row.get("sequence") != index
                for index, row in enumerate(records))):
        return None
    payloads = [row.get("payload") for row in records]
    snapshots = [row for row in payloads if isinstance(row, dict) and row.get("kind") == "execution_snapshot"]
    if len(snapshots) != 1:
        return None
    snapshot = snapshots[0]
    if (snapshot.get("schema") != "natlang.partial_execution_snapshot/1" or snapshot.get("status") != "interrupted" or
            snapshot.get("execution_attempt_id") != partial["execution_attempt_id"] or
            not isinstance(snapshot.get("logical_run_id"), str) or not snapshot["logical_run_id"] or
            snapshot.get("run_id") != f"{snapshot['logical_run_id']}/execution-{partial['execution_attempt_id']}" or
            snapshot.get("pending_children") is not False or
            not isinstance(snapshot.get("failure_reason"), str) or
            not re.fullmatch(r"whole-case model request budget exceeded \([1-9][0-9]*\)",
                             snapshot["failure_reason"])):
        return None
    if sum(row.get("kind") == "execution_snapshot" for row in payloads if isinstance(row, dict)) != 1:
        return None
    # A record over the collector's byte limit is replaced by a `record_omitted` marker. Losing a provider
    # exchange is harmless (nothing below reads it); losing a turn's action/decision frame means the episode's
    # turns are captured only up to that point, so it is trained on the turns before it (a chronological prefix
    # is import-closed: a consumer never precedes its producer). Any other omission (trace, snapshot) is ambiguous.
    omitted = [(index, row) for index, row in enumerate(payloads) if isinstance(row, dict) and
               row.get("kind") == "record_omitted"]
    if any(row.get("original_kind") not in {"provider_exchange", "teacher_action_frame", "decision_action_frame"}
           for _, row in omitted):
        return None
    cut = min((index for index, row in omitted if row.get("original_kind") != "provider_exchange"), default=None)
    kept_payloads = [(index, row) for index, row in enumerate(payloads) if cut is None or index < cut]
    # A replayed (resumed) turn is the same canonical `teacher_action_frame` as a fresh one (source `journal_replay`,
    # bound to its saved turn index). Older collectors wrote it as a `provider_frame` and wrote no frame for a replayed
    # decision; an attempt with that record holds only part of its episode, so it is never a training sample.
    if any(isinstance(row, dict) and row.get("kind") == "provider_frame" for _, row in kept_payloads):
        return None
    all_action_frames = [row for _, row in kept_payloads if isinstance(row, dict) and
                         row.get("kind") == "teacher_action_frame"]
    policy_sources = {"live_provider", "journal_replay", "handoff_replay"}
    known_sources = policy_sources | {"authored_reference_root"}
    if any(row.get("source") not in known_sources for row in all_action_frames):
        return None
    saved_decisions = partial.get("decisions")
    unmatched_decisions = list(saved_decisions) if isinstance(saved_decisions, list) else []
    ordered_frames: list[tuple[dict[str, Any] | None, dict[str, Any]]] = []  # (teacher action row or None, frame)
    for _, row in kept_payloads:
        if not isinstance(row, dict):
            continue
        if row.get("kind") == "teacher_action_frame" and row.get("source") in policy_sources | {"authored_reference_root"}:
            ordered_frames.append((row, row.get("response_frame")))
        elif row.get("kind") == "decision_action_frame":
            frame = _decision_frame(row, unmatched_decisions)
            if frame is None:
                return None
            ordered_frames.append((None, frame))
    frames = [frame for row, frame in ordered_frames if row is None or row.get("source") in policy_sources]
    verified_action_frames = all_action_frames
    verified_frames = [frame for row, frame in ordered_frames]
    if (not frames or any(not isinstance(frame, dict) or frame.get("phase") != "action" or
                          not isinstance(frame.get("invocation_id"), str) or
                          not isinstance(frame.get("context"), list) or
                          not isinstance(frame.get("model_response"), dict) or
                          "raw_calls" not in frame["model_response"] for frame in verified_frames)):
        return None
    saved_turns = partial.get("turns")
    if not isinstance(saved_turns, list):
        return None
    saved_indexes = []
    for evidence_row, frame in zip(all_action_frames, [frame for row, frame in ordered_frames if row is not None]):
        index = evidence_row.get("saved_turn_index")
        if (type(index) is not int or index < 0 or index >= len(saved_turns) or
                index in saved_indexes or not isinstance(evidence_row.get("request_sha256"), str) or
                not re.fullmatch(r"[0-9a-f]{64}", evidence_row["request_sha256"])):
            return None
        saved_indexes.append(index)
        saved = saved_turns[index]
        response = saved.get("response") if isinstance(saved, dict) else None
        model_response = frame.get("model_response")
        if (not isinstance(response, dict) or saved.get("request_sha256") != evidence_row.get("request_sha256") or
                saved.get("invocation_id") != evidence_row.get("invocation_id") or
                response.get("calls") != model_response.get("calls") or
                # The native collector's durable turn envelope stores transport metadata under raw_response;
                # the exact raw call payload is retained in the separately hashed action frame above.
                ("raw_calls" in response and response.get("raw_calls") != model_response.get("raw_calls")) or
                not isinstance(model_response.get("raw_calls"), list) or
                response.get("text", "") != model_response.get("text", "")):
            return None

    trace_rows = [row for row in payloads if isinstance(row, dict) and row.get("kind") == "trace_event"]
    invocation_rows = [row for row in payloads if isinstance(row, dict) and row.get("kind") == "invocation_snapshot"]
    if (sum(row.get("trace_role") == "root" for row in trace_rows) != snapshot.get("root_event_count") or
            len(invocation_rows) != snapshot.get("invocation_count")):
        return None
    call_ids = [row.get("call_id") for row in invocation_rows]
    if (any(not isinstance(call_id, str) or not call_id for call_id in call_ids) or len(call_ids) != len(set(call_ids)) or
            any((row.get("trace_role") == "root" and row.get("invocation_id") != snapshot.get("run_id")) or
                (row.get("trace_role") == "child" and row.get("invocation_id") not in call_ids) or
                row.get("trace_role") not in {"root", "child"} for row in trace_rows)):
        return None
    for invocation in invocation_rows:
        events = [row for row in trace_rows if row.get("trace_role") == "child" and
                  row.get("invocation_id") == invocation.get("call_id")]
        if len(events) != invocation.get("event_count"):
            return None
    graph = [row["event"] for row in trace_rows if isinstance(row.get("event"), dict) and
             isinstance(row["event"].get("node"), str) and isinstance(row["event"].get("inputs"), list)]
    reference_roots = [row for row in verified_action_frames if row.get("source") == "authored_reference_root"]
    if reference_roots:
        # An authored root is harness reference material, not a sampled policy turn. Keep its
        # invocation node as the exact caller edge, but never detach its model/state dependencies
        # from the child's trainable graph.
        if len(reference_roots) != 1:
            return None
        reference = reference_roots[0]
        root_invocation = reference.get("invocation_id")
        if (root_invocation != snapshot.get("run_id") or
                reference.get("response_frame", {}).get("invocation_id") != root_invocation):
            return None
        root_events = [row["event"] for row in trace_rows if row.get("trace_role") == "root" and
                       isinstance(row.get("event"), dict) and isinstance(row["event"].get("node"), str) and
                       isinstance(row["event"].get("inputs"), list)]
        root_turns = [event for event in root_events if event.get("kind") == "model_turn"]
        root_call_node = f"call:{root_invocation}"
        root_calls = [event for event in root_events if event.get("kind") == "invocation" and
                      event.get("node") == root_call_node]
        if (len(root_turns) != 1 or root_turns[0].get("node") != f"{root_invocation}#turn1" or
                len(root_calls) != 1 or any(event.get("kind") in {"block_write", "block_read"}
                                           for event in root_events)):
            return None
        child_events = [row["event"] for row in trace_rows if row.get("trace_role") == "child" and
                        isinstance(row.get("event"), dict) and isinstance(row["event"].get("node"), str) and
                        isinstance(row["event"].get("inputs"), list)]
        for event in child_events:
            for edge in event["inputs"]:
                if isinstance(edge, dict) and edge.get("node") == root_turns[0]["node"]:
                    return None
                if isinstance(edge, dict) and edge.get("node") == root_call_node and not (
                        event.get("kind") == "invocation" and edge.get("port") == "caller"):
                    return None
        graph = child_events + root_calls
    trajectory = {"trajectory": frames}
    turns = rollout_turns(trajectory)
    if not turns:
        return None
    try:
        from .episode_graph import rollout_turn_names, validate_rollout_graph
        if cut is not None:
            graph = _graph_of_turn_prefix(graph, rollout_turn_names(turns))
        validate_rollout_graph({"turns": turns, "execution_graph": graph})
    except (TypeError, ValueError):
        return None
    return {"turns": turns, "trajectory": frames, "execution_graph": graph,
            "evidence": {"path": str(evidence_path), "sha256": manifest["sha256"],
                         "attempt_id": manifest["attempt_id"], "failure_reason": snapshot["failure_reason"],
                         **({"truncated_at_omitted_record": {"record_index": cut, "kept_turns": len(turns)}}
                            if cut is not None else {})}}


def memetic_outcome(task: dict[str, Any], result: dict[str, Any] | None) -> dict[str, Any] | None:
    """The graded outcome of a memetic improvement episode (``natlang.memetic-outcome/1`` in the world score the
    collector records), or None for every other task. An episode that ended without one (a crash or budget stop before
    scoring) is a failed improvement, scored 0 against its group's anchor (the group's shared preparation)."""
    world = (task.get("semantics") or {}).get("world") or {}
    if world.get("kind") != "memetic" or not result:
        return None
    memetic = ((result.get("outcome") or {}).get("world") or {}).get("memetic")
    if isinstance(memetic, dict) and memetic.get("schema") == "natlang.memetic-outcome/1":
        quality = memetic.get("quality")
        if type(quality) not in (int, float) or not 0 <= quality <= 1:
            raise ValueError("memetic outcome quality must be in [0, 1]")
        return memetic
    anchor = 0.
    try:
        spec = json.loads(Path(world["task"]).read_text())
        out = Path(spec["out"])
        # the staged layout keeps the anchor in baselines.json; a preparation before 2026-11-23 in prepared.json
        prepared = json.loads((out / "prepared.json").read_text()) if (out / "prepared.json").exists() else {}
        baselines = (json.loads((out / "baselines.json").read_text()) if (out / "baselines.json").exists()
                     else prepared.get("baselines") or {})
        anchor = max([item["success"] for item in baselines.values() if item and item.get("success") is not None],
                     default=0.)
    except (OSError, KeyError, TypeError, ValueError):
        pass
    return {"schema": "natlang.memetic-outcome/1", "quality": 0., "anchor": anchor, "gain": -anchor, "cost": 0.,
            "query_success": 0., "unscored_workspace": True}


def memetic_reward_record(memetic: dict[str, Any]) -> dict[str, Any]:
    """The reward fields the learner reads (grpo.py ``episode_advantages``): the trivial-improvement anchor and the
    measured gain, cost and query success, with where the committed set and its query runs are. ``without`` and
    ``default`` are the same query worlds and seeds with no guidance and with the mechanical default (paired; None when
    preparation did not run)."""
    baselines = memetic.get("baselines") or {}
    arm = lambda name: (baselines.get(name) or {}).get("success")
    return {"anchor": float(memetic["anchor"]), "gain": float(memetic["gain"]), "cost": float(memetic["cost"]),
            "without": arm("none"), "default": arm("default"),
            "query_success": float(memetic["query_success"]), "query_dir": memetic.get("query_dir"),
            "committed": memetic.get("committed"), "unscored_workspace": bool(memetic.get("unscored_workspace")),
            "leak_refused": bool(memetic.get("leak_refused")), "score_path": memetic.get("score_path")}


def memetic_success(memetic: dict[str, Any] | None) -> bool:
    """Whether a memetic improvement episode is a verified success (its own success for expert iteration): it
    committed a change set, the workspace scored it on the private query worlds (never shown to the supervisor;
    separate from the support worlds it was judged on while working), its texts reproduce no query answer, and its
    quality (query success minus the evaluation cost) beats the best trivial improvement (``anchor``: no change or the
    mechanical default on the same worlds and seeds), the condition under which GRPO gives it a positive advantage.
    ``memetic``: a ``natlang.memetic-outcome/1`` record (``memetic_outcome``)."""
    if not memetic or memetic.get("unscored_workspace") or memetic.get("leak_refused"):
        return False
    if not memetic.get("committed"):
        return False
    quality, anchor = memetic.get("quality"), memetic.get("anchor")
    return type(quality) in (int, float) and type(anchor) in (int, float) and quality > anchor


def build_queue(packet: Path, rows: int, group: int, out: Path, *, temperature: float, case_seconds: int,
                max_model_requests: int, max_turns: int, seed_base: int) -> list[dict[str, Any]]:
    if group < 2:
        raise ValueError("a GRPO group needs at least two rollouts")
    if temperature <= 0:
        raise ValueError("group rollouts need temperature > 0; greedy samples are identical")
    packet_tasks = read_jsonl(packet) if packet.is_file() else []
    packet_sha256 = hashlib.sha256(packet.read_bytes()).hexdigest() if packet.is_file() else None
    if packet_tasks and len(packet_tasks) != rows:
        raise ValueError("queue row count must match the authenticated task packet")
    entries = []
    retries = actor_request_retries()
    for index in range(rows):
        # A memetic improvement episode runs its program on support (and, at the end, query) worlds inside the
        # episode; its world declares the wall budget that needs (rl.memetic_tasks).
        world = ((packet_tasks[index].get("semantics") or {}).get("world") or {}) if packet_tasks else {}
        seconds = max(case_seconds, int(world.get("case_seconds") or 0)) if world.get("kind") == "memetic" else case_seconds
        for sample in range(group):
            case = out / f"row-{index:03d}" / f"sample-{sample:02d}"
            entries.append({
                "key": f"s7-task-{index:03d}-{sample:02d}", "index": index, "count": 1, "source": str(packet),
                "seed": seed_base + 1000 * index + sample, "temperature": temperature,
                "case_seconds": seconds, "max_model_requests": max_model_requests, "max_turns": max_turns,
                "request_retries": retries, "jobs": str(case / "jobs"), "log": str(case / "collector.log"), "output": str(case / "results.jsonl"),
                "group": index, "sample": sample,
                **({"task_id": packet_tasks[index]["id"], "packet_sha256": packet_sha256}
                   if packet_tasks else {}),
            })
    return entries


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    source = read_jsonl(args.cases)
    root_mode = getattr(args, "root_mode", "authored")
    held = bool(getattr(args, "held_rows", False))
    if held and root_mode != "student":
        raise ValueError("held rows are measured with the student at the root")
    transform = authored_root_row if root_mode == "authored" else (lambda row: student_root_row(row, held=held))
    selected = [transform(row) for row in source[args.offset:args.offset + args.rows]]
    packet = out / "packet.jsonl"
    packet.write_text("".join(json_line(row) for row in selected), encoding="utf-8")
    entries = build_queue(packet, len(selected), args.group, out, temperature=args.temperature,
                          case_seconds=args.case_seconds, max_model_requests=args.max_model_requests,
                          max_turns=args.max_turns, seed_base=args.seed_base)
    preparation = {"schema": "natlang.task-rollout-preparation/1", "root_mode": root_mode,
                   "source_cases": str(args.cases.resolve()),
                   "source_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
                   "packet_sha256": hashlib.sha256(packet.read_bytes()).hexdigest()}
    policy_id = getattr(args, 'policy_checkpoint_sha256', None)
    if policy_id:
        if len(policy_id) != 64 or any(c not in '0123456789abcdef' for c in policy_id):
            raise ValueError('policy checkpoint must be a SHA-256 digest')
        preparation['policy_checkpoint_sha256'] = policy_id
    seed_reference = getattr(args, "match_seeds_from", None)
    if seed_reference is not None:
        reference = seed_reference.resolve()
        reference_packet = read_jsonl(reference / "packet.jsonl")
        reference_queue = read_jsonl(reference / "queue.jsonl")
        by_task_sample = {}
        for entry in reference_queue:
            key = (reference_packet[int(entry["index"])]["id"], int(entry["sample"]))
            if key in by_task_sample:
                raise ValueError(f"reference seed is ambiguous for {key}")
            by_task_sample[key] = int(entry["seed"])
        for entry in entries:
            key = (selected[int(entry["index"])]["id"], int(entry["sample"]))
            if key not in by_task_sample:
                raise ValueError(f"reference has no matching task and sample: {key}")
            entry["seed"] = by_task_sample[key]
        preparation["matched_seed_queue"] = str(reference / "queue.jsonl")
        preparation["matched_seed_queue_sha256"] = hashlib.sha256((reference / "queue.jsonl").read_bytes()).hexdigest()
    (out / "preparation.json").write_text(json.dumps(preparation, indent=2) + "\n", encoding="utf-8")
    (out / "queue.jsonl").write_text("".join(json.dumps(e, sort_keys=True) + "\n" for e in entries), encoding="utf-8")
    return {"packet": str(packet), "root_mode": root_mode, "rows": len(selected), "group": args.group, "queue_entries": len(entries),
            "source_cases": str(args.cases), "ids": [row["id"] for row in selected]}


def score(run_dir: Path, emit: Path | None = None, round_: int = 0,
          neuralese_bonus: float | None = None,
          efficiency_weight: float | None = None) -> dict[str, Any]:
    """Classify every rollout; with `emit`, also write `natlang.rollout/1` records for `grpo.py round`.

    ``neuralese_bonus`` (None: DEFAULT_NEURALESE_USE_BONUS; 0 opts out) is added as ``reward.neuralese_bonus`` to a
    successful student episode with at least one real between-call Neuralese edge (`neuralese_exchanges`). The
    ``quality`` field stays the binary task outcome; learners use ``rollout_math.rollout_return``."""
    bonus = neuralese_use_bonus(neuralese_bonus)
    efficiency_weight = (DEFAULT_EXECUTION_EFFICIENCY_WEIGHT if efficiency_weight is None else efficiency_weight)
    # Validate at the boundary, including score runs with only unscored infrastructure outcomes.
    execution_efficiency_cost(model_calls=0, generated_tokens=None, repeated_actions=0, weight=efficiency_weight)
    run_dir = run_dir.resolve()
    emitted: list[dict[str, Any]] = []
    packet_path = run_dir / "packet.jsonl"
    packet = read_jsonl(packet_path)
    packet_bytes = packet_path.read_bytes()
    packet_sha256 = hashlib.sha256(packet_bytes).hexdigest()
    packet_lines = packet_bytes.splitlines(keepends=True)
    if len(packet_lines) != len(packet) or any(not line.endswith(b"\n") for line in packet_lines):
        raise ValueError("task packet must have one newline-terminated JSONL record per row")
    preparation_path = run_dir / "preparation.json"
    preparation = json.loads(preparation_path.read_text()) if preparation_path.exists() else {"root_mode": "authored"}
    root_mode = preparation["root_mode"]
    if emit is not None and preparation.get("packet_sha256") != packet_sha256:
        raise ValueError("scored rollout export requires a packet hash matching its preparation receipt")
    if root_mode == "student":
        for index, task in enumerate(packet):
            if ((task.get("curriculum") or {}).get("trajectory_contract") or {}).get("authored_root_eval"):
                raise ValueError(f"student-root packet row {index} still requests authored-root replay")
    entries = read_jsonl(run_dir / "queue.jsonl")
    journal_path = run_dir / "journal.jsonl"
    journal = read_jsonl(journal_path) if journal_path.exists() else []
    finishes = {row["key"]: row for row in journal if row.get("event") == "finish"}
    rollouts = []
    for entry in entries:
        output = present(entry["output"])     # raw or compacted once scored (episode_files.compact_results)
        results = read_jsonl(output) if output else []
        result = results[0] if results else None
        if root_mode == "student" and result:
            for turn in result.get("trajectory") or []:
                if (turn.get("action_provenance") or {}).get("kind") == "authored_reference_root_eval":
                    raise ValueError(f"student-root rollout {entry['key']} replayed an authored root")
        outcome = classify_case(result, finishes.get(entry["key"]))
        reward = reward_of(outcome, finishes.get(entry["key"]))
        expected_packet_sha256 = packet_sha256 if preparation.get("packet_sha256") == packet_sha256 else None
        budget_prefix = _budget_terminated_prefix(
            entry, packet[entry["index"]], finishes.get(entry["key"]),
            packet_source_path=packet_path, packet_source_sha256=expected_packet_sha256) \
            if result is None and outcome == "budget_exhaustion" and \
            (finishes.get(entry["key"]) or {}).get("resource_limit_reason") == "model_request_budget" else None
        task = packet[entry["index"]]
        memetic = memetic_outcome(task, result)
        if memetic is not None and memetic.get("unscored_workspace") and reward is None:
            memetic = None   # no workspace score and an unresolved run (wall clock, infrastructure): no measurement
        if memetic is not None:
            # An improvement episode's outcome is graded: its committed change set's success on held query worlds
            # (rl.memetic_workspace.Workspace.score), anchored on the best trivial improvement.
            outcome, reward = "memetic_scored", memetic["quality"]
        graded = memetic["quality"] if memetic is not None else graded_reward(outcome, reward, result, task.get("semantics"))
        if emit is not None and (entry.get("task_id") != task.get("id") or
                                 entry.get("packet_sha256") != packet_sha256):
            raise ValueError(f"queue entry {entry.get('key')} is not bound to its exact task packet row")
        detail = ((result or {}).get("outcome") or {}).get("detail") if result else None
        turns = budget_prefix["turns"] if budget_prefix else rollout_turns(result)
        execution_graph = budget_prefix["execution_graph"] if budget_prefix else \
            copy.deepcopy(((result or {}).get("outcome") or {}).get("execution_graph") or [])
        # A run without RL sampling capture (held-out evaluation) is measured over its execution graph alone.
        captured = any('sampling' in turn for turn in turns)
        exchanges = neuralese_exchanges({"turns": turns, "execution_graph": execution_graph},
                                        captured=captured) if turns else 0
        efficiency = execution_efficiency({'turns': turns}, efficiency_weight)
        student = entry.get("jolt_role", "student") == "student"
        # Privileged JOLT+ teacher attempts import harness-placed guidance; only the student policy's own
        # between-call state passing is rewarded, and only when its capture binds the graph.
        episode_bonus = bonus if captured and student and reward == 1.0 and exchanges > 0 and bonus > 0 else 0.
        if emit is not None:
            training_reward = reward
            if result is None and reward is not None and budget_prefix is None:
                # Preserve the task-level failure score in this report, but never
                # offer an empty or unverified partial as a native policy sample.
                training_reward = None
            source_custody = _packet_source_custody(
                task, packet_sha256=packet_sha256, row_index=int(entry["index"]),
                row_bytes=packet_lines[int(entry["index"])])
            emitted.append({"schema": "natlang.rollout/1", "round": round_, "task_id": task["id"],
                            "family": task.get("family"), "group": entry["group"], "sample": entry["sample"],
                            "seed": entry["seed"], "split": task.get("split"),
                            "source_ids": source_custody["source_ids"],
                            "source_groups": source_custody["source_groups"],
                            "source_revisions": source_custody["source_revisions"],
                            "source_custody": source_custody,
                            **({"jolt_role": entry["jolt_role"]} if "jolt_role" in entry else {}),
                            "policy": {"temperature": entry.get("temperature"),
                                "checkpoint_sha256": preparation.get("policy_checkpoint_sha256"),
                                "endpoint": entry.get("rl_endpoint")},
                            # the runtime that produced the turns: a later re-render or replay knows what it changes
                            "code": code_provenance(),
                            "model_facing": os.environ.get("NATLANG_MODEL_FACING") or None,
                            "turns": turns, "execution_graph": execution_graph,
                            "outcome": outcome,
                            **({"memetic_success": memetic_success(memetic)} if memetic is not None else {}),
                            "reward": None if training_reward is None else {"quality": training_reward,
                                "graded": graded,
                                **({"neuralese_bonus": episode_bonus} if episode_bonus else {}),
                                **({"memetic": memetic_reward_record(memetic)} if memetic is not None else {}),
                                "execution_efficiency": efficiency},
                            "unscored": ("budget_prefix_capture_unavailable" if result is None and reward is not None and
                                         budget_prefix is None else outcome if reward is None else None),
                            **({"budget_termination_evidence": budget_prefix["evidence"]} if budget_prefix else {})})
        rollouts.append({"task_id": task["id"], "family": task.get("family"), "group": entry["group"],
                         "sample": entry["sample"], "seed": entry["seed"], "outcome": outcome,
                         "reward": reward, "graded_reward": graded,
                         "detail": detail if isinstance(detail, str) else None,
                         "finish_status": (finishes.get(entry["key"]) or {}).get("status"),
                         "neuralese_exchanges": exchanges,
                         "execution_efficiency": efficiency,
                         **({"neuralese_bonus": episode_bonus} if episode_bonus else {}),
                         **({"memetic_success": memetic_success(memetic)} if memetic is not None else {}),
                         **({"jolt_role": entry["jolt_role"]} if "jolt_role" in entry else {})})
    groups = []
    for index, task in enumerate(packet):
        # Group statistics and routes describe the student policy; privileged teacher attempts are reported apart.
        members = [r for r in rollouts if r["group"] == index and r.get("jolt_role", "student") == "student"]
        advantages, stats = group_advantages([r["reward"] for r in members])
        for member, advantage in zip(members, advantages):
            member["advantage"] = advantage
        scored = [r["reward"] for r in members if r["reward"] is not None]
        efficiency_costs = [r['execution_efficiency']['cost'] for r in members if r['reward'] is not None]
        efficiency_spread = bool(scored) and all(reward == 1.0 for reward in scored) \
            and len(set(efficiency_costs)) > 1
        reporting_varies = reporting_spread([{'quality': r['reward'], 'execution_efficiency': r['execution_efficiency']}
                                            for r in members if r['reward'] is not None])
        efficiency_spread = efficiency_spread or reporting_varies
        learning_signal = stats['spread'] or efficiency_spread
        if learning_signal:
            route = "grpo"
        elif not scored:
            route = "unscored"
        elif scored[0] == 1.0:
            route = "retire"
        else:
            route = "rejection-sampling"
        groups.append({"task_id": task["id"], "route": route, **stats,
                       "efficiency_spread": efficiency_spread, "reporting_spread": reporting_varies,
                       "outcomes": sorted(r["outcome"] for r in members)})
    counts: dict[str, int] = {}
    for r in rollouts:
        counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1
    report = {"schema": "natlang.s7-task-rollout-score/1", "root_mode": root_mode,
              "classification": classification_binding(),
              "rollouts": len(rollouts), "outcome_counts": counts,
              "scored": sum(r["reward"] is not None for r in rollouts),
              "successes": sum(r["reward"] == 1.0 for r in rollouts),
              "groups_with_spread": sum(g["spread"] for g in groups),
              "groups_with_quality_spread": sum(g["spread"] for g in groups),
              "groups_with_efficiency_spread": sum(g["efficiency_spread"] for g in groups),
              "groups_with_reporting_spread": sum(g["reporting_spread"] for g in groups),
              "neuralese": neuralese_metrics(rollouts, bonus),
              "execution_efficiency": efficiency_metrics(rollouts, efficiency_weight), "groups": groups}
    (run_dir / "rewards.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rollouts), encoding="utf-8")
    if emit is not None:
        emit.write_text("".join(json_line(r) for r in emitted), encoding="utf-8")
        from .episode_files import drop_stale_packed
        drop_stale_packed(emit)     # a group scored again: its compacted rollouts (a retention pass's) are stale
        report["emitted_rollouts"] = str(emit)
    (run_dir / "score.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def neuralese_metrics(rollouts: list[dict[str, Any]], bonus: float) -> dict[str, Any]:
    """Per-round Neuralese-use telemetry over scored student episodes (teacher attempts are excluded)."""
    scored = [r for r in rollouts if r["reward"] is not None and r.get("jolt_role", "student") == "student"]
    correct = [r for r in scored if r["reward"] == 1.0]
    using = [r for r in scored if r.get("neuralese_exchanges", 0) > 0]
    correct_using = [r for r in correct if r.get("neuralese_exchanges", 0) > 0]
    awarded = [r.get("neuralese_bonus", 0.) for r in rollouts if r.get("neuralese_bonus")]
    return {"bonus": bonus, "bonus_episodes": len(awarded), "bonus_total": sum(awarded),
            "scored_episodes": len(scored), "episodes_using_neuralese": len(using),
            "neuralese_use_rate": len(using) / len(scored) if scored else 0.,
            "correct_episodes": len(correct), "correct_using_neuralese": len(correct_using),
            "correct_neuralese_rate": len(correct_using) / len(correct) if correct else 0.,
            "exchanges": sum(r.get("neuralese_exchanges", 0) for r in scored)}


def efficiency_metrics(rollouts: list[dict[str, Any]], weight: float) -> dict[str, Any]:
    """Public accounting for shaped cost and observed counters; unknown token totals stay unknown."""
    values = [r['execution_efficiency'] for r in rollouts if r.get('reward') is not None]
    known_tokens = [row['generated_tokens'] for row in values if row['generated_tokens'] is not None]
    return {'weight': float(weight), 'max_cost': 0.05, 'scored_episodes': len(values),
            'total_cost': sum(row['cost'] for row in values),
            'mean_cost': sum(row['cost'] for row in values) / len(values) if values else 0.,
            'model_calls': sum(row['model_calls'] for row in values),
            'generated_tokens': sum(known_tokens) if len(known_tokens) == len(values) else None,
            'generated_tokens_known_episodes': len(known_tokens),
            'repeated_actions': sum(row['repeated_actions'] for row in values)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--cases", type=Path, required=True, help="train-split natlang.program/2 task rows")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--rows", type=int, default=8)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--group", type=int, default=4)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--case-seconds", type=int, default=600)
    p.add_argument("--max-model-requests", type=int, default=32)
    p.add_argument("--max-turns", type=int, default=20)
    p.add_argument("--seed-base", type=int, default=7000)
    p.add_argument("--policy-checkpoint-sha256", help="full checkpoint actually served for this rollout queue")
    p.add_argument("--match-seeds-from", type=Path,
                   help="copy seeds by task ID and sample from an existing prepared run; fail on missing or ambiguous matches")
    p.add_argument("--root-mode", choices=("authored", "student"), default="authored",
                   help="authored replays the reference root and samples children; student samples root and children")
    s = sub.add_parser("score")
    s.add_argument("--run-dir", type=Path, required=True)
    s.add_argument("--emit-rollouts", type=Path, default=None, help="write natlang.rollout/1 records here")
    s.add_argument("--round", type=int, default=0)
    s.add_argument("--neuralese-use-bonus", type=float, default=None,
                   help=f"bonus for correct episodes that pass Neuralese state between calls "
                        f"(default {DEFAULT_NEURALESE_USE_BONUS}; 0 opts out)")
    s.add_argument("--efficiency-weight", type=float, default=None,
                   help=f"bounded whole-graph execution-efficiency cost weight (default "
                        f"{DEFAULT_EXECUTION_EFFICIENCY_WEIGHT}; 0 opts out)")
    args = parser.parse_args(argv)
    result = prepare(args) if args.command == "prepare" else score(args.run_dir, args.emit_rollouts, args.round,
                                                                   args.neuralese_use_bonus, args.efficiency_weight)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
