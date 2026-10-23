"""S7 task-success rollouts: reward mapping, train-only derivation, sampled queue and spread routing."""
import hashlib
import json
from pathlib import Path

import pytest

from natlang_neuralese.rl.task_rollouts import (_budget_terminated_prefix, authored_root_row, build_queue,
                                               reward_of, score, student_root_row)


def _row(split="train", code="return await f(task);"):
    return {"id": f"row-{split}", "split": split, "family": "fam",
            "curriculum": {"reference": {"root": [["eval", {"code": code}], ["return_result", {}]]}}}


def test_reward_is_binary_task_success_and_infrastructure_is_unscored():
    assert reward_of("task_success") == 1.0
    for outcome in ("invalid_action", "repetition", "budget_exhaustion", "model_incomplete", "semantic_mismatch"):
        assert reward_of(outcome) == 0.0
    assert reward_of("infrastructure_error") is None and reward_of("unresolved") is None
    assert reward_of("budget_exhaustion", {"resource_limit_reason": "case_wall_budget"}) is None
    assert reward_of("budget_exhaustion", {"resource_limit_reason": "model_request_budget"}) == 0.0


def test_authored_root_derivation_is_train_only_and_hash_bound():
    derived = authored_root_row(_row())
    contract = derived["curriculum"]["trajectory_contract"]
    assert contract["authored_root_eval"] is True
    assert contract["authored_root_sha256"] == hashlib.sha256(b"return await f(task);").hexdigest()
    with pytest.raises(ValueError, match="non-train"):
        authored_root_row(_row(split="test"))


def test_student_root_derivation_refuses_reference_replay_and_non_train_rows():
    authored = authored_root_row(_row())
    student = student_root_row(authored)
    assert student["curriculum"]["reference"] == authored["curriculum"]["reference"]
    assert not student["curriculum"]["trajectory_contract"].get("authored_root_eval")
    assert "authored_root_sha256" not in student["curriculum"]["trajectory_contract"]
    with pytest.raises(ValueError, match="non-train"):
        student_root_row(_row(split="test"))


def test_queue_samples_distinct_seeds_and_refuses_greedy_groups(tmp_path):
    entries = build_queue(tmp_path / "p.jsonl", 2, 3, tmp_path, temperature=1.0, case_seconds=60,
                          max_model_requests=8, max_turns=4, seed_base=10)
    assert len(entries) == 6 and len({e["seed"] for e in entries}) == 6 and len({e["key"] for e in entries}) == 6
    assert all(e["count"] == 1 and e["temperature"] == 1.0 for e in entries)
    with pytest.raises(ValueError):
        build_queue(tmp_path / "p.jsonl", 1, 3, tmp_path, temperature=0.0, case_seconds=60,
                    max_model_requests=8, max_turns=4, seed_base=0)


def test_score_routes_groups_by_spread(tmp_path):
    packet = [authored_root_row(_row()), {**authored_root_row(_row()), "id": "row-2"}]
    (tmp_path / "packet.jsonl").write_text("".join(json.dumps(r) + "\n" for r in packet))
    entries = build_queue(tmp_path / "packet.jsonl", 2, 2, tmp_path, temperature=1.0, case_seconds=60,
                          max_model_requests=8, max_turns=4, seed_base=0)
    (tmp_path / "queue.jsonl").write_text("".join(json.dumps(e) + "\n" for e in entries))
    done = {"outcome": {"status": "done", "accepted": True}}
    failed = {"outcome": {"status": "failed", "detail": "model_incomplete: terminated_without_result"}}
    for entry, result in zip(entries, [done, failed, failed, failed]):
        output = tmp_path / entry["output"]
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result) + "\n")
    report = score(tmp_path)
    routes = [g["route"] for g in report["groups"]]
    assert routes == ["grpo", "rejection-sampling"]
    assert report["successes"] == 1 and report["groups_with_spread"] == 1


def _budget_prefix_fixture(tmp_path, *, graph_turn="inv#turn1", pending=False, manifest_tamper=False,
                           native_pair=False, stale_attempt=False, missing_frame=False, split="train",
                           saved_raw_calls=False, mismatched_saved_raw_calls=False, root_dependency=False):
    from natlang_neuralese.rl.task_rollouts import build_queue
    task = {**authored_root_row(_row()), "id": "budget-prefix-task", "split": split}
    packet_path = tmp_path / "packet.jsonl"
    packet_path.write_text(json.dumps(task) + "\n")
    packet_sha = hashlib.sha256(packet_path.read_bytes()).hexdigest()
    (tmp_path / "preparation.json").write_text(json.dumps({
        "schema": "natlang.task-rollout-preparation/1", "root_mode": "authored",
        "packet_sha256": packet_sha,
    }) + "\n")
    entries = build_queue(packet_path, 1, 2, tmp_path, temperature=0.7, case_seconds=60,
                          max_model_requests=4, max_turns=8, seed_base=3)
    entry = entries[0]
    jobs = Path(entry["jobs"])
    jobs.mkdir(parents=True, exist_ok=True)
    source_sha, program_sha = packet_sha, "d" * 64
    attempt, execution = "evidence-attempt", "execution-attempt"
    response_calls = [["eval", {"code": "return_result({status:'failed'});"}]]
    raw_calls = [{"id": "call-1", "type": "function", "function": {
        "name": "eval", "arguments": json.dumps({"code": "return_result({status:'failed'});"})}}]
    frame_specs = [("inv", graph_turn, None)]
    graph_events = [{"kind": "model_turn", "node": graph_turn, "call_id": "inv", "inputs": []}]
    if native_pair:
        producer_sampling = {"prompt": [{"kind": "tokens", "token_ids": [7], "producing": [False]}],
                            "writes": [{"state_id": "state"}]}
        consumer_sampling = {"prompt": [{"kind": "native_state", "id": "state"}], "writes": []}
        frame_specs = [("producer", "producer#turn1", producer_sampling),
                       ("consumer", "consumer#turn1", consumer_sampling)]
        graph_events = [
            {"kind": "model_turn", "node": "producer#turn1", "call_id": "producer", "inputs": []},
            {"kind": "model_turn", "node": "consumer#turn1", "call_id": "consumer", "inputs": []},
            {"kind": "block_write", "node": "writer", "turn": "producer#turn1", "block": "state", "inputs": []},
            {"kind": "block_read", "node": "reader", "turn": "consumer#turn1", "block": "state",
             "inputs": [{"node": "writer", "block": "state"}]},
        ]
    frames = []
    for invocation_id, node, sampling in frame_specs:
        frame = {"phase": "action", "invocation_id": invocation_id,
                 "context": [{"role": "user", "content": "q"}],
                 "tools_offered": [{"type": "function", "function": {"name": "eval", "parameters": {"type": "object"}}}],
                 "assistant": {"content": "", "calls": [{"tool": "eval", "arguments": {"code": "return_result({status:'failed'});"}}]},
                 "model_response": {"calls": response_calls, "raw_calls": raw_calls,
                                    **({"rl_sampling": sampling} if sampling else {})}}
        frames.append((invocation_id, frame))
    root_invocation = f"root/execution-{execution}"
    root_frame = {"phase": "action", "invocation_id": root_invocation,
                  "context": [{"role": "user", "content": "reference"}],
                  "tools_offered": [], "assistant": {"content": "", "calls": []},
                  "model_response": {"calls": [], "raw_calls": []}}
    child_events = graph_events
    root_call_node = f"call:{root_invocation}"
    root_events = [
        {"kind": "invocation", "node": root_call_node, "call_id": root_invocation, "inputs": []},
        {"kind": "model_turn", "node": f"{root_invocation}#turn1", "call_id": root_invocation,
         "inputs": [{"node": root_call_node, "port": "invocation"}]},
    ]
    if root_dependency:
        root_events.append({"kind": "block_write", "node": "root-writer", "turn": f"{root_invocation}#turn1",
                            "block": "reference", "inputs": []})
        child_events.append({"kind": "block_read", "node": "child-reader", "turn": f"{frames[0][0]}#turn1",
                             "block": "reference", "inputs": [{"node": "root-writer", "block": "reference"}]})
    child_events = [*({"kind": "invocation", "node": f"call:{invocation}", "call_id": invocation,
                       "inputs": [{"node": root_call_node, "port": "caller"}]}
                      for invocation, _ in frames),
                    *[{**event, "inputs": [{"node": f"call:{event['call_id']}", "port": "invocation"}]}
                      if event.get("kind") == "model_turn" else event for event in child_events]]
    trace_rows = [{"kind": "trace_event", "trace_role": "root", "invocation_id": f"root/execution-{execution}",
                   "event": event} for event in root_events]
    child_counts = {}
    for event in child_events:
        invocation = event.get("call_id") or str(event.get("turn", "")).split("#turn", 1)[0]
        if not invocation:
            invocation = frames[0][0]
        trace_rows.append({"kind": "trace_event", "trace_role": "child", "invocation_id": invocation,
                           "event": event})
        child_counts[invocation] = child_counts.get(invocation, 0) + 1
    payloads = [
        {"kind": "execution_snapshot", "schema": "natlang.partial_execution_snapshot/1",
         "run_id": root_invocation, "logical_run_id": "root",
         "execution_attempt_id": execution, "status": "interrupted",
         "failure_reason": "whole-case model request budget exceeded (4)", "pending_children": pending,
         "root_event_count": len(root_events), "invocation_count": len(frames)},
        *trace_rows,
        *[{"kind": "invocation_snapshot", "call_id": invocation, "event_count": child_counts.get(invocation, 0)}
          for invocation, _ in frames],
        *([] if missing_frame else [
            {"kind": "teacher_action_frame", "source": "authored_reference_root", "request_sha256": f"{8:064x}",
             "invocation_id": root_invocation, "response_frame": root_frame, "saved_turn_index": 0},
            *[{"kind": "teacher_action_frame", "source": "live_provider", "request_sha256": f"{index + 10:064x}",
               "invocation_id": invocation, "response_frame": frame, "saved_turn_index": index + 1}
              for index, (invocation, frame) in enumerate(frames)]
        ]),
    ]
    records = [json.dumps({"version": "natlang.teacher_partial_evidence_record/1", "attempt_id": attempt,
                           "sequence": index, "payload": payload}, separators=(",", ":"))
               for index, payload in enumerate(payloads)]
    evidence = ("\n".join(records) + "\n").encode()
    evidence_path = jobs / f"{entry['key']}.partial.evidence-{attempt}.jsonl"
    evidence_path.write_bytes(evidence)
    (jobs / f"000000-{program_sha[:16]}.partial.json").write_text(json.dumps({
        "version": "natlang.teacher_partial.native/1", "program_id": task["id"],
        "provenance": {"program_ir_sha256": program_sha}, "execution_attempt_id": execution,
        "turns": [{"request_sha256": f"{8:064x}", "invocation_id": root_invocation,
                   "response": {"calls": [], "raw_response": {"transport": "native-collector"}, "text": ""}},
                  *[{"request_sha256": f"{index + 10:064x}", "invocation_id": invocation,
                     "response": {"calls": response_calls,
                                  **({"raw_calls": [] if mismatched_saved_raw_calls else frame["model_response"]["raw_calls"]}
                                     if saved_raw_calls or mismatched_saved_raw_calls else
                                     {"raw_response": {"transport": "native-collector"}}),
                                  "text": ""}}
                    for index, (invocation, frame) in enumerate(frames)]],
        "request_attempt_ids": [attempt, "newer-attempt"] if stale_attempt else [attempt], "evidence_snapshots": [{
            "schema": "natlang.teacher_partial_evidence/1", "path": evidence_path.name,
            "attempt_id": attempt, "status": "execution_interrupted", "records": len(payloads),
            "bytes": len(evidence) + (1 if manifest_tamper else 0), "sha256": hashlib.sha256(evidence).hexdigest()}]},
        ) + "\n")
    error = {"index": 0, "program_id": task["id"], "program_ir_sha256": program_sha,
        "source_sha256": source_sha, "code": "NATLANG_MODEL_REQUEST_BUDGET",
        "error": "Error: whole-case model request budget exceeded (4)"}
    error_path = jobs / "000000.error.json"
    error_bytes = (json.dumps(error) + "\n").encode()
    error_path.write_bytes(error_bytes)
    state = {"index": 0, "program_id": task["id"], "digest": program_sha, "state": "failed",
             "resource_limit_reason": "model_request_budget",
             "terminal_error": {"program_ir_sha256": program_sha, "record_binding_verified": True,
                                "source_sha256": source_sha,
                                "sha256": hashlib.sha256(error_bytes).hexdigest(), "path": str(error_path),
                                "error": error["error"]}}
    finish = {"event": "finish", "key": entry["key"], "status": "incomplete_export",
              "resource_limit_reason": "model_request_budget",
              "output_accounting": {"source_sha256": source_sha, "job_states": [state]}}
    return task, entries, finish


def test_scored_model_budget_prefix_reconstructs_exact_whole_graph(tmp_path):
    task, entries, finish = _budget_prefix_fixture(tmp_path)
    (tmp_path / "packet.jsonl").write_text(json.dumps(task) + "\n")
    (tmp_path / "queue.jsonl").write_text("".join(json.dumps(row) + "\n" for row in entries))
    (tmp_path / "journal.jsonl").write_text(json.dumps(finish) + "\n")
    emitted = tmp_path / "native-rollouts.jsonl"
    score(tmp_path, emit=emitted)
    row = json.loads(emitted.read_text().splitlines()[0])
    assert row["outcome"] == "budget_exhaustion"
    assert row["reward"] == {"quality": 0.0}
    assert len(row["turns"]) == 1 and row["turns"][0]["replay"]["raw_calls_retained"]
    assert row["turns"][0]["target"]["tool_calls"] == [{"id": "call-1", "type": "function", "function": {
        "name": "eval", "arguments": json.dumps({"code": "return_result({status:'failed'});"})}}]
    assert any(event["kind"] == "model_turn" for event in row["execution_graph"])
    assert [event["node"] for event in row["execution_graph"] if event.get("kind") == "model_turn"] == ["inv#turn1"]
    assert any(event.get("kind") == "invocation" and event.get("node", "").startswith("call:root/")
               for event in row["execution_graph"])
    assert row["budget_termination_evidence"]["attempt_id"] == "evidence-attempt"


def test_budget_prefix_keeps_producer_and_consumer_as_one_graph(tmp_path):
    task, entries, finish = _budget_prefix_fixture(tmp_path, native_pair=True)
    (tmp_path / "packet.jsonl").write_text(json.dumps(task) + "\n")
    (tmp_path / "queue.jsonl").write_text("".join(json.dumps(row) + "\n" for row in entries))
    (tmp_path / "journal.jsonl").write_text(json.dumps(finish) + "\n")
    emitted = tmp_path / "native-rollouts.jsonl"
    score(tmp_path, emit=emitted)
    row = json.loads(emitted.read_text().splitlines()[0])
    assert [turn["invocation_id"] for turn in row["turns"]] == ["producer", "consumer"]
    assert {event["kind"] for event in row["execution_graph"]} == {
        "invocation", "model_turn", "block_write", "block_read"}
    from natlang_neuralese.rl.episode_graph import whole_episode_turns
    bound = whole_episode_turns(row)
    assert bound[1]["read_bindings"] == {"state": "writer"}


def test_budget_prefix_checks_raw_calls_when_collector_persists_them(tmp_path):
    task, entries, finish = _budget_prefix_fixture(tmp_path, saved_raw_calls=True)
    row = _budget_terminated_prefix(entries[0], task, finish, packet_source_path=tmp_path / "packet.jsonl",
                                    packet_source_sha256=hashlib.sha256((tmp_path / "packet.jsonl").read_bytes()).hexdigest())
    assert row is not None

    tmp_path2 = tmp_path / "mismatch"
    tmp_path2.mkdir()
    task, entries, finish = _budget_prefix_fixture(tmp_path2, saved_raw_calls=True, mismatched_saved_raw_calls=True)
    row = _budget_terminated_prefix(entries[0], task, finish, packet_source_path=tmp_path2 / "packet.jsonl",
                                    packet_source_sha256=hashlib.sha256((tmp_path2 / "packet.jsonl").read_bytes()).hexdigest())
    assert row is None


def test_budget_prefix_rejects_authored_root_state_dependency_instead_of_detaching_it(tmp_path):
    task, entries, finish = _budget_prefix_fixture(tmp_path, root_dependency=True)
    row = _budget_terminated_prefix(entries[0], task, finish, packet_source_path=tmp_path / "packet.jsonl",
                                    packet_source_sha256=hashlib.sha256((tmp_path / "packet.jsonl").read_bytes()).hexdigest())
    assert row is None


@pytest.mark.parametrize("change", ["graph", "pending", "manifest", "error_code", "stale_attempt", "missing_frame", "held_split", "duplicate_state"])
def test_unverifiable_budget_prefix_is_never_emitted_as_trainable(tmp_path, change):
    task, entries, finish = _budget_prefix_fixture(tmp_path,
        graph_turn="other#turn1" if change == "graph" else "inv#turn1",
        pending=change == "pending", manifest_tamper=change == "manifest",
        stale_attempt=change == "stale_attempt", missing_frame=change == "missing_frame",
        split="test" if change == "held_split" else "train")
    (tmp_path / "packet.jsonl").write_text(json.dumps(task) + "\n")
    (tmp_path / "queue.jsonl").write_text("".join(json.dumps(row) + "\n" for row in entries))
    if change == "error_code":
        error_path = Path(entries[0]["jobs"]) / "000000.error.json"
        error = json.loads(error_path.read_text()); error["code"] = "SOME_OTHER_FAILURE"
        error_path.write_text(json.dumps(error) + "\n")
    if change == "duplicate_state":
        finish["output_accounting"]["job_states"].append(json.loads(json.dumps(finish["output_accounting"]["job_states"][0])))
    (tmp_path / "journal.jsonl").write_text(json.dumps(finish) + "\n")
    emitted = tmp_path / "native-rollouts.jsonl"
    score(tmp_path, emit=emitted)
    row = json.loads(emitted.read_text().splitlines()[0])
    assert row["reward"] is None and row["unscored"] == "budget_prefix_capture_unavailable"
    assert row["turns"] == [] and row["execution_graph"] == []


def test_stale_self_consistent_terminal_is_rejected_when_current_packet_changes(tmp_path):
    task, entries, finish = _budget_prefix_fixture(tmp_path)
    packet_path = tmp_path / "packet.jsonl"
    changed_task = {**task, "current_source_revision": "new packet content with same task identity"}
    packet_path.write_text(json.dumps(changed_task) + "\n")
    preparation = json.loads((tmp_path / "preparation.json").read_text())
    preparation["packet_sha256"] = hashlib.sha256(packet_path.read_bytes()).hexdigest()
    (tmp_path / "preparation.json").write_text(json.dumps(preparation) + "\n")
    (tmp_path / "queue.jsonl").write_text("".join(json.dumps(row) + "\n" for row in entries))
    (tmp_path / "journal.jsonl").write_text(json.dumps(finish) + "\n")
    emitted = tmp_path / "native-rollouts.jsonl"
    score(tmp_path, emit=emitted)
    row = json.loads(emitted.read_text().splitlines()[0])
    assert row["reward"] is None and row["unscored"] == "budget_prefix_capture_unavailable"
    assert row["turns"] == [] and row["execution_graph"] == []


def test_student_root_score_rejects_host_authored_action(tmp_path):
    (tmp_path / "packet.jsonl").write_text(json.dumps(student_root_row(_row())) + "\n")
    (tmp_path / "preparation.json").write_text(json.dumps({"root_mode": "student"}))
    entries = build_queue(tmp_path / "packet.jsonl", 1, 2, tmp_path, temperature=0.6, case_seconds=60,
                          max_model_requests=8, max_turns=4, seed_base=0)
    (tmp_path / "queue.jsonl").write_text("".join(json.dumps(e) + "\n" for e in entries))
    output = tmp_path / entries[0]["output"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"outcome": {"status": "done", "accepted": True}, "trajectory": [
        {"phase": "action", "action_provenance": {"kind": "authored_reference_root_eval"}}
    ]}) + "\n")
    with pytest.raises(ValueError, match="replayed an authored root"):
        score(tmp_path)


def test_rollout_turns_keep_policy_samples_and_skip_authored_actions():
    from natlang_neuralese.rl.task_rollouts import rollout_turns
    context = [{"role": "user", "content": "q"}]
    result = {"trajectory": [
        {"phase": "action", "context": context, "tools_offered": [{"type": "function"}],
         "assistant": {"content": "", "calls": [{"tool": "eval", "arguments": {"code": "1"}}],
                       "action_provenance": {"kind": "authored_reference_root_eval"}}},
        {"phase": "action", "context": context, "tools_offered": [{"type": "function"}],
         "assistant": {"content": "", "calls": [{"tool": "return_result", "source_tool": "return_result",
                                                 "arguments": {"status": "success", "value": 3}}]}},
    ]}
    turns = rollout_turns(result)
    assert len(turns) == 1 and turns[0]["messages"] == context
    call = turns[0]["target"]["tool_calls"][0]["function"]
    assert call["name"] == "return_result" and json.loads(call["arguments"])["value"] == 3


def test_rollout_replay_preserves_reasoning_and_raw_calls_without_normalization():
    from natlang_neuralese.rl.task_rollouts import rollout_turns
    raw = {"id": "actual_17", "type": "function", "function": {
        "name": "wire_eval", "arguments": '{ "code" : "2 + 2" }'}}
    result = {"trajectory": [{"phase": "action", "context": [{"role": "user", "content": "q"}],
        "assistant": {"content": "", "reasoning": "Read the input first.",
                      "calls": [{"tool": "eval", "arguments": {"code": "2 + 2"}}]},
        "model_response": {"raw_calls": [raw]}}]}
    turn = rollout_turns(result)[0]
    assert turn["target"]["reasoning_content"] == "Read the input first."
    assert turn["target"]["tool_calls"] == [raw]
    assert turn["replay"]["raw_calls_retained"]
    turn["target"]["tool_calls"][0]["id"] = "changed"
    assert raw["id"] == "actual_17"


def test_rollout_plain_text_interpreted_as_call_stays_plain_text_for_replay():
    from natlang_neuralese.rl.task_rollouts import rollout_turns
    text = '{"value": 3}'
    turn = rollout_turns({"trajectory": [{"phase": "action", "context": [],
        "assistant": {"content": text, "calls": [{"tool": "return_result", "arguments": {"value": 3}}]},
        "model_response": {"raw_calls": []}}]})[0]
    assert turn["target"] == {"role": "assistant", "content": text}


def test_native_sampling_receipt_and_native_tool_parts_survive_rollout_emission():
    from natlang_neuralese.rl.task_rollouts import rollout_turns
    state_id = "nzs1_" + "a" * 52
    receipt = {"schema": "natlang.native-rl-sampling/1",
        "policy": {"checkpoint_sha256": "1" * 64, "tokenizer_fingerprint": "2" * 64,
                   "embedding_weight_sha256": "3" * 64, "architecture": "lfm2"},
        "prompt": [{"kind": "tokens", "token_ids": [7], "producing": [False]},
                   {"kind": "native_state", "id": state_id,
                    "native_identity": {"checkpoint_sha256": "1" * 64}}],
        "samples": [{"index": 0, "token_id": 4, "logp": -0.1, "input_role": "ordinary", "terminal": False},
                    {"index": 1, "token_id": 9, "logp": -0.25, "input_role": "produce", "terminal": False},
                    {"index": 2, "token_id": 5, "logp": -0.15, "input_role": "ordinary", "terminal": False},
                    {"index": 3, "token_id": 2, "logp": -0.2, "input_role": "ordinary", "terminal": True}],
        "writes": [{"state_id": state_id, "begin_sample": 0, "end_sample": 3,
                    "payload_begin": 1, "payload_end": 2,
                    "native_identity": {"checkpoint_sha256": "1" * 64}}],
        "sampling_law": {"temperature": 1.0, "seed": 14, "guidance": False,
                         "probabilities": "post-processing-normalized"}}
    raw_call = {"id": "call_1", "type": "function", "function": {"name": "consume",
        "arguments": json.dumps({"value": [{"type": "native_state", "id": state_id}]})}}
    result = {"trajectory": [{"phase": "action", "context": [{"role": "user", "content": "q"}],
        "assistant": {"content": "state: " + "\uE002" + state_id + "\uE003", "calls": []},
        "model_response": {"raw_calls": [raw_call], "rl_sampling": receipt}}]}
    turn = rollout_turns(result)[0]
    assert turn["sampling"] == receipt
    assert turn["target"]["content"] == result["trajectory"][0]["assistant"]["content"]
    assert turn["target"]["tool_calls"] == [raw_call]
