"""S7 task-success rollouts: reward mapping, train-only derivation, sampled queue and spread routing."""
import hashlib
import json
from pathlib import Path

import pytest

from natlang_neuralese.rl.task_rollouts import (_budget_terminated_prefix, authored_root_row, build_queue,
                                               reward_of, score, student_root_row)


def _row(split="train", code="return await f(task);"):
    return {"id": f"row-{split}", "split": split, "family": "fam",
            "source_ids": [f"fixture:{split}"], "source_groups": [f"fixture-group:{split}"],
            "source_revisions": ["fixture/v1"],
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
                           saved_raw_calls=False, mismatched_saved_raw_calls=False, root_dependency=False,
                           decision=False, omit_last=False, tamper_decision=False,
                           frame_source="live_provider", legacy_replay_frames=False):
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
    decision_capture = {"prompt": [{"kind": "tokens", "token_ids": [7], "producing": [False]}], "writes": [],
                        "decision_sample": {"chosen": 1}}
    decision_scores = {"log_probs": [-1.0, -0.5], "tokens": [1, 2], "native_decision_capture": decision_capture,
                       "chosen": 1}
    decision_request = {"messages": [{"role": "user", "content": "pick"}], "options": ["a", "b"],
                        "invocation_id": "dec", "invocation_role": "student"}
    decision_sha = f"{99:064x}"
    if decision:
        graph_events.append({"kind": "model_turn", "node": "dec#turn1", "call_id": "dec", "inputs": []})
        frame_specs.append(("dec", "dec#turn1", None))
    frames = []
    for invocation_id, node, sampling in frame_specs:
        if invocation_id == "dec":
            continue
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
    invocations = [invocation for invocation, _ in frames] + (["dec"] if decision else [])
    child_events = [*({"kind": "invocation", "node": f"call:{invocation}", "call_id": invocation,
                       "inputs": [{"node": root_call_node, "port": "caller"}]}
                      for invocation in invocations),
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
         "root_event_count": len(root_events), "invocation_count": len(invocations)},
        *trace_rows,
        *[{"kind": "invocation_snapshot", "call_id": invocation, "event_count": child_counts.get(invocation, 0)}
          for invocation in invocations],
        *([] if missing_frame else [
            {"kind": "teacher_action_frame", "source": "authored_reference_root", "request_sha256": f"{8:064x}",
             "invocation_id": root_invocation, "response_frame": root_frame, "saved_turn_index": 0},
            *[{"kind": "provider_frame" if legacy_replay_frames else "teacher_action_frame",
               "source": "journal_replay" if legacy_replay_frames else frame_source,
               "request_sha256": f"{index + 10:064x}",
               "invocation_id": invocation, "response_frame": frame, "saved_turn_index": index + 1}
              for index, (invocation, frame) in enumerate(frames)]
        ]),
        *([{"kind": "decision_action_frame", "request_sha256": decision_sha, "invocation_id": "dec",
            "request": decision_request, "scores": decision_scores}] if decision else []),
    ]
    if omit_last:
        # the last policy turn's action record exceeded the collector's byte limit
        omitted_index = max(i for i, row in enumerate(payloads) if row["kind"] == "teacher_action_frame")
        payloads[omitted_index] = {"kind": "record_omitted", "reason": "record_exceeded_byte_limit", "bytes": 9,
                                   "payload_sha256": "e" * 64, "original_kind": "teacher_action_frame"}
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
        **({"decisions": [{"request_sha256": decision_sha, "scores": {**decision_scores, "chosen": 0}
                           if tamper_decision else decision_scores}]} if decision else {}),
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
    assert row["reward"]["quality"] == 0.0
    assert row["reward"]["execution_efficiency"]["model_calls"] == len(row["turns"])
    assert len(row["turns"]) == 1 and row["turns"][0]["replay"]["raw_calls_retained"]
    assert row["turns"][0]["target"]["tool_calls"] == [{"id": "call-1", "type": "function", "function": {
        "name": "eval", "arguments": json.dumps({"code": "return_result({status:'failed'});"})}}]
    assert any(event["kind"] == "model_turn" for event in row["execution_graph"])
    assert [event["node"] for event in row["execution_graph"] if event.get("kind") == "model_turn"] == ["inv#turn1"]
    assert any(event.get("kind") == "invocation" and event.get("node", "").startswith("call:root/")
               for event in row["execution_graph"])
    assert row["budget_termination_evidence"]["attempt_id"] == "evidence-attempt"
    assert row["source_groups"] == ["fixture-group:train"]
    assert row["source_ids"] == ["fixture:train"]
    assert row["source_revisions"] == ["fixture/v1"]
    assert row["source_custody"]["packet_sha256"] == hashlib.sha256((tmp_path / "packet.jsonl").read_bytes()).hexdigest()
    assert row["source_custody"]["packet_row_sha256"] == hashlib.sha256((tmp_path / "packet.jsonl").read_bytes()).hexdigest()


def _emit_budget_rollout(tmp_path, **options):
    task, entries, finish = _budget_prefix_fixture(tmp_path, **options)
    (tmp_path / "packet.jsonl").write_text(json.dumps(task) + "\n")
    (tmp_path / "queue.jsonl").write_text("".join(json.dumps(row) + "\n" for row in entries))
    (tmp_path / "journal.jsonl").write_text(json.dumps(finish) + "\n")
    emitted = tmp_path / "native-rollouts.jsonl"
    score(tmp_path, emit=emitted)
    return json.loads(emitted.read_text().splitlines()[0])


def test_budget_prefix_keeps_sampled_decision_turns_as_scored_failure(tmp_path):
    """A closed-set decision is a model turn of the graph (`decision_action_frame`); the episode keeps it."""
    row = _emit_budget_rollout(tmp_path, decision=True)
    assert row["unscored"] is None and row["reward"]["quality"] == 0.0
    assert [turn["invocation_id"] for turn in row["turns"]] == ["inv", "dec"]
    decision = row["turns"][1]
    assert decision["target"] == {"role": "assistant", "content": "b"}
    assert decision["sampling"]["decision_sample"] == {"chosen": 1}
    assert {event["node"] for event in row["execution_graph"] if event.get("kind") == "model_turn"} == {
        "inv#turn1", "dec#turn1"}
    from natlang_neuralese.rl.episode_graph import whole_episode_turns
    whole_episode_turns(row)


def test_resumed_budget_prefix_recovers_replayed_actions_and_decisions(tmp_path):
    """A resumed attempt's evidence holds only journal-replayed turns (canonical `teacher_action_frame`, source
    `journal_replay`, bound to the saved turn index) plus a replayed decision's `decision_action_frame`; the
    prefix is recovered with the same custody checks as a fresh one."""
    row = _emit_budget_rollout(tmp_path, decision=True, frame_source="journal_replay")
    assert row["unscored"] is None and row["reward"]["quality"] == 0.0
    assert [turn["invocation_id"] for turn in row["turns"]] == ["inv", "dec"]
    from natlang_neuralese.rl.episode_graph import whole_episode_turns
    whole_episode_turns(row)


def test_replayed_frame_with_a_mismatched_saved_turn_is_not_recovered(tmp_path):
    task, entries, finish = _budget_prefix_fixture(tmp_path, frame_source="journal_replay")
    evidence = next(Path(entries[0]["jobs"]).glob("*.evidence-*.jsonl"))
    text = evidence.read_text().replace('"saved_turn_index":1', '"saved_turn_index":0')
    assert text != evidence.read_text()
    # re-seal the manifest so only the custody binding (not the file hash) is what fails
    partial_path = next(evidence.parent.glob("*.partial.json"))
    partial = json.loads(partial_path.read_text())
    evidence.write_text(text)
    partial["evidence_snapshots"][0].update(bytes=len(text.encode()), sha256=hashlib.sha256(text.encode()).hexdigest())
    partial_path.write_text(json.dumps(partial) + "\n")
    assert _budget_terminated_prefix(entries[0], task, finish, packet_source_path=tmp_path / "packet.jsonl",
                                     packet_source_sha256=hashlib.sha256((tmp_path / "packet.jsonl").read_bytes()).hexdigest()) is None


def test_legacy_provider_frame_replay_is_never_a_partial_sample(tmp_path):
    """Pre-fix collectors wrote replayed turns as `provider_frame` and nothing for replayed decisions."""
    row = _emit_budget_rollout(tmp_path, legacy_replay_frames=True)
    assert row["reward"] is None and row["unscored"] == "budget_prefix_capture_unavailable"


def test_budget_prefix_decision_must_match_its_saved_decision(tmp_path):
    row = _emit_budget_rollout(tmp_path, decision=True, tamper_decision=True)
    assert row["reward"] is None and row["unscored"] == "budget_prefix_capture_unavailable"


def test_budget_prefix_trains_turns_before_an_omitted_oversized_record(tmp_path):
    """The collector replaces a record over its byte limit with `record_omitted`: the episode is trained on the
    turns captured before it (producer only: its consumer's read and turn leave the graph), still a failure."""
    row = _emit_budget_rollout(tmp_path, native_pair=True, omit_last=True)
    assert row["unscored"] is None and row["reward"]["quality"] == 0.0
    assert [turn["invocation_id"] for turn in row["turns"]] == ["producer"]
    assert {event["node"] for event in row["execution_graph"] if event.get("kind") == "model_turn"} == {
        "producer#turn1"}
    assert not any(event.get("kind") == "block_read" for event in row["execution_graph"])
    assert row["budget_termination_evidence"]["truncated_at_omitted_record"]["kept_turns"] == 1
    from natlang_neuralese.rl.episode_graph import whole_episode_turns
    whole_episode_turns(row)


def test_budget_prefix_with_no_captured_turn_before_omission_stays_unscored(tmp_path):
    row = _emit_budget_rollout(tmp_path, omit_last=True)
    assert row["reward"] is None and row["unscored"] == "budget_prefix_capture_unavailable"


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
    for entry in entries:
        entry["packet_sha256"] = preparation["packet_sha256"]
    (tmp_path / "queue.jsonl").write_text("".join(json.dumps(row) + "\n" for row in entries))
    (tmp_path / "journal.jsonl").write_text(json.dumps(finish) + "\n")
    emitted = tmp_path / "native-rollouts.jsonl"
    score(tmp_path, emit=emitted)
    row = json.loads(emitted.read_text().splitlines()[0])
    assert row["reward"] is None and row["unscored"] == "budget_prefix_capture_unavailable"
    assert row["turns"] == [] and row["execution_graph"] == []


def test_score_rejects_queue_rebound_to_another_source_task(tmp_path):
    task, entries, _ = _budget_prefix_fixture(tmp_path)
    (tmp_path / "packet.jsonl").write_text(json.dumps(task) + "\n")
    entries[0]["task_id"] = "different-task"
    (tmp_path / "queue.jsonl").write_text("".join(json.dumps(row) + "\n" for row in entries))
    with pytest.raises(ValueError, match="not bound to its exact task packet row"):
        score(tmp_path, emit=tmp_path / "rollouts.jsonl")


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


def _neuralese_result(*, accepted, consumer_imports=True, external_writer=False, marker_text=False):
    """A collector result with a producer call and a consumer call; the consumer optionally imports its state."""
    def step(call, sampling, content="ok"):
        return {"phase": "action", "invocation_id": call, "context": [{"role": "user", "content": "q"}],
                "assistant": {"content": content, "calls": []},
                "model_response": {"raw_calls": [], "rl_sampling": sampling}}
    producer = {"prompt": [{"kind": "tokens", "token_ids": [7], "producing": [False]}],
                "writes": [{"state_id": "state"}]}
    consumer = {"prompt": [{"kind": "tokens", "token_ids": [8], "producing": [False]},
                           *([{"kind": "native_state", "id": "state"}] if consumer_imports else [])],
                "writes": []}
    events = [{"kind": "model_turn", "node": "producer#turn1", "call_id": "producer", "inputs": []},
              {"kind": "model_turn", "node": "consumer#turn1", "call_id": "consumer", "inputs": []},
              {"kind": "block_write", "node": "writer", "turn": "producer#turn1", "block": "state", "inputs": []}]
    if consumer_imports:
        events.append({"kind": "block_read", "node": "reader", "turn": "consumer#turn1", "block": "state",
                       "inputs": [{"node": "older-outside-writer" if external_writer else "writer", "block": "state"}]})
    text = "use state here" if marker_text else "ok"
    outcome = ({"status": "done", "accepted": True} if accepted else
               {"status": "failed", "detail": "model_incomplete: terminated_without_result"})
    return {"trajectory": [step("producer", producer), step("consumer", consumer, text)],
            "outcome": {**outcome, "execution_graph": events}}


def _score_results(tmp_path, results, roles=None, **options):
    tmp_path.mkdir(parents=True, exist_ok=True)
    packet = [student_root_row(authored_root_row(_row()))]
    packet_bytes = "".join(json.dumps(r) + "\n" for r in packet).encode()
    (tmp_path / "packet.jsonl").write_bytes(packet_bytes)
    (tmp_path / "preparation.json").write_text(json.dumps({"root_mode": "student",
        "packet_sha256": hashlib.sha256(packet_bytes).hexdigest()}))
    entries = build_queue(tmp_path / "packet.jsonl", 1, len(results), tmp_path, temperature=1.0, case_seconds=60,
                          max_model_requests=8, max_turns=4, seed_base=0)
    for entry, role in zip(entries, roles or []):
        if role:
            entry["jolt_role"] = role
    (tmp_path / "queue.jsonl").write_text("".join(json.dumps(e) + "\n" for e in entries))
    for entry, result in zip(entries, results):
        output = tmp_path / entry["output"]
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result) + "\n")
    report = score(tmp_path, tmp_path / "rollouts.jsonl", **options)
    from natlang_neuralese.eval.task_packet import read_jsonl
    return report, read_jsonl(tmp_path / "rollouts.jsonl")


def test_neuralese_bonus_only_for_correct_episodes_with_a_real_producer_consumer_import(tmp_path):
    from natlang_neuralese.rl.rollout_math import rollout_return
    plain = {"trajectory": [{"phase": "action", "invocation_id": "root", "context": [{"role": "user", "content": "q"}],
                             "assistant": {"content": "ok", "calls": []}, "model_response": {"raw_calls": []}}],
             "outcome": {"status": "done", "accepted": True}}
    report, rows = _score_results(tmp_path, [
        _neuralese_result(accepted=True),                       # correct + real exchange: bonus
        _neuralese_result(accepted=False),                      # failed + real exchange: never a bonus
        plain,                                                  # correct without Neuralese: rewarded, no bonus
        _neuralese_result(accepted=True, consumer_imports=False, marker_text=True),  # markers, no import
        _neuralese_result(accepted=True, external_writer=True),  # import of an older outside state
    ])
    assert [row["outcome"] for row in rows] == ["task_success", "model_incomplete", "task_success",
                                                "task_success", "task_success"]
    assert rows[0]["reward"]["quality"] == 1.0 and rows[0]["reward"]["neuralese_bonus"] == .1
    assert rows[1]["reward"]["quality"] == 0.0
    assert all(row["reward"]["quality"] == 1.0 for row in rows[2:])
    assert all("execution_efficiency" in row["reward"] for row in rows)
    assert [rollout_return(row) for row in rows] == [1.1, 0.0, 1.0, 1.0, 1.0]
    metrics = report["neuralese"]
    assert metrics["bonus"] == 0.1 and metrics["bonus_episodes"] == 1 and metrics["bonus_total"] == 0.1
    assert metrics["correct_episodes"] == 4 and metrics["correct_using_neuralese"] == 1
    assert metrics["correct_neuralese_rate"] == 0.25
    assert metrics["episodes_using_neuralese"] == 2 and metrics["neuralese_use_rate"] == 0.4
    assert report["groups"][0]["route"] == "grpo"  # curriculum routes stay on the task outcome


def test_a_run_without_rl_capture_measures_neuralese_use_from_its_graph_and_never_pays_a_bonus(tmp_path):
    def uncaptured(result):
        for step in result["trajectory"]:
            step["model_response"].pop("rl_sampling")
        return result
    report, rows = _score_results(tmp_path, [uncaptured(_neuralese_result(accepted=True)),
                                             uncaptured(_neuralese_result(accepted=True, external_writer=True)),
                                             _neuralese_result(accepted=True)])
    assert "neuralese_bonus" not in rows[0]["reward"] and rows[2]["reward"]["neuralese_bonus"] == .1
    assert report["neuralese"]["episodes_using_neuralese"] == 2   # the held-out evaluation case and the captured one
    assert report["neuralese"]["bonus_episodes"] == 1             # an outside writer is never an exchange


def test_neuralese_bonus_opt_out_and_teacher_attempts_get_none(tmp_path):
    report, rows = _score_results(tmp_path / "off", [_neuralese_result(accepted=True)] * 2, neuralese_bonus=0)
    assert all(row["reward"]["quality"] == 1.0 and "neuralese_bonus" not in row["reward"]
               for row in rows) and report["neuralese"]["bonus_episodes"] == 0
    assert report["neuralese"]["correct_using_neuralese"] == 2
    report, rows = _score_results(tmp_path / "teacher", [_neuralese_result(accepted=True)] * 2,
                                  roles=[None, "teacher"])
    assert rows[0]["reward"]["quality"] == 1.0 and rows[0]["reward"]["neuralese_bonus"] == .1
    assert rows[1]["reward"]["quality"] == 1.0 and "neuralese_bonus" not in rows[1]["reward"]
    with pytest.raises(ValueError, match="neuralese_use_bonus"):
        _score_results(tmp_path / "bad", [_neuralese_result(accepted=True)] * 2, neuralese_bonus=-0.1)


def test_execution_efficiency_uses_observed_whole_graph_counters_and_is_bounded():
    from natlang_neuralese.rl.task_rollouts import execution_efficiency
    action = {"function": {"name": "eval", "arguments": '{"code":"return 1;"}'}}
    measured = execution_efficiency({'turns': [
        {'invocation_id': 'i', 'usage': {'completion_tokens': 1000}, 'messages': [],
         'target': {'tool_calls': [{**action, 'id': 'a'}]}},
        {'invocation_id': 'i', 'usage': {'completion_tokens': 1000},
         'messages': [{'role': 'tool', 'tool_call_id': 'a', 'content': 'same'}],
         'target': {'tool_calls': [{**action, 'id': 'b'}]}},
        {'invocation_id': 'i', 'usage': {'completion_tokens': 1000},
         'messages': [{'role': 'tool', 'tool_call_id': 'a', 'content': 'same'},
                      {'role': 'tool', 'tool_call_id': 'b', 'content': 'same'}],
         'target': {'tool_calls': [{**action, 'id': 'c'}]}},
    ]})
    assert measured == {'model_calls': 3, 'generated_tokens': 3000, 'repeated_actions': 1,
                        'cost': pytest.approx(.015*3/11+.015*3000/5048+.020/5), 'weight': 1.0,
                        'policy': 'bounded-rational-v2', 'failure_report_credit': 0.0}
    assert measured['cost'] <= .05
    # The same sampled retry after a changed exact tool response is not evidence of no progress.
    progressed = execution_efficiency({'turns': [
        {'invocation_id': 'i', 'target': {'tool_calls': [{**action, 'id': 'a'}]}},
        {'invocation_id': 'i', 'messages': [{'role': 'tool', 'tool_call_id': 'a', 'content': 'same'}],
         'target': {'tool_calls': [{**action, 'id': 'b'}]}},
        {'invocation_id': 'i', 'messages': [{'role': 'tool', 'tool_call_id': 'a', 'content': 'same'},
                                            {'role': 'tool', 'tool_call_id': 'b', 'content': 'changed'}],
         'target': {'tool_calls': [{**action, 'id': 'c'}]}},
    ]})
    assert progressed['repeated_actions'] == 0
    unknown = execution_efficiency({'turns': [{'target': {'tool_calls': [action]}}]})
    assert unknown['model_calls'] == 1 and unknown['generated_tokens'] is None
    assert unknown['cost'] == pytest.approx(.015/9)  # call evidence remains usable; token count stays unknown
    opted_out = execution_efficiency({'turns': [{'target': {'tool_calls': [action]}}]}, weight=0)
    assert opted_out['cost'] == 0 and opted_out['model_calls'] == 1


def test_score_efficiency_ignores_wall_clock_and_reports_unknown_tokens(tmp_path):
    first = {'trajectory': [{'phase': 'action', 'context': [{'role': 'user', 'content': 'x'}],
                            'assistant': {'content': 'ok', 'calls': []}, 'model_response': {'raw_calls': []}}],
             'outcome': {'status': 'done', 'accepted': True}}
    second = {'outcome': {'status': 'done', 'accepted': True}}
    report, rows = _score_results(tmp_path, [first, second])
    telemetry = rows[0]['reward']['execution_efficiency']
    assert telemetry['model_calls'] == 1 and telemetry['generated_tokens'] is None
    assert report['execution_efficiency']['generated_tokens'] is None
    assert report['execution_efficiency']['model_calls'] == 1


def test_all_success_efficiency_spread_routes_to_policy_gradient_update(tmp_path):
    def action(tokens):
        return {'phase': 'action', 'context': [{'role': 'user', 'content': 'q'}],
                'assistant': {'content': 'done', 'calls': []},
                'model_response': {'raw_calls': [], 'completion_tokens': tokens}}
    results = [
        {'trajectory': [action(1)], 'outcome': {'status': 'done', 'accepted': True}},
        {'trajectory': [action(32), action(32)], 'outcome': {'status': 'done', 'accepted': True}},
    ]
    report, rows = _score_results(tmp_path, results)
    group = report['groups'][0]
    assert all(row['reward']['quality'] == 1 for row in rows)
    assert group['spread'] is False and group['efficiency_spread'] is True
    assert group['route'] == 'grpo'
    assert report['groups_with_quality_spread'] == 0
    assert report['groups_with_efficiency_spread'] == 1
    # Outcome spread stays semantically separate; production reads the explicit efficiency signal too.
    assert report['groups_with_spread'] == 0


def test_rollout_turns_keep_collector_completion_token_counter():
    from natlang_neuralese.rl.task_rollouts import rollout_turns
    result = {'trajectory': [{'phase': 'action', 'context': [], 'assistant': {'content': '', 'calls': []},
                             'model_response': {'raw_calls': [], 'completion_tokens': 17}}]}
    assert rollout_turns(result)[0]['usage'] == {'completion_tokens': 17}


def test_expand_captures_rebuilds_prompt_deltas_and_sample_columns():
    from natlang_neuralese.rl.task_rollouts import expand_captures
    system = list(range(100))
    def call(ids, sampled, producing=None):
        return {"phase": "action", "invocation_id": "a", "model_response": {"rl_sampling": {
            "schema": "s", "prompt": [{"kind": "tokens", "token_ids": ids, "producing": producing or [False] * len(ids)}],
            "samples": [{"index": i, "token_id": t, "logp": -0.5 * i, "input_role": "ordinary",
                         "terminal": i == len(sampled) - 1, "query_own_position": len(ids) + i} for i, t in enumerate(sampled)],
            "generated_inputs": [{"sample_index": i, "token_id": t, "input_role": "ordinary"}
                                 for i, t in enumerate(sampled[:-1])]}}}
    first = call(system + [500], [7, 8, 9])
    second = call(system + [500, 7, 8, 9, 600], [10, 11], producing=[False] * 104 + [True])
    compact = {"phase": "action", "invocation_id": "a", "model_response": {"rl_sampling": {
        "schema": "s", "prompt": [{"kind": "tokens_delta", "turn": 0, "keep": 101, "token_ids": [7, 8, 9, 600],
                                   "producing_at": [3]}],
        "samples_columns": {"token_id": [10, 11], "logp": [-0.0, -0.5], "query_own_start": 105, "terminal_at": [1],
                            "generated_inputs": 1}}}}
    assert expand_captures([first, compact]) == [first, second]
    with pytest.raises(ValueError, match="prompt delta"):
        expand_captures([compact])


def test_lowering_puts_a_forced_call_opening_into_the_prompt():
    import sys, collections
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/neuralese_data'))
    from lower_rollout_turns import episode_turns
    def turn(inputs, samples, **extra):
        return {'invocation_id': 'a', 'sampling': {'policy': {'checkpoint_sha256': 'p'}, 'prompt': [
            {'kind': 'tokens', 'token_ids': [1, 2], 'producing': [False, False]}],
            'samples': [{'index': i, 'token_id': t, 'logp': -0.5, 'input_role': 'ordinary'} for i, t in enumerate(samples)],
            'generated_inputs': inputs, **extra}}
    forced = turn([{'token_id': 7, 'sample_index': None}, {'token_id': 8, 'sample_index': None},
                   {'token_id': 9, 'sample_index': 0}], [9, 10], constrained={'tool': 'compact_history'})
    middle = turn([{'token_id': 9, 'sample_index': 0}, {'token_id': 7, 'sample_index': None}], [9, 10])
    stats = collections.Counter()
    rows = episode_turns({'turns': [forced, middle]}, stats)
    assert len(rows) == 1 and rows[0]['prompt'].tolist() == [1, 2, 7, 8] and rows[0]['sample'].tolist() == [9, 10]
    assert stats['forced_prefix_calls'] == 1 and stats['skip_forced_after_sample'] == 1


def test_lowering_keeps_sampled_decisions_as_categorical_actions():
    import sys, collections, math
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/neuralese_data'))
    from lower_rollout_turns import decision_turns, episode_turns
    decision = {'invocation_id': 'd', 'sampling': {'policy': {'checkpoint_sha256': 'p'}, 'prompt': [
        {'kind': 'tokens', 'token_ids': [1, 2, 3]}], 'continuations': [[5, 6], [5, 7, 8]],
        'decision_sample': {'schema': 'natlang.decision-sample/1', 'log_probs': [-3.0, -1.0], 'chosen': 1, 'temperature': 1}}}
    argmax = {'invocation_id': 'e', 'sampling': {'policy': {'checkpoint_sha256': 'p'}, 'prompt': [
        {'kind': 'tokens', 'token_ids': [1]}], 'continuations': [[5], [6]]}}
    stats = collections.Counter()
    assert episode_turns({'turns': [decision, argmax]}, stats) == []
    assert stats['decision_calls'] == 1 and stats['skip_no_sample_stream'] == 1
    rows = decision_turns({'turns': [decision, argmax]}, stats)
    assert len(rows) == 1 and rows[0]['prompt'].tolist() == [1, 2, 3] and rows[0]['chosen'] == 1
    assert [o.tolist() for o in rows[0]['options']] == [[5, 6], [5, 7, 8]]
    assert math.isclose(rows[0]['behavior_logq'], -math.log1p(math.exp(-2.0)), rel_tol=1e-9)
    import grpo
    ids, lengths, positions, targets, owner, counts = grpo.decision_tensors([{'row': rows[0], 'advantage': 1.}], 'cpu')
    assert ids.tolist() == [1, 2, 3, 5, 6, 1, 2, 3, 5, 7, 8] and lengths == [5, 6] and counts == [2]
    assert positions.tolist() == [2, 3, 7, 8, 9] and targets.tolist() == [5, 6, 5, 7, 8] and owner.tolist() == [0, 0, 1, 1, 1]
