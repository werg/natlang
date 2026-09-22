"""IR remains semantic and can be verified after lowering to the current harness."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from generate import run_program  # noqa: E402
from program_ir import VERSION, decision_record, lower, validate  # noqa: E402
from build_program_ir import execute_scene_nodes, parse_numeric_program  # noqa: E402
from natlang.gen.policy import native_text  # noqa: E402
from audit_program_ir import audit  # noqa: E402
from build_synthetic_ir import SUPPORTED, freeze  # noqa: E402
from generate import make_program  # noqa: E402
from build_scenario_ir import scenario_record  # noqa: E402
from generate_failures import matched_cases  # noqa: E402
from audit_trajectory_admission import audit_turns, audit_actions  # noqa: E402
from materialize_ir import materialize_record  # noqa: E402
from build_review_ir import review_record  # noqa: E402
from materialize_review_ir import render_review  # noqa: E402
from generate_agent_support import support_cases  # noqa: E402


def task(i, state, gold):
    return {"id": f"fixture/{i}", "source": "fixture", "group_id": "g",
            "split": "train", "state": state, "instruction": "Is it urgent?",
            "kind": "boolean", "labels": ["false", "true"], "gold": gold,
            "license": "cc0-1.0", "jev": {"request": "must not enter IR"},
            "source_meta": {"question_key": "urgent", "teacher_model": "must not enter IR"}}


def test_decision_ir_discards_provider_payload_and_verifies_composition():
    record = decision_record("decision_map_report", "map:fixture", [
        task(1, "later", "false"), task(2, "now", "true")])
    assert "must not enter IR" not in json.dumps(record)
    assert "tool_calls" not in json.dumps(record)
    samples, episodes = run_program(lower(json.loads(json.dumps(record))))
    assert episodes == 3
    assert {s["skill"] for s in samples} >= {"for_each", "run_code", "write_value", "reply"}


def test_state_sequence_resolves_repeated_instruction_with_history():
    record = {"version": VERSION, "id": "fixture:sequence", "kind": "state_sequence",
              "source": "fixture", "split": "train", "source_ids": ["one"],
              "source_groups": ["one"], "license": "cc0-1.0",
              "semantics": {"domain": "fixture", "initial_state": "same",
                            "steps": [{"utterance": "move it", "before": "same", "after": "same"},
                                      {"utterance": "move it", "before": "same", "after": "other"}]}}
    samples, episodes = run_program(lower(validate(record)))
    assert episodes == 3
    assert sum(s["skill"] == "run_function" for s in samples) == 2


def test_state_sequence_rejects_broken_chain():
    record = {"version": VERSION, "id": "fixture:bad", "kind": "state_sequence",
              "source": "fixture", "split": "train", "source_ids": ["one"],
              "source_groups": ["one"], "license": "cc0-1.0",
              "semantics": {"domain": "fixture", "initial_state": "a",
                            "steps": [{"utterance": "move", "before": "b", "after": "c"}]}}
    with pytest.raises(ValueError, match="broken"):
        validate(record)


def test_numeric_program_uses_intermediate_results_and_native_decimal():
    steps, value = parse_numeric_program("divide(1, 1000000), multiply(#0, 2)")
    assert value == 0.000002
    assert steps[1]["args"][0] == 0
    assert "e-" not in native_text([("write", {"path": "return", "type": "Num", "value": 0.000000932})])
    record = {"version": VERSION, "id": "fixture:numeric", "kind": "numeric_program",
              "source": "fixture", "split": "train", "source_ids": ["one"],
              "source_groups": ["one"], "license": "cc0-1.0",
              "semantics": {"question": "Twice one millionth?", "evidence": "1 and 1000000",
                            "steps": steps, "answer_type": "Num", "answer": value}}
    samples, _ = run_program(lower(validate(record)))
    assert sum(s["skill"] == "run_code" for s in samples) == 2


def test_scene_program_executes_graph_nodes_in_harness():
    scene = {"objects": [{"color": "red", "size": "small", "shape": "cube", "material": "metal"},
                         {"color": "blue", "size": "large", "shape": "sphere", "material": "rubber"}],
             "relationships": {"left": [[1], []]}}
    nodes = [{"function": "scene", "inputs": [], "value_inputs": []},
             {"function": "filter_color", "inputs": [0], "value_inputs": ["red"]},
             {"function": "unique", "inputs": [1], "value_inputs": []},
             {"function": "relate", "inputs": [2], "value_inputs": ["left"]},
             {"function": "count", "inputs": [3], "value_inputs": []}]
    assert execute_scene_nodes(nodes, scene) == 1
    record = {"version": VERSION, "id": "fixture:scene", "kind": "scene_program",
              "source": "fixture", "split": "train", "source_ids": ["one"],
              "source_groups": ["one"], "license": "cc0-1.0",
              "semantics": {"question": "How many objects are left of the red cube?",
                            "scene": scene, "nodes": nodes, "answer_type": "Num", "answer": 1}}
    samples, _ = run_program(lower(validate(record)))
    assert sum(s["skill"] == "run_code" for s in samples) == len(nodes)

    # Larger graphs may compile into one exact
    # program while preserving every semantic node in the IR.
    many = [{"function": "scene", "inputs": [], "value_inputs": []} for _ in range(17)]
    many.append({"function": "count", "inputs": [16], "value_inputs": []})
    large = {**record, "id": "fixture:large-scene",
             "semantics": {**record["semantics"], "nodes": many, "answer": 2}}
    samples, _ = run_program(lower(validate(large)))
    assert sum(s["skill"] == "run_code" for s in samples) == 1


def test_sharded_materializer_resumes_only_same_ir_and_harness(tmp_path):
    record = decision_record("decision_leaf", "leaf:fixture", [task(1, "now", "true")])
    src, dst = tmp_path / "programs.jsonl", tmp_path / "shards"
    src.write_text(json.dumps(record) + "\n")
    script = Path(__file__).resolve().parents[1] / "scripts/materialize_ir_shards.py"
    command = [sys.executable, str(script), str(src), str(dst), "--workers", "1", "--shard-size", "1"]
    first = subprocess.run(command, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    assert len(list(dst.glob("part-*.jsonl.gz"))) == 1
    second = subprocess.run(command, capture_output=True, text=True)
    assert second.returncode == 0 and '"new_shards": 0' in second.stdout
    changed = subprocess.run(command[:-1] + ["2"], capture_output=True, text=True)
    assert changed.returncode != 0 and "changed" in changed.stderr


def test_ir_audit_rejects_duplicate_ids_and_cross_split_groups(tmp_path):
    row = decision_record("decision_leaf", "leaf:fixture", [task(1, "now", "true")])
    first, second = tmp_path / "one.jsonl", tmp_path / "two.jsonl"
    first.write_text(json.dumps(row) + "\n")
    second.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="duplicate program"):
        audit([first, second])
    second.write_text(json.dumps({**row, "source": "another-source"}) + "\n")
    with pytest.raises(ValueError, match="duplicate program"):
        audit([first, second])
    other = {**row, "id": "leaf:other", "split": "dev"}
    second.write_text(json.dumps(other) + "\n")
    with pytest.raises(ValueError, match="crosses splits"):
        audit([first, second])


def test_synthetic_families_round_trip_through_ir():
    provenance = {"seed": 72, "sources": "fixture", "historical_match": False}
    for family in sorted(SUPPORTED):
        _, program = make_program(72, 12, [family])
        record = json.loads(json.dumps(freeze(program, f"72:{family}:12", provenance)))
        samples, _ = run_program(lower(record))
        assert samples and samples[-1]["skill"] == "reply"


def test_scenario_ir_verifies_errors_blockers_faults_and_effects():
    for case in matched_cases(81, 0):
        record = scenario_record(case, "failures", 81, 0, ["fixture"])
        samples, _ = run_program(lower(validate(json.loads(json.dumps(record)))))
        assert samples
        if case["failure"]:
            assert samples[-1]["skill"] == ("report_error" if case["failure"] == "error" else "report_blocker")
        else:
            assert samples[-1]["skill"] == "reply"
        audit_turns(record, samples)


def test_outcome_contract_rejects_destination_change_and_skipped_effect_call():
    cases = {case["name"]: case for case in matched_cases(81, 0)}
    binding = scenario_record(cases["binding_error"], "failures", 81, 0, ["fixture"])
    assert binding["semantics"]["contract"]["constrained_calls"][0]["to"] == "return"
    with pytest.raises(ValueError, match="changed required call destination"):
        audit_actions(binding, [{"tool": "call", "arguments": {
            "function": "size_of", "to": "return/size", "inputs": {"items": "args/items"}}}], complete=False)
    effect = scenario_record(cases["effect_error"], "failures", 81, 0, ["fixture"])
    assert effect["semantics"]["contract"]["effects"]
    with pytest.raises(ValueError, match="expected"):
        audit_actions(effect, [{"tool": "report_error", "arguments": {
            "message": effect["semantics"]["contract"]["explanation"]}}])
    lines, _ = materialize_record(effect)
    assert len(lines) == 2


def test_scenario_materialization_can_link_an_admitted_whole_run_trace(tmp_path):
    case = next(c for c in matched_cases(81, 0) if c["name"] == "effect_error")
    record = scenario_record(case, "failures", 81, 0, ["fixture"])
    lines, _ = materialize_record(record, trace_dir=tmp_path)
    assert lines and all(line["trace"]["admitted"] for line in lines)
    assert len({line["trace"]["trace_sha256"] for line in lines}) == 1
    assert (tmp_path / next(iter(tmp_path.iterdir())).name).exists()


def test_linked_review_renders_from_fresh_base_execution():
    case = next(c for c in support_cases(113, 0) if c["name"] == "binding_repair")
    base = scenario_record(case, "support", 113, 0, ["fixture"])
    samples, _ = run_program(lower(base))
    proposed = [(c["function"]["name"], json.loads(c["function"]["arguments"]))
                for c in samples[0]["target"]["tool_calls"]]
    old = {"id": "support:113:0:binding_repair:0:review0",
           "contrast_group": "support:113:0:binding_repair:0",
           "proposal": proposed, "expected": "approve", "proposal_justified": True,
           "task_feasible": True, "lesson_ids": ["exact_destination"],
           "target": {"tool_calls": [{"function": {"arguments": json.dumps({
               "decision": "approve", "reason": "The destination matches."})}}]}}
    review = json.loads(json.dumps(review_record(old, base, "fixture")))
    rendered = render_review(review, base, samples)
    assert rendered["target"]["tool_calls"][0]["function"]["name"] == "review_write"
    assert "The destination matches." in rendered["native_target"]
