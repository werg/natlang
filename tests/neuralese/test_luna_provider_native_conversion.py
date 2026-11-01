import copy
import json
from pathlib import Path

import pytest

from scripts.neuralese_data.build_luna_provider_native_conversion import build
from natlang_neuralese.common.hashing import sha256_hex
from natlang_neuralese.common.jsonio import canonical_json_bytes


def _write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = data if isinstance(data, bytes) else canonical_json_bytes(data) + b"\n"
    path.write_bytes(raw)
    return raw


def _fixture(tmp_path, *, raw_override=None, decision_index=0, judge="gpt-6-sol"):
    source = tmp_path / "source"
    output = tmp_path / "output"
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    runtime_bytes = _write(runtime / "frozen-runtime.json", {"schema": "test-runtime"})
    groups = ["upstream:train:world"]
    program = {"id": "codebase:test:world", "split": "train", "source_groups": groups}
    selected_path = tmp_path / "selected.jsonl"
    selected_bytes = _write(selected_path, canonical_json_bytes(program) + b"\n")
    protection_path = tmp_path / "protection.json"
    protection_bytes = _write(protection_path, {
        "schema": "natlang.luna-codebase-protection-preflight/1",
        "selected_train_rows": 1,
        "selected_receipts": [{"case_id": program["id"], "source_groups": groups,
            "receipt": {"split": "train", "protected_sha256": "p" * 64}}],
    })
    raw = {
        "version": "natlang.teacher_trajectory.native/1", "id": "teacher-case",
        "task": {"program_ir": program},
        "provenance": {"model": "gpt-6-luna", "provider": "openai-codex",
            "transport": "pi-provider", "collection_role": "teacher", "neuralese_server": False},
        "outcome": {"status": "done", "accepted": True,
            "oracle": {"accepted": True, "level": "judged"}},
        "request_telemetry": {"starts": [{"role": "judge", "purpose": "judge", "status": "completed"}]},
        "trajectory": [{"context": [{"role": "user", "content": "inspect"}]}],
    }
    if raw_override:
        raw = raw_override(raw)
    raw_path = source / "raw.jsonl"
    raw_bytes = _write(raw_path, canonical_json_bytes(raw) + b"\n")
    messages = raw["trajectory"][0]["context"]
    decisions_path = source / "decisions.jsonl"
    decision = {
        "id": "teacher-case:decision:0000", "split": "train", "source_groups": [*groups, program["id"]],
        "outcome": {"accepted": True}, "training_admission": {"approved": True}, "target": {"role": "assistant", "content": "done"},
        "source_ref": {"trajectory_id": "teacher-case", "provider_input_context": {
            "schema": "natlang.captured-provider-input-context/1",
            "source_trajectory_index": decision_index,
            "student_context_sha256": sha256_hex(canonical_json_bytes(messages))}},
    }
    decisions_bytes = _write(decisions_path, canonical_json_bytes(decision) + b"\n")
    plan_path = tmp_path / "plan.json"
    plan = {"root_approved": True, "model_id": "gpt-6-luna", "provider": "openai-codex",
        "judge_model": {"model_id": judge, "provider": "openai-codex"},
        "runtime": str(runtime.resolve()), "runtime_receipt_sha256": sha256_hex(runtime_bytes),
        "source": str(selected_path.resolve()),
        "artifact_hashes": {str(selected_path.resolve()): sha256_hex(selected_bytes),
            str(protection_path.resolve()): sha256_hex(protection_bytes),
            str((runtime / "frozen-runtime.json").resolve()): sha256_hex(runtime_bytes)}}
    plan_bytes = _write(plan_path, plan)
    manifest_path = source / "provider-source.json"
    source_manifest = {"schema": "natlang.provider-campaign-source/1", "model": "gpt-6-luna",
        "provider": "openai-codex", "training_admission": False, "training_exposure": False,
        "oracle_regrades": [], "source": {"path": str(raw_path.resolve()), "sha256": sha256_hex(raw_bytes), "bytes": len(raw_bytes)},
        "results": [{"result": {"sha256": sha256_hex(raw_bytes), "bytes": len(raw_bytes)}}],
        "plan": {"path": str(plan_path.resolve()), "sha256": sha256_hex(plan_bytes)}}
    _write(manifest_path, source_manifest)
    return {"raw": raw_path, "decisions": decisions_path, "source": manifest_path,
        "plan": plan_path, "runtime": runtime, "protection": protection_path, "output": output}


def _build(paths):
    return build(paths["raw"], paths["decisions"], paths["source"], paths["plan"],
        paths["runtime"], paths["protection"], paths["output"])


def test_generic_audit_serializes_one_complete_judged_provider_episode(tmp_path):
    paths = _fixture(tmp_path)
    _build(paths)
    audit = json.loads((paths["output"] / "ingestion-audit.json").read_bytes())
    assert audit["schema"] == "natlang.luna-provider-native-ingestion-audit/1"
    assert audit["trajectory_count"] == audit["converted_decision_count"] == 1
    assert audit["approved_decision_count"] == 1
    assert audit["oracle_level"] == "judged" and audit["exact_oracle_claimed"] is False
    assert audit["training_admission"] is False


def test_generic_audit_rejects_noncontiguous_decision_chain(tmp_path):
    paths = _fixture(tmp_path, decision_index=1)
    with pytest.raises(ValueError, match="unique turn of the source episode"):
        _build(paths)


def test_generic_audit_rejects_changed_judge_binding(tmp_path):
    paths = _fixture(tmp_path, judge="gpt-6-luna")
    with pytest.raises(ValueError, match="independent Sol judgment"):
        _build(paths)


def test_generic_audit_rejects_changed_raw_judgment_bytes(tmp_path):
    paths = _fixture(tmp_path)
    raw = json.loads(paths["raw"].read_bytes())
    raw["outcome"]["oracle"]["verdict"] = "changed after source snapshot"
    _write(paths["raw"], canonical_json_bytes(raw) + b"\n")
    with pytest.raises(ValueError, match="exact Luna artifact"):
        _build(paths)


def test_generic_audit_rejects_non_luna_source_provenance(tmp_path):
    def changed(raw):
        value = copy.deepcopy(raw)
        value["provenance"]["model"] = "gpt-6-sol"
        return value
    paths = _fixture(tmp_path, raw_override=changed)
    with pytest.raises(ValueError, match="actual Luna teacher result"):
        _build(paths)
