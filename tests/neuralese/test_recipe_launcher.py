import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_neuralese_recipe.py"
SPEC = importlib.util.spec_from_file_location("run_neuralese_recipe", SCRIPT)
LAUNCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LAUNCHER)


def resume_fixture(tmp_path):
    output = tmp_path / "stage-output"
    stage_output = output / "direct"
    stage_output.mkdir(parents=True)
    (stage_output / "checkpoint.pt").write_bytes(b"committed checkpoint")
    source = tmp_path / "source.jsonl"
    source.write_bytes(b"pinned input\n")
    code = {"train/direct_sft.py": "a" * 64}
    stage = {"id": "direct", "kind": "neuralese_direct"}
    inputs = {"text_data": {"path": str(source), "sha256": LAUNCHER.sha256(source)}}
    plan = {
        "schema": "natlang.neuralese-direct-stage-plan/1",
        "stage": stage,
        "inputs": inputs,
        "device": "cuda",
        "runtime_image": "sha256:" + "1" * 64,
        "frozen_code": code,
        "recipe_id": "resume-fixture-v1",
        "parameters": {},
        "lineage": {"evidence": []},
        "lineage_evidence": [],
        "stage_output": str(stage_output.resolve()),
    }
    (output / "direct-stage-plan.json").write_text(json.dumps(plan))
    recipe = {"id": plan["recipe_id"], "stage": stage, "runtime": {"image": plan["runtime_image"]},
              "inputs": inputs, "parameters": {}, "lineage": plan["lineage"]}
    return output, recipe, code


def test_exact_existing_direct_resume_binds_plan_and_checkpoint(tmp_path):
    output, recipe, code = resume_fixture(tmp_path)

    binding = LAUNCHER.validate_existing_direct_resume(output, recipe, "cuda", code, False)

    assert binding["frozen_code_changed"] is False
    assert binding["checkpoint_sha256"] == LAUNCHER.sha256(Path(binding["checkpoint_path"]))
    assert binding["plan_sha256"] == LAUNCHER.sha256(Path(binding["plan_path"]))


def test_resume_requires_handoff_only_for_changed_code(tmp_path):
    output, recipe, code = resume_fixture(tmp_path)
    changed = {"train/direct_sft.py": "b" * 64}

    with pytest.raises(ValueError, match="requires --handoff"):
        LAUNCHER.validate_existing_direct_resume(output, recipe, "cuda", changed, False)

    binding = LAUNCHER.validate_existing_direct_resume(output, recipe, "cuda", changed, True)
    assert binding["frozen_code_changed"] is True

    with pytest.raises(ValueError, match="only valid when"):
        LAUNCHER.validate_existing_direct_resume(output, recipe, "cuda", code, True)


def test_resume_rejects_missing_checkpoint(tmp_path):
    output, recipe, code = resume_fixture(tmp_path)
    (output / "direct" / "checkpoint.pt").unlink()

    with pytest.raises(ValueError, match="committed handler checkpoint"):
        LAUNCHER.validate_existing_direct_resume(output, recipe, "cuda", code, False)
