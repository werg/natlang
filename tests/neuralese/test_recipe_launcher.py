import importlib.util
import json
from pathlib import Path

import pytest
import natlang_neuralese.train.recipe as recipe_module


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


def test_output_launch_lock_rejects_a_second_owner(tmp_path):
    output = tmp_path / "stage-output"
    first, _ = LAUNCHER.acquire_output_launch_lock(output)
    try:
        with pytest.raises(ValueError, match="another recipe launcher owns"):
            LAUNCHER.acquire_output_launch_lock(output)
    finally:
        import os
        os.close(first)

    second, _ = LAUNCHER.acquire_output_launch_lock(output)
    import os
    os.close(second)


def test_shared_recipe_requires_pinned_runtime_image_and_package_code(tmp_path, monkeypatch):
    recipe_path = tmp_path / "shared.json"
    recipe_path.write_text(json.dumps({"schema": "natlang.neuralese-training-recipe/1"}))
    recipe = {
        "schema": "natlang.neuralese-training-recipe/1",
        "id": "two-phase-fixture",
        "runtime": {"image": "sha256:" + "1" * 64, "package_code": {"train/direct_sft.py": "a" * 64}},
        "outputs": {"directory": str(tmp_path / "output")},
        "input_bindings": {"heads": {"path": str(tmp_path / "heads.pt"), "sha256": "b" * 64}},
        "lineage": {"evidence": []},
        "stages": [{"id": "header", "kind": "neuralese_direct", "requires": [], "inputs": {"heads": "heads"}, "parameters": {}},
                   {"id": "whole", "kind": "neuralese_direct", "requires": ["header"], "inputs": {"heads": "heads"}, "parameters": {}}],
    }
    monkeypatch.setattr(recipe_module, "load_recipe", lambda _path: recipe)

    assert LAUNCHER.read_shared_recipe(recipe_path) is recipe

    recipe["runtime"].pop("image")
    with pytest.raises(ValueError, match="runtime.image"):
        LAUNCHER.read_shared_recipe(recipe_path)


def test_shared_recipe_inputs_resolve_heads_from_first_stage_binding(monkeypatch):
    path = "/tmp/pinned-heads.pt"
    recipe = {"stages": [
        {"id": "header", "kind": "neuralese_direct", "parameters": {}, "inputs": {"heads": "base_heads"}},
        {"id": "whole", "kind": "neuralese_direct", "parameters": {}, "inputs": {"heads": "header_heads"}},
        {"id": "held", "kind": "neuralese_direct", "admitted": False, "parameters": {}, "inputs": {"heads": "held_heads"}},
    ]}
    calls = []

    def resolve(_recipe, stage, defaults, overrides):
        calls.append(stage["id"])
        return {"heads": {"path": path, "sha256": "c" * 64, "binding": stage["inputs"]["heads"]}}

    monkeypatch.setattr(recipe_module, "resolve_stage_inputs", resolve)
    stage_inputs, heads = LAUNCHER.resolve_shared_inputs(recipe)

    assert calls == ["header", "whole"]
    assert heads == {"path": path, "sha256": "c" * 64, "binding": "base_heads"}
    assert stage_inputs["held"] == {}


@pytest.mark.parametrize("value", ["0", "1:0", "4:3", "1:2:3", "x:2", " 1:2"])
def test_profile_range_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        LAUNCHER.validated_profile_updates(value)


def test_profile_range_and_container_names_are_forwardable():
    assert LAUNCHER.validated_profile_updates("2:6") == "2:6"
    assert LAUNCHER.validated_profile_updates("7") == "7"
    assert LAUNCHER.validated_container_name("pop-header2-whole1-1234") == "pop-header2-whole1-1234"

    with pytest.raises(ValueError, match="Docker-safe"):
        LAUNCHER.validated_container_name("bad name")


def test_profile_environment_is_added_to_the_named_training_container():
    command = LAUNCHER.build_training_docker_command(
        ["docker", "run", "--rm"], "sha256:" + "1" * 64, "pop-header2-whole1-1234",
        "cuda", ["-m", "natlang_neuralese.train.recipe"], "2:6")

    assert command[command.index("--name") + 1] == "pop-header2-whole1-1234"
    assert command[command.index("-e") + 1] == "NATLANG_PROFILE_UPDATES=2:6"
    assert command.index("NATLANG_PROFILE_UPDATES=2:6") < command.index("sha256:" + "1" * 64)
