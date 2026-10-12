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


def test_direct_epoch_budget_change_is_only_allowed_for_exact_epoch_only_direct_handoff(tmp_path):
    output, recipe, code = resume_fixture(tmp_path)
    plan_path = output / "direct-stage-plan.json"
    plan = json.loads(plan_path.read_text())
    plan["parameters"] = {"epochs": 2.0, "batch": 4}
    plan_path.write_text(json.dumps(plan))
    recipe["parameters"] = {"epochs": 1.0, "batch": 4}
    changed_code = {"train/direct_sft.py": "b" * 64}

    with pytest.raises(ValueError, match="training parameters"):
        LAUNCHER.validate_existing_direct_resume(output, recipe, "cuda", changed_code, False)
    binding = LAUNCHER.validate_existing_direct_resume(output, recipe, "cuda", changed_code, True)
    assert binding["epoch_budget_handoff"] == {
        "old_epochs": 2.0,
        "new_epochs": 1.0,
        "verification": "trainer-reconstructs-exact-schedule-prefix-and-checkpoint-step",
    }

    recipe["parameters"]["batch"] = 8
    with pytest.raises(ValueError, match="training parameters"):
        LAUNCHER.validate_existing_direct_resume(output, recipe, "cuda", changed_code, True)
    recipe["parameters"] = {"epochs": 1.0, "batch": 4}
    recipe["stage"]["kind"] = "core_text_warmup"
    with pytest.raises(ValueError, match="changes stage"):
        LAUNCHER.validate_existing_direct_resume(output, recipe, "cuda", changed_code, True)


@pytest.mark.parametrize("old,new", [(1.0, 1.0), (0.0, 1.0), (2.0, 0.0), ("2", 1.0), (True, 1.0)])
def test_direct_epoch_budget_change_rejects_invalid_or_unchanged_epochs(old, new):
    assert recipe_module.direct_epoch_budget_change(
        "neuralese_direct", {"epochs": old, "batch": 4}, {"epochs": new, "batch": 4}) is None


def test_direct_epoch_budget_change_rejects_other_parameter_changes():
    assert recipe_module.direct_epoch_budget_change(
        "neuralese_direct", {"epochs": 2.0, "batch": 4}, {"epochs": 1.0, "batch": 8}) is None
    assert recipe_module.direct_epoch_budget_change(
        "core_text_warmup", {"epochs": 2.0, "batch": 4}, {"epochs": 1.0, "batch": 4}) is None


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


def shared_resume_fixture(tmp_path, monkeypatch):
    output = tmp_path / "shared-output"
    frozen = output / "runtime" / "natlang_neuralese"
    frozen.mkdir(parents=True)
    code_file = frozen / "train" / "direct_sft.py"
    code_file.parent.mkdir(parents=True)
    code_file.write_bytes(b"pinned trainer code\n")
    code = {"train/direct_sft.py": LAUNCHER.sha256(code_file)}
    recipe_path = tmp_path / "shared-recipe.json"
    recipe_path.write_text("{}\n")
    recipe = {
        "schema": "natlang.neuralese-training-recipe/1",
        "id": "shared-resume-fixture-v1",
        "runtime": {"image": "sha256:" + "1" * 64, "package_code": code},
        "outputs": {"directory": str(output)},
        "input_bindings": {"heads": {"path": str(tmp_path / "heads.pt"), "sha256": "b" * 64}},
        "lineage": {"evidence": []},
        "stages": [
            {"id": "header", "kind": "neuralese_direct", "requires": [],
             "inputs": {"heads": "heads"}, "parameters": {"text_data": "records.jsonl"}},
            {"id": "whole", "kind": "neuralese_direct", "requires": ["header"],
             "inputs": {"heads": "header_heads"}, "parameters": {"text_data": "records.jsonl"}},
        ],
    }
    stage_inputs = {
        "header": {"heads": {"path": str(tmp_path / "heads.pt"), "sha256": "b" * 64,
                              "binding": "heads"}},
        "whole": {"heads": {"path": str(output / "header" / "checkpoint.pt"),
                             "sha256": "c" * 64, "binding": "header_heads"}},
    }
    heads = {"path": str(tmp_path / "heads.pt"), "sha256": "b" * 64}
    inputs = {heads["path"]: heads["sha256"],
              str(output / "header" / "checkpoint.pt"): "c" * 64}
    monkeypatch.setattr(recipe_module, "recipe_identity_sha256", lambda _path: "d" * 64)
    plan = {"schema": "natlang.neuralese-recipe-plan/1", "recipe": recipe,
            "recipe_sha256": "d" * 64, "inputs": inputs, "device": "cuda",
            "stage_inputs": stage_inputs, "code": code}
    (output / "recipe-plan.json").write_text(json.dumps(plan))
    return output, recipe, recipe_path, heads, stage_inputs, code


def test_exact_existing_shared_resume_binds_recipe_inputs_device_and_frozen_runtime(tmp_path, monkeypatch):
    output, recipe, recipe_path, heads, stage_inputs, code = shared_resume_fixture(tmp_path, monkeypatch)

    binding = LAUNCHER.validate_existing_shared_resume(
        output, recipe, recipe_path, heads, stage_inputs, "cuda", code)

    assert binding["frozen_code"] == code
    assert binding["plan_sha256"] == LAUNCHER.sha256(output / "recipe-plan.json")
    assert binding["device"] == "cuda"
    assert binding["frozen_code_changed"] is False


def test_shared_resume_code_handoff_allows_only_a_pinned_package_code_change(tmp_path, monkeypatch):
    output, recipe, recipe_path, heads, stage_inputs, _old_code = shared_resume_fixture(tmp_path, monkeypatch)
    plan_path = output / "recipe-plan.json"
    plan = json.loads(plan_path.read_text())
    new_code = {"train/direct_sft.py": "e" * 64}
    recipe = json.loads(json.dumps(recipe))
    recipe["runtime"]["package_code"] = new_code

    binding = LAUNCHER.validate_existing_shared_resume(
        output, recipe, recipe_path, heads, stage_inputs, "cuda", new_code, handoff=True)

    assert binding["frozen_code_changed"] is True
    assert binding["frozen_code"] == plan["code"]

    recipe["stages"][0]["parameters"]["learning_rate"] = 0.5
    with pytest.raises(ValueError, match="changes more than pinned runtime code"):
        LAUNCHER.validate_existing_shared_resume(
            output, recipe, recipe_path, heads, stage_inputs, "cuda", new_code, handoff=True)


def test_shared_resume_handoff_allows_epoch_only_neuralese_direct_budget_change(tmp_path, monkeypatch):
    output, recipe, recipe_path, heads, stage_inputs, _old_code = shared_resume_fixture(tmp_path, monkeypatch)
    plan_path = output / "recipe-plan.json"
    plan = json.loads(plan_path.read_text())
    plan["recipe"]["stages"][0]["parameters"]["epochs"] = 2.0
    plan_path.write_text(json.dumps(plan))
    new_code = {"train/direct_sft.py": "e" * 64}
    recipe = json.loads(json.dumps(recipe))
    recipe["runtime"]["package_code"] = new_code
    recipe["stages"][0]["parameters"]["epochs"] = 1.0

    binding = LAUNCHER.validate_existing_shared_resume(
        output, recipe, recipe_path, heads, stage_inputs, "cuda", new_code, handoff=True)
    assert binding["epoch_budget_changes"] == {
        "header": {
            "old_epochs": 2.0,
            "new_epochs": 1.0,
            "verification": "trainer-reconstructs-exact-schedule-prefix-and-checkpoint-step",
        }
    }

    recipe["stages"][0]["parameters"]["batch"] = 2
    with pytest.raises(ValueError, match="changes more than pinned runtime code"):
        LAUNCHER.validate_existing_shared_resume(
            output, recipe, recipe_path, heads, stage_inputs, "cuda", new_code, handoff=True)


def test_shared_resume_handoff_requires_changed_package_code(tmp_path, monkeypatch):
    output, recipe, recipe_path, heads, stage_inputs, code = shared_resume_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="requires changed shared package code"):
        LAUNCHER.validate_existing_shared_resume(
            output, recipe, recipe_path, heads, stage_inputs, "cuda", code, handoff=True)


@pytest.mark.parametrize("field", ["recipe", "inputs", "device", "stage_inputs"])
def test_shared_resume_rejects_changed_committed_plan_fields(tmp_path, monkeypatch, field):
    output, recipe, recipe_path, heads, stage_inputs, code = shared_resume_fixture(tmp_path, monkeypatch)
    plan_path = output / "recipe-plan.json"
    plan = json.loads(plan_path.read_text())
    if field == "recipe":
        plan[field]["id"] = "different-recipe"
    elif field == "inputs":
        plan[field]["/elsewhere/heads.pt"] = "e" * 64
    elif field == "device":
        plan[field] = "cpu"
    else:
        plan[field]["whole"]["heads"]["sha256"] = "e" * 64
    plan_path.write_text(json.dumps(plan))

    with pytest.raises(ValueError, match=f"committed {field}"):
        LAUNCHER.validate_existing_shared_resume(
            output, recipe, recipe_path, heads, stage_inputs, "cuda", code)


def test_shared_resume_rejects_changed_or_missing_frozen_runtime(tmp_path, monkeypatch):
    output, recipe, recipe_path, heads, stage_inputs, code = shared_resume_fixture(tmp_path, monkeypatch)
    frozen_file = output / "runtime" / "natlang_neuralese" / "train" / "direct_sft.py"
    frozen_file.write_bytes(b"different code\n")

    with pytest.raises(ValueError, match="frozen recipe runtime differs"):
        LAUNCHER.validate_existing_shared_resume(
            output, recipe, recipe_path, heads, stage_inputs, "cuda", code)

    frozen_file.unlink()
    with pytest.raises(ValueError, match="committed code inventory"):
        LAUNCHER.validate_existing_shared_resume(
            output, recipe, recipe_path, heads, stage_inputs, "cuda", code)


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
