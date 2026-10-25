import hashlib
import ast
import importlib.util
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from scripts.coreweave import stage_teacher_helper_continuation as stage


def _loaded_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _execute_staged_proof_calls(path, manifest):
    source = Path(path).read_text()
    tree = ast.parse(source)
    functions = {node.name: node for node in ast.walk(tree)
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    api_path = Path(manifest["successor_helper"]["path"])
    verifier_path = Path(manifest["successor_verifier"]["path"])
    api = _loaded_module("scripts.coreweave.teacher_tranche_admission", api_path)
    import scripts.coreweave as coreweave
    coreweave.teacher_tranche_admission = api
    verifier = _loaded_module("teacher_helper_continuation", verifier_path)
    env = {
        "HELPER_TRANSITION": manifest["successor_transition"],
        "SUCCESSOR_HELPER_BINDING": manifest["successor_helper"],
        "SUCCESSOR_VERIFIER_BINDING": manifest["successor_verifier"],
        "ADMISSION_API_PATH": api_path,
        "ROOT": stage.ROOT,
        "pathlib": __import__("pathlib"),
        "_helper_continuation": verifier,
    }
    for name in ("verify_new_helpers", "verify_execution_helper"):
        node = functions[name]
        exec(compile(textwrap.dedent(ast.get_source_segment(source, node)), str(path), "exec"), env)
    env["verify_new_helpers"]()
    builder = stage.ROOT / "scripts/neuralese_data/build_verified_teacher_trajectory_graphs.py"
    builder_ref = stage.file_binding(builder)
    execution = {"helper_continuation": manifest["successor_transition"],
                 "converter": {"path": builder_ref["path"], "sha256": builder_ref["sha256"]}}
    assert env["verify_execution_helper"](execution) == "successor"
    api_digest_assertions = [node for node in ast.walk(tree) if isinstance(node, ast.Assert) and
                             any(isinstance(sub, ast.Name) and sub.id == "ADMISSION_API_SHA256"
                                 for sub in ast.walk(node.test)) and
                             any(isinstance(sub, ast.Name) and sub.id == "resolver"
                                 for sub in ast.walk(node.test))]
    if Path(path).name in {"convert-flint-both-v2-0aad.py", "convert-current-teacher-both-v3-0aad.py"}:
        assert api_digest_assertions
        assertion_source = textwrap.dedent(ast.get_source_segment(source, api_digest_assertions[0]))
        from natlang_neuralese.common.hashing import sha256_file_hex
        resolver_env = {"sha256_file_hex": sha256_file_hex, "resolver": api_path,
                        "ADMISSION_API_SHA256": manifest["successor_helper"]["sha256"]}
        exec(compile(assertion_source, str(path), "exec"), resolver_env)


def _review(path: Path):
    old = json.loads(Path(stage.custody.LEGACY_OBSERVATION["path"]).read_bytes())
    successor_inventory = {
        relative: stage.file_binding(stage.ROOT / relative)
        for relative in old["reviewed_external_helpers"]["files"]
        if relative != stage.custody.LEGACY_API_RELATIVE
    }
    successor_source_revision = stage._source_revision(stage.ROOT)
    builder = stage.file_binding(stage.ROOT / "scripts/neuralese_data/build_verified_teacher_trajectory_graphs.py")
    evidence = {}
    for name in ("focused", "full"):
        result = path.with_name(name + "-result.json")
        result.write_text(json.dumps({
            "schema": "natlang.teacher-helper-transition-check-result/2", "status": "passed",
            "exit_code": 0, "source_commit": "8c3358fac0355d462652f1b3352970ff2f5ca0f9",
            "successor_source_revision": successor_source_revision,
            "successor_implementation_files": successor_inventory,
            "unit": "review-fixture-" + name,
        }) + "\n")
        evidence["focused_evidence" if name == "focused" else "full_check_evidence"] = stage.file_binding(result)
    path.write_text(json.dumps({
        "schema": "natlang.teacher-helper-transition-review/2", "passed": True,
        "full_check": True, "training_admission": False, "exposure": 0,
        "source_commit": "8c3358fac0355d462652f1b3352970ff2f5ca0f9",
        "successor_helper_sha256": stage.API_SOURCE_BINDING["sha256"],
        "successor_source_revision": successor_source_revision,
        "successor_converter": builder,
        "successor_implementation_files": successor_inventory,
        **evidence,
    }) + "\n")


def test_exact_active_closures_stage_inactive_with_pinned_successor(tmp_path, monkeypatch):
    monkeypatch.setattr(stage, "DGXOPS", tmp_path)
    destination = tmp_path / "candidate-v1"
    review = tmp_path / "review.json"
    _review(review)
    result = stage.stage(destination, review)
    assert result["activation"] is False
    manifest = json.loads((destination / "operator-transition.json").read_bytes())
    assert manifest["status"] == "prepared-inactive"
    assert set(manifest["outputs"]) == {
        "selected-convert", "selected-validate", "flint-convert", "flint-validate",
        "ready202-convert", "ready202-validate", "drainer",
    }
    assert manifest["training_admission"] is False and manifest["exposure"] == 0
    verifier_ref = manifest["successor_verifier"]
    assert verifier_ref["path"] == str((destination / "selected/scripts/coreweave/teacher_helper_continuation.py").resolve())
    for key, spec in stage.INPUTS.items():
        old = stage.file_binding(spec["path"])
        assert old["sha256"] == spec["sha256"] and old["bytes"] == spec["bytes"]
    outputs = manifest["outputs"]
    for key in ("selected-convert", "selected-validate", "flint-convert", "flint-validate",
                "ready202-convert", "ready202-validate"):
        source = Path(outputs[key]["path"])
        _execute_staged_proof_calls(source, manifest)
        subprocess.run([sys.executable, "-m", "py_compile", str(source)], check=True)
        text = source.read_text()
        assert "HELPER_TRANSITION" in text and "verify_execution_helper(execution)" in text if "validate" in key else "HELPER_TRANSITION" in text
        assert "teacher-helper-continuation-transition.json" in text
        assert "SUCCESSOR_VERIFIER_BINDING" in text
        if key in {"flint-convert", "ready202-convert"}:
            assert "sha256_file_hex(resolver)==ADMISSION_API_SHA256" in text
    drainer = Path(outputs["drainer"]["path"]).read_text()
    subprocess.run([sys.executable, "-m", "py_compile", str(outputs["drainer"]["path"])], check=True)
    assert str(destination / "selected/scripts/coreweave") in drainer
    assert str(destination / "flint/convert-flint-both-v2-0aad.py") in drainer
    assert str(destination / "ready202/strict-current-teacher-both-v4-classified-siblings-0aad.py") in drainer
    for key in ("selected-convert", "selected-validate", "flint-convert", "flint-validate",
                "ready202-convert", "ready202-validate"):
        completed = subprocess.run([sys.executable, str(outputs[key]["path"]), "--help"], check=False,
                                   capture_output=True, text=True,
                                   env={"PATH": "/usr/bin:/bin", "PYTHONPATH": f"{stage.ROOT}:{stage.ROOT / 'training/neuralese'}"})
        assert completed.returncode == 0, completed.stderr[-2000:]

    # A hostile ambient module cannot replace the staged verifier: operators load
    # only the exact path+hash bound into their source and transition receipt.
    poison = tmp_path / "poison"
    poison.mkdir()
    (poison / "sitecustomize.py").write_text(
        "import sys, types\n"
        "m=types.ModuleType('teacher_helper_continuation')\n"
        "m.verify_successor_bundle=lambda **kw: (_ for _ in ()).throw(RuntimeError('poisoned verifier'))\n"
        "sys.modules['teacher_helper_continuation']=m\n")
    env = {"PATH": "/usr/bin:/bin", "PYTHONPATH": f"{poison}:{stage.ROOT}:{stage.ROOT / 'training/neuralese'}"}
    completed = subprocess.run([sys.executable, str(outputs["flint-convert"]["path"]), "--help"],
                               check=False, capture_output=True, text=True, env=env)
    assert completed.returncode == 0, completed.stderr[-2000:]


def test_operator_loader_rejects_changed_staged_verifier(tmp_path, monkeypatch):
    monkeypatch.setattr(stage, "DGXOPS", tmp_path)
    destination = tmp_path / "candidate-tamper"
    review = tmp_path / "review.json"
    _review(review)
    result = stage.stage(destination, review)
    helper = Path(result["successor_verifier"]["path"])
    helper.write_bytes(helper.read_bytes() + b"# tampered\n")
    completed = subprocess.run([sys.executable, str(result["outputs"]["flint-convert"]["path"]), "--help"],
                               check=False, capture_output=True, text=True,
                               env={"PATH": "/usr/bin:/bin", "PYTHONPATH": f"{stage.ROOT}:{stage.ROOT / 'training/neuralese'}"})
    assert completed.returncode != 0
    assert "pinned teacher helper continuation verifier changed" in completed.stderr


def test_stage_rejects_review_without_exact_current_inventory(tmp_path, monkeypatch):
    monkeypatch.setattr(stage, "DGXOPS", tmp_path)
    destination = tmp_path / "candidate-stale-review"
    review = tmp_path / "review.json"
    _review(review)
    report = json.loads(review.read_bytes())
    builder = "scripts/neuralese_data/build_verified_teacher_trajectory_graphs.py"
    report["successor_implementation_files"][builder]["sha256"] = "0" * 64
    review.write_text(json.dumps(report) + "\n")
    with pytest.raises(ValueError, match="exact active source revision and implementation inventory"):
        stage.stage(destination, review)
    assert not destination.exists()


def test_stage_refuses_incomplete_review_and_modified_source(tmp_path, monkeypatch):
    monkeypatch.setattr(stage, "DGXOPS", tmp_path)
    review = tmp_path / "review.json"
    _review(review)
    value = json.loads(review.read_bytes())
    value["full_check"] = False
    review.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="review report"):
        stage.stage(tmp_path / "bad-review", review)


def test_transformer_fails_closed_on_unrecognized_helper_control():
    with pytest.raises(ValueError, match="exactly one HELPER_OBSERVATION"):
        stage.transform_operator("def verify_new_helpers():\n    pass\n", kind="validator",
                                 transition_ref={"path": "x", "sha256": "a" * 64, "bytes": 1},
                                 api_ref={"path": "y", "sha256": "b" * 64, "bytes": 1},
                                 verifier_ref={"path": "z", "sha256": "c" * 64, "bytes": 1})
