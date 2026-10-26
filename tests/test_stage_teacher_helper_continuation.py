import ast
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from scripts.coreweave import stage_teacher_helper_continuation as stage


def _loaded_module(name, path, monkeypatch):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def _execute_staged_proof_calls(path, manifest, monkeypatch):
    source = Path(path).read_text()
    tree = ast.parse(source)
    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    api_path = Path(manifest["successor_helper"]["path"])
    verifier_path = Path(manifest["successor_verifier"]["path"])
    api = _loaded_module(
        "scripts.coreweave.teacher_tranche_admission", api_path, monkeypatch
    )
    from scripts import coreweave

    monkeypatch.setattr(coreweave, "teacher_tranche_admission", api, raising=False)
    verifier = _loaded_module("teacher_helper_continuation", verifier_path, monkeypatch)
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
        exec(  # noqa: S102 - execute the generated operator function under its bound test environment
            compile(
                textwrap.dedent(ast.get_source_segment(source, node)), str(path), "exec"
            ),
            env,
        )
    env["verify_new_helpers"]()
    builder = (
        stage.ROOT
        / "scripts/neuralese_data/build_verified_teacher_trajectory_graphs.py"
    )
    builder_ref = stage.file_binding(builder)
    execution = {
        "helper_continuation": manifest["successor_transition"],
        "converter": {"path": builder_ref["path"], "sha256": builder_ref["sha256"]},
    }
    assert env["verify_execution_helper"](execution) == "successor"
    api_digest_assertions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assert)
        and any(
            isinstance(sub, ast.Name) and sub.id == "ADMISSION_API_SHA256"
            for sub in ast.walk(node.test)
        )
        and any(
            isinstance(sub, ast.Name) and sub.id == "resolver"
            for sub in ast.walk(node.test)
        )
    ]
    if Path(path).name in {
        "convert-flint-both-v2-0aad.py",
        "convert-current-teacher-both-v3-0aad.py",
    }:
        assert api_digest_assertions
        assertion_source = textwrap.dedent(
            ast.get_source_segment(source, api_digest_assertions[0])
        )
        from natlang_neuralese.common.hashing import sha256_file_hex

        resolver_env = {
            "sha256_file_hex": sha256_file_hex,
            "resolver": api_path,
            "ADMISSION_API_SHA256": manifest["successor_helper"]["sha256"],
        }
        exec(compile(assertion_source, str(path), "exec"), resolver_env)  # noqa: S102 - evaluate the generated digest assertion


def _canonical_sha256(value):
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def _test_verifier_source(source, bindings):
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    targets = {
        node.targets[0].id: node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in bindings
    }
    assert set(targets) == set(bindings)
    for name, node in sorted(
        targets.items(), key=lambda item: item[1].lineno, reverse=True
    ):
        lines[node.lineno - 1 : node.end_lineno] = [f"{name} = {bindings[name]!r}\n"]
    result = "".join(lines)
    patched = ast.parse(result)
    original_functions = {
        node.name: ast.get_source_segment(source, node)
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    patched_functions = {
        node.name: ast.get_source_segment(result, node)
        for node in ast.walk(patched)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert patched_functions == original_functions
    return result


def _operator_fixture(path: Path, *, selected=False, api_assert=False):
    lines = ["import argparse", "import hashlib", "import pathlib", "import sys"]
    if selected:
        lines += [
            "import scripts.coreweave as _coreweave_package",
            "import scripts.coreweave.teacher_tranche_admission as _api_module",
            "_coreweave_package.teacher_tranche_admission = _api_module",
            "ADMISSION_API_PATH = pathlib.Path(__file__).with_name('teacher_tranche_admission.py')",
        ]
    else:
        lines += [
            "from scripts.coreweave.teacher_tranche_admission import file_binding"
        ]
    if api_assert:
        lines += ["from natlang_neuralese.common.hashing import sha256_file_hex"]
    lines += [
        "HELPER_OBSERVATION = None",
        "",
        "def verify_new_helpers():",
        "    return None",
    ]
    if api_assert:
        lines += [
            "",
            "def verify_admission_api(resolver):",
            "    assert sha256_file_hex(resolver)=='e62b8d9ccaaf397d067706f15cc00417d08b4591aaa04542f273675dfe756054'",
        ]
    lines += [
        "",
        "def current_execution():",
        "    execution = {'helper_continuation': HELPER_OBSERVATION}",
        "    return execution.get('helper_continuation') == HELPER_OBSERVATION",
        "",
        "if __name__ == '__main__':",
        "    argparse.ArgumentParser().parse_args()",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def _stage_portable_inputs(path: Path, monkeypatch):
    """Use real temporary files and repository source, with explicitly unit-only provenance."""
    monkeypatch.setattr(stage, "ROOT", Path(__file__).resolve().parents[1])
    fixture_root = path.parent / "stage-unit-test-only"
    fixture_root.mkdir()
    source_root = fixture_root / "source-closure"
    source_root.mkdir()
    api_source = stage.ROOT / "scripts/coreweave/teacher_tranche_admission.py"
    builder = (
        stage.ROOT
        / "scripts/neuralese_data/build_verified_teacher_trajectory_graphs.py"
    )
    old_api = fixture_root / "historical/scripts/coreweave/teacher_tranche_admission.py"
    old_builder = (
        fixture_root
        / "historical/scripts/neuralese_data/build_verified_teacher_trajectory_graphs.py"
    )
    old_api.parent.mkdir(parents=True)
    old_builder.parent.mkdir(parents=True)
    # These are deliberately tiny, test-only historical-source stand-ins. The
    # portable case exercises binding, staging, and tamper detection against
    # real bytes without requiring the historical 0aad212b5 Git object to be
    # present in this checkout. Strict historical coverage still uses the
    # independently pinned assets.
    original_check_output = subprocess.check_output

    def reject_historical_git_show(args, *call_args, **call_kwargs):
        if isinstance(args, (list, tuple)) and list(args[:2]) == ["git", "show"]:
            raise AssertionError(
                "portable staging must not retrieve historical Git objects"
            )
        return original_check_output(args, *call_args, **call_kwargs)

    monkeypatch.setattr(subprocess, "check_output", reject_historical_git_show)
    old_api.write_text(
        "# unit-test-only legacy API fixture; not an admission implementation\n"
        "def file_binding(path):\n"
        "    raise RuntimeError('test fixture is not executable authority')\n"
    )
    old_builder.write_text(
        "# unit-test-only legacy builder fixture; not a production converter\n"
        "def build(*args, **kwargs):\n"
        "    raise RuntimeError('test fixture is not executable authority')\n"
    )
    files = {
        stage.custody.LEGACY_API_RELATIVE: stage.file_binding(old_api),
        "scripts/neuralese_data/build_verified_teacher_trajectory_graphs.py": stage.file_binding(
            old_builder
        ),
    }
    authority_path = fixture_root / "immutable-authority.unit-test-only.json"
    authority_path.write_text(
        json.dumps(
            {
                "schema": "natlang.immutable-source-helper-authority/1",
                # The verifier requires this historical identity even in the
                # portable unit case. The adjacent marker and synthetic files
                # make clear that no production authority is being asserted.
                "commit": "0aad212b5",
                "training_admission": False,
                "unit_test_only": True,
                "files": files,
            },
            sort_keys=True,
        )
        + "\n"
    )
    authority_ref = stage.file_binding(authority_path)

    accounting = {"unit_test_only": True, "training_admission": False}
    input_artifacts = {"unit_test_only": "stage-helper-fixture"}
    transition_path = source_root / "transition.unit-test-only.json"
    transition = {
        "schema": "natlang.acquired-source-closure-code-transition/6",
        "training_admission": False,
        "unit_test_only": True,
        "closure_verifier_sha256": "unit-test-only-verifier",
        "protection_sha256": "unit-test-only-protection",
        "accounting_sha256": _canonical_sha256(accounting),
        "input_artifacts_sha256": _canonical_sha256(input_artifacts),
    }
    transition_path.write_text(json.dumps(transition, sort_keys=True) + "\n")
    transition_ref = stage.file_binding(transition_path)
    receipt_path = source_root / "receipt.unit-test-only.json"
    receipt = {
        "schema": "natlang.acquired-source-closure-receipt/1",
        "code_transition": {key: transition_ref[key] for key in ("path", "sha256")},
        "training_admission": False,
        "unit_test_only": True,
        "verifier_sha256": transition["closure_verifier_sha256"],
        "protection_sha256": transition["protection_sha256"],
        "accounting": accounting,
        "input_artifacts": input_artifacts,
    }
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n")
    receipt_ref = stage.file_binding(receipt_path)
    observation_path = fixture_root / "legacy-observation.unit-test-only.json"
    observation_path.write_text(
        json.dumps(
            {
                "passed": True,
                "source_closure_implementation_unchanged": True,
                "normal_full_artifact_verifier_passed": True,
                "source_rows_parsed": 0,
                "training_admission": False,
                "unit_test_only": True,
                "original_receipt_sha256": receipt_ref["sha256"],
                "reviewed_external_helpers": {"files": files},
            },
            sort_keys=True,
        )
        + "\n"
    )
    observation_ref = stage.file_binding(observation_path)

    bindings = {
        "LEGACY_OBSERVATION": observation_ref,
        "LEGACY_AUTHORITY": authority_ref,
        "SOURCE_CLOSURE_RECEIPT": receipt_ref,
        "SOURCE_CLOSURE_TRANSITION": transition_ref,
    }
    verifier_path = fixture_root / "teacher_helper_continuation.unit-test-only.py"
    verifier_path.write_text(
        _test_verifier_source(Path(stage.custody.__file__).read_text(), bindings)
    )
    for name, value in bindings.items():
        monkeypatch.setattr(stage.custody, name, value)
    monkeypatch.setattr(stage.custody, "__file__", str(verifier_path))
    monkeypatch.setattr(stage, "API_SOURCE", api_source)
    monkeypatch.setattr(stage, "API_SOURCE_BINDING", stage.file_binding(api_source))
    source_dgxops = fixture_root / "source-operational-data"
    monkeypatch.setattr(stage, "SOURCE_DGXOPS", source_dgxops)

    inputs = {}
    for key, spec in stage.INPUTS.items():
        target = fixture_root / "operator-inputs" / key / Path(spec["path"]).name
        if spec["kind"] == "drainer":
            old_selected = (
                source_dgxops
                / "teacher-zero-guided-control-closure-8c3358fa-20261022/scripts/coreweave"
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                "\n".join(
                    (
                        "from pathlib import Path",
                        "def choose(**kwargs): return kwargs",
                        f"def selected(): return choose(selected_operators=Path({str(old_selected)!r}))",
                        "def flint(): return choose(convert='convert-flint-both-v2-0aad.py',strict='strict-flint-both-v3-classified-siblings-0aad.py')",
                        "def ready202(): return choose(convert='convert-current-teacher-both-v3-0aad.py',strict='strict-current-teacher-both-v4-classified-siblings-0aad.py')",
                        "",
                    )
                )
            )
        else:
            _operator_fixture(
                target,
                selected=spec["tree"] == "selected/scripts/coreweave",
                api_assert=key in {"flint-convert", "ready202-convert"},
            )
        inputs[key] = {**spec, **stage.file_binding(target), "path": target}
    monkeypatch.setattr(stage, "INPUTS", inputs)
    successor_inventory = {
        relative: stage.file_binding(stage.ROOT / relative)
        for relative in files
        if relative != stage.custody.LEGACY_API_RELATIVE
    }
    successor_source_revision = stage._source_revision(stage.ROOT)
    builder = stage.file_binding(
        stage.ROOT
        / "scripts/neuralese_data/build_verified_teacher_trajectory_graphs.py"
    )
    return _write_review(
        path,
        successor_inventory,
        successor_source_revision,
        builder,
        stage.API_SOURCE_BINDING["sha256"],
    )


def _write_review(
    path, successor_inventory, successor_source_revision, builder, api_sha256
):
    evidence = {}
    for name in ("focused", "full"):
        result = path.with_name(name + "-result.json")
        result.write_text(
            json.dumps(
                {
                    "schema": "natlang.teacher-helper-transition-check-result/2",
                    "status": "passed",
                    "exit_code": 0,
                    "source_commit": "8c3358fac0355d462652f1b3352970ff2f5ca0f9",
                    "successor_source_revision": successor_source_revision,
                    "successor_implementation_files": successor_inventory,
                    "unit": "unit-test-only-review-fixture-" + name,
                }
            )
            + "\n"
        )
        evidence["focused_evidence" if name == "focused" else "full_check_evidence"] = (
            stage.file_binding(result)
        )
    path.write_text(
        json.dumps(
            {
                "schema": "natlang.teacher-helper-transition-review/2",
                "passed": True,
                "full_check": True,
                "training_admission": False,
                "exposure": 0,
                "source_commit": "8c3358fac0355d462652f1b3352970ff2f5ca0f9",
                "successor_helper_sha256": api_sha256,
                "successor_source_revision": successor_source_revision,
                "successor_converter": builder,
                "successor_implementation_files": successor_inventory,
                **evidence,
            }
        )
        + "\n"
    )
    return path


def _stage_historical_inputs(path, monkeypatch):
    monkeypatch.setattr(stage, "ROOT", Path(__file__).resolve().parents[1])
    old = json.loads(Path(stage.custody.LEGACY_OBSERVATION["path"]).read_bytes())
    successor_inventory = {
        relative: stage.file_binding(stage.ROOT / relative)
        for relative in old["reviewed_external_helpers"]["files"]
        if relative != stage.custody.LEGACY_API_RELATIVE
    }
    return _write_review(
        path,
        successor_inventory,
        stage._source_revision(stage.ROOT),
        stage.file_binding(
            stage.ROOT
            / "scripts/neuralese_data/build_verified_teacher_trajectory_graphs.py"
        ),
        stage.API_SOURCE_BINDING["sha256"],
    )


_HISTORICAL_INPUTS = [
    stage.custody.LEGACY_OBSERVATION,
    stage.custody.LEGACY_AUTHORITY,
    stage.custody.SOURCE_CLOSURE_RECEIPT,
    stage.custody.SOURCE_CLOSURE_TRANSITION,
]
_HISTORICAL_INPUTS += [{"path": str(stage.API_SOURCE)}]
_HISTORICAL_INPUTS += [{"path": str(spec["path"])} for spec in stage.INPUTS.values()]
_MISSING_HISTORICAL_INPUTS = [
    ref["path"] for ref in _HISTORICAL_INPUTS if not Path(ref["path"]).is_file()
]


@pytest.fixture(params=["portable", "historical"])
def stage_mode(request):
    if request.param == "historical" and _MISSING_HISTORICAL_INPUTS:
        message = "historical stage assets are unavailable: " + ", ".join(
            _MISSING_HISTORICAL_INPUTS
        )
        if os.environ.get("NATLANG_REQUIRE_TEACHER_HELPER_HISTORY") == "1":
            pytest.fail(message)
        pytest.skip(message)
    return request.param


def _review(path, mode, monkeypatch):
    if mode == "portable":
        return _stage_portable_inputs(path, monkeypatch)
    return _stage_historical_inputs(path, monkeypatch)


def test_exact_active_closures_stage_inactive_with_pinned_successor(
    tmp_path, monkeypatch, stage_mode
):
    monkeypatch.setattr(stage, "DGXOPS", tmp_path)
    destination = tmp_path / "candidate-v1"
    review = tmp_path / "review.json"
    _review(review, stage_mode, monkeypatch)
    result = stage.stage(destination, review)
    assert result["activation"] is False
    manifest = json.loads((destination / "operator-transition.json").read_bytes())
    assert manifest["status"] == "prepared-inactive"
    assert set(manifest["outputs"]) == {
        "selected-convert",
        "selected-validate",
        "flint-convert",
        "flint-validate",
        "ready202-convert",
        "ready202-validate",
        "drainer",
    }
    assert manifest["training_admission"] is False and manifest["exposure"] == 0
    verifier_ref = manifest["successor_verifier"]
    assert verifier_ref["path"] == str(
        (
            destination / "selected/scripts/coreweave/teacher_helper_continuation.py"
        ).resolve()
    )
    for key, spec in stage.INPUTS.items():
        old = stage.file_binding(spec["path"])
        assert old["sha256"] == spec["sha256"] and old["bytes"] == spec["bytes"]
    outputs = manifest["outputs"]
    for key in (
        "selected-convert",
        "selected-validate",
        "flint-convert",
        "flint-validate",
        "ready202-convert",
        "ready202-validate",
    ):
        source = Path(outputs[key]["path"])
        _execute_staged_proof_calls(source, manifest, monkeypatch)
        subprocess.run([sys.executable, "-m", "py_compile", str(source)], check=True)
        text = source.read_text()
        assert (
            "HELPER_TRANSITION" in text and "verify_execution_helper(execution)" in text
            if "validate" in key
            else "HELPER_TRANSITION" in text
        )
        assert "teacher-helper-continuation-transition.json" in text
        assert "SUCCESSOR_VERIFIER_BINDING" in text
        if key in {"flint-convert", "ready202-convert"}:
            assert "sha256_file_hex(resolver)==ADMISSION_API_SHA256" in text
    drainer = Path(outputs["drainer"]["path"]).read_text()
    subprocess.run(
        [sys.executable, "-m", "py_compile", str(outputs["drainer"]["path"])],
        check=True,
    )
    assert str(destination / "selected/scripts/coreweave") in drainer
    assert str(destination / "flint/convert-flint-both-v2-0aad.py") in drainer
    assert (
        str(
            destination
            / "ready202/strict-current-teacher-both-v4-classified-siblings-0aad.py"
        )
        in drainer
    )
    for key in (
        "selected-convert",
        "selected-validate",
        "flint-convert",
        "flint-validate",
        "ready202-convert",
        "ready202-validate",
    ):
        completed = subprocess.run(
            [sys.executable, str(outputs[key]["path"]), "--help"],
            check=False,
            capture_output=True,
            text=True,
            env={
                "PATH": "/usr/bin:/bin",
                "PYTHONPATH": f"{stage.ROOT}:{stage.ROOT / 'training/neuralese'}",
            },
        )
        assert completed.returncode == 0, completed.stderr[-2000:]

    # A hostile ambient module cannot replace the staged verifier: operators load
    # only the exact path+hash bound into their source and transition receipt.
    poison = tmp_path / "poison"
    poison.mkdir()
    (poison / "sitecustomize.py").write_text(
        "import sys, types\n"
        "m=types.ModuleType('teacher_helper_continuation')\n"
        "m.verify_successor_bundle=lambda **kw: (_ for _ in ()).throw(RuntimeError('poisoned verifier'))\n"
        "sys.modules['teacher_helper_continuation']=m\n"
    )
    env = {
        "PATH": "/usr/bin:/bin",
        "PYTHONPATH": f"{poison}:{stage.ROOT}:{stage.ROOT / 'training/neuralese'}",
    }
    completed = subprocess.run(
        [sys.executable, str(outputs["flint-convert"]["path"]), "--help"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]


def test_operator_loader_rejects_changed_staged_verifier(
    tmp_path, monkeypatch, stage_mode
):
    monkeypatch.setattr(stage, "DGXOPS", tmp_path)
    destination = tmp_path / "candidate-tamper"
    review = tmp_path / "review.json"
    _review(review, stage_mode, monkeypatch)
    result = stage.stage(destination, review)
    helper = Path(result["successor_verifier"]["path"])
    helper.write_bytes(helper.read_bytes() + b"# tampered\n")
    completed = subprocess.run(
        [sys.executable, str(result["outputs"]["flint-convert"]["path"]), "--help"],
        check=False,
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin",
            "PYTHONPATH": f"{stage.ROOT}:{stage.ROOT / 'training/neuralese'}",
        },
    )
    assert completed.returncode != 0
    assert "pinned teacher helper continuation verifier changed" in completed.stderr


def test_stage_rejects_review_without_exact_current_inventory(
    tmp_path, monkeypatch, stage_mode
):
    monkeypatch.setattr(stage, "DGXOPS", tmp_path)
    destination = tmp_path / "candidate-stale-review"
    review = tmp_path / "review.json"
    _review(review, stage_mode, monkeypatch)
    report = json.loads(review.read_bytes())
    builder = "scripts/neuralese_data/build_verified_teacher_trajectory_graphs.py"
    report["successor_implementation_files"][builder]["sha256"] = "0" * 64
    review.write_text(json.dumps(report) + "\n")
    with pytest.raises(
        ValueError, match="exact active source revision and implementation inventory"
    ):
        stage.stage(destination, review)
    assert not destination.exists()


def test_stage_refuses_incomplete_review_and_modified_source(
    tmp_path, monkeypatch, stage_mode
):
    monkeypatch.setattr(stage, "DGXOPS", tmp_path)
    review = tmp_path / "review.json"
    _review(review, stage_mode, monkeypatch)
    value = json.loads(review.read_bytes())
    value["full_check"] = False
    review.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="review report"):
        stage.stage(tmp_path / "bad-review", review)


def test_transformer_fails_closed_on_unrecognized_helper_control():
    with pytest.raises(ValueError, match="exactly one HELPER_OBSERVATION"):
        stage.transform_operator(
            "def verify_new_helpers():\n    pass\n",
            kind="validator",
            transition_ref={"path": "x", "sha256": "a" * 64, "bytes": 1},
            api_ref={"path": "y", "sha256": "b" * 64, "bytes": 1},
            verifier_ref={"path": "z", "sha256": "c" * 64, "bytes": 1},
        )
