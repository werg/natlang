import hashlib
import importlib.util
import json
import shutil
from pathlib import Path
from natlang_neuralese.common import paths as machine_paths

import pytest

from scripts.coreweave import teacher_helper_continuation as custody


STAGED_API = Path(str(machine_paths.resolve('data_nvme', 'dgxops/teacher-zero-guided-control-closure-8c3358fa-20261022/scripts/coreweave/teacher_tranche_admission.py')))
AUTHORITY_PATH = Path(custody.LEGACY_AUTHORITY["path"])
OLD = json.loads(Path(custody.LEGACY_OBSERVATION["path"]).read_bytes())


def _ref(path: Path):
    raw = path.read_bytes()
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def _builder(root: Path):
    relative = "scripts/neuralese_data/build_verified_teacher_trajectory_graphs.py"
    ref = _ref(root / relative)
    return {"path": ref["path"], "sha256": ref["sha256"]}


def _successor_fixture(tmp_path, *, bad_accounting=False):
    assert STAGED_API.exists()
    repo = tmp_path / "repo"
    relroot = repo / "scripts/coreweave"
    relroot.mkdir(parents=True)
    staged_api = tmp_path / "operators/teacher_tranche_admission.py"
    staged_api.parent.mkdir(parents=True)
    shutil.copyfile(STAGED_API, staged_api)
    old_api = Path(json.loads(AUTHORITY_PATH.read_bytes())["files"][custody.LEGACY_API_RELATIVE]["path"])
    shutil.copyfile(old_api, relroot / "teacher_tranche_admission.py")
    authority = json.loads(AUTHORITY_PATH.read_bytes())
    inventory = {relative: _ref(Path(ref["path"])) for relative, ref in authority["files"].items()}
    for relative, old_ref in OLD["reviewed_external_helpers"]["files"].items():
        if relative == custody.LEGACY_API_RELATIVE:
            continue
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(authority["files"][relative]["path"]), target)
    api_ref = _ref(staged_api)
    successor_inventory = {relative: _ref(repo / relative)
        for relative in OLD["reviewed_external_helpers"]["files"]
        if relative != custody.LEGACY_API_RELATIVE}
    successor_converter = successor_inventory["scripts/neuralese_data/build_verified_teacher_trajectory_graphs.py"]
    successor_source_revision = "a" * 40
    verifier_path = tmp_path / "operators/teacher_helper_continuation.py"
    shutil.copyfile(Path(custody.__file__), verifier_path)
    verifier_ref = _ref(verifier_path)
    closure = {
        "receipt": _ref(Path(custody.SOURCE_CLOSURE_RECEIPT["path"])),
        "transition": _ref(Path(custody.SOURCE_CLOSURE_TRANSITION["path"])),
        "accounting": json.loads(Path(custody.SOURCE_CLOSURE_RECEIPT["path"]).read_bytes())["accounting"],
    }
    review_path = tmp_path / "review.json"
    review_evidence = {}
    for name in ("focused", "full"):
        report_path = tmp_path / f"{name}-check.json"
        report_path.write_text(json.dumps({
            "schema": "natlang.teacher-helper-transition-check-result/2", "status": "passed",
            "exit_code": 0, "source_commit": custody.SUCCESSOR_COMMIT, "unit": "unit-" + name,
            "successor_source_revision": successor_source_revision,
            "successor_implementation_files": successor_inventory,
        }) + "\n")
        review_evidence["focused_evidence" if name == "focused" else "full_check_evidence"] = _ref(report_path)
    review_path.write_text(json.dumps({
        "schema": "natlang.teacher-helper-transition-review/2", "passed": True,
        "full_check": True, "training_admission": False, "exposure": 0,
        "successor_helper_sha256": api_ref["sha256"],
        "source_commit": custody.SUCCESSOR_COMMIT,
        "successor_source_revision": successor_source_revision,
        "successor_converter": successor_converter,
        "successor_implementation_files": successor_inventory,
        **review_evidence,
    }, sort_keys=True) + "\n")
    proof = {
        "schema": "natlang.teacher-helper-continuation-transition/2",
        "legacy_observation": custody.LEGACY_OBSERVATION,
        "legacy_authority": custody.LEGACY_AUTHORITY,
        "successor_helper": api_ref,
        "successor_verifier": verifier_ref,
        "successor_converter": successor_converter,
        "implementation_files": inventory,
        "successor_implementation_files": successor_inventory,
        "successor_source_revision": successor_source_revision,
        "source_closure": closure,
        "review": _ref(review_path),
        "training_admission": False,
        "exposure": 0,
    }
    if bad_accounting:
        proof["source_closure"]["accounting"] = {"normalized_sessions": 999}
    proof_path = tmp_path / "transition.json"
    proof_path.write_text(json.dumps(proof, sort_keys=True) + "\n")
    return repo, staged_api, verifier_path, proof_path, proof, api_ref, verifier_ref


def test_exact_historical_helper_receipt_remains_verifiable():
    root = Path(__file__).resolve().parents[1]
    authority = json.loads(AUTHORITY_PATH.read_bytes())
    builder_relative = "scripts/neuralese_data/build_verified_teacher_trajectory_graphs.py"
    historical_builder = authority["files"][builder_relative]
    assert _builder(root)["sha256"] != historical_builder["sha256"]
    execution = {"helper_continuation": custody.LEGACY_OBSERVATION,
                 "converter": {"path": str((root / builder_relative).resolve()),
                               "sha256": historical_builder["sha256"]}}
    assert custody.verify_helper_continuation(
        execution, transition_binding={"path": "/not-read", "sha256": "b" * 64, "bytes": 1},
        successor_helper_binding={"path": "/not-read", "sha256": "c" * 64, "bytes": 1},
        successor_verifier_binding={"path": "/not-read", "sha256": "d" * 64, "bytes": 1},
        loaded_helper_path=Path(custody.LEGACY_AUTHORITY["path"]), repository_root=root,
        loaded_verifier_path=Path(custody.LEGACY_AUTHORITY["path"]),
        expected_executor=execution["converter"]) == "historical"


def test_successor_receipt_requires_exact_assembled_helper_and_accounting(tmp_path):
    repo, helper_path, verifier_path, proof_path, proof, api_ref, verifier_ref = _successor_fixture(tmp_path)
    transition = _ref(proof_path)
    execution = {"helper_continuation": transition, "converter": _builder(repo)}
    assert custody.verify_helper_continuation(
        execution, transition_binding=transition, successor_helper_binding=api_ref,
        successor_verifier_binding=verifier_ref, loaded_helper_path=helper_path,
        loaded_verifier_path=verifier_path, repository_root=repo,
        expected_executor=execution["converter"]) == "successor"
    assert custody.verify_successor_bundle(
        transition_binding=transition, successor_helper_binding=api_ref,
        successor_verifier_binding=verifier_ref, loaded_helper_path=helper_path,
        loaded_verifier_path=verifier_path, repository_root=repo) is None

    proof["successor_helper"]["sha256"] = "0" * 64
    proof_path.write_text(json.dumps(proof, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="changed after it was bound"):
        custody.verify_helper_continuation(
            execution, transition_binding=transition, successor_helper_binding=api_ref,
            successor_verifier_binding=verifier_ref, loaded_helper_path=helper_path,
            loaded_verifier_path=verifier_path, repository_root=repo,
            expected_executor=execution["converter"])


def test_successor_proof_cannot_claim_different_source_closure_accounting(tmp_path):
    repo, helper_path, verifier_path, proof_path, _, api_ref, verifier_ref = _successor_fixture(tmp_path, bad_accounting=True)
    transition = _ref(proof_path)
    execution = {"helper_continuation": transition, "converter": _builder(repo)}
    with pytest.raises(ValueError, match="source-closure accounting"):
        custody.verify_helper_continuation(
            execution, transition_binding=transition, successor_helper_binding=api_ref,
            successor_verifier_binding=verifier_ref, loaded_helper_path=helper_path,
            loaded_verifier_path=verifier_path, repository_root=repo,
            expected_executor=execution["converter"])


@pytest.mark.parametrize("tamper", ["helper", "implementation", "builder", "review_inventory", "receipt", "executor", "transition"])
def test_successor_refuses_tampered_or_unrelated_provenance(tmp_path, tamper):
    repo, helper_path, verifier_path, proof_path, proof, api_ref, verifier_ref = _successor_fixture(tmp_path)
    transition = _ref(proof_path)
    execution = {"helper_continuation": transition, "converter": _builder(repo)}
    if tamper == "helper":
        helper_path.write_bytes(helper_path.read_bytes() + b"# changed\n")
    elif tamper == "implementation":
        target = repo / "training/neuralese/natlang_neuralese/train/text_supervision.py"
        target.write_bytes(target.read_bytes() + b"# changed\n")
    elif tamper == "builder":
        target = repo / "scripts/neuralese_data/build_verified_teacher_trajectory_graphs.py"
        target.write_bytes(target.read_bytes() + b"# changed\n")
    elif tamper == "review_inventory":
        review_path = Path(proof["review"]["path"])
        review = json.loads(review_path.read_bytes())
        review["successor_implementation_files"]["scripts/neuralese_data/build_verified_teacher_trajectory_graphs.py"]["sha256"] = "0" * 64
        review_path.write_text(json.dumps(review, sort_keys=True) + "\n")
        proof["review"] = _ref(review_path)
        proof_path.write_text(json.dumps(proof, sort_keys=True) + "\n")
    elif tamper == "receipt":
        proof["legacy_observation"] = {**custody.LEGACY_OBSERVATION, "sha256": "0" * 64}
        proof_path.write_text(json.dumps(proof, sort_keys=True) + "\n")
        transition = _ref(proof_path)
        execution["helper_continuation"] = transition
    elif tamper == "executor":
        execution["converter"] = {"path": "/operators/unrelated.py", "sha256": "e" * 64}
    else:
        execution["helper_continuation"] = {**transition, "sha256": "f" * 64}
    with pytest.raises(ValueError):
        custody.verify_helper_continuation(
            execution, transition_binding=transition, successor_helper_binding=api_ref,
            successor_verifier_binding=verifier_ref, loaded_helper_path=helper_path,
            loaded_verifier_path=verifier_path, repository_root=repo,
            expected_executor=_builder(repo))


def test_sibling_loaded_verifier_uses_same_assembled_successor_authority(tmp_path):
    repo, helper_path, sibling, proof_path, _, api_ref, verifier_ref = _successor_fixture(tmp_path)
    spec = importlib.util.spec_from_file_location("sealed_teacher_helper_continuation", sibling)
    assert spec and spec.loader
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    transition = _ref(proof_path)
    executor = _builder(repo)
    receipt = {"helper_continuation": transition, "converter": executor}
    assert loaded.verify_helper_continuation(
        receipt, transition_binding=transition, successor_helper_binding=api_ref,
        successor_verifier_binding=verifier_ref, loaded_helper_path=helper_path,
        loaded_verifier_path=sibling, repository_root=repo, expected_executor=executor) == "successor"
