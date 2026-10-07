import json
from pathlib import Path

import pytest

from scripts.prepare_training_exclusion_plan import prepare, rewrite_training_argv


ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = ROOT / "runs/training-source-exclusion-20261003/tatqa-and-bugabula-v1"


def test_rewrites_exact_inventory_bind_and_trainer_path():
    old = Path("/runs/old-run")
    new = Path("/runs/new-run")
    runtime = Path("/runs/runtime")
    old_mount = str(old / "frozen-inventory-policy/data_sources.json") + ":/home/werg/natlang/training/data_sources.json:ro"
    argv = ["docker", "run", "-v", old_mount, "-v", f"{old}:/workspace:rw",
            f"{old}/code-snapshot/scripts/train_lora.py", f"{old}/trainer-output"]
    rewritten = rewrite_training_argv(argv, old_run=old, target_run=new, runtime_root=runtime)
    assert rewritten[3] == str(runtime / "inventory-policy/data_sources.json") + ":/home/werg/natlang/training/data_sources.json:ro"
    assert rewritten[-2] == str(runtime / "code-snapshot/scripts/train_lora.py")
    assert rewritten[-1] == f"{new}/trainer-output"
    assert rewritten[5] == f"{new}:/workspace:rw"


@pytest.mark.skipif(not (CASE_DIR / "future-runtime-candidate-v8.manifest.json").exists(),
                    reason="needs the local runs/training-source-exclusion-20261003 artifacts")
def test_unapproved_transition_is_rejected_without_writing_plan(tmp_path):
    base_plan = ROOT / "runs/lfm25-350m-broad-20261002/full-v13-muon-epoch1/training-plan-v3.json"
    manifest = CASE_DIR / "transition-candidate-v4.json"
    runtime_manifest = CASE_DIR / "future-runtime-candidate-v8.manifest.json"
    runtime_doc = json.loads(runtime_manifest.read_text())
    supervisor = Path(runtime_doc["candidate_path"]) / runtime_doc["supervisor_relative_path"]
    application = tmp_path / "application.json"
    quiescence = tmp_path / "quiescence.json"
    application.write_text("{}\n")
    quiescence.write_text("{}\n")
    output = tmp_path / "must-not-exist/plan.json"
    service = tmp_path / "must-not-exist/training.service"

    with pytest.raises(ValueError, match="does not have root approval"):
        prepare(base_plan_path=base_plan.resolve(), manifest_path=manifest.resolve(),
                manifest_sha256=__import__("hashlib").sha256(manifest.read_bytes()).hexdigest(),
                application_path=application.resolve(), quiescence_path=quiescence.resolve(),
                runtime_manifest_path=runtime_manifest.resolve(), supervisor_path=supervisor.resolve(),
                supervisor_sha256=runtime_doc["supervisor_sha256"],
                output_plan=output, service_draft=service)

    assert not output.exists()
    assert not service.exists()
