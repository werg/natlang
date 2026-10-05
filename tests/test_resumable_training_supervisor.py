import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.resumable_training_supervisor import is_complete


SUPERVISOR = Path(__file__).resolve().parents[1] / "scripts" / "resumable_training_supervisor.py"


def test_status_reads_live_checkpoint_without_mutating_lifecycle_status(tmp_path, capsys):
    from scripts.resumable_training_supervisor import control
    child = tmp_path / "unused.py"
    child.write_text("raise SystemExit(0)")
    path, run = _write_plan(tmp_path, child, {"step": 10})
    status_path = run / "supervisor-status.json"
    status_path.write_text(json.dumps({"state": "running", "checkpoint_step": None,
        "plan_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "pid": 99999999,
        "pid_start_time": "stale"}))
    original = status_path.read_bytes()
    ck = run / "checkpoint"
    ck.mkdir()
    state = {"step": 8, "trained_examples": 64, "corpus": {"fixture": "fake-training-v1"}}
    (ck / "state.json").write_text(json.dumps(state))
    for name in ("optimizer.pt", "scheduler.pt", "rng.pt"):
        (ck / name).write_text("toy")
    (ck / "weights").mkdir()
    (ck / "weights" / "model.bin").write_text("toy")
    assert control(path, "status") == 0
    observed = json.loads(capsys.readouterr().out)
    assert observed["checkpoint_step"] == 8
    assert observed["trained_examples"] == 64
    assert observed["supervisor_alive"] is False
    assert observed["checkpoint_complete"] is False
    assert status_path.read_bytes() == original
    state["corpus"]["fixture"] = "wrong"
    (ck / "state.json").write_text(json.dumps(state))
    with __import__("pytest").raises(ValueError, match="identity differs"):
        control(path, "status")


def _write_plan(tmp_path, child, completion, *, pins=None):
    run = tmp_path / "run"
    run.mkdir()
    artifacts = pins or {str(child): hashlib.sha256(child.read_bytes()).hexdigest()}
    plan = {
        "schema": "natlang-resumable-training/1",
        "run_dir": str(run),
        "checkpoint_state": str(run / "checkpoint" / "state.json"),
        "status_file": str(run / "supervisor-status.json"),
        "lock_file": str(run / "supervisor.lock"),
        "operator_stop_file": str(run / "operator-stop.json"),
        "argv": [sys.executable, str(child)],
        "cwd": str(tmp_path),
        "env": {"CHECKPOINT_STATE": str(run / "checkpoint" / "state.json"),
                "ATTEMPTS": str(run / "attempts")},
        "artifact_sha256": artifacts,
        "checkpoint_identity": {"fixture": "fake-training-v1"},
        "completion": completion,
        "retry": {"max_attempts": 3, "initial_delay_seconds": 1, "max_delay_seconds": 1},
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    return path, run


def test_handoff_uses_durable_complete_state_and_verified_checkpoint(tmp_path):
    import pytest
    from scripts.resumable_training_supervisor import completed_checkpoint_receipt, load_plan
    child = tmp_path / "unused.py"
    child.write_text("raise SystemExit(0)")
    path, run = _write_plan(tmp_path, child, {"step": 10, "final_evaluation_required": True})
    plan, sha = load_plan(path)
    status = run / "supervisor-status.json"
    status.write_text(json.dumps({"state": "complete", "plan_sha256": sha}))
    ck = run / "checkpoint"
    ck.mkdir()
    for name in ("optimizer.pt", "scheduler.pt", "rng.pt"):
        (ck / name).write_text("toy")
    (ck / "weights").mkdir()
    (ck / "weights/model.bin").write_text("weights")
    state = {"step": 10, "trained_examples": 80, "heldout_after": 0.5,
             "corpus": {"fixture": "fake-training-v1"}}
    (ck / "state.json").write_text(json.dumps(state))
    receipt = completed_checkpoint_receipt(plan, sha)
    assert receipt["step"] == 10 and len(receipt["weight_sha256"]) == 1
    status.write_text(json.dumps({"status": "complete", "plan_sha256": sha}))
    with pytest.raises(ValueError, match="supervisor has not completed"):
        completed_checkpoint_receipt(plan, sha)
    status.write_text(json.dumps({"state": "complete", "plan_sha256": sha}))
    state.pop("heldout_after")
    (ck / "state.json").write_text(json.dumps(state))
    with pytest.raises(ValueError, match="complete training target"):
        completed_checkpoint_receipt(plan, sha)


def test_supervisor_retries_from_checkpoint_and_never_restarts_complete(tmp_path):
    child = tmp_path / "fake_train.py"
    child.write_text('''import json, os, pathlib, sys
from pathlib import Path
ck=Path(os.environ["CHECKPOINT_STATE"]); attempts=Path(os.environ["ATTEMPTS"]); ck.parent.mkdir(parents=True,exist_ok=True)
def save(s):
 ck.parent.mkdir(parents=True,exist_ok=True); ck.write_text(json.dumps(s))
 for name in ("optimizer.pt","scheduler.pt","rng.pt"): (ck.parent/name).write_text("toy")
 (ck.parent/"weights").mkdir(exist_ok=True); (ck.parent/"weights"/"model.bin").write_text("toy")
a=int(attempts.read_text())+1 if attempts.exists() else 1; attempts.write_text(str(a))
s=json.loads(ck.read_text()) if ck.exists() else {"step":0,"trained_examples":0,"corpus":{"fixture":"fake-training-v1"}}
if a==1:
 s.update(step=1,trained_examples=2); save(s); sys.exit(7)
for step in range(s["step"]+1,4):
 s.update(step=step,trained_examples=step*2); save(s)
''')
    plan, run = _write_plan(tmp_path, child, {"step": 3})
    wrong_pin = subprocess.run([sys.executable, str(SUPERVISOR), "run", str(plan),
                                "--expected-plan-sha256", "0" * 64],
                               text=True, capture_output=True, timeout=5)
    assert wrong_pin.returncode == 2
    assert "plan SHA256 differs" in wrong_pin.stderr
    assert not (run / "attempts").exists()
    result = subprocess.run([sys.executable, str(SUPERVISOR), "run", str(plan)],
                            text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert json.loads(Path(json.loads(plan.read_text())["checkpoint_state"]).read_text())["step"] == 3
    assert (run / "attempts").read_text() == "2"
    assert json.loads((run / "supervisor-status.json").read_text())["state"] == "complete"

    # A later service start verifies the target and exits without spawning trainer.
    again = subprocess.run([sys.executable, str(SUPERVISOR), "run", str(plan)],
                           text=True, capture_output=True, timeout=5)
    assert again.returncode == 0, again.stderr
    assert (run / "attempts").read_text() == "2"


def test_operator_stop_latch_blocks_restart_until_explicit_resume(tmp_path):
    child = tmp_path / "signal_train.py"
    child.write_text('''import json, os, signal, time
from pathlib import Path
ck=Path(os.environ["CHECKPOINT_STATE"]); ck.parent.mkdir(parents=True,exist_ok=True)
def save(s):
 ck.parent.mkdir(parents=True,exist_ok=True); ck.write_text(json.dumps(s))
 for name in ("optimizer.pt","scheduler.pt","rng.pt"): (ck.parent/name).write_text("toy")
 (ck.parent/"weights").mkdir(exist_ok=True); (ck.parent/"weights"/"model.bin").write_text("toy")
if ck.exists() and json.loads(ck.read_text()).get("step",0) >= 1:
 s=json.loads(ck.read_text()); s.update(step=2,trained_examples=4); save(s); raise SystemExit(0)
def stop(*_):
 save({"step":1,"trained_examples":2,"corpus":{"fixture":"fake-training-v1"}}); raise SystemExit(0)
signal.signal(signal.SIGTERM,stop)
while True: time.sleep(.05)
''')
    plan, run = _write_plan(tmp_path, child, {"step": 2})
    proc = subprocess.Popen([sys.executable, str(SUPERVISOR), "run", str(plan)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    status = run / "supervisor-status.json"
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if status.exists() and json.loads(status.read_text()).get("state") == "running":
            break
        time.sleep(.02)
    else:
        proc.kill()
        raise AssertionError("supervisor never entered running state")

    # A competing launch cannot acquire the run lock or overwrite its status.
    concurrent = subprocess.run([sys.executable, str(SUPERVISOR), "run", str(plan)],
                                text=True, capture_output=True, timeout=5)
    assert concurrent.returncode == 2
    assert "another supervisor owns" in concurrent.stderr
    assert json.loads(status.read_text())["state"] == "running"

    stopped = subprocess.run([sys.executable, str(SUPERVISOR), "stop", str(plan)],
                             text=True, capture_output=True, timeout=5)
    assert stopped.returncode == 0, stopped.stderr
    assert proc.wait(timeout=5) == 0
    assert json.loads(status.read_text())["state"] == "operator_stopped"

    blocked = subprocess.run([sys.executable, str(SUPERVISOR), "run", str(plan)],
                             text=True, capture_output=True, timeout=5)
    assert blocked.returncode == 0
    assert json.loads(status.read_text())["state"] == "operator_stopped"
    subprocess.run([sys.executable, str(SUPERVISOR), "resume", str(plan)], check=True,
                   text=True, capture_output=True, timeout=5)
    # Resumed command observes step 1 and reaches the target.
    resumed = subprocess.run([sys.executable, str(SUPERVISOR), "run", str(plan)],
                             text=True, capture_output=True, timeout=5)
    assert resumed.returncode == 0, resumed.stderr
    assert json.loads(status.read_text())["state"] == "complete"


def test_completion_uses_checkpoint_target_or_corpus_example_target():
    plan = {"checkpoint_identity": {"fixture": "toy"}}
    assert is_complete({**plan, "completion": {"step": 5}},
                       {"step": 5, "corpus": {"fixture": "toy"}})
    assert is_complete({**plan, "completion": {"trained_examples": 20}},
                       {"step": 4, "trained_examples": 20, "corpus": {"fixture": "toy"}})
    assert not is_complete({**plan, "completion": {"step": 5}},
                           {"step": 4, "corpus": {"fixture": "toy"}})
    with __import__("pytest").raises(ValueError, match="must pin step"):
        is_complete({**plan, "completion": {}}, {"step": 0, "corpus": {"fixture": "toy"}})
    assert not is_complete({**plan, "completion": {"step": 5, "final_evaluation_required": True}},
                           {"step": 5, "heldout_before": 0.2, "corpus": {"fixture": "toy"}})
