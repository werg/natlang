"""Numerical preconditions and CPU self-check for safe training boundaries."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import sys
import tempfile
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def require_finite_loss(loss, step=None):
    """Reject an invalid scalar loss before backward can create bad gradients."""
    if not torch.isfinite(loss).all().item():
        where = f" at step {step}" if step is not None else ""
        raise FloatingPointError(f"non-finite training loss{where}")


def clip_finite_grad_norm_(parameters, max_norm):
    """Clip gradients while making a non-finite norm a hard pre-step failure."""
    return torch.nn.utils.clip_grad_norm_(parameters, max_norm, error_if_nonfinite=True)


def validate_training_audit(data_path, max_len, model=None, revision=None):
    """Require an audit manifest matching the exact training bytes and length."""
    data_path = Path(data_path)
    manifest_path = data_path.with_name(data_path.name + ".manifest.json")
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read training audit manifest {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"training audit manifest must be an object: {manifest_path}")
    audit = manifest.get("audit")
    hasher = hashlib.sha256()
    with data_path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    if not isinstance(audit, dict) or audit.get("ready") is not True:
        raise ValueError(f"training audit is not ready in {manifest_path}")
    if audit.get("max_len") != max_len:
        raise ValueError(f"training audit max_len {audit.get('max_len')!r} does not match --max-len {max_len}")
    if manifest.get("sha256") != hasher.hexdigest():
        raise ValueError(f"training audit SHA-256 does not match {data_path}")
    renderer = manifest.get("renderer")
    if not isinstance(renderer, dict):
        raise ValueError("training audit manifest has no renderer identity")
    if model is not None and renderer.get("model") != model:
        raise ValueError(f"training audit renderer model {renderer.get('model')!r} does not match --model {model!r}")
    if model is not None and renderer.get("revision") != revision:
        raise ValueError("training audit renderer revision does not match --model-revision")
    return manifest


def validate_training_audit_tokenizer(manifest, tokenizer, model, revision):
    """Bind audited token counts to the tokenizer actually loaded by training."""
    from scripts.render_training_corpus import _tokenizer_info

    _, actual = _tokenizer_info(tokenizer, model, revision)
    expected = manifest["renderer"]
    for key in ("model", "revision", "template_sha256", "tokenizer_fingerprint_sha256",
                "local_tokenizer_artifacts_sha256", "end_token"):
        if expected.get(key) != actual.get(key):
            raise ValueError(f"training audit tokenizer mismatch: {key}")


def _cpu_checkpoint_self_check():
    # Import the production writer and collator lazily; this remains a CPU-only
    # gate and runs the exact serialization/swap helper used by the trainer.
    from scripts.train_lora import (collate_completions, restore_rng_state,
                                    write_checkpoint_directory)

    encoded = collate_completions([([11, 12], [13, 14])], pad_id=0, device="cpu")
    if encoded["labels"].tolist() != [[-100, 13, 14]]:
        raise RuntimeError("completion mask self-check failed")

    try:
        require_finite_loss(torch.tensor(float("nan")), "self-check")
    except FloatingPointError:
        pass
    else:
        raise RuntimeError("non-finite loss self-check failed")
    parameter = torch.nn.Parameter(torch.ones(1))
    parameter.grad = torch.tensor([float("inf")])
    try:
        clip_finite_grad_norm_([parameter], 1.0)
    except RuntimeError:
        pass
    else:
        raise RuntimeError("non-finite gradient self-check failed")

    def components():
        model = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.Dropout(0.2), torch.nn.Linear(5, 2))
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 / (step + 1))
        return model, optimizer, scheduler

    def update(model, optimizer, scheduler, inputs, targets):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(inputs), targets)
        require_finite_loss(loss)
        loss.backward()
        clip_finite_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    original_torch, original_python = torch.get_rng_state(), random.getstate()
    try:
        torch.manual_seed(471)
        random.seed(479)
        inputs, targets = torch.randn(4, 3), torch.randn(4, 2)
        torch.manual_seed(487)
        uninterrupted, uninterrupted_opt, uninterrupted_sched = components()
        update(uninterrupted, uninterrupted_opt, uninterrupted_sched, inputs, targets)
        update(uninterrupted, uninterrupted_opt, uninterrupted_sched, inputs, targets)
        expected_model = {name: value.clone() for name, value in uninterrupted.state_dict().items()}
        expected_opt, expected_sched = uninterrupted_opt.state_dict(), uninterrupted_sched.state_dict()

        # Reset to the same initial state and produce one update/checkpoint.
        torch.manual_seed(487)
        interrupted, interrupted_opt, interrupted_sched = components()
        update(interrupted, interrupted_opt, interrupted_sched, inputs, targets)
        with tempfile.TemporaryDirectory(prefix="natlang-readiness-") as temp:
            root = Path(temp)

            def save_weights(weights):
                weights.mkdir()
                torch.save(interrupted.state_dict(), weights / "model.pt")

            # Keep the self-check CPU-only even in a container with visible GPUs.
            cpu_rng = {"torch_cpu": torch.get_rng_state().clone(), "torch_cuda": [],
                       "python": random.getstate()}
            write_checkpoint_directory(root, save_weights, interrupted_opt.state_dict(),
                                       interrupted_sched.state_dict(), cpu_rng, {"step": 1})
            checkpoint = root / "checkpoint"
            resumed, resumed_opt, resumed_sched = components()
            resumed.load_state_dict(torch.load(checkpoint / "weights/model.pt", weights_only=True))
            resumed_opt.load_state_dict(torch.load(checkpoint / "optimizer.pt", weights_only=False))
            resumed_sched.load_state_dict(torch.load(checkpoint / "scheduler.pt", weights_only=False))
            restore_rng_state(torch.load(checkpoint / "rng.pt", weights_only=False))
            update(resumed, resumed_opt, resumed_sched, inputs, targets)

        if any(not torch.equal(resumed.state_dict()[name], value)
               for name, value in expected_model.items()):
            raise RuntimeError("model continuation self-check failed")
        if resumed_sched.state_dict() != expected_sched:
            raise RuntimeError("scheduler continuation self-check failed")
        actual_opt = resumed_opt.state_dict()
        if actual_opt["param_groups"] != expected_opt["param_groups"]:
            raise RuntimeError("optimizer parameter-group self-check failed")
        for param_id, expected_values in expected_opt["state"].items():
            for name, value in expected_values.items():
                actual = actual_opt["state"][param_id][name]
                same = torch.equal(actual, value) if isinstance(value, torch.Tensor) else actual == value
                if not same:
                    raise RuntimeError("optimizer state continuation self-check failed")
    finally:
        torch.set_rng_state(original_torch)
        random.setstate(original_python)

    return {"completion_mask": "passed", "finite_loss_guard": "passed",
            "finite_gradient_guard": "passed", "adamw_scheduler_rng_resume": "passed"}


def write_immutable_report(path, report):
    """Publish a report atomically without replacing an existing readiness record."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    if path.exists():
        if path.read_bytes() == payload:
            return
        raise FileExistsError(f"readiness report already exists with different content: {path}")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    except FileExistsError:
        if path.exists() and path.read_bytes() == payload:
            return
        raise FileExistsError(f"readiness report already exists with different content: {path}")
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run CPU-only training checkpoint and numerical readiness checks.")
    parser.add_argument("--output", required=True, type=Path,
                        help="new report path; existing reports are never overwritten")
    args = parser.parse_args(argv)
    checks = _cpu_checkpoint_self_check()
    def source_sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    packages = {}
    for name in ("torch", "transformers", "peft"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    report = {"schema": "natlang-training-readiness-v1", "ready": True,
              "device": "cpu", "python": sys.version.split()[0], "packages": packages,
              "sources": {"training_readiness.py": source_sha(__file__),
                          "train_lora.py": source_sha(REPO_ROOT / "scripts/train_lora.py")},
              "checks": checks}
    write_immutable_report(args.output, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
