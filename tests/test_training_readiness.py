import pytest
import torch
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from scripts.training_readiness import (clip_finite_grad_norm_, require_finite_loss,
                                        validate_training_audit, validate_training_audit_tokenizer)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_loss_is_rejected_before_backward(value):
    loss = torch.tensor(value, requires_grad=True)
    with pytest.raises(FloatingPointError, match="non-finite training loss"):
        require_finite_loss(loss, 9)
    assert loss.grad is None


def test_finite_loss_and_gradients_pass_readiness_guard():
    parameter = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    loss = (parameter ** 2).sum()
    require_finite_loss(loss, 1)
    loss.backward()
    norm = clip_finite_grad_norm_([parameter], 1.0)
    assert torch.isfinite(norm)
    assert torch.linalg.vector_norm(parameter.grad) <= 1.000001


def test_nonfinite_gradient_norm_is_rejected_before_optimizer_step():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    parameter.grad = torch.tensor([float("nan")])
    with pytest.raises(RuntimeError, match="non-finite"):
        clip_finite_grad_norm_([parameter], 1.0)


def test_audit_validation_binds_readiness_to_data_bytes_and_length(tmp_path):
    data = tmp_path / "tiny.sft.jsonl"
    data.write_bytes(b'{"prompt":"p","completion":"c"}\n')
    manifest_path = tmp_path / "tiny.sft.jsonl.manifest.json"
    manifest = {"sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
                "renderer": {"model": "org/model", "revision": "abc123"},
                "audit": {"max_len": 16, "ready": True}}
    manifest_path.write_text(json.dumps(manifest))

    assert validate_training_audit(data, 16, "org/model", "abc123") == manifest
    with pytest.raises(ValueError, match="renderer model"):
        validate_training_audit(data, 16, "other/model", "abc123")
    with pytest.raises(ValueError, match="renderer revision"):
        validate_training_audit(data, 16, "org/model", "changed")
    with pytest.raises(ValueError, match="max_len"):
        validate_training_audit(data, 17, "org/model", "abc123")
    manifest["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="SHA-256"):
        validate_training_audit(data, 16, "org/model", "abc123")
    manifest["sha256"] = hashlib.sha256(data.read_bytes()).hexdigest()
    manifest["audit"]["ready"] = False
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="not ready"):
        validate_training_audit(data, 16, "org/model", "abc123")


def test_audit_validation_binds_loaded_tokenizer_fingerprint():
    class Backend:
        @staticmethod
        def to_str():
            return '{"model":"tiny"}'

    class Tokenizer:
        chat_template = "{{ messages }}"
        backend_tokenizer = Backend()
        special_tokens_map = {"eos_token": "<eos>"}
        eos_token = "<eos>"
        name_or_path = "org/model"

        @staticmethod
        def get_vocab():
            return {"<eos>": 0, "x": 1}

        @staticmethod
        def get_added_vocab():
            return {}

    from scripts.render_training_corpus import _tokenizer_info
    _, renderer = _tokenizer_info(Tokenizer(), "org/model", "abc123")
    manifest = {"renderer": renderer}
    validate_training_audit_tokenizer(manifest, Tokenizer(), "org/model", "abc123")
    manifest["renderer"]["tokenizer_fingerprint_sha256"] = "wrong"
    with pytest.raises(ValueError, match="tokenizer_fingerprint_sha256"):
        validate_training_audit_tokenizer(manifest, Tokenizer(), "org/model", "abc123")


def test_cpu_readiness_cli_writes_immutable_report(tmp_path):
    report = tmp_path / "readiness.json"
    script = Path(__file__).resolve().parents[1] / "scripts" / "training_readiness.py"
    command = [sys.executable, str(script), "--output", str(report)]
    first = subprocess.run(command, cwd=tmp_path, text=True, capture_output=True)
    assert first.returncode == 0, first.stderr
    result = json.loads(report.read_text())
    assert result["ready"] is True
    assert result["device"] == "cpu"
    assert all(status == "passed" for status in result["checks"].values())
    content = report.read_bytes()
    retry = subprocess.run(command, cwd=tmp_path, text=True, capture_output=True)
    assert retry.returncode == 0, retry.stderr
    assert report.read_bytes() == content
    report.write_text('{"ready": false}\n')
    changed = subprocess.run(command, cwd=tmp_path, text=True, capture_output=True)
    assert changed.returncode != 0
