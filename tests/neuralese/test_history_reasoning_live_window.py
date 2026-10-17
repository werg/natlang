import importlib.util
from pathlib import Path

import pytest


_SCRIPT = (Path(__file__).resolve().parents[2] / "scripts" / "neuralese_data" /
           "run_history_reasoning_live_window.py")
_SPEC = importlib.util.spec_from_file_location("history_reasoning_live_window", _SCRIPT)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_live_window_output_binding_accepts_exact_receipt():
    output = {"path": "/data/train.zst", "sha256": "a" * 64, "bytes": 8,
              "rows": 2, "tokens": 16, "max_row_tokens": 9}
    _MODULE._assert_output_binding(output, dict(output), description="train output")


@pytest.mark.parametrize("field,value", [
    ("path", "/data/other.zst"),
    ("sha256", "b" * 64),
    ("bytes", 9),
    ("rows", 3),
    ("tokens", 17),
    ("max_row_tokens", 10),
])
def test_live_window_output_binding_rejects_proof_receipt_mismatch(field, value):
    proof_output = {"path": "/data/train.zst", "sha256": "a" * 64, "bytes": 8,
                    "rows": 2, "tokens": 16, "max_row_tokens": 9}
    receipt_output = dict(proof_output)
    receipt_output[field] = value
    with pytest.raises(ValueError, match="proof/receipt bindings disagree"):
        _MODULE._assert_output_binding(proof_output, receipt_output, description="train output")


def test_live_window_output_binding_rejects_missing_side():
    with pytest.raises(ValueError, match="missing"):
        _MODULE._assert_output_binding(None, {}, description="train output")
