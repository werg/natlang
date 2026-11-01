import hashlib
import json

import pytest

from scripts.neuralese_data.build_native_graph_target_mask import build
from natlang_neuralese.data.native_graph_schema import parse_graph
from natlang_neuralese.data.native_sources import _row_bindings


FP = "f" * 64


def _graph():
    return {
        "schema": "natlang.neuralese-native-graph/1", "id": "rate-case",
        "split": "train", "source_groups": ["ext-unit:tatqa:context:train-1"],
        "tokenizer_fingerprint": FP,
        "nodes": [
            {"name": "root9", "token_ids": [10, 11, 12, 13],
             "target_weights": [0, 1, 1, 1]},
            {"name": "child10", "token_ids": [20, 21, 22],
             "target_weights": [0, 1, 1]},
            {"name": "writer", "token_ids": [30, 31, 32],
             "producing": [[1, 3]], "target_weights": [0, 1, 1]},
            {"name": "reader", "token_ids": [40, 41, 42],
             "imports": [{"at": 1, "source": "writer", "start": 1, "stop": 3}],
             "target_weights": [0, 0, 1]},
        ],
    }


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _request(tmp_path):
    row = _graph()
    parsed = parse_graph(row, tokenizer_fingerprint=FP)
    binding = _row_bindings([parsed])[row["id"]]
    source_path = tmp_path / "source.jsonl"
    source_path.write_text(json.dumps(row, separators=(",", ":")) + "\n")
    token_hash = _sha(json.dumps([11, 12, 13], separators=(",", ":")).encode())
    request = {
        "schema": "natlang.native-graph-target-mask-request/1",
        "source": {"path": str(source_path), "sha256": _sha(source_path.read_bytes())},
        "tokenizer_fingerprint": FP,
        "review": {"review_id": "review-rate-repeat-v1",
                    "rationale": "Repeated classification delegation is not useful target behavior."},
        "targets": [{"id": "rate-case", "binding": binding, "spans": [
            {"node": "root9", "token_start": 1, "token_stop": 4,
             "expected_positive_tokens": 3, "expected_token_ids_sha256": token_hash},
            {"node": "child10", "token_start": 1, "token_stop": 3,
             "expected_positive_tokens": 2,
             "expected_token_ids_sha256": _sha(json.dumps([21, 22], separators=(",", ":")).encode())},
        ]}],
    }
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request))
    return request_path


def test_build_exact_mask_binds_source_and_preserves_all_graph_dependencies(tmp_path):
    request = _request(tmp_path)
    output, receipt_path = tmp_path / "masked.jsonl", tmp_path / "receipt.json"
    receipt = build(request, output, receipt_path)
    row = json.loads(output.read_text())
    parsed = parse_graph(row, tokenizer_fingerprint=FP)
    assert row["nodes"][0]["target_weights"] == [0, 0.0, 0.0, 0.0]
    assert row["nodes"][1]["target_weights"] == [0, 0.0, 0.0]
    assert row["nodes"][2]["target_weights"] == [0, 1, 1]
    assert row["nodes"][3]["imports"][0]["source"] == "writer"
    assert len(parsed.plan.invocations) == 4
    assert receipt["training_admission"] is False
    assert receipt["masked_graphs"][0]["binding_before"] != receipt["masked_graphs"][0]["binding_after"]
    with pytest.raises(ValueError, match="new paths"):
        build(request, output, tmp_path / "second-receipt.json")


def test_build_exact_mask_rejects_changed_source_without_writing_output(tmp_path):
    request = _request(tmp_path)
    # Mutate an action token after the review request bound the source bytes.
    source_path = tmp_path / "source.jsonl"
    row = json.loads(source_path.read_text())
    row["nodes"][0]["token_ids"][1] = 999
    source_path.write_text(json.dumps(row, separators=(",", ":")) + "\n")
    output, receipt = tmp_path / "masked.jsonl", tmp_path / "receipt.json"
    with pytest.raises(ValueError, match="source JSONL bytes"):
        build(request, output, receipt)
    assert not output.exists() and not receipt.exists()


def test_build_exact_mask_rejects_held_graph(tmp_path):
    request = _request(tmp_path)
    value = json.loads(request.read_text())
    row_path = tmp_path / "source.jsonl"
    row = json.loads(row_path.read_text())
    row["split"] = "test"
    row_path.write_text(json.dumps(row, separators=(",", ":")) + "\n")
    value["source"]["sha256"] = _sha(row_path.read_bytes())
    # Recompute the graph binding because split is part of graph identity.
    parsed = parse_graph(row, tokenizer_fingerprint=FP)
    value["targets"][0]["binding"] = _row_bindings([parsed])[row["id"]]
    request.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="train graphs"):
        build(request, tmp_path / "masked.jsonl", tmp_path / "receipt.json")
