import hashlib
import json

import pytest

from scripts.periodic_heldout import hydrate_support_metadata
from scripts.report_training_wandb import read_audited_support_sidecar


def _row(row_id, family, source_id, groups):
    return {
        "id": row_id,
        "program_id": "program:" + row_id,
        "task_family": family,
        "family": "teacher_program",
        "task_kind": "lambda_source",
        "task_modality": "function",
        "source": "reference-native",
        "source_ids": [source_id],
        "source_groups": groups,
        "messages": [{"role": "assistant", "content": "private training text"}],
        "gold": "private target",
    }


def test_support_hydration_seeks_only_selected_rows_without_changing_identity_or_order(tmp_path):
    path = tmp_path / "corpus.jsonl"
    rows = [_row("a", "family-a", "source-a", ["group-a"]),
            _row("b", "family-b", "source-b", ["group-b"]),
            _row("c", "family-c", "source-c", ["group-c"])]
    offsets = []
    offset = 0
    encoded = []
    for row in rows:
        raw = (json.dumps(row) + "\n").encode()
        offsets.append(offset)
        offset += len(raw)
        encoded.append(raw)
    path.write_bytes(b"".join(encoded))
    selected = [{"id": rows[i]["id"], "program_id": rows[i]["program_id"],
                 "offset": offsets[i], "source_groups": rows[i]["source_groups"],
                 "split": "test"} for i in (2, 0)]
    hydrated = hydrate_support_metadata(path, selected)
    assert [row["id"] for row in hydrated] == ["c", "a"]
    assert [row["offset"] for row in hydrated] == [offsets[2], offsets[0]]
    assert [row["split"] for row in hydrated] == ["test", "test"]
    assert [row["task_family"] for row in hydrated] == ["family-c", "family-a"]
    assert [row["source_ids"] for row in hydrated] == [["source-c"], ["source-a"]]
    assert all("messages" not in row and "gold" not in row for row in hydrated)


@pytest.mark.parametrize("mutate", [
    lambda indexed: indexed.update(id="other"),
    lambda indexed: indexed.update(program_id="other-program"),
    lambda indexed: indexed.update(source_groups=["wrong-group"]),
    lambda indexed: indexed.update(offset=10_000),
])
def test_support_hydration_fails_closed_on_bad_identity_or_offset(tmp_path, mutate):
    row = _row("a", "family-a", "source-a", ["group-a"])
    path = tmp_path / "corpus.jsonl"
    path.write_text(json.dumps(row) + "\n")
    indexed = {"id": "a", "program_id": "program:a", "offset": 0,
               "source_groups": ["group-a"]}
    mutate(indexed)
    with pytest.raises(ValueError):
        hydrate_support_metadata(path, [indexed])


def test_audited_support_observer_is_hashbound_scalar_only_and_allowlisted(tmp_path):
    sidecar = tmp_path / "support.json"
    identity = {"data_sha256": "d" * 64, "split_sha256": "s" * 64,
                "model_identity_sha256": "m" * 64, "subset_ids_sha256": "i" * 64,
                "trainable_weights_sha256": "w" * 64, "step": 500}
    identity_sha = hashlib.sha256(json.dumps(
        identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    item = {
        "schema": "natlang.periodic_heldout_support_aggregate/1",
        "status": "audited", "identity_sha256": identity_sha,
        "identity": identity, "step": 500,
        "counts": {"rows": 128, "class_count": 15, "source_id_count": 108,
                   "source_id_links": 162, "source_group_count": 196,
                   "source_group_links": 379, "modality_count": 4},
        "examples": [{"id": "never-export"}],
    }
    sidecar.write_text(json.dumps(item))
    spec = {"path": str(sidecar), "sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest(),
            "step": 500, "identity_sha256": identity_sha}
    digest, identity, step, counts = read_audited_support_sidecar(spec)
    assert digest == spec["sha256"] and identity == identity_sha
    assert step == 500
    assert counts["rows"] == 128 and "examples" not in counts
    sidecar.write_text(json.dumps({**item, "counts": {**item["counts"], "unsafe": 1}}))
    spec["sha256"] = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="exactly the approved count fields"):
        read_audited_support_sidecar(spec)
    sidecar.write_text(json.dumps(item))
    spec["sha256"] = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    spec["step"] = 501
    with pytest.raises(ValueError, match="pinned plan identity"):
        read_audited_support_sidecar(spec)
    spec["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        read_audited_support_sidecar(spec)
