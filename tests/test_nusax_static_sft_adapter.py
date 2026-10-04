import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/prepare_nusax_translation_ir.py"
SPEC = importlib.util.spec_from_file_location("nusax_static_sft_adapter", SCRIPT)
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _jsonl(rows):
    return b"".join(adapter.canonical_bytes(row) + b"\n" for row in rows)


def _fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(adapter, "checkout_revision", lambda _path: adapter.NUSAX_REVISION)
    source = tmp_path / "nusax"
    (source / "datasets/mt").mkdir(parents=True)
    (source / "datasets/LICENSE").parent.mkdir(exist_ok=True)
    (source / "datasets/LICENSE").write_text("fixture license", encoding="utf-8")
    all_langs = {lang: f"{lang}-0" for lang in adapter.LANGUAGES}
    all_langs.update(english="A source sentence.", indonesian="Kalimat sumber.")
    csv_header = ",".join(["", *adapter.LANGUAGES])
    split_files = {}
    for split, row_id in (("train", "0"), ("valid", "0"), ("test", "0")):
        raw = (csv_header + "\n" + ",".join([row_id, *(all_langs[lang] for lang in adapter.LANGUAGES)]) + "\n").encode()
        relative = f"datasets/mt/{split}.csv"
        (source / relative).write_bytes(raw)
        split_files[relative] = {"sha256": _sha(raw), "bytes": len(raw)}
    license_path = source / "datasets/LICENSE"
    split_files["datasets/LICENSE"] = {"sha256": adapter.sha256_file(license_path), "bytes": license_path.stat().st_size}

    packet = tmp_path / "source-packet"
    packet.mkdir()
    tasks, refs = [], []
    for src, dst, text, target in (("english", "indonesian", "A source sentence.", "Kalimat sumber."),
                                   ("indonesian", "english", "Kalimat sumber.", "A source sentence.")):
        task_id = f"task-{src}-{dst}"
        tasks.append({"id": task_id, "source_group": "nusax/mt/train/0", "role": "train_support",
                      "original_split": "train", "source_language": src, "target_language": dst,
                      "input": {"text": text}, "provenance": {"row_id": "0"}})
        refs.append({"task_id": task_id, "source_group": "nusax/mt/train/0", "reference_target": target,
                     "visibility": "host-only"})
    task_blob, ref_blob = _jsonl(tasks), _jsonl(refs)
    (packet / "translation-source-tasks.jsonl").write_bytes(task_blob)
    (packet / "translation-source-references.host-only.jsonl").write_bytes(ref_blob)
    source_files = {name: {"sha256": value["sha256"], "bytes": value["bytes"]} for name, value in split_files.items()}
    manifest = {"schema": "natlang.translation-source-manifest/1", "task_count": len(tasks),
                "host_reference_count": len(refs), "source": {"revision": adapter.NUSAX_REVISION, "files": source_files},
                "readiness": {"self_improvement_reward": {"admission": "held"}},
                "artifacts": {"translation-source-tasks.jsonl": {"sha256": _sha(task_blob)},
                              "translation-source-references.host-only.jsonl": {"sha256": _sha(ref_blob)}}}
    (packet / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return packet, source, tasks, refs


def test_ir_keeps_reference_host_only_and_preserves_shared_group_roles(tmp_path, monkeypatch):
    packet, source, _, _ = _fixture(tmp_path, monkeypatch)
    ir_rows, refs, manifest = adapter.prepare_packet(packet, source)
    assert len(ir_rows) == len(refs) == 2
    assert all(row["version"] == "natlang.program/2" for row in ir_rows)
    assert all(row["source_groups"] == ["nusax/mt/train/0"] for row in ir_rows)
    assert all(row["generation"]["original_task_role"] == "train_support" for row in ir_rows)
    assert all(row["external_source"]["quality"]["status"] == "held" for row in ir_rows)
    assert refs[0]["target_text"] == "Kalimat sumber."
    assert "Kalimat sumber." not in json.dumps(ir_rows[0], ensure_ascii=False)
    assert manifest["protected_original_test"]["rows_emitted"] == 0
    assert manifest["rights_review"]["training_admission"] == "held"
    assert manifest["readiness"]["self_improvement_reward"]["exact_reference_string_grading"] is False


def test_source_or_reference_mismatch_fails_closed(tmp_path, monkeypatch):
    packet, source, _, refs = _fixture(tmp_path, monkeypatch)
    refs[0]["reference_target"] = "wrong target"
    ref_blob = _jsonl(refs)
    (packet / "translation-source-references.host-only.jsonl").write_bytes(ref_blob)
    manifest_path = packet / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"]["translation-source-references.host-only.jsonl"]["sha256"] = _sha(ref_blob)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="host_reference_source_mismatch"):
        adapter.prepare_packet(packet, source)
