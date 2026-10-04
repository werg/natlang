import importlib.util
import sys
import zipfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("nllb_seed_translation_adapter",
                                              SCRIPTS / "prepare_nllb_seed_translation_ir.py")
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)


def _archive(path: Path, *, include_flores=False):
    directions = {
        "ace_Latn-eng_Latn": {
            "ace_Latn": "Ace 0\nEcho\nEcho\nAce 3\n",
            "eng_Latn": "English\u2028line\nSame sentence\nSame sentence\nEnglish 3\n",
        },
        "eng_Latn-ace_Latn": {
            "eng_Latn": "Reverse English 0\nUnique 1\nUnique 2\nUnique 3\nUnique 4\n",
            "ace_Latn": "Ace ref 0\nAnother 1\nEcho\nEcho\nAce ref 4\n",
        },
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("NLLB-Seed/", b"")
        for folder, files in directions.items():
            zf.writestr(f"NLLB-Seed/{folder}/", b"")
            for name, text in files.items():
                zf.writestr(f"NLLB-Seed/{folder}/{name}", text.encode("utf-8"))
        if include_flores:
            zf.writestr("FLORES-200/test.txt", b"must not be read")


def test_candidate_keeps_host_references_separate_and_links_exact_text_across_directions(tmp_path):
    archive = tmp_path / "tiny.zip"
    _archive(archive)
    ir_rows, refs, _, manifest = adapter.prepare_candidate(
        archive, max_parallel_rows=5, verify_pin=False, allowed_languages={"ace_Latn", "eng_Latn"})

    assert len(ir_rows) == len(refs) == 9
    assert manifest["candidate_counts"]["by_direction_and_split"]
    ir_by_source_id = {row["source_ids"][0]: row for row in ir_rows}
    ref_by_id = {row["task_id"]: row for row in refs}
    forward = "nllb-seed:ace_Latn->eng_Latn:line-1"
    reverse = "nllb-seed:eng_Latn->ace_Latn:line-1"
    record = ir_by_source_id[forward]
    assert record["version"] == "natlang.program/2"
    assert record["semantics"]["inputs"]["source_text"] == "Ace 0"
    assert "English\u2028line" not in str(record["semantics"]["inputs"])
    assert ref_by_id[forward]["target_text"] == "English\u2028line"
    assert record["external_source"]["source_line_content_sha256"]
    assert record["external_source"]["target_line_content_sha256"]

    # Equal row indexes across direction folders are not treated as a shared identity.
    # This pair has different input/reference strings; only exact sentence reuse may link it.
    assert ir_by_source_id[forward]["source_groups"] != ir_by_source_id[reverse]["source_groups"]

    # Exact sentence reuse joins across different direction folders and row indexes.
    shifted_echo = "nllb-seed:eng_Latn->ace_Latn:line-3"
    forward_echo = "nllb-seed:ace_Latn->eng_Latn:line-2"
    assert ir_by_source_id[shifted_echo]["generation"]["parallel_row_index_zero_based"] == 2
    assert ir_by_source_id[forward_echo]["generation"]["parallel_row_index_zero_based"] == 1
    assert ir_by_source_id[shifted_echo]["source_groups"] == ir_by_source_id[forward_echo]["source_groups"]
    assert ir_by_source_id[shifted_echo]["split"] == ir_by_source_id[forward_echo]["split"]

    # Equal row indexes alone do not connect unrelated source/reference pairs.
    assert ir_by_source_id[forward]["source_groups"] != ir_by_source_id[reverse]["source_groups"]
    assert manifest["readiness"]["static_sft"]["training_admission"] == "held"
    assert manifest["readiness"]["self_improvement_reward"]["exact_reference_string_grading"] is False
    assert manifest["protected_evaluation"]["flores_evaluation_used"] is False


def test_archive_pin_and_flores_paths_fail_closed(tmp_path):
    archive = tmp_path / "tiny.zip"
    _archive(archive)
    with pytest.raises(ValueError, match="archive_hash_mismatch"):
        adapter.prepare_candidate(archive, max_parallel_rows=2, verify_pin=True,
                                  expected_sha256="0" * 64,
                                  allowed_languages={"ace_Latn", "eng_Latn"})
    _archive(archive, include_flores=True)
    with pytest.raises(ValueError, match="unsafe_or_unexpected_archive_path"):
        adapter.prepare_candidate(archive, max_parallel_rows=2, verify_pin=False,
                                  allowed_languages={"ace_Latn", "eng_Latn"})
